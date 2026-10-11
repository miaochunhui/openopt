"""Offline SBOM controls; fixtures are real Git, wheel and dist metadata bytes."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import build_sbom as sbom


SCHEMA = Path(
    "/hf3fs-jd/data/tmp/openopt-release-20261011/sbom-delivery/standard/"
    "spdx-2.3.1-schema.json"
)


def git(root, *args):
    result = subprocess.run(
        ["git", "-C", str(root), "-c", "core.hooksPath=/dev/null", *args],
        env=sbom.git_env(), capture_output=True, check=False,
    )
    assert result.returncode == 0, "isolated Git fixture failed"
    return result.stdout.decode("ascii").strip()


def commit(root):
    git(root, "add", "pyproject.toml", "LICENSE", "src/openopt/__init__.py")
    git(root, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-m", "SBOM fixture")
    return git(root, "rev-parse", "HEAD")


def distribution(root, name, version="1.0", license_expression=None, requires=()):
    directory = root / f"{name.replace('-', '_')}-{version}.dist-info"
    directory.mkdir(parents=True)
    fields = [
        "Metadata-Version: 2.4", f"Name: {name}", f"Version: {version}",
        *(["License-Expression: " + license_expression] if license_expression else []),
        *["Requires-Dist: " + item for item in requires],
    ]
    path = directory / "METADATA"
    path.write_text("\n".join(fields) + "\n\n", encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "src/openopt").mkdir(parents=True)
    (root / "src/openopt/__init__.py").write_bytes(b'__version__ = "0.1.0"\n')
    (root / "LICENSE").write_bytes(b"Independent fixture license; not legal evidence.\n")
    (root / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=77"]\n'
        '[project]\nname = "openopt"\nversion = "0.1.0"\nlicense = "0BSD"\n'
        'dependencies = ["alpha>=1"]\n'
        '[project.optional-dependencies]\nextra = ["missing-addon>=2"]\n',
        encoding="ascii",
    )
    git(root, "init", "-b", "master")
    head = commit(root)
    metadata_root = tmp_path / "metadata"
    distribution(metadata_root, "alpha", license_expression="MIT",
                 requires=("beta>=1",))
    distribution(metadata_root, "beta")
    return root, head, metadata_root


def build(inputs, **kwargs):
    root, head, metadata_root = inputs
    return sbom.build(
        root, head, [metadata_root], (), "3.10",
        SCHEMA, sbom.SCHEMA_SHA256, **kwargs,
    )


def test_real_git_metadata_schema_unknown_license_and_optional_absent(inputs):
    document, evidence = build(inputs)
    assert document["spdxVersion"] == "SPDX-2.3"
    assert document["dataLicense"] == "CC0-1.0"
    assert evidence["verdict"] == "PASS"
    assert evidence["scope"] == "selected_python_metadata_closure"
    assert evidence["legal_qualification"] == "UNVERIFIED"
    assert evidence["runtime_qualification"] == "NOT_TESTED"
    packages = {item["name"]: item for item in document["packages"]}
    assert packages["openopt-source"]["licenseDeclared"] == "0BSD"
    assert packages["alpha"]["licenseDeclared"] == "MIT"
    assert packages["beta"]["licenseDeclared"] == "NOASSERTION"
    assert packages["beta"]["licenseConcluded"] == "NOASSERTION"
    absent = [row for row in evidence["declarations"]
              if row["name"] == "missing-addon"]
    assert absent[0]["resolution"] == "OPTIONAL_ABSENT"
    assert absent[0]["active"] is False
    assert evidence["unknown_licenses"] == ["beta"]
    assert len(evidence["resolved"]) == 2
    assert all(row["metadata_sha256"] for row in evidence["resolved"])
    assert evidence["build_requirements"] == [
        {"requirement": "setuptools>=77", "resolution": "DECLARED_NOT_RESOLVED"}]


def test_selected_optional_missing_is_explicit_incomplete(inputs):
    root, head, metadata_root = inputs
    _, evidence = sbom.build(
        root, head, [metadata_root], ("extra",), "3.10",
        SCHEMA, sbom.SCHEMA_SHA256,
    )
    assert evidence["verdict"] == "INCOMPLETE"
    assert "missing-addon" in evidence["unresolved"]
    assert evidence["first_cause"] == "active_dependency_absent"


def test_missing_required_is_not_success(inputs):
    root, head, _ = inputs
    _, evidence = sbom.build(
        root, head, [], (), "3.10", SCHEMA, sbom.SCHEMA_SHA256,
    )
    assert evidence["verdict"] == "INCOMPLETE"
    assert evidence["unresolved"] == ["alpha"]


@pytest.mark.parametrize("expected", [None, "", "short", True])
def test_explicit_full_source_pin_required(inputs, expected):
    root, _, metadata_root = inputs
    with pytest.raises(sbom.SbomError, match="^source_pin_required$"):
        sbom.build(root, expected, [metadata_root], (), "3.10",
                   SCHEMA, sbom.SCHEMA_SHA256)


@pytest.mark.parametrize("payload", [
    b"", b"not-metadata", b"Name: alpha\nName: beta\nVersion: 1\n\n",
    b"Name: alpha\nVersion: invalid version!\n\n",
    b"Name: alpha\nVersion: 1\nRequires-Dist: broken ???\n\n",
    b"Name: alpha\nVersion: 1\nLicense-Expression: MIT OR\n\n",
    b"Name: alpha\nVersion: 1\nLicense-Expression: MIT\nLicense-Expression: BSD\n\n",
])
def test_malformed_metadata_rejects_without_raw_content(tmp_path, payload):
    path = tmp_path / "METADATA"
    path.write_bytes(payload)
    with pytest.raises(sbom.SbomError) as error:
        sbom.parse_metadata(path)
    assert str(error.value) in {
        "metadata_invalid", "metadata_identity_invalid",
        "metadata_requirement_invalid", "metadata_license_invalid",
    }
    assert "???" not in str(error.value)


def test_legacy_license_is_hash_not_relicensed_or_copied(tmp_path):
    path = distribution(tmp_path, "legacy")
    path.write_bytes(path.read_bytes().replace(
        b"\n\n", b"\nLicense: A historical multi-license statement.\n\n"))
    row = sbom.parse_metadata(path)
    assert row["license_declared"] == "NOASSERTION"
    assert row["license_metadata"]["legacy_sha256"]
    assert "historical" not in json.dumps(row)


def test_duplicate_metadata_is_not_first_match_wins(inputs):
    root, head, metadata_root = inputs
    distribution(metadata_root, "alpha", "2.0")
    with pytest.raises(sbom.SbomError, match="^ambiguous_dependency_metadata:alpha$"):
        build(inputs)


def test_inactive_duplicate_dependency_is_visible_not_active_failure(inputs):
    distribution(inputs[2], "missing-addon", "2.0")
    distribution(inputs[2], "missing-addon", "3.0")
    _, evidence = build(inputs)
    assert evidence["verdict"] == "PASS"
    row = next(item for item in evidence["declarations"]
               if item["name"] == "missing-addon")
    assert row["active"] is False
    assert row["resolution"] == "INACTIVE_AMBIGUOUS"
    assert "missing-addon" not in evidence["unresolved"]

def test_incompatible_installed_version_is_not_resolved(inputs):
    path = next(inputs[2].glob("alpha-*.dist-info/METADATA"))
    path.write_bytes(path.read_bytes().replace(b"Version: 1.0", b"Version: 0.1"))
    _, evidence = build(inputs)
    assert evidence["verdict"] == "INCOMPLETE"
    assert "alpha" in evidence["unresolved"]
    assert next(row for row in evidence["declarations"]
                if row["name"] == "alpha")["resolution"] == "VERSION_MISMATCH"


def test_direct_url_requirement_refused_without_network(inputs):
    path = next(inputs[2].glob("alpha-*.dist-info/METADATA"))
    path.write_bytes(path.read_bytes().replace(
        b"Requires-Dist: beta>=1", b"Requires-Dist: beta @ https://example.invalid/beta.whl"))
    with pytest.raises(sbom.SbomError, match="^direct_url_dependency_unsupported$"):
        build(inputs)


def test_unknown_extra_is_fail_loud(inputs):
    root, head, metadata_root = inputs
    with pytest.raises(sbom.SbomError, match="^unknown_project_extra$"):
        sbom.build(root, head, [metadata_root], ("not-declared",), "3.10",
                   SCHEMA, sbom.SCHEMA_SHA256)


def test_metadata_symlink_refused(inputs, tmp_path):
    path = next(inputs[2].glob("alpha-*.dist-info/METADATA"))
    external = tmp_path / "external-metadata"
    external.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(external)
    with pytest.raises(sbom.SbomError, match="^input_file_unavailable$"):
        build(inputs)


def test_dirty_source_refused(inputs):
    (inputs[0] / "src/openopt/__init__.py").write_bytes(b"changed\n")
    with pytest.raises(sbom.SbomError, match="^source_bytes_changed$"):
        build(inputs)


def test_assume_unchanged_cannot_hide_source_bytes(inputs):
    git(inputs[0], "update-index", "--assume-unchanged", "src/openopt/__init__.py")
    (inputs[0] / "src/openopt/__init__.py").write_bytes(b"hidden change\n")
    assert git(inputs[0], "status", "--porcelain") == ""
    with pytest.raises(sbom.SbomError, match="^source_bytes_changed$"):
        build(inputs)


def test_empty_declarations_refused(inputs):
    root, _, _ = inputs
    text = (root / "pyproject.toml").read_text().replace('dependencies = ["alpha>=1"]',
                                                       "dependencies = []")
    (root / "pyproject.toml").write_text(text)
    head = commit(root)
    with pytest.raises(sbom.SbomError, match="^empty_project_dependencies$"):
        sbom.build(root, head, [inputs[2]], (), "3.10", SCHEMA, sbom.SCHEMA_SHA256)


def test_build_detects_metadata_drift_after_read(inputs, monkeypatch):
    original = sbom.parse_metadata
    changed = False

    def mutate(path):
        nonlocal changed
        result = original(path)
        if not changed:
            changed = True
            path.write_bytes(path.read_bytes() + b"mutation\n")
        return result

    monkeypatch.setattr(sbom, "parse_metadata", mutate)
    with pytest.raises(sbom.SbomError, match="^metadata_changed$"):
        build(inputs)


def test_schema_pin_and_standard_rejection(inputs):
    with pytest.raises(sbom.SbomError, match="^schema_pin_mismatch$"):
        sbom.validate_standard({}, SCHEMA, "0" * 64)
    with pytest.raises(sbom.SbomError, match="^spdx_schema_rejected$"):
        sbom.validate_standard({}, SCHEMA, sbom.SCHEMA_SHA256)


def make_wheel(inputs, tmp_path, *, bad_record=False):
    root, _, _ = inputs
    metadata = (
        b"Metadata-Version: 2.4\nName: openopt\nVersion: 0.1.0\n"
        b"License-Expression: 0BSD\nRequires-Dist: alpha>=1\n"
        b"Provides-Extra: extra\nRequires-Dist: missing-addon>=2; extra == 'extra'\n\n"
    )
    files = {
        "openopt/__init__.py": (root / "src/openopt/__init__.py").read_bytes(),
        "openopt-0.1.0.dist-info/METADATA": metadata,
        "openopt-0.1.0.dist-info/licenses/LICENSE": (root / "LICENSE").read_bytes(),
        "openopt-0.1.0.dist-info/WHEEL": b"Wheel-Version: 1.0\nTag: py3-none-any\n",
    }
    record = io.StringIO(newline="")
    writer = csv.writer(record)
    for name, payload in files.items():
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=")
        writer.writerow((name, "sha256=" + digest.decode(), len(payload)))
    writer.writerow(("openopt-0.1.0.dist-info/RECORD", "", ""))
    files["openopt-0.1.0.dist-info/RECORD"] = record.getvalue().encode()
    if bad_record:
        files["openopt/__init__.py"] += b"mutation\n"
    wheel = tmp_path / "openopt-0.1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, payload in files.items():
            archive.writestr(name, payload)
    installed = tmp_path / "installed"
    installed.mkdir()
    for name, payload in files.items():
        path = installed / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    return wheel, installed


def test_real_wheel_installed_membership_checksums(inputs, tmp_path):
    wheel, installed = make_wheel(inputs, tmp_path)
    document, evidence = build(
        inputs, wheel=wheel, wheel_sha256=sbom.digest(wheel.read_bytes()),
        installed_root=installed,
    )
    assert evidence["wheel"]["record_verified"] is True
    assert evidence["wheel"]["source_binding"] == "PACKAGE_BYTES_MATCH_ONLY"
    assert evidence["installed"]["membership_missing"] == []
    assert evidence["installed"]["membership_extra"] == []
    assert any(item["name"] == "openopt-wheel" for item in document["packages"])


def test_wheel_record_mutation_refused(inputs, tmp_path):
    wheel, installed = make_wheel(inputs, tmp_path, bad_record=True)
    with pytest.raises(sbom.SbomError, match="^wheel_record_mismatch$"):
        build(inputs, wheel=wheel, wheel_sha256=sbom.digest(wheel.read_bytes()),
              installed_root=installed)


def test_installed_module_changed_refused(inputs, tmp_path):
    wheel, installed = make_wheel(inputs, tmp_path)
    (installed / "openopt/__init__.py").write_bytes(b"changed installed\n")
    with pytest.raises(sbom.SbomError, match="^installed_content_mismatch$"):
        build(inputs, wheel=wheel, wheel_sha256=sbom.digest(wheel.read_bytes()),
              installed_root=installed)


def test_installed_extra_module_refused(inputs, tmp_path):
    wheel, installed = make_wheel(inputs, tmp_path)
    (installed / "openopt/foreign.py").write_bytes(b"foreign\n")
    with pytest.raises(sbom.SbomError, match="^installed_membership_mismatch$"):
        build(inputs, wheel=wheel, wheel_sha256=sbom.digest(wheel.read_bytes()),
              installed_root=installed)


def test_output_create_once_and_reader_integrity(inputs, tmp_path):
    document, evidence = build(inputs)
    target = tmp_path / "delivery"
    path = sbom.save_delivery(target, document, evidence, require_shared=False)
    assert sbom.read_delivery(path, sbom.digest(path.read_bytes()), SCHEMA,
                              sbom.SCHEMA_SHA256)["verdict"] == "PASS"
    with pytest.raises(sbom.SbomError, match="^output_already_exists$"):
        sbom.save_delivery(target, document, evidence, require_shared=False)
    (target / "sbom.spdx.json").write_bytes(b"{}\n")
    with pytest.raises(sbom.SbomError, match="^artifact_pin_mismatch$"):
        sbom.read_delivery(path, sbom.digest(path.read_bytes()), SCHEMA,
                           sbom.SCHEMA_SHA256)


def test_reader_rejects_new_installed_member(inputs, tmp_path):
    wheel, installed = make_wheel(inputs, tmp_path)
    document, evidence = build(
        inputs, wheel=wheel, wheel_sha256=sbom.digest(wheel.read_bytes()),
        installed_root=installed)
    path = sbom.save_delivery(tmp_path / "delivery", document, evidence,
                              require_shared=False)
    (installed / "openopt/foreign.py").write_bytes(b"foreign\n")
    with pytest.raises(sbom.SbomError, match="^reader_installed_membership_changed$"):
        sbom.read_delivery(path, sbom.digest(path.read_bytes()), SCHEMA,
                           sbom.SCHEMA_SHA256)


def test_no_sdk_network_or_environment_credential_discovery(inputs, monkeypatch):
    import builtins
    import socket

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in {"torch", "gurobipy", "cplex", "mosek", "cuopt"}
        return original(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("network forbidden")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    _, evidence = build(inputs)
    assert evidence["network_attempts"] == evidence["sdk_imports"] == 0
    assert evidence["credential_files_read"] == 0


if __name__ == "__main__":
    raise SystemExit(sbom.run_tests(Path(__file__).resolve(), sys.argv[1:]))
