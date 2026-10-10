"""Portable renderer controls; synthetic inputs are never GPU evidence."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

import pytest

from scripts import prepare_publication as safe
from scripts import render_project_site as site


def sample():
    return {
        "schema_version": 1, "project": "openopt",
        "content_source_commit": "a" * 40,
        "modified_at": "2026-10-10T23:20:10+08:00", "license": "0BSD",
        "boundary_sha256": {"AGENTS.md": "b" * 64, "LICENSE": "c" * 64},
        "access": "existing-super-only", "publication": "NOT_PUBLISHED",
        "qualification": dict(site.CLAIMS),
        "evidence": {"cpu-smoke": {
            "source_commit": "d" * 40, "receipt_sha256": "e" * 64,
            "finished_at": "2026-10-10T22:30:01+08:00",
            "tests": 12, "passed": 10, "failures": 0, "errors": 0, "skipped": 2,
            "relation": "historical", "kind": "reported-owner-cpu-checks",
        }},
    }


def test_deterministic_complete_two_output_contract():
    payload = safe.encoded(sample())
    first, second = site.render(payload), site.render(payload)
    assert first == second and set(first) == set(site.OUTPUTS)
    page = first[site.OUTPUTS[0]].decode()
    assert "10<span> passed / 12 selected" in page and "2 skipped" in page
    assert "2026-10-10 23:20:10 +08:00" in page
    assert safe.sha(payload) in page
    for text in ("super-only", "NOT_PUBLISHED", "PENDING", "UNVERIFIED",
                 "not online acceptance", "not general native", "arbitrary-precision"):
        assert text in page
    assert "<script" not in page and "onchange=" not in page


def test_catalog_exact_existing_admin_contract():
    files = site.render(safe.encoded(sample()))
    catalog = safe.parse(files["docs.publish.json"], "catalog")
    assert catalog["version"] == 1 and catalog["project"] == "openopt"
    assert catalog["audiences"] == {"admin": {"path_prefix": ""}}
    assert len(catalog["pages"]) == 1
    page = catalog["pages"][0]
    assert set(page) == {"slug", "title", "source", "section", "order",
                         "audience", "status", "owner", "aliases"}
    assert page["source"] == "docs/site/index.html"
    assert page["audience"] == "admin" and page["aliases"] == []


@pytest.mark.parametrize("mutation", [
    "schema-bool", "schema-new", "unknown-top", "raw-array", "private-path",
    "sdk", "free-prose", "commercial-ready", "general-ready", "cuda-pass",
    "speed-win", "public-access", "published", "license", "digest-short",
    "digest-nonhex", "timestamp-minute", "timestamp-offset", "timestamp-invalid",
    "empty-evidence", "unknown-evidence", "count-bool", "count-negative",
    "count-denominator", "count-zero", "relation", "evidence-certified",
    "evidence-after-edit", "boundary-extra",
])
def test_unsafe_or_promoted_input_rejected(mutation):
    record = sample()
    item = record["evidence"]["cpu-smoke"]
    if mutation == "schema-bool":
        record["schema_version"] = True
    elif mutation == "schema-new":
        record["schema_version"] = 2
    elif mutation in {"unknown-top", "raw-array", "private-path", "sdk", "free-prose"}:
        record[mutation] = {"raw-array": [1, 2], "private-path": "/private/example",
                            "sdk": "proprietary-runtime", "free-prose": "<script>bad</script>"
                            }.get(mutation, 1)
    elif mutation in {"commercial-ready", "general-ready", "cuda-pass", "speed-win"}:
        key = {"commercial-ready": "commercial", "general-ready": "general_optimizer",
               "cuda-pass": "current_source_cuda", "speed-win": "fair_speed_comparison"}[mutation]
        record["qualification"][key] = "READY"
    elif mutation == "public-access":
        record["access"] = "public"
    elif mutation == "published":
        record["publication"] = "ONLINE_PASS"
    elif mutation == "license":
        record["license"] = "dependency-relicensed"
    elif mutation == "digest-short":
        record["content_source_commit"] = "a" * 7
    elif mutation == "digest-nonhex":
        item["receipt_sha256"] = "g" * 64
    elif mutation == "timestamp-minute":
        record["modified_at"] = "2026-10-10T23:20+08:00"
    elif mutation == "timestamp-offset":
        record["modified_at"] = "2026-10-10T23:20:10+00:00"
    elif mutation == "timestamp-invalid":
        record["modified_at"] = "2026-99-10T23:20:10+08:00"
    elif mutation == "empty-evidence":
        record["evidence"] = {}
    elif mutation == "unknown-evidence":
        record["evidence"]["commercial-solve"] = deepcopy(item)
    elif mutation == "count-bool":
        item["failures"] = False
    elif mutation == "count-negative":
        item["skipped"] = -1
    elif mutation == "count-denominator":
        item["passed"] = 12
    elif mutation == "count-zero":
        item.update(tests=0, passed=0, skipped=0)
    elif mutation == "relation":
        item["relation"] = "same-content-source"
    elif mutation == "evidence-certified":
        item["kind"] = "CERTIFIED"
    elif mutation == "evidence-after-edit":
        item["finished_at"] = "2026-10-11T01:00:00+08:00"
    elif mutation == "boundary-extra":
        record["boundary_sha256"]["internal"] = "e" * 64
    with pytest.raises(safe.Rejected):
        site.render(safe.encoded(record))


@pytest.mark.parametrize("payload", [
    b"", b"{}", b'{"schema_version":1,"schema_version":1}',
    b'{"a":NaN}', b'{"a":Infinity}', b"\xff",
])
def test_empty_duplicate_or_nonfinite_json_rejected(payload):
    with pytest.raises(safe.Rejected):
        site.render(payload)


def test_scalar_changes_come_only_from_input():
    record = sample()
    first = site.render(safe.encoded(record))
    record["evidence"]["cpu-smoke"].update(tests=13, passed=11)
    second = site.render(safe.encoded(record))
    assert first["docs.publish.json"] == second["docs.publish.json"]
    assert first["docs/site/index.html"] != second["docs/site/index.html"]
    assert b"11<span> passed / 13 selected" in second["docs/site/index.html"]


def test_timestamp_exemption_is_exact_and_never_masks_network_or_active_content():
    record = site.validate_input(sample())
    site.scan_content(safe.encoded(record), record, "test")
    for suffix in (b" https://example.invalid", b" 2001:db8:abcd::1",
                   b"<script>bad</script>", b" /private/unknown"):
        with pytest.raises(safe.Rejected):
            site.scan_content(safe.encoded(record) + suffix, record, "test")


def test_never_overwrite_partial_existing_outputs(tmp_path):
    files = site.render(safe.encoded(sample()))
    site.create_once(tmp_path / "docs.publish.json", b"existing\n")
    with pytest.raises(safe.Rejected, match="output_already_exists"):
        site.write_outputs(tmp_path, files)
    assert not (tmp_path / "docs/site/index.html").exists()
    assert (tmp_path / "docs.publish.json").read_bytes() == b"existing\n"


def test_exact_output_tamper_and_extra_catalog_fields_rejected(tmp_path):
    files = site.render(safe.encoded(sample()))
    site.write_outputs(tmp_path, files)
    site.check_outputs(tmp_path, files)
    catalog = tmp_path / "docs.publish.json"
    changed = json.loads(catalog.read_text())
    changed["audiences"]["public"] = {"path_prefix": "public"}
    catalog.write_bytes(safe.encoded(changed))
    with pytest.raises(safe.Rejected, match="generated_output_mismatch"):
        site.check_outputs(tmp_path, files)
    assert (tmp_path / "docs/site/index.html").read_bytes() == files["docs/site/index.html"]


@pytest.mark.parametrize("symlink_kind", ["parent", "leaf"])
def test_symlink_output_rejected(tmp_path, symlink_kind):
    files = site.render(safe.encoded(sample()))
    if symlink_kind == "parent":
        target = tmp_path / "elsewhere"
        target.mkdir()
        (tmp_path / "docs").symlink_to(target, target_is_directory=True)
    else:
        target = tmp_path / "elsewhere"
        target.write_bytes(b"untouched")
        (tmp_path / "docs.publish.json").symlink_to(target)
    with pytest.raises(safe.Rejected):
        site.write_outputs(tmp_path, files)
    if symlink_kind == "leaf":
        assert target.read_bytes() == b"untouched"


@pytest.mark.parametrize("private", [
    "/home/user/evidence", "http://internal.example", "Bearer abcdefgh",
    "%2Fdata%2Fprivate", "-----BEGIN PRIVATE KEY-----",
])
def test_private_payload_is_not_echoed(private):
    record = sample()
    record["evidence"]["cpu-smoke"]["receipt_sha256"] = private
    with pytest.raises(safe.Rejected) as found:
        site.render(safe.encoded(record))
    assert private not in str(found.value)


def init_repo(path):
    path.mkdir()
    for arguments in (
        ["init", "-q"], ["config", "user.name", "Synthetic test"],
        ["config", "user.email", "synthetic@example.invalid"],
    ):
        result = subprocess.run(["git", "-C", str(path), *arguments],
                                capture_output=True, check=False)
        assert result.returncode == 0
    (path / "LICENSE").write_text("BSD Zero Clause License\n")
    (path / "AGENTS.md").write_text("Synthetic bounded research; no qualification.\n")
    for name in site.RUNTIME_PATHS:
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((site.ROOT / name).read_bytes())
    controls = path / "tests/test_control.py"
    controls.parent.mkdir(parents=True, exist_ok=True)
    controls.write_text("\n".join(f"def test_case_{i}():\n    assert True\n" for i in range(12)))
    result = subprocess.run(["git", "-C", str(path), "add", "--", "LICENSE", "AGENTS.md", *site.RUNTIME_PATHS,
                             "tests/test_control.py"],
                            capture_output=True, check=False)
    assert result.returncode == 0
    result = subprocess.run(["git", "-C", str(path), "commit", "-qm", "synthetic source"],
                            capture_output=True, check=False)
    assert result.returncode == 0
    return site.git(path, "rev-parse", "HEAD").decode().strip()


def synthetic_receipt(repo, commit):
    return {
        "schema_version": 1, "verdict": "PASS", "mode": "smoke",
        "cuda_available": False, "finished_at": "2026-10-10T14:30:01+00:00",
        "source": {"commit": commit, "clean": True,
                   "tree": site.git(repo, "rev-parse", "HEAD^{tree}").decode().strip(),
                   "sha256": site.tree_hashes(repo, commit)},
        "source_consistency": {"verdict": "PASS", "same_commit": True,
                               "changed": [], "missing": [], "extra": [], "files": len(site.tree_hashes(repo, commit))},
        "tests": {"tests": 12, "failures": 0, "errors": 0, "skipped": 2},
    }


def cpu_authority(repo, receipt, name="cpu-smoke"):
    """Explicit synthetic caller pins; no real receipt is changed or trusted."""
    raw = json.loads(receipt.read_bytes())
    root = receipt.parent / (receipt.stem + "-caller")
    root.mkdir()
    counts = {"tests": 12, "failures": 0, "errors": 0, "skipped": 2}
    members = [f"tests.test_control::test_case_{i}" for i in range(12)]
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", {key: str(value) for key, value in counts.items()})
    for i, node in enumerate(members):
        cls, method = node.split("::", 1)
        case = ET.SubElement(suite, "testcase", classname=cls, name=method)
        if i >= 10:
            ET.SubElement(case, "skipped")
    payloads = {
        "receipt": (receipt, receipt.read_bytes()),
        "junit": (root / "pytest.xml", ET.tostring(xml)),
        "pytest_log": (root / "pytest.log", b"10 passed, 2 skipped in 1.0s\nreturncode=0\n"),
        "collection": (root / "collection.log", ("\n".join(
            node.replace("tests.test_control", "tests/test_control.py") for node in members) + "\n").encode()),
    }
    files = {}
    for role, (path, payload) in payloads.items():
        if role != "receipt":
            path.write_bytes(payload)
        files[role] = {"path": str(path), "sha256": safe.sha(payload), "bytes": len(payload)}
    manifest = {"schema_version": 1, "evidence": {name: {
        "source_commit": raw["source"]["commit"], "files": files, "membership": members,
        "pytest_exit_code": 0,
        "runtime": {"environment": {"CUDA_VISIBLE_DEVICES": "-1"}, "gpu_qualification": "UNVERIFIED"},
    }}}
    path = root / "authority.json"
    payload = safe.encoded(manifest)
    path.write_bytes(payload)
    return (path, safe.sha(payload))


@pytest.fixture(autouse=True)
def isolated_authority_root(request, tmp_path, monkeypatch):
    if request.node.name != "test_external_storage_rejects_local_or_repository_paths":
        monkeypatch.setattr(safe, "SHARED", tmp_path)


@pytest.mark.parametrize("mutation", [
    "none", "hash-pin", "source-pin", "coordinated-source-hash",
    "clean-bool", "consistency", "count-bool", "failed", "all-skipped",
    "cuda", "schema-bool", "tree", "missing-count",
])
def test_receipt_extraction_real_git_source_controls(tmp_path, mutation):
    repo = tmp_path / "repo"
    commit = init_repo(repo)
    raw = synthetic_receipt(repo, commit)
    if mutation == "coordinated-source-hash":
        raw["source"]["sha256"]["AGENTS.md"] = "a" * 64
    elif mutation == "clean-bool":
        raw["source"]["clean"] = 1
    elif mutation == "consistency":
        raw["source_consistency"]["same_commit"] = 1
    elif mutation == "count-bool":
        raw["tests"]["errors"] = False
    elif mutation == "failed":
        raw["tests"]["failures"] = 1
    elif mutation == "all-skipped":
        raw["tests"]["skipped"] = 12
    elif mutation == "cuda":
        raw["cuda_available"] = True
    elif mutation == "schema-bool":
        raw["schema_version"] = True
    elif mutation == "tree":
        raw["source"]["tree"] = "e" * 40
    elif mutation == "missing-count":
        del raw["tests"]["errors"]
    payload = safe.encoded(raw)
    path = tmp_path / "receipt.json"
    path.write_bytes(payload)
    expected_hash = "f" * 64 if mutation == "hash-pin" else safe.sha(payload)
    expected_commit = "f" * 40 if mutation == "source-pin" else commit
    authority = cpu_authority(repo, path)
    if mutation == "none":
        result = site.extract_receipt(repo, "cpu-smoke", path, expected_hash, expected_commit, authority)
        assert result["passed"] == 10 and result["tests"] == 12
        assert result["kind"] == "reported-owner-cpu-checks"
    else:
        with pytest.raises(safe.Rejected):
            site.extract_receipt(repo, "cpu-smoke", path, expected_hash, expected_commit, authority)


def test_collect_and_boundary_pin_without_historical_files(tmp_path):
    repo = tmp_path / "repo"
    commit = init_repo(repo)
    payload = safe.encoded(synthetic_receipt(repo, commit))
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(payload)
    arguments = [("cpu-smoke", receipt, safe.sha(payload), commit)]
    authority = cpu_authority(repo, receipt)
    result = site.collect(repo, commit, sample()["modified_at"], arguments, authority=authority)
    record = safe.parse(result, "input")
    assert record["evidence"]["cpu-smoke"]["relation"] == "same-content-source"
    site.check_boundary(record, repo)
    record["boundary_sha256"]["LICENSE"] = "d" * 64
    with pytest.raises(safe.Rejected, match="boundary_source_mismatch"):
        site.check_boundary(record, repo)
    with pytest.raises(safe.Rejected, match="duplicate_evidence"):
        site.collect(repo, commit, sample()["modified_at"], arguments * 2, authority=authority)


def test_external_storage_rejects_local_or_repository_paths(tmp_path):
    with pytest.raises(safe.Rejected, match="external_storage_required"):
        site.external(tmp_path / "evidence.json", tmp_path)
    accepted = site.external(Path(site.DEFAULT_ARTIFACTS) / "input.json", tmp_path)
    assert accepted.is_relative_to(safe.SHARED)


def test_failed_cuda_render_is_red_and_cannot_be_effective_acceptance():
    record = sample()
    item = deepcopy(record["evidence"].pop("cpu-smoke"))
    item.update(tests=5, passed=3, failures=2, skipped=0,
                kind="reported-failed-cuda-owner-checks",
                junit_sha256="f" * 64, pytest_log_sha256="a" * 64)
    record["evidence"]["operator-cuda-failure"] = item
    page = site.render(safe.encoded(record))[site.OUTPUTS[0]].decode()
    assert 'class="evidence failed"' in page and "2 failed" in page
    assert "FAILED; no CUDA acceptance." in page
    assert "No fixed13 evidence is included" in page
    item.update(failures=0, passed=5)
    with pytest.raises(safe.Rejected, match="cuda_failure_not_red"):
        site.render(safe.encoded(record))


@pytest.mark.parametrize("mutation", [
    "none", "pin", "bool-exit", "pass-status", "source-rehash", "junit-count",
    "duplicate-case", "log-summary", "result-count", "source-extra", "target",
])
def test_failed_cuda_extraction_has_actual_frozen_git_and_three_byte_pins(tmp_path, mutation):
    repo = tmp_path / "repo"
    init_repo(repo)
    for name in sorted(site.CUDA_FAILURE_SOURCE_FILES):
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic source for scalar readback controls\n")
    assert subprocess.run(["git", "-C", str(repo), "add", "--",
                           *sorted(site.CUDA_FAILURE_SOURCE_FILES)],
                          capture_output=True, check=False).returncode == 0
    assert subprocess.run(["git", "-C", str(repo), "commit", "-qm",
                           "synthetic failed readback source"],
                          capture_output=True, check=False).returncode == 0
    commit = site.git(repo, "rev-parse", "HEAD").decode().strip()
    counts = {"tests": 53, "failures": 51, "errors": 0, "skipped": 0}
    raw = {
        "verdict": "FAIL", "returncode": 1,
        "source_sha256": {name: value for name, value in site.tree_hashes(repo, commit).items()
                          if name in site.CUDA_FAILURE_SOURCE_FILES},
        "targets": ["tests/test_operator_original_cuda.py::test_cuda_collection_entry"],
        "selection": "test_cuda_collection_entry",
        "finished_at": "2026-10-10T14:30:01+00:00",
        "error": f"ValueError: CUDA tests failed or skipped: {counts}",
    }
    if mutation == "bool-exit":
        raw["returncode"] = True
    elif mutation == "pass-status":
        raw["verdict"] = "PASS"
    elif mutation == "source-rehash":
        raw["source_sha256"]["src/openopt/qp.py"] = "a" * 64
    elif mutation == "source-extra":
        raw["source_sha256"]["unexpected.py"] = "a" * 64
    elif mutation == "result-count":
        raw["error"] = "ValueError: no actual count summary"
    elif mutation == "target":
        raw["targets"] = ["tests/test_rotated_conic_cuda.py"]
    tree = ET.Element("testsuites")
    attributes = {key: str(value) for key, value in counts.items()}
    if mutation == "junit-count":
        attributes["tests"] = "54"
    suite = ET.SubElement(tree, "testsuite", attributes)
    for i, identity in enumerate(site.operator_membership()):
        cls, method = identity.split("::", 1)
        if mutation == "duplicate-case":
            method = "test_cuda_collection_entry[one]"
        case = ET.SubElement(suite, "testcase", classname=cls, name=method)
        if i < 51:
            ET.SubElement(case, "failure")
    xml = ET.tostring(tree)
    log = (b"51 failed, 2 passed in 1.0s\nreturncode=1\n"
           if mutation != "log-summary" else b"partial dots without terminal summary")
    payloads = [safe.encoded(raw), xml, log]
    arguments = []
    for filename, payload in zip(("result.json", "pytest.xml", "pytest.log"), payloads):
        path = tmp_path / filename
        path.write_bytes(payload)
        arguments.extend([path, safe.sha(payload)])
    if mutation == "pin":
        arguments[1] = "e" * 64
    arguments.append(commit)
    if mutation == "none":
        item = site.extract_cuda_failure(repo, arguments)
        assert item["passed"] == 2 and item["failures"] == 51
        assert item["kind"] == "reported-failed-cuda-owner-checks"
        collected = site.collect(repo, commit, sample()["modified_at"], [], arguments)
        assert safe.parse(collected, "input")["evidence"][
            "operator-cuda-failure"]["relation"] == "same-content-source"
    else:
        with pytest.raises(safe.Rejected):
            site.extract_cuda_failure(repo, arguments)


def test_cli_runs_from_foreign_directory_without_solver_import(tmp_path):
    result = subprocess.run([sys.executable, str(Path(site.__file__)), "--help"],
                            cwd=tmp_path, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "never publish" in result.stdout
    assert "torch" not in site.__dict__ and "numpy" not in site.__dict__


@pytest.mark.parametrize("mutation", [
    "none", "input-pin", "owner-pin", "dirty-owner", "output-drift",
    "refuse-second-render",
])
def test_actual_collect_render_check_receipts_and_negative_cli_paths(tmp_path, monkeypatch, mutation):
    from argparse import Namespace

    repo = tmp_path / "repo"
    content_commit = init_repo(repo)
    external = tmp_path / "external"
    external.mkdir()
    monkeypatch.setattr(safe, "SHARED", external)
    original_receipt = external / "original.json"
    payload = safe.encoded(synthetic_receipt(repo, content_commit))
    original_receipt.write_bytes(payload)
    args = Namespace(
        repository=repo, operation="collect", input=external / "input.json",
        input_sha256=None, owner_commit=None, content_source_commit=content_commit,
        modified_at=sample()["modified_at"],
        evidence=[("cpu-smoke", original_receipt, safe.sha(payload), content_commit)],
        authority=cpu_authority(repo, original_receipt),
    )
    collect_artifacts = external / "collect"
    collect_artifacts.mkdir()
    assert site.execute(args, collect_artifacts) == 0
    input_payload = args.input.read_bytes()
    args.operation, args.input_sha256 = "render", safe.sha(input_payload)
    render_artifacts = external / "render"
    render_artifacts.mkdir()
    assert site.execute(args, render_artifacts) == 0
    assert json.loads((render_artifacts / "receipt.json").read_text())[
        "source_consistency"]["extra"] == sorted(site.OUTPUTS)
    for arguments in (
        ["add", "--", *site.OUTPUTS],
        ["commit", "-qm", "synthetic generated outputs"],
    ):
        assert subprocess.run(["git", "-C", str(repo), *arguments],
                              capture_output=True, check=False).returncode == 0
    owner_commit = site.git(repo, "rev-parse", "HEAD").decode().strip()
    args.operation, args.owner_commit = "check", owner_commit
    if mutation == "input-pin":
        args.input_sha256 = "f" * 64
    elif mutation == "owner-pin":
        args.owner_commit = content_commit
    elif mutation in {"dirty-owner", "output-drift"}:
        changed = repo / ("AGENTS.md" if mutation == "dirty-owner" else site.OUTPUTS[0])
        changed.write_bytes(changed.read_bytes() + b"\nchanged")
    elif mutation == "refuse-second-render":
        args.operation = "render"
    check_artifacts = external / "check"
    check_artifacts.mkdir()
    rc = site.execute(args, check_artifacts)
    report = json.loads((check_artifacts / "receipt.json").read_text())
    assert report["exit_code"] == rc
    assert report["publication"] == "NOT_PUBLISHED"
    assert report["qualification"] == site.CLAIMS
    assert report["access"] == "existing-super-only"
    if mutation == "none":
        assert rc == 0 and report["source_before"]["clean"] is True
        assert report["source_before"]["commit"] == owner_commit
        assert report["source_consistency"] == {
            "verdict": "PASS", "changed": [], "missing": [], "extra": [],
        }
    else:
        assert rc == 1 and report["verdict"] == "FAIL" and report["first_cause"]
        assert "verdict=FAIL" in (check_artifacts / "execution.log").read_text()


@pytest.mark.parametrize("replacement", ["commit", "blob", "tree"])
def test_repair_raw_object_graph_ignores_real_replacements(tmp_path, replacement):
    repo = tmp_path / "repo"
    original = init_repo(repo)
    expected = site.tree_hashes(repo, original)
    before = {kind: site.git(repo, "rev-parse", original + suffix).decode().strip()
              for kind, suffix in (("commit", ""), ("blob", ":AGENTS.md"), ("tree", "^{tree}"))}
    (repo / "AGENTS.md").write_bytes(b"Untrusted replacement bytes.\n")
    assert subprocess.run(["git", "-C", str(repo), "add", "--", "AGENTS.md"],
                          capture_output=True).returncode == 0
    assert subprocess.run(["git", "-C", str(repo), "commit", "-qm", "replacement"],
                          capture_output=True).returncode == 0
    after = {kind: site.git(repo, "rev-parse", "HEAD" + suffix).decode().strip()
             for kind, suffix in (("commit", ""), ("blob", ":AGENTS.md"), ("tree", "^{tree}"))}
    assert subprocess.run(["git", "-C", str(repo), "replace", before[replacement], after[replacement]],
                          capture_output=True).returncode == 0
    replaced = subprocess.run(["git", "-C", str(repo), "cat-file", "blob", original + ":AGENTS.md"],
                              capture_output=True)
    assert replaced.returncode == 0 and safe.sha(replaced.stdout) != expected["AGENTS.md"]
    assert site.tree_hashes(repo, original) == expected
    raw = synthetic_receipt(repo, original)
    raw["source"]["sha256"]["AGENTS.md"] = safe.sha(replaced.stdout)
    path = tmp_path / "replaced.json"
    path.write_bytes(safe.encoded(raw))
    authority = cpu_authority(repo, path)
    with pytest.raises(safe.Rejected, match="receipt_source_bytes"):
        site.extract_receipt(repo, "cpu-smoke", path, safe.sha(path.read_bytes()), original, authority)


@pytest.mark.parametrize("mutation", ["none", "missing-authority", "authority-pin", "exit", "junit",
    "runtime", "membership", "bool-junit", "log-failed", "log-skip", "log-exit", "collection",
    "pinned-old-manifest", "coordinated-count"])
def test_repair_cpu_actual_bytes_and_caller_authority(tmp_path, mutation):
    repo = tmp_path / "repo"
    commit = init_repo(repo)
    raw = synthetic_receipt(repo, commit)
    if mutation == "exit":
        raw["exit_code"] = 1
    elif mutation in {"junit", "bool-junit"}:
        raw["junit"] = {"tests": 12, "failures": 0, "errors": False if mutation == "bool-junit" else 1, "skipped": 2}
    elif mutation == "runtime":
        raw["runtime"] = {"environment": {"CUDA_VISIBLE_DEVICES": "0"}}
    elif mutation == "membership":
        raw["exact_membership"] = False
    elif mutation == "coordinated-count":
        raw["tests"].update(tests=999)
    receipt = tmp_path / "original.json"
    receipt.write_bytes(safe.encoded(raw))
    authority = cpu_authority(repo, receipt)
    manifest = json.loads(authority[0].read_bytes())
    entry = manifest["evidence"]["cpu-smoke"]
    if mutation in {"log-failed", "log-skip", "log-exit", "collection", "pinned-old-manifest"}:
        role = "collection" if mutation == "collection" else "pytest_log"
        target = Path(entry["files"][role]["path"])
        payload = {"log-failed": b"10 passed, 1 failed, 2 skipped in 1.0s\nreturncode=0\n",
                   "log-skip": b"10 passed in 1.0s\nreturncode=0\n",
                   "log-exit": b"10 passed, 2 skipped in 1.0s\nreturncode=2\n",
                   "collection": b"tests/unrelated.py::test_one\n",
                   "pinned-old-manifest": b"12 passed in 1.0s\nreturncode=0\n"}[mutation]
        target.write_bytes(payload)
        if mutation != "pinned-old-manifest":
            entry["files"][role].update(sha256=safe.sha(payload), bytes=len(payload))
            authority[0].write_bytes(safe.encoded(manifest))
            authority = (authority[0], safe.sha(authority[0].read_bytes()))
    if mutation == "authority-pin":
        authority = (authority[0], "f" * 64)
    if mutation == "missing-authority":
        authority = None
    if mutation == "none":
        result = site.extract_receipt(repo, "cpu-smoke", receipt, safe.sha(receipt.read_bytes()), commit, authority)
        assert result["tests"] == 12 and result["passed"] == 10
        assert result["authority_sha256"] == authority[1]
    else:
        with pytest.raises(safe.Rejected):
            site.extract_receipt(repo, "cpu-smoke", receipt, safe.sha(receipt.read_bytes()), commit, authority)


@pytest.mark.parametrize("spelling", ["iso", "space"])
@pytest.mark.parametrize("prefix", ["2001:db8:", "https://", "//network:", "2001%3Adb8%3A"])
def test_repair_timestamp_network_adjacent_token_rejected(spelling, prefix):
    record = sample()
    value = record["modified_at"] if spelling == "iso" else "2026-10-10 23:20:10 +08:00"
    with pytest.raises(safe.Rejected, match="network_entry"):
        site.scan_content((prefix + value).encode(), record, "development")


@pytest.mark.parametrize("mutation", ["none", "renamed", "classname", "exit", "conflicting-status",
    "ambiguous-outcome", "error", "skip", "bool-count", "log-extra-fail", "log-exit"])
def test_repair_original53_complete_identity_and_outcome(tmp_path, mutation):
    repo = tmp_path / "repo"
    init_repo(repo)
    for name in site.CUDA_FAILURE_SOURCE_FILES:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"# nonexecuted synthetic identity fixture\n")
    assert subprocess.run(["git", "-C", str(repo), "add", "--", *sorted(site.CUDA_FAILURE_SOURCE_FILES)],
                          capture_output=True).returncode == 0
    assert subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], capture_output=True).returncode == 0
    commit = site.git(repo, "rev-parse", "HEAD").decode().strip()
    counts = {"tests": 53, "failures": 9, "errors": 0, "skipped": 0}
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", {key: str(value) for key, value in counts.items()})
    for i, node in enumerate(site.operator_membership()):
        cls, name = node.split("::", 1)
        case = ET.SubElement(suite, "testcase", classname=cls, name=name)
        if i < 9:
            ET.SubElement(case, "failure")
    raw = {"verdict": "FAIL", "returncode": 1, "source_sha256": {
        name: pin for name, pin in site.tree_hashes(repo, commit).items() if name in site.CUDA_FAILURE_SOURCE_FILES},
        "targets": ["tests/test_operator_original_cuda.py::test_cuda_collection_entry"],
        "selection": "test_cuda_collection_entry", "finished_at": "2026-10-10T14:30:01+00:00",
        "error": f"ValueError: CUDA tests failed or skipped: {counts}"}
    cases = list(xml.iter("testcase"))
    if mutation == "renamed":
        for i, case in enumerate(cases):
            case.set("name", f"test_cuda_collection_entry[invented-{i}]")
    elif mutation == "classname":
        for case in cases:
            case.set("classname", "unrelated.module")
    elif mutation == "exit":
        raw["returncode"] = 2
    elif mutation in {"conflicting-status", "bool-count"}:
        raw["tests"] = dict(counts)
        raw["tests"]["errors" if mutation == "bool-count" else "failures"] = False if mutation == "bool-count" else 0
    elif mutation == "ambiguous-outcome":
        ET.SubElement(cases[0], "skipped")
    elif mutation in {"error", "skip"}:
        tag, field = ("error", "errors") if mutation == "error" else ("skipped", "skipped")
        ET.SubElement(cases[-1], tag)
        counts[field] = 1
        suite.set(field, "1")
        raw["error"] = f"ValueError: CUDA tests failed or skipped: {counts}"
    log = b"9 failed, 44 passed in 1.0s\nreturncode=1\n"
    if mutation == "log-extra-fail":
        log = b"9 failed, 44 passed, 1 error in 1.0s\nreturncode=1\n"
    elif mutation == "log-exit":
        log = log.replace(b"returncode=1", b"returncode=2")
    args = []
    for name, payload in (("result.json", safe.encoded(raw)), ("pytest.xml", ET.tostring(xml)), ("pytest.log", log)):
        path = tmp_path / name
        path.write_bytes(payload)
        args.extend((path, safe.sha(payload)))
    args.append(commit)
    if mutation == "none":
        result = site.extract_cuda_failure(repo, args)
        assert result["passed"] == 44 and result["failures"] == 9
    else:
        with pytest.raises(safe.Rejected):
            site.extract_cuda_failure(repo, args)


@pytest.mark.parametrize("mutation", ["none", "function", "method", "policy", "disk", "foreign-repo"])
def test_repair_loaded_renderer_source_binding(tmp_path, monkeypatch, mutation):
    repo = tmp_path / "repo"
    init_repo(repo)
    if mutation == "function":
        monkeypatch.setattr(site.render, "__code__", site.normalized_time.__code__)
    elif mutation == "method":
        code = safe.Rejected.__init__.__code__
        monkeypatch.setattr(safe.Rejected.__init__, "__code__", code.replace(co_consts=code.co_consts + ("drift",)))
    elif mutation == "policy":
        monkeypatch.setattr(site, "CLAIMS", dict(site.CLAIMS, commercial="PASS"))
    elif mutation in {"disk", "foreign-repo"}:
        (repo / site.RUNTIME_PATHS[0]).write_bytes(b"# unrelated nonexecuted owner bytes\n")
    if mutation == "none":
        assert set(site.loaded_source(repo)) == set(site.RUNTIME_PATHS)
    else:
        with pytest.raises(safe.Rejected):
            site.loaded_source(repo)


@pytest.mark.parametrize("mutation", ["none", "missing", "bad-pin", "duplicate", "tamper"])
def test_repair_explicit_generated_refresh_preserves_create_once(tmp_path, mutation):
    files = site.render(safe.encoded(sample()))
    site.write_outputs(tmp_path, files)
    old = [(name, safe.sha(payload)) for name, payload in files.items()]
    updated = sample()
    updated["evidence"]["cpu-smoke"].update(tests=13, passed=11)
    new = site.render(safe.encoded(updated))
    if mutation == "missing":
        old = None
    elif mutation == "bad-pin":
        old[0] = (old[0][0], "a" * 64)
    elif mutation == "duplicate":
        old[1] = old[0]
    elif mutation == "tamper":
        (tmp_path / site.OUTPUTS[0]).write_bytes(b"untrusted old page\n")
    if mutation == "none":
        site.refresh_outputs(tmp_path, new, old)
        site.check_outputs(tmp_path, new)
        with pytest.raises(safe.Rejected, match="output_already_exists"):
            site.write_outputs(tmp_path, files)
    else:
        before = {name: (tmp_path / name).read_bytes() for name in site.OUTPUTS}
        with pytest.raises(safe.Rejected):
            site.refresh_outputs(tmp_path, new, old)
        assert before == {name: (tmp_path / name).read_bytes() for name in site.OUTPUTS}


def read_owner_receipt(path, pin, owner, root):
    """Read-only exact byte/membership reader. Does not attest execution."""
    site.digest(pin, "owner_receipt")
    site.digest(owner, "owner", commit=True)
    path = site.external(path, root)
    payload = safe.read(path, "owner_receipt").payload
    site.require(safe.sha(payload) == pin, "owner_receipt_pin", "reader")
    report = safe.parse(payload, "owner_receipt")
    source = site.source_state(root)
    site.require(source["clean"] is True and source["commit"] == owner and
                 report["source"] == report["source_after"] == source,
                 "owner_receipt_source", "reader")
    site.require(report["owned"] == list(site.OWNED) and report["owned_sha256"] == {
        name: source["sha256"][name] for name in site.OWNED}, "owner_receipt_owned", "reader")
    site.require(report["qualification"] == site.CLAIMS and report["publication"] == "NOT_PUBLISHED",
                 "owner_receipt_ceiling", "reader")
    site.require(report["pytest_exit_code"] == 0 and type(report["pytest_exit_code"]) is int,
                 "owner_receipt_exit", "reader")
    files = {}
    for name in ("pytest.xml", "pytest.log", "collection.log"):
        data = safe.read(path.parent / name, "owner_file").payload
        site.require(report["artifact_sha256"][name] == safe.sha(data), "owner_file_pin", "reader")
        files[name] = data
    counts, nodes = site.junit_record(files["pytest.xml"], "owner_junit")
    members = [node.split("::", 1)[1] for node in nodes]
    collection = [line.split("::", 1)[1] for line in files["collection.log"].decode().splitlines()
                  if line.startswith("tests/test_render_project_site.py::")]
    site.require(all(node.startswith("tests.test_render_project_site::") for node in nodes)
                 and members == collection == report["membership"] == report["collection_membership"]
                 and report["exact_membership"] is True and report["counts"] == counts,
                 "owner_reader_membership", "reader")
    site.require(counts["failures"] == counts["errors"] == counts["skipped"] == 0,
                 "owner_reader_not_all_pass", "reader")
    log = files["pytest.log"].decode()
    summaries = [line.strip("= ") for line in log.splitlines() if site.re.fullmatch(
        r"[0-9]+ passed in [0-9]+(?:\.[0-9]+)?s(?: \([^\n]+\))?", line.strip("= "))]
    site.require(len(summaries) == 1 and summaries[0].startswith(str(counts["tests"]) + " passed in "),
                 "owner_reader_summary", "reader")
    env = report["runtime"]["environment"]
    site.require(env["CUDA_VISIBLE_DEVICES"] == "-1" and all(env[key] == "1" for key in (
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD")), "owner_reader_runtime", "reader")
    site.require(report["source_consistency"] == {"verdict": "PASS", "same_complete_source": True}
                 and report["verdict"] == "PASS" and type(report["exit_code"]) is int
                 and report["exit_code"] == 0, "owner_reader_not_pass", "reader")
    site.normalized_time(report["finished_at"])
    return {"integrity": "PASS", "acceptance": "PASS", "tests": counts["tests"],
            "execution": "UNVERIFIED", "receipt_sha256": pin, "owner_commit": owner}


@pytest.mark.parametrize("mutation", ["none", "pin", "source", "log", "collection", "junit", "membership", "runtime"])
def test_owner_reader_is_read_only_and_rejects_rehashed_conflicts(tmp_path, mutation):
    repo = tmp_path / "repo"
    init_repo(repo)
    for name in site.OWNED:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((site.ROOT / name).read_bytes())
    assert subprocess.run(["git", "-C", str(repo), "add", "--", *site.OWNED], capture_output=True).returncode == 0
    assert subprocess.run(["git", "-C", str(repo), "commit", "-qm", "synthetic reader scope"], capture_output=True).returncode == 0
    source = site.source_state(repo)
    owner = source["commit"]
    counts = {"tests": 2, "failures": 0, "errors": 0, "skipped": 0}
    members = ["test_one", "test_two"]
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", {key: str(value) for key, value in counts.items()})
    for node in members:
        ET.SubElement(suite, "testcase", classname="tests.test_render_project_site", name=node)
    out = tmp_path / "synthetic-reader"
    out.mkdir()
    payloads = {"pytest.xml": ET.tostring(xml), "pytest.log": b"2 passed in 1.0s\n",
                "collection.log": b"tests/test_render_project_site.py::test_one\ntests/test_render_project_site.py::test_two\n"}
    for name, payload in payloads.items():
        (out / name).write_bytes(payload)
    env = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                                  "PYTHONDONTWRITEBYTECODE", "PYTEST_DISABLE_PLUGIN_AUTOLOAD")}
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    report = {"source": source, "source_after": deepcopy(source), "owned": list(site.OWNED),
              "owned_sha256": {name: source["sha256"][name] for name in site.OWNED},
              "qualification": dict(site.CLAIMS), "publication": "NOT_PUBLISHED", "pytest_exit_code": 0,
              "artifact_sha256": {name: safe.sha(payload) for name, payload in payloads.items()},
              "membership": members, "collection_membership": members, "counts": counts,
              "exact_membership": True, "runtime": {"environment": env},
              "source_consistency": {"verdict": "PASS", "same_complete_source": True},
              "verdict": "PASS", "exit_code": 0, "finished_at": "2026-10-10T14:30:01+00:00"}
    if mutation == "source":
        report["source_after"]["commit"] = "a" * 40
    elif mutation == "membership":
        report["membership"] = ["invented", "test_two"]
    elif mutation == "runtime":
        report["runtime"]["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    elif mutation in {"log", "collection", "junit"}:
        role = {"log": "pytest.log", "collection": "collection.log", "junit": "pytest.xml"}[mutation]
        changed = {"log": b"99 passed in 1.0s\n", "collection": b"tests/unrelated.py::invented\n",
                   "junit": ET.tostring(xml).replace(b"test_two", b"unrelated")}[mutation]
        (out / role).write_bytes(changed)
        report["artifact_sha256"][role] = safe.sha(changed)
    receipt = out / "receipt.json"
    receipt.write_bytes(safe.encoded(report))
    pin = "e" * 64 if mutation == "pin" else safe.sha(receipt.read_bytes())
    before = {path.name: path.read_bytes() for path in out.iterdir()}
    if mutation == "none":
        result = read_owner_receipt(receipt, pin, owner, repo)
        assert result["integrity"] == result["acceptance"] == "PASS"
        assert result["execution"] == "UNVERIFIED" and result["tests"] == 2
    else:
        with pytest.raises(safe.Rejected):
            read_owner_receipt(receipt, pin, owner, repo)
    assert before == {path.name: path.read_bytes() for path in out.iterdir()}


def main(argv=None):
    """External owner tests/scan receipt; no default historical evidence input."""
    import argparse
    from datetime import datetime, timezone
    import hashlib
    import importlib.util
    import os
    from types import SimpleNamespace
    from xml.etree import ElementTree

    parser = argparse.ArgumentParser(description=main.__doc__, allow_abbrev=False)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--check-receipt", type=Path)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--owner-commit")
    parser.add_argument("--scan-shapes", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--publisher-check", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    if args.check_receipt:
        try:
            result = read_owner_receipt(args.check_receipt, args.receipt_sha256, args.owner_commit, root)
        except (safe.Rejected, KeyError, OSError, ValueError, TypeError) as error:
            code = error.code if isinstance(error, safe.Rejected) else type(error).__name__
            print(f"PROJECT_SITE_OWNER_READER integrity=FAIL acceptance=BLOCKED first_cause={code}", flush=True)
            return 1
        print("PROJECT_SITE_OWNER_READER " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    if args.artifact_dir is None:
        parser.error("--artifact-dir is required for producing new evidence")
    out = site.external(args.artifact_dir, root)
    out.mkdir(parents=True, exist_ok=False)
    before = site.source_state(root)
    receipt = {
        "schema_version": 1, "source": before, "owned": list(site.OWNED),
        "qualification": dict(site.CLAIMS), "publication": "NOT_PUBLISHED",
        "verdict": "FAIL", "exit_code": 1,
    }
    try:
        if args.scan_shapes:
            import unicodedata
            shape = site.re.compile("IDENTIFIED" + r" BY|" + "pass" + r"word[\x22\x27]?\s*[:=]\s*[\x22\x27][^\x22\x27]{6,}", site.re.I)
            bad = [name for name in site.OWNED if shape.search((root / name).read_text()) or any(
                (ord(c) < 32 and c not in "\n\r\t") or unicodedata.category(c) == "Cf"
                for c in (root / name).read_text())]
            if bad:
                raise ValueError("named source shape/hidden scan failed")
            receipt["scan"] = {"shape_hidden": "PASS", "files": list(site.OWNED),
                               "credential_files_read": False, "literal_coverage": "UNVERIFIED"}
        elif args.scan_only:
            spec = importlib.util.spec_from_file_location(
                "site_existing_scan", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
            scanner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(scanner)
            needles = scanner.read_needles(SimpleNamespace(
                secret_hex=None, secrets_file=None, no_needle=False,
                needle_from=str(Path.home() / ".config/dsh-n600-token")))
            if not needles:
                raise ValueError("missing literal scanner material; no scan qualification")
            hits, _ = scanner.scan_literal([str(root / name) for name in site.OWNED], needles)
            shape = site.re.compile(
                "IDENTIFIED" + r""" BY|password["']?\s*[:=]\s*["'][^"']{6,}""",
                site.re.IGNORECASE)
            shape_paths = [name for name in site.OWNED
                           if shape.search((root / name).read_text())]
            canary = out / "synthetic-sensitivity.txt"
            anchor = "site-sensitivity-" + "not-a-credential"
            canary.write_text(anchor)
            seen, _ = scanner.scan_literal([str(canary)], [anchor])
            canary.write_text("cleared synthetic sensitivity control\n")
            cleared, _ = scanner.scan_literal([str(canary)], [anchor])
            if hits or shape_paths or seen != 1 or cleared != 0:
                raise ValueError("scan or sensitivity failure; values withheld")
            receipt["scan"] = {
                "literal_hits": hits, "shape_hit_paths": shape_paths,
                "sensitivity_seen": seen, "sensitivity_cleared": cleared,
                "values_disclosed": False,
                "tool_sha256": hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),
            }
        else:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="-1", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
                       PYTHONPATH=str(root / "src") + os.pathsep + str(root))
            receipt["runtime"] = {
                "python": sys.version, "executable": sys.executable,
                "environment": {key: env[key] for key in (
                    "CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "PYTHONDONTWRITEBYTECODE",
                    "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTHONPATH")},
            }
            collect_command = [
                sys.executable, "-m", "pytest", "--collect-only", "-q",
                "-o", f"cache_dir={out / 'collect-cache'}", str(Path(__file__)),
            ]
            collected = subprocess.run(collect_command, cwd=root, env=env,
                                       capture_output=True, check=False)
            (out / "collection.log").write_bytes(collected.stdout + collected.stderr)
            expected = [line.split("::", 1)[1] for line in collected.stdout.decode().splitlines()
                        if line.startswith("tests/test_render_project_site.py::")]
            if collected.returncode or not expected or len(expected) != len(set(expected)):
                raise ValueError("empty, duplicate or failed collection")
            command = [
                sys.executable, "-m", "pytest", "-q", "-ra",
                "-o", f"cache_dir={out / 'cache'}", "--basetemp", str(out / "tmp"),
                f"--junitxml={out / 'pytest.xml'}", str(Path(__file__)),
            ]
            process = subprocess.run(command, cwd=root, env=env,
                                     capture_output=True, check=False)
            (out / "pytest.log").write_bytes(process.stdout + process.stderr)
            receipt["command"], receipt["pytest_exit_code"] = command, process.returncode
            xml = ElementTree.parse(out / "pytest.xml").getroot()
            suites = list(xml.iter("testsuite"))
            counts = {key: sum(int(suite.attrib[key]) for suite in suites)
                      for key in ("tests", "failures", "errors", "skipped")}
            membership = [case.attrib["name"] for case in xml.iter("testcase")]
            receipt.update(counts=counts, membership=membership,
                           collection_membership=expected,
                           exact_membership=(membership == expected))
            receipt["artifact_sha256"] = {name: safe.sha((out / name).read_bytes())
                                         for name in ("pytest.xml", "pytest.log", "collection.log")}
            if (process.returncode or counts["failures"] or counts["errors"]
                    or counts["skipped"] or len(membership) != counts["tests"]
                    or membership != expected):
                raise ValueError("owner tests or exact full membership failure")
        if args.publisher_check:
            spec = importlib.util.spec_from_file_location("readonly_publisher",
                                                         args.publisher_check)
            publisher = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(publisher)
            audiences, pages = publisher.read_catalog(
                {"catalog_path": root / "docs.publish.json", "slug": "openopt"}, root)
            if audiences != {"admin": ""} or len(pages) != 1 or pages[0]["slug"] != "overview":
                raise ValueError("existing publisher catalog contract mismatch")
            receipt["readonly_publisher_catalog"] = {
                "verdict": "PASS", "pages": 1, "audiences": audiences,
                "source_sha256": hashlib.sha256(args.publisher_check.read_bytes()).hexdigest(),
                "published": False,
            }
        after = site.source_state(root)
        receipt["source_after"] = after
        if before != after:
            raise ValueError("actual complete source changed during owner validation")
        receipt["owned_sha256"] = {name: after["sha256"].get(name) for name in site.OWNED}
        receipt["missing_generated_outputs"] = [
            name for name in site.OUTPUTS if name not in after["sha256"]]
        receipt["source_consistency"] = {"verdict": "PASS", "same_complete_source": True}
        receipt["verdict"], receipt["exit_code"] = "PASS", 0
    except Exception as error:
        receipt["first_cause"] = (
            {"code": error.code, "coordinate": error.coordinate}
            if isinstance(error, safe.Rejected) else {"type": type(error).__name__}
        )
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    site.create_once(out / "receipt.json", safe.encoded(receipt))
    line = f"PROJECT_SITE_OWNER verdict={receipt['verdict']} scan={args.scan_only}"
    site.create_once(out / "execution.log", (line + "\n").encode())
    print(f"{line} receipt={out / 'receipt.json'}", flush=True)
    if receipt.get("counts"):
        print(f"PROJECT_SITE_OWNER counts={receipt['counts']}", flush=True)
    return receipt["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
