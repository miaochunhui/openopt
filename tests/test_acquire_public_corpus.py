"""Offline acquisition gates; every archive and test artifact is external.

Synthetic transport/format fixtures validate gates, never count as real public
source evidence. The live download/import is a separate explicit acceptance.
"""

import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from scripts import acquire_public_corpus as acquire
from scripts import public_qp_corpus as importer

COMMIT = "a" * 40
LP = """NAME TEST
ROWS
 N OBJ
 L CAP
COLUMNS
 X OBJ -1 CAP 1
RHS
 RHS1 CAP 2
ENDATA
"""


def archive(tmp_path, members):
    path = tmp_path / "fixture.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        for name, raw, kind in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = len(raw) if kind == tarfile.REGTYPE else 0
            if kind == tarfile.SYMTYPE:
                info.linkname = "/external"
            tar.addfile(info, io.BytesIO(raw) if kind == tarfile.REGTYPE else None)
    return path


def test_safe_selective_extract_no_upstream_execution_or_other_files(tmp_path):
    path = archive(tmp_path, [
        (f"HiGHS-{COMMIT}/LICENSE", b"MIT License\n", tarfile.REGTYPE),
        (f"HiGHS-{COMMIT}/README.md", b"Official fixture\n", tarfile.REGTYPE),
        (f"HiGHS-{COMMIT}/build.py", b"do not execute", tarfile.REGTYPE),
        (f"HiGHS-{COMMIT}/check/instances/afiro.mps", LP.encode(), tarfile.REGTYPE),
    ])
    output = tmp_path / "docs"
    result = acquire.extract_selected(path, COMMIT, output, acquire.DOCS,
                                      ("LICENSE", "README.md"))
    assert set(result) == {"LICENSE", "README.md"}
    assert (output / "LICENSE").read_bytes() == b"MIT License\n"
    assert not (output / "build.py").exists()
    assert not (output / "check").exists()
    assert result["LICENSE"]["sha256"] == importer.sha256(b"MIT License\n")


@pytest.mark.parametrize("mutation", ["traversal", "absolute", "wrong-commit", "symlink",
                                     "hardlink", "duplicate", "oversized", "missing"])
def test_archive_negative_controls(tmp_path, mutation, monkeypatch):
    name = f"HiGHS-{COMMIT}/LICENSE"
    kind, raw = tarfile.REGTYPE, b"MIT License\n"
    if mutation == "traversal":
        name = f"HiGHS-{COMMIT}/../escape"
    elif mutation == "absolute":
        name = "/escape"
    elif mutation == "wrong-commit":
        name = "HiGHS-wrong/LICENSE"
    elif mutation == "symlink":
        kind = tarfile.SYMTYPE
    elif mutation == "hardlink":
        kind = tarfile.LNKTYPE
    elif mutation == "oversized":
        monkeypatch.setattr(acquire, "MAX_MEMBER_BYTES", 1)
    elif mutation == "missing":
        name = f"HiGHS-{COMMIT}/not-selected"
    members = [(name, raw, kind)]
    if mutation == "duplicate":
        members.append((name, raw, kind))
    path = archive(tmp_path, members)
    with pytest.raises(ValueError):
        acquire.extract_selected(path, COMMIT, tmp_path / "out",
                                 ("LICENSE",), ("LICENSE",))
    assert not (tmp_path / "escape").exists()


@pytest.mark.parametrize("selected", [(), ("LICENSE", "LICENSE"), ("build.py",)])
def test_empty_or_unapproved_extraction(tmp_path, selected):
    with pytest.raises(ValueError):
        acquire.extract_selected(tmp_path / "unused", COMMIT, tmp_path / "out",
                                 selected, ())


def test_commit_resolution_requires_full_sha_official_identity():
    result = {"sha": COMMIT, "html_url": f"{acquire.REPOSITORY}/commit/{COMMIT}"}
    assert acquire.resolve_commit(result) == COMMIT
    assert acquire.archive_url(COMMIT).endswith(COMMIT)
    for value in [{}, dict(result, sha="main"), dict(result, html_url="https://example.invalid")]:
        with pytest.raises(ValueError):
            acquire.resolve_commit(value)


def marker(path, remote="0.250", local="NA", url=acquire.API, rate=acquire.RATE):
    return (f"ASTRiLL_FETCH ok=1 mode=url out={path} url={url} "
            f"bytes={path.stat().st_size} sha256={acquire.hash_file(path)} "
            f"rate={rate} remote_max_mib_s={remote} local_max_mib_s={local}\n")


def test_transfer_actual_bytes_and_explicit_unmeasured_state(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"actual bytes")
    result = acquire.transfer_markers(marker(path), path, acquire.API)
    assert result["sampler"]["remote_max_mib_s"] == 0.25
    assert result["sampler"]["local_max_mib_s"] is None
    assert result["configured_mib_s"] <= 4
    assert result["sha256"] == acquire.hash_file(path)


def test_rate_ruler_does_not_measure_policy_text_but_sees_actual_wget_average(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"actual bytes")
    policy = "[c14] hard upper 8MB/s\n"
    observed = "[remote] 2026-10-10 17:02:16 (57.3 KB/s) - payload saved [12/12]\n"
    result = acquire.transfer_markers(policy + observed + marker(path), path, acquire.API)
    assert result["wget_progress_mib_s"] == [57.3 / 1024]
    with pytest.raises(ValueError, match="wget observed rate"):
        acquire.transfer_markers(observed.replace("57.3 KB/s", "9.0 MB/s") + marker(path),
                                 path, acquire.API)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "high-rate", "nan-rate",
                                     "wrong-url", "wrong-config", "wrong-hash", "empty"])
def test_transfer_negative_controls(tmp_path, mutation):
    path = tmp_path / "payload"
    path.write_bytes(b"actual bytes")
    text = marker(path)
    if mutation == "missing":
        text = ""
    elif mutation == "duplicate":
        text += text
    elif mutation == "high-rate":
        text = marker(path, remote="5.0")
    elif mutation == "nan-rate":
        text = marker(path, remote="nan")
    elif mutation == "wrong-url":
        text = marker(path, url="https://example.invalid")
    elif mutation == "wrong-config":
        text = marker(path, rate="6m")
    elif mutation == "wrong-hash":
        text = text.replace(acquire.hash_file(path), "0" * 64)
    else:
        path.write_bytes(b"")
        text = marker(path)
    with pytest.raises(ValueError):
        acquire.transfer_markers(text, path, acquire.API)


def test_no_network_automatic_retry_and_rate_explicit(tmp_path, monkeypatch):
    calls = []

    def broken(argv, log):
        calls.append(argv)
        raise RuntimeError("transport-negative-control rc=8")

    monkeypatch.setattr(acquire, "command", broken)
    with pytest.raises(RuntimeError, match="rc=8"):
        acquire.fetch_url(acquire.API, tmp_path / "payload", tmp_path, "fetch")
    assert len(calls) == 1
    assert calls[0][calls[0].index("--tries") + 1] == "1"
    assert calls[0][calls[0].index("--rate") + 1] == "512k"
    assert calls[0][2] == str(acquire.FETCH)
    assert "url" in calls[0] and "git" not in calls[0]


def test_missing_execution_chain_does_not_download_install_or_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "SHARED_TMP", tmp_path.resolve())
    calls = []

    def missing(output):
        calls.append("preflight")
        raise RuntimeError("missing node; no install authorized")

    def never(*args):
        pytest.fail("fetch attempted despite missing chain")

    monkeypatch.setattr(acquire, "preflight", missing)
    monkeypatch.setattr(acquire, "fetch_url", never)
    output = tmp_path / "review"
    with pytest.raises(RuntimeError, match="missing node"):
        acquire.source_review(output)
    report = importer.read_json(output / "source.json")
    assert report["verdict"] == "ERROR" and "missing node" in report["error"]["message"]
    assert calls == ["preflight"] and (output / "failure.log").is_file()


def test_review_pin_acknowledgement_and_archive_hash_gates(tmp_path):
    with pytest.raises(ValueError, match="acknowledgement"):
        acquire.freeze_reviewed(tmp_path, "0" * 64, tmp_path / "out", False)
    source = tmp_path / "source"
    source.mkdir()
    (source / "source.json").write_text("{}")
    with pytest.raises(ValueError, match="caller pin"):
        acquire.freeze_reviewed(source, "0" * 64, tmp_path / "out", True)


def test_independent_dimensions_offset_bounds_decoys(tmp_path):
    audited = importer.audit_records(LP.encode())
    audited["model_sha256"] = importer.model_hash(audited)
    summary = acquire.independent_summary(audited, LP.encode(), "format-fixture")
    assert summary["variables"] == summary["constraint_rows"] == 1
    assert summary["offset"] == 0 and not summary["license_verified"]
    assert not summary["public_redistribution"]
    for field in ("offset", "variable_lower", "P"):
        changed = {**audited, "arrays": {key: value.copy() for key, value in audited["arrays"].items()}}
        if field == "offset":
            changed["arrays"][field] = np.asarray(1.0)
        else:
            changed["arrays"][field].flat[0] = 1
        with pytest.raises(ValueError):
            acquire.independent_summary(changed, LP.encode(), "format-fixture")


def test_benchmark_declaration_never_promotes_mit_to_instance_license():
    assert "NOT VERIFIED" in acquire.LICENSE_DECLARATION
    assert "no public redistribution" in acquire.LICENSE_DECLARATION
    assert "not asserted" in acquire.LICENSE_DECLARATION


def test_known_netlib_name_size_annotation_preserves_bytes_and_exact_reader_semantics(tmp_path):
    raw = LP.replace("NAME TEST", "NAME TEST SIZE: N=1, M=2, NZ=2").encode()
    snapshot = tmp_path / "annotation.mps"
    snapshot.write_bytes(raw)
    model = importer.read_model(snapshot, tmp_path / "reader.log")
    assert snapshot.read_bytes() == raw
    assert model["model_name"] == "TEST"
    assert model["reader"]["raw_record_semantics"] == "exactly matched"
    assert model["arrays"]["q"][0] == -1
    assert model["arrays"]["A"][0, 0] == 1
    assert model["arrays"]["offset"] == 0
    with pytest.raises(ValueError, match="NAME requires"):
        importer.audit_records(raw.replace(b"SIZE:", b"UNSUPPORTED:"))


def test_reviewed_archive_to_real_reader_freeze_pipeline_without_network(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "SHARED_TMP", tmp_path.resolve())
    source = tmp_path / "source"
    source.mkdir()
    packed = archive(source, [
        (f"HiGHS-{COMMIT}/check/instances/afiro.mps", LP.encode(), tarfile.REGTYPE),
        (f"HiGHS-{COMMIT}/check/instances/adlittle.mps",
         LP.replace("OBJ -1", "OBJ -2").encode(), tarfile.REGTYPE),
    ])
    packed.rename(source / "official.tar.gz")
    review = {"verdict": "REVIEW_REQUIRED", "license_verified": False,
              "public_redistribution": False, "resolved_commit": COMMIT,
              "archive_transfer": {"sha256": acquire.hash_file(source / "official.tar.gz")}}
    importer.write_json(source / "source.json", review)
    pin = acquire.hash_file(source / "source.json")
    output = tmp_path / "frozen"
    result = acquire.freeze_reviewed(source, pin, output, True)
    assert result["case_count"] == 2
    assert result["verdict"] == "ACQUIRED_FROZEN_NOT_SOLVER_QUALIFIED"
    assert result["replay"]["verdict"] == "VERIFIED"
    assert result["replay"]["public_qualified_count"] == 0
    assert not result["license_verified"] and not result["public_redistribution"]
    assert all(row["independent_dimensions_bounds_offset"] == "PASS" for row in result["cases"])
    assert result == importer.read_json(output / "acquisition.json")
    json.dumps(result, allow_nan=False)
    with pytest.raises(FileExistsError):
        acquire.freeze_reviewed(source, pin, output, True)
    raw = (source / "official.tar.gz").read_bytes()
    (source / "official.tar.gz").write_bytes(raw + b"x")
    with pytest.raises(ValueError, match="archive byte hash"):
        acquire.freeze_reviewed(source, pin, tmp_path / "bad", True)
