"""Function-only offline trust-boundary fixtures, not public corpus coverage."""

import io
import json
from pathlib import Path
import tarfile

import numpy as np
import pytest

from scripts import accept_public_source as gate
from scripts import acquire_public_corpus as acquire
from scripts import public_qp_corpus as importer

COMMIT = "a" * 40
LP = b"""NAME FUNCTION_ONLY
ROWS
 N COST
 L CAP
COLUMNS
 X COST -1 CAP 1
RHS
 RHS1 CAP 2
ENDATA
"""


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def repin(bundle):
    acquisition = importer.read_json(bundle["acquisition"])
    source_pin = gate.sha(bundle["source"].read_bytes())
    acquisition["source_manifest_sha256"] = source_pin
    acquisition["corpus_manifest_sha256"] = gate.sha((bundle["corpus"] / "corpus.json").read_bytes())
    acquisition["replay"]["corpus_sha256"] = acquisition["corpus_manifest_sha256"]
    save_json(bundle["acquisition"], acquisition)
    return source_pin


def update_frozen_pin(bundle, metadata=None):
    corpus_path = bundle["corpus"] / "corpus.json"
    frozen = importer.read_json(corpus_path)
    if metadata is not None:
        metadata_path = bundle["corpus"] / "afiro/metadata.json"
        save_json(metadata_path, metadata)
        frozen["cases"][0]["metadata_sha256"] = gate.sha(metadata_path.read_bytes())
        frozen["cases"][0]["model_sha256"] = metadata["model_sha256"]
    frozen["input_manifest_sha256"] = gate.sha((bundle["corpus"] / "input-manifest.json").read_bytes())
    save_json(corpus_path, frozen)
    repin(bundle)


def check(bundle):
    return gate.build_evidence(
        bundle["acquisition"], gate.sha(bundle["acquisition"].read_bytes()),
        bundle["source"], gate.sha(bundle["source"].read_bytes()),
        bundle["corpus"], gate.sha((bundle["corpus"] / "corpus.json").read_bytes()),
    )


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    monkeypatch.setattr(importer, "SHARED_TMP", tmp_path.resolve())
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    review_dir = source_dir / "review"
    review_dir.mkdir()
    docs = {"LICENSE.txt": b"MIT License\nSoftware only. Function-only fixture.\n",
            "README.md": b"Function-only archive fixture, not public provenance.\n"}
    for name, raw in docs.items():
        (review_dir / name).write_bytes(raw)
    archive = source_dir / "official.tar.gz"
    with tarfile.open(archive, "w:gz") as packed:
        for name, raw in {**docs, **{name: LP for name in gate.MEMBERS.values()}}.items():
            member = tarfile.TarInfo(f"HiGHS-{COMMIT}/{name}")
            member.size = len(raw)
            packed.addfile(member, io.BytesIO(raw))
    api = {"sha": COMMIT, "html_url": f"{gate.REPOSITORY}/commit/{COMMIT}",
           "url": f"{gate.API_ROOT}/commits/{COMMIT}",
           "commit": {"url": f"{gate.API_ROOT}/git/commits/{COMMIT}"}}
    save_json(source_dir / "commit.json", api)
    source = {
        "schema_version": 1, "version": "public-acquisition-v1", "verdict": "REVIEW_REQUIRED",
        "repository": gate.REPOSITORY, "requested_ref": gate.REF, "resolved_commit": COMMIT,
        "license_status": "benchmark-use-declaration", "license_verified": False,
        "public_redistribution": False,
        "source_hashes": {"scripts/acquire_public_corpus.py": acquire.hash_file(Path(acquire.__file__))},
        "documents": {name: gate.record(raw) for name, raw in docs.items()},
        "commit_transfer": {**gate.record((source_dir / "commit.json").read_bytes()),
                            "url": f"{gate.API_ROOT}/commits/{gate.REF}"},
        "archive_transfer": {**gate.record(archive.read_bytes()),
                             "url": f"https://codeload.github.com/ERGO-Code/HiGHS/tar.gz/{COMMIT}"},
    }
    save_json(source_dir / "source.json", source)
    acquisition_dir = tmp_path / "acquisition"
    acquire.freeze_reviewed(source_dir, acquire.hash_file(source_dir / "source.json"),
                            acquisition_dir, True)
    return {"source": source_dir / "source.json", "acquisition": acquisition_dir / "acquisition.json",
            "corpus": acquisition_dir / "frozen", "output": tmp_path / "accepted"}


def test_source_checks_and_schema_json_roundtrip_without_network(bundle, monkeypatch):
    def no_download(*args):
        pytest.fail("offline source acceptance attempted network")

    monkeypatch.setattr(acquire, "fetch_url", no_download)
    evidence = check(bundle)
    assert evidence["verdict"] == "SOURCE_EVIDENCE_VERIFIED"
    assert [row["case_id"] for row in evidence["cases"]] == list(gate.MEMBERS)
    assert evidence["documents"]["LICENSE.txt"]["sha256"] == gate.sha(
        (bundle["source"].parent / "review/LICENSE.txt").read_bytes())
    result = gate.produce(
        bundle["acquisition"], gate.sha(bundle["acquisition"].read_bytes()),
        bundle["source"], gate.sha(bundle["source"].read_bytes()),
        bundle["corpus"], gate.sha((bundle["corpus"] / "corpus.json").read_bytes()), bundle["output"],
    )
    accepted = importer.read_json(bundle["output"] / "acceptance.json")
    assert set(accepted) == {"schema_version", "verdict", "corpus_sha256", "cases"}
    assert accepted["verdict"] == result["verdict"] == "PUBLIC_SOURCE_ACCEPTED"
    assert all(set(row) == gate.CASE_FIELDS for row in accepted["cases"])
    assert evidence["license_verified"] is False and evidence["public_redistribution"] is False
    assert all("no legal license verified" in row["source_review"]
               and "no public redistribution" in row["source_review"] for row in accepted["cases"])
    assert all(row["evidence_sha256"] == gate.sha((bundle["output"] / "evidence.json").read_bytes())
               for row in accepted["cases"])
    assert importer.read_json(bundle["output"] / "status.json") == result
    assert gate.verify_acceptance(bundle["output"] / "acceptance.json",
                                  result["acceptance_sha256"])["case_count"] == 2
    assert "PUBLIC_SOURCE_ACCEPTED" in (bundle["output"] / "acceptance.log").read_text()
    with pytest.raises(FileExistsError):
        gate.produce(bundle["acquisition"], "0" * 64, bundle["source"], "0" * 64,
                     bundle["corpus"], "0" * 64, bundle["output"])


@pytest.mark.parametrize("target", ["acquisition", "source", "corpus"])
def test_each_caller_pin_is_mandatory(bundle, target):
    args = [bundle["acquisition"], gate.sha(bundle["acquisition"].read_bytes()),
            bundle["source"], gate.sha(bundle["source"].read_bytes()),
            bundle["corpus"], gate.sha((bundle["corpus"] / "corpus.json").read_bytes())]
    args[{"acquisition": 1, "source": 3, "corpus": 5}[target]] = "0" * 64
    with pytest.raises(ValueError, match="caller pin mismatch"):
        gate.build_evidence(*args)


@pytest.mark.parametrize("mutation", [
    "archive-bytes", "document-bytes", "commit-api", "raw-local", "raw-frozen", "npz",
])
def test_pinned_original_byte_tamper_negative_controls(bundle, mutation):
    paths = {
        "archive-bytes": bundle["source"].parent / "official.tar.gz",
        "document-bytes": bundle["source"].parent / "review/LICENSE.txt",
        "commit-api": bundle["source"].parent / "commit.json",
        "raw-local": bundle["acquisition"].parent / "local/check/instances/afiro.mps",
        "raw-frozen": bundle["corpus"] / "afiro/raw.mps",
        "npz": bundle["corpus"] / "afiro/arrays.npz",
    }
    path = paths[mutation]
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError):
        check(bundle)


@pytest.mark.parametrize("mutation", [
    "repo", "archive-url", "api-url", "short-commit", "documents-membership",
    "source-rights", "acquisition-rights", "case-rights", "model-hash",
    "case-membership", "summary", "importer-source", "cross-pin",
])
def test_rehashed_claim_tamper_still_rejected(bundle, mutation):
    source = importer.read_json(bundle["source"])
    acquisition = importer.read_json(bundle["acquisition"])
    if mutation == "repo":
        source["repository"] = "https://github.com/untrusted/HiGHS"
    elif mutation == "archive-url":
        source["archive_transfer"]["url"] += "?untrusted"
    elif mutation == "api-url":
        source["commit_transfer"]["url"] = "https://example.invalid/commit"
    elif mutation == "short-commit":
        source["resolved_commit"] = COMMIT[:7]
    elif mutation == "documents-membership":
        del source["documents"]["LICENSE.txt"]
    elif mutation == "source-rights":
        source["license_verified"] = True
    elif mutation == "acquisition-rights":
        acquisition["public_redistribution"] = True
    elif mutation == "case-rights":
        acquisition["cases"][0]["license_verified"] = True
    elif mutation == "model-hash":
        acquisition["cases"][0]["model_sha256"] = "0" * 64
    elif mutation == "case-membership":
        acquisition["cases"].reverse()
    elif mutation == "summary":
        acquisition["cases"][0]["variables"] += 1
    elif mutation == "importer-source":
        acquisition["source_hashes"]["scripts/public_qp_corpus.py"] = "0" * 64
    else:
        acquisition["corpus_manifest_sha256"] = "0" * 64
    save_json(bundle["source"], source)
    acquisition["source_manifest_sha256"] = gate.sha(bundle["source"].read_bytes())
    save_json(bundle["acquisition"], acquisition)
    with pytest.raises(ValueError):
        check(bundle)


def test_archive_scan_rejects_duplicates_traversal_links_and_missing():
    for mutation in ("duplicate", "traversal", "link", "missing"):
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as packed:
            member = tarfile.TarInfo(f"HiGHS-{COMMIT}/LICENSE.txt")
            if mutation == "traversal":
                member.name = f"HiGHS-{COMMIT}/../escape"
            if mutation == "link":
                member.type = tarfile.SYMTYPE
                member.linkname = "/external"
            else:
                member.size = 1
            packed.addfile(member, io.BytesIO(b"x") if mutation != "link" else None)
            if mutation == "duplicate":
                packed.addfile(member, io.BytesIO(b"x"))
        with pytest.raises(ValueError):
            gate.archive_members(buffer.getvalue(), COMMIT)


def test_rehashed_case_url_cannot_override_official_archive_membership(bundle):
    declared = importer.read_json(bundle["acquisition"].parent / "input.json")
    declared["cases"][0]["source_url"] = "https://github.com/ERGO-Code/HiGHS/main/fake.mps"
    save_json(bundle["acquisition"].parent / "input.json", declared)
    save_json(bundle["corpus"] / "input-manifest.json", declared)
    metadata = importer.read_json(bundle["corpus"] / "afiro/metadata.json")
    metadata["provenance"] = declared["cases"][0]
    update_frozen_pin(bundle, metadata)
    with pytest.raises(ValueError, match="official URL/path/origin"):
        check(bundle)


def test_rehashed_model_array_decoy_cannot_break_original_raw_semantics(bundle):
    path = bundle["corpus"] / "afiro/arrays.npz"
    with np.load(path, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in importer.ARRAY_NAMES}
    arrays["q"][0] -= 1
    np.savez(path, **arrays)
    metadata = importer.read_json(bundle["corpus"] / "afiro/metadata.json")
    metadata["array_hashes"] = importer.array_hashes(arrays)
    metadata["npz_sha256"] = gate.sha(path.read_bytes())
    metadata["model_sha256"] = importer.model_hash({**metadata, "arrays": arrays})
    acquisition = importer.read_json(bundle["acquisition"])
    acquisition["cases"][0]["model_sha256"] = metadata["model_sha256"]
    save_json(bundle["acquisition"], acquisition)
    update_frozen_pin(bundle, metadata)
    with pytest.raises(ValueError, match="frozen/raw semantics"):
        check(bundle)


def test_rehashed_commit_api_false_identity_is_not_a_trusted_url(bundle):
    path = bundle["source"].parent / "commit.json"
    api = importer.read_json(path)
    api["html_url"] = "https://github.com/untrusted/HiGHS/commit/" + COMMIT
    save_json(path, api)
    source = importer.read_json(bundle["source"])
    source["commit_transfer"].update(gate.record(path.read_bytes()))
    save_json(bundle["source"], source)
    repin(bundle)
    with pytest.raises(ValueError, match="API identity"):
        check(bundle)


def test_second_pass_failure_never_publishes_final_acceptance(bundle, monkeypatch):
    original = gate.build_evidence
    calls = []

    def changed(*args):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("changed between source passes")
        return original(*args)

    monkeypatch.setattr(gate, "build_evidence", changed)
    with pytest.raises(ValueError, match="between source passes"):
        gate.produce(
            bundle["acquisition"], gate.sha(bundle["acquisition"].read_bytes()),
            bundle["source"], gate.sha(bundle["source"].read_bytes()), bundle["corpus"],
            gate.sha((bundle["corpus"] / "corpus.json").read_bytes()), bundle["output"],
        )
    assert len(calls) == 2
    assert not (bundle["output"] / "acceptance.json").exists()
    assert importer.read_json(bundle["output"] / "status.json")["verdict"] == "ERROR"


def test_failure_retains_first_cause_without_acceptance(bundle):
    with pytest.raises(ValueError, match="caller pin"):
        gate.produce(bundle["acquisition"], "0" * 64, bundle["source"],
                     gate.sha(bundle["source"].read_bytes()), bundle["corpus"],
                     gate.sha((bundle["corpus"] / "corpus.json").read_bytes()), bundle["output"])
    status = importer.read_json(bundle["output"] / "status.json")
    assert status["verdict"] == "ERROR" and status["exit_code"] == 2
    assert "caller pin" in status["error"]["message"]
    assert (bundle["output"] / "failure.log").is_file()
    assert not (bundle["output"] / "acceptance.json").exists()


@pytest.mark.parametrize("mutation", ["rights", "evidence", "url", "reader-log"])
def test_acceptance_replay_rejects_repinned_output_and_changed_sidecar(bundle, mutation):
    result = gate.produce(
        bundle["acquisition"], gate.sha(bundle["acquisition"].read_bytes()),
        bundle["source"], gate.sha(bundle["source"].read_bytes()), bundle["corpus"],
        gate.sha((bundle["corpus"] / "corpus.json").read_bytes()), bundle["output"],
    )
    path = bundle["output"] / "acceptance.json"
    accepted = importer.read_json(path)
    if mutation == "rights":
        accepted["license_verified"] = True
    elif mutation == "evidence":
        accepted["cases"][0]["evidence_sha256"] = "0" * 64
    elif mutation == "url":
        accepted["cases"][0]["source_url"] += "/false"
    else:
        log = bundle["corpus"] / "afiro/reader.log"
        log.write_bytes(log.read_bytes() + b"altered")
    save_json(path, accepted)
    with pytest.raises(ValueError):
        gate.verify_acceptance(path, gate.sha(path.read_bytes()))
    assert result["verdict"] == "PUBLIC_SOURCE_ACCEPTED"


def test_duplicate_nonfinite_json_and_symlink_are_refused(tmp_path):
    for raw in (b'{"a":1,"a":2}', b'{"a":NaN}', b"[]"):
        with pytest.raises(ValueError):
            gate.json_object(raw)
    outside = tmp_path / "outside"
    outside.write_bytes(b"x")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        gate.file_bytes(root, "link")
