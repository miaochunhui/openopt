"""Offline source acceptance for the caller-pinned HiGHS afiro/adlittle bundle.

No network, extraction, solve, installation or legal permission is performed.
The three externally trusted pins are the authenticity boundary, not labels in
a JSON document. Archive members, review documents and original frozen models
are checked against that boundary. All output and failure logs are external.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import traceback

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import public_qp_corpus as importer

REPOSITORY = "https://github.com/ERGO-Code/HiGHS"
API_ROOT = "https://api.github.com/repos/ERGO-Code/HiGHS"
REF = "v1.15.1"
MEMBERS = {"afiro": "check/instances/afiro.mps",
           "adlittle": "check/instances/adlittle.mps"}
DOCS = {"LICENSE", "LICENSE.txt", "README.md", "check/README.md",
        "check/instances/README.md", "check/instances/LICENSE",
        "check/instances/LICENSE.txt"}
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
MAX_SCAN_BYTES = 256 * 1024 * 1024
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_MEMBERS = 20000
REVIEW_SCOPE = (
    "Caller-pinned official distribution bytes, selected documents and original "
    "model membership only; no legal license verified; no public redistribution "
    "of raw or derived arrays; no independent TLS/PGP authentication, solver "
    "or industrial qualification"
)
VERSION = "public-source-acceptance-v1"
CASE_FIELDS = {"case_id", "source_url", "input_sha256", "model_sha256",
               "source_review", "evidence_sha256"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def json_object(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON token: {value}")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    require(isinstance(result, dict), "expected JSON object")
    return result


def file_bytes(root: Path, relative: str, limit: int = MAX_MEMBER_BYTES) -> bytes:
    parts = PurePosixPath(relative).parts
    require(bool(parts) and not relative.startswith("/") and ".." not in parts
            and "\\" not in relative, "unsafe evidence relative path")
    path = root
    for part in parts:
        path = path / part
        require(not path.is_symlink(), f"symlink evidence refused: {path}")
    require(path.resolve(strict=True).is_relative_to(root) and path.is_file(),
            f"evidence file escaped root: {path}")
    require(0 < path.stat().st_size <= limit, f"empty/oversized evidence: {path}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    require(0 < len(raw) <= limit, f"empty/oversized evidence read: {path}")
    return raw


def pinned_json(path: Path, expected: str) -> tuple[dict, Path]:
    importer._sha(expected, "caller pin")
    require(not path.is_symlink(), "symlink pinned manifest refused")
    path = path.expanduser().resolve(strict=True)
    require(not path.is_relative_to(ROOT), "input evidence must be external")
    raw = file_bytes(path.parent, path.name)
    require(sha(raw) == expected, f"caller pin mismatch: {path.name}")
    return json_object(raw), path


def record(raw: bytes) -> dict:
    return {"bytes": len(raw), "sha256": sha(raw)}


def rights(value: dict, label: str) -> None:
    require(value.get("license_verified") is False
            and value.get("public_redistribution") is False,
            f"{label}: source acceptance cannot elevate license/redistribution")
    if "license_status" in value:
        require(value["license_status"] == "benchmark-use-declaration",
                f"{label}: unsupported license declaration")


def archive_members(raw: bytes, commit: str) -> dict[str, bytes]:
    """Inspect every header without extracting or executing upstream content."""
    selected = DOCS | set(MEMBERS.values())
    result, seen, total, count = {}, set(), 0, 0
    prefix = f"HiGHS-{commit}"
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            canonical = member.name.rstrip("/")
            require(bool(parts) and parts[0] == prefix
                    and not member.name.startswith("/") and ".." not in parts
                    and "\\" not in member.name and canonical == "/".join(parts),
                    "unsafe/wrong-commit archive member")
            relative = "/".join(parts[1:])
            require(relative not in seen, "duplicate archive member")
            seen.add(relative)
            require(member.isfile() or member.isdir(), "special/link archive member")
            require(member.size >= 0, "negative archive member size")
            count += 1
            total += member.size
            require(count <= MAX_MEMBERS and total <= MAX_SCAN_BYTES,
                    "archive scan resource limit")
            if relative not in selected:
                continue
            require(member.isfile() and 0 < member.size <= MAX_MEMBER_BYTES,
                    "invalid selected archive member")
            stream = archive.extractfile(member)
            require(stream is not None, "missing archive member stream")
            with stream:
                data = stream.read(MAX_MEMBER_BYTES + 1)
            require(len(data) == member.size, "truncated selected archive member")
            result[relative] = data
    require({"LICENSE.txt", "README.md", *MEMBERS.values()} <= set(result),
            "missing required source archive members")
    return result


def ordered_cases(value: dict, label: str) -> list[dict]:
    cases = value.get("cases")
    require(isinstance(cases, list) and bool(cases)
            and all(isinstance(row, dict) for row in cases),
            f"{label}: empty/invalid membership")
    require([row.get("case_id") for row in cases] == list(MEMBERS),
            f"{label}: case membership/order mismatch")
    return cases


def build_evidence(acquisition_path: Path, acquisition_pin: str, source_path: Path,
                   source_pin: str, corpus_dir: Path, corpus_pin: str) -> dict:
    acquisition, acquisition_path = pinned_json(acquisition_path, acquisition_pin)
    source, source_path = pinned_json(source_path, source_pin)
    corpus_dir = corpus_dir.expanduser().resolve(strict=True)
    corpus, _ = pinned_json(corpus_dir / "corpus.json", corpus_pin)
    for value, label, verdict in (
        (source, "source", "REVIEW_REQUIRED"),
        (acquisition, "acquisition", "ACQUIRED_FROZEN_NOT_SOLVER_QUALIFIED"),
    ):
        require(type(value.get("schema_version")) is int and value["schema_version"] == 1
                and value.get("version") == "public-acquisition-v1"
                and value.get("verdict") == verdict, f"{label}: schema/verdict mismatch")
        rights(value, label)
        require(value.get("repository") == REPOSITORY, f"{label}: unofficial repository")
    commit = source.get("resolved_commit")
    require(isinstance(commit, str) and re.fullmatch("[0-9a-f]{40}", commit) is not None,
            "full official commit required")
    require(source.get("requested_ref") == REF and acquisition.get("resolved_commit") == commit,
            "source/acquisition commit/ref mismatch")
    require(acquisition.get("source_manifest_sha256") == source_pin
            and acquisition.get("corpus_manifest_sha256") == corpus_pin,
            "acquisition cross-pin mismatch")
    archive_url = f"https://codeload.github.com/ERGO-Code/HiGHS/tar.gz/{commit}"
    for key, filename, expected_url, limit in (
        ("commit_transfer", "commit.json", f"{API_ROOT}/commits/{REF}", MAX_MEMBER_BYTES),
        ("archive_transfer", "official.tar.gz", archive_url, MAX_ARCHIVE_BYTES),
    ):
        raw = file_bytes(source_path.parent, filename, limit)
        transfer = source.get(key)
        require(isinstance(transfer, dict) and transfer.get("url") == expected_url,
                f"{key}: official URL mismatch")
        require(record(raw) == {name: transfer.get(name) for name in ("bytes", "sha256")},
                f"{filename}: transfer byte/hash mismatch")
        if filename == "commit.json":
            api = json_object(raw)
            require(api.get("sha") == commit
                    and api.get("html_url") == f"{REPOSITORY}/commit/{commit}"
                    and api.get("url") == f"{API_ROOT}/commits/{commit}"
                    and api.get("commit", {}).get("url") == f"{API_ROOT}/git/commits/{commit}",
                    "commit API identity mismatch")
            commit_record = record(raw)
        else:
            members = archive_members(raw, commit)
            archive_record = record(raw)
    documents = {name: record(data) for name, data in members.items() if name in DOCS}
    require(source.get("documents") == documents, "review document membership/hash mismatch")
    for name in documents:
        require(file_bytes(source_path.parent, f"review/{name}") == members[name],
                f"review document bytes differ from archive: {name}")
    require("MIT License" in members["LICENSE.txt"].decode("utf-8"),
            "unexpected repository license text; not a data license verdict")
    members["README.md"].decode("utf-8")
    code_hashes = {
        "scripts/acquire_public_corpus.py": sha((ROOT / "scripts/acquire_public_corpus.py").read_bytes()),
        "scripts/public_qp_corpus.py": sha(Path(importer.__file__).read_bytes()),
    }
    require(acquisition.get("source_hashes") == code_hashes
            and source.get("source_hashes") == {
                "scripts/acquire_public_corpus.py": code_hashes["scripts/acquire_public_corpus.py"]}
            and corpus.get("source_hashes") == {
                "scripts/public_qp_corpus.py": code_hashes["scripts/public_qp_corpus.py"]},
            "acquisition/importer source hash mismatch")
    replay = importer.verify_frozen(corpus_dir, corpus_pin)
    require(replay["case_count"] == 2 and replay["public_declared_count"] == 2
            and replay["public_qualified_count"] == 0 and replay["industrial_qualified_count"] == 0,
            "unsupported frozen source qualification/membership")
    require({path.name for path in corpus_dir.iterdir()}
            == {"corpus.json", "input-manifest.json", *MEMBERS}, "frozen root membership mismatch")
    acquisition_cases = ordered_cases(acquisition, "acquisition")
    frozen_cases = ordered_cases(corpus, "corpus")
    declarations_raw = file_bytes(acquisition_path.parent, "input.json")
    require(declarations_raw == file_bytes(corpus_dir, "input-manifest.json"),
            "input declaration bytes differ from frozen manifest")
    declarations = json_object(declarations_raw)
    require(set(declarations) == {"schema_version", "cases"} and declarations["schema_version"] == 1,
            "unsupported declaration schema")
    declared_cases = ordered_cases(declarations, "input declaration")
    expected_extract = {member: record(members[member]) for member in MEMBERS.values()}
    require(acquisition.get("extracted") == expected_extract
            and acquisition.get("case_count") == len(MEMBERS), "extracted member/hash mismatch")
    previous_replay = acquisition.get("replay")
    require(isinstance(previous_replay, dict)
            and previous_replay.get("verdict") == "VERIFIED"
            and previous_replay.get("corpus_sha256") == corpus_pin
            and previous_replay.get("case_count") == 2
            and previous_replay.get("public_declared_count") == 2
            and previous_replay.get("public_qualified_count") == 0
            and previous_replay.get("industrial_qualified_count") == 0,
            "acquisition replay identity/membership mismatch")
    checked = []
    for summary, frozen, declared in zip(acquisition_cases, frozen_cases, declared_cases):
        case_id = summary["case_id"]
        member = MEMBERS[case_id]
        url = f"https://raw.githubusercontent.com/ERGO-Code/HiGHS/{commit}/{member}"
        require(declared.get("path") == member and declared.get("source_url") == url
                and declared.get("origin") == "public", "case official URL/path/origin mismatch")
        declaration = importer.declaration(declared)
        require(declaration["license"].startswith("benchmark-use-declaration:")
                and "NOT VERIFIED" in declaration["license"]
                and "no public redistribution" in declaration["license"],
                "case license declaration cannot elevate permissions")
        require(file_bytes(acquisition_path.parent, f"local/{member}") == members[member]
                and file_bytes(corpus_dir, f"{case_id}/raw.mps") == members[member],
                "case raw bytes differ from official archive")
        require({path.name for path in (corpus_dir / case_id).iterdir()}
                == {"raw.mps", "arrays.npz", "metadata.json", "reader.log"},
                "frozen model file membership mismatch")
        metadata_raw = file_bytes(corpus_dir, f"{case_id}/metadata.json")
        metadata = json_object(metadata_raw)
        arrays_raw = file_bytes(corpus_dir, f"{case_id}/arrays.npz", MAX_ARCHIVE_BYTES)
        with np.load(io.BytesIO(arrays_raw), allow_pickle=False) as saved:
            require(len(saved.files) == len(importer.ARRAY_NAMES)
                    and set(saved.files) == set(importer.ARRAY_NAMES), "frozen array membership mismatch")
            arrays = {name: saved[name] for name in importer.ARRAY_NAMES}
        model = {**metadata, "arrays": arrays}
        model_pin = importer.model_hash(model)
        require(metadata["provenance"] == declared and metadata["model_sha256"] == model_pin
                and frozen["model_sha256"] == model_pin and summary["model_sha256"] == model_pin,
                "original model hash/provenance mismatch")
        require(metadata["array_hashes"] == importer.array_hashes(arrays)
                and metadata["npz_sha256"] == sha(arrays_raw)
                and frozen["metadata_sha256"] == sha(metadata_raw), "frozen model byte/hash mismatch")
        raw_pin = sha(members[member])
        require(declared["input_sha256"] == raw_pin and summary["input_sha256"] == raw_pin
                and frozen["raw_sha256"] == raw_pin and metadata["raw_sha256"] == raw_pin,
                "case input hash mismatch")
        rights(summary, case_id)
        facts = {
            "variables": len(model["variable_names"]), "constraint_rows": len(model["row_names"]),
            "A_nnz": int(np.count_nonzero(arrays["A"])), "P_nnz": int(np.count_nonzero(arrays["P"])),
            "offset": float(arrays["offset"]), "objective_sense": model["objective_sense"],
            "origin": "public", "independent_dimensions_bounds_offset": "PASS",
        }
        require(all(summary.get(key) == value for key, value in facts.items()),
                "acquisition original model summary mismatch")
        require(facts["P_nnz"] == 0 and facts["offset"] == 0
                and np.all(arrays["variable_lower"] == 0)
                and np.all(np.isposinf(arrays["variable_upper"])), "known LP bounds/offset mismatch")
        reader_raw = file_bytes(corpus_dir, f"{case_id}/reader.log")
        checked.append({"case_id": case_id, "archive_member": member, "source_url": url,
                        "input_sha256": raw_pin, "model_sha256": model_pin, **facts,
                        "metadata": record(metadata_raw), "npz": record(arrays_raw),
                        "array_hashes": importer.array_hashes(arrays),
                        "reader_log": record(reader_raw)})
    return {
        "schema_version": 1, "version": VERSION, "verdict": "SOURCE_EVIDENCE_VERIFIED",
        "anchors": {"acquisition_sha256": acquisition_pin, "source_sha256": source_pin,
                    "corpus_sha256": corpus_pin},
        "inputs": {"acquisition": str(acquisition_path), "source": str(source_path),
                   "corpus": str(corpus_dir)},
        "official_repository": REPOSITORY, "resolved_commit": commit, "requested_ref": REF,
        "archive_url": archive_url, "archive": archive_record, "commit_api": commit_record,
        "documents": documents, "selected_members": expected_extract, "cases": checked,
        "license_verified": False, "public_redistribution": False,
        "license_status": "benchmark-use-declaration", "review_scope": REVIEW_SCOPE,
        "verified_code_sha256": code_hashes,
        "producer_sha256": sha(Path(__file__).read_bytes()),
    }


def acceptance_document(evidence: dict, evidence_pin: str) -> dict:
    return {
        "schema_version": 1, "verdict": "PUBLIC_SOURCE_ACCEPTED",
        "corpus_sha256": evidence["anchors"]["corpus_sha256"],
        "cases": [{**{key: row[key] for key in
                     ("case_id", "source_url", "input_sha256", "model_sha256")},
                   "source_review": REVIEW_SCOPE, "evidence_sha256": evidence_pin}
                  for row in evidence["cases"]],
    }


def verify_acceptance(path: Path, acceptance_pin: str) -> dict:
    acceptance, path = pinned_json(path, acceptance_pin)
    require(set(acceptance) == {"schema_version", "verdict", "corpus_sha256", "cases"},
            "acceptance schema mismatch")
    cases = ordered_cases(acceptance, "acceptance")
    require(all(set(row) == CASE_FIELDS for row in cases), "acceptance case schema mismatch")
    evidence_pin = cases[0]["evidence_sha256"]
    require(all(row["evidence_sha256"] == evidence_pin for row in cases),
            "acceptance evidence hash membership mismatch")
    evidence, _ = pinned_json(path.parent / "evidence.json", evidence_pin)
    anchors, inputs = evidence["anchors"], evidence["inputs"]
    recomputed = build_evidence(Path(inputs["acquisition"]), anchors["acquisition_sha256"],
                                Path(inputs["source"]), anchors["source_sha256"],
                                Path(inputs["corpus"]), anchors["corpus_sha256"])
    require(evidence == recomputed, "source evidence differs from independent replay")
    require(acceptance == acceptance_document(recomputed, evidence_pin),
            "acceptance differs from checked source evidence")
    return {"verdict": "PUBLIC_SOURCE_REPLAYED", "case_count": len(cases), "checked_at": now(),
            "acceptance_sha256": acceptance_pin, "evidence_sha256": evidence_pin,
            "license_verified": False, "public_redistribution": False}


def produce(acquisition: Path, acquisition_pin: str, source: Path, source_pin: str,
            corpus: Path, corpus_pin: str, output: Path) -> dict:
    output = importer.external(output)
    output.mkdir(parents=True, exist_ok=False)
    status = {"schema_version": 1, "verdict": "RUNNING", "started_at": now()}
    with (output / "acceptance.log").open("x", encoding="utf-8") as log:
        log.write(f"PUBLIC_SOURCE_START checked_at={status['started_at']}\n")
        log.flush()
        try:
            evidence = build_evidence(acquisition, acquisition_pin, source, source_pin, corpus, corpus_pin)
            importer.write_json(output / "evidence.json", evidence)
            evidence_pin = sha((output / "evidence.json").read_bytes())
            require(json_object((output / "evidence.json").read_bytes()) == build_evidence(
                acquisition, acquisition_pin, source, source_pin, corpus, corpus_pin),
                "source evidence differs from independent replay before publication")
            acceptance = acceptance_document(evidence, evidence_pin)
            importer.write_json(output / "acceptance.json", acceptance)
            acceptance_pin = sha((output / "acceptance.json").read_bytes())
            status.update({"verdict": "PUBLIC_SOURCE_ACCEPTED", "case_count": len(acceptance["cases"]),
                           "acceptance": str(output / "acceptance.json"),
                           "acceptance_sha256": acceptance_pin, "evidence_sha256": evidence_pin,
                           "license_verified": False, "public_redistribution": False})
        except Exception as error:
            status.update({"verdict": "ERROR", "error": {"type": type(error).__name__,
                                                       "message": str(error)}, "exit_code": 2})
            (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            status["finished_at"] = now()
            importer.write_json(output / "status.json", status)
            log.write(json.dumps(status, sort_keys=True, allow_nan=False) + "\n")
    return status


def check_consumer(path: Path, acceptance_pin: str, runner: Path, runner_pin: str,
                   output: Path) -> dict:
    """Run only the explicitly selected local consumer's source gate, never solve."""
    output = importer.external(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "verdict": "RUNNING", "started_at": now(),
              "scope": "source_eligibility interoperability only; zero solves"}
    try:
        replay = verify_acceptance(path, acceptance_pin)
        acceptance, path = pinned_json(path, acceptance_pin)
        evidence = json_object(file_bytes(path.parent, "evidence.json"))
        runner = runner.expanduser().resolve(strict=True)
        importer._sha(runner_pin, "caller runner source pin")
        require(sha(file_bytes(runner.parent, runner.name)) == runner_pin, "runner source pin mismatch")
        spec = importlib.util.spec_from_file_location("_public_source_consumer_" + runner_pin, runner)
        require(spec is not None and spec.loader is not None, "cannot load selected consumer")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        require(sha(Path(module.corpus.__file__).read_bytes())
                == evidence["verified_code_sha256"]["scripts/public_qp_corpus.py"],
                "consumer importer source hash mismatch")
        corpus_dir = Path(evidence["inputs"]["corpus"])
        frozen = json_object(file_bytes(corpus_dir, "corpus.json"))
        rows = module.source_eligibility(frozen, corpus_dir, acceptance, acceptance["corpus_sha256"])
        require(isinstance(rows, list) and [row["case_id"] for row in rows] == list(MEMBERS)
                and all(row["eligible"] is True and row["reason"] is None for row in rows),
                "consumer source eligibility did not accept exact real membership")
        require(sha(file_bytes(runner.parent, runner.name)) == runner_pin,
                "runner source changed during consumer check")
        result.update({"verdict": "PUBLIC_SOURCE_CONSUMER_ACCEPTED", "case_count": len(rows),
                       "cases": rows, "acceptance_sha256": replay["acceptance_sha256"],
                       "runner": str(runner), "runner_sha256": runner_pin,
                       "license_verified": False, "public_redistribution": False})
    except Exception as error:
        result.update({"verdict": "ERROR", "exit_code": 2,
                       "error": {"type": type(error).__name__, "message": str(error)}})
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        result["finished_at"] = now()
        importer.write_json(output / "consumer.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    accept = commands.add_parser("accept")
    accept.add_argument("--acquisition", type=Path, required=True)
    accept.add_argument("--acquisition-sha256", required=True)
    accept.add_argument("--source", type=Path, required=True)
    accept.add_argument("--source-sha256", required=True)
    accept.add_argument("--corpus", type=Path, required=True)
    accept.add_argument("--corpus-sha256", required=True)
    accept.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("verify")
    replay.add_argument("--acceptance", type=Path, required=True)
    replay.add_argument("--acceptance-sha256", required=True)
    consumer = commands.add_parser("check-consumer")
    consumer.add_argument("--acceptance", type=Path, required=True)
    consumer.add_argument("--acceptance-sha256", required=True)
    consumer.add_argument("--runner", type=Path, required=True)
    consumer.add_argument("--runner-sha256", required=True)
    consumer.add_argument("--artifact-dir", type=Path, required=True)
    args = cli.parse_args(argv)
    try:
        if args.action == "accept":
            result = produce(args.acquisition, args.acquisition_sha256, args.source,
                             args.source_sha256, args.corpus, args.corpus_sha256, args.artifact_dir)
        elif args.action == "verify":
            result = verify_acceptance(args.acceptance, args.acceptance_sha256)
        else:
            result = check_consumer(args.acceptance, args.acceptance_sha256, args.runner,
                                    args.runner_sha256, args.artifact_dir)
        print("PUBLIC_SOURCE " + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
