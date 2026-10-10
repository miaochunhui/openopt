"""Acquire a tiny official HiGHS subset through the sole astrill-fetch entry.

Serial rate-limited URL fetches only. Existing remote tools are preflighted
without installation. No upstream code execution, broad archive extraction,
dataset relicensing, redistribution, solves, or automatic network retry.
Runtime evidence lives outside the worktree. The existing importer is reused.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tarfile
import time
import traceback

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import public_qp_corpus as importer

FETCH = Path("/data/dev/infra/ops/astrill-jumphost/astrill-fetch.py")
API = "https://api.github.com/repos/ERGO-Code/HiGHS/commits/v1.15.1"
REPOSITORY = "https://github.com/ERGO-Code/HiGHS"
REF = "v1.15.1"
RATE = "512k"
RATE_MIB_S = 0.5
DOCS = ("LICENSE", "LICENSE.txt", "README.md", "check/README.md", "check/instances/README.md",
        "check/instances/LICENSE", "check/instances/LICENSE.txt")
INSTANCES = ("check/instances/afiro.mps", "check/instances/adlittle.mps")
LICENSE_DECLARATION = (
    "benchmark-use-declaration: public official HiGHS test distribution; "
    "third-party Netlib instance license NOT VERIFIED; repository MIT is "
    "not asserted to cover these data; internal benchmark only, "
    "no public redistribution of raw or derived arrays"
)
MAX_MEMBER_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_BYTES = 128 * 1024 * 1024
VERSION = "public-acquisition-v1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_file(path: Path) -> str:
    return importer.sha256(path.read_bytes())


def command(argv: list[str], log_path: Path) -> subprocess.CompletedProcess:
    with log_path.open("x", encoding="utf-8") as log:
        log.write(f"command={argv!r}\n")
        log.flush()
        result = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                check=False)
        log.write(f"\nreturncode={result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"command failed rc={result.returncode}; log={log_path}")
    return result


def preflight(output: Path) -> dict:
    """Use the approved module's existing remote helper, never hand-built SSH.

    This checks prerequisites only. Missing tools stop before cmd_url can reach
    its legacy ensure_remote_tools() auto-install branch.
    """
    spec = importlib.util.spec_from_file_location("approved_astrill_fetch", FETCH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the sole approved fetch entry")
    fetch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fetch)
    result = fetch.remote(
        "command -v node && node --version && command -v git && git --version "
        "&& command -v wget && command -v rsync && command -v setsid "
        "&& command -v timeout && id -un"
    )
    with (output / "preflight.log").open("x", encoding="utf-8") as stream:
        stream.write(result.stdout + result.stderr + f"\nreturncode={result.returncode}\n")
    if result.returncode:
        raise RuntimeError(f"execution chain missing; no installation authorized; rc={result.returncode}")
    if result.stdout.strip().splitlines()[-1] != "infra":
        raise RuntimeError("remote execution identity is not infra")
    if not fetch.shutil.which("rsync"):
        raise RuntimeError("local rsync missing; no installation authorized")
    return {"verdict": "PREFLIGHTED", "identity": "infra", "installation": "forbidden",
            "fetch_entry_sha256": hash_file(FETCH)}


def transfer_markers(text: str, path: Path, expected_url: str) -> dict:
    matches = re.findall(r"^ASTRiLL_FETCH ok=1 mode=url (.*)$", text, re.M)
    if len(matches) != 1:
        raise ValueError("missing/ambiguous ASTRiLL_FETCH ok=1 terminal evidence")
    fields = dict(re.findall(r"(\w+)=([^\s]+)", matches[0]))
    if fields.get("url") != expected_url or fields.get("rate") != RATE:
        raise ValueError("fetch URL/rate differs from authorized request")
    if fields.get("sha256") != hash_file(path) or int(fields["bytes"]) != path.stat().st_size:
        raise ValueError("fetch terminal hash/bytes differ from actual payload")
    if not path.stat().st_size or path.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("empty/oversized acquisition payload")
    measured = {}
    for key in ("remote_max_mib_s", "local_max_mib_s"):
        value = fields.get(key, "NA")
        if value == "NA":
            measured[key] = None
        else:
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0 or numeric > 4:
                raise ValueError(f"observed transfer rate exceeds 4 MiB/s: {key}={value}")
            measured[key] = numeric
    # wget progress is independent evidence when the built-in sampler has too
    # few samples. Short metadata transfers can remain explicitly NA.
    progress_rates = []
    for number, unit in re.findall(
            r"^\[remote\] \d{4}-\d\d-\d\d .*?\((\d+(?:\.\d+)?) ([KMG])B/s\) - .*? saved \[",
            text, re.M):
        mib = float(number) * {"K": 1 / 1024, "M": 1, "G": 1024}[unit]
        if mib > 4:
            raise ValueError(f"wget observed rate exceeds 4 MiB/s: {number}{unit}B/s")
        progress_rates.append(mib)
    return {"url": expected_url, "bytes": path.stat().st_size, "sha256": hash_file(path),
            "configured_rate": RATE, "configured_mib_s": RATE_MIB_S,
            "sampler": measured, "wget_progress_mib_s": progress_rates,
            "sampler_scope": "upstream sampler stops before pull; local samples not a pull speed proof"}


def fetch_url(url: str, path: Path, output: Path, label: str) -> dict:
    if path.exists():
        raise FileExistsError(f"no overwrite/resume of a local acquisition artifact: {path}")
    argv = [sys.executable, "-B", str(FETCH), "url", "--url", url, "--out", str(path),
            "--rate", RATE, "--tries", "1", "--hard-timeout", "1800",
            "--log-dir", str(output / "fetch-logs")]
    started = time.monotonic()
    log = output / f"{label}-command.log"
    command(argv, log)
    result = transfer_markers(log.read_text(encoding="utf-8"), path, url)
    result["wall_seconds"] = time.monotonic() - started
    result["whole_command_average_mib_s"] = path.stat().st_size / result["wall_seconds"] / 1024**2
    result["average_scope"] = "whole command including SSH, remote fetch and serial pull, not peak rate"
    return result


def resolve_commit(value: dict) -> str:
    commit = value.get("sha")
    if not isinstance(commit, str) or not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("official commit API did not resolve a full SHA")
    if value.get("html_url") != f"{REPOSITORY}/commit/{commit}":
        raise ValueError("commit API repository identity mismatch")
    return commit


def archive_url(commit: str) -> str:
    if not re.fullmatch("[0-9a-f]{40}", commit):
        raise ValueError("full source commit required")
    return f"https://codeload.github.com/ERGO-Code/HiGHS/tar.gz/{commit}"


def extract_selected(archive: Path, commit: str, target: Path,
                     selected: tuple[str, ...], required: tuple[str, ...]) -> dict:
    """Never extractall: reject dangerous names/types, write selected bytes only."""
    if not selected or len(selected) != len(set(selected)) or not set(required) <= set(selected):
        raise ValueError("empty/duplicate/invalid extraction selection")
    approved = set(DOCS + INSTANCES)
    if not set(selected) <= approved:
        raise ValueError("unapproved archive member selection")
    prefix = f"HiGHS-{commit}"
    found = {}
    with tarfile.open(archive, "r:gz") as tar:
        seen = set()
        for member in tar:
            parts = PurePosixPath(member.name).parts
            if (not parts or member.name.startswith("/") or ".." in parts
                    or "\\" in member.name or parts[0] != prefix):
                raise ValueError(f"unsafe/wrong-commit archive path: {member.name}")
            relative = "/".join(parts[1:])
            if relative in seen:
                raise ValueError(f"duplicate archive member: {relative}")
            seen.add(relative)
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"link/device/special archive member refused: {member.name}")
            if relative not in selected:
                continue
            if not member.isfile() or member.size < 1 or member.size > MAX_MEMBER_BYTES:
                raise ValueError(f"invalid selected member size/type: {relative}")
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"missing member byte stream: {relative}")
            raw = stream.read(MAX_MEMBER_BYTES + 1)
            if len(raw) != member.size:
                raise ValueError(f"selected member truncated: {relative}")
            path = target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as output:
                output.write(raw)
            if path.read_bytes() != raw:
                raise OSError(f"extracted byte readback mismatch: {relative}")
            found[relative] = {"bytes": len(raw), "sha256": importer.sha256(raw)}
    if set(required) - set(found):
        raise ValueError(f"missing required archive members: {sorted(set(required) - set(found))}")
    return found


def source_review(output: Path, reuse_commit: Path | None = None,
                  reuse_commit_sha256: str | None = None,
                  reuse_archive: Path | None = None,
                  reuse_archive_sha256: str | None = None) -> dict:
    output = importer.external(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "version": VERSION, "verdict": "RUNNING",
              "started_at": now(), "repository": REPOSITORY, "requested_ref": REF,
              "source_hashes": {"scripts/acquire_public_corpus.py": hash_file(Path(__file__))},
              "network_parallelism": 1, "configured_rate": RATE,
              "license_status": "benchmark-use-declaration",
              "license_verified": False, "public_redistribution": False}
    try:
        report["preflight"] = preflight(output)
        if reuse_commit is None:
            if reuse_commit_sha256 is not None:
                raise ValueError("reuse commit hash without source path")
            report["commit_transfer"] = fetch_url(API, output / "commit.json", output, "commit")
        else:
            if hash_file(reuse_commit) != reuse_commit_sha256:
                raise ValueError("reuse commit caller-pinned SHA256 mismatch")
            previous_log = reuse_commit.parent / "commit-command.log"
            report["commit_transfer"] = transfer_markers(previous_log.read_text(), reuse_commit, API)
            with (output / "commit.json").open("xb") as stream:
                stream.write(reuse_commit.read_bytes())
            report["commit_reuse"] = {
                "path": str(reuse_commit), "sha256": reuse_commit_sha256,
                "reason": "offline recovery after diagnosed log-rate false positive; no network retry",
            }
        commit = resolve_commit(importer.read_json(output / "commit.json"))
        report["resolved_commit"] = commit
        url = archive_url(commit)
        if reuse_archive is None:
            if reuse_archive_sha256 is not None:
                raise ValueError("reuse archive hash without source path")
            report["archive_transfer"] = fetch_url(url, output / "official.tar.gz", output, "archive")
        else:
            if hash_file(reuse_archive) != reuse_archive_sha256:
                raise ValueError("reuse archive caller-pinned SHA256 mismatch")
            previous_log = reuse_archive.parent / "archive-command.log"
            report["archive_transfer"] = transfer_markers(previous_log.read_text(), reuse_archive, url)
            with (output / "official.tar.gz").open("xb") as stream:
                stream.write(reuse_archive.read_bytes())
            report["archive_reuse"] = {
                "path": str(reuse_archive), "sha256": reuse_archive_sha256,
                "reason": "offline recovery after actual LICENSE.txt path diagnosis; no network retry",
            }
        report["documents"] = extract_selected(output / "official.tar.gz", commit,
                                               output / "review", DOCS, ("LICENSE.txt", "README.md"))
        license_text = (output / "review/LICENSE.txt").read_text(encoding="utf-8")
        if "MIT License" not in license_text:
            raise ValueError("unexpected upstream repository license; human review required")
        report["repository_license_observed"] = "MIT text in root LICENSE.txt; data scope NOT inferred"
        report["verdict"] = "REVIEW_REQUIRED"
    except Exception as error:
        report["verdict"] = "ERROR"
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        with (output / "failure.log").open("x") as log:
            log.write(traceback.format_exc())
        raise
    finally:
        report["finished_at"] = now()
        importer.write_json(output / "source.json", report)
    return report


def independent_summary(model: dict, raw: bytes, case_id: str) -> dict:
    arrays = model["arrays"]
    # Raw ROWS/COLUMNS counts independently check the frozen dimensions.
    section = None
    variables, rows = set(), []
    for line in raw.decode("ascii").splitlines():
        if not line.strip() or line.lstrip().startswith("*"):
            continue
        tokens = line.split()
        if not line[0].isspace():
            section = tokens[0]
        elif section == "ROWS" and tokens[0] != "N":
            rows.append(tokens[1])
        elif section == "COLUMNS":
            variables.add(tokens[0])
    n, m = len(variables), len(rows)
    if arrays["q"].shape != (n,) or arrays["A"].shape != (m, n):
        raise ValueError(f"independent raw dimension check failed: {case_id}")
    if len(set(rows)) != m or set(model["variable_names"]) != variables or model["row_names"] != rows:
        raise ValueError("independent raw coordinate membership mismatch")
    if np_count_nonzero(arrays["P"]) != 0:
        raise ValueError("selected known LP unexpectedly has a quadratic objective")
    if float(arrays["offset"]) != 0:
        raise ValueError("selected known LP unexpectedly has an objective offset")
    if not (arrays["variable_lower"] == 0).all() or not np_isposinf(arrays["variable_upper"]).all():
        raise ValueError("selected known LP implicit nonnegative bounds differ")
    return {"case_id": case_id, "variables": n, "constraint_rows": m,
            "A_nnz": np_count_nonzero(arrays["A"]), "P_nnz": 0, "offset": 0.0,
            "objective_sense": model["objective_sense"], "input_sha256": importer.sha256(raw),
            "model_sha256": model["model_sha256"], "origin": "public",
            "license_verified": False, "license_status": "benchmark-use-declaration",
            "public_redistribution": False, "independent_dimensions_bounds_offset": "PASS"}


def np_count_nonzero(value) -> int:
    import numpy as np
    return int(np.count_nonzero(value))


def np_isposinf(value):
    import numpy as np
    return np.isposinf(value)


def freeze_reviewed(source: Path, source_sha256: str, output: Path,
                    acknowledge_benchmark_only: bool) -> dict:
    if not acknowledge_benchmark_only:
        raise ValueError("explicit benchmark-only/no-public-redistribution acknowledgement required")
    source = source.resolve(strict=True)
    if hash_file(source / "source.json") != source_sha256:
        raise ValueError("review source manifest caller pin mismatch")
    review = importer.read_json(source / "source.json")
    if review["verdict"] != "REVIEW_REQUIRED" or review["license_verified"] or review["public_redistribution"]:
        raise ValueError("unsupported source review/permission state")
    commit = review["resolved_commit"]
    if hash_file(source / "official.tar.gz") != review["archive_transfer"]["sha256"]:
        raise ValueError("official archive byte hash mismatch")
    output = importer.external(output)
    output.mkdir(parents=True, exist_ok=False)
    result = {"schema_version": 1, "version": VERSION, "verdict": "RUNNING",
              "started_at": now(), "source_manifest_sha256": source_sha256,
              "resolved_commit": commit, "repository": REPOSITORY, "cases": [],
              "license_verified": False, "public_redistribution": False,
              "source_hashes": {"scripts/acquire_public_corpus.py": hash_file(Path(__file__)),
                                "scripts/public_qp_corpus.py": hash_file(Path(importer.__file__))}}
    try:
        result["extracted"] = extract_selected(source / "official.tar.gz", commit,
                                               output / "local", INSTANCES, INSTANCES)
        cases = []
        for member in INSTANCES:
            path = output / "local" / member
            cases.append({"case_id": Path(member).stem, "path": member,
                          "source_url": f"https://raw.githubusercontent.com/ERGO-Code/HiGHS/{commit}/{member}",
                          "license": LICENSE_DECLARATION, "input_sha256": hash_file(path), "origin": "public"})
        importer.write_json(output / "input.json", {"schema_version": 1, "cases": cases})
        manifest = importer.freeze(output / "input.json", output / "local", output / "frozen")
        pinned = hash_file(manifest)
        result["corpus_manifest_sha256"] = pinned
        result["replay"] = importer.verify_frozen(output / "frozen", pinned)
        import numpy as np
        for entry in cases:
            case_dir = output / "frozen" / entry["case_id"]
            metadata = importer.read_json(case_dir / "metadata.json")
            with np.load(case_dir / "arrays.npz", allow_pickle=False) as loaded:
                model = {**metadata, "arrays": {key: loaded[key] for key in importer.ARRAY_NAMES}}
            summary = independent_summary(model, (case_dir / "raw.mps").read_bytes(), entry["case_id"])
            result["cases"].append(summary)
        if not result["cases"]:
            raise ValueError("empty real public acquisition rejected")
        result["case_count"] = len(result["cases"])
        result["verdict"] = "ACQUIRED_FROZEN_NOT_SOLVER_QUALIFIED"
    except Exception as error:
        result["verdict"] = "ERROR"
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        with (output / "failure.log").open("x") as log:
            log.write(traceback.format_exc())
        raise
    finally:
        result["finished_at"] = now()
        importer.write_json(output / "acquisition.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="action", required=True)
    review = sub.add_parser("review")
    review.add_argument("--artifact-dir", type=Path, required=True)
    review.add_argument("--reuse-commit", type=Path)
    review.add_argument("--reuse-commit-sha256")
    review.add_argument("--reuse-archive", type=Path)
    review.add_argument("--reuse-archive-sha256")
    freeze = sub.add_parser("freeze-reviewed")
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--source-sha256", required=True)
    freeze.add_argument("--artifact-dir", type=Path, required=True)
    freeze.add_argument("--acknowledge-benchmark-only", action="store_true")
    return cli


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.action == "review":
            result = source_review(args.artifact_dir, args.reuse_commit, args.reuse_commit_sha256,
                                   args.reuse_archive, args.reuse_archive_sha256)
        else:
            result = freeze_reviewed(args.source, args.source_sha256, args.artifact_dir,
                                     args.acknowledge_benchmark_only)
        print(f"PUBLIC_ACQUISITION verdict={result['verdict']} "
              f"commit={result['resolved_commit']} output={args.artifact_dir}", flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
