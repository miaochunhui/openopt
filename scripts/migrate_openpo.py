"""Import an explicit committed OpenPO snapshot, or verify its local manifest.

This is a one-time extraction tool, not a bidirectional synchronization tool.
It reads source Git objects without checkout, reset, stash, add or commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "a0300f07efaf9669e18d236f71c739490d3afe2a"
MANIFEST = ROOT / "docs/migration-manifest.json"
BENCHMARKS = {
    "benchmark_solver.py",
    "compare_open_source.py",
    "compare_constrained.py",
    "generate_constrained_cases.py",
    "compare_warm_start.py",
    "profile_router.py",
    "mpo_t1_benchmark.py",
    "benchmark_trade_style_qp.py",
    "benchmark_cntr_6000.py",
}
CORE_TESTS = {
    "test_admm_solver.py", "test_barra_adapter.py", "test_barra_io.py",
    "test_cardinality_failures.py", "test_certification_policy.py",
    "test_constrained_benchmark.py", "test_constrained_cases.py",
    "test_constrained_solver.py", "test_constrained_warm_start.py",
    "test_constraints.py", "test_device_contract.py", "test_durable_shadow.py",
    "test_exact_solver.py", "test_facade.py", "test_hardware.py",
    "test_mosek_prepared_lp.py", "test_open_source_comparison.py",
    "test_projection_adversarial.py", "test_projection_correction.py",
    "test_public_api.py", "test_router.py", "test_service_time_calibration.py",
    "test_shadow.py", "test_shadow_codec.py", "test_shadow_journal.py",
    "test_shadow_server.py", "test_solver.py", "test_trade_optimizer.py",
    "test_trade_style_optimizer.py", "test_triton_projection.py",
    "test_warm_start_benchmark.py", "test_benchmark_trade_style_qp.py",
}
DOCS = {
    "mpo-t1-SPEC.md", "mpo-t1-delaycost-design.md",
    "mpo-t1-rollforward-design.md", "mpo-t1-shrinkage-design.md",
}


def git(source: Path, *args: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(source), *args], capture_output=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(
            f"source Git command failed rc={result.returncode}: "
            f"{result.stderr.decode(errors='replace').strip()}"
        )
    return result.stdout


def included(path: str) -> bool:
    parts = Path(path).parts
    if path.startswith("src/openpo/") and path.endswith(".py"):
        return True
    if path.startswith("tests/mpo/") or path.startswith("tests/helpers/"):
        return path.endswith(".py")
    return (
        len(parts) == 2
        and (
            (parts[0] == "tests" and parts[1] in CORE_TESTS)
            or (parts[0] == "benchmarks" and parts[1] in BENCHMARKS)
            or (parts[0] == "docs" and parts[1] in DOCS)
        )
    )


def local_path(path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError(f"unsafe manifest path: {path}")
    destination = ROOT / relative
    if not destination.resolve().is_relative_to(ROOT):
        raise ValueError(f"destination escapes repository: {path}")
    return destination


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def import_snapshot(source: Path) -> None:
    names = git(source, "ls-tree", "-r", "--name-only", SOURCE_COMMIT).decode().splitlines()
    paths = sorted(path for path in names if included(path))
    if not paths or not any(path.startswith("src/openpo/mpo/") for path in paths):
        raise ValueError("source selection is empty or lacks the expected MPO implementation")
    expected = (
        {f"benchmarks/{name}" for name in BENCHMARKS}
        | {f"tests/{name}" for name in CORE_TESTS}
        | {f"docs/{name}" for name in DOCS}
    )
    missing = sorted(expected.difference(paths))
    if missing:
        raise ValueError(f"required selected files missing from source commit: {missing}")
    pending: list[tuple[Path, bytes]] = []
    files = []
    for path in paths:
        mode = git(source, "ls-tree", SOURCE_COMMIT, "--", path).decode().split()[0]
        if mode not in {"100644", "100755"}:
            raise ValueError(f"source is not a regular file: {path} mode={mode}")
        content = git(source, "show", f"{SOURCE_COMMIT}:{path}")
        if not content:
            raise ValueError(f"unexpected empty import: {path}")
        if (source / path).is_symlink() or (source / path).read_bytes() != content:
            raise ValueError(f"selected source file has uncommitted changes: {path}")
        destination = local_path(path)
        if destination.is_symlink():
            raise ValueError(f"destination must not be a symlink: {path}")
        if destination.exists() and destination.read_bytes() != content:
            raise ValueError(f"refusing to overwrite divergent destination: {path}")
        pending.append((destination, content))
        files.append({"path": path, "bytes": len(content), "sha256": digest(content)})
    excluded = [
        {
            "path": path,
            "reason": (
                "legacy gate, historical-artifact renderer, or excluded private delivery/deployment"
                if path.startswith("tests/")
                else "outside the standalone foundation whitelist"
            ),
        }
        for path in names
        if path.startswith(("tests/", "benchmarks/", "scripts/", "docs/", "deploy/", "deliverables/"))
        and path not in paths
    ]
    manifest = {
        "schema_version": 1,
        "migration_date": "2026-10-10",
        "source_project": "optimizer",
        "source_commit": SOURCE_COMMIT,
        "method": "explicit committed snapshot; no source history imported",
        "licensing": "NO_LICENSE_DECLARATION_IN_SOURCE; rights-holder decision required",
        "files": files,
        "excluded": excluded,
    }
    rendered = (json.dumps(manifest, indent=2, ensure_ascii=True) + "\n").encode()
    manifest_extension = False
    if MANIFEST.exists() and MANIFEST.read_bytes() != rendered:
        previous = json.loads(MANIFEST.read_text())
        previous_files = {row["path"]: row for row in previous.get("files", [])}
        new_files = {row["path"]: row for row in files}
        if (
            previous.get("source_commit") != SOURCE_COMMIT
            or not previous_files
            or not previous_files.keys() < new_files.keys()
            or any(new_files[path] != row for path, row in previous_files.items())
        ):
            raise ValueError("refusing to overwrite a divergent migration manifest")
        manifest_extension = True
    for destination, content in pending:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not destination.exists():
            with destination.open("xb") as output:
                output.write(content)
    if not MANIFEST.exists():
        with MANIFEST.open("xb") as output:
            output.write(rendered)
    elif manifest_extension:
        temporary = MANIFEST.with_suffix(".json.new")
        with temporary.open("xb") as output:
            output.write(rendered)
        temporary.replace(MANIFEST)
    verify_snapshot()


def verify_snapshot(*, exact_worktree: bool = True) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("source_commit") != SOURCE_COMMIT:
        raise ValueError("unsupported or divergent migration manifest")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest must contain a nonempty file list")
    seen: set[str] = set()
    total = 0
    origin_commit = None
    if not exact_worktree:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-list", "--max-parents=0", "HEAD"],
            capture_output=True, check=False,
        )
        if result.returncode == 0:
            roots = result.stdout.decode().splitlines()
            if len(roots) != 1:
                raise ValueError("provenance expects a single initial import commit")
            origin_commit = roots[0]
        else:
            # A newly initialized repository has no baseline commit yet.
            unborn = subprocess.run(
                ["git", "-C", str(ROOT), "symbolic-ref", "--quiet", "HEAD"],
                capture_output=True, check=False,
            )
            ref_exists = subprocess.run(
                ["git", "-C", str(ROOT), "show-ref", "--verify", "--quiet",
                 unborn.stdout.decode().strip()], capture_output=True, check=False,
            ) if unborn.returncode == 0 else None
            if ref_exists is None or ref_exists.returncode != 1:
                raise RuntimeError(f"cannot read provenance Git history: {result.stderr.decode().strip()}")
    for row in files:
        path = row["path"]
        if path in seen or not included(path):
            raise ValueError(f"duplicate or unapproved manifest file: {path}")
        seen.add(path)
        destination = local_path(path)
        if destination.is_symlink():
            raise ValueError(f"import must be a regular file: {path}")
        content = (
            git(ROOT, "show", f"{origin_commit}:{path}")
            if origin_commit else destination.read_bytes()
        )
        if len(content) != row["bytes"] or digest(content) != row["sha256"]:
            raise ValueError(f"imported snapshot differs from manifest: {path}")
        total += len(content)
    source_files = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/openpo").rglob("*.py")
    }
    expected_source = {path for path in seen if path.startswith("src/openpo/")}
    if exact_worktree and source_files != expected_source:
        raise ValueError(
            f"source set mismatch: added={sorted(source_files - expected_source)} "
            f"missing={sorted(expected_source - source_files)}"
        )
    print(
        f"OPENPO_SNAPSHOT verdict=PASS files={len(files)} bytes={total} "
        f"source_commit={SOURCE_COMMIT} "
        f"mode={'exact_worktree' if exact_worktree else 'provenance'} "
        f"baseline={origin_commit or 'uncommitted_snapshot'}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--provenance", action="store_true")
    args = parser.parse_args()
    try:
        if args.apply:
            if args.source is None or not args.source.is_dir():
                raise ValueError("--apply requires an existing explicit --source repository")
            if args.source.resolve() == ROOT:
                raise ValueError("source must differ from destination")
            import_snapshot(args.source.resolve())
        else:
            verify_snapshot(exact_worktree=not args.provenance)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        print(f"OPENPO_SNAPSHOT verdict=FAIL reason={error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
