"""Freeze eight disjoint development seats and external release evidence.

This creates new worktrees only; it never deletes evidence, pushes Git, installs
dependencies, submits GPU jobs or changes the infrastructure publication scope.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SEATS = {
    "scaling": ["src/openopt/scaling.py", "tests/test_qp_scaling.py", "docs/research/scaling.md"],
    "termination": ["src/openopt/qp.py", "tests/test_qp_termination.py", "docs/research/termination.md"],
    "precondition": ["src/openopt/operators.py", "tests/test_qp_precondition.py", "docs/research/precondition.md"],
    "conic": ["src/openopt/conic.py", "tests/test_conic.py", "docs/research/conic.md"],
    "integer": ["src/openopt/integer.py", "tests/test_integer.py", "docs/research/integer.md"],
    "commercial": ["scripts/benchmark_commercial.py", "tests/test_benchmark_commercial.py", "docs/research/commercial-run.md"],
    "public-corpus": ["scripts/public_qp_corpus.py", "tests/test_public_qp_corpus.py", "docs/research/public-corpus.md"],
    "publication": ["scripts/prepare_publication.py", "tests/test_prepare_publication.py", "docs/research/publication.md"],
}


def git(*args, cwd=ROOT):
    result = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"Git failed rc={result.returncode}: {result.stderr}")
    return result.stdout.strip()


def setup(root: Path, name: str) -> Path:
    root = root.resolve()
    if not root.is_relative_to(Path("/3fs/research3fs/data/tmp")):
        raise ValueError("campaign artifacts must be external shared 3FS tmp")
    if not name or not all(c.isascii() and (c.isalnum() or c == "-") for c in name):
        raise ValueError("invalid campaign name")
    root.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix=name + "-", dir=root))
    record = {"baseline": git("rev-parse", "HEAD"),
              "created_at": datetime.now(timezone.utc).isoformat(), "seats": {}}
    for seat, scope in SEATS.items():
        path = ROOT.parent / f"openopt-wt-{name}-{seat}"
        if path.exists():
            raise ValueError(f"worktree already exists: {path}")
        branch = f"codex/{name}-{seat}"
        git("worktree", "add", "-b", branch, str(path), record["baseline"])
        record["seats"][seat] = {"path": str(path), "branch": branch, "scope": scope}
        (run / "campaign.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"RELEASE_SEAT seat={seat} path={path}", flush=True)
    print(f"RELEASE_CAMPAIGN verdict=PREPARED manifest={run / 'campaign.json'}", flush=True)
    return run / "campaign.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="release4-20261010")
    parser.add_argument("--artifact-root", type=Path, default=Path("/hf3fs-jd/data/tmp/openopt-release-20261010"))
    args = parser.parse_args()
    setup(args.artifact_root, args.name)


if __name__ == "__main__":
    main()
