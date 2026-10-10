"""Persist an explicit experiment/report ledger on shared storage."""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path


def freeze_evidence(source: Path, root: Path) -> Path:
    """Content-address an evidence read so later live refreshes cannot change it."""
    payload = source.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    directory = root / "report-inputs"
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"{digest}.json"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError("content-addressed evidence mismatch")
    else:
        with destination.open("xb") as handle:
            handle.write(payload)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--paired-run", type=Path, required=True)
    parser.add_argument("--reference", type=Path, action="append", required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--cuda-regression", type=Path, required=True)
    parser.add_argument("--final-smoke", type=Path)
    parser.add_argument("--final-snapshot", type=Path, required=True)
    parser.add_argument("--environment-probe", type=Path, required=True)
    args = parser.parse_args()
    record = json.loads((args.paired_run / "run.json").read_text())
    bundle = {
        "modified_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "historical_sha": record["source_sha"],
        "paired_results": [str(Path(item["output"]) / "result.json") for item in record["jobs"]],
        "reference_results": [str(path) for path in args.reference],
        "verification": str(args.verification), "wheel": str(args.wheel),
        "cuda_regression": str(args.cuda_regression),
        "final_smoke": str(args.final_smoke) if args.final_smoke else None,
        "final_snapshot": str(freeze_evidence(args.final_snapshot, args.root)),
        "environment_probe": str(args.environment_probe),
        "failures": [],
        "run_records": [],
        "ladder_results": [],
    }
    for run_file in sorted(args.root.glob("*/run.json")):
        run = json.loads(run_file.read_text())
        bundle["run_records"].append(str(run_file))
        if run_file.parent.name == "probe-gkc7keyc":
            bundle["failures"].append({
                "stage": "submission", "reason": "observer identity denied Job creation; no Pod created",
                "evidence": str(run_file.parent),
            })
        for item in run["jobs"]:
            result_path = Path(item["output"]) / "result.json"
            if result_path.is_file():
                result = json.loads(result_path.read_text())
                if result["verdict"] == "FAIL":
                    bundle["failures"].append({
                        "stage": run["stage"], "reason": result["error"],
                        "evidence": str(result_path),
                    })
                elif run["stage"] in ("smoke", "ladder"):
                    bundle["ladder_results"].append(str(result_path))
            elif run_file.parent.name == "probe-wy6qvauf":
                bundle["failures"].append({
                    "stage": "container startup", "reason": "StartError/128: CUDA base has no Python",
                    "evidence": str(run_file.parent),
                })
    path = args.root / "report-bundle.json"
    path.write_text(json.dumps(bundle, indent=2, allow_nan=False) + "\n")
    print(f"QP_REPORT_BUNDLE path={path} failures={len(bundle['failures'])}")


if __name__ == "__main__":
    main()
