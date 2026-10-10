"""Read bounded campaign evidence without dumping raw samples or claiming wins."""

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    result = {"path": str(path), "verdict": data.get("verdict"),
              "at": data.get("finished_at", data.get("checked_at"))}
    if "arms" in data:
        result["arms"] = {
            arm["id"]: {
                mode: row.get("latency", {}).get("p50_s")
                if isinstance(row.get("latency"), dict) else None
                for mode, row in arm["modes"].items()
            }
            for arm in data["arms"]
        }
        result["coverage"] = data.get("reference_coverage")
        result["failures"] = data.get("failures")
    elif "measurements" in data.get("result", {}):
        result["mode"] = data["mode"]
        result["p50_s"] = {
            mode: {arm: row["latency"]["p50_s"] for arm, row in arms.items()}
            for mode, arms in data["result"]["measurements"].items()
        }
        result["nsight"] = data.get("nsight")
    elif "counts" in data:
        result["counts"] = data["counts"]
        result["failures"] = data.get("failures")
    elif "rows" in data:
        result["counts"] = {}
        result["failures"] = []
        for row in data["rows"]:
            solver = row["solver"]
            counts = result["counts"].setdefault(solver, {"qualified": 0, "unqualified": 0})
            counts["qualified" if row["verdict"] == "QUALIFIED" else "unqualified"] += 1
            if row["verdict"] != "QUALIFIED":
                result["failures"].append({
                    "case": row["case_id"], "solver": solver, "status": row["solver_status"],
                    "reason": row.get("reason"),
                })
    else:
        result["tests"] = data.get("tests", data.get("result", {}).get("tests"))
        result["error"] = data.get("error")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()
    for path in args.paths:
        print(json.dumps(summarize(path), allow_nan=False))
