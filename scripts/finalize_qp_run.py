"""Read live cluster/evidence state and verify measured core source identity."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def get(args):
    result = subprocess.run(["kubectl", *args, "-o", "json"], text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"kubectl read failed rc={result.returncode}: {result.stderr[:1000]}")
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--namespace", required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text())
    nodes = get(["get", "nodes"])
    pods = get(["-n", args.namespace, "get", "pods", "-l", "repo=openopt"])
    jobs = get(["-n", args.namespace, "get", "jobs", "-l", "repo=openopt"])
    ready = {
        node["metadata"]["name"]: next(
            c["status"] for c in node["status"]["conditions"] if c["type"] == "Ready"
        ) for node in nodes["items"]
    }
    active = [pod["metadata"]["name"] for pod in pods["items"] if pod["status"]["phase"] in ("Pending", "Running")]
    if active or any(value != "True" for value in ready.values()):
        raise ValueError(f"live acceptance failed ready={ready} active={active}")
    core_hash = hashlib.sha256((ROOT / "src/openopt/qp.py").read_bytes()).hexdigest()
    paired_paths = {str(Path(path).parent) for path in bundle["paired_results"]}
    matched = 0
    stage_walls = []
    for run_file in bundle["run_records"]:
        record = json.loads(Path(run_file).read_text())
        if not any(item["output"] in paired_paths for item in record["jobs"]):
            continue
        if record["source_hashes"]["src/openopt/qp.py"] != core_hash:
            raise ValueError("current QP core differs from measured paired source")
        matched += 1
        walls = []
        for item in record["jobs"]:
            matching = [pod for pod in pods["items"]
                        if pod["metadata"]["labels"].get("job-name") == item["job"]]
            if len(matching) != 1:
                raise ValueError(f"missing/duplicate pod evidence: {item['job']}")
            pod = matching[0]
            if pod["status"]["phase"] != "Succeeded":
                raise ValueError(f"paired pod not successful: {item['job']}")
            terminated = pod["status"]["containerStatuses"][0]["state"]["terminated"]
            begin = datetime.fromisoformat(terminated["startedAt"].replace("Z", "+00:00"))
            end = datetime.fromisoformat(terminated["finishedAt"].replace("Z", "+00:00"))
            created = datetime.fromisoformat(pod["metadata"]["creationTimestamp"].replace("Z", "+00:00"))
            walls.append({
                "job": item["job"], "node": pod["spec"]["nodeName"],
                "gpu_count": pod["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"],
                "image_id": pod["status"]["containerStatuses"][0]["imageID"],
                "scheduling_startup_s": (begin - created).total_seconds(),
                "container_wall_s": (end - begin).total_seconds(),
                "started_at": begin.isoformat(), "finished_at": end.isoformat(),
                "restart_count": pod["status"]["containerStatuses"][0]["restartCount"],
            })
        stage_walls.append(walls)
    if matched != 1:
        raise ValueError(f"expected one measured paired source run, found {matched}")
    report = {
        "verdict": "PASS", "read_at": datetime.now(timezone.utc).isoformat(),
        "nodes_ready": ready, "active_openopt_pods": active,
        "core_sha256": core_hash, "measured_source_match": True,
        "paired_stage_pods": stage_walls,
        "failed_jobs": [job["metadata"]["name"] for job in jobs["items"] if job.get("status", {}).get("failed", 0)],
        "failures": "retained original startup/profiling failures; see diagnosis and ledger",
    }
    (args.bundle.parent / "final-snapshot.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"QP_FINAL verdict=PASS read_at={report['read_at']} ready_nodes={len(ready)} "
          f"active_cards=0 measured_source_match=true retained_failed_jobs={len(report['failed_jobs'])}")


if __name__ == "__main__":
    main()
