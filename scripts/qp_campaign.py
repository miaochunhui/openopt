"""Create isolated eight-seat worktrees and audit their real on-disk changes.

No cleanup or remote push is performed. Reusing an existing campaign is refused.
Artifacts are outside Git; source worktrees contain only code/config/docs.
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
    "adaptive": ["src/openopt/qp.py", "tests/test_qp_adaptive.py", "docs/research/adaptive.md"],
    "certificates": ["src/openopt/certificates.py", "tests/test_qp_certificates.py", "docs/research/certificates.md"],
    "operators": ["src/openopt/operators.py", "tests/test_qp_operators.py", "docs/research/operators.md"],
    "batch": ["src/openopt/batch.py", "tests/test_qp_batch.py", "docs/research/batch.md"],
    "benchmarks": ["scripts/benchmark_solver_comparison.py", "tests/test_solver_comparison_tools.py", "docs/research/benchmark-contract.md"],
    "corpus": ["src/openopt/corpus.py", "tests/test_qp_corpus.py", "scripts/qualify_qp_corpus.py", "docs/research/corpus.md"],
    "commercial": ["scripts/commercial_probe.py", "tests/test_commercial_probe.py", "docs/research/commercial-gap.md"],
    "acceptance": ["scripts/accept_qp_campaign.py", "tests/test_qp_campaign_acceptance.py", "docs/research/acceptance.md"],
}


def git(*args: str, cwd: Path = ROOT) -> str:
    completed = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if completed.returncode:
        raise RuntimeError(f"git {args} rc={completed.returncode}: {completed.stderr}")
    return completed.stdout.strip()


def setup(name: str, artifact_root: Path) -> None:
    if not name.replace("-", "").isalnum():
        raise ValueError("campaign name must be alphanumeric/hyphen")
    artifact_root = artifact_root.resolve()
    if not str(artifact_root).startswith("/3fs/research3fs/data/tmp/"):
        raise ValueError("campaign evidence must be on shared 3FS tmp")
    artifact_root.mkdir(parents=True, exist_ok=True)
    baseline = git("rev-parse", "HEAD")
    run = Path(tempfile.mkdtemp(prefix=name + "-", dir=artifact_root))
    record = {"baseline": baseline, "started_at": datetime.now(timezone.utc).isoformat(), "seats": {}}
    for seat, scope in SEATS.items():
        path = ROOT.parent / f"openopt-wt-{name}-{seat}"
        branch = f"codex/{name}-{seat}"
        if path.exists():
            raise ValueError(f"worktree already exists: {path}")
        git("worktree", "add", "-b", branch, str(path), baseline)
        record["seats"][seat] = {"path": str(path), "branch": branch, "scope": scope}
        (run / "campaign.json").write_text(json.dumps(record, indent=2) + "\n")
        print(f"QP_SEAT seat={seat} path={path} branch={branch}", flush=True)
    print(f"QP_CAMPAIGN state=PREPARED manifest={run / 'campaign.json'}", flush=True)


def audit(manifest: Path) -> None:
    record = json.loads(manifest.read_text())
    results = []
    for seat, item in record["seats"].items():
        path = Path(item["path"])
        committed = git("diff", "--name-only", record["baseline"], "HEAD", cwd=path).splitlines()
        uncommitted = git("diff", "--name-only", "HEAD", cwd=path).splitlines()
        untracked = git("ls-files", "--others", "--exclude-standard", cwd=path).splitlines()
        changes = sorted(set(committed + uncommitted + untracked))
        outside = sorted(set(changes) - set(item["scope"]))
        result = {
            "seat": seat, "head": git("rev-parse", "HEAD", cwd=path),
            "changes": changes, "outside_scope": outside,
            "index": git("rev-parse", "--git-path", "index", cwd=path),
            "verdict": "FAIL" if outside else "PASS",
        }
        results.append(result)
        print(f"QP_SEAT_AUDIT seat={seat} verdict={result['verdict']} changed={len(changes)} head={result['head'][:10]}")
    report = {"checked_at": datetime.now(timezone.utc).isoformat(), "seats": results}
    output = manifest.parent / f"audit-{datetime.now(timezone.utc).strftime('%H%M%S%f')}.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    if len(results) != 8 or any(result["outside_scope"] for result in results):
        raise RuntimeError(f"eight-seat scope audit failed: {output}")
    print(f"QP_CAMPAIGN_AUDIT verdict=PASS seats={len(results)} evidence={output}")


def live(namespace: str, artifact_root: Path, result_paths: list[Path]) -> None:
    def read_resource(*args: str) -> dict:
        completed = subprocess.run(
            ["kubectl", *args, "-o", "json"], capture_output=True, text=True,
        )
        if completed.returncode:
            raise RuntimeError(f"cluster read rc={completed.returncode}: {completed.stderr[:1000]}")
        return json.loads(completed.stdout)

    nodes = read_resource("get", "nodes")
    pods = read_resource("-n", namespace, "get", "pods", "-l", "repo=openopt")
    health = subprocess.run(["kubectl", "get", "--raw", "/readyz"], capture_output=True, text=True)
    if health.returncode or health.stdout.strip() != "ok":
        raise RuntimeError(f"apiserver readiness failed rc={health.returncode}")
    ready = {node["metadata"]["name"]: next(
        c["status"] for c in node["status"]["conditions"] if c["type"] == "Ready"
    ) for node in nodes["items"]}
    cordoned = [node["metadata"]["name"] for node in nodes["items"]
                if node["spec"].get("unschedulable", False)]
    active = [p["metadata"]["name"] for p in pods["items"] if p["status"]["phase"] in ("Pending", "Running")]
    retained_failures = []
    historical = {
        "openopt-probe-wy6qvauf-0": "Historical preflight: selected CUDA image had no Python; changed-image probe passed.",
        "openopt-smoke-cebpgx8a-0": "Historical Nsight FUSE scratch failure; local shared-memory scratch smoke passed.",
    }
    for pod in pods["items"]:
        if pod["status"]["phase"] != "Failed":
            continue
        job = pod["metadata"]["labels"].get("job-name")
        states = pod["status"].get("containerStatuses", [])
        retained_failures.append({
            "job": job, "owner": "openopt earlier 4500 experiment" if job in historical else "UNKNOWN",
            "cause": historical.get(job, "unattributed failure"),
            "containers": [{"termination": state["state"].get("terminated"),
                            "restart_count": state["restartCount"]} for state in states],
        })
    evidence = []
    for result_path in result_paths:
        record = json.loads((result_path.parent.parent / "run.json").read_text())
        item = next(j for j in record["jobs"] if Path(j["output"]) == result_path.parent)
        pod = next(p for p in pods["items"] if p["metadata"]["labels"].get("job-name") == item["job"])
        terminal = pod["status"]["containerStatuses"][0]["state"].get("terminated")
        if pod["status"]["phase"] != "Succeeded" or not terminal or terminal["exitCode"] != 0:
            raise RuntimeError(f"campaign job not successful: {item['job']}")
        for relative, digest in record["source_hashes"].items():
            if relative.startswith("src/openopt/") and hashlib_file(ROOT / relative) != digest:
                raise RuntimeError(f"measured source mismatch: {relative}")
        evidence.append({
            "result": str(result_path), "job": item["job"], "node": pod["spec"]["nodeName"],
            "started_at": terminal["startedAt"], "finished_at": terminal["finishedAt"],
            "created_at": pod["metadata"]["creationTimestamp"],
            "image_id": pod["status"]["containerStatuses"][0]["imageID"],
            "restart_count": pod["status"]["containerStatuses"][0]["restartCount"],
        })
    artifact_root = artifact_root.resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    report = {
        "verdict": "PASS" if ready and not active and not cordoned
        and all(v == "True" for v in ready.values())
        and all(failure["owner"] != "UNKNOWN" for failure in retained_failures) else "FAIL",
        "read_at": datetime.now(timezone.utc).isoformat(), "ready": ready, "active": active,
        "measurements": evidence, "cordoned": cordoned, "apiserver_ready": True,
        "retained_failures": retained_failures,
    }
    path = artifact_root / f"live-{datetime.now(timezone.utc).strftime('%H%M%S%f')}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"QP_CAMPAIGN_LIVE verdict={report['verdict']} ready={len(ready)} active={len(active)} evidence={path}")
    if report["verdict"] != "PASS":
        raise RuntimeError(f"live acceptance failed: {path}")


def hashlib_file(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("setup", "audit", "live"))
    parser.add_argument("--name", default="qp3-20261010")
    parser.add_argument("--artifact-root", type=Path, default=Path("/hf3fs-jd/data/tmp/openopt-campaign"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--namespace", default="devtest")
    parser.add_argument("--result", type=Path, action="append", default=[])
    args = parser.parse_args()
    if args.action == "setup":
        setup(args.name, args.artifact_root)
    elif args.action == "live":
        live(args.namespace, args.artifact_root, args.result)
    elif args.manifest is None:
        parser.error("audit requires --manifest")
    else:
        audit(args.manifest)


if __name__ == "__main__":
    main()
