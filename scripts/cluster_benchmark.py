"""Snapshot public source and submit bounded, non-retrying GPU experiments.

Infrastructure values are supplied by the caller, never shipped as credentials.
Only this run's Jobs are created. No existing objects are modified or deleted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def command(args: list[str]) -> str:
    result = subprocess.run(args, text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(
            f"{args[0]} failed rc={result.returncode}: {result.stderr[:1500]}"
        )
    return result.stdout


def dump(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def snapshot(destination: Path) -> dict[str, str]:
    paths = command(["git", "-C", str(ROOT), "ls-files", "-z"]).split("\0")
    paths += command(
        ["git", "-C", str(ROOT), "ls-files", "--others", "--exclude-standard", "-z"]
    ).split("\0")
    hashes = {}
    for relative in sorted(set(paths)):
        if not relative or not relative.startswith(("src/", "scripts/", "tests/", "examples/")):
            continue
        source = ROOT / relative
        if source.is_symlink():
            raise ValueError(f"snapshot refuses symlink: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"snapshot mismatch: {relative}")
        hashes[relative] = digest
    if not hashes:
        raise RuntimeError("empty source snapshot")
    dump(destination / "source-hashes.json", hashes)
    return hashes


def submit(args: argparse.Namespace) -> None:
    root = args.artifact_root.resolve()
    if not str(root).startswith("/3fs/research3fs/data/tmp/"):
        raise ValueError("artifact root must resolve to shared 3FS tmp")
    root.mkdir(parents=True, exist_ok=True)
    active = json.loads(command(["kubectl", "-n", args.namespace, "get", "pods", "-l", "repo=openopt", "-o", "json"]))
    active_cards = sum(
        int(container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
        for pod in active["items"] if pod["status"]["phase"] in ("Pending", "Running")
        for container in pod["spec"]["containers"]
    )
    if active_cards + len(args.sizes) > 3:
        raise RuntimeError(f"three-card cap exceeded: active={active_cards} requested={len(args.sizes)}")
    validate_performance_prerequisite(args)
    if args.stage == "ladder":
        if args.prerequisite is None:
            raise ValueError("ladder requires a passing previous run")
        previous = json.loads((args.prerequisite / "run.json").read_text())
        for item in previous["jobs"]:
            result = json.loads((Path(item["output"]) / "result.json").read_text())
            if result["verdict"] != "PASS" or result.get("nsight", {}).get("status") != "PASS":
                raise ValueError("previous run numerical/Nsight gate did not pass")
    with tempfile.NamedTemporaryFile(dir=root, prefix="write-check-") as probe:
        probe.write(b"openopt shared write check")
        probe.flush()
        if Path(probe.name).read_bytes() != b"openopt shared write check":
            raise RuntimeError("shared write verification failed")
    run = Path(tempfile.mkdtemp(prefix=args.stage + "-", dir=root))
    source = run / "source"
    source.mkdir()
    hashes = snapshot(source)
    outputs = []
    script = {
        "probe": "gpu_probe.py", "regression": "cluster_regression.py",
        "smoke": "benchmark_general_qp.py", "ladder": "benchmark_general_qp.py",
        "campaign": "gpu_qp_campaign.py",
        "comparison": "benchmark_solver_comparison.py",
    }[args.stage]
    for index, size in enumerate(args.sizes):
        output = run / f"case-{index}"
        output.mkdir()
        name = f"openopt-{args.stage}-{run.name.split('-')[-1].replace('_', '')}-{index}"
        labels = {"repo": "openopt", "owner": "infra-openopt", "purpose": args.stage}
        main_command = [
            "python3", str(source / "scripts" / script),
            "--artifact-dir", str(output),
        ]
        if args.stage in ("smoke", "ladder"):
            main_command += ["--size", str(size), "--repeats", str(args.repeats)]
        if args.stage == "campaign":
            main_command += ["--size", str(size), "--repeats", str(args.repeats),
                             "--mode", args.campaign_mode]
        if args.stage == "comparison":
            main_command += ["--size", str(size), "--repeats", str(args.repeats),
                             "--devices", "cpu", "cuda:0"]
        env = [
            {"name": "PYTHONPATH", "value": str(source / "src")},
            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "OMP_NUM_THREADS", "value": str(args.cpu)},
            {"name": "OPENBLAS_NUM_THREADS", "value": str(args.cpu)},
            {"name": "MKL_NUM_THREADS", "value": str(args.cpu)},
            {"name": "OPENOPT_NODE", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
            {"name": "TMPDIR", "value": str(output)},
        ]
        resources = {"cpu": str(args.cpu), "memory": "12Gi", "nvidia.com/gpu": 1}
        pod = {
            "restartPolicy": "Never", "runtimeClassName": "nvidia",
            "serviceAccountName": args.service_account,
            "priorityClassName": args.priority_class,
            "nodeSelector": {"nvidia.com/gpu.product": args.gpu_product},
            "securityContext": {
                "runAsNonRoot": True, "runAsUser": 1002, "runAsGroup": 1002,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [{
                "name": "benchmark", "image": args.image, "imagePullPolicy": "Never",
                "command": main_command, "env": env,
                "securityContext": {
                    "allowPrivilegeEscalation": False, "capabilities": {"drop": ["ALL"]},
                },
                "resources": {"requests": resources, "limits": resources},
                "volumeMounts": [
                    {"name": "threefs", "mountPath": "/3fs", "mountPropagation": "HostToContainer"},
                    {"name": "shared-memory", "mountPath": "/dev/shm"},
                ],
            }],
            "volumes": [
                {"name": "threefs", "persistentVolumeClaim": {"claimName": args.pvc}},
                {"name": "shared-memory", "emptyDir": {"medium": "Memory", "sizeLimit": "512Mi"}},
            ],
        }
        manifest = {
            "apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"name": name, "namespace": args.namespace, "labels": labels},
            "spec": {
                "backoffLimit": 0, "activeDeadlineSeconds": 300,
                "ttlSecondsAfterFinished": 86400,
                "template": {"metadata": {"labels": labels}, "spec": pod},
            },
        }
        manifest_path = run / f"job-{index}.json"
        dump(manifest_path, manifest)
        outputs.append({"job": name, "size": size, "output": str(output), "manifest": str(manifest_path)})
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(), "namespace": args.namespace,
        "stage": args.stage, "source_sha": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip(),
        "source_hashes": hashes, "jobs": outputs, "state": "PREPARED",
    }
    dump(run / "run.json", record)
    try:
        for item in outputs:
            text = command(["kubectl", "create", "-f", item["manifest"]])
            with (run / "submission.log").open("a") as log:
                log.write(text)
            print(text.strip(), flush=True)
    except Exception as error:
        record["state"] = "SUBMISSION_FAILED"
        record["error"] = str(error)
        dump(run / "run.json", record)
        raise
    record["state"] = "SUBMITTED"
    dump(run / "run.json", record)
    print(f"OPENOPT_CLUSTER state=SUBMITTED cards={len(outputs)} run={run}", flush=True)


def collect(run: Path) -> None:
    record = json.loads((run / "run.json").read_text())
    namespace = record["namespace"]
    states = []
    for item in record["jobs"]:
        job = json.loads(command(["kubectl", "-n", namespace, "get", "job", item["job"], "-o", "json"]))
        pods = json.loads(command([
            "kubectl", "-n", namespace, "get", "pods", "-l", f"job-name={item['job']}", "-o", "json",
        ]))
        dump(run / f"{item['job']}-status.json", {"job": job, "pods": pods})
        status = job.get("status", {})
        state = "FAIL" if status.get("failed", 0) else "PASS" if status.get("succeeded", 0) else "RUNNING"
        for pod in pods["items"]:
            name = pod["metadata"]["name"]
            if pod["status"]["phase"] in ("Running", "Succeeded", "Failed"):
                logs = command(["kubectl", "-n", namespace, "logs", name])
                (run / f"{name}.log").write_text(logs)
        if state == "PASS":
            path = Path(item["output"]) / "result.json"
            if not path.is_file() or json.loads(path.read_text()).get("verdict") != "PASS":
                state = "FAIL_ARTIFACT"
        states.append(state)
        print(f"OPENOPT_JOB job={item['job']} state={state} output={item['output']}", flush=True)
    record["collected_at"] = datetime.now(timezone.utc).isoformat()
    record["states"] = states
    dump(run / "collection.json", record)
    if any(state.startswith("FAIL") for state in states):
        raise RuntimeError(f"cluster failure; evidence={run}")


def validate_performance_prerequisite(args: argparse.Namespace) -> None:
    if args.stage not in ("campaign", "comparison") or (
        args.stage == "campaign" and args.campaign_mode == "correctness"
    ):
        return
    if args.prerequisite is None:
        raise ValueError("new performance paths require a passing campaign correctness run")
    previous = json.loads((args.prerequisite / "run.json").read_text())
    if previous["stage"] != "campaign" or len(previous["jobs"]) != 1:
        raise ValueError("prerequisite must be a one-card campaign gate")
    item = previous["jobs"][0]
    result = json.loads((Path(item["output"]) / "result.json").read_text())
    if result.get("verdict") != "PASS" or result.get("mode") != "correctness":
        raise ValueError("campaign correctness prerequisite did not pass")
    if args.acceptance_bundle is None or not args.acceptance_sha256:
        raise ValueError("performance admission requires a pinned independent acceptance bundle")
    if __package__:
        from .accept_qp_campaign import accept
    else:
        from accept_qp_campaign import accept
    accepted = accept(args.acceptance_bundle, args.acceptance_sha256)
    correct_path = str((Path(item["output"]) / "result.json").resolve())
    if not any(str(Path(entry["result_path"]).resolve()) == correct_path
               and entry.get("tests", {}).get("tests", 0) > 0
               for entry in accepted["entries"]):
        raise ValueError("independent acceptance does not cover this correctness prerequisite")
    hashes = previous["source_hashes"]
    if not any(relative.startswith("src/openopt/") for relative in hashes):
        raise ValueError("prerequisite contains no openopt core source")
    for relative, digest in hashes.items():
        if relative.startswith("src/openopt/") and hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f"core changed since campaign prerequisite: {relative}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    launch = sub.add_parser("submit")
    launch.add_argument("--stage", choices=("probe", "smoke", "ladder", "regression", "campaign", "comparison"), required=True)
    launch.add_argument("--campaign-mode", choices=("correctness", "adaptive", "batch", "operators"), default="correctness")
    launch.add_argument("--artifact-root", type=Path, required=True)
    launch.add_argument("--namespace", required=True)
    launch.add_argument("--pvc", required=True)
    launch.add_argument("--service-account", required=True)
    launch.add_argument("--priority-class", required=True)
    launch.add_argument("--image", required=True)
    launch.add_argument("--gpu-product", required=True)
    launch.add_argument("--sizes", type=int, nargs="+", default=[64])
    launch.add_argument("--repeats", type=int, default=5)
    launch.add_argument("--cpu", type=int, default=4)
    launch.add_argument("--prerequisite", type=Path)
    launch.add_argument("--acceptance-bundle", type=Path)
    launch.add_argument("--acceptance-sha256")
    read = sub.add_parser("collect")
    read.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.action == "submit":
        if not 1 <= len(args.sizes) <= 3 or (args.stage in ("probe", "smoke", "regression") and len(args.sizes) != 1):
            parser.error("probe/smoke require one card; any launch is limited to three")
        if args.stage == "campaign" and args.campaign_mode == "correctness" and len(args.sizes) != 1:
            parser.error("campaign correctness is a one-card gate")
        submit(args)
    else:
        collect(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
