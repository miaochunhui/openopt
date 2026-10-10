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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_PREFIXES = ("src/", "scripts/", "tests/", "examples/")
SNAPSHOT_FILES = ("docs/research/refinement.md",)


def proof_cases() -> tuple[str, ...]:
    return proof_module().CASES


def proof_module():
    if __package__:
        from . import gpu_release_proof
    else:
        import gpu_release_proof
    return gpu_release_proof


def freeze_proof_expectations(run: Path, source: Path, cases: list[str]) -> dict:
    """Freeze model bytes before dispatch, never derive them during replay."""
    proof = proof_module()
    frozen_sources = proof.source_map(source)
    proof.same_sources(frozen_sources, proof.source_map(ROOT), "pre-dispatch source")
    models = {}
    for case in cases:
        problem = proof.make_problem(case)
        if case.startswith("jacobi_"):
            P, A = (getattr(problem, name).to_sparse() for name in ("P", "A"))
            if case == "jacobi_csr":
                P, A = P.to_sparse_csr(), A.to_sparse_csr()
            problem = SimpleNamespace(P=P, q=problem.q, A=A, l=problem.l, u=problem.u)
        model = {
            name: proof.save_tensor(run, case + "-original-" + name, getattr(problem, name))
            for name in proof.FIELDS
        }
        models[case] = {"model_sha256": proof.model_hash(model), "model": model}
    path = run / "proof-expectations.json"
    proof.write(path, {"schema_version": 1, "source_sha256": frozen_sources, "cases": models})
    proof.same_sources(frozen_sources, proof.source_map(ROOT), "source after model freeze")
    payload = path.read_bytes()
    return {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}


def submission_units(args: argparse.Namespace) -> list[tuple[int, str | None]]:
    """One independently fail-able case per one-card proof Job."""
    if args.stage == "release-proof":
        cases = args.proof_case
        if not cases or len(cases) > 3 or len(set(cases)) != len(cases):
            raise ValueError("release proof requires one to three distinct explicit cases")
        if not set(cases) <= set(proof_cases()):
            raise ValueError("unknown release proof case")
        if args.target or args.expression:
            raise ValueError("release proof cannot mix test targets or expressions")
        return [(0, case) for case in cases]
    if getattr(args, "proof_case", []):
        raise ValueError("proof cases require the release-proof stage")
    if not 1 <= len(args.sizes) <= 3:
        raise ValueError("any launch is limited to one to three cards")
    if args.stage in ("probe", "smoke", "regression") and len(args.sizes) != 1:
        raise ValueError("probe/smoke/regression require one card")
    if args.stage == "campaign" and args.campaign_mode == "correctness" and len(args.sizes) != 1:
        raise ValueError("campaign correctness is a one-card gate")
    if args.stage == "release-regression" and (len(args.sizes) != 1 or not args.target):
        raise ValueError("release regression requires one card and explicit test targets")
    return [(size, None) for size in args.sizes]


def worker_command(
    args: argparse.Namespace, source: Path, output: Path, size: int, proof_case: str | None,
    expectations: dict | None = None,
) -> list[str]:
    script = {
        "probe": "gpu_probe.py", "regression": "cluster_regression.py",
        "smoke": "benchmark_general_qp.py", "ladder": "benchmark_general_qp.py",
        "campaign": "gpu_qp_campaign.py",
        "comparison": "benchmark_solver_comparison.py",
        "release-regression": "gpu_release_check.py",
        "release-proof": "gpu_release_proof.py",
    }[args.stage]
    result = ["python3", str(source / "scripts" / script)]
    if args.stage == "release-proof":
        if proof_case not in proof_cases() or expectations is None:
            raise ValueError("explicit release proof case and pre-dispatch expectations required")
        return result + [
            "run", "--artifact-dir", str(output), "--device", "cuda:0", "--case", proof_case,
            "--expectations", expectations["path"],
            "--expectations-sha256", expectations["sha256"],
        ]
    result += ["--artifact-dir", str(output)]
    if args.stage in ("smoke", "ladder"):
        result += ["--size", str(size), "--repeats", str(args.repeats)]
    if args.stage == "campaign":
        result += ["--size", str(size), "--repeats", str(args.repeats),
                   "--mode", args.campaign_mode]
    if args.stage == "comparison":
        result += ["--size", str(size), "--repeats", str(args.repeats),
                   "--devices", "cpu", "cuda:0"]
    if args.stage == "release-regression":
        for target in args.target:
            result += ["--target", target]
        if args.expression:
            result += ["--expression", args.expression]
    return result


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
        if not relative or not (
            relative.startswith(SNAPSHOT_PREFIXES) or relative in SNAPSHOT_FILES
        ):
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
    if "tests/test_qp_refinement.py" in hashes and SNAPSHOT_FILES[0] not in hashes:
        raise ValueError("refinement evidence requires its explicit research-document source")
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
    requested = len(args.proof_case) if getattr(args, "stage", "") == "release-proof" else len(args.sizes)
    if active_cards + requested > 3:
        raise RuntimeError(f"three-card cap exceeded: active={active_cards} requested={requested}")
    units = submission_units(args)
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
    identity = None
    if args.stage in ("release-proof", "release-regression"):
        if __package__:
            from .verify import source_snapshot, check_source_consistency
        else:
            from verify import source_snapshot, check_source_consistency
        identity = source_snapshot(ROOT)
        if not identity["clean"]:
            raise ValueError("release GPU submissions require clean committed source")
    hashes = snapshot(source)
    if identity is not None:
        consistency = check_source_consistency(identity, source_snapshot(ROOT))
        if consistency["verdict"] != "PASS":
            raise ValueError(f"source changed during release snapshot: {consistency}")
        if any(identity["sha256"].get(name) != digest for name, digest in hashes.items()):
            raise ValueError("release snapshot bytes differ from frozen source identity")
    expectations = None
    if args.stage == "release-proof":
        expectations = freeze_proof_expectations(run, source, args.proof_case)
        consistency = check_source_consistency(identity, source_snapshot(ROOT))
        if consistency["verdict"] != "PASS":
            raise ValueError(f"source changed during model freeze: {consistency}")
    outputs = []
    for index, (size, proof_case) in enumerate(units):
        output = run / f"case-{index}"
        output.mkdir()
        name = f"openopt-{args.stage}-{run.name.split('-')[-1].replace('_', '')}-{index}"
        labels = {"repo": "openopt", "owner": "infra-openopt", "purpose": args.stage}
        main_command = worker_command(args, source, output, size, proof_case, expectations)
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
        if proof_case is not None:
            outputs[-1]["proof_case"] = proof_case
    record = {
        "started_at": datetime.now(timezone.utc).isoformat(), "namespace": args.namespace,
        "stage": args.stage, "source_sha": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).strip(),
        "source_hashes": hashes, "jobs": outputs, "state": "PREPARED",
    }
    if identity is not None:
        record["source_identity"] = identity
    if expectations is not None:
        record["expectations"] = expectations
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
        collected_logs = []
        for pod in pods["items"]:
            name = pod["metadata"]["name"]
            if pod["status"]["phase"] in ("Running", "Succeeded", "Failed"):
                logs = command(["kubectl", "-n", namespace, "logs", name])
                (run / f"{name}.log").write_text(logs)
                collected_logs.append(logs)
        if state == "PASS":
            path = collected_result(record, item)
            if path is None:
                state = "FAIL_ARTIFACT"
            else:
                item["result_path"] = str(path)
                item["result_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                if record["stage"] == "release-proof":
                    if len(collected_logs) != 1 or not proof_completion_matches(
                        collected_logs[0], path, item["result_sha256"],
                    ):
                        state = "FAIL_EXECUTION_PIN"
        states.append(state)
        print(f"OPENOPT_JOB job={item['job']} state={state} output={item['output']}", flush=True)
    record["collected_at"] = datetime.now(timezone.utc).isoformat()
    record["states"] = states
    dump(run / "collection.json", record)
    if any(state.startswith("FAIL") for state in states):
        raise RuntimeError(f"cluster failure; evidence={run}")


def collected_result(record: dict, item: dict) -> Path | None:
    root = Path(item["output"])
    if record["stage"] == "release-proof":
        paths = sorted(root.glob("cuda-proof-*/result.json"))
        if len(paths) != 1:
            return None
        path = paths[0]
    else:
        path = root / "result.json"
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        return None
    value = json.loads(path.read_text())
    expected_verdict = "SEMANTIC_NUMERICAL_PASS" if record["stage"] == "release-proof" else "PASS"
    if value.get("verdict") != expected_verdict:
        return None
    if record["stage"] == "release-proof" and (
        value.get("returncode") != 0 or value.get("device") != "cuda:0"
        or value.get("requested_cases") != [item["proof_case"]]
        or set(value.get("cases", {})) != {item["proof_case"]}
        or value.get("expectations") != record.get("expectations")
        or value.get("numerical_verdict") != "PASS"
        or value.get("execution_verdict") != "UNVERIFIED"
    ):
        return None
    return path


def proof_completion_matches(log: str, result: Path, expected_sha256: str) -> bool:
    value = json.loads(result.read_text())
    completion = (
        f"OPENOPT_GPU_PROOF verdict=SEMANTIC_NUMERICAL_PASS at={value['finished_at']} "
        f"result={result} result_sha256={expected_sha256}"
    )
    markers = [line for line in log.splitlines() if line.startswith("OPENOPT_GPU_PROOF verdict=")]
    lines = [line for line in log.splitlines() if line.strip()]
    return markers == [completion] and bool(lines) and lines[-1] == completion


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
    launch.add_argument("--stage", choices=("probe", "smoke", "ladder", "regression", "campaign", "comparison", "release-regression", "release-proof"), required=True)
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
    launch.add_argument("--target", action="append", default=[])
    launch.add_argument("--expression", default="")
    launch.add_argument("--proof-case", action="append", default=[])
    launch.add_argument("--prerequisite", type=Path)
    launch.add_argument("--acceptance-bundle", type=Path)
    launch.add_argument("--acceptance-sha256")
    read = sub.add_parser("collect")
    read.add_argument("run", type=Path)
    args = parser.parse_args()
    if args.action == "submit":
        submit(args)
    else:
        collect(args.run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
