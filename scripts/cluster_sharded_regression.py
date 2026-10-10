"""Prepare and coordinate zero-GPU, source-pinned CPU regression Jobs.

Only an explicitly authorized coordinator uses run --submit. collect/check are
read-only on Kubernetes. No installation, retries, deletion or credential reads.
The container probes itself; host Python is never substituted for its runtime.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
from xml.etree import ElementTree as ET

if __package__:
    from . import verify_sharded as shard
else:
    import verify_sharded as shard

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/shard-cluster")
PHASES = {"PROBE": "probe", "SMOKE": "shard", "SHARDS": "shard", "CHECK": "check"}
IMPORTS = ("numpy", "torch", "pytest", "packaging", "scipy", "osqp", "clarabel",
           "highspy", "openopt", "openpo")
FIXTURE_TRANSPORT = "pytest-source-pinned-fixture-transport-v2"


class RuntimeProbeError(shard.GateError):
    """Preserve partial inventory without accepting a failed runtime."""

    def __init__(self, message, inventory):
        super().__init__(message)
        self.inventory = inventory


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".new")
    with temporary.open("xb") as stream:
        stream.write(shard.encoded(value))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def shared(path):
    value = Path(path).expanduser().resolve()
    shard.require(value.is_relative_to(Path("/3fs/research3fs/data/tmp")),
                  "cluster_artifacts_must_be_shared_3fs_tmp")
    return value


def local_command(args, repo, log):
    result = subprocess.run(args, cwd=repo, capture_output=True, text=True, check=False)
    save(log, {"command": args, "returncode": result.returncode,
               "stdout": result.stdout, "stderr": result.stderr})
    shard.require(result.returncode == 0, f"source_command_failed_rc={result.returncode}: {log}")
    return result.stdout.strip()


def freeze(repo, identity, destination, artifacts):
    branch = local_command(["git", "symbolic-ref", "--short", "HEAD"], repo,
                           artifacts / "branch.json")
    local_command(["git", "clone", "--depth", "1", "--single-branch", "--no-checkout",
                   "--branch", branch, repo.as_uri(), str(destination)],
                  repo, artifacts / "clone.json")
    local_command(["git", "checkout", "--detach", identity["commit"]],
                  destination, artifacts / "checkout.json")
    local_command(["git", "remote", "remove", "origin"], destination, artifacts / "remote.json")
    shard.require(shard.snapshot(destination) == identity, "frozen_source_not_identical")
    shard.consistent(repo, identity)


def runtime_probe(repo):
    base = shard.verifier(repo)
    sys.path[:0] = [str(repo / "src"), str(repo), str(repo / "benchmarks")]
    inventory = {
        "python": sys.version, "executable": sys.executable, "dependencies": {},
        "optional_commercial_sdks": {},
        "commercial_qualification": "UNVERIFIED; discovery is not licensed execution",
    }
    for name in (*base.DEPENDENCIES, "packaging"):
        try:
            version = importlib.metadata.version(name)
            inventory["dependencies"][name] = {"status": "FOUND", "version": version}
        except importlib.metadata.PackageNotFoundError:
            inventory["dependencies"][name] = {"status": "MISSING", "version": None}
    for name in ("gurobipy", "cplex", "mosek", "cuopt"):
        inventory["optional_commercial_sdks"][name] = (
            "DISCOVERABLE" if importlib.util.find_spec(name) is not None else "MISSING")
    try:
        found = base.versions()
    except Exception as error:
        raise RuntimeProbeError(f"{type(error).__name__}: {error}", inventory) from error
    modules = {}
    for name in IMPORTS:
        try:
            module = importlib.import_module(name)
        except Exception as error:
            raise RuntimeProbeError(
                f"runtime_import_failed: {name}: {type(error).__name__}: {error}", inventory,
            ) from error
        modules[name] = {"file": str(Path(module.__file__).resolve())}
    for name in ("openopt", "openpo"):
        shard.require(Path(modules[name]["file"]).is_relative_to(repo / "src"),
                      f"runtime_import_not_frozen_source: {name}")
    import torch
    shard.require(not torch.cuda.is_available(), "runtime_observed_gpu")
    git = subprocess.run(["git", "--version"], capture_output=True, text=True, check=False)
    shard.require(git.returncode == 0, f"runtime_git_failed_rc={git.returncode}")
    return {
        "python": sys.version, "executable": sys.executable,
        "implementation": platform.python_implementation(),
        "packages": found, "modules": modules, "git": git.stdout.strip(),
        "cuda_available": False, "inventory": inventory,
    }


def value_pin(value):
    return hashlib.sha256(shard.encoded(value)).hexdigest()


def fixture_transport(plan):
    """Describe runtime source identities without treating host paths as Pod paths."""
    audit = plan["grouping"]["audit"]
    modules = {}
    for node in audit["nodes"]:
        for fixture in node["closure"]:
            identity = fixture["identity"]
            if identity.startswith("builtin:"):
                module = identity.split(":")[1]
                shard.require(module.startswith("_pytest."), "invalid_fixture_runtime_module")
                source = Path(importlib.import_module(module).__file__).resolve()
                shard.require(str(source) in audit["runtime_sha256"],
                              "fixture_runtime_origin_not_in_audit")
                modules[str(source)] = module
    shard.require(set(modules) == set(audit["runtime_sha256"]),
                  "fixture_runtime_source_membership_mismatch")
    return {
        "contract": FIXTURE_TRANSPORT, "opt_in": True,
        "pytest_version": audit["pytest_version"],
        "host_audit_sha256": value_pin(audit),
        "runtime_files": [
            {"host_path": path, "module": modules[path], "sha256": pin}
            for path, pin in sorted(audit["runtime_sha256"].items())
        ],
    }


def transport_contract(plan):
    contract = plan.get("cluster_fixture_transport")
    shard.require(plan["schema_version"] == 2 and isinstance(contract, dict)
                  and contract.get("contract") == FIXTURE_TRANSPORT
                  and contract.get("opt_in") is True, "cluster_fixture_opt_in_required")
    audit = plan["grouping"]["audit"]
    shard.require(contract["host_audit_sha256"] == value_pin(audit)
                  and contract["pytest_version"] == audit["pytest_version"],
                  "transport_host_audit_pin_mismatch")
    files = contract["runtime_files"]
    shard.require(isinstance(files, list), "invalid_transport_runtime_files")
    paths = [item["host_path"] for item in files]
    modules = [item["module"] for item in files]
    shard.require(len(paths) == len(set(paths)) and len(modules) == len(set(modules))
                  and {item["host_path"]: item["sha256"] for item in files}
                  == audit["runtime_sha256"], "transport_runtime_membership_mismatch")
    expected_modules = {
        fixture["identity"].split(":")[1]
        for node in audit["nodes"] for fixture in node["closure"]
        if fixture["identity"].startswith("builtin:")
    }
    shard.require(set(modules) == expected_modules
                  and all(module.startswith("_pytest.") for module in modules),
                  "transport_runtime_module_mismatch")
    for name, isolation in audit["isolated_builtins"].items():
        shard.require(any(item["host_path"] == isolation["origin"]
                          and item["module"] == isolation["module"]
                          and item["sha256"] == isolation["sha256"] for item in files),
                      f"transport_builtin_origin_mismatch: {name}")
    sidecar = shard.read(plan["collection"]["path"], plan["collection"]["sha256"])
    shard.require(sidecar == {"nodeids": plan["collection"]["nodeids"],
                              "fixture_audit": audit}, "transport_full_collection_mismatch")
    return contract


def mapped_plan(plan, mapping, collection_path):
    """Only runtime installation paths change; dependencies/groups/costs do not."""
    contract = transport_contract(plan)
    shard.require(len(mapping) == len(contract["runtime_files"]), "transport_mapping_size_mismatch")
    for expected, actual in zip(contract["runtime_files"], mapping):
        shard.require(all(actual.get(key) == value for key, value in expected.items())
                      and isinstance(actual.get("runtime_path"), str)
                      and Path(actual["runtime_path"]).is_absolute(),
                      "transport_mapping_identity_mismatch")
    replacements = {item["host_path"]: item["runtime_path"] for item in mapping}
    shard.require(len(set(replacements.values())) == len(replacements),
                  "transport_runtime_path_collision")
    result = copy.deepcopy(plan)
    audit = result["grouping"]["audit"]
    audit["runtime_sha256"] = {
        replacements[path]: pin for path, pin in audit["runtime_sha256"].items()
    }
    for isolation in audit["isolated_builtins"].values():
        isolation["origin"] = replacements[isolation["origin"]]
    for node in audit["nodes"]:
        for fixture in node["closure"]:
            isolation = fixture["isolated_builtin"]
            if isolation is not None:
                isolation["origin"] = replacements[isolation["origin"]]
    collection = {"nodeids": result["collection"]["nodeids"], "fixture_audit": audit}
    result["collection"].update(path=str(collection_path), sha256=value_pin(collection))
    # The source transport descriptor is deliberately retained for host replay.
    return result, collection


def local_runtime_mapping(plan):
    contract = transport_contract(plan)
    import pytest
    shard.require(pytest.__version__ == contract["pytest_version"],
                  "fixture_transport_pytest_version_drift")
    mapping = []
    for item in contract["runtime_files"]:
        module = importlib.import_module(item["module"])
        path = Path(module.__file__).resolve()
        origin = Path(pytest.__file__).resolve().parent.parent / (
            item["module"].replace(".", "/") + ".py")
        shard.require(module.__name__ == item["module"] and path == origin,
                      f"fixture_transport_runtime_origin_drift: {item['module']}")
        shard.require(path.suffix == ".py" and shard.digest(path) == item["sha256"],
                      f"fixture_transport_runtime_source_drift: {item['module']}")
        mapping.append(dict(item, runtime_path=str(path)))
    return mapping


def execution_plan(config, plan, runtime):
    if plan["schema_version"] == 1:
        return plan, Path(config["plan"]), config["plan_sha256"]
    shard.require(runtime is not None, "fixture_runtime_probe_required")
    path = Path(runtime["execution_plan"])
    shard.require(path == Path(config["artifacts"]) / "worker-probe/runtime-plan.json",
                  "runtime_plan_location_mismatch")
    effective = shard.read(path, runtime["execution_plan_sha256"])
    collection_path = path.parent / "runtime-collection.json"
    expected, collection = mapped_plan(plan, runtime["fixture_runtime_mapping"], collection_path)
    shard.require(effective == expected, "runtime_plan_semantic_drift")
    shard.require(shard.read(collection_path, effective["collection"]["sha256"]) == collection,
                  "runtime_collection_transport_drift")
    return effective, path, runtime["execution_plan_sha256"]


def worker(plan_path, plan_pin, phase, output, shard_id=None, runtime_path=None,
           runtime_pin=None, report_path=None, report_pin=None):
    output.mkdir(parents=True, exist_ok=False)
    report = {"schema_version": 1, "phase": phase, "started_at": stamp(), "verdict": "RUNNING"}
    try:
        plan = shard.read(plan_path, plan_pin)
        if plan["schema_version"] == 1:
            shard.validate_plan(plan)
        repo = Path(plan["repo"])
        shard.consistent(repo, plan["source"])
        report.update(source=plan["source"], runtime=runtime_probe(repo), plan_sha256=plan_pin)
        effective, effective_path, effective_pin = plan, plan_path, plan_pin
        if plan["schema_version"] == 2:
            mapping = local_runtime_mapping(plan)
            report["fixture_runtime_mapping"] = mapping
            if phase == "probe":
                effective_path = output / "runtime-plan.json"
                effective, collection = mapped_plan(
                    plan, mapping, output / "runtime-collection.json")
                shard.save(output / "runtime-collection.json", collection)
                shard.save(effective_path, effective)
                effective_pin = shard.digest(effective_path)
            else:
                expected = shard.read(runtime_path, runtime_pin)
                shard.require(expected["plan_sha256"] == plan_pin
                              and expected["source"] == plan["source"],
                              "fixture_probe_source_plan_mismatch")
                shard.require(expected["fixture_runtime_mapping"] == mapping,
                              "fixture_runtime_mapping_drift")
                effective, effective_path, effective_pin = execution_plan(
                    {"plan": str(plan_path), "plan_sha256": plan_pin,
                     "artifacts": str(plan_path.parent)}, plan, expected)
            report.update(execution_plan=str(effective_path), execution_plan_sha256=effective_pin)
        shard.validate_plan(effective)
        if phase != "probe":
            expected = shard.read(runtime_path, runtime_pin)
            shard.require(expected["verdict"] == "PASS"
                          and expected["phase"] == "probe", "runtime_probe_not_pass")
            shard.require(report["runtime"] == expected["runtime"], "container_runtime_drift")
            report["runtime_sha256"] = runtime_pin
        if phase == "probe":
            collection = output / "collection"
            collection.mkdir()
            shard.recollect_plan(repo, effective, collection)
            report["collection_sha256"] = shard.digest(collection / "collection.json")
        elif phase == "shard":
            shard.require(type(shard_id) is int and 0 <= shard_id < len(plan["shards"]),
                          "invalid_worker_shard_id")
            result_dir = output.parent / f"shard-{shard_id}"
            result_dir.mkdir(exist_ok=False)
            if plan["schema_version"] == 2:
                collection = output / "collection"
                collection.mkdir()
                shard.recollect_plan(repo, effective, collection)
                report["collection_sha256"] = shard.digest(collection / "collection.json")
            report["shard"] = shard.launch_shard(
                repo, effective, effective["shards"][shard_id], result_dir)
        elif phase == "check":
            shard.require(report_path is not None and report_pin is not None,
                          "caller_pinned_report_required")
            report["acceptance_dir"] = str(shard.check(
                repo, effective_path, effective_pin, report_path, report_pin, output / "replay"))
            report["acceptance_sha256"] = shard.digest(
                Path(report["acceptance_dir"]) / "acceptance.json")
        else:
            raise shard.GateError("invalid_worker_phase")
        report["source_consistency"] = shard.consistent(repo, plan["source"])
        report["verdict"] = "PASS"
    except Exception as error:
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        if isinstance(error, RuntimeProbeError):
            report["runtime_probe_inventory"] = error.inventory
        raise
    finally:
        report["finished_at"] = stamp()
        save(output / "worker.json", report)
        print(f"OPENOPT_CLUSTER_WORKER verdict={report['verdict']} phase={phase} "
              f"report={output / 'worker.json'}", flush=True)
    return report


def prepare(repo, plan_path, plan_pin, artifact_root, image, python="python3",
            namespace="devtest", pvc="devtest-3fs", service_account="devbox",
            priority_class="devtest-batch", cpu=4, memory="16Gi", node=None,
            fixture_aware=False):
    artifacts = shard.new_artifacts(shared(artifact_root), repo, "cluster-")
    source_plan = shard.read(plan_path, plan_pin)
    shard.validate_plan(source_plan)
    shard.require(type(fixture_aware) is bool and fixture_aware == (
        source_plan["schema_version"] == 2), "cluster_fixture_explicit_opt_in_required")
    shard.require({"scripts/verify.py", "scripts/verify_sharded.py",
                   "scripts/cluster_sharded_regression.py"}.issubset(
                       source_plan["source"]["sha256"]), "frozen_plan_missing_worker_entry")
    shard.require(Path(source_plan["repo"]).resolve() == repo, "source_plan_repo_mismatch")
    shard.consistent(repo, source_plan["source"])
    shard.require(isinstance(image, str) and image and not any(c.isspace() for c in image),
                  "invalid_image")
    shard.require(python and not any(c.isspace() for c in python)
                  and "/hf3fs" not in python and "/3fs" not in python,
                  "container_python_must_not_be_host_environment")
    shard.require(type(cpu) is int and 1 <= cpu <= 32, "invalid_cpu_limit")
    shard.require(re.fullmatch(r"[1-9][0-9]*(?:Mi|Gi)", memory) is not None,
                  "invalid_memory_limit")
    source = artifacts / "source"
    freeze(repo, source_plan["source"], source, artifacts)
    sidecar = artifacts / "collection.json"
    collection = shard.read(source_plan["collection"]["path"], source_plan["collection"]["sha256"])
    shard.require(collection["nodeids"] == source_plan["collection"]["nodeids"],
                  "source_collection_membership_mismatch")
    shard.save(sidecar, collection)
    plan = dict(source_plan)
    plan.update(repo=str(source), collection={
        "nodeids": source_plan["collection"]["nodeids"],
        "path": str(sidecar), "sha256": shard.digest(sidecar)})
    if fixture_aware:
        plan["cluster_fixture_transport"] = fixture_transport(source_plan)
        transport_contract(plan)
    shard.validate_plan(plan)
    shard.save(artifacts / "plan.json", plan)
    config = {
        "schema_version": plan["schema_version"], "source_repo": str(repo), "source": plan["source"],
        "source_plan": str(plan_path.resolve()), "source_plan_sha256": plan_pin,
        "plan": str(artifacts / "plan.json"), "plan_sha256": shard.digest(artifacts / "plan.json"),
        "image": image, "python": python, "namespace": namespace, "pvc": pvc,
        "service_account": service_account, "priority_class": priority_class,
        "cpu": cpu, "memory": memory, "node": node, "artifacts": str(artifacts),
        "gpu_count": 0, "created_at": stamp(),
        "smoke_id": min(plan["shards"], key=lambda item: (-item["predicted_test_s"], item["id"]))["id"],
    }
    if fixture_aware:
        config["fixture_transport_contract"] = FIXTURE_TRANSPORT
    shard.save(artifacts / "cluster-plan.json", config)
    save(artifacts / "run.json", {
        "state": "PREPARED", "jobs": [], "events": [], "first_failure": None,
        "config_sha256": shard.digest(artifacts / "cluster-plan.json"),
    })
    print(f"OPENOPT_CLUSTER_PLAN state=PREPARED plan={artifacts / 'cluster-plan.json'} "
          f"sha256={shard.digest(artifacts / 'cluster-plan.json')}", flush=True)
    return artifacts


def manifest(config, phase, item_id, runtime_pin=None, report_pin=None):
    artifacts = Path(config["artifacts"])
    plan = shard.read(config["plan"], config["plan_sha256"])
    label = phase if item_id is None else f"{phase}-{item_id}"
    suffix = re.sub(r"[^a-z0-9-]", "-", artifacts.name[-8:].lower())
    name = f"openopt-cpu-{suffix}-{label}"
    output = artifacts / f"worker-{label}"
    command = [config["python"], "-B",
               str(Path(plan["repo"]) / "scripts/cluster_sharded_regression.py"),
               "worker", "--plan", config["plan"], "--plan-sha256", config["plan_sha256"],
               "--phase", phase, "--output", str(output)]
    if item_id is not None:
        command.extend(["--shard-id", str(item_id)])
    if phase != "probe":
        shard.require(runtime_pin is not None, "pinned_runtime_required")
        command.extend(["--runtime", str(artifacts / "worker-probe/worker.json"),
                        "--runtime-sha256", runtime_pin])
    if phase == "check":
        shard.require(report_pin is not None, "pinned_report_required")
        command.extend(["--report", str(artifacts / "verification.json"),
                        "--report-sha256", report_pin])
    resources = {"cpu": str(config["cpu"]), "memory": config["memory"]}
    env = shard.cpu_env(Path(plan["repo"]), output)
    keys = ("CUDA_VISIBLE_DEVICES", "PYTHONDONTWRITEBYTECODE",
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD", "PYTEST_ADDOPTS", "PYTEST_PLUGINS",
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "TMPDIR", "TRITON_CACHE_DIR", "PYTHONPATH")
    pod = {
        "restartPolicy": "Never", "automountServiceAccountToken": False,
        "serviceAccountName": config["service_account"],
        "priorityClassName": config["priority_class"],
        "securityContext": {"runAsNonRoot": True, "runAsUser": 1002,
                            "runAsGroup": 1002, "seccompProfile": {"type": "RuntimeDefault"}},
        "containers": [{
            "name": "regression", "image": config["image"], "imagePullPolicy": "Never",
            "workingDir": plan["repo"], "command": command,
            "env": [{"name": key, "value": env[key]} for key in keys]
                   + [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "void"}],
            "securityContext": {"allowPrivilegeEscalation": False,
                                "capabilities": {"drop": ["ALL"]}},
            "resources": {"requests": resources, "limits": resources},
            "volumeMounts": [{"name": "threefs", "mountPath": "/3fs",
                             "mountPropagation": "HostToContainer"},
                            {"name": "shared-memory", "mountPath": "/dev/shm"}],
        }],
        "volumes": [{"name": "threefs", "persistentVolumeClaim": {"claimName": config["pvc"]}},
                    {"name": "shared-memory", "emptyDir": {"medium": "Memory",
                                                         "sizeLimit": "512Mi"}}],
    }
    if config["node"]:
        pod["nodeName"] = config["node"]
    labels = {"repo": "openopt", "component": "cpu-regression", "campaign": artifacts.name}
    spec = {"apiVersion": "batch/v1", "kind": "Job",
            "metadata": {"name": name, "namespace": config["namespace"], "labels": labels},
            "spec": {"backoffLimit": 0,
                     "activeDeadlineSeconds": int(math.ceil(plan["shard_timeout_s"]))
                     if phase == "shard" else 300,
                     "template": {"metadata": {"labels": labels}, "spec": pod}}}
    return spec, {"name": name, "phase": phase, "id": item_id, "output": str(output),
                  "manifest": str(artifacts / f"job-{label}.json")}


class Kubernetes:
    """Credentials remain in the existing caller environment, never read here."""

    def __init__(self, namespace):
        self.namespace = namespace

    def raw(self, args, artifacts):
        artifacts.mkdir(parents=True, exist_ok=False)
        command = ["kubectl", "--request-timeout=8s", "-n", self.namespace, *args]
        try:
            result = subprocess.run(command, capture_output=True, check=False, timeout=10)
        except subprocess.TimeoutExpired as error:
            (artifacts / "stdout.log").write_bytes(error.stdout or b"")
            (artifacts / "stderr.log").write_bytes(error.stderr or b"")
            shard.save(artifacts / "query.json", {
                "args": args, "returncode": None, "timeout": True,
            })
            raise shard.GateError(f"kubectl_timeout: {artifacts}") from error
        # No raw config, Secret or arbitrary object queries exist in this tool.
        (artifacts / "stdout.log").write_bytes(result.stdout)
        (artifacts / "stderr.log").write_bytes(result.stderr)
        shard.save(artifacts / "query.json", {"args": args, "returncode": result.returncode})
        shard.require(result.returncode == 0,
                      f"kubectl_failed_rc={result.returncode}: {artifacts}")
        return result.stdout

    def query(self, args, artifacts):
        return json.loads(self.raw(args, artifacts))

    def logs(self, name, artifacts):
        return self.raw(["logs", name, "-c", "regression"], artifacts)


def subset(expected, actual):
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and subset(value, actual[key]) for key, value in expected.items())
    if isinstance(expected, list):
        return isinstance(actual, list) and len(expected) == len(actual) and all(
            subset(first, last) for first, last in zip(expected, actual))
    return expected == actual


def terminal_job(job, pods, expected, uid, image_id=None):
    shard.require(job["metadata"]["uid"] == uid, "job_uid_mismatch")
    shard.require(subset(expected, job), "job_manifest_drift")
    expected_pod = expected["spec"]["template"]["spec"]

    def cpu_only(pod_spec):
        shard.require("runtimeClassName" not in pod_spec and
                      not pod_spec.get("initContainers") and
                      not pod_spec.get("ephemeralContainers"), "unexpected_cpu_pod_runtime")
        shard.require(len(pod_spec["containers"]) == 1 and
                      pod_spec["containers"][0]["resources"] ==
                      expected_pod["containers"][0]["resources"], "cpu_resources_manifest_drift")

    cpu_only(job["spec"]["template"]["spec"])
    status = job.get("status", {})
    shard.require(not status.get("failed") and not any(
        item.get("type") == "Failed" and item.get("status") == "True"
        for item in status.get("conditions", [])), "kubernetes_job_failed")
    candidates = pods.get("items", [])
    shard.require(len(candidates) <= 1, "unexpected_retry_or_extra_pod")
    if candidates:
        pod = candidates[0]
        shard.require(any(owner.get("uid") == uid and owner.get("controller") is True
                          for owner in pod["metadata"].get("ownerReferences", [])),
                      "pod_owner_mismatch")
        shard.require(subset(expected["spec"]["template"]["spec"], pod["spec"]),
                      "pod_manifest_drift")
        cpu_only(pod["spec"])
        states = pod.get("status", {}).get("containerStatuses", [])
        shard.require(len(states) <= 1, "extra_container_status")
        if states:
            container = states[0]
            shard.require(container.get("restartCount") == 0, "container_restarted")
            stopped = container.get("state", {}).get("terminated")
            if stopped:
                shard.require(stopped.get("exitCode") == 0,
                              f"container_failed_rc={stopped.get('exitCode')}")
        shard.require(pod.get("status", {}).get("phase") != "Failed", "pod_failed")
    complete = status.get("succeeded") == 1 and any(
        item.get("type") == "Complete" and item.get("status") == "True"
        for item in status.get("conditions", []))
    if not complete:
        return None
    shard.require(len(candidates) == 1 and len(states) == 1, "complete_job_missing_pod")
    shard.require(pod["status"]["phase"] == "Succeeded"
                  and container.get("state", {}).get("terminated", {}).get("exitCode") == 0,
                  "complete_job_without_successful_container")
    observed_image = container.get("imageID", "")
    shard.require(re.search(r"sha256:[a-f0-9]{64}", observed_image) is not None,
                  "missing_resolved_image_digest")
    if image_id is not None:
        shard.require(observed_image == image_id, "container_image_drift")
    return {"image_id": observed_image, "pod_uid": pod["metadata"]["uid"],
            "node": pod["spec"].get("nodeName"), "exit_code": 0}


def load_config(path, pin):
    config = shard.read(path, pin)
    shard.require(config["schema_version"] in (1, 2) and config["gpu_count"] == 0,
                  "invalid_cluster_config")
    shard.require(Path(config["artifacts"]).resolve() == path.parent.resolve(),
                  "cluster_config_location_mismatch")
    plan = shard.read(config["plan"], config["plan_sha256"])
    shard.validate_plan(plan)
    shard.require(config["schema_version"] == plan["schema_version"],
                  "cluster_plan_version_mismatch")
    if plan["schema_version"] == 2:
        shard.require(config.get("fixture_transport_contract") == FIXTURE_TRANSPORT,
                      "cluster_fixture_opt_in_required")
        transport_contract(plan)
        original = shard.read(config["source_plan"], config["source_plan_sha256"])
        expected = copy.deepcopy(original)
        expected.update(repo=plan["repo"], collection=plan["collection"])
        expected["cluster_fixture_transport"] = fixture_transport(original)
        shard.require(plan == expected, "transport_source_plan_semantic_drift")
    shard.require(plan["source"] == config["source"], "cluster_plan_source_mismatch")
    shard.consistent(Path(config["source_repo"]), config["source"])
    shard.consistent(Path(plan["repo"]), config["source"])
    return config, plan


def stage_members(config, plan, state):
    seen = [(job["phase"], job["id"]) for job in state["jobs"]]
    shard.require(len(seen) == len(set(seen)), "duplicate_stage_job")
    shard.require(len({job["name"] for job in state["jobs"]}) == len(seen)
                  and len({job["uid"] for job in state["jobs"]}) == len(seen),
                  "duplicate_job_name_or_uid")
    passed = {
        "PREPARED": [],
        "PROBE": [("probe", None)],
        "PROBE_PASSED": [("probe", None)],
        "SMOKE": [("probe", None), ("shard", config["smoke_id"])],
        "SMOKE_PASSED": [("probe", None), ("shard", config["smoke_id"])],
        "SHARDS": [("probe", None), *[("shard", item["id"]) for item in plan["shards"]]],
        "SHARDS_PASSED": [("probe", None), *[("shard", item["id"]) for item in plan["shards"]]],
        "CHECK": [("probe", None), *[("shard", item["id"]) for item in plan["shards"]],
                  ("check", None)],
        "PASS": [("probe", None), *[("shard", item["id"]) for item in plan["shards"]],
                 ("check", None)],
    }
    shard.require(state["state"] in passed and Counter(seen) == Counter(passed[state["state"]]),
                  "stage_job_membership_mismatch")
    active_phase = PHASES.get(state["state"])
    for job in state["jobs"]:
        if job["phase"] != active_phase:
            shard.require(job.get("accepted") is True, "prior_stage_not_accepted")


def locked_update(path, expected_pin, new):
    with (path.parent / "coordinator.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        shard.require(shard.digest(path) == expected_pin, "coordinator_state_changed")
        save(path, new)


def stop(config, state, error, coordinate):
    if state["first_failure"] is None:
        state["first_failure"] = {"at": stamp(), "error": f"{type(error).__name__}: {error}",
                                  "coordinate": coordinate}
    state["state"] = "BLOCKED"
    save(Path(config["artifacts"]) / "run.json", state)


def admit(config_path, config_pin, submit=False, client=None):
    shard.require(submit is True, "coordinator_explicit_submit_required")
    config, plan = load_config(config_path, config_pin)
    artifacts = Path(config["artifacts"])
    record_path = artifacts / "run.json"
    state_pin = shard.digest(record_path)
    state = shard.read(record_path)
    shard.require(state["config_sha256"] == config_pin, "run_config_pin_mismatch")
    shard.require(state["first_failure"] is None, "admission_blocked_by_first_failure")
    stage_members(config, plan, state)
    try:
        verify_accepted(config, plan, state)
    except Exception as error:
        stop(config, state, error, "prior_stage_evidence")
        raise
    stages = {"PREPARED": ("PROBE", [None]), "PROBE_PASSED": ("SMOKE", [config["smoke_id"]]),
              "SMOKE_PASSED": ("SHARDS", [item["id"] for item in plan["shards"]
                                        if item["id"] != config["smoke_id"]]),
              "SHARDS_PASSED": ("CHECK", [None])}
    shard.require(state["state"] in stages, f"stage_not_ready_for_admission: {state['state']}")
    stage, ids = stages[state["state"]]
    if stage == "SHARDS" and not ids:
        state["report_sha256"] = combined_report(config, plan, state["jobs"])
        state["state"] = "SHARDS_PASSED"
        locked_update(record_path, state_pin, state)
        return state
    prepared = []
    for item_id in ids:
        spec, item = manifest(config, PHASES[stage], item_id, state.get("runtime_sha256"),
                              state.get("report_sha256"))
        shard.save(item["manifest"], spec)
        item["manifest_sha256"] = shard.digest(item["manifest"])
        prepared.append(item)
    state["state"] = "ADMITTING"
    state["events"].append({"at": stamp(), "stage": stage, "action": "reserve"})
    locked_update(record_path, state_pin, state)
    client = client or Kubernetes(config["namespace"])
    try:
        for item in prepared:
            receipt = client.query(["create", "-f", item["manifest"], "-o", "json"],
                                   artifacts / f"create-{item['name']}")
            shard.require(receipt["metadata"]["name"] == item["name"]
                          and receipt["metadata"]["namespace"] == config["namespace"],
                          "submission_receipt_mismatch")
            item["uid"] = receipt["metadata"]["uid"]
            state["jobs"].append(item)
            save(record_path, state)
        state["state"] = stage
        save(record_path, state)
    except Exception as error:
        stop(config, state, error, f"admission:{stage}")
        raise
    print(f"OPENOPT_CLUSTER_RUN state={stage} jobs={len(prepared)} gpu=0", flush=True)
    return state


def reaccept_result(config, plan, job, runtime):
    worker_path = Path(job["output"]) / "worker.json"
    result = shard.read(worker_path)
    shard.require(result["verdict"] == "PASS" and result["phase"] == job["phase"],
                  "worker_not_pass")
    shard.require(result["plan_sha256"] == config["plan_sha256"]
                  and result["source"] == plan["source"], "worker_source_mismatch")
    expected = shard.verifier(Path(plan["repo"])).check_source_consistency(
        plan["source"], plan["source"])
    shard.require(result["source_consistency"] == expected, "worker_source_not_consistent")
    effective = plan
    if plan["schema_version"] == 2:
        effective, effective_path, effective_pin = execution_plan(
            config, plan, result if job["phase"] == "probe" else runtime)
        shard.require(result["execution_plan"] == str(effective_path)
                      and result["execution_plan_sha256"] == effective_pin,
                      "worker_execution_plan_pin_mismatch")
        if runtime is not None:
            shard.require(result["fixture_runtime_mapping"] == runtime["fixture_runtime_mapping"],
                          "worker_fixture_runtime_mapping_drift")
    if runtime is not None:
        shard.require(result["runtime"] == runtime["runtime"], "worker_runtime_drift")
        shard.require(result["runtime_sha256"] == shard.digest(
            Path(config["artifacts"]) / "worker-probe/worker.json"), "worker_runtime_pin_mismatch")
    if job["phase"] == "probe" or job["phase"] == "shard" and plan["schema_version"] == 2:
        collection_path = Path(job["output"]) / "collection/collection.json"
        collection = shard.read(collection_path, result["collection_sha256"])
        shard.require(collection["nodeids"] == plan["collection"]["nodeids"],
                      "probe_collection_membership_mismatch")
        if plan["schema_version"] == 2:
            shard.require(collection == shard.read(effective["collection"]["path"],
                          effective["collection"]["sha256"]), "container_fixture_audit_drift")
    if job["phase"] == "shard":
        item = result["shard"]
        folder = Path(config["artifacts"]) / f"shard-{job['id']}"
        shard.require(Path(item["artifact_dir"]) == folder, "shard_output_mismatch")
        process = shard.read(folder / "process.json", item["process_sha256"])
        # Interpret the container executable from its probe, never host sys.executable.
        expected_command = shard.shard_command(Path(plan["repo"]), plan["shards"][job["id"]], folder)
        expected_command[0] = runtime["runtime"]["executable"]
        shard.require(process["command"] == expected_command and process["returncode"] == 0
                      and process["timeout"] is False, "container_process_mismatch")
        shard.require(process["timeout_s"] == plan["shard_timeout_s"]
                      and type(process["pid"]) is int and process["pid"] > 0
                      and math.isfinite(process["wall_s"]) and process["wall_s"] > 0,
                      "container_process_measurement_mismatch")
        report = shard.read(item["verification"], item["verification_sha256"])
        shard.require(report["source"] == plan["source"]
                      and report["source_consistency"] == expected
                      and report["verdict"] == "PASS" and report["cuda_available"] is False,
                      "container_verification_not_pass")
        shard.require(report["dependencies"] == runtime["runtime"]["packages"]
                      and report["python"] == runtime["runtime"]["python"]
                      and report["python_executable"] == runtime["runtime"]["executable"],
                      "container_verification_runtime_mismatch")
        shard.require(report["targets"] == plan["shards"][job["id"]]["targets"]
                      and report["mode"] == "tests" and report["coverage"] == "explicit_targets",
                      "container_target_mismatch")
        shard.require(Path(item["verification"]).parent.parent == folder
                      and Path(item["junit"]) == Path(item["verification"]).parent / "pytest.xml",
                      "container_sidecar_location_mismatch")
        shard.require(shard.digest(item["junit"]) == item["junit_sha256"], "junit_pin_mismatch")
        records, counts = shard.parse_junit(item["junit"])
        expected_nodes = plan["shards"][job["id"]]["nodeids"]
        shard.require(Counter(record["key"] for record in records) ==
                      Counter(shard.junit_key(node) for node in expected_nodes),
                      "container_junit_membership_mismatch")
        shard.require(counts == item["tests"] == report["tests"]
                      and not counts["errors"] and not counts["failures"],
                      "container_junit_count_mismatch")
        shard.require(Counter(item["nodeids"]) == Counter(expected_nodes), "worker_nodeid_mismatch")
    return result, shard.digest(worker_path)


def worker_failure(job):
    path = Path(job["output"]) / "worker.json"
    if path.is_file():
        result = shard.read(path)
        if result.get("verdict") == "FAIL":
            raise shard.GateError(
                f"worker_failed: {result.get('error', 'missing_failure_reason')}; evidence={path}")


def combined_report(config, plan, jobs):
    artifacts = Path(config["artifacts"])
    runtime = shard.read(artifacts / "worker-probe/worker.json")
    _, effective_path, effective_pin = execution_plan(config, plan, runtime)
    results = [shard.read(Path(job["output"]) / "worker.json")["shard"]
               for job in jobs if job["phase"] == "shard"]
    summary = {
        "schema_version": 1, "verdict": "PASS", "mode": "tests", "coverage": "tests",
        "targets": ["tests"], "source": plan["source"], "shards": results,
        "tests": shard.combine(plan, results), "plan": str(effective_path),
        "plan_sha256": effective_pin, "failures": [], "not_admitted": [],
        "qualification": "full CPU correctness only; independent container replay pending",
        "source_consistency": shard.consistent(Path(plan["repo"]), plan["source"]),
        "finished_at": stamp(), "speedup": None,
    }
    if plan["schema_version"] == 2:
        summary.update(plan_schema_version=2, source_transport_plan_sha256=config["plan_sha256"],
                       fixture_transport_contract=FIXTURE_TRANSPORT)
    combined = ET.Element("testsuites", name="openopt cluster CPU regression")
    for item in sorted(results, key=lambda value: value["id"]):
        root = ET.parse(item["junit"]).getroot()
        for suite in ([root] if root.tag == "testsuite" else root.findall("testsuite")):
            combined.append(suite)
    ET.ElementTree(combined).write(artifacts / "pytest.xml", encoding="utf-8", xml_declaration=True)
    summary["junit_sha256"] = shard.digest(artifacts / "pytest.xml")
    shard.save(artifacts / "verification.json", summary)
    return shard.digest(artifacts / "verification.json")


def verify_accepted(config, plan, state):
    artifacts = Path(config["artifacts"])
    runtime = shard.read(artifacts / "worker-probe/worker.json",
                         state["runtime_sha256"]) if "runtime_sha256" in state else None
    for job in state["jobs"]:
        if not job.get("accepted"):
            continue
        shard.require(shard.digest(Path(job["output"]) / "worker.json") ==
                      job["worker_sha256"], "worker_pin_mismatch")
        evidence = job["cluster_evidence"]
        observed = shard.read(evidence["job"], evidence["job_sha256"])
        pods = shard.read(evidence["pods"], evidence["pods_sha256"])
        spec = shard.read(job["manifest"], job["manifest_sha256"])
        shard.require(terminal_job(observed, pods, spec, job["uid"], state["image_id"])
                      == job["hardware"], "archived_cluster_evidence_mismatch")
        reaccept_result(config, plan, job, runtime if job["phase"] != "probe" else None)


def collect(config_path, config_pin, client=None):
    config, plan = load_config(config_path, config_pin)
    artifacts = Path(config["artifacts"])
    state_path = artifacts / "run.json"
    state_pin, state = shard.digest(state_path), shard.read(state_path)
    shard.require(state["config_sha256"] == config_pin, "run_config_pin_mismatch")
    shard.require(state["state"] in PHASES, f"stage_not_collectable: {state['state']}")
    stage_members(config, plan, state)
    phase = PHASES[state["state"]]
    active = [job for job in state["jobs"] if job["phase"] == phase and not job.get("accepted")]
    shard.require(bool(active), "no_active_stage_jobs")
    client = client or Kubernetes(config["namespace"])
    sample = Path(tempfile.mkdtemp(prefix="collect-", dir=artifacts))
    runtime = shard.read(artifacts / "worker-probe/worker.json",
                         state["runtime_sha256"]) if "runtime_sha256" in state else None
    original_stage = state["state"]
    reservation = dict(state, state="COLLECTING")
    locked_update(state_path, state_pin, reservation)
    state_pin = shard.digest(state_path)
    try:
        verify_accepted(config, plan, state)
        for job in active:
            spec = shard.read(job["manifest"], job["manifest_sha256"])
            observed = client.query(["get", "job", job["name"], "-o", "json"],
                                    sample / f"{job['name']}-job")
            pods = client.query(["get", "pods", "-l", f"job-name={job['name']}", "-o", "json"],
                                sample / f"{job['name']}-pods")
            job_path = sample / f"{job['name']}.json"
            pods_path = sample / f"{job['name']}-pods.json"
            shard.save(job_path, observed)
            shard.save(pods_path, pods)
            for pod in pods.get("items", []):
                if pod.get("status", {}).get("phase") in ("Running", "Succeeded", "Failed"):
                    try:
                        client.logs(pod["metadata"]["name"],
                                    sample / f"{pod['metadata']['name']}-logs")
                    except Exception:
                        worker_failure(job)
                        raise
            worker_failure(job)
            evidence = terminal_job(observed, pods, spec, job["uid"], state.get("image_id"))
            if evidence is None:
                continue
            result, pin = reaccept_result(config, plan, job, runtime)
            job.update(accepted=True, worker_sha256=pin, hardware=evidence)
            job["cluster_evidence"] = {
                "job": str(job_path), "job_sha256": shard.digest(job_path),
                "pods": str(pods_path), "pods_sha256": shard.digest(pods_path),
            }
            if phase == "probe":
                state.update(runtime_sha256=pin, image_id=evidence["image_id"])
            if phase == "check":
                audit_path = Path(result["acceptance_dir"]) / "acceptance.json"
                audit = shard.read(audit_path, result["acceptance_sha256"])
                shard.require(audit["verdict"] == "PASS" and audit["tests"] ==
                              shard.read(artifacts / "verification.json",
                                         state["report_sha256"])["tests"],
                              "independent_container_replay_not_pass")
                state["acceptance"] = str(audit_path)
                state["acceptance_sha256"] = result["acceptance_sha256"]
        if all(job.get("accepted") for job in active):
            state["state"] = {"PROBE": "PROBE_PASSED", "SMOKE": "SMOKE_PASSED",
                              "SHARDS": "SHARDS_PASSED", "CHECK": "PASS"}[original_stage]
            if state["state"] == "SHARDS_PASSED":
                state["report_sha256"] = combined_report(config, plan, state["jobs"])
        state["events"].append({"at": stamp(), "action": "collect", "evidence": str(sample)})
        locked_update(state_path, state_pin, state)
    except Exception as error:
        stop(config, state, error, str(sample))
        raise
    print(f"OPENOPT_CLUSTER_COLLECT state={state['state']} gpu=0 evidence={sample}", flush=True)
    return state


def check(config_path, config_pin):
    config, plan = load_config(config_path, config_pin)
    artifacts = Path(config["artifacts"])
    state = shard.read(artifacts / "run.json")
    shard.require(state["state"] == "PASS" and state["first_failure"] is None
                  and state["config_sha256"] == config_pin, "cluster_not_qualified")
    stage_members(config, plan, state)
    shard.require(all(job.get("accepted") is True for job in state["jobs"]), "unaccepted_job")
    verify_accepted(config, plan, state)
    audit = shard.read(state["acceptance"], state["acceptance_sha256"])
    summary = shard.read(artifacts / "verification.json", state["report_sha256"])
    if plan["schema_version"] == 2:
        runtime = shard.read(artifacts / "worker-probe/worker.json", state["runtime_sha256"])
        _, effective_path, effective_pin = execution_plan(config, plan, runtime)
        shard.require(summary["plan"] == str(effective_path)
                      and summary["plan_sha256"] == effective_pin
                      and summary["plan_schema_version"] == 2
                      and summary["source_transport_plan_sha256"] == config["plan_sha256"]
                      and summary["fixture_transport_contract"] == FIXTURE_TRANSPORT,
                      "combined_fixture_transport_binding_mismatch")
    shard.require(audit["source"] == plan["source"] and audit["tests"] == summary["tests"]
                  and audit["verdict"] == "PASS", "container_acceptance_mismatch")
    print(f"OPENOPT_CLUSTER_CHECK verdict=PASS coverage=tests gpu=0 "
          f"report={state['acceptance']}", flush=True)
    return audit


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("plan", "run", "collect", "check", "worker"))
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--artifact-dir", "--log-dir", type=Path, default=ARTIFACTS)
    parser.add_argument("--image")
    parser.add_argument("--python", default="python3")
    parser.add_argument("--namespace", default="devtest")
    parser.add_argument("--pvc", default="devtest-3fs")
    parser.add_argument("--service-account", default="devbox")
    parser.add_argument("--priority-class", default="devtest-batch")
    parser.add_argument("--cpu", type=int, default=4)
    parser.add_argument("--memory", default="16Gi")
    parser.add_argument("--node")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--fixture-aware", action="store_true",
                        help="plan-only opt-in to pinned v2 fixture/runtime transport")
    parser.add_argument("--phase", choices=("probe", "shard", "check"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-id", type=int)
    parser.add_argument("--runtime", type=Path)
    parser.add_argument("--runtime-sha256")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--report-sha256")
    args = parser.parse_args(argv)
    try:
        if args.action == "plan":
            prepare(args.repo.resolve(), args.plan, args.plan_sha256, args.artifact_dir,
                    args.image, args.python, args.namespace, args.pvc, args.service_account,
                    args.priority_class, args.cpu, args.memory, args.node, args.fixture_aware)
        elif args.action == "run":
            shard.require(not args.fixture_aware, "fixture_opt_in_is_plan_only")
            admit(args.plan, args.plan_sha256, args.submit)
        elif args.action == "collect":
            shard.require(not args.fixture_aware, "fixture_opt_in_is_plan_only")
            collect(args.plan, args.plan_sha256)
        elif args.action == "check":
            shard.require(not args.fixture_aware, "fixture_opt_in_is_plan_only")
            check(args.plan, args.plan_sha256)
        else:
            shard.require(not args.fixture_aware, "fixture_opt_in_is_plan_only")
            shard.require(args.output is not None and args.phase is not None,
                          "worker_output_and_phase_required")
            worker(args.plan, args.plan_sha256, args.phase, args.output, args.shard_id,
                   args.runtime, args.runtime_sha256, args.report, args.report_sha256)
    except Exception as error:
        print(f"OPENOPT_CLUSTER_CPU error={type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
