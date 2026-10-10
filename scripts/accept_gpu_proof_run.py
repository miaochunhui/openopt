"""Caller-pinned, CPU-only acceptance of one-to-three CUDA proof Jobs.

Cross-check collected records, actual Job/Pod terminal status, exact commands,
committed coordinator identity, both source snapshots and numerical seal/replay.
Missing evidence fails. No solver, snapshot-code execution or cluster call.
This integrity contract is not a remote hardware attestation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import tempfile
import traceback

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/hf3fs-jd/data/tmp").resolve()
ARTIFACT_ALIAS = Path("/hf3fs-jd/data/tmp")
PREFIXES = ("src/", "scripts/", "tests/", "examples/")
SNAPSHOT_FILES = {"docs/research/refinement.md"}
PRODUCER = "scripts/gpu_release_proof.py"
COORDINATOR = "scripts/cluster_benchmark.py"
SCOPE = "collected one-to-three single-case CUDA numerical proof Jobs"
PRODUCT = "NVIDIA-RTX-PRO-4500-Blackwell"


class AuditError(ValueError):
    """A missing/contradictory artifact, with its first coordinate."""


def require(condition, coordinate, cause):
    if not condition:
        raise AuditError(f"{coordinate}: {cause}")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(value, length=64):
    require(isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value),
            "hash", f"caller/source {length}-character digest required")
    return value


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def parse(data, coordinate):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, coordinate, f"duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value):
        raise AuditError(f"{coordinate}: nonfinite JSON {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), path, "regular non-symlink file required")
    data = path.read_bytes()
    return data


def relative(value):
    require(isinstance(value, str) and value, "path", "nonempty relative path required")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value,
            value, "unsafe relative path")
    return value


def external(path, tree=None):
    path = Path(path)
    require(path.is_absolute(), path, "absolute shared artifact path required")
    require(".." not in path.parts, path, "parent traversal rejected")
    # Allow the configured mount alias, not symlinks beneath the evidence root.
    lexical_root = next((root for root in (ARTIFACT_ALIAS, ARTIFACT_ROOT)
                         if path.is_relative_to(root)), None)
    require(lexical_root is not None, path, "outside shared 3FS tmp")
    component = lexical_root
    for part in path.relative_to(lexical_root).parts:
        component /= part
        require(not component.is_symlink(), component, "symlink evidence component rejected")
    resolved = path.resolve()
    require(resolved.is_relative_to(ARTIFACT_ROOT), path, "outside shared 3FS tmp")
    if tree is not None:
        require(resolved.is_relative_to(Path(tree).resolve()), path, "artifact escapes run tree")
    return resolved


def descriptor(root, path):
    path = external(path, root)
    data = read(path)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(data), "bytes": len(data)}


def bound(root, item):
    require(isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"},
            "descriptor", "exact path/hash/bytes descriptor required")
    path = external(Path(root) / relative(item["path"]), root)
    data = read(path)
    require(type(item["bytes"]) is int and item["bytes"] == len(data)
            and sha(data) == pin(item["sha256"]), path, "artifact byte/hash mutation")
    return path


def write(path, value):
    with Path(path).open("xb") as stream:
        stream.write(encode(value))


def file_map(root, *, prefixes=None):
    root = Path(root).resolve()
    result = {}
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), path, "source/inventory symlink rejected")
        if path.is_file():
            name = path.relative_to(root).as_posix()
            if prefixes is None or name.startswith(prefixes):
                result[name] = sha(read(path))
    require(result, root, "empty source/inventory")
    return result


def source_map(value, coordinate):
    require(isinstance(value, dict) and value, coordinate, "empty source membership")
    for name, digest in value.items():
        relative(name)
        pin(digest)
    return value


def equal_maps(first, second, coordinate):
    delta = {"missing": sorted(set(first) - set(second)),
             "extra": sorted(set(second) - set(first)),
             "changed": sorted(k for k in set(first) & set(second) if first[k] != second[k])}
    require(not any(delta.values()), coordinate, f"bidirectional source difference {delta}")


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    require(result.returncode == 0, "source identity",
            f"Git query failed rc={result.returncode} stderr={result.stderr.decode()[:500]}")
    return result.stdout


def current_identity(root):
    root = Path(root).resolve()
    require(not git(root, "status", "--porcelain").strip(), root, "current source must be clean")
    names = git(root, "ls-files", "-z").decode().split("\0")
    hashes = {}
    for name in sorted(set(names) - {""}):
        path = root / relative(name)
        require(not path.is_symlink() and path.resolve().is_relative_to(root),
                name, "current source symlink/escape")
        hashes[name] = sha(read(path))
    require(hashes, root, "empty committed current source")
    return {"commit": git(root, "rev-parse", "HEAD").decode().strip(),
            "tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
            "clean": True, "sha256": hashes}


def check_sources(run, tree, current_root):
    identity = run.get("source_identity")
    require(isinstance(identity, dict) and set(identity) == {"commit", "tree", "clean", "sha256"}
            and identity["clean"] is True, "source_identity", "clean committed identity required")
    pin(identity["commit"], 40)
    pin(identity["tree"], 40)
    require(run["source_sha"] == identity["commit"], "source_sha", "coordinator commit mismatch")
    expected = source_map(identity["sha256"], "source identity files")
    current = current_identity(current_root)
    require(current["commit"] == identity["commit"] and current["tree"] == identity["tree"],
            "current identity", "current commit/tree differs from coordinator")
    equal_maps(expected, current["sha256"], "current committed source")
    hashes = source_map(run["source_hashes"], "run source hashes")
    selected = {name: value for name, value in expected.items()
                if name.startswith(PREFIXES) or name in SNAPSHOT_FILES}
    equal_maps(selected, hashes, "coordinator snapshot selection")
    snapshot = tree / "source"
    require(snapshot.is_dir() and not snapshot.is_symlink(), snapshot, "source snapshot missing")
    actual = file_map(snapshot)
    manifest = snapshot / "source-hashes.json"
    require("source-hashes.json" in actual, manifest, "snapshot manifest missing")
    del actual["source-hashes.json"]
    equal_maps(hashes, actual, "coordinator snapshot bytes")
    equal_maps(hashes, source_map(parse(read(manifest), manifest), "manifest"), "snapshot manifest")
    require(PRODUCER in hashes and COORDINATOR in hashes, "sources", "producer/coordinator missing")
    return identity, hashes


def trusted_producer(current_root, digest):
    """Import explicitly trusted current source, never the artifact snapshot."""
    path = Path(current_root).resolve() / PRODUCER
    require(sha(read(path)) == pin(digest), path, "trusted current producer differs from coordinator")
    name = "_openopt_gpu_run_acceptance_producer"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, path, "producer loader missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def timestamp(value, coordinate):
    require(isinstance(value, str) and value, coordinate, "explicit timestamp required")
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AuditError(f"{coordinate}: invalid timestamp") from error
    require(date.tzinfo is not None, coordinate, "timezone missing")
    return date


def single_container(pod, coordinate):
    require(isinstance(pod, dict), coordinate, "Pod spec missing")
    require(pod.get("restartPolicy") == "Never" and pod.get("runtimeClassName") == "nvidia"
            and not pod.get("initContainers") and not pod.get("ephemeralContainers"),
            coordinate, "non-retrying single-container NVIDIA runtime required")
    containers = pod.get("containers")
    require(isinstance(containers, list) and len(containers) == 1,
            coordinate, "exactly one application container required")
    container = containers[0]
    require(container.get("name") == "benchmark" and container.get("image")
            and not container.get("args"), coordinate, "unknown container/argument override")
    require(pod.get("nodeSelector", {}).get("nvidia.com/gpu.product") == PRODUCT,
            coordinate, "requested RTX PRO 4500 product missing")
    for key in ("requests", "limits"):
        count = container.get("resources", {}).get(key, {}).get("nvidia.com/gpu")
        require((type(count) is int and count == 1) or count == "1",
                coordinate, "exactly one requested/limited GPU")
    return container


def environment(container, coordinate):
    result = {}
    for item in container.get("env", []):
        name = item.get("name")
        require(isinstance(name, str) and name not in result, coordinate, "duplicate environment key")
        result[name] = {key: value for key, value in item.items() if key != "name"}
    node = result.get("OPENOPT_NODE")
    if node == {"valueFrom": {"fieldRef": {"apiVersion": "v1", "fieldPath": "spec.nodeName"}}}:
        result["OPENOPT_NODE"] = {"valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}
    require(result.get("OPENOPT_NODE") == {
        "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        coordinate, "actual node downward API absent")
    require("HOSTNAME" not in result and "envFrom" not in container,
            coordinate, "Pod hostname/environment override rejected")
    return result


def execution_policy(spec, coordinate):
    require(type(spec.get("backoffLimit")) is int and spec["backoffLimit"] == 0
            and all(type(spec.get(key, 1)) is int and spec.get(key, 1) == 1
                    for key in ("completions", "parallelism")),
            coordinate, "one execution with zero retries required")


def pod_bindings(original_pod, actual_pod, original, actual, coordinate):
    require(environment(actual, coordinate) == environment(original, coordinate),
            coordinate, "actual environment differs from original execution")
    for key in ("runtimeClassName", "nodeSelector", "securityContext",
                "serviceAccountName", "priorityClassName"):
        require(actual_pod.get(key) == original_pod.get(key),
                coordinate + "/" + key, "actual execution policy differs from manifest")
    for key in ("securityContext", "workingDir", "imagePullPolicy"):
        require(actual.get(key) == original.get(key),
                coordinate + "/" + key, "actual container policy differs from manifest")
    mounts = original.get("volumeMounts", [])
    require(isinstance(mounts, list)
            and any(m.get("name") == "threefs" and m.get("mountPath") == "/3fs"
                    and not m.get("subPath") and not m.get("subPathExpr") for m in mounts),
            coordinate, "original shared-source mount missing/substituted")
    actual_mounts = actual.get("volumeMounts", [])
    actual_volumes = actual_pod.get("volumes", [])
    require(len({m.get("name") for m in actual_mounts}) == len(actual_mounts)
            and len({v.get("name") for v in actual_volumes}) == len(actual_volumes),
            coordinate, "duplicate mount/volume names")
    for mount in mounts:
        require(mount in actual_mounts, coordinate, "original mount absent/changed")
    for volume in original_pod.get("volumes", []):
        require(volume in actual_volumes, coordinate, "original backing volume absent/changed")
    extras = [m for m in actual_mounts if m not in mounts]
    # Kubernetes may inject only its service-account projection.
    for mount in extras:
        require(mount.get("name", "").startswith("kube-api-access-")
                and mount.get("mountPath") == "/var/run/secrets/kubernetes.io/serviceaccount"
                and mount.get("readOnly") is True and not mount.get("subPath")
                and any(v.get("name") == mount["name"] and "projected" in v
                        for v in actual_volumes),
                coordinate, "unrequested execution mount")
    allowed = {v["name"] for v in original_pod.get("volumes", [])} | {m["name"] for m in extras}
    require({v.get("name") for v in actual_volumes} == allowed,
            coordinate, "unrequested backing volume")


def check_status(item, manifest, status, namespace, source, output, result, expectations):
    name = item["job"]
    require(manifest.get("kind") == "Job" and manifest.get("apiVersion") == "batch/v1",
            name, "original Job manifest missing")
    meta = manifest.get("metadata", {})
    require(meta.get("name") == name and meta.get("namespace") == namespace,
            name, "original manifest Job/namespace mismatch")
    spec = manifest.get("spec", {})
    execution_policy(spec, name + "/manifest")
    original_pod = spec.get("template", {}).get("spec")
    original = single_container(original_pod, name + "/manifest")
    case = item["proof_case"]
    command = ["python3", str(source / PRODUCER), "run", "--artifact-dir", str(output),
               "--device", "cuda:0", "--case", case,
               "--expectations", expectations["path"],
               "--expectations-sha256", expectations["sha256"]]
    require(original.get("command") == command, name, "original command differs from exact one-case run")
    env = environment(original, name)
    require(env.get("PYTHONPATH") == {"value": str(source / "src")},
            name, "executed PYTHONPATH differs from frozen snapshot")
    require(isinstance(status, dict) and set(status) == {"job", "pods"},
            name, "actual Job/Pod status missing")
    job = status["job"]
    job_meta = job.get("metadata", {})
    require(job.get("kind") == "Job" and job_meta.get("name") == name
            and job_meta.get("namespace") == namespace and job_meta.get("uid"),
            name, "actual Job identity missing/mismatched")
    job_spec = job.get("spec", {})
    execution_policy(job_spec, name + "/actual Job")
    live_template = job_spec.get("template", {}).get("spec")
    live_container = single_container(live_template, name + "/actual Job")
    require(live_container["command"] == command and environment(live_container, name) == env
            and live_container["image"] == original["image"], name, "actual Job command/environment/image mismatch")
    pod_bindings(original_pod, live_template, original, live_container, name + "/actual Job")
    js = job.get("status", {})
    require(type(js.get("succeeded")) is int and js["succeeded"] == 1
            and all(type(js.get(key, 0)) is int and js.get(key, 0) == 0
                    for key in ("failed", "active")),
            name, "actual Job not exactly succeeded once with zero active/failed")
    conditions = js.get("conditions", [])
    require(any(c.get("type") == "Complete" and c.get("status") == "True" for c in conditions)
            and not any(c.get("type") in ("Failed", "FailureTarget") and c.get("status") == "True"
                        for c in conditions), name, "Job terminal Complete condition missing/contradictory")
    pods = status["pods"]
    require(pods.get("kind") in ("List", "PodList") and isinstance(pods.get("items"), list)
            and len(pods["items"]) == 1, name, "exactly one actual Pod required")
    pod = pods["items"][0]
    pm = pod.get("metadata", {})
    pod_name = pm.get("name")
    require(pod.get("kind") == "Pod" and pod_name and pm.get("uid")
            and pm.get("namespace") == namespace and pm.get("labels", {}).get("job-name") == name
            and any(o.get("kind") == "Job" and o.get("name") == name
                    and o.get("uid") == job_meta["uid"] and o.get("controller") is True
                    for o in pm.get("ownerReferences", [])),
            name, "actual Pod ownership/name/namespace missing")
    ps = pod.get("spec", {})
    observed_container = single_container(ps, name + "/actual Pod")
    require(observed_container["command"] == command
            and observed_container["image"] == original["image"], name, "actual Pod command/image mismatch")
    pod_bindings(original_pod, ps, original, observed_container, name + "/actual Pod")
    st = pod.get("status", {})
    require(st.get("phase") == "Succeeded" and ps.get("nodeName"),
            name, "actual Pod not Succeeded or actual node absent")
    states = st.get("containerStatuses")
    require(isinstance(states, list) and len(states) == 1 and states[0].get("name") == "benchmark",
            name, "actual container terminal status missing")
    container_status = states[0]
    require(type(container_status.get("restartCount")) is int and container_status["restartCount"] == 0
            and not container_status.get("lastState")
            and container_status.get("imageID") and container_status.get("containerID"),
            name, "container restarted or runtime identity missing")
    state = container_status.get("state", {})
    require(set(state) == {"terminated"}, name, "container is not terminal")
    terminal = state["terminated"]
    require(type(terminal.get("exitCode")) is int and terminal["exitCode"] == 0
            and terminal.get("reason") == "Completed"
            and type(terminal.get("signal", 0)) is int and terminal.get("signal", 0) == 0,
            name, "actual exit is not Completed/0")
    start = timestamp(terminal.get("startedAt"), name + "/startedAt")
    finish = timestamp(terminal.get("finishedAt"), name + "/finishedAt")
    require(start <= timestamp(result["started_at"], "result start")
            <= timestamp(result["finished_at"], "result finish") <= finish,
            name, "result timestamps outside actual container lifetime")
    hw = result.get("hardware", {})
    require(hw.get("node") == ps["nodeName"] and hw.get("pod") == pod_name,
            name, "producer actual node/Pod differs from cluster status")
    return {"job": name, "pod": pod_name, "node": ps["nodeName"],
            "job_uid": job_meta["uid"], "pod_uid": pm["uid"], "image_id": container_status["imageID"],
            "container_id": container_status["containerID"], "exit_code": 0, "restart_count": 0,
            "command": command}


def expected_models(run, tree, hashes, producer, cases, expected_sha256):
    item = run.get("expectations")
    require(isinstance(item, dict), "expectations", "pre-dispatch model manifest missing")
    path = external(item.get("path"), tree)
    require(path == tree / "proof-expectations.json", path, "pre-dispatch manifest substitution")
    data = read(path)
    require(set(item) == {"path", "sha256", "bytes"}
            and type(item["bytes"]) is int and len(data) == item["bytes"]
            and sha(data) == pin(item["sha256"]) == pin(expected_sha256),
            path, "independent pre-dispatch caller model pin mismatch")
    manifest = parse(data, path)
    require(set(manifest) == {"schema_version", "source_sha256", "cases"}
            and type(manifest["schema_version"]) is int and manifest["schema_version"] == 1,
            path, "unknown/incomplete expected model schema")
    equal_maps(hashes, source_map(manifest["source_sha256"], "expectations sources"),
               "expectations source binding")
    require(isinstance(manifest["cases"], dict) and set(manifest["cases"]) == set(cases),
            path, "pre-dispatch expected model denominator differs from Jobs")
    for case, record in manifest["cases"].items():
        require(set(record) == {"model_sha256", "model"}
                and set(record["model"]) == set(producer.FIELDS)
                and pin(record["model_sha256"]) == producer.model_hash(record["model"]),
                case, "pre-dispatch model metadata/hash invalid")
        for field, tensor in record["model"].items():
            require(tensor["device"] == "cpu" and tensor["dtype"] == "torch.float64",
                    case + "/" + field, "expected model must be pre-dispatch float64 CPU data")
            for component in tensor["components"].values():
                require(relative(component["path"]).startswith("arrays/"),
                        case + "/" + field, "expected raw array outside pre-dispatch inventory")
            producer.load_tensor(tree, tensor, case + "/expected/" + field)
    return manifest


def runtime_binding(log, tree, result_path, result, placement, digest):
    lines = [line for line in log.splitlines() if line.strip()]
    completion = (f"OPENOPT_GPU_PROOF verdict={result['verdict']} at={result['finished_at']} "
                  f"result={result_path} result_sha256={digest}")
    require(lines and lines[-1] == completion
            and lines.count(completion) == 1
            and sum(line.startswith("OPENOPT_GPU_PROOF verdict=") for line in lines) == 1,
            "Pod log", "last completion line does not bind exact producer result path/hash")
    runtime_lines = [line for line in lines if line.startswith("OPENOPT_GPU_PROOF_RUNTIME ")]
    require(len(runtime_lines) == 1, "Pod log", "one independently collected runtime identity required")
    runtime = parse(runtime_lines[0].split(" ", 1)[1], "Pod runtime")
    require(set(runtime) == {"device", "gpu_uuid", "gpu_name", "gpu_count", "node", "pod"}
            and runtime["device"] == "cuda:0" and type(runtime["gpu_count"]) is int
            and runtime["gpu_count"] == 1 and runtime["node"] == placement["node"]
            and runtime["pod"] == placement["pod"] and "RTX PRO 4500" in runtime["gpu_name"]
            and re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}",
                             runtime["gpu_uuid"]),
            "Pod runtime", "one-card GPU UUID/name/device/actual node/Pod missing or mismatched")
    hardware = result["hardware"]
    require(hardware.get("gpu") == runtime["gpu_name"]
            and type(hardware.get("gpu_count")) is int and hardware["gpu_count"] == 1
            and hardware.get("requested_device") == "cuda:0"
            and hardware.get("node") == runtime["node"] and hardware.get("pod") == runtime["pod"],
            "Pod runtime", "result hardware differs from independently collected runtime")
    if "gpu_uuid" in hardware:
        require(hardware["gpu_uuid"] == runtime["gpu_uuid"], "hardware", "result GPU UUID differs")
    samples = parse(read(bound(result_path.parent, result["telemetry"])), "telemetry")
    require(isinstance(samples, list) and len(samples) >= 2, "telemetry", "missing runtime samples")
    for sample in samples:
        rows = sample.get("stdout", "").splitlines()
        require(type(sample.get("returncode")) is int and sample["returncode"] == 0
                and len(rows) == 1, "telemetry", "missing actual single-card sample")
        values = [part.strip() for part in rows[0].split(",")]
        require(len(values) == 6 and values[:2] == [runtime["gpu_uuid"], runtime["gpu_name"]],
                "telemetry", "sample GPU UUID/name differs from coordinator-fetched Pod log")
    return {**placement, "gpu_uuid": runtime["gpu_uuid"], "gpu_name": runtime["gpu_name"],
            "gpu_count": 1, "device": "cuda:0", "log_result_sha256": digest}


def control_audit(collection_path, expected_sha256, current_root, expectations_sha256):
    path = external(collection_path)
    require(path.name == "collection.json", path, "explicit collection.json required")
    data = read(path)
    require(sha(data) == pin(expected_sha256), path, "caller collection pin mismatch")
    tree = path.parent
    collection = parse(data, path)
    run = parse(read(tree / "run.json"), "run.json")
    require(run.get("stage") == "release-proof" and run.get("state") == "SUBMITTED",
            "run", "submitted release-proof run required")
    require(set(collection) == set(run) | {"collected_at", "states"}, "collection",
            "collection added/dropped run fields")
    require({key: value for key, value in collection.items() if key not in ("jobs", "collected_at", "states")}
            == {key: value for key, value in run.items() if key != "jobs"},
            "collection", "collection metadata differs from original run")
    jobs = run.get("jobs")
    require(isinstance(jobs, list) and 1 <= len(jobs) <= 3, "jobs", "exact one-to-three denominator required")
    require(collection["states"] == ["PASS"] * len(jobs)
            and len(collection.get("jobs", [])) == len(jobs),
            "collection", "incomplete/failed/RUNNING case denominator")
    require(timestamp(run["started_at"], "run start")
            <= timestamp(collection["collected_at"], "collected_at"), "collection", "collection predates run")
    identity, hashes = check_sources(run, tree, current_root)
    producer = trusted_producer(current_root, hashes[PRODUCER])
    producer_sources = {k: v for k, v in hashes.items()
                        if (k.startswith("src/") and k.endswith(".py")) or k == PRODUCER}
    expectations = expected_models(run, tree, producer_sources, producer,
                                   [item["proof_case"] for item in jobs], expectations_sha256)
    names, cases, entries = set(), set(), []
    namespace = run["namespace"]
    for index, (item, collected) in enumerate(zip(jobs, collection["jobs"])):
        require(set(item) == {"job", "size", "output", "manifest", "proof_case"}
                and set(collected) == set(item) | {"result_path", "result_sha256"}
                and {key: collected[key] for key in item} == item,
                f"job-{index}", "collected job entry differs from original")
        name, case = item["job"], item["proof_case"]
        require(isinstance(name, str) and re.fullmatch(r"[a-z0-9][a-z0-9-]*", name)
                and name not in names and isinstance(case, str) and case not in cases,
                f"job-{index}", "duplicate/invalid job or case")
        names.add(name)
        cases.add(case)
        require(type(item["size"]) is int and item["size"] == 0,
                name, "proof Job cannot mix size ladder")
        output = external(item["output"], tree)
        require(output == tree / f"case-{index}", name, "original output mapping mismatch")
        manifest_path = external(item["manifest"], tree)
        require(manifest_path == tree / f"job-{index}.json", name, "original manifest mapping mismatch")
        results = sorted(output.glob("cuda-proof-*/result.json"))
        require(len(results) == 1, name, "exactly one immutable cuda-proof result required")
        result_path = external(collected["result_path"], output)
        require(result_path == results[0] and result_path.parent.parent == output,
                name, "collection result is not exact immutable producer output")
        require(sha(read(result_path)) == pin(collected["result_sha256"]),
                result_path, "collected result pin mismatch")
        result = parse(read(result_path), result_path)
        require(type(result.get("schema_version")) is int and result["schema_version"] == 1
                and result.get("verdict") == producer.NUMERICAL_PASS
                and result.get("numerical_verdict") == "PASS"
                and result.get("execution_verdict") == "UNVERIFIED"
                and result.get("expectations") == run["expectations"]
                and result.get("device") == "cuda:0" and type(result.get("returncode")) is int
                and result["returncode"] == 0 and result.get("requested_cases") == [case]
                and set(result.get("cases", {})) == {case},
                name, "immutable result not exactly one successful explicit CUDA case")
        equal_maps(producer_sources, source_map(result.get("source_sha256"), "producer sources"),
                   name + "/producer-coordinator source")
        saved = parse(read(bound(result_path.parent, result["cases"][case])), case)
        model = expectations["cases"][case]
        require(saved.get("case") == case and saved.get("model_sha256") == model["model_sha256"]
                and producer.model_hash(saved["model"]) == model["model_sha256"],
                case, "executed whole model differs from independently pinned pre-dispatch case")
        for field in producer.FIELDS:
            require(producer.tensor_hash(saved["model"][field])
                    == producer.tensor_hash(model["model"][field]),
                    case + "/" + field, "original model raw bytes/shape/dtype/layout substituted")
        placement = check_status(
            item, parse(read(manifest_path), manifest_path),
            parse(read(tree / (name + "-status.json")), name + "/status"),
            namespace, tree / "source", output, result, run["expectations"])
        log_path = tree / (placement["pod"] + ".log")
        log = read(log_path).decode()
        placement = runtime_binding(log, tree, result_path, result, placement, collected["result_sha256"])
        require(timestamp(result["finished_at"], "result finish")
                <= timestamp(collection["collected_at"], "collected_at"),
                name, "collection predates producer completion")
        entries.append({"case": case, "expected_model_sha256": model["model_sha256"],
                        "result": descriptor(tree, result_path),
                        "placement": placement})
    return tree, identity, hashes, entries


def inventory(tree):
    return {name: descriptor(tree, tree / name) for name in file_map(tree)}


def validate(collection_path, expected_sha256, current_root, *, expectations_sha256,
             numerical_dir, replay_bundles=None):
    tree, identity, hashes, entries = control_audit(collection_path, expected_sha256, current_root,
                                                  expectations_sha256)
    producer = trusted_producer(current_root, hashes[PRODUCER])
    results, proof_bundles = [], {}
    if replay_bundles is not None:
        require(set(replay_bundles) == {entry["case"] for entry in entries},
                "numerical bundles", "missing/extra numerical case denominator")
    numerical_dir = external(numerical_dir)
    require(not numerical_dir.is_relative_to(tree), numerical_dir, "numerical output cannot mutate run evidence")
    numerical_dir.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        case = entry["case"]
        require(case in producer.CASES, case, "unknown producer case")
        result_path = bound(tree, entry["result"])
        if replay_bundles is None:
            sealed = producer.seal(result_path, entry["result"]["sha256"], current_root,
                                   expectations=tree / "proof-expectations.json",
                                   expectations_sha256=expectations_sha256)
            path = numerical_dir / (case + "-bundle.json")
            write(path, sealed)
            item = {"path": str(path), "sha256": sha(read(path)), "bytes": len(read(path))}
        else:
            item = replay_bundles[case]
            require(set(item) == {"path", "sha256", "bytes"},
                    case, "numerical bundle descriptor incomplete")
            path = external(item["path"])
            require(type(item["bytes"]) is int and len(read(path)) == item["bytes"]
                    and sha(read(path)) == pin(item["sha256"]),
                    path, "numerical seal byte/hash mutation")
        numerical = producer.replay(path, item["sha256"], current_root,
                                    expectations=tree / "proof-expectations.json",
                                    expectations_sha256=expectations_sha256)
        require(numerical.get("verdict") == producer.NUMERICAL_PASS
                and numerical.get("numerical_verdict") == "PASS"
                and numerical.get("execution_verdict") == "UNVERIFIED"
                and numerical.get("device") == "cuda:0"
                and numerical.get("scope") == producer.GPU_SCOPE and numerical.get("case_count") == 1
                and [c["case"] for c in numerical.get("cases", [])] == [case]
                and all(c.get("passed") is True for c in numerical["cases"])
                and numerical.get("result_sha256") == entry["result"]["sha256"],
                case, "numerical replay missing, failed or outside exact one-case denominator")
        results.append({**entry, "numerical": numerical})
        proof_bundles[case] = item
    # Check again after all numerical reads. No racing rewrite is accepted.
    second = control_audit(collection_path, expected_sha256, current_root, expectations_sha256)
    require(second[1:] == (identity, hashes, entries), "run", "control evidence changed during replay")
    return {"verdict": "PASS", "numerical_verdict": "PASS", "execution_verdict": "PASS",
            "scope": SCOPE, "case_count": len(results),
            "requested_cases": [entry["case"] for entry in entries], "entries": results,
            "source_commit": identity["commit"], "source_tree": identity["tree"],
            "coordinator_source_count": len(hashes), "collection_sha256": expected_sha256,
            "expectations_sha256": expectations_sha256,
            "qualification": "selected synthetic numerical Jobs only; not remote attestation, "
                             "performance, commercial parity or comprehensive capability"}, proof_bundles


def seal(collection_path, expected_sha256, current_root, output, *, expectations_sha256):
    tree = external(collection_path).parent
    output = external(output)
    require(not output.is_relative_to(tree), output, "seal output cannot mutate original run")
    before = inventory(tree)
    outcome, numerical = validate(collection_path, expected_sha256, current_root,
                                  expectations_sha256=expectations_sha256,
                                  numerical_dir=output / "numerical")
    require(before == inventory(tree), "run inventory", "run evidence changed while sealing")
    bundle = {"schema_version": 1, "scope": SCOPE, "run_root": str(tree),
              "collection": descriptor(tree, external(collection_path)),
              "expectations_sha256": expectations_sha256, "files": before, "numerical_bundles": numerical}
    return bundle, outcome


def accept(bundle_path, expected_sha256, current_root, output, *, expectations_sha256):
    bundle_path = external(bundle_path)
    data = read(bundle_path)
    require(sha(data) == pin(expected_sha256), bundle_path, "caller bundle pin mismatch")
    bundle = parse(data, bundle_path)
    require(set(bundle) == {"schema_version", "scope", "run_root", "collection", "files",
                           "expectations_sha256", "numerical_bundles"}
            and bundle["schema_version"] == 1 and bundle["scope"] == SCOPE,
            "bundle", "unknown/incomplete run-acceptance scope")
    tree = external(bundle["run_root"])
    require(pin(bundle["expectations_sha256"]) == pin(expectations_sha256),
            "expectations", "accept must preserve independently pinned pre-dispatch manifest")
    require(bundle["files"] == inventory(tree), "run inventory", "sealed evidence added/removed/changed")
    for item in bundle["numerical_bundles"].values():
        require(external(item["path"]).parent == bundle_path.parent / "numerical",
                "numerical bundles", "numerical seal must belong to caller-pinned acceptance bundle")
    collection_path = bound(tree, bundle["collection"])
    require(collection_path == tree / "collection.json", "collection", "sealed collection substitution")
    outcome, _ = validate(collection_path, bundle["collection"]["sha256"], current_root,
                          expectations_sha256=expectations_sha256,
                          numerical_dir=output / "numerical", replay_bundles=bundle["numerical_bundles"])
    require(bundle["files"] == inventory(tree), "run inventory", "sealed evidence raced acceptance")
    return outcome


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "accept"))
    parser.add_argument("--collection", type=Path)
    parser.add_argument("--collection-sha256")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--expectations-sha256", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    parent = external(args.artifact_dir)
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.action + "-", dir=parent))
    report, code = {"started_at": now(), "verdict": "RUNNING",
                    "numerical_verdict": "NOT_CHECKED", "execution_verdict": "UNVERIFIED"}, 0
    try:
        if args.action == "seal":
            require(args.collection and args.collection_sha256, "seal", "explicit caller collection pin required")
            bundle, outcome = seal(args.collection, args.collection_sha256, args.current_root, output,
                                   expectations_sha256=args.expectations_sha256)
            write(output / "bundle.json", bundle)
            report.update(outcome, verdict="SEALED", bundle=str(output / "bundle.json"),
                          bundle_sha256=sha(read(output / "bundle.json")))
        else:
            require(args.bundle and args.sha256, "accept", "explicit caller bundle pin required")
            report.update(accept(args.bundle, args.sha256, args.current_root, output,
                                 expectations_sha256=args.expectations_sha256))
    except Exception as error:
        code = 1
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        with (output / "failure.log").open("x") as stream:
            stream.write(traceback.format_exc())
    report.update(finished_at=now(), returncode=code)
    write(output / "acceptance.json", report)
    line = (f"OPENOPT_GPU_PROOF_RUN_ACCEPT verdict={report['verdict']} "
            f"at={report['finished_at']} cases={report.get('case_count', 'NA')} "
            f"evidence={output / 'acceptance.json'}")
    with (output / "execution.log").open("x") as stream:
        stream.write(line + f"\nreturncode={code}\n")
    print(line, flush=True)
    if code:
        print(report["error"], flush=True)
    elif args.action == "seal":
        print(f"bundle={report['bundle']} sha256={report['bundle_sha256']}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
