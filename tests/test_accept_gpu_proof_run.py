"""CPU-only integrity fixtures, not evidence of CUDA or cluster execution.

Numerical fixtures use analytic solutions (no solve/factory). A committed base
plus explicitly selected local source bytes form a clean test-only repository.
All generated records and
repositories live in pytest's external artifact directory. Fake Pod/GPU records
exercise the verifier only and must never be promoted to GPU release evidence.
"""

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tarfile
import io

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("proof_run_audit", ROOT / "scripts/accept_gpu_proof_run.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
REVISION = os.environ.get("OPENOPT_PROOF_TEST_REVISION", "c451dcf")
UUID = "GPU-12345678-1234-5678-9abc-123456789abc"
GPU = "NVIDIA RTX PRO 4500 Blackwell"


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.encode(value))


@pytest.fixture(scope="module")
def trusted_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("trusted-source")
    archive = git(ROOT, "archive", REVISION)
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        for member in stream.getmembers():
            assert member.isfile() or member.isdir()
            name = audit.relative(member.name.rstrip("/"))
            if member.isdir():
                continue
            if not (name.startswith("src/") or name in {
                    audit.PRODUCER, audit.COORDINATOR, "LICENSE",
                    "docs/research/refinement.md"}):
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(stream.extractfile(member).read())
    source = Path(os.environ.get("OPENOPT_PROOF_TEST_SOURCE_ROOT", str(ROOT)))
    if (source / audit.PRODUCER).is_file():
        for path in sorted((source / "src").rglob("*.py")) + [
                source / audit.PRODUCER, source / audit.COORDINATOR]:
            target = root / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
        for name in audit.SNAPSHOT_FILES:
            if (source / name).is_file():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source / name).read_bytes())
    assert "NUMERICAL_PASS" in (root / audit.PRODUCER).read_text(), "fixed producer required"
    git(root, "init", "-q")
    names = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                   if path.is_file() and not path.is_relative_to(root / ".git"))
    git(root, "add", "--", *names)
    git(root, "-c", "user.name=CPU Integrity Fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-q", "-m", "Explicitly simulated CPU-only integrity source")
    assert audit.current_identity(root)["clean"] is True
    return root


@pytest.fixture
def producer(trusted_root, monkeypatch):
    digest = audit.sha((trusted_root / audit.PRODUCER).read_bytes())
    real = audit.trusted_producer(trusted_root, digest)

    def forbidden(*args, **kwargs):
        raise AssertionError("acceptance attempted a solve, factory, CUDA or cluster action")

    import openopt.qp as qp
    import openopt.operators as operators
    import openopt.scaling as scaling

    for module, names in (
        (qp, ("solve_qp", "prepare_qp")),
        (operators, ("prepare_operator_qp", "conjugate_gradient")),
        (scaling, ("solve_scaled_qp",)),
        (real, ("run_cases", "execute_case", "make_problem", "hardware")),
        (torch.cuda, ("is_available", "device_count", "get_device_name")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    # Keep actual seal/replay intact; prevent importing a second unguarded copy.
    monkeypatch.setattr(audit, "trusted_producer", lambda root, digest: real)
    assert real.seal.__module__ == real.replay.__module__
    return real


def model_arrays(case):
    if case == "gap_lp":
        return (np.zeros((3, 3)), np.array([1., -2., .5]), np.eye(3),
                np.array([0., 0., -1.]), np.array([2., 3., 1.])), np.array([0., 3., -1.])
    if case == "gap_singular":
        return (np.diag([2., 0., 4.]), np.array([-1., -2., 1.]), np.eye(3),
                -np.ones(3), np.ones(3)), np.array([.5, 1., -.25])
    # A different SPD model is legal only as a labelled simulation fixture.
    return (np.eye(2), np.array([-1., 2.]), np.eye(2), -np.ones(2), np.ones(2)), np.array([1., -1.])


def tensor_records(producer, root, case, arrays, label):
    return {field: producer.save_tensor(root, label + "-" + case + "-" + field,
                                       torch.from_numpy(array))
            for field, array in zip(producer.FIELDS, arrays)}


def fake_result(producer, root, case, sources, model, expectation):
    root.mkdir(parents=True)
    for name in sources:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((producer.ROOT / name).read_bytes())
    arrays, x = model_arrays(case)
    actual_model = tensor_records(producer, root, case, arrays, "original")
    assert producer.model_hash(actual_model) == producer.model_hash(model)
    resident = deepcopy(actual_model)
    for record in resident.values():
        record["device"] = "cuda:0"
    y = -(arrays[0] @ x + arrays[1])
    vectors = {}
    for name, value in (("x", x), ("y", y)):
        vectors[name] = producer.save_tensor(root, case + "-" + name, torch.from_numpy(value))
        vectors[name]["device"] = "cuda:0"
    config, scaling, atol, rtol = producer.policy(case, "cuda:0")
    checks = producer.original_checks(dict(zip(producer.FIELDS, arrays)), {"x": x, "y": y},
                                      "solved", atol, rtol)
    assert checks["passed"] is True
    saved = {
        "case": case, "status": "solved", "expected_status": "solved",
        "dtype": "torch.float64", "device": "cuda:0", "config": producer.config_dict(config),
        "scaling_config": None, "atol": atol, "rtol": rtol,
        "model": actual_model, "model_sha256": producer.model_hash(actual_model),
        "resident_model": resident, "vectors": vectors,
        "observed": {name: {"device": "cuda:0", "dtype": "torch.float64", "layout": "torch.strided"}
                     for name in (*producer.FIELDS, "factor", "x", "y")},
        "extensions": {}, "iterations": 25, "objective": checks["metrics"]["primal_objective"],
        "independent_original": checks, "numpy_original": checks,
        "wall_s": .1, "qualification": "SIMULATED CPU fixture only, not GPU evidence",
    }
    put(root / (case + ".json"), saved)
    samples = [
        {"at": f"2026-10-10T08:00:0{i}+00:00", "cpu": {"usage_usec": i + 1},
         "returncode": 0, "stdout": f"{UUID}, {GPU}, 0, 1, 24576, 595.91.07", "stderr": ""}
        for i in (1, 2)
    ]
    put(root / "telemetry.json", samples)
    result = {
        "schema_version": 1, "scope": producer.GPU_SCOPE, "verdict": producer.NUMERICAL_PASS,
        "numerical_verdict": "PASS", "execution_verdict": "UNVERIFIED", "expectations": expectation,
        "started_at": "2026-10-10T08:00:01+00:00", "finished_at": "2026-10-10T08:00:02+00:00",
        "device": "cuda:0", "requested_cases": [case], "returncode": 0, "wall_s": 1,
        "cases": {case: producer.descriptor(root, root / (case + ".json"))},
        "source_sha256": sources,
        "hardware": {
            "requested_device": "cuda:0", "torch": torch.__version__, "numpy": np.__version__,
            "cuda": "13.0", "gpu": GPU, "gpu_uuid": UUID, "gpu_count": 1, "node": "fixture-node",
            "pod": f"fixture-{case.replace('_', '-')}-pod", "cpu_quota": "400000 100000",
            "cpu_scope": "Pod cgroup v2", "cuda_capability": [12, 0],
        },
        "telemetry": producer.descriptor(root, root / "telemetry.json"),
    }
    put(root / "result.json", result)
    digest = audit.sha((root / "result.json").read_bytes())
    completion = (f"OPENOPT_GPU_PROOF verdict={result['verdict']} at={result['finished_at']} "
                  f"result={root / 'result.json'} result_sha256={digest}")
    (root / "execution.log").write_text(completion + "\nreturncode=0\n")
    return result, completion, digest


def fake_manifest(tree, index, case, expectation):
    source, output = tree / "source", tree / f"case-{index}"
    item = {"job": f"fixture-{case.replace('_', '-')}", "size": 0, "proof_case": case,
            "output": str(output), "manifest": str(tree / f"job-{index}.json")}
    command = ["python3", str(source / audit.PRODUCER), "run", "--artifact-dir", str(output),
               "--device", "cuda:0", "--case", case, "--expectations", expectation["path"],
               "--expectations-sha256", expectation["sha256"]]
    container = {
        "name": "benchmark", "image": "fixture-runtime:never-executed", "command": command,
        "imagePullPolicy": "Never",
        "env": [{"name": "PYTHONPATH", "value": str(source / "src")},
                {"name": "OPENOPT_NODE", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}}],
        "resources": {"requests": {"nvidia.com/gpu": 1}, "limits": {"nvidia.com/gpu": 1}},
        "volumeMounts": [{"name": "threefs", "mountPath": "/3fs", "mountPropagation": "HostToContainer"}],
    }
    pod = {"containers": [container], "restartPolicy": "Never", "runtimeClassName": "nvidia",
           "nodeSelector": {"nvidia.com/gpu.product": audit.PRODUCT},
           "volumes": [{"name": "threefs", "persistentVolumeClaim": {"claimName": "fixture-only"}}]}
    job = {"apiVersion": "batch/v1", "kind": "Job", "metadata": {
        "name": item["job"], "namespace": "fixture-only"}, "spec": {
        "backoffLimit": 0, "template": {"metadata": {}, "spec": pod}}}
    put(Path(item["manifest"]), job)
    actual_job = deepcopy(job)
    actual_job["metadata"]["uid"] = "fixture-job-uid-" + case
    actual_job["status"] = {"succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}]}
    live_container = actual_job["spec"]["template"]["spec"]["containers"][0]
    live_container["env"][1]["valueFrom"]["fieldRef"]["apiVersion"] = "v1"
    for counts in live_container["resources"].values():
        counts["nvidia.com/gpu"] = "1"
    actual_pod = {
        "apiVersion": "v1", "kind": "Pod",
        "metadata": {"name": f"fixture-{case.replace('_', '-')}-pod", "uid": "fixture-pod-uid-" + case,
                     "namespace": "fixture-only", "labels": {"job-name": item["job"]},
                     "ownerReferences": [{"kind": "Job", "name": item["job"],
                                          "uid": actual_job["metadata"]["uid"], "controller": True}]},
        "spec": deepcopy(actual_job["spec"]["template"]["spec"]),
        "status": {"phase": "Succeeded", "containerStatuses": [{
            "name": "benchmark", "restartCount": 0, "lastState": {},
            "imageID": "fixture-sha256-not-an-image", "containerID": "fixture-runtime-id",
            "state": {"terminated": {"exitCode": 0, "reason": "Completed", "signal": 0,
                      "startedAt": "2026-10-10T08:00:00Z", "finishedAt": "2026-10-10T08:00:03Z"}}}]},
    }
    actual_pod["spec"]["nodeName"] = "fixture-node"
    status = {"job": actual_job, "pods": {"kind": "PodList", "items": [actual_pod]}}
    put(tree / (item["job"] + "-status.json"), status)
    return item


def fixture_run(tmp_path, trusted_root, producer, cases=("gap_lp",)):
    tree = tmp_path / "simulated-release-proof"
    tree.mkdir()
    identity = audit.current_identity(trusted_root)
    selected = {k: v for k, v in identity["sha256"].items()
                if k.startswith(audit.PREFIXES) or k in audit.SNAPSHOT_FILES}
    producer_sources = {k: v for k, v in selected.items()
                        if (k.startswith("src/") and k.endswith(".py")) or k == audit.PRODUCER}
    for name in selected:
        path = tree / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((trusted_root / name).read_bytes())
    put(tree / "source/source-hashes.json", selected)
    model_root = tree
    expected = {"schema_version": 1, "source_sha256": producer_sources, "cases": {}}
    for case in cases:
        arrays, _ = model_arrays(case)
        records = tensor_records(producer, model_root, case, arrays, "expected")
        expected["cases"][case] = {"model": records, "model_sha256": producer.model_hash(records)}
    path = tree / "proof-expectations.json"
    put(path, expected)
    expectation = {"path": str(path), "sha256": audit.sha(path.read_bytes()), "bytes": path.stat().st_size}
    jobs, collected = [], []
    for index, case in enumerate(cases):
        item = fake_manifest(tree, index, case, expectation)
        result_path = tree / f"case-{index}/cuda-proof-explicit-simulation/result.json"
        result, completion, digest = fake_result(
            producer, result_path.parent, case, producer_sources, expected["cases"][case]["model"], expectation)
        runtime = {"device": "cuda:0", "gpu_uuid": UUID, "gpu_name": GPU, "gpu_count": 1,
                   "node": "fixture-node", "pod": result["hardware"]["pod"]}
        (tree / (runtime["pod"] + ".log")).write_text(
            "SIMULATED CPU fixture, NEVER cluster/CUDA evidence\n"
            + "OPENOPT_GPU_PROOF_RUNTIME " + json.dumps(runtime, sort_keys=True) + "\n" + completion + "\n")
        jobs.append(item)
        collected.append({**item, "result_path": str(result_path), "result_sha256": digest})
    run = {"started_at": "2026-10-10T07:59:00Z", "namespace": "fixture-only",
           "stage": "release-proof", "state": "SUBMITTED", "source_sha": identity["commit"],
           "source_identity": identity, "source_hashes": selected, "jobs": jobs, "expectations": expectation}
    put(tree / "run.json", run)
    collection = {**run, "jobs": collected, "states": ["PASS"] * len(cases),
                  "collected_at": "2026-10-10T08:00:04Z"}
    put(tree / "collection.json", collection)
    return tree, expectation["sha256"]


def control(tree, expected_pin, trusted_root):
    path = tree / "collection.json"
    return audit.control_audit(path, audit.sha(path.read_bytes()), trusted_root, expected_pin)


def load(path):
    return json.loads(path.read_bytes())


@pytest.mark.parametrize("count", [1, 2, 3])
def test_real_numerical_seal_replay_without_any_solve(count, tmp_path, trusted_root, producer):
    cases = ("gap_lp", "gap_singular", "gap_spd")[:count]
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer, cases)
    before = audit.inventory(tree)
    collection = tree / "collection.json"
    bundle, sealed = audit.seal(collection, audit.sha(collection.read_bytes()), trusted_root,
                                tmp_path / "sealed", expectations_sha256=expected_pin)
    bundle_path = tmp_path / "sealed/bundle.json"
    put(bundle_path, bundle)
    accepted = audit.accept(bundle_path, audit.sha(bundle_path.read_bytes()), trusted_root,
                            tmp_path / "accepted", expectations_sha256=expected_pin)
    assert sealed["case_count"] == accepted["case_count"] == count
    assert accepted["numerical_verdict"] == accepted["execution_verdict"] == "PASS"
    assert all(entry["numerical"]["cases"][0]["passed"] for entry in accepted["entries"])
    assert accepted["entries"][0]["placement"]["gpu_uuid"] == UUID
    assert audit.inventory(tree) == before


@pytest.mark.parametrize("mutation", [
    "restart", "exit", "phase", "succeeded", "complete", "owner", "node", "image",
    "gpu_count", "retry", "command", "env", "mount", "bool_count", "extra_pod",
])
def test_actual_terminal_and_execution_mutations_rejected(mutation, tmp_path, trusted_root, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer)
    path = tree / "fixture-gap-lp-status.json"
    status = load(path)
    pod = status["pods"]["items"][0]
    cs = pod["status"]["containerStatuses"][0]
    container = pod["spec"]["containers"][0]
    if mutation == "restart":
        cs["restartCount"] = 1
    elif mutation == "exit":
        cs["state"]["terminated"]["exitCode"] = 1
    elif mutation == "phase":
        pod["status"]["phase"] = "Running"
    elif mutation == "succeeded":
        status["job"]["status"]["succeeded"] = 2
    elif mutation == "complete":
        status["job"]["status"]["conditions"] = []
    elif mutation == "owner":
        pod["metadata"]["ownerReferences"][0]["uid"] = "other-job"
    elif mutation == "node":
        pod["spec"]["nodeName"] = "other-node"
    elif mutation == "image":
        container["image"] = "other-image"
    elif mutation == "gpu_count":
        container["resources"]["requests"]["nvidia.com/gpu"] = "2"
    elif mutation == "retry":
        status["job"]["spec"]["backoffLimit"] = 1
    elif mutation == "command":
        container["command"][-1] = "0" * 64
    elif mutation == "env":
        container["env"].append({"name": "PYTHONHOME", "value": "/other-source"})
    elif mutation == "mount":
        container["volumeMounts"][0]["subPath"] = "other-source"
    elif mutation == "bool_count":
        status["job"]["status"]["succeeded"] = True
    else:
        status["pods"]["items"].append(deepcopy(pod))
    put(path, status)
    with pytest.raises(audit.AuditError):
        control(tree, expected_pin, trusted_root)


@pytest.mark.parametrize("mutation", ["missing", "sha", "runtime", "uuid", "trailing", "duplicate"])
def test_collected_log_origin_is_required(mutation, tmp_path, trusted_root, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer)
    path = tree / "fixture-gap-lp-pod.log"
    text = path.read_text()
    if mutation == "missing":
        path.unlink()
    elif mutation == "sha":
        digest = load(tree / "collection.json")["jobs"][0]["result_sha256"]
        path.write_text(text.replace(digest, "0" * 64))
    elif mutation == "runtime":
        path.write_text("\n".join(line for line in text.splitlines() if "RUNTIME " not in line) + "\n")
    elif mutation == "uuid":
        path.write_text(text.replace(UUID, "GPU-aaaaaaaa-1234-5678-9abc-123456789abc"))
    elif mutation == "trailing":
        path.write_text(text + "execution was not completed\n")
    else:
        path.write_text(text + text.splitlines()[-1] + "\n")
    with pytest.raises(audit.AuditError):
        control(tree, expected_pin, trusted_root)


@pytest.mark.parametrize("mutation", ["empty", "four", "dropped", "duplicate", "running", "result_twice"])
def test_exact_full_case_denominator(mutation, tmp_path, trusted_root, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer)
    run, collection = load(tree / "run.json"), load(tree / "collection.json")
    if mutation == "empty":
        run["jobs"] = collection["jobs"] = collection["states"] = []
    elif mutation == "four":
        run["jobs"] *= 4
        collection["jobs"] *= 4
        collection["states"] *= 4
    elif mutation == "dropped":
        collection["jobs"] = []
    elif mutation == "duplicate":
        run["jobs"] *= 2
        collection["jobs"] *= 2
        collection["states"] *= 2
    elif mutation == "running":
        collection["states"] = ["RUNNING"]
    else:
        put(tree / "case-0/cuda-proof-duplicate/result.json", {})
    put(tree / "run.json", run)
    put(tree / "collection.json", collection)
    with pytest.raises(audit.AuditError):
        control(tree, expected_pin, trusted_root)


def repin_result(tree, producer, saved, result):
    root = tree / "case-0/cuda-proof-explicit-simulation"
    put(root / "gap_lp.json", saved)
    result["cases"]["gap_lp"] = producer.descriptor(root, root / "gap_lp.json")
    put(root / "result.json", result)
    digest = audit.sha((root / "result.json").read_bytes())
    completion = (f"OPENOPT_GPU_PROOF verdict={result['verdict']} at={result['finished_at']} "
                  f"result={root / 'result.json'} result_sha256={digest}")
    (root / "execution.log").write_text(completion + "\nreturncode=0\n")
    collection = load(tree / "collection.json")
    collection["jobs"][0]["result_sha256"] = digest
    put(tree / "collection.json", collection)
    return root, digest, completion


def test_coordinated_cpu_to_cuda_new_artifact_pin_cannot_replace_collected_log(
        tmp_path, trusted_root, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer)
    root = tree / "case-0/cuda-proof-explicit-simulation"
    saved, result = load(root / "gap_lp.json"), load(root / "result.json")
    cpu_saved, cpu_result = deepcopy(saved), deepcopy(result)
    cpu_saved["device"] = cpu_saved["config"]["device"] = "cpu"
    for records in ("resident_model", "vectors", "observed"):
        for record in cpu_saved[records].values():
            record["device"] = "cpu"
    cpu_result.update(device="cpu", scope=producer.CPU_SCOPE, verdict="CPU_SMOKE_PASS")
    cpu_result["hardware"].update(requested_device="cpu", gpu=None, gpu_uuid=None,
                                  gpu_count=0, node=None, pod=None, cuda=None,
                                  cuda_capability=None, cpu_quota=None,
                                  cpu_scope="current process rusage; not Pod cgroup")
    samples = [{"cpu": {"usage_usec": value}} for value in (1, 2)]
    put(root / "cpu-telemetry.json", samples)
    cpu_result["telemetry"] = producer.descriptor(root, root / "cpu-telemetry.json")
    _, cpu_digest, cpu_completion = repin_result(tree, producer, cpu_saved, cpu_result)
    cpu_replay = producer.validate_result(
        root / "result.json", cpu_digest, trusted_root, allow_cpu_smoke=True,
        expectations=tree / "proof-expectations.json", expectations_sha256=expected_pin)
    assert cpu_replay["verdict"] == "CPU_SMOKE_REPLAY_PASS"
    assert cpu_replay["execution_verdict"] == "UNVERIFIED"
    log_path = tree / "fixture-gap-lp-pod.log"
    log_path.write_text("Explicit CPU fixture completion, not GPU execution\n" + cpu_completion + "\n")
    original_log = log_path.read_bytes()
    # Relabel every local device/runtime record and repin result/collection,
    # but preserve the independently collected CPU completion.
    result["hardware"]["torch"] = "synthetic CPU relabel"
    root, digest, _ = repin_result(tree, producer, saved, result)
    sealed = producer.seal(root / "result.json", digest, trusted_root,
                          expectations=tree / "proof-expectations.json",
                          expectations_sha256=expected_pin)
    assert sealed["scope"] == producer.GPU_SCOPE
    numerical = producer.validate_result(
        root / "result.json", digest, trusted_root,
        expectations=tree / "proof-expectations.json", expectations_sha256=expected_pin)
    assert numerical["execution_verdict"] == "UNVERIFIED"
    with pytest.raises(audit.AuditError, match="last completion"):
        control(tree, expected_pin, trusted_root)
    assert log_path.read_bytes() == original_log


def test_coordinated_whole_model_vectors_metrics_replacement_rejected_by_expected_pin(
        tmp_path, trusted_root, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer)
    root = tree / "case-0/cuda-proof-explicit-simulation"
    saved, result = load(root / "gap_lp.json"), load(root / "result.json")
    arrays = (np.eye(1), np.zeros(1), np.eye(1), -np.ones(1), np.ones(1))
    saved["model"] = tensor_records(producer, root, "gap_lp", arrays, "replacement")
    saved["resident_model"] = deepcopy(saved["model"])
    for record in saved["resident_model"].values():
        record["device"] = "cuda:0"
    saved["model_sha256"] = producer.model_hash(saved["model"])
    for name in ("x", "y"):
        saved["vectors"][name] = producer.save_tensor(root, "replacement-" + name, torch.zeros(1, dtype=torch.float64))
        saved["vectors"][name]["device"] = "cuda:0"
    checks = producer.original_checks(dict(zip(producer.FIELDS, arrays)),
                                      {"x": np.zeros(1), "y": np.zeros(1)}, "solved",
                                      saved["atol"], saved["rtol"])
    saved.update(independent_original=checks, numpy_original=checks, objective=0.)
    root, digest, completion = repin_result(tree, producer, saved, result)
    pod_log = tree / "fixture-gap-lp-pod.log"
    pod_log.write_text("\n".join(pod_log.read_text().splitlines()[:-1]) + "\n" + completion + "\n")
    assert producer.replay_case(root, result["cases"]["gap_lp"], "cuda:0")["passed"] is True
    with pytest.raises(audit.AuditError, match="whole model differs"):
        control(tree, expected_pin, trusted_root)


@pytest.mark.parametrize("mutation", ["expected_bytes", "expected_repin", "snapshot", "missing_source", "extra_source"])
def test_independent_expected_model_and_bidirectional_source_pins(
        mutation, tmp_path, trusted_root, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer)
    if mutation == "expected_bytes":
        path = next((tree / "arrays").glob("*.npy"))
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "expected_repin":
        path = tree / "proof-expectations.json"
        value = load(path)
        value["cases"]["gap_lp"]["model_sha256"] = "0" * 64
        put(path, value)
        run, collection = load(tree / "run.json"), load(tree / "collection.json")
        descriptor = {"path": str(path), "sha256": audit.sha(path.read_bytes()), "bytes": path.stat().st_size}
        run["expectations"] = collection["expectations"] = descriptor
        put(tree / "run.json", run)
        put(tree / "collection.json", collection)
    elif mutation == "extra_source":
        (tree / "source/src/openopt/unexpected.py").write_text("# unexpected\n")
    elif mutation == "missing_source":
        (tree / "source/scripts/cluster_benchmark.py").unlink()
    else:
        path = tree / "source/scripts/cluster_benchmark.py"
        path.write_bytes(path.read_bytes() + b"\n# changed\n")
    with pytest.raises((audit.AuditError, producer.ProofError)):
        control(tree, expected_pin, trusted_root)


@pytest.mark.parametrize("mutation", ["bytes", "added", "removed", "numerical_bundle"])
def test_caller_pinned_whole_bundle_inventory_rejects_tampering(
        mutation, tmp_path, trusted_root, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer)
    collection = tree / "collection.json"
    bundle, _ = audit.seal(collection, audit.sha(collection.read_bytes()), trusted_root,
                           tmp_path / "sealed", expectations_sha256=expected_pin)
    path = tmp_path / "sealed/bundle.json"
    put(path, bundle)
    if mutation == "bytes":
        target = tree / "fixture-gap-lp-status.json"
        target.write_bytes(target.read_bytes() + b" ")
    elif mutation == "added":
        (tree / "new-evidence.json").write_text("{}")
    elif mutation == "removed":
        (tree / "fixture-gap-lp-pod.log").unlink()
    else:
        target = Path(bundle["numerical_bundles"]["gap_lp"]["path"])
        target.write_bytes(target.read_bytes() + b" ")
    with pytest.raises(audit.AuditError):
        audit.accept(path, audit.sha(path.read_bytes()), trusted_root, tmp_path / "accepted",
                     expectations_sha256=expected_pin)


def test_current_root_actual_bytes_even_when_git_ignores_modification(tmp_path, trusted_root, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer)
    path = trusted_root / "LICENSE"
    original = path.read_bytes()
    git(trusted_root, "update-index", "--assume-unchanged", "--", "LICENSE")
    try:
        path.write_bytes(original + b"\n# ignored change\n")
        assert not git(trusted_root, "status", "--porcelain").strip()
        with pytest.raises(audit.AuditError, match="current committed source"):
            control(tree, expected_pin, trusted_root)
    finally:
        path.write_bytes(original)
        git(trusted_root, "update-index", "--no-assume-unchanged", "--", "LICENSE")
    assert audit.current_identity(trusted_root)["clean"]


def test_internal_symlink_and_ambiguous_json_rejected(tmp_path):
    directory = tmp_path / "directory"
    directory.mkdir()
    link = tmp_path / "symlink"
    link.symlink_to(directory, target_is_directory=True)
    with pytest.raises(audit.AuditError, match="symlink"):
        audit.external(link / "file.json")
    for value in ('{"k":1,"k":2}', '{"k":NaN}', '{"k":Infinity}'):
        with pytest.raises(audit.AuditError):
            audit.parse(value, "ambiguous")


def test_cli_failure_is_logged_and_never_unverified_then_pass(tmp_path, trusted_root):
    parent = tmp_path / "cli"
    code = audit.main(["seal", "--collection", str(tmp_path / "absent/collection.json"),
                       "--collection-sha256", "0" * 64, "--expectations-sha256", "1" * 64,
                       "--current-root", str(trusted_root), "--artifact-dir", str(parent)])
    assert code == 1
    reports = list(parent.glob("seal-*/acceptance.json"))
    assert len(reports) == 1
    report = load(reports[0])
    assert report["verdict"] == "FAIL" and report["returncode"] == 1
    assert report["execution_verdict"] == "UNVERIFIED"
    assert (reports[0].parent / "failure.log").stat().st_size > 0
    assert (reports[0].parent / "execution.log").read_text().rstrip().endswith("returncode=1")
