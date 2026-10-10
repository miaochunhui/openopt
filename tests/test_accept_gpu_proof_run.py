"""CPU-only integrity fixtures, not evidence of CUDA or cluster execution.

Legacy gap fixtures use analytic solutions. Outer-allnine fixtures reuse one
module-level real CPU production with pre-solve original-model pins. Audits
never solve or recreate models. A committed base plus explicitly selected local
source bytes form a clean test-only repository.
All generated records and
repositories live in pytest's external artifact directory. Fake Pod/GPU records
exercise the verifier only and must never be promoted to GPU release evidence.
"""

from copy import deepcopy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import io

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("proof_run_audit", ROOT / "scripts/accept_gpu_proof_run.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
REVIEW_SPEC = importlib.util.spec_from_file_location(
    "outer_review_loader", ROOT / "scripts/review_gpu_proof_contract.py")
review = importlib.util.module_from_spec(REVIEW_SPEC)
REVIEW_SPEC.loader.exec_module(review)
REVISION = os.environ.get("OPENOPT_PROOF_TEST_REVISION", "c451dcf")
UUID = "GPU-12345678-1234-5678-9abc-123456789abc"
GPU = "NVIDIA RTX PRO 4500 Blackwell"
ALL_NINE_GROUPS = (
    ("scaling_basic", "scaling_ill_conditioned", "jacobi_coo"),
    ("jacobi_csr", "gap_spd", "gap_lp"),
    ("gap_singular", "infeasible", "unbounded"),
)
RAY_MUTATIONS = (
    "status", "iteration_zero", "iteration_boolean", "iteration_fraction",
    "iteration_limit", "iteration_interval", "iteration_disagrees",
    "ray_sign", "ray_zero", "ray_missing", "base", "checkpoint_missing",
    "checkpoint_current", "checkpoint_delta", "checkpoint_initial",
    "rho", "selected_summary", "opposite_summary",
)


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(audit.encode(value))


def write_fixture_files(root, files):
    """Independent byte copies only; no links, metadata copying or gate cache."""
    paths = {root / audit.relative(name): data for name, data in files.items()}
    parents = {path.parent for path in paths}
    for parent in sorted(parents):
        parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(path.write_bytes, data) for path, data in paths.items()]
        for future, data in zip(futures, paths.values()):
            assert future.result() == len(data)


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
                    "docs/research/refinement.md", "tests/test_gpu_release_proof.py",
                    "docs/research/gpu-release-proof.md"}):
                continue
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(stream.extractfile(member).read())
    source = Path(os.environ.get("OPENOPT_PROOF_TEST_SOURCE_ROOT", str(ROOT)))
    if "OPENOPT_PROOF_TEST_SOURCE_ROOT" in os.environ:
        assert len(REVISION) == 40 and all(char in "0123456789abcdef" for char in REVISION)
        selected_identity = audit.current_identity(source)
        assert selected_identity["clean"] is True
        assert selected_identity["commit"] == REVISION, "external selector requires pinned clean HEAD"
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


@pytest.fixture(scope="module")
def frozen_source_files(trusted_root):
    """Freeze fixture inputs once; real acceptance still rechecks current bytes."""
    identity = audit.current_identity(trusted_root)
    selected = {name: digest for name, digest in identity["sha256"].items()
                if name.startswith(audit.PREFIXES) or name in audit.SNAPSHOT_FILES}
    files = {name: (trusted_root / name).read_bytes() for name in selected}
    assert {name: audit.sha(data) for name, data in files.items()} == selected
    yield {"identity": identity, "selected": selected, "files": files}
    assert audit.current_identity(trusted_root) == identity


@pytest.fixture
def producer(trusted_root, trusted_runtime, frozen_cpu_nine, frozen_source_files, monkeypatch):
    real = trusted_runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("acceptance attempted a solve, factory, CUDA or cluster action")

    import openopt.qp as qp
    import openopt.operators as operators
    import openopt.scaling as scaling

    for module, names in (
        (qp, ("solve_qp", "prepare_qp")),
        (qp.PreparedQP, ("solve",)),
        (operators, ("prepare_operator_qp", "conjugate_gradient")),
        (operators.PreparedOperatorQP, ("solve",)),
        (scaling, ("solve_scaled_qp",)),
        (real, ("run_cases", "execute_case", "make_problem", "hardware", "freeze_sources")),
        (torch.cuda, ("is_available", "device_count", "get_device_name", "synchronize")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    original_run = subprocess.run

    def git_only(command, *args, **kwargs):
        assert command and command[0] == "git", "audit attempted non-Git external execution"
        return original_run(command, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", git_only)
    monkeypatch.setattr(audit, "trusted_producer", lambda root, digest: real)
    monkeypatch.setattr(real, "_fixture_source_files", frozen_source_files, raising=False)
    assert real.seal.__module__ == real.replay.__module__
    assert real.ROOT == trusted_root
    return real


@contextmanager
def isolated_openopt(root):
    """Select one real source root, without bypassing loaded-source guards."""
    previous = {key: value for key, value in sys.modules.items()
                if key == "openopt" or key.startswith("openopt.")}
    search_path = sys.path[:]
    for name in previous:
        del sys.modules[name]
    sys.path.insert(0, str(root / "src"))
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if name == "openopt" or name.startswith("openopt."):
                del sys.modules[name]
        sys.modules.update(previous)
        sys.path[:] = search_path


@pytest.fixture(scope="module")
def trusted_runtime(trusted_root):
    digest = audit.sha((trusted_root / audit.PRODUCER).read_bytes())
    with isolated_openopt(trusted_root):
        real = audit.trusted_producer(trusted_root, digest)
        import openopt.qp as qp
        import openopt.certificates as certificates

        assert Path(qp.__file__).is_relative_to(trusted_root)
        assert Path(certificates.__file__).is_relative_to(trusted_root)
        assert qp.QPConfig.validate.__code__.co_filename == str(trusted_root / "src/openopt/qp.py")
        assert certificates.check_unbounded_direction.__code__.co_filename == str(
            trusted_root / "src/openopt/certificates.py")
        yield real


@pytest.fixture(scope="module")
def frozen_cpu_nine(tmp_path_factory, trusted_root, trusted_runtime):
    """One genuine CPU run, expectations frozen before the first solve."""
    proof = trusted_runtime
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    parent = tmp_path_factory.mktemp("real-cpu-nine-not-cuda")
    expected = {"schema_version": 1, "source_sha256": proof.source_map(trusted_root), "cases": {}}
    for case in proof.CASES:
        problem = proof.make_problem(case)
        model = {}
        for field in proof.FIELDS:
            tensor = getattr(problem, field)
            if case.startswith("jacobi_") and field in ("P", "A"):
                tensor = tensor.to_sparse()
                if case == "jacobi_csr":
                    tensor = tensor.to_sparse_csr()
            model[field] = proof.save_tensor(parent, "expected-" + case + "-" + field, tensor)
        expected["cases"][case] = {"model": model, "model_sha256": proof.model_hash(model)}
    expectation_path = parent / "proof-expectations.json"
    proof.write(expectation_path, expected)
    expected_pin = audit.sha(expectation_path.read_bytes())
    result = proof.run_cases(parent / "production", list(proof.CASES), "cpu", cpu_smoke=True,
                             expectations=expectation_path, expectations_sha256=expected_pin)
    result_pin = audit.sha(result.read_bytes())
    assert load(result)["verdict"] == "CPU_SMOKE_PASS"
    assert load(result)["execution_verdict"] == "UNVERIFIED"
    assert set(load(result)["cases"]) == set(proof.CASES) == set(sum(ALL_NINE_GROUPS, ()))
    saved = load(result.parent / "unbounded.json")
    base = proof.load_tensor(result.parent, saved["vectors"]["feasible_base"], "actual base")
    x = proof.load_tensor(result.parent, saved["vectors"]["x"], "terminal x")
    assert np.array_equal(base, x) and base[1] > 2
    frozen = {"fixture": "REAL_CPU_ONLY; coordinator/Pod/GPU records simulated separately",
              "result": str(result), "result_sha256": result_pin,
              "expectations": str(expectation_path), "expectations_sha256": expected_pin,
              "case_count": 9, "source_sha256": proof.source_map(trusted_root)}
    put(parent / "cpu-receipt.json", frozen)
    before = proof.inventory(parent)
    sidecars = {case: load(result.parent / (case + ".json")) for case in proof.CASES}
    expected_files = frozen_raw_files(expected, parent)
    case_files = {case: frozen_raw_files(saved, result.parent)
                  for case, saved in sidecars.items()}
    yield {"root": parent, "result": result, "expected": expected,
           "sidecars": sidecars, "expected_files": expected_files, "case_files": case_files}
    assert proof.inventory(parent) == before


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


def fake_result(producer, root, case, sources, model, expectation, cpu=None):
    root.mkdir(parents=True)
    write_fixture_files(root / "source",
                        {name: producer._fixture_source_files["files"][name] for name in sources})
    if cpu is not None:
        saved = deepcopy(cpu["sidecars"][case])
        copy_raw_descriptors(saved, cpu["result"].parent, root, cpu["case_files"][case])
        relabel_runtime(saved)
        saved["device"] = saved["config"]["device"] = "cuda:0"
        for record in saved["model"].values():
            record["device"] = "cpu"
        for key in ("infeasibility_direction", "unbounded_direction", "feasible_base"):
            if key in saved["vectors"]:
                saved["vectors"][key]["device"] = "cpu"
        saved["qualification"] = "REAL CPU numerical solve; SIMULATED CUDA/coordinator metadata"
        assert saved["model_sha256"] == producer.model_hash(model)
    else:
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


def raw_descriptors(value):
    records = {}

    def collect(item):
        if isinstance(item, dict):
            if set(item) == {"path", "sha256", "bytes"}:
                name = audit.relative(item["path"])
                if name in records:
                    assert records[name] == item, "conflicting fixture raw descriptor"
                else:
                    records[name] = item
            else:
                for nested in item.values():
                    collect(nested)
        elif isinstance(item, list):
            for nested in item:
                collect(nested)

    collect(value)
    return records


def frozen_raw_files(value, source):
    """Cache verified immutable fixture inputs, not replay/inventory outcomes."""
    records = raw_descriptors(value)

    def checked(item):
        name, record = item
        data = audit.bound(source, record).read_bytes()
        assert len(data) == record["bytes"] and audit.sha(data) == record["sha256"]
        return name, data

    with ThreadPoolExecutor(max_workers=8) as executor:
        return dict(executor.map(checked, records.items()))


def copy_raw_descriptors(value, source, target, frozen_files=None):
    records = raw_descriptors(value)
    inputs = frozen_raw_files(value, source) if frozen_files is None else frozen_files
    files = {name: inputs[name] for name in records}
    assert all(len(files[name]) == record["bytes"] and audit.sha(files[name]) == record["sha256"]
               for name, record in records.items())
    write_fixture_files(target, files)


def relabel_runtime(value):
    """Fixture metadata only; raw numerical bytes are never changed."""
    if isinstance(value, dict):
        if "device" in value:
            value["device"] = "cuda:0"
        if "observed_device" in value:
            value["observed_device"] = "cuda:0"
        for item in value.values():
            relabel_runtime(item)
    elif isinstance(value, list):
        for item in value:
            relabel_runtime(item)


def fixture_run(tmp_path, trusted_root, producer, cases=("gap_lp",), cpu=None):
    tree = tmp_path / "simulated-release-proof"
    tree.mkdir()
    frozen_sources = producer._fixture_source_files
    identity = deepcopy(frozen_sources["identity"])
    selected = frozen_sources["selected"]
    producer_sources = {k: v for k, v in selected.items()
                        if (k.startswith("src/") and k.endswith(".py")) or k == audit.PRODUCER}
    write_fixture_files(tree / "source", frozen_sources["files"])
    put(tree / "source/source-hashes.json", selected)
    model_root = tree
    expected = {"schema_version": 1, "source_sha256": producer_sources, "cases": {}}
    for case in cases:
        if cpu is None:
            arrays, _ = model_arrays(case)
            records = tensor_records(producer, model_root, case, arrays, "expected")
            expected["cases"][case] = {"model": records, "model_sha256": producer.model_hash(records)}
        else:
            entry = deepcopy(cpu["expected"]["cases"][case])
            copy_raw_descriptors(entry, cpu["root"], model_root, cpu["expected_files"])
            expected["cases"][case] = entry
    path = tree / "proof-expectations.json"
    put(path, expected)
    expectation = {"path": str(path), "sha256": audit.sha(path.read_bytes()), "bytes": path.stat().st_size}
    jobs, collected = [], []
    for index, case in enumerate(cases):
        item = fake_manifest(tree, index, case, expectation)
        result_path = tree / f"case-{index}/cuda-proof-explicit-simulation/result.json"
        result, completion, digest = fake_result(
            producer, result_path.parent, case, producer_sources, expected["cases"][case]["model"],
            expectation, cpu=cpu)
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
    case = saved["case"]
    put(root / (case + ".json"), saved)
    result["cases"][case] = producer.descriptor(root, root / (case + ".json"))
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


# Outer coverage groups share the single real CPU producer fixture above.
@pytest.mark.parametrize("cases", ALL_NINE_GROUPS, ids=("scaling-coo", "csr-gaps", "singular-rays"))
def test_outer_allnine_real_cpu_frozen_seal_accept(
        cases, tmp_path, trusted_root, frozen_cpu_nine, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer, cases, cpu=frozen_cpu_nine)
    expectation = load(tree / "proof-expectations.json")
    before = audit.inventory(tree)
    for case in cases:
        assert expectation["cases"][case]["model_sha256"] == \
            frozen_cpu_nine["expected"]["cases"][case]["model_sha256"]
        root = tree / f"case-{cases.index(case)}/cuda-proof-explicit-simulation"
        saved = load(root / (case + ".json"))
        original = load(frozen_cpu_nine["result"].parent / (case + ".json"))
        assert saved["iterations"] == original["iterations"]
        assert saved["independent_original"] == original["independent_original"]
        assert saved["numpy_original"] == original["numpy_original"]
        assert "SIMULATED" in saved["qualification"]
    path = tree / "collection.json"
    bundle, sealed = audit.seal(path, audit.sha(path.read_bytes()), trusted_root,
                                tmp_path / "sealed", expectations_sha256=expected_pin)
    bundle_path = tmp_path / "sealed/bundle.json"
    put(bundle_path, bundle)
    accepted = audit.accept(bundle_path, audit.sha(bundle_path.read_bytes()), trusted_root,
                            tmp_path / "accepted", expectations_sha256=expected_pin)
    assert sealed["case_count"] == accepted["case_count"] == 3
    assert accepted["requested_cases"] == list(cases)
    assert accepted["verdict"] == accepted["numerical_verdict"] == "PASS"
    assert accepted["execution_verdict"] == "PASS"  # Simulated coordinator contract only.
    for entry in accepted["entries"]:
        numerical = entry["numerical"]
        assert numerical["case_count"] == 1 and numerical["verdict"] == producer.NUMERICAL_PASS
        assert numerical["execution_verdict"] == "UNVERIFIED"
        assert numerical["cases"][0]["case"] == entry["case"]
        assert numerical["cases"][0]["passed"] is True
        assert entry["placement"]["pod"].startswith("fixture-")
    assert audit.inventory(tree) == before
    put(tmp_path / "outer-coverage-receipt.json", {
        "fixture": "REAL_CPU_NUMERICS_FAKE_COORDINATOR; NOT CUDA EXECUTION EVIDENCE",
        "cases": list(cases), "cpu_result_sha256": audit.sha(frozen_cpu_nine["result"].read_bytes()),
        "expectations_sha256": expected_pin, "collection_sha256": audit.sha(path.read_bytes()),
        "outer_contract_verdict": accepted["verdict"], "actual_execution_verdict": "UNVERIFIED",
    })


@pytest.mark.parametrize("cases", [("infeasible",), ("infeasible", "unbounded")], ids=("one", "two"))
def test_outer_allnine_direction_run_denominators(
        cases, tmp_path, trusted_root, frozen_cpu_nine, producer):
    tree, pin = fixture_run(tmp_path, trusted_root, producer, cases, cpu=frozen_cpu_nine)
    path = tree / "collection.json"
    outcome, _ = audit.validate(path, audit.sha(path.read_bytes()), trusted_root,
                                expectations_sha256=pin, numerical_dir=tmp_path / "numerical")
    assert outcome["case_count"] == len(cases)
    assert outcome["requested_cases"] == list(cases)
    assert all(entry["numerical"]["cases"][0]["status"] == entry["case"]
               for entry in outcome["entries"])


def mutate_ray(producer, root, saved, mutation):
    case = saved["case"]
    trace = saved["extensions"]["checkpoint"]
    name = "infeasibility_direction" if case == "infeasible" else "unbounded_direction"

    def vector(group, key, values):
        group[key] = producer.save_tensor(
            root, "mutated-" + mutation + "-" + key, torch.from_numpy(np.asarray(values)))
        group[key]["device"] = "cpu" if group is saved["vectors"] and key not in ("x", "y") else "cuda:0"

    if mutation == "status":
        saved["status"] = saved["expected_status"] = "unbounded" if case == "infeasible" else "infeasible"
    elif mutation.startswith("iteration_"):
        values = {"iteration_zero": 0, "iteration_boolean": True, "iteration_fraction": 25.5,
                  "iteration_limit": saved["config"]["max_iterations"] + 1, "iteration_interval": 1}
        if mutation == "iteration_disagrees":
            trace["iteration"] += 1
        else:
            saved["iterations"] = trace["iteration"] = values[mutation]
    elif mutation in ("ray_sign", "ray_zero"):
        raw = producer.load_tensor(root, saved["vectors"][name], name)
        vector(saved["vectors"], name, -raw if mutation == "ray_sign" else np.zeros_like(raw))
    elif mutation == "ray_missing":
        del saved["vectors"][name]
    elif mutation == "base":
        if case == "unbounded":
            # Numerically feasible but not the solver's actual terminal base.
            vector(saved["vectors"], "feasible_base", np.array([0., 2.]))
        else:
            vector(saved["vectors"], "feasible_base", np.zeros(saved["model"]["q"]["shape"]))
    elif mutation == "checkpoint_missing":
        del saved["extensions"]["checkpoint"]
    elif mutation in ("checkpoint_current", "checkpoint_delta", "checkpoint_initial"):
        key = {"checkpoint_current": "current_x", "checkpoint_delta": "dx",
               "checkpoint_initial": "previous_y"}[mutation]
        raw = producer.load_tensor(root, trace["tensors"][key], key)
        vector(trace["tensors"], key, raw + 1)
    elif mutation == "rho":
        trace["rho"] *= 2
    elif mutation == "selected_summary":
        selected = trace[case + "_evidence"]
        metric = next(iter(selected["metrics"]))
        selected["metrics"][metric] += 1
    else:
        assert mutation == "opposite_summary"
        opposite = "unbounded" if case == "infeasible" else "infeasible"
        trace[opposite + "_evidence"]["passed"] = True
        trace[opposite + "_evidence"]["checks"] = {
            key: True for key in trace[opposite + "_evidence"]["checks"]}


@pytest.mark.parametrize("case", ("infeasible", "unbounded"))
@pytest.mark.parametrize("mutation", RAY_MUTATIONS)
def test_outer_allnine_ray_semantics_resealed_rejected(
        case, mutation, tmp_path, trusted_root, frozen_cpu_nine, producer):
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer, (case,), cpu=frozen_cpu_nine)
    root = tree / "case-0/cuda-proof-explicit-simulation"
    saved, result = load(root / (case + ".json")), load(root / "result.json")
    mutate_ray(producer, root, saved, mutation)
    root, digest, completion = repin_result(tree, producer, saved, result)
    pod_log = tree / f"fixture-{case}-pod.log"
    old_lines = pod_log.read_text().splitlines()
    pod_log.write_text("\n".join(old_lines[:-1]) + "\n" + completion + "\n")
    before = audit.inventory(tree)
    control_result = control(tree, expected_pin, trusted_root)
    assert len(control_result[3]) == 1  # All independent control pins agree.
    collection = tree / "collection.json"
    with pytest.raises(producer.ProofError) as caught:
        audit.seal(collection, audit.sha(collection.read_bytes()), trusted_root,
                   tmp_path / "sealed", expectations_sha256=expected_pin)
    assert case in str(caught.value) or mutation in ("ray_sign", "ray_zero")
    assert "hash mutation" not in str(caught.value)
    assert audit.inventory(tree) == before
    put(tmp_path / "outer-ray-rejection-receipt.json", {
        "fixture": "REAL_CPU_NUMERICS_FAKE_COORDINATOR; NOT CUDA EXECUTION EVIDENCE",
        "case": case, "mutation": mutation, "verdict": "REJECTED",
        "reason": str(caught.value), "result_sha256": digest, "expectations_sha256": expected_pin,
        "caller_collection_sha256": audit.sha(collection.read_bytes()),
    })


def test_outer_allnine_loaded_foreign_module_rejected_not_guard_bypassed(
        tmp_path, trusted_root, frozen_cpu_nine, producer, monkeypatch):
    if not hasattr(producer, "loaded_source_check"):
        pytest.skip("Mendel loaded-source guard not present in this pinned producer")
    tree, expected_pin = fixture_run(tmp_path, trusted_root, producer, ("infeasible",), cpu=frozen_cpu_nine)
    import openopt.certificates as current
    from types import ModuleType
    foreign = ModuleType("openopt.certificates")
    foreign.__dict__.update(current.__dict__)
    foreign.__file__ = str(tmp_path / "not-trusted-certificates.py")
    monkeypatch.setitem(sys.modules, "openopt.certificates", foreign)
    monkeypatch.setattr(sys.modules["openopt"], "certificates", foreign)
    collection = tree / "collection.json"
    with pytest.raises(producer.ProofError, match="loaded source"):
        audit.seal(collection, audit.sha(collection.read_bytes()), trusted_root,
                   tmp_path / "sealed", expectations_sha256=expected_pin)


@pytest.mark.parametrize("exceptional", (False, True), ids=("normal", "exception"))
def test_review_loader_selects_verified_root_and_restores_cache(
        exceptional, trusted_root, monkeypatch, tmp_path):
    from types import ModuleType
    foreign = ModuleType("openopt")
    foreign.__file__ = str(tmp_path / "foreign-package.py")
    monkeypatch.setitem(sys.modules, "openopt", foreign)
    before = {name: module for name, module in sys.modules.items()
              if name == "openopt" or name.startswith("openopt.")}
    search = sys.path[:]
    commit = audit.current_identity(trusted_root)["commit"]

    def selected():
        with review.verified_source_selection(trusted_root, commit):
            import openopt.qp as qp
            import openopt.certificates as certificates
            assert Path(qp.__file__) == trusted_root / "src/openopt/qp.py"
            assert Path(certificates.__file__) == trusted_root / "src/openopt/certificates.py"
            assert sys.modules["openopt"] is not foreign
            real = review.load_module(trusted_root / audit.PRODUCER, "loader_selected_proof")
            if hasattr(real, "loaded_source_check"):
                identity = real.loaded_source_check(trusted_root)
                assert identity["root"] == str(trusted_root)
            if exceptional:
                raise RuntimeError("selector restoration negative")

    if exceptional:
        with pytest.raises(RuntimeError, match="selector restoration negative"):
            selected()
    else:
        selected()
    assert sys.path == search
    assert {name: module for name, module in sys.modules.items()
            if name == "openopt" or name.startswith("openopt.")} == before


def test_review_loader_wrong_commit_does_not_replace_modules(trusted_root):
    before = dict(sys.modules)
    search = sys.path[:]
    with pytest.raises(review.ReviewError, match="full commit"):
        with review.verified_source_selection(trusted_root, "not-a-full-commit"):
            pytest.fail("unverified source must not be selected")
    assert sys.path == search
    assert all(sys.modules.get(name) is module for name, module in before.items()
               if name == "openopt" or name.startswith("openopt."))


@pytest.mark.parametrize("mutation", ("wrong_pin", "added", "removed", "changed"))
def test_review_loader_pins_fail_before_source_selection(mutation, tmp_path, monkeypatch):
    root = tmp_path / "immutable-review"
    root.mkdir()
    put(root / "review.json", {"verdict": "CONTRACT_REVIEW_PASS"})
    put(root / "original.json", {"fixture": "CPU-only"})
    bundle = {"schema_version": 1, "evidence_root": str(root),
              "files": review.inventory(root),
              "report": review.descriptor(root, root / "review.json")}
    path = tmp_path / "bundle.json"
    put(path, bundle)
    digest = audit.sha(path.read_bytes())
    reason = "caller pin mismatch" if mutation == "wrong_pin" else "added, removed or changed"
    if mutation == "wrong_pin":
        digest = "0" * 64
    elif mutation == "added":
        put(root / "added.json", {})
    elif mutation == "removed":
        (root / "review.json").unlink()
    else:
        (root / "review.json").write_bytes(b"mutated")
    monkeypatch.setattr(review, "external", lambda value: Path(value))

    def forbidden(*args, **kwargs):
        pytest.fail("invalid bundle must fail before source selection/import")

    monkeypatch.setattr(review, "pinned_producer", forbidden)
    monkeypatch.setattr(review, "load_module", forbidden)
    with pytest.raises(review.ReviewError, match=reason):
        review.replay(path, digest, tmp_path, "1" * 40, tmp_path, "2" * 40)
