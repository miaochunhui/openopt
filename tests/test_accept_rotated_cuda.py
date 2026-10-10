"""Self-contained CPU controls. Synthetic archives never establish CUDA."""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

from scripts import accept_rotated_cuda as auditor


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    auditor.write_json(path, value)
    return path


def evidence(replay, atol, rtol):
    checks = replay["checks"]
    metrics = {key: 0. for key in checks if key not in ("dual_objective_available", "duality_gap")}
    metrics.update(primal_objective=float(auditor.Fraction(replay["metrics"]["primal_objective"])),
                   dual_objective=float(auditor.Fraction(replay["metrics"]["dual_objective"]))
                   if replay["metrics"]["dual_objective"] is not None else None,
                   duality_gap=float(auditor.Fraction(replay["metrics"]["signed_duality_gap"]))
                   if replay["metrics"]["signed_duality_gap"] is not None else None,
                   range_residual=0. if checks["dual_cone"] else None)
    thresholds = {"primal_equation_max": atol, "stationarity_max": atol,
                  "spectral": 0., "range_max": atol}
    thresholds.update({key: atol for key in checks if key.startswith("complementarity_")})
    if checks["dual_objective_available"]:
        thresholds["duality_gap"] = atol
    return {"schema_version": 1, "kind": "conic_candidate_optimality",
            "passed": replay["passed"], "checks": checks, "metrics": metrics,
            "thresholds": thresholds, "unavailable": {} if checks["dual_objective_available"]
            else {"dual_objective": "synthetic numerical rejection"},
            "qualification": "approximate numerical evidence; not an exact proof"}


def model(qp=False):
    d = 1. if qp else 0.
    return {"P": [[d, 0., 0.], [0., d, 0.], [0., 0., 0.]],
            "q": [0., 0., 0.] if qp else [1., 1., 0.],
            "A": [[0., 0., 1.], [-1., -0., -0.], [-0., -1., -0.], [-0., -0., -1.]],
            "b": [2., 0., 0., 0.],
            "cones": [{"kind": "zero", "dimension": 1}, {"kind": "rsoc", "dimension": 3}]}


def analytic(qp=False):
    r = math.sqrt(2)
    return {"x": [r, r, 2.], "s": [0., r, r, 2.],
            "z": [-2., r, r, -2.] if qp else [-r, 1., 1., -r]}


def sidecar(identity, sources=None):
    kind, case = auditor.IDENTITIES[identity]
    dtype = "torch.float64" if kind == "original_rejection" else "torch." + case.rsplit("-", 1)[1]
    atol = 1e-12 if kind == "projection" else 3e-5 if dtype == "torch.float32" else 1e-8
    rtol = (3e-6 if dtype == "torch.float32" else 3e-14) if kind == "projection" else 0
    qp = case.startswith("rotated_qp")
    problem, vectors = model(qp), analytic(qp)
    baseline = deepcopy(vectors)
    if kind == "projection":
        u = auditor.represented(auditor.INPUTS[case], dtype)
        s = auditor.represented(auditor.projection_oracle(u), dtype)
        vectors = {"x": [0.], "s": s, "z": [0.] * len(s)}
        problem = {"P": [[0.]], "q": [0.], "A": [[0.]] * len(s), "b": s,
                   "cones": [{"kind": "rsoc", "dimension": len(s)}]}
    elif kind == "original_rejection":
        if case == "wrong-equality-dual":
            vectors["z"][0] *= -1
        elif case == "planted-slack":
            vectors["x"] = [0., 0., 2.]
        else:
            vectors["z"][1:] = [-v for v in vectors["z"][1:]]
    for key in vectors:
        vectors[key] = auditor.represented(vectors[key], dtype)
    replay = auditor.candidate(problem, vectors, atol, rtol)
    saved_evidence = evidence(replay, atol, rtol)
    for key, value in replay["metrics"].items():
        if key.startswith("complementarity_"):
            saved_evidence["metrics"][key] = float(auditor.Fraction(value))
    saved_evidence["thresholds"] = auditor.recorded_thresholds(saved_evidence, problem, vectors, atol, rtol)
    n, m = len(problem["q"]), len(problem["b"])
    shapes = {"x": [n], "s": [m], "z": [m], "original.P": [n, n], "original.q": [n],
              "original.A": [m, n], "original.b": [m]}
    if kind == "projection":
        shapes.update({"projection.input": [m], "projection.output": [m]})
    else:
        shapes.update({"prepared.P": [n, n], "prepared.q": [n], "prepared.A": [m, n],
                       "prepared.b": [m], "prepared.factor": [n, n]})
    metadata = {key: {"shape": shape, "device": "cuda:0", "dtype": "torch.float64"
                     if key.startswith("original.") else dtype, "layout": "torch.strided"}
                for key, shape in shapes.items()}
    cone_evidence = {}
    for key, checks in replay["cone_checks"].items():
        cone_evidence[key] = {
            "schema_version": 1, "kind": "dual_cone_feasibility" if key == "dual" else "cone_feasibility",
            "passed": all(checks.values()), "checks": checks,
            "metrics": {name: 0. for name in checks},
            "thresholds": {name: atol for name in checks}, "unavailable": {},
            "qualification": "approximate numerical evidence; not an exact proof"}
    report = {
        "schema_version": 1, "family": "rotated_conic_cuda", "verdict": "PASS",
        "checked_at": "2026-10-10T12:00:05+00:00",
        "test_identity": identity, "case_kind": kind, "case_id": case,
        "device": "cuda:0", "dtype": dtype,
        "config": {"enabled": True, "device": "cuda:0", "dtype": dtype, "rho": 1.,
                   "sigma": 1e-6, "max_iterations": 10000, "check_interval": 25,
                   "eps_abs": atol, "eps_rel": rtol},
        "runtime": {"python": "synthetic", "implementation": "synthetic", "executable": "/synthetic",
                    "numpy": "synthetic", "torch": "synthetic", "compiled_cuda": "synthetic",
                    "cuda_available": True, "cuda_execution": True,
                    "gpu_name": "NVIDIA RTX PRO 4500 Blackwell", "capability": [12, 0],
                    "environment": {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                                    "MKL_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": None}},
        "source_sha256": sources or {}, "loaded_sources": {},
        "tensor_metadata": metadata, "problem": problem, **vectors,
        "status": {"original_model": "solved", "projection": "projected",
                   "original_rejection": "candidate_rejected"}[kind],
        "iterations": 0 if kind == "projection" else 25,
        "evidence": saved_evidence, "cone_evidence": cone_evidence,
        "objective": saved_evidence["metrics"]["primal_objective"],
        "certified": False, "qualified": False, "commercial": "UNVERIFIED", "performance": "UNVERIFIED",
    }
    if sources:
        report["loaded_sources"] = {
            name: {"source": "src/" + name.replace(".", "/") + ".py",
                   "sha256": sources["src/" + name.replace(".", "/") + ".py"]}
            for name in ("openopt.conic", "openopt.native_conic", "openopt.certificates")}
    if kind == "original_model":
        av = analytic(qp)
        ae = evidence(auditor.candidate(problem, av, 1e-12, 0), 1e-12, 0)
        ae["thresholds"] = auditor.recorded_thresholds(ae, problem, av, 1e-12, 0)
        report["analytic"] = {"vectors": av, "evidence": ae}
    elif kind == "projection":
        report["projection"] = {"input": u, "output": vectors["s"], "input_unchanged": True,
                                "oracle": {"algorithm": "bounded_decimal_analytic_rsoc_v1",
                                           "precision": 800, "expected": auditor.projection_oracle(u),
                                           "atol": 0., "rtol": rtol, "passed": True}}
    else:
        report["negative"] = {
            "mutation": case, "expected_failed_checks": auditor.NEGATIVES[case],
            "baseline": {**baseline, "evidence": evidence(
                auditor.candidate(problem, baseline, atol, rtol), atol, rtol),
                "status": "solved", "tensor_metadata": {key: metadata[key] for key in baseline}}}
    return report


@pytest.mark.parametrize("identity", auditor.IDENTITIES)
def test_thirteen_synthetic_math_controls(identity):
    saved = sidecar(identity)
    before = deepcopy(saved)
    result = auditor.replay_sidecar(saved)
    assert result["identity"] == identity
    assert saved == before
    assert result["passed"] is (result["kind"] != "original_rejection")


@pytest.mark.parametrize("mutation,reason", [
    ("placement", "placement mismatch"), ("missing-model", "complete original model"),
    ("config", "full config"), ("false-check", "verdict mismatch"),
    ("tiny-head", "component oracle"), ("negative-baseline", "baseline"),
    ("threshold", "threshold contract"),
])
def test_numeric_negative_controls(mutation, reason):
    if mutation == "tiny-head":
        identity = next(i for i in auditor.IDENTITIES if "[small-head-float64]" in i)
    elif mutation == "negative-baseline":
        identity = next(i for i in auditor.IDENTITIES if "[wrong-equality-dual]" in i)
    else:
        identity = next(i for i in auditor.IDENTITIES if "[rotated_lp-float64]" in i)
    saved = sidecar(identity)
    if mutation == "placement":
        saved["tensor_metadata"]["s"]["device"] = "cpu"
    elif mutation == "missing-model":
        del saved["problem"]["P"]
    elif mutation == "config":
        saved["config"]["eps_abs"] *= 2
    elif mutation == "false-check":
        saved["evidence"]["checks"]["duality_gap"] = False
    elif mutation == "threshold":
        saved["evidence"]["thresholds"]["duality_gap"] *= 2
    elif mutation == "tiny-head":
        saved["s"][0] *= 2
        saved["problem"]["b"] = saved["s"]
        saved["projection"]["output"] = saved["s"]
    else:
        saved["negative"]["baseline"]["x"] = [0., 0., 2.]
    with pytest.raises(ValueError, match=reason):
        auditor.replay_sidecar(saved)


def test_exact_head_sign_and_small_head_underflow():
    cones = [{"kind": "rsoc", "dimension": 3}]
    assert not auditor.cone_checks(list(map(auditor.number, [-1., 1e300, 0.])),
                                   cones, auditor.number(1e-8), auditor.number(0))["block_0"]
    assert not auditor.cone_checks(list(map(auditor.number, [0., 1e308, 1e-200])),
                                   cones, auditor.number(0), auditor.number(0))["block_0"]


def test_no_third_party_or_producer_imports():
    import ast
    tree = ast.parse(Path(auditor.__file__).read_text())
    modules = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    modules += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    assert not any(name and name.split(".")[0] in ("torch", "numpy", "openopt", "tests")
                   for name in modules)
    assert not any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                   and node.func.attr in ("solve", "project_cones") for node in ast.walk(tree))


def fixture_archive(tmp_path):
    """Fully local synthetic provenance/control, never emitted as real GPU receipt."""
    source = tmp_path / "synthetic-source"
    names = ["src/openopt/conic.py", "src/openopt/native_conic.py", "src/openopt/certificates.py",
             auditor.PRODUCER, "scripts/gpu_release_check.py"]
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Synthetic unexecuted source bytes, not an optimizer."""\n')
    for args in (["init"], ["add", *names], ["-c", "user.name=Synthetic", "-c",
                                          "user.email=synthetic@example.invalid", "commit", "-m", "fixture"]):
        result = subprocess.run(["git", "-C", str(source), *args], capture_output=True)
        assert result.returncode == 0, result.stderr
    commit = auditor.git(source, "rev-parse", "HEAD").decode().strip()
    tree = auditor.git(source, "rev-parse", "HEAD^{tree}").decode().strip()
    root = tmp_path / "synthetic-run"
    hashes = {name: auditor.sha(source / name) for name in names}
    for name in names:
        path = root / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    write(root / "source/source-hashes.json", hashes)
    scope = {name: digest for name, digest in hashes.items()
             if name.startswith("src/") or name == auditor.PRODUCER}
    result_root = root / "case-0/cuda-check-synthetic"
    suite = ET.Element("testsuite", tests="13", failures="0", errors="0", skipped="0")
    for index, identity in enumerate(auditor.IDENTITIES):
        path = write(result_root / "pytest-tmp" / f"case-{index}" / "rotated-conic.json",
                     sidecar(identity, scope))
        classname, name = identity.split("::")
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        props = ET.SubElement(case, "properties")
        for key, value in (("rotated_conic_acceptance_path", str(path)),
                           ("rotated_conic_acceptance_sha256", auditor.sha(path))):
            ET.SubElement(props, "property", name=key, value=value)
    suites = ET.Element("testsuites")
    suites.append(suite)
    (result_root / "pytest.xml").write_bytes(ET.tostring(suites))
    (result_root / "pytest.log").write_text("13 passed\nreturncode=0\n")
    log = f"OPENOPT_RELEASE_CUDA verdict=PASS tests=13 evidence={result_root}\n"
    (result_root / "execution.log").write_text(log)
    result = {"verdict": "PASS", "returncode": 0, "targets": [auditor.PRODUCER],
              "selection": "cuda_rotated",
              "tests": {"tests": 13, "failures": 0, "errors": 0, "skipped": 0},
              "source_sha256": hashes,
              "started_at": "2026-10-10T12:00:00+00:00", "finished_at": "2026-10-10T12:00:10+00:00",
              "hardware": {"node": "synthetic-node", "gpu": "NVIDIA RTX PRO 4500 Blackwell",
                           "torch": "synthetic", "cuda": "synthetic"}, "telemetry_samples": 1}
    write(result_root / "result.json", result)
    write(root / "case-0/result.json", result)
    write(result_root / "telemetry.json", [{
        "at": "2026-10-10T12:00:05+00:00", "monotonic_s": 5., "rc": 0, "error": None,
        "gpu": "GPU-00000000-0000-0000-0000-000000000000, 0, 0, 0", "cpu": {"usage_usec": 1}}])
    name = "synthetic-job"
    spec = {"containers": [{
        "name": "benchmark", "image": "synthetic-image", "imagePullPolicy": "Never",
        "command": ["python3", str(root / "source/scripts/gpu_release_check.py"), "--artifact-dir",
                    str(root / "case-0"), "--target", auditor.PRODUCER, "--expression", "cuda_rotated"],
        "env": auditor.expected_environment(root / "source", root / "case-0"),
        "resources": {key: {"cpu": "4", "memory": "12Gi", "nvidia.com/gpu": 1}
                      for key in ("requests", "limits")},
        "securityContext": {"allowPrivilegeEscalation": False}}],
        "restartPolicy": "Never", "runtimeClassName": "nvidia",
        "nodeSelector": {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-4500-Blackwell"}}
    job = {"apiVersion": "batch/v1", "kind": "Job",
           "metadata": {"name": name, "namespace": "synthetic"},
           "spec": {"template": {"spec": spec}, "backoffLimit": 0}}
    write(root / "job-0.json", job)
    actual_job = deepcopy(job)
    actual_job["metadata"]["uid"] = "synthetic-job-uid"
    actual_job["spec"].update(parallelism=1, completions=1)
    actual_job["status"] = {"succeeded": 1, "startTime": "2026-10-10T12:00:00Z",
                            "completionTime": "2026-10-10T12:00:11Z",
                            "conditions": [{"type": "Complete", "status": "True"}]}
    pod = {"apiVersion": "v1", "kind": "Pod",
           "metadata": {"name": "synthetic-pod", "uid": "synthetic-pod-uid", "namespace": "synthetic",
                        "ownerReferences": [{"apiVersion": "batch/v1", "kind": "Job", "name": name,
                                             "uid": "synthetic-job-uid", "controller": True}]},
           "spec": deepcopy(spec), "status": {"phase": "Succeeded", "containerStatuses": [{
               "name": "benchmark", "restartCount": 0, "lastState": {},
               "imageID": "synthetic@sha256:" + "0" * 64,
               "state": {"terminated": {"exitCode": 0, "reason": "Completed",
                                       "startedAt": "2026-10-10T12:00:00Z",
                                       "finishedAt": "2026-10-10T12:00:10Z"}}}]}}
    pod["spec"]["nodeName"] = "synthetic-node"
    write(root / f"{name}-status.json", {
        "job": actual_job, "pods": {"apiVersion": "v1", "kind": "List", "items": [pod]}})
    (root / "synthetic-pod.log").write_text(log)
    run = {"namespace": "synthetic", "source_sha": commit, "source_hashes": hashes,
           "source_identity": {"commit": commit, "tree": tree, "clean": True, "sha256": hashes},
           "jobs": [{"job": name, "output": str(root / "case-0"), "manifest": str(root / "job-0.json")}]}
    write(root / "run.json", run)
    files = {"run": root / "run.json", "result": result_root / "result.json",
             "junit": result_root / "pytest.xml", "submitted_job": root / "job-0.json",
             "status": root / f"{name}-status.json", "pod_log": root / "synthetic-pod.log",
             "pytest_log": result_root / "pytest.log", "execution_log": result_root / "execution.log",
             "telemetry": result_root / "telemetry.json", "source_manifest": root / "source/source-hashes.json",
             "outer_result": root / "case-0/result.json"}
    pins = {"schema_version": 1, "run_root": str(root),
            "files": {key: {"path": str(path), "sha256": auditor.sha(path)} for key, path in files.items()}}
    return root, source, pins


def repin(pins, name):
    pins["files"][name]["sha256"] = auditor.sha(pins["files"][name]["path"])


def test_synthetic_archive_positive_is_not_cuda_evidence(tmp_path):
    _, source, pins = fixture_archive(tmp_path)
    result = auditor.audit(pins, source)
    assert result["denominator"] == {"positive": 10, "negative": 3, "total": 13}
    assert result["runtime"]["python"] == "synthetic"
    assert result["commercial"] == result["performance"] == "UNVERIFIED"
    assert result["old_proof_authority"] is False


@pytest.mark.parametrize("mutation,reason", [
    ("owner-apiVersion", "owner UID"),
    ("job-kind", "resource identity"),
    ("image-digest", "image digest"),
    ("exit-bool", "image digest"),
    ("restart-bool", "image digest"),
    ("succeeded-bool", "completed once"),
    ("backoff-bool", "completed once"),
    ("parallelism-bool", "completed once"),
    ("completion-bool", "completed once"),
    ("result-bool", "fixed13 result"),
])
def test_original_falseaccepts_coordinated_repin_reject(tmp_path, mutation, reason):
    """Original ten archive reds, with caller pins updated, not pin-mismatch tests."""
    _, source, pins = fixture_archive(tmp_path)
    assert auditor.audit(pins, source)["verdict"] == "PASS"
    status = auditor.load(pins["files"]["status"]["path"])
    state = status["pods"]["items"][0]["status"]["containerStatuses"][0]
    if mutation == "owner-apiVersion":
        status["pods"]["items"][0]["metadata"]["ownerReferences"][0]["apiVersion"] = "foreign/v9"
    elif mutation == "job-kind":
        status["job"]["kind"] = "Deployment"
    elif mutation == "image-digest":
        state["imageID"] = "foreign@sha256:not-a-digest"
    elif mutation == "exit-bool":
        state["state"]["terminated"]["exitCode"] = False
    elif mutation == "restart-bool":
        state["restartCount"] = False
    elif mutation == "succeeded-bool":
        status["job"]["status"]["succeeded"] = True
    elif mutation in ("backoff-bool", "parallelism-bool", "completion-bool"):
        key = {"backoff-bool": "backoffLimit", "parallelism-bool": "parallelism",
               "completion-bool": "completions"}[mutation]
        status["job"]["spec"][key] = False if key == "backoffLimit" else True
    else:
        for name in ("result", "outer_result"):
            result = auditor.load(pins["files"][name]["path"])
            result["returncode"] = False
            result["tests"]["failures"] = False
            write(Path(pins["files"][name]["path"]), result)
            repin(pins, name)
    write(Path(pins["files"]["status"]["path"]), status)
    repin(pins, "status")
    write(tmp_path / "mutated-pins.json", pins)
    with pytest.raises(ValueError, match=reason) as rejected:
        auditor.audit(pins, source)
    write(tmp_path / "rejection.json", {
        "mutation": mutation, "accepted": False, "reason": str(rejected.value),
        "auditor_sha256": auditor.sha(auditor.__file__),
        "fixture": "synthetic CPU control; not CUDA origin qualification"})


@pytest.mark.parametrize("location", ["sidecar", "evidence", "analytic", "cone"])
def test_boolean_schema_rejects_without_reinterpreting_coordinates(location):
    saved = sidecar(next(i for i in auditor.IDENTITIES if "[rotated_lp-float64]" in i))
    assert auditor.replay_sidecar(saved)["passed"] is True
    target = {"sidecar": saved, "evidence": saved["evidence"],
              "analytic": saved["analytic"]["evidence"],
              "cone": saved["cone_evidence"]["dual"]}[location]
    target["schema_version"] = True
    with pytest.raises(ValueError, match="schema"):
        auditor.replay_sidecar(saved)


@pytest.mark.parametrize("value,expected,accepted", [
    (0, 0, True), (1, 1, True), (13, 13, True),
    (False, 0, False), (True, 1, False), (0., 0, False), (1., 1, False),
    ("0", 0, False), (None, 0, False),
])
def test_integer_contract_has_no_json_aliases(value, expected, accepted):
    assert auditor.integer(value, expected) is accepted


@pytest.mark.parametrize("value,accepted", [
    ("synthetic@sha256:" + "0" * 64, True),
    ("docker-pullable://registry/repository@sha256:" + "a" * 64, True),
    ("foreign@sha256:not-a-digest", False),
    ("foreign@sha256:" + "a" * 63, False),
    ("foreign@sha256:" + "a" * 65, False),
    ("foreign@sha256:" + "A" * 64, False),
    ("foreign@sha256:" + "a" * 64 + "\n", False),
    ("foreign image@sha256:" + "a" * 64, False),
    ("@sha256:" + "a" * 64, False),
    (True, False),
])
def test_image_identity_requires_complete_digest(value, accepted):
    assert auditor.image_digest(value) is accepted


@pytest.fixture(scope="module")
def execution_archive(tmp_path_factory):
    """Shared read-only metadata input; each mutation receives deep copies."""
    root, _, pins = fixture_archive(tmp_path_factory.mktemp("execution-types"))
    return (
        root, auditor.load(pins["files"]["run"]["path"]),
        Path(pins["files"]["result"]["path"]), auditor.load(pins["files"]["result"]["path"]),
        auditor.load(pins["files"]["submitted_job"]["path"]),
        auditor.load(pins["files"]["status"]["path"]),
        Path(pins["files"]["pod_log"]["path"]).read_text(),
    )


@pytest.mark.parametrize("resource,field", [
    (resource, field) for resource in ("submitted", "job", "pod")
    for field in ("apiVersion", "kind")
])
def test_resource_identity_fields_reject_foreign_or_missing(execution_archive, resource, field):
    root, run, path, result, submitted, status, log = execution_archive
    assert auditor.archive_execution(root, run, path, result, submitted, status, log)
    for missing in (False, True):
        submitted_copy, status_copy = deepcopy(submitted), deepcopy(status)
        target = {"submitted": submitted_copy, "job": status_copy["job"],
                  "pod": status_copy["pods"]["items"][0]}[resource]
        if missing:
            del target[field]
        else:
            target[field] = "foreign/v9" if field == "apiVersion" else "Deployment"
        with pytest.raises(ValueError, match="resource identity"):
            auditor.archive_execution(root, run, path, result, submitted_copy, status_copy, log)


@pytest.mark.parametrize("field", [
    "backoffLimit", "parallelism", "completions", "succeeded", "failed", "active",
    "restartCount", "exitCode",
])
def test_archive_counters_reject_bool_and_float(execution_archive, field):
    root, run, path, result, submitted, status, log = execution_archive
    expected = 1 if field in ("parallelism", "completions", "succeeded") else 0
    for value in (bool(expected), float(expected)):
        submitted_copy, status_copy = deepcopy(submitted), deepcopy(status)
        state = status_copy["pods"]["items"][0]["status"]["containerStatuses"][0]
        if field in ("backoffLimit", "parallelism", "completions"):
            target = status_copy["job"]["spec"]
            submitted_copy["spec"][field] = value
        elif field in ("succeeded", "failed", "active"):
            target = status_copy["job"]["status"]
        elif field == "restartCount":
            target = state
        else:
            target = state["state"]["terminated"]
        target[field] = value
        with pytest.raises(ValueError, match="completed once|failure/restart"):
            auditor.archive_execution(root, run, path, result, submitted_copy, status_copy, log)


@pytest.mark.parametrize("field,value", [("workingDir", "/foreign"), ("envFrom", []), ("args", [])])
def test_unexpected_actual_execution_fields_reject(tmp_path, field, value):
    _, source, pins = fixture_archive(tmp_path)
    status = auditor.load(pins["files"]["status"]["path"])
    status["pods"]["items"][0]["spec"]["containers"][0][field] = value
    write(Path(pins["files"]["status"]["path"]), status)
    repin(pins, "status")
    with pytest.raises(ValueError, match="spec mismatch|args/envFrom"):
        auditor.audit(pins, source)


@pytest.mark.parametrize("mutation", ["args", "env-value", "env-membership", "envFrom"])
def test_coordinated_rehashed_job_pod_substitution_rejects(tmp_path, mutation):
    _, source, pins = fixture_archive(tmp_path)
    auditor.audit(pins, source)
    submitted = auditor.load(pins["files"]["submitted_job"]["path"])
    status = auditor.load(pins["files"]["status"]["path"])
    specs = [submitted["spec"]["template"]["spec"], status["job"]["spec"]["template"]["spec"],
             status["pods"]["items"][0]["spec"]]
    for spec in specs:
        container = spec["containers"][0]
        if mutation == "args":
            container["args"] = ["--not-in-command-contract"]
        elif mutation == "env-value":
            next(item for item in container["env"] if item["name"] == "OMP_NUM_THREADS")["value"] = "8"
        elif mutation == "env-membership":
            container["env"].append({"name": "UNEXPECTED", "value": "1"})
        else:
            container["envFrom"] = [{"configMapRef": {"name": "untrusted"}}]
    write(Path(pins["files"]["submitted_job"]["path"]), submitted)
    write(Path(pins["files"]["status"]["path"]), status)
    repin(pins, "submitted_job")
    repin(pins, "status")
    with pytest.raises(ValueError, match="args/envFrom|environment mismatch"):
        auditor.audit(pins, source)


@pytest.mark.parametrize("mutation,reason", [
    ("skip", "denominator"), ("duplicate", "identity duplicate"), ("outside", "outside pinned root"),
    ("source", "source byte"), ("uid", "owner UID"), ("telemetry", "telemetry count"),
    ("outer", "inner/outer"), ("pin", "byte pin"),
])
def test_archive_negative_controls(tmp_path, mutation, reason):
    root, source, pins = fixture_archive(tmp_path)
    if mutation in ("skip", "duplicate", "outside"):
        junit_path = Path(pins["files"]["junit"]["path"])
        doc = ET.fromstring(junit_path.read_bytes())
        if mutation == "skip":
            doc[0].set("skipped", "1")
        elif mutation == "duplicate":
            doc[0][1].set("name", doc[0][0].get("name"))
        else:
            prop = doc[0][0][0][0]
            prop.set("value", str(root / "run.json"))
        junit_path.write_bytes(ET.tostring(doc))
        repin(pins, "junit")
    elif mutation == "source":
        (root / "source/src/openopt/conic.py").write_text("foreign bytes\n")
    elif mutation == "uid":
        status = auditor.load(pins["files"]["status"]["path"])
        status["pods"]["items"][0]["metadata"]["ownerReferences"][0]["uid"] = "different-uid"
        write(Path(pins["files"]["status"]["path"]), status)
        repin(pins, "status")
    elif mutation == "telemetry":
        write(Path(pins["files"]["telemetry"]["path"]), [])
        repin(pins, "telemetry")
    elif mutation == "outer":
        write(Path(pins["files"]["outer_result"]["path"]), {})
        repin(pins, "outer_result")
    else:
        pins["files"]["run"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match=reason):
        auditor.audit(pins, source)


@pytest.mark.skipif(os.environ.get("OPENOPT_RUN_ROTATED_EVIDENCE_TESTS") != "1",
                    reason="historical evidence opt-in absent; not CUDA qualification")
def test_opt_in_historical_exact_thirteen_math():
    root_value = os.environ.get("OPENOPT_ROTATED_RESULT_ROOT")
    expected = os.environ.get("OPENOPT_ROTATED_JUNIT_SHA256")
    assert root_value and expected, "opt-in requires explicit result root and JUnit caller pin"
    root = Path(root_value)
    path = root / "pytest.xml"
    assert auditor.sha(path) == expected
    selected, _ = auditor.junit_sidecars(path, root)
    results = [auditor.replay_sidecar(auditor.load(path)) for path in selected.values()]
    assert len(results) == 13


@pytest.mark.skipif(os.environ.get("OPENOPT_RUN_ROTATED_ARCHIVE_TESTS") != "1",
                    reason="historical archive opt-in absent; not CUDA qualification")
def test_opt_in_historical_complete_archive():
    pin_path = os.environ.get("OPENOPT_ROTATED_PINS")
    expected = os.environ.get("OPENOPT_ROTATED_PINS_SHA256")
    assert pin_path and expected, "archive opt-in requires explicit caller pins"
    assert auditor.sha(pin_path) == expected
    report = auditor.audit(auditor.load(pin_path), auditor.ROOT)
    assert report["verdict"] == "PASS" and len(report["numeric_cases"]) == 13


def prepare_pins(argv):
    """Read-only historical pin inventory anchored by two explicit caller hashes."""
    import argparse
    parser = argparse.ArgumentParser(description=prepare_pins.__doc__)
    parser.add_argument("--prepare-pins", action="store_true", required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--run-sha256", required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    assert auditor.sha(args.run) == args.run_sha256, "caller run pin mismatch"
    assert auditor.sha(args.result) == args.result_sha256, "caller result pin mismatch"
    assert not args.artifact_dir.exists(), "fresh external inventory required"
    root = args.run.parent
    run = auditor.load(args.run)
    status_path = root / (run["jobs"][0]["job"] + "-status.json")
    pod = auditor.load(status_path)["pods"]["items"][0]["metadata"]["name"]
    result_root = args.result.parent
    files = {"run": args.run, "result": args.result, "junit": result_root / "pytest.xml",
             "submitted_job": root / "job-0.json", "status": status_path, "pod_log": root / (pod + ".log"),
             "pytest_log": result_root / "pytest.log", "execution_log": result_root / "execution.log",
             "telemetry": result_root / "telemetry.json", "source_manifest": root / "source/source-hashes.json",
             "outer_result": root / "case-0/result.json"}
    args.artifact_dir.mkdir(parents=True)
    full = write(args.artifact_dir / "pins.json", {
        "schema_version": 1, "run_root": str(root),
        "files": {name: {"path": str(path), "sha256": auditor.sha(path)} for name, path in files.items()}})
    numeric = write(args.artifact_dir / "numeric-pins.json", {
        "schema_version": 1, "result_root": str(result_root),
        "junit": {"path": str(files["junit"]), "sha256": auditor.sha(files["junit"])}})
    receipt = {"verdict": "INVENTORY_ONLY_NOT_ACCEPTANCE", "run_sha256": args.run_sha256,
               "result_sha256": args.result_sha256, "pins_sha256": auditor.sha(full),
               "numeric_pins_sha256": auditor.sha(numeric), "junit_sha256": auditor.sha(files["junit"])}
    write(args.artifact_dir / "inventory.json", receipt)
    (args.artifact_dir / "execution.log").write_text(json.dumps(receipt) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0


def scan_task(argv):
    """Credential needles remain in memory; only counts and file paths leave."""
    import argparse
    import importlib.util
    import re
    parser = argparse.ArgumentParser(description=scan_task.__doc__)
    parser.add_argument("--scan-only", action="store_true", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    assert not args.artifact_dir.exists() and args.artifact_dir.is_absolute()
    spec = importlib.util.spec_from_file_location(
        "safe_fixed13_scan", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(
        secret_hex=None, secrets_file=None, no_needle=False,
        needle_from=str(Path.home() / ".config/dsh-n600-token")))
    assert needles, "nonempty known-credential needle required"
    paths = [auditor.ROOT / "scripts/accept_rotated_cuda.py", Path(__file__).resolve(),
             auditor.ROOT / "docs/research/rotated-cuda-acceptance.md"]
    literal, _ = scanner.scan_literal([str(path) for path in paths], needles)
    pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                         r"""(?i:password|token|secret)\s*[\"']?\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']""")
    shape = sum(bool(pattern.search(path.read_text())) for path in paths)
    receipt = {"verdict": "PASS" if literal == shape == 0 else "FAIL",
               "literal_hits": literal, "shape_hits": shape, "files": [str(path) for path in paths]}
    args.artifact_dir.mkdir(parents=True)
    write(args.artifact_dir / "scan.json", receipt)
    (args.artifact_dir / "execution.log").write_text(json.dumps(receipt) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if literal == shape == 0 else 1


if __name__ == "__main__":
    raise SystemExit(scan_task(sys.argv[1:]) if "--scan-only" in sys.argv else prepare_pins(sys.argv[1:]))
