"""Portable independent fixed13 controls; historical CLI is explicit opt-in.

Synthetic CUDA-shaped records below are unexecuted coordinator fixtures, not
GPU evidence. No producer, optimizer, Torch, or NumPy is imported or executed.
"""

from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction as F
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts/accept_rotated_cuda.py"
GATE_PIN = "b249e9a3f13e422c5d992c78411aaeb070735750b73900eee323dd8c5b6809c0"
PRODUCER = "tests/test_rotated_conic_cuda.py"
PROJECTIONS = {
    "interior-asymmetric-float64": [1., 1e20, 0.],
    "interior-overflow-float64": [1.3e308, 1.3e308, 0.],
    "near-polar-float64": [-1e20, 0., 1e8],
    "small-head-float64": [-1e300, 0., 1e210],
    "small-head-float32": [-1e30, 0., 1e17],
    "tail-overflow-float64": [0., 0., 1.3e308, 1.3e308],
}
NEGATIVES = {
    "wrong-equality-dual": ["stationarity"],
    "planted-slack": ["primal_equation", "original_slack_cone"],
    "wrong-dual-cone": ["dual_cone"],
}
IDS = {
    **{f"tests.test_rotated_conic_cuda::test_cuda_rotated_original_model[{case}]":
       ("original_model", case) for case in (
           "rotated_lp-float32", "rotated_lp-float64",
           "rotated_qp-float32", "rotated_qp-float64")},
    **{f"tests.test_rotated_conic_cuda::test_cuda_rotated_projection_extremes[{case}]":
       ("projection", case) for case in PROJECTIONS},
    **{f"tests.test_rotated_conic_cuda::test_cuda_rotated_original_rejections[{case}]":
       ("original_rejection", case) for case in NEGATIVES},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="ascii")
    return path


def git(root, *args):
    process = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    assert process.returncode == 0, process.stderr
    return process.stdout.decode().strip()


@pytest.fixture(scope="module")
def gate():
    expected = os.environ.get("OPENOPT_FIXED13_GATE_SHA256", GATE_PIN)
    assert sha(GATE) == expected, "caller-reviewed gate bytes changed"
    spec = importlib.util.spec_from_file_location("fixed13_independent_target", GATE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PRODUCER == PRODUCER and set(module.IDENTITIES) == set(IDS)
    return module


def cast(values, dtype):
    return [struct.unpack("!f", struct.pack("!f", v))[0] if dtype.endswith("32") else float(v)
            for v in values]


def oracle(values):
    """1100-digit SOC isometry, independent of target's RSOC elimination."""
    with localcontext() as context:
        context.prec = 1100
        a, b, *w = [Decimal.from_float(v) for v in values]
        if a >= 0 and b >= 0 and 2 * a * b >= sum(v * v for v in w):
            return values[:]
        root2 = Decimal(2).sqrt()
        head, tail0 = (a + b) / root2, (a - b) / root2
        radius = (tail0 * tail0 + sum(v * v for v in w)).sqrt()
        if head <= -radius:
            return [0.] * len(values)
        alpha = (head + radius) / (2 * radius)
        t = alpha * radius
        return [float((t + alpha * tail0) / root2),
                float((t - alpha * tail0) / root2), *[float(alpha * v) for v in w]]


def original_model(qp):
    d = 1. if qp else 0.
    return {"P": [[d, 0., 0.], [0., d, 0.], [0., 0., 0.]],
            "q": [0., 0., 0.] if qp else [1., 1., 0.],
            "A": [[0., 0., 1.], [-1., -0., -0.], [-0., -1., -0.], [-0., -0., -1.]],
            "b": [2., 0., 0., 0.],
            "cones": [{"kind": "zero", "dimension": 1}, {"kind": "rsoc", "dimension": 3}]}


def analytic(qp):
    with localcontext() as context:
        context.prec = 1100
        r = float(Decimal(2).sqrt())
    return {"x": [r, r, 2.], "s": [0., r, r, 2.],
            "z": [-2., r, r, -2.] if qp else [-r, 1., 1., -r]}


def cone(values, blocks, tolerance, relative, dual=False):
    """Exact original RSOC product test with the required-small-head tolerance."""
    out, offset = {}, 0
    for i, block in enumerate(blocks):
        v = values[offset:offset + block["dimension"]]
        offset += block["dimension"]
        if block["kind"] == "zero":
            passed = dual or all(abs(x) <= tolerance for x in v)
        else:
            a, b, *w = v
            small, large = min(a, b), max(a, b)
            radial = sum(x * x for x in w)
            signs = all(x >= -tolerance - relative * abs(x) for x in (a, b))
            if large > 0:
                required = radial / (2 * large)
                passed = signs and required <= small + tolerance + relative * max(abs(small), required)
            else:
                bound = (small + tolerance) / (1 - relative)
                passed = signs and bound >= 0 and radial <= bound * bound
        out[f"block_{i}"] = bool(passed)
    return out


def evidence(model, vectors, atol, rtol):
    """Independent exact residuals; legacy metric thresholds are retained separately."""
    P, q, A, b = (model[k] for k in ("P", "q", "A", "b"))
    x, s, z = ([F.from_float(v) for v in vectors[k]] for k in ("x", "s", "z"))
    dot = lambda a, v: sum((F.from_float(ai) * vi for ai, vi in zip(a, v)), F(0))
    Ax, Px = [dot(row, x) for row in A], [dot(row, x) for row in P]
    ATz = [dot([row[i] for row in A], z) for i in range(len(x))]
    t, r = F.from_float(atol), F.from_float(float(rtol))
    eq = [a + si - F.from_float(bi) for a, si, bi in zip(Ax, s, b)]
    st = [p + F.from_float(qi) + a for p, qi, a in zip(Px, q, ATz)]
    cones = {
        "supplied_slack": cone(s, model["cones"], t, r),
        "original_slack": cone([F.from_float(bi) - a for bi, a in zip(b, Ax)], model["cones"], t, r),
        "dual": cone(z, model["cones"], t, r, True),
    }
    checks = {
        "primal_equation": all(abs(v) <= t + r * max(abs(a), abs(si), abs(F.from_float(bi)))
                               for v, a, si, bi in zip(eq, Ax, s, b)),
        "primal_cone": all(cones["supplied_slack"].values()),
        "original_slack_cone": all(cones["original_slack"].values()),
        "dual_cone": all(cones["dual"].values()),
        "stationarity": all(abs(v) <= t + r * max(abs(p), abs(F.from_float(qi)), abs(a))
                            for v, p, qi, a in zip(st, Px, q, ATz)),
    }
    metrics = {"primal_equation": float(max(map(abs, eq))), "stationarity": float(max(map(abs, st))),
               "primal_cone": 0., "original_slack_cone": 0., "dual_cone": 0.}
    thresholds = {"primal_equation_max": float(max(t + r * max(abs(a), abs(si), abs(F.from_float(bi)))
                                                   for a, si, bi in zip(Ax, s, b))),
                  "stationarity_max": float(max(t + r * max(abs(p), abs(F.from_float(qi)), abs(a))
                                               for p, qi, a in zip(Px, q, ATz))),
                  "spectral": 64 * 2. ** -52 * len(x) * max(P[i][i] for i in range(len(x))),
                  "range_max": float(max(t + r * max(abs(F.from_float(qi)), abs(a))
                                        for qi, a in zip(q, ATz)))}
    legacy_Ax = [sum(a * v for a, v in zip(row, vectors["x"])) for row in A]
    legacy_Px = [sum(a * v for a, v in zip(row, vectors["x"])) for row in P]
    legacy_ATz = [sum(row[i] * v for row, v in zip(A, vectors["z"])) for i in range(len(x))]
    thresholds.update(
        primal_equation_max=max(atol + rtol * max(abs(a), abs(si), abs(bi))
                                for a, si, bi in zip(legacy_Ax, vectors["s"], b)),
        stationarity_max=max(atol + rtol * max(abs(p), abs(qi), abs(a))
                             for p, qi, a in zip(legacy_Px, q, legacy_ATz)),
        range_max=max(atol + rtol * max(abs(qi), abs(a)) for qi, a in zip(q, legacy_ATz)),
    )
    offset = 0
    for i, block in enumerate(model["cones"]):
        size = block["dimension"]
        products = [a * v for a, v in zip(s[offset:offset + size], z[offset:offset + size])]
        name = f"complementarity_block_{i}"
        checks[name] = abs(sum(products)) <= t + r * sum(map(abs, products))
        metrics[name] = float(abs(sum(products)))
        # The archived threshold algorithm used binary64 products, not rational rounding.
        legacy = [a * v for a, v in zip(vectors["s"][offset:offset + size],
                                       vectors["z"][offset:offset + size])]
        thresholds[name] = atol + rtol * sum(map(abs, legacy))
        offset += size
    h = [F.from_float(qi) + a for qi, a in zip(q, ATz)]
    in_range = all(P[i][i] or abs(h[i]) <= t + r * max(abs(F.from_float(q[i])), abs(ATz[i]))
                   for i in range(len(x)))
    available = checks["dual_cone"] and in_range
    primal = sum(xi * pi for xi, pi in zip(x, Px)) / 2 + dot(q, x)
    dual = -sum(h[i] ** 2 / 2 for i in range(len(x)) if P[i][i]) - dot(b, z)
    checks["dual_objective_available"] = bool(available)
    checks["duality_gap"] = bool(available and abs(primal - dual) <= t + r * max(abs(primal), abs(dual)))
    metrics.update(primal_objective=float(primal), dual_objective=float(dual) if available else None,
                   duality_gap=float(primal - dual) if available else None,
                   range_residual=0. if in_range else None)
    if available:
        thresholds["duality_gap"] = atol + rtol * max(abs(float(primal)), abs(float(dual)))
    result = {"schema_version": 1, "kind": "conic_candidate_optimality",
              "passed": all(checks.values()), "checks": checks, "metrics": metrics,
              "thresholds": thresholds, "unavailable": {} if available else
              {"dual_objective": "independent synthetic dual unavailable"},
              "qualification": "approximate numerical evidence; not an exact proof"}
    return result, cones


def record(identity, sources):
    kind, case = IDS[identity]
    dtype = "torch.float64" if kind == "original_rejection" else "torch." + case.rsplit("-", 1)[1]
    atol = 1e-12 if kind == "projection" else 3e-5 if dtype.endswith("32") else 1e-8
    rtol = (3e-6 if dtype.endswith("32") else 3e-14) if kind == "projection" else 0
    qp = case.startswith("rotated_qp")
    model, vectors = original_model(qp), analytic(qp)
    baseline = deepcopy(vectors)
    if kind == "projection":
        value = cast(PROJECTIONS[case], dtype)
        projected = cast(oracle(value), dtype)
        vectors = {"x": [0.], "s": projected, "z": [0.] * len(projected)}
        model = {"P": [[0.]], "q": [0.], "A": [[0.]] * len(projected), "b": projected,
                 "cones": [{"kind": "rsoc", "dimension": len(projected)}]}
    elif kind == "original_rejection":
        if case == "wrong-equality-dual":
            vectors["z"][0] *= -1
        elif case == "planted-slack":
            vectors["x"] = [0., 0., 2.]
        else:
            vectors["z"][1:] = [-v for v in vectors["z"][1:]]
    vectors = {key: cast(v, dtype) for key, v in vectors.items()}
    measured, cones = evidence(model, vectors, atol, rtol)
    n, m = len(model["q"]), len(model["b"])
    shapes = {"x": [n], "s": [m], "z": [m], "original.P": [n, n],
              "original.q": [n], "original.A": [m, n], "original.b": [m]}
    shapes.update({"projection.input": [m], "projection.output": [m]} if kind == "projection"
                  else {"prepared.P": [n, n], "prepared.q": [n], "prepared.A": [m, n],
                        "prepared.b": [m], "prepared.factor": [n, n]})
    metadata = {key: {"shape": v, "device": "cuda:0", "layout": "torch.strided",
                      "dtype": "torch.float64" if key.startswith("original.") else dtype}
                for key, v in shapes.items()}
    runtime = {"python": "synthetic-unexecuted", "implementation": "synthetic",
               "executable": "/synthetic", "numpy": "synthetic", "torch": "synthetic",
               "compiled_cuda": "synthetic", "cuda_available": True, "cuda_execution": True,
               "gpu_name": "NVIDIA RTX PRO 4500 Blackwell", "capability": [12, 0],
               "environment": {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                               "MKL_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": None}}
    saved = {"schema_version": 1, "family": "rotated_conic_cuda", "verdict": "PASS",
             "checked_at": "2026-10-10T12:00:05+00:00", "test_identity": identity,
             "case_id": case, "case_kind": kind, "device": "cuda:0", "dtype": dtype,
             "config": {"enabled": True, "device": "cuda:0", "dtype": dtype, "rho": 1.,
                        "sigma": 1e-6, "max_iterations": 10000, "check_interval": 25,
                        "eps_abs": atol, "eps_rel": rtol},
             "runtime": runtime, "source_sha256": sources,
             "loaded_sources": {name: {"source": "src/" + name.replace(".", "/") + ".py",
                                      "sha256": sources["src/" + name.replace(".", "/") + ".py"]}
                                for name in ("openopt.conic", "openopt.native_conic", "openopt.certificates")},
             "tensor_metadata": metadata, "problem": model, **vectors,
             "status": {"projection": "projected", "original_model": "solved",
                        "original_rejection": "candidate_rejected"}[kind],
             "iterations": 0 if kind == "projection" else 25,
             "evidence": measured,
             "cone_evidence": {name: {"schema_version": 1, "kind": "dual_cone_feasibility"
                                    if name == "dual" else "cone_feasibility",
                                    "passed": all(checks.values()), "checks": checks,
                                    "metrics": {k: 0. for k in checks}, "thresholds": {k: atol for k in checks},
                                    "unavailable": {}, "qualification": measured["qualification"]}
                               for name, checks in cones.items()},
             "objective": measured["metrics"]["primal_objective"], "certified": False,
             "qualified": False, "commercial": "UNVERIFIED", "performance": "UNVERIFIED"}
    if kind == "projection":
        saved["projection"] = {"input": value, "output": vectors["s"], "input_unchanged": True,
                               "oracle": {"algorithm": "bounded_decimal_analytic_rsoc_v1",
                                          "precision": 800, "expected": oracle(value),
                                          "atol": 0., "rtol": rtol, "passed": True}}
    elif kind == "original_model":
        av = analytic(qp)
        saved["analytic"] = {"vectors": av, "evidence": evidence(model, av, 1e-12, 0)[0]}
    else:
        saved["negative"] = {"mutation": case, "expected_failed_checks": NEGATIVES[case],
                             "baseline": {**baseline, "evidence": evidence(model, baseline, atol, rtol)[0],
                                          "status": "solved",
                                          "tensor_metadata": {k: metadata[k] for k in baseline}}}
    return saved


def environment(root):
    return [{"name": "PYTHONPATH", "value": str(root / "source/src")},
            {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            *[{"name": name, "value": "4"} for name in
              ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")],
            {"name": "OPENOPT_NODE", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
            {"name": "TMPDIR", "value": str(root / "case-0")}]


@pytest.fixture(scope="module")
def archive(tmp_path_factory, gate):
    base = tmp_path_factory.mktemp("independent-synthetic")
    root, source = base / "run", base / "source"
    names = ["src/openopt/conic.py", "src/openopt/native_conic.py",
             "src/openopt/certificates.py", PRODUCER, "scripts/gpu_release_check.py"]
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('"""Unexecuted independent synthetic source."""\n', encoding="ascii")
    git(source, "init")
    git(source, "add", *names)
    git(source, "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid",
        "commit", "-m", "independent transport fixture")
    hashes = {name: sha(source / name) for name in names}
    for name in names:
        path = root / "source" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    put(root / "source/source-hashes.json", hashes)
    scope = {k: v for k, v in hashes.items() if k.startswith("src/") or k == PRODUCER}
    result_root = root / "case-0/cuda-check-synthetic"
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite", tests="13", failures="0", errors="0", skipped="0")
    for index, identity in enumerate(IDS):
        path = put(result_root / "pytest-tmp" / f"case-{index}" / "rotated-conic.json",
                   record(identity, scope))
        classname, name = identity.split("::")
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        props = ET.SubElement(case, "properties")
        for key, value in (("rotated_conic_acceptance_path", str(path)),
                           ("rotated_conic_acceptance_sha256", sha(path))):
            ET.SubElement(props, "property", name=key, value=value)
    (result_root / "pytest.xml").write_bytes(ET.tostring(suites))
    (result_root / "pytest.log").write_text("13 passed\nreturncode=0\n")
    log = f"OPENOPT_RELEASE_CUDA verdict=PASS tests=13 evidence={result_root}\n"
    (result_root / "execution.log").write_text(log)
    result = {"verdict": "PASS", "returncode": 0, "targets": [PRODUCER], "selection": "cuda_rotated",
              "tests": {"tests": 13, "failures": 0, "errors": 0, "skipped": 0},
              "source_sha256": hashes, "started_at": "2026-10-10T12:00:00+00:00",
              "finished_at": "2026-10-10T12:00:10+00:00", "telemetry_samples": 1,
              "hardware": {"node": "synthetic-node", "gpu": "NVIDIA RTX PRO 4500 Blackwell",
                           "torch": "synthetic", "cuda": "synthetic"}}
    files = {"run": root / "run.json", "result": result_root / "result.json",
             "outer_result": root / "case-0/result.json", "junit": result_root / "pytest.xml",
             "pytest_log": result_root / "pytest.log", "execution_log": result_root / "execution.log",
             "telemetry": result_root / "telemetry.json", "source_manifest": root / "source/source-hashes.json",
             "submitted_job": root / "job-0.json", "status": root / "synthetic-job-status.json",
             "pod_log": root / "synthetic-pod.log"}
    put(files["result"], result)
    put(files["outer_result"], result)
    put(files["telemetry"], [{"at": "2026-10-10T12:00:05+00:00", "monotonic_s": 5.,
                            "rc": 0, "error": None, "cpu": {"usage_usec": 1},
                            "gpu": "GPU-00000000-0000-0000-0000-000000000000, 0, 0, 0"}])
    container = {"name": "benchmark", "image": "synthetic-image", "imagePullPolicy": "Never",
                 "command": ["python3", str(root / "source/scripts/gpu_release_check.py"),
                             "--artifact-dir", str(root / "case-0"), "--target", PRODUCER,
                             "--expression", "cuda_rotated"],
                 "env": environment(root), "securityContext": {"allowPrivilegeEscalation": False},
                 "resources": {k: {"cpu": "4", "memory": "12Gi", "nvidia.com/gpu": 1}
                               for k in ("requests", "limits")}}
    spec = {"containers": [container], "restartPolicy": "Never", "runtimeClassName": "nvidia",
            "nodeSelector": {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-4500-Blackwell"}}
    submitted = {"apiVersion": "batch/v1", "kind": "Job",
                 "metadata": {"name": "synthetic-job", "namespace": "synthetic"},
                 "spec": {"template": {"spec": spec}, "backoffLimit": 0}}
    put(files["submitted_job"], submitted)
    job = deepcopy(submitted)
    job["metadata"]["uid"] = "synthetic-job-uid"
    job["spec"].update(parallelism=1, completions=1)
    job["status"] = {"succeeded": 1, "startTime": "2026-10-10T12:00:00Z",
                     "completionTime": "2026-10-10T12:00:11Z",
                     "conditions": [{"type": "Complete", "status": "True"}]}
    pod = {"kind": "Pod", "apiVersion": "v1",
           "metadata": {"name": "synthetic-pod", "namespace": "synthetic", "uid": "synthetic-pod-uid",
                        "ownerReferences": [{"kind": "Job", "apiVersion": "batch/v1",
                                             "uid": "synthetic-job-uid", "name": "synthetic-job",
                                             "controller": True}]},
           "spec": {**deepcopy(spec), "nodeName": "synthetic-node"},
           "status": {"phase": "Succeeded", "containerStatuses": [{
               "name": "benchmark", "restartCount": 0, "lastState": {},
               "imageID": "synthetic@sha256:" + "0" * 64,
               "state": {"terminated": {"exitCode": 0, "reason": "Completed",
                                       "startedAt": "2026-10-10T12:00:00Z",
                                       "finishedAt": "2026-10-10T12:00:10Z"}}}]}}
    put(files["status"], {"job": job, "pods": {"items": [pod]}})
    files["pod_log"].write_text(log)
    put(files["run"], {"namespace": "synthetic", "source_sha": git(source, "rev-parse", "HEAD"),
                      "source_hashes": hashes,
                      "source_identity": {"commit": git(source, "rev-parse", "HEAD"),
                                          "tree": git(source, "rev-parse", "HEAD^{tree}"),
                                          "clean": True, "sha256": hashes},
                      "jobs": [{"job": "synthetic-job", "output": str(root / "case-0"),
                                "manifest": str(root / "job-0.json")}]})
    pins = {"schema_version": 1, "run_root": str(root),
            "files": {k: {"path": str(p), "sha256": sha(p)} for k, p in files.items()}}
    checked = gate.audit(pins, source)
    assert checked["denominator"] == {"positive": 10, "negative": 3, "total": 13}
    return root, source, pins


def observation(path, call):
    try:
        result = call()
        outcome = {"accepted": True, "verdict": result.get("verdict", "RETURNED"), "result": result}
    except (ValueError, KeyError, TypeError, OverflowError) as error:
        outcome = {"accepted": False, "error_type": type(error).__name__, "reason": str(error)}
    put(path, outcome)
    return outcome


def copy_archive(archive, tmp_path):
    """Relocate only controlled synthetic files; never touch historical inputs."""
    root, source, old = archive
    target = tmp_path / "mutated-run"
    shutil.copytree(root, target, copy_function=shutil.copyfile)
    for path in target.rglob("*"):
        if path.is_file() and path.suffix in (".json", ".xml", ".log"):
            path.write_text(path.read_text().replace(str(root), str(target)))
    junit = target / "case-0/cuda-check-synthetic/pytest.xml"
    doc = ET.fromstring(junit.read_bytes())
    for case in doc[0]:
        props = {item.get("name"): item for item in case[0]}
        props["rotated_conic_acceptance_sha256"].set(
            "value", sha(props["rotated_conic_acceptance_path"].get("value")))
    junit.write_bytes(ET.tostring(doc))
    pins = deepcopy(old)
    pins["run_root"] = str(target)
    for binding in pins["files"].values():
        binding["path"] = binding["path"].replace(str(root), str(target))
        binding["sha256"] = sha(binding["path"])
    return target, source, pins


def repin(pins, *names):
    for name in names:
        pins["files"][name]["sha256"] = sha(pins["files"][name]["path"])


def rehash_sidecar(junit, path):
    doc = ET.fromstring(Path(junit).read_bytes())
    for case in doc[0]:
        props = {item.get("name"): item for item in case[0]}
        if Path(props["rotated_conic_acceptance_path"].get("value")) == path:
            props["rotated_conic_acceptance_sha256"].set("value", sha(path))
    Path(junit).write_bytes(ET.tostring(doc))


@pytest.mark.parametrize("identity", IDS)
def test_original_thirteen_and_baselines_independent(identity, archive, gate):
    root, _, _ = archive
    selected, _ = gate.junit_sidecars(root / "case-0/cuda-check-synthetic/pytest.xml",
                                    root / "case-0/cuda-check-synthetic")
    saved = gate.load(selected[identity])
    independent, _ = evidence(saved["problem"], saved, saved["config"]["eps_abs"], saved["config"]["eps_rel"])
    assert saved["evidence"]["checks"] == independent["checks"]
    checked = gate.replay_sidecar(saved)
    assert checked["passed"] is (IDS[identity][0] != "original_rejection")
    if IDS[identity][0] == "original_rejection":
        baseline, _ = evidence(saved["problem"], saved["negative"]["baseline"], 1e-8, 0)
        assert baseline["passed"] and checked["baseline_passed"]
    if IDS[identity][0] == "projection":
        expected = oracle(saved["projection"]["input"])
        assert all(abs(F.from_float(a) - F.from_float(b)) <= F.from_float(saved["config"]["eps_rel"])
                   * abs(F.from_float(b)) for a, b in zip(saved["s"], expected))


@pytest.mark.parametrize("mutation", [
    "envFrom", "args", "coordinated-env", "workingDir", "mount", "placement", "namespace",
    "owner-uid", "owner-apiVersion", "job-kind", "image-digest", "exit-bool", "restart-bool",
    "succeeded-bool", "backoff-bool", "parallelism-bool", "completion-bool",
])
def test_execution_rehash_rejects(mutation, archive, tmp_path, gate):
    root, source, pins = copy_archive(archive, tmp_path)
    assert gate.audit(pins, source)["verdict"] == "PASS"
    status = gate.load(pins["files"]["status"]["path"])
    submitted = gate.load(pins["files"]["submitted_job"]["path"])
    pod = status["pods"]["items"][0]
    container = pod["spec"]["containers"][0]
    state = pod["status"]["containerStatuses"][0]
    if mutation in ("envFrom", "args"):
        container[mutation] = [{"configMapRef": {"name": "foreign"}}] if mutation == "envFrom" else ["foreign"]
    elif mutation == "coordinated-env":
        for spec in (submitted["spec"]["template"]["spec"], status["job"]["spec"]["template"]["spec"], pod["spec"]):
            spec["containers"][0]["env"][3]["value"] = "8"
    elif mutation == "workingDir":
        container["workingDir"] = "/foreign"
    elif mutation == "mount":
        container["volumeMounts"] = [{"name": "foreign", "mountPath": "/source"}]
        pod["spec"]["volumes"] = [{"name": "foreign", "persistentVolumeClaim": {"claimName": "foreign"}}]
    elif mutation == "placement":
        pod["spec"]["nodeSelector"] = {"nvidia.com/gpu.product": "foreign"}
    elif mutation == "namespace":
        pod["metadata"]["namespace"] = "foreign"
    elif mutation == "owner-uid":
        pod["metadata"]["ownerReferences"][0]["uid"] = "foreign"
    elif mutation == "owner-apiVersion":
        pod["metadata"]["ownerReferences"][0]["apiVersion"] = "foreign/v9"
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
    else:
        key = {"backoff-bool": "backoffLimit", "parallelism-bool": "parallelism",
               "completion-bool": "completions"}[mutation]
        status["job"]["spec"][key] = False if key == "backoffLimit" else True
    put(pins["files"]["status"]["path"], status)
    put(pins["files"]["submitted_job"]["path"], submitted)
    repin(pins, "status", "submitted_job")
    put(tmp_path / "mutated-pins.json", pins)
    outcome = observation(tmp_path / "observation.json", lambda: gate.audit(pins, source))
    assert not outcome["accepted"], f"falseaccept={mutation}; receipt={tmp_path / 'observation.json'}"


@pytest.mark.parametrize("mutation", [
    "model", "property", "negative-baseline", "dtype", "placement", "threshold",
    "schema-bool", "check-bool", "projection-input", "tiny-head",
])
def test_original_coordinate_mutations_reject(mutation, archive, tmp_path, gate):
    root, _, _ = archive
    selected, _ = gate.junit_sidecars(root / "case-0/cuda-check-synthetic/pytest.xml",
                                    root / "case-0/cuda-check-synthetic")
    case = "small-head-float64" if mutation in ("tiny-head", "projection-input") else (
        "wrong-equality-dual" if mutation == "negative-baseline" else "rotated_lp-float64")
    saved = gate.load(next(path for identity, path in selected.items() if f"[{case}]" in identity))
    if mutation == "model":
        saved["problem"]["b"][0] = 3.
    elif mutation == "property":
        saved["status"] = "unknown"
    elif mutation == "negative-baseline":
        saved["negative"]["baseline"]["x"] = [0., 0., 2.]
    elif mutation == "dtype":
        saved["dtype"] = "torch.float32"
    elif mutation == "placement":
        saved["tensor_metadata"]["s"]["device"] = "cpu"
    elif mutation == "threshold":
        saved["evidence"]["thresholds"]["duality_gap"] *= 2
    elif mutation == "schema-bool":
        saved["schema_version"] = True
    elif mutation == "check-bool":
        saved["evidence"]["checks"]["stationarity"] = 1
    elif mutation == "projection-input":
        saved["projection"]["input"][2] *= 2
    else:
        saved["s"][0] *= 2
        saved["projection"]["output"] = saved["s"][:]
        saved["problem"]["b"] = saved["s"][:]
    path = put(tmp_path / "mutated-sidecar.json", saved)
    outcome = observation(tmp_path / "observation.json", lambda: gate.replay_sidecar(gate.load(path)))
    assert not outcome["accepted"], f"falseaccept={mutation}; receipt={tmp_path / 'observation.json'}"


def test_portable_complete_archive_no_solver_import(archive, gate):
    import ast
    tree = ast.parse(GATE.read_text())
    imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imports += [name.name for node in ast.walk(tree) if isinstance(node, ast.Import) for name in node.names]
    assert not any(name and name.split(".")[0] in ("torch", "numpy", "openopt", "tests") for name in imports)
    _, source, pins = archive
    result = gate.audit(pins, source)
    assert result["denominator"]["total"] == 13 and result["runtime"]["python"] == "synthetic-unexecuted"
    assert result["commercial"] == result["performance"] == "UNVERIFIED"


@pytest.mark.parametrize("mutation", [
    "drop-negative", "duplicate-identity", "extra-sidecar", "missing-property",
    "reused-property", "cpu-complete", "model-rehash", "property-rehash",
    "source-rehash", "source-current", "loaded-source", "result-bool",
])
def test_complete_archive_coordinated_bindings_reject(mutation, archive, tmp_path, gate):
    root, source, pins = copy_archive(archive, tmp_path)
    result_root = Path(pins["files"]["result"]["path"]).parent
    junit = Path(pins["files"]["junit"]["path"])
    doc = ET.fromstring(junit.read_bytes())
    if mutation == "drop-negative":
        doc[0].remove(doc[0][-1])
        doc[0].set("tests", "12")
    elif mutation == "duplicate-identity":
        doc[0][1].set("name", doc[0][0].get("name"))
    elif mutation == "extra-sidecar":
        put(result_root / "pytest-tmp/undeclared/rotated-conic.json", {})
    elif mutation == "missing-property":
        doc[0][0][0].remove(doc[0][0][0][0])
    elif mutation == "reused-property":
        for a, b in zip(doc[0][1][0], doc[0][0][0]):
            a.set("value", b.get("value"))
    elif mutation == "result-bool":
        for name in ("result", "outer_result"):
            result = gate.load(pins["files"][name]["path"])
            result["returncode"] = False
            result["tests"]["failures"] = False
            put(pins["files"][name]["path"], result)
        repin(pins, "result", "outer_result")
    elif mutation == "source-current":
        current = tmp_path / "mutated-current"
        shutil.copytree(source, current, copy_function=shutil.copyfile)
        changed = current / "src/openopt/conic.py"
        changed.write_bytes(changed.read_bytes() + b"\n# scoped source mutation\n")
        source = current
    elif mutation == "source-rehash":
        name = "src/openopt/conic.py"
        changed = root / "source" / name
        changed.write_bytes(changed.read_bytes() + b"\n# coordinated snapshot mutation\n")
        digest = sha(changed)
        run = gate.load(pins["files"]["run"]["path"])
        run["source_hashes"][name] = run["source_identity"]["sha256"][name] = digest
        put(pins["files"]["run"]["path"], run)
        manifest = gate.load(pins["files"]["source_manifest"]["path"])
        manifest[name] = digest
        put(pins["files"]["source_manifest"]["path"], manifest)
        for case in doc[0]:
            props = {p.get("name"): p for p in case[0]}
            path = Path(props["rotated_conic_acceptance_path"].get("value"))
            saved = gate.load(path)
            saved["source_sha256"][name] = digest
            saved["loaded_sources"]["openopt.conic"]["sha256"] = digest
            put(path, saved)
            props["rotated_conic_acceptance_sha256"].set("value", sha(path))
        for key in ("result", "outer_result"):
            result = gate.load(pins["files"][key]["path"])
            result["source_sha256"][name] = digest
            put(pins["files"][key]["path"], result)
        repin(pins, "run", "source_manifest", "result", "outer_result")
    else:
        cases = list(doc[0]) if mutation == "cpu-complete" else [doc[0][1]]
        for case in cases:
            props = {p.get("name"): p for p in case[0]}
            path = Path(props["rotated_conic_acceptance_path"].get("value"))
            saved = gate.load(path)
            if mutation == "cpu-complete":
                saved["device"] = saved["config"]["device"] = "cpu"
                saved["runtime"].update(cuda_execution=False, gpu_name=None, capability=None)
                for value in saved["tensor_metadata"].values():
                    value["device"] = "cpu"
            elif mutation == "model-rehash":
                saved["problem"]["b"][0] = 3.
                saved["x"][2] = saved["s"][3] = 3.
            elif mutation == "property-rehash":
                saved["status"] = "unknown"
            else:
                saved["loaded_sources"].pop("openopt.certificates")
            put(path, saved)
            props["rotated_conic_acceptance_sha256"].set("value", sha(path))
    junit.write_bytes(ET.tostring(doc))
    repin(pins, "junit")
    put(tmp_path / "mutated-pins.json", pins)
    outcome = observation(tmp_path / "observation.json", lambda: gate.audit(pins, source))
    assert not outcome["accepted"], f"falseaccept={mutation}; receipt={tmp_path / 'observation.json'}"


@pytest.mark.skipif(os.environ.get("OPENOPT_FIXED13_INDEPENDENT_HISTORY") != "1",
                    reason="portable default; historical CPU archive replay is explicit opt-in")
def test_actual_historical_cli_seal_accept_and_wrong_pin(tmp_path, gate, request):
    pins = Path(os.environ["OPENOPT_FIXED13_INDEPENDENT_PINS"])
    pin = os.environ["OPENOPT_FIXED13_INDEPENDENT_PINS_SHA256"]
    assert sha(pins) == pin
    assert not git(ROOT, "status", "--porcelain"), "historical CLI requires frozen clean source"
    outcomes = {}
    for mode in ("seal", "accept", "wrong-pin"):
        output = tmp_path / mode
        command = [sys.executable, "-B", str(GATE), "seal" if mode == "wrong-pin" else mode,
                   "--pins", str(pins), "--pins-sha256", "0" * 64 if mode == "wrong-pin" else pin,
                   "--artifact-dir", str(output)]
        if mode == "accept":
            bundle = tmp_path / "seal/bundle.json"
            command += ["--bundle", str(bundle), "--bundle-sha256", sha(bundle)]
        process = subprocess.run(command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1",
                                                        "PYTHONDONTWRITEBYTECODE": "1"},
                                 capture_output=True, text=True, timeout=120, check=False)
        (tmp_path / f"{mode}.log").write_text(process.stdout + process.stderr)
        outcomes[mode] = {"returncode": process.returncode, "acceptance": gate.load(output / "acceptance.json")}
    put(tmp_path / "historical-cli.json", outcomes)
    request.node.user_properties.append(("historical_cli_receipt", str(tmp_path / "historical-cli.json")))
    assert outcomes["seal"]["returncode"] == outcomes["accept"]["returncode"] == 0
    assert outcomes["wrong-pin"]["returncode"] == 1
    assert outcomes["wrong-pin"]["acceptance"]["verdict"] == "FAIL"


def run_review(argv):
    """Freeze test/source bytes, run all controls once, preserve failed nodes."""
    import argparse
    import tempfile
    parser = argparse.ArgumentParser(description=run_review.__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args(argv)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="review-", dir=args.artifact_dir))
    if args.scan_only:
        import re
        spec = importlib.util.spec_from_file_location(
            "independent_safe_scan", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
        scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scanner)
        needles = scanner.read_needles(argparse.Namespace(
            secret_hex=None, secrets_file=None, no_needle=False,
            needle_from=str(Path.home() / ".config/dsh-n600-token")))
        assert needles, "nonempty in-memory credential scan required"
        paths = [str(Path(__file__).resolve()),
                 str(ROOT / "docs/research/rotated-cuda-acceptance-independent.md")]
        literal, _ = scanner.scan_literal(paths, needles)
        pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                             r"""(?i:password|token|secret)\s*[\"']?\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']""")
        shape = sum(bool(pattern.search(Path(path).read_text())) for path in paths)
        receipt = {"verdict": "PASS" if literal == shape == 0 else "FAIL",
                   "literal_hits": literal, "shape_hits": shape, "paths": paths}
        put(output / "scan.json", receipt)
        print(json.dumps({"receipt": str(output / "scan.json"), **receipt}))
        return 0 if receipt["verdict"] == "PASS" else 1
    before = {"commit": git(ROOT, "rev-parse", "HEAD"), "status": git(ROOT, "status", "--porcelain"),
              "gate": sha(GATE), "test": sha(__file__)}
    command = [sys.executable, "-m", "pytest", "-q", "-ra", str(Path(__file__).resolve()),
               "--basetemp", str(output / "tmp"), "--junitxml", str(output / "pytest.xml"),
               "-o", f"cache_dir={output / 'cache'}"]
    with (output / "pytest.log").open("x") as log:
        process = subprocess.run(command, cwd=ROOT,
                                 env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1",
                                      "PYTHONDONTWRITEBYTECODE": "1", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"},
                                 stdout=log, stderr=subprocess.STDOUT, check=False, timeout=180)
    after = {"commit": git(ROOT, "rev-parse", "HEAD"), "status": git(ROOT, "status", "--porcelain"),
             "gate": sha(GATE), "test": sha(__file__)}
    suite = ET.parse(output / "pytest.xml").getroot()[0]
    receipt = {"verdict": "PASS" if process.returncode == 0 and before == after else "FAIL",
               "returncode": process.returncode, "source_consistency": before == after,
               "checked_at": datetime.now(timezone.utc).isoformat(),
               "source": before, "counts": {k: int(suite.get(k)) for k in ("tests", "failures", "errors", "skipped")},
               "seconds": float(suite.get("time")), "command": command,
               "failed_nodes": [case.get("name") for case in suite if case.find("failure") is not None],
               "qualification": "independent CPU regression; synthetic fixtures are not GPU origin evidence",
               "artifacts": {path.relative_to(output).as_posix(): sha(path)
                             for path in output.rglob("observation.json")},
               "junit_sha256": sha(output / "pytest.xml"), "log_sha256": sha(output / "pytest.log")}
    put(output / "receipt.json", receipt)
    print(json.dumps({"receipt": str(output / "receipt.json"), **receipt}, sort_keys=True))
    return 0 if receipt["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(run_review(sys.argv[1:]))
