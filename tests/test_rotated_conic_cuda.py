"""Version-1 bounded RSOC CUDA producer; CPU mechanisms never qualify CUDA.

The --replay CLI only decodes pinned original evidence, never solves.
The --collect-contract CLI checks the exact thirteen-member CUDA selection.
Both retain external logs. See docs/research/rotated-conic-cuda.md.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import traceback

import numpy as np
import pytest
import torch

from openopt.conic import (
    ConicProblem, RSOC, Zero, check_cone_feasibility, check_conic_candidate,
)
from openopt.native_conic import NativeConicConfig, prepare_native_conic, project_cones

ROOT = Path(__file__).resolve().parents[1]
TEST_PATH = "tests/test_rotated_conic_cuda.py"
FAMILY = "rotated_conic_cuda"
DTYPES = {"torch.float32": torch.float32, "torch.float64": torch.float64}
MODEL_IDS = (
    "rotated_lp-float32", "rotated_lp-float64",
    "rotated_qp-float32", "rotated_qp-float64",
)
PROJECTION_IDS = (
    "interior-asymmetric-float64", "interior-overflow-float64",
    "near-polar-float64", "small-head-float64", "small-head-float32",
    "tail-overflow-float64",
)
NEGATIVE_IDS = ("wrong-equality-dual", "planted-slack", "wrong-dual-cone")
TARGETS = {
    "test_cuda_rotated_original_model": MODEL_IDS,
    "test_cuda_rotated_projection_extremes": PROJECTION_IDS,
    "test_cuda_rotated_original_rejections": NEGATIVE_IDS,
}
SKIP_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable: all 13 RSOC GPU cases remain UNVERIFIED",
)
CHECK_NAMES = (
    "primal_equation", "primal_cone", "original_slack_cone", "dual_cone",
    "stationarity", "complementarity_block_0", "dual_objective_available",
    "duality_gap",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _exact(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False,
    )


def _host(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu", dtype=torch.float64).numpy().copy()
    return np.asarray(value, dtype=np.float64).copy()


def _source_hashes(root=ROOT):
    paths = sorted(path.relative_to(root).as_posix() for path in (root / "src").rglob("*.py"))
    _require(bool(paths), "empty src source membership")
    paths.append(TEST_PATH)
    for name in paths:
        path = root / name
        _require(path.is_file() and not path.is_symlink(), f"unsafe source: {name}")
    return {name: _sha(root / name) for name in paths}


def _loaded_sources():
    observed = {}
    for name in ("openopt.conic", "openopt.native_conic", "openopt.certificates"):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        expected = (ROOT / "src" / Path(*name.split("."))).with_suffix(".py").resolve()
        _require(path == expected, f"foreign loaded numerical module: {name}")
        observed[name] = {"source": path.relative_to(ROOT).as_posix(), "sha256": _sha(path)}
    return observed


def _runtime(device):
    cuda = torch.device(device).type == "cuda"
    return {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "executable": sys.executable, "numpy": np.__version__, "torch": torch.__version__,
        "compiled_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "cuda_execution": cuda,
        "gpu_name": torch.cuda.get_device_name(device) if cuda else None,
        "capability": list(torch.cuda.get_device_capability(device)) if cuda else None,
        "environment": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
        )},
    }


def _metadata(tensors):
    return {name: {
        "device": str(tensor.device), "dtype": str(tensor.dtype),
        "shape": list(tensor.shape), "layout": str(tensor.layout),
    } for name, tensor in tensors.items()}


def _freeze(problem):
    return {
        **{name: _host(getattr(problem, name)).tolist() for name in ("P", "q", "A", "b")},
        "cones": [{"kind": "zero" if type(cone) is Zero else "rsoc",
                   "dimension": cone.dimension} for cone in problem.cones],
    }


def _decode(saved):
    _require(isinstance(saved, dict) and set(saved) == {"P", "q", "A", "b", "cones"},
             "complete frozen original model required")
    cones = []
    for cone in saved["cones"]:
        _require(set(cone) == {"kind", "dimension"} and cone["kind"] in ("zero", "rsoc"),
                 "unsupported frozen original cone")
        cones.append({"zero": Zero, "rsoc": RSOC}[cone["kind"]](cone["dimension"]))
    return ConicProblem(*(np.asarray(saved[name], dtype=np.float64)
                          for name in ("P", "q", "A", "b")), tuple(cones))


def _analytic_model(name):
    root = math.sqrt(2.0)
    if name == "rotated_lp":
        P, q, dual = np.zeros((3, 3)), [1., 1., 0.], [-root, 1., 1., -root]
    elif name == "rotated_qp":
        P, q, dual = np.diag([1., 1., 0.]), [0., 0., 0.], [-2., root, root, -2.]
    else:
        raise ValueError(f"unknown analytic case: {name}")
    return ConicProblem(
        P, np.asarray(q), np.vstack(([[0., 0., 1.]], -np.eye(3))),
        np.array([2., 0., 0., 0.]), (Zero(1), RSOC(3)),
    ), {"x": [root, root, 2.], "s": [0., root, root, 2.], "z": dual}


def _config(device, dtype):
    return NativeConicConfig(
        enabled=True, device=device, dtype=dtype,
        eps_abs=3e-5 if dtype == torch.float32 else 1e-8, eps_rel=0,
    )


def _config_dict(config):
    return {field.name: str(config.dtype) if field.name == "dtype" else getattr(config, field.name)
            for field in fields(config)}


def _checks(problem, vectors, atol, rtol):
    values = {name: _host(vectors[name]) for name in ("x", "s", "z")}
    evidence = check_conic_candidate(problem, **values, atol=atol, rtol=rtol).to_dict()
    expected_checks = set(CHECK_NAMES) | {
        f"complementarity_block_{index}" for index in range(len(problem.cones))
    }
    _require(set(evidence["checks"]) == expected_checks, "incomplete original KKT/gap gate")
    cones = {
        "supplied_slack": check_cone_feasibility(values["s"], problem.cones, atol=atol, rtol=rtol),
        "original_slack": check_cone_feasibility(
            _host(problem.b) - _host(problem.A) @ values["x"], problem.cones, atol=atol, rtol=rtol,
        ),
        "dual": check_cone_feasibility(values["z"], problem.cones, dual=True, atol=atol, rtol=rtol),
    }
    return evidence, {name: item.to_dict() for name, item in cones.items()}


def _gpu_problem(problem, device):
    return replace(problem, **{name: torch.tensor(
        getattr(problem, name), device=device, dtype=torch.float64,
    ) for name in ("P", "q", "A", "b")})


def _solve(name, device, dtype):
    problem, analytic = _analytic_model(name)
    original = _gpu_problem(problem, device)
    before = _freeze(original)
    config = _config(device, dtype)
    prepared = prepare_native_conic(original, config)
    result = prepared.solve()
    if device == "cuda:0":
        torch.cuda.synchronize(device)
    assert result.status == "solved" and result.evidence.passed, result.evidence.to_dict()
    assert not result.certified
    assert _freeze(original) == before
    np.testing.assert_allclose(_host(result.x), analytic["x"], atol=20 * config.eps_abs, rtol=0)
    tensors = {
        **{f"original.{key}": getattr(original, key) for key in ("P", "q", "A", "b")},
        **{f"prepared.{key}": getattr(prepared, "_" + key) for key in ("P", "q", "A", "b", "factor")},
        **{key: getattr(result, key) for key in ("x", "s", "z")},
    }
    for key, tensor in tensors.items():
        assert str(tensor.device) == device
        assert tensor.dtype == (torch.float64 if key.startswith("original.") else dtype)
    return problem, config, result, tensors, analytic


def _projection_input(case_id):
    values = {
        "interior-asymmetric-float64": [1., 1e20, 0.],
        "interior-overflow-float64": [1.3e308, 1.3e308, 0.],
        "near-polar-float64": [-1e20, 0., 1e8],
        "small-head-float64": [-1e300, 0., 1e210],
        "small-head-float32": [-1e30, 0., 1e17],
        "tail-overflow-float64": [0., 0., 1.3e308, 1.3e308],
    }
    return values[case_id], DTYPES["torch." + case_id.rsplit("-", 1)[1]]


def _projection_oracle(values):
    # Bounded independent analytic oracle, never part of the optimizer.
    with localcontext() as context:
        context.prec = 800
        a, b, *tail = [Decimal.from_float(float(value)) for value in values]
        radius_squared = sum((value * value for value in tail), Decimal(0))
        if a >= 0 and b >= 0 and 2 * a * b >= radius_squared:
            return [float(value) for value in (a, b, *tail)]
        if a <= 0 and b <= 0 and 2 * a * b >= radius_squared:
            return [0.] * len(values)
        root_two = Decimal(2).sqrt()
        head, difference = (a + b) / root_two, (a - b) / root_two
        radius = (difference * difference + radius_squared).sqrt()
        projected_head = (head + radius) / 2
        weight = projected_head / radius
        return [float((projected_head + weight * difference) / root_two),
                float((projected_head - weight * difference) / root_two),
                *(float(weight * value) for value in tail)]


def _projection_check(values, projected, dtype, case_id):
    expected = np.asarray(_projection_oracle(values))
    tolerance = 3e-6 if dtype == torch.float32 else 3e-14
    actual = np.asarray(projected, dtype=np.float64)
    _require(actual.shape == expected.shape and bool(np.isfinite(actual).all()),
             "invalid projection coordinates")
    np.testing.assert_allclose(actual, expected, atol=0, rtol=tolerance)
    if case_id.startswith("interior-"):
        _require(_exact(list(values), list(projected)), "interior projection changed input")
    return {"algorithm": "bounded_decimal_analytic_rsoc_v1", "precision": 800,
            "expected": expected.tolist(), "atol": 0., "rtol": tolerance,
            "passed": True}


def _report(problem, config, vectors, tensors, identity, case_id, kind, status, iterations):
    evidence, cones = _checks(problem, vectors, config.eps_abs, config.eps_rel)
    return {
        "schema_version": 1, "family": FAMILY, "verdict": "PASS",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "test_identity": identity, "case_id": case_id, "case_kind": kind,
        "device": config.device, "dtype": str(config.dtype), "config": _config_dict(config),
        "runtime": _runtime(config.device), "loaded_sources": _loaded_sources(),
        "source_sha256": _source_hashes(), "tensor_metadata": _metadata(tensors),
        "problem": _freeze(problem),
        **{key: _host(vectors[key]).tolist() for key in ("x", "s", "z")},
        "status": status, "iterations": iterations, "evidence": evidence,
        "cone_evidence": cones, "objective": evidence["metrics"]["primal_objective"],
        "certified": False, "qualified": False, "commercial": "UNVERIFIED",
        "performance": "UNVERIFIED",
    }


def _identity(request):
    return "tests.test_rotated_conic_cuda::" + request.node.name


def _save(report, tmp_path, request):
    path = tmp_path / "rotated-conic.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert _exact(json.loads(path.read_text()), report)
    receipt = replay_sidecar(path, _sha(path), ROOT, require_cuda=report["device"] == "cuda:0")
    assert receipt["verdict"] == "PASS"
    request.node.user_properties.extend((
        ("rotated_conic_acceptance_path", str(path)),
        ("rotated_conic_acceptance_sha256", _sha(path)),
    ))
    return path


def _produce_model(case_id, device, tmp_path, request):
    name, dtype_name = case_id.rsplit("-", 1)
    problem, config, result, tensors, analytic = _solve(name, device, DTYPES["torch." + dtype_name])
    report = _report(problem, config, tensors, tensors, _identity(request), case_id,
                     "original_model", result.status, result.iterations)
    assert _exact(report["evidence"], result.evidence.to_dict())
    assert report["evidence"]["passed"]
    analytic_evidence, _ = _checks(problem, analytic, 1e-12, 0)
    assert analytic_evidence["passed"]
    report["analytic"] = {"vectors": analytic, "evidence": analytic_evidence}
    return _save(report, tmp_path, request)


def _produce_projection(case_id, device, tmp_path, request):
    values, dtype = _projection_input(case_id)
    value = torch.tensor(values, device=device, dtype=dtype)
    before = value.clone()
    projected = project_cones(value, (RSOC(len(values)),))
    if device == "cuda:0":
        torch.cuda.synchronize(device)
    assert torch.equal(value, before)
    assert projected.device == value.device and projected.dtype == dtype
    oracle = _projection_check(_host(value).tolist(), _host(projected).tolist(), dtype, case_id)
    # A full original feasibility witness avoids an overflowing projection-QP objective.
    problem = ConicProblem(np.zeros((1, 1)), np.zeros(1), np.zeros((len(values), 1)),
                           _host(projected), (RSOC(len(values)),))
    original = _gpu_problem(problem, device)
    vectors = {"x": torch.zeros(1, device=device, dtype=dtype), "s": projected,
               "z": torch.zeros_like(projected)}
    tensors = {**vectors, "projection.input": value, "projection.output": projected,
               **{f"original.{key}": getattr(original, key) for key in ("P", "q", "A", "b")}}
    # No native iterations/configured solve is claimed for projection-only cases.
    config = _config(device, dtype)
    config = replace(config, eps_abs=1e-12, eps_rel=3e-6 if dtype == torch.float32 else 3e-14)
    report = _report(problem, config, vectors, tensors, _identity(request), case_id,
                     "projection", "projected", 0)
    assert report["evidence"]["passed"], report["evidence"]
    report["projection"] = {"input": _host(value).tolist(), "output": _host(projected).tolist(),
                            "oracle": oracle, "input_unchanged": True}
    return _save(report, tmp_path, request)


def _mutate(vectors, case_id):
    changed = {name: value.clone() if isinstance(value, torch.Tensor) else np.array(value, copy=True)
               for name, value in vectors.items() if name in ("x", "s", "z")}
    if case_id == "wrong-equality-dual":
        changed["z"][0] *= -1
        failed = ["stationarity"]
    elif case_id == "planted-slack":
        changed["x"][0:2] = 0
        changed["x"][2] = 2
        failed = ["primal_equation", "original_slack_cone"]
    elif case_id == "wrong-dual-cone":
        changed["z"][1:] *= -1
        failed = ["dual_cone"]
    else:
        raise ValueError(f"unknown negative case: {case_id}")
    return changed, failed


def _produce_negative(case_id, device, tmp_path, request):
    problem, config, result, tensors, _ = _solve("rotated_lp", device, torch.float64)
    baseline = {name: _host(tensors[name]).tolist() for name in ("x", "s", "z")}
    changed, failed = _mutate(tensors, case_id)
    report = _report(problem, config, changed, {**tensors, **changed}, _identity(request), case_id,
                     "original_rejection", "candidate_rejected", result.iterations)
    assert not report["evidence"]["passed"]
    assert all(not report["evidence"]["checks"][name] for name in failed)
    assert baseline == {name: _host(tensors[name]).tolist() for name in ("x", "s", "z")}
    report["negative"] = {
        "mutation": case_id, "expected_failed_checks": failed,
        "baseline": {**baseline, "evidence": result.evidence.to_dict(), "status": result.status,
                     "tensor_metadata": _metadata({name: tensors[name] for name in ("x", "s", "z")})},
    }
    return _save(report, tmp_path, request)


def _pairs(items):
    result = {}
    for key, value in items:
        _require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def replay_sidecar(path, expected_sha256, source_root, *, require_cuda=True):
    _require(isinstance(expected_sha256, str) and re.fullmatch("[0-9a-f]{64}", expected_sha256),
             "caller-pinned sidecar SHA256 required")
    _require(_sha(path) == expected_sha256, "caller sidecar hash mismatch")
    saved = json.loads(path.read_bytes(), object_pairs_hook=_pairs, parse_constant=_nonfinite)
    base_fields = {
        "schema_version", "family", "verdict", "checked_at", "test_identity", "case_id",
        "case_kind", "device", "dtype", "config", "runtime", "loaded_sources", "source_sha256",
        "tensor_metadata", "problem", "x", "s", "z", "status", "iterations", "evidence",
        "cone_evidence", "objective", "certified", "qualified", "commercial", "performance",
    }
    extension = {"original_model": "analytic", "projection": "projection",
                 "original_rejection": "negative"}.get(saved.get("case_kind"))
    _require(extension is not None and set(saved) == base_fields | {extension},
             "exact complete schema1 fields required")
    _require(saved["schema_version"] == 1 and saved["family"] == FAMILY
             and saved["verdict"] == "PASS", "unsupported successful execution schema")
    _require(source_root.resolve() == ROOT, "replay must use its explicit trusted current root")
    _require(_exact(saved["source_sha256"], _source_hashes(source_root)), "source membership/hash drift")
    for module, binding in saved["loaded_sources"].items():
        _require(module in ("openopt.conic", "openopt.native_conic", "openopt.certificates")
                 and set(binding) == {"source", "sha256"}
                 and binding["source"] == "src/" + module.replace(".", "/") + ".py"
                 and binding["sha256"] == saved["source_sha256"][binding["source"]],
                 "loaded source binding mismatch")
    _require(set(saved["loaded_sources"]) == {
        "openopt.conic", "openopt.native_conic", "openopt.certificates",
    }, "incomplete loaded source binding")
    _loaded_sources()
    device, dtype = saved["device"], saved["dtype"]
    _require(device in ("cpu", "cuda:0") and dtype in DTYPES, "unsupported explicit device/dtype")
    _require(not require_cuda or device == "cuda:0", "CPU mechanism is not CUDA evidence")
    runtime = saved["runtime"]
    _require(set(runtime) == {
        "python", "implementation", "executable", "numpy", "torch", "compiled_cuda",
        "cuda_available", "cuda_execution", "gpu_name", "capability", "environment",
    } and all(isinstance(runtime[name], str) and bool(runtime[name]) for name in (
        "python", "implementation", "executable", "numpy", "torch",
    )) and set(runtime["environment"]) == {
        "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "CUDA_VISIBLE_DEVICES",
    } and all(value is None or isinstance(value, str) for value in runtime["environment"].values()),
             "incomplete observed runtime contract")
    _require(runtime["cuda_execution"] is (device == "cuda:0"), "runtime execution mismatch")
    if device == "cuda:0":
        _require(runtime["cuda_available"] is True and isinstance(runtime["gpu_name"], str)
                 and bool(runtime["gpu_name"]) and runtime["compiled_cuda"] is not None
                 and len(runtime["capability"]) == 2, "incomplete actual CUDA runtime")
    else:
        _require(runtime["gpu_name"] is None and runtime["capability"] is None,
                 "CPU mechanism cannot report actual GPU execution")
    config = saved["config"]
    _require(set(config) == {field.name for field in fields(NativeConicConfig)}
             and config["enabled"] is True and config["device"] == device and config["dtype"] == dtype,
             "incomplete or contradictory config")
    case_id, kind = saved["case_id"], saved["case_kind"]
    if kind == "original_model":
        _require(case_id in MODEL_IDS, "unknown original case")
        _require(dtype == "torch." + case_id.rsplit("-", 1)[1], "original case dtype mismatch")
        expected_config = _config_dict(_config(device, DTYPES[dtype]))
        target = "test_cuda_rotated_original_model"
    elif kind == "projection":
        _require(case_id in PROJECTION_IDS, "unknown projection case")
        _, expected_dtype = _projection_input(case_id)
        _require(DTYPES[dtype] == expected_dtype, "projection dtype mismatch")
        expected_config = _config_dict(replace(
            _config(device, expected_dtype), eps_abs=1e-12,
            eps_rel=3e-6 if expected_dtype == torch.float32 else 3e-14,
        ))
        target = "test_cuda_rotated_projection_extremes"
    else:
        _require(kind == "original_rejection" and case_id in NEGATIVE_IDS
                 and dtype == "torch.float64", "unknown negative case")
        expected_config = _config_dict(_config(device, torch.float64))
        target = "test_cuda_rotated_original_rejections"
    _require(_exact(config, expected_config), "fixed full config contract mismatch")
    if device == "cuda:0":
        _require(saved["test_identity"] == f"tests.test_rotated_conic_cuda::{target}[{case_id}]",
                 "exact selected JUnit identity mismatch")
    else:
        _require(saved["test_identity"].startswith("tests.test_rotated_conic_cuda::test_cpu_"),
                 "CPU mechanism identity required")
    problem = _decode(saved["problem"])
    if kind != "projection":
        # Validate the small exact case contract directly, without a solve/factory.
        model_name = case_id.rsplit("-", 1)[0] if kind == "original_model" else "rotated_lp"
        expected_P = np.zeros((3, 3)) if model_name == "rotated_lp" else np.diag([1., 1., 0.])
        expected_q = [1., 1., 0.] if model_name == "rotated_lp" else [0., 0., 0.]
        _require(np.array_equal(problem.P, expected_P) and _exact(problem.q.tolist(), expected_q)
                 and np.array_equal(problem.A, np.vstack(([[0., 0., 1.]], -np.eye(3))))
                 and _exact(problem.b.tolist(), [2., 0., 0., 0.])
                 and problem.cones == (Zero(1), RSOC(3)), "frozen analytic original model changed")
    vectors = {name: saved[name] for name in ("x", "s", "z")}
    required = set(vectors) | {f"original.{name}" for name in ("P", "q", "A", "b")}
    if kind == "projection":
        required |= {"projection.input", "projection.output"}
    else:
        required |= {f"prepared.{name}" for name in ("P", "q", "A", "b", "factor")}
    metadata = saved["tensor_metadata"]
    _require(set(metadata) == required, "incomplete actual tensor placement membership")
    shapes = {**{name: list(np.asarray(vectors[name]).shape) for name in vectors},
              **{f"original.{name}": list(_host(getattr(problem, name)).shape)
                 for name in ("P", "q", "A", "b")}}
    if kind == "projection":
        shapes.update({"projection.input": list(problem.b.shape),
                       "projection.output": list(problem.b.shape)})
    else:
        shapes.update({f"prepared.{name}": shapes["original." + name] for name in ("P", "q", "A", "b")})
        shapes["prepared.factor"] = shapes["original.P"]
    for name, item in metadata.items():
        _require(_exact(item, {"device": device, "dtype": "torch.float64" if name.startswith("original.")
                             else dtype, "shape": shapes[name], "layout": "torch.strided"}),
                 f"actual placement/dtype/shape mismatch: {name}")
    evidence, cone_evidence = _checks(problem, vectors, config["eps_abs"], config["eps_rel"])
    _require(_exact(saved["evidence"], evidence) and _exact(saved["cone_evidence"], cone_evidence),
             "exact independent original checker replay mismatch")
    _require(_exact(saved["objective"], evidence["metrics"]["primal_objective"]),
             "original objective mismatch")
    _require(saved["certified"] is False and saved["qualified"] is False
             and saved["commercial"] == saved["performance"] == "UNVERIFIED",
             "unsupported qualification claim")
    if kind == "original_rejection":
        negative = saved["negative"]
        baseline = negative["baseline"]
        baseline_evidence, _ = _checks(problem, baseline, config["eps_abs"], config["eps_rel"])
        _require(baseline_evidence["passed"] and _exact(baseline["evidence"], baseline_evidence)
                 and baseline["status"] == "solved", "negative baseline not independently accepted")
        _require(_exact(baseline["tensor_metadata"], {name: metadata[name] for name in vectors}),
                 "baseline actual tensor placement mismatch")
        changed, failed = _mutate(baseline, case_id)
        _require(negative["mutation"] == case_id and negative["expected_failed_checks"] == failed
                 and _exact({name: _host(changed[name]).tolist() for name in changed}, vectors),
                 "negative mutation contract mismatch")
        _require(not evidence["passed"] and all(not evidence["checks"][name] for name in failed)
                 and saved["status"] == "candidate_rejected", "original rejection not observed")
    else:
        _require(evidence["passed"], "independent original candidate rejected")
        if kind == "original_model":
            _require(saved["status"] == "solved", "native solve not completed")
            analytic = saved["analytic"]
            root = math.sqrt(2.0)
            expected_dual = ([-root, 1., 1., -root] if model_name == "rotated_lp"
                             else [-2., root, root, -2.])
            _require(_exact(analytic["vectors"], {
                "x": [root, root, 2.], "s": [0., root, root, 2.], "z": expected_dual,
            }), "original analytic coordinates changed")
            original_analytic, _ = _checks(problem, analytic["vectors"], 1e-12, 0)
            _require(original_analytic["passed"] and _exact(analytic["evidence"], original_analytic),
                     "analytic original KKT/gap mismatch")
            np.testing.assert_allclose(vectors["x"], analytic["vectors"]["x"],
                                       atol=20 * config["eps_abs"], rtol=0)
        else:
            projection = saved["projection"]
            values, projection_dtype = _projection_input(case_id)
            represented = np.asarray(values, dtype=np.float32 if projection_dtype == torch.float32
                                     else np.float64).astype(np.float64).tolist()
            _require(_exact(projection["input"], represented)
                     and problem.cones == (RSOC(len(values)),), "frozen projection case input changed")
            _require(saved["status"] == "projected" and projection["input_unchanged"] is True
                     and _exact(projection["output"], vectors["s"]), "projection witness mismatch")
            oracle = _projection_check(projection["input"], projection["output"], DTYPES[dtype], case_id)
            _require(_exact(projection["oracle"], oracle), "independent projection oracle mismatch")
            _require(bool(np.all(problem.P == 0)) and bool(np.all(problem.q == 0))
                     and bool(np.all(problem.A == 0)) and _exact(problem.b.tolist(), vectors["s"])
                     and vectors["x"] == [0.] and vectors["z"] == [0.] * len(vectors["s"]),
                     "projection original feasibility witness changed")
    _require(type(saved["iterations"]) is int
             and (saved["iterations"] == 0 if kind == "projection" else
                  0 < saved["iterations"] <= config["max_iterations"]),
             "invalid actual iteration count")
    return {"verdict": "PASS", "family": FAMILY, "sidecar_sha256": expected_sha256,
            "test_identity": saved["test_identity"], "cuda_execution": device == "cuda:0",
            "candidate_passed": evidence["passed"], "qualified": False,
            "commercial": "UNVERIFIED", "performance": "UNVERIFIED"}


@SKIP_CUDA
@pytest.mark.parametrize("case_id", MODEL_IDS, ids=MODEL_IDS)
def test_cuda_rotated_original_model(case_id, tmp_path, request):
    _produce_model(case_id, "cuda:0", tmp_path, request)


@SKIP_CUDA
@pytest.mark.parametrize("case_id", PROJECTION_IDS, ids=PROJECTION_IDS)
def test_cuda_rotated_projection_extremes(case_id, tmp_path, request):
    _produce_projection(case_id, "cuda:0", tmp_path, request)


@SKIP_CUDA
@pytest.mark.parametrize("case_id", NEGATIVE_IDS, ids=NEGATIVE_IDS)
def test_cuda_rotated_original_rejections(case_id, tmp_path, request):
    _produce_negative(case_id, "cuda:0", tmp_path, request)


@pytest.mark.parametrize("case_id", MODEL_IDS, ids=MODEL_IDS)
def test_cpu_original_mechanism(case_id, tmp_path, request):
    path = _produce_model(case_id, "cpu", tmp_path, request)
    with pytest.raises(ValueError, match="not CUDA evidence"):
        replay_sidecar(path, _sha(path), ROOT)


@pytest.mark.parametrize("case_id", PROJECTION_IDS, ids=PROJECTION_IDS)
def test_cpu_projection_mechanism(case_id, tmp_path, request):
    _produce_projection(case_id, "cpu", tmp_path, request)


@pytest.mark.parametrize("case_id", NEGATIVE_IDS, ids=NEGATIVE_IDS)
def test_cpu_original_rejection_mechanism(case_id, tmp_path, request):
    _produce_negative(case_id, "cpu", tmp_path, request)


@pytest.mark.parametrize("mutation", [
    "vector", "placement", "config", "source", "identity", "evidence",
    "model", "loaded-source", "runtime", "case-dtype",
])
def test_cpu_rehashed_corruption_rejected(mutation, tmp_path, request):
    path = _produce_model("rotated_lp-float64", "cpu", tmp_path, request)
    saved = json.loads(path.read_text())
    if mutation == "vector":
        saved["x"][0] += 1
    elif mutation == "placement":
        saved["tensor_metadata"]["x"]["device"] = "cuda:0"
    elif mutation == "config":
        saved["config"]["eps_abs"] *= 10
    elif mutation == "source":
        saved["source_sha256"]["src/openopt/conic.py"] = "0" * 64
    elif mutation == "identity":
        saved["test_identity"] = "tests.test_rotated_conic_cuda::test_cuda_rotated_original_model[x]"
    elif mutation == "evidence":
        saved["evidence"]["metrics"]["duality_gap"] += 1
    elif mutation == "model":
        saved["problem"]["b"][0] = 3
    elif mutation == "loaded-source":
        saved["loaded_sources"]["openopt.certificates"] = saved["loaded_sources"]["openopt.conic"]
    elif mutation == "runtime":
        saved["runtime"]["cuda_execution"] = True
    else:
        saved["case_id"] = "rotated_lp-float32"
    altered = tmp_path / "corrupted.json"
    altered.write_text(json.dumps(saved, allow_nan=False))
    with pytest.raises(ValueError):
        replay_sidecar(altered, _sha(altered), ROOT, require_cuda=False)


def test_cpu_replay_never_solves_or_reconstructs_factory(tmp_path, request, monkeypatch):
    path = _produce_model("rotated_qp-float64", "cpu", tmp_path, request)

    def forbidden(*args, **kwargs):
        raise AssertionError("replay consulted a solve or model factory")

    monkeypatch.setattr(sys.modules[__name__], "_analytic_model", forbidden)
    monkeypatch.setattr(sys.modules[__name__], "prepare_native_conic", forbidden)
    monkeypatch.setattr(sys.modules[__name__], "project_cones", forbidden)
    monkeypatch.setattr("openopt.conic.solve_conic", forbidden)
    assert replay_sidecar(path, _sha(path), ROOT, require_cuda=False)["verdict"] == "PASS"


def test_cpu_cli_replay_has_external_terminal_and_rejects_cuda_claim(tmp_path, request):
    if not tmp_path.resolve().is_relative_to(Path("/hf3fs-jd/data/tmp").resolve()):
        pytest.skip("CLI log-placement test requires external 3FS basetemp; not CUDA qualification")
    path = _produce_model("rotated_lp-float64", "cpu", tmp_path, request)
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"}
    outcomes = {}
    for mode in ("cpu", "cuda-required"):
        output = tmp_path / ("cli-" + mode)
        command = [
            sys.executable, str(ROOT / TEST_PATH), "--replay", str(path),
            "--sha256", _sha(path), "--source-root", str(ROOT), "--artifact-dir", str(output),
        ]
        if mode == "cpu":
            command.append("--cpu-mechanism")
        result = subprocess.run(command, cwd=ROOT, env=environment,
                                capture_output=True, text=True, check=False)
        (tmp_path / ("cli-" + mode + ".log")).write_text(result.stdout + result.stderr)
        outcomes[mode] = {"returncode": result.returncode,
                          "report": json.loads((output / "acceptance.json").read_text())}
    assert outcomes["cpu"]["returncode"] == 0
    assert outcomes["cpu"]["report"]["cuda_execution"] is False
    assert outcomes["cuda-required"]["returncode"] == 1
    assert outcomes["cuda-required"]["report"]["verdict"] == "FAIL"
    assert "not CUDA evidence" in outcomes["cuda-required"]["report"]["error"]


@pytest.mark.parametrize("mutation", ["input", "oracle", "membership"])
def test_cpu_projection_replay_rejects_rehashed_mutation(mutation, tmp_path, request):
    path = _produce_projection("near-polar-float64", "cpu", tmp_path, request)
    saved = json.loads(path.read_text())
    if mutation == "input":
        saved["projection"]["input"][2] *= 2
    elif mutation == "oracle":
        saved["projection"]["oracle"]["rtol"] *= 10
    else:
        saved["tensor_metadata"].pop("projection.input")
    altered = tmp_path / "projection-corrupted.json"
    altered.write_text(json.dumps(saved, allow_nan=False))
    with pytest.raises(ValueError):
        replay_sidecar(altered, _sha(altered), ROOT, require_cuda=False)


def test_cpu_duplicate_json_rejected(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate JSON key"):
        replay_sidecar(path, _sha(path), ROOT, require_cuda=False)


def test_cpu_exact_denominator_contract():
    assert tuple(map(len, TARGETS.values())) == (4, 6, 3)
    assert len(expected_selection()) == len(set(expected_selection())) == 13


def expected_selection():
    return [f"{TEST_PATH}::{target}[{case_id}]"
            for target, cases in TARGETS.items() for case_id in cases]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--collect-contract", action="store_true")
    modes.add_argument("--replay", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--cpu-mechanism", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.artifact_dir.resolve()
    _require(out.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve()) and not out.is_relative_to(ROOT),
             "logs must be external shared 3FS tmp")
    out.mkdir(parents=True, exist_ok=False)
    rc = 0
    try:
        if args.collect_contract:
            command = [sys.executable, "-m", "pytest", "--collect-only", "-q",
                       "-o", f"cache_dir={out / 'pytest-cache'}",
                       *[f"{TEST_PATH}::{target}" for target in TARGETS]]
            environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1",
                           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
                           "PYTHONPATH": str(ROOT / "src") + os.pathsep + str(ROOT)}
            collected = subprocess.run(command, cwd=ROOT, env=environment,
                                       capture_output=True, text=True, check=False)
            (out / "collection.log").write_text(collected.stdout + collected.stderr)
            _require(collected.returncode == 0, f"pytest collection failed rc={collected.returncode}")
            observed = [line for line in collected.stdout.splitlines() if line.startswith(TEST_PATH + "::")]
            _require(observed == expected_selection(), "exact CUDA selection mismatch")
            report = {"verdict": "PASS", "selection": observed, "positive": 10, "negative": 3,
                      "selected": 13, "cuda_execution": False, "qualified": False,
                      "source_sha256": _source_hashes()}
        else:
            report = replay_sidecar(args.replay, args.sha256, args.source_root.resolve(),
                                    require_cuda=not args.cpu_mechanism)
    except Exception as error:
        rc = 1
        report = {"verdict": "FAIL", "error": f"{type(error).__name__}: {error}"}
        (out / "failure.log").write_text(traceback.format_exc())
    report["checked_at"] = datetime.now(timezone.utc).isoformat()
    (out / "acceptance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    line = f"ROTATED_CONIC_EVIDENCE verdict={report['verdict']} at={report['checked_at']} report={out / 'acceptance.json'}"
    (out / "execution.log").write_text(line + f"\nreturncode={rc}\n")
    print(line)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
