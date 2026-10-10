"""Native conic numerical, fail-loud and original-model replay acceptance.

CUDA test names contain cuda and write full JSON proof sidecars to external
pytest tmp paths. This module's --replay CLI independently rechecks a pinned
sidecar; it never invokes a solver or reconstructs a model from a test factory.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import traceback

import numpy as np
import pytest
import torch

from openopt.conic import (
    ConicConfig, ConicProblem, Nonnegative, SOC, Zero,
    check_cone_feasibility, check_conic_candidate, solve_conic,
)
from openopt.native_conic import (
    NativeConicConfig, prepare_native_conic, project_cones, solve_native_conic,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/openopt/native_conic.py", "src/openopt/conic.py",
    "src/openopt/certificates.py", "tests/test_native_conic.py",
)
CONE_TYPES = {"zero": Zero, "nonnegative": Nonnegative, "soc": SOC}
CONFIG = NativeConicConfig(enabled=True, eps_abs=1e-8, eps_rel=0)


def case(name: str) -> tuple[ConicProblem, list[float]]:
    if name == "soc_lp":
        return ConicProblem(
            np.zeros((3, 3)), np.array([1., 0., 0.]),
            np.vstack([[[0., 1., 0.], [0., 0., 1.]], -np.eye(3)]),
            np.array([3., 4., 0., 0., 0.]), (Zero(2), SOC(3)),
        ), [5., 3., 4.]
    if name == "mixed":
        return ConicProblem(
            np.eye(4), np.array([0., -3., -4., 0.]),
            np.vstack([np.hstack([-np.eye(3), np.zeros((3, 1))]),
                       [[0., 0., 0., 1.]] * 2]),
            np.array([0., 0., 0., 0., -2.]), (SOC(3), Nonnegative(1), Zero(1)),
        ), [2.5, 1.5, 2., -2.]
    if name == "nonnegative":
        return ConicProblem(
            np.eye(2), np.array([-.2, -3.]),
            np.array([[1., 1.], [-1., 0.], [0., -1.]]),
            np.array([1., 0., 0.]), (Zero(1), Nonnegative(2)),
        ), [0., 1.]
    if name == "equality":
        return ConicProblem(
            np.array([[4., 1.], [1., 2.]]), np.array([3., -2.]),
            np.eye(2), np.array([1., -1.]), (Zero(2),),
        ), [1., -1.]
    if name == "redundant":
        return ConicProblem(
            np.eye(1), np.zeros(1), np.array([[1.], [1.], [0.]]),
            np.array([2., 2., 0.]), (Zero(3),),
        ), [2.]
    if name == "empty":
        return ConicProblem(
            np.array([[4., 1.], [1., 2.]]), np.array([-9., -5.]),
            np.empty((0, 2)), np.empty(0), (),
        ), [13. / 7, 11. / 7]
    if name == "singular":
        return ConicProblem(
            np.diag([2., 0.]), np.array([-4., 0.]),
            np.empty((0, 2)), np.empty(0), (),
        ), [2., 0.]
    if name == "zero_lp":
        return ConicProblem(
            np.zeros((2, 2)), np.zeros(2), np.empty((0, 2)), np.empty(0), (),
        ), [0., 0.]
    if name == "soc_origin":
        return ConicProblem(
            np.eye(3), np.array([1., 0., 0.]), -np.eye(3),
            np.zeros(3), (SOC(3),),
        ), [0., 0., 0.]
    if name == "soc2":
        return ConicProblem(
            np.eye(2), np.array([0., -2.]), -np.eye(2), np.zeros(2), (SOC(2),),
        ), [1., 1.]
    raise ValueError(name)


def host(value):
    return value.detach().cpu().double().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)


def frozen_problem(problem):
    kinds = {Zero: "zero", Nonnegative: "nonnegative", SOC: "soc"}
    return {
        **{name: host(getattr(problem, name)).tolist() for name in ("P", "q", "A", "b")},
        "cones": [{"kind": kinds[type(cone)], "dimension": cone.dimension} for cone in problem.cones],
    }


def decode_problem(saved):
    if not isinstance(saved, dict) or set(saved) != {"P", "q", "A", "b", "cones"}:
        raise ValueError("complete frozen original conic model required")
    q, b = np.asarray(saved["q"], dtype=float), np.asarray(saved["b"], dtype=float)
    A = np.asarray(saved["A"], dtype=float)
    if A.size == 0 and saved["A"] == []:
        A = A.reshape(0, q.size)
    cones = []
    for block in saved["cones"]:
        if set(block) != {"kind", "dimension"} or block["kind"] not in CONE_TYPES:
            raise ValueError("unknown frozen cone descriptor")
        cones.append(CONE_TYPES[block["kind"]](block["dimension"]))
    return ConicProblem(np.asarray(saved["P"], dtype=float), q, A, b, tuple(cones))


def save_sidecar(problem, result, config, tmp_path, request, *, reference=None):
    vectors = {name: host(getattr(result, name)).tolist() for name in ("x", "s", "z")}
    report = {
        "schema_version": 1, "verdict": "PASS", "family": "native_conic",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "device": result.device, "dtype": str(result.dtype),
        "placements": {name: str(getattr(result, name).device) for name in ("x", "s", "z")},
        "problem": frozen_problem(problem), **vectors, "status": result.status,
        "iterations": result.iterations, "objective": result.objective,
        "atol": config.eps_abs, "rtol": config.eps_rel,
        "config": {
            "enabled": config.enabled, "device": config.device, "dtype": str(config.dtype),
            "rho": config.rho, "sigma": config.sigma,
            "max_iterations": config.max_iterations, "check_interval": config.check_interval,
        },
        "evidence": result.evidence.to_dict(),
        "source_sha256": {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in SOURCE_PATHS},
        "qualification": "original-coordinate numerical candidate; not an exact proof or speed claim",
    }
    if result.x.device.type == "cuda":
        report["gpu_name"] = torch.cuda.get_device_name(result.x.device)
    if reference is not None:
        report["reference"] = {
            "backend": reference.backend, "status": reference.status,
            "version": reference.backend_version, "objective": reference.objective,
            **{name: getattr(reference, name).tolist() for name in ("x", "s", "z")},
            "evidence": reference.evidence.to_dict(),
        }
    path = tmp_path / "native-conic.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    readback = json.loads(path.read_text())
    assert readback == report
    assert replay_sidecar(path, hashlib.sha256(path.read_bytes()).hexdigest())["verdict"] == "PASS"
    request.node.user_properties.append(("native_conic_acceptance_path", str(path)))
    request.node.user_properties.append(("native_conic_acceptance_sha256",
                                         hashlib.sha256(path.read_bytes()).hexdigest()))
    return path


def replay_sidecar(path: Path, expected_sha256: str, source_root: Path | None = None):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate evidence key: {key}")
            result[key] = value
        return result

    def nonfinite(value):
        raise ValueError(f"nonfinite evidence constant: {value}")

    if not re.fullmatch("[0-9a-f]{64}", expected_sha256):
        raise ValueError("caller-pinned sha256 required")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError("sidecar hash mismatch")
    saved = json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    if (saved.get("schema_version") != 1 or saved.get("family") != "native_conic"
            or saved.get("verdict") != "PASS" or saved.get("status") != "solved"):
        raise ValueError("successful native conic candidate required")
    for name in ("atol", "rtol"):
        value = saved[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value > 1e-3:
            raise ValueError("finite bounded replay tolerance required")
    if type(saved["iterations"]) is not int or saved["iterations"] <= 0:
        raise ValueError("actual positive iteration count required")
    sources = saved["source_sha256"]
    if set(sources) != set(SOURCE_PATHS):
        raise ValueError("complete numerical source hashes required")
    for name, digest in sources.items():
        if not isinstance(digest, str) or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("invalid source hash")
        if source_root is not None and hashlib.sha256((source_root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"source hash mismatch: {name}")
    if saved["device"] not in ("cpu", "cuda:0"):
        raise ValueError("only explicitly observed cpu/cuda:0 replay supported")
    if set(saved["placements"]) != {"x", "s", "z"} or any(
            value != saved["device"] for value in saved["placements"].values()):
        raise ValueError("saved vector placement contradiction")
    problem = decode_problem(saved["problem"])
    evidence = check_conic_candidate(
        problem, saved["x"], saved["s"], saved["z"], atol=saved["atol"], rtol=saved["rtol"],
    )
    if not evidence.passed:
        raise ValueError(f"independent native conic candidate rejected: {evidence.to_dict()}")
    if saved["evidence"] != evidence.to_dict():
        raise ValueError("saved evidence disagrees with independent recomputation")
    if saved["objective"] != evidence.metrics["primal_objective"]:
        raise ValueError("saved objective disagrees with original model")
    return {
        "verdict": "PASS", "checked_at": datetime.now(timezone.utc).isoformat(),
        "sidecar_sha256": expected_sha256, "family": "native_conic",
        "evidence": evidence.to_dict(), "source_checked": source_root is not None,
        "qualification": "frozen-model numerical replay, not comprehensive qualification",
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value,cone,dual,expected", [
    ([2., -3.], Zero(2), False, [0., 0.]),
    ([2., -3.], Zero(2), True, [2., -3.]),
    ([2., -3.], Nonnegative(2), False, [2., 0.]),
    ([5., 3., 4.], SOC(3), False, [5., 3., 4.]),
    ([-5., 3., 4.], SOC(3), False, [0., 0., 0.]),
    ([0., 3., 4.], SOC(3), False, [2.5, 1.5, 2.]),
    ([1., 0., 0.], SOC(3), True, [1., 0., 0.]),
    ([0., 0.], SOC(2), False, [0., 0.]),
])
def test_projection_exact_cases(value, cone, dual, expected, dtype):
    v = torch.tensor(value, dtype=dtype)
    before = v.clone()
    projected = project_cones(v, (cone,), dual=dual)
    assert torch.allclose(projected, torch.tensor(expected, dtype=dtype))
    assert torch.equal(v, before)
    assert projected.dtype == dtype and projected.device == v.device
    assert check_cone_feasibility(projected, (cone,), dual=dual).passed
    assert torch.equal(project_cones(projected, (cone,), dual=dual), projected)


def test_product_projection_moreau_and_nonexpansive():
    generator = torch.Generator().manual_seed(81)
    cones = (SOC(3), Zero(2), Nonnegative(2), SOC(2))
    for _ in range(15):
        u, v = torch.randn((2, 9), dtype=torch.float64, generator=generator)
        pu, pv = project_cones(u, cones), project_cones(v, cones)
        negative_dual = project_cones(-u, cones, dual=True)
        assert torch.allclose(u, pu - negative_dual, atol=1e-14, rtol=1e-14)
        assert abs(float(pu @ negative_dual)) < 1e-14
        assert torch.linalg.vector_norm(pu - pv) <= torch.linalg.vector_norm(u - v) + 1e-14


def test_projection_extreme_representable_boundary_and_empty():
    for scale in (1e300, 1e-300):
        projected = project_cones(torch.tensor([0., 3 * scale, 4 * scale], dtype=torch.float64), (SOC(3),))
        assert torch.allclose(projected / scale, torch.tensor([2.5, 1.5, 2.], dtype=torch.float64),
                              atol=1e-14, rtol=1e-14)
    empty = project_cones(torch.empty(0, dtype=torch.float64), ())
    assert empty.shape == (0,)


@pytest.mark.parametrize("value,cones,kwargs", [
    (torch.tensor([True]), (Nonnegative(1),), {}),
    (torch.ones((1, 2)), (SOC(2),), {}),
    (torch.ones(2, dtype=torch.complex64), (SOC(2),), {}),
    (torch.ones(2).to_sparse(), (SOC(2),), {}),
    (torch.ones(1), (SOC(2),), {}),
    (torch.ones(2), ("soc",), {}),
    (torch.ones(2), (SOC(2),), {"dual": 1}),
    (torch.tensor([float("inf"), 0.]), (SOC(2),), {}),
    (torch.tensor([float("nan"), 0.]), (SOC(2),), {}),
    ([1., 0.], (SOC(2),), {}),
])
def test_projection_reject_malformed(value, cones, kwargs):
    with pytest.raises((TypeError, ValueError)):
        project_cones(value, cones, **kwargs)


@pytest.mark.parametrize("name", [
    "soc_lp", "mixed", "nonnegative", "equality", "redundant", "empty",
    "singular", "zero_lp", "soc_origin", "soc2",
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_analytic_same_model_clarabel(name, dtype, tmp_path, request):
    p, expected = case(name)
    config = replace(CONFIG, dtype=dtype, eps_abs=2e-5 if dtype == torch.float32 else 1e-8)
    result = solve_native_conic(p, config)
    ref = solve_conic(p, ConicConfig(backend="clarabel_reference", eps_abs=1e-10, eps_rel=0))
    assert result.status == "solved", result.evidence.to_dict()
    assert ref.status == "reference_solved", ref.evidence.to_dict()
    assert result.evidence.passed and not result.certified and result.device == "cpu"
    assert torch.allclose(result.x.double(), torch.tensor(expected, dtype=torch.float64),
                          atol=20 * config.eps_abs, rtol=0)
    assert np.allclose(host(result.x), ref.x, atol=20 * config.eps_abs, rtol=0)
    assert abs(result.objective - ref.objective) <= 20 * config.eps_abs
    assert check_conic_candidate(p, result.x, result.s, result.z,
                                 atol=config.eps_abs, rtol=config.eps_rel).passed
    save_sidecar(p, result, config, tmp_path, request, reference=ref)


@pytest.mark.parametrize("seed", [191, 192, 193])
def test_cpu_unplanted_random_same_model_reference(seed, tmp_path, request):
    rng = np.random.default_rng(seed)
    B = rng.normal(size=(5, 5))
    P = B.T @ B + np.eye(5)
    A = rng.normal(size=(9, 5))
    interior = np.array([0., 0., 3., .2, -.3, .1, 2., 1., 1.])
    feasible = rng.normal(size=5)
    p = ConicProblem(P, rng.normal(size=5), A, A @ feasible + interior,
                     (Zero(2), SOC(4), Nonnegative(3)))
    result = solve_native_conic(p, CONFIG)
    ref = solve_conic(p, ConicConfig(backend="clarabel_reference", eps_abs=1e-11, eps_rel=0))
    assert result.status == "solved", result.evidence.to_dict()
    assert ref.status == "reference_solved", ref.evidence.to_dict()
    assert np.allclose(host(result.x), ref.x, atol=2e-6, rtol=0)
    assert abs(result.objective - ref.objective) < 2e-6
    save_sidecar(p, result, CONFIG, tmp_path, request, reference=ref)


def test_prepare_frozen_reset_and_no_gradient():
    p, _ = case("mixed")
    p = replace(p, **{name: torch.tensor(getattr(p, name), requires_grad=True)
                     for name in ("P", "q", "A", "b")})
    before = frozen_problem(p)
    prepared = prepare_native_conic(p, CONFIG)
    first = prepared.solve()
    assert first.status == "solved"
    assert frozen_problem(p) == before
    assert all(getattr(p, name).grad is None for name in ("P", "q", "A", "b"))
    with torch.no_grad():
        p.q.add_(100)
    original_copy = prepared.original_problem
    original_copy.q.add_(200)
    second = prepared.solve()
    assert torch.equal(first.x, second.x) and torch.equal(first.s, second.s) and torch.equal(first.z, second.z)
    assert frozen_problem(prepared.original_problem) == before
    assert not second.x.requires_grad and not second.z.requires_grad


@pytest.mark.parametrize("kwargs", [
    {}, {"enabled": 1}, {"enabled": True, "dtype": torch.float16},
    {"enabled": True, "rho": 0}, {"enabled": True, "rho": True},
    {"enabled": True, "sigma": float("nan")}, {"enabled": True, "sigma": -1},
    {"enabled": True, "max_iterations": 0}, {"enabled": True, "max_iterations": True},
    {"enabled": True, "check_interval": 1.5}, {"enabled": True, "eps_abs": True},
    {"enabled": True, "eps_abs": float("inf")}, {"enabled": True, "eps_rel": -1},
    {"enabled": True, "eps_rel": 1}, {"enabled": True, "device": "meta"},
    {"enabled": True, "device": "cpu:0"},
    {"enabled": True, "dtype": torch.float32, "sigma": 1e-100},
    {"enabled": True, "dtype": torch.float32, "rho": 1e100},
])
def test_reject_bad_config_without_routing(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        solve_native_conic(case("mixed")[0], NativeConicConfig(**kwargs))


def test_explicit_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA requested"):
        solve_native_conic(case("soc_lp")[0], replace(CONFIG, device="cuda:0"))


def test_reject_cuda_out_of_range(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="index out of range"):
        solve_native_conic(case("soc_lp")[0], replace(CONFIG, device="cuda:99"))


@pytest.mark.parametrize("field", ["P", "q", "A", "b"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_reject_nonfinite_model(field, value):
    p, _ = case("soc_lp")
    data = getattr(p, field).copy()
    data.flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        prepare_native_conic(replace(p, **{field: data}), CONFIG)


@pytest.mark.parametrize("field,value", [
    ("P", [[2., 1., 0.], [0., 2., 0.], [0., 0., 2.]]),
    ("P", -np.eye(3)), ("P", np.eye(2)), ("q", []), ("q", [[1, 2, 3]]),
    ("A", np.zeros((5, 2))), ("b", np.zeros(4)), ("cones", (SOC(3),)),
    ("cones", ("exponential",)), ("P", np.eye(3, dtype=complex)),
    ("P", torch.eye(3).to_sparse()), ("q", [True, False, True]),
])
def test_reject_malformed_full_psd_model(field, value):
    p, _ = case("soc_lp")
    with pytest.raises((ValueError, TypeError)):
        prepare_native_conic(replace(p, **{field: value}), CONFIG)


def test_reject_psd_negative_relative_not_absolute_floor():
    p = ConicProblem(np.array([[-1e-30]]), np.zeros(1), np.empty((0, 1)), np.empty(0), ())
    with pytest.raises(ValueError, match="positive semidefinite"):
        prepare_native_conic(p, CONFIG)


@pytest.mark.parametrize("rho,sigma", [(0.1, 0.01), (1., 1.), (10., 0.2)])
def test_weighted_proximal_admm_not_model_regularization(rho, sigma):
    p, expected = case("soc_lp")
    result = solve_native_conic(p, replace(CONFIG, rho=rho, sigma=sigma))
    assert result.status == "solved", result.evidence.to_dict()
    assert np.allclose(host(result.x), expected, atol=1e-6, rtol=0)
    assert result.evidence.checks["duality_gap"]
    assert abs(result.objective - 5.) < 1e-7


def test_unknown_iteration_limits_infeasible_unbounded_and_unresolved_spectrum():
    cases = [
        ConicProblem(np.zeros((1, 1)), np.zeros(1), np.array([[1.], [-1.]]),
                     np.array([0., -1.]), (Nonnegative(2),)),
        ConicProblem(np.zeros((1, 1)), np.array([-1.]), np.array([[-1.]]),
                     np.zeros(1), (Nonnegative(1),)),
        ConicProblem(np.diag([1., 1e-16]), np.zeros(2), np.empty((0, 2)), np.empty(0), ()),
    ]
    for p in cases:
        result = solve_native_conic(p, replace(CONFIG, max_iterations=50))
        assert result.status == "max_iterations" and not result.evidence.passed
        assert result.certified is False and result.iterations == 50
    result = solve_native_conic(case("soc_lp")[0], replace(CONFIG, max_iterations=1))
    assert result.status == "max_iterations" and not result.evidence.passed


def test_no_third_party_solver_or_fallback(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native solve consulted a third-party/other solver")
    monkeypatch.setattr("openopt.conic._reference_dependencies", forbidden)
    monkeypatch.setattr("openopt.conic.solve_conic", forbidden)
    monkeypatch.setattr("openopt.qp.solve_qp", forbidden)
    monkeypatch.setattr("openopt.qp.prepare_qp", forbidden)
    assert solve_native_conic(case("soc_lp")[0], CONFIG).status == "solved"


def test_checker_denial_cannot_accept_self_reported_success(monkeypatch):
    import openopt.native_conic as module
    real = module.check_conic_candidate
    def denied(*args, **kwargs):
        evidence = real(*args, **kwargs)
        return replace(evidence, passed=False, checks={**evidence.checks, "forced_negative": False})
    monkeypatch.setattr(module, "check_conic_candidate", denied)
    result = solve_native_conic(case("zero_lp")[0], replace(CONFIG, max_iterations=5, check_interval=1))
    assert result.status == "max_iterations" and not result.evidence.passed


def test_original_not_dtype_rounded_model_cannot_accept():
    p = ConicProblem(np.eye(1), np.array([-1.00000003]), np.empty((0, 1)), np.empty(0), ())
    result = solve_native_conic(p, replace(CONFIG, dtype=torch.float32, eps_abs=1e-10, max_iterations=25))
    assert result.status == "max_iterations" and not result.evidence.checks["stationarity"]
    assert result.evidence.metrics["stationarity"] == pytest.approx(3e-8)


def test_original_python_lists_preserve_float64_before_cast():
    p = ConicProblem([[1.]], [-1.00000003], np.empty((0, 1)), [], ())
    prepared = prepare_native_conic(
        p, replace(CONFIG, dtype=torch.float32, eps_abs=1e-10, max_iterations=25),
    )
    assert prepared.original_problem.q.item() == -1.00000003
    result = prepared.solve()
    assert result.status == "max_iterations" and not result.evidence.checks["stationarity"]
    assert result.evidence.metrics["stationarity"] == pytest.approx(3e-8)


def test_arithmetic_nonfinite_fail_loud_and_preserves_cause(monkeypatch):
    import openopt.native_conic as module
    p, _ = case("mixed")
    prepared = prepare_native_conic(p, CONFIG)
    def nonfinite(*args, **kwargs):
        return torch.full((4, 1), float("nan"), dtype=torch.float64)
    monkeypatch.setattr(torch, "cholesky_solve", nonfinite)
    with pytest.raises(RuntimeError, match="nonfinite x iteration=1"):
        prepared.solve()
    cause = ValueError("deliberate independent arithmetic cause")
    def failure(*args, **kwargs):
        raise cause
    monkeypatch.undo()
    monkeypatch.setattr(module, "check_conic_candidate", failure)
    with pytest.raises(RuntimeError, match="check failed iteration=25") as raised:
        prepared.solve()
    assert raised.value.__cause__ is cause


def test_reject_factorization_failure_and_overflow(monkeypatch):
    p, _ = case("mixed")
    cause = RuntimeError("deliberate factor cause")
    def failure(*args, **kwargs):
        raise cause
    monkeypatch.setattr(torch.linalg, "cholesky_ex", failure)
    with pytest.raises(RuntimeError, match="factorization failed") as raised:
        prepare_native_conic(p, CONFIG)
    assert raised.value.__cause__ is cause
    monkeypatch.undo()
    p = ConicProblem(np.eye(1), np.zeros(1), np.array([[1e308]]), np.zeros(1), (Zero(1),))
    with pytest.raises(RuntimeError, match="nonfinite linear system"):
        prepare_native_conic(p, CONFIG)
    p = ConicProblem(np.eye(1), np.array([1e100]), np.empty((0, 1)), np.empty(0), ())
    with pytest.raises(RuntimeError, match="nonfinite device-cast"):
        prepare_native_conic(p, replace(CONFIG, dtype=torch.float32))
    p = replace(p, q=np.array([1e-100]))
    with pytest.raises(ValueError, match="underflow lost nonzero"):
        prepare_native_conic(p, replace(CONFIG, dtype=torch.float32))


@pytest.mark.parametrize("mutation", ["x", "s", "z", "problem", "objective", "evidence", "placement"])
def test_pinned_replay_reject_tamper_even_rehashed(mutation, tmp_path, request):
    p, _ = case("soc_lp")
    result = solve_native_conic(p, CONFIG)
    path = save_sidecar(p, result, CONFIG, tmp_path, request)
    saved = json.loads(path.read_text())
    if mutation in ("x", "s", "z"):
        saved[mutation][0] += 1
    elif mutation == "problem":
        saved["problem"]["q"][0] += 1
    elif mutation == "objective":
        saved["objective"] += 1
    elif mutation == "evidence":
        saved["evidence"]["metrics"]["duality_gap"] = 0.123
    else:
        saved["placements"]["x"] = "cuda:0"
    altered = tmp_path / "tampered.json"
    altered.write_text(json.dumps(saved, allow_nan=False))
    with pytest.raises(ValueError, match="rejected|disagrees|contradiction"):
        replay_sidecar(altered, hashlib.sha256(altered.read_bytes()).hexdigest())
    with pytest.raises(ValueError, match="hash mismatch"):
        replay_sidecar(altered, hashlib.sha256(path.read_bytes()).hexdigest())


def test_pinned_replay_reject_source_drift_and_duplicate_json(tmp_path, request):
    p, _ = case("nonnegative")
    path = save_sidecar(p, solve_native_conic(p, CONFIG), CONFIG, tmp_path, request)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert replay_sidecar(path, digest, ROOT)["source_checked"]
    saved = json.loads(path.read_text())
    saved["source_sha256"]["src/openopt/native_conic.py"] = "0" * 64
    changed = tmp_path / "source-drift.json"
    changed.write_text(json.dumps(saved, allow_nan=False))
    with pytest.raises(ValueError, match="source hash mismatch"):
        replay_sidecar(changed, hashlib.sha256(changed.read_bytes()).hexdigest(), ROOT)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}')
    with pytest.raises(ValueError, match="duplicate evidence key"):
        replay_sidecar(duplicate, hashlib.sha256(duplicate.read_bytes()).hexdigest())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; native conic GPU execution UNVERIFIED")
@pytest.mark.parametrize("name", [
    "soc_lp", "mixed", "nonnegative", "equality", "empty", "singular", "soc_origin", "soc2",
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_native_conic_frozen_original_model(name, dtype, tmp_path, request):
    p, expected = case(name)
    gpu_problem = replace(p, **{
        field: torch.tensor(getattr(p, field), device="cuda:0", dtype=torch.float64)
        for field in ("P", "q", "A", "b")
    })
    before = frozen_problem(gpu_problem)
    config = replace(CONFIG, device="cuda:0", dtype=dtype,
                     eps_abs=3e-5 if dtype == torch.float32 else 1e-8)
    prepared = prepare_native_conic(gpu_problem, config)
    assert all(getattr(prepared, field).device == torch.device("cuda:0")
               for field in ("_P", "_q", "_A", "_b", "_factor"))
    result = prepared.solve()
    torch.cuda.synchronize("cuda:0")
    assert result.status == "solved" and result.evidence.passed, result.evidence.to_dict()
    for field in ("x", "s", "z"):
        assert getattr(result, field).device == torch.device("cuda:0")
        assert getattr(result, field).dtype == dtype
    assert np.allclose(host(result.x), expected, atol=20 * config.eps_abs, rtol=0)
    assert frozen_problem(gpu_problem) == before
    save_sidecar(p, result, config, tmp_path, request)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; native conic GPU negative controls UNVERIFIED")
@pytest.mark.parametrize("mutation", ["x", "s", "z"])
def test_cuda_native_conic_reject_numerical_tamper(mutation, tmp_path, request):
    p, _ = case("soc_lp")
    config = replace(CONFIG, device="cuda:0")
    result = solve_native_conic(p, config)
    assert result.status == "solved" and result.evidence.passed
    path = save_sidecar(p, result, config, tmp_path, request)
    saved = json.loads(path.read_text())
    saved[mutation][0] += 1
    bad = tmp_path / "tampered.json"
    bad.write_text(json.dumps(saved, allow_nan=False))
    with pytest.raises(ValueError, match="candidate rejected"):
        replay_sidecar(bad, hashlib.sha256(bad.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.artifact_dir.resolve()
    if not out.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve()) or out.is_relative_to(ROOT):
        raise ValueError("replay artifacts must be external shared 3FS tmp")
    out.mkdir(parents=True, exist_ok=False)
    rc = 0
    try:
        report = replay_sidecar(args.replay, args.sha256, args.source_root)
    except Exception as error:
        rc = 1
        report = {"verdict": "FAIL", "checked_at": datetime.now(timezone.utc).isoformat(),
                  "error": f"{type(error).__name__}: {error}"}
        (out / "failure.log").write_text(traceback.format_exc())
    (out / "acceptance.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    line = f"NATIVE_CONIC_REPLAY verdict={report['verdict']} at={report['checked_at']} report={out / 'acceptance.json'}"
    (out / "execution.log").write_text(line + f"\nreturncode={rc}\n")
    print(line)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
