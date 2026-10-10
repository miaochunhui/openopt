"""Reversible scaling, planted KKT solutions and independent negative controls."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from openopt.certificates import check_qp_candidate
from openopt.qp import QPConfig, QPProblem, QPResult, solve_qp
from openopt.scaling import (
    DiagonalScaling, ScalingConfig, equilibrate_qp, solve_scaled_qp,
)


def qp(P, q, A, l, u):
    return QPProblem(*[
        torch.tensor(value, dtype=torch.float64) for value in (P, q, A, l, u)
    ])


def planted():
    P = np.array([[4., 1.], [1., 2.]])
    A = np.array([[2., 0.], [0., 3.], [1., 1.], [0., 0.]])
    x, y = np.array([1., -0.5]), np.array([2., 0., -1., 0.])
    Ax = A @ x
    p = qp(
        P, -P @ x - A.T @ y, A,
        [-np.inf, Ax[1] - 1, Ax[2], -np.inf], [Ax[0], np.inf, Ax[2], np.inf],
    )
    return p, torch.tensor(x), torch.tensor(y)


def ill_conditioned():
    # A fixed, known KKT QP represented with disparate variable/row units.
    base, x, y = planted()
    units = DiagonalScaling(
        torch.tensor([1e-3, 1e3], dtype=torch.float64),
        torch.tensor([1e3, 1e-3, 1., 1.], dtype=torch.float64),
    )
    p = units.apply(base)
    x, y = units.to_scaled(x, y)
    return p, x, y


def reference(p):
    import osqp
    from scipy import sparse

    solver = osqp.OSQP()
    solver.setup(
        P=sparse.csc_matrix(p.P.numpy()), q=p.q.numpy(), A=sparse.csc_matrix(p.A.numpy()),
        l=p.l.numpy(), u=p.u.numpy(), eps_abs=1e-10, eps_rel=1e-10,
        max_iter=100000, polishing=True, verbose=False,
    )
    result = solver.solve(raise_error=True)
    assert result.info.status == "solved"
    return result


def strict_config(**kwargs):
    return QPConfig(eps_abs=1e-10, eps_rel=1e-10, max_iterations=20000, **kwargs)


def ill_config():
    return strict_config(
        adaptive_rho=True, adaptive_rho_interval=25, adaptive_rho_tolerance=2.,
        rho_min=1e-10,
    )


@pytest.mark.parametrize("objective_scaling", [False, True])
@pytest.mark.parametrize("rho_mode", ["fixed", "adaptive"])
def test_original_ill_conditioned_rho_cost_ablation(rho_mode, objective_scaling, tmp_path, request):
    p, known_x, known_y = ill_conditioned()
    config = strict_config() if rho_mode == "fixed" else ill_config()
    prepared = equilibrate_qp(p, config, ScalingConfig(iterations=30, objective_scaling=objective_scaling))
    result = solve_scaled_qp(p, config, ScalingConfig(iterations=30, objective_scaling=objective_scaling))
    ref = reference(p)
    ref_evidence = check_qp_candidate(p, ref.x, ref.y, atol=1e-7, rtol=1e-7)
    assert check_qp_candidate(p, known_x, known_y).passed and ref_evidence.passed
    assert result.status != "solved" or result.evidence.passed
    report = {
        "schema_version": 1, "kind": "original_ill_conditioned_ablation",
        "rho_mode": rho_mode, "objective_scaling": objective_scaling,
        "status": result.status, "iterations": result.iterations,
        "rho_updates": result.scaled_result.rho_updates,
        "final_rho": result.scaled_result.final_rho,
        "original_evidence": result.evidence.to_dict(),
        "osqp_evidence": ref_evidence.to_dict(),
        "D": result.scaling.variable_scale.tolist(),
        "E": result.scaling.constraint_scale.tolist(), "c": result.scaling.objective_scale,
        "scaled_P": prepared.scaled_problem.P.tolist(),
        "scaled_A": prepared.scaled_problem.A.tolist(),
        "scaled_q": prepared.scaled_problem.q.tolist(),
        "scaled_condition_P": float(torch.linalg.cond(prepared.scaled_problem.P)),
        "x": result.x.tolist(), "y": result.y.tolist(),
        "known_x": known_x.tolist(), "known_y": known_y.tolist(),
        "qualification": "diagnostic measurement; passing test does not mean every ablation solved",
    }
    artifact = tmp_path / "rho-cost-ablation.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    request.node.user_properties.append(("rho_cost_ablation_path", str(artifact)))
    if rho_mode == "adaptive":
        assert result.status == "solved" and result.evidence.passed, report


def test_manual_transform_equations_roundtrip_objective_and_kkt():
    p, x, y = planted()
    d, e = torch.tensor([0.2, 5.], dtype=torch.float64), torch.tensor([3., 0.1, 2., 1.], dtype=torch.float64)
    transform = DiagonalScaling(d, e, 0.25)
    scaled = transform.apply(p)
    xhat, yhat = transform.to_scaled(x, y)
    actual_x, actual_y = transform.restore(xhat, yhat)
    assert torch.allclose(actual_x, x, atol=1e-15, rtol=1e-15)
    assert torch.allclose(actual_y, y, atol=1e-15, rtol=1e-15)
    assert torch.allclose(scaled.P, 0.25 * d[:, None] * p.P * d[None, :])
    assert torch.allclose(scaled.q, 0.25 * d * p.q)
    assert torch.allclose(scaled.A, e[:, None] * p.A * d[None, :])
    assert torch.equal(torch.isinf(scaled.l), torch.isinf(p.l))
    assert torch.equal(torch.isinf(scaled.u), torch.isinf(p.u))
    assert check_qp_candidate(p, actual_x, actual_y).passed
    assert check_qp_candidate(scaled, xhat, yhat).passed
    original_objective = 0.5 * x @ p.P @ x + p.q @ x
    scaled_objective = 0.5 * xhat @ scaled.P @ xhat + scaled.q @ xhat
    assert float(scaled_objective) == pytest.approx(0.25 * float(original_objective))
    wrong = yhat / e * 0.25
    assert not check_qp_candidate(p, actual_x, wrong).passed
    d.add_(10)
    e.add_(10)
    assert torch.allclose(transform.restore(xhat, yhat)[0], x)


@pytest.mark.parametrize("objective_scaling", [False, True])
@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
def test_known_kkt_and_original_coordinates_against_osqp(objective_scaling, linear_solver):
    p, expected_x, expected_y = planted()
    result = solve_scaled_qp(
        p, strict_config(linear_solver=linear_solver),
        ScalingConfig(objective_scaling=objective_scaling),
    )
    ref = reference(p)
    assert result.status == result.scaled_result.status == "solved"
    assert result.evidence.passed
    assert torch.allclose(result.x, expected_x, atol=1e-7, rtol=1e-7)
    assert torch.allclose(result.y, expected_y, atol=1e-7, rtol=1e-7)
    assert np.allclose(result.x.numpy(), ref.x, atol=1e-7, rtol=1e-7)
    assert np.allclose(result.y.numpy(), ref.y, atol=1e-7, rtol=1e-7)
    assert result.objective == pytest.approx(ref.info.obj_val, abs=1e-7)
    assert result.iterations == result.scaled_result.iterations
    if not objective_scaling:
        assert result.scaling.objective_scale == 1


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_none_keeps_dense_path_results_and_does_not_mutate_inputs(dtype):
    p, _, _ = planted()
    copies = [value.clone() for value in (p.P, p.q, p.A, p.l, p.u)]
    c = QPConfig(dtype=dtype, eps_abs=1e-6, eps_rel=1e-6)
    ordinary = solve_qp(p, c)
    result = solve_scaled_qp(p, c, ScalingConfig(method="none", check_atol=1e-4, check_rtol=1e-4))
    assert result.status == ordinary.status == "solved"
    assert torch.equal(ordinary.x, result.x) and torch.equal(ordinary.y, result.y)
    assert result.evidence.passed
    assert result.x.dtype == result.y.dtype == dtype
    assert all(torch.equal(a, b) for a, b in zip(copies, (p.P, p.q, p.A, p.l, p.u)))
    prepared = equilibrate_qp(p, c, ScalingConfig(method="none"))
    assert prepared.iterations == 0
    assert bool((prepared.scaling.variable_scale == 1).all())
    assert bool((prepared.scaling.constraint_scale == 1).all())
    p.q.add_(100)
    assert torch.equal(prepared.original_problem.q, copies[1])


@pytest.mark.parametrize("kind", ["unconstrained", "lp", "zero_column"])
def test_empty_rows_semidefinite_lp_and_structural_zeros(kind):
    if kind == "unconstrained":
        p = qp(np.diag([1e-3, 1e3]), [-2e-3, 3e3], np.empty((0, 2)), [], [])
        expected = torch.tensor([2., -3.], dtype=torch.float64)
    elif kind == "lp":
        p = qp(np.zeros((2, 2)), [-1., 2.], np.eye(2), [0, 0], [1, 1])
        expected = torch.tensor([1., 0.], dtype=torch.float64)
    else:
        p = qp(np.diag([2., 0.]), [-4., 0.], [[1., 0.], [0., 0.]], [-np.inf, -1], [5, 1])
        expected = torch.tensor([2., 0.], dtype=torch.float64)
    result = solve_scaled_qp(p, strict_config())
    assert result.status == "solved" and result.evidence.passed
    assert torch.allclose(result.x, expected, atol=1e-7, rtol=1e-7)
    if kind == "zero_column":
        assert result.scaling.variable_scale[1] == result.scaling.constraint_scale[1] == 1
    if kind == "unconstrained":
        assert result.y.numel() == result.scaling.constraint_scale.numel() == 0


def test_ruiz_factors_bound_and_match_explicit_transform():
    p, _, _ = ill_conditioned()
    scaling = ScalingConfig(iterations=20, scale_min=1e-2, scale_max=1e2)
    prepared = equilibrate_qp(p, scaling=scaling)
    t = prepared.scaling
    for factors in (t.variable_scale, t.constraint_scale, torch.tensor([t.objective_scale])):
        assert bool((factors >= scaling.scale_min).all() & (factors <= scaling.scale_max).all())
    explicit = t.apply(p)
    for name in ("P", "q", "A", "l", "u"):
        assert torch.allclose(getattr(explicit, name), getattr(prepared.scaled_problem, name), atol=1e-12, rtol=1e-12)
    assert prepared.iterations == scaling.iterations


def test_ill_conditioned_restoration_and_original_osqp_candidate():
    p, expected_x, expected_y = ill_conditioned()
    result = solve_scaled_qp(p, ill_config(), ScalingConfig(iterations=30))
    ref = reference(p)
    ref_evidence = check_qp_candidate(p, ref.x, ref.y, atol=1e-7, rtol=1e-7)
    assert result.status == "solved" and result.evidence.passed and ref_evidence.passed
    assert torch.allclose(result.x, expected_x, atol=1e-6, rtol=1e-7)
    assert torch.allclose(result.y, expected_y, atol=1e-7, rtol=1e-7)
    assert np.allclose(result.x.numpy(), ref.x, atol=1e-6, rtol=1e-7)
    assert np.allclose(result.y.numpy(), ref.y, atol=1e-7, rtol=1e-7)
    assert result.objective == pytest.approx(ref.info.obj_val, abs=1e-7)


def test_extreme_row_curvature_competition_retains_iteration_limit():
    base, x, y = planted()
    units = DiagonalScaling(
        torch.tensor([1e-3, 1e3], dtype=torch.float64),
        torch.tensor([1e3, 1e-3, 1., 1.], dtype=torch.float64),
    )
    p = units.apply(base)
    known_x, known_y = units.to_scaled(x, y)
    assert check_qp_candidate(p, known_x, known_y).passed
    result = solve_scaled_qp(
        p, QPConfig(max_iterations=2000, eps_abs=1e-10, eps_rel=1e-10),
        ScalingConfig(iterations=30),
    )
    assert result.status == result.scaled_result.status == "max_iterations"
    assert not result.evidence.passed


@pytest.mark.parametrize("mutation", [
    "P_dims", "q_dims", "A_dims", "l_dims", "u_dims", "empty",
    "non_tensor", "complex", "sparse", "indefinite", "asymmetric",
])
def test_bad_problem_dimensions_types_and_convexity(mutation):
    p, _, _ = planted()
    changes = {
        "P_dims": {"P": p.P[:1]}, "q_dims": {"q": p.q[None]},
        "A_dims": {"A": p.A[:, :1]}, "l_dims": {"l": p.l[:1]},
        "u_dims": {"u": p.u[:, None]}, "empty": {"q": torch.empty(0)},
        "non_tensor": {"q": [1, 2]}, "complex": {"P": p.P.to(torch.complex128)},
        "sparse": {"A": p.A.to_sparse()}, "indefinite": {"P": -p.P},
        "asymmetric": {"P": torch.tensor([[4., 9.], [1., 2.]])},
    }
    with pytest.raises((ValueError, TypeError)):
        solve_scaled_qp(replace(p, **changes[mutation]))


@pytest.mark.parametrize("name", ["P", "q", "A"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_model_rejected(name, value):
    p, _, _ = planted()
    getattr(p, name).reshape(-1)[0] = value
    with pytest.raises(ValueError, match="finite"):
        solve_scaled_qp(p)


@pytest.mark.parametrize("mutation", ["nan_l", "nan_u", "positive_inf_l", "negative_inf_u", "reversed"])
def test_invalid_bounds_rejected(mutation):
    p, _, _ = planted()
    if mutation == "nan_l":
        p.l[0] = np.nan
    elif mutation == "nan_u":
        p.u[0] = np.nan
    elif mutation == "positive_inf_l":
        p.l[0] = np.inf
    elif mutation == "negative_inf_u":
        p.u[0] = -np.inf
    else:
        p.l[0] = p.u[0] + 1
    with pytest.raises(ValueError):
        solve_scaled_qp(p)


@pytest.mark.parametrize("kwargs", [
    {"method": "invalid"}, {"iterations": 0}, {"iterations": -1}, {"iterations": True},
    {"iterations": 1.5}, {"scale_min": 0}, {"scale_min": -1}, {"scale_max": -1},
    {"scale_min": np.nan}, {"scale_max": np.inf}, {"scale_min": 2},
    {"scale_max": 0.5}, {"scale_max": True}, {"objective_scaling": 1},
    {"check_atol": 0}, {"check_atol": np.inf}, {"check_rtol": -1},
    {"check_rtol": 1}, {"check_rtol": np.nan}, {"check_rtol": True},
    {"scale_min": 1e-320},
])
def test_bad_scaling_config_and_extreme_reciprocals(kwargs):
    p, _, _ = planted()
    with pytest.raises(ValueError):
        solve_scaled_qp(p, scaling=ScalingConfig(**kwargs))


@pytest.mark.parametrize("name", ["variable_scale", "constraint_scale", "objective_scale"])
@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf, 1e-320])
def test_manual_factors_must_be_positive_finite_reversible(name, value):
    kwargs = {
        "variable_scale": torch.ones(2, dtype=torch.float64),
        "constraint_scale": torch.ones(4, dtype=torch.float64),
        "objective_scale": 1.,
    }
    kwargs[name] = value if name == "objective_scale" else torch.full_like(kwargs[name], value)
    with pytest.raises(ValueError):
        DiagonalScaling(**kwargs)


@pytest.mark.parametrize("mutation", ["matrix_factor", "empty_d", "dtype", "problem_dims", "candidate_dims", "candidate_nonfinite", "candidate_dtype", "mutated_factor"])
def test_manual_transform_bad_dims_and_candidates(mutation):
    p, x, y = planted()
    with pytest.raises(ValueError):
        if mutation == "matrix_factor":
            DiagonalScaling(torch.ones((1, 2)), torch.ones(4))
        elif mutation == "empty_d":
            DiagonalScaling(torch.empty(0), torch.ones(4))
        elif mutation == "dtype":
            DiagonalScaling(torch.ones(2, dtype=torch.float32), torch.ones(4, dtype=torch.float64))
        else:
            transform = DiagonalScaling(torch.ones_like(x), torch.ones_like(y))
            if mutation == "problem_dims":
                transform.apply(replace(p, A=p.A[:1], l=p.l[:1], u=p.u[:1]))
            elif mutation == "candidate_dims":
                transform.restore(x[:1], y)
            elif mutation == "candidate_nonfinite":
                transform.restore(x * np.inf, y)
            elif mutation == "candidate_dtype":
                transform.restore(x.float(), y)
            else:
                transform.variable_scale[0] = 0
                transform.restore(x, y)


@pytest.mark.parametrize("kind", ["P_overflow", "bound_overflow", "matrix_underflow", "restore_overflow", "restore_underflow", "dtype_overflow", "dtype_underflow"])
def test_extreme_arithmetic_fails_loud_without_losing_nonzero_or_finite_bounds(kind):
    p = qp([[1.]], [0.], [[1.]], [0.], [1.])
    with pytest.raises(ValueError, match="nonfinite|underflow"):
        if kind == "P_overflow":
            DiagonalScaling(torch.tensor([1e200], dtype=torch.float64), torch.ones(1, dtype=torch.float64)).apply(p)
        elif kind == "bound_overflow":
            p = replace(p, u=torch.tensor([1e200], dtype=torch.float64))
            DiagonalScaling(torch.ones(1, dtype=torch.float64), torch.tensor([1e200], dtype=torch.float64)).apply(p)
        elif kind == "matrix_underflow":
            p = replace(p, P=torch.tensor([[1e-200]], dtype=torch.float64))
            DiagonalScaling(torch.tensor([1e-100], dtype=torch.float64), torch.ones(1, dtype=torch.float64)).apply(p)
        elif kind == "restore_overflow":
            DiagonalScaling(torch.tensor([1e200], dtype=torch.float64), torch.ones(1, dtype=torch.float64)).restore(
                torch.tensor([1e200], dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            )
        elif kind == "restore_underflow":
            DiagonalScaling(torch.tensor([1e-200], dtype=torch.float64), torch.ones(1, dtype=torch.float64)).restore(
                torch.tensor([1e-200], dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            )
        elif kind == "dtype_overflow":
            solve_scaled_qp(replace(p, u=torch.tensor([1e100], dtype=torch.float64)), QPConfig(dtype=torch.float32))
        else:
            solve_scaled_qp(replace(p, q=torch.tensor([1e-100], dtype=torch.float64)), QPConfig(dtype=torch.float32))


def test_float32_scaling_limits_must_be_representable():
    p, _, _ = planted()
    for kwargs in ({"scale_max": 1e100}, {"scale_min": 1e-100}):
        with pytest.raises(ValueError, match="finite|positive"):
            solve_scaled_qp(p, QPConfig(dtype=torch.float32), ScalingConfig(**kwargs))


def test_representable_large_factors_are_not_rejected_by_arbitrary_magnitude():
    transform = DiagonalScaling(
        torch.tensor([1e100], dtype=torch.float64),
        torch.tensor([1e-100], dtype=torch.float64), 1e100,
    )
    p = qp([[1e-200]], [1e-100], [[1.]], [-np.inf], [1e100])
    scaled = transform.apply(p)
    assert bool(torch.isfinite(scaled.P).all() & torch.isfinite(scaled.q).all())
    x, y = torch.tensor([1e-100], dtype=torch.float64), torch.tensor([1e100], dtype=torch.float64)
    restored_x, restored_y = transform.restore(x, y)
    assert restored_x.item() == pytest.approx(1.)
    assert restored_y.item() == pytest.approx(1e-100)


def test_explicit_cuda_never_falls_back(monkeypatch):
    p, _, _ = planted()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA"):
        solve_scaled_qp(p, QPConfig(device="cuda:0"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; original-coordinate GPU restoration is UNVERIFIED")
@pytest.mark.parametrize("kind,dtype,tolerance", [
    ("known_kkt", torch.float64, 1e-7),
    ("known_kkt", torch.float32, 2e-4),
    ("ill_conditioned", torch.float64, 1e-7),
])
def test_cuda_original_coordinate_restoration_kkt_only(kind, dtype, tolerance, tmp_path, request):
    p, expected_x, expected_y = planted() if kind == "known_kkt" else ill_conditioned()
    gpu_problem = QPProblem(*[
        getattr(p, name).to("cuda:0") for name in ("P", "q", "A", "l", "u")
    ])
    copies = [getattr(gpu_problem, name).clone() for name in ("P", "q", "A", "l", "u")]
    eps = 1e-10 if dtype == torch.float64 else 1e-6
    c = QPConfig(
        device="cuda:0", dtype=dtype, eps_abs=eps, eps_rel=eps, max_iterations=20000,
        adaptive_rho=kind == "ill_conditioned", adaptive_rho_interval=25,
        adaptive_rho_tolerance=2., rho_min=1e-10,
    )
    result = solve_scaled_qp(
        gpu_problem, c, ScalingConfig(iterations=30, check_atol=tolerance, check_rtol=tolerance),
    )
    torch.cuda.synchronize("cuda:0")
    assert result.status == result.scaled_result.status == "solved" and result.evidence.passed
    for value in (
        result.x, result.y, result.scaled_result.x, result.scaled_result.y,
        result.scaling.variable_scale, result.scaling.constraint_scale,
    ):
        assert value.device == torch.device("cuda:0") and value.dtype == dtype
    x, y = result.x.cpu().double(), result.y.cpu().double()
    assert torch.allclose(x, expected_x, atol=10 * tolerance, rtol=tolerance)
    assert torch.allclose(y, expected_y, atol=10 * tolerance, rtol=tolerance)
    xhat, yhat = result.scaling.to_scaled(result.x, result.y)
    assert torch.allclose(xhat, result.scaled_result.x, atol=tolerance, rtol=tolerance)
    assert torch.allclose(yhat, result.scaled_result.y, atol=tolerance, rtol=tolerance)
    independent = check_qp_candidate(p, x, y, atol=tolerance, rtol=tolerance)
    assert independent.passed
    known_evidence = check_qp_candidate(p, expected_x, expected_y, atol=tolerance, rtol=tolerance)
    assert known_evidence.passed
    assert result.objective == pytest.approx(
        known_evidence.metrics["primal_objective"], abs=10 * tolerance,
    )
    assert not check_qp_candidate(p, x, -y, atol=tolerance, rtol=tolerance).passed
    assert all(torch.equal(before, getattr(gpu_problem, name)) for before, name in zip(
        copies, ("P", "q", "A", "l", "u"),
    ))
    report = {
        "schema_version": 1, "verdict": "PASS", "case": kind, "dtype": str(dtype),
        "device": str(result.x.device), "gpu_name": torch.cuda.get_device_name(0),
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "original-coordinate numerical CUDA evidence; no speed claim",
        "x": x.tolist(), "y": y.tolist(), "iterations": result.iterations,
        "original_evidence": independent.to_dict(), "known_evidence": known_evidence.to_dict(),
        "oracle": "independent CPU float64 KKT/duality gap; OSQP/SciPy not imported",
    }
    artifact = tmp_path / "cuda-scaling-acceptance.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(artifact.read_text()) == report
    request.node.user_properties.append(("cuda_scaling_acceptance_path", str(artifact)))


def test_scaled_success_is_rejected_when_original_coordinates_fail(monkeypatch):
    import openopt.scaling as module

    p = qp([[1.]], [0.], [[1e12]], [0.], [0.])
    c = QPConfig(eps_abs=1e-6, eps_rel=1e-6)
    s = ScalingConfig(objective_scaling=False, check_atol=1e-8, check_rtol=1e-8)
    prepared = equilibrate_qp(p, c, s)
    bad_x = torch.tensor([1e-15], dtype=torch.float64)
    bad_y = -(p.P @ bad_x + p.q) / 1e12
    xhat, yhat = prepared.scaling.to_scaled(bad_x, bad_y)
    assert check_qp_candidate(prepared.scaled_problem, xhat, yhat, atol=1e-6, rtol=1e-6).passed
    assert not check_qp_candidate(p, bad_x, bad_y, atol=1e-8, rtol=1e-8).passed
    raw = QPResult(xhat, yhat, "solved", 25, 0., 0., 0., 0.)
    monkeypatch.setattr(module, "solve_qp", lambda *args: raw)
    result = module.solve_scaled_qp(p, c, s)
    assert result.status == "candidate_rejected"
    assert result.scaled_result.status == "solved" and result.scaled_result.primal_residual == 0
    assert not result.evidence.passed
    assert not result.evidence.checks["primal_feasibility"]
    assert result.objective != raw.objective


def test_iteration_limit_is_not_promoted_even_for_valid_kkt_candidate(monkeypatch):
    import openopt.scaling as module

    p, x, y = planted()
    prepared = equilibrate_qp(p)
    xhat, yhat = prepared.scaling.to_scaled(x, y)
    raw = QPResult(xhat, yhat, "max_iterations", 1, 999., 999., 999., 999.)
    monkeypatch.setattr(module, "solve_qp", lambda *args: raw)
    result = module.solve_scaled_qp(p)
    assert result.evidence.passed
    assert result.status == "max_iterations" and result.objective != raw.objective


def test_original_checker_uses_input_not_dtype_rounded_model(monkeypatch):
    import openopt.scaling as module

    p = qp([[1.]], [-1.00000003], np.empty((0, 1)), [], [])
    raw = QPResult(torch.ones(1, dtype=torch.float32), torch.empty(0), "solved", 1, 0., 0., 0., 0.)
    monkeypatch.setattr(module, "solve_qp", lambda *args: raw)
    result = solve_scaled_qp(
        p, QPConfig(dtype=torch.float32),
        ScalingConfig(method="none", check_atol=1e-10, check_rtol=0),
    )
    assert result.status == "candidate_rejected"
    assert result.evidence.metrics["stationarity"] == pytest.approx(3e-8)


def test_independent_check_does_not_consult_solver_again(monkeypatch):
    import openopt.qp as solver

    def forbidden(*args, **kwargs):
        raise AssertionError("independent check must not call a solver")

    monkeypatch.setattr(solver, "solve_qp", forbidden)
    monkeypatch.setattr(solver, "prepare_qp", forbidden)
    p, x, y = planted()
    prepared = equilibrate_qp(p)
    xhat, yhat = prepared.scaling.to_scaled(x, y)
    x, y = prepared.scaling.restore(xhat, yhat)
    assert check_qp_candidate(p, x, y).passed


def test_machine_readable_original_coordinate_acceptance(tmp_path, request):
    root = Path(__file__).resolve().parents[1]
    cases = {}
    for name, factory in (("known_kkt", planted), ("ill_conditioned", ill_conditioned)):
        p, x, y = factory()
        config = ill_config() if name == "ill_conditioned" else strict_config()
        result = solve_scaled_qp(p, config, ScalingConfig(iterations=30))
        ref = reference(p)
        ref_evidence = check_qp_candidate(p, ref.x, ref.y, atol=1e-7, rtol=1e-7)
        wrong_y = check_qp_candidate(p, result.x, -result.y, atol=1e-7, rtol=1e-7)
        wrong_x = check_qp_candidate(p, result.x + 1, result.y, atol=1e-7, rtol=1e-7)
        assert result.status == "solved" and result.evidence.passed and ref_evidence.passed
        assert not wrong_y.passed and not wrong_x.passed
        cases[name] = {
            "status": result.status, "scaled_status": result.scaled_result.status,
            "iterations": result.iterations, "x": result.x.tolist(), "y": result.y.tolist(),
            "known_x": x.tolist(), "known_y": y.tolist(),
            "scaling": {
                "D": result.scaling.variable_scale.tolist(),
                "E": result.scaling.constraint_scale.tolist(),
                "c": result.scaling.objective_scale,
            },
            "original_evidence": result.evidence.to_dict(),
            "osqp_evidence": ref_evidence.to_dict(),
            "negative_wrong_y": wrong_y.to_dict(), "negative_wrong_x": wrong_x.to_dict(),
        }
    paths = [
        "src/openopt/scaling.py", "tests/test_qp_scaling.py", "docs/research/scaling.md",
        "src/openopt/qp.py", "src/openopt/certificates.py", "src/openopt/__init__.py",
    ]
    report = {
        "schema_version": 1, "verdict": "PASS", "owner": "scaling", "device": "cpu",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "qualification": "original-coordinate numerical evidence; no exact proof or speed claim",
        "case_count": len(cases), "negative_control_count": 2 * len(cases),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
        },
        "cases": cases,
    }
    artifact = tmp_path / "scaling-acceptance.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(artifact.read_text()) == report
    request.node.user_properties.append(("scaling_acceptance_path", str(artifact)))
