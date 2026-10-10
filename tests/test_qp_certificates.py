"""Independent numerical QP evidence, planted witnesses and mutation controls."""

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import osqp
import pytest
from scipy import sparse
import torch

from openopt.qp import QPConfig, QPProblem, solve_qp
from openopt.certificates import (
    REPRODUCIBLE_ARITHMETIC, check_primal_infeasibility, check_qp_candidate,
    check_unbounded_direction, reproducible_checker_contract,
)


def qp(P, q, A, l, u):
    return QPProblem(*[
        torch.tensor(value, dtype=torch.float64) for value in (P, q, A, l, u)
    ])


def bounded_lp():
    return qp(np.zeros((2, 2)), [-1, 2], np.eye(2), [0, 0], [1, 1])


def infeasible():
    return qp([[1]], [0], [[1], [1]], [-np.inf, 2], [1, np.inf])


def unbounded():
    return qp(
        np.diag([2, 0]), [0, -1],
        [[1, 0], [0, 1], [0, -1], [0, 2]],
        [0, 0, -np.inf, -np.inf], [0, np.inf, 0, np.inf],
    )


def test_planted_dense_original_coordinates_and_osqp():
    rng = np.random.default_rng(710)
    B = rng.normal(size=(5, 5))
    P = B.T @ B + np.eye(5)
    A = rng.normal(size=(4, 5)) * np.array([1e-3, 1, 20, 100])[:, None]
    x, y = rng.normal(size=5), np.array([2., -1., 0., -0.5])
    Ax = A @ x
    l, u = Ax - 1, Ax + 1
    u[0], l[1], l[3], u[3] = Ax[0], Ax[1], Ax[3], Ax[3]
    p = qp(P, -P @ x - A.T @ y, A, l, u)
    evidence = check_qp_candidate(p, x, y)
    assert evidence.passed and all(evidence.checks.values())
    assert abs(evidence.metrics["duality_gap"]) < 1e-9
    ref = osqp.OSQP()
    ref.setup(
        P=sparse.csc_matrix(P), q=p.q.numpy(), A=sparse.csc_matrix(A),
        l=l, u=u, eps_abs=1e-10, eps_rel=1e-10, verbose=False, polishing=True,
    )
    result = ref.solve(raise_error=True)
    assert result.info.status == "solved"
    independent = check_qp_candidate(p, result.x, result.y, atol=1e-7, rtol=1e-7)
    assert independent.passed
    assert abs(independent.metrics["primal_objective"] - result.info.obj_val) < 1e-8


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cpu_tensor_candidates_no_solver_status_change_or_input_mutation(dtype):
    p = bounded_lp()
    result = solve_qp(p, QPConfig(dtype=dtype))
    copies = [value.clone() for value in (p.P, p.q, p.A, p.l, p.u, result.x, result.y)]
    original_status = result.status
    evidence = check_qp_candidate(p, result.x, result.y, atol=1e-5, rtol=1e-5)
    assert evidence.passed
    assert result.status == original_status == "solved"
    assert all(torch.equal(a, b) for a, b in zip(
        copies, (p.P, p.q, p.A, p.l, p.u, result.x, result.y),
    ))
    assert "exact proof" in evidence.qualification
    json.dumps(evidence.to_dict(), allow_nan=False)


def test_singular_psd_dual_range_and_nullspace_invariance():
    p = qp(np.diag([2, 0]), [-4, 0], np.empty((0, 2)), [], [])
    first = check_qp_candidate(p, [2, 7], [])
    second = check_qp_candidate(p, [2, -200], [])
    assert first.passed and second.passed
    assert first.metrics["spectral_rank"] == 1
    assert first.metrics["dual_objective"] == pytest.approx(-4)
    assert first.metrics["range_residual"] == 0
    assert first.metrics["duality_gap"] == second.metrics["duality_gap"] == 0
    bad = check_qp_candidate(replace(p, q=torch.tensor([-4., 1.])), [2, 0], [])
    assert not bad.passed
    assert bad.metrics["dual_objective"] is None
    assert bad.metrics["duality_gap"] is None
    assert bad.metrics["range_residual"] == 1
    assert "range(P)" in bad.unavailable["dual_objective"]
    json.dumps(bad.to_dict(), allow_nan=False)


def test_rotated_singular_psd_candidate_and_recession():
    p = qp([[1, 1], [1, 1]], [-2, -2], np.empty((0, 2)), [], [])
    evidence = check_qp_candidate(p, [3, -1], [])
    assert evidence.passed and evidence.metrics["spectral_rank"] == 1
    assert evidence.metrics["dual_objective"] == pytest.approx(-2)
    p = replace(p, q=torch.tensor([-1., 1.], dtype=torch.float64))
    assert check_unbounded_direction(p, [1, -1], feasible_x=[0, 0]).passed


def test_independence_from_solver_and_inverse_or_regularization(monkeypatch):
    import openopt.qp as solver

    def forbidden(*args, **kwargs):
        raise AssertionError("independent evidence must not call a solver or inverse")

    monkeypatch.setattr(solver, "solve_qp", forbidden)
    monkeypatch.setattr(solver, "prepare_qp", forbidden)
    monkeypatch.setattr(solver.PreparedQP, "__init__", forbidden)
    for name in ("inv", "pinv", "solve", "cholesky"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    assert check_qp_candidate(bounded_lp(), [1, 0], [1, -2]).passed
    assert check_primal_infeasibility(infeasible(), [1, -1]).passed
    assert check_unbounded_direction(unbounded(), [0, 1], feasible_x=[0, 0]).passed


def test_lp_dual_requires_zero_linear_lagrangian():
    p = bounded_lp()
    good = check_qp_candidate(p, [1, 0], [1, -2])
    assert good.passed
    assert good.metrics["dual_objective"] == good.metrics["primal_objective"] == -1
    bad = check_qp_candidate(p, [1, 0], [0, 0])
    assert not bad.passed and bad.metrics["dual_objective"] is None
    assert bad.metrics["range_residual"] == 2


def test_unconstrained_and_fully_free_rows_do_not_multiply_zero_infinity():
    p = qp([[2]], [-4], [[1]], [-np.inf], [np.inf])
    good = check_qp_candidate(p, [2], [0])
    assert good.passed and good.metrics["duality_gap"] == 0
    for y in ([1], [-1], [1e-300]):
        bad = check_qp_candidate(p, [2], y)
        assert not bad.passed and not bad.checks["bound_sign"]
        assert bad.metrics["dual_objective"] is None
        json.dumps(bad.to_dict(), allow_nan=False)
    empty = qp([[2]], [-4], np.empty((0, 1)), [], [])
    assert check_qp_candidate(empty, [2], []).passed


@pytest.mark.parametrize("upper", [False, True])
def test_one_sided_multiplier_sign(upper):
    p = qp(
        [[1]], [-2] if upper else [2], [[1]],
        [-np.inf] if upper else [0], [0] if upper else [np.inf],
    )
    assert check_qp_candidate(p, [0], [2 if upper else -2]).passed
    bad = check_qp_candidate(p, [0], [-2 if upper else 2])
    assert not bad.passed and not bad.checks["bound_sign"]


def test_inactive_wrong_multiplier_and_signed_negative_gap_are_not_clipped():
    p = qp([[1]], [-1], [[1]], [0], [2])
    bad = check_qp_candidate(p, [0], [1])
    assert bad.checks["stationarity"]
    assert not bad.checks["complementarity"] and not bad.checks["duality_gap"]
    assert not bad.passed
    violation = check_qp_candidate(bounded_lp(), [2, 0], [1, -2])
    assert violation.metrics["duality_gap"] == -1
    assert not violation.passed and not violation.checks["duality_gap"]


def test_original_feasibility_uses_per_bound_not_global_large_bound_scale():
    p = qp(np.eye(2), [0, 0], np.eye(2), [0, -1e20], [1, 1e20])
    bad = check_qp_candidate(p, [2, 0], [-2, 0])
    assert bad.metrics["primal_feasibility"] == 1
    assert not bad.checks["primal_feasibility"]


@pytest.mark.parametrize("scale", [1., 1e-200, 1e200])
def test_farkas_witness_normalization_and_strict_separation(scale):
    evidence = check_primal_infeasibility(infeasible(), np.array([1., -1.]) * scale)
    assert evidence.passed
    assert evidence.metrics["annihilation"] == 0
    assert evidence.metrics["box_support"] == -1


@pytest.mark.parametrize("mutation", ["flip", "annihilation", "zero", "infinite_support"])
def test_farkas_mutation_negative_controls(mutation):
    v = {
        "flip": [-1, 1], "annihilation": [1, -0.5],
        "zero": [0, 0], "infinite_support": [1, 1],
    }[mutation]
    assert not check_primal_infeasibility(infeasible(), v).passed


def test_zero_margin_farkas_does_not_certify_a_feasible_problem():
    p = qp([[0]], [0], [[1], [1]], [-np.inf, 1], [1, np.inf])
    evidence = check_primal_infeasibility(p, [1, -1])
    assert evidence.checks["annihilation"] and evidence.checks["bound_sign"]
    assert evidence.metrics["box_support"] == 0
    assert not evidence.checks["strict_separation"] and not evidence.passed
    p = replace(p, l=torch.tensor([-np.inf, 1 + 1e-10], dtype=torch.float64))
    assert not check_primal_infeasibility(p, [1, -1]).passed


@pytest.mark.parametrize("scale", [1., 1e-200, 1e200])
def test_singular_psd_unbounded_with_all_recession_row_kinds(scale):
    evidence = check_unbounded_direction(
        unbounded(), np.array([0., 1.]) * scale, feasible_x=[0, 0],
    )
    assert evidence.passed
    assert evidence.metrics["zero_curvature"] == 0
    assert evidence.metrics["descent_slope"] == -1
    assert evidence.metrics["base_feasibility"] == 0


@pytest.mark.parametrize("upper", [False, True])
def test_tiny_wrong_one_sided_recession_sign_is_rejected_strictly(upper):
    A = [[0, 1e-12 if upper else -1e-12]]
    p = qp(
        np.zeros((2, 2)), [0, -1], A,
        [-np.inf] if upper else [0], [0] if upper else [np.inf],
    )
    evidence = check_unbounded_direction(p, [0, 1], feasible_x=[0, 0])
    assert evidence.checks["zero_curvature"] and evidence.checks["strict_descent"]
    assert 0 < evidence.metrics["recession"] < 1e-8
    assert not evidence.checks["recession"] and not evidence.passed


@pytest.mark.parametrize("mutation", ["flip", "curvature", "zero", "base", "recession"])
def test_unbounded_mutation_negative_controls(mutation):
    p, d, base = unbounded(), [0, 1], [0, 0]
    if mutation == "flip":
        d = [0, -1]
    if mutation == "curvature":
        d = [1, 1]
    if mutation == "zero":
        d = [0, 0]
    if mutation == "base":
        base = [1, 0]
    if mutation == "recession":
        p = replace(p, u=torch.tensor([0., 1., 0., np.inf], dtype=torch.float64))
    assert not check_unbounded_direction(p, d, feasible_x=base).passed


def test_descent_ray_without_feasible_base_is_not_unboundedness():
    ray_only = check_unbounded_direction(unbounded(), [0, 1])
    assert not ray_only.passed and not ray_only.checks["feasible_base"]
    assert all(value for key, value in ray_only.checks.items() if key != "feasible_base")
    p = qp(np.zeros((2, 2)), [0, -1], [[1, 0], [1, 0]], [0, 2], [1, 3])
    evidence = check_unbounded_direction(p, [0, 1], feasible_x=[0, 0])
    assert not evidence.passed
    assert evidence.checks["strict_descent"] and not evidence.checks["feasible_base"]


def test_unconstrained_lp_unbounded_and_small_positive_curvature_rejected():
    p = qp([[0]], [-1], np.empty((0, 1)), [], [])
    assert check_unbounded_direction(p, [1], feasible_x=[0]).passed
    curved = replace(p, P=torch.tensor([[1e-12]], dtype=torch.float64))
    bad = check_unbounded_direction(curved, [1], feasible_x=[0])
    assert bad.metrics["zero_curvature"] < 1e-8
    assert not bad.checks["zero_curvature"] and not bad.passed
    assert check_qp_candidate(curved, [1e12], []).passed


def test_unresolved_spectrum_is_unavailable_not_regularized():
    p = qp(np.diag([1., 1e-16]), [0, -1], np.empty((0, 2)), [], [])
    candidate = check_qp_candidate(p, [0, 1e16], [])
    assert candidate.metrics["dual_objective"] is None and not candidate.passed
    assert "no regularized inverse" in candidate.unavailable["dual_objective"]
    ray = check_unbounded_direction(p, [0, 1], feasible_x=[0, 0])
    assert not ray.passed and "kernel" in ray.unavailable


@pytest.mark.parametrize("which", ["candidate_x", "candidate_y", "farkas", "ray", "base"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_candidates_raise(which, value):
    with pytest.raises(ValueError, match="finite"):
        if which == "candidate_x":
            check_qp_candidate(bounded_lp(), [value, 0], [1, -2])
        elif which == "candidate_y":
            check_qp_candidate(bounded_lp(), [1, 0], [value, -2])
        elif which == "farkas":
            check_primal_infeasibility(infeasible(), [value, -1])
        elif which == "ray":
            check_unbounded_direction(unbounded(), [0, value], feasible_x=[0, 0])
        else:
            check_unbounded_direction(unbounded(), [0, 1], feasible_x=[value, 0])


@pytest.mark.parametrize("which", ["x", "y", "farkas", "ray", "base"])
def test_wrong_candidate_dimensions_raise(which):
    with pytest.raises(ValueError, match="shape"):
        if which == "x":
            check_qp_candidate(bounded_lp(), [[1, 0]], [1, -2])
        elif which == "y":
            check_qp_candidate(bounded_lp(), [1, 0], [1])
        elif which == "farkas":
            check_primal_infeasibility(infeasible(), [1])
        elif which == "ray":
            check_unbounded_direction(unbounded(), [1], feasible_x=[0, 0])
        else:
            check_unbounded_direction(unbounded(), [0, 1], feasible_x=[0])


@pytest.mark.parametrize("mutation", [
    "nan", "inf", "bounds_nan", "lower_inf", "upper_inf", "reversed",
    "P_shape", "A_shape", "l_shape", "empty_q", "indefinite", "asymmetric", "complex", "sparse",
])
def test_invalid_problem_is_rejected_independently(mutation):
    p = bounded_lp()
    if mutation == "nan":
        p.q[0] = np.nan
    elif mutation == "inf":
        p.A[0, 0] = np.inf
    elif mutation == "bounds_nan":
        p.l[0] = np.nan
    elif mutation == "lower_inf":
        p.l[0] = np.inf
    elif mutation == "upper_inf":
        p.u[0] = -np.inf
    elif mutation == "reversed":
        p.l[0] = 2
    elif mutation == "P_shape":
        p = replace(p, P=p.P[:, :1])
    elif mutation == "A_shape":
        p = replace(p, A=p.A[:, :1])
    elif mutation == "l_shape":
        p = replace(p, l=p.l[:1])
    elif mutation == "empty_q":
        p = replace(p, q=torch.empty(0))
    elif mutation == "indefinite":
        p = replace(p, P=-torch.eye(2))
    elif mutation == "asymmetric":
        p.P[0, 1] = 1
    elif mutation == "complex":
        p = replace(p, P=p.P.to(torch.complex128))
    else:
        p = replace(p, P=p.P.to_sparse())
    with pytest.raises(ValueError):
        check_qp_candidate(p, [1, 0], [1, -2])


@pytest.mark.parametrize("kwargs", [
    {"atol": 0}, {"atol": -1}, {"atol": np.nan}, {"atol": np.inf},
    {"rtol": -1}, {"rtol": np.nan}, {"rtol": 1}, {"rtol": True},
])
def test_invalid_tolerances_raise(kwargs):
    with pytest.raises(ValueError):
        check_qp_candidate(bounded_lp(), [1, 0], [1, -2], **kwargs)


def test_non_cpu_tensor_is_rejected_without_device_transfer():
    with pytest.raises(ValueError, match="CPU"):
        check_qp_candidate(bounded_lp(), torch.empty(2, device="meta"), [1, -2])


@pytest.mark.parametrize("kind", ["farkas", "ray"])
def test_direction_normalization_underflow_is_fail_loud(kind):
    with pytest.raises(ValueError, match="underflow"):
        if kind == "farkas":
            p = qp([[0]], [0], [[0], [0]], [-np.inf, -np.inf], [-1, np.inf])
            check_primal_infeasibility(p, [1e300, -1e-300])
        else:
            check_unbounded_direction(unbounded(), [1e-300, 1e300], feasible_x=[0, 0])


def test_overflow_is_fail_loud_not_a_nan_report():
    p = qp([[1e308]], [0], np.empty((0, 1)), [], [])
    with pytest.raises(ValueError, match="arithmetic failed"):
        check_qp_candidate(p, [1e308], [])


def test_machine_readable_acceptance_with_named_mutation_controls(tmp_path, request):
    singular = qp(np.diag([2, 0]), [-4, 0], np.empty((0, 2)), [], [])
    free = qp([[2]], [-4], [[1]], [-np.inf], [np.inf])
    cases = {
        "lp_optimal": (True, check_qp_candidate(bounded_lp(), [1, 0], [1, -2])),
        "singular_range": (True, check_qp_candidate(singular, [2, 100], [])),
        "singular_out_of_range": (False, check_qp_candidate(
            replace(singular, q=torch.tensor([-4., 1.])), [2, 0], [],
        )),
        "lp_out_of_range": (False, check_qp_candidate(bounded_lp(), [1, 0], [0, 0])),
        "free_row": (True, check_qp_candidate(free, [2], [0])),
        "mutation_infinite_bound_sign": (False, check_qp_candidate(free, [2], [1e-300])),
        "singular_unbounded": (True, check_unbounded_direction(
            unbounded(), [0, 1], feasible_x=[0, 0],
        )),
        "primal_infeasible": (True, check_primal_infeasibility(infeasible(), [1, -1])),
        "mutation_wrong_x": (False, check_qp_candidate(bounded_lp(), [2, 0], [1, -2])),
        "mutation_wrong_y": (False, check_qp_candidate(bounded_lp(), [1, 0], [-1, 2])),
        "mutation_farkas_annihilation": (False, check_primal_infeasibility(
            infeasible(), [1, -0.5],
        )),
        "mutation_ray_curvature": (False, check_unbounded_direction(
            unbounded(), [1, 1], feasible_x=[0, 0],
        )),
        "mutation_ray_recession": (False, check_unbounded_direction(
            unbounded(), [0, -1], feasible_x=[0, 0],
        )),
        "mutation_missing_base": (False, check_unbounded_direction(unbounded(), [0, 1])),
    }
    assert all(evidence.passed == expected for expected, evidence in cases.values())
    assert all(
        type(evidence.passed) is bool
        and all(type(value) is bool for value in evidence.checks.values())
        and all(value is None or type(value) is float for value in evidence.metrics.values())
        and all(type(value) is float for value in evidence.thresholds.values())
        for _, evidence in cases.values()
    )
    root = Path(__file__).resolve().parents[1]
    paths = [
        "src/openopt/certificates.py", "tests/test_qp_certificates.py",
        "docs/research/certificates.md", "src/openopt/qp.py", "src/openopt/__init__.py",
    ]
    report = {
        "schema_version": 1, "verdict": "PASS", "device": "cpu",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "owner": "certificates", "baseline_commit": "fcefac1",
        "qualification": "numerical evidence only; no solver status promotion or exact proof",
        "case_count": len(cases),
        "negative_control_count": sum(not expected for expected, _ in cases.values()),
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths
        },
        "cases": {
            name: {"expected_passed": expected, "evidence": evidence.to_dict()}
            for name, (expected, evidence) in cases.items()
        },
    }
    artifact = tmp_path / "numerical-acceptance.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(artifact.read_text()) == report
    request.node.user_properties.append(("numerical_acceptance_path", str(artifact)))


def reproducible_cases():
    """Tiny fixed same-model witnesses, no solve or random BLAS model generation."""
    return [
        (bounded_lp(), [1., 0.], [1., -2.], True),
        (qp([[2.1, .3], [.3, 1.7]], [-2.1, -.3],
            [[.7, 1.2]], [.7], [.7]), [1., 0.], [0.], True),
        (qp([[1., 1.], [1., 1.]], [-2., -2.], np.empty((0, 2)), [], []),
         [3., -1.], [], True),
        (qp(np.diag([1., 1e-16]), [0., 0.], np.empty((0, 2)), [], []),
         [0., 0.], [], False),
        (qp([[2.]], [-4.], [[1.]], [-np.inf], [np.inf]), [2.], [1e-300], False),
        (bounded_lp(), [2., 0.], [1., -2.], False),
        (bounded_lp(), [1., 0.], [-1., 2.], False),
        (qp([[2.]], [-3.], np.empty((0, 1)), [], []), [2.], [], False),
    ]


def test_reproducible_complete_original_gates_and_independence(monkeypatch, tmp_path):
    import openopt.certificates as module
    import openopt.qp as solver

    def forbidden(*args, **kwargs):
        raise AssertionError("reproducible checker cannot call BLAS/LAPACK/solver")

    for name in ("eigh", "eigvalsh", "solve", "inv", "pinv", "lstsq", "cholesky"):
        monkeypatch.setattr(np.linalg, name, forbidden)
    monkeypatch.setattr(np, "dot", forbidden)
    monkeypatch.setattr(np, "sum", forbidden)
    monkeypatch.setattr(solver, "solve_qp", forbidden)
    monkeypatch.setattr(solver, "prepare_qp", forbidden)
    monkeypatch.setattr(module.math, "sqrt", forbidden)
    rows = []
    for p, x, y, expected in reproducible_cases():
        copies = [value.clone() for value in (p.P, p.q, p.A, p.l, p.u)]
        evidence = check_qp_candidate(p, x, y, arithmetic=REPRODUCIBLE_ARITHMETIC)
        assert evidence.passed == expected
        assert set(evidence.checks) == {
            "primal_feasibility", "stationarity", "bound_sign", "complementarity",
            "dual_objective_available", "duality_gap",
        }
        assert all(torch.equal(a, b) for a, b in zip(copies, (p.P, p.q, p.A, p.l, p.u)))
        rows.append(evidence.to_dict())
    (tmp_path / "tiny-original-gates.json").write_text(json.dumps({
        "verdict": "PASS", "contract": reproducible_checker_contract(), "cases": rows,
        "qualification": "CPU only, no CUDA or final seal",
    }, allow_nan=False))


@pytest.mark.parametrize("mutation", ["component", "sign", "gap", "range"])
def test_reproducible_negative_components_sign_gap_and_range(mutation):
    p, x, y = bounded_lp(), [1., 0.], [1., -2.]
    if mutation == "component":
        p = qp(np.eye(2), [0., 0.], np.eye(2), [0., -1e20], [1., 1e20])
        x, y = [2., 0.], [-2., 0.]
    elif mutation == "sign":
        p, x, y = qp([[2.]], [-4.], [[1.]], [-np.inf], [np.inf]), [2.], [1e-300]
    elif mutation == "gap":
        p, x, y = qp([[1.]], [-1.], [[1.]], [0.], [2.]), [0.], [1.]
    else:
        p, x, y = qp(np.diag([2., 0.]), [-4., 1.], np.empty((0, 2)), [], []), [2., 0.], []
    evidence = check_qp_candidate(p, x, y, arithmetic=REPRODUCIBLE_ARITHMETIC)
    assert not evidence.passed
    if mutation == "component":
        assert not evidence.checks["primal_feasibility"]
    elif mutation == "sign":
        assert not evidence.checks["bound_sign"] and evidence.metrics["bound_sign"] > 0
    elif mutation == "gap":
        assert evidence.checks["stationarity"] and not evidence.checks["duality_gap"]
    else:
        assert evidence.metrics["range_residual"] == 1 and evidence.metrics["dual_objective"] is None


@pytest.mark.parametrize("mutation", ["n", "m", "subnormal", "nonfinite", "rounding", "spectrum"])
def test_reproducible_unsupported_is_fail_loud_never_default(monkeypatch, mutation):
    import openopt.certificates as module

    p, x, y = bounded_lp(), [1., 0.], [1., -2.]
    if mutation == "n":
        p, x, y = qp(np.eye(9), np.zeros(9), np.empty((0, 9)), [], []), np.zeros(9), []
    elif mutation == "m":
        p, x, y = qp([[1.]], [0.], np.zeros((65, 1)), np.zeros(65), np.zeros(65)), [0.], np.zeros(65)
    elif mutation == "subnormal":
        p.P[0, 0] = float.fromhex("0x0.0000000000001p-1022")
    elif mutation == "nonfinite":
        x[0] = float("nan")
    elif mutation == "rounding":
        def unsupported():
            raise ValueError("unsupported reproducible rounding")
        monkeypatch.setattr(module, "_binary64_runtime", unsupported)
    else:
        def unresolved(*args):
            raise ValueError("unresolved reproducible Jacobi spectrum")
        monkeypatch.setattr(module, "_jacobi", unresolved)
    with pytest.raises(ValueError, match="unsupported|finite|unresolved"):
        check_qp_candidate(p, x, y, arithmetic=REPRODUCIBLE_ARITHMETIC)


def test_reproducible_spectral_verification_detects_broken_dot(monkeypatch):
    import openopt.certificates as module

    original = module._bdot
    monkeypatch.setattr(module, "_bdot", lambda a, b: original(a, b) + .01)
    with pytest.raises(ValueError, match="residual/orthogonality"):
        module._jacobi([[2., .3], [.3, 1.7]])


def test_reproducible_computed_zero_cannot_hide_original_positive_curvature(monkeypatch):
    import openopt.certificates as module

    monkeypatch.setattr(module, "_jacobi", lambda matrix: ([0., 1.], [[0., 1.], [1., 0.]]))
    p = qp(np.diag([1., 1e-16]), [0., 0.], np.empty((0, 2)), [], [])
    result = check_qp_candidate(p, [0., 0.], [], arithmetic=REPRODUCIBLE_ARITHMETIC)
    assert not result.passed and not result.checks["dual_objective_available"]


def test_reproducible_actual_rounding_mode_rejected_and_restored():
    import ctypes
    import ctypes.util
    import openopt.certificates as module

    if sys.platform != "linux":
        pytest.skip("reproducible checker runtime contract requires Linux")
    library = ctypes.CDLL(ctypes.util.find_library("m"))
    getround, setround = library.fegetround, library.fesetround
    getround.argtypes, getround.restype = (), ctypes.c_int
    setround.argtypes, setround.restype = (ctypes.c_int,), ctypes.c_int
    original = getround()
    try:
        if setround(0x400) != 0:
            pytest.skip("runtime does not expose this reversible rounding control")
        assert getround() != 0
        with pytest.raises(ValueError, match="rounding mode"):
            module._binary64_runtime()
    finally:
        assert setround(original) == 0
    assert getround() == original
    module._binary64_runtime()


def test_reproducible_binary64_sqrt_exact_rounding():
    import math
    import openopt.certificates as module
    from decimal import Decimal, localcontext

    with localcontext() as context:
        context.prec = 800
        for value in (0., 1., 2., 3., .1, 1e-300, 1e300, sys.float_info.min,
                      sys.float_info.max, float.fromhex("0x1.fffffffffffffp+1")):
            assert module._broot(value).hex() == float(Decimal(value).sqrt()).hex()
        rng = np.random.default_rng(819)
        for value in rng.uniform(.01, 100., 60):
            assert module._broot(float(value)).hex() == math.sqrt(float(value)).hex()


def test_default_checker_evidence_unchanged_after_opt_in():
    for p, x, y, _ in reproducible_cases():
        before = check_qp_candidate(p, x, y).to_dict()
        explicit = check_qp_candidate(p, x, y, arithmetic="numpy").to_dict()
        check_qp_candidate(p, x, y, arithmetic=REPRODUCIBLE_ARITHMETIC)
        assert check_qp_candidate(p, x, y).to_dict() == before == explicit
    with pytest.raises(ValueError, match="unsupported original checker arithmetic"):
        check_qp_candidate(bounded_lp(), [1., 0.], [1., -2.], arithmetic="fma_if_matches")


def test_reproducible_n8_cost_and_spectral_residual(tmp_path):
    import openopt.certificates as module

    matrix = [[float(5 + i) if i == j else .125 for j in range(8)] for i in range(8)]
    eigenvalues, vectors = module._jacobi(matrix)
    assert np.max(np.abs(np.asarray(matrix) @ np.asarray(vectors)
                         - np.asarray(vectors) * eigenvalues)) < 1e-12
    p = qp(matrix, np.zeros(8), np.zeros((64, 8)), np.zeros(64), np.zeros(64))
    started = time.perf_counter()
    evidence = check_qp_candidate(p, np.zeros(8), np.zeros(64), arithmetic=REPRODUCIBLE_ARITHMETIC)
    elapsed = time.perf_counter() - started
    assert evidence.passed
    (tmp_path / "n8-cost.json").write_text(json.dumps({
        "verdict": "PASS", "n": 8, "m": 64, "wall_seconds": elapsed,
        "qualification": "one CPU checker call, not optimizer performance",
    }))


def test_reproducible_cross_process_different_blas_environments_exact(tmp_path):
    arms = [
        {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
         "OPENBLAS_CORETYPE": "Haswell", "NPY_DISABLE_CPU_FEATURES": "AVX2,FMA3"},
        {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
         "OPENBLAS_CORETYPE": "Sandybridge", "NPY_DISABLE_CPU_FEATURES": ""},
    ]
    reports = []
    for index, arm in enumerate(arms):
        output = tmp_path / f"arm-{index}.json"
        environment = {**os.environ, **arm, "CUDA_VISIBLE_DEVICES": "-1",
                       "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run([sys.executable, "-B", str(Path(__file__).resolve()),
                                 "--reproducible-worker-output", str(output)],
                                env=environment, capture_output=True, check=False)
        (tmp_path / f"arm-{index}.log").write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0, result.stderr.decode()
        report = json.loads(output.read_text())
        assert report["verdict"] == "PASS"
        assert all(report["runtime"]["environment"][name] == value for name, value in arm.items())
        reports.append(report)
    assert reports[0]["runtime"]["pid"] != reports[1]["runtime"]["pid"]
    assert reports[0]["runtime"]["numpy"] == reports[1]["runtime"]["numpy"]
    assert reports[0]["evidence"] == reports[1]["evidence"]
    assert reports[0]["evidence_sha256"] == reports[1]["evidence_sha256"]
    (tmp_path / "cross-process-receipt.json").write_text(json.dumps({
        "verdict": "PASS", "arms": reports,
        "qualification": "two CPU processes, same installed NumPy, distinct BLAS environments; no GPU",
    }, allow_nan=False))


def reproducible_worker_main():
    """External-output subprocess probe; no inline code or solver invocation."""
    import argparse
    from tests.test_qp_refinement import checker_runtime

    parser = argparse.ArgumentParser(description=reproducible_worker_main.__doc__)
    parser.add_argument("--reproducible-worker-output", type=Path, required=True)
    parser.add_argument("--refinement-sidecar", type=Path, action="append", default=[])
    args = parser.parse_args()
    evidence = [check_qp_candidate(p, x, y, arithmetic=REPRODUCIBLE_ARITHMETIC).to_dict()
                for p, x, y, _ in reproducible_cases()]
    if args.refinement_sidecar:
        from tests.test_qp_refinement import check_sidecar
        root = Path(__file__).resolve().parents[1]
        for path in args.refinement_sidecar:
            raw = path.read_bytes()
            evidence.append(check_sidecar(path, hashlib.sha256(raw).hexdigest(), root))
    raw = json.dumps(evidence, sort_keys=True, allow_nan=False).encode()
    report = {"verdict": "PASS", "contract": reproducible_checker_contract(),
              "runtime": {**checker_runtime(), "pid": os.getpid()},
              "evidence": evidence, "evidence_sha256": hashlib.sha256(raw).hexdigest()}
    with args.reproducible_worker_output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, allow_nan=False)
    print("REPRODUCIBLE_CPU_WORKER verdict=PASS evidence_sha256=" + report["evidence_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(reproducible_worker_main())
