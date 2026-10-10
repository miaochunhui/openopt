"""Exact reductions, original-model postsolve gates and frozen CUDA sidecars."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from openopt.certificates import check_primal_infeasibility, check_qp_candidate
from openopt.presolve import PresolveConfig, presolve_qp, solve_presolved_qp
from openopt.qp import QPConfig, QPProblem, solve_qp

ON = PresolveConfig(enabled=True)
STRICT = QPConfig(
    eps_abs=1e-10, eps_rel=1e-10, max_iterations=20000,
    dual_gap_termination=True, adaptive_rho=True,
)


def problem(P, q, A, l, u):
    n = len(q)
    return QPProblem(
        torch.tensor(P, dtype=torch.float64).reshape(n, n),
        torch.tensor(q, dtype=torch.float64),
        torch.tensor(A, dtype=torch.float64).reshape(len(l), n),
        torch.tensor(l, dtype=torch.float64), torch.tensor(u, dtype=torch.float64),
    )


def coupled():
    # x0=2 is fixed; the free optimum is x1=-1 with inactive inequalities.
    return problem(
        [[4, 1], [1, 2]], [-3, 0],
        [[1, 0], [0, 0], [0, 1], [1, 1], [1, 0]],
        [2, -float("inf"), -3, 0, 2], [2, 5, 3, 4, 2],
    )


def assert_complete(evidence):
    assert evidence.passed, evidence.to_dict()
    assert set(evidence.checks) == {
        "primal_feasibility", "stationarity", "bound_sign", "complementarity",
        "dual_objective_available", "duality_gap",
    }
    assert all(evidence.checks.values())
    assert evidence.metrics["duality_gap"] is not None


def write_sidecar(tmp_path, request, prepared, result, device):
    arrays = {}
    for prefix, p in (
        ("original", prepared.original_problem), ("working", prepared.working_problem),
        ("reduced", prepared.reduced_problem),
    ):
        arrays.update({
            f"{prefix}_{name}": getattr(p, name).detach().cpu().numpy()
            for name in ("P", "q", "A", "l", "u")
        })
    for name in (
        "free_variables", "fixed_variables", "fixed_values", "fixed_rows",
        "kept_rows", "removed_rows", "row_shifts",
    ):
        arrays[name] = getattr(prepared, name).detach().cpu().numpy()
    arrays["x"], arrays["y"] = result.x.detach().cpu().numpy(), result.y.detach().cpu().numpy()
    arrays["objective_offset"] = np.array(prepared.objective_offset)
    if result.infeasibility_direction is not None:
        arrays["infeasibility_direction"] = result.infeasibility_direction.detach().cpu().numpy()
    data = tmp_path / "full-model.npz"
    np.savez(data, **arrays)
    with np.load(data, allow_pickle=False) as saved:
        assert set(saved.files) == set(arrays)
        assert all(np.array_equal(saved[name], value) for name, value in arrays.items())
        replay_problem = QPProblem(*[
            torch.from_numpy(saved[f"original_{name}"].copy())
            for name in ("P", "q", "A", "l", "u")
        ])
        replay = check_qp_candidate(
            replay_problem, saved["x"], saved["y"],
            atol=prepared.options.check_atol, rtol=prepared.options.check_rtol,
        )
        assert replay.to_dict() == result.evidence.to_dict()
        if result.infeasibility_direction is not None:
            assert check_primal_infeasibility(
                replay_problem, saved["infeasibility_direction"],
            ).passed
    root = Path(__file__).resolve().parents[1]
    report = {
        "schema_version": 1, "kind": "presolve_original_coordinate_replay",
        "checked_at": datetime.now(timezone.utc).isoformat(), "device": device,
        "status": result.status, "objective": result.objective,
        "presolve_config": asdict(prepared.options),
        "data": {"path": data.name, "sha256": hashlib.sha256(data.read_bytes()).hexdigest()},
        "source_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in ("src/openopt/presolve.py", "src/openopt/qp.py",
                         "src/openopt/certificates.py", "tests/test_qp_presolve.py")
        },
        "evidence": replay.to_dict(),
        "infeasibility_evidence": (
            result.infeasibility_evidence.to_dict()
            if result.infeasibility_evidence is not None else None
        ),
        "qualification": "numerical original-model replay; not an exact proof or performance claim",
    }
    artifact = tmp_path / "presolve-acceptance.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert json.loads(artifact.read_text()) == report
    request.node.user_properties.extend([
        ("presolve_acceptance_path", str(artifact)),
        ("presolve_acceptance_sha256", hashlib.sha256(artifact.read_bytes()).hexdigest()),
        ("status", result.status),
    ])


def test_coupled_dual_restore_objective_and_original_solver_parity(tmp_path, request):
    p = coupled()
    before = [getattr(p, name).clone() for name in ("P", "q", "A", "l", "u")]
    prepared = presolve_qp(p, STRICT, ON)
    assert prepared.free_variables.tolist() == [1]
    assert prepared.fixed_variables.tolist() == [0]
    assert prepared.fixed_rows.tolist() == [0]
    assert prepared.removed_rows.tolist() == [0, 1, 4]
    assert prepared.kept_rows.tolist() == [2, 3]
    assert prepared.objective_offset == 2
    assert torch.equal(prepared.row_shifts, torch.tensor([2, 0, 0, 2, 2], dtype=torch.float64))
    raw = solve_qp(prepared.reduced_problem, STRICT)
    restored_x, restored_y = prepared.restore(raw.x, raw.y)
    assert_complete(prepared.check(restored_x, restored_y))
    result = solve_presolved_qp(p, STRICT, ON)
    ordinary = solve_qp(p, STRICT)
    assert result.status == ordinary.status == "solved"
    assert_complete(result.evidence)
    assert torch.allclose(result.x, ordinary.x, atol=1e-7, rtol=1e-7)
    assert result.objective == pytest.approx(ordinary.objective, abs=1e-7)
    assert torch.allclose(result.x, torch.tensor([2., -1.], dtype=torch.float64), atol=1e-7)
    assert result.y[0] == pytest.approx(-4., abs=1e-7)
    assert result.y[4] == 0  # The equality multiplier need not be unique.
    assert raw.objective + prepared.objective_offset == pytest.approx(result.objective, abs=1e-10)
    assert all(torch.equal(value, getattr(p, name)) for value, name in zip(before, ("P", "q", "A", "l", "u")))
    write_sidecar(tmp_path, request, prepared, result, "cpu")


@pytest.mark.parametrize("sign", [-1, 1])
def test_nonzero_retained_multiplier_contributes_to_fixed_dual(sign):
    # At x=(2,0), row (x0+x1) is active and has signed multiplier 4*sign.
    p = problem(
        [[4, 1], [1, 2]], [-3, -2 - 4 * sign],
        [[1, 0], [1, 1], [0, 0]],
        [2, 2 if sign < 0 else -float("inf"), -1],
        [2, 2 if sign > 0 else float("inf"), 1],
    )
    result = solve_presolved_qp(p, STRICT, ON)
    assert result.status == "solved"
    assert_complete(result.evidence)
    assert torch.allclose(result.x, torch.tensor([2., 0.], dtype=torch.float64), atol=1e-7)
    assert result.y[1] == pytest.approx(4 * sign, abs=1e-7)
    assert result.y[0] == pytest.approx(-5 - 4 * sign, abs=1e-7)
    # Omitting the retained multiplier from the fixed dual must fail.
    wrong_y = result.y.clone()
    wrong_y[0] = -5
    assert not result.presolved.check(result.x, wrong_y).passed
    ordinary = solve_qp(p, STRICT)
    assert ordinary.status == "solved"
    assert result.objective == pytest.approx(ordinary.objective, abs=1e-7)


def test_multiple_fixed_variables_permuted_rows_and_active_coupling():
    p = problem(
        [[4, 1, 0], [1, 4, 1], [0, 1, 4]], [3, -2, 1],
        [[0, 0, 1], [1, 1, 1], [1, 0, 0], [0, 0, 0]],
        [-1, -float("inf"), 2, -1], [-1, 1, 2, 1],
    )
    result = solve_presolved_qp(p, STRICT, ON)
    assert result.status == "solved"
    assert_complete(result.evidence)
    assert result.presolved.fixed_variables.tolist() == [0, 2]
    assert result.presolved.fixed_rows.tolist() == [2, 0]
    assert result.presolved.free_variables.tolist() == [1]
    assert result.presolved.objective_offset == 15
    assert torch.allclose(result.x, torch.tensor([2., 0., -1.], dtype=torch.float64), atol=1e-7)
    assert torch.allclose(result.y, torch.tensor([2., 1., -12., 0.], dtype=torch.float64), atol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_disabled_is_bitwise_original_path(dtype):
    p = coupled()
    c = QPConfig(dtype=dtype, eps_abs=1e-6, eps_rel=1e-6)
    ordinary = solve_qp(p, c)
    result = solve_presolved_qp(p, c, PresolveConfig(check_atol=1e-4, check_rtol=1e-4))
    assert result.status == ordinary.status
    assert torch.equal(result.x, ordinary.x) and torch.equal(result.y, ordinary.y)
    assert result.presolved.removed_rows.numel() == result.presolved.fixed_variables.numel() == 0
    assert result.presolved.objective_offset == 0


@pytest.mark.parametrize("kind", ["empty_rows", "zero_column", "lp", "all_fixed", "constant_inequalities"])
def test_empty_degenerate_lp_and_zero_variable_semantics(kind):
    if kind == "empty_rows":
        p = problem([[2]], [-4], [], [], [])
    elif kind == "zero_column":
        p = problem([[2, 0], [0, 0]], [-4, 0], [[0, 0]], [-1], [1])
    elif kind == "lp":
        p = problem([[0, 0], [0, 0]], [3, -1], [[1, 0], [0, 1]], [2, 0], [2, 1])
    elif kind == "all_fixed":
        p = problem([[4, 1], [1, 2]], [3, -1], [[1, 0], [0, 1]], [2, -1], [2, -1])
    else:
        p = problem([[2]], [3], [[1], [2]], [2, 3], [2, 5])
    result = solve_presolved_qp(p, STRICT, ON)
    assert result.status == "solved"
    assert_complete(result.evidence)
    if kind in ("all_fixed", "constant_inequalities"):
        assert result.presolved.is_constant and result.reduced_result is None
        assert result.iterations == 0
        assert result.presolved.reduced_problem.P.shape == (0, 0)
    if kind == "zero_column":
        assert result.presolved.free_variables.numel() == 2  # Not a column eliminator.


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("fixed", [False, True])
def test_zero_row_farkas_original_model_and_mutated_direction(sign, fixed, tmp_path, request):
    if fixed:
        p = problem([[2]], [0], [[1], [2]], [2, 5] if sign < 0 else [2, -9],
                    [2, 9] if sign < 0 else [2, 3])
    else:
        p = problem([[2]], [0], [[0]], [1] if sign < 0 else [-2],
                    [2] if sign < 0 else [-1])
    result = solve_presolved_qp(p, STRICT, ON)
    assert result.status == "infeasible_numerical" and result.iterations == 0
    assert result.infeasibility_evidence.passed
    ray = result.infeasibility_direction.to("cpu", torch.float64)
    assert check_primal_infeasibility(p, ray).passed
    assert not check_primal_infeasibility(p, -ray).passed
    assert result.presolved.kept_rows.numel() == 1
    assert "zero_row_contradiction_retained" in result.presolved.notes
    write_sidecar(tmp_path, request, result.presolved, result, "cpu")


def test_tiny_zero_row_contradiction_is_not_false_certificate():
    p = problem([[1]], [0], [[0]], [1e-12], [2e-12])
    result = solve_presolved_qp(p, STRICT, ON)
    assert result.status == "infeasible_unverified"
    assert not result.infeasibility_evidence.passed
    assert result.presolved.kept_rows.tolist() == [0]


@pytest.mark.parametrize("kind", [
    "nearly_zero", "non_identity", "negative_identity", "near_equality", "conflict",
    "rounded_cast", "rounding", "overflow", "underflow",
])
def test_unsupported_or_unrepresentable_reductions_keep_original_model(kind):
    p = coupled()
    c = STRICT
    if kind == "nearly_zero":
        p = problem([[1]], [0], [[1e-30]], [-1], [1])
    elif kind == "non_identity":
        p = problem([[1]], [0], [[2]], [4], [4])
    elif kind == "negative_identity":
        p = problem([[1]], [0], [[-1]], [-2], [-2])
    elif kind == "near_equality":
        p = problem([[1]], [0], [[1]], [2], [2 + 1e-12])
    elif kind == "conflict":
        p = problem([[1]], [0], [[1], [1]], [1, 2], [1, 2])
    elif kind == "rounded_cast":
        p = problem([[1]], [1 + 1e-12], [[1]], [2], [2])
        c = replace(STRICT, dtype=torch.float32)
    elif kind == "rounding":
        p = problem([[1, 0], [0, 1]], [0, 0.1], [[1, 0], [0.1, 1]], [0.1, 0.1], [0.1, 2])
    elif kind == "overflow":
        p = problem([[1]], [1e308], [[1]], [1e308], [1e308])
    else:
        p = problem([[1]], [0], [[1]], [1e-200], [1e-200])
    prepared = presolve_qp(p, c, ON)
    assert prepared.fixed_variables.numel() == prepared.removed_rows.numel() == 0
    assert prepared.free_variables.numel() == p.q.numel()
    assert prepared.objective_offset == 0
    for name in ("P", "q", "A", "l", "u"):
        assert torch.equal(getattr(prepared.reduced_problem, name), getattr(p, name).to(c.dtype))


@pytest.mark.parametrize("field", [
    "original_q", "working_P", "reduced_l", "free_variables", "fixed_variables",
    "fixed_values", "fixed_rows", "kept_rows", "removed_rows", "row_shifts", "offset",
])
def test_sealed_mapping_and_models_reject_mutation(field):
    prepared = presolve_qp(coupled(), STRICT, ON)
    if "_" in field and field.split("_", 1)[0] in ("original", "working", "reduced"):
        prefix, name = field.split("_", 1)
        getattr(getattr(prepared, f"{prefix}_problem"), name).reshape(-1)[0] += 1
    elif field == "offset":
        prepared = replace(prepared, objective_offset=prepared.objective_offset + 1)
    else:
        getattr(prepared, field).reshape(-1)[0] += 1
    with pytest.raises(ValueError, match="mutation"):
        prepared.restore(torch.tensor([-1.], dtype=torch.float64), torch.zeros(2, dtype=torch.float64))


@pytest.mark.parametrize("mutation", ["wrong_x", "wrong_y", "wrong_fixed_dual"])
def test_full_original_gap_gate_detects_wrong_primal_and_signed_dual(mutation):
    prepared = presolve_qp(coupled(), STRICT, ON)
    x, y = prepared.restore(torch.tensor([-1.], dtype=torch.float64), torch.zeros(2, dtype=torch.float64))
    assert_complete(prepared.check(x, y))
    if mutation == "wrong_x":
        x[1] += 0.5
    elif mutation == "wrong_y":
        y[2] = 2
    else:
        y[0] *= -1
    assert not prepared.check(x, y).passed


def test_reduced_solved_candidate_failing_original_gap_is_rejected(monkeypatch):
    import openopt.presolve as module

    p = coupled()
    ordinary = solve_qp(p, STRICT)
    bad = replace(ordinary, x=torch.tensor([0.], dtype=torch.float64),
                  y=torch.zeros(2, dtype=torch.float64), status="solved")
    monkeypatch.setattr(module, "solve_qp", lambda *_: bad)
    result = solve_presolved_qp(p, STRICT, ON)
    assert result.status == "candidate_rejected" and not result.evidence.passed


def test_original_unresolved_spectrum_is_not_hidden_by_reduction():
    p = problem([[1e-20, 0], [0, 2]], [0, -4], [[1, 0]], [0], [0])
    result = solve_presolved_qp(p, STRICT, ON)
    assert result.reduced_result.status == "solved"
    assert result.status == "candidate_rejected"
    assert result.evidence.checks["stationarity"]
    assert not result.evidence.checks["dual_objective_available"]
    assert not result.evidence.checks["duality_gap"]
    assert "unresolved PSD spectrum" in result.evidence.unavailable["dual_objective"]


def test_reduced_gap_not_substitute_for_original_gap(tmp_path, request):
    p = coupled()
    result = solve_presolved_qp(p, STRICT, ON)
    assert_complete(result.evidence)
    report = {
        "schema_version": 1, "kind": "presolve_negative_controls",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "checks": {},
    }
    for mutation in ("wrong_x", "wrong_kept_y", "wrong_fixed_dual"):
        x, y = result.x.clone(), result.y.clone()
        if mutation == "wrong_x":
            x[1] += 0.5
        elif mutation == "wrong_kept_y":
            y[2] = 2
        else:
            y[0] *= -1
        evidence = result.presolved.check(x, y)
        assert not evidence.passed
        report["checks"][mutation] = evidence.to_dict()
    artifact = tmp_path / "negative-controls.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert json.loads(artifact.read_text()) == report
    request.node.user_properties.append(("presolve_negative_controls_path", str(artifact)))


def test_limits_and_reduced_directions_are_not_promoted():
    result = solve_presolved_qp(coupled(), replace(STRICT, max_iterations=1), ON)
    assert result.status == "max_iterations" and result.iterations == 1
    p = problem([[0, 0], [0, 0]], [0, -1], [[1, 0]], [2], [2])
    result = solve_presolved_qp(p, replace(STRICT, direction_termination=True), ON)
    assert result.reduced_result.status == "unbounded"
    assert result.status == "reduced_direction_unverified"
    assert result.infeasibility_direction is None


@pytest.mark.parametrize("field", ["P", "q", "A"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_model_fail_loud(field, value):
    p = coupled()
    getattr(p, field).reshape(-1)[0] = value
    with pytest.raises(ValueError, match="finite"):
        presolve_qp(p, STRICT, ON)


@pytest.mark.parametrize("kind", [
    "empty", "P_shape", "A_shape", "l_shape", "u_shape", "complex",
    "sparse", "integer", "asymmetric", "nonconvex", "reversed", "nan_bound",
    "inf_lower", "inf_upper", "mixed_dtype", "not_tensor",
])
def test_invalid_model_fail_loud(kind):
    p = coupled()
    if kind == "empty":
        p = replace(p, q=torch.empty(0, dtype=torch.float64))
    elif kind == "P_shape":
        p = replace(p, P=p.P[:1])
    elif kind == "A_shape":
        p = replace(p, A=p.A[:, :1])
    elif kind == "l_shape":
        p = replace(p, l=p.l[:1])
    elif kind == "u_shape":
        p = replace(p, u=p.u[None])
    elif kind == "complex":
        p = replace(p, P=p.P.to(torch.complex128))
    elif kind == "sparse":
        p = replace(p, A=p.A.to_sparse())
    elif kind == "integer":
        p = replace(p, q=p.q.to(torch.int64))
    elif kind == "asymmetric":
        p.P[0, 1] += 1e-15
    elif kind == "nonconvex":
        p = replace(p, P=-p.P)
    elif kind == "reversed":
        p.l[0] = 3
    elif kind == "nan_bound":
        p.u[0] = float("nan")
    elif kind == "inf_lower":
        p.l[0] = float("inf")
    elif kind == "inf_upper":
        p.u[0] = -float("inf")
    elif kind == "mixed_dtype":
        p = replace(p, q=p.q.float())
    else:
        p = replace(p, q=[1, 2])
    with pytest.raises((ValueError, TypeError)):
        presolve_qp(p, STRICT, ON)


@pytest.mark.parametrize("name", ["enabled", "remove_zero_rows", "fix_identity_equalities"])
def test_config_boolean_strict(name):
    with pytest.raises(ValueError, match="boolean"):
        presolve_qp(coupled(), STRICT, replace(ON, **{name: 1}))


@pytest.mark.parametrize("name,value", [
    ("check_atol", 0), ("check_atol", True), ("check_atol", float("nan")),
    ("check_atol", float("inf")), ("check_rtol", -1), ("check_rtol", 1),
])
def test_invalid_check_thresholds(name, value):
    with pytest.raises(ValueError):
        presolve_qp(coupled(), STRICT, replace(ON, **{name: value}))


def test_snapshot_survives_input_mutation_and_before_cast_precision():
    p = coupled()
    p.q[1] = 1e-12
    prepared = presolve_qp(p, replace(STRICT, dtype=torch.float32), ON)
    assert prepared.original_problem.q[1] == p.q[1]
    before = prepared.original_problem.q.clone()
    p.q.add_(100)
    assert torch.equal(prepared.original_problem.q, before)
    prepared.validate()


@pytest.mark.parametrize("field", ["q", "A"])
@pytest.mark.parametrize("value", [1e-100, 1e100])
def test_cast_underflow_overflow_does_not_create_structural_zeros(field, value):
    p = coupled()
    getattr(p, field).reshape(-1)[0] = value
    with pytest.raises(ValueError, match="conversion"):
        presolve_qp(p, replace(STRICT, dtype=torch.float32), ON)


@pytest.mark.parametrize("kind", ["nan_x", "inf_y", "dtype", "shape", "not_tensor"])
def test_restore_candidate_strict(kind):
    prepared = presolve_qp(coupled(), STRICT, ON)
    x, y = torch.zeros(1, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    if kind == "nan_x":
        x[0] = float("nan")
    elif kind == "inf_y":
        y[0] = float("inf")
    elif kind == "dtype":
        x = x.float()
    elif kind == "shape":
        y = y[:1]
    else:
        x = [0]
    with pytest.raises((ValueError, TypeError)):
        prepared.restore(x, y)


def test_explicit_cuda_unavailable_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA"):
        solve_presolved_qp(coupled(), replace(STRICT, device="cuda:0"), ON)


@pytest.mark.parametrize("option", ["remove_zero_rows", "fix_identity_equalities"])
def test_independent_opt_out_switches_and_unbounded_zero_row(option):
    p = coupled()
    prepared = presolve_qp(p, STRICT, replace(ON, **{option: False}))
    if option == "remove_zero_rows":
        assert prepared.fixed_variables.tolist() == [0]
        assert prepared.removed_rows.tolist() == [0]
        assert prepared.kept_rows.tolist() == [1, 2, 3, 4]
    else:
        assert prepared.fixed_variables.numel() == 0
        assert prepared.removed_rows.tolist() == [1]
    p = problem([[1]], [0], [[0]], [-float("inf")], [float("inf")])
    result = solve_presolved_qp(p, STRICT, ON)
    assert result.status == "solved"
    assert result.presolved.removed_rows.tolist() == [0]
    assert_complete(result.evidence)


def test_infeasibility_direction_mutation_rejected():
    p = problem([[1]], [0], [[0]], [1], [2])
    prepared = presolve_qp(p, STRICT, ON)
    prepared.infeasibility_direction.mul_(-1)
    with pytest.raises(ValueError, match="mutation"):
        prepared.validate()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; presolve CUDA is UNVERIFIED")
@pytest.mark.parametrize("kind", ["coupled", "active", "constant", "infeasible"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_presolve_original_dual_gap_full_model_sidecar(kind, dtype, tmp_path, request):
    if kind == "coupled":
        p = coupled()
    elif kind == "active":
        p = problem([[4, 1], [1, 2]], [-3, -6], [[1, 0], [1, 1]],
                    [2, -float("inf")], [2, 2])
    elif kind == "constant":
        p = problem([[4, 1], [1, 2]], [3, -1], [[1, 0], [0, 1]], [2, -1], [2, -1])
    else:
        p = problem([[2]], [0], [[1], [2]], [2, 5], [2, 9])
    p = QPProblem(*[getattr(p, name).to(device="cuda:0", dtype=dtype)
                    for name in ("P", "q", "A", "l", "u")])
    tolerance = 2e-4 if dtype == torch.float32 else 1e-7
    eps = 1e-6 if dtype == torch.float32 else 1e-10
    c = replace(STRICT, device="cuda:0", dtype=dtype, eps_abs=eps, eps_rel=eps)
    options = replace(ON, check_atol=tolerance, check_rtol=tolerance)
    result = solve_presolved_qp(p, c, options)
    torch.cuda.synchronize("cuda:0")
    assert result.x.device == result.y.device == torch.device("cuda:0")
    assert result.x.dtype == result.y.dtype == dtype
    for name in ("free_variables", "fixed_variables", "fixed_values", "fixed_rows", "kept_rows", "row_shifts"):
        assert getattr(result.presolved, name).device == torch.device("cuda:0")
    if kind == "infeasible":
        assert result.status == "infeasible_numerical"
        assert result.infeasibility_direction.device == torch.device("cuda:0")
        assert result.infeasibility_evidence.passed
    else:
        assert result.status == "solved"
        assert_complete(result.evidence)
    write_sidecar(tmp_path, request, result.presolved, result, "cuda:0")
