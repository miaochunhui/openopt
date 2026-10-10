"""Real HiGHS CPU acceptance and independent MILP negative controls."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import highspy
import numpy as np
import pytest
import torch

import openopt.integer as integer
from openopt.integer import (
    MILPConfig, MILPProblem, check_milp_candidate, solve_milp, solve_miqp,
)


def analytic():
    # Integer x=1, continuous y=1.5 is uniquely optimal, objective 1.375.
    return MILPProblem(
        q=[1.0, 0.25], A=[[2.0, 1.0]], l=[3.5], u=[np.inf],
        lower=[0.0, 0.0], upper=[2.0, 2.0], integrality=[1, 0],
    )


def scalar(**changes):
    p = MILPProblem(
        q=[1.0], A=[[1.0]], l=[0.5], u=[np.inf],
        lower=[0.0], upper=[3.0], integrality=[1],
    )
    return replace(p, **changes)


def config(**changes):
    return MILPConfig(backend="highspy", **changes)


def test_real_analytic_milp():
    p = analytic()
    result = solve_milp(p, config())
    assert result.status == result.termination_reason == "optimal"
    assert result.backend == "highspy"
    assert result.backend_version
    np.testing.assert_allclose(result.x, [1, 1.5], atol=1e-8)
    assert result.incumbent == result.objective == pytest.approx(1.375)
    assert result.dual_bound == pytest.approx(1.375)
    assert result.abs_gap == pytest.approx(0)
    assert result.rel_gap == pytest.approx(0)
    assert all(result.validation.checks.values())
    # These checks do not reuse backend feasibility, objective, gap or residuals.
    x = result.x
    assert 2 * x[0] + x[1] >= 3.5 - 1e-8
    assert np.all(x >= 0) and np.all(x <= 2)
    assert abs(x[0] - round(x[0])) <= 1e-8
    assert x[0] + 0.25 * x[1] == pytest.approx(1.375)


def test_real_binary_and_two_sided_equality():
    p = MILPProblem(
        q=[-3, -2, -1], A=[[1, 1, 1], [1, 0, -1]],
        l=[2, -np.inf], u=[2, 1], lower=[0, 0, 0], upper=[1, 1, 1],
        integrality=[1, 1, 1],
    )
    result = solve_milp(p, config())
    assert result.status == "optimal"
    np.testing.assert_allclose(result.x, [1, 1, 0], atol=1e-8)
    assert result.incumbent == result.dual_bound == pytest.approx(-5)


def test_real_signed_bounds_and_empty_constraints():
    p = scalar(q=[1], A=np.empty((0, 1)), l=[], u=[], lower=[-3.5], upper=[-0.5])
    result = solve_milp(p, config())
    assert result.status == "optimal"
    assert result.x[0] == pytest.approx(-3)
    assert result.incumbent == result.dual_bound == pytest.approx(-3)


def test_real_zero_objective():
    result = solve_milp(scalar(q=[0]), config())
    assert result.status == "optimal"
    assert result.incumbent == result.dual_bound == 0
    assert result.abs_gap == result.rel_gap == 0


def test_real_all_continuous_has_independent_lp_dual():
    result = solve_milp(scalar(integrality=[0]), config())
    assert result.status == "optimal"
    assert result.x[0] == result.incumbent == pytest.approx(0.5)
    assert result.dual_bound == pytest.approx(0.5)
    assert result.node_count is None


def test_real_lp_variable_bound_dual():
    p = scalar(A=np.empty((0, 1)), l=[], u=[], lower=[-2], upper=[3], integrality=[0])
    result = solve_milp(p, config())
    assert result.status == "optimal"
    assert result.incumbent == result.dual_bound == pytest.approx(-2)


@pytest.mark.parametrize("p", [
    scalar(l=[1.5], u=[1.5]),
    scalar(l=[4], upper=[3]),
])
def test_real_infeasible(p):
    result = solve_milp(p, config())
    assert result.status == "infeasible"
    assert result.x is result.incumbent is result.dual_bound is None
    assert result.abs_gap is result.rel_gap is result.validation is None


def test_real_unbounded():
    p = scalar(q=[-1], A=np.empty((0, 1)), l=[], u=[], upper=[np.inf])
    result = solve_milp(p, config(presolve=False))
    assert result.status == "unbounded"
    assert result.status != "optimal"
    assert result.dual_bound is None


def test_real_presolve_ambiguous_unbounded_is_not_promoted():
    p = scalar(q=[-1], A=np.empty((0, 1)), l=[], u=[], upper=[np.inf])
    result = solve_milp(p, config())
    assert result.status in ("unknown", "unbounded")
    if result.backend_status == "kUnboundedOrInfeasible":
        assert result.status == "unknown"
        assert result.termination_reason == "unbounded_or_infeasible"
    else:
        assert result.backend_status == "kUnbounded"
        assert result.status == "unbounded"


@pytest.mark.parametrize("limit", [{"time_limit": 0.0}, {"node_limit": 0}])
def test_real_zero_limits_do_not_report_optimal(limit):
    result = solve_milp(scalar(), config(presolve=False, **limit))
    assert result.status == "unknown"
    assert result.x is result.incumbent is None
    assert result.abs_gap is result.rel_gap is None
    assert result.termination_reason == ("time_limit" if "time_limit" in limit else "node_limit")


def test_real_positive_limits_may_finish_optimal():
    result = solve_milp(analytic(), config(time_limit=10, node_limit=100))
    assert result.status == "optimal"
    assert all(result.validation.checks.values())


@pytest.mark.parametrize("field,value", [
    ("q", [np.nan]), ("q", [np.inf]), ("q", [-np.inf]),
    ("A", [[np.nan]]), ("A", [[np.inf]]),
    ("l", [np.nan]), ("u", [np.nan]), ("lower", [np.nan]), ("upper", [np.nan]),
    ("l", [np.inf]), ("u", [-np.inf]), ("lower", [np.inf]), ("upper", [-np.inf]),
    ("integrality", [np.inf]), ("integrality", [np.nan]),
])
def test_nonfinite_model_rejected_before_backend(monkeypatch, field, value):
    def forbidden():
        pytest.fail("invalid model reached backend")
    monkeypatch.setattr(integer, "_load_highspy", forbidden)
    with pytest.raises(ValueError):
        solve_milp(replace(scalar(), **{field: value}), config())


@pytest.mark.parametrize("field,value", [
    ("q", []), ("q", [[1]]), ("A", [[1, 2]]), ("A", [1]),
    ("l", []), ("u", []), ("lower", [0, 1]), ("upper", []),
    ("integrality", []), ("integrality", [2]), ("integrality", [-1]),
    ("integrality", [0.5]), ("integrality", [True]),
    ("l", [2]), ("lower", [4]), ("q", [1j]), ("q", ["1"]),
])
def test_malformed_models(field, value):
    p = scalar(u=[1]) if field == "l" else scalar()
    with pytest.raises(ValueError):
        solve_milp(replace(p, **{field: value}), config())


@pytest.mark.parametrize("field,value", [
    ("backend", "gpu"), ("backend", "auto"), ("backend", "scipy"),
    ("device", "cuda"), ("device", "cuda:0"), ("device", "meta"),
    ("time_limit", np.nan), ("time_limit", np.inf), ("time_limit", -1),
    ("time_limit", True), ("node_limit", -1), ("node_limit", 1.5),
    ("node_limit", True), ("node_limit", 2**31),
    ("abs_gap", np.inf), ("abs_gap", -1), ("rel_gap", np.nan), ("rel_gap", -1),
    ("feasibility_tol", 0), ("integrality_tol", 0.5), ("objective_tol", np.inf),
    ("presolve", 1),
])
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        solve_milp(scalar(), replace(config(), **{field: value}))


def test_explicit_backend_is_required():
    with pytest.raises(TypeError):
        MILPConfig()
    with pytest.raises(TypeError):
        solve_milp(scalar())


def test_miqp_is_explicitly_unsupported():
    with pytest.raises(NotImplementedError, match="MIQP"):
        solve_milp(replace(scalar(), P=[[0]]), config())
    with pytest.raises(NotImplementedError, match="MIQP"):
        solve_miqp()
    with pytest.raises(TypeError, match="MIQP"):
        solve_milp(SimpleNamespace(P=[[1]], q=[1]), config())


def test_cpu_tensors_supported_without_mutation():
    p = analytic()
    tensors = {
        name: torch.tensor(getattr(p, name), dtype=torch.float64)
        for name in ("q", "A", "l", "u", "lower", "upper", "integrality")
    }
    before = {name: value.clone() for name, value in tensors.items()}
    result = solve_milp(MILPProblem(**tensors), config())
    assert result.status == "optimal"
    for name, value in tensors.items():
        torch.testing.assert_close(value, before[name])
    with pytest.raises(ValueError, match="CPU"):
        solve_milp(replace(p, q=torch.empty(2, device="meta")), config())
    with pytest.raises(ValueError, match="dense"):
        solve_milp(replace(p, A=tensors["A"].to_sparse()), config())


@pytest.mark.parametrize("field,value", [
    ("q", [1e20]), ("lower", [-1e20]), ("upper", [1e20]),
    ("A", [[1e-13]]), ("A", [[1e15]]),
])
def test_reference_range_does_not_silently_change_model(field, value):
    with pytest.raises(ValueError, match="reference range"):
        solve_milp(replace(scalar(), **{field: value}), config())


def test_independent_constraints_integrality_bounds_and_gap(monkeypatch):
    monkeypatch.setattr(integer, "_load_highspy", lambda: pytest.fail("checker consulted backend"))
    p = analytic()
    good = check_milp_candidate(p, [1, 1.5], dual_bound=1.375)
    assert good.feasible and all(good.checks.values())
    fractional = check_milp_candidate(p, [0.75, 2], dual_bound=1.25)
    assert fractional.checks["constraints"] and fractional.checks["bounds"]
    assert not fractional.checks["integrality"]
    row_bad = check_milp_candidate(p, [1, 1], dual_bound=1.25)
    assert row_bad.checks["integrality"] and row_bad.checks["bounds"]
    assert not row_bad.checks["constraints"]
    bound_bad = check_milp_candidate(p, [1, 2.1], dual_bound=1.525)
    assert bound_bad.checks["constraints"] and bound_bad.checks["integrality"]
    assert not bound_bad.checks["bounds"]
    gap_bad = check_milp_candidate(p, [1, 1.5], dual_bound=0)
    assert gap_bad.feasible and not gap_bad.checks["gap"]
    missing = check_milp_candidate(p, [1, 1.5])
    assert missing.feasible and not missing.checks["dual_bound_available"]
    assert missing.abs_gap is missing.rel_gap is None
    inverted = check_milp_candidate(p, [1, 1.5], dual_bound=2)
    assert inverted.signed_gap < 0 and not inverted.checks["dual_bound_order"]
    assert inverted.abs_gap == pytest.approx(0.625)


def test_unrelated_large_bounds_cannot_mask_bad_row():
    p = MILPProblem(
        q=[0], A=[[1], [1]], l=[0, -1e12], u=[1, 1e12],
        lower=[-1e12], upper=[1e12], integrality=[0],
    )
    report = check_milp_candidate(p, [1.001], dual_bound=0)
    assert not report.checks["constraints"]
    assert report.constraint_violation == pytest.approx(0.001)


@pytest.mark.parametrize("objective,bound,absolute,relative", [
    (10, 8, 2, 0.2), (-10, -12, 2, 0.2), (0, -2, 2, 2), (0.1, 0, 0.1, 0.1),
])
def test_gap_formula_and_abs_or_rel_threshold(objective, bound, absolute, relative):
    p = scalar(q=[objective], l=[1], u=[1])
    report = check_milp_candidate(p, [1], dual_bound=bound)
    assert report.abs_gap == pytest.approx(absolute)
    assert report.rel_gap == pytest.approx(relative)
    assert not report.checks["gap"]
    assert check_milp_candidate(p, [1], dual_bound=bound, abs_gap=absolute).checks["gap"]
    assert check_milp_candidate(p, [1], dual_bound=bound, rel_gap=relative).checks["gap"]


@pytest.mark.parametrize("x,bound", [([np.nan], 0), ([np.inf], 0), ([1], np.nan), ([1], np.inf)])
def test_checker_rejects_nonfinite_candidate_or_bound(x, bound):
    with pytest.raises(ValueError, match="nonfinite"):
        check_milp_candidate(scalar(), x, dual_bound=bound)


def test_checker_overflow_underflow_and_integer_precision_fail_loud():
    with pytest.raises(ValueError, match="arithmetic|nonfinite"):
        check_milp_candidate(scalar(q=[1e308], upper=[np.inf]), [2])
    with pytest.raises(ValueError, match="arithmetic"):
        check_milp_candidate(scalar(q=[1e-300]), [1e-300])
    with pytest.raises(ValueError, match="integer range"):
        check_milp_candidate(scalar(upper=[np.inf]), [float(2**53)])
    with pytest.raises(ValueError, match="dimensions"):
        check_milp_candidate(scalar(), [1, 2])


def fake_backend(monkeypatch, *, status, x=None, bound=-np.inf, objective=None,
                 info_valid=True, run_status=None, nodes=0, primal_feasible=True):
    state = {}
    native = highspy.Highs
    class Fake:
        def __init__(self):
            state["options"] = {}
        def setOptionValue(self, name, value):
            state["options"][name] = value
            return highspy.HighsStatus.kOk
        def passModel(self, lp):
            state["lp"] = lp
            return highspy.HighsStatus.kOk
        def run(self):
            state["ran"] = True
            return highspy.HighsStatus.kOk if run_status is None else run_status
        def getModelStatus(self):
            return status
        def getInfo(self):
            return SimpleNamespace(
                valid=info_valid, mip_node_count=nodes, mip_dual_bound=bound,
                primal_solution_status=(
                    highspy.kSolutionStatusFeasible if x is not None and primal_feasible
                    else highspy.kSolutionStatusNone
                ), objective_function_value=objective,
            )
        def getSolution(self):
            return SimpleNamespace(value_valid=x is not None, col_value=x, dual_valid=False)
        def version(self):
            return "controlled-negative-test"
    monkeypatch.setattr(highspy, "Highs", Fake)
    state["native"] = native
    return state


S = highspy.HighsModelStatus


@pytest.mark.parametrize("status", [
    S.kUnknown, S.kNotset, S.kTimeLimit, S.kIterationLimit,
    S.kSolutionLimit, S.kObjectiveTarget, S.kObjectiveBound, S.kInterrupt, S.kMemoryLimit,
])
@pytest.mark.parametrize("incumbent", [False, True])
def test_unknown_and_limits_never_upgrade_to_optimal(monkeypatch, status, incumbent):
    fake_backend(
        monkeypatch, status=status, x=[1] if incumbent else None,
        bound=1, objective=1 if incumbent else None,
    )
    result = solve_milp(scalar(), config(node_limit=0, time_limit=0))
    assert result.status == ("feasible" if incumbent else "unknown")
    assert result.status != "optimal"
    assert (result.incumbent is not None) == incumbent
    if incumbent:
        assert result.abs_gap == 0  # Even a zero gap cannot change the raw status.


def test_ambiguous_status_stays_unknown(monkeypatch):
    fake_backend(monkeypatch, status=S.kUnboundedOrInfeasible)
    result = solve_milp(scalar(), config())
    assert result.status == "unknown"
    assert result.termination_reason == "unbounded_or_infeasible"


def test_options_and_model_are_explicit(monkeypatch):
    state = fake_backend(monkeypatch, status=S.kOptimal, x=[1, 1.5], bound=1.375, objective=1.375)
    p = analytic()
    result = solve_milp(p, config(time_limit=2.5, node_limit=12, abs_gap=0.1, rel_gap=0.01))
    assert result.status == "optimal" and state["ran"]
    assert state["options"]["time_limit"] == 2.5
    assert state["options"]["mip_max_nodes"] == 12
    assert state["options"]["mip_abs_gap"] == 0.1
    assert state["options"]["mip_rel_gap"] == 0.01
    assert state["options"]["threads"] == 1
    lp = state["lp"]
    np.testing.assert_allclose(lp.col_cost_, p.q)
    np.testing.assert_allclose(lp.col_lower_, p.lower)
    np.testing.assert_allclose(lp.col_upper_, p.upper)
    assert lp.integrality_ == [highspy.HighsVarType.kInteger, highspy.HighsVarType.kContinuous]
    assert lp.sense_ == highspy.ObjSense.kMinimize
    np.testing.assert_array_equal(lp.a_matrix_.start_, [0, 2])
    np.testing.assert_array_equal(lp.a_matrix_.index_, [0, 1])
    np.testing.assert_array_equal(lp.a_matrix_.value_, [2, 1])


@pytest.mark.parametrize("status", [S.kOptimal, S.kTimeLimit, S.kUnknown])
@pytest.mark.parametrize("x,objective,bound", [
    ([0.5], 0.5, 0.5), ([0], 0, 0), ([4], 4, 4),
    ([np.nan], 1, 1), ([np.inf], 1, 1), ([1, 2], 1, 1),
    ([1], np.nan, 1), ([1], np.inf, 1), ([1], 2, 1),
    ([1], 1, np.nan), ([1], 1, np.inf), ([1], 1, 2),
])
def test_backend_bad_incumbents_and_bounds_fail_loud(monkeypatch, status, x, objective, bound):
    fake_backend(monkeypatch, status=status, x=x, objective=objective, bound=bound)
    with pytest.raises((ValueError, RuntimeError)):
        solve_milp(scalar(), config())


@pytest.mark.parametrize("x,bound,valid", [
    (None, 1, True), ([1], -np.inf, True), ([1], 0, True), ([1], 1, False),
])
def test_backend_optimal_needs_incumbent_bound_gap_and_valid_info(monkeypatch, x, bound, valid):
    fake_backend(monkeypatch, status=S.kOptimal, x=x, bound=bound, objective=1, info_valid=valid)
    with pytest.raises(RuntimeError, match="optimal"):
        solve_milp(scalar(), config())


def test_backend_nonoptimal_can_have_unavailable_bound(monkeypatch):
    fake_backend(monkeypatch, status=S.kTimeLimit, x=[1], objective=1)
    result = solve_milp(scalar(), config())
    assert result.status == "feasible" and result.incumbent == 1
    assert result.dual_bound is result.abs_gap is result.rel_gap is None


def test_backend_solution_valid_is_not_sufficient(monkeypatch):
    fake_backend(monkeypatch, status=S.kUnknown, x=[1], objective=1, primal_feasible=False)
    result = solve_milp(scalar(), config())
    assert result.status == "unknown" and result.x is None


def test_backend_infeasible_contradiction(monkeypatch):
    fake_backend(monkeypatch, status=S.kInfeasible, x=[1], objective=1)
    with pytest.raises(RuntimeError, match="contradicts"):
        solve_milp(scalar(), config())


@pytest.mark.parametrize("status", [S.kModelError, S.kPresolveError, S.kSolveError, S.kPostsolveError])
def test_backend_errors_surface(monkeypatch, status):
    fake_backend(monkeypatch, status=status)
    with pytest.raises(RuntimeError, match="HiGHS failed"):
        solve_milp(scalar(), config())


def test_backend_run_error_surfaces(monkeypatch):
    fake_backend(monkeypatch, status=S.kUnknown, run_status=highspy.HighsStatus.kError)
    with pytest.raises(RuntimeError, match="run failed"):
        solve_milp(scalar(), config())


@pytest.mark.parametrize("operation", ["setOptionValue", "passModel"])
def test_backend_load_and_option_failures_surface(monkeypatch, operation):
    fake_backend(monkeypatch, status=S.kUnknown)
    monkeypatch.setattr(
        highspy.Highs, operation, lambda *args: highspy.HighsStatus.kError,
    )
    with pytest.raises(RuntimeError, match=operation):
        solve_milp(scalar(), config())


def test_backend_warning_is_not_swallowed(monkeypatch):
    fake_backend(monkeypatch, status=S.kUnknown, run_status=highspy.HighsStatus.kWarning)
    with pytest.warns(RuntimeWarning, match="warning"):
        assert solve_milp(scalar(), config()).status == "unknown"


def test_backend_missing_is_fail_loud_without_fallback(monkeypatch):
    import builtins
    original = builtins.__import__
    def missing(name, *args, **kwargs):
        if name == "highspy":
            raise ModuleNotFoundError("controlled missing backend")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(ImportError, match="no fallback") as error:
        solve_milp(scalar(), config())
    assert isinstance(error.value.__cause__, ModuleNotFoundError)


def test_external_acceptance_artifact(tmp_path, record_property):
    root = Path(__file__).resolve().parents[1]
    assert not tmp_path.resolve().is_relative_to(root)
    p = analytic()
    result = solve_milp(p, config())
    unbounded_problem = scalar(q=[-1], A=np.empty((0, 1)), l=[], u=[], upper=[np.inf])
    unbounded = solve_milp(unbounded_problem, config(presolve=False))
    presolved = solve_milp(unbounded_problem, config())
    infeasible = solve_milp(scalar(l=[1.5], u=[1.5]), config())
    with pytest.warns(RuntimeWarning, match="HiGHS run warning"):
        time_stop = solve_milp(scalar(), config(presolve=False, time_limit=0))
    with pytest.warns(RuntimeWarning, match="HiGHS run warning"):
        node_stop = solve_milp(scalar(), config(presolve=False, node_limit=0))
    cases = {
        "analytic_optimal": result.status == "optimal",
        "fractional_rejected": not check_milp_candidate(p, [0.75, 2]).checks["integrality"],
        "constraint_rejected": not check_milp_candidate(p, [1, 1]).checks["constraints"],
        "bounds_rejected": not check_milp_candidate(p, [1, 2.1]).checks["bounds"],
        "gap_rejected": not check_milp_candidate(p, [1, 1.5], dual_bound=0).checks["gap"],
        "real_unbounded_without_presolve": unbounded.status == "unbounded",
        "real_presolve_ambiguity_preserved": (
            presolved.status == "unknown" and presolved.backend_status == "kUnboundedOrInfeasible"
        ) or (presolved.status == "unbounded" and presolved.backend_status == "kUnbounded"),
        "real_infeasible": infeasible.status == "infeasible",
        "real_time_limit": time_stop.status == "unknown" and time_stop.termination_reason == "time_limit",
        "real_node_limit": node_stop.status == "unknown" and node_stop.termination_reason == "node_limit",
    }
    report = {
        "schema_version": 1, "verdict": "PASS" if all(cases.values()) else "FAIL",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "backend": result.backend, "backend_version": result.backend_version,
        "qualification": result.qualification, "cases": cases,
        "incumbent": result.incumbent, "dual_bound": result.dual_bound,
        "abs_gap": result.abs_gap, "rel_gap": result.rel_gap,
        "validation": asdict(result.validation), "x": result.x.tolist(),
        "real_terminations": {
            name: {
                "status": item.status, "backend_status": item.backend_status,
                "termination_reason": item.termination_reason,
                "incumbent": item.incumbent, "dual_bound": item.dual_bound,
                "node_count": item.node_count,
            } for name, item in (
                ("unbounded_no_presolve", unbounded), ("unbounded_presolved", presolved),
                ("infeasible", infeasible), ("time_limit_zero", time_stop),
                ("node_limit_zero", node_stop),
            )
        },
        "source_sha256": {
            path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in ("src/openopt/integer.py", "tests/test_integer.py", "docs/research/integer.md")
        },
    }
    artifact = tmp_path / "integer-acceptance.json"
    with artifact.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert json.loads(artifact.read_text()) == report
    assert report["verdict"] == "PASS" and len(cases) == 10
    record_property("integer_acceptance_artifact", str(artifact))
