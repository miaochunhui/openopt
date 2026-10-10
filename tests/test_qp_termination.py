"""CPU original-coordinate termination, independent replay and frozen controls."""

from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from openopt.certificates import (
    check_primal_infeasibility, check_qp_candidate, check_unbounded_direction,
)
from openopt.corpus import (
    check_infeasibility, check_unboundedness, generate_case, validate_oracle,
)
from openopt.qp import QPConfig, QPProblem, prepare_qp, solve_qp

BASELINE = "a698c6744a0c5a667217ea683ac5c898d65a0d67"


def qp(P, q, A, l, u):
    return QPProblem(*[
        torch.tensor(v, dtype=torch.float64) for v in (P, q, A, l, u)
    ])


@pytest.fixture(scope="module")
def frozen_baseline():
    root = Path(__file__).resolve().parents[1]
    if not (root / ".git").exists():
        pytest.skip("frozen bitwise baseline requires .git; unavailable in source-only GPU snapshot")
    source = subprocess.run(
        ["git", "show", f"{BASELINE}:src/openopt/qp.py"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout
    assert "direction_termination" not in source
    spec = importlib.util.spec_from_loader("_termination_frozen_qp", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        exec(compile(source, f"{BASELINE}:src/openopt/qp.py", "exec"), module.__dict__)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
@pytest.mark.parametrize("adaptive", [False, True])
@pytest.mark.parametrize("family", ["dense_spd", "lp", "psd_singular", "infeasible", "unbounded"])
def test_disabled_bitwise_baseline(frozen_baseline, dtype, linear_solver, adaptive, family):
    p = generate_case(family, 8).problem()
    c = QPConfig(
        dtype=dtype, linear_solver=linear_solver, adaptive_rho=adaptive,
        adaptive_rho_interval=7, max_iterations=100,
    )
    old = frozen_baseline.QPConfig(**{
        f.name: getattr(c, f.name) for f in fields(frozen_baseline.QPConfig)
    })
    expected, actual = frozen_baseline.solve_qp(p, old), solve_qp(p, c)
    for f in fields(frozen_baseline.QPResult):
        a, b = getattr(actual, f.name), getattr(expected, f.name)
        assert torch.equal(a, b) if isinstance(a, torch.Tensor) else a == b
    assert actual.termination_evidence is None
    assert actual.dual_objective is actual.duality_gap is None
    assert actual.infeasibility_direction is actual.unbounded_direction is None


@pytest.mark.parametrize("family", ["dense_spd", "lp", "psd_singular", "equality_redundant"])
@pytest.mark.parametrize("adaptive", [False, True])
def test_gap_solved_only_after_independent_original_replay(family, adaptive):
    p = generate_case(family, 8).problem()
    c = QPConfig(dual_gap_termination=True, adaptive_rho=adaptive)
    result = solve_qp(p, c)
    replay = check_qp_candidate(p, result.x, result.y, atol=c.eps_abs, rtol=c.eps_rel)
    assert result.status == "solved"
    assert replay.passed and result.termination_evidence.to_dict() == replay.to_dict()
    assert result.dual_objective == replay.metrics["dual_objective"]
    assert result.duality_gap == replay.metrics["duality_gap"]
    assert abs(result.duality_gap) <= replay.thresholds["duality_gap"]


def test_gap_blocks_residual_only_false_solved():
    # The constrained optimum is zero: relative gap cannot hide the offset error.
    p = qp([[1]], [-5e3], [[1]], [1e4], [1e4])
    c = QPConfig(eps_abs=1e-7, eps_rel=1e-7, check_interval=1, max_iterations=20)
    fixed = solve_qp(p, c)
    guarded = solve_qp(p, replace(c, dual_gap_termination=True))
    assert fixed.status == "solved"
    fixed_evidence = check_qp_candidate(p, fixed.x, fixed.y, atol=c.eps_abs, rtol=c.eps_rel)
    assert not fixed_evidence.checks["duality_gap"]
    assert guarded.iterations > fixed.iterations
    assert guarded.status == "max_iterations" or guarded.termination_evidence.passed


@pytest.mark.parametrize("kind", ["unresolved", "range"])
def test_singular_dual_uncertainty_remains_max_iterations(kind):
    P, q = np.diag([1., 1e-16]), [0., 0.]
    if kind == "range":
        P, q = np.diag([1., 0.]), [0., 1e-9]
    p = qp(P, q, np.empty((0, 2)), [], [])
    result = solve_qp(p, QPConfig(
        dual_gap_termination=True, eps_abs=1e-12 if kind == "range" else 1e-7,
        max_iterations=25,
    ))
    assert result.status == "max_iterations"
    assert result.dual_objective is result.duality_gap is None
    assert not result.termination_evidence.passed
    assert "dual_objective" in result.termination_evidence.unavailable


@pytest.mark.parametrize("family", ["infeasible", "unbounded"])
@pytest.mark.parametrize("n", [3, 8, 24, 64, 128])
@pytest.mark.parametrize("seed", [20261010, 20261011])
@pytest.mark.parametrize("adaptive", [False, True])
def test_known_direction_corpus(family, n, seed, adaptive, tmp_path, request):
    case = generate_case(family, n, seed)
    oracle = validate_oracle(case)
    p = case.problem()
    c = QPConfig(
        direction_termination=True, dual_gap_termination=True,
        adaptive_rho=adaptive, adaptive_rho_interval=7, max_iterations=2000,
    )
    result = solve_qp(p, c)
    assert result.status == family
    if family == "infeasible":
        direction = result.infeasibility_direction
        independent = check_primal_infeasibility(
            p, direction, atol=c.direction_atol, rtol=c.direction_rtol,
        )
        second = check_infeasibility(case, direction.numpy(), tol=c.direction_atol)
        assert result.unbounded_direction is result.feasible_base is None
    else:
        direction = result.unbounded_direction
        independent = check_unbounded_direction(
            p, direction, feasible_x=result.feasible_base,
            atol=c.direction_atol, rtol=c.direction_rtol,
        )
        second = check_unboundedness(
            case, direction.numpy(), result.feasible_base.numpy(), tol=c.direction_atol,
        )
        assert result.infeasibility_direction is None
    assert direction.device.type == "cpu" and direction.dtype == torch.float64
    assert float(direction.abs().max()) == 1
    assert independent.passed and second["verdict"] == "PASS"
    assert not (
        check_primal_infeasibility(p, -direction).passed if family == "infeasible"
        else check_unbounded_direction(p, -direction, feasible_x=result.feasible_base).passed
    )
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(), "owner": "termination",
        "verdict": "PASS", "case_id": case.case_id, "problem_hash": case.problem_hash,
        "adaptive_rho": adaptive, "iterations": result.iterations,
        "rho_updates": result.rho_updates, "final_rho": result.final_rho,
        "oracle": oracle, "solver_evidence": result.termination_evidence.to_dict(),
        "independent_replay": independent.to_dict(), "second_checker": second,
        "source_sha256": {
            path: hashlib.sha256((Path(__file__).resolve().parents[1] / path).read_bytes()).hexdigest()
            for path in ("src/openopt/qp.py", "src/openopt/certificates.py", "src/openopt/corpus.py")
        },
    }
    vectors = {"direction": direction.numpy()}
    if result.feasible_base is not None:
        vectors["feasible_base"] = result.feasible_base.numpy()
    np.savez(tmp_path / "vectors.npz", **vectors)
    with np.load(tmp_path / "vectors.npz", allow_pickle=False) as saved:
        assert set(saved.files) == set(vectors)
        assert all(np.array_equal(saved[name], v) for name, v in vectors.items())
    artifact = tmp_path / "direction-acceptance.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    assert json.loads(artifact.read_text()) == report
    request.node.user_properties.extend([
        ("direction_acceptance_path", str(artifact)), ("case_id", case.case_id),
        ("status", result.status), ("iterations", result.iterations),
    ])


@pytest.mark.parametrize("sign", [-1, 1])
def test_one_sided_recession_bounds_signs(sign):
    p = qp(
        np.zeros((2, 2)), [0, -sign],
        [[1, 0], [0, sign], [0, -sign], [0, 2 * sign]],
        [0, 2, -np.inf, -np.inf], [0, np.inf, -2, np.inf],
    )
    result = solve_qp(p, QPConfig(direction_termination=True))
    assert result.status == "unbounded"
    assert check_unbounded_direction(
        p, result.unbounded_direction, feasible_x=result.feasible_base,
    ).passed


@pytest.mark.parametrize("P", [np.diag([1., 1e-16]), np.diag([0., 1e-12])])
def test_small_positive_curvature_never_unbounded(P):
    p = qp(P, [0, -1], np.empty((0, 2)), [], [])
    result = solve_qp(p, QPConfig(direction_termination=True, max_iterations=50))
    assert result.status == "max_iterations"
    assert result.unbounded_direction is None


def test_descent_without_feasible_base_never_unbounded():
    p = qp(np.zeros((2, 2)), [0, -1], [[1, 0], [1, 0]], [0, 2], [1, 3])
    result = solve_qp(p, QPConfig(direction_termination=True))
    assert result.status == "infeasible"
    assert result.unbounded_direction is result.feasible_base is None


def test_direction_generation_crosses_real_adaptive_rho_updates():
    p = generate_case("infeasible", 8, 20261011).problem()
    c = QPConfig(
        direction_termination=True, adaptive_rho=True, rho=1e-4,
        adaptive_rho_interval=1, adaptive_rho_tolerance=1.1, max_iterations=2000,
    )
    result = solve_qp(p, c)
    assert result.rho_updates > 0
    assert result.final_rho != c.rho
    assert result.status == "infeasible"
    assert check_primal_infeasibility(p, result.infeasibility_direction).passed


def test_checks_are_original_precast_data_and_workspace_owns_inputs(monkeypatch):
    p = qp(np.diag([2., 0.]), [-4., 0.], np.empty((0, 2)), [], [])
    workspace = prepare_qp(p, QPConfig(dual_gap_termination=True, dtype=torch.float32))
    p.q.add_(100)
    import openopt.certificates as certificates
    original = certificates.check_qp_candidate
    calls = []

    def track(problem, x, y, **kwargs):
        assert problem.P.dtype == problem.q.dtype == torch.float64
        assert torch.equal(problem.q, torch.tensor([-4., 0.], dtype=torch.float64))
        assert float(problem.P[1, 1]) == 0
        assert x.device.type == y.device.type == "cpu"
        calls.append(1)
        return original(problem, x, y, **kwargs)

    monkeypatch.setattr(certificates, "check_qp_candidate", track)
    first, second = workspace.solve(), workspace.solve()
    assert calls and first.status == second.status == "solved"
    assert torch.equal(first.x, second.x)


@pytest.mark.parametrize("kind", ["gap", "infeasible", "unbounded"])
def test_independent_failure_is_loud_with_first_cause(monkeypatch, kind):
    import openopt.certificates as certificates
    names = {
        "gap": "check_qp_candidate", "infeasible": "check_primal_infeasibility",
        "unbounded": "check_unbounded_direction",
    }

    def fail(*args, **kwargs):
        raise ValueError("injected independent arithmetic first cause")

    monkeypatch.setattr(certificates, names[kind], fail)
    family = "dense_spd" if kind == "gap" else kind
    with pytest.raises(RuntimeError, match=f"kind={kind} iteration=.*rho=") as error:
        solve_qp(generate_case(family, 8).problem(), QPConfig(
            dual_gap_termination=kind == "gap", direction_termination=kind != "gap",
        ))
    assert "first cause" in str(error.value.__cause__)


def test_conflicting_evidence_remains_ambiguous(monkeypatch):
    import openopt.certificates as certificates
    p = generate_case("infeasible", 8).problem()
    good = check_primal_infeasibility(p, torch.tensor([1., -1.] + [0.] * 8))
    assert good.passed
    monkeypatch.setattr(certificates, "check_primal_infeasibility", lambda *a, **kw: good)
    monkeypatch.setattr(certificates, "check_unbounded_direction", lambda *a, **kw: good)
    result = solve_qp(p, QPConfig(direction_termination=True, max_iterations=50))
    assert result.status == "max_iterations"
    assert result.infeasibility_direction is result.unbounded_direction is None


@pytest.mark.parametrize("family", ["infeasible", "unbounded"])
def test_rejected_direction_never_promotes_status(monkeypatch, family):
    import openopt.certificates as certificates
    p = generate_case(family, 8).problem()
    if family == "infeasible":
        bad = check_primal_infeasibility(p, torch.zeros_like(p.l))
        name = "check_primal_infeasibility"
    else:
        bad = check_unbounded_direction(p, torch.zeros_like(p.q), feasible_x=torch.zeros_like(p.q))
        name = "check_unbounded_direction"
    assert not bad.passed
    monkeypatch.setattr(certificates, name, lambda *a, **kw: bad)
    result = solve_qp(p, QPConfig(direction_termination=True, max_iterations=100))
    assert result.status == "max_iterations"
    assert result.infeasibility_direction is result.unbounded_direction is None


@pytest.mark.parametrize("family", ["dense_spd", "lp", "psd_singular", "equality_redundant"])
def test_directions_do_not_misclassify_finite_corpus(family):
    p = generate_case(family, 24).problem()
    result = solve_qp(p, QPConfig(
        dual_gap_termination=True, direction_termination=True, adaptive_rho=True,
    ))
    assert result.status == "solved"
    assert result.termination_evidence.passed
    assert result.infeasibility_direction is result.unbounded_direction is None


@pytest.mark.parametrize("option", ["dual_gap_termination", "direction_termination"])
def test_nonfinite_opt_in_iterate_is_fail_loud(monkeypatch, option):
    workspace = prepare_qp(
        generate_case("dense_spd", 3).problem(), replace(QPConfig(), **{option: True}),
    )
    monkeypatch.setattr(torch, "cholesky_solve", lambda rhs, factor: torch.full_like(rhs, np.nan))
    with pytest.raises(RuntimeError, match="nonfinite termination iterate iteration=25 rho="):
        workspace.solve()


@pytest.mark.parametrize("field,value", [
    ("dual_gap_termination", 1), ("direction_termination", 1),
    ("direction_atol", 0), ("direction_atol", True), ("direction_atol", np.inf),
    ("direction_rtol", -1), ("direction_rtol", True), ("direction_rtol", 1),
    ("direction_rtol", np.nan), ("eps_rel", 1),
])
def test_invalid_termination_config(field, value):
    with pytest.raises(ValueError):
        prepare_qp(generate_case("dense_spd", 3).problem(), replace(
            QPConfig(dual_gap_termination=True), **{field: value},
        ))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; opt-in gap unverified")
@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
@pytest.mark.parametrize("family", ["dense_spd", "lp", "psd_singular"])
def test_cuda_opt_in_gap_uses_explicit_device_and_independent_replay(family, linear_solver):
    p = generate_case(family, 24).problem()
    c = QPConfig(
        device="cuda:0", dual_gap_termination=True, adaptive_rho=True,
        linear_solver=linear_solver,
    )
    result = solve_qp(p, c)
    assert result.x.device == result.y.device == torch.device("cuda:0")
    assert result.status == "solved" and result.termination_evidence.passed
    replay = check_qp_candidate(
        p, result.x.cpu(), result.y.cpu(), atol=c.eps_abs, rtol=c.eps_rel,
    )
    assert replay.passed
    assert replay.to_dict() == result.termination_evidence.to_dict()
    assert result.duality_gap == replay.metrics["duality_gap"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; opt-in directions unverified")
@pytest.mark.parametrize("family", ["infeasible", "unbounded"])
@pytest.mark.parametrize("seed", [20261010, 20261011])
@pytest.mark.parametrize("adaptive", [False, True])
def test_cuda_opt_in_direction_status_and_independent_replay(family, seed, adaptive):
    p = generate_case(family, 24, seed).problem()
    c = QPConfig(
        device="cuda:0", dual_gap_termination=True, direction_termination=True,
        adaptive_rho=adaptive, adaptive_rho_interval=7, max_iterations=2000,
    )
    result = solve_qp(p, c)
    assert result.x.device == result.y.device == torch.device("cuda:0")
    assert result.status == family and result.termination_evidence.passed
    if family == "infeasible":
        assert result.infeasibility_direction.device.type == "cpu"
        replay = check_primal_infeasibility(
            p, result.infeasibility_direction, atol=c.direction_atol, rtol=c.direction_rtol,
        )
    else:
        assert result.unbounded_direction.device.type == result.feasible_base.device.type == "cpu"
        replay = check_unbounded_direction(
            p, result.unbounded_direction, feasible_x=result.feasible_base,
            atol=c.direction_atol, rtol=c.direction_rtol,
        )
    assert replay.passed


@pytest.mark.parametrize("option", ["dual_gap_termination", "direction_termination"])
def test_opt_in_never_falls_back_from_explicit_cuda(monkeypatch, option):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA requested but unavailable"):
        solve_qp(
            generate_case("dense_spd", 3).problem(),
            replace(QPConfig(device="cuda:0"), **{option: True}),
        )
