"""Analytic and real Clarabel acceptance for the explicit conic CPU contract.

Running this file as a script probes a fresh import with optional dependencies
blocked. Pytest writes case/source evidence only into its external temp folder.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from xml.etree import ElementTree

from openopt.conic import (
    ConicConfig, ConicProblem, Nonnegative, SOC, Zero,
    check_cone_feasibility, check_conic_candidate, solve_conic,
)

REFERENCE = ConicConfig(backend="clarabel_reference", eps_abs=1e-8, eps_rel=1e-8)


def known_soc() -> ConicProblem:
    """min t subject to u=3, v=4, (t,u,v) in SOC; optimum is (5,3,4)."""
    return ConicProblem(
        np.zeros((3, 3)), np.array([1., 0., 0.]),
        np.vstack([[[0., 1., 0.], [0., 0., 1.]], -np.eye(3)]),
        np.array([3., 4., 0., 0., 0.]), (Zero(2), SOC(3)),
    )


def bounded_qp() -> ConicProblem:
    """min .5||x||^2 - .2*x0 - 3*x1, x0+x1=1, x>=0; optimum (0,1)."""
    return ConicProblem(
        np.eye(2), np.array([-.2, -3.]),
        np.array([[1., 1.], [-1., 0.], [0., -1.]]),
        np.array([1., 0., 0.]), (Zero(1), Nonnegative(2)),
    )


def unconstrained(P=None, q=None) -> ConicProblem:
    return ConicProblem(
        np.diag([2., 4.]) if P is None else np.asarray(P),
        np.array([-4., 8.]) if q is None else np.asarray(q),
        np.empty((0, 2)), np.empty(0), (),
    )


def infeasible() -> ConicProblem:
    return ConicProblem(
        np.zeros((1, 1)), np.zeros(1), np.array([[1.], [-1.]]),
        np.array([0., -1.]), (Nonnegative(2),),
    )


def unbounded() -> ConicProblem:
    return ConicProblem(
        np.zeros((1, 1)), np.array([-1.]), np.array([[-1.]]),
        np.zeros(1), (Nonnegative(1),),
    )


def mixed_soc_qp() -> ConicProblem:
    """Projection of (0,3,4) on SOC, plus fixed w=-2 and inactive w<=0."""
    return ConicProblem(
        np.eye(4), np.array([0., -3., -4., 0.]),
        np.vstack([np.hstack([-np.eye(3), np.zeros((3, 1))]), [[0., 0., 0., 1.]] * 2]),
        np.array([0., 0., 0., 0., -2.]), (SOC(3), Nonnegative(1), Zero(1)),
    )


@pytest.mark.parametrize("kind,dimension", [(Zero, 0), (Nonnegative, -1), (SOC, 1), (SOC, True), (Zero, 2.5)])
def test_invalid_cone_dimensions(kind, dimension):
    with pytest.raises(ValueError, match="cone dimension"):
        kind(dimension)


@pytest.mark.parametrize("value,cone,dual,expected", [
    ([0., 0.], Zero(2), False, True),
    ([-3., 4.], Zero(2), True, True),
    ([0., 1.], Zero(2), False, False),
    ([1., 0.], Nonnegative(2), False, True),
    ([1., -1.], Nonnegative(2), False, False),
    ([1., -1.], Nonnegative(2), True, False),
    ([5., 3., 4.], SOC(3), False, True),
    ([5., -3., -4.], SOC(3), True, True),
    ([4., 3., 4.], SOC(3), False, False),
    ([-5., 3., 4.], SOC(3), True, False),
    ([0., 0.], SOC(2), False, True),
])
def test_cone_membership_and_dual_contract(value, cone, dual, expected):
    evidence = check_cone_feasibility(value, (cone,), dual=dual, atol=1e-12, rtol=0)
    assert evidence.passed is expected
    assert json.loads(json.dumps(evidence.to_dict(), allow_nan=False))["passed"] is expected


def test_unrelated_large_block_cannot_mask_small_cone_violation():
    assert not check_cone_feasibility([1e20, -1.], (Nonnegative(2),)).passed
    assert not check_cone_feasibility([1e20, 0., 0., -1.], (SOC(2), SOC(2))).passed
    assert not check_cone_feasibility([1e20, 1.], (Nonnegative(1), Zero(1))).passed


@pytest.mark.parametrize("value,cones", [
    ([1.], (SOC(2),)), ([[1., 0.]], (SOC(2),)),
    ([1.], ("nonnegative",)), ([1.], None), ([True], (Nonnegative(1),)),
])
def test_malformed_cone_inputs_raise(value, cones):
    with pytest.raises((TypeError, ValueError)):
        check_cone_feasibility(value, cones)


def test_analytic_soc_primal_dual_and_gap():
    report = check_conic_candidate(
        known_soc(), [5, 3, 4], [0, 0, 5, 3, 4], [-.6, -.8, 1, -.6, -.8],
        atol=1e-12, rtol=0,
    )
    assert report.passed
    assert report.metrics["primal_objective"] == 5
    assert report.metrics["dual_objective"] == 5
    assert report.metrics["duality_gap"] == 0
    assert report.metrics["stationarity"] < 1e-15


def test_qp_analytic_and_unrestricted_equality_multiplier():
    report = check_conic_candidate(bounded_qp(), [0, 1], [0, 0, 1], [2, 1.8, 0])
    assert report.passed
    assert report.metrics["primal_objective"] == -2.5
    assert report.metrics["dual_objective"] == -2.5
    p = ConicProblem(np.eye(1), np.array([3.]), np.eye(1), np.ones(1), (Zero(1),))
    assert check_conic_candidate(p, [1], [0], [-4]).passed


def test_mixed_cone_order_and_quadratic_soc_analytic_witness():
    evidence = check_conic_candidate(
        mixed_soc_qp(), [2.5, 1.5, 2., -2.], [2.5, 1.5, 2., 2., 0.],
        [2.5, -1.5, -2., 0., 2.],
    )
    assert evidence.passed
    assert evidence.metrics["primal_objective"] == -4.25
    assert evidence.metrics["dual_objective"] == -4.25
    assert evidence.metrics["duality_gap"] == 0


def test_repeated_soc_blocks_real_planted_primal_dual():
    rng = np.random.default_rng(811)
    x = rng.normal(size=6)
    B = rng.normal(size=(6, 6))
    P = np.eye(6) + B.T @ B
    A = rng.normal(size=(8, 6))
    s = np.array([5., 3., 4., 2., 2., 0., 0., 1.])
    z = np.array([1., -.6, -.8, .5, -.5, -2., 2., 0.])
    p = ConicProblem(P, -P @ x - A.T @ z, A, A @ x + s, (SOC(3), SOC(2), Zero(1), Nonnegative(2)))
    assert check_conic_candidate(p, x, s, z).passed
    result = solve_conic(p, REFERENCE)
    # A global relative objective gap can leave a small cone block outside its
    # independent local complementarity limit. Retain that rejection as evidence.
    assert result.backend_status == "Solved"
    assert result.status == "reference_candidate_rejected"
    assert not result.evidence.checks["complementarity_block_1"]
    result = solve_conic(p, replace(REFERENCE, eps_abs=1e-9, eps_rel=0))
    assert result.status == "reference_solved", result.evidence.to_dict()
    assert np.allclose(result.x, x, atol=2e-5, rtol=0)


@pytest.mark.parametrize("mutation", ["x", "s", "dual_sign", "axis", "stationarity"])
def test_analytic_soc_mutation_controls(mutation):
    x, s, z = [5., 3., 4.], [0., 0., 5., 3., 4.], [-.6, -.8, 1., -.6, -.8]
    if mutation == "x":
        x[0] = 4
    elif mutation == "s":
        s[0] = 1
    elif mutation == "dual_sign":
        z = [-value for value in z]
    elif mutation == "axis":
        s[2], s[4] = s[4], s[2]
    else:
        z[0] += .1
    assert not check_conic_candidate(known_soc(), x, s, z).passed


def test_dual_objective_does_not_use_candidate_quadratic_when_not_stationary():
    p = unconstrained()
    bad = check_conic_candidate(p, [0, 0], [], [])
    assert not bad.passed and not bad.checks["stationarity"]
    assert bad.metrics["primal_objective"] == 0
    assert bad.metrics["dual_objective"] == -12
    assert bad.metrics["duality_gap"] == 12
    assert check_conic_candidate(p, [2, -2], [], []).passed


def test_signed_negative_gap_is_not_clipped():
    p = ConicProblem(np.zeros((1, 1)), np.array([1.]), np.array([[-1.]]), np.zeros(1), (Nonnegative(1),))
    bad = check_conic_candidate(p, [-1], [-1], [1])
    assert not bad.passed
    assert bad.metrics["duality_gap"] == -1


def test_singular_psd_and_lp_range_are_checked_not_regularized():
    p = unconstrained(P=np.diag([2., 0.]), q=[-4., 0.])
    assert check_conic_candidate(p, [2, 100], [], []).passed
    out_of_range = check_conic_candidate(replace(p, q=np.array([-4., 1.])), [2, 0], [], [])
    assert not out_of_range.passed and out_of_range.metrics["dual_objective"] is None
    assert "range(P)" in out_of_range.unavailable["dual_objective"]
    unresolved = check_conic_candidate(unconstrained(P=np.diag([1., 1e-16])), [4, -8e16], [], [])
    assert not unresolved.passed and unresolved.metrics["dual_objective"] is None
    assert "unresolved PSD" in unresolved.unavailable["dual_objective"]
    tiny = ConicProblem(np.array([[1e-12]]), np.array([-1.]), np.empty((0, 1)), np.empty(0), ())
    assert check_conic_candidate(tiny, [1e12], [], []).passed


def test_componentwise_equation_and_stationarity_not_global_scale():
    p = unconstrained(P=np.eye(2), q=[-1e20, 0.])
    assert not check_conic_candidate(p, [1e20, 1], [], []).checks["stationarity"]
    p = ConicProblem(np.zeros((2, 2)), np.zeros(2), np.eye(2), np.array([1e20, 0.]), (Zero(2),))
    bad = check_conic_candidate(p, [1e20, 1], [0, 0], [0, 0])
    assert not bad.checks["primal_equation"] and not bad.checks["original_slack_cone"]


def test_complementarity_checked_per_block_without_cancellation():
    p = ConicProblem(
        np.zeros((1, 1)), np.zeros(1), np.zeros((2, 1)),
        np.array([1., -1.]), (Zero(1), Zero(1)),
    )
    bad = check_conic_candidate(p, [0], [1, -1], [1, 1])
    assert bad.metrics["duality_gap"] == 0
    assert not bad.checks["complementarity_block_0"]
    assert not bad.checks["complementarity_block_1"] and not bad.passed


@pytest.mark.parametrize("field", ["P", "q", "A", "b"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_model_raises(field, value):
    p = known_soc()
    array = getattr(p, field).copy()
    array.flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        check_conic_candidate(replace(p, **{field: array}), [5, 3, 4], [0] * 5, [0] * 5)


@pytest.mark.parametrize("field", ["x", "s", "z"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_candidate_raises(field, value):
    candidates = {"x": [5., 3., 4.], "s": [0., 0., 5., 3., 4.], "z": [-.6, -.8, 1., -.6, -.8]}
    candidates[field][0] = value
    with pytest.raises(ValueError, match="finite"):
        check_conic_candidate(known_soc(), **candidates)


@pytest.mark.parametrize("field,value", [
    ("P", -np.eye(3)), ("P", np.eye(2)), ("q", np.empty(0)),
    ("q", np.ones((3, 1))), ("A", np.empty((5, 2))),
    ("b", np.ones(4)), ("cones", (SOC(3),)), ("cones", ("exp",)),
    ("P", np.eye(3, dtype=np.complex128)), ("P", torch.eye(3).to_sparse()),
    ("P", torch.empty((3, 3), device="meta")),
])
def test_invalid_model_raises_before_loading_backend(field, value, monkeypatch):
    def forbidden():
        raise AssertionError("invalid model must not load backend")
    monkeypatch.setattr("openopt.conic._reference_dependencies", forbidden)
    with pytest.raises((ValueError, TypeError)):
        solve_conic(replace(known_soc(), **{field: value}), REFERENCE)


def test_full_symmetric_P_required_not_upper_triangle():
    P = np.array([[2., 1., 0.], [0., 2., 0.], [0., 0., 2.]])
    with pytest.raises(ValueError, match="full symmetric"):
        solve_conic(replace(known_soc(), P=P), REFERENCE)


@pytest.mark.parametrize("kwargs", [
    {"atol": 0}, {"atol": True}, {"atol": np.inf}, {"atol": np.nan},
    {"rtol": -1}, {"rtol": 1}, {"rtol": True}, {"rtol": np.nan},
])
def test_invalid_checker_tolerances(kwargs):
    with pytest.raises(ValueError):
        check_cone_feasibility([1, 0], (SOC(2),), **kwargs)


@pytest.mark.parametrize("kwargs", [
    {}, {"backend": "auto"}, {"backend": "clarabel"}, {"backend": "native"},
    {"backend": "clarabel_reference", "device": "meta"},
    {"backend": "clarabel_reference", "max_iterations": 0},
    {"backend": "clarabel_reference", "max_iterations": True},
    {"backend": "clarabel_reference", "eps_abs": np.nan},
    {"backend": "clarabel_reference", "eps_rel": 1},
])
def test_invalid_config_never_implicitly_routes(kwargs):
    with pytest.raises((ValueError, RuntimeError)):
        solve_conic(known_soc(), ConicConfig(**kwargs))


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:99"])
@pytest.mark.parametrize("available", [True, False])
def test_explicit_cuda_refused_even_if_hardware_available(device, available, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    def forbidden():
        raise AssertionError("CUDA rejection must precede loading CPU reference")
    monkeypatch.setattr("openopt.conic._reference_dependencies", forbidden)
    with pytest.raises(RuntimeError, match="explicit CUDA"):
        solve_conic(known_soc(), replace(REFERENCE, device=device))


@pytest.mark.parametrize("module", ["clarabel", "scipy.sparse"])
def test_missing_optional_dependency_is_strict_and_preserves_cause(module, monkeypatch):
    original = importlib.import_module
    cause = ModuleNotFoundError(f"blocked dependency {module}", name=module)
    def blocked(name, *args, **kwargs):
        if name == module:
            raise cause
        return original(name, *args, **kwargs)
    monkeypatch.setattr(importlib, "import_module", blocked)
    with pytest.raises(RuntimeError, match="optional clarabel and scipy") as raised:
        solve_conic(known_soc(), REFERENCE)
    assert raised.value.__cause__ is cause
    assert check_cone_feasibility([5, 3, 4], (SOC(3),)).passed


def test_import_and_checker_are_lazy_in_fresh_process(tmp_path):
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--lazy-probe"],
        env=env, capture_output=True, text=True, check=False,
    )
    artifact = tmp_path / "lazy-probe.log"
    artifact.write_text(f"rc={result.returncode}\n{result.stdout}\n{result.stderr}", encoding="utf-8")
    assert result.returncode == 0, artifact.read_text()
    assert "CONIC_LAZY_PROBE PASS" in result.stdout


def test_checker_is_independent_of_backend_and_solver_metrics(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("independent checker consulted a backend/factorization")
    monkeypatch.setattr("openopt.conic._reference_dependencies", forbidden)
    monkeypatch.setattr(np.linalg, "inv", forbidden)
    monkeypatch.setattr(np.linalg, "pinv", forbidden)
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    assert check_conic_candidate(known_soc(), [5, 3, 4], [0, 0, 5, 3, 4], [-.6, -.8, 1, -.6, -.8]).passed


def test_overflow_and_soc_extreme_range_are_fail_loud():
    p = ConicProblem(np.array([[1e308]]), np.zeros(1), np.empty((0, 1)), np.empty(0), ())
    with pytest.raises(ValueError, match="arithmetic failed"):
        check_conic_candidate(p, [1e308], [], [])
    assert check_cone_feasibility([1e300, 6e299, 8e299], (SOC(3),)).passed
    # Stable hypot need not underflow merely because its inputs differ in scale.
    assert not check_cone_feasibility([1e-300, 1e300, 1e-300], (SOC(3),)).passed
    with pytest.raises(ValueError, match="arithmetic failed"):
        check_cone_feasibility([0., 1.7e308, 1.7e308], (SOC(3),))
    p = ConicProblem(np.zeros((1, 1)), np.zeros(1), np.zeros((1, 1)), np.zeros(1), (Nonnegative(1),))
    with pytest.raises(ValueError, match="arithmetic failed"):
        check_conic_candidate(p, [0], [1e-300], [1e-300])


@pytest.mark.parametrize("factory,expected,objective", [
    (known_soc, [5., 3., 4.], 5.),
    (bounded_qp, [0., 1.], -2.5),
    (unconstrained, [2., -2.], -12.),
    (mixed_soc_qp, [2.5, 1.5, 2., -2.], -4.25),
])
def test_real_clarabel_matches_analytic_solutions(factory, expected, objective):
    result = solve_conic(factory(), REFERENCE)
    assert result.backend_status == "Solved"
    assert result.status == "reference_solved", result.evidence.to_dict()
    assert result.backend == "clarabel_reference" and result.device == "cpu"
    assert result.evidence.passed and result.certified is False
    assert result.backend_version
    assert np.allclose(result.x, expected, atol=2e-6, rtol=0)
    assert abs(result.objective - objective) < 1e-6
    assert check_conic_candidate(factory(), result.x, result.s, result.z).passed


def test_real_clarabel_direct_run_and_full_offdiagonal_qp():
    import clarabel
    from scipy import sparse
    p = unconstrained(P=[[4., 1.], [1., 2.]], q=[-9., -5.])
    expected = np.linalg.solve(p.P, -p.q)
    result = solve_conic(p, REFERENCE)
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.tol_feas = settings.tol_gap_abs = settings.tol_gap_rel = 1e-8
    direct = clarabel.DefaultSolver(
        sparse.csc_matrix(np.triu(p.P)), p.q, sparse.csc_matrix(p.A), p.b, [], settings,
    ).solve()
    assert result.status == "reference_solved" and str(direct.status) == "Solved"
    assert np.allclose(result.x, expected, atol=1e-8, rtol=0)
    assert np.allclose(result.x, direct.x, atol=1e-8, rtol=0)
    assert abs(result.objective - direct.obj_val) < 1e-8


def test_cpu_tensors_copied_without_mutation_or_grad():
    p = bounded_qp()
    tensors = {name: torch.tensor(getattr(p, name), requires_grad=True) for name in ("P", "q", "A", "b")}
    snapshot = {name: value.detach().clone() for name, value in tensors.items()}
    result = solve_conic(replace(p, **tensors), REFERENCE)
    assert result.status == "reference_solved"
    for name, tensor in tensors.items():
        assert torch.equal(tensor.detach(), snapshot[name]) and tensor.grad is None


@pytest.mark.parametrize("factory,raw,status", [
    (infeasible, "PrimalInfeasible", "reference_primal_infeasible_unverified"),
    (unbounded, "DualInfeasible", "reference_dual_infeasible_unverified"),
])
def test_real_infeasible_unbounded_statuses_not_fake_certificates(factory, raw, status):
    result = solve_conic(factory(), REFERENCE)
    assert result.backend_status == raw and result.status == status
    assert result.certified is False and result.evidence is None and result.objective is None
    assert "unbounded" not in result.status


def test_real_iteration_limit_not_infeasibility():
    result = solve_conic(known_soc(), replace(REFERENCE, max_iterations=1))
    assert result.backend_status == "MaxIterations" and result.status == "max_iterations"
    assert not result.certified and result.evidence is None and result.objective is None


@pytest.mark.parametrize("raw", [
    "Solved", "AlmostSolved", "AlmostPrimalInfeasible", "AlmostDualInfeasible",
    "NumericalError", "InsufficientProgress", "MaxTime", "Mystery",
])
def test_reference_status_is_not_success_and_self_report_not_evidence(raw, monkeypatch):
    import clarabel
    solution = SimpleNamespace(
        status=raw, x=[0., 0.], s=[], z=[], iterations=1,
        obj_val=-12., r_prim=0., r_dual=0.,
    )
    monkeypatch.setattr(clarabel, "DefaultSolver", lambda *args: SimpleNamespace(solve=lambda: solution))
    if raw == "Mystery":
        with pytest.raises(RuntimeError, match="unrecognized"):
            solve_conic(unconstrained(), REFERENCE)
    else:
        result = solve_conic(unconstrained(), REFERENCE)
        assert result.status != "reference_solved" and result.certified is False
        if raw in ("Solved", "AlmostSolved"):
            assert result.objective == 0 and not result.evidence.passed
        else:
            assert result.objective is None and result.evidence is None


def test_solver_failure_propagates_not_fallback(monkeypatch):
    import clarabel
    cause = RuntimeError("backend factorization failure")
    def failed(*args):
        raise cause
    monkeypatch.setattr(clarabel, "DefaultSolver", failed)
    with pytest.raises(RuntimeError) as raised:
        solve_conic(known_soc(), REFERENCE)
    assert raised.value is cause


@pytest.mark.parametrize("field,value", [
    ("x", [np.nan, 0.]), ("x", [np.inf, 0.]), ("x", [0.]),
    ("iterations", 1.5), ("iterations", -1), ("iterations", True),
])
def test_invalid_backend_return_is_fail_loud(field, value, monkeypatch):
    import clarabel
    solution = SimpleNamespace(status="Solved", x=[2., -2.], s=[], z=[], iterations=1)
    setattr(solution, field, value)
    monkeypatch.setattr(clarabel, "DefaultSolver", lambda *args: SimpleNamespace(solve=lambda: solution))
    with pytest.raises((ValueError, RuntimeError)):
        solve_conic(unconstrained(), REFERENCE)


def test_machine_readable_conic_acceptance(tmp_path, request):
    factories = {"known_soc": known_soc, "bounded_qp": bounded_qp, "unconstrained_qp": unconstrained,
                 "mixed_soc_qp": mixed_soc_qp, "infeasible": infeasible, "unbounded": unbounded}
    cases = {}
    for name, factory in factories.items():
        result = solve_conic(factory(), REFERENCE)
        optimal = name not in ("infeasible", "unbounded")
        assert (result.status == "reference_solved") is optimal
        assert result.certified is False
        cases[name] = {
            "backend": result.backend, "backend_version": result.backend_version,
            "backend_status": result.backend_status, "status": result.status,
            "certified": result.certified, "iterations": result.iterations,
            "objective": result.objective, "x": result.x.tolist(), "s": result.s.tolist(),
            "z": result.z.tolist(), "evidence": None if result.evidence is None else result.evidence.to_dict(),
        }
    negative = check_conic_candidate(known_soc(), [4, 3, 4], [0, 0, 5, 3, 4], [-.6, -.8, 1, -.6, -.8])
    assert not negative.passed
    root = Path(__file__).resolve().parents[1]
    paths = ["src/openopt/conic.py", "tests/test_conic.py", "docs/research/conic.md"]
    report = {
        "schema_version": 1, "verdict": "PASS", "owner": "conic", "device": "cpu",
        "checked_at": datetime.now(timezone.utc).isoformat(), "python_executable": sys.executable,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "qualification": "Clarabel CPU reference plus independent numerical checks; no GPU or exact certificate",
        "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
        "cases": cases, "negative_control_wrong_x": negative.to_dict(),
    }
    artifact = tmp_path / "conic-acceptance.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(artifact.read_text()) == report
    request.node.user_properties.append(("conic_acceptance_path", str(artifact)))


def _audit_artifacts(directory: Path) -> None:
    """Re-read final JUnit, numerical evidence and current scoped source hashes."""
    directory = directory.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    if directory.is_relative_to(root):
        raise ValueError("artifacts must be outside this worktree")
    verification = json.loads((directory / "verification.json").read_text())
    assert verification["verdict"] == "PASS"
    suites = ElementTree.parse(directory / "pytest.xml").getroot()
    counts = {
        name: sum(int(suite.attrib.get(name, 0)) for suite in suites.findall("testsuite"))
        for name in ("tests", "failures", "errors", "skipped")
    }
    assert counts == verification["tests"]
    assert counts["tests"] > 0 and counts["failures"] == counts["errors"] == counts["skipped"] == 0
    matches = [
        prop.attrib["value"]
        for prop in suites.findall(".//property")
        if prop.attrib["name"] == "conic_acceptance_path"
    ]
    assert len(matches) == 1
    evidence_path = Path(matches[0]).resolve(strict=True)
    assert evidence_path.is_relative_to(directory)
    report = json.loads(evidence_path.read_text())
    assert report["verdict"] == "PASS" and report["device"] == "cpu"
    assert report["cuda_visible_devices"] == "-1"
    paths = {"src/openopt/conic.py", "tests/test_conic.py", "docs/research/conic.md"}
    assert report["source_sha256"].keys() == paths
    for path, digest in report["source_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == digest, path
    cases = report["cases"]
    assert len(cases) == 6
    for name, case in cases.items():
        assert not case["certified"] and case["backend"] == "clarabel_reference"
        if name in ("infeasible", "unbounded"):
            assert case["evidence"] is None and case["objective"] is None
            assert case["status"].endswith("_unverified")
        else:
            assert case["evidence"]["passed"] and case["status"] == "reference_solved"
    assert cases["known_soc"]["evidence"]["metrics"]["primal_objective"] == pytest.approx(5., abs=1e-6)
    assert not report["negative_control_wrong_x"]["passed"]
    print(
        f"CONIC_ARTIFACT_AUDIT verdict=PASS checked_at={datetime.now(timezone.utc).isoformat()} "
        f"tests={counts['tests']} failures=0 errors=0 skipped=0 cases={len(cases)} "
        f"source_hashes={len(paths)} evidence={evidence_path}"
    )


class _BlockOptional(importlib.abc.MetaPathFinder):
    def __init__(self, blocked):
        self.blocked = blocked

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in self.blocked:
            raise ModuleNotFoundError(f"lazy probe blocked {fullname}", name=fullname)
        return None


def _lazy_probe() -> None:
    # The existing compatibility facade imports SciPy eagerly, outside this
    # seat's scope. Test its Clarabel laziness, then the actual conic module
    # under a namespace-only parent to measure its SciPy import independently.
    for name in tuple(sys.modules):
        if name.split(".")[0] in ("openopt", "openpo", "clarabel"):
            del sys.modules[name]
    blocker = _BlockOptional({"clarabel"})
    sys.meta_path.insert(0, blocker)
    module = importlib.import_module("openopt.conic")
    assert "clarabel" not in sys.modules
    assert module.check_cone_feasibility([5, 3, 4], (module.SOC(3),)).passed
    blocker.blocked.add("scipy")
    for name in tuple(sys.modules):
        if name.split(".")[0] == "scipy":
            del sys.modules[name]
    parent = ModuleType("conic_scoped_probe")
    parent.__path__ = [str(Path(__file__).resolve().parents[1] / "src/openopt")]
    sys.modules[parent.__name__] = parent
    module = importlib.import_module("conic_scoped_probe.conic")
    assert "clarabel" not in sys.modules and "scipy" not in sys.modules
    assert module.check_cone_feasibility([5, 3, 4], (module.SOC(3),)).passed
    p = module.ConicProblem(np.eye(1), np.zeros(1), np.empty((0, 1)), np.empty(0), ())
    assert module.check_conic_candidate(p, [0], [], []).passed
    try:
        module.solve_conic(p, module.ConicConfig(backend="clarabel_reference"))
    except RuntimeError as error:
        assert isinstance(error.__cause__, ModuleNotFoundError)
    else:
        raise AssertionError("missing optional dependency must not produce a solution")
    print("CONIC_LAZY_PROBE PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["--lazy-probe"]:
        _lazy_probe()
    elif len(sys.argv) == 3 and sys.argv[1] == "--audit-artifacts":
        _audit_artifacts(Path(sys.argv[2]))
    else:
        raise SystemExit("use pytest, --lazy-probe, or --audit-artifacts DIRECTORY")
