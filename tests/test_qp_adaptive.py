"""CPU adaptive-rho acceptance, frozen baseline and fault-injection controls."""

from dataclasses import fields, replace
import importlib.util
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import osqp
import pytest
from scipy import sparse
import torch

from openopt.qp import QPConfig, QPProblem, _balanced_rho, prepare_qp, solve_qp


def ill_conditioned(kind="upper"):
    diagonal = torch.logspace(-4, 4, 8, dtype=torch.float64)
    target = torch.linspace(-0.4, 0.4, 8, dtype=torch.float64)
    target[-1] = -2 if kind == "lower" else 2
    l, u = -torch.ones(8, dtype=torch.float64), torch.ones(8, dtype=torch.float64)
    if kind == "equality":
        l[-1] = u[-1] = 0.5
    return QPProblem(torch.diag(diagonal), -diagonal * target, torch.eye(8), l, u)


def assert_original_kkt(p, result, config):
    P, q, A, l, u = [t.numpy().astype(np.float64) for t in (p.P, p.q, p.A, p.l, p.u)]
    x, y = result.x.numpy().astype(np.float64), result.y.numpy().astype(np.float64)
    Ax, Px, ATy = A @ x, P @ x, A.T @ y
    z = np.clip(Ax, l, u)
    primal = np.max(np.abs(Ax - z), initial=0)
    dual = np.max(np.abs(Px + q + ATy), initial=0)
    normal = np.max(np.abs(z - np.clip(z + y, l, u)), initial=0)
    ptol = config.eps_abs + config.eps_rel * max(np.max(np.abs(Ax)), np.max(np.abs(z)))
    dtol = config.eps_abs + config.eps_rel * max(
        np.max(np.abs(Px)), np.max(np.abs(q)), np.max(np.abs(ATy)),
    )
    assert primal <= 1.01 * ptol
    assert dual <= 1.01 * dtol
    assert normal <= 2.01 * ptol
    assert result.primal_residual <= 1.01 * ptol
    assert result.dual_residual <= 1.01 * dtol


@pytest.fixture(scope="module")
def frozen_baseline():
    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "show", "fcefac1:src/openopt/qp.py"], cwd=root,
        check=True, capture_output=True, text=True,
    ).stdout
    assert "adaptive_rho" not in source
    spec = importlib.util.spec_from_loader("_adaptive_frozen_qp", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        exec(compile(source, "fcefac1:src/openopt/qp.py", "exec"), module.__dict__)
        yield module
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_disabled_is_bitwise_frozen_baseline(frozen_baseline, linear_solver, dtype):
    p = ill_conditioned()
    c = QPConfig(dtype=dtype, linear_solver=linear_solver, max_iterations=300)
    legacy_names = [f.name for f in fields(frozen_baseline.QPConfig)]
    defaults = QPConfig()
    for f in fields(frozen_baseline.QPConfig):
        assert getattr(defaults, f.name) == f.default
    old_config = frozen_baseline.QPConfig(**{name: getattr(c, name) for name in legacy_names})
    expected = frozen_baseline.solve_qp(p, old_config)
    actual = solve_qp(p, c)
    assert torch.equal(actual.x, expected.x)
    assert torch.equal(actual.y, expected.y)
    for f in fields(frozen_baseline.QPResult):
        if f.name not in ("x", "y"):
            assert getattr(actual, f.name) == getattr(expected, f.name)
    assert actual.rho_updates == 0
    assert actual.final_rho == c.rho


@pytest.mark.parametrize("kind", ["upper", "lower", "equality"])
@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
def test_ill_conditioned_matches_osqp_and_witness(kind, linear_solver, request):
    p = ill_conditioned(kind)
    c = QPConfig(
        adaptive_rho=True, rho=1e-4, adaptive_rho_interval=25,
        max_iterations=5000, eps_abs=1e-8, eps_rel=1e-9,
        linear_solver=linear_solver,
    )
    fixed = solve_qp(p, replace(c, adaptive_rho=False))
    result = solve_qp(p, c)
    ref = osqp.OSQP()
    ref.setup(
        P=sparse.csc_matrix(p.P.numpy()), q=p.q.numpy(),
        A=sparse.csc_matrix(p.A.numpy()), l=p.l.numpy(), u=p.u.numpy(),
        eps_abs=1e-10, eps_rel=0, max_iter=100000,
        polishing=False, verbose=False,
    )
    expected = ref.solve(raise_error=True)
    witness = torch.minimum(torch.maximum(-p.q / p.P.diag(), p.l), p.u)
    assert expected.info.status == "solved"
    assert np.max(np.abs(expected.x - witness.numpy())) < 1e-4
    assert result.status == "solved"
    assert result.x.device.type == result.y.device.type == "cpu"
    assert result.rho_updates > 0
    assert c.rho_min <= result.final_rho <= c.rho_max
    assert_original_kkt(p, result, c)
    assert torch.max(torch.abs(result.x - witness)) < 1e-4
    assert np.max(np.abs(result.x.numpy() - expected.x)) < 1e-4
    assert abs(result.objective - expected.info.obj_val) < 1e-3
    # Fixed tiny rho is the deliberate negative control, not a tuned baseline.
    assert fixed.status == "max_iterations"
    assert fixed.primal_residual > 0.1
    assert result.iterations < fixed.iterations
    request.node.user_properties.extend([
        ("fixed_iterations", fixed.iterations),
        ("adaptive_iterations", result.iterations),
        ("rho_updates", result.rho_updates),
        ("final_rho", result.final_rho),
        ("primal_residual", result.primal_residual),
        ("dual_residual", result.dual_residual),
        ("osqp_objective_gap", abs(result.objective - expected.info.obj_val)),
    ])


def test_repeat_solve_resets_rho_factor_and_owns_inputs():
    p = ill_conditioned()
    c = QPConfig(adaptive_rho=True, rho=1e-4, adaptive_rho_interval=25)
    workspace = prepare_qp(p, c)
    factor = workspace.factor.clone()
    first = workspace.solve()
    p.P.mul_(2)
    p.A.mul_(3)
    second = workspace.solve()
    assert first.status == second.status == "solved"
    assert first.rho_updates == second.rho_updates > 0
    assert first.final_rho == second.final_rho
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.y, second.y)
    assert torch.equal(workspace.factor, factor)
    assert workspace.config.rho == 1e-4


@pytest.mark.parametrize("field,value", [
    ("adaptive_rho", 1), ("adaptive_rho_interval", 0),
    ("adaptive_rho_interval", True), ("adaptive_rho_interval", 2.5),
    ("adaptive_rho_tolerance", 1), ("adaptive_rho_tolerance", float("nan")),
    ("rho_min", 0), ("rho_max", float("inf")), ("rho_min", 1e7),
    ("rho", 1e-8),
])
def test_invalid_adaptive_config_fails(field, value):
    c = replace(QPConfig(adaptive_rho=True), **{field: value})
    with pytest.raises(ValueError):
        prepare_qp(ill_conditioned(), c)


@pytest.mark.parametrize("primal,dual,expected", [
    (0.0, 0.0, 1.0), (0.0, 1.0, 1e-6), (1.0, 0.0, 1e6),
    (1e300, 1e-300, 1e6), (1e-300, 1e300, 1e-6),
    (1.0, 1.0, 1.0), (4.0, 1.0, 1.0),
])
def test_balancer_is_bounded_zero_safe_and_has_hysteresis(primal, dual, expected):
    c = QPConfig(adaptive_rho=True)
    actual = _balanced_rho(1.0, primal, dual, 1.0, 1.0, c)
    assert math.isfinite(actual)
    assert c.rho_min <= actual <= c.rho_max
    assert actual == pytest.approx(expected)


def test_balancer_uses_normalized_not_raw_residuals():
    c = QPConfig(adaptive_rho=True)
    assert _balanced_rho(1.0, 100.0, 1.0, 100.0, 1.0, c) == 1.0
    with pytest.raises(RuntimeError, match="finite"):
        _balanced_rho(1.0, float("inf"), 1.0, 1.0, 1.0, c)


def test_update_interval_is_independent_of_stop_check(monkeypatch):
    c = QPConfig(
        adaptive_rho=True, rho=1e-4, adaptive_rho_interval=7,
        check_interval=11, max_iterations=50,
    )
    workspace = prepare_qp(ill_conditioned(), c)
    original = workspace._factorize_rho
    calls = []

    def track(rho, iteration):
        calls.append((rho, iteration))
        return original(rho, iteration)

    monkeypatch.setattr(workspace, "_factorize_rho", track)
    result = workspace.solve()
    assert calls and calls[0][1] == 7
    assert all(iteration % 7 == 0 and iteration < 50 for _, iteration in calls)
    assert result.rho_updates == len(calls)
    assert result.status != "solved" or result.iterations % 11 == 0 or result.iterations == 50


@pytest.mark.parametrize("failure", ["info", "nonfinite", "exception"])
def test_refactor_failure_is_loud_with_coordinates(monkeypatch, failure):
    workspace = prepare_qp(
        ill_conditioned(), QPConfig(adaptive_rho=True, rho=1e-4, adaptive_rho_interval=1),
    )
    original = torch.linalg.cholesky_ex

    def fail(system):
        if failure == "exception":
            raise RuntimeError("injected backend first cause")
        factor, info = original(system)
        if failure == "info":
            info = torch.tensor(1)
        else:
            factor.fill_(float("nan"))
        return factor, info

    monkeypatch.setattr(torch.linalg, "cholesky_ex", fail)
    with pytest.raises(RuntimeError, match="iteration=1 rho=") as error:
        workspace.solve()
    if failure == "exception":
        assert "injected backend first cause" in str(error.value.__cause__)


def test_no_adaptive_cuda_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA"):
        solve_qp(ill_conditioned(), QPConfig(device="cuda:0", adaptive_rho=True))


@pytest.mark.parametrize("kind", ["infeasible", "unbounded"])
def test_adaptive_does_not_invent_certificates(kind):
    if kind == "infeasible":
        p = QPProblem(
            torch.eye(1), torch.zeros(1), torch.ones((2, 1)),
            torch.tensor([0., 2.]), torch.tensor([1., 3.]),
        )
    else:
        p = QPProblem(
            torch.zeros((1, 1)), -torch.ones(1), torch.empty((0, 1)),
            torch.empty(0), torch.empty(0),
        )
    result = solve_qp(p, QPConfig(adaptive_rho=True, adaptive_rho_interval=1, max_iterations=30))
    assert result.status == "max_iterations"


def test_unconstrained_adaptive_does_not_refactor():
    p = QPProblem(
        torch.eye(2), torch.tensor([-2., 3.]), torch.empty((0, 2)),
        torch.empty(0), torch.empty(0),
    )
    result = solve_qp(p, QPConfig(adaptive_rho=True, adaptive_rho_interval=1))
    assert result.status == "solved"
    assert result.rho_updates == 0
    assert torch.allclose(result.x, -p.q.double())


@pytest.mark.parametrize("seed", [17, 29, 81])
def test_coupled_dense_qp_matches_independent_reference(seed):
    rng = np.random.default_rng(seed)
    rotation, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    P = (rotation * np.logspace(-2, 2, 6)) @ rotation.T
    A = np.concatenate([np.eye(6), rng.normal(size=(4, 6))])
    x = rng.uniform(-0.3, 0.3, 6)
    Ax = A @ x
    l, u = Ax - 0.4, Ax + 0.4
    l[0], u[1], l[2], u[2] = Ax[0], Ax[1], Ax[2], Ax[2]
    y = np.zeros(10)
    y[:3] = [-1.5, 2.0, 0.7]
    q = -P @ x - A.T @ y
    p = QPProblem(*[torch.tensor(t, dtype=torch.float64) for t in (P, q, A, l, u)])
    c = QPConfig(adaptive_rho=True, rho=1e-3, adaptive_rho_interval=25)
    result = solve_qp(p, c)
    ref = osqp.OSQP()
    ref.setup(
        P=sparse.csc_matrix(P), q=q, A=sparse.csc_matrix(A), l=l, u=u,
        eps_abs=1e-10, eps_rel=1e-10, max_iter=100000, verbose=False,
    )
    expected = ref.solve(raise_error=True)
    assert result.status == "solved"
    assert result.rho_updates > 0
    assert_original_kkt(p, result, c)
    assert np.max(np.abs(result.x.numpy() - x)) < 2e-5
    assert np.max(np.abs(result.x.numpy() - expected.x)) < 2e-5
    assert abs(result.objective - expected.info.obj_val) < 2e-6


def test_nonfinite_refactor_system_is_rejected_before_backend(monkeypatch):
    workspace = prepare_qp(ill_conditioned(), QPConfig(adaptive_rho=True))
    workspace.gram.fill_(float("inf"))

    def forbidden(system):
        pytest.fail("nonfinite system reached Cholesky")

    monkeypatch.setattr(torch.linalg, "cholesky_ex", forbidden)
    with pytest.raises(RuntimeError, match="iteration=100 rho=10.0: nonfinite system"):
        workspace._factorize_rho(10.0, 100)


def test_real_objective_overflow_cannot_be_solved(frozen_baseline):
    p = QPProblem(
        torch.eye(1, dtype=torch.float64), torch.tensor([-1e200], dtype=torch.float64),
        torch.empty((0, 1)), torch.empty(0), torch.empty(0),
    )
    legacy = frozen_baseline.solve_qp(p)
    assert legacy.status == "solved"
    assert torch.isfinite(legacy.x).all() and torch.isfinite(legacy.y).all()
    assert not math.isfinite(legacy.objective)
    with pytest.raises(RuntimeError, match="nonfinite objective"):
        solve_qp(p)


def test_real_stop_tolerance_overflow_is_not_accepted(frozen_baseline):
    p = QPProblem(
        torch.eye(1), torch.tensor([-3.]), torch.empty((0, 1)),
        torch.empty(0), torch.empty(0),
    )
    legacy = frozen_baseline.solve_qp(
        p, frozen_baseline.QPConfig(eps_rel=1e308, check_interval=1),
    )
    assert legacy.status == "solved"
    assert 1e308 * float(p.q.abs().max()) == math.inf
    with pytest.raises(RuntimeError, match="nonfinite residual/tolerance iteration=1"):
        solve_qp(p, QPConfig(eps_rel=1e308, check_interval=1))


def test_nonfinite_residual_input_is_rejected_even_with_finite_iterates(monkeypatch):
    p = QPProblem(
        torch.tensor([[1e4]]), torch.zeros(1), torch.empty((0, 1)),
        torch.empty(0), torch.empty(0),
    )
    workspace = prepare_qp(p)

    def finite_x(rhs, factor):
        return torch.full_like(rhs, 1e308)

    monkeypatch.setattr(torch, "cholesky_solve", finite_x)
    with pytest.raises(RuntimeError, match="nonfinite residual input"):
        workspace.solve()
