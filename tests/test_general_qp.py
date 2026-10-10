"""Independent dense QP correctness and honest termination tests."""

from dataclasses import replace

import numpy as np
import osqp
import pytest
from scipy import sparse
import torch

from openopt import QPProblem, QPConfig, prepare_qp, solve_qp


def problem(n=12):
    rng = np.random.default_rng(81)
    B = rng.normal(size=(n, n))
    P = np.eye(n) + B.T @ B / n
    A = np.concatenate([np.eye(n), rng.normal(size=(3, n))])
    feasible = rng.uniform(-0.4, 0.4, n)
    Ax = A @ feasible
    return QPProblem(*[
        torch.tensor(x, dtype=torch.float64) for x in (
            P, rng.normal(size=n), A, Ax - 0.5, Ax + 0.5,
        )
    ])


@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-7), (torch.float32, 1e-5)])
def test_matches_osqp_and_kkt(dtype, tolerance, linear_solver):
    p = problem()
    config = QPConfig(dtype=dtype, eps_abs=tolerance, eps_rel=tolerance, linear_solver=linear_solver)
    result = solve_qp(p, config)
    ref = osqp.OSQP()
    ref.setup(
        P=sparse.csc_matrix(p.P.numpy()), q=p.q.numpy(),
        A=sparse.csc_matrix(p.A.numpy()), l=p.l.numpy(), u=p.u.numpy(),
        eps_abs=1e-10, eps_rel=1e-10, verbose=False, polishing=True,
    )
    expected = ref.solve()
    assert result.status == "solved"
    assert result.x.device.type == "cpu"
    assert result.x.dtype == dtype
    assert np.max(np.abs(result.x.numpy() - expected.x)) < 50 * tolerance
    assert abs(result.objective - expected.info.obj_val) < 50 * tolerance
    assert result.primal_residual < 10 * tolerance
    assert result.dual_residual < 10 * tolerance


def test_workspace_is_repeatable_and_owns_inputs():
    p = problem()
    workspace = prepare_qp(p)
    first = workspace.solve()
    p.q.add_(100)
    second = workspace.solve()
    assert torch.equal(first.x, second.x)
    assert first.status == second.status == "solved"


def test_equalities_and_one_sided_infinite_bounds():
    p = QPProblem(
        torch.eye(2, dtype=torch.float64), torch.tensor([-2., -3.], dtype=torch.float64),
        torch.tensor([[1., 1.], [1., 0.]], dtype=torch.float64),
        torch.tensor([1., -float("inf")], dtype=torch.float64),
        torch.tensor([1., 0.2], dtype=torch.float64),
    )
    r = solve_qp(p)
    assert r.status == "solved"
    assert torch.allclose(r.x, torch.tensor([0., 1.], dtype=torch.float64), atol=1e-6)


def test_unconstrained_spd():
    p = QPProblem(
        torch.eye(2, dtype=torch.float64), torch.tensor([-2., 3.], dtype=torch.float64),
        torch.empty((0, 2), dtype=torch.float64),
        torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64),
    )
    r = solve_qp(p)
    assert r.status == "solved"
    assert torch.allclose(r.x, -p.q)


def test_lp_semidefinite_and_bounded():
    p = QPProblem(
        torch.zeros((2, 2)), torch.tensor([-1., 2.]), torch.eye(2),
        torch.zeros(2), torch.ones(2),
    )
    r = solve_qp(p)
    assert r.status == "solved"
    assert torch.allclose(r.x, torch.tensor([1., 0.], dtype=torch.float64), atol=1e-6)


@pytest.mark.parametrize("kind", ["infeasible", "unbounded", "iteration_limit"])
def test_no_false_certificate(kind):
    p = problem()
    if kind == "infeasible":
        A = torch.ones((2, 1), dtype=torch.float64)
        p = QPProblem(torch.eye(1), torch.zeros(1), A, torch.tensor([0., 2.]), torch.tensor([1., 3.]))
    if kind == "unbounded":
        p = QPProblem(
            torch.zeros((1, 1)), -torch.ones(1), torch.empty((0, 1)),
            torch.empty(0), torch.empty(0),
        )
    r = solve_qp(p, QPConfig(max_iterations=2, check_interval=1))
    assert r.status == "max_iterations"


@pytest.mark.parametrize("field,value", [
    ("rho", 0), ("sigma", -1), ("eps_abs", float("nan")),
    ("relaxation", 2), ("max_iterations", 0), ("check_interval", 1.5),
    ("dtype", torch.float16), ("device", "meta"),
    ("linear_solver", "invalid"),
])
def test_invalid_config(field, value):
    with pytest.raises((ValueError, RuntimeError)):
        solve_qp(problem(), replace(QPConfig(), **{field: value}))


@pytest.mark.parametrize("mutation", ["nonfinite", "nan_bound", "reversed", "indefinite", "asymmetric", "shape", "empty"])
def test_invalid_problem(mutation):
    p = problem()
    if mutation == "nonfinite":
        p.q[0] = float("inf")
    if mutation == "nan_bound":
        p.l[0] = float("nan")
    if mutation == "reversed":
        p.l[0] = p.u[0] + 1
    if mutation == "indefinite":
        p = replace(p, P=-torch.eye(p.q.numel()))
    if mutation == "asymmetric":
        p.P[0, 1] += 1
    if mutation == "shape":
        p = replace(p, A=p.A[:, :-1])
    if mutation == "empty":
        p = replace(p, q=torch.empty(0))
    with pytest.raises(ValueError):
        solve_qp(p)


def test_explicit_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA"):
        solve_qp(problem(), QPConfig(device="cuda:0"))


@pytest.mark.parametrize("variant", ["complex", "sparse"])
def test_unsupported_tensor_layout_is_not_silently_converted(variant):
    p = problem()
    P = p.P.to(torch.complex128) if variant == "complex" else p.P.to_sparse()
    with pytest.raises(ValueError, match="real dense"):
        solve_qp(replace(p, P=P))
