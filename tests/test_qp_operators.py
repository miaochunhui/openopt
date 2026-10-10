"""CPU sparse/matrix-free correctness, true inner residual and negative controls."""

from dataclasses import replace

import pytest
import torch

from openopt.operators import (
    CGConvergenceError, MatrixFreeOperator, OperatorQPConfig, OperatorQPProblem,
    TensorOperator, conjugate_gradient, prepare_operator_qp, solve_operator_qp,
)
from openopt.qp import QPConfig, QPProblem, solve_qp


def known_kkt(dtype=torch.float64):
    P = torch.tensor([[4., 1., 0.], [1., 3., 0.], [0., 0., 2.]], dtype=dtype)
    A = torch.tensor([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [1., 1., 1.]], dtype=dtype)
    x = torch.tensor([0.2, -0.3, 0.1], dtype=dtype)
    y = torch.tensor([0.7, -0.4, 0., 0.2], dtype=dtype)
    Ax = A @ x
    l, u = Ax - 1, Ax + 1
    u[0], l[1] = Ax[0], Ax[1]
    l[3] = u[3] = Ax[3]
    q = -P @ x - A.T @ y
    assert torch.max(torch.abs(P @ x + q + A.T @ y)) <= 4 * torch.finfo(dtype).eps
    assert bool(((l <= Ax) & (Ax <= u)).all())
    return OperatorQPProblem(P, q, A, l, u, assume_psd=True), x, y


def matrix_free(matrix):
    return MatrixFreeOperator(
        tuple(matrix.shape), matrix.device, matrix.dtype,
        lambda v: matrix @ v, lambda v: matrix.T @ v,
    )


def variant(matrix, kind):
    if kind == "dense":
        return matrix
    if kind == "coo":
        return matrix.to_sparse_coo()
    if kind == "csr":
        return matrix.to_sparse_csr()
    if kind == "tensor_operator":
        return TensorOperator(matrix)
    return matrix_free(matrix)


@pytest.mark.parametrize("kind", ["dense", "coo", "csr", "matrix_free", "tensor_operator"])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 2e-7), (torch.float32, 2e-5)])
def test_known_kkt_matches_dense_and_independent_residuals(kind, dtype, tolerance):
    p, expected_x, expected_y = known_kkt(dtype)
    config = OperatorQPConfig(
        dtype=dtype, eps_abs=tolerance / 10, eps_rel=tolerance / 10,
        cg_eps_abs=1e-11 if dtype == torch.float64 else 1e-7,
        cg_eps_rel=1e-11 if dtype == torch.float64 else 1e-7,
    )
    result = solve_operator_qp(
        replace(p, P=variant(p.P, kind), A=variant(p.A, kind)), config,
    )
    dense = solve_qp(
        QPProblem(p.P, p.q, p.A, p.l, p.u),
        QPConfig(dtype=dtype, eps_abs=config.eps_abs, eps_rel=config.eps_rel),
    )
    assert result.status == dense.status == "solved"
    assert result.x.device.type == "cpu"
    assert result.x.dtype == dtype
    assert result.experimental and not result.convexity_certified
    assert torch.allclose(result.x, expected_x, atol=5 * tolerance, rtol=0)
    assert torch.allclose(result.y, expected_y, atol=10 * tolerance, rtol=0)
    assert torch.allclose(result.x, dense.x, atol=5 * tolerance, rtol=0)
    Px, Ax, ATy = p.P @ result.x, p.A @ result.x, p.A.T @ result.y
    assert float((Px + p.q + ATy).abs().max()) < 5 * tolerance
    assert float((Ax - torch.clamp(Ax, min=p.l, max=p.u)).abs().max()) < 5 * tolerance
    assert float((Ax - torch.clamp(Ax + result.y, min=p.l, max=p.u)).abs().max()) < 5 * tolerance
    expected_objective = float(0.5 * expected_x @ p.P @ expected_x + p.q @ expected_x)
    assert abs(result.objective - expected_objective) < 10 * tolerance
    assert abs(result.objective - dense.objective) < 10 * tolerance
    assert result.cg_iterations > 0
    assert result.inner_residual <= result.inner_tolerance


@pytest.mark.parametrize("kind", ["coo", "csr", "matrix_free"])
@pytest.mark.parametrize("case", ["unconstrained", "lp", "infinite_bounds"])
def test_edge_problem_matches_dense(kind, case):
    P, q, A = torch.eye(2, dtype=torch.float64), torch.tensor([-2., 3.], dtype=torch.float64), torch.eye(2, dtype=torch.float64)
    l, u = torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64)
    if case == "unconstrained":
        A, l, u = torch.empty((0, 2), dtype=torch.float64), l[:0], u[:0]
    elif case == "lp":
        P = torch.zeros_like(P)
    else:
        l[0], u[1] = -float("inf"), float("inf")
    dense = solve_qp(QPProblem(P, q, A, l, u))
    result = solve_operator_qp(
        OperatorQPProblem(variant(P, kind), q, variant(A, kind), l, u, assume_psd=True),
    )
    assert result.status == dense.status == "solved"
    assert torch.allclose(result.x, dense.x, atol=1e-6, rtol=0)


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_sparse_path_cannot_densify_or_form_dense_gram(monkeypatch, layout):
    p, expected, _ = known_kkt()
    P, A = variant(p.P, layout), variant(p.A, layout)
    sparse_mm = torch.sparse.mm
    calls = []

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse path attempted a dense conversion/factorization")

    def checked_mm(matrix, vector, *args, **kwargs):
        assert matrix.layout in (torch.sparse_coo, torch.sparse_csr)
        assert vector.layout == torch.strided and vector.shape[1] == 1
        calls.append(matrix.layout)
        return sparse_mm(matrix, vector, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)
    monkeypatch.setattr(torch.sparse, "mm", checked_mm)
    workspace = prepare_operator_qp(replace(p, P=P, A=A))
    assert workspace.P.operator.matrix.layout == P.layout
    assert workspace.A.operator.matrix.layout == A.layout
    result = workspace.solve()
    assert result.status == "solved"
    assert torch.allclose(result.x, expected, atol=1e-6, rtol=0)
    assert calls
    if layout == "csr":
        assert torch.sparse_csr in calls and torch.sparse_coo in calls


def test_coo_duplicates_are_coalesced_without_losing_values():
    indices = torch.tensor([[0, 0, 1], [0, 0, 1]])
    values = torch.tensor([1., 2., 4.], dtype=torch.float64)
    op = TensorOperator(torch.sparse_coo_tensor(indices, values, (2, 2)))
    assert op.matrix.is_coalesced()
    assert op.matrix._nnz() == 2
    assert torch.equal(op.matvec(torch.ones(2, dtype=torch.float64)), torch.tensor([3., 4.], dtype=torch.float64))


@pytest.mark.parametrize("kind", ["coo", "csr", "matrix_free"])
def test_direct_sparse_or_vector_only_diagonal_kkt_without_dense_source(kind, monkeypatch):
    n = 257
    index = torch.arange(n)
    diagonal = torch.linspace(1., 4., n, dtype=torch.float64)
    expected = torch.linspace(-0.4, 0.4, n, dtype=torch.float64)
    y = torch.where(index % 2 == 0, 0.3, -0.2).to(torch.float64)
    q = -diagonal * expected - y
    l = torch.where(y < 0, expected, expected - 1)
    u = torch.where(y > 0, expected, expected + 1)
    if kind == "matrix_free":
        P = MatrixFreeOperator((n, n), "cpu", q.dtype, lambda v: diagonal * v, lambda v: diagonal * v)
        A = MatrixFreeOperator((n, n), "cpu", q.dtype, lambda v: v.clone(), lambda v: v.clone())
    else:
        indices = torch.stack((index, index))
        P = torch.sparse_coo_tensor(indices, diagonal, (n, n)).coalesce()
        A = torch.sparse_coo_tensor(indices, torch.ones_like(diagonal), (n, n)).coalesce()
        if kind == "csr":
            P, A = P.to_sparse_csr(), A.to_sparse_csr()

    def forbidden(*args, **kwargs):
        raise AssertionError("vector/nnz-only problem attempted densification")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    result = solve_operator_qp(OperatorQPProblem(P, q, A, l, u, assume_psd=True))
    assert result.status == "solved"
    assert torch.allclose(result.x, expected, atol=2e-6, rtol=0)
    assert torch.allclose(result.y, y, atol=2e-6, rtol=0)
    assert float((diagonal * result.x + q + result.y).abs().max()) < 2e-6
    assert result.inner_residual <= result.inner_tolerance


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_invalid_sparse_indices_rejected_before_mm(layout):
    if layout == "coo":
        matrix = torch.sparse_coo_tensor(
            torch.tensor([[0, 3], [0, 1]]), torch.ones(2, dtype=torch.float64),
            (2, 2), check_invariants=False,
        )
    else:
        matrix = torch.sparse_csr_tensor(
            torch.tensor([0, 1, 2]), torch.tensor([0, 3]),
            torch.ones(2, dtype=torch.float64), (2, 2), check_invariants=False,
        )
    with pytest.raises(RuntimeError):
        TensorOperator(matrix)


@pytest.mark.parametrize("kind", ["dense", "coo", "csr"])
def test_prepared_tensor_workspace_owns_inputs_and_zero_starts(kind):
    p, _, _ = known_kkt()
    P, A = variant(p.P, kind), variant(p.A, kind)
    workspace = prepare_operator_qp(replace(p, P=P, A=A))
    first = workspace.solve()
    p.q.add_(100)
    p.l.fill_(-10)
    if kind == "dense":
        P.zero_()
        A.zero_()
    else:
        P.values().zero_()
        A.values().zero_()
    second = workspace.solve()
    assert torch.equal(first.x, second.x)
    assert first.cg_iterations == second.cg_iterations


def test_cg_true_residual_against_direct_solve():
    K = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    rhs = torch.tensor([1., 2.], dtype=torch.float64)
    result = conjugate_gradient(TensorOperator(K), rhs)
    true_residual = float(torch.linalg.vector_norm(rhs - K @ result.x))
    assert result.residual == true_residual
    assert result.residual <= result.tolerance
    assert torch.allclose(result.x, torch.linalg.solve(K, rhs), atol=1e-10, rtol=0)
    initial = conjugate_gradient(TensorOperator(K), rhs, initial=result.x)
    assert initial.iterations == 0
    zero = conjugate_gradient(TensorOperator(K), torch.zeros_like(rhs))
    assert zero.iterations == 0 and zero.residual == 0


def test_cg_iteration_limit_is_loud_with_true_residual():
    K = torch.diag(torch.tensor([1., 2., 4.], dtype=torch.float64))
    rhs = torch.ones(3, dtype=torch.float64)
    with pytest.raises(CGConvergenceError, match="iteration limit") as caught:
        conjugate_gradient(TensorOperator(K), rhs, max_iterations=1)
    error = caught.value
    one_step = rhs * (rhs @ rhs) / (rhs @ K @ rhs)
    assert error.residual == float(torch.linalg.vector_norm(rhs - K @ one_step))
    assert error.residual > error.tolerance and error.iterations == 1


def test_cg_does_not_trust_zero_recurrence_residual():
    # Broken/stateful operator mimics an exact first CG step, then changes Kx.
    calls = []

    def changing(v):
        calls.append(1)
        return v if len(calls) < 3 else v * 0.5

    op = MatrixFreeOperator((2, 2), "cpu", torch.float64, changing, changing)
    with pytest.raises(CGConvergenceError, match="true_residual") as caught:
        conjugate_gradient(op, torch.ones(2, dtype=torch.float64), max_iterations=1)
    assert len(calls) == 3
    assert caught.value.residual == pytest.approx(2 ** 0.5 / 2)


def test_cg_true_residual_restart_stays_inside_original_budget():
    calls = []

    def changing(v):
        calls.append(1)
        return v if len(calls) < 3 else v * 0.5

    op = MatrixFreeOperator((2, 2), "cpu", torch.float64, changing, changing)
    result = conjugate_gradient(op, torch.ones(2, dtype=torch.float64), max_iterations=2)
    assert result.iterations == 2 and len(calls) == 5
    assert torch.equal(result.x, torch.full((2,), 2., dtype=torch.float64))
    assert result.residual == 0


def test_cg_negative_curvature_is_loud():
    with pytest.raises(CGConvergenceError, match="curvature") as caught:
        conjugate_gradient(TensorOperator(-torch.eye(2, dtype=torch.float64)), torch.ones(2, dtype=torch.float64))
    assert caught.value.residual > caught.value.tolerance


def test_outer_cannot_promote_failed_inner_to_result():
    p, _, _ = known_kkt()
    with pytest.raises(CGConvergenceError, match="ADMM iteration=1") as caught:
        solve_operator_qp(p, OperatorQPConfig(cg_max_iterations=1))
    assert caught.value.outer_iteration == 1
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("case", ["iteration_limit", "infeasible", "unbounded"])
def test_max_iterations_is_not_solved_or_a_certificate(case):
    p, _, _ = known_kkt()
    if case == "infeasible":
        p = OperatorQPProblem(
            torch.eye(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64),
            torch.ones((2, 1), dtype=torch.float64),
            torch.tensor([0., 2.], dtype=torch.float64), torch.tensor([1., 3.], dtype=torch.float64),
            assume_psd=True,
        )
    elif case == "unbounded":
        p = OperatorQPProblem(
            torch.zeros((1, 1), dtype=torch.float64), -torch.ones(1, dtype=torch.float64),
            torch.empty((0, 1), dtype=torch.float64),
            torch.empty(0, dtype=torch.float64), torch.empty(0, dtype=torch.float64),
            assume_psd=True,
        )
    result = solve_operator_qp(p, OperatorQPConfig(max_iterations=1, check_interval=100))
    assert result.status == "max_iterations" and result.iterations == 1
    assert result.inner_residual <= result.inner_tolerance
    assert not result.convexity_certified
    assert result.dual_residual > 1e-7 or result.primal_residual > 1e-7


@pytest.mark.parametrize("assumption", [False, None, 1, "yes"])
def test_requires_explicit_psd_declaration(assumption):
    p, _, _ = known_kkt()
    with pytest.raises(ValueError, match="assume_psd=True"):
        solve_operator_qp(replace(p, assume_psd=assumption))


@pytest.mark.parametrize("broken", ["adjoint", "affine", "nonlinear", "output_shape", "output_dtype", "nonfinite", "callback_failure"])
def test_operator_negative_controls(broken):
    p, _, _ = known_kkt()
    A = p.A
    mv, rmv = lambda v: A @ v, lambda v: A.T @ v
    match = "adjoint"
    if broken == "adjoint":
        rmv = lambda v: -(A.T @ v)
    elif broken == "affine":
        mv = lambda v: A @ v + 1
        match = "linear"
    elif broken == "nonlinear":
        mv = lambda v: A @ (v * v)
        match = "adjoint|linearity"
    elif broken == "output_shape":
        mv = lambda v: (A @ v)[:, None]
        match = "shape"
    elif broken == "output_dtype":
        mv = lambda v: (A @ v).float()
        match = "device/dtype"
    elif broken == "nonfinite":
        mv = lambda v: (A @ v) * float("nan")
        match = "finite"
    else:
        def mv(v):
            raise RuntimeError("intentional operator first cause")
        match = "intentional operator first cause"
    op = MatrixFreeOperator(tuple(A.shape), "cpu", A.dtype, mv, rmv)
    with pytest.raises((ValueError, RuntimeError), match=match):
        prepare_operator_qp(replace(p, A=op))


@pytest.mark.parametrize("kind", ["dense", "coo", "csr", "matrix_free"])
@pytest.mark.parametrize("broken", ["asymmetric", "negative_curvature"])
def test_false_quadratic_assumptions_caught_when_observable(kind, broken):
    p, _, _ = known_kkt()
    P = p.P.clone()
    if broken == "asymmetric":
        P[0, 1] += 2
    else:
        P = -torch.eye(3, dtype=P.dtype)
    with pytest.raises(ValueError, match="symmetr|negative curvature"):
        prepare_operator_qp(replace(p, P=variant(P, kind)))


@pytest.mark.parametrize("field,value", [
    ("rho", 0), ("sigma", -1), ("max_iterations", True), ("check_interval", 0),
    ("cg_max_iterations", 1.2), ("cg_eps_abs", float("nan")), ("cg_eps_rel", 0),
    ("eps_abs", float("inf")), ("relaxation", 2), ("dtype", torch.float16),
    ("device", "meta"),
])
def test_invalid_config(field, value):
    p, _, _ = known_kkt()
    with pytest.raises(ValueError):
        solve_operator_qp(p, replace(OperatorQPConfig(), **{field: value}))


@pytest.mark.parametrize("broken", ["empty", "shape", "q_nan", "bound_nan", "reversed", "invalid_inf", "complex", "sparse_vector"])
def test_invalid_problem(broken):
    p, _, _ = known_kkt()
    if broken == "empty":
        p = replace(p, q=torch.empty(0, dtype=torch.float64))
    elif broken == "shape":
        p = replace(p, A=p.A[:, :-1])
    elif broken == "q_nan":
        p.q[0] = float("nan")
    elif broken == "bound_nan":
        p.l[0] = float("nan")
    elif broken == "reversed":
        p.l[0] = p.u[0] + 1
    elif broken == "invalid_inf":
        p.u[0] = -float("inf")
    elif broken == "complex":
        p = replace(p, P=p.P.to(torch.complex128))
    else:
        p = replace(p, q=p.q.to_sparse_coo())
    with pytest.raises(ValueError):
        prepare_operator_qp(p)


def test_matrix_free_dtype_mismatch_does_not_convert():
    p, _, _ = known_kkt()
    with pytest.raises(ValueError, match="matrix-free device/dtype"):
        prepare_operator_qp(replace(p, A=matrix_free(p.A.float())))


def test_explicit_cuda_unavailable_is_strict_before_transfer(monkeypatch):
    p, _, _ = known_kkt()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA.*no CPU fallback"):
        prepare_operator_qp(p, OperatorQPConfig(device="cuda:0"))


def test_explicit_cuda_invalid_index_is_strict_before_transfer(monkeypatch):
    p, _, _ = known_kkt()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="index out of range"):
        prepare_operator_qp(p, OperatorQPConfig(device="cuda:1"))
