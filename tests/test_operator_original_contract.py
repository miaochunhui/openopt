"""Original input semantics and componentwise sparse acceptance regressions."""

from dataclasses import replace

import pytest
import torch

from openopt.operators import (
    MatrixFreeOperator, OperatorQPConfig, OperatorQPProblem,
    prepare_operator_qp, solve_operator_qp,
)


def representation(matrix, kind):
    if kind == "coo":
        return matrix.to_sparse_coo()
    if kind == "csr":
        return matrix.to_sparse_csr()
    if kind == "matrix_free":
        return MatrixFreeOperator(
            tuple(matrix.shape), matrix.device, matrix.dtype,
            lambda v: matrix @ v, lambda v: matrix.T @ v,
        )
    return matrix


def model(kind="dense"):
    P = torch.ones((1, 1), dtype=torch.float64)
    A = torch.tensor([[1.], [0.]], dtype=torch.float64)
    return OperatorQPProblem(
        representation(P, kind), torch.tensor([-1e9], dtype=torch.float64),
        representation(A, kind), torch.tensor([1e9, 1.], dtype=torch.float64),
        torch.tensor([1e9, 2.], dtype=torch.float64), assume_psd=True,
    )


@pytest.mark.parametrize("kind", ["dense", "coo", "csr", "matrix_free"])
def test_zero_row_contradiction_cannot_use_other_rows_scale(kind):
    p = model(kind)
    result = solve_operator_qp(p, OperatorQPConfig(max_iterations=100))
    assert result.status == "max_iterations"
    assert result.primal_residual >= 1
    assert float(p.l[1]) == 1 and float(p.u[1]) == 2


@pytest.mark.parametrize("kind", ["dense", "coo", "csr"])
@pytest.mark.parametrize("loss", ["reversed", "lower_overflow", "upper_overflow", "nonzero_underflow"])
def test_original_bound_semantics_checked_before_float32_cast(kind, loss):
    p = model(kind)
    if loss == "reversed":
        p.l[1], p.u[1] = 1 + 1e-12, 1.
    elif loss == "lower_overflow":
        p.l[1] = -1e100
    elif loss == "upper_overflow":
        p.u[1] = 1e100
    else:
        p.l[1], p.u[1] = 1e-100, 2e-100
    before = [v.clone() for v in (p.q, p.l, p.u)]
    with pytest.raises(ValueError, match="original bounds|conversion"):
        prepare_operator_qp(p, OperatorQPConfig(dtype=torch.float32))
    assert all(torch.equal(left, right) for left, right in zip(before, (p.q, p.l, p.u)))


@pytest.mark.parametrize("kind", ["dense", "coo", "csr"])
@pytest.mark.parametrize("field,value", [("P", 1e-100), ("A", 1e-100), ("q", 1e100)])
def test_finite_matrix_and_objective_cast_loss_rejected(kind, field, value):
    p = model()
    changed = getattr(p, field).clone()
    changed.flatten()[0] = value
    p = replace(p, **{field: changed})
    p = replace(p, P=representation(p.P, kind), A=representation(p.A, kind))
    with pytest.raises(ValueError, match="conversion"):
        prepare_operator_qp(p, OperatorQPConfig(dtype=torch.float32))


@pytest.mark.parametrize("kind", ["dense", "coo", "csr", "matrix_free"])
def test_feasible_companion_still_solves_without_densifying(kind, monkeypatch):
    p = model(kind)
    p.l[1], p.u[1] = -1., 2.
    if kind != "dense":
        def forbidden(*args, **kwargs):
            raise AssertionError("original acceptance must not densify")
        monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
        monkeypatch.setattr(torch.linalg, "cholesky_ex", forbidden)
    result = solve_operator_qp(p, OperatorQPConfig(max_iterations=100))
    assert result.status == "solved"
    assert abs(float(result.x[0]) - 1e9) <= 1e2


def test_preparation_owns_original_vectors_for_later_acceptance():
    p = model()
    p.l[1], p.u[1] = -1., 2.
    workspace = prepare_operator_qp(p)
    expected = workspace.solve()
    p.l.fill_(100)
    p.q.zero_()
    assert torch.equal(workspace.solve().x, expected.x)

