"""CPU-only preconditioner contracts, frozen default parity and independent KKT."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

import numpy as np
import pytest
import torch

from openopt.operators import (
    CGConvergenceError, JacobiPreconditioner, MatrixFreeOperator,
    OperatorQPConfig, OperatorQPProblem, TensorOperator, conjugate_gradient,
    prepare_operator_qp, solve_operator_qp,
)


def as_operator(matrix, kind):
    if kind == "matrix_free":
        return MatrixFreeOperator(
            tuple(matrix.shape), matrix.device, matrix.dtype,
            lambda v: matrix @ v, lambda v: matrix.T @ v,
        )
    if kind == "coo":
        return matrix.to_sparse_coo()
    if kind == "csr":
        return matrix.to_sparse_csr()
    return matrix


def forbid_dense(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("preconditioner attempted densification/Gram factorization")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", forbidden)
    monkeypatch.setattr(torch.linalg, "eigvalsh", forbidden)


@pytest.mark.parametrize("kind", ["dense", "coo", "csr"])
def test_tensor_diagonal_and_gram_diagonal_are_exact_without_densify(kind, monkeypatch):
    P = torch.tensor([[4., 1., 0.], [1., 0., 2.], [0., 2., 9.]], dtype=torch.float64)
    A = torch.tensor([[2., 3., 0.], [-1., 4., 0.]], dtype=torch.float64)
    p_op, a_op = TensorOperator(as_operator(P, kind)), TensorOperator(as_operator(A, kind))
    forbid_dense(monkeypatch)
    assert torch.equal(p_op.diagonal(), torch.tensor([4., 0., 9.], dtype=P.dtype))
    assert torch.equal(a_op.column_squared_norms(), torch.tensor([5., 25., 0.], dtype=P.dtype))
    returned = p_op.diagonal()
    returned.zero_()
    assert p_op.diagonal()[0] == 4
    with pytest.raises(ValueError, match="square"):
        a_op.diagonal()


def test_duplicate_coo_entries_are_summed_before_squaring(monkeypatch):
    A = torch.sparse_coo_tensor(
        torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]]),
        torch.tensor([2., -1., 3., 1.], dtype=torch.float64), (2, 3),
    )
    forbid_dense(monkeypatch)
    op = TensorOperator(A)
    assert torch.equal(op.column_squared_norms(), torch.tensor([1., 16., 0.], dtype=A.dtype))


@pytest.mark.parametrize("kind", ["coo", "csr"])
def test_direct_sparse_empty_rows_and_missing_diagonal(kind, monkeypatch):
    n = 4096
    P = torch.sparse_coo_tensor(
        torch.tensor([[0, n - 1], [0, n - 1]]),
        torch.tensor([2., 3.], dtype=torch.float64), (n, n),
    ).coalesce()
    A = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.int64), torch.empty(0, dtype=P.dtype), (0, n),
    ).coalesce()
    if kind == "csr":
        P, A = P.to_sparse_csr(), A.to_sparse_csr()
    forbid_dense(monkeypatch)
    p_op, a_op = TensorOperator(P), TensorOperator(A)
    diagonal = p_op.diagonal()
    assert diagonal.shape == (n,) and diagonal[0] == 2 and diagonal[-1] == 3
    assert torch.count_nonzero(diagonal) == 2
    assert torch.equal(a_op.column_squared_norms(), torch.zeros(n, dtype=P.dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_jacobi_owned_diagonal_and_direct_cg(dtype):
    diagonal = torch.tensor([1., 4., 64., 4096.], dtype=dtype)
    rhs = torch.ones(4, dtype=dtype)
    preconditioner = JacobiPreconditioner(diagonal)
    diagonal.zero_()
    returned = preconditioner.diagonal
    returned.zero_()
    actual = torch.tensor([1., 4., 64., 4096.], dtype=dtype)
    result = conjugate_gradient(TensorOperator(torch.diag(actual)), rhs, preconditioner=preconditioner)
    assert result.iterations == 1
    assert torch.equal(result.x, rhs / actual)
    assert result.residual == float(torch.linalg.vector_norm(rhs - actual * result.x)) == 0
    assert result.tolerance == pytest.approx(1e-10 + 1e-10 * 2)
    exact = conjugate_gradient(
        TensorOperator(torch.diag(actual)), rhs, initial=result.x, preconditioner=preconditioner,
    )
    assert exact.iterations == 0


@pytest.mark.parametrize("bad", [
    torch.tensor([-1., 1.], dtype=torch.float64),
    torch.tensor([0., 1.], dtype=torch.float64),
    torch.tensor([float("nan"), 1.], dtype=torch.float64),
    torch.tensor([float("inf"), 1.], dtype=torch.float64),
    torch.empty(0, dtype=torch.float64),
    torch.ones((2, 1), dtype=torch.float64),
    torch.ones(2, dtype=torch.float16),
    torch.ones(2, dtype=torch.complex128),
    torch.ones(2, dtype=torch.float64).to_sparse_coo(),
])
def test_invalid_jacobi_diagonal_is_loud(bad):
    with pytest.raises((ValueError, TypeError), match="positive|finite|shape|float32"):
        JacobiPreconditioner(bad)


def test_jacobi_division_overflow_is_loud_even_outside_cg():
    preconditioner = JacobiPreconditioner(torch.tensor([1e-300], dtype=torch.float64))
    with pytest.raises(ValueError, match="Jacobi output must be finite"):
        preconditioner.apply(torch.tensor([1e100], dtype=torch.float64))


def test_preconditioner_does_not_hide_negative_system_curvature():
    with pytest.raises(CGConvergenceError, match="curvature") as caught:
        conjugate_gradient(
            TensorOperator(-torch.eye(2, dtype=torch.float64)),
            torch.ones(2, dtype=torch.float64),
            preconditioner=JacobiPreconditioner(torch.ones(2, dtype=torch.float64)),
        )
    assert caught.value.residual > caught.value.tolerance


@dataclass
class CallbackPreconditioner:
    callback: object
    shape: tuple[int, int] = (2, 2)
    device: torch.device = torch.device("cpu")
    dtype: torch.dtype = torch.float64

    def apply(self, residual):
        return self.callback(residual)


@pytest.mark.parametrize("case", ["shape", "dtype", "device", "not_contract"])
def test_preconditioner_metadata_is_strict_even_for_zero_rhs(case):
    preconditioner = CallbackPreconditioner(lambda r: r.clone())
    if case == "shape":
        preconditioner.shape = (3, 3)
    elif case == "dtype":
        preconditioner.dtype = torch.float32
    elif case == "device":
        preconditioner.device = torch.device("cuda:0")
    else:
        preconditioner = lambda r: r
    with pytest.raises((ValueError, TypeError), match="preconditioner"):
        conjugate_gradient(
            TensorOperator(torch.eye(2, dtype=torch.float64)),
            torch.zeros(2, dtype=torch.float64), preconditioner=preconditioner,
        )


@pytest.mark.parametrize("case", [
    "nonfinite", "negative", "zero", "shape", "dtype", "device", "nontensor", "exception",
])
def test_invalid_preconditioner_action_preserves_first_cause_and_coordinates(case):
    callbacks = {
        "nonfinite": lambda r: r * float("nan"),
        "negative": lambda r: -r,
        "zero": lambda r: torch.zeros_like(r),
        "shape": lambda r: r[:, None],
        "dtype": lambda r: r.float(),
        "device": lambda r: torch.empty_like(r, device="meta"),
        "nontensor": lambda r: None,
    }
    if case == "exception":
        def broken(r):
            raise RuntimeError("intentional preconditioner first cause")
        callback = broken
    else:
        callback = callbacks[case]
    with pytest.raises(CGConvergenceError, match="invalid preconditioner") as caught:
        conjugate_gradient(
            TensorOperator(torch.eye(2, dtype=torch.float64)),
            torch.ones(2, dtype=torch.float64),
            preconditioner=CallbackPreconditioner(callback),
        )
    assert caught.value.iterations == 0
    assert caught.value.residual == pytest.approx(2 ** 0.5)
    assert caught.value.residual > caught.value.tolerance
    assert caught.value.__cause__ is not None
    if case == "exception":
        assert "intentional preconditioner first cause" in str(caught.value)


@pytest.mark.parametrize("budget", [1, 2])
def test_preconditioned_true_residual_restart_uses_original_budget(budget):
    calls, applied = [], []

    def changing(v):
        calls.append(1)
        return v if len(calls) < 3 else 0.5 * v

    def inverse(r):
        applied.append(r.clone())
        return 0.25 * r

    op = MatrixFreeOperator((2, 2), "cpu", torch.float64, changing, changing)
    kwargs = dict(max_iterations=budget, preconditioner=CallbackPreconditioner(inverse))
    rhs = torch.ones(2, dtype=torch.float64)
    if budget == 1:
        with pytest.raises(CGConvergenceError, match="iteration limit") as caught:
            conjugate_gradient(op, rhs, **kwargs)
        assert caught.value.residual == pytest.approx(2 ** 0.5 / 2)
        assert len(calls) == 3 and len(applied) == 1
    else:
        result = conjugate_gradient(op, rhs, **kwargs)
        assert result.iterations == 2 and len(calls) == 5 and len(applied) == 2
        assert torch.equal(applied[1], torch.full((2,), 0.5, dtype=rhs.dtype))
        assert torch.equal(result.x, torch.full((2,), 2., dtype=rhs.dtype))
        assert result.residual == 0


def test_tiny_preconditioned_norm_never_relaxes_true_residual_threshold():
    K = torch.diag(torch.tensor([1., 2., 4.], dtype=torch.float64))
    rhs = torch.ones(3, dtype=torch.float64)
    with pytest.raises(CGConvergenceError, match="iteration limit") as caught:
        conjugate_gradient(
            TensorOperator(K), rhs, max_iterations=1,
            preconditioner=JacobiPreconditioner(torch.full((3,), 1e20, dtype=rhs.dtype)),
        )
    assert caught.value.tolerance == pytest.approx(1e-10 + 1e-10 * 3 ** 0.5)
    assert caught.value.residual > 0.5


def small_problem(dtype=torch.float64):
    P = torch.tensor([[4., 1.], [1., 3.]], dtype=dtype)
    A = torch.eye(2, dtype=dtype)
    return OperatorQPProblem(
        P, torch.tensor([-1., 2.], dtype=dtype), A,
        -torch.ones(2, dtype=dtype), torch.ones(2, dtype=dtype), assume_psd=True,
    )


@pytest.mark.parametrize("kind", ["dense", "coo", "csr"])
def test_automatic_jacobi_is_exact_owned_and_opt_in(kind, monkeypatch):
    p = small_problem()
    p = replace(p, P=as_operator(p.P, kind), A=as_operator(p.A, kind))
    forbid_dense(monkeypatch)
    assert prepare_operator_qp(p).preconditioner is None
    workspace = prepare_operator_qp(p, OperatorQPConfig(cg_preconditioner="jacobi", rho=2, sigma=0.5))
    assert torch.equal(workspace.preconditioner.diagonal, torch.tensor([6.5, 5.5], dtype=p.q.dtype))
    first, second = workspace.solve(), workspace.solve()
    assert first.status == second.status == "solved"
    assert torch.equal(first.x, second.x) and torch.equal(first.y, second.y)
    assert first.cg_iterations == second.cg_iterations


@pytest.mark.parametrize("case", ["P", "A", "both"])
def test_matrix_free_automatic_jacobi_is_rejected_and_explicit_is_supported(case):
    p = small_problem()
    diagonal = p.P.diagonal() + p.A.square().sum(0) + 1e-6
    if case in ("P", "both"):
        p = replace(p, P=as_operator(p.P, "matrix_free"))
    if case in ("A", "both"):
        p = replace(p, A=as_operator(p.A, "matrix_free"))
    with pytest.raises(ValueError, match="matrix-free callers must explicitly"):
        prepare_operator_qp(p, OperatorQPConfig(cg_preconditioner="jacobi"))
    result = solve_operator_qp(p, preconditioner=JacobiPreconditioner(diagonal))
    assert result.status == "solved" and result.inner_residual <= result.inner_tolerance


@pytest.mark.parametrize("assumption", [False, None, 1, "yes"])
def test_jacobi_never_bypasses_assume_psd(assumption):
    with pytest.raises(ValueError, match="assume_psd=True"):
        prepare_operator_qp(
            replace(small_problem(), assume_psd=assumption),
            OperatorQPConfig(cg_preconditioner="jacobi"),
        )


@pytest.mark.parametrize("kind", ["dense", "coo", "csr"])
def test_negative_P_diagonal_cannot_be_hidden_by_rho_or_sigma(kind):
    p = small_problem()
    p.P[0, 0] = -1e-12
    with pytest.raises(ValueError, match="P Jacobi diagonal is negative"):
        prepare_operator_qp(
            replace(p, P=as_operator(p.P, kind)),
            OperatorQPConfig(cg_preconditioner="jacobi", sigma=100),
        )


@pytest.mark.parametrize("kind", ["dense", "coo", "csr"])
def test_gram_diagonal_overflow_is_loud(kind):
    p = small_problem()
    p.A[0, 0] = 1e200
    with pytest.raises(ValueError, match="Jacobi diagonal must be finite"):
        prepare_operator_qp(
            replace(p, A=as_operator(p.A, kind)), OperatorQPConfig(cg_preconditioner="jacobi"),
        )


def test_conflicting_or_unknown_preconditioner_is_not_ignored():
    p = small_problem()
    with pytest.raises(ValueError, match="not both"):
        prepare_operator_qp(
            p, OperatorQPConfig(cg_preconditioner="jacobi"),
            preconditioner=JacobiPreconditioner(torch.ones(2, dtype=p.q.dtype)),
        )
    with pytest.raises(ValueError, match="cg_preconditioner"):
        prepare_operator_qp(p, OperatorQPConfig(cg_preconditioner="automatic"))


def test_outer_preconditioner_failure_retains_admm_coordinate():
    with pytest.raises(CGConvergenceError, match="ADMM iteration=1") as caught:
        solve_operator_qp(small_problem(), preconditioner=CallbackPreconditioner(lambda r: -r))
    assert caught.value.outer_iteration == 1 and caught.value.__cause__ is not None


@pytest.fixture(scope="module")
def frozen_operators():
    """Execute the committed pre-change module, not a second rewritten recurrence."""
    root = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "show", "a698c67:src/openopt/operators.py"], cwd=root,
        capture_output=True, text=True, check=True,
    ).stdout
    name = "_openopt_frozen_precondition_default"
    module = ModuleType(name)
    sys.modules[name] = module
    try:
        exec(compile(source, "a698c67:src/openopt/operators.py", "exec"), module.__dict__)
        yield module
    finally:
        del sys.modules[name]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["dense", "coo", "csr", "matrix_free"])
def test_default_cg_and_qp_are_bitwise_equal_to_frozen_baseline(frozen_operators, dtype, kind):
    p = small_problem(dtype)
    P, A = as_operator(p.P, kind), as_operator(p.A, kind)
    cg_kwargs = dict(eps_abs=1e-7, eps_rel=1e-7) if dtype == torch.float32 else {}
    rhs = -p.q
    cg = conjugate_gradient(TensorOperator(p.P), rhs, **cg_kwargs)
    old_cg = frozen_operators.conjugate_gradient(frozen_operators.TensorOperator(p.P), rhs, **cg_kwargs)
    assert torch.equal(cg.x, old_cg.x)
    assert (cg.iterations, cg.residual, cg.tolerance) == (
        old_cg.iterations, old_cg.residual, old_cg.tolerance,
    )
    cfg_kwargs = dict(dtype=dtype)
    if dtype == torch.float32:
        cfg_kwargs.update(eps_abs=1e-5, eps_rel=1e-5, cg_eps_abs=1e-7, cg_eps_rel=1e-7)
    result = solve_operator_qp(replace(p, P=P, A=A), OperatorQPConfig(**cfg_kwargs))
    old = frozen_operators.solve_operator_qp(
        frozen_operators.OperatorQPProblem(P, p.q, A, p.l, p.u, assume_psd=True),
        frozen_operators.OperatorQPConfig(**cfg_kwargs),
    )
    assert torch.equal(result.x, old.x) and torch.equal(result.y, old.y)
    for name in result.__dataclass_fields__:
        if name not in ("x", "y"):
            assert getattr(result, name) == getattr(old, name)


def illconditioned_kkt(kind):
    n = 48
    diagonal = torch.logspace(0, 6, n, dtype=torch.float64)
    expected_x = torch.linspace(-0.4, 0.4, n, dtype=torch.float64)
    expected_y = torch.tensor([0.7, -0.4, 0., 0.2], dtype=torch.float64)
    rows, columns = torch.arange(4), torch.arange(n)
    P = torch.sparse_coo_tensor(torch.stack((columns, columns)), diagonal, (n, n)).coalesce()
    A = torch.sparse_coo_tensor(
        torch.stack((rows, rows)), torch.ones(4, dtype=diagonal.dtype), (4, n),
    ).coalesce()
    Ax = expected_x[:4]
    l, u = Ax - 1, Ax + 1
    u[0], l[1], l[3], u[3] = Ax[0], Ax[1], Ax[3], Ax[3]
    ATy = torch.zeros(n, dtype=diagonal.dtype)
    ATy[:4] = expected_y
    q = -diagonal * expected_x - ATy
    system_diagonal = diagonal.clone() + 1e-6
    system_diagonal[:4] += 1
    if kind == "matrix_free":
        def transpose(v):
            result = torch.zeros(n, dtype=v.dtype, device=v.device)
            result[:4] = v
            return result
        P = MatrixFreeOperator((n, n), "cpu", diagonal.dtype, lambda v: diagonal * v, lambda v: diagonal * v)
        A = MatrixFreeOperator((4, n), "cpu", diagonal.dtype, lambda v: v[:4].clone(), transpose)
    elif kind == "csr":
        P, A = P.to_sparse_csr(), A.to_sparse_csr()
    elif kind == "dense":
        P = torch.diag(diagonal)
        A = torch.eye(n, dtype=diagonal.dtype)[:4]
    return (
        OperatorQPProblem(P, q, A, l, u, assume_psd=True),
        diagonal, system_diagonal, expected_x, expected_y,
    )


def kkt_comparison(kind, config, tmp_path):
    """Persist both arms and original-coordinate errors before any acceptance."""
    p, diagonal, system_diagonal, expected_x, expected_y = illconditioned_kkt(kind)
    control = solve_operator_qp(p, config)
    if kind == "matrix_free":
        preconditioned = solve_operator_qp(p, config, preconditioner=JacobiPreconditioner(system_diagonal))
    else:
        preconditioned = solve_operator_qp(p, replace(config, cg_preconditioner="jacobi"))
    d, q, lo, hi = diagonal.numpy(), p.q.numpy(), p.l.numpy(), p.u.numpy()
    x_star, y_star = expected_x.numpy(), expected_y.numpy()
    objective_star = float(0.5 * np.dot(x_star, d * x_star) + np.dot(q, x_star))
    arms = {}
    for name, result in (("control", control), ("jacobi", preconditioned)):
        x, y = result.x.numpy(), result.y.numpy()
        Ax = x[:4]
        ATy = np.zeros_like(x)
        ATy[:4] = y
        stationarity = float(np.max(np.abs(d * x + q + ATy)))
        feasibility = float(np.max(np.abs(Ax - np.clip(Ax, lo, hi))))
        normal = float(np.max(np.abs(Ax - np.clip(Ax + y, lo, hi))))
        x_error, y_error = float(np.max(np.abs(x - x_star))), float(np.max(np.abs(y - y_star)))
        objective = float(0.5 * np.dot(x, d * x) + np.dot(q, x))
        primal_tol = config.eps_abs + config.eps_rel * float(np.max(np.abs(Ax)))
        dual_tol = config.eps_abs + config.eps_rel * max(
            float(np.max(np.abs(d * x))), float(np.max(np.abs(q))), float(np.max(np.abs(ATy))),
        )
        component_stationarity_limits = config.eps_abs + config.eps_rel * np.maximum.reduce([
            np.abs(d * x), np.abs(q), np.abs(ATy),
        ])
        arms[name] = dict(
            status=result.status, outer_iterations=result.iterations,
            cg_iterations=result.cg_iterations, stationarity=stationarity,
            feasibility=feasibility, normal_cone=normal, x_error=x_error, y_error=y_error,
            objective_error=abs(objective - objective_star),
            inner_residual=result.inner_residual, inner_tolerance=result.inner_tolerance,
            primal_tolerance=primal_tol, dual_tolerance=dual_tol,
            feasibility_excess=feasibility - primal_tol,
            normal_cone_excess=normal - primal_tol, stationarity_excess=stationarity - dual_tol,
            component_stationarity_excess=float(np.max(
                np.abs(d * x + q + ATy) - component_stationarity_limits,
            )),
            solver_primal=result.primal_residual, solver_dual=result.dual_residual,
            objective_report_error=abs(objective - result.objective),
            x_error_coordinate=int(np.argmax(np.abs(x - x_star))),
            y_error_coordinate=int(np.argmax(np.abs(y - y_star))),
            convexity_certified=result.convexity_certified, x=x.tolist(), y=y.tolist(),
        )
    report = dict(
        schema_version=1, device="cpu", dtype="float64", layout=kind, n=48,
        system_diagonal_condition=float(system_diagonal.max() / system_diagonal.min()),
        qualification="synthetic CPU correctness and iteration counts only; GPU UNVERIFIED",
        tolerances=dict(
            cg_eps_abs=config.cg_eps_abs, cg_eps_rel=config.cg_eps_rel,
            eps_abs=config.eps_abs, eps_rel=config.eps_rel,
        ),
        witness=dict(
            diagonal=d.tolist(), q=q.tolist(), l=lo.tolist(), u=hi.tolist(),
            x=x_star.tolist(), y=y_star.tolist(),
            stationarity=float(np.max(np.abs(d * x_star + q + np.r_[y_star, np.zeros(44)]))),
        ), arms=arms,
    )
    evidence = tmp_path / f"known-kkt-{kind}.json"
    evidence.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="ascii")
    assert json.loads(evidence.read_text()) == report
    return report


@pytest.mark.parametrize("kind", ["dense", "coo", "csr", "matrix_free"])
def test_illconditioned_known_kkt_control_has_independent_numpy_evidence(kind, monkeypatch, tmp_path):
    if kind != "dense":
        forbid_dense(monkeypatch)
    config = OperatorQPConfig(
        cg_max_iterations=2000, cg_eps_abs=1e-12, cg_eps_rel=1e-15,
        eps_abs=1e-9, eps_rel=1e-11,
    )
    report = kkt_comparison(kind, config, tmp_path)
    for arm in report["arms"].values():
        assert arm["status"] == "solved" and not arm["convexity_certified"], arm
        assert arm["inner_residual"] <= arm["inner_tolerance"]
        assert arm["stationarity_excess"] <= 0
        assert arm["feasibility_excess"] <= 0 and arm["normal_cone_excess"] <= 0
        assert arm["x_error"] <= 1e-7 and arm["y_error"] <= 1e-6
        assert arm["objective_error"] <= 1e-6 and arm["objective_report_error"] <= 1e-8
    assert report["arms"]["jacobi"]["cg_iterations"] < report["arms"]["control"]["cg_iterations"]


@pytest.mark.parametrize("kind", ["dense", "coo", "csr", "matrix_free"])
def test_original_inner_accuracy_negative_is_preserved_not_promoted(kind, monkeypatch, tmp_path):
    if kind != "dense":
        forbid_dense(monkeypatch)
    original_config = OperatorQPConfig(
        cg_max_iterations=2000, cg_eps_abs=1e-11, cg_eps_rel=1e-13,
        eps_abs=1e-9, eps_rel=1e-11,
    )
    report = kkt_comparison(kind, original_config, tmp_path)
    control = report["arms"]["control"]
    jacobi = report["arms"]["jacobi"]
    assert control["status"] == "max_iterations" and control["outer_iterations"] == 5000
    assert control["inner_residual"] <= control["inner_tolerance"]
    assert control["stationarity_excess"] <= 0
    assert control["component_stationarity_excess"] > 0
    assert jacobi["status"] == "max_iterations" and jacobi["outer_iterations"] == 5000
    assert jacobi["inner_residual"] <= jacobi["inner_tolerance"]
    assert jacobi["normal_cone_excess"] > 0
    assert report["witness"]["stationarity"] < 1e-12
    for arm in report["arms"].values():
        assert not arm["convexity_certified"]
        if arm["status"] == "solved":
            assert arm["stationarity_excess"] <= 0
            assert arm["normal_cone_excess"] <= 0 and arm["feasibility_excess"] <= 0


@pytest.mark.parametrize("kind", ["coo", "csr"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_cuda_sparse_jacobi_true_residual_and_kkt(kind, dtype, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit CUDA sparse Jacobi acceptance UNVERIFIED")
    device = torch.device("cuda:0")
    p = small_problem(dtype)
    problem = replace(p, P=as_operator(p.P, kind), A=as_operator(p.A, kind))
    forbid_dense(monkeypatch)
    config = OperatorQPConfig(
        device="cuda:0", dtype=dtype, cg_preconditioner="jacobi",
        eps_abs=1e-5 if dtype == torch.float32 else 1e-9,
        eps_rel=1e-5 if dtype == torch.float32 else 1e-9,
        cg_eps_abs=1e-7 if dtype == torch.float32 else 1e-12,
        cg_eps_rel=1e-7 if dtype == torch.float32 else 1e-12,
    )
    workspace = prepare_operator_qp(problem, config)
    assert workspace.P.device == workspace.A.device == device
    assert workspace.P.operator.matrix.device == workspace.A.operator.matrix.device == device
    assert workspace.P.operator.matrix.layout == problem.P.layout
    assert workspace.A.operator.matrix.layout == problem.A.layout
    assert workspace.preconditioner.device == device
    rhs = -workspace.q
    inner = conjugate_gradient(
        workspace.system, rhs, preconditioner=workspace.preconditioner,
        eps_abs=config.cg_eps_abs, eps_rel=config.cg_eps_rel,
    )
    assert inner.x.device == device and inner.x.dtype == dtype
    true_residual = float(torch.linalg.vector_norm(rhs - workspace.system.matvec(inner.x)))
    assert inner.residual == true_residual and true_residual <= inner.tolerance
    K_cpu = p.P + config.rho * (p.A.T @ p.A) + config.sigma * torch.eye(2, dtype=dtype)
    oracle = np.linalg.norm(
        rhs.cpu().numpy().astype(np.float64)
        - K_cpu.numpy().astype(np.float64) @ inner.x.cpu().numpy().astype(np.float64),
    )
    rounding = 8 * torch.finfo(dtype).eps * float(torch.linalg.vector_norm(rhs))
    assert oracle <= inner.tolerance + rounding
    result = workspace.solve()
    assert result.status == "solved" and result.x.device == result.y.device == device
    assert result.inner_residual <= result.inner_tolerance
    x, y = result.x.cpu().numpy().astype(np.float64), result.y.cpu().numpy().astype(np.float64)
    P, A, q = p.P.numpy().astype(np.float64), p.A.numpy().astype(np.float64), p.q.numpy().astype(np.float64)
    lo, hi = p.l.numpy().astype(np.float64), p.u.numpy().astype(np.float64)
    Px, Ax, ATy = P @ x, A @ x, A.T @ y
    primal_tol = config.eps_abs + config.eps_rel * np.max(np.abs(Ax))
    dual_tol = config.eps_abs + config.eps_rel * max(
        np.max(np.abs(Px)), np.max(np.abs(q)), np.max(np.abs(ATy)),
    )
    assert np.max(np.abs(Ax - np.clip(Ax, lo, hi))) <= primal_tol + rounding
    assert np.max(np.abs(Ax - np.clip(Ax + y, lo, hi))) <= primal_tol + rounding
    assert np.max(np.abs(Px + q + ATy)) <= dual_tol + rounding
    assert not result.convexity_certified


@pytest.mark.parametrize("kind", ["coo", "csr"])
def test_explicit_cuda_sparse_jacobi_cannot_accept_false_recurrence(kind):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; explicit CUDA true-residual negative UNVERIFIED")
    device = torch.device("cuda:0")
    matrix = as_operator(torch.eye(2, dtype=torch.float64), kind).to(device)
    tensor_operator = TensorOperator(matrix)
    calls = []

    def changing(v):
        calls.append(1)
        product = tensor_operator.matvec(v)
        return product if len(calls) < 3 else 0.5 * product

    system = MatrixFreeOperator((2, 2), device, torch.float64, changing, changing)
    with pytest.raises(CGConvergenceError, match="iteration limit") as caught:
        conjugate_gradient(
            system, torch.ones(2, dtype=torch.float64, device=device), max_iterations=1,
            preconditioner=JacobiPreconditioner(tensor_operator.diagonal()),
        )
    assert caught.value.residual == pytest.approx(2 ** 0.5 / 2)
    assert caught.value.residual > caught.value.tolerance and len(calls) == 3
