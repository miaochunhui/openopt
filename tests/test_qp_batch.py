"""Batch acceptance: scalar oracle, independent KKT and fail-loud controls."""

from dataclasses import dataclass, replace
import inspect

import pytest
import torch

from openopt.batch import BatchQPProblem, BatchQPResult, PreparedBatchQP, prepare_batch_qp, solve_batch_qp
from openopt.qp import QPConfig, QPProblem, solve_qp


def problem(seed=41, n=5):
    generator = torch.Generator().manual_seed(seed)
    B = torch.randn((n, n), generator=generator, dtype=torch.float64)
    P = torch.eye(n, dtype=torch.float64) + B.T @ B / n
    A = torch.cat((torch.eye(n, dtype=torch.float64),
                   torch.randn((2, n), generator=generator, dtype=torch.float64)))
    witness = torch.rand(n, generator=generator, dtype=torch.float64) - 0.5
    Ax = A @ witness
    return QPProblem(P, torch.randn(n, generator=generator, dtype=torch.float64), A, Ax - 0.3, Ax + 0.3)


def norm(vector):
    return float(vector.abs().max()) if vector.numel() else 0.0


def check_kkt(p, result, index, config):
    x, y = result.x[index], result.y[index]
    P, q, A, l, u = (t.to(x) for t in (p.P, p.q, p.A, p.l, p.u))
    Ax, Px, ATy = A @ x, P @ x, A.T @ y
    projected = torch.minimum(torch.maximum(Ax, l), u)
    feasible = norm(Ax - projected)
    stationary = norm(Px + q + ATy)
    cone = norm(Ax - torch.minimum(torch.maximum(Ax + y, l), u))
    primal_tol = config.eps_abs + config.eps_rel * max(norm(Ax), norm(projected))
    dual_tol = config.eps_abs + config.eps_rel * max(norm(Px), norm(q), norm(ATy))
    assert feasible < 4 * primal_tol
    assert stationary <= 2 * dual_tol
    assert cone < 6 * primal_tol
    assert float(result.objective[index]) == pytest.approx(float(0.5 * x @ Px + q @ x), abs=1e-6)


@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
@pytest.mark.parametrize("dtype,tolerance", [(torch.float64, 1e-7), (torch.float32, 1e-5)])
def test_batch_matches_each_original_solver_and_independent_kkt(dtype, tolerance, linear_solver):
    problems = [problem(seed) for seed in (41, 82, 123, 164)]
    config = QPConfig(dtype=dtype, eps_abs=tolerance, eps_rel=tolerance, linear_solver=linear_solver)
    batch = BatchQPProblem.stack(problems)
    result = solve_batch_qp(batch, config)
    assert result.status == ("solved",) * len(problems)
    assert result.x.shape == (4, 5) and result.y.shape == (4, 7)
    assert result.iterations.dtype == torch.int64
    for name in ("x", "y", "iterations", "objective", "primal_residual", "dual_residual", "normal_cone_residual"):
        assert getattr(result, name).device.type == "cpu"
    assert result.x.dtype == dtype and not result.x.requires_grad
    for index, p in enumerate(problems):
        reference = solve_qp(p, config)
        assert reference.status == result.status[index]
        assert int(result.iterations[index]) == reference.iterations
        assert torch.allclose(result.x[index], reference.x, atol=30 * tolerance, rtol=30 * tolerance)
        assert torch.allclose(result.y[index], reference.y, atol=30 * tolerance, rtol=30 * tolerance)
        for name in ("objective", "primal_residual", "dual_residual", "normal_cone_residual"):
            assert float(getattr(result, name)[index]) == pytest.approx(getattr(reference, name), abs=30 * tolerance)
        check_kkt(p, result, index, config)


@pytest.mark.parametrize("check_interval", [1, 3, 100])
def test_mixed_solved_unconverged_freezes_each_item_and_real_iterations(check_interval):
    P, A = torch.ones((1, 1), dtype=torch.float64), torch.ones((2, 1), dtype=torch.float64)
    easy = QPProblem(P, torch.zeros(1, dtype=torch.float64), A,
                     -torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))
    hard = QPProblem(P, torch.zeros(1, dtype=torch.float64), A,
                     torch.tensor([0., 2.], dtype=torch.float64),
                     torch.tensor([1., 3.], dtype=torch.float64))
    config = QPConfig(max_iterations=7, check_interval=check_interval)
    result = solve_batch_qp(BatchQPProblem.stack([easy, hard]), config)
    assert result.status == ("solved", "max_iterations")
    assert result.iterations.tolist() == [min(check_interval, 7), 7]
    for index, p in enumerate((easy, hard)):
        reference = solve_qp(p, config)
        assert int(result.iterations[index]) == reference.iterations
        assert torch.allclose(result.x[index], reference.x)
        assert torch.allclose(result.y[index], reference.y)
        assert float(result.primal_residual[index]) == pytest.approx(reference.primal_residual)
    assert float(result.primal_residual[1]) > 0.1
    assert torch.equal(result.x[0], torch.zeros_like(result.x[0]))


def test_unconstrained_and_unbounded_batch_is_not_a_false_certificate():
    P = torch.tensor([[[1.]], [[0.]]], dtype=torch.float64)
    p = BatchQPProblem(P, -torch.ones((2, 1)), torch.empty((2, 0, 1)),
                       torch.empty((2, 0)), torch.empty((2, 0)))
    result = solve_batch_qp(p, QPConfig(max_iterations=8, check_interval=1))
    assert result.status == ("solved", "max_iterations")
    assert result.iterations.tolist() == [2, 8]
    assert result.y.shape == (2, 0)
    assert torch.equal(result.primal_residual, torch.zeros(2, dtype=torch.float64))
    assert float(result.dual_residual[1]) == 1.0


def test_equalities_one_sided_and_fully_infinite_bounds():
    p = QPProblem(
        torch.eye(2, dtype=torch.float64), torch.tensor([-2., -3.], dtype=torch.float64),
        torch.tensor([[1., 1.], [1., 0.], [0., 1.]], dtype=torch.float64),
        torch.tensor([1., -float("inf"), -float("inf")], dtype=torch.float64),
        torch.tensor([1., 0.2, float("inf")], dtype=torch.float64),
    )
    result = solve_batch_qp(BatchQPProblem.stack([p, replace(p, q=p.q + 1)]))
    assert result.status == ("solved", "solved")
    assert torch.allclose(result.x, torch.tensor([[0., 1.], [0., 1.]], dtype=torch.float64), atol=1e-6)
    assert torch.equal(result.y[:, 2], torch.zeros(2, dtype=torch.float64))


def test_semidefinite_bounded_lp():
    p = BatchQPProblem(torch.zeros((2, 2, 2)), torch.tensor([[-1., 2.], [1., -2.]]),
                       torch.eye(2).expand(2, -1, -1), torch.zeros((2, 2)), torch.ones((2, 2)))
    result = solve_batch_qp(p)
    assert result.status == ("solved", "solved")
    assert torch.allclose(result.x, torch.tensor([[1., 0.], [0., 1.]], dtype=torch.float64), atol=1e-6)


def test_prepared_workspace_owns_inputs_and_independent_solve_results():
    p = BatchQPProblem.stack([problem(), problem(82)])
    p.q.requires_grad_(True)
    workspace = prepare_batch_qp(p)
    other = PreparedBatchQP(p)
    first = workspace.solve()
    saved_x, saved_y = first.x.clone(), first.y.clone()
    for name in ("P", "q", "A", "l", "u"):
        source, owned, second_owned = getattr(p, name), getattr(workspace, name), getattr(other, name)
        assert source.data_ptr() != owned.data_ptr() != second_owned.data_ptr()
        assert not owned.requires_grad
        with torch.no_grad():
            source.add_(10)
    second = workspace.solve()
    assert torch.equal(first.x, second.x) and torch.equal(first.y, second.y)
    second.x.add_(99)
    second.y.add_(99)
    third = other.solve()
    assert torch.equal(first.x, saved_x) and torch.equal(first.y, saved_y)
    assert torch.equal(third.x, saved_x) and torch.equal(third.y, saved_y)
    assert first.status == third.status == ("solved", "solved")
    assert torch.equal(first.iterations, third.iterations)


@pytest.mark.parametrize("field", ["P", "q", "A"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_matrix_input_names_bad_item(field, value):
    p = BatchQPProblem.stack([problem(), problem(82)])
    getattr(p, field)[1].flatten()[0] = value
    with pytest.raises(ValueError, match=rf"{field} must be finite; batch items=\[1\]"):
        prepare_batch_qp(p)


@pytest.mark.parametrize("mutation", [
    "nan_lower", "nan_upper", "positive_infinite_lower", "negative_infinite_upper",
    "reversed", "indefinite", "asymmetric",
])
def test_invalid_bounds_and_nonconvex_item_fail_loud(mutation):
    p = BatchQPProblem.stack([problem(), problem(82)])
    if mutation == "nan_lower":
        p.l[1, 0] = float("nan")
    elif mutation == "nan_upper":
        p.u[1, 0] = float("nan")
    elif mutation == "positive_infinite_lower":
        p.l[1, 0] = float("inf")
    elif mutation == "negative_infinite_upper":
        p.u[1, 0] = -float("inf")
    elif mutation == "reversed":
        p.l[1, 0] = p.u[1, 0] + 1
    elif mutation == "indefinite":
        p.P[1] = -torch.eye(5, dtype=torch.float64)
    elif mutation == "asymmetric":
        p.P[1, 0, 1] += 1
    with pytest.raises(ValueError, match=r"batch items=\[1\]"):
        prepare_batch_qp(p)


def test_psd_tolerance_is_per_item_not_largest_matrix_in_batch():
    p = BatchQPProblem(torch.tensor([[[1e20]], [[-1.]]], dtype=torch.float64),
                       torch.zeros((2, 1)), torch.zeros((2, 0, 1)),
                       torch.zeros((2, 0)), torch.zeros((2, 0)))
    with pytest.raises(ValueError, match=r"positive semidefinite; batch items=\[1\]"):
        prepare_batch_qp(p)


@pytest.mark.parametrize("field,value", [
    ("P", torch.ones((2, 5, 4))), ("P", torch.ones((5, 5))),
    ("q", torch.ones(5)), ("q", torch.empty((0, 5))), ("q", torch.empty((2, 0))),
    ("A", torch.ones((2, 7, 4))), ("A", torch.ones((1, 7, 5))),
    ("l", torch.ones((2, 6))), ("u", torch.ones(7)),
])
def test_invalid_shapes_do_not_broadcast(field, value):
    p = BatchQPProblem.stack([problem(), problem(82)])
    with pytest.raises(ValueError, match="shape"):
        prepare_batch_qp(replace(p, **{field: value}))


@pytest.mark.parametrize("variant", ["complex", "sparse", "not_tensor", "meta"])
def test_invalid_tensor_types(variant):
    p = BatchQPProblem.stack([problem(), problem(82)])
    value = {"complex": p.P.to(torch.complex128), "sparse": p.P.to_sparse(),
             "not_tensor": [[1]], "meta": torch.empty(p.P.shape, device="meta")}[variant]
    with pytest.raises((ValueError, TypeError)):
        prepare_batch_qp(replace(p, P=value))


@pytest.mark.parametrize("variant", ["empty", "shape", "dtype", "not_problem", "bad_first"])
def test_stack_rejects_empty_and_implicit_broadcasts(variant):
    p = problem()
    variants = {
        "empty": [],
        "shape": [p, problem(n=6)],
        "dtype": [p, replace(p, q=p.q.float())],
        "not_problem": [p, None],
        "bad_first": [replace(p, A=p.A[:, :-1])],
    }
    with pytest.raises((ValueError, TypeError)):
        BatchQPProblem.stack(variants[variant])


@pytest.mark.parametrize("field,value", [
    ("rho", 0), ("sigma", -1), ("eps_abs", float("nan")), ("eps_rel", float("inf")),
    ("relaxation", 2), ("max_iterations", 0), ("max_iterations", True),
    ("check_interval", 1.5), ("dtype", torch.float16), ("device", "meta"),
    ("linear_solver", "invalid"),
])
def test_invalid_config(field, value):
    with pytest.raises((ValueError, RuntimeError)):
        solve_batch_qp(BatchQPProblem.stack([problem()]), replace(QPConfig(), **{field: value}))


@pytest.mark.parametrize("adaptive_rho", [False, True])
def test_optional_adaptive_config_is_not_silently_ignored(adaptive_rho):
    @dataclass(frozen=True)
    class AdaptiveConfig(QPConfig):
        adaptive_rho: bool = False

    config = AdaptiveConfig(adaptive_rho=adaptive_rho)
    p = BatchQPProblem.stack([problem()])
    if adaptive_rho:
        with pytest.raises(ValueError, match="fixed rho only; adaptive_rho must be False"):
            solve_batch_qp(p, config)
    else:
        assert solve_batch_qp(p, config).status == ("solved",)


def test_explicit_unavailable_cuda_does_not_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA requested but unavailable"):
        prepare_batch_qp(BatchQPProblem.stack([problem()]), QPConfig(device="cuda:0"))


def test_explicit_cuda_bad_index_rejected_before_transfer(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="CUDA device index out of range"):
        prepare_batch_qp(BatchQPProblem.stack([problem()]), QPConfig(device="cuda:1"))


def test_finite_input_system_overflow_has_item_coordinates():
    p = BatchQPProblem(torch.ones((2, 1, 1)), torch.zeros((2, 1)),
                       torch.tensor([[[1.]], [[1e30]]]), torch.zeros((2, 1)), torch.ones((2, 1)))
    with pytest.raises(RuntimeError, match=r"linear system is nonfinite; batch items=\[1\]"):
        prepare_batch_qp(p, QPConfig(dtype=torch.float32))


def test_numerical_failure_is_per_item_not_fake_solved(monkeypatch):
    p = BatchQPProblem.stack([problem(), problem(82)])
    original = torch.cholesky_solve

    def damaged(rhs, factor):
        value = original(rhs, factor)
        value[1] = float("nan")
        return value

    monkeypatch.setattr(torch, "cholesky_solve", damaged)
    result = solve_batch_qp(p, QPConfig(max_iterations=500, check_interval=5))
    assert result.status == ("solved", "numerical_error")
    assert int(result.iterations[1]) == 5
    assert int(result.iterations[0]) > 5
    assert bool(torch.isfinite(result.x[0]).all())
    assert bool(torch.isnan(result.x[1]).all())
    assert not bool(torch.isfinite(result.objective[1]))


def test_actual_overflow_in_iterate_objective_not_marked_solved():
    p = BatchQPProblem(torch.ones((2, 1, 1)), torch.tensor([[0.], [1e30]]),
                       torch.empty((2, 0, 1)), torch.empty((2, 0)), torch.empty((2, 0)))
    result = solve_batch_qp(p, QPConfig(dtype=torch.float32, max_iterations=3, check_interval=1))
    assert result.status == ("solved", "numerical_error")
    assert result.iterations.tolist() == [1, 1]
    assert not bool(torch.isfinite(result.objective[1]))


def test_preparation_performs_one_batched_psd_check_and_factorization(monkeypatch):
    calls = []
    original_eig, original_chol = torch.linalg.eigvalsh, torch.linalg.cholesky_ex

    def eig(matrix):
        calls.append(("eig", tuple(matrix.shape)))
        return original_eig(matrix)

    def chol(matrix):
        calls.append(("chol", tuple(matrix.shape)))
        return original_chol(matrix)

    monkeypatch.setattr(torch.linalg, "eigvalsh", eig)
    monkeypatch.setattr(torch.linalg, "cholesky_ex", chol)
    workspace = prepare_batch_qp(BatchQPProblem.stack([problem(), problem(82), problem(123)]))
    assert calls == [("eig", (3, 5, 5)), ("chol", (3, 5, 5))]
    workspace.solve()
    workspace.solve()
    assert len(calls) == 2


def test_factorization_failure_preserves_item_coordinates(monkeypatch):
    original = torch.linalg.cholesky_ex

    def failed(matrix):
        factor, info = original(matrix)
        info[1] = 1
        return factor, info

    monkeypatch.setattr(torch.linalg, "cholesky_ex", failed)
    with pytest.raises(RuntimeError, match=r"factorization failed; batch items=\[1\]"):
        prepare_batch_qp(BatchQPProblem.stack([problem(), problem(82)]))


@pytest.mark.parametrize("mutation", ["primal", "dual", "objective"])
def test_independent_kkt_ruler_rejects_corrupted_solved_result(mutation):
    p, config = problem(), QPConfig()
    result = solve_batch_qp(BatchQPProblem.stack([p]), config)
    assert isinstance(result, BatchQPResult) and result.status == ("solved",)
    check_kkt(p, result, 0, config)
    changes = {"primal": {"x": result.x + 10},
               "dual": {"y": result.y + 10},
               "objective": {"objective": result.objective + 10}}
    with pytest.raises(AssertionError):
        check_kkt(p, replace(result, **changes[mutation]), 0, config)


@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
def test_factor_application_is_batched_not_per_problem(monkeypatch, linear_solver):
    name = "cholesky_solve" if linear_solver == "cholesky" else "solve_triangular"
    owner = torch if linear_solver == "cholesky" else torch.linalg
    original = getattr(owner, name)
    shapes = []

    def record(*args, **kwargs):
        rhs, factor = (args[0], args[1]) if linear_solver == "cholesky" else (args[1], args[0])
        shapes.append((tuple(rhs.shape), tuple(factor.shape)))
        return original(*args, **kwargs)

    monkeypatch.setattr(owner, name, record)
    result = solve_batch_qp(BatchQPProblem.stack([problem(), problem(82)]),
                            QPConfig(linear_solver=linear_solver, max_iterations=3, check_interval=1))
    assert shapes == [((2, 5, 1), (2, 5, 5))] * (3 if linear_solver == "cholesky" else 6)
    assert result.iterations.tolist() == [3, 3]
    assert "solve_qp(" not in inspect.getsource(PreparedBatchQP.solve)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; batch GPU path unverified")
@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
def test_cuda_batch_resident_outputs_and_cpu_scalar_oracle(linear_solver):
    problems = [problem(), problem(82), problem(123)]
    config = QPConfig(device="cuda:0", linear_solver=linear_solver)
    workspace = prepare_batch_qp(BatchQPProblem.stack(problems), config)
    for name in ("P", "q", "A", "AT", "l", "u", "factor"):
        assert getattr(workspace, name).device == torch.device("cuda:0")
    result = workspace.solve()
    assert result.status == ("solved",) * 3
    for name in ("x", "y", "iterations", "objective", "primal_residual", "dual_residual", "normal_cone_residual"):
        assert getattr(result, name).device == torch.device("cuda:0")
    for index, p in enumerate(problems):
        reference = solve_qp(p, replace(config, device="cpu"))
        assert torch.allclose(result.x[index].cpu(), reference.x, atol=1e-6, rtol=1e-6)
        assert float(result.objective[index]) == pytest.approx(reference.objective, abs=1e-6)
        check_kkt(p, result, index, config)
