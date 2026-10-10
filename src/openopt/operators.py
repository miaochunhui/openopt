"""Experimental sparse/matrix-free QP, separate from the dense QP core.

Only real float32/float64, fixed-rho ADMM and unpreconditioned CG are supported.
Finite operator probes are diagnostics, NOT a convexity certificate. Callers
must explicitly declare that P is symmetric positive semidefinite.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Protocol, runtime_checkable

import torch

__all__ = [
    "LinearOperator", "TensorOperator", "MatrixFreeOperator",
    "CGResult", "CGConvergenceError", "conjugate_gradient",
    "OperatorQPProblem", "OperatorQPConfig", "OperatorQPResult",
    "PreparedOperatorQP", "prepare_operator_qp", "solve_operator_qp",
]


@runtime_checkable
class LinearOperator(Protocol):
    shape: tuple[int, int]
    device: torch.device
    dtype: torch.dtype

    def matvec(self, vector: torch.Tensor) -> torch.Tensor: ...

    def rmatvec(self, vector: torch.Tensor) -> torch.Tensor: ...


def _metadata(shape: tuple[int, int], device: torch.device, dtype: torch.dtype) -> None:
    if (
        not isinstance(shape, tuple) or len(shape) != 2
        or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in shape)
    ):
        raise ValueError("operator shape must be a pair of nonnegative integers")
    if not isinstance(device, torch.device) or device.type not in ("cpu", "cuda"):
        raise ValueError("operator device must be a CPU/CUDA torch.device")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("operators support real float32/float64 only")


def _vector(
    vector: torch.Tensor, size: int, device: torch.device, dtype: torch.dtype, label: str,
) -> None:
    if not isinstance(vector, torch.Tensor):
        raise TypeError(f"{label} must be a tensor")
    if vector.layout != torch.strided or vector.shape != (size,):
        raise ValueError(f"{label} must be a dense vector of shape ({size},)")
    if vector.device != device or vector.dtype != dtype:
        raise ValueError(f"{label} device/dtype differs from operator metadata")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError(f"{label} must be finite")


class TensorOperator:
    """Owned dense, sparse COO or sparse CSR matrix; never densifies.

    The CSR transpose is cached as COO because sparse-mm CSC support varies
    between runtimes. This is a sparse-to-sparse conversion, not an A' A cache.
    """

    def __init__(self, matrix: torch.Tensor):
        if not isinstance(matrix, torch.Tensor):
            raise TypeError("operator matrix must be a tensor")
        if matrix.layout not in (torch.strided, torch.sparse_coo, torch.sparse_csr):
            raise ValueError("operator matrix must be dense, sparse COO or sparse CSR")
        _metadata(tuple(matrix.shape), matrix.device, matrix.dtype)
        if matrix.layout == torch.sparse_coo and (
            matrix.sparse_dim() != 2 or matrix.dense_dim() != 0
        ):
            raise ValueError("hybrid sparse tensors are unsupported")
        # PyTorch sparse constructors normally skip structural validation.
        # Reconstruct with invariants enabled before coalescing or invoking mm.
        if matrix.layout == torch.sparse_coo:
            matrix = torch.sparse_coo_tensor(
                matrix._indices(), matrix._values(), matrix.shape,
                dtype=matrix.dtype, device=matrix.device, check_invariants=True,
            )
        elif matrix.layout == torch.sparse_csr:
            matrix = torch.sparse_csr_tensor(
                matrix.crow_indices(), matrix.col_indices(), matrix.values(), matrix.shape,
                dtype=matrix.dtype, device=matrix.device, check_invariants=True,
            )
        owned = matrix.detach().clone()
        if owned.layout == torch.sparse_coo:
            owned = owned.coalesce()
        values = owned if owned.layout == torch.strided else owned.values()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("operator matrix must be finite")
        self.matrix = owned
        self.shape, self.device, self.dtype = tuple(owned.shape), owned.device, owned.dtype
        if owned.layout == torch.strided:
            self._transpose = owned.T
        elif owned.layout == torch.sparse_csr:
            self._transpose = owned.to_sparse_coo().transpose(0, 1).coalesce()
        else:
            self._transpose = owned.transpose(0, 1).coalesce()

    @staticmethod
    def _multiply(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        if matrix.layout == torch.strided:
            return matrix @ vector
        return torch.sparse.mm(matrix, vector[:, None]).squeeze(1)

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        _vector(vector, self.shape[1], self.device, self.dtype, "matvec input")
        return self._multiply(self.matrix, vector)

    def rmatvec(self, vector: torch.Tensor) -> torch.Tensor:
        _vector(vector, self.shape[0], self.device, self.dtype, "rmatvec input")
        return self._multiply(self._transpose, vector)


@dataclass(frozen=True)
class MatrixFreeOperator:
    """Borrowed deterministic linear callbacks, already resident on device.

    Callbacks must not mutate inputs or their mathematical operator during a
    prepared solve. Arbitrary callback state cannot be copied or certified.
    """

    shape: tuple[int, int]
    device: torch.device
    dtype: torch.dtype
    matvec: Callable[[torch.Tensor], torch.Tensor]
    rmatvec: Callable[[torch.Tensor], torch.Tensor]

    def __post_init__(self) -> None:
        object.__setattr__(self, "device", torch.device(self.device))
        _metadata(self.shape, self.device, self.dtype)
        if not callable(self.matvec) or not callable(self.rmatvec):
            raise TypeError("matvec/rmatvec must be callable")


class _CheckedOperator:
    def __init__(self, operator: LinearOperator, label: str):
        if not isinstance(operator, LinearOperator):
            raise TypeError(f"{label} must implement the LinearOperator contract")
        _metadata(operator.shape, operator.device, operator.dtype)
        self.operator, self.label = operator, label
        self.shape, self.device, self.dtype = operator.shape, operator.device, operator.dtype

    def _apply(self, vector: torch.Tensor, transpose: bool) -> torch.Tensor:
        method = "rmatvec" if transpose else "matvec"
        incoming, outgoing = self.shape if transpose else self.shape[::-1]
        _vector(vector, incoming, self.device, self.dtype, f"{self.label}.{method} input")
        try:
            result = getattr(self.operator, method)(vector)
        except Exception as error:
            raise RuntimeError(f"{self.label}.{method} callback failed: {error}") from error
        _vector(result, outgoing, self.device, self.dtype, f"{self.label}.{method} output")
        return result

    def matvec(self, vector: torch.Tensor) -> torch.Tensor:
        return self._apply(vector, False)

    def rmatvec(self, vector: torch.Tensor) -> torch.Tensor:
        return self._apply(vector, True)


def _norm(vector: torch.Tensor) -> float:
    return float(vector.abs().max()) if vector.numel() else 0.0


def _l2(vector: torch.Tensor) -> float:
    value = float(torch.linalg.vector_norm(vector))
    if not math.isfinite(value):
        raise RuntimeError("nonfinite L2 residual")
    return value


def _positive(name: str, value: float) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


def _integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class CGResult:
    x: torch.Tensor
    iterations: int
    residual: float
    tolerance: float


class CGConvergenceError(RuntimeError):
    """Inner failure with true residual and exact inner/outer coordinates."""

    def __init__(
        self, reason: str, iterations: int, residual: float, tolerance: float,
        outer_iteration: int | None = None,
    ):
        self.reason, self.iterations = reason, iterations
        self.residual, self.tolerance = residual, tolerance
        self.outer_iteration = outer_iteration
        coordinate = "" if outer_iteration is None else f" ADMM iteration={outer_iteration};"
        super().__init__(
            f"CG failed:{coordinate} {reason}; iterations={iterations}; "
            f"true_residual={residual:.17g}; tolerance={tolerance:.17g}"
        )


@torch.no_grad()
def conjugate_gradient(
    operator: LinearOperator, rhs: torch.Tensor, *, initial: torch.Tensor | None = None,
    max_iterations: int = 1000, eps_abs: float = 1e-10, eps_rel: float = 1e-10,
) -> CGResult:
    """Solve an assumed SPD system; every accepted residual is recomputed.

    Recurrence residuals only nominate convergence. If a nominated iterate
    fails the true b-Kx test, restart the direction with the true residual,
    within the original iteration budget. Never return a failed inner solve.
    """
    _integer("CG max_iterations", max_iterations)
    _positive("CG eps_abs", eps_abs)
    _positive("CG eps_rel", eps_rel)
    K = _CheckedOperator(operator, "CG system")
    if K.shape[0] != K.shape[1] or not K.shape[0]:
        raise ValueError("CG system must be nonempty and square")
    _vector(rhs, K.shape[0], K.device, K.dtype, "CG rhs")
    b = rhs.detach().clone()
    if initial is not None:
        _vector(initial, K.shape[1], K.device, K.dtype, "CG initial")
    x = torch.zeros_like(b) if initial is None else initial.detach().clone()
    tolerance = eps_abs + eps_rel * _l2(b)
    if not math.isfinite(tolerance):
        raise ValueError("CG tolerance overflow")
    residual = b - K.matvec(x)
    true_norm = _l2(residual)
    if true_norm <= tolerance:
        return CGResult(x, 0, true_norm, tolerance)
    direction = residual.clone()
    squared = residual @ residual
    for iteration in range(1, max_iterations + 1):
        product = K.matvec(direction)
        curvature = float(direction @ product)
        if not math.isfinite(curvature) or curvature <= 0 or not bool(torch.isfinite(squared)):
            true_norm = _l2(b - K.matvec(x))
            raise CGConvergenceError(
                f"nonpositive/nonfinite curvature ({curvature}) or residual square",
                iteration, true_norm, tolerance,
            )
        alpha = squared / curvature
        x = x + alpha * direction
        _vector(x, K.shape[1], K.device, K.dtype, "CG iterate")
        residual = residual - alpha * product
        if _l2(residual) <= tolerance or iteration == max_iterations:
            true_residual = b - K.matvec(x)
            true_norm = _l2(true_residual)
            if true_norm <= tolerance:
                return CGResult(x, iteration, true_norm, tolerance)
            if iteration == max_iterations:
                raise CGConvergenceError("iteration limit", iteration, true_norm, tolerance)
            residual = true_residual
            squared = residual @ residual
            direction = residual.clone()
            continue
        updated = residual @ residual
        if not bool(torch.isfinite(updated)) or float(squared) <= 0:
            true_norm = _l2(b - K.matvec(x))
            raise CGConvergenceError("invalid residual recurrence", iteration, true_norm, tolerance)
        direction = residual + (updated / squared) * direction
        squared = updated
    raise AssertionError("validated positive CG iteration budget was not entered")


@dataclass(frozen=True)
class OperatorQPProblem:
    P: torch.Tensor | LinearOperator
    q: torch.Tensor
    A: torch.Tensor | LinearOperator
    l: torch.Tensor
    u: torch.Tensor
    assume_psd: bool = False


@dataclass(frozen=True)
class OperatorQPConfig:
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    rho: float = 1.0
    sigma: float = 1e-6
    relaxation: float = 1.6
    max_iterations: int = 5000
    check_interval: int = 25
    eps_abs: float = 1e-7
    eps_rel: float = 1e-7
    cg_max_iterations: int = 1000
    cg_eps_abs: float = 1e-10
    cg_eps_rel: float = 1e-10

    def validate(self) -> torch.device:
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("operator QP supports float32/float64 only")
        for name in ("rho", "sigma", "eps_abs", "eps_rel", "cg_eps_abs", "cg_eps_rel"):
            _positive(name, getattr(self, name))
        for name in ("max_iterations", "check_interval", "cg_max_iterations"):
            _integer(name, getattr(self, name))
        if not math.isfinite(self.relaxation) or not 0 < self.relaxation < 2:
            raise ValueError("relaxation must be in (0, 2)")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("operator QP supports CPU or CUDA only")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("explicit CUDA requested but unavailable; no CPU fallback")
            index = torch.cuda.current_device() if device.index is None else device.index
            if index >= torch.cuda.device_count():
                raise ValueError("CUDA device index out of range; no CPU fallback")
            return torch.device("cuda", index)
        return torch.device("cpu")


@dataclass(frozen=True)
class OperatorQPResult:
    x: torch.Tensor
    y: torch.Tensor
    status: str
    iterations: int
    objective: float
    primal_residual: float
    dual_residual: float
    normal_cone_residual: float
    cg_iterations: int
    inner_residual: float
    inner_tolerance: float
    experimental: bool = True
    convexity_certified: bool = False


def _coerce(
    value: torch.Tensor | LinearOperator, device: torch.device, dtype: torch.dtype, label: str,
) -> _CheckedOperator:
    if isinstance(value, torch.Tensor):
        # Validate source before conversion: complex values must never be discarded.
        value = TensorOperator(value)
    if isinstance(value, TensorOperator):
        value = TensorOperator(value.matrix.to(device=device, dtype=dtype))
    checked = _CheckedOperator(value, label)
    if checked.device != device or checked.dtype != dtype:
        raise ValueError(f"{label} matrix-free device/dtype must match config; no fallback/transfer")
    return checked


def _probe(operator: _CheckedOperator, *, quadratic: bool = False) -> None:
    """Small deterministic diagnostics, deliberately not full operator/PSD proof."""
    rows, columns = operator.shape
    device, dtype = operator.device, operator.dtype
    tolerance = 100 * torch.finfo(dtype).eps * math.sqrt(max(rows, columns, 1))
    zeros_n = torch.zeros(columns, device=device, dtype=dtype)
    zeros_m = torch.zeros(rows, device=device, dtype=dtype)
    if _norm(operator.matvec(zeros_n)) > tolerance or _norm(operator.rmatvec(zeros_m)) > tolerance:
        raise ValueError(f"{operator.label} is not linear at zero")
    column_index = torch.arange(columns, device=device, dtype=dtype)
    row_index = torch.arange(rows, device=device, dtype=dtype)
    for trial in range(3):
        x = torch.cos(column_index * math.sqrt(2) + trial)
        v = torch.sin(column_index * math.sqrt(3) + trial + 0.7)
        y = torch.sin(row_index * math.sqrt(5) + trial + 0.3)
        Ax, ATy = operator.matvec(x), operator.rmatvec(y)
        left, right = float(y @ Ax), float(x @ ATy)
        scale = max(1.0, _l2(y) * _l2(Ax), _l2(x) * _l2(ATy))
        if not math.isfinite(left) or not math.isfinite(right) or abs(left - right) > tolerance * scale:
            raise ValueError(f"{operator.label} adjoint probe failed (trial={trial})")
        for method, first, second in (
            (operator.matvec, x, v),
            (operator.rmatvec, y, torch.cos(row_index + trial)),
        ):
            expected = method(first) + 0.37 * method(second)
            actual = method(first + 0.37 * second)
            if _norm(actual - expected) > tolerance * max(1.0, _norm(actual), _norm(expected)):
                raise ValueError(f"{operator.label} linearity probe failed (trial={trial})")
        if quadratic:
            transpose_x = operator.rmatvec(x)
            if _norm(Ax - transpose_x) > tolerance * max(1.0, _norm(Ax), _norm(transpose_x)):
                raise ValueError(f"P symmetry probe failed (trial={trial})")
            curvature = float(x @ Ax)
            if not math.isfinite(curvature) or curvature < -tolerance * max(1.0, _l2(x) * _l2(Ax)):
                raise ValueError(f"P negative curvature probe failed (trial={trial}); assumption contradicted")
    if quadratic and isinstance(operator.operator, TensorOperator):
        matrix = operator.operator.matrix
        if matrix.layout == torch.strided:
            difference, values = matrix - matrix.T, matrix
        else:
            coo = matrix.to_sparse_coo().coalesce()
            difference = (coo - coo.transpose(0, 1)).coalesce().values()
            values = coo.values()
        if _norm(difference) > 20 * torch.finfo(dtype).eps * max(1.0, _norm(values)):
            raise ValueError("P matrix must be symmetric")


class PreparedOperatorQP:
    """Owned tensor workspace, borrowed matrix-free callbacks, zero-start solves."""

    @torch.no_grad()
    def __init__(
        self, problem: OperatorQPProblem, config: OperatorQPConfig = OperatorQPConfig(),
    ):
        device = config.validate()
        if problem.assume_psd is not True:
            raise ValueError(
                "experimental operator QP requires assume_psd=True: caller declares "
                "P symmetric positive semidefinite; convexity is NOT certified"
            )
        vectors = (problem.q, problem.l, problem.u)
        if not all(isinstance(t, torch.Tensor) for t in vectors):
            raise TypeError("q/l/u must be tensors")
        if any(
            t.layout != torch.strided or t.dtype not in (torch.float32, torch.float64)
            for t in vectors
        ):
            raise ValueError("q/l/u must be real dense float32/float64 vectors")
        q, l, u = [t.detach().to(device=device, dtype=config.dtype).clone() for t in vectors]
        if q.ndim != 1 or not q.numel():
            raise ValueError("q must be a nonempty vector")
        if not bool(torch.isfinite(q).all()):
            raise ValueError("q must be finite")
        P, A = _coerce(problem.P, device, config.dtype, "P"), _coerce(problem.A, device, config.dtype, "A")
        if P.shape != (q.numel(), q.numel()) or A.shape[1] != q.numel():
            raise ValueError("P/A dimensions must match q")
        if l.shape != (A.shape[0],) or u.shape != l.shape:
            raise ValueError("bounds must match A rows")
        if bool(torch.isnan(l).any() | torch.isnan(u).any()):
            raise ValueError("bounds cannot contain NaN")
        if bool(torch.isposinf(l).any() | torch.isneginf(u).any() | (l > u).any()):
            raise ValueError("invalid/reversed bounds")
        _probe(P, quadratic=True)
        _probe(A)
        self.config, self.P, self.A = config, P, A
        self.q, self.l, self.u = q, l, u
        self.system = MatrixFreeOperator(
            P.shape, device, config.dtype,
            self._system_matvec, self._system_matvec,
        )

    def _system_matvec(self, vector: torch.Tensor) -> torch.Tensor:
        return (
            self.P.matvec(vector)
            + self.config.rho * self.A.rmatvec(self.A.matvec(vector))
            + self.config.sigma * vector
        )

    @torch.no_grad()
    def solve(self) -> OperatorQPResult:
        c, P, A, q, l, u = self.config, self.P, self.A, self.q, self.l, self.u
        x, y = torch.zeros_like(q), torch.zeros_like(l)
        z = torch.minimum(torch.maximum(torch.zeros_like(l), l), u)
        status = "max_iterations"
        cg_iterations = 0
        primal = dual = normal = math.inf
        for iteration in range(1, c.max_iterations + 1):
            rhs = c.sigma * x - q + A.rmatvec(c.rho * z - y)
            try:
                inner = conjugate_gradient(
                    self.system, rhs, initial=x, max_iterations=c.cg_max_iterations,
                    eps_abs=c.cg_eps_abs, eps_rel=c.cg_eps_rel,
                )
            except CGConvergenceError as error:
                raise CGConvergenceError(
                    error.reason, error.iterations, error.residual, error.tolerance, iteration,
                ) from error
            x = inner.x
            cg_iterations += inner.iterations
            Ax = A.matvec(x)
            relaxed = c.relaxation * Ax + (1 - c.relaxation) * z
            z = torch.minimum(torch.maximum(relaxed + y / c.rho, l), u)
            y = y + c.rho * (relaxed - z)
            if not bool(torch.isfinite(z).all() & torch.isfinite(y).all()):
                raise RuntimeError(f"operator ADMM nonfinite iterate at iteration={iteration}")
            if iteration % c.check_interval == 0 or iteration == c.max_iterations:
                Px, ATy = P.matvec(x), A.rmatvec(y)
                primal = max(
                    _norm(Ax - z), _norm(Ax - torch.minimum(torch.maximum(Ax, l), u)),
                )
                dual = _norm(Px + q + ATy)
                normal = _norm(Ax - torch.minimum(torch.maximum(Ax + y, l), u))
                if not all(math.isfinite(value) for value in (primal, dual, normal)):
                    raise RuntimeError(f"operator ADMM nonfinite KKT residual at iteration={iteration}")
                primal_tol = c.eps_abs + c.eps_rel * max(_norm(Ax), _norm(z))
                dual_tol = c.eps_abs + c.eps_rel * max(_norm(Px), _norm(q), _norm(ATy))
                if primal <= primal_tol and dual <= dual_tol and normal <= primal_tol:
                    status = "solved"
                    break
        objective = float(0.5 * x @ P.matvec(x) + q @ x)
        if not math.isfinite(objective):
            raise RuntimeError(f"operator QP nonfinite objective at iteration={iteration}")
        return OperatorQPResult(
            x, y, status, iteration, objective, primal, dual, normal,
            cg_iterations, inner.residual, inner.tolerance,
        )


def prepare_operator_qp(
    problem: OperatorQPProblem, config: OperatorQPConfig = OperatorQPConfig(),
) -> PreparedOperatorQP:
    return PreparedOperatorQP(problem, config)


def solve_operator_qp(
    problem: OperatorQPProblem, config: OperatorQPConfig = OperatorQPConfig(),
) -> OperatorQPResult:
    return prepare_operator_qp(problem, config).solve()
