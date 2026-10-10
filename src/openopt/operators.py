"""Experimental sparse/matrix-free QP, separate from the dense QP core.

Only real float32/float64, fixed-rho ADMM and opt-in preconditioned CG are supported.
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
    "Preconditioner", "JacobiPreconditioner",
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


def _checked_device_coo(matrix: torch.Tensor) -> torch.Tensor:
    """Check 2-D COO components on their device before unchecked reconstruction.

    CUDA backend invariant validation can allocate CPU range tensors. Replace
    that validator, not the invariants, without trusting the coalesced flag.
    Scalar boolean synchronization is explicit; no index tensor moves to host.
    """
    if not isinstance(matrix, torch.Tensor) or matrix.layout != torch.sparse_coo:
        raise RuntimeError("device COO validation requires a sparse COO tensor")
    _metadata(tuple(matrix.shape), matrix.device, matrix.dtype)
    if matrix.sparse_dim() != 2 or matrix.dense_dim() != 0:
        raise RuntimeError("device COO must have two sparse dimensions and no dense dimensions")
    indices, values = matrix._indices(), matrix._values()
    if (
        indices.layout != torch.strided or indices.dtype != torch.int64
        or indices.device != matrix.device or indices.ndim != 2 or indices.shape[0] != 2
    ):
        raise RuntimeError("device COO indices must be same-device dense int64 with shape (2, nnz)")
    if (
        values.layout != torch.strided or values.device != matrix.device
        or values.dtype != matrix.dtype or values.ndim != 1
        or values.numel() != indices.shape[1] or matrix._nnz() != values.numel()
    ):
        raise RuntimeError("device COO values must match original device/dtype and index nnz")
    for axis, size in enumerate(matrix.shape):
        index = indices[axis]
        if bool((index < 0).any()) or bool((index >= size).any()):
            raise RuntimeError(f"device COO indices out of original bounds on axis {axis}")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("device COO original values must be finite")
    # Unsorted indices and duplicates are legal uncoalesced COO. Reset a
    # potentially false flag, then TensorOperator coalesces its owned copy.
    return torch.sparse_coo_tensor(
        indices, values, matrix.shape, dtype=matrix.dtype, device=matrix.device,
        check_invariants=False, is_coalesced=False,
    )


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
        # Validate before coalescing/mm. Keep CPU/CSR backend checks unchanged;
        # CUDA COO uses equivalent device-local component checks instead.
        if matrix.layout == torch.sparse_coo:
            if matrix.device.type == "cuda":
                matrix = _checked_device_coo(matrix)
            else:
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

    def diagonal(self) -> torch.Tensor:
        """Extract an owned diagonal in O(nnz + n) sparse storage, never densify."""
        if self.shape[0] != self.shape[1]:
            raise ValueError("diagonal requires a square tensor operator")
        if self.matrix.layout == torch.strided:
            return self.matrix.diagonal().clone()
        coo = self.matrix.to_sparse_coo().coalesce()
        rows, columns = coo.indices()
        selected = rows == columns
        result = torch.zeros(self.shape[1], device=self.device, dtype=self.dtype)
        result.index_add_(0, rows[selected], coo.values()[selected])
        return result

    def column_squared_norms(self) -> torch.Tensor:
        """Return diag(A' A) without forming a Gram matrix."""
        if self.matrix.layout == torch.strided:
            return self.matrix.square().sum(dim=0)
        coo = self.matrix.to_sparse_coo().coalesce()
        result = torch.zeros(self.shape[1], device=self.device, dtype=self.dtype)
        result.index_add_(0, coo.indices()[1], coo.values().square())
        return result


@runtime_checkable
class Preconditioner(Protocol):
    """Fixed, deterministic SPD inverse action z=M^-1 r, resident on device.

    Callers declare symmetry/positive definiteness and must not mutate inputs
    or change the action during CG. Positive r'z checks are not an SPD proof.
    """

    shape: tuple[int, int]
    device: torch.device
    dtype: torch.dtype

    def apply(self, residual: torch.Tensor) -> torch.Tensor: ...


class JacobiPreconditioner:
    """Owned finite, strictly positive system diagonal; no clipping or shifts."""

    def __init__(self, diagonal: torch.Tensor):
        if not isinstance(diagonal, torch.Tensor):
            raise TypeError("Jacobi diagonal must be a tensor")
        _metadata((diagonal.numel(), diagonal.numel()), diagonal.device, diagonal.dtype)
        _vector(diagonal, diagonal.numel(), diagonal.device, diagonal.dtype, "Jacobi diagonal")
        if not diagonal.numel() or bool((diagonal <= 0).any()):
            raise ValueError("Jacobi diagonal must be nonempty and strictly positive")
        self._diagonal = diagonal.detach().clone()
        self.shape = (diagonal.numel(), diagonal.numel())
        self.device, self.dtype = diagonal.device, diagonal.dtype

    @property
    def diagonal(self) -> torch.Tensor:
        return self._diagonal.clone()

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        _vector(residual, self.shape[1], self.device, self.dtype, "Jacobi input")
        result = residual / self._diagonal
        _vector(result, self.shape[0], self.device, self.dtype, "Jacobi output")
        return result


class _CheckedPreconditioner:
    def __init__(self, preconditioner: Preconditioner, system: LinearOperator):
        if not isinstance(preconditioner, Preconditioner):
            raise TypeError("preconditioner must implement the Preconditioner contract")
        _metadata(preconditioner.shape, preconditioner.device, preconditioner.dtype)
        if (
            preconditioner.shape != system.shape or preconditioner.device != system.device
            or preconditioner.dtype != system.dtype
        ):
            raise ValueError("preconditioner shape/device/dtype must match system; no transfer")
        self.preconditioner, self.system = preconditioner, system

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        try:
            result = self.preconditioner.apply(residual)
        except Exception as error:
            raise RuntimeError(f"preconditioner.apply callback failed: {error}") from error
        _vector(
            result, self.system.shape[0], self.system.device, self.system.dtype,
            "preconditioner.apply output",
        )
        return result


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


def _preconditioned_residual(
    preconditioner: _CheckedPreconditioner, residual: torch.Tensor,
    system: LinearOperator, rhs: torch.Tensor, x: torch.Tensor,
    iteration: int, tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    try:
        transformed = preconditioner.apply(residual)
        squared = residual @ transformed
        if not bool(torch.isfinite(squared)) or float(squared) <= 0:
            raise ValueError("nonpositive/nonfinite preconditioned residual product r'z")
    except (TypeError, ValueError, RuntimeError) as error:
        raise CGConvergenceError(
            f"invalid preconditioner: {error}", iteration,
            _l2(rhs - system.matvec(x)), tolerance,
        ) from error
    return transformed, squared


@torch.no_grad()
def conjugate_gradient(
    operator: LinearOperator, rhs: torch.Tensor, *, initial: torch.Tensor | None = None,
    max_iterations: int = 1000, eps_abs: float = 1e-10, eps_rel: float = 1e-10,
    preconditioner: Preconditioner | None = None,
) -> CGResult:
    """Solve an assumed SPD system; every accepted residual is recomputed.

    Recurrence residuals only nominate convergence. If a nominated iterate
    fails the true b-Kx test, restart the direction with the true residual,
    within the original iteration budget, reapplying M^-1 on restart. Acceptance
    uses the original unpreconditioned L2 residual/tolerance, never r'M^-1 r.
    With preconditioner=None the original arithmetic path is unchanged.
    """
    _integer("CG max_iterations", max_iterations)
    _positive("CG eps_abs", eps_abs)
    _positive("CG eps_rel", eps_rel)
    K = _CheckedOperator(operator, "CG system")
    if K.shape[0] != K.shape[1] or not K.shape[0]:
        raise ValueError("CG system must be nonempty and square")
    M = None if preconditioner is None else _CheckedPreconditioner(preconditioner, K)
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
    if M is None:
        direction = residual.clone()
        squared = residual @ residual
    else:
        transformed, squared = _preconditioned_residual(M, residual, K, b, x, 0, tolerance)
        direction = transformed.clone()
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
            if M is None:
                squared = residual @ residual
                direction = residual.clone()
            else:
                transformed, squared = _preconditioned_residual(
                    M, residual, K, b, x, iteration, tolerance,
                )
                direction = transformed.clone()
            continue
        if M is None:
            transformed = residual
            updated = residual @ residual
        else:
            transformed, updated = _preconditioned_residual(
                M, residual, K, b, x, iteration, tolerance,
            )
        if not bool(torch.isfinite(updated)) or float(squared) <= 0:
            true_norm = _l2(b - K.matvec(x))
            raise CGConvergenceError("invalid residual recurrence", iteration, true_norm, tolerance)
        direction = transformed + (updated / squared) * direction
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
    cg_preconditioner: str = "none"

    def validate(self) -> torch.device:
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("operator QP supports float32/float64 only")
        if self.cg_preconditioner not in ("none", "jacobi"):
            raise ValueError("cg_preconditioner must be 'none' or 'jacobi'")
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
        converted = value.matrix.to(device=device, dtype=dtype)
        original_values = (
            value.matrix if value.matrix.layout == torch.strided else value.matrix.values()
        )
        converted_values = converted if converted.layout == torch.strided else converted.values()
        _check_conversion(original_values, converted_values, label)
        value = TensorOperator(converted)
    checked = _CheckedOperator(value, label)
    if checked.device != device or checked.dtype != dtype:
        raise ValueError(f"{label} matrix-free device/dtype must match config; no fallback/transfer")
    return checked


def _check_conversion(original: torch.Tensor, converted: torch.Tensor, label: str) -> None:
    source = original.detach().to(device=converted.device)
    if bool((torch.isfinite(source) & ~torch.isfinite(converted)).any()):
        raise ValueError(f"{label} finite input became nonfinite during conversion")
    if bool(((source != 0) & (converted == 0)).any()):
        raise ValueError(f"{label} nonzero input became zero during conversion")


def _original_operator(
    value: torch.Tensor | LinearOperator, device: torch.device,
    working: _CheckedOperator, label: str,
) -> _CheckedOperator:
    if isinstance(value, (torch.Tensor, TensorOperator)):
        return _coerce(value, device, torch.float64, label)
    # Callback contracts already require the working dtype/device. Their
    # mathematical action is borrowed, not copied or certified by these checks.
    return working


def _original_candidate(
    P: _CheckedOperator, A: _CheckedOperator, q: torch.Tensor,
    l: torch.Tensor, u: torch.Tensor, x: torch.Tensor, y: torch.Tensor,
    config: OperatorQPConfig,
) -> bool:
    Px = P.matvec(x.to(dtype=P.dtype)).to(dtype=torch.float64)
    Ax = A.matvec(x.to(dtype=A.dtype)).to(dtype=torch.float64)
    ATy = A.rmatvec(y.to(dtype=A.dtype)).to(dtype=torch.float64)
    y = y.to(dtype=torch.float64)
    if not all(bool(torch.isfinite(v).all()) for v in (Px, Ax, ATy)):
        raise RuntimeError("operator original-model products are nonfinite")
    lower, upper = torch.isfinite(l), torch.isfinite(u)
    lower_limit = config.eps_abs + config.eps_rel * torch.maximum(l[lower].abs(), Ax[lower].abs())
    upper_limit = config.eps_abs + config.eps_rel * torch.maximum(u[upper].abs(), Ax[upper].abs())
    stationary = Px + q + ATy
    stationary_limit = config.eps_abs + config.eps_rel * torch.maximum(
        torch.maximum(Px.abs(), q.abs()), ATy.abs(),
    )
    projected = torch.minimum(torch.maximum(Ax + y, l), u)
    normal_limit = config.eps_abs + config.eps_rel * torch.maximum(Ax.abs(), projected.abs())
    positive, negative = y > 0, y < 0
    if bool((positive & ~upper).any() | (negative & ~lower).any()):
        return False
    # Mask before multiplication: zero multipliers on infinite bounds are legal.
    support = torch.cat((y[positive] * u[positive], y[negative] * l[negative]))
    products = torch.cat((y[positive] * Ax[positive], y[negative] * Ax[negative]))
    complementarity_limit = config.eps_abs + config.eps_rel * torch.maximum(
        support.abs(), products.abs(),
    )
    quantities = (
        stationary, stationary_limit, projected, normal_limit,
        lower_limit, upper_limit, support, products, complementarity_limit,
    )
    if not all(bool(torch.isfinite(v).all()) for v in quantities):
        raise RuntimeError("operator original-model acceptance arithmetic is nonfinite")
    return (
        bool((l[lower] - Ax[lower] <= lower_limit).all())
        and bool((Ax[upper] - u[upper] <= upper_limit).all())
        and bool((stationary.abs() <= stationary_limit).all())
        and bool(((Ax - projected).abs() <= normal_limit).all())
        and bool(((support - products).abs() <= complementarity_limit).all())
    )


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
        *, preconditioner: Preconditioner | None = None,
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
        original_q, original_l, original_u = [
            t.detach().to(device=device, dtype=torch.float64).clone() for t in vectors
        ]
        if original_q.ndim != 1 or not original_q.numel():
            raise ValueError("q must be a nonempty vector")
        if not bool(torch.isfinite(original_q).all()):
            raise ValueError("q must be finite")
        if original_l.shape != original_u.shape:
            raise ValueError("bounds must have matching shapes")
        if bool(torch.isnan(original_l).any() | torch.isnan(original_u).any()):
            raise ValueError("bounds cannot contain NaN")
        if bool(
            torch.isposinf(original_l).any() | torch.isneginf(original_u).any()
            | (original_l > original_u).any()
        ):
            raise ValueError("invalid/reversed original bounds")
        q, l, u = [t.detach().to(device=device, dtype=config.dtype).clone() for t in vectors]
        for label, original, converted in zip(("q", "l", "u"), vectors, (q, l, u)):
            _check_conversion(original, converted, label)
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
        if config.cg_preconditioner == "jacobi":
            if preconditioner is not None:
                raise ValueError("choose automatic Jacobi OR an explicit preconditioner, not both")
            if not isinstance(P.operator, TensorOperator) or not isinstance(A.operator, TensorOperator):
                raise ValueError(
                    "automatic Jacobi requires tensor P/A; matrix-free callers must explicitly "
                    "supply preconditioner=JacobiPreconditioner(diag_K) or an SPD inverse action"
                )
            diagonal = P.operator.diagonal()
            _vector(diagonal, q.numel(), device, config.dtype, "P Jacobi diagonal")
            if bool((diagonal < 0).any()):
                raise ValueError("P Jacobi diagonal is negative; assume_psd contradicted")
            diagonal = diagonal + config.rho * A.operator.column_squared_norms() + config.sigma
            preconditioner = JacobiPreconditioner(diagonal)
        _probe(P, quadratic=True)
        _probe(A)
        self.config, self.P, self.A = config, P, A
        self.q, self.l, self.u = q, l, u
        self.original_P = P if P.dtype == torch.float64 else _original_operator(
            problem.P, device, P, "original P",
        )
        self.original_A = A if A.dtype == torch.float64 else _original_operator(
            problem.A, device, A, "original A",
        )
        self.original_q, self.original_l, self.original_u = original_q, original_l, original_u
        self.system = MatrixFreeOperator(
            P.shape, device, config.dtype,
            self._system_matvec, self._system_matvec,
        )
        if preconditioner is not None:
            _CheckedPreconditioner(preconditioner, self.system)
        self.preconditioner = preconditioner

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
                    preconditioner=self.preconditioner,
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
                if (
                    primal <= primal_tol and dual <= dual_tol and normal <= primal_tol
                    and _original_candidate(
                        self.original_P, self.original_A,
                        self.original_q, self.original_l, self.original_u, x, y, c,
                    )
                ):
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
    *, preconditioner: Preconditioner | None = None,
) -> PreparedOperatorQP:
    return PreparedOperatorQP(problem, config, preconditioner=preconditioner)


def solve_operator_qp(
    problem: OperatorQPProblem, config: OperatorQPConfig = OperatorQPConfig(),
    *, preconditioner: Preconditioner | None = None,
) -> OperatorQPResult:
    return prepare_operator_qp(problem, config, preconditioner=preconditioner).solve()
