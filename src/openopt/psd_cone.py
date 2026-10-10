"""Bounded native PSD projection and independent represented-coordinate checks.

This module is opt-in and is not wired into any conic or SDP solver. Projection
uses torch on the requested device. Verification explicitly copies at most a
6-by-6 matrix to the host for direct rational principal-minor arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
import math
import re

import torch

MAX_ORDER = 6
CHECKER_VERSION = "psd-principal-minors-moreau-v1"


def _placement(device: str, dtype: torch.dtype) -> torch.device:
    if type(device) is not str or not re.fullmatch(r"cpu|cuda:(0|[1-9][0-9]*)", device):
        raise ValueError("device must be explicit cpu or cuda:<index>")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("PSD primitive requires float32 or float64")
    target = torch.device(device)
    if target.type == "cuda":
        if not torch.cuda.is_available() or target.index >= torch.cuda.device_count():
            raise RuntimeError("requested CUDA device unavailable; no CPU fallback")
    return target


def _tensor(value: torch.Tensor, target: torch.device, dtype: torch.dtype) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError("PSD coordinates must be a torch tensor")
    if value.layout != torch.strided:
        raise ValueError("PSD primitive requires dense strided coordinates")
    if value.device != target or value.dtype != dtype:
        raise ValueError("actual tensor device/dtype differs from explicit request")
    if value.requires_grad:
        raise ValueError("PSD primitive does not provide an autograd contract")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("PSD coordinates must be finite")


def _matrix(value: torch.Tensor, target: torch.device, dtype: torch.dtype) -> None:
    _tensor(value, target, dtype)
    if value.ndim != 2 or value.shape[0] != value.shape[1]:
        raise ValueError("PSD coordinates must be one square matrix, not a batch")
    if not 1 <= value.shape[0] <= MAX_ORDER:
        raise NotImplementedError(f"PSD primitive supports matrix order 1..{MAX_ORDER}")
    if not torch.equal(value, value.T):
        raise ValueError("original PSD matrix must be exactly symmetric")


def _tolerances(atol: float, rtol: float | None, dtype: torch.dtype) -> tuple[Fraction, Fraction]:
    if rtol is None:
        rtol = 128 * torch.finfo(dtype).eps
    for name, value in (("atol", atol), ("rtol", rtol)):
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if rtol >= 1:
        raise ValueError("rtol must be less than one")
    return Fraction(float(atol)), Fraction(float(rtol))


def svec(matrix: torch.Tensor, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Lower triangle in column order, off-diagonals multiplied by sqrt(2).

    In real arithmetic this is an isometry from symmetric matrices with the
    Frobenius inner product to Euclidean vectors. Binary sqrt(2) is rounded.
    """
    target = _placement(device, dtype)
    _matrix(matrix, target, dtype)
    n = matrix.shape[0]
    result = torch.stack([
        matrix[i, j] if i == j else matrix[i, j] * math.sqrt(2)
        for j in range(n) for i in range(j, n)
    ])
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("svec overflow; original matrix is not replaced or clipped")
    return result


def smat(vector: torch.Tensor, order: int, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    """Inverse mathematical svec convention; no solver or implicit conversion."""
    target = _placement(device, dtype)
    _tensor(vector, target, dtype)
    if type(order) is not int or not 1 <= order <= MAX_ORDER:
        raise ValueError(f"order must be an integer in 1..{MAX_ORDER}")
    if vector.ndim != 1 or vector.numel() != order * (order + 1) // 2:
        raise ValueError("svec length must exactly match the triangular order")
    result = torch.zeros((order, order), device=target, dtype=dtype)
    k = 0
    for j in range(order):
        for i in range(j, order):
            coordinate = vector[k] if i == j else vector[k] / math.sqrt(2)
            if bool((vector[k] != 0) & (coordinate == 0)):
                raise RuntimeError("smat nonzero coordinate underflow")
            result[i, j] = result[j, i] = coordinate
            k += 1
    return result


def _host(matrix: torch.Tensor) -> list[list[Fraction]]:
    return [[Fraction(float(value)) for value in row]
            for row in matrix.detach().to(device="cpu", dtype=torch.float64).tolist()]


def _determinant(matrix: list[list[Fraction]]) -> Fraction:
    """Direct exact elimination, not eigen iteration or a linear-system solve."""
    a = [row[:] for row in matrix]
    result = Fraction(1)
    for k in range(len(a)):
        pivot = next((i for i in range(k, len(a)) if a[i][k]), None)
        if pivot is None:
            return Fraction(0)
        if pivot != k:
            a[k], a[pivot] = a[pivot], a[k]
            result = -result
        diagonal = a[k][k]
        result *= diagonal
        for i in range(k + 1, len(a)):
            multiplier = a[i][k] / diagonal
            for j in range(k + 1, len(a)):
                a[i][j] -= multiplier * a[k][j]
    return result


def _minor_signs(matrix: list[list[Fraction]], shift: list[Fraction]) -> tuple[int, ...]:
    n = len(matrix)
    signs = []
    for size in range(1, n + 1):
        for selected in combinations(range(n), size):
            minor = [[matrix[i][j] + (shift[i] if i == j else 0)
                      for j in selected] for i in selected]
            value = _determinant(minor)
            signs.append(int(value > 0) - int(value < 0))
    return tuple(signs)


def _psd(matrix: list[list[Fraction]], atol: Fraction, rtol: Fraction) -> dict:
    n = len(matrix)
    exact = _minor_signs(matrix, [Fraction(0)] * n)
    shift = [atol + rtol * abs(matrix[i][i]) for i in range(n)]
    shifted = _minor_signs(matrix, shift) if any(shift) else exact
    return {
        "exact_psd": all(sign >= 0 for sign in exact),
        "numerical_psd": all(sign >= 0 for sign in shifted),
        "original_minor_signs": exact,
        "shifted_minor_signs": shifted,
        "principal_minor_count": (1 << n) - 1,
    }


def _ratio(value: Fraction, limit: Fraction) -> str:
    if not value:
        return "0x0.0p+0"
    if not limit:
        return "unbounded: positive residual with zero tolerance"
    try:
        return float(value / limit).hex()
    except OverflowError:
        return "above binary64 display range"


@dataclass(frozen=True)
class PSDProjectionEvidence:
    passed: bool
    checks: dict[str, bool]
    projected_psd: dict
    residual_psd: dict
    component_ratios: tuple[tuple[str, ...], ...]
    trace_ratio: str
    order: int
    device: str
    dtype: str
    atol: float
    rtol: float
    checker: str = CHECKER_VERSION
    verification_device: str = "cpu: bounded represented-coordinate Fraction"
    qualification: str = "numerical projection only; not SDP or exact certification"


@dataclass(frozen=True)
class PSDProjection:
    matrix: torch.Tensor
    evidence: PSDProjectionEvidence
    algorithm: str


class PSDProjectionRejected(RuntimeError):
    def __init__(self, evidence: PSDProjectionEvidence):
        self.evidence = evidence
        failed = [key for key, passed in evidence.checks.items() if not passed]
        super().__init__(f"PSD candidate rejected by original-matrix checks: {failed}")


def check_psd_projection(
    original: torch.Tensor, candidate: torch.Tensor, *,
    device: str, dtype: torch.dtype, atol: float = 0.0, rtol: float | None = None,
) -> PSDProjectionEvidence:
    """Check Y and G=Y-X are PSD to local tolerance, and YG/trace(YG) vanish.

    Independent of the producer: no eigendecomposition, projection, iterative
    refinement or solver call. Original represented coordinates are not rounded
    again during subtraction, products or principal-minor sign evaluation.
    A numerical pass with negative unshifted minors is explicitly NOT exact PSD.
    """
    target = _placement(device, dtype)
    _matrix(original, target, dtype)
    _matrix(candidate, target, dtype)
    if original.shape != candidate.shape:
        raise ValueError("candidate must have the original matrix shape")
    absolute, relative = _tolerances(atol, rtol, dtype)
    x, y = _host(original), _host(candidate)
    n = len(x)
    g = [[y[i][j] - x[i][j] for j in range(n)] for i in range(n)]
    primal, residual = _psd(y, absolute, relative), _psd(g, absolute, relative)
    products, limits, ratios = [], [], []
    for i in range(n):
        product_row, limit_row, ratio_row = [], [], []
        for j in range(n):
            product = sum((y[i][k] * g[k][j] for k in range(n)), Fraction(0))
            scale = sum((abs(y[i][k] * g[k][j])
                         + abs(y[i][k] * x[k][j])
                         + abs(x[i][k] * g[k][j]) for k in range(n)), Fraction(0))
            linear = sum((abs(y[i][k]) + abs(g[k][j]) for k in range(n)), Fraction(0))
            limit = relative * scale + absolute * linear + n * absolute * absolute
            product_row.append(product)
            limit_row.append(limit)
            ratio_row.append(_ratio(abs(product), limit))
        products.append(product_row)
        limits.append(limit_row)
        ratios.append(tuple(ratio_row))
    trace = abs(sum((products[i][i] for i in range(n)), Fraction(0)))
    trace_limit = sum((limits[i][i] for i in range(n)), Fraction(0))
    checks = {
        "projected_psd_with_local_tolerance": primal["numerical_psd"],
        "moreau_residual_psd_with_local_tolerance": residual["numerical_psd"],
        "moreau_component_orthogonality": all(
            abs(products[i][j]) <= limits[i][j] for i in range(n) for j in range(n)),
        "moreau_trace_orthogonality": trace <= trace_limit,
    }
    return PSDProjectionEvidence(
        all(checks.values()), checks, primal, residual, tuple(ratios),
        _ratio(trace, trace_limit), n, device, str(dtype), float(absolute), float(relative),
    )


def project_psd(
    matrix: torch.Tensor, *,
    device: str, dtype: torch.dtype, atol: float = 0.0, rtol: float | None = None,
) -> PSDProjection:
    """Native Frobenius projection candidate, accepted only by the original gate.

    Exact original PSD/negative-PSD matrices have an identity/zero fast path.
    Otherwise scale by a represented power of two and use torch.linalg.eigh on
    the requested device. Unsupported scaling or a rejected result fails loud;
    it does not use any CPU/third-party solver as a fallback.
    """
    target = _placement(device, dtype)
    _matrix(matrix, target, dtype)
    _tolerances(atol, rtol, dtype)
    original = matrix.detach().clone()
    host = _host(original)
    zero = [Fraction(0)] * len(host)
    if all(sign >= 0 for sign in _minor_signs(host, zero)):
        candidate = original.clone()
        algorithm = "exact-membership identity; tensor on requested device"
    elif all(sign >= 0 for sign in _minor_signs(
            [[-value for value in row] for row in host], zero)):
        candidate = torch.zeros_like(original)
        algorithm = "exact-negative-membership zero; tensor on requested device"
    else:
        maximum = float(original.abs().max())
        exponent = math.frexp(maximum)[1] - 1
        scale = math.ldexp(1.0, exponent)
        normalized = original / scale
        if not bool(torch.isfinite(normalized).all()):
            raise RuntimeError("PSD normalization nonfinite; no fallback")
        if bool(((original != 0) & (normalized == 0)).any()):
            raise NotImplementedError("PSD normalization loses a nonzero original coordinate")
        if not torch.equal(normalized * scale, original):
            raise NotImplementedError("PSD normalization changes a represented original coordinate")
        eigenvalues, vectors = torch.linalg.eigh(normalized)
        for value in (eigenvalues, vectors):
            _tensor(value, target, dtype)
        if eigenvalues.shape != (len(host),) or vectors.shape != original.shape:
            raise RuntimeError("PSD eigendecomposition returned wrong shape")
        # Clipping proposes the Euclidean projection. It does not certify PSD.
        positive = eigenvalues.clamp_min(0)
        proposal = (vectors * positive.unsqueeze(0)) @ vectors.T
        candidate = ((proposal + proposal.T) * 0.5) * scale
        if not bool(torch.isfinite(candidate).all()):
            raise RuntimeError("PSD projected coordinate overflow; no fallback")
        algorithm = "power-of-two scaled torch.linalg.eigh on requested device"
    evidence = check_psd_projection(
        original, candidate, device=device, dtype=dtype, atol=atol, rtol=rtol,
    )
    if not evidence.passed:
        raise PSDProjectionRejected(evidence)
    return PSDProjection(candidate, evidence, algorithm)
