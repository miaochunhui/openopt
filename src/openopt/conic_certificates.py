"""Opt-in, bounded host checks of original conic directions, not a solver.

The represented binary64 model is checked without scaling, regularization,
eigensolver tolerances, solver statuses or producer metadata. Bounded Fraction
arithmetic protects necessary identities and cone boundaries. Passing is only
numerical evidence for the supplied data, never an exact certificate for an
underlying real model, GPU execution, HSD implementation or commercial scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import sys

import numpy as np
import torch

from .certificates import NumericalEvidence
from .conic import (
    ConicProblem, NonnegativeCone, RotatedSecondOrderCone, SecondOrderCone, ZeroCone,
)

__all__ = [
    "check_conic_infeasibility_direction", "check_conic_unbounded_direction",
    "conic_direction_contract",
]

MAX_N = 16
MAX_M = 128
MAX_RATIONAL_BITS = 32768
ARITHMETIC = "bounded_host_binary64_fraction_v1"
QUALIFICATION = (
    "approximate original-coordinate numerical evidence only; host CPU verification; "
    "not an exact certificate, solver, HSD, GPU execution or commercial qualification"
)


def conic_direction_contract() -> dict[str, object]:
    """Return the independently selected contract; existing APIs stay unchanged."""
    return {
        "algorithm": ARITHMETIC, "device": "cpu", "max_n": MAX_N, "max_m": MAX_M,
        "max_rational_bits": MAX_RATIONAL_BITS, "input": "finite dense binary64",
        "psd": "full original symmetry and bounded rational Schur complements",
        "identities": "exact on supplied binary64 values; no residual-only acceptance",
        "cones": ["zero", "nonnegative", "SOC", "RSOC"],
        "base": "caller-supplied x0 AND s0 required for primal unbounded evidence",
        "negative_margin": "upper(dot) < -(upper(atol + rtol * sum(abs(products))))",
        "certified": False, "qualification": QUALIFICATION,
    }


def _device(device: str) -> None:
    requested = torch.device(device)
    if requested.type == "cuda":
        raise RuntimeError("explicit CUDA requested: direction checker is host-only; no fallback")
    if requested != torch.device("cpu"):
        raise ValueError("direction checker requires device=cpu")
    if (sys.float_info.radix, sys.float_info.mant_dig, sys.float_info.max_exp) != (2, 53, 1024):
        raise ValueError("unsupported host binary64 runtime")


def _bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_RATIONAL_BITS:
        raise ValueError("unavailable arithmetic bound: rational bit budget exceeded")
    return value


def _f(value: float) -> Fraction:
    return _bounded(Fraction.from_float(float(value)))


def _sum(values) -> Fraction:
    result = Fraction(0)
    for value in values:
        result = _bounded(result + value)
    return result


def _bounds(value: Fraction, name: str) -> tuple[float, float]:
    """One outward binary64 enclosure, never use the nearest float as exact."""
    try:
        nearest = float(value)
    except OverflowError as error:
        raise ValueError(f"unavailable {name} bound: binary64 overflow") from error
    if not math.isfinite(nearest):
        raise ValueError(f"unavailable {name} bound: binary64 overflow")
    if value and nearest == 0:
        raise ValueError(f"unavailable {name} bound: binary64 underflow")
    represented = _f(nearest)
    lower = math.nextafter(nearest, -math.inf) if represented > value else nearest
    upper = math.nextafter(nearest, math.inf) if represented < value else nearest
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError(f"unavailable {name} outward bound: binary64 overflow")
    return lower, upper


def _array(value: object, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device != torch.device("cpu"):
            raise ValueError(f"{name}: implicit device transfer is forbidden")
        if value.layout != torch.strided or value.dtype != torch.float64:
            raise ValueError(f"{name} must be real dense binary64")
        value = value.detach().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or (array.dtype.kind == "f" and array.dtype != np.float64):
        raise ValueError(f"{name} must be real dense binary64 or exactly representable integers")
    if array.size > max(MAX_N * MAX_N, MAX_N * MAX_M):
        raise ValueError(f"{name}: unsupported bounded model size")
    try:
        result = np.array(array, dtype=np.float64, copy=True)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name}: unavailable binary64 conversion") from error
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    if array.dtype.kind in "iu" and any(int(a) != int(b) for a, b in zip(array.flat, result.flat)):
        raise ValueError(f"{name}: integer conversion is not exact binary64")
    return result


def _vector(value: object, name: str, size: int) -> tuple[Fraction, ...]:
    array = _array(value, name)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return tuple(_f(v) for v in array)


def _dot(left, right, name: str) -> tuple[Fraction, Fraction]:
    if len(left) != len(right):
        raise ValueError(f"{name}: dot dimensions mismatch")
    products = [_bounded(a * b) for a, b in zip(left, right)]
    # Guard actual product and signed/absolute sum bounds, not just cancellation.
    for product in products:
        _bounds(product, f"{name} product")
    value, scale = _sum(products), _sum(abs(v) for v in products)
    _bounds(value, name)
    _bounds(scale, f"{name} absolute product sum")
    return value, scale


@dataclass(frozen=True)
class _Model:
    P: tuple[tuple[Fraction, ...], ...]
    q: tuple[Fraction, ...]
    A: tuple[tuple[Fraction, ...], ...]
    b: tuple[Fraction, ...]
    cones: tuple


def _psd(matrix: tuple[tuple[Fraction, ...], ...]) -> None:
    """PSD iff successive positive pivots have PSD Schur complements.

    A zero diagonal in a PSD matrix must have an entirely zero row. Thus a
    nonzero off-diagonal at a zero pivot is indefiniteness, not a nullspace
    tolerance. This also handles LPs and singular, non-DD dense PSD matrices.
    """
    work = [list(row) for row in matrix]
    for k in range(len(work)):
        pivot = work[k][k]
        if pivot < 0:
            raise ValueError("original P must be positive semidefinite")
        if pivot == 0:
            if any(work[k][j] != 0 for j in range(k + 1, len(work))):
                raise ValueError("original P must be positive semidefinite: nonzero zero-pivot row")
            continue
        for i in range(k + 1, len(work)):
            for j in range(i, len(work)):
                product = _bounded(work[i][k] * work[k][j])
                correction = _bounded(product / pivot)
                work[i][j] = work[j][i] = _bounded(work[i][j] - correction)


def _model(problem: ConicProblem) -> _Model:
    if not isinstance(problem, ConicProblem):
        raise TypeError("problem must be a ConicProblem")
    P, q, A, b = [_array(getattr(problem, name), name) for name in ("P", "q", "A", "b")]
    if q.ndim != 1 or not 1 <= q.size <= MAX_N:
        raise ValueError(f"q: supported n is 1..{MAX_N}")
    n = q.size
    if P.shape != (n, n) or A.ndim != 2 or A.shape[1] != n:
        raise ValueError("P/A dimensions must match q")
    if b.ndim != 1 or b.shape != (A.shape[0],) or b.size > MAX_M:
        raise ValueError(f"b: supported m is 0..{MAX_M} and must match A")
    if not isinstance(problem.cones, (tuple, list)):
        raise TypeError("cones must be an ordered tuple or list")
    cones = tuple(problem.cones)
    if any(type(c) not in (ZeroCone, NonnegativeCone, SecondOrderCone, RotatedSecondOrderCone) for c in cones):
        raise TypeError("only original zero/nonnegative/SOC/RSOC cones are supported")
    if sum(c.dimension for c in cones) != b.size:
        raise ValueError("cone dimensions must equal original rows")
    if not bool(np.array_equal(P, P.T)):
        raise ValueError("original P must be full symmetric; no averaging or triangular storage")
    matrix = tuple(tuple(_f(v) for v in row) for row in P)
    _psd(matrix)
    return _Model(matrix, tuple(_f(v) for v in q),
                  tuple(tuple(_f(v) for v in row) for row in A), tuple(_f(v) for v in b), cones)


class _Audit:
    def __init__(self, atol: float, rtol: float):
        if isinstance(atol, bool) or isinstance(rtol, bool):
            raise ValueError("tolerances cannot be booleans")
        atol, rtol = float(atol), float(rtol)
        if not math.isfinite(atol) or atol <= 0 or not math.isfinite(rtol) or not 0 <= rtol < 1:
            raise ValueError("atol must be finite positive; rtol must be finite in [0,1)")
        self.atol, self.rtol = _f(atol), _f(rtol)
        self.checks = {"original_full_symmetry": True, "original_psd": True}
        self.metrics: dict[str, float | None] = {}
        self.thresholds: dict[str, float] = {"atol": atol, "rtol": rtol}
        self.unavailable: dict[str, str] = {}

    def record(self, name: str, value: Fraction) -> tuple[float, float]:
        lower, upper = _bounds(value, name)
        self.metrics[f"{name}_lower"], self.metrics[f"{name}_upper"] = lower, upper
        return lower, upper

    def threshold(self, name: str, scale: Fraction) -> float:
        threshold = _bounded(self.atol + _bounded(self.rtol * scale))
        upper = _bounds(threshold, f"{name} threshold")[1]
        self.thresholds[name] = upper
        return upper

    def identity(self, name: str, value: Fraction, scale: Fraction) -> None:
        self.record(name, value)
        limit = self.threshold(name, scale)
        self.checks[f"{name}_residual"] = abs(value) <= _f(limit)
        self.checks[f"{name}_identity"] = value == 0

    def negative(self, name: str, value: Fraction, scale: Fraction) -> None:
        _, upper = self.record(name, value)
        margin = self.threshold(name, scale)
        self.checks[f"{name}_strict_sign"] = upper < 0
        self.checks[f"{name}_strict_margin"] = upper < -margin

    def cone(self, prefix: str, values: tuple[Fraction, ...], cones: tuple, dual: bool = False) -> None:
        offset = 0
        for block, cone in enumerate(cones):
            part = values[offset:offset + cone.dimension]
            offset += cone.dimension
            name = f"{prefix}_block_{block}"
            for j, value in enumerate(part):
                self.record(f"{name}_original_component_{j}", value)
            if type(cone) is ZeroCone:
                for j, value in enumerate(part):
                    if dual:
                        self.checks[f"{name}_component_{j}_unrestricted"] = True
                    else:
                        self.identity(f"{name}_component_{j}", value, abs(value))
            elif type(cone) is NonnegativeCone:
                for j, value in enumerate(part):
                    self.record(f"{name}_component_{j}", value)
                    self.checks[f"{name}_component_{j}_nonnegative"] = value >= 0
            elif type(cone) is SecondOrderCone:
                self.checks[f"{name}_head_nonnegative"] = part[0] >= 0
                squared = _sum(_bounded(v * v) for v in part[1:])
                _bounds(squared, f"{name} tail square sum")
                _bounds(_bounded(part[0] * part[0]), f"{name} head square")
                slack = _bounded(_bounded(part[0] * part[0]) - squared)
                self.record(f"{name}_squared_slack", slack)
                self.checks[f"{name}_original_soc"] = slack >= 0
            else:
                for j in (0, 1):
                    self.record(f"{name}_head_{j}", part[j])
                    self.checks[f"{name}_head_{j}_nonnegative"] = part[j] >= 0
                squared = _sum(_bounded(v * v) for v in part[2:])
                _bounds(squared, f"{name} tail square sum")
                _bounds(_bounded(2 * _bounded(part[0] * part[1])), f"{name} head product")
                slack = _bounded(_bounded(2 * _bounded(part[0] * part[1])) - squared)
                self.record(f"{name}_squared_slack", slack)
                self.checks[f"{name}_original_rsoc"] = slack >= 0

    def finish(self, kind: str) -> NumericalEvidence:
        return NumericalEvidence(
            kind, all(self.checks.values()) and not self.unavailable, self.checks,
            self.metrics, self.thresholds, self.unavailable, QUALIFICATION,
        )


def check_conic_infeasibility_direction(
    problem: ConicProblem, y: object, *, device: str = "cpu",
    atol: float = 1e-8, rtol: float = 1e-8,
) -> NumericalEvidence:
    """Check original Farkas y: y in K*, A.T y=0, b.T y strictly negative.

    The dual of zero is unrestricted; all other supported cones are self-dual.
    No ray generation, status promotion or solver/source trust is involved.
    """
    _device(device)
    audit = _Audit(atol, rtol)
    model = _model(problem)
    vector = _vector(y, "y", len(model.b))
    audit.cone("dual_cone", vector, model.cones, dual=True)
    for j in range(len(model.q)):
        value, scale = _dot(tuple(row[j] for row in model.A), vector, f"Aty_{j}")
        audit.identity(f"Aty_{j}", value, scale)
    value, scale = _dot(model.b, vector, "bty")
    audit.negative("bty", value, scale)
    return audit.finish("conic_primal_infeasibility_direction")


def check_conic_unbounded_direction(
    problem: ConicProblem, d: object, *, x0: object | None = None, s0: object | None = None,
    device: str = "cpu", atol: float = 1e-8, rtol: float = 1e-8,
) -> NumericalEvidence:
    """Check a base AND recession direction; dual infeasibility is insufficient.

    A*x0+s0=b, s0 in K, P*d=0, -A*d in K, q.T*d<0 are independently
    checked on original coordinates. Both missing base and any failed check
    mean passed=False, never an unbounded status or exact certificate.
    """
    _device(device)
    audit = _Audit(atol, rtol)
    model = _model(problem)
    ray = _vector(d, "d", len(model.q))
    point = None if x0 is None else _vector(x0, "x0", len(model.q))
    slack = None if s0 is None else _vector(s0, "s0", len(model.b))
    for i, row in enumerate(model.P):
        value, scale = _dot(row, ray, f"Pd_{i}")
        audit.identity(f"Pd_{i}", value, scale)
    recession = tuple(-_dot(row, ray, f"Ad_{i}")[0] for i, row in enumerate(model.A))
    audit.cone("recession_cone", recession, model.cones)
    value, scale = _dot(model.q, ray, "qtd")
    audit.negative("qtd", value, scale)
    audit.checks["base_point_present"] = point is not None and slack is not None
    if point is None or slack is None:
        audit.unavailable["base_point"] = "caller must supply BOTH original feasible x0 and s0; dual infeasibility is not primal unboundedness"
    else:
        audit.cone("base_slack_cone", slack, model.cones)
        quadratic = tuple(_dot(row, point, f"Px0_{i}")[0] for i, row in enumerate(model.P))
        base_quadratic = _dot(point, quadratic, "x0Px0")[0]
        base_linear = _dot(model.q, point, "qx0")[0]
        audit.record("base_objective", _bounded(base_quadratic / 2 + base_linear))
        for i, row in enumerate(model.A):
            value, scale = _dot(row, point, f"Ax0_{i}")
            residual = _bounded(_bounded(value + slack[i]) - model.b[i])
            scale = _sum((scale, abs(slack[i]), abs(model.b[i])))
            _bounds(scale, f"base_equation_{i} absolute sum")
            audit.identity(f"base_equation_{i}", residual, scale)
    return audit.finish("conic_primal_unbounded_direction")
