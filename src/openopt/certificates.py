"""Independent CPU checks for dense QP numerical evidence, never exact proofs.

No solver residual, status, scaling, regularization or factorization is reused.
See docs/research/certificates.md for conventions and conservative limitations.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
from typing import Iterator

import numpy as np
import torch

__all__ = [
    "NumericalEvidence", "check_qp_candidate",
    "check_primal_infeasibility", "check_unbounded_direction",
]


@dataclass(frozen=True)
class NumericalEvidence:
    kind: str
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float | None]
    thresholds: dict[str, float]
    unavailable: dict[str, str]
    qualification: str = "approximate numerical evidence; not an exact proof"

    def to_dict(self) -> dict[str, object]:
        """JSON-compatible report: unavailable values are null, never NaN/Inf."""
        return {"schema_version": 1, **asdict(self)}


@dataclass(frozen=True)
class _Problem:
    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    l: np.ndarray
    u: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_threshold: float

    @property
    def spectrum_resolved(self) -> bool:
        # Do not silently truncate a tiny positive eigenvalue into a nullspace.
        return bool(np.all(
            (self.eigenvalues == 0) | (self.eigenvalues > self.spectral_threshold)
        ))


@contextmanager
def _arithmetic() -> Iterator[None]:
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            yield
    except FloatingPointError as error:
        raise ValueError(f"certificate arithmetic failed: {error}") from error


def _array(value: object, name: str, *, bounds: bool = False) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU; implicit device transfer is forbidden")
        if value.layout != torch.strided or value.is_complex():
            raise ValueError(f"{name} must be real dense data")
        value = value.detach().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric dense data")
    result = np.array(array, dtype=np.float64, copy=True)
    if bool(np.isnan(result).any()) or (not bounds and not bool(np.isfinite(result).all())):
        raise ValueError(f"{name} must be finite" if not bounds else f"{name} cannot contain NaN")
    return result


def _norm(value: np.ndarray) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def _scalar(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite computed {name}")
    return result


def _tolerances(atol: float, rtol: float) -> tuple[float, float]:
    if isinstance(atol, bool) or isinstance(rtol, bool):
        raise ValueError("tolerances must be real numbers, not booleans")
    atol, rtol = float(atol), float(rtol)
    if not math.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if not math.isfinite(rtol) or not 0 <= rtol < 1:
        raise ValueError("rtol must be finite and in [0, 1)")
    return atol, rtol


def _problem(problem: object) -> _Problem:
    P, q, A, l, u = [
        _array(getattr(problem, name), name, bounds=name in ("l", "u"))
        for name in ("P", "q", "A", "l", "u")
    ]
    if q.ndim != 1 or not q.size:
        raise ValueError("q must be a nonempty vector")
    n = q.size
    if P.shape != (n, n) or A.ndim != 2 or A.shape[1] != n:
        raise ValueError("P/A dimensions must match q")
    if l.shape != (A.shape[0],) or u.shape != l.shape:
        raise ValueError("l/u dimensions must match A rows")
    if bool(np.isposinf(l).any() | np.isneginf(u).any()):
        raise ValueError("lower +inf and upper -inf are invalid")
    if bool(np.any(l > u)):
        raise ValueError("lower bound exceeds upper bound")
    symmetry_threshold = _scalar(
        64 * np.finfo(np.float64).eps * n * _norm(P), "symmetry threshold",
    )
    if _norm(P - P.T) > symmetry_threshold:
        raise ValueError("P must be symmetric")
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * P + 0.5 * P.T)
    spectral_threshold = _scalar(
        64 * np.finfo(np.float64).eps * n * _norm(eigenvalues), "spectral threshold",
    )
    if float(eigenvalues[0]) < -spectral_threshold:
        raise ValueError("P must be positive semidefinite")
    return _Problem(P, q, A, l, u, eigenvalues, eigenvectors, spectral_threshold)


def _vector(value: object, name: str, size: int) -> np.ndarray:
    result = _array(value, name)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return result


def _feasibility(
    p: _Problem, x: np.ndarray, atol: float, rtol: float,
) -> tuple[np.ndarray, float, float, bool]:
    Ax = p.A @ x
    lower, upper = np.isfinite(p.l), np.isfinite(p.u)
    violations = np.concatenate([
        np.maximum(p.l[lower] - Ax[lower], 0),
        np.maximum(Ax[upper] - p.u[upper], 0),
    ])
    scales = np.concatenate([
        np.maximum(np.abs(p.l[lower]), np.abs(Ax[lower])),
        np.maximum(np.abs(p.u[upper]), np.abs(Ax[upper])),
    ])
    limits = atol + rtol * scales
    return Ax, _norm(violations), _norm(limits), bool(np.all(violations <= limits))


def _support(
    p: _Problem, y: np.ndarray,
) -> tuple[float | None, float, np.ndarray]:
    positive, negative = y > 0, y < 0
    bad = (positive & ~np.isfinite(p.u)) | (negative & ~np.isfinite(p.l))
    sign_residual = _norm(y[bad])
    if bool(np.any(bad)):
        return None, sign_residual, np.empty(0)
    # Mask before multiplication: 0 * infinity must never occur.
    terms = np.concatenate([p.u[positive] * y[positive], p.l[negative] * y[negative]])
    return _scalar(np.sum(terms), "box support"), sign_residual, terms


def _report(
    kind: str, checks: dict[str, bool], metrics: dict[str, float | None],
    thresholds: dict[str, float], unavailable: dict[str, str],
) -> NumericalEvidence:
    checks = {name: bool(value) for name, value in checks.items()}
    metrics = {
        name: None if value is None else _scalar(value, name)
        for name, value in metrics.items()
    }
    thresholds = {name: _scalar(value, name) for name, value in thresholds.items()}
    return NumericalEvidence(kind, all(checks.values()), checks, metrics, thresholds, unavailable)


def check_qp_candidate(
    problem: object, x: object, y: object, *, atol: float = 1e-8, rtol: float = 1e-8,
) -> NumericalEvidence:
    """Recompute original-coordinate feasibility, KKT, dual objective and gap.

    Signed y is positive at an upper bound and negative at a lower bound.
    Wrong candidates return failed evidence; malformed/nonfinite inputs raise.
    """
    with _arithmetic():
        atol, rtol = _tolerances(atol, rtol)
        p = _problem(problem)
        x, y = _vector(x, "x", p.q.size), _vector(y, "y", p.l.size)
        Ax, feasibility, feasibility_limit, feasible = _feasibility(p, x, atol, rtol)
        Px, ATy = p.P @ x, p.A.T @ y
        stationarity = Px + p.q + ATy
        stationarity_limits = atol + rtol * np.maximum.reduce([
            np.abs(Px), np.abs(p.q), np.abs(ATy),
        ])
        support, sign_residual, _ = _support(p, y)
        positive, negative = y > 0, y < 0
        upper = positive & np.isfinite(p.u)
        lower = negative & np.isfinite(p.l)
        complementarity = np.concatenate([
            y[upper] * (p.u[upper] - Ax[upper]),
            y[lower] * (p.l[lower] - Ax[lower]),
        ])
        complementarity_scales = np.concatenate([
            np.maximum(np.abs(y[upper] * p.u[upper]), np.abs(y[upper] * Ax[upper])),
            np.maximum(np.abs(y[lower] * p.l[lower]), np.abs(y[lower] * Ax[lower])),
        ])
        complementarity_limits = atol + rtol * complementarity_scales
        primal = _scalar(0.5 * x @ Px + p.q @ x, "primal objective")
        b = p.q + ATy
        range_limit = _scalar(atol + rtol * _norm(b), "range threshold")
        range_residual = dual = gap = gap_limit = None
        unavailable: dict[str, str] = {}
        if support is None:
            unavailable["dual_objective"] = "multiplier has an infinite box support"
        elif not p.spectrum_resolved:
            unavailable["dual_objective"] = "unresolved PSD spectrum; no regularized inverse used"
        else:
            positive_eigenvalues = p.eigenvalues > 0
            coordinates = p.eigenvectors.T @ b
            inverse_coordinates = np.zeros_like(coordinates)
            inverse_coordinates[positive_eigenvalues] = (
                coordinates[positive_eigenvalues] / p.eigenvalues[positive_eigenvalues]
            )
            inverse_b = p.eigenvectors @ inverse_coordinates
            range_residual = _norm(b - p.P @ inverse_b)
            if range_residual > range_limit:
                unavailable["dual_objective"] = "q + A.T @ y is outside numerical range(P)"
            else:
                dual = _scalar(-0.5 * b @ inverse_b - support, "dual objective")
                gap = _scalar(np.subtract(primal, dual), "signed duality gap")
                gap_limit = _scalar(atol + rtol * max(abs(primal), abs(dual)), "gap threshold")
        checks = {
            "primal_feasibility": feasible,
            "stationarity": bool(np.all(np.abs(stationarity) <= stationarity_limits)),
            "bound_sign": support is not None,
            "complementarity": bool(np.all(
                np.abs(complementarity) <= complementarity_limits,
            )) and support is not None,
            "dual_objective_available": dual is not None,
            "duality_gap": gap is not None and abs(gap) <= gap_limit,
        }
        metrics = {
            "primal_feasibility": feasibility, "stationarity": _norm(stationarity),
            "bound_sign": sign_residual, "complementarity": _norm(complementarity),
            "primal_objective": primal, "dual_objective": dual,
            "duality_gap": gap, "range_residual": range_residual,
            "spectral_rank": float(np.count_nonzero(p.eigenvalues > p.spectral_threshold)),
        }
        thresholds = {
            "primal_feasibility_max": feasibility_limit,
            "stationarity_max": _norm(stationarity_limits),
            "complementarity_max": _norm(complementarity_limits),
            "bound_sign": 0.0, "range": range_limit, "spectral": p.spectral_threshold,
        }
        if gap_limit is not None:
            thresholds["duality_gap"] = gap_limit
        return _report("candidate_optimality", checks, metrics, thresholds, unavailable)


def check_primal_infeasibility(
    problem: object, direction: object, *, atol: float = 1e-8, rtol: float = 1e-8,
) -> NumericalEvidence:
    """Check a supplied Farkas direction, without generating one or solving."""
    with _arithmetic():
        atol, rtol = _tolerances(atol, rtol)
        p = _problem(problem)
        v = _vector(direction, "direction", p.l.size)
        norm = _norm(v)
        if norm:
            v = v / norm
        annihilation = _norm(p.A.T @ v)
        support, sign_residual, terms = _support(p, v)
        margin = _scalar(atol + rtol * np.sum(np.abs(terms)), "support margin")
        checks = {
            "nonzero_direction": norm > 0,
            "annihilation": annihilation <= atol,
            "bound_sign": support is not None,
            "strict_separation": support is not None and support < -margin,
        }
        return _report(
            "primal_infeasibility_direction", checks,
            {"input_direction_norm": norm, "annihilation": annihilation,
             "box_support": support, "bound_sign": sign_residual},
            {"annihilation": atol, "separation_margin": margin, "bound_sign": 0.0},
            {} if support is not None else {"box_support": "infinite box support"},
        )


def check_unbounded_direction(
    problem: object, direction: object, *, feasible_x: object | None = None,
    atol: float = 1e-8, rtol: float = 1e-8,
) -> NumericalEvidence:
    """Check a recession/descent direction and an independent feasible base.

    Without feasible_x, direction checks are available but passed stays False:
    dual infeasibility alone is not primal unboundedness.
    """
    with _arithmetic():
        atol, rtol = _tolerances(atol, rtol)
        p = _problem(problem)
        d = _vector(direction, "direction", p.q.size)
        norm = _norm(d)
        if norm:
            d = d / norm
        Pd, Ad = p.P @ d, p.A @ d
        lower, upper = np.isfinite(p.l), np.isfinite(p.u)
        both = lower & upper
        lower_only, upper_only = lower & ~upper, upper & ~lower
        recession = _norm(np.concatenate([
            np.abs(Ad[both]), np.maximum(-Ad[lower_only], 0),
            np.maximum(Ad[upper_only], 0),
        ]))
        spectral_coordinates = p.eigenvectors.T @ d
        positive_component = _norm(spectral_coordinates[p.eigenvalues > 0])
        kernel_limit = 64 * np.finfo(np.float64).eps * p.q.size
        slope = _scalar(p.q @ d, "descent slope")
        margin = _scalar(atol + rtol * np.sum(np.abs(p.q * d)), "descent margin")
        base_residual = base_limit = None
        base_feasible = False
        unavailable = {}
        if feasible_x is None:
            unavailable["feasible_base"] = "feasible_x required for primal unboundedness evidence"
        else:
            base = _vector(feasible_x, "feasible_x", p.q.size)
            _, base_residual, base_limit, base_feasible = _feasibility(p, base, atol, rtol)
        if not p.spectrum_resolved:
            unavailable["kernel"] = "unresolved PSD spectrum; cannot establish a numerical nullspace"
        checks = {
            "nonzero_direction": norm > 0,
            "zero_curvature": p.spectrum_resolved and _norm(Pd) <= atol
            and positive_component <= kernel_limit,
            "recession": bool(np.all(np.abs(Ad[both]) <= atol))
            and bool(np.all(Ad[lower_only] >= 0)) and bool(np.all(Ad[upper_only] <= 0)),
            "strict_descent": slope < -margin,
            "feasible_base": base_feasible,
        }
        thresholds = {
            "zero_curvature": atol, "positive_eigenspace": kernel_limit,
            "two_sided_recession": atol, "one_sided_recession": 0.0,
            "descent_margin": margin,
        }
        if base_limit is not None:
            thresholds["base_feasibility_max"] = base_limit
        return _report(
            "primal_unboundedness_direction", checks,
            {"input_direction_norm": norm, "zero_curvature": _norm(Pd),
             "positive_eigenspace": positive_component, "recession": recession,
             "descent_slope": slope, "base_feasibility": base_residual},
            thresholds, unavailable,
        )
