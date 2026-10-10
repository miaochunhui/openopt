"""Dense conic contracts and independent CPU numerical checks.

min 0.5*x.T*P*x + q.T*x, subject to A*x + s = b, s in K.
Zero, nonnegative, Lorentz and rotated Lorentz cones are supported here.
The solver in this module is an explicitly selected Clarabel CPU reference;
the separate native_conic module supplies opt-in PyTorch execution.
No exact certificate or automatic fallback is provided.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
from typing import Iterator

import numpy as np
import torch

from .certificates import NumericalEvidence

__all__ = [
    "ZeroCone", "NonnegativeCone", "SecondOrderCone", "RotatedSecondOrderCone",
    "Zero", "Nonnegative", "SOC", "RSOC",
    "ConicProblem", "ConicConfig", "ConicResult",
    "check_cone_feasibility", "check_conic_candidate", "solve_conic",
]


def _dimension(value: int, minimum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"cone dimension must be an integer >= {minimum}")


@dataclass(frozen=True)
class ZeroCone:
    """s = 0 (equalities); its dual is unrestricted R^dimension."""

    dimension: int

    def __post_init__(self) -> None:
        _dimension(self.dimension, 1)


@dataclass(frozen=True)
class NonnegativeCone:
    """s >= 0, hence the corresponding rows mean A*x <= b; self-dual."""

    dimension: int

    def __post_init__(self) -> None:
        _dimension(self.dimension, 1)


@dataclass(frozen=True)
class SecondOrderCone:
    """Lorentz cone (t, u) with t >= ||u||_2; self-dual, not rotated."""

    dimension: int

    def __post_init__(self) -> None:
        _dimension(self.dimension, 2)


@dataclass(frozen=True)
class RotatedSecondOrderCone:
    """(a, b, w): a,b >= 0 and 2*a*b >= ||w||_2^2; self-dual."""

    dimension: int

    def __post_init__(self) -> None:
        _dimension(self.dimension, 3)


Zero = ZeroCone
Nonnegative = NonnegativeCone
SOC = SecondOrderCone
RSOC = RotatedSecondOrderCone
Cone = ZeroCone | NonnegativeCone | SecondOrderCone | RotatedSecondOrderCone


@dataclass(frozen=True)
class ConicProblem:
    P: object
    q: object
    A: object
    b: object
    cones: tuple[Cone, ...]


@dataclass(frozen=True)
class ConicConfig:
    backend: str | None = None
    device: str = "cpu"
    max_iterations: int = 200
    eps_abs: float = 1e-8
    eps_rel: float = 1e-8

    def validate(self) -> None:
        device = torch.device(self.device)
        if device.type == "cuda":
            raise RuntimeError("explicit CUDA requested: conic has no native GPU backend")
        if device != torch.device("cpu"):
            raise ValueError("conic reference supports device='cpu' only")
        if self.backend != "clarabel_reference":
            raise ValueError("explicit backend='clarabel_reference' is required; no auto routing")
        _dimension(self.max_iterations, 1)
        _tolerances(self.eps_abs, self.eps_rel)


@dataclass(frozen=True)
class ConicResult:
    x: np.ndarray
    s: np.ndarray
    z: np.ndarray
    status: str
    backend_status: str
    iterations: int
    objective: float | None
    evidence: NumericalEvidence | None
    backend_version: str
    backend: str = "clarabel_reference"
    device: str = "cpu"
    certified: bool = False
    qualification: str = "third-party CPU reference; numerical checks only, not exact certificates"


@contextmanager
def _arithmetic() -> Iterator[None]:
    try:
        with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
            yield
    except FloatingPointError as error:
        raise ValueError(f"conic arithmetic failed: {error}") from error


def _array(value: object, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be on CPU; implicit device transfer is forbidden")
        if value.layout != torch.strided or value.is_complex():
            raise ValueError(f"{name} must be real dense data")
        value = value.detach().to(dtype=torch.float64).numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real numeric dense data")
    result = np.array(array, dtype=np.float64, copy=True)
    if not bool(np.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


def _vector(value: object, name: str, size: int) -> np.ndarray:
    result = _array(value, name)
    if result.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
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
        raise ValueError("tolerances cannot be booleans")
    atol, rtol = float(atol), float(rtol)
    if not math.isfinite(atol) or atol <= 0:
        raise ValueError("atol must be finite and positive")
    if not math.isfinite(rtol) or not 0 <= rtol < 1:
        raise ValueError("rtol must be finite and in [0, 1)")
    return atol, rtol


def _cones(values: object, rows: int) -> tuple[Cone, ...]:
    if not isinstance(values, (tuple, list)):
        raise TypeError("cones must be an ordered tuple or list")
    cones = tuple(values)
    if any(type(cone) not in (
        ZeroCone, NonnegativeCone, SecondOrderCone, RotatedSecondOrderCone,
    ) for cone in cones):
        raise TypeError("supported cones are Zero, Nonnegative, SOC and RSOC only")
    if sum(cone.dimension for cone in cones) != rows:
        raise ValueError("sum of cone dimensions must equal A/b rows")
    return cones


@dataclass(frozen=True)
class _Model:
    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    b: np.ndarray
    cones: tuple[Cone, ...]
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    spectral_threshold: float


def _model(problem: ConicProblem) -> _Model:
    if not isinstance(problem, ConicProblem):
        raise TypeError("problem must be a ConicProblem")
    P, q, A, b = [_array(getattr(problem, name), name) for name in ("P", "q", "A", "b")]
    if q.ndim != 1 or not q.size:
        raise ValueError("q must be a nonempty vector")
    n = q.size
    if P.shape != (n, n) or A.ndim != 2 or A.shape[1] != n:
        raise ValueError("P/A dimensions must match q")
    if b.shape != (A.shape[0],):
        raise ValueError("b must match A rows")
    cones = _cones(problem.cones, b.size)
    symmetry = _scalar(64 * np.finfo(np.float64).eps * n * _norm(P), "symmetry threshold")
    if _norm(P - P.T) > symmetry:
        raise ValueError("P must be a full symmetric matrix, not upper-triangular storage")
    P = 0.5 * P + 0.5 * P.T
    eigenvalues, eigenvectors = np.linalg.eigh(P)
    spectral = _scalar(
        64 * np.finfo(np.float64).eps * n * _norm(eigenvalues), "spectral threshold",
    )
    if eigenvalues[0] < -spectral:
        raise ValueError("P must be positive semidefinite")
    return _Model(P, q, A, b, cones, eigenvalues, eigenvectors, spectral)


def _evidence(
    kind: str, checks: dict[str, bool], metrics: dict[str, float | None],
    thresholds: dict[str, float], unavailable: dict[str, str] | None = None,
) -> NumericalEvidence:
    checks = {name: bool(value) for name, value in checks.items()}
    return NumericalEvidence(
        kind, all(checks.values()), checks,
        {name: None if value is None else _scalar(value, name) for name, value in metrics.items()},
        {name: _scalar(value, name) for name, value in thresholds.items()},
        {} if unavailable is None else unavailable,
    )


def _rotated_required_head(
    first: float, second: float, tail: np.ndarray, atol: float, rtol: float,
) -> tuple[float, bool]:
    """Compare original binary64 products in a bounded dyadic accumulator.

    The unit is the smallest binary64 subnormal. Squared input integers need
    at most 4198 bits plus log2(tail size); this is not an optimizer certificate.
    """
    def integer(value):
        numerator, denominator = float(value).as_integer_ratio()
        return numerator << (1074 - (denominator.bit_length() - 1))

    a, b = integer(first), integer(second)
    squared = sum(integer(value) ** 2 for value in tail)
    large = max(a, b)
    small = min(a, b)
    if large > 0:
        unit = 1 << 1074
        radial_pass = (
            (squared - 2 * large * small) * unit
            <= 2 * large * integer(atol) * unit
            + integer(rtol) * max(2 * large * abs(small), squared)
        )
        try:
            required = squared / ((2 * large) << 1074)
        except OverflowError as error:
            raise ValueError("RSOC required head is not representable in binary64") from error
    else:
        required = _norm(tail)
        radial_pass = required - min(first, second) <= (
            atol + rtol * max(abs(min(first, second)), required)
        )
    return required, radial_pass


def _cone_check(
    value: np.ndarray, cones: tuple[Cone, ...], dual: bool, atol: float, rtol: float,
) -> NumericalEvidence:
    checks, metrics, thresholds = {}, {}, {}
    offset = 0
    for index, cone in enumerate(cones):
        block = value[offset:offset + cone.dimension]
        offset += cone.dimension
        name = f"block_{index}"
        if isinstance(cone, ZeroCone) and dual:
            violation, limits = np.zeros_like(block), np.full_like(block, atol)
        elif isinstance(cone, ZeroCone):
            violation, limits = np.abs(block), np.full_like(block, atol)
        elif isinstance(cone, NonnegativeCone):
            violation = np.maximum(-block, 0)
            limits = atol + rtol * np.abs(block)
        elif isinstance(cone, RotatedSecondOrderCone):
            first, second = float(block[0]), float(block[1])
            small = min(first, second)
            required, radial_pass = _rotated_required_head(first, second, block[2:], atol, rtol)
            violation = np.array([
                max(-first, 0.0), max(-second, 0.0),
                max(required - small, 0.0),
            ])
            limits = np.array([
                atol + rtol * abs(first), atol + rtol * abs(second),
                atol + rtol * max(abs(small), required),
            ])
        else:
            tail = _scalar(np.hypot.reduce(block[1:]), f"{name} SOC norm")
            violation = np.array([max(tail - block[0], 0.0)])
            limits = np.array([atol + rtol * max(abs(block[0]), tail)])
        checks[name] = (
            bool(np.all(violation[:2] <= limits[:2])) and radial_pass
            if isinstance(cone, RotatedSecondOrderCone)
            else bool(np.all(violation <= limits))
        )
        metrics[name] = _norm(violation)
        thresholds[name] = _norm(limits)
    # An empty product is R^0, but callers must explicitly supply a size-0 vector.
    return _evidence("dual_cone_feasibility" if dual else "cone_feasibility", checks, metrics, thresholds)


def check_cone_feasibility(
    value: object, cones: tuple[Cone, ...], *, dual: bool = False,
    atol: float = 1e-8, rtol: float = 1e-8,
) -> NumericalEvidence:
    """Check ordered cone blocks independently of any solver or projection."""
    with _arithmetic():
        atol, rtol = _tolerances(atol, rtol)
        if not isinstance(dual, bool):
            raise ValueError("dual must be a boolean")
        value = _array(value, "cone vector")
        if value.ndim != 1:
            raise ValueError("cone vector must be one-dimensional")
        return _cone_check(value, _cones(cones, value.size), dual, atol, rtol)


def _candidate(
    p: _Model, x: np.ndarray, s: np.ndarray, z: np.ndarray, atol: float, rtol: float,
) -> NumericalEvidence:
    Ax, Px, ATz = p.A @ x, p.P @ x, p.A.T @ z
    equation = Ax + s - p.b
    equation_limits = atol + rtol * np.maximum.reduce([np.abs(Ax), np.abs(s), np.abs(p.b)])
    primal_cone = _cone_check(s, p.cones, False, atol, rtol)
    actual_cone = _cone_check(p.b - Ax, p.cones, False, atol, rtol)
    dual_cone = _cone_check(z, p.cones, True, atol, rtol)
    stationarity = Px + p.q + ATz
    stationarity_limits = atol + rtol * np.maximum.reduce([np.abs(Px), np.abs(p.q), np.abs(ATz)])
    checks = {
        "primal_equation": bool(np.all(np.abs(equation) <= equation_limits)),
        "primal_cone": primal_cone.passed,
        "original_slack_cone": actual_cone.passed,
        "dual_cone": dual_cone.passed,
        "stationarity": bool(np.all(np.abs(stationarity) <= stationarity_limits)),
    }
    metrics: dict[str, float | None] = {
        "primal_equation": _norm(equation), "stationarity": _norm(stationarity),
        "primal_cone": max(primal_cone.metrics.values(), default=0.0),
        "original_slack_cone": max(actual_cone.metrics.values(), default=0.0),
        "dual_cone": max(dual_cone.metrics.values(), default=0.0),
    }
    thresholds = {
        "primal_equation_max": _norm(equation_limits),
        "stationarity_max": _norm(stationarity_limits),
        "spectral": p.spectral_threshold,
    }
    offset = 0
    for index, cone in enumerate(p.cones):
        block = slice(offset, offset + cone.dimension)
        offset += cone.dimension
        products = s[block] * z[block]
        complementarity = abs(_scalar(np.sum(products), "block complementarity"))
        limit = _scalar(atol + rtol * np.sum(np.abs(products)), "complementarity threshold")
        name = f"complementarity_block_{index}"
        checks[name] = complementarity <= limit
        metrics[name], thresholds[name] = complementarity, limit
    primal = _scalar(0.5 * x @ Px + p.q @ x, "primal objective")
    h = p.q + ATz
    range_limits = atol + rtol * np.maximum(np.abs(p.q), np.abs(ATz))
    dual = gap = range_residual = None
    unavailable = {}
    resolved = bool(np.all((p.eigenvalues == 0) | (p.eigenvalues > p.spectral_threshold)))
    if not dual_cone.passed:
        unavailable["dual_objective"] = "z is outside the numerical dual cone"
    elif not resolved:
        unavailable["dual_objective"] = "unresolved PSD spectrum; no regularized inverse used"
    else:
        coordinates = p.eigenvectors.T @ h
        inverse_coordinates = np.zeros_like(coordinates)
        positive = p.eigenvalues > 0
        inverse_coordinates[positive] = coordinates[positive] / p.eigenvalues[positive]
        inverse_h = p.eigenvectors @ inverse_coordinates
        range_error = h - p.P @ inverse_h
        range_residual = _norm(range_error)
        if not bool(np.all(np.abs(range_error) <= range_limits)):
            unavailable["dual_objective"] = "q + A.T @ z is outside numerical range(P)"
        else:
            dual = _scalar(-0.5 * h @ inverse_h - p.b @ z, "dual objective")
            gap = _scalar(primal - dual, "signed duality gap")
            limit = _scalar(atol + rtol * max(abs(primal), abs(dual)), "gap threshold")
            thresholds["duality_gap"] = limit
    checks["dual_objective_available"] = dual is not None
    checks["duality_gap"] = gap is not None and abs(gap) <= thresholds["duality_gap"]
    metrics.update(primal_objective=primal, dual_objective=dual, duality_gap=gap, range_residual=range_residual)
    thresholds["range_max"] = _norm(range_limits)
    return _evidence("conic_candidate_optimality", checks, metrics, thresholds, unavailable)


def check_conic_candidate(
    problem: ConicProblem, x: object, s: object, z: object, *,
    atol: float = 1e-8, rtol: float = 1e-8,
) -> NumericalEvidence:
    """Recompute original-coordinate feasibility, stationarity and PSD dual gap.

    z uses L = objective + z.T*(Ax+s-b), so z belongs to K* and
    P*x + q + A.T*z = 0. A passing report is approximate, never exact proof.
    """
    with _arithmetic():
        atol, rtol = _tolerances(atol, rtol)
        p = _model(problem)
        return _candidate(
            p, _vector(x, "x", p.q.size), _vector(s, "s", p.b.size),
            _vector(z, "z", p.b.size), atol, rtol,
        )


def _reference_dependencies():
    try:
        clarabel = importlib.import_module("clarabel")
        sparse = importlib.import_module("scipy.sparse")
    except ImportError as error:
        raise RuntimeError(
            "clarabel_reference requires working optional clarabel and scipy dependencies"
        ) from error
    return clarabel, sparse


def _rotated_coordinates(value: np.ndarray, cones: tuple[Cone, ...]) -> np.ndarray:
    """Apply symmetric orthogonal T; T is also its inverse and dual map."""
    result = value.copy()
    offset = 0
    inverse_root_two = 1.0 / math.sqrt(2.0)
    for cone in cones:
        if isinstance(cone, RotatedSecondOrderCone):
            first, second = value[offset].copy(), value[offset + 1].copy()
            result[offset] = inverse_root_two * first + inverse_root_two * second
            result[offset + 1] = inverse_root_two * first - inverse_root_two * second
        offset += cone.dimension
    if not np.isfinite(result).all():
        raise ValueError("nonfinite rotated-cone coordinate transformation")
    return result


def solve_conic(problem: ConicProblem, config: ConicConfig = ConicConfig()) -> ConicResult:
    """Execute only the explicitly selected third-party CPU reference backend.

    Raw infeasibility statuses are retained, not independently certified here.
    DualInfeasible is not promoted to primal unboundedness without a base point.
    """
    config.validate()
    with _arithmetic():
        p = _model(problem)
    clarabel, sparse = _reference_dependencies()
    cone_types = {
        ZeroCone: clarabel.ZeroConeT,
        NonnegativeCone: clarabel.NonnegativeConeT,
        SecondOrderCone: clarabel.SecondOrderConeT,
        RotatedSecondOrderCone: clarabel.SecondOrderConeT,
    }
    settings = clarabel.DefaultSettings()
    for name, value in {
        "verbose": False, "max_iter": config.max_iterations, "max_threads": 1,
        "tol_gap_abs": config.eps_abs, "tol_gap_rel": config.eps_rel,
        "tol_feas": config.eps_abs,
    }.items():
        if not hasattr(settings, name):
            raise RuntimeError(f"Clarabel settings field unavailable: {name}")
        setattr(settings, name, value)
    # Clarabel consumes the upper triangle; independent checks use the full P.
    with _arithmetic():
        reference_A = _rotated_coordinates(p.A, p.cones)
        reference_b = _rotated_coordinates(p.b, p.cones)
    solver = clarabel.DefaultSolver(
        sparse.csc_matrix(np.triu(p.P)), p.q.copy(), sparse.csc_matrix(reference_A),
        reference_b, [cone_types[type(cone)](cone.dimension) for cone in p.cones], settings,
    )
    solution = solver.solve()
    raw = str(solution.status)
    iterations = solution.iterations
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise RuntimeError(f"invalid Clarabel iteration count: {iterations!r}")
    with _arithmetic():
        x, s, z = (
            _vector(getattr(solution, name), name, size)
            for name, size in (("x", p.q.size), ("s", p.b.size), ("z", p.b.size))
        )
        s = _rotated_coordinates(s, p.cones)
        z = _rotated_coordinates(z, p.cones)
    statuses = {
        "Solved": "reference_candidate_rejected",
        "AlmostSolved": "reference_almost_solved",
        "PrimalInfeasible": "reference_primal_infeasible_unverified",
        "AlmostPrimalInfeasible": "reference_almost_primal_infeasible_unverified",
        "DualInfeasible": "reference_dual_infeasible_unverified",
        "AlmostDualInfeasible": "reference_almost_dual_infeasible_unverified",
        "MaxIterations": "max_iterations", "MaxTime": "time_limit",
        "NumericalError": "numerical_error", "InsufficientProgress": "insufficient_progress",
    }
    if raw not in statuses:
        raise RuntimeError(f"unrecognized Clarabel termination status: {raw}")
    evidence = None
    objective = None
    status = statuses[raw]
    if raw in ("Solved", "AlmostSolved"):
        with _arithmetic():
            evidence = _candidate(p, x, s, z, config.eps_abs, config.eps_rel)
        objective = evidence.metrics["primal_objective"]
        if raw == "Solved" and evidence.passed:
            status = "reference_solved"
    return ConicResult(
        x=x, s=s, z=z, status=status, backend_status=raw,
        iterations=iterations, objective=objective, evidence=evidence,
        backend_version=str(clarabel.__version__),
    )
