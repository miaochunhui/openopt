"""Experimental MILP contract backed explicitly by the HiGHS CPU reference.

Minimize q.T @ x with l <= A @ x <= u, lower <= x <= upper, and
integrality[i] in {0: continuous, 1: integer}. This is not a native/GPU solver.
See docs/research/integer.md for numerical evidence and termination semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
import warnings

import numpy as np
import torch

__all__ = [
    "MILPProblem", "MILPConfig", "MILPValidation", "MILPResult",
    "check_milp_candidate", "solve_milp", "solve_miqp",
]


@dataclass(frozen=True)
class MILPProblem:
    q: object
    A: object
    l: object
    u: object
    lower: object
    upper: object
    integrality: object
    P: object | None = None


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


@dataclass(frozen=True)
class MILPConfig:
    backend: str
    device: str = "cpu"
    time_limit: float | None = None
    node_limit: int | None = None
    abs_gap: float = 0.0
    rel_gap: float = 0.0
    feasibility_tol: float = 1e-7
    integrality_tol: float = 1e-7
    objective_tol: float = 1e-7
    presolve: bool = True

    def validate(self) -> None:
        if self.backend != "highspy":
            raise ValueError("backend must explicitly be 'highspy' (CPU reference only)")
        if torch.device(self.device).type != "cpu":
            raise ValueError("MILP supports CPU reference only; CUDA/device fallback is forbidden")
        if self.time_limit is not None:
            _number(self.time_limit, "time_limit")
        if self.node_limit is not None and (
            isinstance(self.node_limit, (bool, np.bool_))
            or not isinstance(self.node_limit, int)
            or not 0 <= self.node_limit <= 2_147_483_647
        ):
            raise ValueError("node_limit must be a nonnegative 32-bit integer")
        for name in ("abs_gap", "rel_gap"):
            _number(getattr(self, name), name)
        for name in ("feasibility_tol", "integrality_tol", "objective_tol"):
            _number(getattr(self, name), name, positive=True)
        if not 1e-10 <= self.feasibility_tol < 0.5:
            raise ValueError("feasibility_tol must be in [1e-10, 0.5)")
        if not 1e-10 <= self.integrality_tol < 0.5:
            raise ValueError("integrality_tol must be in [1e-10, 0.5)")
        if not isinstance(self.presolve, bool):
            raise ValueError("presolve must be boolean")


@dataclass(frozen=True)
class MILPValidation:
    checks: dict[str, bool]
    objective: float
    constraint_violation: float
    bound_violation: float
    integrality_violation: float
    dual_bound: float | None
    signed_gap: float | None
    abs_gap: float | None
    rel_gap: float | None

    @property
    def feasible(self) -> bool:
        return all(self.checks[name] for name in ("constraints", "bounds", "integrality"))


@dataclass(frozen=True)
class MILPResult:
    x: np.ndarray | None
    status: str
    termination_reason: str
    incumbent: float | None
    dual_bound: float | None
    abs_gap: float | None
    rel_gap: float | None
    node_count: int | None
    backend_status: str
    backend: str
    backend_version: str
    validation: MILPValidation | None
    qualification: str = "HiGHS CPU reference numerical evidence; not native GPU or exact proof"

    @property
    def objective(self) -> float | None:
        return self.incumbent


def _array(value: object, name: str, *, bounds: bool = False) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be CPU data; implicit CUDA transfer is forbidden")
        if value.layout != torch.strided or value.is_complex():
            raise ValueError(f"{name} must be real dense data")
        value = value.detach().numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu" or array.ndim == 0:
        raise ValueError(f"{name} must be a real numeric dense array")
    result = np.array(array, dtype=np.float64, copy=True)
    if np.isnan(result).any() or (not bounds and not np.isfinite(result).all()):
        raise ValueError(f"{name} contains nonfinite data")
    return result


def _problem(problem: MILPProblem) -> MILPProblem:
    if not isinstance(problem, MILPProblem):
        raise TypeError("problem must be MILPProblem; QP/MIQP objects are not accepted")
    if problem.P is not None:
        raise NotImplementedError("MIQP is unsupported; quadratic P must not be supplied")
    values = {
        name: _array(getattr(problem, name), name, bounds=name in ("l", "u", "lower", "upper"))
        for name in ("q", "A", "l", "u", "lower", "upper", "integrality")
    }
    p = MILPProblem(**values)
    if p.q.ndim != 1 or not p.q.size:
        raise ValueError("q must be a nonempty vector")
    n = p.q.size
    if p.A.ndim != 2 or p.A.shape[1] != n:
        raise ValueError("A dimensions must match q")
    if p.l.shape != (p.A.shape[0],) or p.u.shape != p.l.shape:
        raise ValueError("l/u must match A rows")
    for name in ("lower", "upper", "integrality"):
        if getattr(p, name).shape != (n,):
            raise ValueError(f"{name} must match q")
    for lower, upper in ((p.l, p.u), (p.lower, p.upper)):
        if np.isposinf(lower).any() or np.isneginf(upper).any():
            raise ValueError("lower +inf and upper -inf are invalid")
        if np.any(lower > upper):
            raise ValueError("lower bound exceeds upper bound")
    if not np.isin(p.integrality, (0, 1)).all():
        raise ValueError("integrality entries must be 0 (continuous) or 1 (integer)")
    return p


def _finite(value: object, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {name}")
    return result


def _violation(value: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    lo, hi = np.isfinite(lower), np.isfinite(upper)
    errors = np.concatenate((
        np.maximum(lower[lo] - value[lo], 0), np.maximum(value[hi] - upper[hi], 0),
    ))
    return _finite(np.max(errors, initial=0), "bound/constraint violation")


def check_milp_candidate(
    problem: MILPProblem, x: object, *, dual_bound: float | None = None,
    feasibility_tol: float = 1e-7, integrality_tol: float = 1e-7,
    objective_tol: float = 1e-7, abs_gap: float = 0.0, rel_gap: float = 0.0,
) -> MILPValidation:
    """Independent original-coordinate checks; a supplied bound is not certified.

    No backend status/residual is consulted. Gap = |q.T x - dual_bound|;
    relative gap divides by max(1, |q.T x|). Missing bounds do not prove optimality.
    """
    config = MILPConfig(
        backend="highspy", feasibility_tol=feasibility_tol, integrality_tol=integrality_tol,
        objective_tol=objective_tol, abs_gap=abs_gap, rel_gap=rel_gap,
    )
    config.validate()
    p = _problem(problem)
    candidate = _array(x, "x")
    if candidate.shape != p.q.shape:
        raise ValueError("x dimensions must match q")
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            activity = p.A @ candidate
            if not np.isfinite(activity).all():
                raise ValueError("nonfinite constraint activity")
            constraints = _violation(activity, p.l, p.u)
            bounds = _violation(candidate, p.lower, p.upper)
            integers = candidate[p.integrality == 1]
            if np.any(np.abs(integers) > 2**53 - 1):
                raise ValueError("integer candidate exceeds exact float64 integer range")
            integer_error = _finite(
                np.max(np.abs(integers - np.rint(integers)), initial=0), "integrality violation",
            )
            objective = _finite(p.q @ candidate, "incumbent objective")
            bound = None if dual_bound is None else _finite(dual_bound, "dual_bound")
            signed = None if bound is None else _finite(
                np.subtract(objective, bound), "signed gap",
            )
            absolute = None if signed is None else abs(signed)
            scale = max(1.0, abs(objective))
            relative = None if absolute is None else _finite(absolute / scale, "relative gap")
            checks = {
                "constraints": constraints <= feasibility_tol,
                "bounds": bounds <= feasibility_tol,
                "integrality": integer_error <= integrality_tol,
                "dual_bound_available": bound is not None,
                "dual_bound_order": signed is not None and signed >= -objective_tol,
                "gap": absolute is not None and (
                    absolute <= abs_gap + objective_tol
                    or relative <= rel_gap + objective_tol / scale
                ),
            }
    except FloatingPointError as error:
        raise ValueError(f"MILP candidate arithmetic failed: {error}") from error
    return MILPValidation(
        checks, objective, constraints, bounds, integer_error, bound, signed, absolute, relative,
    )


def _load_highspy():
    try:
        import highspy
    except ImportError as error:
        raise ImportError(
            "explicit highspy CPU reference backend unavailable; no fallback or installation"
        ) from error
    return highspy


def _ok(highspy, status, operation: str) -> None:
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS {operation} failed status={status!s}")


def _reference_range(p: MILPProblem) -> None:
    # HiGHS interprets very large finite bounds as infinity and drops tiny A.
    for name in ("q", "l", "u", "lower", "upper"):
        value = getattr(p, name)
        if np.any(np.abs(value[np.isfinite(value)]) >= 1e20):
            raise ValueError(f"{name} exceeds HiGHS finite reference range (<1e20)")
    nonzero = np.abs(p.A[p.A != 0])
    if np.any((nonzero <= 1e-12) | (nonzero >= 1e15)):
        raise ValueError("nonzero A exceeds HiGHS reference range (1e-12, 1e15)")


def _lp_dual_bound(p: MILPProblem, solution, tol: float) -> float:
    row = _array(solution.row_dual, "LP row_dual")
    col = _array(solution.col_dual, "LP col_dual")
    if row.shape != p.l.shape or col.shape != p.q.shape:
        raise RuntimeError("HiGHS LP dual dimensions mismatch")
    try:
        with np.errstate(over="raise", invalid="raise", under="raise"):
            residual = p.q - p.A.T @ row - col
            if np.max(np.abs(residual), initial=0) > tol:
                raise RuntimeError("HiGHS LP dual stationarity check failed")
            total = 0.0
            for dual, lower, upper in ((row, p.l, p.u), (col, p.lower, p.upper)):
                positive, negative = dual > 0, dual < 0
                if np.any(~np.isfinite(lower[positive])) or np.any(~np.isfinite(upper[negative])):
                    raise RuntimeError("HiGHS LP dual has infinite bound support")
                total += _finite(
                    np.sum(dual[positive] * lower[positive])
                    + np.sum(dual[negative] * upper[negative]), "LP dual objective",
                )
            return _finite(total, "LP dual bound")
    except FloatingPointError as error:
        raise RuntimeError(f"HiGHS LP dual arithmetic failed: {error}") from error


def solve_milp(problem: MILPProblem, config: MILPConfig) -> MILPResult:
    """Run an explicitly requested CPU reference; never enumerate or route to CUDA."""
    config.validate()
    p = _problem(problem)
    _reference_range(p)
    highspy = _load_highspy()
    solver = highspy.Highs()
    options = {
        "output_flag": False, "threads": 1, "random_seed": 0,
        "presolve": "on" if config.presolve else "off",
        "allow_unbounded_or_infeasible": False,
        "small_matrix_value": 1e-12,
        "mip_abs_gap": config.abs_gap, "mip_rel_gap": config.rel_gap,
        "primal_feasibility_tolerance": config.feasibility_tol,
        "mip_feasibility_tolerance": min(config.feasibility_tol, config.integrality_tol),
    }
    if config.time_limit is not None:
        options["time_limit"] = config.time_limit
    if config.node_limit is not None:
        options["mip_max_nodes"] = config.node_limit
    for name, value in options.items():
        _ok(highspy, solver.setOptionValue(name, value), f"setOptionValue({name}={value!r})")
    lp = highspy.HighsLp()
    lp.num_col_, lp.num_row_ = p.q.size, p.A.shape[0]
    lp.sense_ = highspy.ObjSense.kMinimize
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = p.q, p.lower, p.upper
    lp.row_lower_, lp.row_upper_ = p.l, p.u
    lp.integrality_ = [
        highspy.HighsVarType.kInteger if value else highspy.HighsVarType.kContinuous
        for value in p.integrality
    ]
    rows, cols = np.nonzero(p.A)
    lp.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = lp.num_col_, lp.num_row_
    lp.a_matrix_.start_ = np.concatenate(([0], np.cumsum(np.bincount(rows, minlength=lp.num_row_))))
    lp.a_matrix_.index_, lp.a_matrix_.value_ = cols, p.A[rows, cols]
    _ok(highspy, solver.passModel(lp), "passModel")
    run_status = solver.run()
    raw = solver.getModelStatus()
    S = highspy.HighsModelStatus
    limits = {
        S.kTimeLimit: "time_limit", S.kIterationLimit: "iteration_limit",
        S.kSolutionLimit: "node_or_solution_limit", S.kInterrupt: "interrupt",
        S.kMemoryLimit: "memory_limit", S.kObjectiveBound: "objective_bound",
        S.kObjectiveTarget: "objective_target",
    }
    if run_status == highspy.HighsStatus.kError:
        raise RuntimeError(f"HiGHS run failed status={run_status!s} model_status={raw!s}")
    if run_status == highspy.HighsStatus.kWarning:
        warnings.warn(f"HiGHS run warning model_status={raw!s}", RuntimeWarning, stacklevel=2)
    if raw in (S.kLoadError, S.kModelError, S.kPresolveError, S.kSolveError, S.kPostsolveError):
        raise RuntimeError(f"HiGHS failed model_status={raw!s} run_status={run_status!s}")
    info, solution = solver.getInfo(), solver.getSolution()
    is_mip = bool(np.any(p.integrality))
    nodes = int(info.mip_node_count) if info.valid and is_mip else None
    if nodes is not None and nodes < 0:
        raise RuntimeError(f"HiGHS invalid mip_node_count={nodes}")
    reason = limits.get(raw, raw.name.removeprefix("k").lower())
    if raw == S.kSolutionLimit and config.node_limit is not None and nodes is not None:
        if nodes >= config.node_limit:
            reason = "node_limit"
    bound = None
    if info.valid and is_mip and raw not in (S.kInfeasible, S.kUnbounded, S.kUnboundedOrInfeasible):
        value = float(info.mip_dual_bound)
        if math.isnan(value) or value == math.inf:
            raise RuntimeError(f"HiGHS nonfinite/invalid mip_dual_bound={value!r} status={raw!s}")
        if value != -math.inf:
            bound = _finite(value, "HiGHS mip_dual_bound")
    candidate = evidence = None
    incumbent = absolute = relative = None
    has_candidate = (
        info.valid and solution.value_valid
        and info.primal_solution_status == highspy.kSolutionStatusFeasible
    )
    if has_candidate:
        if raw == S.kInfeasible:
            raise RuntimeError("HiGHS infeasible status contradicts a feasible incumbent")
        candidate = _array(solution.col_value, "HiGHS incumbent")
        if not is_mip and solution.dual_valid and raw == S.kOptimal:
            bound = _lp_dual_bound(p, solution, config.objective_tol)
        evidence = check_milp_candidate(
            p, candidate, dual_bound=bound, feasibility_tol=config.feasibility_tol,
            integrality_tol=config.integrality_tol, objective_tol=config.objective_tol,
            abs_gap=config.abs_gap, rel_gap=config.rel_gap,
        )
        if not evidence.feasible:
            raise RuntimeError(f"HiGHS invalid incumbent: independent checks={evidence.checks}")
        native_objective = _finite(info.objective_function_value, "HiGHS incumbent objective")
        if abs(evidence.objective - native_objective) > config.objective_tol:
            raise RuntimeError("HiGHS incumbent objective disagrees with independent q.T @ x")
        if bound is not None and not evidence.checks["dual_bound_order"]:
            raise RuntimeError("HiGHS dual_bound exceeds independently recomputed incumbent")
        incumbent, absolute, relative = evidence.objective, evidence.abs_gap, evidence.rel_gap
    if raw == S.kOptimal:
        if evidence is None or not all(evidence.checks.values()):
            raise RuntimeError("HiGHS optimal status lacks independent incumbent/bound/gap evidence")
        status, reason = "optimal", "optimal"
    elif raw == S.kInfeasible:
        status = "infeasible"
    elif raw == S.kUnbounded:
        status = "unbounded"
    elif raw == S.kUnboundedOrInfeasible:
        status, reason = "unknown", "unbounded_or_infeasible"
    else:
        status = "feasible" if evidence is not None else "unknown"
    return MILPResult(
        candidate, status, reason, incumbent, bound, absolute, relative, nodes,
        raw.name, "highspy", solver.version(), evidence,
    )


def solve_miqp(*args, **kwargs):
    """MIQP is deliberately unsupported, including zero-Hessian MIQP requests."""
    raise NotImplementedError("MIQP is unsupported; use the explicitly linear MILP contract")
