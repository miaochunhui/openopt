"""Opt-in dense QP polishing with explicitly checked original-coordinate accuracy.

The factorization and residual corrections run on the requested PyTorch device.
CPU float64 copies are used only by the independent numerical checker, not by
a solver. No regularization, arbitrary precision, least-squares or solver
fallback is used. Neither the ordinary QP solver nor its facade imports this.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .certificates import NumericalEvidence, check_qp_candidate
from .qp import QPProblem

__all__ = ["RefinementConfig", "RefinedQPCandidate", "refine_qp_candidate"]


@dataclass(frozen=True)
class RefinementConfig:
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    factor_dtype: torch.dtype = torch.float64
    max_refinement_steps: int = 3
    active_atol: float = 1e-5
    active_rtol: float = 1e-7
    check_atol: float = 1e-8
    check_rtol: float = 1e-8

    def validate(self) -> torch.device:
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("refinement supports float32/float64 only")
        if self.factor_dtype not in (torch.float32, torch.float64):
            raise ValueError("factor_dtype supports float32/float64 only")
        if self.dtype == torch.float32 and self.factor_dtype == torch.float64:
            raise ValueError("factor_dtype must not exceed residual dtype precision")
        if (
            isinstance(self.max_refinement_steps, bool)
            or not isinstance(self.max_refinement_steps, int)
            or self.max_refinement_steps < 0
        ):
            raise ValueError("max_refinement_steps must be a nonnegative integer")
        for name in ("active_atol", "active_rtol", "check_atol", "check_rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.check_atol <= 0 or self.check_rtol >= 1 or self.active_rtol >= 1:
            raise ValueError("check_atol > 0 and relative tolerances < 1 are required")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("refinement supports CPU or CUDA only")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("explicit CUDA requested but unavailable")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if device.index >= torch.cuda.device_count():
                raise ValueError("CUDA device index out of range")
        return device


@dataclass(frozen=True)
class RefinedQPCandidate:
    """Rejected/unknown results contain untouched baseline-value clones.

    ``accepted`` means numerical candidate acceptance, not a solver status or
    an exact certificate. ``attempted_*`` retain the last proposed candidate.
    """

    x: torch.Tensor
    y: torch.Tensor
    status: str
    reason: str
    baseline_evidence: NumericalEvidence
    evidence: NumericalEvidence
    attempted_x: torch.Tensor | None
    attempted_y: torch.Tensor | None
    attempted_evidence: NumericalEvidence | None
    diagnostics: dict[str, object]

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"


def _tensor(value: object, name: str) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (
        value.layout != torch.strided or value.is_complex()
        or value.dtype not in (torch.float32, torch.float64)
        or value.device.type not in ("cpu", "cuda")
    ):
        raise ValueError(f"{name} must be real dense CPU/CUDA float32/float64 data")
    return value.detach()


def _copy(value: torch.Tensor, device: torch.device, dtype: torch.dtype, name: str) -> torch.Tensor:
    result = value.to(device=device, dtype=dtype).clone()
    before = value.to(device=device)
    finite = torch.isfinite(before)
    if not bool(torch.isfinite(result[finite]).all()):
        raise ValueError(f"nonfinite dtype conversion: {name}")
    if bool(((before != 0) & finite & (result == 0)).any()):
        raise ValueError(f"underflow in dtype conversion: {name}")
    if not bool(torch.equal(torch.isinf(before), torch.isinf(result))):
        raise ValueError(f"lost infinite bounds in dtype conversion: {name}")
    return result


def _norm(value: torch.Tensor) -> float:
    return float(value.abs().max()) if value.numel() else 0.0


def _nonworsening(before: NumericalEvidence, after: NumericalEvidence) -> list[str]:
    # Objective decrease alone can reward an infeasible candidate. Compare
    # original feasibility, stationarity, sign, complementarity and |gap|.
    worsened = []
    for name in (
        "primal_feasibility", "stationarity", "bound_sign",
        "complementarity", "duality_gap",
    ):
        old, new = before.metrics[name], after.metrics[name]
        if new is None or (old is not None and abs(new) > abs(old)):
            worsened.append(name)
    return worsened


@torch.no_grad()
def refine_qp_candidate(
    problem: QPProblem, x: torch.Tensor, y: torch.Tensor,
    config: RefinementConfig = RefinementConfig(),
) -> RefinedQPCandidate:
    """Guess active bounds, solve their unregularized KKT system and refine.

    All equalities are active. A non-equality within the explicit distance
    tolerance of one finite bound is guessed active, regardless of multiplier
    sign; wrong guesses must survive the full independent check. Two near
    bounds are ambiguous and rejected. No active-set search is attempted.
    """
    device = config.validate()
    names = ("P", "q", "A", "l", "u")
    tensors = [_tensor(getattr(problem, name), name) for name in names]
    x, y = _tensor(x, "x"), _tensor(y, "y")
    if x.device != device or y.device != device:
        raise ValueError("x/y must reside on the explicit refinement device; no solver fallback")
    baseline_x, baseline_y = x.clone(), y.clone()
    original = QPProblem(*[
        _copy(value, torch.device("cpu"), torch.float64, name)
        for name, value in zip(names, tensors)
    ])
    baseline = check_qp_candidate(
        original, baseline_x.cpu().double(), baseline_y.cpu().double(),
        atol=config.check_atol, rtol=config.check_rtol,
    )
    p = QPProblem(*[
        _copy(value, device, config.dtype, name) for name, value in zip(names, tensors)
    ])
    working_x = _copy(x, device, config.dtype, "x")
    _copy(y, device, config.dtype, "y")
    diagnostics: dict[str, object] = {
        "device": str(device), "dtype": str(config.dtype),
        "factor_dtype": str(config.factor_dtype),
        "checker": "explicit CPU float64 original-coordinate numerical check",
        "solver_fallback": False, "regularization": False,
        "arbitrary_precision": False, "active_rows": [], "active_sides": [],
        "iterations": [], "accepted_step": None,
    }
    attempted_x = attempted_y = attempted_evidence = None

    def result(status: str, reason: str, best=None) -> RefinedQPCandidate:
        bx, by, evidence = best if best is not None else (baseline_x, baseline_y, baseline)
        return RefinedQPCandidate(
            bx.clone(), by.clone(), status, reason, baseline, evidence,
            attempted_x, attempted_y, attempted_evidence, diagnostics,
        )

    Ax = p.A @ working_x
    if not bool(torch.isfinite(Ax).all()):
        raise ValueError("nonfinite active-set arithmetic")
    equality = torch.isfinite(p.l) & (p.l == p.u)
    near = []
    for bound in (p.l, p.u):
        finite = torch.isfinite(bound)
        close = torch.zeros_like(finite)
        limits = config.active_atol + config.active_rtol * torch.maximum(
            Ax[finite].abs(), bound[finite].abs(),
        )
        distance = (Ax[finite] - bound[finite]).abs()
        if not bool(torch.isfinite(limits).all() & torch.isfinite(distance).all()):
            raise ValueError("nonfinite active-set tolerance/distance")
        close[finite] = distance <= limits
        near.append(close & ~equality)
    lower, upper = near
    if bool((lower & upper).any()):
        diagnostics["ambiguous_rows"] = torch.where(lower & upper)[0].cpu().tolist()
        return result("rejected", "ambiguous_active_bounds")
    rows = torch.where(equality | lower | upper)[0]
    sides = [
        "equality" if bool(equality[row]) else "lower" if bool(lower[row]) else "upper"
        for row in rows
    ]
    diagnostics["active_rows"] = rows.cpu().tolist()
    diagnostics["active_sides"] = sides
    active_A = p.A[rows]
    bounds = torch.where(equality[rows] | lower[rows], p.l[rows], p.u[rows])
    n, k = p.q.numel(), rows.numel()
    K = torch.zeros((n + k, n + k), device=device, dtype=config.dtype)
    K[:n, :n] = p.P
    K[:n, n:] = active_A.T
    K[n:, :n] = active_A
    rhs = torch.cat((-p.q, bounds))
    if not bool(torch.isfinite(K).all() & torch.isfinite(rhs).all()):
        raise ValueError("nonfinite KKT model")
    # The original checker validates convexity/symmetry before any device solve.
    singular_values = torch.linalg.svdvals(K)
    if not bool(torch.isfinite(singular_values).all()):
        return result("unknown", "nonfinite_kkt_spectrum")
    rank_limit = 64 * torch.finfo(config.dtype).eps * (n + k) * float(singular_values.max())
    if not math.isfinite(rank_limit):
        raise ValueError("nonfinite KKT rank threshold")
    diagnostics["rank_threshold"] = rank_limit
    diagnostics["smallest_singular_value"] = float(singular_values.min())
    if float(singular_values.min()) <= rank_limit:
        return result("unknown", "singular_or_unresolved_kkt")
    factor_K = _copy(K, device, config.factor_dtype, "KKT factor")
    factor, pivots, info = torch.linalg.lu_factor_ex(factor_K, check_errors=False)
    diagnostics["factor_info"] = int(info)
    if int(info) != 0:
        return result("unknown", "singular_kkt_factorization")
    if not bool(torch.isfinite(factor).all()):
        return result("unknown", "nonfinite_kkt_factorization")
    factor_rhs = _copy(rhs, device, config.factor_dtype, "KKT rhs")
    solution = torch.linalg.lu_solve(factor, pivots, factor_rhs[:, None]).flatten().to(config.dtype)
    best = None
    last_reason = "original_check_failed"
    previous_residual = None
    # Each correction depends on the previous residual; these are not
    # independent benchmark units that could be scheduled in separate Pods.
    for step in range(config.max_refinement_steps + 1):
        if not bool(torch.isfinite(solution).all()):
            diagnostics["stop_reason"] = "nonfinite_linear_solution"
            return result("unknown", "nonfinite_linear_solution")
        attempted_x = solution[:n].clone()
        attempted_y = torch.zeros_like(p.l)
        attempted_y[rows] = solution[n:]
        residual = rhs - K @ solution
        if not bool(torch.isfinite(residual).all()):
            raise ValueError(f"nonfinite residual correction step={step}")
        residual_norm = _norm(residual)
        attempted_evidence = check_qp_candidate(
            original, attempted_x.cpu().double(), attempted_y.cpu().double(),
            atol=config.check_atol, rtol=config.check_rtol,
        )
        worsened = _nonworsening(baseline, attempted_evidence)
        accepted = attempted_evidence.passed and not worsened
        entry = {
            "step": step, "linear_residual": residual_norm,
            "original_check_passed": attempted_evidence.passed,
            "worsened_metrics": worsened, "acceptable": accepted,
            "evidence": attempted_evidence.to_dict(),
        }
        diagnostics["iterations"].append(entry)
        if accepted and (best is None or not _nonworsening(best[2], attempted_evidence)):
            best = (attempted_x.clone(), attempted_y.clone(), attempted_evidence)
            diagnostics["accepted_step"] = step
        if not all(attempted_evidence.checks[name] for name in (
            "primal_feasibility", "stationarity", "bound_sign", "complementarity",
        )):
            last_reason = "original_check_failed"
        elif attempted_evidence.unavailable:
            last_reason = "original_check_unavailable"
        elif not attempted_evidence.passed:
            last_reason = "original_check_failed"
        elif worsened:
            last_reason = "accuracy_worsened"
        if residual_norm == 0:
            diagnostics["stop_reason"] = "zero_linear_residual"
            break
        if previous_residual is not None and residual_norm >= previous_residual:
            diagnostics["stop_reason"] = "linear_refinement_stagnated"
            break
        if step == config.max_refinement_steps:
            diagnostics["stop_reason"] = "refinement_step_limit"
            break
        previous_residual = residual_norm
        correction_rhs = _copy(residual, device, config.factor_dtype, "correction rhs")
        correction = torch.linalg.lu_solve(factor, pivots, correction_rhs[:, None])
        solution = solution + correction.flatten().to(config.dtype)
    if best is not None:
        return result("accepted", "original_check_passed_nonworsening", best)
    return result(
        "unknown" if last_reason == "original_check_unavailable" else "rejected", last_reason,
    )
