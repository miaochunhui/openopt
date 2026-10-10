"""Opt-in, reversible dense QP equilibration with original-coordinate checks.

Import this module explicitly; neither the dense solver nor the facade changes.
The independent CPU check includes an explicit host copy for CUDA candidates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .certificates import NumericalEvidence, check_qp_candidate
from .qp import QPConfig, QPProblem, QPResult, solve_qp

__all__ = [
    "ScalingConfig", "DiagonalScaling", "EquilibratedQP", "ScaledQPResult",
    "equilibrate_qp", "solve_scaled_qp",
]


@dataclass(frozen=True)
class ScalingConfig:
    method: str = "ruiz"
    iterations: int = 10
    scale_min: float = 1e-6
    scale_max: float = 1e6
    objective_scaling: bool = True
    check_atol: float = 1e-7
    check_rtol: float = 1e-7

    def validate(self, dtype: torch.dtype) -> None:
        if self.method not in ("none", "ruiz"):
            raise ValueError("scaling method must be none or ruiz")
        if isinstance(self.iterations, bool) or not isinstance(self.iterations, int):
            raise ValueError("scaling iterations must be a positive integer")
        if self.iterations <= 0:
            raise ValueError("scaling iterations must be a positive integer")
        if not isinstance(self.objective_scaling, bool):
            raise ValueError("objective_scaling must be a boolean")
        for name in ("scale_min", "scale_max", "check_atol", "check_rtol"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a finite real number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be a finite real number")
        if not 0 < self.scale_min <= 1 <= self.scale_max:
            raise ValueError("scaling limits must satisfy 0 < scale_min <= 1 <= scale_max")
        if self.check_atol <= 0 or not 0 <= self.check_rtol < 1:
            raise ValueError("check_atol > 0 and 0 <= check_rtol < 1 are required")
        _positive_scale(torch.tensor([self.scale_min, self.scale_max], dtype=dtype), "limits")


def _positive_scale(value: torch.Tensor, name: str) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if (
        value.layout != torch.strided or value.is_complex()
        or value.dtype not in (torch.float32, torch.float64)
        or value.device.type not in ("cpu", "cuda")
    ):
        raise ValueError(f"{name} must be a real dense CPU/CUDA float32/float64 tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must be a vector")
    if not bool(torch.isfinite(value).all() & (value > 0).all()):
        raise ValueError(f"{name} must be finite and strictly positive")
    inverse = value.reciprocal()
    if not bool(torch.isfinite(inverse).all() & (inverse > 0).all()):
        raise ValueError(f"{name} reciprocal is not representable")


def _product(value: torch.Tensor, factor: torch.Tensor | float, context: str) -> torch.Tensor:
    result = value * factor
    finite = torch.isfinite(value)
    if not bool(torch.isfinite(result[finite]).all()):
        raise ValueError(f"nonfinite scaling arithmetic: {context}")
    if bool(((value != 0) & finite & (result == 0)).any()):
        raise ValueError(f"underflow in scaling arithmetic: {context}")
    if not bool(torch.equal(torch.isinf(value), torch.isinf(result))):
        raise ValueError(f"invalid infinite bound scaling: {context}")
    return result


def _norm(value: torch.Tensor) -> float:
    return float(value.abs().max()) if value.numel() else 0.0


def _copy_problem(problem: QPProblem, device: str, dtype: torch.dtype) -> QPProblem:
    tensors = (problem.P, problem.q, problem.A, problem.l, problem.u)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("QP inputs must be tensors")
    for name, value in zip(("P", "q", "A", "l", "u"), tensors):
        if (
            value.layout != torch.strided or value.is_complex()
            or value.device.type not in ("cpu", "cuda")
        ):
            raise ValueError(f"{name} must be real dense CPU/CUDA data")
    P, q, A, l, u = [value.detach().to(device=device, dtype=dtype).clone() for value in tensors]
    if q.ndim != 1 or not q.numel():
        raise ValueError("q must be a nonempty vector")
    n = q.numel()
    if P.shape != (n, n) or A.ndim != 2 or A.shape[1] != n:
        raise ValueError("P/A dimensions must match q")
    if l.shape != (A.shape[0],) or u.shape != l.shape:
        raise ValueError("constraint bounds must match A rows")
    if not all(bool(torch.isfinite(value).all()) for value in (P, q, A)):
        raise ValueError("P/q/A must be finite")
    if bool(torch.isnan(l).any() | torch.isnan(u).any()):
        raise ValueError("bounds cannot contain NaN")
    if bool(torch.isposinf(l).any() | torch.isneginf(u).any()):
        raise ValueError("lower +inf or upper -inf is invalid")
    if bool((l > u).any()):
        raise ValueError("lower bound exceeds upper bound")
    for name, before, after in zip(("P", "q", "A", "l", "u"), tensors, (P, q, A, l, u)):
        before = before.detach().to(device=device)
        finite = torch.isfinite(before)
        if not bool(torch.isfinite(after[finite]).all()):
            raise ValueError(f"nonfinite dtype conversion: {name}")
        if bool(((before != 0) & finite & (after == 0)).any()):
            raise ValueError(f"underflow in dtype conversion: {name}")
    tolerance = 20 * torch.finfo(dtype).eps * max(1.0, _norm(P))
    difference = P - P.T
    if not bool(torch.isfinite(difference).all()):
        raise ValueError("nonfinite symmetry arithmetic")
    if _norm(difference) > tolerance:
        raise ValueError("P must be symmetric")
    eigenvalues = torch.linalg.eigvalsh(0.5 * P + 0.5 * P.T)
    if not bool(torch.isfinite(eigenvalues).all()):
        raise ValueError("nonfinite PSD spectrum")
    if float(eigenvalues.min()) < -tolerance:
        raise ValueError("P must be positive semidefinite")
    return QPProblem(P, q, A, l, u)


@dataclass(frozen=True)
class DiagonalScaling:
    """x = D*x_hat, y = E*y_hat/c; all factors are strictly positive."""

    variable_scale: torch.Tensor
    constraint_scale: torch.Tensor
    objective_scale: float = 1.0

    def __post_init__(self) -> None:
        self.validate()
        object.__setattr__(self, "variable_scale", self.variable_scale.detach().clone())
        object.__setattr__(self, "constraint_scale", self.constraint_scale.detach().clone())

    def validate(self) -> None:
        d, e = self.variable_scale, self.constraint_scale
        _positive_scale(d, "variable_scale")
        _positive_scale(e, "constraint_scale")
        if d.numel() == 0:
            raise ValueError("variable_scale must be nonempty")
        if d.device != e.device or d.dtype != e.dtype:
            raise ValueError("scaling factors must share dtype and device")
        c = self.objective_scale
        if isinstance(c, bool) or not isinstance(c, (float, int)) or not math.isfinite(c) or c <= 0:
            raise ValueError("objective_scale must be finite and strictly positive")
        _positive_scale(torch.tensor([c], dtype=d.dtype, device=d.device), "objective_scale")

    def apply(self, problem: QPProblem) -> QPProblem:
        self.validate()
        d, e, c = self.variable_scale, self.constraint_scale, self.objective_scale
        p = _copy_problem(problem, str(d.device), d.dtype)
        if p.q.shape != d.shape or p.l.shape != e.shape:
            raise ValueError("scaling dimensions must match q and A rows")
        P = _product(p.P, d[:, None], "D P")
        P = _product(_product(P, d[None, :], "D P D"), c, "c D P D")
        q = _product(_product(p.q, d, "D q"), c, "c D q")
        A = _product(_product(p.A, d[None, :], "A D"), e[:, None], "E A D")
        l, u = _product(p.l, e, "E l"), _product(p.u, e, "E u")
        return QPProblem(P, q, A, l, u)

    def _candidate(self, value: torch.Tensor, factor: torch.Tensor, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a tensor")
        if (
            value.layout != torch.strided or value.is_complex()
            or value.dtype != factor.dtype or value.device != factor.device
        ):
            raise ValueError(f"{name} must share real dense scaling dtype and device")
        if value.shape != factor.shape:
            raise ValueError(f"{name} dimensions must match scaling")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
        return value.detach()

    def restore(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate()
        d, e, c = self.variable_scale, self.constraint_scale, self.objective_scale
        x, y = self._candidate(x, d, "x"), self._candidate(y, e, "y")
        restored_x = _product(x, d, "restore x")
        restored_y = _product(_product(y, e, "restore E y"), 1 / c, "restore y/c")
        return restored_x, restored_y

    def to_scaled(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.validate()
        d, e, c = self.variable_scale, self.constraint_scale, self.objective_scale
        x, y = self._candidate(x, d, "x"), self._candidate(y, e, "y")
        scaled_x = _product(x, d.reciprocal(), "x/D")
        scaled_y = _product(_product(y, e.reciprocal(), "y/E"), c, "c y/E")
        return scaled_x, scaled_y


@dataclass(frozen=True)
class EquilibratedQP:
    original_problem: QPProblem
    scaled_problem: QPProblem
    scaling: DiagonalScaling
    iterations: int


@dataclass(frozen=True)
class ScaledQPResult:
    x: torch.Tensor
    y: torch.Tensor
    status: str
    objective: float
    scaling: DiagonalScaling
    evidence: NumericalEvidence
    scaled_result: QPResult

    @property
    def iterations(self) -> int:
        return self.scaled_result.iterations


def _update_scale(
    accumulated: torch.Tensor, norm: torch.Tensor, config: ScalingConfig, power: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Zero rows/columns retain identity updates; logs avoid intermediate overflow.
    safe_norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    log_target = accumulated.log() - power * safe_norm.log()
    updated = log_target.clamp(math.log(config.scale_min), math.log(config.scale_max)).exp()
    updated = updated.clamp(config.scale_min, config.scale_max)
    step = updated / accumulated
    _positive_scale(updated, "accumulated scaling")
    _positive_scale(step, "scaling step")
    return updated, step


@torch.no_grad()
def equilibrate_qp(
    problem: QPProblem, config: QPConfig = QPConfig(), scaling: ScalingConfig = ScalingConfig(),
) -> EquilibratedQP:
    """Bounded infinity-norm Ruiz iterations on [P A.T; A 0], plus optional cost scaling."""
    config.validate()
    scaling.validate(config.dtype)
    # Preserve original values, not the solver's rounded/scaled/symmetrized model.
    original = _copy_problem(problem, "cpu", torch.float64)
    current = _copy_problem(problem, config.device, config.dtype)
    d, e = torch.ones_like(current.q), torch.ones_like(current.l)
    c = torch.ones(1, device=current.q.device, dtype=current.q.dtype)
    if scaling.method == "none":
        return EquilibratedQP(original, current, DiagonalScaling(d, e), 0)
    for iteration in range(1, scaling.iterations + 1):
        P, q, A, l, u = current.P, current.q, current.A, current.l, current.u
        column_norm = P.abs().amax(dim=0)
        row_norm = A.abs().amax(dim=1)
        if e.numel():
            column_norm = torch.maximum(column_norm, A.abs().amax(dim=0))
        d, ds = _update_scale(d, column_norm, scaling, 0.5)
        e, es = _update_scale(e, row_norm, scaling, 0.5)
        P = _product(P, ds[:, None], f"iteration={iteration} D P")
        P = _product(P, ds[None, :], f"iteration={iteration} D P D")
        q = _product(q, ds, f"iteration={iteration} D q")
        A = _product(A, ds[None, :], f"iteration={iteration} A D")
        A = _product(A, es[:, None], f"iteration={iteration} E A D")
        l = _product(l, es, f"iteration={iteration} E l")
        u = _product(u, es, f"iteration={iteration} E u")
        if scaling.objective_scaling:
            cost_norm = torch.maximum(P.abs().max(), q.abs().max()).reshape(1)
            c, cs = _update_scale(c, cost_norm, scaling, 1.0)
            P = _product(P, cs, f"iteration={iteration} cost P")
            q = _product(q, cs, f"iteration={iteration} cost q")
        current = QPProblem(P, q, A, l, u)
    transform = DiagonalScaling(d, e, float(c[0]))
    return EquilibratedQP(original, current, transform, iteration)


@torch.no_grad()
def solve_scaled_qp(
    problem: QPProblem, config: QPConfig = QPConfig(), scaling: ScalingConfig = ScalingConfig(),
) -> ScaledQPResult:
    """Call the unchanged dense solver, restore x/y, and independently check them.

    A scaled ``solved`` candidate that fails the original check is returned as
    ``candidate_rejected``. Iteration limits are never promoted to success.
    Malformed or unrepresentable arithmetic raises, without retry/fallback.
    """
    prepared = equilibrate_qp(problem, config, scaling)
    raw = solve_qp(prepared.scaled_problem, config)
    x, y = prepared.scaling.restore(raw.x, raw.y)
    evidence = check_qp_candidate(
        prepared.original_problem,
        x.detach().to(device="cpu", dtype=torch.float64),
        y.detach().to(device="cpu", dtype=torch.float64),
        atol=scaling.check_atol, rtol=scaling.check_rtol,
    )
    status = raw.status
    if status == "solved" and not evidence.passed:
        status = "candidate_rejected"
    return ScaledQPResult(
        x=x, y=y, status=status, objective=evidence.metrics["primal_objective"],
        scaling=prepared.scaling, evidence=evidence, scaled_result=raw,
    )
