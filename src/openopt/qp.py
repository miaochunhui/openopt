"""Experimental domain-neutral dense convex quadratic programming.

Minimize 0.5*x.T*P*x + q.T*x subject to l <= A*x <= u. Optional termination
uses independent original-coordinate numerical evidence, never exact proofs.
Explicit devices never fall back. No portfolio assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .certificates import NumericalEvidence


@dataclass(frozen=True)
class QPProblem:
    P: torch.Tensor
    q: torch.Tensor
    A: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor


@dataclass(frozen=True)
class QPConfig:
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    rho: float = 1.0
    sigma: float = 1e-6
    relaxation: float = 1.6
    max_iterations: int = 5000
    check_interval: int = 25
    eps_abs: float = 1e-7
    eps_rel: float = 1e-7
    linear_solver: str = "cholesky"
    adaptive_rho: bool = False
    adaptive_rho_interval: int = 100
    adaptive_rho_tolerance: float = 5.0
    rho_min: float = 1e-6
    rho_max: float = 1e6
    dual_gap_termination: bool = False
    direction_termination: bool = False
    direction_atol: float = 1e-8
    direction_rtol: float = 1e-8

    def validate(self) -> None:
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("QP supports float32/float64 only")
        if self.linear_solver not in ("cholesky", "triangular"):
            raise ValueError("linear_solver must be cholesky or triangular")
        for name in ("rho", "sigma", "eps_abs", "eps_rel"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.relaxation) or not 0 < self.relaxation < 2:
            raise ValueError("relaxation must be in (0, 2)")
        for name in ("max_iterations", "check_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.adaptive_rho, bool):
            raise ValueError("adaptive_rho must be a boolean")
        interval = self.adaptive_rho_interval
        if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0:
            raise ValueError("adaptive_rho_interval must be a positive integer")
        for name in ("rho_min", "rho_max"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.rho_min > self.rho_max:
            raise ValueError("rho_min must not exceed rho_max")
        if not math.isfinite(self.adaptive_rho_tolerance) or self.adaptive_rho_tolerance <= 1:
            raise ValueError("adaptive_rho_tolerance must be finite and greater than one")
        if self.adaptive_rho and not self.rho_min <= self.rho <= self.rho_max:
            raise ValueError("adaptive initial rho must be within rho_min/rho_max")
        for name in ("dual_gap_termination", "direction_termination"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        if self.dual_gap_termination and self.eps_rel >= 1:
            raise ValueError("dual-gap eps_rel must be less than one")
        if (isinstance(self.direction_atol, bool)
                or not math.isfinite(self.direction_atol) or self.direction_atol <= 0):
            raise ValueError("direction_atol must be finite and positive")
        if (isinstance(self.direction_rtol, bool)
                or not math.isfinite(self.direction_rtol) or not 0 <= self.direction_rtol < 1):
            raise ValueError("direction_rtol must be finite and in [0, 1)")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda"):
            raise ValueError("QP supports CPU or CUDA only")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("explicit CUDA requested but unavailable")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError("CUDA device index out of range")


@dataclass(frozen=True)
class QPResult:
    x: torch.Tensor
    y: torch.Tensor
    status: str
    iterations: int
    objective: float
    primal_residual: float
    dual_residual: float
    normal_cone_residual: float
    rho_updates: int = 0
    final_rho: float = 1.0
    dual_objective: float | None = None
    duality_gap: float | None = None
    infeasibility_direction: torch.Tensor | None = None
    unbounded_direction: torch.Tensor | None = None
    feasible_base: torch.Tensor | None = None
    termination_evidence: NumericalEvidence | None = None


def _norm(x: torch.Tensor) -> float:
    return float(x.abs().max()) if x.numel() else 0.0


def _balanced_rho(
    rho: float, primal: float, dual: float,
    primal_scale: float, dual_scale: float, config: QPConfig,
) -> float:
    """Balance normalized KKT residuals without overflow or a zero division."""
    if not all(math.isfinite(value) for value in (primal, dual, primal_scale, dual_scale)):
        raise RuntimeError("adaptive rho residuals/scales must be finite")
    if primal == dual == 0:
        return rho
    if primal == 0:
        candidate = config.rho_min
    elif dual == 0:
        candidate = config.rho_max
    else:
        # Log arithmetic bounds even extreme finite ratios before exponentiation.
        log_candidate = math.log(rho) + 0.5 * (
            math.log(primal) - math.log(max(primal_scale, config.eps_abs))
            - math.log(dual) + math.log(max(dual_scale, config.eps_abs))
        )
        candidate = math.exp(min(
            math.log(config.rho_max), max(math.log(config.rho_min), log_candidate),
        ))
        candidate = min(config.rho_max, max(config.rho_min, candidate))
    change = abs(math.log(candidate) - math.log(rho))
    return candidate if change > math.log(config.adaptive_rho_tolerance) else rho


class PreparedQP:
    """Validated, device-resident workspace. Each solve starts from zero."""

    def __init__(self, problem: QPProblem, config: QPConfig = QPConfig()):
        config.validate()
        self.config = config
        tensors = (problem.P, problem.q, problem.A, problem.l, problem.u)
        if not all(isinstance(tensor, torch.Tensor) for tensor in tensors):
            raise TypeError("QP inputs must be tensors")
        if any(tensor.layout != torch.strided or tensor.is_complex() for tensor in tensors):
            raise ValueError("QP inputs must be real dense tensors")
        P, q, A, l, u = [
            tensor.detach().to(device=config.device, dtype=config.dtype).clone()
            for tensor in tensors
        ]
        if q.ndim != 1 or not q.numel():
            raise ValueError("q must be a nonempty vector")
        n = q.numel()
        if P.shape != (n, n) or A.ndim != 2 or A.shape[1] != n:
            raise ValueError("P/A dimensions must match q")
        if l.shape != (A.shape[0],) or u.shape != l.shape:
            raise ValueError("constraint bounds must match A rows")
        if not all(bool(torch.isfinite(t).all()) for t in (P, q, A)):
            raise ValueError("P/q/A must be finite")
        if bool(torch.isnan(l).any() | torch.isnan(u).any()):
            raise ValueError("bounds cannot contain NaN")
        if bool(torch.isposinf(l).any() | torch.isneginf(u).any()):
            raise ValueError("lower +inf or upper -inf is invalid")
        if bool((l > u).any()):
            raise ValueError("lower bound exceeds upper bound")
        symmetry_tol = 20 * torch.finfo(P.dtype).eps * max(1.0, _norm(P))
        if _norm(P - P.T) > symmetry_tol:
            raise ValueError("P must be symmetric")
        P = (P + P.T) * 0.5
        # Validation is deliberately included in cold end-to-end timing.
        eigen_min = float(torch.linalg.eigvalsh(P).min())
        if eigen_min < -symmetry_tol:
            raise ValueError("P must be positive semidefinite")
        self.P, self.q, self.A, self.l, self.u = P, q, A, l, u
        if config.dual_gap_termination or config.direction_termination:
            # Preserve pre-cast, pre-symmetrization data for independent CPU replay.
            self.original_problem = QPProblem(*[
                tensor.detach().to(device="cpu", dtype=torch.float64).clone()
                for tensor in tensors
            ])
        gram = A.T @ A
        system = P + config.rho * gram
        system.diagonal().add_(config.sigma)
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info) != 0:
            raise RuntimeError("QP linear system factorization failed")
        if config.adaptive_rho:
            if not bool(torch.isfinite(system).all() & torch.isfinite(factor).all()):
                raise RuntimeError("adaptive QP initial factorization is nonfinite")
            self.gram = gram
        self.factor = factor

    def _termination_check(
        self, kind: str, iteration: int, rho: float, *vectors: torch.Tensor,
    ) -> NumericalEvidence:
        from .certificates import (
            check_primal_infeasibility, check_qp_candidate, check_unbounded_direction,
        )

        c = self.config
        host = [v.detach().to(device="cpu", dtype=torch.float64) for v in vectors]
        try:
            if kind == "gap":
                return check_qp_candidate(
                    self.original_problem, *host, atol=c.eps_abs, rtol=c.eps_rel,
                )
            if kind == "infeasible":
                return check_primal_infeasibility(
                    self.original_problem, host[0],
                    atol=c.direction_atol, rtol=c.direction_rtol,
                )
            if kind == "unbounded":
                return check_unbounded_direction(
                    self.original_problem, host[0], feasible_x=host[1],
                    atol=c.direction_atol, rtol=c.direction_rtol,
                )
            raise ValueError(f"unknown termination check: {kind}")
        except (ValueError, RuntimeError, ArithmeticError) as error:
            raise RuntimeError(
                f"QP termination check failed kind={kind} iteration={iteration} rho={rho!r}",
            ) from error

    def _factorize_rho(self, rho: float, iteration: int) -> torch.Tensor:
        context = f"adaptive rho factorization failed iteration={iteration} rho={rho!r}"
        system = self.P + rho * self.gram
        system.diagonal().add_(self.config.sigma)
        if not bool(torch.isfinite(system).all()):
            raise RuntimeError(f"{context}: nonfinite system")
        try:
            factor, info = torch.linalg.cholesky_ex(system)
        except RuntimeError as error:
            raise RuntimeError(context) from error
        if int(info) != 0 or not bool(torch.isfinite(factor).all()):
            raise RuntimeError(f"{context}: info={int(info)} or nonfinite factor")
        return factor

    @torch.no_grad()
    def solve(self) -> QPResult:
        c = self.config
        P, q, A, l, u = self.P, self.q, self.A, self.l, self.u
        x = torch.zeros_like(q)
        z = torch.minimum(torch.maximum(torch.zeros_like(l), l), u)
        y = torch.zeros_like(l)
        status = "max_iterations"
        primal = dual = normal = math.inf
        rho, factor, rho_updates = c.rho, self.factor, 0
        evidence = None
        infeasibility_direction = unbounded_direction = feasible_base = None
        if c.direction_termination:
            checkpoint_x, checkpoint_y = x.clone(), y.clone()
        for iteration in range(1, c.max_iterations + 1):
            rhs = c.sigma * x - q + A.T @ (rho * z - y)
            if c.linear_solver == "triangular":
                forward = torch.linalg.solve_triangular(factor, rhs[:, None], upper=False)
                x = torch.linalg.solve_triangular(factor.T, forward, upper=True).squeeze(1)
            else:
                x = torch.cholesky_solve(rhs[:, None], factor).squeeze(1)
            Ax = A @ x
            relaxed = c.relaxation * Ax + (1 - c.relaxation) * z
            z = torch.minimum(torch.maximum(relaxed + y / rho, l), u)
            y = y + rho * (relaxed - z)
            check = iteration % c.check_interval == 0 or iteration == c.max_iterations
            adapt = c.adaptive_rho and l.numel() > 0 and iteration % c.adaptive_rho_interval == 0
            if check or adapt:
                if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
                    if c.dual_gap_termination or c.direction_termination:
                        raise RuntimeError(
                            f"QP nonfinite termination iterate iteration={iteration} rho={rho!r}",
                        )
                    status = "numerical_error"
                    break
                Px, ATy = P @ x, A.T @ y
                if not all(bool(torch.isfinite(t).all()) for t in (Ax, z, Px, ATy)):
                    raise RuntimeError(f"QP nonfinite residual input iteration={iteration} rho={rho!r}")
                primal = _norm(Ax - z)
                dual = _norm(Px + q + ATy)
                normal = _norm(z - torch.minimum(torch.maximum(z + y, l), u))
                primal_tol = c.eps_abs + c.eps_rel * max(_norm(Ax), _norm(z))
                dual_tol = c.eps_abs + c.eps_rel * max(_norm(Px), _norm(q), _norm(ATy))
                if not all(math.isfinite(value) for value in (
                    primal, dual, normal, primal_tol, dual_tol,
                )):
                    raise RuntimeError(
                        f"QP nonfinite residual/tolerance iteration={iteration} rho={rho!r}",
                    )
                if check and c.dual_gap_termination:
                    evidence = self._termination_check("gap", iteration, rho, x, y)
                if check and primal <= primal_tol and dual <= dual_tol and normal <= primal_tol:
                    if not c.dual_gap_termination or evidence.passed:
                        status = "solved"
                        break
                if check and c.direction_termination:
                    # Checkpoint differences average ADMM increments; y is unscaled
                    # even across adaptive-rho updates. No factor or z is checked.
                    dx, dy = x - checkpoint_x, y - checkpoint_y
                    checkpoint_x, checkpoint_y = x.clone(), y.clone()
                    farkas = self._termination_check("infeasible", iteration, rho, dy)
                    recession = self._termination_check("unbounded", iteration, rho, dx, x)
                    # Conflicting tolerance-level evidence is ambiguous, not a status.
                    if farkas.passed != recession.passed:
                        if farkas.passed:
                            status, evidence = "infeasible", farkas
                            v = dy.detach().to(device="cpu", dtype=torch.float64)
                            infeasibility_direction = v / _norm(v)
                        else:
                            status, evidence = "unbounded", recession
                            d = dx.detach().to(device="cpu", dtype=torch.float64)
                            unbounded_direction = d / _norm(d)
                            feasible_base = x.detach().to(device="cpu", dtype=torch.float64).clone()
                        break
                if adapt and iteration < c.max_iterations:
                    new_rho = _balanced_rho(
                        rho, primal, dual, max(_norm(Ax), _norm(z)),
                        max(_norm(Px), _norm(q), _norm(ATy)), c,
                    )
                    if new_rho != rho:
                        factor = self._factorize_rho(new_rho, iteration)
                        # y is the unscaled Lagrange multiplier, not y/rho.
                        rho = new_rho
                        rho_updates += 1
        objective = float(0.5 * x @ (P @ x) + q @ x)
        if not math.isfinite(objective):
            raise RuntimeError(f"QP nonfinite objective iteration={iteration} rho={rho!r}")
        return QPResult(
            x=x, y=y, status=status, iterations=iteration, objective=objective,
            primal_residual=primal, dual_residual=dual, normal_cone_residual=normal,
            rho_updates=rho_updates, final_rho=rho,
            dual_objective=evidence.metrics.get("dual_objective") if evidence else None,
            duality_gap=evidence.metrics.get("duality_gap") if evidence else None,
            infeasibility_direction=infeasibility_direction,
            unbounded_direction=unbounded_direction, feasible_base=feasible_base,
            termination_evidence=evidence,
        )


def prepare_qp(problem: QPProblem, config: QPConfig = QPConfig()) -> PreparedQP:
    return PreparedQP(problem, config)


def solve_qp(problem: QPProblem, config: QPConfig = QPConfig()) -> QPResult:
    return prepare_qp(problem, config).solve()
