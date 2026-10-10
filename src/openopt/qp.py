"""Experimental domain-neutral dense convex quadratic programming.

Minimize 0.5*x.T*P*x + q.T*x subject to l <= A*x <= u. ADMM uses a
regularized positive-definite system; this module does not certify infeasibility
or unboundedness. Explicit devices never fall back. No portfolio assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


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


def _norm(x: torch.Tensor) -> float:
    return float(x.abs().max()) if x.numel() else 0.0


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
        system = P + config.rho * (A.T @ A)
        system.diagonal().add_(config.sigma)
        factor, info = torch.linalg.cholesky_ex(system)
        if int(info) != 0:
            raise RuntimeError("QP linear system factorization failed")
        self.factor = factor

    @torch.no_grad()
    def solve(self) -> QPResult:
        c = self.config
        P, q, A, l, u = self.P, self.q, self.A, self.l, self.u
        x = torch.zeros_like(q)
        z = torch.minimum(torch.maximum(torch.zeros_like(l), l), u)
        y = torch.zeros_like(l)
        status = "max_iterations"
        primal = dual = normal = math.inf
        for iteration in range(1, c.max_iterations + 1):
            rhs = c.sigma * x - q + A.T @ (c.rho * z - y)
            if c.linear_solver == "triangular":
                forward = torch.linalg.solve_triangular(self.factor, rhs[:, None], upper=False)
                x = torch.linalg.solve_triangular(self.factor.T, forward, upper=True).squeeze(1)
            else:
                x = torch.cholesky_solve(rhs[:, None], self.factor).squeeze(1)
            Ax = A @ x
            relaxed = c.relaxation * Ax + (1 - c.relaxation) * z
            z = torch.minimum(torch.maximum(relaxed + y / c.rho, l), u)
            y = y + c.rho * (relaxed - z)
            if iteration % c.check_interval == 0 or iteration == c.max_iterations:
                if not bool(torch.isfinite(x).all() & torch.isfinite(y).all()):
                    status = "numerical_error"
                    break
                Px, ATy = P @ x, A.T @ y
                primal = _norm(Ax - z)
                dual = _norm(Px + q + ATy)
                normal = _norm(z - torch.minimum(torch.maximum(z + y, l), u))
                primal_tol = c.eps_abs + c.eps_rel * max(_norm(Ax), _norm(z))
                dual_tol = c.eps_abs + c.eps_rel * max(_norm(Px), _norm(q), _norm(ATy))
                if primal <= primal_tol and dual <= dual_tol and normal <= primal_tol:
                    status = "solved"
                    break
        objective = float(0.5 * x @ (P @ x) + q @ x)
        return QPResult(
            x=x, y=y, status=status, iterations=iteration, objective=objective,
            primal_residual=primal, dual_residual=dual, normal_cone_residual=normal,
        )


def prepare_qp(problem: QPProblem, config: QPConfig = QPConfig()) -> PreparedQP:
    return PreparedQP(problem, config)


def solve_qp(problem: QPProblem, config: QPConfig = QPConfig()) -> QPResult:
    return prepare_qp(problem, config).solve()
