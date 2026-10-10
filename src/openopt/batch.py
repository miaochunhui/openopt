"""Experimental same-shape dense QPs using batched resident tensor operations.

This opt-in module does not change the single-problem dense solver. Preparation
owns inputs and includes spectral PSD validation and batched factorization.
Termination is approximate and per item; no infeasible/unbounded certificate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .qp import QPConfig, QPProblem


@dataclass(frozen=True)
class BatchQPProblem:
    P: torch.Tensor
    q: torch.Tensor
    A: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor

    @classmethod
    def stack(cls, problems: Sequence[QPProblem]) -> BatchQPProblem:
        """Stack a nonempty same-shape sequence; no device/dtype broadcasting."""
        if not isinstance(problems, Sequence) or not len(problems):
            raise ValueError("stack requires a nonempty sequence of QPProblem")
        fields = ("P", "q", "A", "l", "u")
        for index, problem in enumerate(problems):
            if not isinstance(problem, QPProblem):
                raise TypeError(f"stack item {index} must be QPProblem")
            tensors = tuple(getattr(problem, field) for field in fields)
            _validate_tensor_types(tensors)
            if problem.q.ndim != 1 or not problem.q.numel():
                raise ValueError(f"stack item {index}: q must be a nonempty vector")
            n = problem.q.numel()
            if problem.P.shape != (n, n) or problem.A.ndim != 2 or problem.A.shape[1] != n:
                raise ValueError(f"stack item {index}: P/A dimensions must match q")
            if problem.l.shape != (problem.A.shape[0],) or problem.u.shape != problem.l.shape:
                raise ValueError(f"stack item {index}: bounds must match A rows")
            if any(t.device != problems[0].P.device for t in tensors):
                raise ValueError(f"stack item {index}: all inputs must share a device")
            if any(t.dtype != problems[0].P.dtype for t in tensors):
                raise ValueError(f"stack item {index}: all inputs must share a dtype")
            if any(t.shape != getattr(problems[0], field).shape for field, t in zip(fields, tensors)):
                raise ValueError(f"stack item {index}: all problems must have the same shape")
        return cls(*(torch.stack([getattr(p, field) for p in problems]) for field in fields))


@dataclass(frozen=True)
class BatchQPResult:
    """Vectors/metrics/iterations stay on the requested device; status is host."""

    x: torch.Tensor
    y: torch.Tensor
    status: tuple[str, ...]
    iterations: torch.Tensor
    objective: torch.Tensor
    primal_residual: torch.Tensor
    dual_residual: torch.Tensor
    normal_cone_residual: torch.Tensor


def _validate_tensor_types(tensors: tuple[torch.Tensor, ...]) -> None:
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise TypeError("batch QP inputs must be tensors")
    if any(t.layout != torch.strided or t.is_complex() for t in tensors):
        raise ValueError("batch QP inputs must be real dense tensors")
    if any(t.device.type not in ("cpu", "cuda") for t in tensors):
        raise ValueError("batch QP input tensors must be on CPU or CUDA")


def _reject_items(mask: torch.Tensor, message: str, error_type: type[Exception] = ValueError) -> None:
    if bool(mask.any()):
        indices = mask.nonzero().flatten().cpu().tolist()
        raise error_type(f"{message}; batch items={indices}")


def _norm(x: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] == 0:
        return x.new_zeros(x.shape[0])
    return x.abs().amax(dim=-1)


def _mv(matrix: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    return torch.bmm(matrix, vector.unsqueeze(-1)).squeeze(-1)


class PreparedBatchQP:
    """Owned resident factors; each solve has independent zero-start state."""

    @torch.no_grad()
    def __init__(self, problem: BatchQPProblem, config: QPConfig = QPConfig()):
        config.validate()
        if getattr(config, "adaptive_rho", False):
            raise ValueError("batch QP supports fixed rho only; adaptive_rho must be False")
        if getattr(config, "dual_gap_termination", False) or getattr(config, "direction_termination", False):
            raise ValueError("batch QP does not support dual-gap or direction termination")
        if not isinstance(problem, BatchQPProblem):
            raise TypeError("problem must be BatchQPProblem")
        tensors = (problem.P, problem.q, problem.A, problem.l, problem.u)
        _validate_tensor_types(tensors)
        if problem.q.ndim != 2 or not all(problem.q.shape):
            raise ValueError("q must have nonempty shape (batch, variables)")
        batch, n = problem.q.shape
        if problem.P.shape != (batch, n, n):
            raise ValueError("P must have shape (batch, variables, variables)")
        if problem.A.ndim != 3 or problem.A.shape[0] != batch or problem.A.shape[2] != n:
            raise ValueError("A must have shape (batch, constraints, variables)")
        if problem.l.shape != (batch, problem.A.shape[1]) or problem.u.shape != problem.l.shape:
            raise ValueError("bounds must have shape (batch, constraints)")
        P, q, A, l, u = (
            t.detach().to(device=config.device, dtype=config.dtype).clone()
            for t in tensors
        )
        for name, tensor in (("P", P), ("q", q), ("A", A)):
            _reject_items(~torch.isfinite(tensor).reshape(batch, -1).all(dim=-1),
                          f"{name} must be finite")
        _reject_items(torch.isnan(l).any(dim=-1) | torch.isnan(u).any(dim=-1),
                      "bounds cannot contain NaN")
        _reject_items(torch.isposinf(l).any(dim=-1) | torch.isneginf(u).any(dim=-1),
                      "lower +inf or upper -inf is invalid")
        _reject_items((l > u).any(dim=-1), "lower bound exceeds upper bound")
        tolerance = (20 * torch.finfo(P.dtype).eps) * _norm(P.flatten(1)).clamp_min(1)
        _reject_items(_norm((P - P.transpose(-1, -2)).flatten(1)) > tolerance,
                      "P must be symmetric")
        P = 0.5 * P + 0.5 * P.transpose(-1, -2)
        # Deliberately part of cold preparation: no unchecked PSD fast path.
        eigen_min = torch.linalg.eigvalsh(P).amin(dim=-1)
        _reject_items(~torch.isfinite(eigen_min), "PSD validation is nonfinite")
        _reject_items(eigen_min < -tolerance, "P must be positive semidefinite")
        AT = A.transpose(-1, -2)
        system = P + config.rho * torch.bmm(AT, A)
        system.diagonal(dim1=-2, dim2=-1).add_(config.sigma)
        _reject_items(~torch.isfinite(system).flatten(1).all(dim=-1),
                      "batch QP linear system is nonfinite", RuntimeError)
        factor, info = torch.linalg.cholesky_ex(system)
        _reject_items((info != 0) | ~torch.isfinite(factor).flatten(1).all(dim=-1),
                      "batch QP linear system factorization failed", RuntimeError)
        self.config = config
        self.P, self.q, self.A, self.l, self.u = P, q, A, l, u
        self.AT, self.factor = AT, factor
        self.device = q.device

    @torch.no_grad()
    def solve(self) -> BatchQPResult:
        c = self.config
        P, q, A, AT, l, u = self.P, self.q, self.A, self.AT, self.l, self.u
        x = torch.zeros_like(q)
        z = torch.minimum(torch.maximum(torch.zeros_like(l), l), u)
        y = torch.zeros_like(l)
        batch = q.shape[0]
        active = torch.ones(batch, dtype=torch.bool, device=self.device)
        codes = torch.zeros(batch, dtype=torch.int8, device=self.device)
        iterations = torch.zeros(batch, dtype=torch.int64, device=self.device)
        primal = q.new_full((batch,), float("inf"))
        dual, normal, objective = primal.clone(), primal.clone(), primal.clone()
        q_norm = _norm(q)
        for iteration in range(1, c.max_iterations + 1):
            rhs = c.sigma * x - q + _mv(AT, c.rho * z - y)
            rhs = torch.where(active[:, None], rhs, 0).unsqueeze(-1)
            if c.linear_solver == "triangular":
                forward = torch.linalg.solve_triangular(self.factor, rhs, upper=False)
                next_x = torch.linalg.solve_triangular(
                    self.factor.transpose(-1, -2), forward, upper=True,
                ).squeeze(-1)
            else:
                next_x = torch.cholesky_solve(rhs, self.factor).squeeze(-1)
            x = torch.where(active[:, None], next_x, x)
            Ax = _mv(A, x)
            relaxed = c.relaxation * Ax + (1 - c.relaxation) * z
            next_z = torch.minimum(torch.maximum(relaxed + y / c.rho, l), u)
            next_y = y + c.rho * (relaxed - next_z)
            z = torch.where(active[:, None], next_z, z)
            y = torch.where(active[:, None], next_y, y)
            if iteration % c.check_interval == 0 or iteration == c.max_iterations:
                Px, ATy = _mv(P, x), _mv(AT, y)
                now_primal = _norm(Ax - z)
                now_dual = _norm(Px + q + ATy)
                now_normal = _norm(z - torch.minimum(torch.maximum(z + y, l), u))
                now_objective = (0.5 * x * Px + q * x).sum(dim=-1)
                primal_tol = c.eps_abs + c.eps_rel * torch.maximum(_norm(Ax), _norm(z))
                dual_tol = c.eps_abs + c.eps_rel * torch.maximum(
                    torch.maximum(_norm(Px), q_norm), _norm(ATy),
                )
                metrics = torch.stack(
                    (now_primal, now_dual, now_normal, now_objective, primal_tol, dual_tol),
                    dim=-1,
                )
                finite = (
                    torch.isfinite(x).all(dim=-1) & torch.isfinite(y).all(dim=-1)
                    & torch.isfinite(z).all(dim=-1) & torch.isfinite(metrics).all(dim=-1)
                )
                solved = finite & (now_primal <= primal_tol) & (now_dual <= dual_tol) & (
                    now_normal <= primal_tol
                )
                primal = torch.where(active, now_primal, primal)
                dual = torch.where(active, now_dual, dual)
                normal = torch.where(active, now_normal, normal)
                objective = torch.where(active, now_objective, objective)
                iterations = torch.where(active, iteration, iterations)
                codes = torch.where(active & ~finite, 2, codes)
                codes = torch.where(active & solved, 1, codes)
                active = active & finite & ~solved
                # One aggregate host check, not a Python solve/check per item.
                if not bool(active.any()):
                    break
        names = ("max_iterations", "solved", "numerical_error")
        status = tuple(names[code] for code in codes.cpu().tolist())
        return BatchQPResult(
            x=x, y=y, status=status, iterations=iterations, objective=objective,
            primal_residual=primal, dual_residual=dual, normal_cone_residual=normal,
        )


def prepare_batch_qp(problem: BatchQPProblem, config: QPConfig = QPConfig()) -> PreparedBatchQP:
    return PreparedBatchQP(problem, config)


def solve_batch_qp(problem: BatchQPProblem, config: QPConfig = QPConfig()) -> BatchQPResult:
    return prepare_batch_qp(problem, config).solve()
