from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import torch

from .constraints import LinearConstraints
from .model import PortfolioProblem
from .projection import project_bounded_simplex_unchecked
from .solver import objective


@dataclass(frozen=True)
class AdmmConfig:
    iterations: int = 200
    tolerance: float = 1e-6
    check_interval: int = 10
    rho: float = 1.0
    bisection_steps: int = 24
    cholesky_jitter: float = 0.0
    row_scaling: bool = False
    adaptive_rho: bool = False
    adaptive_rho_interval: int = 25
    adaptive_rho_mu: float = 10.0
    adaptive_rho_tau: float = 2.0
    adaptive_rho_increase: float = 2.0
    adaptive_rho_decrease: float = 2.0
    rho_min: float = 1e-6
    rho_max: float = 1e6
    rho_weight_scale: float = 1.0
    rho_equality_scale: float = 1.0
    rho_inequality_scale: float = 1.0

    def __post_init__(self) -> None:
        if type(self.row_scaling) is not bool:
            raise ValueError("row_scaling must be bool")
        if type(self.adaptive_rho) is not bool:
            raise ValueError("adaptive_rho must be bool")
        if (
            self.iterations <= 0
            or self.check_interval <= 0
            or self.bisection_steps <= 0
        ):
            raise ValueError("iteration settings must be positive")
        if self.adaptive_rho_interval <= 0:
            raise ValueError("adaptive_rho_interval must be positive")
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("tolerance must be finite and nonnegative")
        if not math.isfinite(self.rho) or self.rho <= 0:
            raise ValueError("rho must be finite and positive")
        if (
            not math.isfinite(self.rho_weight_scale)
            or self.rho_weight_scale <= 0
        ):
            raise ValueError("rho_weight_scale must be finite and positive")
        if (
            not math.isfinite(self.rho_equality_scale)
            or self.rho_equality_scale <= 0
        ):
            raise ValueError("rho_equality_scale must be finite and positive")
        if (
            not math.isfinite(self.rho_inequality_scale)
            or self.rho_inequality_scale <= 0
        ):
            raise ValueError("rho_inequality_scale must be finite and positive")
        if not math.isfinite(self.cholesky_jitter) or self.cholesky_jitter < 0:
            raise ValueError("cholesky_jitter must be finite and nonnegative")
        if (
            not math.isfinite(self.adaptive_rho_mu)
            or self.adaptive_rho_mu <= 1
        ):
            raise ValueError("adaptive_rho_mu must be finite and greater than 1")
        if not math.isfinite(self.adaptive_rho_tau) or self.adaptive_rho_tau <= 1:
            raise ValueError("adaptive_rho_tau must be finite and greater than 1")
        if (
            not math.isfinite(self.adaptive_rho_increase)
            or self.adaptive_rho_increase <= 1
        ):
            raise ValueError("adaptive_rho_increase must be finite and greater than 1")
        if (
            not math.isfinite(self.adaptive_rho_decrease)
            or self.adaptive_rho_decrease <= 1
        ):
            raise ValueError("adaptive_rho_decrease must be finite and greater than 1")
        if (
            not math.isfinite(self.rho_min)
            or not math.isfinite(self.rho_max)
            or self.rho_min <= 0
            or self.rho_max <= 0
        ):
            raise ValueError("rho_min and rho_max must be finite positive values")
        if self.rho_min > self.rho_max:
            raise ValueError("rho_min must be less than or equal to rho_max")
        if self.rho < self.rho_min or self.rho > self.rho_max:
            raise ValueError("rho must lie within rho_min/rho_max bounds")


@dataclass(frozen=True)
class AdmmSolveResult:
    weights: torch.Tensor
    objective: torch.Tensor
    primal_residual: float
    dual_residual: float
    stationarity_residual: float
    max_constraint_violation: float
    iterations: int
    converged: bool
    status: str
    elapsed_seconds: float
    device: str
    rho_update_count: int = 0
    final_rho: float = 1.0
    rho_updates: int = 0
    block_residuals: dict[str, float] = field(default_factory=dict)
    block_status: dict[str, str] = field(default_factory=dict)
    final_block_rho: dict[str, float] = field(default_factory=dict)


def _stack_constraints(
    constraints: LinearConstraints,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrices: list[torch.Tensor] = []
    lower_bounds: list[torch.Tensor] = []
    upper_bounds: list[torch.Tensor] = []
    if constraints.A_eq is not None:
        matrices.append(constraints.A_eq)
        lower_bounds.append(constraints.b_eq)
        upper_bounds.append(constraints.b_eq)
    if constraints.A_ineq is not None:
        matrices.append(constraints.A_ineq)
        lower_bounds.append(constraints.lower)
        upper_bounds.append(constraints.upper)
    return (
        torch.cat(matrices, dim=0),
        torch.cat(lower_bounds, dim=0),
        torch.cat(upper_bounds, dim=0),
    )


def _require_finite(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")


def _broadcast_rows(
    tensor: torch.Tensor,
    batch_size: int,
    assets: int,
    name: str,
) -> torch.Tensor:
    if tensor.ndim == 1:
        if tensor.shape[0] != assets:
            raise ValueError(f"{name} must have shape [assets]")
        return tensor.unsqueeze(0).expand(batch_size, -1)
    if tensor.ndim == 2:
        if tensor.shape[1] != assets:
            raise ValueError(f"{name} must have shape [batch, assets]")
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, -1)
    raise ValueError(
        f"{name} must have shape [assets] or [batch, assets], got {tuple(tensor.shape)}"
    )


def _batch_norm(values: torch.Tensor) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros((values.shape[0],))
    return torch.linalg.vector_norm(values, ord=2, dim=1)


def _scale_constraint_rows(
    matrix: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    enabled: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if matrix.shape[0] == 0:
        scale = matrix.new_empty((0,))
        return matrix, lower, upper, scale

    if not enabled:
        scale = torch.ones((matrix.shape[0],), device=matrix.device, dtype=matrix.dtype)
        return matrix, lower, upper, scale

    row_norms = torch.linalg.vector_norm(matrix, ord=2, dim=1)
    if not torch.isfinite(row_norms).all().item():
        raise ValueError("row scaling requires finite general linear constraint row norms")
    zero_rows = row_norms == 0
    if torch.any(zero_rows):
        sample = zero_rows.nonzero(as_tuple=False).flatten()[:8].detach().cpu().tolist()
        raise ValueError(
            "row scaling requires nonzero general linear constraint rows; "
            f"zero row indices sample={sample}"
        )
    scale = row_norms.reciprocal()
    if not torch.isfinite(scale).all().item() or torch.any(scale <= 0):
        raise ValueError("row scaling produced nonfinite or nonpositive scales")
    return (
        matrix * scale.unsqueeze(1),
        lower * scale,
        upper * scale,
        scale,
    )


def _block_rho_values(
    current_rho: float,
    config: AdmmConfig,
) -> dict[str, float]:
    return {
        "weights": current_rho * config.rho_weight_scale,
        "equality": current_rho * config.rho_equality_scale,
        "inequality": current_rho * config.rho_inequality_scale,
    }


def _linear_rho_vector(
    constraints: LinearConstraints,
    matrix: torch.Tensor,
    current_rho: float,
    config: AdmmConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    block_rho = _block_rho_values(current_rho, config)
    row_rhos: list[torch.Tensor] = []
    if constraints.A_eq is not None:
        row_rhos.append(
            matrix.new_full((constraints.A_eq.shape[0],), block_rho["equality"])
        )
    if constraints.A_ineq is not None:
        row_rhos.append(
            matrix.new_full((constraints.A_ineq.shape[0],), block_rho["inequality"])
        )
    if row_rhos:
        return torch.cat(row_rhos, dim=0), block_rho
    return matrix.new_empty((0,)), block_rho


def _max_final_constraint_violation(
    problem: PortfolioProblem,
    matrix: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    budget_violation = (weights.sum(dim=1) - problem.budget).abs().amax()
    box_lower_violation = torch.clamp(problem.lower.unsqueeze(0) - weights, min=0).amax()
    box_upper_violation = torch.clamp(weights - problem.upper.unsqueeze(0), min=0).amax()
    if matrix.shape[0]:
        linear_values = weights @ matrix.T
        linear_violation = torch.maximum(
            torch.clamp(lower - linear_values, min=0),
            torch.clamp(linear_values - upper, min=0),
        ).amax()
    else:
        linear_violation = weights.new_zeros(())
    return torch.stack(
        (
            budget_violation,
            box_lower_violation,
            box_upper_violation,
            linear_violation,
        )
    ).amax()


def _cholesky_with_jitter(
    inner: torch.Tensor,
    configured_jitter: float,
) -> torch.Tensor:
    eye = torch.eye(inner.shape[0], device=inner.device, dtype=inner.dtype)
    scale = inner.diagonal().abs().amax().clamp_min(1)
    jitter_floor = scale * torch.finfo(inner.dtype).eps
    base_jitter = torch.maximum(jitter_floor, inner.new_tensor(configured_jitter))
    attempts = 6
    failures: list[tuple[int, torch.Tensor | None, int]] = []

    for attempt in range(attempts):
        jitter = None
        candidate = inner
        if configured_jitter or attempt:
            multiplier = 10 ** attempt if configured_jitter else 10 ** (attempt - 1)
            jitter = base_jitter * multiplier
            candidate = inner + jitter * eye
        chol, info = torch.linalg.cholesky_ex(candidate, check_errors=False)
        info_max = int(info.max().item())
        if info_max == 0:
            return chol
        failures.append((attempt, jitter, info_max))

    failure_details = [
        {
            "attempt": attempt,
            "jitter": 0.0 if jitter is None else float(jitter.detach().cpu()),
            "info": info_max,
        }
        for attempt, jitter, info_max in failures
    ]
    raise ValueError(
        "x-step Woodbury Cholesky factorization failed after jitter retries; "
        f"attempts={failure_details}"
    )


def _build_x_solver(
    problem: PortfolioProblem,
    config: AdmmConfig,
    trade_cost: float,
    matrix: torch.Tensor,
    rho_z: float,
    rho_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    exposures = problem.risk.exposures
    factors = exposures.shape[1]
    rows = matrix.shape[0]

    factor_update = exposures.new_empty((exposures.shape[0], 0))
    if factors:
        factor_scale = math.sqrt(problem.risk_aversion)
        factor_update = factor_scale * exposures * problem.risk.factor_variance.sqrt().unsqueeze(0)

    constraint_update = matrix.new_empty((matrix.shape[1], 0))
    if rows:
        constraint_update = matrix.T * rho_y.sqrt().unsqueeze(0)

    update = torch.cat((factor_update, constraint_update), dim=1)
    diagonal = (
        problem.risk_aversion * problem.risk.specific_variance
        + trade_cost
        + rho_z
    )
    _require_finite("x-step diagonal", diagonal)
    if torch.any(diagonal <= 0):
        raise ValueError("x-step diagonal must be strictly positive")

    if update.shape[1] == 0:
        return diagonal, update, update.new_empty((0, 0))

    inverse_diagonal = diagonal.reciprocal()
    weighted_update = inverse_diagonal.unsqueeze(1) * update
    inner = torch.eye(
        update.shape[1], device=update.device, dtype=update.dtype
    ) + update.T @ weighted_update
    chol = _cholesky_with_jitter(inner, config.cholesky_jitter)
    return diagonal, weighted_update, chol


def _solve_x_batch(
    rhs: torch.Tensor,
    diagonal: torch.Tensor,
    weighted_update: torch.Tensor,
    chol: torch.Tensor,
) -> torch.Tensor:
    if weighted_update.shape[1] == 0:
        return rhs * diagonal.reciprocal().unsqueeze(0)

    inverse_diagonal = diagonal.reciprocal()
    rhs_t = rhs.T
    weighted_rhs = inverse_diagonal.unsqueeze(1) * rhs_t
    coeff = weighted_update.T @ rhs_t
    coeff = torch.cholesky_solve(coeff, chol)
    correction = weighted_update @ coeff
    return (weighted_rhs - correction).T


def _block_residuals_and_status(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    weights: torch.Tensor,
    admm_weight_residual: torch.Tensor,
    admm_linear_residual: torch.Tensor,
    dual_weight_residual: torch.Tensor,
    dual_linear_residual: torch.Tensor,
    stationarity_residual: torch.Tensor,
    tolerance: float,
) -> tuple[dict[str, float], dict[str, str]]:
    residuals: dict[str, float] = {}

    residuals["budget"] = float((weights.sum(dim=1) - problem.budget).abs().amax().item())
    residuals["box"] = float(
        torch.maximum(
            torch.clamp(problem.lower.unsqueeze(0) - weights, min=0),
            torch.clamp(weights - problem.upper.unsqueeze(0), min=0),
        )
        .amax()
        .item()
    )

    if constraints.A_eq is not None:
        if constraints.A_eq.shape[0] == 0:
            residuals["linear_equality"] = 0.0
        else:
            equality = weights @ constraints.A_eq.T - constraints.b_eq
            residuals["linear_equality"] = float(equality.abs().amax().item())
    if constraints.A_ineq is not None:
        if constraints.A_ineq.shape[0] == 0:
            residuals["linear_inequality"] = 0.0
        else:
            values = weights @ constraints.A_ineq.T
            violation = torch.maximum(
                torch.clamp(constraints.lower - values, min=0),
                torch.clamp(values - constraints.upper, min=0),
            )
            residuals["linear_inequality"] = float(violation.amax().item())

    residuals["admm_consensus_weights"] = float(admm_weight_residual.amax().item())
    residuals["admm_consensus_linear"] = float(admm_linear_residual.amax().item())
    residuals["dual_weights"] = float(dual_weight_residual.amax().item())
    residuals["dual_linear"] = float(dual_linear_residual.amax().item())
    residuals["stationarity"] = float(stationarity_residual.amax().item())

    block_status = {
        name: "ok" if value <= tolerance else "violated"
        for name, value in residuals.items()
    }
    if constraints.A_eq is None:
        block_status["linear_equality"] = "not_applicable"
    if constraints.A_ineq is None:
        block_status["linear_inequality"] = "not_applicable"
    return residuals, block_status


@torch.no_grad()
def solve_admm(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    config: AdmmConfig = AdmmConfig(),
) -> AdmmSolveResult:
    assets = problem.risk.assets
    if constraints.num_variables != assets:
        raise ValueError("constraints and problem have inconsistent asset counts")
    if constraints.device != problem.alpha.device:
        raise ValueError("constraints and problem must be on the same device")
    if constraints.dtype != problem.alpha.dtype:
        raise ValueError("constraints and problem must use the same dtype")

    alpha = problem.batched_alpha
    if alpha.device != problem.risk.exposures.device or alpha.dtype != problem.risk.exposures.dtype:
        raise ValueError("problem tensors must share device and dtype")
    if alpha.shape[-1] != assets:
        raise ValueError("alpha and risk model asset counts differ")

    tensors_to_check = [
        alpha,
        problem.risk.exposures,
        problem.risk.factor_variance,
        problem.risk.specific_variance,
        problem.lower,
        problem.upper,
        constraints.A_eq,
        constraints.b_eq,
        constraints.A_ineq,
        constraints.lower,
        constraints.upper,
    ]
    if problem.current is not None:
        tensors_to_check.append(problem.current)
    for tensor in tensors_to_check:
        if tensor is not None:
            if not tensor.is_floating_point():
                raise TypeError("all problem and constraint tensors must be floating-point")
            _require_finite("problem or constraint tensor", tensor)

    original_matrix, original_lower, original_upper = _stack_constraints(constraints)
    matrix, lower, upper, row_scale = _scale_constraint_rows(
        original_matrix,
        original_lower,
        original_upper,
        config.row_scaling,
    )
    batch_size = alpha.shape[0]
    if problem.current is not None:
        current_rows = problem.current.shape[0] if problem.current.ndim == 2 else 1
        if current_rows not in (1, batch_size):
            if batch_size == 1:
                batch_size = current_rows
            else:
                raise ValueError("current holdings batch size is incompatible with alpha")
    current = None
    if problem.current is not None:
        current = _broadcast_rows(problem.current, batch_size, assets, "current")

    if alpha.shape[0] not in (1, batch_size):
        raise ValueError("alpha batch size is incompatible with current holdings")
    if alpha.shape[0] == 1 and batch_size > 1:
        alpha = alpha.expand(batch_size, -1)

    trade_cost = problem.trade_cost if current is not None else 0.0
    if alpha.is_cuda:
        torch.cuda.synchronize(alpha.device)
    start = time.perf_counter()
    current_rho = config.rho
    rho_y, block_rho = _linear_rho_vector(constraints, matrix, current_rho, config)
    rho_z = block_rho["weights"]
    rho_updates = 0
    diagonal, weighted_update, chol = _build_x_solver(
        problem, config, trade_cost, matrix, rho_z, rho_y
    )

    initial = (
        current
        if current is not None
        else alpha.new_full((batch_size, assets), problem.budget / assets)
    )
    x = project_bounded_simplex_unchecked(
        initial,
        problem.lower,
        problem.upper,
        problem.budget,
        config.bisection_steps,
    )
    z = x.clone()
    y = torch.clamp(x @ matrix.T, min=lower, max=upper)
    dual_z = alpha.new_zeros((batch_size, assets))
    dual_y = alpha.new_zeros((batch_size, matrix.shape[0]))

    converged = False
    iterations = config.iterations
    primal_residual_tensor = alpha.new_tensor(float("inf"))
    dual_residual_tensor = alpha.new_tensor(float("inf"))
    stationarity_tensor = alpha.new_tensor(float("inf"))
    admm_weight_residual_tensor = alpha.new_tensor(float("inf"))
    admm_linear_residual_tensor = alpha.new_tensor(float("inf"))
    dual_weight_residual_tensor = alpha.new_tensor(float("inf"))
    dual_linear_residual_tensor = alpha.new_tensor(float("inf"))

    for iteration in range(1, config.iterations + 1):
        z_prev = z
        y_prev = y

        rhs = alpha + rho_z * (z - dual_z)
        if current is not None and trade_cost:
            rhs = rhs + trade_cost * current
        if matrix.shape[0]:
            rhs = rhs + ((y - dual_y) * rho_y.unsqueeze(0)) @ matrix

        x = _solve_x_batch(rhs, diagonal, weighted_update, chol)
        z = project_bounded_simplex_unchecked(
            x + dual_z,
            problem.lower,
            problem.upper,
            problem.budget,
            config.bisection_steps,
        )
        ax = x @ matrix.T
        y = torch.clamp(ax + dual_y, min=lower, max=upper)
        dual_z = dual_z + x - z
        dual_y = dual_y + ax - y

        needs_residuals = (
            bool(config.tolerance) and iteration % config.check_interval == 0
        ) or (
            config.adaptive_rho
            and iteration % config.adaptive_rho_interval == 0
        )
        if needs_residuals:
            gradient = problem.risk_aversion * problem.risk.matvec(x) - alpha
            if current is not None and trade_cost:
                gradient = gradient + trade_cost * (x - current)
            admm_weight_residual_tensor = _batch_norm(x - z)
            admm_linear_residual_tensor = _batch_norm((ax - y) / row_scale.unsqueeze(0))
            primal_residual_tensor = torch.maximum(
                admm_weight_residual_tensor,
                admm_linear_residual_tensor,
            )
            dual_weight_residual_tensor = rho_z * _batch_norm(z - z_prev)
            dual_linear_residual_tensor = _batch_norm(
                ((y - y_prev) * rho_y.unsqueeze(0)) @ matrix
            )
            dual_residual_tensor = torch.sqrt(
                dual_weight_residual_tensor.square()
                + dual_linear_residual_tensor.square()
            )
            stationarity_tensor = _batch_norm(
                gradient + rho_z * dual_z + (dual_y * rho_y.unsqueeze(0)) @ matrix
            )
            original_violation_tensor = _max_final_constraint_violation(
                problem,
                original_matrix,
                original_lower,
                original_upper,
                z,
            )

            if (
                config.adaptive_rho
                and iteration % config.adaptive_rho_interval == 0
            ):
                primal_max = float(primal_residual_tensor.amax().item())
                dual_max = float(dual_residual_tensor.amax().item())
                rho_increase = (
                    config.adaptive_rho_tau
                    if config.adaptive_rho_increase == 2.0
                    else config.adaptive_rho_increase
                )
                rho_decrease = (
                    config.adaptive_rho_tau
                    if config.adaptive_rho_decrease == 2.0
                    else config.adaptive_rho_decrease
                )
                next_rho = current_rho
                if primal_max > config.adaptive_rho_mu * dual_max:
                    next_rho = min(
                        config.rho_max,
                        current_rho * rho_increase,
                    )
                elif dual_max > config.adaptive_rho_mu * primal_max:
                    next_rho = max(
                        config.rho_min,
                        current_rho / rho_decrease,
                    )
                if next_rho != current_rho:
                    dual_scale = current_rho / next_rho
                    dual_z = dual_z * dual_scale
                    dual_y = dual_y * dual_scale
                    current_rho = next_rho
                    rho_y, block_rho = _linear_rho_vector(
                        constraints,
                        matrix,
                        current_rho,
                        config,
                    )
                    rho_z = block_rho["weights"]
                    diagonal, weighted_update, chol = _build_x_solver(
                        problem, config, trade_cost, matrix, rho_z, rho_y
                    )
                    rho_updates += 1
                    dual_weight_residual_tensor = rho_z * _batch_norm(z - z_prev)
                    dual_linear_residual_tensor = _batch_norm(
                        ((y - y_prev) * rho_y.unsqueeze(0)) @ matrix
                    )
                    dual_residual_tensor = torch.sqrt(
                        dual_weight_residual_tensor.square()
                        + dual_linear_residual_tensor.square()
                    )
                    stationarity_tensor = _batch_norm(
                        gradient + rho_z * dual_z + (dual_y * rho_y.unsqueeze(0)) @ matrix
                    )

        if config.tolerance and iteration % config.check_interval == 0:
            residual_max_tensor = torch.stack(
                (
                    primal_residual_tensor.amax(),
                    dual_residual_tensor.amax(),
                    stationarity_tensor.amax(),
                    original_violation_tensor,
                )
            ).amax()
            if float(residual_max_tensor.item()) <= config.tolerance:
                converged = True
                iterations = iteration
                break

    if not converged:
        gradient = problem.risk_aversion * problem.risk.matvec(x) - alpha
        if current is not None and trade_cost:
            gradient = gradient + trade_cost * (x - current)
        admm_weight_residual_tensor = _batch_norm(x - z)
        admm_linear_residual_tensor = _batch_norm(
            (x @ matrix.T - y) / row_scale.unsqueeze(0)
        )
        primal_residual_tensor = torch.maximum(
            admm_weight_residual_tensor,
            admm_linear_residual_tensor,
        )
        dual_weight_residual_tensor = rho_z * _batch_norm(z - z_prev)
        dual_linear_residual_tensor = _batch_norm(
            ((y - y_prev) * rho_y.unsqueeze(0)) @ matrix
        )
        dual_residual_tensor = torch.sqrt(
            dual_weight_residual_tensor.square()
            + dual_linear_residual_tensor.square()
        )
        stationarity_tensor = _batch_norm(
            gradient + rho_z * dual_z + (dual_y * rho_y.unsqueeze(0)) @ matrix
        )

    objective_tensor = objective(problem, z)
    max_constraint_violation_tensor = _max_final_constraint_violation(
        problem,
        original_matrix,
        original_lower,
        original_upper,
        z,
    )
    block_residuals, block_status = _block_residuals_and_status(
        problem,
        constraints,
        z,
        admm_weight_residual_tensor,
        admm_linear_residual_tensor,
        dual_weight_residual_tensor,
        dual_linear_residual_tensor,
        stationarity_tensor,
        config.tolerance,
    )
    final_block_rho = _block_rho_values(current_rho, config)
    if z.is_cuda:
        torch.cuda.synchronize(z.device)
    elapsed = time.perf_counter() - start

    return AdmmSolveResult(
        weights=z,
        objective=objective_tensor,
        primal_residual=float(primal_residual_tensor.max().item()),
        dual_residual=float(dual_residual_tensor.max().item()),
        stationarity_residual=float(stationarity_tensor.max().item()),
        max_constraint_violation=float(max_constraint_violation_tensor.item()),
        iterations=iterations,
        converged=converged,
        status="converged" if converged else "max_iterations",
        elapsed_seconds=elapsed,
        device=str(z.device),
        rho_update_count=rho_updates,
        rho_updates=rho_updates,
        final_rho=current_rho,
        block_residuals=block_residuals,
        block_status=block_status,
        final_block_rho=final_block_rho,
    )
