from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import time

import torch

from .constraints import LinearConstraints
from .model import PortfolioProblem
from .projection import project_bounded_simplex_unchecked
from .solver import _lipschitz_bound, objective


@dataclass(frozen=True)
class PrimalDualState:
    weights: torch.Tensor
    dual: torch.Tensor
    alpha: torch.Tensor
    template_signature: str


@dataclass(frozen=True)
class PrimalDualConfig:
    iterations: int = 1000
    tolerance: float = 1e-6
    constraint_tolerance: float = 2e-5
    check_interval: int = 20
    bisection_steps: int = 24
    step_safety: float = 0.95
    warm_start_max_alpha_delta: float | None = None

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.check_interval <= 0 or self.bisection_steps <= 0:
            raise ValueError("iteration settings must be positive")
        if self.tolerance < 0 or self.constraint_tolerance < 0:
            raise ValueError("tolerances must be nonnegative")
        if not 0 < self.step_safety < 1:
            raise ValueError("step_safety must be in (0, 1)")
        if (
            self.warm_start_max_alpha_delta is not None
            and (
                not math.isfinite(self.warm_start_max_alpha_delta)
                or self.warm_start_max_alpha_delta < 0
            )
        ):
            raise ValueError("warm_start_max_alpha_delta must be finite and nonnegative")


@dataclass(frozen=True)
class ConstrainedSolveResult:
    weights: torch.Tensor
    objective: torch.Tensor
    iterations: int
    converged: bool
    max_constraint_violation: float
    elapsed_seconds: float
    device: str
    state: PrimalDualState


def _stack_constraints(
    constraints: LinearConstraints,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    matrices = []
    lower_bounds = []
    upper_bounds = []
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


def _constraint_violation(
    weights: torch.Tensor,
    matrix: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> torch.Tensor:
    values = weights @ matrix.T
    return torch.maximum(
        torch.clamp(lower - values, min=0).amax(),
        torch.clamp(values - upper, min=0).amax(),
    )


def _template_signature(
    problem: PortfolioProblem,
    matrix: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> str:
    digest = hashlib.sha256()
    tensors = [
        matrix,
        lower,
        upper,
        problem.risk.exposures,
        problem.risk.factor_variance,
        problem.risk.specific_variance,
        problem.lower,
        problem.upper,
    ]
    if problem.current is not None:
        tensors.append(problem.current)
    for tensor in tensors:
        values = tensor.detach().contiguous().cpu().numpy()
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(str(values.shape).encode("utf-8"))
        digest.update(values.tobytes())
    for scalar in (problem.budget, problem.risk_aversion, problem.trade_cost):
        digest.update(float(scalar).hex().encode("ascii"))
    return digest.hexdigest()


def _validate_initial_state(
    initial_state: PrimalDualState,
    problem: PortfolioProblem,
    matrix: torch.Tensor,
    template_signature: str,
    config: PrimalDualConfig,
) -> PrimalDualState:
    alpha = problem.batched_alpha
    weights = initial_state.weights
    dual = initial_state.dual
    expected_primal_shape = alpha.shape
    expected_dual_shape = (alpha.shape[0], matrix.shape[0])

    for name, tensor in (
        ("initial_state.weights", weights),
        ("initial_state.dual", dual),
        ("initial_state.alpha", initial_state.alpha),
    ):
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must use a floating-point dtype")
        if tensor.device != alpha.device:
            raise ValueError(f"{name} must be on the same device as the problem")
        if tensor.dtype != alpha.dtype:
            raise ValueError(f"{name} must use the same dtype as the problem")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must be finite")

    if weights.ndim != 2 or weights.shape != expected_primal_shape:
        raise ValueError(
            "initial_state.weights must have shape "
            f"{expected_primal_shape}, got {tuple(weights.shape)}"
        )
    if dual.ndim != 2 or dual.shape != expected_dual_shape:
        raise ValueError(
            "initial_state.dual must have shape "
            f"{expected_dual_shape}, got {tuple(dual.shape)}"
        )
    if initial_state.alpha.shape != expected_primal_shape:
        raise ValueError(
            "initial_state.alpha must have shape "
            f"{expected_primal_shape}, got {tuple(initial_state.alpha.shape)}"
        )
    if initial_state.template_signature != template_signature:
        raise ValueError(
            "initial_state problem template does not match the current problem"
        )
    if config.warm_start_max_alpha_delta is not None:
        alpha_delta = float(torch.max(torch.abs(alpha - initial_state.alpha)))
        if alpha_delta > config.warm_start_max_alpha_delta:
            raise ValueError(
                "warm-start alpha drift exceeds configured limit: "
                f"delta={alpha_delta:.12g}, "
                f"limit={config.warm_start_max_alpha_delta:.12g}"
            )

    projected_weights = project_bounded_simplex_unchecked(
        weights,
        problem.lower,
        problem.upper,
        problem.budget,
        config.bisection_steps,
    )
    return PrimalDualState(
        weights=projected_weights,
        dual=dual,
        alpha=initial_state.alpha,
        template_signature=template_signature,
    )


@torch.no_grad()
def solve_constrained(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    config: PrimalDualConfig = PrimalDualConfig(),
    initial_state: PrimalDualState | None = None,
) -> ConstrainedSolveResult:
    if constraints.num_variables != problem.risk.assets:
        raise ValueError("constraints and problem have inconsistent asset counts")
    if constraints.device != problem.alpha.device:
        raise ValueError("constraints and problem must be on the same device")
    if constraints.dtype != problem.alpha.dtype:
        raise ValueError("constraints and problem must use the same dtype")

    matrix, constraint_lower, constraint_upper = _stack_constraints(constraints)
    template_signature = _template_signature(
        problem, matrix, constraint_lower, constraint_upper
    )
    operator_norm = float(torch.linalg.matrix_norm(matrix, ord=2))
    if operator_norm <= 0:
        raise ValueError("linear constraint matrix must have positive operator norm")
    smooth_lipschitz = _lipschitz_bound(problem)
    dual_step = 1.0 / operator_norm
    primal_step = config.step_safety / (operator_norm + 0.5 * smooth_lipschitz)

    alpha = problem.batched_alpha
    if not torch.isfinite(alpha).all().item():
        raise ValueError("problem alpha must be finite")
    if not torch.isfinite(problem.lower).all().item() or not torch.isfinite(problem.upper).all().item():
        raise ValueError("problem bounds must be finite")
    if initial_state is None:
        initial = alpha.new_full(alpha.shape, problem.budget / alpha.shape[1])
        weights = project_bounded_simplex_unchecked(
            initial,
            problem.lower,
            problem.upper,
            problem.budget,
            config.bisection_steps,
        )
        dual = alpha.new_zeros((alpha.shape[0], matrix.shape[0]))
    else:
        warmed = _validate_initial_state(
            initial_state, problem, matrix, template_signature, config
        )
        weights = warmed.weights
        dual = warmed.dual
    converged = False
    iterations = config.iterations
    start = time.perf_counter()

    for iteration in range(1, config.iterations + 1):
        gradient = problem.risk_aversion * problem.risk.matvec(weights) - alpha
        if problem.current is not None and problem.trade_cost:
            current = problem.current.unsqueeze(0) if problem.current.ndim == 1 else problem.current
            gradient = gradient + problem.trade_cost * (weights - current)
        next_weights = project_bounded_simplex_unchecked(
            weights - primal_step * (gradient + dual @ matrix),
            problem.lower,
            problem.upper,
            problem.budget,
            config.bisection_steps,
        )
        extrapolated = 2.0 * next_weights - weights
        dual_candidate = dual + dual_step * (extrapolated @ matrix.T)
        projected_interval = torch.clamp(
            dual_candidate / dual_step,
            min=constraint_lower,
            max=constraint_upper,
        )
        next_dual = dual_candidate - dual_step * projected_interval

        if config.tolerance and iteration % config.check_interval == 0:
            delta = torch.max(torch.abs(next_weights - weights))
            violation = _constraint_violation(
                next_weights, matrix, constraint_lower, constraint_upper
            )
            if float(delta) <= config.tolerance and float(violation) <= config.constraint_tolerance:
                converged = True
                iterations = iteration
                weights = next_weights
                dual = next_dual
                break
        weights = next_weights
        dual = next_dual

    final_violation = float(
        _constraint_violation(weights, matrix, constraint_lower, constraint_upper)
    )
    if weights.is_cuda:
        torch.cuda.synchronize(weights.device)
    elapsed = time.perf_counter() - start
    return ConstrainedSolveResult(
        weights=weights,
        objective=objective(problem, weights),
        iterations=iterations,
        converged=converged,
        max_constraint_violation=final_violation,
        elapsed_seconds=elapsed,
        device=str(weights.device),
        state=PrimalDualState(
            weights=weights.detach().clone(),
            dual=dual.detach().clone(),
            alpha=alpha.detach().clone(),
            template_signature=template_signature,
        ),
    )
