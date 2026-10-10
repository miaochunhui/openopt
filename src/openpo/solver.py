from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import math
import time

import torch

from .model import PortfolioProblem
from .projection import project_bounded_simplex_unchecked


@dataclass(frozen=True)
class SolveConfig:
    iterations: int = 200
    tolerance: float = 1e-6
    check_interval: int = 10
    bisection_steps: int = 24

    def __post_init__(self) -> None:
        if self.iterations <= 0 or self.check_interval <= 0 or self.bisection_steps <= 0:
            raise ValueError("iteration settings must be positive")
        if self.tolerance < 0:
            raise ValueError("tolerance must be nonnegative")


@dataclass(frozen=True)
class SolveResult:
    weights: torch.Tensor
    objective: torch.Tensor
    iterations: int
    converged: bool
    elapsed_seconds: float
    device: str


def objective(problem: PortfolioProblem, weights: torch.Tensor) -> torch.Tensor:
    alpha = problem.batched_alpha
    risk = 0.5 * problem.risk_aversion * (weights * problem.risk.matvec(weights)).sum(dim=1)
    utility = (alpha * weights).sum(dim=1)
    if problem.current is None or problem.trade_cost == 0:
        trade = torch.zeros_like(risk)
    else:
        current = problem.current.unsqueeze(0) if problem.current.ndim == 1 else problem.current
        trade = 0.5 * problem.trade_cost * ((weights - current) ** 2).sum(dim=1)
    return risk - utility + trade


def _lipschitz_bound(problem: PortfolioProblem) -> float:
    exposure_norm = torch.linalg.matrix_norm(problem.risk.exposures, ord=2)
    risk_bound = (
        problem.risk.specific_variance.max()
        + exposure_norm.square() * problem.risk.factor_variance.max()
    )
    return float(problem.risk_aversion * risk_bound + problem.trade_cost)


def solve(problem: PortfolioProblem, config: SolveConfig = SolveConfig()) -> SolveResult:
    alpha = problem.batched_alpha
    if alpha.device != problem.risk.exposures.device:
        raise ValueError("all problem tensors must be on the same device")
    start = time.perf_counter()
    initial = alpha.new_full(alpha.shape, problem.budget / alpha.shape[1])
    weights = project_bounded_simplex_unchecked(
        initial, problem.lower, problem.upper, problem.budget, config.bisection_steps
    )
    accelerated = weights.clone()
    momentum = 1.0
    step_size = 1.0 / _lipschitz_bound(problem)
    converged = False
    iterations = config.iterations

    for iteration in range(1, config.iterations + 1):
        gradient = problem.risk_aversion * problem.risk.matvec(accelerated) - alpha
        if problem.current is not None and problem.trade_cost:
            current = problem.current.unsqueeze(0) if problem.current.ndim == 1 else problem.current
            gradient = gradient + problem.trade_cost * (accelerated - current)
        next_weights = project_bounded_simplex_unchecked(
            accelerated - step_size * gradient,
            problem.lower,
            problem.upper,
            problem.budget,
            config.bisection_steps,
        )
        next_momentum = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * momentum * momentum))
        accelerated = next_weights + ((momentum - 1.0) / next_momentum) * (next_weights - weights)

        if config.tolerance and iteration % config.check_interval == 0:
            delta = torch.max(torch.abs(next_weights - weights))
            if float(delta) <= config.tolerance:
                converged = True
                iterations = iteration
                weights = next_weights
                break
        weights = next_weights
        momentum = next_momentum

    if weights.is_cuda:
        torch.cuda.synchronize(weights.device)
    elapsed = time.perf_counter() - start
    return SolveResult(
        weights=weights,
        objective=objective(problem, weights),
        iterations=iterations,
        converged=converged,
        elapsed_seconds=elapsed,
        device=str(weights.device),
    )


def solve_cardinality(
    problem: PortfolioProblem,
    max_holdings: int,
    minimum_holding: float = 0.0,
    config: SolveConfig = SolveConfig(),
) -> SolveResult:
    if problem.alpha.ndim != 1:
        raise ValueError("cardinality prototype currently accepts one portfolio at a time")
    if not 0 < max_holdings <= problem.risk.assets:
        raise ValueError("max_holdings must be in [1, assets]")
    if minimum_holding < 0 or minimum_holding * max_holdings > problem.budget:
        raise ValueError("minimum holding is infeasible")

    relaxed = solve(problem, config)
    selected = torch.topk(relaxed.weights[0], max_holdings).indices
    selected_capacity = float(problem.upper[selected].sum())
    if selected_capacity < problem.budget:
        raise ValueError(
            "selected holding capacity is below budget: "
            f"capacity={selected_capacity:.12g}, budget={problem.budget:.12g}, "
            f"max_holdings={max_holdings}"
        )
    lower = torch.zeros_like(problem.lower)
    upper = torch.zeros_like(problem.upper)
    lower[selected] = minimum_holding
    upper[selected] = problem.upper[selected]
    restricted = PortfolioProblem(
        alpha=problem.alpha,
        risk=problem.risk,
        lower=lower,
        upper=upper,
        budget=problem.budget,
        risk_aversion=problem.risk_aversion,
        current=problem.current,
        trade_cost=problem.trade_cost,
    )
    result = solve(restricted, config)
    return SolveResult(
        weights=result.weights,
        objective=result.objective,
        iterations=relaxed.iterations + result.iterations,
        converged=relaxed.converged and result.converged,
        elapsed_seconds=relaxed.elapsed_seconds + result.elapsed_seconds,
        device=result.device,
    )


def solve_multi_gpu(
    problem: PortfolioProblem,
    devices: list[str],
    config: SolveConfig = SolveConfig(),
) -> SolveResult:
    if problem.alpha.ndim != 2:
        raise ValueError("multi-GPU solving requires batched alpha")
    if not devices or any(not device.startswith("cuda:") for device in devices):
        raise ValueError("devices must be a nonempty list of explicit CUDA devices")
    for device in devices:
        index = int(device.split(":", 1)[1])
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device is unavailable: {device}")
    if problem.alpha.shape[0] < len(devices):
        raise ValueError(
            "batch must be at least the number of CUDA devices to avoid empty shards: "
            f"batch={problem.alpha.shape[0]}, devices={len(devices)}"
        )

    shards = torch.tensor_split(problem.alpha, len(devices), dim=0)
    start = time.perf_counter()

    def run_shard(device: str, alpha: torch.Tensor) -> SolveResult:
        shard_problem = PortfolioProblem(
            alpha=alpha.to(device),
            risk=problem.risk.to(device),
            lower=problem.lower.to(device),
            upper=problem.upper.to(device),
            budget=problem.budget,
            risk_aversion=problem.risk_aversion,
            current=None if problem.current is None else problem.current.to(device),
            trade_cost=problem.trade_cost,
        )
        return solve(shard_problem, config)

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(run_shard, device, shard) for device, shard in zip(devices, shards)]
        results = [future.result() for future in futures]
    weights = torch.cat([result.weights.cpu() for result in results], dim=0)
    cpu_problem = problem.to("cpu")
    elapsed = time.perf_counter() - start
    return SolveResult(
        weights=weights,
        objective=objective(cpu_problem, weights),
        iterations=max(result.iterations for result in results),
        converged=all(result.converged for result in results),
        elapsed_seconds=elapsed,
        device=",".join(devices),
    )
