from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real

import torch


_TRITON_TRADE_PROJECTION_DISABLED: set[tuple[int, int]] = set()


@dataclass(frozen=True)
class Func2LPSolveResult:
    dw: torch.Tensor
    objective: torch.Tensor
    traded_volume: torch.Tensor
    budget_residual: torch.Tensor
    box_residual: torch.Tensor
    device: torch.device


@dataclass(frozen=True)
class TradeStyleQPSolveResult:
    dw: torch.Tensor
    objective: torch.Tensor
    mean_score: torch.Tensor
    trade_cost: torch.Tensor
    style_utility: torch.Tensor
    exposure: torch.Tensor
    lower_style_violation: torch.Tensor
    upper_style_violation: torch.Tensor
    budget_residual: torch.Tensor
    box_residual: torch.Tensor
    stationarity_residual: torch.Tensor
    iterations: int
    status: str
    projection_backend: str
    device: torch.device


def _require_finite_output(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} is non-finite")


def _validate_inputs(
    score: torch.Tensor,
    max_sell: torch.Tensor,
    max_buy: torch.Tensor,
    spread: float | torch.Tensor,
) -> torch.Tensor:
    for name, tensor in (
        ("score", score),
        ("max_sell", max_sell),
        ("max_buy", max_buy),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.ndim != 1:
            raise ValueError(f"{name} must have shape [assets]")
        if tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} must use torch.float32 or torch.float64")
        if tensor.device.type not in ("cpu", "cuda"):
            raise ValueError(f"{name} must be on a CPU or CUDA device")
    if score.numel() == 0:
        raise ValueError("score must contain at least one asset")
    if max_sell.shape != score.shape or max_buy.shape != score.shape:
        raise ValueError("score, max_sell, and max_buy must have the same shape")
    if max_sell.device != score.device or max_buy.device != score.device:
        raise ValueError("score, max_sell, and max_buy must be on the same device")
    if max_sell.dtype != score.dtype or max_buy.dtype != score.dtype:
        raise ValueError("score, max_sell, and max_buy must use the same dtype")
    if not bool(torch.isfinite(score).all()):
        raise ValueError("score must contain only finite values")
    if not bool(torch.isfinite(max_sell).all()) or bool(torch.any(max_sell < 0)):
        raise ValueError("max_sell must contain only finite nonnegative values")
    if not bool(torch.isfinite(max_buy).all()) or bool(torch.any(max_buy < 0)):
        raise ValueError("max_buy must contain only finite nonnegative values")

    if isinstance(spread, torch.Tensor):
        if spread.ndim != 0:
            raise ValueError("spread must be a scalar")
        if spread.device != score.device:
            raise ValueError("spread must be on the same device as score")
        if spread.dtype != score.dtype:
            raise ValueError("spread must use the same dtype as score")
        spread_tensor = spread.detach()
    elif isinstance(spread, Real) and not isinstance(spread, bool):
        spread_value = float(spread)
        if not math.isfinite(spread_value):
            raise ValueError("spread must be finite")
        spread_tensor = score.new_tensor(spread_value)
    else:
        raise TypeError("spread must be a real scalar or scalar torch.Tensor")
    if not bool(torch.isfinite(spread_tensor)) or bool(spread_tensor < 0):
        raise ValueError("spread must be finite and nonnegative")
    return spread_tensor


def solve_func_2_lp(
    score: torch.Tensor,
    max_sell: torch.Tensor,
    max_buy: torch.Tensor,
    spread: float | torch.Tensor,
) -> Func2LPSolveResult:
    """Solve the market-neutral box LP by sorting marginal trade values."""
    spread_tensor = _validate_inputs(score, max_sell, max_buy, spread)

    with torch.no_grad():
        buy_order = torch.argsort(score, descending=True, stable=True)
        sell_order = torch.argsort(score, descending=False, stable=True)
        sorted_buy_capacity = max_buy[buy_order]
        sorted_sell_capacity = max_sell[sell_order]
        cumulative_buy = torch.cumsum(sorted_buy_capacity, dim=0)
        cumulative_sell = torch.cumsum(sorted_sell_capacity, dim=0)
        if not bool(torch.isfinite(cumulative_buy[-1])) or not bool(
            torch.isfinite(cumulative_sell[-1])
        ):
            raise ValueError("cumulative trade capacity overflowed the input dtype")

        total_capacity = torch.minimum(cumulative_buy[-1], cumulative_sell[-1])
        zero = score.new_zeros(1)
        breakpoints = torch.cat(
            (
                zero,
                torch.minimum(cumulative_buy, total_capacity),
                torch.minimum(cumulative_sell, total_capacity),
                total_capacity.reshape(1),
            )
        )
        breakpoints = torch.unique_consecutive(torch.sort(breakpoints).values)
        interval_ends = breakpoints[1:]
        interval_capacity = interval_ends - breakpoints[:-1]

        buy_indices = torch.searchsorted(cumulative_buy, interval_ends, right=False)
        sell_indices = torch.searchsorted(cumulative_sell, interval_ends, right=False)
        sorted_buy_score = score[buy_order]
        sorted_sell_score = score[sell_order]
        positive_margin = (
            sorted_buy_score[buy_indices]
            - sorted_sell_score[sell_indices]
            - spread_tensor
            > 0
        )
        traded_capacity = interval_capacity * positive_margin.to(score.dtype)

        matched_volume = traded_capacity.sum()
        _require_finite_output("matched_volume", matched_volume)
        buy_starts = torch.cat((zero, cumulative_buy[:-1]))
        sell_starts = torch.cat((zero, cumulative_sell[:-1]))
        sorted_buy_volume = torch.minimum(
            (matched_volume - buy_starts).clamp_min(0), sorted_buy_capacity
        )
        sorted_sell_volume = torch.minimum(
            (matched_volume - sell_starts).clamp_min(0), sorted_sell_capacity
        )
        buy_volume = torch.zeros_like(score).scatter(
            0, buy_order, sorted_buy_volume
        )
        sell_volume = torch.zeros_like(score).scatter(
            0, sell_order, sorted_sell_volume
        )
        dw = buy_volume - sell_volume
        _require_finite_output("dw", dw)

        traded_volume = dw.abs().sum()
        _require_finite_output("traded_volume", traded_volume)
        objective = torch.dot(score, dw) - 0.5 * spread_tensor * traded_volume
        _require_finite_output("objective", objective)
        budget_residual = dw.sum().abs()
        _require_finite_output("budget_residual", budget_residual)
        lower_violation = (-max_sell - dw).clamp_min(0).amax()
        upper_violation = (dw - max_buy).clamp_min(0).amax()
        box_residual = torch.maximum(lower_violation, upper_violation)
        _require_finite_output("box_residual", box_residual)

    return Func2LPSolveResult(
        dw=dw,
        objective=objective,
        traded_volume=traded_volume,
        budget_residual=budget_residual,
        box_residual=box_residual,
        device=score.device,
    )


def _prox_l1_box_neutral_torch(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    threshold: torch.Tensor,
    bisection_steps: int,
) -> torch.Tensor:
    lower_lambda = torch.min(value - threshold - upper) - 1.0
    upper_lambda = torch.max(value + threshold - lower) + 1.0
    for _ in range(bisection_steps):
        middle = 0.5 * (lower_lambda + upper_lambda)
        shifted = value - middle
        candidate = torch.sign(shifted) * torch.clamp(
            torch.abs(shifted) - threshold, min=0
        )
        candidate = torch.minimum(torch.maximum(candidate, lower), upper)
        positive_sum = candidate.sum() > 0
        lower_lambda = torch.where(positive_sum, middle, lower_lambda)
        upper_lambda = torch.where(positive_sum, upper_lambda, middle)
    shifted = value - 0.5 * (lower_lambda + upper_lambda)
    candidate = torch.sign(shifted) * torch.clamp(
        torch.abs(shifted) - threshold, min=0
    )
    return torch.minimum(torch.maximum(candidate, lower), upper)


def _prox_l1_box_neutral(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    threshold: torch.Tensor,
    bisection_steps: int,
) -> torch.Tensor:
    if (
        value.is_cuda
        and value.dtype == torch.float64
        and value.is_contiguous()
        and lower.is_contiguous()
        and upper.is_contiguous()
    ):
        device_index = value.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        dispatch_key = (device_index, value.numel())
        if dispatch_key not in _TRITON_TRADE_PROJECTION_DISABLED:
            try:
                from .triton_trade_projection import (
                    TritonTradeProjectionUnavailable,
                    triton_prox_l1_box_neutral,
                )

                return triton_prox_l1_box_neutral(
                    value, lower, upper, threshold, bisection_steps
                )
            except (ImportError, TritonTradeProjectionUnavailable):
                _TRITON_TRADE_PROJECTION_DISABLED.add(dispatch_key)
    return _prox_l1_box_neutral_torch(
        value, lower, upper, threshold, bisection_steps
    )


def _projection_backend(value: torch.Tensor) -> str:
    if not value.is_cuda or value.dtype != torch.float64 or not value.is_contiguous():
        return "torch"
    device_index = value.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    return (
        "torch"
        if (device_index, value.numel()) in _TRITON_TRADE_PROJECTION_DISABLED
        else "triton"
    )


def solve_trade_style_qp(
    score: torch.Tensor,
    max_sell: torch.Tensor,
    max_buy: torch.Tensor,
    spread: float | torch.Tensor,
    weights: torch.Tensor,
    market_weights: torch.Tensor,
    factor_exposure: torch.Tensor,
    style_ranges: torch.Tensor,
    *,
    style_penalty: float = 5.0,
    max_iterations: int = 500,
    tolerance: float = 1e-8,
    bisection_steps: int = 60,
) -> TradeStyleQPSolveResult:
    """Solve the HandleScore Barra hinge-squared QP with accelerated proximal gradient."""
    spread_tensor = _validate_inputs(score, max_sell, max_buy, spread)
    for name, tensor in (("weights", weights), ("market_weights", market_weights)):
        if not isinstance(tensor, torch.Tensor) or tensor.shape != score.shape:
            raise ValueError(f"{name} must match score shape")
        if tensor.device != score.device or tensor.dtype != score.dtype:
            raise ValueError(f"{name} must match score device and dtype")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")
    if (
        not isinstance(factor_exposure, torch.Tensor)
        or factor_exposure.ndim != 2
        or factor_exposure.shape[0] != score.numel()
        or factor_exposure.shape[1] == 0
    ):
        raise ValueError("factor_exposure must have shape [assets, styles]")
    if factor_exposure.device != score.device or factor_exposure.dtype != score.dtype:
        raise ValueError("factor_exposure must match score device and dtype")
    if not bool(torch.isfinite(factor_exposure).all()):
        raise ValueError("factor_exposure must contain only finite values")
    if (
        not isinstance(style_ranges, torch.Tensor)
        or style_ranges.shape != (factor_exposure.shape[1], 2)
        or style_ranges.device != score.device
        or style_ranges.dtype != score.dtype
    ):
        raise ValueError("style_ranges must have shape [styles, 2] on the score device and dtype")
    if not bool(torch.isfinite(style_ranges).all()) or bool(
        torch.any(style_ranges[:, 0] > style_ranges[:, 1])
    ):
        raise ValueError("style_ranges must be finite with lower <= upper")
    if not isinstance(style_penalty, Real) or isinstance(style_penalty, bool):
        raise TypeError("style_penalty must be a real scalar")
    style_penalty_value = float(style_penalty)
    if not math.isfinite(style_penalty_value) or style_penalty_value <= 0:
        raise ValueError("style_penalty must be finite and positive")
    if max_iterations <= 0 or bisection_steps <= 0:
        raise ValueError("max_iterations and bisection_steps must be positive")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")

    with torch.no_grad():
        gram = factor_exposure.T @ factor_exposure
        largest_eigenvalue = torch.linalg.eigvalsh(gram)[-1]
        lipschitz = 2.0 * style_penalty_value * largest_eigenvalue
        if not bool(torch.isfinite(lipschitz)) or bool(lipschitz <= 0):
            raise ValueError("factor_exposure must have positive finite spectral norm")
        step = 0.95 * lipschitz.reciprocal()
        threshold = step * 0.5 * spread_tensor
        lower = -max_sell
        upper = max_buy
        current = _prox_l1_box_neutral(
            torch.zeros_like(score), lower, upper, threshold, bisection_steps
        )
        accelerated = current.clone()
        momentum = score.new_tensor(1.0)
        converged = False
        iterations = max_iterations
        stationarity_residual = score.new_tensor(math.inf)
        base_exposure = factor_exposure.T @ (weights - market_weights)
        lower_range = style_ranges[:, 0]
        upper_range = style_ranges[:, 1]
        for iteration in range(1, max_iterations + 1):
            exposure = base_exposure + factor_exposure.T @ accelerated
            lower_violation = torch.clamp(lower_range - exposure, min=0)
            upper_violation = torch.clamp(exposure - upper_range, min=0)
            gradient = -score + factor_exposure @ (
                2.0 * style_penalty_value * (upper_violation - lower_violation)
            )
            candidate = _prox_l1_box_neutral(
                accelerated - step * gradient,
                lower,
                upper,
                threshold,
                bisection_steps,
            )
            next_momentum = 0.5 * (1.0 + torch.sqrt(1.0 + 4.0 * momentum.square()))
            accelerated = candidate + ((momentum - 1.0) / next_momentum) * (
                candidate - current
            )
            if iteration % 10 == 0:
                check_exposure = base_exposure + factor_exposure.T @ candidate
                check_lower_violation = torch.clamp(
                    lower_range - check_exposure, min=0
                )
                check_upper_violation = torch.clamp(
                    check_exposure - upper_range, min=0
                )
                check_gradient = -score + factor_exposure @ (
                    2.0
                    * style_penalty_value
                    * (check_upper_violation - check_lower_violation)
                )
                check_prox = _prox_l1_box_neutral(
                    candidate - step * check_gradient,
                    lower,
                    upper,
                    threshold,
                    bisection_steps,
                )
                mapping_norm = torch.linalg.vector_norm(
                    (candidate - check_prox) / step, ord=torch.inf
                )
                gradient_scale = torch.maximum(
                    score.new_tensor(1.0),
                    torch.maximum(
                        torch.linalg.vector_norm(check_gradient, ord=torch.inf),
                        0.5 * spread_tensor,
                    ),
                )
                stationarity_residual = mapping_norm / gradient_scale
                if bool(stationarity_residual <= tolerance):
                    converged = True
                    iterations = iteration
                    current = candidate
                    break
            current = candidate
            momentum = next_momentum

        dw = current
        exposure = base_exposure + factor_exposure.T @ dw
        lower_violation = torch.clamp(lower_range - exposure, min=0)
        upper_violation = torch.clamp(exposure - upper_range, min=0)
        mean_score = torch.dot(score, dw)
        trade_cost = 0.5 * spread_tensor * torch.abs(dw).sum()
        style_utility = style_penalty_value * (
            lower_violation.square().sum() + upper_violation.square().sum()
        )
        objective = mean_score - trade_cost - style_utility
        budget_residual = torch.abs(dw.sum())
        box_residual = torch.maximum(
            torch.clamp(lower - dw, min=0).amax(),
            torch.clamp(dw - upper, min=0).amax(),
        )
        final_gradient = -score + factor_exposure @ (
            2.0 * style_penalty_value * (upper_violation - lower_violation)
        )
        final_prox = _prox_l1_box_neutral(
            dw - step * final_gradient,
            lower,
            upper,
            threshold,
            bisection_steps,
        )
        stationarity_residual = torch.linalg.vector_norm(
            (dw - final_prox) / step, ord=torch.inf
        ) / torch.maximum(
            score.new_tensor(1.0),
            torch.maximum(
                torch.linalg.vector_norm(final_gradient, ord=torch.inf),
                0.5 * spread_tensor,
            ),
        )
        converged = bool(stationarity_residual <= tolerance)
        for name, value in (
            ("dw", dw),
            ("objective", objective),
            ("exposure", exposure),
            ("budget_residual", budget_residual),
            ("box_residual", box_residual),
            ("stationarity_residual", stationarity_residual),
        ):
            _require_finite_output(name, value)

    return TradeStyleQPSolveResult(
        dw=dw,
        objective=objective,
        mean_score=mean_score,
        trade_cost=trade_cost,
        style_utility=style_utility,
        exposure=exposure,
        lower_style_violation=lower_violation,
        upper_style_violation=upper_violation,
        budget_residual=budget_residual,
        box_residual=box_residual,
        stationarity_residual=stationarity_residual,
        iterations=iterations,
        status="converged" if converged else "fixed_iterations",
        projection_backend=_projection_backend(score),
        device=score.device,
    )
