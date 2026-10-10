from __future__ import annotations

import torch


def _project_bounded_simplex_torch(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    budget: float = 1.0,
    bisection_steps: int = 48,
) -> torch.Tensor:
    threshold_low = (values - upper).amin(dim=1, keepdim=True)
    threshold_high = (values - lower).amax(dim=1, keepdim=True)
    for _ in range(bisection_steps):
        threshold = (threshold_low + threshold_high) * 0.5
        projected = torch.clamp(values - threshold, min=lower, max=upper)
        too_large = projected.sum(dim=1, keepdim=True) > budget
        threshold_low = torch.where(too_large, threshold, threshold_low)
        threshold_high = torch.where(too_large, threshold_high, threshold)
    projected = torch.clamp(values - threshold_high, min=lower, max=upper)
    residual = budget - projected.sum(dim=1, keepdim=True)
    upward_capacity = upper - projected
    downward_capacity = projected - lower
    capacity = torch.where(residual >= 0, upward_capacity, downward_capacity)
    total_capacity = capacity.sum(dim=1, keepdim=True)
    safe_capacity = torch.where(
        total_capacity > 0, total_capacity, torch.ones_like(total_capacity)
    )
    corrected = projected + residual * capacity / safe_capacity
    return torch.clamp(corrected, min=lower, max=upper)


def project_bounded_simplex_unchecked(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    budget: float = 1.0,
    bisection_steps: int = 48,
) -> torch.Tensor:
    if (
        values.is_cuda
        and values.dtype == torch.float32
        and values.shape[1] <= 4096
        and values.is_contiguous()
        and lower.is_contiguous()
        and upper.is_contiguous()
    ):
        from .triton_projection import triton_project_bounded_simplex

        return triton_project_bounded_simplex(
            values, lower, upper, budget, bisection_steps
        )
    return _project_bounded_simplex_torch(
        values, lower, upper, budget, bisection_steps
    )


def project_bounded_simplex(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    budget: float = 1.0,
    bisection_steps: int = 48,
) -> torch.Tensor:
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, assets]")
    if lower.shape != (values.shape[1],) or upper.shape != (values.shape[1],):
        raise ValueError("bounds must have shape [assets]")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")
    if values.device != lower.device or values.device != upper.device:
        raise ValueError("values and bounds must be on the same device")
    if values.dtype != lower.dtype or values.dtype != upper.dtype:
        raise ValueError("values and bounds must use the same dtype")
    if torch.any(lower > upper):
        raise ValueError("lower bounds exceed upper bounds")
    if float(lower.sum()) > budget or float(upper.sum()) < budget:
        raise ValueError("budget is infeasible under supplied bounds")
    return project_bounded_simplex_unchecked(
        values, lower, upper, budget, bisection_steps
    )
