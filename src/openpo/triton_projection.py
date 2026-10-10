from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _bounded_simplex_kernel(
    values_pointer,
    lower_pointer,
    upper_pointer,
    output_pointer,
    row_stride: tl.constexpr,
    assets: tl.constexpr,
    budget: tl.constexpr,
    steps: tl.constexpr,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_size)
    mask = offsets < assets
    values = tl.load(values_pointer + row * row_stride + offsets, mask=mask, other=0.0)
    lower = tl.load(lower_pointer + offsets, mask=mask, other=0.0)
    upper = tl.load(upper_pointer + offsets, mask=mask, other=0.0)
    positive_infinity = float("inf")
    negative_infinity = -float("inf")
    threshold_low = tl.min(
        tl.where(mask, values - upper, positive_infinity), axis=0
    )
    threshold_high = tl.max(
        tl.where(mask, values - lower, negative_infinity), axis=0
    )

    for _ in range(steps):
        threshold = 0.5 * (threshold_low + threshold_high)
        projected = tl.minimum(tl.maximum(values - threshold, lower), upper)
        total = tl.sum(tl.where(mask, projected, 0.0), axis=0)
        threshold_low = tl.where(total > budget, threshold, threshold_low)
        threshold_high = tl.where(total > budget, threshold_high, threshold)

    projected = tl.minimum(tl.maximum(values - threshold_high, lower), upper)
    total = tl.sum(tl.where(mask, projected, 0.0), axis=0)
    residual = budget - total
    capacity = tl.where(residual >= 0.0, upper - projected, projected - lower)
    total_capacity = tl.sum(tl.where(mask, capacity, 0.0), axis=0)
    safe_capacity = tl.maximum(total_capacity, 1.0e-20)
    corrected = projected + residual * capacity / safe_capacity
    corrected = tl.minimum(tl.maximum(corrected, lower), upper)
    tl.store(output_pointer + row * row_stride + offsets, corrected, mask=mask)


def triton_project_bounded_simplex(
    values: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    budget: float = 1.0,
    bisection_steps: int = 20,
) -> torch.Tensor:
    if not values.is_cuda:
        raise ValueError("Triton projection requires CUDA values")
    if values.dtype != torch.float32:
        raise TypeError("Triton projection currently requires float32 tensors")
    if values.ndim != 2:
        raise ValueError("values must have shape [batch, assets]")
    if lower.shape != (values.shape[1],) or upper.shape != (values.shape[1],):
        raise ValueError("bounds must have shape [assets]")
    if values.device != lower.device or values.device != upper.device:
        raise ValueError("values and bounds must be on the same CUDA device")
    if not values.is_contiguous() or not lower.is_contiguous() or not upper.is_contiguous():
        raise ValueError("Triton projection requires contiguous tensors")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")
    assets = values.shape[1]
    if assets > 4096:
        raise ValueError("Triton projection currently supports at most 4096 assets")
    with torch.cuda.device(values.device):
        output = torch.empty_like(values)
        block_size = triton.next_power_of_2(assets)
        _bounded_simplex_kernel[(values.shape[0],)](
            values,
            lower,
            upper,
            output,
            values.stride(0),
            assets,
            budget,
            bisection_steps,
            block_size,
            num_warps=8 if block_size >= 1024 else 4,
        )
    return output
