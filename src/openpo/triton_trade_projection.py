from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.runtime.errors import TritonError


class TritonTradeProjectionUnavailable(RuntimeError):
    pass


@triton.jit
def _l1_box_neutral_kernel(
    value_pointer,
    lower_pointer,
    upper_pointer,
    output_pointer,
    threshold_pointer,
    assets: tl.constexpr,
    steps: tl.constexpr,
    block_size: tl.constexpr,
):
    offsets = tl.arange(0, block_size)
    mask = offsets < assets
    value = tl.load(value_pointer + offsets, mask=mask, other=0.0)
    lower = tl.load(lower_pointer + offsets, mask=mask, other=0.0)
    upper = tl.load(upper_pointer + offsets, mask=mask, other=0.0)
    threshold = tl.load(threshold_pointer)
    positive_infinity = float("inf")
    negative_infinity = -float("inf")
    lower_lambda = (
        tl.min(
            tl.where(mask, value - threshold - upper, positive_infinity), axis=0
        )
        - 1.0
    )
    upper_lambda = (
        tl.max(
            tl.where(mask, value + threshold - lower, negative_infinity), axis=0
        )
        + 1.0
    )

    for _ in range(steps):
        middle = 0.5 * (lower_lambda + upper_lambda)
        shifted = value - middle
        candidate = tl.where(
            shifted > threshold,
            shifted - threshold,
            tl.where(shifted < -threshold, shifted + threshold, 0.0),
        )
        candidate = tl.minimum(tl.maximum(candidate, lower), upper)
        total = tl.sum(tl.where(mask, candidate, 0.0), axis=0)
        lower_lambda = tl.where(total > 0.0, middle, lower_lambda)
        upper_lambda = tl.where(total > 0.0, upper_lambda, middle)

    shifted = value - 0.5 * (lower_lambda + upper_lambda)
    candidate = tl.where(
        shifted > threshold,
        shifted - threshold,
        tl.where(shifted < -threshold, shifted + threshold, 0.0),
    )
    candidate = tl.minimum(tl.maximum(candidate, lower), upper)
    tl.store(output_pointer + offsets, candidate, mask=mask)


def triton_prox_l1_box_neutral(
    value: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    threshold: torch.Tensor,
    bisection_steps: int,
) -> torch.Tensor:
    if not value.is_cuda:
        raise ValueError("Triton trade projection requires CUDA values")
    if value.dtype != torch.float64:
        raise TypeError("Triton trade projection requires float64 tensors")
    if value.ndim != 1:
        raise ValueError("value must have shape [assets]")
    if lower.shape != value.shape or upper.shape != value.shape:
        raise ValueError("bounds must match value shape")
    if threshold.ndim != 0:
        raise ValueError("threshold must be a scalar")
    if value.device != lower.device or value.device != upper.device:
        raise ValueError("value and bounds must be on the same CUDA device")
    if threshold.device != value.device or threshold.dtype != value.dtype:
        raise ValueError("threshold must match value device and dtype")
    if lower.dtype != value.dtype or upper.dtype != value.dtype:
        raise ValueError("value and bounds must use the same dtype")
    if (
        not value.is_contiguous()
        or not lower.is_contiguous()
        or not upper.is_contiguous()
    ):
        raise ValueError("Triton trade projection requires contiguous tensors")
    if bisection_steps <= 0:
        raise ValueError("bisection_steps must be positive")
    assets = value.numel()
    if assets > 8192:
        raise TritonTradeProjectionUnavailable(
            "Triton trade projection supports at most 8192 assets"
        )
    capability = torch.cuda.get_device_capability(value.device)
    if capability[0] < 8:
        raise TritonTradeProjectionUnavailable(
            "Triton trade projection requires CUDA compute capability 8.0 or newer"
        )

    output = torch.empty_like(value)
    block_size = triton.next_power_of_2(assets)
    try:
        with torch.cuda.device(value.device):
            _l1_box_neutral_kernel[(1,)](
                value,
                lower,
                upper,
                output,
                threshold,
                assets,
                bisection_steps,
                block_size,
                num_warps=8 if block_size >= 1024 else 4,
            )
    except TritonError as error:
        raise TritonTradeProjectionUnavailable(str(error)) from error
    return output
