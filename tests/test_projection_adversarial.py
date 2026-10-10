from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpo.projection import project_bounded_simplex

REFERENCE_STEPS = 48
STEP_CASES = (
    (16, 3.0e-4),
    (20, 2.0e-4),
    (24, 1.5e-4),
)
CUDA_AVAILABLE = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
DEVICE_CASES = (
    "cpu",
    pytest.param(
        "cuda:0",
        marks=pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable"),
    ),
)
ADVERSARIAL_CASES = (
    (20260809, 0.0, 2),
    (20260810, 0.99, 2),
    (20260811, 0.05, 3),
    (20260812, 0.5, 3),
)


def build_adversarial_projection_case(
    *,
    seed: int,
    device: str,
    budget_fraction: float,
    free_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    generator = torch.Generator().manual_seed(seed)
    batch = 3
    assets = 10

    lower = torch.rand(assets, generator=generator, dtype=torch.float32) * 0.04
    slack = torch.rand(assets, generator=generator, dtype=torch.float32) * 0.12
    free_indices = torch.randperm(assets, generator=generator)[:free_count]
    free_mask = torch.zeros(assets, dtype=torch.bool)
    free_mask[free_indices] = True
    slack = torch.where(free_mask, slack.clamp_min(0.01), torch.zeros_like(slack)).to(device)
    lower = lower.to(device)
    upper = lower + slack

    values = torch.randn(batch, assets, generator=generator, dtype=torch.float32) * 0.25
    free_bias = torch.tensor([1.8, -1.5, 0.9], dtype=torch.float32)[:free_count]
    values[:, free_mask] += free_bias
    fixed_count = int((~free_mask).sum())
    if fixed_count > 0:
        values[:, ~free_mask] += torch.linspace(
            -0.35,
            0.35,
            steps=fixed_count,
            dtype=torch.float32,
        )

    values = values.to(device)
    budget = float(lower.sum().item() + budget_fraction * (upper - lower).sum().item())
    return values, lower, upper, budget


def assert_feasible(
    projected: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    budget: float,
) -> None:
    residual = projected.sum(dim=1) - budget
    assert projected.dtype == torch.float32
    assert torch.isfinite(projected).all()
    assert float(residual.abs().max()) <= 6.0e-7
    assert torch.all(projected >= lower - 1.0e-6)
    assert torch.all(projected <= upper + 1.0e-6)


@pytest.mark.parametrize("device", DEVICE_CASES)
@pytest.mark.parametrize("seed, budget_fraction, free_count", ADVERSARIAL_CASES)
def test_projection_adversarial_cpu_cuda_against_48_steps(
    seed: int,
    budget_fraction: float,
    free_count: int,
    device: str,
) -> None:
    values, lower, upper, budget = build_adversarial_projection_case(
        seed=seed,
        device=device,
        budget_fraction=budget_fraction,
        free_count=free_count,
    )

    reference = project_bounded_simplex(
        values,
        lower,
        upper,
        budget=budget,
        bisection_steps=REFERENCE_STEPS,
    )
    assert_feasible(reference, lower, upper, budget)

    expected = None
    if budget_fraction == 0.0:
        expected = lower.unsqueeze(0).expand_as(reference)
    elif budget_fraction == 1.0:
        expected = upper.unsqueeze(0).expand_as(reference)

    if expected is not None:
        assert torch.allclose(reference, expected, atol=6.0e-7, rtol=0.0)

    for bisection_steps, atol in STEP_CASES:
        projected = project_bounded_simplex(
            values,
            lower,
            upper,
            budget=budget,
            bisection_steps=bisection_steps,
        )
        assert_feasible(projected, lower, upper, budget)
        assert torch.allclose(projected, reference, atol=atol, rtol=0.0)
