from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpo import FactorRiskModel, PortfolioProblem
from openpo.projection import project_bounded_simplex


def build_problem(assets: int = 96, factors: int = 12, batch: int = 24, seed: int = 20260809) -> PortfolioProblem:
    generator = torch.Generator().manual_seed(seed)
    exposures = torch.randn(assets, factors, generator=generator) * 0.05
    risk = FactorRiskModel(
        exposures=exposures,
        factor_variance=torch.linspace(0.05, 0.25, factors),
        specific_variance=torch.linspace(0.02, 0.10, assets),
    )
    alpha = torch.randn(batch, assets, generator=generator) * 0.002
    return PortfolioProblem(
        alpha=alpha,
        risk=risk,
        lower=torch.zeros(assets),
        upper=torch.full((assets,), min(0.05, 10.0 / assets)),
        risk_aversion=10.0,
    )


def build_projection_inputs(problem: PortfolioProblem) -> torch.Tensor:
    batch, assets = problem.batched_alpha.shape
    values = torch.linspace(
        -0.35,
        0.85,
        steps=batch * assets,
        dtype=problem.alpha.dtype,
        device=problem.alpha.device,
    )
    return values.reshape(batch, assets)


def assert_projection_close(
    *,
    problem: PortfolioProblem,
    values: torch.Tensor,
    bisection_steps: int,
    reference_steps: int = 40,
) -> None:
    projected = project_bounded_simplex(
        values,
        problem.lower,
        problem.upper,
        budget=problem.budget,
        bisection_steps=bisection_steps,
    )
    reference = project_bounded_simplex(
        values,
        problem.lower,
        problem.upper,
        budget=problem.budget,
        bisection_steps=reference_steps,
    )

    residual = (projected.sum(dim=1) - problem.budget).abs()
    reference_residual = (reference.sum(dim=1) - problem.budget).abs()

    assert float(residual.max()) <= 3e-7
    assert float(reference_residual.max()) <= 3e-7
    assert torch.all(projected >= problem.lower - 1e-7)
    assert torch.all(projected <= problem.upper + 1e-7)
    assert torch.allclose(projected, reference, atol=2e-6 if bisection_steps == 16 else 5e-7, rtol=0.0)


@pytest.mark.parametrize("bisection_steps", [16, 20])
def test_projection_correction_matches_40_step_reference_on_cpu_and_cuda(
    bisection_steps: int,
) -> None:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")

    for device in devices:
        problem = build_problem().to(device)
        values = build_projection_inputs(problem)
        assert_projection_close(
            problem=problem,
            values=values,
            bisection_steps=bisection_steps,
        )


@pytest.mark.parametrize("bisection_steps", [16, 20])
@pytest.mark.parametrize("budget_source", ["lower", "upper"])
def test_projection_hits_exact_budget_boundaries(
    budget_source: str,
    bisection_steps: int,
) -> None:
    values = torch.tensor([[0.9, -0.4, 0.1, 0.7]], dtype=torch.float32)
    lower = torch.tensor([0.1, 0.05, 0.2, 0.15], dtype=torch.float32)
    upper = torch.tensor([0.4, 0.45, 0.5, 0.35], dtype=torch.float32)
    budget = float(lower.sum() if budget_source == "lower" else upper.sum())
    expected = lower.unsqueeze(0) if budget_source == "lower" else upper.unsqueeze(0)

    result = project_bounded_simplex(
        values,
        lower,
        upper,
        budget=budget,
        bisection_steps=bisection_steps,
    )

    assert torch.allclose(result, expected, atol=2e-7, rtol=0.0)
    assert torch.allclose(result.sum(dim=1), torch.tensor([budget]), atol=2e-7, rtol=0.0)
    assert torch.all(result >= lower - 1e-7)
    assert torch.all(result <= upper + 1e-7)


def test_capacity_correction_handles_coarse_bisection_without_free_variables() -> None:
    values = torch.tensor([[10.0, 9.0, -8.0, -9.0]], dtype=torch.float32)
    lower = torch.zeros(4)
    upper = torch.tensor([0.35, 0.35, 0.35, 0.35])
    result = project_bounded_simplex(
        values, lower, upper, budget=1.0, bisection_steps=1
    )
    assert torch.allclose(result.sum(dim=1), torch.ones(1), atol=2e-7)
    assert torch.all(result >= lower)
    assert torch.all(result <= upper)
