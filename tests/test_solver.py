import pytest
import torch

from openpo import FactorRiskModel, PortfolioProblem, SolveConfig, solve, solve_cardinality
from openpo.projection import project_bounded_simplex


def make_problem(device: str = "cpu", batch: int = 3) -> PortfolioProblem:
    generator = torch.Generator(device=device).manual_seed(7)
    assets, factors = 32, 5
    exposures = torch.randn(assets, factors, generator=generator, device=device) * 0.1
    risk = FactorRiskModel(
        exposures=exposures,
        factor_variance=torch.linspace(0.1, 0.3, factors, device=device),
        specific_variance=torch.linspace(0.05, 0.15, assets, device=device),
    )
    alpha = torch.randn(batch, assets, generator=generator, device=device) * 0.01
    return PortfolioProblem(
        alpha=alpha,
        risk=risk,
        lower=torch.zeros(assets, device=device),
        upper=torch.full((assets,), 0.12, device=device),
        risk_aversion=5.0,
    )


def test_projection_respects_budget_and_bounds() -> None:
    values = torch.tensor([[0.9, -0.2, 0.4], [0.1, 0.2, 0.3]])
    lower = torch.zeros(3)
    upper = torch.tensor([0.6, 0.7, 0.8])
    result = project_bounded_simplex(values, lower, upper)
    assert torch.allclose(result.sum(dim=1), torch.ones(2), atol=1e-6)
    assert torch.all(result >= lower) and torch.all(result <= upper)


def test_solver_returns_feasible_improved_portfolios() -> None:
    problem = make_problem()
    result = solve(problem, SolveConfig(iterations=300, tolerance=1e-7))
    equal = torch.full_like(result.weights, 1.0 / result.weights.shape[1])
    from openpo.solver import objective

    assert torch.allclose(result.weights.sum(dim=1), torch.ones(3), atol=2e-5)
    assert torch.all(result.weights >= -1e-7)
    assert torch.all(result.weights <= 0.120001)
    assert torch.all(result.objective <= objective(problem, equal) + 1e-7)


def test_solver_matches_closed_form_isotropic_solution() -> None:
    assets = 10
    alpha = torch.linspace(-0.04, 0.05, assets)
    risk_aversion = 2.5
    specific_variance = 0.2
    problem = PortfolioProblem(
        alpha=alpha,
        risk=FactorRiskModel(
            exposures=torch.zeros(assets, 1),
            factor_variance=torch.ones(1),
            specific_variance=torch.full((assets,), specific_variance),
        ),
        lower=torch.zeros(assets),
        upper=torch.full((assets,), 0.3),
        risk_aversion=risk_aversion,
    )
    expected = project_bounded_simplex(
        (alpha / (risk_aversion * specific_variance)).unsqueeze(0),
        problem.lower,
        problem.upper,
    )
    result = solve(problem, SolveConfig(iterations=30, tolerance=1e-7))
    assert torch.allclose(result.weights, expected, atol=2e-6)


def test_cardinality_and_minimum_holding() -> None:
    batched = make_problem(batch=1)
    problem = PortfolioProblem(
        alpha=batched.alpha[0],
        risk=batched.risk,
        lower=batched.lower,
        upper=torch.full_like(batched.upper, 0.3),
        risk_aversion=batched.risk_aversion,
    )
    result = solve_cardinality(
        problem, max_holdings=8, minimum_holding=0.02, config=SolveConfig(iterations=250)
    )
    active = result.weights[0] > 1e-6
    assert int(active.sum()) <= 8
    assert torch.all(result.weights[0, active] >= 0.01999)
    assert result.weights.sum().item() == pytest.approx(1.0, abs=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_gpu_parity() -> None:
    cpu_problem = make_problem(batch=2)
    config = SolveConfig(iterations=250, tolerance=0)
    cpu = solve(cpu_problem, config)
    gpu = solve(cpu_problem.to("cuda:0"), config)
    assert torch.allclose(cpu.weights, gpu.weights.cpu(), atol=3e-5, rtol=3e-4)
