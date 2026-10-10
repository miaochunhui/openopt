import pytest
import torch

from openpo import FactorRiskModel, PortfolioProblem, SolveConfig, solve, solve_cardinality


def test_cardinality_selected_capacity_shortfall_fails_loud() -> None:
    assets = 4
    problem = PortfolioProblem(
        alpha=torch.tensor([0.04, 0.03, 0.02, 0.01]),
        risk=FactorRiskModel(
            exposures=torch.zeros(assets, 1),
            factor_variance=torch.ones(1),
            specific_variance=torch.full((assets,), 0.2),
        ),
        lower=torch.zeros(assets),
        upper=torch.full((assets,), 0.31),
        risk_aversion=1.0,
    )

    assert problem.lower.sum().item() <= problem.budget <= problem.upper.sum().item()

    relaxed = solve(problem, SolveConfig(iterations=30, tolerance=1e-7))
    selected = torch.topk(relaxed.weights[0], k=3).indices
    selected_capacity = float(problem.upper[selected].sum())
    assert selected_capacity < problem.budget

    with pytest.raises(ValueError, match=r"selected.*capacity.*budget"):
        solve_cardinality(
            problem,
            max_holdings=3,
            minimum_holding=0.0,
            config=SolveConfig(iterations=30, tolerance=1e-7),
        )
