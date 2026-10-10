from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpo import BarraRiskData, SolveConfig, active_objective, solve
from openpo.solver import objective


def make_barra_data(device: str = "cpu") -> BarraRiskData:
    exposures = torch.tensor(
        [
            [0.10, -0.20],
            [0.30, 0.05],
            [-0.15, 0.25],
        ],
        device=device,
    )
    factor_covariance = torch.tensor(
        [
            [1.40, 0.12],
            [0.12, 0.90],
        ],
        device=device,
    )
    specific_variance = torch.tensor([0.20, 0.30, 0.40], device=device)
    benchmark = torch.tensor([0.30, 0.40, 0.30], device=device)
    return BarraRiskData(
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variance=specific_variance,
        benchmark=benchmark,
    )


def dense_q(data: BarraRiskData) -> torch.Tensor:
    return data.exposures @ data.factor_covariance @ data.exposures.T + torch.diag(
        data.specific_variance
    )


def test_barra_risk_model_matches_dense_matvec() -> None:
    data = make_barra_data()
    risk_model = data.to_factor_risk_model()
    weights = torch.tensor([[0.20, 0.50, 0.30], [0.10, 0.25, 0.65]])
    reference = weights @ dense_q(data).T

    torch.testing.assert_close(risk_model.matvec(weights), reference, atol=1e-7, rtol=0.0)


def test_barra_portfolio_problem_matches_active_objective_and_broadcasts() -> None:
    data = make_barra_data()
    alpha = torch.tensor(
        [
            [0.02, -0.01, 0.03],
            [0.01, 0.00, 0.04],
        ]
    )
    current = torch.tensor([0.25, 0.45, 0.30])
    lower = torch.zeros(3)
    upper = torch.full((3,), 0.8)
    problem = data.to_portfolio_problem(
        alpha,
        lower,
        upper,
        budget=1.0,
        risk_aversion=1.7,
        current=current,
        trade_cost=0.15,
    )

    assert problem.alpha.shape == (2, 3)
    assert problem.current is not None
    assert problem.current.shape == (2, 3)
    expected_adjusted = alpha + 1.7 * (dense_q(data) @ data.benchmark)
    torch.testing.assert_close(problem.alpha, expected_adjusted, atol=1e-7, rtol=0.0)

    weights = torch.tensor([[0.30, 0.40, 0.30], [0.18, 0.32, 0.50]])
    active = active_objective(
        data,
        alpha,
        weights,
        risk_aversion=1.7,
        current=current,
        trade_cost=0.15,
    )
    problem_objective = objective(problem, weights)
    constant = 0.5 * 1.7 * (data.benchmark * (dense_q(data) @ data.benchmark)).sum()
    torch.testing.assert_close(problem_objective + constant, active, atol=1e-7, rtol=0.0)


def test_barra_portfolio_problem_broadcasts_benchmark_over_alpha() -> None:
    data = make_barra_data()
    alpha = torch.tensor([0.02, -0.01, 0.03])
    benchmark = torch.stack([data.benchmark, data.benchmark + torch.tensor([0.05, -0.02, -0.03])])
    data = replace(data, benchmark=benchmark)
    problem = data.to_portfolio_problem(
        alpha,
        torch.zeros(3),
        torch.full((3,), 0.7),
        risk_aversion=2.0,
    )

    assert problem.alpha.shape == (2, 3)
    expected = alpha.unsqueeze(0) + 2.0 * (dense_q(data) @ benchmark.T).T
    torch.testing.assert_close(problem.alpha, expected, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        pytest.param(
            lambda data: replace(
                data,
                factor_covariance=data.factor_covariance.clone().index_put(
                    (torch.tensor(0), torch.tensor(1)), torch.tensor(0.5)
                ),
            ),
            "symmetric",
            id="non-symmetric",
        ),
        pytest.param(lambda data: replace(data, factor_covariance=torch.tensor([[1.0, 0.9], [0.9, 0.1]])), "positive definite", id="non-pd"),
        pytest.param(lambda data: replace(data, factor_covariance=data.factor_covariance.double()), "same dtype", id="dtype"),
        pytest.param(lambda data: replace(data, benchmark=data.benchmark[:-1]), "shape", id="shape"),
        pytest.param(lambda data: replace(data, exposures=data.exposures * torch.tensor(float("nan"))), "finite", id="nonfinite"),
    ],
)
def test_barra_risk_data_rejects_invalid_inputs(mutate, match: str) -> None:
    with pytest.raises((TypeError, ValueError), match=match):
        mutate(make_barra_data())


def test_barra_risk_data_to_device_round_trips() -> None:
    data = make_barra_data()
    moved = data.to("cpu")

    assert moved is not data
    torch.testing.assert_close(moved.exposures, data.exposures)
    torch.testing.assert_close(moved.factor_covariance, data.factor_covariance)
    torch.testing.assert_close(moved.specific_variance, data.specific_variance)
    torch.testing.assert_close(moved.benchmark, data.benchmark)


def test_barra_adapter_solves_small_problem() -> None:
    data = make_barra_data()
    alpha = torch.tensor([0.02, -0.01, 0.03])
    problem = data.to_portfolio_problem(
        alpha,
        torch.zeros(3),
        torch.full((3,), 0.8),
        risk_aversion=1.3,
        trade_cost=0.05,
    )
    result = solve(problem, SolveConfig(iterations=80, tolerance=1e-8))

    assert result.weights.shape == (1, 3)
    assert result.objective.shape == (1,)
    assert torch.isfinite(result.objective).all()
