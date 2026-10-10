from __future__ import annotations

import pytest
import torch

from openpo import FactorRiskModel, PortfolioProblem, SolveConfig, solve_multi_gpu


def _make_risk_model(*, device: str = "cpu", dtype: torch.dtype = torch.float32) -> FactorRiskModel:
    assets, factors = 4, 2
    exposures = torch.zeros((assets, factors), device=device, dtype=dtype)
    factor_variance = torch.ones(factors, device=device, dtype=dtype)
    specific_variance = torch.ones(assets, device=device, dtype=dtype)
    return FactorRiskModel(
        exposures=exposures,
        factor_variance=factor_variance,
        specific_variance=specific_variance,
    )


def _make_problem(
    *,
    alpha_device: str = "cpu",
    alpha_dtype: torch.dtype = torch.float32,
    risk_device: str = "cpu",
    risk_dtype: torch.dtype = torch.float32,
    batched_alpha: bool = False,
    batch_size: int = 2,
) -> PortfolioProblem:
    risk = _make_risk_model(device=risk_device, dtype=risk_dtype)
    alpha_shape = (batch_size, risk.assets) if batched_alpha else (risk.assets,)
    alpha = torch.ones(alpha_shape, device=alpha_device, dtype=alpha_dtype)
    return PortfolioProblem(
        alpha=alpha,
        risk=risk,
        lower=torch.zeros(risk.assets, device=alpha_device, dtype=alpha_dtype),
        upper=torch.full((risk.assets,), 0.5, device=alpha_device, dtype=alpha_dtype),
        risk_aversion=2.0,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_factor_risk_model_rejects_mixed_device() -> None:
    with pytest.raises(ValueError, match="same device"):
        FactorRiskModel(
            exposures=torch.zeros((4, 2), device="cpu"),
            factor_variance=torch.ones(2, device="cuda:0"),
            specific_variance=torch.ones(4, device="cuda:0"),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_portfolio_problem_rejects_mixed_device() -> None:
    risk = _make_risk_model(device="cuda:0")
    with pytest.raises(ValueError, match="same device"):
        PortfolioProblem(
            alpha=torch.ones(risk.assets, device="cpu"),
            risk=risk,
            lower=torch.zeros(risk.assets, device="cpu"),
            upper=torch.full((risk.assets,), 0.5, device="cpu"),
            risk_aversion=2.0,
        )


def test_factor_risk_model_rejects_mixed_dtype() -> None:
    with pytest.raises(ValueError, match="same dtype"):
        FactorRiskModel(
            exposures=torch.zeros((4, 2), dtype=torch.float32),
            factor_variance=torch.ones(2, dtype=torch.float64),
            specific_variance=torch.ones(4, dtype=torch.float32),
        )


def test_portfolio_problem_rejects_mixed_dtype() -> None:
    risk = _make_risk_model(dtype=torch.float32)
    with pytest.raises(ValueError, match="same dtype"):
        PortfolioProblem(
            alpha=torch.ones(risk.assets, dtype=torch.float64),
            risk=risk,
            lower=torch.zeros(risk.assets, dtype=torch.float32),
            upper=torch.full((risk.assets,), 0.5, dtype=torch.float32),
            risk_aversion=2.0,
        )


def test_solve_multi_gpu_rejects_small_batch_before_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    problem = _make_problem(batched_alpha=True, batch_size=1)

    with pytest.raises(ValueError, match="avoid empty shards"):
        solve_multi_gpu(problem, ["cuda:0", "cuda:1"], SolveConfig())
