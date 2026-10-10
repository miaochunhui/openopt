"""Verify actual solver placement; CUDA compatibility is not a speedup claim."""

import pytest
import torch

from examples.portfolio_qp import run
from openopt import FactorRiskModel, PortfolioProblem, SolveConfig, solve


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_explicit_solver_device_is_preserved(device):
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable; GPU execution is unverified")
    target = torch.device(device)
    dtype = torch.float64
    problem = PortfolioProblem(
        alpha=torch.tensor([0.01, 0.02], device=target, dtype=dtype),
        risk=FactorRiskModel(
            exposures=torch.zeros((2, 1), device=target, dtype=dtype),
            factor_variance=torch.ones(1, device=target, dtype=dtype),
            specific_variance=torch.ones(2, device=target, dtype=dtype),
        ),
        lower=torch.zeros(2, device=target, dtype=dtype),
        upper=torch.ones(2, device=target, dtype=dtype),
        risk_aversion=2.0,
    )
    result = solve(problem, SolveConfig(iterations=300, tolerance=1e-9))
    assert result.converged
    assert result.weights.device == target
    assert result.objective.device == target
    assert result.weights.dtype == dtype
    assert torch.isfinite(result.weights).all()
    assert result.device == device
    assert float((result.weights.sum(dim=1) - 1.0).abs().max()) < 1e-6


def test_example_rejects_explicit_unavailable_gpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requested CUDA device is unavailable"):
        run("cuda:0")
