"""Run one synthetic portfolio QP; fail on infeasibility or unavailable CUDA."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from openopt import FactorRiskModel, PortfolioProblem, SolveConfig, solve  # noqa: E402
from openpo.solver import objective  # noqa: E402


def run(device: str) -> dict[str, object]:
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("this example supports CPU or explicit CUDA devices")
    if target.type == "cuda":
        if target.index is None:
            raise ValueError("select an explicit CUDA device, such as cuda:0")
        if not torch.cuda.is_available() or target.index >= torch.cuda.device_count():
            raise RuntimeError(f"requested CUDA device is unavailable: {target}")
    assets = 32
    dtype = torch.float64
    problem = PortfolioProblem(
        alpha=torch.linspace(-0.01, 0.03, assets, dtype=dtype, device=target),
        risk=FactorRiskModel(
            exposures=torch.zeros((assets, 2), dtype=dtype, device=target),
            factor_variance=torch.ones(2, dtype=dtype, device=target),
            specific_variance=torch.full((assets,), 0.2, dtype=dtype, device=target),
        ),
        lower=torch.zeros(assets, dtype=dtype, device=target),
        upper=torch.full((assets,), 0.1, dtype=dtype, device=target),
        risk_aversion=2.0,
    )
    result = solve(problem, SolveConfig(iterations=300, tolerance=1e-9))
    weights = result.weights
    residual = max(
        float((weights.sum(dim=1) - problem.budget).abs().max()),
        float(torch.clamp(problem.lower - weights, min=0).max()),
        float(torch.clamp(weights - problem.upper, min=0).max()),
    )
    equal = torch.full_like(weights, 1.0 / assets)
    if not torch.isfinite(weights).all() or residual > 2e-6:
        raise RuntimeError(f"example feasibility failed: residual={residual}")
    if not torch.isfinite(result.objective).all():
        raise RuntimeError("example returned non-finite objective")
    if not result.converged:
        raise RuntimeError("example did not converge")
    if torch.any(result.objective > objective(problem, equal) + 1e-9):
        raise RuntimeError("example did not improve the equal-weight baseline")
    return {
        "verdict": "PASS",
        "source": "synthetic",
        "device": result.device,
        "converged": result.converged,
        "objective": result.objective.tolist(),
        "max_constraint_residual": residual,
        "weights": weights.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args.device), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
