from __future__ import annotations

import argparse
import json
import platform

import torch

from openpo import FactorRiskModel, PortfolioProblem, SolveConfig, solve, solve_multi_gpu


def build_problem(assets: int, factors: int, batch: int, seed: int) -> PortfolioProblem:
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


def verify(problem: PortfolioProblem, weights: torch.Tensor) -> dict[str, float]:
    return {
        "max_budget_error": float((weights.sum(dim=1) - problem.budget).abs().max()),
        "max_lower_violation": float(torch.clamp(problem.lower - weights, min=0).max()),
        "max_upper_violation": float(torch.clamp(weights - problem.upper, min=0).max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, default=1000)
    parser.add_argument("--factors", type=int, default=40)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--devices", default="cpu,cuda:0,cuda:1")
    parser.add_argument("--seed", type=int, default=20260809)
    args = parser.parse_args()
    problem = build_problem(args.assets, args.factors, args.batch, args.seed)
    config = SolveConfig(iterations=args.iterations, tolerance=0, bisection_steps=32)
    requested = list(dict.fromkeys(args.devices.split(",")))
    rows = []
    cpu_weights = None

    for label in requested:
        if label == "cpu":
            solve(problem, SolveConfig(iterations=5, tolerance=0, bisection_steps=8))
            result = solve(problem, config)
            cpu_weights = result.weights.cpu()
        elif "," not in label and label.startswith("cuda:"):
            index = int(label.split(":", 1)[1])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise RuntimeError(f"requested CUDA device is unavailable: {label}")
            device_problem = problem.to(label)
            solve(device_problem, SolveConfig(iterations=5, tolerance=0, bisection_steps=8))
            result = solve(device_problem, config)
        else:
            raise ValueError(f"invalid device entry: {label}")
        weights = result.weights.cpu()
        row = {
                "device": label,
                "seconds": result.elapsed_seconds,
                "portfolios_per_second": args.batch / result.elapsed_seconds,
                "median_objective": float(result.objective.median()),
                **verify(problem, weights),
            }
        if cpu_weights is not None:
            row["max_weight_delta_vs_cpu"] = float((weights - cpu_weights).abs().max())
        rows.append(row)

    unique_cuda = list(dict.fromkeys(device for device in requested if device.startswith("cuda:")))
    if len(unique_cuda) > 1:
        solve_multi_gpu(
            problem,
            unique_cuda,
            SolveConfig(iterations=5, tolerance=0, bisection_steps=8),
        )
        result = solve_multi_gpu(problem, unique_cuda, config)
        row = {
                "device": "+".join(unique_cuda),
                "seconds": result.elapsed_seconds,
                "portfolios_per_second": args.batch / result.elapsed_seconds,
                "median_objective": float(result.objective.median()),
                **verify(problem, result.weights),
            }
        if cpu_weights is not None:
            row["max_weight_delta_vs_cpu"] = float((result.weights - cpu_weights).abs().max())
        rows.append(row)

    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "torch_arch_list": torch.cuda.get_arch_list(),
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "problem": vars(args),
        "results": rows,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
