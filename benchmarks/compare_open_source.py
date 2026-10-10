from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import clarabel
import numpy as np
import osqp
import scipy
import scipy.sparse as sparse
import torch

from openpo import FactorRiskModel, PortfolioProblem, SolveConfig, solve
from openpo.exact_solver import configured_clarabel_settings


@dataclass(frozen=True)
class Case:
    assets: int
    factors: int
    batch: int
    exposures: np.ndarray
    factor_variance: np.ndarray
    specific_variance: np.ndarray
    alpha: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    risk_aversion: float
    budget: float

def build_case(assets: int, factors: int, batch: int, seed: int) -> Case:
    generator = np.random.default_rng(seed + assets)
    upper_value = min(0.05, 10.0 / assets)
    return Case(
        assets=assets,
        factors=factors,
        batch=batch,
        exposures=generator.normal(0.0, 0.05, size=(assets, factors)),
        factor_variance=np.linspace(0.05, 0.25, factors),
        specific_variance=np.linspace(0.02, 0.10, assets),
        alpha=generator.normal(0.0, 0.002, size=(batch, assets)),
        lower=np.zeros(assets),
        upper=np.full(assets, upper_value),
        risk_aversion=10.0,
        budget=1.0,
    )


def objective(case: Case, weights: np.ndarray) -> np.ndarray:
    factor_load = weights @ case.exposures
    risk = 0.5 * case.risk_aversion * (
        np.sum(weights * weights * case.specific_variance, axis=1)
        + np.sum(factor_load * factor_load * case.factor_variance, axis=1)
    )
    return risk - np.sum(case.alpha * weights, axis=1)


def feasibility(case: Case, weights: np.ndarray) -> dict[str, float]:
    return {
        "max_budget_error": float(np.max(np.abs(weights.sum(axis=1) - case.budget))),
        "max_lower_violation": float(np.max(np.maximum(case.lower - weights, 0.0))),
        "max_upper_violation": float(np.max(np.maximum(weights - case.upper, 0.0))),
    }


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values), quantile))


def summarize(
    name: str,
    case: Case,
    weights: np.ndarray,
    statuses: list[str],
    setup_seconds: float,
    cold_seconds: float,
    batch_timings: list[float],
    per_portfolio_timings: list[float] | None,
) -> dict[str, object]:
    objectives = objective(case, weights)
    residuals = feasibility(case, weights)
    feasible_rows = (
        np.isfinite(weights).all(axis=1)
        & (np.abs(weights.sum(axis=1) - case.budget) <= 2e-5)
        & (np.max(case.lower - weights, axis=1) <= 2e-6)
        & (np.max(weights - case.upper, axis=1) <= 2e-6)
    )
    batch_seconds = statistics.median(batch_timings)
    result: dict[str, object] = {
        "solver": name,
        "setup_seconds": setup_seconds,
        "cold_seconds": cold_seconds,
        "batch_seconds": batch_seconds,
        "portfolios_per_second": case.batch / batch_seconds,
        "success_rate": float(np.mean(feasible_rows)),
        "median_objective": float(np.median(objectives[feasible_rows]))
        if np.any(feasible_rows)
        else None,
        "objectives": objectives.tolist(),
        "feasible": feasible_rows.tolist(),
        "statuses": statuses,
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "hot_batch_p50_ms": 1000.0 * statistics.median(batch_timings),
        "hot_batch_p95_ms": 1000.0 * percentile(batch_timings, 0.95),
        **residuals,
    }
    if per_portfolio_timings:
        result["per_portfolio_p50_ms"] = 1000.0 * statistics.median(
            per_portfolio_timings
        )
        result["per_portfolio_p95_ms"] = 1000.0 * percentile(
            per_portfolio_timings, 0.95
        )
    else:
        result["per_portfolio_p50_ms"] = None
        result["per_portfolio_p95_ms"] = None
    return result


def openpo_problem(case: Case, device: str, alpha: np.ndarray) -> PortfolioProblem:
    return PortfolioProblem(
        alpha=torch.as_tensor(alpha, dtype=torch.float32, device=device),
        risk=FactorRiskModel(
            exposures=torch.as_tensor(case.exposures, dtype=torch.float32, device=device),
            factor_variance=torch.as_tensor(
                case.factor_variance, dtype=torch.float32, device=device
            ),
            specific_variance=torch.as_tensor(
                case.specific_variance, dtype=torch.float32, device=device
            ),
        ),
        lower=torch.as_tensor(case.lower, dtype=torch.float32, device=device),
        upper=torch.as_tensor(case.upper, dtype=torch.float32, device=device),
        risk_aversion=case.risk_aversion,
        budget=case.budget,
    )


def run_openpo(
    case: Case,
    device: str,
    iterations: int,
    bisection_steps: int = 20,
    repeats: int = 1,
) -> dict[str, object]:
    config = SolveConfig(
        iterations=iterations, tolerance=0, bisection_steps=bisection_steps
    )
    cold_start = time.perf_counter()
    cold = solve(openpo_problem(case, device, case.alpha[:1]), config)
    cold_seconds = time.perf_counter() - cold_start
    batch_problem = openpo_problem(case, device, case.alpha)
    solve(batch_problem, SolveConfig(iterations=5, tolerance=0, bisection_steps=8))
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    batch_timings = []
    result = None
    for _ in range(repeats):
        batch_start = time.perf_counter()
        result = solve(batch_problem, config)
        batch_timings.append(time.perf_counter() - batch_start)
    if result is None:
        raise RuntimeError("OpenPO benchmark completed without a result")
    weights = result.weights.detach().cpu().double().numpy()
    return summarize(
        f"OpenPO {device}",
        case,
        weights,
        ["fixed_iterations"] * case.batch,
        setup_seconds=0.0,
        cold_seconds=cold_seconds,
        batch_timings=batch_timings,
        per_portfolio_timings=None,
    )


def osqp_solver(case: Case) -> tuple[osqp.OSQP, float]:
    variables = case.assets + case.factors
    matrix = sparse.diags(
        case.risk_aversion
        * np.concatenate((case.specific_variance, case.factor_variance)),
        format="csc",
    )
    budget_row = sparse.hstack(
        [sparse.csc_matrix(np.ones((1, case.assets))), sparse.csc_matrix((1, case.factors))],
        format="csc",
    )
    bound_rows = sparse.hstack(
        [sparse.eye(case.assets), sparse.csc_matrix((case.assets, case.factors))],
        format="csc",
    )
    factor_rows = sparse.hstack(
        [-sparse.csc_matrix(case.exposures.T), sparse.eye(case.factors)], format="csc"
    )
    constraints = sparse.vstack(
        [budget_row, bound_rows, factor_rows],
        format="csc",
    )
    lower = np.concatenate(([case.budget], case.lower, np.zeros(case.factors)))
    upper = np.concatenate(([case.budget], case.upper, np.zeros(case.factors)))
    linear = np.concatenate((-case.alpha[0], np.zeros(case.factors)))
    solver = osqp.OSQP()
    started = time.perf_counter()
    solver.setup(
        P=matrix,
        q=linear,
        A=constraints,
        l=lower,
        u=upper,
        eps_abs=1e-7,
        eps_rel=1e-7,
        max_iter=20_000,
        polishing=True,
        verbose=False,
    )
    return solver, time.perf_counter() - started


def run_osqp(case: Case, repeats: int = 1) -> dict[str, object]:
    cold_start = time.perf_counter()
    cold_solver, setup_seconds = osqp_solver(case)
    cold_solver.solve(raise_error=True)
    cold_seconds = time.perf_counter() - cold_start
    solver, _ = osqp_solver(case)
    weights: list[np.ndarray] = []
    statuses: list[str] = []
    timings: list[float] = []
    batch_timings = []
    for _ in range(repeats):
        weights = []
        statuses = []
        batch_start = time.perf_counter()
        for alpha in case.alpha:
            solver.update(q=np.concatenate((-alpha, np.zeros(case.factors))))
            started = time.perf_counter()
            result = solver.solve(raise_error=True)
            timings.append(time.perf_counter() - started)
            weights.append(result.x[: case.assets])
            statuses.append(result.info.status)
        batch_timings.append(time.perf_counter() - batch_start)
    return summarize(
        f"OSQP {osqp.__version__} CPU",
        case,
        np.asarray(weights),
        statuses,
        setup_seconds,
        cold_seconds,
        batch_timings,
        timings,
    )


def clarabel_solver(case: Case) -> tuple[clarabel.DefaultSolver, float]:
    matrix = sparse.diags(
        case.risk_aversion
        * np.concatenate((case.specific_variance, case.factor_variance)),
        format="csc",
    )
    budget_row = sparse.hstack(
        [sparse.csc_matrix(np.ones((1, case.assets))), sparse.csc_matrix((1, case.factors))],
        format="csc",
    )
    factor_rows = sparse.hstack(
        [-sparse.csc_matrix(case.exposures.T), sparse.eye(case.factors)], format="csc"
    )
    identity = sparse.hstack(
        [sparse.eye(case.assets), sparse.csc_matrix((case.assets, case.factors))],
        format="csc",
    )
    constraints = sparse.vstack(
        [budget_row, factor_rows, identity, -identity], format="csc"
    )
    right_hand_side = np.concatenate(
        ([case.budget], np.zeros(case.factors), case.upper, -case.lower)
    )
    cones = [
        clarabel.ZeroConeT(1 + case.factors),
        clarabel.NonnegativeConeT(case.assets),
        clarabel.NonnegativeConeT(case.assets),
    ]
    settings = configured_clarabel_settings(clarabel)
    started = time.perf_counter()
    solver = clarabel.DefaultSolver(
        matrix,
        np.concatenate((-case.alpha[0], np.zeros(case.factors))),
        constraints,
        right_hand_side,
        cones,
        settings,
    )
    return solver, time.perf_counter() - started


def run_clarabel(case: Case, repeats: int = 1) -> dict[str, object]:
    cold_start = time.perf_counter()
    cold_solver, setup_seconds = clarabel_solver(case)
    cold_solver.solve()
    cold_seconds = time.perf_counter() - cold_start
    solver, _ = clarabel_solver(case)
    weights: list[np.ndarray] = []
    statuses: list[str] = []
    timings: list[float] = []
    batch_timings = []
    for _ in range(repeats):
        weights = []
        statuses = []
        batch_start = time.perf_counter()
        for alpha in case.alpha:
            solver.update(q=np.concatenate((-alpha, np.zeros(case.factors))))
            started = time.perf_counter()
            solution = solver.solve()
            timings.append(time.perf_counter() - started)
            weights.append(solution.x[: case.assets])
            statuses.append(str(solution.status))
        batch_timings.append(time.perf_counter() - batch_start)
    return summarize(
        f"Clarabel {clarabel.__version__} CPU",
        case,
        np.asarray(weights),
        statuses,
        setup_seconds,
        cold_seconds,
        batch_timings,
        timings,
    )


def add_quality_gaps(results: list[dict[str, object]]) -> None:
    objective_matrix = np.asarray([result.pop("objectives") for result in results], dtype=float)
    feasible_matrix = np.asarray([result.pop("feasible") for result in results], dtype=bool)
    masked = np.where(feasible_matrix, objective_matrix, np.inf)
    best = np.min(masked, axis=0)
    for index, result in enumerate(results):
        valid = feasible_matrix[index] & np.isfinite(best)
        gaps = objective_matrix[index, valid] - best[valid]
        result["median_objective_gap"] = float(np.median(gaps)) if gaps.size else None
        result["max_objective_gap"] = float(np.max(gaps)) if gaps.size else None


def case_sha256(case: Case) -> str:
    digest = hashlib.sha256()
    for values in (
        case.exposures,
        case.factor_variance,
        case.specific_variance,
        case.alpha,
        case.lower,
        case.upper,
    ):
        digest.update(values.tobytes())
    return digest.hexdigest()


def run_case(
    assets: int,
    factors: int,
    batch: int,
    seed: int,
    iterations: int,
    cpu_bisection_steps: int,
    cuda_bisection_steps: int,
    repeats: int,
) -> dict[str, object]:
    case = build_case(assets, factors, batch, seed)
    results = [
        run_osqp(case, repeats),
        run_clarabel(case, repeats),
        run_openpo(case, "cpu", iterations, cpu_bisection_steps, repeats),
    ]
    if torch.cuda.is_available():
        results.append(
            run_openpo(case, "cuda:0", iterations, cuda_bisection_steps, repeats)
        )
    add_quality_gaps(results)
    return {
        "assets": assets,
        "factors": factors,
        "batch": batch,
        "input_sha256": case_sha256(case),
        "risk_model": "shared diagonal factor covariance plus specific risk",
        "reference_formulation": "lifted sparse QP with explicit factor exposure variables",
        "constraints": "long-only budget and box bounds",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, nargs="+", default=[250, 1000])
    parser.add_argument("--factors", type=int, default=40)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--bisection-steps",
        type=int,
        default=None,
        help="Override both CPU and CUDA projection steps.",
    )
    parser.add_argument("--cpu-bisection-steps", type=int, default=24)
    parser.add_argument("--cuda-bisection-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmark-results/open-source-comparison.json")
    )
    args = parser.parse_args()
    if args.bisection_steps is not None:
        args.cpu_bisection_steps = args.bisection_steps
        args.cuda_bisection_steps = args.bisection_steps
    started = time.perf_counter()
    cases = [
        run_case(
            assets,
            min(args.factors, assets),
            args.batch,
            args.seed,
            args.iterations,
            args.cpu_bisection_steps,
            args.cuda_bisection_steps,
            args.repeats,
        )
        for assets in args.assets
    ]
    report = {
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "osqp": osqp.__version__,
            "clarabel": clarabel.__version__,
            "cpu_threads": torch.get_num_threads(),
            "gpus": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": ".".join(
                        map(str, torch.cuda.get_device_capability(index))
                    ),
                }
                for index in range(torch.cuda.device_count())
            ],
        },
        "methodology": {
            "objective": "0.5 * risk_aversion * x'Qx - alpha'x",
            "success_tolerance": {
                "budget": 2e-5,
                "bounds": 2e-6,
            },
            "quality_reference": "best feasible objective returned by any compared solver",
            "openpo_mode": "fixed iterations, FP32",
            "openpo_projection_steps": {
                "cpu": args.cpu_bisection_steps,
                "cuda": args.cuda_bisection_steps,
            },
            "general_solver_mode": "FP64, warm q updates after one symbolic/numeric setup",
            "timing": f"median of {args.repeats} hot batch repeats; p95 from same repeats",
        },
        "cases": cases,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if not all(
        result["success_rate"] == 1.0
        for case in cases
        for result in case["results"]
    ):
        raise RuntimeError("at least one solver returned an infeasible or non-finite portfolio")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
