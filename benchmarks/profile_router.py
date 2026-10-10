from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import clarabel
import numpy as np
import osqp
import scipy
import scipy.sparse as sparse
import torch

from openpo import FactorRiskModel, PortfolioProblem, SolveConfig, solve
from openpo.exact_solver import configured_clarabel_settings
from openpo.router import RouterThresholds, route_backend


QUALITY_GAP_LIMIT = 1e-6

SMOKE_ASSETS = (64, 512, 1024)
SMOKE_BATCHES = (8, 128)
SMOKE_FACTORS = (16,)

FULL_ASSETS = (64, 128, 256, 512, 1024, 2048)
FULL_BATCHES = (8, 32, 128, 512, 2048)
FULL_FACTORS = (8, 16, 32, 64)


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
    generator = np.random.default_rng(seed + assets * 1_000_003 + factors * 1_009 + batch * 97)
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


def percentile(values: Iterable[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(list(values), dtype=float), quantile))


def synchronize(device: str) -> None:
    if not device.startswith("cuda:"):
        return
    torch.cuda.synchronize(int(device.split(":", 1)[1]))


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
    summary: dict[str, object] = {
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
        "hot_batch_p50_ms": 1000.0 * batch_seconds,
        "hot_batch_p95_ms": 1000.0 * percentile(batch_timings, 0.95),
        **residuals,
    }
    if per_portfolio_timings:
        summary["per_portfolio_p50_ms"] = 1000.0 * statistics.median(
            per_portfolio_timings
        )
        summary["per_portfolio_p95_ms"] = 1000.0 * percentile(
            per_portfolio_timings, 0.95
        )
    else:
        summary["per_portfolio_p50_ms"] = None
        summary["per_portfolio_p95_ms"] = None
    return summary


def osqp_solver(case: Case) -> tuple[osqp.OSQP, float]:
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
    constraints = sparse.vstack([budget_row, bound_rows, factor_rows], format="csc")
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
    batch_timings: list[float] = []
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
    constraints = sparse.vstack([budget_row, factor_rows, identity, -identity], format="csc")
    right_hand_side = np.concatenate(([case.budget], np.zeros(case.factors), case.upper, -case.lower))
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
    batch_timings: list[float] = []
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


def run_openpo(
    case: Case,
    device: str,
    iterations: int,
    bisection_steps: int,
    repeats: int,
) -> dict[str, object]:
    config = SolveConfig(
        iterations=iterations,
        tolerance=0,
        bisection_steps=bisection_steps,
    )
    cold_start = time.perf_counter()
    cold = solve(openpo_problem(case, device, case.alpha[:1]), config)
    del cold
    cold_seconds = time.perf_counter() - cold_start
    batch_problem = openpo_problem(case, device, case.alpha)
    solve(batch_problem, SolveConfig(iterations=5, tolerance=0, bisection_steps=8))
    synchronize(device)
    batch_timings: list[float] = []
    result = None
    for _ in range(repeats):
        synchronize(device)
        batch_start = time.perf_counter()
        result = solve(batch_problem, config)
        synchronize(device)
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


def build_grid(mode: str, assets: list[int] | None, batches: list[int] | None, factors: list[int] | None) -> tuple[list[int], list[int], list[int]]:
    if assets is None:
        assets = list(SMOKE_ASSETS if mode == "smoke" else FULL_ASSETS)
    if batches is None:
        batches = list(SMOKE_BATCHES if mode == "smoke" else FULL_BATCHES)
    if factors is None:
        factors = list(SMOKE_FACTORS if mode == "smoke" else FULL_FACTORS)
    if not assets or not batches or not factors:
        raise ValueError("assets, batches, and factors must each contain at least one value")
    if any(value <= 0 for value in assets + batches + factors):
        raise ValueError("assets, batches, and factors must be positive integers")
    return sorted(set(assets)), sorted(set(batches)), sorted(set(factors))


def quality_ok(result: dict[str, object]) -> bool:
    return (
        result["success_rate"] == 1.0
        and float(result["max_budget_error"]) <= 2e-5
        and float(result["max_lower_violation"]) <= 2e-6
        and float(result["max_upper_violation"]) <= 2e-6
        and (
            result["median_objective_gap"] is None
            or float(result["max_objective_gap"]) <= QUALITY_GAP_LIMIT
        )
    )


def observed_backend(case_results: dict[str, dict[str, object]], gpu_count: int) -> dict[str, object]:
    if gpu_count == 0:
        candidates = {
            name: result
            for name, result in case_results.items()
            if name != "openpo_cuda"
        }
    elif gpu_count == 1:
        candidates = case_results
    else:
        raise ValueError("observed_backend only supports gpu_count 0 or 1")
    eligible = {
        name: result
        for name, result in candidates.items()
        if quality_ok(result)
    }
    if not eligible:
        raise RuntimeError("no quality-eligible solver remained for backend comparison")
    backend, result = max(
        eligible.items(),
        key=lambda item: (float(item[1]["portfolios_per_second"]), -float(item[1]["cold_seconds"])),
    )
    return {"backend": backend, "result": result}


def route_match_score(
    case_rows: list[dict[str, object]],
    thresholds: RouterThresholds,
) -> dict[str, object]:
    matches = 0
    weighted_ratio_sum = 0.0
    decisions = []
    for row in case_rows:
        decision = route_backend(
            assets=int(row["assets"]),
            batch=int(row["batch"]),
            gpu_count=int(row["gpu_count"]),
            gpu_is_warm=int(row["gpu_count"]) > 0,
            thresholds=thresholds,
        )
        decisions.append(
            {
                "assets": row["assets"],
                "batch": row["batch"],
                "factors": row["factors"],
                "gpu_count": row["gpu_count"],
                "predicted_backend": decision.backend,
                "predicted_reason": decision.reason,
                "observed_backend": row["observed_backend"],
                "observed_throughput": row["observed_throughput"],
                "predicted_throughput": row["throughputs"][decision.backend],
                "throughput_ratio": row["throughputs"][decision.backend]
                / row["observed_throughput"],
            }
        )
        if decision.backend == row["observed_backend"]:
            matches += 1
        weighted_ratio_sum += row["throughputs"][decision.backend] / row["observed_throughput"]
    total = len(case_rows)
    return {
        "exact_matches": matches,
        "total": total,
        "exact_match_rate": matches / total if total else None,
        "mean_throughput_ratio": weighted_ratio_sum / total if total else None,
        "decisions": decisions,
    }


def candidate_values(values: list[int], include_above: bool = False) -> list[int]:
    unique = sorted(set(values))
    if include_above:
        unique = unique + [unique[-1] + 1]
    return unique


def search_thresholds(case_rows: list[dict[str, object]], current_multi_gpu: int) -> tuple[RouterThresholds, dict[str, object]]:
    asset_candidates = candidate_values([int(row["assets"]) for row in case_rows if row["gpu_count"] == 0])
    batch_candidates = candidate_values([int(row["batch"]) for row in case_rows], include_above=True)
    best_thresholds: RouterThresholds | None = None
    best_score: tuple[float, float, int, int, int] | None = None
    best_details: dict[str, object] | None = None
    for gpu_batch_threshold in batch_candidates:
        multi_gpu_threshold = max(current_multi_gpu, gpu_batch_threshold)
        for gpu_asset_threshold in asset_candidates:
            for osqp_asset_threshold in asset_candidates:
                for clarabel_asset_threshold in asset_candidates:
                    if clarabel_asset_threshold < osqp_asset_threshold:
                        continue
                    thresholds = RouterThresholds(
                        gpu_batch_threshold=gpu_batch_threshold,
                        gpu_asset_threshold=gpu_asset_threshold,
                        multi_gpu_batch_threshold=multi_gpu_threshold,
                        osqp_asset_threshold=osqp_asset_threshold,
                        clarabel_asset_threshold=clarabel_asset_threshold,
                    )
                    details = route_match_score(case_rows, thresholds)
                    score = (
                        float(details["exact_match_rate"]),
                        float(details["mean_throughput_ratio"]),
                        -thresholds.gpu_batch_threshold,
                        -thresholds.gpu_asset_threshold,
                        -thresholds.clarabel_asset_threshold,
                    )
                    if best_score is None or score > best_score:
                        best_score = score
                        best_thresholds = thresholds
                        best_details = details
    if best_thresholds is None or best_details is None:
        raise RuntimeError("unable to derive routing thresholds from benchmark data")
    return best_thresholds, best_details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile router crossover thresholds.")
    parser.add_argument("mode", nargs="?", choices=("smoke", "full"), default=None)
    parser.add_argument("--mode", dest="mode_flag", choices=("smoke", "full"))
    parser.add_argument("--assets", type=int, nargs="+", default=None)
    parser.add_argument("--batches", type=int, nargs="+", default=None)
    parser.add_argument("--factors", type=int, nargs="+", default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--cpu-bisection-steps", type=int, default=None)
    parser.add_argument("--cuda-bisection-steps", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path in addition to stdout.",
    )
    return parser.parse_args()


def validate_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for router profiling but torch.cuda.is_available() is false")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("CUDA is required for router profiling but no CUDA devices are visible")


def main() -> None:
    args = parse_args()
    mode = args.mode_flag or args.mode or "smoke"
    assets, batches, factors = build_grid(mode, args.assets, args.batches, args.factors)
    iterations = args.iterations if args.iterations is not None else (60 if mode == "smoke" else 200)
    cpu_bisection_steps = args.cpu_bisection_steps if args.cpu_bisection_steps is not None else 24
    cuda_bisection_steps = args.cuda_bisection_steps if args.cuda_bisection_steps is not None else 20
    repeats = args.repeats if args.repeats is not None else (1 if mode == "smoke" else 5)
    if iterations <= 0 or cpu_bisection_steps <= 0 or cuda_bisection_steps <= 0 or repeats <= 0:
        raise ValueError("iterations, bisection steps, and repeats must be positive")
    validate_cuda()
    started = time.perf_counter()
    cases = []
    for asset_count in assets:
        for factor_count in factors:
            for batch_size in batches:
                case = build_case(asset_count, factor_count, batch_size, args.seed)
                case_results = {
                    "osqp": run_osqp(case, repeats),
                    "clarabel": run_clarabel(case, repeats),
                    "openpo_cpu": run_openpo(case, "cpu", iterations, cpu_bisection_steps, repeats),
                    "openpo_cuda": run_openpo(case, "cuda:0", iterations, cuda_bisection_steps, repeats),
                }
                solver_results = [case_results[name] for name in ("osqp", "clarabel", "openpo_cpu", "openpo_cuda")]
                add_quality_gaps(solver_results)
                if not all(quality_ok(result) for result in case_results.values()):
                    failing = [
                        {
                            "solver": name,
                            "success_rate": result["success_rate"],
                            "max_budget_error": result["max_budget_error"],
                            "max_lower_violation": result["max_lower_violation"],
                            "max_upper_violation": result["max_upper_violation"],
                            "median_objective_gap": result["median_objective_gap"],
                            "max_objective_gap": result["max_objective_gap"],
                        }
                        for name, result in case_results.items()
                        if not quality_ok(result)
                    ]
                    raise RuntimeError(
                        "benchmark quality gate failed for "
                        f"assets={asset_count} factors={factor_count} batch={batch_size}: "
                        f"{json.dumps(failing, ensure_ascii=False)}"
                    )
                cpu_choice = observed_backend(case_results, gpu_count=0)
                gpu_choice = observed_backend(case_results, gpu_count=1)
                case_row = {
                    "assets": asset_count,
                    "factors": factor_count,
                    "batch": batch_size,
                    "input_sha256": case_sha256(case),
                    "risk_model": "shared diagonal factor covariance plus specific risk",
                    "reference_formulation": "lifted sparse QP with explicit factor exposure variables",
                    "constraints": "long-only budget and box bounds",
                    "results": solver_results,
                    "best_cpu_backend": cpu_choice["backend"],
                    "best_gpu_backend": gpu_choice["backend"],
                    "best_cpu_throughput": cpu_choice["result"]["portfolios_per_second"],
                    "best_gpu_throughput": gpu_choice["result"]["portfolios_per_second"],
                    "solver_throughputs": {
                        name: result["portfolios_per_second"]
                        for name, result in case_results.items()
                    },
                }
                cases.append(
                    {
                        **case_row,
                        "current_route_cpu": route_backend(
                            assets=asset_count,
                            batch=batch_size,
                            gpu_count=0,
                        ).backend,
                        "current_route_gpu": route_backend(
                            assets=asset_count,
                            batch=batch_size,
                            gpu_count=1,
                            gpu_is_warm=True,
                        ).backend,
                    }
                )
    current_thresholds = RouterThresholds()
    case_rows = []
    for case in cases:
        case_rows.append(
            {
                "assets": case["assets"],
                "batch": case["batch"],
                "factors": case["factors"],
                "gpu_count": 0,
                "observed_backend": case["best_cpu_backend"],
                "observed_throughput": case["best_cpu_throughput"],
                "throughputs": case["solver_throughputs"],
            }
        )
        case_rows.append(
            {
                "assets": case["assets"],
                "batch": case["batch"],
                "factors": case["factors"],
                "gpu_count": 1,
                "observed_backend": case["best_gpu_backend"],
                "observed_throughput": case["best_gpu_throughput"],
                "throughputs": case["solver_throughputs"],
            }
        )
    recommended_thresholds, recommended_details = search_thresholds(
        case_rows, current_thresholds.multi_gpu_batch_threshold
    )
    current_details = route_match_score(case_rows, current_thresholds)
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
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
        "scan": {
            "assets": assets,
            "batches": batches,
            "factors": factors,
            "iterations": iterations,
            "cpu_bisection_steps": cpu_bisection_steps,
            "cuda_bisection_steps": cuda_bisection_steps,
            "repeats": repeats,
            "seed": args.seed,
            "quality_gap_limit": QUALITY_GAP_LIMIT,
        },
        "current_thresholds": {
            "gpu_batch_threshold": current_thresholds.gpu_batch_threshold,
            "gpu_asset_threshold": current_thresholds.gpu_asset_threshold,
            "multi_gpu_batch_threshold": current_thresholds.multi_gpu_batch_threshold,
            "osqp_asset_threshold": current_thresholds.osqp_asset_threshold,
            "clarabel_asset_threshold": current_thresholds.clarabel_asset_threshold,
        },
        "recommended_thresholds": {
            "gpu_batch_threshold": recommended_thresholds.gpu_batch_threshold,
            "gpu_asset_threshold": recommended_thresholds.gpu_asset_threshold,
            "multi_gpu_batch_threshold": recommended_thresholds.multi_gpu_batch_threshold,
            "osqp_asset_threshold": recommended_thresholds.osqp_asset_threshold,
            "clarabel_asset_threshold": recommended_thresholds.clarabel_asset_threshold,
        },
        "route_quality": {
            "current": current_details,
            "recommended": recommended_details,
        },
        "cases": cases,
        "wall_seconds": time.perf_counter() - started,
    }
    rendered = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    print(rendered, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
