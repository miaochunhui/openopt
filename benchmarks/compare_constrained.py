from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
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

import compare_open_source as open_source_benchmark
from generate_constrained_cases import build_case as build_constrained_case
from openpo import FactorRiskModel, LinearConstraints, PortfolioProblem, PrimalDualConfig, solve_constrained
from openpo.exact_solver import configured_clarabel_settings


@dataclass(frozen=True)
class BenchmarkCase:
    assets: int
    batch: int
    seed: int
    risk_case: open_source_benchmark.Case
    constrained_case: object

    @property
    def portfolio_mode(self) -> str:
        return "long_short" if self.constrained_case.long_short else "long_only"

    def input_sha256(self) -> str:
        digest = hashlib.sha256()
        for value in (
            self.assets,
            self.batch,
            self.seed,
            self.risk_case.assets,
            self.risk_case.factors,
            self.risk_case.batch,
            self.risk_case.risk_aversion,
            self.constrained_case.budget,
            self.constrained_case.long_short,
        ):
            _update_digest(digest, type(value).__name__, value)
        for values in (
            self.risk_case.exposures,
            self.risk_case.factor_variance,
            self.risk_case.specific_variance,
            self.risk_case.alpha,
            self.risk_case.lower,
            self.risk_case.upper,
            self.constrained_case.box_lower,
            self.constrained_case.box_upper,
            self.constrained_case.industry_loadings,
            self.constrained_case.industry_lower,
            self.constrained_case.industry_upper,
            self.constrained_case.style_loadings,
            self.constrained_case.style_lower,
            self.constrained_case.style_upper,
            self.constrained_case.market_loading,
            self.constrained_case.equality_target,
        ):
            _update_digest(digest, values.dtype.str, values)
        return digest.hexdigest()


@dataclass(frozen=True)
class OpenPOConfig:
    iterations: int
    tolerance: float
    constraint_tolerance: float
    cpu_bisection_steps: int
    cuda_bisection_steps: int


@dataclass(frozen=True)
class SolverContext:
    name: str
    setup_seconds: float
    cold_seconds: float
    batch_timings: list[float]
    per_portfolio_timings: list[float] | None
    weights: np.ndarray
    statuses: list[str]


def _update_digest(digest: hashlib._Hash, label: str, value: np.ndarray | int | float | bool | str) -> None:
    digest.update(label.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        return
    if isinstance(value, str):
        digest.update(value.encode("utf-8"))
        return
    if isinstance(value, bool):
        digest.update(np.asarray([value], dtype=np.bool_).tobytes())
        return
    if isinstance(value, int):
        digest.update(np.asarray([value], dtype=np.int64).tobytes())
        return
    if isinstance(value, float):
        digest.update(np.asarray([value], dtype=np.float64).tobytes())
        return
    raise TypeError(f"unsupported digest value type: {type(value)!r}")


def percentile(values: list[float], quantile: float) -> float:
    return float(np.quantile(np.asarray(values), quantile))


def build_risk_case(assets: int, batch: int, seed: int) -> open_source_benchmark.Case:
    factors = min(40, assets)
    return open_source_benchmark.build_case(assets=assets, factors=factors, batch=batch, seed=seed)


def build_benchmark_case(
    assets: int, batch: int, seed: int, *, long_short: bool = False
) -> BenchmarkCase:
    risk_case = build_risk_case(assets, batch, seed)
    constrained_case = build_constrained_case(
        seed=seed, long_short=long_short, assets=assets
    )
    return BenchmarkCase(
        assets=assets,
        batch=batch,
        seed=seed,
        risk_case=risk_case,
        constrained_case=constrained_case,
    )


def openpo_problem(case: BenchmarkCase, device: str, alpha: np.ndarray) -> PortfolioProblem:
    constrained = case.constrained_case
    risk = FactorRiskModel(
        exposures=torch.as_tensor(case.risk_case.exposures, dtype=torch.float32, device=device),
        factor_variance=torch.as_tensor(case.risk_case.factor_variance, dtype=torch.float32, device=device),
        specific_variance=torch.as_tensor(case.risk_case.specific_variance, dtype=torch.float32, device=device),
    )
    return PortfolioProblem(
        alpha=torch.as_tensor(alpha, dtype=torch.float32, device=device),
        risk=risk,
        lower=torch.as_tensor(constrained.box_lower, dtype=torch.float32, device=device),
        upper=torch.as_tensor(constrained.box_upper, dtype=torch.float32, device=device),
        risk_aversion=case.risk_case.risk_aversion,
        budget=constrained.budget,
    )


def linear_constraints(case: BenchmarkCase, device: str, dtype: torch.dtype) -> LinearConstraints:
    constrained = case.constrained_case
    return LinearConstraints(
        A_eq=torch.as_tensor(constrained.market_loading.reshape(1, -1), dtype=dtype, device=device),
        b_eq=torch.as_tensor([constrained.market_target], dtype=dtype, device=device),
        A_ineq=torch.as_tensor(
            np.vstack([constrained.industry_loadings, constrained.style_loadings]),
            dtype=dtype,
            device=device,
        ),
        lower=torch.as_tensor(
            np.concatenate([constrained.industry_lower, constrained.style_lower]),
            dtype=dtype,
            device=device,
        ),
        upper=torch.as_tensor(
            np.concatenate([constrained.industry_upper, constrained.style_upper]),
            dtype=dtype,
            device=device,
        ),
    )


def benchmark_objective(case: BenchmarkCase, weights: np.ndarray) -> np.ndarray:
    factor_load = weights @ case.risk_case.exposures
    risk = 0.5 * case.risk_case.risk_aversion * (
        np.sum(weights * weights * case.risk_case.specific_variance, axis=1)
        + np.sum(factor_load * factor_load * case.risk_case.factor_variance, axis=1)
    )
    return risk - np.sum(case.risk_case.alpha * weights, axis=1)


def feasibility(case: BenchmarkCase, weights: np.ndarray) -> dict[str, float]:
    constrained = case.constrained_case
    linear_values = np.vstack(
        [constrained.market_loading, constrained.industry_loadings, constrained.style_loadings]
    ) @ weights.T
    market_error = np.abs(linear_values[0] - constrained.market_target)
    interval_values = linear_values[1:]
    box_lower_violation = np.maximum(constrained.box_lower - weights, 0.0)
    box_upper_violation = np.maximum(weights - constrained.box_upper, 0.0)
    linear_lower = np.concatenate([constrained.industry_lower, constrained.style_lower])[:, None]
    linear_upper = np.concatenate([constrained.industry_upper, constrained.style_upper])[:, None]
    linear_lower_violation = np.maximum(linear_lower - interval_values, 0.0)
    linear_upper_violation = np.maximum(interval_values - linear_upper, 0.0)
    budget_error = np.abs(weights.sum(axis=1) - constrained.budget)
    return {
        "max_budget_error": float(np.max(budget_error)),
        "max_box_lower_violation": float(np.max(box_lower_violation)),
        "max_box_upper_violation": float(np.max(box_upper_violation)),
        "max_box_violation": float(np.max(np.maximum(box_lower_violation, box_upper_violation))),
        "max_linear_equality_error": float(np.max(market_error)),
        "max_linear_lower_violation": float(np.max(linear_lower_violation)),
        "max_linear_upper_violation": float(np.max(linear_upper_violation)),
        "max_linear_violation": float(np.max(np.maximum(linear_lower_violation, linear_upper_violation))),
    }


def build_lifted_qp(case: BenchmarkCase) -> tuple[sparse.csc_matrix, sparse.csc_matrix, np.ndarray, np.ndarray, np.ndarray]:
    assets = case.assets
    factors = case.risk_case.factors
    constrained = case.constrained_case
    p_diag = case.risk_case.risk_aversion * np.concatenate(
        (case.risk_case.specific_variance, case.risk_case.factor_variance)
    )
    p_matrix = sparse.diags(p_diag, format="csc")
    budget_row = sparse.hstack(
        [sparse.csc_matrix(np.ones((1, assets))), sparse.csc_matrix((1, factors))],
        format="csc",
    )
    market_row = sparse.hstack(
        [sparse.csc_matrix(constrained.market_loading.reshape(1, -1)), sparse.csc_matrix((1, factors))],
        format="csc",
    )
    factor_rows = sparse.hstack(
        [-sparse.csc_matrix(case.risk_case.exposures.T), sparse.eye(factors)], format="csc"
    )
    box_rows = sparse.hstack(
        [sparse.eye(assets), sparse.csc_matrix((assets, factors))],
        format="csc",
    )
    linear_rows = sparse.hstack(
        [sparse.csc_matrix(np.vstack([constrained.industry_loadings, constrained.style_loadings])), sparse.csc_matrix((constrained.industry_loadings.shape[0] + constrained.style_loadings.shape[0], factors))],
        format="csc",
    )
    matrix = sparse.vstack([budget_row, market_row, factor_rows, box_rows, -box_rows, linear_rows, -linear_rows], format="csc")
    lower = np.concatenate(
        [
            np.array([constrained.budget, constrained.market_target]),
            np.zeros(factors),
            constrained.box_lower,
            -constrained.box_upper,
            constrained.industry_lower,
            constrained.style_lower,
            -constrained.industry_upper,
            -constrained.style_upper,
        ]
    )
    upper = np.concatenate(
        [
            np.array([constrained.budget, constrained.market_target]),
            np.zeros(factors),
            constrained.box_upper,
            -constrained.box_lower,
            constrained.industry_upper,
            constrained.style_upper,
            -constrained.industry_lower,
            -constrained.style_lower,
        ]
    )
    linear_term = np.concatenate((-case.risk_case.alpha[0], np.zeros(factors)))
    return p_matrix, matrix, linear_term, lower, upper


def osqp_solver(case: BenchmarkCase) -> tuple[osqp.OSQP, float]:
    p_matrix, matrix, linear_term, lower, upper = build_lifted_qp(case)
    solver = osqp.OSQP()
    started = time.perf_counter()
    solver.setup(
        P=p_matrix,
        q=linear_term,
        A=matrix,
        l=lower,
        u=upper,
        eps_abs=1e-7,
        eps_rel=1e-7,
        max_iter=20_000,
        polishing=True,
        verbose=False,
    )
    return solver, time.perf_counter() - started


def clarabel_solver(case: BenchmarkCase) -> tuple[clarabel.DefaultSolver, float]:
    p_matrix, matrix, linear_term, lower, upper = build_lifted_qp(case)
    constrained = case.constrained_case
    eq_rows = 2 + case.risk_case.factors
    box_rows = constrained.box_lower.shape[0]
    linear_rows = constrained.industry_lower.shape[0] + constrained.style_lower.shape[0]
    cones = [
        clarabel.ZeroConeT(eq_rows),
        clarabel.NonnegativeConeT(2 * box_rows),
        clarabel.NonnegativeConeT(2 * linear_rows),
    ]
    settings = configured_clarabel_settings(clarabel)
    right_hand_side = np.concatenate(
        [
            np.array([constrained.budget, constrained.market_target]),
            np.zeros(case.risk_case.factors),
            constrained.box_upper,
            -constrained.box_lower,
            constrained.industry_upper,
            constrained.style_upper,
            -constrained.industry_lower,
            -constrained.style_lower,
        ]
    )
    started = time.perf_counter()
    solver = clarabel.DefaultSolver(
        p_matrix,
        linear_term,
        matrix,
        right_hand_side,
        cones,
        settings,
    )
    return solver, time.perf_counter() - started


def solve_openpo(case: BenchmarkCase, device: str, config: OpenPOConfig, repeats: int = 1) -> SolverContext:
    constrained = linear_constraints(case, device, torch.float32)
    problem = openpo_problem(case, device, case.risk_case.alpha)
    cold_start = time.perf_counter()
    cold_result = solve_constrained(
        openpo_problem(case, device, case.risk_case.alpha[:1]),
        constrained,
        PrimalDualConfig(
            iterations=config.iterations,
            tolerance=config.tolerance,
            constraint_tolerance=config.constraint_tolerance,
            bisection_steps=config.cuda_bisection_steps if device.startswith("cuda") else config.cpu_bisection_steps,
        ),
    )
    cold_seconds = time.perf_counter() - cold_start
    warm_problem = openpo_problem(case, device, case.risk_case.alpha)
    solve_constrained(
        warm_problem,
        constrained,
        PrimalDualConfig(
            iterations=max(5, config.iterations // 20),
            tolerance=0,
            bisection_steps=config.cuda_bisection_steps if device.startswith("cuda") else config.cpu_bisection_steps,
        ),
    )
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    batch_timings: list[float] = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = solve_constrained(
            problem,
            constrained,
            PrimalDualConfig(
                iterations=config.iterations,
                tolerance=config.tolerance,
                constraint_tolerance=config.constraint_tolerance,
                bisection_steps=config.cuda_bisection_steps if device.startswith("cuda") else config.cpu_bisection_steps,
            ),
        )
        if device.startswith("cuda"):
            torch.cuda.synchronize(device)
        batch_timings.append(time.perf_counter() - started)
    if result is None:
        raise RuntimeError("OpenPO benchmark completed without a result")
    weights = result.weights.detach().cpu().double().numpy()
    statuses = ["converged" if result.converged else "max_iterations"] * case.batch
    return SolverContext(
        name=f"OpenPO {device}",
        setup_seconds=0.0,
        cold_seconds=cold_seconds,
        batch_timings=batch_timings,
        per_portfolio_timings=None,
        weights=weights,
        statuses=statuses,
    )


def run_osqp(case: BenchmarkCase, repeats: int = 1) -> SolverContext:
    solver, setup_seconds = osqp_solver(case)
    cold_start = time.perf_counter()
    cold_result = solver.solve(raise_error=True)
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
        for alpha in case.risk_case.alpha:
            solver.update(q=np.concatenate((-alpha, np.zeros(case.risk_case.factors))))
            started = time.perf_counter()
            result = solver.solve(raise_error=True)
            timings.append(time.perf_counter() - started)
            weights.append(result.x[: case.assets])
            statuses.append(result.info.status)
        batch_timings.append(time.perf_counter() - batch_start)
    return SolverContext(
        name=f"OSQP {osqp.__version__} CPU",
        setup_seconds=setup_seconds,
        cold_seconds=cold_seconds,
        batch_timings=batch_timings,
        per_portfolio_timings=timings,
        weights=np.asarray(weights),
        statuses=statuses,
    )


def run_clarabel(case: BenchmarkCase, repeats: int = 1) -> SolverContext:
    solver, setup_seconds = clarabel_solver(case)
    cold_start = time.perf_counter()
    cold_result = solver.solve()
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
        for alpha in case.risk_case.alpha:
            solver.update(q=np.concatenate((-alpha, np.zeros(case.risk_case.factors))))
            started = time.perf_counter()
            result = solver.solve()
            timings.append(time.perf_counter() - started)
            weights.append(result.x[: case.assets])
            statuses.append(str(result.status))
        batch_timings.append(time.perf_counter() - batch_start)
    return SolverContext(
        name=f"Clarabel {clarabel.__version__} CPU",
        setup_seconds=setup_seconds,
        cold_seconds=cold_seconds,
        batch_timings=batch_timings,
        per_portfolio_timings=timings,
        weights=np.asarray(weights),
        statuses=statuses,
    )


def summarize(
    solver_name: str,
    case: BenchmarkCase,
    weights: np.ndarray,
    statuses: list[str],
    setup_seconds: float,
    cold_seconds: float,
    batch_timings: list[float],
    per_portfolio_timings: list[float] | None,
) -> dict[str, object]:
    objectives = benchmark_objective(case, weights)
    residuals = feasibility(case, weights)
    constrained = case.constrained_case
    constraints = linear_constraints(
        case,
        "cpu",
        torch.float64 if weights.dtype == np.float64 else torch.float32,
    )
    # The constraint residual must be computed on the same dtype/device as the weights.
    weight_tensor = torch.as_tensor(weights, dtype=constraints.dtype, device=constraints.device)
    linear_residual = constraints.residual(weight_tensor)
    interval_matrix = np.vstack(
        [constrained.industry_loadings, constrained.style_loadings]
    )
    interval_lower = np.concatenate(
        [constrained.industry_lower, constrained.style_lower]
    )[:, None]
    interval_upper = np.concatenate(
        [constrained.industry_upper, constrained.style_upper]
    )[:, None]
    interval_values = interval_matrix @ weights.T
    feasible_rows = (
        np.isfinite(weights).all(axis=1)
        & (np.abs(weights.sum(axis=1) - constrained.budget) <= 2e-5)
        & (np.max(np.maximum(constrained.box_lower - weights, 0.0), axis=1) <= 2e-6)
        & (np.max(np.maximum(weights - constrained.box_upper, 0.0), axis=1) <= 2e-6)
        & (np.abs((constrained.market_loading @ weights.T) - constrained.market_target) <= 2e-6)
        & (np.max(np.maximum(interval_lower - interval_values, 0.0), axis=0) <= 2e-6)
        & (np.max(np.maximum(interval_values - interval_upper, 0.0), axis=0) <= 2e-6)
    )
    batch_seconds = statistics.median(batch_timings)
    result: dict[str, object] = {
        "solver": solver_name,
        "setup_seconds": setup_seconds,
        "cold_seconds": cold_seconds,
        "batch_seconds": batch_seconds,
        "portfolios_per_second": case.batch / batch_seconds,
        "success_rate": float(np.mean(feasible_rows)),
        "median_objective": float(np.median(objectives[feasible_rows])) if np.any(feasible_rows) else None,
        "objectives": objectives.tolist(),
        "feasible": feasible_rows.tolist(),
        "statuses": statuses,
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        "hot_batch_p50_ms": 1000.0 * statistics.median(batch_timings),
        "hot_batch_p95_ms": 1000.0 * percentile(batch_timings, 0.95),
        **residuals,
        "max_linear_equality_error": float(linear_residual.max_abs_equality.detach().cpu().item()),
        "max_linear_lower_violation": float(
            0.0 if linear_residual.lower_violation is None else linear_residual.lower_violation.detach().cpu().amax().item()
        ),
        "max_linear_upper_violation": float(
            0.0 if linear_residual.upper_violation is None else linear_residual.upper_violation.detach().cpu().amax().item()
        ),
    }
    result["max_linear_violation"] = max(
        result["max_linear_equality_error"],
        result["max_linear_lower_violation"],
        result["max_linear_upper_violation"],
    )
    result["max_box_violation"] = max(
        result["max_box_lower_violation"],
        result["max_box_upper_violation"],
    )
    if per_portfolio_timings:
        result["per_portfolio_p50_ms"] = 1000.0 * statistics.median(per_portfolio_timings)
        result["per_portfolio_p95_ms"] = 1000.0 * percentile(per_portfolio_timings, 0.95)
    else:
        result["per_portfolio_p50_ms"] = None
        result["per_portfolio_p95_ms"] = None
    return result


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


def run_case(
    assets: int,
    batch: int,
    seed: int,
    iterations: int,
    cpu_bisection_steps: int,
    cuda_bisection_steps: int,
    repeats: int,
    long_short: bool = False,
) -> dict[str, object]:
    case = build_benchmark_case(assets, batch, seed, long_short=long_short)
    config = OpenPOConfig(
        iterations=iterations,
        tolerance=1e-6,
        constraint_tolerance=2e-6,
        cpu_bisection_steps=cpu_bisection_steps,
        cuda_bisection_steps=cuda_bisection_steps,
    )
    results = [
        summarize(
            context.name,
            case,
            context.weights,
            context.statuses,
            context.setup_seconds,
            context.cold_seconds,
            context.batch_timings,
            context.per_portfolio_timings,
        )
        for context in [
            run_osqp(case, repeats),
            run_clarabel(case, repeats),
            solve_openpo(case, "cpu", config, repeats),
        ]
    ]
    if torch.cuda.is_available():
        context = solve_openpo(case, "cuda:0", config, repeats)
        results.append(
            summarize(
                context.name,
                case,
                context.weights,
                context.statuses,
                context.setup_seconds,
                context.cold_seconds,
                context.batch_timings,
                context.per_portfolio_timings,
            )
        )
    add_quality_gaps(results)
    return {
        "assets": assets,
        "batch": batch,
        "portfolio_mode": case.portfolio_mode,
        "seed": seed,
        "input_sha256": case.input_sha256(),
        "risk_model": "shared diagonal factor covariance plus specific risk",
        "reference_formulation": "lifted sparse QP with explicit factor exposure variables and linear interval constraints",
        "constraints": (
            "market-neutral long-short box bounds, market equality, and industry/style intervals"
            if long_short
            else "budgeted long-only box bounds, market equality, and industry/style intervals"
        ),
        "results": results,
    }


def solver_environment() -> dict[str, object]:
    gpus = []
    for index in range(torch.cuda.device_count()):
        gpus.append(
            {
                "name": torch.cuda.get_device_name(index),
                "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(index))),
            }
        )
    try:
        openpo_version = importlib.metadata.version("openpo")
    except importlib.metadata.PackageNotFoundError:
        openpo_version = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "openpo": openpo_version,
        "cuda_runtime": torch.version.cuda,
        "osqp": osqp.__version__,
        "clarabel": clarabel.__version__,
        "cpu_threads": torch.get_num_threads(),
        "gpus": gpus,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, nargs="+", default=[16, 64, 250, 1000])
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    parser.add_argument(
        "--portfolio-modes",
        nargs="+",
        choices=("long_only", "long_short"),
        default=["long_only", "long_short"],
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--bisection-steps", type=int, default=None)
    parser.add_argument("--cpu-bisection-steps", type=int, default=24)
    parser.add_argument("--cuda-bisection-steps", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/constrained-comparison.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bisection_steps is not None:
        args.cpu_bisection_steps = args.bisection_steps
        args.cuda_bisection_steps = args.bisection_steps
    started = time.perf_counter()
    cases = [
        run_case(
            assets,
            batch,
            args.seed,
            args.iterations,
            args.cpu_bisection_steps,
            args.cuda_bisection_steps,
            args.repeats,
            long_short=portfolio_mode == "long_short",
        )
        for assets in args.assets
        for batch in args.batch
        for portfolio_mode in args.portfolio_modes
    ]
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": solver_environment(),
        "methodology": {
            "objective": "0.5 * risk_aversion * x'Qx - alpha'x",
            "success_tolerance": {
                "budget": 2e-5,
                "box": 2e-6,
                "linear": 2e-6,
            },
            "quality_reference": "best feasible objective returned by any compared solver",
            "openpo_mode": "bounded-simplex PDHG, residual-based stopping with an iteration cap, FP32",
            "openpo_budget_handling": "budget is enforced once by bounded-simplex projection; LinearConstraints omit the budget row",
            "openpo_projection_steps": {
                "cpu": args.cpu_bisection_steps,
                "cuda": args.cuda_bisection_steps,
            },
            "general_solver_mode": "FP64, warm q updates after one symbolic/numeric setup",
            "timing": f"median of {args.repeats} hot batch repeats; p95 from the same repeats",
        },
        "cases": cases,
        "wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.write_text(rendered)
    print(rendered, end="")
    if not all(result["success_rate"] == 1.0 for case in cases for result in case["results"]):
        raise RuntimeError("at least one solver returned an infeasible or non-finite portfolio")


if __name__ == "__main__":
    main()
