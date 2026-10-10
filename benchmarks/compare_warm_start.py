from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import clarabel
import numpy as np
import osqp
import torch

import compare_constrained as constrained_benchmark
from openpo import PortfolioProblem, PrimalDualConfig, PrimalDualState, solve_constrained
from openpo.exact_solver import configured_clarabel_settings


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
    return float(np.quantile(np.asarray(values, dtype=float), quantile))


def resolve_devices(device_spec: str | None) -> list[str]:
    if device_spec is None or device_spec == "auto":
        devices = ["cpu"]
        if torch.cuda.is_available():
            devices.append("cuda:0")
        return devices

    devices = []
    for entry in device_spec.split(","):
        label = entry.strip()
        if not label:
            continue
        if label == "cuda":
            label = "cuda:0"
        if label == "cpu":
            devices.append(label)
            continue
        if label.startswith("cuda:"):
            index = int(label.split(":", 1)[1])
            if not torch.cuda.is_available() or index >= torch.cuda.device_count():
                raise RuntimeError(f"requested CUDA device is unavailable: {label}")
            devices.append(label)
            continue
        raise ValueError(f"invalid device entry: {entry!r}")

    if not devices:
        raise ValueError("at least one device must be provided")
    return list(dict.fromkeys(devices))


def build_alpha_path(
    *,
    assets: int,
    batch: int,
    horizon: int,
    seed: int,
    ar1_phi: float,
    alpha_scale: float,
) -> np.ndarray:
    if not 0.0 <= ar1_phi < 1.0:
        raise ValueError("ar1_phi must be in [0, 1)")
    if alpha_scale <= 0:
        raise ValueError("alpha_scale must be positive")
    generator = np.random.default_rng(seed)
    path = np.empty((horizon, batch, assets), dtype=np.float64)
    path[0] = generator.normal(0.0, alpha_scale, size=(batch, assets))
    innovation_scale = alpha_scale * math.sqrt(1.0 - ar1_phi * ar1_phi)
    for step in range(1, horizon):
        noise = generator.normal(0.0, innovation_scale, size=(batch, assets))
        path[step] = ar1_phi * path[step - 1] + noise
    return path


def path_sha256(alpha_path: np.ndarray, ar1_phi: float, alpha_scale: float) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, "ar1_phi", ar1_phi)
    _update_digest(digest, "alpha_scale", alpha_scale)
    _update_digest(digest, alpha_path.dtype.str, alpha_path)
    return digest.hexdigest()


def combined_sha256(case_sha256: str, alpha_path_sha256: str) -> str:
    digest = hashlib.sha256()
    _update_digest(digest, "case_sha256", case_sha256)
    _update_digest(digest, "alpha_path_sha256", alpha_path_sha256)
    return digest.hexdigest()


def objective_from_alpha(
    case: constrained_benchmark.BenchmarkCase,
    alpha_rows: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    factor_load = weights @ case.risk_case.exposures
    risk = 0.5 * case.risk_case.risk_aversion * (
        np.sum(weights * weights * case.risk_case.specific_variance, axis=1)
        + np.sum(factor_load * factor_load * case.risk_case.factor_variance, axis=1)
    )
    return risk - np.sum(alpha_rows * weights, axis=1)


def feasible_rows(
    case: constrained_benchmark.BenchmarkCase,
    weights: np.ndarray,
) -> np.ndarray:
    constrained = case.constrained_case
    interval_values = np.vstack(
        [constrained.industry_loadings, constrained.style_loadings]
    ) @ weights.T
    lower_bounds = np.concatenate(
        [constrained.industry_lower, constrained.style_lower]
    )[:, None]
    upper_bounds = np.concatenate(
        [constrained.industry_upper, constrained.style_upper]
    )[:, None]
    return (
        np.isfinite(weights).all(axis=1)
        & (np.abs(weights.sum(axis=1) - constrained.budget) <= 2e-5)
        & (np.max(np.maximum(constrained.box_lower - weights, 0.0), axis=1) <= 2e-6)
        & (np.max(np.maximum(weights - constrained.box_upper, 0.0), axis=1) <= 2e-6)
        & (np.abs((constrained.market_loading @ weights.T) - constrained.market_target) <= 2e-6)
        & (np.max(np.maximum(lower_bounds - interval_values, 0.0), axis=0) <= 2e-6)
        & (np.max(np.maximum(interval_values - upper_bounds, 0.0), axis=0) <= 2e-6)
    )


def summarize_result(
    *,
    solver_name: str,
    mode: str,
    device: str,
    case: constrained_benchmark.BenchmarkCase,
    alpha_rows: np.ndarray,
    weights: np.ndarray,
    statuses: list[str],
    iterations: list[int],
    step_seconds: list[float],
    setup_seconds: float,
) -> dict[str, object]:
    objectives = objective_from_alpha(case, alpha_rows, weights)
    feasible = feasible_rows(case, weights)
    residuals = constrained_benchmark.feasibility(case, weights)
    constraints = constrained_benchmark.linear_constraints(
        case,
        "cpu",
        torch.float64 if weights.dtype == np.float64 else torch.float32,
    )
    weight_tensor = torch.as_tensor(weights, dtype=constraints.dtype, device=constraints.device)
    linear_residual = constraints.residual(weight_tensor)
    result: dict[str, object] = {
        "solver": solver_name,
        "mode": mode,
        "device": device,
        "setup_seconds": setup_seconds,
        "step_seconds_p50_ms": 1000.0 * statistics.median(step_seconds),
        "step_seconds_p95_ms": 1000.0 * percentile(step_seconds, 0.95),
        "iterations_p50": percentile([float(value) for value in iterations], 0.5),
        "iterations_p95": percentile([float(value) for value in iterations], 0.95),
        "success_rate": float(np.mean(feasible)),
        "median_objective": float(np.median(objectives[feasible])) if np.any(feasible) else None,
        "statuses": statuses,
        "weights_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
        **residuals,
        "max_linear_equality_error": float(linear_residual.max_abs_equality.detach().cpu().item()),
        "max_linear_lower_violation": float(
            0.0
            if linear_residual.lower_violation is None
            else linear_residual.lower_violation.detach().cpu().amax().item()
        ),
        "max_linear_upper_violation": float(
            0.0
            if linear_residual.upper_violation is None
            else linear_residual.upper_violation.detach().cpu().amax().item()
        ),
        "objectives": objectives.tolist(),
        "feasible": feasible.tolist(),
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


def make_openpo_config(
    *,
    iterations: int,
    bisection_steps: int,
    warm_start_max_alpha_delta: float,
) -> PrimalDualConfig:
    config = PrimalDualConfig(
        iterations=iterations,
        tolerance=1e-6,
        constraint_tolerance=2e-6,
        bisection_steps=bisection_steps,
        warm_start_max_alpha_delta=warm_start_max_alpha_delta,
    )
    return config


def max_step_delta(alpha_path: np.ndarray) -> float:
    if alpha_path.shape[0] < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(alpha_path, axis=0))))


def run_q_update_path(
    *,
    case: constrained_benchmark.BenchmarkCase,
    alpha_path: np.ndarray,
    solver_name: str,
    setup_solver,
    update_solver,
    solve_solver,
    status_fn,
    iteration_fn,
) -> dict[str, object]:
    setup_seconds = 0.0
    solvers = []
    for track_alpha in alpha_path[0]:
        started = time.perf_counter()
        solver = setup_solver(track_alpha)
        setup_seconds += time.perf_counter() - started
        solvers.append(solver)

    for solver in solvers:
        solve_solver(solver)

    weights: list[np.ndarray] = []
    statuses: list[str] = []
    iterations: list[int] = []
    step_seconds: list[float] = []
    alpha_rows: list[np.ndarray] = []
    for step_alpha in alpha_path[1:]:
        step_start = time.perf_counter()
        for track_index, alpha in enumerate(step_alpha):
            solver = solvers[track_index]
            update_solver(solver, alpha)
            result = solve_solver(solver)
            weights.append(result.x[: case.assets])
            statuses.append(status_fn(result))
            iterations.append(int(iteration_fn(result)))
            alpha_rows.append(alpha)
        step_seconds.append(time.perf_counter() - step_start)

    return summarize_result(
        solver_name=solver_name,
        mode="q_update_hot",
        device="cpu",
        case=case,
        alpha_rows=np.asarray(alpha_rows, dtype=np.float64),
        weights=np.asarray(weights, dtype=np.float64),
        statuses=statuses,
        iterations=iterations,
        step_seconds=step_seconds,
        setup_seconds=setup_seconds,
    )


def run_osqp_path(
    case: constrained_benchmark.BenchmarkCase,
    alpha_path: np.ndarray,
) -> dict[str, object]:
    p_matrix, matrix, _, lower, upper = constrained_benchmark.build_lifted_qp(case)

    def setup_solver(alpha: np.ndarray) -> osqp.OSQP:
        solver = osqp.OSQP()
        solver.setup(
            P=p_matrix,
            q=np.concatenate((-alpha, np.zeros(case.risk_case.factors))),
            A=matrix,
            l=lower,
            u=upper,
            eps_abs=1e-7,
            eps_rel=1e-7,
            max_iter=20_000,
            polishing=True,
            verbose=False,
        )
        return solver

    def update_solver(solver: osqp.OSQP, alpha: np.ndarray) -> None:
        solver.update(q=np.concatenate((-alpha, np.zeros(case.risk_case.factors))))

    def solve_solver(solver: osqp.OSQP):
        return solver.solve(raise_error=True)

    return run_q_update_path(
        case=case,
        alpha_path=alpha_path,
        solver_name=f"OSQP {osqp.__version__} CPU",
        setup_solver=setup_solver,
        update_solver=update_solver,
        solve_solver=solve_solver,
        status_fn=lambda result: result.info.status,
        iteration_fn=lambda result: result.info.iter,
    )


def run_clarabel_path(
    case: constrained_benchmark.BenchmarkCase,
    alpha_path: np.ndarray,
) -> dict[str, object]:
    p_matrix, matrix, _, lower, upper = constrained_benchmark.build_lifted_qp(case)
    constrained = case.constrained_case
    eq_rows = 2 + case.risk_case.factors
    box_rows = constrained.box_lower.shape[0]
    linear_rows = constrained.industry_lower.shape[0] + constrained.style_lower.shape[0]
    cones = [
        clarabel.ZeroConeT(eq_rows),
        clarabel.NonnegativeConeT(2 * box_rows),
        clarabel.NonnegativeConeT(2 * linear_rows),
    ]
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

    def setup_solver(alpha: np.ndarray) -> clarabel.DefaultSolver:
        local_settings = configured_clarabel_settings(clarabel)
        return clarabel.DefaultSolver(
            p_matrix,
            np.concatenate((-alpha, np.zeros(case.risk_case.factors))),
            matrix,
            right_hand_side,
            cones,
            local_settings,
        )

    def update_solver(solver: clarabel.DefaultSolver, alpha: np.ndarray) -> None:
        solver.update(q=np.concatenate((-alpha, np.zeros(case.risk_case.factors))))

    def solve_solver(solver: clarabel.DefaultSolver):
        return solver.solve()

    return run_q_update_path(
        case=case,
        alpha_path=alpha_path,
        solver_name=f"Clarabel {clarabel.__version__} CPU",
        setup_solver=setup_solver,
        update_solver=update_solver,
        solve_solver=solve_solver,
        status_fn=lambda result: str(result.status),
        iteration_fn=lambda result: result.iterations,
    )


def run_openpo_path(
    case: constrained_benchmark.BenchmarkCase,
    alpha_path: np.ndarray,
    device: str,
    mode: str,
    iterations: int,
    cpu_bisection_steps: int,
    cuda_bisection_steps: int,
    warm_start_max_alpha_delta: float,
) -> dict[str, object]:
    setup_started = time.perf_counter()
    bisection_steps = cuda_bisection_steps if device.startswith("cuda") else cpu_bisection_steps
    config = make_openpo_config(
        iterations=iterations,
        bisection_steps=bisection_steps,
        warm_start_max_alpha_delta=warm_start_max_alpha_delta,
    )
    constraints = constrained_benchmark.linear_constraints(
        case,
        device,
        torch.float32,
    )
    problem_template = constrained_benchmark.openpo_problem(
        case, device, alpha_path[0]
    )
    setup_seconds = time.perf_counter() - setup_started

    def problem_with_alpha(alpha: np.ndarray) -> PortfolioProblem:
        return PortfolioProblem(
            alpha=torch.as_tensor(alpha, dtype=torch.float32, device=device),
            risk=problem_template.risk,
            lower=problem_template.lower,
            upper=problem_template.upper,
            budget=problem_template.budget,
            risk_aversion=problem_template.risk_aversion,
            current=problem_template.current,
            trade_cost=problem_template.trade_cost,
        )

    initial_result = solve_constrained(
        problem_template,
        constraints,
        config,
    )
    previous_state: PrimalDualState | None = (
        initial_result.state if mode == "previous_primal_dual" else None
    )

    step_seconds: list[float] = []
    statuses: list[str] = []
    iterations: list[int] = []
    alpha_rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []

    for step_alpha in alpha_path[1:]:
        started = time.perf_counter()
        problem = problem_with_alpha(step_alpha)
        warm_state = previous_state if mode == "previous_primal_dual" else None
        result = solve_constrained(
            problem,
            constraints,
            config,
            initial_state=warm_state,
        )
        step_seconds.append(time.perf_counter() - started)
        statuses.extend(
            ["converged" if result.converged else "max_iterations"] * case.batch
        )
        iterations.extend([int(result.iterations)] * case.batch)
        weights.append(result.weights.detach().cpu().double().numpy())
        alpha_rows.append(step_alpha)
        previous_state = result.state if mode == "previous_primal_dual" else None

    return summarize_result(
        solver_name=f"OpenPO {device} {mode}",
        mode=mode,
        device=device,
        case=case,
        alpha_rows=np.vstack(alpha_rows),
        weights=np.vstack(weights),
        statuses=statuses,
        iterations=iterations,
        step_seconds=step_seconds,
        setup_seconds=setup_seconds,
    )


def build_case_report(
    *,
    assets: int,
    batch: int,
    horizon: int,
    portfolio_mode: str,
    seed: int,
    devices: list[str],
    ar1_phi: float,
    alpha_scale: float,
    iterations: int,
    cpu_bisection_steps: int,
    cuda_bisection_steps: int,
) -> dict[str, object]:
    if horizon < 2:
        raise ValueError("horizon must be at least 2 so step0 can be excluded from metrics")
    long_short = portfolio_mode == "long_short"
    case = constrained_benchmark.build_benchmark_case(
        assets=assets,
        batch=batch,
        seed=seed,
        long_short=long_short,
    )
    path_seed = seed + assets * 31 + batch * 17 + horizon * 13 + (1 if long_short else 0)
    alpha_path = build_alpha_path(
        assets=assets,
        batch=batch,
        horizon=horizon,
        seed=path_seed,
        ar1_phi=ar1_phi,
        alpha_scale=alpha_scale,
    )
    alpha_delta = max_step_delta(alpha_path)
    warm_start_max_alpha_delta = alpha_delta + max(1e-12, 1e-6 * alpha_delta)
    results: list[dict[str, object]] = [
        run_osqp_path(case, alpha_path),
        run_clarabel_path(case, alpha_path),
    ]
    for device in devices:
        results.append(
            run_openpo_path(
                case,
                alpha_path,
                device,
                "cold",
                iterations,
                cpu_bisection_steps,
                cuda_bisection_steps,
                warm_start_max_alpha_delta,
            )
        )
        results.append(
            run_openpo_path(
                case,
                alpha_path,
                device,
                "previous_primal_dual",
                iterations,
                cpu_bisection_steps,
                cuda_bisection_steps,
                warm_start_max_alpha_delta,
            )
        )

    add_quality_gaps(results)
    alpha_path_sha = path_sha256(alpha_path, ar1_phi, alpha_scale)
    return {
        "assets": assets,
        "batch": batch,
        "horizon": horizon,
        "portfolio_mode": portfolio_mode,
        "seed": seed,
        "path_seed": path_seed,
        "case_sha256": case.input_sha256(),
        "path_sha256": alpha_path_sha,
        "alpha_path_max_delta": alpha_delta,
        "openpo_warm_start_max_alpha_delta": warm_start_max_alpha_delta,
        "input_sha256": combined_sha256(case.input_sha256(), alpha_path_sha),
        "risk_model": "shared diagonal factor covariance plus specific risk",
        "reference_formulation": "lifted sparse QP with explicit factor exposure variables and linear interval constraints",
        "alpha_process": {
            "kind": "AR(1)",
            "phi": ar1_phi,
            "alpha_scale": alpha_scale,
        },
        "constraints": (
            "market-neutral long-short box bounds, market equality, and industry/style intervals"
            if long_short
            else "budgeted long-only box bounds, market equality, and industry/style intervals"
        ),
        "results": results,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, nargs="+", default=[250, 1000])
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument(
        "--portfolio-modes",
        nargs="+",
        choices=("long_only", "long_short"),
        default=["long_only", "long_short"],
    )
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--ar1-phi", type=float, default=0.98)
    parser.add_argument("--alpha-scale", type=float, default=0.002)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--cpu-bisection-steps", type=int, default=24)
    parser.add_argument("--cuda-bisection-steps", type=int, default=20)
    parser.add_argument(
        "--devices",
        default=None,
        help="Comma-separated devices for OpenPO warm-start runs; default is auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark-results/warm-start-comparison.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    devices = resolve_devices(args.devices)
    started = time.perf_counter()
    cases = [
        build_case_report(
            assets=assets,
            batch=args.batch,
            horizon=args.horizon,
            portfolio_mode=portfolio_mode,
            seed=args.seed,
            devices=devices,
            ar1_phi=args.ar1_phi,
            alpha_scale=args.alpha_scale,
            iterations=args.iterations,
            cpu_bisection_steps=args.cpu_bisection_steps,
            cuda_bisection_steps=args.cuda_bisection_steps,
        )
        for assets in args.assets
        for portfolio_mode in args.portfolio_modes
    ]
    report = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": constrained_benchmark.solver_environment(),
        "methodology": {
            "objective": "0.5 * risk_aversion * x'Qx - alpha'x",
            "alpha_process": "rolling AR(1) path shared across all solvers and modes",
            "warm_start_max_alpha_delta": "max observed step delta plus a small margin",
            "success_tolerance": {
                "budget": 2e-5,
                "box": 2e-6,
                "linear": 2e-6,
            },
            "quality_reference": "best feasible objective returned by any compared solver/mode",
            "osqp_mode": "official q-update hot path after one symbolic/numeric setup",
            "clarabel_mode": "official q-update hot path after one symbolic/numeric setup",
            "openpo_mode": "bounded-simplex PDHG, cold start vs previous_primal_dual warm start",
            "openpo_projection_steps": {
                "cpu": args.cpu_bisection_steps,
                "cuda": args.cuda_bisection_steps,
            },
            "timing": f"p50/p95 across rolling steps 1..{args.horizon - 1}; iterations pooled across measured solves",
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
