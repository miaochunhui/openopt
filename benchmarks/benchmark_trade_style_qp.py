#!/usr/bin/env python3
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import statistics
import sys
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpo import trade_optimizer
from openpo.trade_optimizer import TradeStyleQPSolveResult, solve_trade_style_qp


DEFAULT_SEEDS = (20260809, 20260810, 20260811)


@dataclasses.dataclass(frozen=True)
class Sample:
    elapsed_seconds: float
    success: bool
    objective: float
    budget_residual: float
    box_residual: float
    stationarity_residual: float
    iterations: int


@dataclasses.dataclass(frozen=True)
class NumericalResult:
    objective: float
    dw: list[float]
    exposure: list[float]


@dataclasses.dataclass(frozen=True)
class BenchmarkCase:
    seed: int
    input_sha256: str
    problem: dict[str, Any]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def stats(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "max": max(values),
    }


def tensor_hash(hasher: Any, name: str, value: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(value)
    hasher.update(name.encode("utf-8"))
    hasher.update(str(contiguous.dtype).encode("ascii"))
    hasher.update(str(contiguous.shape).encode("ascii"))
    hasher.update(contiguous.tobytes())


def build_problem(assets: int, styles: int, seed: int) -> tuple[dict[str, Any], str]:
    rng = np.random.default_rng(seed)
    score = rng.normal(0.0, 1.0, assets).astype(np.float64)
    weights = rng.uniform(0.1, 1.0, assets).astype(np.float64)
    market_weights = rng.uniform(0.1, 1.0, assets).astype(np.float64)
    weights /= weights.sum()
    market_weights /= market_weights.sum()
    max_buy = np.full(assets, 0.002, dtype=np.float64)
    max_sell = np.minimum(weights, 0.002).astype(np.float64)
    factor_exposure = rng.normal(0.0, 0.2, (assets, styles)).astype(np.float64)
    base_exposure = factor_exposure.T @ (weights - market_weights)
    half_width = np.full(styles, 0.02, dtype=np.float64)
    style_ranges = np.column_stack((base_exposure - half_width, base_exposure + half_width))
    spread = np.asarray(0.6, dtype=np.float64)

    hasher = hashlib.sha256()
    for name, value in (
        ("score", score),
        ("max_sell", max_sell),
        ("max_buy", max_buy),
        ("weights", weights),
        ("market_weights", market_weights),
        ("factor_exposure", factor_exposure),
        ("style_ranges", style_ranges),
        ("spread", spread),
    ):
        tensor_hash(hasher, name, value)
    return {
        "score": score,
        "max_sell": max_sell,
        "max_buy": max_buy,
        "weights": weights,
        "market_weights": market_weights,
        "factor_exposure": factor_exposure,
        "style_ranges": style_ranges,
        "spread": float(spread),
    }, hasher.hexdigest()


def build_case(args: argparse.Namespace) -> BenchmarkCase:
    seed = int(getattr(args, "seed", args.seeds[0]))
    problem, input_sha256 = build_problem(args.assets, args.styles, seed)
    return BenchmarkCase(seed=seed, input_sha256=input_sha256, problem=problem)


def summarize_route(
    *,
    name: str,
    device: str,
    cold: list[Sample],
    warmup: list[Sample],
    hot: list[Sample],
) -> dict[str, Any]:
    if not hot:
        raise ValueError("hot samples must not be empty")
    for sample in (*cold, *warmup, *hot):
        numeric = (
            sample.elapsed_seconds,
            sample.objective,
            sample.budget_residual,
            sample.box_residual,
            sample.stationarity_residual,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("sample values must be finite")
        if sample.elapsed_seconds < 0 or sample.iterations <= 0:
            raise ValueError("elapsed time must be nonnegative and iterations positive")
    hot_ms = [sample.elapsed_seconds * 1000.0 for sample in hot]
    objectives = [sample.objective for sample in hot]
    iterations = [float(sample.iterations) for sample in hot]
    hot_statistics = stats(hot_ms)
    return {
        "name": name,
        "device": device,
        "cold": {"samples": len(cold)},
        "warmup": {"samples": len(warmup)},
        "hot": {
            "samples": len(hot),
            **hot_statistics,
            "p50_ms": hot_statistics["p50"],
            "p95_ms": hot_statistics["p95"],
        },
        "success_rate": sum(sample.success for sample in hot) / len(hot),
        "objective": stats(objectives),
        "residuals": {
            "budget_max": max(sample.budget_residual for sample in hot),
            "box_max": max(sample.box_residual for sample in hot),
            "stationarity_max": max(sample.stationarity_residual for sample in hot),
        },
        "iterations": stats(iterations),
        "estimated_30_day_seconds": statistics.median(hot_ms) * 30.0 / 1000.0,
    }


def compare_numerical(
    fused: NumericalResult, fallback: NumericalResult
) -> dict[str, float]:
    fused_dw = np.asarray(fused.dw, dtype=np.float64)
    fallback_dw = np.asarray(fallback.dw, dtype=np.float64)
    fused_exposure = np.asarray(fused.exposure, dtype=np.float64)
    fallback_exposure = np.asarray(fallback.exposure, dtype=np.float64)
    if fused_dw.shape != fallback_dw.shape or fused_exposure.shape != fallback_exposure.shape:
        raise ValueError("numerical result shapes must match")
    return {
        "objective_abs_gap": abs(fused.objective - fallback.objective),
        "dw_linf": float(np.max(np.abs(fused_dw - fallback_dw), initial=0.0)),
        "exposure_linf": float(
            np.max(np.abs(fused_exposure - fallback_exposure), initial=0.0)
        ),
    }


def to_device(problem: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: torch.as_tensor(value, dtype=torch.float64, device=device)
        if name != "spread"
        else value
        for name, value in problem.items()
    }


@contextmanager
def projection_route(route: str) -> Iterator[None]:
    original = trade_optimizer._prox_l1_box_neutral
    if route == "torch":
        selected = trade_optimizer._prox_l1_box_neutral_torch
    elif route == "triton":
        from openpo.triton_trade_projection import triton_prox_l1_box_neutral

        selected = triton_prox_l1_box_neutral
    else:
        raise ValueError(f"unknown route: {route}")
    trade_optimizer._prox_l1_box_neutral = selected
    try:
        yield
    finally:
        trade_optimizer._prox_l1_box_neutral = original


def solve(problem: dict[str, Any], args: argparse.Namespace) -> TradeStyleQPSolveResult:
    return solve_trade_style_qp(
        problem["score"],
        problem["max_sell"],
        problem["max_buy"],
        problem["spread"],
        problem["weights"],
        problem["market_weights"],
        problem["factor_exposure"],
        problem["style_ranges"],
        style_penalty=args.style_penalty,
        max_iterations=args.max_iterations,
        tolerance=args.tolerance,
        bisection_steps=args.bisection_steps,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def residuals(result: TradeStyleQPSolveResult) -> dict[str, float]:
    return {
        "budget": float(result.budget_residual.item()),
        "box": float(result.box_residual.item()),
        "style": float(torch.maximum(result.lower_style_violation, result.upper_style_violation).amax().item()),
        "stationarity": float(result.stationarity_residual.item()),
    }


def valid(result: TradeStyleQPSolveResult, args: argparse.Namespace) -> bool:
    metrics = residuals(result)
    return (
        bool(torch.isfinite(result.dw).all())
        and result.status == "converged"
        and np.isfinite(float(result.objective.item()))
        and metrics["budget"] <= args.residual_tolerance
        and metrics["box"] <= args.residual_tolerance
        and metrics["stationarity"] <= args.stationarity_tolerance
    )


def timed_solve(problem: dict[str, Any], device: torch.device, args: argparse.Namespace) -> tuple[float, TradeStyleQPSolveResult]:
    synchronize(device)
    started = time.perf_counter_ns()
    result = solve(problem, args)
    synchronize(device)
    return (time.perf_counter_ns() - started) / 1e6, result


def benchmark_route(problem: dict[str, Any], device: torch.device, route: str, args: argparse.Namespace) -> tuple[dict[str, Any], TradeStyleQPSolveResult]:
    with projection_route(route):
        cold_ms, cold_result = timed_solve(problem, device, args)
        if not valid(cold_result, args):
            raise RuntimeError(f"cold solve failed route={route} device={device} residuals={residuals(cold_result)}")
        for _ in range(args.warmups):
            warmup = solve(problem, args)
            if not valid(warmup, args):
                raise RuntimeError(f"warmup failed route={route} device={device}")
        synchronize(device)
        timings: list[float] = []
        results: list[TradeStyleQPSolveResult] = []
        for repeat in range(args.repeats):
            elapsed_ms, result = timed_solve(problem, device, args)
            if not valid(result, args):
                raise RuntimeError(f"hot solve failed route={route} device={device} repeat={repeat} residuals={residuals(result)}")
            timings.append(elapsed_ms)
            results.append(result)
    objectives = [float(result.objective.item()) for result in results]
    iterations = [float(result.iterations) for result in results]
    residual_rows = [residuals(result) for result in results]
    report = {
        "route": route,
        "cold": {
            "milliseconds": cold_ms,
            "success": True,
            "objective": float(cold_result.objective.item()),
            "residual": residuals(cold_result),
            "iterations": cold_result.iterations,
            "status": cold_result.status,
        },
        "hot": {
            "milliseconds": stats(timings),
            "success_rate": len(results) / args.repeats,
            "objective": stats(objectives),
            "residual": {name: stats([row[name] for row in residual_rows]) for name in residual_rows[0]},
            "iterations": stats(iterations),
            "samples": args.repeats,
        },
        "thirty_day_estimate": {
            "hot_p50_seconds": stats(timings)["p50"] * 30.0 / 1000.0,
            "hot_p95_seconds": stats(timings)["p95"] * 30.0 / 1000.0,
            "assumption": "one independent solve per trading day",
        },
    }
    return report, results[-1]


def environment(devices: list[str]) -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    gpu_rows = []
    for name in devices:
        device = torch.device(name)
        properties = torch.cuda.get_device_properties(device)
        gpu_rows.append({
            "device": name,
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
            "total_memory_bytes": properties.total_memory,
        })
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_driver": torch.cuda.driver_version() if hasattr(torch.cuda, "driver_version") else None,
        "triton": triton_version,
        "gpus": gpu_rows,
    }


def collect_environment() -> dict[str, Any]:
    devices = (
        [f"cuda:{index}" for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    result = environment(devices)
    result["torch"] = result["pytorch"]
    result["machine"] = platform.machine()
    return result


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_json(path, payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark FP64 trade style-QP Triton and Torch projection routes.")
    parser.add_argument("--assets", type=int, default=6000)
    parser.add_argument("--styles", type=int, default=10)
    parser.add_argument("--devices", nargs="+", default=["cuda:0", "cuda:1"])
    parser.add_argument("--device", action="append", dest="device_alias")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--seed", action="append", type=int, dest="seed_alias")
    parser.add_argument("--routes", nargs="+", choices=("triton", "torch"), default=["triton", "torch"])
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--bisection-steps", type=int, default=60)
    parser.add_argument("--style-penalty", type=float, default=5.0)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    parser.add_argument("--residual-tolerance", type=float, default=1e-8)
    parser.add_argument("--stationarity-tolerance", type=float, default=1e-6)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results" / "trade-style-qp.json")
    args = parser.parse_args(argv)
    if args.device_alias:
        args.devices = args.device_alias
    if args.seed_alias:
        args.seeds = args.seed_alias
    args.device = args.devices[0]
    args.seed = args.seeds[0]
    if args.smoke:
        args.assets = min(args.assets, 128)
        args.styles = min(args.styles, 3)
        args.seeds = args.seeds[:1]
        args.warmups = min(args.warmups, 1)
        args.repeats = min(args.repeats, 2)
        args.max_iterations = min(args.max_iterations, 40)
    if args.assets <= 0 or args.styles <= 0 or args.warmups < 0 or args.repeats <= 0:
        parser.error("assets/styles/repeats must be positive and warmups nonnegative")
    cuda_requested = any(str(device).startswith("cuda") for device in args.devices)
    if not args.smoke and cuda_requested and args.repeats < 20:
        parser.error("full benchmark requires repeats >= 20; use --smoke for smaller validation")
    if len(args.seeds) != 3 and not args.smoke and cuda_requested:
        parser.error("full benchmark requires exactly 3 seeds")
    return args


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    for device_name in args.devices:
        device = torch.device(device_name)
        if device.type != "cuda" or device.index is None or device.index >= torch.cuda.device_count():
            raise ValueError(f"unavailable CUDA device: {device_name}")

    lanes = []
    pairwise = []
    for seed in args.seeds:
        host_problem, input_sha256 = build_problem(args.assets, args.styles, seed)
        for device_name in args.devices:
            device = torch.device(device_name)
            device_problem = to_device(host_problem, device)
            route_outputs: dict[str, TradeStyleQPSolveResult] = {}
            for route in args.routes:
                route_report, final_result = benchmark_route(device_problem, device, route, args)
                route_report.update({"seed": seed, "device": device_name, "input_sha256": input_sha256})
                lanes.append(route_report)
                route_outputs[route] = final_result
            if {"triton", "torch"}.issubset(route_outputs):
                fused = route_outputs["triton"]
                fallback = route_outputs["torch"]
                pairwise.append({
                    "seed": seed,
                    "device": device_name,
                    "input_sha256": input_sha256,
                    "objective_abs_diff": abs(float(fused.objective.item() - fallback.objective.item())),
                    "dw_max_abs_diff": float((fused.dw - fallback.dw).abs().amax().item()),
                    "exposure_max_abs_diff": float((fused.exposure - fallback.exposure).abs().amax().item()),
                })

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "assets": args.assets,
            "styles": args.styles,
            "dtype": "float64",
            "devices": args.devices,
            "seeds": args.seeds,
            "routes": args.routes,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "max_iterations": args.max_iterations,
            "bisection_steps": args.bisection_steps,
            "osqp_large_scale": False,
        },
        "environment": environment(args.devices),
        "lanes": lanes,
        "pairwise": pairwise,
    }
    atomic_json(args.output, payload)
    print(json.dumps({"output": str(args.output), "lanes": len(lanes), "pairwise": len(pairwise)}, sort_keys=True))


if __name__ == "__main__":
    main()
