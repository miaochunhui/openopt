from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist

from openpo import (
    BarraRiskData,
    LinearConstraints,
    PortfolioProblem,
    PrimalDualConfig,
    active_objective as barra_active_objective,
    solve_constrained,
)
from openpo.projection import project_bounded_simplex_unchecked


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = 6000
DEFAULT_FACTORS = 40
DEFAULT_INDUSTRY_FACTORS = 30
DEFAULT_STYLE_FACTORS = 10
DEFAULT_BATCH = 32
DEFAULT_ITERATIONS = 2000
DEFAULT_REPEATS = 3
DEFAULT_SEED = 20260809
HOT_SLA_SECONDS = 10.0


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be in [0, 1], got {probability}")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered[lower_index]
    fraction = position - lower_index
    return ordered[lower_index] * (1.0 - fraction) + ordered[upper_index] * fraction


def split_batch_sizes(batch: int, world_size: int) -> list[int]:
    if batch <= 0:
        raise ValueError("batch must be positive")
    if world_size <= 0:
        raise ValueError("world_size must be positive")
    base, remainder = divmod(batch, world_size)
    return [base + (1 if rank < remainder else 0) for rank in range(world_size)]


def _tensor_digest(tensor: torch.Tensor) -> str:
    digest = hashlib.sha256()
    cpu_tensor = tensor.detach().contiguous().cpu()
    digest.update(str(cpu_tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
    digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()


def _combined_digest(parts: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for key in sorted(parts):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(parts[key].encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _torch_version() -> str:
    return torch.__version__


def _openpo_version() -> str | None:
    try:
        return importlib.metadata.version("openpo")
    except importlib.metadata.PackageNotFoundError:
        return None


@dataclass(frozen=True)
class CntrProxyCase:
    data: BarraRiskData
    alpha: torch.Tensor
    current: torch.Tensor
    box_lower: torch.Tensor
    box_upper: torch.Tensor
    factor_exposure_lower: torch.Tensor
    factor_exposure_upper: torch.Tensor
    factor_labels: tuple[str, ...]


def _positive_weights(generator: torch.Generator, assets: int) -> torch.Tensor:
    raw = torch.rand(assets, generator=generator, dtype=torch.float32).clamp_min(1e-6)
    return raw / raw.sum()


def _build_factor_exposures(
    *,
    generator: torch.Generator,
    assets: int,
    industry_factors: int,
    style_factors: int,
) -> torch.Tensor:
    industry_assignments = torch.arange(assets) % industry_factors
    industry_block = torch.nn.functional.one_hot(industry_assignments, num_classes=industry_factors)
    industry_block = industry_block.to(dtype=torch.float32)
    industry_block = industry_block + 0.04 * torch.randn(
        assets, industry_factors, generator=generator, dtype=torch.float32
    )
    style_block = torch.randn(assets, style_factors, generator=generator, dtype=torch.float32)
    style_block = style_block / style_block.std(dim=0, keepdim=True).clamp_min(1e-6)
    style_block = 0.18 * style_block
    return torch.cat([industry_block, style_block], dim=1)


def _build_spd_factor_covariance(generator: torch.Generator, factors: int) -> torch.Tensor:
    scale = 0.65 + 1.60 * torch.rand(factors, generator=generator, dtype=torch.float32)
    indices = torch.arange(factors)
    distance = (indices[:, None] - indices[None, :]).abs()
    correlation = torch.pow(torch.tensor(0.20, dtype=torch.float32), distance)
    lower = torch.tril(scale[:, None] * correlation * scale[None, :], diagonal=-1)
    covariance = lower + lower.T + torch.diag(scale.square() + 0.15)
    return covariance


def build_cntr_proxy(
    *,
    assets: int,
    factors: int,
    batch: int,
    seed: int,
    industry_factors: int = DEFAULT_INDUSTRY_FACTORS,
    style_factors: int = DEFAULT_STYLE_FACTORS,
) -> CntrProxyCase:
    if factors != industry_factors + style_factors:
        raise ValueError("factors must equal industry_factors + style_factors")
    if batch <= 0:
        raise ValueError("batch must be positive")
    if assets <= 0:
        raise ValueError("assets must be positive")
    generator = torch.Generator().manual_seed(seed)
    exposures = _build_factor_exposures(
        generator=generator,
        assets=assets,
        industry_factors=industry_factors,
        style_factors=style_factors,
    )
    factor_covariance = _build_spd_factor_covariance(generator, factors)
    factor_variance = torch.diag(factor_covariance).clamp_min(1e-4)
    specific_variance = (0.015 + 0.04 * torch.rand(assets, generator=generator)).clamp_min(1e-4)

    benchmark = _positive_weights(generator, assets)
    benchmark_exposure = benchmark.unsqueeze(0) @ exposures
    benchmark_exposure = benchmark_exposure.squeeze(0)

    box_radius = (0.0008 + 0.45 * benchmark).clamp_min(5e-5)
    box_lower = torch.clamp(benchmark - box_radius, min=0.0)
    box_upper = torch.clamp(benchmark + box_radius, max=1.0)
    box_upper = torch.maximum(box_upper, box_lower + 1e-6)
    if float(box_lower.sum()) > 1.0 or float(box_upper.sum()) < 1.0:
        raise ValueError("benchmark box is infeasible under the generated bounds")

    current_guess = benchmark + 0.25 * box_radius * torch.randn(
        assets, generator=generator, dtype=torch.float32
    )
    current = project_bounded_simplex_unchecked(
        current_guess.unsqueeze(0),
        box_lower,
        box_upper,
        1.0,
        24,
    ).squeeze(0)

    factor_width = torch.cat(
        [
            0.010 + 0.035 * benchmark_exposure[:industry_factors].abs(),
            0.020 + 0.050 * benchmark_exposure[industry_factors:].abs(),
        ]
    )
    factor_exposure_lower = benchmark_exposure - factor_width
    factor_exposure_upper = benchmark_exposure + factor_width

    alpha = 0.0015 * torch.randn(batch, assets, generator=generator, dtype=torch.float32)
    alpha = alpha + 0.45 * (benchmark - current).unsqueeze(0)

    factor_labels = tuple(
        [f"industry_{index:02d}" for index in range(industry_factors)]
        + [f"style_{index:02d}" for index in range(style_factors)]
    )
    risk_data = BarraRiskData(
        exposures=exposures,
        factor_covariance=factor_covariance,
        specific_variance=specific_variance,
        benchmark=benchmark,
    )
    return CntrProxyCase(
        data=risk_data,
        alpha=alpha,
        current=current,
        box_lower=box_lower,
        box_upper=box_upper,
        factor_exposure_lower=factor_exposure_lower,
        factor_exposure_upper=factor_exposure_upper,
        factor_labels=factor_labels,
    )


def build_barra_problem(
    case: CntrProxyCase,
    alpha: torch.Tensor,
    *,
    device: torch.device | str,
) -> tuple[PortfolioProblem, LinearConstraints]:
    moved_data = case.data.to(device)
    problem = moved_data.to_portfolio_problem(
        alpha.to(device),
        case.box_lower.to(device),
        case.box_upper.to(device),
        budget=1.0,
        risk_aversion=10.0,
        current=case.current.to(device),
        trade_cost=0.0025,
    )
    constraints = LinearConstraints(
        A_ineq=moved_data.exposures.T,
        lower=case.factor_exposure_lower.to(device),
        upper=case.factor_exposure_upper.to(device),
    )
    return problem, constraints


def active_objective(
    case: CntrProxyCase,
    alpha: torch.Tensor,
    weights: torch.Tensor,
    *,
    benchmark: torch.Tensor | None = None,
) -> torch.Tensor:
    device = weights.device
    risk_data = case.data.to(device)
    if benchmark is not None:
        risk_data = BarraRiskData(
            exposures=risk_data.exposures,
            factor_covariance=risk_data.factor_covariance,
            specific_variance=risk_data.specific_variance,
            benchmark=benchmark.to(device),
        )
    return barra_active_objective(
        risk_data,
        alpha.to(device),
        weights,
        risk_aversion=10.0,
        current=case.current.to(device),
        trade_cost=0.0025,
    )


def _max_budget_error(problem: PortfolioProblem, weights: torch.Tensor) -> float:
    return float((weights.sum(dim=1) - problem.budget).abs().amax().item())


def _max_box_violation(problem: PortfolioProblem, weights: torch.Tensor) -> float:
    lower = torch.clamp(problem.lower - weights, min=0).amax()
    upper = torch.clamp(weights - problem.upper, min=0).amax()
    return float(torch.maximum(lower, upper).item())


def _tensor_batch_summary(tensor: torch.Tensor) -> dict[str, object]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": _tensor_digest(tensor),
    }


def build_input_summary(case: CntrProxyCase) -> dict[str, object]:
    data = case.data
    tensor_hashes = {
        "exposures": _tensor_digest(data.exposures),
        "factor_covariance": _tensor_digest(data.factor_covariance),
        "specific_variance": _tensor_digest(data.specific_variance),
        "benchmark": _tensor_digest(data.benchmark),
        "current": _tensor_digest(case.current),
        "alpha": _tensor_digest(case.alpha),
        "box_lower": _tensor_digest(case.box_lower),
        "box_upper": _tensor_digest(case.box_upper),
        "factor_exposure_lower": _tensor_digest(case.factor_exposure_lower),
        "factor_exposure_upper": _tensor_digest(case.factor_exposure_upper),
    }
    return {
        "assets": int(data.exposures.shape[0]),
        "factors": int(data.exposures.shape[1]),
        "industry_factors": int(sum(label.startswith("industry_") for label in case.factor_labels)),
        "style_factors": int(sum(label.startswith("style_") for label in case.factor_labels)),
        "benchmark": _tensor_batch_summary(data.benchmark),
        "current": _tensor_batch_summary(case.current),
        "alpha": _tensor_batch_summary(case.alpha),
        "factor_covariance": _tensor_batch_summary(data.factor_covariance),
        "specific_variance": _tensor_batch_summary(data.specific_variance),
        "box_lower": _tensor_batch_summary(case.box_lower),
        "box_upper": _tensor_batch_summary(case.box_upper),
        "factor_exposure_lower": _tensor_batch_summary(case.factor_exposure_lower),
        "factor_exposure_upper": _tensor_batch_summary(case.factor_exposure_upper),
        "combined_sha256": _combined_digest(tensor_hashes),
    }


def _quality_metrics(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    weights: torch.Tensor,
) -> dict[str, float]:
    residuals = constraints.residual(weights)
    max_linear_lower = 0.0 if residuals.lower_violation is None else float(residuals.lower_violation.amax().item())
    max_linear_upper = 0.0 if residuals.upper_violation is None else float(residuals.upper_violation.amax().item())
    max_linear_equality = 0.0 if residuals.equality is None else float(residuals.equality.abs().amax().item())
    return {
        "max_budget_error": _max_budget_error(problem, weights),
        "max_box_violation": _max_box_violation(problem, weights),
        "max_linear_equality_error": max_linear_equality,
        "max_linear_lower_violation": max_linear_lower,
        "max_linear_upper_violation": max_linear_upper,
        "max_linear_violation": max(max_linear_equality, max_linear_lower, max_linear_upper),
    }


def _success_rate(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    weights: torch.Tensor,
    *,
    budget_tol: float,
    box_tol: float,
    linear_tol: float,
) -> float:
    row_budget = (weights.sum(dim=1) - problem.budget).abs()
    row_box = torch.maximum(
        torch.clamp(problem.lower.unsqueeze(0) - weights, min=0).amax(dim=1),
        torch.clamp(weights - problem.upper.unsqueeze(0), min=0).amax(dim=1),
    )
    linear_residual = constraints.residual(weights)
    row_linear = torch.zeros_like(row_budget)
    if linear_residual.equality is not None:
        row_linear = torch.maximum(row_linear, linear_residual.equality.abs().amax(dim=1))
    if linear_residual.lower_violation is not None:
        row_linear = torch.maximum(row_linear, linear_residual.lower_violation.amax(dim=1))
    if linear_residual.upper_violation is not None:
        row_linear = torch.maximum(row_linear, linear_residual.upper_violation.amax(dim=1))
    feasible = (row_budget <= budget_tol) & (row_box <= box_tol) & (row_linear <= linear_tol)
    return float(feasible.to(dtype=torch.float32).mean().item())


def measure_rank(
    *,
    case: CntrProxyCase,
    rank: int,
    world_size: int,
    device: str,
    iterations: int,
    repeats: int,
    input_seconds: float,
    local_rank: int | None = None,
) -> dict[str, object]:
    rank_batch_sizes = split_batch_sizes(case.alpha.shape[0], world_size)
    alpha_shards = torch.tensor_split(case.alpha, world_size, dim=0)
    alpha = alpha_shards[rank]

    build_started = time.perf_counter()
    problem, constraints = build_barra_problem(case, alpha, device=device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    build_h2d_seconds = time.perf_counter() - build_started

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(device))

    config = PrimalDualConfig(
        iterations=iterations,
        tolerance=0.0,
        constraint_tolerance=2e-5,
        check_interval=max(20, iterations // 20),
        bisection_steps=24,
    )

    cold_started = time.perf_counter()
    cold_result = solve_constrained(problem, constraints, config)
    if device.startswith("cuda"):
        torch.cuda.synchronize(torch.device(device))
    cold_seconds = time.perf_counter() - cold_started

    hot_seconds: list[float] = []
    hot_result = cold_result
    for _ in range(repeats):
        hot_started = time.perf_counter()
        hot_result = solve_constrained(problem, constraints, config, initial_state=cold_result.state)
        if device.startswith("cuda"):
            torch.cuda.synchronize(torch.device(device))
        hot_seconds.append(time.perf_counter() - hot_started)

    weights = hot_result.weights
    active = active_objective(
        case,
        alpha,
        weights,
        benchmark=case.data.benchmark.to(weights.device),
    )
    current_weights = case.current.to(weights.device).unsqueeze(0).expand_as(weights)
    current_active = active_objective(
        case,
        alpha,
        current_weights,
        benchmark=case.data.benchmark.to(weights.device),
    )
    benchmark_weights = case.data.benchmark.to(weights.device).unsqueeze(0)
    active_l1 = (weights - benchmark_weights).abs().sum(dim=1)
    quality = _quality_metrics(problem, constraints, weights)
    success_rate = _success_rate(
        problem,
        constraints,
        weights,
        budget_tol=2e-5,
        box_tol=2e-6,
        linear_tol=2e-6,
    )
    device_properties = torch.cuda.get_device_properties(torch.device(device)) if device.startswith("cuda") else None
    peak_allocated = (
        int(torch.cuda.max_memory_allocated(torch.device(device))) if device.startswith("cuda") else None
    )
    peak_reserved = (
        int(torch.cuda.max_memory_reserved(torch.device(device))) if device.startswith("cuda") else None
    )
    result_hash = hashlib.sha256()
    result_hash.update(weights.detach().contiguous().cpu().numpy().tobytes())
    result_hash.update(hot_result.objective.detach().contiguous().cpu().numpy().tobytes())

    return {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "per_rank_batch": rank_batch_sizes[rank],
        "device": device,
        "gpu": None
        if device_properties is None
        else {
            "name": torch.cuda.get_device_name(torch.device(device)),
            "device_capability": ".".join(str(part) for part in torch.cuda.get_device_capability(torch.device(device))),
            "total_memory_bytes": int(device_properties.total_memory),
        },
        "versions": {
            "python": platform.python_version(),
            "torch": _torch_version(),
            "cuda_runtime": torch.version.cuda,
            "openpo": _openpo_version(),
        },
        "timings": {
            "input_seconds": input_seconds,
            "build_h2d_seconds": build_h2d_seconds,
            "cold_seconds": cold_seconds,
            "hot_seconds": hot_seconds,
            "hot_p50_seconds": percentile(hot_seconds, 0.5),
            "hot_p95_seconds": percentile(hot_seconds, 0.95),
        },
        "success_rate": success_rate,
        "active_objective": float(active.mean().item()),
        "current_active_objective": float(current_active.mean().item()),
        "active_objective_improvement": float((current_active - active).mean().item()),
        "active_weight_l1_mean": float(active_l1.mean().item()),
        "objective_mean": float(hot_result.objective.mean().item()),
        "quality": quality,
        "peak_memory": {
            "allocated_bytes": peak_allocated,
            "reserved_bytes": peak_reserved,
        },
        "input_hash": build_input_summary(case)["combined_sha256"],
        "result_hash": result_hash.hexdigest(),
    }


def assemble_report(
    *,
    args: argparse.Namespace,
    case: CntrProxyCase,
    rank_summaries: Sequence[dict[str, object]],
    environment: dict[str, object],
    input_summary: dict[str, object] | None = None,
) -> dict[str, object]:
    if not rank_summaries:
        raise ValueError("rank_summaries must not be empty")
    current_input_summary = build_input_summary(case)
    if input_summary is None:
        input_summary = current_input_summary
    elif current_input_summary != input_summary:
        raise RuntimeError("CNTR input tensors changed during solve")
    input_hash = str(input_summary["combined_sha256"])
    if args.batch != int(case.alpha.shape[0]):
        raise ValueError("args.batch must match the case alpha batch")
    expected_rank_batches = split_batch_sizes(args.batch, len(rank_summaries))
    actual_rank_batches = [int(summary["per_rank_batch"]) for summary in rank_summaries]
    if actual_rank_batches != expected_rank_batches:
        raise ValueError(
            f"per-rank batches must equal the deterministic split {expected_rank_batches}"
        )
    for rank, summary in enumerate(rank_summaries):
        if summary.get("rank") != rank:
            raise ValueError("rank_summaries must be ordered by unique rank")
        if summary.get("world_size") != len(rank_summaries):
            raise ValueError("rank summary world_size is inconsistent")
        if summary.get("input_hash") != input_hash:
            raise RuntimeError(f"rank {rank} input hash does not match the frozen request")
    cold_seconds = [float(summary["timings"]["cold_seconds"]) for summary in rank_summaries]
    build_h2d_seconds = [float(summary["timings"]["build_h2d_seconds"]) for summary in rank_summaries]
    input_seconds = [float(summary["timings"]["input_seconds"]) for summary in rank_summaries]
    repeat_counts = {len(summary["timings"]["hot_seconds"]) for summary in rank_summaries}
    if len(repeat_counts) != 1:
        raise ValueError("all ranks must report the same number of hot repeats")
    repeat_count = repeat_counts.pop()
    if repeat_count <= 0:
        raise ValueError("hot repeats must not be empty")
    hot_seconds = [
        max(float(summary["timings"]["hot_seconds"][index]) for summary in rank_summaries)
        for index in range(repeat_count)
    ]
    quality = {
        "max_budget_error": max(float(summary["quality"]["max_budget_error"]) for summary in rank_summaries),
        "max_box_violation": max(float(summary["quality"]["max_box_violation"]) for summary in rank_summaries),
        "max_linear_equality_error": max(
            float(summary["quality"]["max_linear_equality_error"]) for summary in rank_summaries
        ),
        "max_linear_lower_violation": max(
            float(summary["quality"]["max_linear_lower_violation"]) for summary in rank_summaries
        ),
        "max_linear_upper_violation": max(
            float(summary["quality"]["max_linear_upper_violation"]) for summary in rank_summaries
        ),
        "max_linear_violation": max(float(summary["quality"]["max_linear_violation"]) for summary in rank_summaries),
    }
    success_rate = sum(
        float(summary["success_rate"]) * int(summary["per_rank_batch"])
        for summary in rank_summaries
    ) / float(sum(int(summary["per_rank_batch"]) for summary in rank_summaries))
    hot_p50_seconds = percentile(hot_seconds, 0.5)
    hot_p95_seconds = percentile(hot_seconds, 0.95)
    cold_p50_seconds = percentile(cold_seconds, 0.5)
    cold_p95_seconds = percentile(cold_seconds, 0.95)
    build_h2d_p50_seconds = percentile(build_h2d_seconds, 0.5)
    build_h2d_p95_seconds = percentile(build_h2d_seconds, 0.95)
    input_p50_seconds = percentile(input_seconds, 0.5)
    input_p95_seconds = percentile(input_seconds, 0.95)
    e2e_cold_seconds = max(
        float(summary["timings"]["input_seconds"])
        + float(summary["timings"]["build_h2d_seconds"])
        + float(summary["timings"]["cold_seconds"])
        for summary in rank_summaries
    )
    e2e_hot_seconds = [
        max(input_seconds) + max(build_h2d_seconds) + seconds for seconds in hot_seconds
    ]
    e2e_hot_p50_seconds = percentile(e2e_hot_seconds, 0.5)
    e2e_hot_p95_seconds = percentile(e2e_hot_seconds, 0.95)
    decision = (
        "KEEP"
        if success_rate == 1.0 and e2e_hot_p95_seconds < HOT_SLA_SECONDS
        else "NO_KEEP"
    )
    result_hash = hashlib.sha256()
    for summary in rank_summaries:
        result_hash.update(str(summary["result_hash"]).encode("ascii"))

    return {
        "schema_version": 2,
        "status": "MEASURED",
        "decision": decision,
        "methodology": {
            "note": (
                "CNTR-shaped synthetic proxy; no authorized CNTR samples are used. "
                "The benchmark deterministically synthesizes a full SPD factor covariance, "
                "specific variance, benchmark/current/alpha/box, and benchmark-adjacent factor exposure intervals."
            ),
            "assets": args.assets,
            "factors": args.factors,
            "industry_factors": args.industry_factors,
            "style_factors": args.style_factors,
            "batch": args.batch,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "world_size": len(rank_summaries),
            "per_rank_batch": [int(summary["per_rank_batch"]) for summary in rank_summaries],
            "sharding": "portfolio batch only; every rank materializes the full static risk model",
            "hot_sla_seconds": HOT_SLA_SECONDS,
        },
        "environment": environment,
        "input": {
            **input_summary,
            "batch": args.batch,
            "seed": args.seed,
            "budget": 1.0,
            "risk_aversion": 10.0,
            "trade_cost": 0.0025,
            "input_hash": input_hash,
        },
        "summary": {
            "world_size": len(rank_summaries),
            "success_rate": success_rate,
            "active_objective": float(statistics.mean(float(summary["active_objective"]) for summary in rank_summaries)),
            "current_active_objective": float(
                statistics.mean(float(summary["current_active_objective"]) for summary in rank_summaries)
            ),
            "active_objective_improvement": float(
                statistics.mean(float(summary["active_objective_improvement"]) for summary in rank_summaries)
            ),
            "active_weight_l1_mean": float(
                statistics.mean(float(summary["active_weight_l1_mean"]) for summary in rank_summaries)
            ),
            "objective_mean": float(statistics.mean(float(summary["objective_mean"]) for summary in rank_summaries)),
            "input_p50_seconds": input_p50_seconds,
            "input_p95_seconds": input_p95_seconds,
            "build_h2d_p50_seconds": build_h2d_p50_seconds,
            "build_h2d_p95_seconds": build_h2d_p95_seconds,
            "cold_p50_seconds": cold_p50_seconds,
            "cold_p95_seconds": cold_p95_seconds,
            "hot_p50_seconds": hot_p50_seconds,
            "hot_p95_seconds": hot_p95_seconds,
            "e2e_cold_seconds": e2e_cold_seconds,
            "e2e_hot_p50_seconds": e2e_hot_p50_seconds,
            "e2e_hot_p95_seconds": e2e_hot_p95_seconds,
            "hot_sla_seconds": HOT_SLA_SECONDS,
            "hot_sla_met": e2e_hot_p95_seconds < HOT_SLA_SECONDS,
            **quality,
        },
        "runs": list(rank_summaries),
        "result_hash": result_hash.hexdigest(),
    }


def collect_environment(world_size: int) -> dict[str, object]:
    visible_devices: list[dict[str, object]] = []
    gpu_total_memory: list[int] = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            visible_devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "device_capability": ".".join(str(part) for part in torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": int(properties.total_memory),
                }
            )
            gpu_total_memory.append(int(properties.total_memory))
    compiled_architectures = torch.cuda.get_arch_list() if torch.cuda.is_available() else []
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _torch_version(),
        "cuda_runtime": torch.version.cuda,
        "openpo": _openpo_version(),
        "world_size": world_size,
        "visible_gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "compiled_architectures": compiled_architectures,
        "visible_devices": visible_devices,
        "gpu_total_memory_bytes": gpu_total_memory,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": (
            ".".join(str(part) for part in torch.cuda.get_device_capability(0))
            if torch.cuda.is_available()
            else None
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark a CNTR-shaped synthetic proxy.")
    parser.add_argument("--assets", type=int, default=DEFAULT_ASSETS)
    parser.add_argument("--factors", type=int, default=DEFAULT_FACTORS)
    parser.add_argument("--industry-factors", type=int, default=DEFAULT_INDUSTRY_FACTORS)
    parser.add_argument("--style-factors", type=int, default=DEFAULT_STYLE_FACTORS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmark-results" / "cntr-6000-benchmark.json")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.assets <= 0:
        parser.error("--assets must be positive")
    if args.factors <= 0:
        parser.error("--factors must be positive")
    if args.industry_factors <= 0 or args.style_factors <= 0:
        parser.error("--industry-factors and --style-factors must be positive")
    if args.factors != args.industry_factors + args.style_factors:
        parser.error("--factors must equal --industry-factors + --style-factors")
    if args.batch <= 0:
        parser.error("--batch must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.batch < 1:
        parser.error("--batch must be at least 1")


def _run_mode(args: argparse.Namespace, *, world_size: int, rank: int | None = None, device: str | None = None) -> dict[str, object]:
    input_started = time.perf_counter()
    case = build_cntr_proxy(
        assets=args.assets,
        factors=args.factors,
        batch=args.batch,
        seed=args.seed,
        industry_factors=args.industry_factors,
        style_factors=args.style_factors,
    )
    input_summary = build_input_summary(case)
    input_seconds = time.perf_counter() - input_started
    if world_size == 1:
        local_rank = 0 if rank is None else rank
        target_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        summary = measure_rank(
            case=case,
            rank=0,
            world_size=1,
            device=target_device,
            iterations=args.iterations,
            repeats=args.repeats,
            input_seconds=input_seconds,
            local_rank=local_rank,
        )
        return assemble_report(
            args=args,
            case=case,
            rank_summaries=[summary],
            environment=collect_environment(1),
            input_summary=input_summary,
        )
    raise RuntimeError("distributed execution is handled by run_worker")


def run_worker(args: argparse.Namespace) -> None:
    if "LOCAL_RANK" not in os.environ:
        report = _run_mode(args, world_size=1)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    dist.init_process_group(backend="nccl", device_id=torch.device(device))
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size < 1:
            raise RuntimeError("distributed world_size must be positive")
        if args.batch < world_size:
            raise RuntimeError(
                f"batch={args.batch} is too small for world_size={world_size}; every rank needs at least one portfolio"
            )
        input_started = time.perf_counter()
        case = build_cntr_proxy(
            assets=args.assets,
            factors=args.factors,
            batch=args.batch,
            seed=args.seed,
            industry_factors=args.industry_factors,
            style_factors=args.style_factors,
        )
        input_summary = build_input_summary(case)
        input_seconds = time.perf_counter() - input_started
        dist.barrier()
        summary = measure_rank(
            case=case,
            rank=rank,
            world_size=world_size,
            device=device,
            iterations=args.iterations,
            repeats=args.repeats,
            input_seconds=input_seconds,
            local_rank=local_rank,
        )
        gathered: list[dict[str, object]] = [dict() for _ in range(world_size)]
        dist.all_gather_object(gathered, summary)
        if rank == 0:
            report = assemble_report(
                args=args,
                case=case,
                rank_summaries=gathered,
                environment=collect_environment(world_size),
                input_summary=input_summary,
            )
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    run_worker(args)


if __name__ == "__main__":
    main()
