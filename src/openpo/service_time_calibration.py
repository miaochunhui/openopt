from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ServiceTimeCalibrationConfig:
    window_size: int = 32
    minimum_samples: int = 4
    ewma_alpha: float = 0.25
    quantile_probability: float = 0.90
    safety_multiplier: float = 1.0
    minimum_budget_ratio: float = 0.50
    maximum_budget_ratio: float = 2.0

    def __post_init__(self) -> None:
        for name, value in (
            ("window_size", self.window_size),
            ("minimum_samples", self.minimum_samples),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.minimum_samples > self.window_size:
            raise ValueError("minimum_samples must not exceed window_size")
        for name, value in (
            ("ewma_alpha", self.ewma_alpha),
            ("quantile_probability", self.quantile_probability),
        ):
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite in (0, 1]")
        for name, value in (
            ("safety_multiplier", self.safety_multiplier),
            ("minimum_budget_ratio", self.minimum_budget_ratio),
            ("maximum_budget_ratio", self.maximum_budget_ratio),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.minimum_budget_ratio > self.maximum_budget_ratio:
            raise ValueError(
                "minimum_budget_ratio must not exceed maximum_budget_ratio"
            )


@dataclass(frozen=True)
class ServiceTimeCalibrationSnapshot:
    source: str
    static_budget_seconds: float
    effective_budget_seconds: float
    total_sample_count: int
    retained_sample_count: int
    window_size: int
    minimum_samples: int
    ewma_alpha: float
    ewma_seconds: float | None
    quantile_probability: float
    quantile_seconds: float | None
    safety_multiplier: float
    minimum_budget_seconds: float
    maximum_budget_seconds: float
    route_key: str | None = None


def calibrate_service_time_budget(
    *,
    static_budget_seconds: float,
    samples: tuple[float, ...],
    total_sample_count: int,
    config: ServiceTimeCalibrationConfig,
    route_key: str | None = None,
) -> ServiceTimeCalibrationSnapshot:
    if not math.isfinite(static_budget_seconds) or static_budget_seconds <= 0.0:
        raise ValueError("static_budget_seconds must be finite and positive")
    if (
        not isinstance(total_sample_count, int)
        or isinstance(total_sample_count, bool)
        or total_sample_count < 0
    ):
        raise ValueError("total_sample_count must be a nonnegative integer")
    if route_key is not None and (
        not isinstance(route_key, str) or not route_key or len(route_key) > 256
    ):
        raise ValueError("route_key must be a nonempty string of at most 256 characters or null")
    if len(samples) > config.window_size:
        raise ValueError("samples exceed calibration window_size")
    if total_sample_count < len(samples):
        raise ValueError("total_sample_count must cover retained samples")
    values: list[float] = []
    for value in samples:
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError("service-time samples must be finite and positive")
        values.append(number)

    minimum_budget = static_budget_seconds * config.minimum_budget_ratio
    maximum_budget = static_budget_seconds * config.maximum_budget_ratio
    ewma: float | None = None
    quantile: float | None = None
    source = "static_cold_start"
    effective = static_budget_seconds
    if len(values) >= config.minimum_samples:
        ewma = values[0]
        for value in values[1:]:
            ewma = config.ewma_alpha * value + (1.0 - config.ewma_alpha) * ewma
        quantile = _linear_quantile(values, config.quantile_probability)
        calibrated = max(ewma, quantile) * config.safety_multiplier
        effective = min(max(calibrated, minimum_budget), maximum_budget)
        source = "adaptive_quantile"

    return ServiceTimeCalibrationSnapshot(
        source=source,
        static_budget_seconds=static_budget_seconds,
        effective_budget_seconds=effective,
        total_sample_count=total_sample_count,
        retained_sample_count=len(values),
        window_size=config.window_size,
        minimum_samples=config.minimum_samples,
        ewma_alpha=config.ewma_alpha,
        ewma_seconds=ewma,
        quantile_probability=config.quantile_probability,
        quantile_seconds=quantile,
        safety_multiplier=config.safety_multiplier,
        minimum_budget_seconds=minimum_budget,
        maximum_budget_seconds=maximum_budget,
        route_key=route_key,
    )


def _linear_quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight
