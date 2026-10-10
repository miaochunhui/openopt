from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


BackendName = Literal["osqp", "clarabel", "openpo_cpu", "openpo_cuda", "openpo_multi_gpu"]


@dataclass(frozen=True)
class RouterThresholds:
    gpu_batch_threshold: int = 128
    gpu_asset_threshold: int = 512
    multi_gpu_batch_threshold: int = 4096
    osqp_asset_threshold: int = 256
    clarabel_asset_threshold: int = 1024

    def __post_init__(self) -> None:
        _require_positive_int("gpu_batch_threshold", self.gpu_batch_threshold)
        _require_positive_int("gpu_asset_threshold", self.gpu_asset_threshold)
        _require_positive_int("multi_gpu_batch_threshold", self.multi_gpu_batch_threshold)
        _require_positive_int("osqp_asset_threshold", self.osqp_asset_threshold)
        _require_positive_int("clarabel_asset_threshold", self.clarabel_asset_threshold)
        if self.multi_gpu_batch_threshold < self.gpu_batch_threshold:
            raise ValueError("multi_gpu_batch_threshold must be >= gpu_batch_threshold")
        if self.clarabel_asset_threshold < self.osqp_asset_threshold:
            raise ValueError("clarabel_asset_threshold must be >= osqp_asset_threshold")


@dataclass(frozen=True)
class RouteDecision:
    backend: BackendName
    reason: str


def _require_positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def route_backend(
    assets: int,
    batch: int,
    gpu_count: int,
    gpu_is_warm: bool = False,
    thresholds: RouterThresholds = RouterThresholds(),
) -> RouteDecision:
    _require_positive_int("assets", assets)
    _require_positive_int("batch", batch)
    _require_nonnegative_int("gpu_count", gpu_count)
    if not isinstance(gpu_is_warm, bool):
        raise TypeError("gpu_is_warm must be a boolean")

    if gpu_count >= 2 and batch >= thresholds.multi_gpu_batch_threshold:
        return RouteDecision(
            backend="openpo_multi_gpu",
            reason=(
                f"gpu_count={gpu_count} and batch={batch} >= "
                f"multi_gpu_batch_threshold={thresholds.multi_gpu_batch_threshold}"
            ),
        )
    if gpu_count >= 1 and (
        batch >= thresholds.gpu_batch_threshold
        or (gpu_is_warm and assets >= thresholds.gpu_asset_threshold)
    ):
        return RouteDecision(
            backend="openpo_cuda",
            reason=(
                f"gpu_count={gpu_count}; batch={batch}; assets={assets}; "
                f"gpu_is_warm={gpu_is_warm}; gpu_batch_threshold="
                f"{thresholds.gpu_batch_threshold}; gpu_asset_threshold="
                f"{thresholds.gpu_asset_threshold}"
            ),
        )

    if batch < thresholds.gpu_batch_threshold:
        if assets <= thresholds.osqp_asset_threshold:
            return RouteDecision(
                backend="osqp",
                reason=(
                    f"batch={batch} < gpu_batch_threshold={thresholds.gpu_batch_threshold} "
                    f"and assets={assets} <= osqp_asset_threshold={thresholds.osqp_asset_threshold}"
                ),
            )
        if assets <= thresholds.clarabel_asset_threshold:
            return RouteDecision(
                backend="clarabel",
                reason=(
                    f"batch={batch} < gpu_batch_threshold={thresholds.gpu_batch_threshold} "
                    f"and assets={assets} <= clarabel_asset_threshold={thresholds.clarabel_asset_threshold}"
                ),
            )

    return RouteDecision(
        backend="openpo_cpu",
        reason=(
            f"using CPU fallback with batch={batch}; "
            f"gpu_count={gpu_count}; "
            f"assets={assets}; "
            f"gpu_batch_threshold={thresholds.gpu_batch_threshold}"
        ),
    )


select_backend = route_backend
