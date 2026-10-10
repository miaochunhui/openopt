import importlib.util
import sys
from pathlib import Path

import pytest

from openpo.exact_solver import configured_clarabel_settings
from openpo.router import RouteDecision, RouterThresholds, route_backend


PROFILE_ROUTER_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "profile_router.py"


def test_clarabel_adapter_reuses_production_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for package in ("clarabel", "osqp", "scipy", "torch"):
        pytest.importorskip(package)
    spec = importlib.util.spec_from_file_location("profile_router", PROFILE_ROUTER_PATH)
    assert spec is not None and spec.loader is not None
    profile_router = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = profile_router
    spec.loader.exec_module(profile_router)

    assert profile_router.configured_clarabel_settings is configured_clarabel_settings
    captured_settings = []

    def capture_settings(backend_module):
        settings = configured_clarabel_settings(backend_module)
        captured_settings.append(settings)
        return settings

    monkeypatch.setattr(profile_router, "configured_clarabel_settings", capture_settings)
    profile_router.clarabel_solver(
        profile_router.build_case(assets=24, factors=4, batch=1, seed=20260809)
    )

    assert len(captured_settings) == 1
    assert captured_settings[0].direct_solve_method == "qdldl"
    assert captured_settings[0].max_threads == 1


@pytest.mark.parametrize(
    ("assets", "batch", "gpu_count", "backend", "reason_snippet"),
    [
        (64, 64, 0, "osqp", "osqp_asset_threshold=256"),
        (512, 64, 0, "clarabel", "clarabel_asset_threshold=1024"),
        (2048, 64, 0, "openpo_cpu", "CPU fallback"),
        (128, 128, 1, "openpo_cuda", "gpu_batch_threshold=128"),
        (128, 2048, 2, "openpo_cuda", "gpu_batch_threshold=128"),
        (128, 4096, 2, "openpo_multi_gpu", "multi_gpu_batch_threshold=4096"),
    ],
)
def test_route_backend_defaults(
    assets: int,
    batch: int,
    gpu_count: int,
    backend: str,
    reason_snippet: str,
) -> None:
    decision = route_backend(assets=assets, batch=batch, gpu_count=gpu_count)
    assert isinstance(decision, RouteDecision)
    assert decision.backend == backend
    assert reason_snippet in decision.reason


def test_route_backend_uses_custom_thresholds() -> None:
    thresholds = RouterThresholds(
        gpu_batch_threshold=64,
        gpu_asset_threshold=96,
        multi_gpu_batch_threshold=256,
        osqp_asset_threshold=32,
        clarabel_asset_threshold=128,
    )

    cpu_decision = route_backend(assets=48, batch=32, gpu_count=0, thresholds=thresholds)
    gpu_decision = route_backend(assets=48, batch=128, gpu_count=1, thresholds=thresholds)
    multi_gpu_decision = route_backend(assets=48, batch=256, gpu_count=2, thresholds=thresholds)

    assert cpu_decision.backend == "clarabel"
    assert gpu_decision.backend == "openpo_cuda"
    assert multi_gpu_decision.backend == "openpo_multi_gpu"


def test_warm_gpu_routes_large_assets_at_small_batch() -> None:
    cold = route_backend(assets=1000, batch=8, gpu_count=1, gpu_is_warm=False)
    warm = route_backend(assets=1000, batch=8, gpu_count=1, gpu_is_warm=True)
    assert cold.backend == "clarabel"
    assert warm.backend == "openpo_cuda"
    assert "gpu_is_warm=True" in warm.reason


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: route_backend(assets=0, batch=1, gpu_count=0), "assets must be positive"),
        (lambda: route_backend(assets=1, batch=0, gpu_count=0), "batch must be positive"),
        (lambda: route_backend(assets=1, batch=1, gpu_count=-1), "gpu_count must be nonnegative"),
        (
            lambda: route_backend(
                assets=1,
                batch=1,
                gpu_count=0,
                thresholds=RouterThresholds(
                    gpu_batch_threshold=128,
                    multi_gpu_batch_threshold=64,
                    osqp_asset_threshold=32,
                    clarabel_asset_threshold=128,
                ),
            ),
            "multi_gpu_batch_threshold must be >= gpu_batch_threshold",
        ),
        (
            lambda: route_backend(
                assets=1,
                batch=1,
                gpu_count=0,
                thresholds=RouterThresholds(
                    gpu_batch_threshold=128,
                    multi_gpu_batch_threshold=256,
                    osqp_asset_threshold=128,
                    clarabel_asset_threshold=64,
                ),
            ),
            "clarabel_asset_threshold must be >= osqp_asset_threshold",
        ),
    ],
)
def test_route_backend_fail_loud(factory, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()
