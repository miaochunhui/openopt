from __future__ import annotations

import pytest

from openpo.service_time_calibration import (
    ServiceTimeCalibrationConfig,
    calibrate_service_time_budget,
)


def test_calibration_uses_static_budget_until_minimum_samples() -> None:
    config = ServiceTimeCalibrationConfig(window_size=4, minimum_samples=3)
    snapshot = calibrate_service_time_budget(
        static_budget_seconds=10.0,
        samples=(7.0, 8.0),
        total_sample_count=2,
        config=config,
    )

    assert snapshot.source == "static_cold_start"
    assert snapshot.effective_budget_seconds == 10.0
    assert snapshot.ewma_seconds is None
    assert snapshot.quantile_seconds is None


def test_calibration_combines_ewma_quantile_and_bounds() -> None:
    config = ServiceTimeCalibrationConfig(
        window_size=4,
        minimum_samples=3,
        ewma_alpha=0.5,
        quantile_probability=0.75,
        safety_multiplier=1.0,
        minimum_budget_ratio=0.5,
        maximum_budget_ratio=1.5,
    )
    snapshot = calibrate_service_time_budget(
        static_budget_seconds=10.0,
        samples=(4.0, 8.0, 12.0, 8.0),
        total_sample_count=9,
        config=config,
        route_key="route-a",
    )

    assert snapshot.source == "adaptive_quantile"
    assert snapshot.ewma_seconds == pytest.approx(8.5)
    assert snapshot.quantile_seconds == pytest.approx(9.0)
    assert snapshot.effective_budget_seconds == pytest.approx(9.0)
    assert snapshot.total_sample_count == 9
    assert snapshot.retained_sample_count == 4
    assert snapshot.route_key == "route-a"


@pytest.mark.parametrize(
    "samples",
    ((0.0,), (-1.0,), (float("nan"),), (float("inf"),)),
)
def test_calibration_rejects_invalid_samples(samples: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        calibrate_service_time_budget(
            static_budget_seconds=10.0,
            samples=samples,
            total_sample_count=1,
            config=ServiceTimeCalibrationConfig(minimum_samples=1),
        )
