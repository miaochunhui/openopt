from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture(scope="module")
def benchmark_module() -> ModuleType:
    path = Path(__file__).parents[1] / "benchmarks" / "benchmark_trade_style_qp.py"
    spec = importlib.util.spec_from_file_location("benchmark_trade_style_qp_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parse_args_and_fixed_seed_input_hash(benchmark_module: ModuleType) -> None:
    args = benchmark_module.parse_args(
        [
            "--assets",
            "12",
            "--styles",
            "3",
            "--seed",
            "20260812",
            "--warmups",
            "2",
            "--repeats",
            "4",
            "--device",
            "cpu",
        ]
    )
    assert (args.assets, args.styles, args.seed, args.warmups, args.repeats) == (
        12,
        3,
        20260812,
        2,
        4,
    )
    first = benchmark_module.build_case(args)
    second = benchmark_module.build_case(args)
    assert first.input_sha256 == second.input_sha256
    assert len(first.input_sha256) == 64
    assert first.seed == second.seed == 20260812


def test_summary_excludes_cold_and_warmup_and_reports_quality(
    benchmark_module: ModuleType,
) -> None:
    route = benchmark_module.summarize_route(
        name="fallback",
        device="cpu",
        cold=[benchmark_module.Sample(9.0, True, 1.0, 0.1, 0.2, 0.3, 90)],
        warmup=[benchmark_module.Sample(8.0, True, 2.0, 0.2, 0.3, 0.4, 80)],
        hot=[
            benchmark_module.Sample(0.001, True, 3.0, 1e-9, 2e-9, 3e-9, 30),
            benchmark_module.Sample(0.002, False, 5.0, 2e-9, 3e-9, 4e-9, 40),
            benchmark_module.Sample(0.004, True, 7.0, 3e-9, 4e-9, 5e-9, 50),
        ],
    )
    assert route["cold"]["samples"] == 1
    assert route["warmup"]["samples"] == 1
    assert route["hot"]["samples"] == 3
    assert route["hot"]["p50_ms"] == pytest.approx(2.0)
    assert route["hot"]["p95_ms"] == pytest.approx(3.8)
    assert route["success_rate"] == pytest.approx(2 / 3)
    assert route["objective"]["p50"] == pytest.approx(5.0)
    assert route["residuals"] == {
        "budget_max": pytest.approx(3e-9),
        "box_max": pytest.approx(4e-9),
        "stationarity_max": pytest.approx(5e-9),
    }
    assert route["iterations"]["p50"] == pytest.approx(40.0)
    assert route["iterations"]["p95"] == pytest.approx(49.0)
    assert route["estimated_30_day_seconds"] == pytest.approx(0.060)


def test_fused_fallback_difference_and_environment_contract(
    benchmark_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fused = benchmark_module.NumericalResult(
        objective=1.0,
        dw=[0.2, -0.2],
        exposure=[0.1, -0.1],
    )
    fallback = benchmark_module.NumericalResult(
        objective=1.0 + 1e-10,
        dw=[0.2 + 2e-10, -0.2 - 2e-10],
        exposure=[0.1 + 3e-10, -0.1 - 3e-10],
    )
    difference = benchmark_module.compare_numerical(fused, fallback)
    assert difference["objective_abs_gap"] == pytest.approx(1e-10)
    assert difference["dw_linf"] == pytest.approx(2e-10)
    assert difference["exposure_linf"] == pytest.approx(3e-10)

    monkeypatch.setattr(benchmark_module.torch.cuda, "is_available", lambda: False)
    environment = benchmark_module.collect_environment()
    for key in ("python", "torch", "triton", "platform", "machine", "gpus"):
        assert key in environment
    assert environment["gpus"] == []


@pytest.mark.parametrize(
    "sample",
    [
        lambda module: module.Sample(math.nan, True, 1.0, 0.0, 0.0, 0.0, 1),
        lambda module: module.Sample(0.1, True, math.inf, 0.0, 0.0, 0.0, 1),
        lambda module: module.Sample(0.1, True, 1.0, 0.0, 0.0, 0.0, 0),
    ],
)
def test_invalid_samples_fail_loud(
    benchmark_module: ModuleType,
    sample: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="finite|positive|iterations"):
        benchmark_module.summarize_route(
            name="fallback",
            device="cpu",
            cold=[],
            warmup=[],
            hot=[sample(benchmark_module)],
        )


def test_atomic_json_preserves_previous_file_on_failure(
    benchmark_module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output = tmp_path / "report.json"
    benchmark_module.atomic_write_json(output, {"status": "ok", "value": 1})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "ok",
        "value": 1,
    }
    assert not list(tmp_path.glob("*.tmp"))

    def fail_dump(*unused_args: object, **unused_kwargs: object) -> None:
        raise RuntimeError("forced json failure")

    monkeypatch.setattr(benchmark_module.json, "dump", fail_dump)
    with pytest.raises(RuntimeError, match="forced json failure"):
        benchmark_module.atomic_write_json(output, {"status": "bad"})
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "status": "ok",
        "value": 1,
    }
    assert not list(tmp_path.glob("*.tmp"))
