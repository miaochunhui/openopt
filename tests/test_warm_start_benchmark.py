from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("clarabel")
pytest.importorskip("osqp")
pytest.importorskip("scipy")
pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"

for path in (SRC, BENCHMARKS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from openpo.exact_solver import configured_clarabel_settings  # noqa: E402


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("compare_open_source", BENCHMARKS / "compare_open_source.py")
COMPARE_CONSTRAINED = _load_module("compare_constrained", BENCHMARKS / "compare_constrained.py")
COMPARE_WARM_START = _load_module("compare_warm_start", BENCHMARKS / "compare_warm_start.py")


def _result_by_solver_prefix(report: dict[str, object], prefix: str) -> dict[str, object]:
    case = report["cases"][0]
    for result in case["results"]:
        if str(result["solver"]).startswith(prefix):
            return result
    raise AssertionError(f"missing result for solver prefix: {prefix}")


def _result_by_mode(report: dict[str, object], mode: str) -> dict[str, object]:
    case = report["cases"][0]
    for result in case["results"]:
        if result["mode"] == mode:
            return result
    raise AssertionError(f"missing result for mode: {mode}")


def _small_report() -> dict[str, object]:
    return COMPARE_WARM_START.build_case_report(
        assets=8,
        batch=2,
        horizon=3,
        portfolio_mode="long_only",
        seed=20260809,
        devices=["cpu"],
        ar1_phi=0.97,
        alpha_scale=0.002,
        iterations=800,
        cpu_bisection_steps=8,
        cuda_bisection_steps=8,
    )


def test_clarabel_adapter_reuses_production_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert COMPARE_WARM_START.configured_clarabel_settings is configured_clarabel_settings
    captured_settings = []

    def capture_settings(backend_module):
        settings = configured_clarabel_settings(backend_module)
        captured_settings.append(settings)
        return settings

    monkeypatch.setattr(COMPARE_WARM_START, "configured_clarabel_settings", capture_settings)
    case = COMPARE_CONSTRAINED.build_benchmark_case(assets=8, batch=1, seed=20260809)
    alpha_path = COMPARE_WARM_START.build_alpha_path(
        assets=8,
        batch=1,
        horizon=2,
        seed=20260809,
        ar1_phi=0.97,
        alpha_scale=0.002,
    )
    COMPARE_WARM_START.run_clarabel_path(case, alpha_path)

    assert len(captured_settings) == 1
    assert captured_settings[0].direct_solve_method == "qdldl"
    assert captured_settings[0].max_threads == 1


def test_ar1_path_and_hash_are_deterministic() -> None:
    params = dict(
        assets=8,
        batch=2,
        horizon=3,
        seed=20260809,
        ar1_phi=0.97,
        alpha_scale=0.002,
    )

    path_a = COMPARE_WARM_START.build_alpha_path(**params)
    path_b = COMPARE_WARM_START.build_alpha_path(**params)

    assert path_a.shape == (3, 2, 8)
    np.testing.assert_array_equal(path_a, path_b)
    assert COMPARE_WARM_START.path_sha256(path_a, params["ar1_phi"], params["alpha_scale"]) == COMPARE_WARM_START.path_sha256(
        path_b, params["ar1_phi"], params["alpha_scale"]
    )


@pytest.fixture(scope="module")
def warm_start_report() -> dict[str, object]:
    case = _small_report()
    return {
        "schema_version": 1,
        "generated_at": "2026-08-09T00:00:00+0800",
        "environment": COMPARE_CONSTRAINED.solver_environment(),
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
                "cpu": 8,
                "cuda": 8,
            },
            "timing": "p50/p95 across rolling steps 1..2; iterations pooled across measured solves",
        },
        "cases": [case],
        "wall_seconds": 0.0,
    }


def test_report_schema_has_key_fields(warm_start_report: dict[str, object]) -> None:
    report = warm_start_report
    case = report["cases"][0]

    assert report["schema_version"] == 1
    assert {"schema_version", "generated_at", "environment", "methodology", "cases", "wall_seconds"} <= report.keys()
    assert report["cases"]
    assert report["environment"]["cpu_threads"] >= 1

    assert {
        "assets",
        "batch",
        "horizon",
        "portfolio_mode",
        "seed",
        "path_seed",
        "case_sha256",
        "path_sha256",
        "alpha_path_max_delta",
        "openpo_warm_start_max_alpha_delta",
        "input_sha256",
        "risk_model",
        "reference_formulation",
        "alpha_process",
        "constraints",
        "results",
    } <= case.keys()

    assert case["assets"] == 8
    assert case["batch"] == 2
    assert case["horizon"] == 3
    assert case["portfolio_mode"] == "long_only"
    assert case["path_seed"] == 20260809 + 8 * 31 + 2 * 17 + 3 * 13
    assert case["alpha_process"] == {"kind": "AR(1)", "phi": 0.97, "alpha_scale": 0.002}
    assert case["input_sha256"] == COMPARE_WARM_START.combined_sha256(case["case_sha256"], case["path_sha256"])
    assert case["openpo_warm_start_max_alpha_delta"] == pytest.approx(
        case["alpha_path_max_delta"] + max(1e-12, 1e-6 * case["alpha_path_max_delta"])
    )

    assert len(case["results"]) == 4
    result_fields = {
        "solver",
        "mode",
        "device",
        "setup_seconds",
        "step_seconds_p50_ms",
        "step_seconds_p95_ms",
        "iterations_p50",
        "iterations_p95",
        "success_rate",
        "median_objective",
        "statuses",
        "weights_sha256",
        "max_budget_error",
        "max_box_lower_violation",
        "max_box_upper_violation",
        "max_box_violation",
        "max_linear_equality_error",
        "max_linear_lower_violation",
        "max_linear_upper_violation",
        "max_linear_violation",
        "median_objective_gap",
        "max_objective_gap",
    }
    for result in case["results"]:
        assert result_fields <= result.keys()
        assert len(result["statuses"]) == case["batch"] * (case["horizon"] - 1)
        assert result["device"] == "cpu"
        assert result["step_seconds_p50_ms"] > 0.0
        assert result["step_seconds_p95_ms"] >= result["step_seconds_p50_ms"]


def test_osqp_and_clarabel_small_case_are_feasible(warm_start_report: dict[str, object]) -> None:
    osqp_result = _result_by_solver_prefix(warm_start_report, "OSQP ")
    clarabel_result = _result_by_solver_prefix(warm_start_report, "Clarabel ")

    for result in (osqp_result, clarabel_result):
        assert result["mode"] == "q_update_hot"
        assert result["success_rate"] == pytest.approx(1.0)
        assert result["median_objective_gap"] is not None
        assert result["max_objective_gap"] is not None
        assert result["max_budget_error"] <= 2e-5
        assert result["max_box_violation"] <= 2e-6
        assert result["max_linear_violation"] <= 2e-6
        assert result["max_objective_gap"] <= 1e-4


def test_openpo_cold_and_warm_share_input_and_pass_quality(warm_start_report: dict[str, object]) -> None:
    cold = _result_by_mode(warm_start_report, "cold")
    warm = _result_by_mode(warm_start_report, "previous_primal_dual")

    assert cold["solver"] == "OpenPO cpu cold"
    assert warm["solver"] == "OpenPO cpu previous_primal_dual"
    assert cold["device"] == warm["device"] == "cpu"
    assert cold["success_rate"] == pytest.approx(1.0)
    assert warm["success_rate"] == pytest.approx(1.0)
    assert cold["median_objective_gap"] is not None
    assert warm["median_objective_gap"] is not None
    assert cold["max_objective_gap"] is not None
    assert warm["max_objective_gap"] is not None
    assert warm["median_objective_gap"] <= cold["median_objective_gap"] + 1e-8
    assert warm["max_objective_gap"] <= cold["max_objective_gap"] + 1e-8
    assert warm["max_budget_error"] <= 2e-5
    assert warm["max_box_violation"] <= 2e-6
    assert warm["max_linear_violation"] <= 2e-6
    assert warm["max_objective_gap"] <= 1e-4
