import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("clarabel")
pytest.importorskip("osqp")
pytest.importorskip("scipy")

from openpo.projection import project_bounded_simplex  # noqa: E402
from openpo.exact_solver import configured_clarabel_settings  # noqa: E402

COMPARE_OPEN_SOURCE = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_open_source.py"
COMPARE_OPEN_SOURCE_SPEC = importlib.util.spec_from_file_location(
    "compare_open_source", COMPARE_OPEN_SOURCE
)
assert COMPARE_OPEN_SOURCE_SPEC is not None and COMPARE_OPEN_SOURCE_SPEC.loader is not None
COMPARE_OPEN_SOURCE_MODULE = importlib.util.module_from_spec(COMPARE_OPEN_SOURCE_SPEC)
sys.modules[COMPARE_OPEN_SOURCE_SPEC.name] = COMPARE_OPEN_SOURCE_MODULE
COMPARE_OPEN_SOURCE_SPEC.loader.exec_module(COMPARE_OPEN_SOURCE_MODULE)

add_quality_gaps = COMPARE_OPEN_SOURCE_MODULE.add_quality_gaps
build_case = COMPARE_OPEN_SOURCE_MODULE.build_case
benchmark_objective = COMPARE_OPEN_SOURCE_MODULE.objective
run_clarabel = COMPARE_OPEN_SOURCE_MODULE.run_clarabel
run_openpo = COMPARE_OPEN_SOURCE_MODULE.run_openpo
run_osqp = COMPARE_OPEN_SOURCE_MODULE.run_osqp


def lifted_objective(case, weights, factor_load) -> np.ndarray:
    return (
        0.5
        * case.risk_aversion
        * (
            np.sum(weights * weights * case.specific_variance, axis=1)
            + np.sum(factor_load * factor_load * case.factor_variance, axis=1)
        )
        - np.sum(case.alpha * weights, axis=1)
    )


def assert_solver_quality(result: dict[str, object]) -> None:
    assert result["success_rate"] == pytest.approx(1.0)
    assert result["max_budget_error"] <= 2e-5
    assert result["max_lower_violation"] <= 2e-6
    assert result["max_upper_violation"] <= 2e-6
    assert result["median_objective_gap"] is not None
    assert result["max_objective_gap"] is not None
    assert result["median_objective_gap"] >= -1e-10
    assert result["max_objective_gap"] >= result["median_objective_gap"]
    assert result["max_objective_gap"] <= 1e-4


def test_clarabel_adapter_reuses_production_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert COMPARE_OPEN_SOURCE_MODULE.configured_clarabel_settings is configured_clarabel_settings
    captured_settings = []

    def capture_settings(backend_module):
        settings = configured_clarabel_settings(backend_module)
        captured_settings.append(settings)
        return settings

    monkeypatch.setattr(
        COMPARE_OPEN_SOURCE_MODULE,
        "configured_clarabel_settings",
        capture_settings,
    )
    COMPARE_OPEN_SOURCE_MODULE.clarabel_solver(
        build_case(assets=24, factors=4, batch=1, seed=20260809)
    )

    assert len(captured_settings) == 1
    assert captured_settings[0].direct_solve_method == "qdldl"
    assert captured_settings[0].max_threads == 1


def test_lifted_sparse_qp_objective_matches_factor_objective() -> None:
    case = build_case(assets=24, factors=4, batch=3, seed=20260809)
    values = torch.linspace(-0.4, 0.8, steps=case.batch * case.assets, dtype=torch.float32).reshape(
        case.batch, case.assets
    )
    weights = project_bounded_simplex(
        values,
        torch.as_tensor(case.lower, dtype=torch.float32),
        torch.as_tensor(case.upper, dtype=torch.float32),
        budget=case.budget,
    ).cpu().numpy()
    factor_load = weights @ case.exposures

    reduced = benchmark_objective(case, weights)
    lifted = lifted_objective(case, weights, factor_load)

    assert np.allclose(reduced, lifted, rtol=1e-12, atol=1e-12)
    assert factor_load.shape == (case.batch, case.exposures.shape[1])


def test_small_case_solvers_respect_constraints_and_gap() -> None:
    case = build_case(assets=24, factors=4, batch=3, seed=20260809)
    results = [
        run_osqp(case),
        run_clarabel(case),
        run_openpo(case, "cpu", iterations=250),
    ]
    add_quality_gaps(results)

    by_solver = {result["solver"]: result for result in results}
    osqp_name = f"OSQP {COMPARE_OPEN_SOURCE_MODULE.osqp.__version__} CPU"
    clarabel_name = f"Clarabel {COMPARE_OPEN_SOURCE_MODULE.clarabel.__version__} CPU"
    assert set(by_solver) == {
        osqp_name,
        clarabel_name,
        "OpenPO cpu",
    }

    for result in by_solver.values():
        assert_solver_quality(result)

    assert by_solver[osqp_name]["max_objective_gap"] <= 1e-8
    assert by_solver[clarabel_name]["max_objective_gap"] <= 1e-8
    assert by_solver["OpenPO cpu"]["max_objective_gap"] <= 1e-4
