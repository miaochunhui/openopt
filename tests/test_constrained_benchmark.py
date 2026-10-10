from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("clarabel")
pytest.importorskip("osqp")
pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpo.exact_solver import configured_clarabel_settings  # noqa: E402


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("generate_constrained_cases", BENCHMARKS / "generate_constrained_cases.py")
_load_module("compare_open_source", BENCHMARKS / "compare_open_source.py")
COMPARE_CONSTRAINED = _load_module("compare_constrained", BENCHMARKS / "compare_constrained.py")

build_benchmark_case = COMPARE_CONSTRAINED.build_benchmark_case
build_lifted_qp = COMPARE_CONSTRAINED.build_lifted_qp
run_case = COMPARE_CONSTRAINED.run_case


def test_build_benchmark_case_hash_is_stable() -> None:
    case_a = build_benchmark_case(assets=16, batch=1, seed=20260809)
    case_b = build_benchmark_case(assets=16, batch=1, seed=20260809)

    assert case_a.input_sha256() == case_b.input_sha256()


def test_clarabel_adapter_reuses_production_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    assert COMPARE_CONSTRAINED.configured_clarabel_settings is configured_clarabel_settings
    captured_settings = []

    def capture_settings(backend_module):
        settings = configured_clarabel_settings(backend_module)
        captured_settings.append(settings)
        return settings

    monkeypatch.setattr(COMPARE_CONSTRAINED, "configured_clarabel_settings", capture_settings)
    COMPARE_CONSTRAINED.clarabel_solver(
        build_benchmark_case(assets=16, batch=1, seed=20260809)
    )

    assert len(captured_settings) == 1
    assert captured_settings[0].direct_solve_method == "qdldl"
    assert captured_settings[0].max_threads == 1


def test_lifted_qp_and_clarabel_rhs_dimensions() -> None:
    case = build_benchmark_case(assets=16, batch=1, seed=20260809)
    p_matrix, matrix, linear_term, lower, upper = build_lifted_qp(case)
    constrained = case.constrained_case
    factors = case.risk_case.factors
    box_rows = constrained.box_lower.shape[0]
    linear_rows = constrained.industry_lower.shape[0] + constrained.style_lower.shape[0]
    rhs = np.concatenate(
        [
            np.array([case.risk_case.budget, constrained.market_target]),
            np.zeros(factors),
            constrained.box_upper,
            -constrained.box_lower,
            constrained.industry_upper,
            constrained.style_upper,
            -constrained.industry_lower,
            -constrained.style_lower,
        ]
    )

    assert p_matrix.shape == (case.assets + factors, case.assets + factors)
    assert matrix.shape[0] == rhs.shape[0]
    assert matrix.shape[0] == 2 + factors + 2 * box_rows + 2 * linear_rows
    assert matrix.shape[1] == case.assets + factors
    assert linear_term.shape == (case.assets + factors,)
    assert lower.shape == upper.shape == rhs.shape


def test_16_asset_batch1_solvers_are_feasible_and_close(monkeypatch) -> None:
    monkeypatch.setattr(COMPARE_CONSTRAINED.torch.cuda, "is_available", lambda: False)

    report = run_case(
        assets=16,
        batch=1,
        seed=20260809,
        iterations=800,
        cpu_bisection_steps=16,
        cuda_bisection_steps=16,
        repeats=1,
    )

    by_solver = {result["solver"]: result for result in report["results"]}
    expected_solvers = {
        f"OSQP {COMPARE_CONSTRAINED.osqp.__version__} CPU",
        f"Clarabel {COMPARE_CONSTRAINED.clarabel.__version__} CPU",
        "OpenPO cpu",
    }

    assert report["input_sha256"] == build_benchmark_case(16, 1, 20260809).input_sha256()
    assert set(by_solver) == expected_solvers

    for result in by_solver.values():
        assert result["success_rate"] == pytest.approx(1.0)
        assert result["median_objective_gap"] is not None
        assert result["max_objective_gap"] is not None
        assert result["median_objective_gap"] >= -1e-10
        assert result["max_objective_gap"] <= 1e-4
