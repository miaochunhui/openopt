from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


GENERATE_CONSTRAINED_CASES = Path(__file__).resolve().parents[1] / "benchmarks" / "generate_constrained_cases.py"
SPEC = importlib.util.spec_from_file_location("generate_constrained_cases", GENERATE_CONSTRAINED_CASES)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

build_case = MODULE.build_case
build_suite = MODULE.build_suite
save_suite = MODULE.save_suite


def test_build_case_is_deterministic_and_hash_stable() -> None:
    case_a = build_case(seed=20260809, long_short=False)
    case_b = build_case(seed=20260809, long_short=False)

    assert case_a.input_sha256() == case_b.input_sha256()
    bundle_a = case_a.to_numpy_bundle()
    bundle_b = case_b.to_numpy_bundle()
    assert set(bundle_a) == set(bundle_b)
    for key in bundle_a:
        assert np.array_equal(bundle_a[key], bundle_b[key]), key


def test_cases_have_known_strict_feasible_point() -> None:
    for case in build_suite(seed=20260809):
        residuals = case.residual_report()
        assert residuals["budget_error"] <= 1e-12
        assert residuals["market_error"] <= 1e-12
        assert residuals["max_box_lower_violation"] <= 1e-12
        assert residuals["max_box_upper_violation"] <= 1e-12
        assert residuals["max_inequality_lower_violation"] <= 1e-12
        assert residuals["max_inequality_upper_violation"] <= 1e-12
        assert residuals["max_equality_error"] <= 1e-12
        assert residuals["min_box_slack"] > 0.0
        assert residuals["min_inequality_slack"] > 0.0


def test_long_short_case_is_signed_and_market_neutral() -> None:
    case = build_case(seed=20260809, long_short=True)
    assert case.long_short is True
    assert case.budget == 0.0
    assert np.any(case.feasible_weights < 0.0)
    assert np.any(case.feasible_weights > 0.0)
    assert np.any(case.box_lower < 0.0)
    assert np.any(case.box_upper > 0.0)
    assert abs(float(case.feasible_weights.sum())) <= 1e-12
    assert abs(float(case.market_loading @ case.feasible_weights)) <= 1e-12


def test_save_suite_round_trips_numpy_payload(tmp_path: Path) -> None:
    manifests = save_suite(tmp_path, seed=20260809)
    assert len(manifests) == 2
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.is_file()

    case = build_case(seed=20260809, long_short=False)
    npz_path = tmp_path / f"{case.name}.npz"
    assert npz_path.is_file()

    payload = np.load(npz_path, allow_pickle=False)
    assert payload["input_sha256"].item() == case.input_sha256()
    assert np.array_equal(payload["feasible_weights"], case.feasible_weights)
    assert np.array_equal(payload["equality_matrix"], case.equality_matrix)
