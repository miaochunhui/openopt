"""Original-coordinate RSOC checks and native projection, not GPU qualification."""

from dataclasses import replace
import math

import numpy as np
import pytest
import torch

from openopt import RotatedSecondOrderCone
from openopt.conic import (
    ConicConfig, ConicProblem, Nonnegative, RSOC, SOC, Zero,
    check_cone_feasibility, check_conic_candidate, solve_conic,
)
from openopt.native_conic import NativeConicConfig, project_cones, solve_native_conic


def rotated_lp():
    # Fix w=2; minimizing a+b in 2ab>=w^2 gives a=b=sqrt(2).
    return ConicProblem(
        np.zeros((3, 3)), np.array([1.0, 1.0, 0.0]),
        np.vstack((np.array([[0.0, 0.0, 1.0]]), -np.eye(3))),
        np.array([2.0, 0.0, 0.0, 0.0]), (Zero(1), RSOC(3)),
    )


@pytest.mark.parametrize("dimension", [True, 2, 0, -1, 3.0])
def test_invalid_rotated_dimension(dimension):
    with pytest.raises(ValueError):
        RSOC(dimension)


def test_public_identity():
    assert RSOC is RotatedSecondOrderCone
    assert RSOC(4).dimension == 4


@pytest.mark.parametrize("value,passed", [
    ([0.0, 0.0, 0.0], True),
    ([0.0, 2.0, 0.0], True),
    ([1.0, 2.0, 2.0], True),
    ([1.0, 2.0, 2.01], False),
    ([-1.0, -2.0, 0.0], False),
    ([-1.0, 2.0, 0.0], False),
    ([1e150, 1e150, 1e150], True),
    ([1e150, 1e150, 2e150], False),
])
def test_original_rotated_cone_feasibility(value, passed):
    primal = check_cone_feasibility(value, (RSOC(3),), atol=1e-12, rtol=1e-12)
    dual = check_cone_feasibility(value, (RSOC(3),), dual=True, atol=1e-12, rtol=1e-12)
    assert primal.passed is passed
    assert dual.passed is passed


def transform(value):
    result = value.clone()
    first, second = value[0], value[1]
    result[0] = first / math.sqrt(2.0) + second / math.sqrt(2.0)
    result[1] = first / math.sqrt(2.0) - second / math.sqrt(2.0)
    return result


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("value", [
    [1.0, 2.0, 2.0], [-1.0, -2.0, 0.0], [-1.0, 2.0, 3.0],
    [0.0, 0.0, 4.0], [3.0, 1.0, -2.0],
])
def test_projection_is_orthogonal_soc_projection_and_moreau(dtype, value):
    original = torch.tensor(value, dtype=dtype)
    before = original.clone()
    projected = project_cones(original, (RSOC(3),))
    reference = transform(project_cones(transform(original), (SOC(3),)))
    tolerance = 2e-6 if dtype == torch.float32 else 2e-14
    torch.testing.assert_close(projected, reference, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(original, before, atol=0, rtol=0)
    torch.testing.assert_close(
        projected, project_cones(original, (RSOC(3),), dual=True), atol=0, rtol=0,
    )
    negative_dual = project_cones(-original, (RSOC(3),), dual=True)
    torch.testing.assert_close(original, projected - negative_dual, atol=tolerance, rtol=tolerance)
    assert abs(float(projected @ negative_dual)) <= tolerance * max(1.0, float(original @ original))
    assert check_cone_feasibility(projected, (RSOC(3),), atol=tolerance, rtol=tolerance).passed


def test_mixed_product_preserves_offsets():
    value = torch.tensor([9.0, -2.0, -1.0, 2.0, 3.0, 4.0, 0.0], dtype=torch.float64)
    cones = (Zero(1), Nonnegative(1), RSOC(3), SOC(2))
    result = project_cones(value, cones)
    assert result[0] == result[1] == 0
    torch.testing.assert_close(result[2:5], project_cones(value[2:5], (RSOC(3),)))
    torch.testing.assert_close(result[5:], value[5:])


def test_analytic_original_kkt_and_wrong_dual_rejected():
    model = rotated_lp()
    root = math.sqrt(2.0)
    x = np.array([root, root, 2.0])
    s = np.array([0.0, root, root, 2.0])
    z = np.array([-root, 1.0, 1.0, -root])
    evidence = check_conic_candidate(model, x, s, z, atol=1e-12, rtol=1e-12)
    assert evidence.passed
    assert abs(evidence.metrics["duality_gap"]) < 1e-12
    wrong_equality = z.copy()
    wrong_equality[0] *= -1
    assert not check_conic_candidate(model, x, s, wrong_equality).checks["stationarity"]
    changed = z.copy()
    changed[1:] *= -1
    bad = check_conic_candidate(model, x, s, changed, atol=1e-12, rtol=1e-12)
    assert not bad.passed and not bad.checks["dual_cone"]
    assert not check_conic_candidate(model, x, s + 0.1, z).passed


def test_native_original_model_and_reference_agree():
    model = rotated_lp()
    before = {name: getattr(model, name).copy() for name in ("P", "q", "A", "b")}
    config = NativeConicConfig(enabled=True, eps_abs=1e-8, eps_rel=0)
    native = solve_native_conic(model, config)
    reference = solve_conic(model, ConicConfig(backend="clarabel_reference", eps_abs=1e-9))
    assert native.status == "solved" and native.evidence.passed
    assert reference.status == "reference_solved" and reference.evidence.passed
    np.testing.assert_allclose(native.x.numpy(), [math.sqrt(2.0), math.sqrt(2.0), 2.0], atol=2e-7)
    np.testing.assert_allclose(native.x.numpy(), reference.x, atol=2e-7)
    assert abs(native.objective - reference.objective) < 2e-7
    for name, value in before.items():
        np.testing.assert_array_equal(getattr(model, name), value)
    assert check_conic_candidate(model, reference.x, reference.s, reference.z).passed


def test_explicit_cuda_never_falls_back(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = NativeConicConfig(enabled=True, device="cuda:0")
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        solve_native_conic(rotated_lp(), config)
    with pytest.raises(RuntimeError, match="no native GPU backend"):
        solve_conic(rotated_lp(), ConicConfig(backend="clarabel_reference", device="cuda:0"))


def test_unfinished_native_is_not_solved():
    result = solve_native_conic(
        rotated_lp(), NativeConicConfig(enabled=True, max_iterations=1, check_interval=1),
    )
    assert result.status == "max_iterations" and not result.evidence.passed


def test_false_cone_slack_cannot_hide_original_infeasibility():
    model = rotated_lp()
    x = np.array([0.0, 0.0, 2.0])
    planted = np.array([0.0, 2.0, 2.0, 2.0])
    evidence = check_conic_candidate(model, x, planted, np.zeros(4))
    assert not evidence.passed
    assert not evidence.checks["primal_equation"]
    assert not evidence.checks["original_slack_cone"]


def test_nonfinite_and_wrong_dimensions_rejected():
    with pytest.raises(ValueError, match="finite"):
        check_cone_feasibility([1.0, 1.0, np.inf], (RSOC(3),))
    with pytest.raises(ValueError, match="sum of cone dimensions"):
        project_cones(torch.ones(4, dtype=torch.float64), (RSOC(3),))
    with pytest.raises(ValueError, match="sum of cone dimensions"):
        solve_native_conic(
            replace(rotated_lp(), cones=(Zero(1), RSOC(4))),
            NativeConicConfig(enabled=True),
        )


@pytest.mark.parametrize("value", [
    [-1.0, 1e20, 0.0], [1e20, -1.0, 0.0],
])
def test_large_opposite_head_cannot_hide_negative_coordinate(value):
    assert not check_cone_feasibility(value, (RSOC(3),)).passed
    model = ConicProblem(
        np.zeros((1, 1)), np.zeros(1), np.zeros((3, 1)),
        np.array(value), (RSOC(3),),
    )
    evidence = check_conic_candidate(model, [0.0], value, [0.0, 0.0, 0.0])
    assert not evidence.passed and not evidence.checks["original_slack_cone"]


@pytest.mark.parametrize("value", [
    [0.0, 1e20, 1e8], [1e20, 0.0, 1e8],
])
def test_cancellation_does_not_hide_original_head_deficit(value):
    evidence = check_cone_feasibility(value, (RSOC(3),), atol=1e-12, rtol=0)
    assert not evidence.passed


@pytest.mark.parametrize("value", [
    [1.0, 1e20, 0.0], [1.3e308, 1.3e308, 0.0],
])
def test_projection_preserves_extreme_feasible_coordinates(value):
    original = torch.tensor(value, dtype=torch.float64)
    torch.testing.assert_close(
        project_cones(original, (RSOC(3),)), original, atol=0, rtol=0,
    )


def test_projection_near_negative_cone_retains_small_nonzero_solution():
    original = torch.tensor([-1e20, 0.0, 1e8], dtype=torch.float64)
    projected = project_cones(original, (RSOC(3),))
    expected = torch.tensor([2.5e-29, 5e-5, 5e-17], dtype=torch.float64)
    torch.testing.assert_close(projected, expected, atol=0, rtol=2e-14)


@pytest.mark.parametrize("dtype,value,expected,tolerance", [
    (torch.float64, [-1e300, 0.0, 1e210], [2.5e-61, 5e119, 5e29], 3e-14),
    (torch.float32, [-1e30, 0.0, 1e17], [2.5e-23, 5e3, 5e-10], 3e-6),
])
def test_small_projected_head_survives_intermediate_square(dtype, value, expected, tolerance):
    projected = project_cones(torch.tensor(value, dtype=dtype), (RSOC(3),))
    torch.testing.assert_close(
        projected, torch.tensor(expected, dtype=dtype), atol=0, rtol=tolerance,
    )


def test_projection_does_not_materialize_overflowing_tail_norm():
    value = torch.tensor([0.0, 0.0, 1.3e308, 1.3e308], dtype=torch.float64)
    projected = project_cones(value, (RSOC(4),))
    torch.testing.assert_close(
        projected, torch.full((4,), 6.5e307, dtype=torch.float64), atol=0, rtol=3e-14,
    )


def test_original_checker_accepts_extreme_boundary_without_tail_norm_overflow():
    value = [1.3e308, 1.3e308, 1.3e308, 1.3e308]
    assert check_cone_feasibility(value, (RSOC(4),), atol=1e-12, rtol=1e-12).passed
