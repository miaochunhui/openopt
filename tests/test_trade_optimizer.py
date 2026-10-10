from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from openpo import solve_func_2_lp


DTYPE = torch.float64


def _assert_result_evidence(
    result: object,
    score: torch.Tensor,
    max_sell: torch.Tensor,
    max_buy: torch.Tensor,
    spread: float,
    *,
    atol: float = 1e-12,
) -> None:
    dw = result.dw
    expected_objective = torch.dot(score, dw) - 0.5 * spread * dw.abs().sum()
    expected_budget_residual = dw.sum().abs()
    expected_box_residual = torch.maximum(
        (-max_sell - dw).clamp_min(0).amax(),
        (dw - max_buy).clamp_min(0).amax(),
    )

    assert result.device == score.device
    for value in (
        result.dw,
        result.objective,
        result.traded_volume,
        result.budget_residual,
        result.box_residual,
    ):
        assert value.device == score.device
    torch.testing.assert_close(result.objective, expected_objective, atol=atol, rtol=0.0)
    torch.testing.assert_close(result.traded_volume, dw.abs().sum(), atol=atol, rtol=0.0)
    torch.testing.assert_close(
        result.budget_residual,
        expected_budget_residual,
        atol=atol,
        rtol=0.0,
    )
    torch.testing.assert_close(result.box_residual, expected_box_residual, atol=atol, rtol=0.0)


def test_solve_func_2_lp_matches_hand_calculation() -> None:
    score = torch.tensor([0.10, 0.04, -0.02], dtype=DTYPE)
    max_sell = torch.tensor([0.0, 0.1, 0.5], dtype=DTYPE)
    max_buy = torch.tensor([0.3, 0.4, 0.0], dtype=DTYPE)
    spread = 0.02

    result = solve_func_2_lp(score, max_sell, max_buy, spread)

    torch.testing.assert_close(
        result.dw,
        torch.tensor([0.3, 0.2, -0.5], dtype=DTYPE),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(result.objective, torch.tensor(0.038, dtype=DTYPE))
    torch.testing.assert_close(result.traded_volume, torch.tensor(1.0, dtype=DTYPE))
    assert result.budget_residual.item() <= 1e-12
    assert result.box_residual.item() <= 1e-12
    _assert_result_evidence(result, score, max_sell, max_buy, spread)


def test_solve_func_2_lp_returns_zero_when_no_trade_is_profitable() -> None:
    score = torch.tensor([0.03, 0.02, 0.01], dtype=DTYPE)
    max_sell = torch.tensor([0.4, 0.4, 0.4], dtype=DTYPE)
    max_buy = torch.tensor([0.4, 0.4, 0.4], dtype=DTYPE)
    spread = 0.03

    result = solve_func_2_lp(score, max_sell, max_buy, spread)

    torch.testing.assert_close(result.dw, torch.zeros_like(score))
    assert result.objective.item() == 0.0
    assert result.traded_volume.item() == 0.0
    assert result.budget_residual.item() == 0.0
    assert result.box_residual.item() == 0.0
    _assert_result_evidence(result, score, max_sell, max_buy, spread)


def test_solve_func_2_lp_partially_fills_larger_buy_capacity() -> None:
    score = torch.tensor([0.08, 0.03, -0.05], dtype=DTYPE)
    max_sell = torch.tensor([0.0, 0.0, 0.25], dtype=DTYPE)
    max_buy = torch.tensor([0.6, 0.5, 0.0], dtype=DTYPE)
    spread = 0.01

    result = solve_func_2_lp(score, max_sell, max_buy, spread)

    torch.testing.assert_close(
        result.dw,
        torch.tensor([0.25, 0.0, -0.25], dtype=DTYPE),
        atol=1e-12,
        rtol=0.0,
    )
    torch.testing.assert_close(result.objective, torch.tensor(0.03, dtype=DTYPE))
    torch.testing.assert_close(result.traded_volume, torch.tensor(0.5, dtype=DTYPE))
    _assert_result_evidence(result, score, max_sell, max_buy, spread)


@pytest.mark.parametrize(
    ("score", "max_sell", "max_buy", "spread", "error", "match"),
    [
        ([0.1, -0.1], torch.ones(2), torch.ones(2), 0.01, TypeError, "score"),
        (
            torch.ones((1, 2), dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            0.01,
            ValueError,
            "score",
        ),
        (
            torch.empty(0, dtype=DTYPE),
            torch.empty(0, dtype=DTYPE),
            torch.empty(0, dtype=DTYPE),
            0.01,
            ValueError,
            "score",
        ),
        (
            torch.ones(2, dtype=DTYPE),
            torch.ones(3, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            0.01,
            ValueError,
            "same shape",
        ),
        (
            torch.ones(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int64),
            0.01,
            TypeError,
            "score",
        ),
        (
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=torch.float32),
            0.01,
            ValueError,
            "same dtype",
        ),
        (
            torch.tensor([math.nan, 0.0], dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            0.01,
            ValueError,
            "score",
        ),
        (
            torch.ones(2, dtype=DTYPE),
            torch.tensor([0.5, -0.1], dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            0.01,
            ValueError,
            "max_sell",
        ),
        (
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            torch.tensor([0.5, math.inf], dtype=DTYPE),
            0.01,
            ValueError,
            "max_buy",
        ),
        (
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            -0.01,
            ValueError,
            "spread",
        ),
        (
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            math.inf,
            ValueError,
            "spread",
        ),
        (
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            torch.ones(2, dtype=DTYPE),
            True,
            TypeError,
            "spread",
        ),
    ],
)
def test_solve_func_2_lp_rejects_invalid_inputs(
    score: object,
    max_sell: torch.Tensor,
    max_buy: torch.Tensor,
    spread: object,
    error: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error, match=match):
        solve_func_2_lp(score, max_sell, max_buy, spread)


def test_solve_func_2_lp_matches_scipy_linprog_objective_and_constraints() -> None:
    scipy_optimize = pytest.importorskip("scipy.optimize")
    generator = np.random.default_rng(20260812)
    assets = 9
    score_array = generator.normal(loc=0.0, scale=0.07, size=assets)
    max_sell_array = generator.uniform(0.02, 0.35, size=assets)
    max_buy_array = generator.uniform(0.02, 0.35, size=assets)
    spread = 0.017
    score = torch.tensor(score_array, dtype=DTYPE)
    max_sell = torch.tensor(max_sell_array, dtype=DTYPE)
    max_buy = torch.tensor(max_buy_array, dtype=DTYPE)

    result = solve_func_2_lp(score, max_sell, max_buy, spread)

    profit = np.concatenate((score_array - 0.5 * spread, -score_array - 0.5 * spread))
    scipy_result = scipy_optimize.linprog(
        c=-profit,
        A_eq=np.concatenate((np.ones(assets), -np.ones(assets))).reshape(1, -1),
        b_eq=np.zeros(1),
        bounds=[(0.0, value) for value in max_buy_array]
        + [(0.0, value) for value in max_sell_array],
        method="highs",
    )

    assert scipy_result.success, scipy_result.message
    assert result.objective.item() == pytest.approx(-scipy_result.fun, abs=1e-11)
    assert result.budget_residual.item() <= 1e-12
    assert result.box_residual.item() <= 1e-12
    assert abs(result.dw.sum().item()) <= 1e-12
    assert bool(torch.all(result.dw >= -max_sell - 1e-12))
    assert bool(torch.all(result.dw <= max_buy + 1e-12))
    _assert_result_evidence(result, score, max_sell, max_buy, spread)


@pytest.mark.parametrize("dtype,large", [(torch.float32, 1e8), (torch.float64, 1e16)])
def test_solve_func_2_lp_does_not_reconstruct_rounded_capacity(
    dtype: torch.dtype,
    large: float,
) -> None:
    result = solve_func_2_lp(
        torch.tensor([2.0, 1.0, -2.0, -1.0], dtype=dtype),
        torch.tensor([0.0, 0.0, large, 8.0], dtype=dtype),
        torch.tensor([large, 7.0, 0.0, 0.0], dtype=dtype),
        0.0,
    )

    assert result.dw[1].item() == pytest.approx(7.0)
    assert result.box_residual.item() == 0.0


@pytest.mark.parametrize("dtype,capacity", [(torch.float32, 2e38), (torch.float64, 1e308)])
def test_solve_func_2_lp_rejects_nonfinite_output_evidence(
    dtype: torch.dtype,
    capacity: float,
) -> None:
    with pytest.raises(FloatingPointError, match="traded_volume is non-finite"):
        solve_func_2_lp(
            torch.tensor([1.0, -1.0], dtype=dtype),
            torch.tensor([0.0, capacity], dtype=dtype),
            torch.tensor([capacity, 0.0], dtype=dtype),
            0.0,
        )


@pytest.mark.parametrize(
    "cuda_device",
    [torch.device(f"cuda:{index}") for index in range(min(2, torch.cuda.device_count()))],
)
def test_solve_func_2_lp_cuda_matches_cpu_objective_and_owns_outputs(
    cuda_device: torch.device,
) -> None:
    score_cpu = torch.tensor([0.09, 0.04, 0.01, -0.03, -0.08], dtype=DTYPE)
    max_sell_cpu = torch.tensor([0.0, 0.1, 0.15, 0.3, 0.4], dtype=DTYPE)
    max_buy_cpu = torch.tensor([0.35, 0.2, 0.1, 0.0, 0.0], dtype=DTYPE)
    spread = 0.013
    cpu_result = solve_func_2_lp(score_cpu, max_sell_cpu, max_buy_cpu, spread)
    score_cuda = score_cpu.to(cuda_device)
    max_sell_cuda = max_sell_cpu.to(cuda_device)
    max_buy_cuda = max_buy_cpu.to(cuda_device)

    cuda_result = solve_func_2_lp(score_cuda, max_sell_cuda, max_buy_cuda, spread)

    assert cuda_result.device == cuda_device
    for value in (
        cuda_result.dw,
        cuda_result.objective,
        cuda_result.traded_volume,
        cuda_result.budget_residual,
        cuda_result.box_residual,
    ):
        assert value.device == cuda_device
    torch.testing.assert_close(cuda_result.objective.cpu(), cpu_result.objective, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        cuda_result.traded_volume.cpu(),
        cpu_result.traded_volume,
        atol=1e-12,
        rtol=0.0,
    )
    assert cuda_result.budget_residual.item() <= 1e-12
    assert cuda_result.box_residual.item() <= 1e-12
    _assert_result_evidence(
        cuda_result,
        score_cuda,
        max_sell_cuda,
        max_buy_cuda,
        spread,
    )
