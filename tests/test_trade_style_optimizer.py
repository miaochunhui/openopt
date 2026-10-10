from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import openpo.trade_optimizer as trade_optimizer
from openpo.trade_optimizer import solve_func_2_lp, solve_trade_style_qp


DTYPE = torch.float64
DW_ATOL = 2e-10
OBJECTIVE_ATOL = 2e-11
NEUTRALITY_ATOL = 1e-11
BOX_ATOL = 1e-12


def _problem(device: torch.device | str = "cpu") -> dict[str, torch.Tensor | float]:
    return {
        "score": torch.tensor(
            [0.11, 0.07, 0.02, -0.03, -0.08, -0.12],
            dtype=DTYPE,
            device=device,
        ),
        "max_sell": torch.tensor(
            [0.01, 0.03, 0.08, 0.14, 0.18, 0.20],
            dtype=DTYPE,
            device=device,
        ),
        "max_buy": torch.tensor(
            [0.16, 0.13, 0.09, 0.05, 0.02, 0.01],
            dtype=DTYPE,
            device=device,
        ),
        "spread": 0.014,
        "weights": torch.tensor(
            [0.18, 0.17, 0.16, 0.17, 0.16, 0.16],
            dtype=DTYPE,
            device=device,
        ),
        "market_weights": torch.tensor(
            [0.16, 0.17, 0.17, 0.16, 0.17, 0.17],
            dtype=DTYPE,
            device=device,
        ),
        "factor_exposure": torch.tensor(
            [
                [1.20, -0.30],
                [0.80, 0.50],
                [0.25, 1.10],
                [-0.20, 0.90],
                [-0.75, -0.10],
                [-1.10, -0.80],
            ],
            dtype=DTYPE,
            device=device,
        ),
    }


def _osqp_reference(
    *,
    score: np.ndarray,
    max_sell: np.ndarray,
    max_buy: np.ndarray,
    spread: float,
    weights: np.ndarray,
    market_weights: np.ndarray,
    factor_exposure: np.ndarray,
    style_ranges: np.ndarray,
    style_penalty: float,
) -> tuple[np.ndarray, float]:
    osqp = pytest.importorskip("osqp")
    scipy_sparse = pytest.importorskip("scipy.sparse")
    assets = score.size
    styles = factor_exposure.shape[1]
    variables = assets * 2 + styles * 2
    dw_slice = slice(0, assets)
    abs_slice = slice(assets, assets * 2)
    lower_slack_slice = slice(assets * 2, assets * 2 + styles)
    upper_slack_slice = slice(assets * 2 + styles, variables)

    quadratic = np.zeros(variables)
    quadratic[lower_slack_slice] = 2.0 * style_penalty
    quadratic[upper_slack_slice] = 2.0 * style_penalty
    linear = np.zeros(variables)
    linear[dw_slice] = -score
    linear[abs_slice] = 0.5 * spread

    rows: list[np.ndarray] = []
    lower: list[float] = []
    upper: list[float] = []

    budget = np.zeros(variables)
    budget[dw_slice] = 1.0
    rows.append(budget)
    lower.append(0.0)
    upper.append(0.0)

    for asset in range(assets):
        box = np.zeros(variables)
        box[asset] = 1.0
        rows.append(box)
        lower.append(-max_sell[asset])
        upper.append(max_buy[asset])

        positive_abs = np.zeros(variables)
        positive_abs[asset] = 1.0
        positive_abs[assets + asset] = -1.0
        rows.append(positive_abs)
        lower.append(-np.inf)
        upper.append(0.0)

        negative_abs = np.zeros(variables)
        negative_abs[asset] = -1.0
        negative_abs[assets + asset] = -1.0
        rows.append(negative_abs)
        lower.append(-np.inf)
        upper.append(0.0)

        abs_nonnegative = np.zeros(variables)
        abs_nonnegative[assets + asset] = 1.0
        rows.append(abs_nonnegative)
        lower.append(0.0)
        upper.append(np.inf)

    base_exposure = factor_exposure.T @ (weights - market_weights)
    for style in range(styles):
        lower_hinge = np.zeros(variables)
        lower_hinge[dw_slice] = -factor_exposure[:, style]
        lower_hinge[assets * 2 + style] = -1.0
        rows.append(lower_hinge)
        lower.append(-np.inf)
        upper.append(base_exposure[style] - style_ranges[style, 0])

        upper_hinge = np.zeros(variables)
        upper_hinge[dw_slice] = factor_exposure[:, style]
        upper_hinge[assets * 2 + styles + style] = -1.0
        rows.append(upper_hinge)
        lower.append(-np.inf)
        upper.append(style_ranges[style, 1] - base_exposure[style])

        for slack_index in (
            assets * 2 + style,
            assets * 2 + styles + style,
        ):
            slack_nonnegative = np.zeros(variables)
            slack_nonnegative[slack_index] = 1.0
            rows.append(slack_nonnegative)
            lower.append(0.0)
            upper.append(np.inf)

    solver = osqp.OSQP()
    solver.setup(
        P=scipy_sparse.diags(quadratic, format="csc"),
        q=linear,
        A=scipy_sparse.csc_matrix(np.vstack(rows)),
        l=np.asarray(lower),
        u=np.asarray(upper),
        eps_abs=1e-11,
        eps_rel=1e-11,
        max_iter=100_000,
        polishing=True,
        verbose=False,
    )
    result = solver.solve()
    assert result.info.status == "solved", result.info.status
    dw = np.asarray(result.x[dw_slice])
    exposure = base_exposure + factor_exposure.T @ dw
    lower_violation = np.maximum(style_ranges[:, 0] - exposure, 0.0)
    upper_violation = np.maximum(exposure - style_ranges[:, 1], 0.0)
    objective = (
        score @ dw
        - 0.5 * spread * np.abs(dw).sum()
        - style_penalty
        * (np.square(lower_violation).sum() + np.square(upper_violation).sum())
    )
    return dw, float(objective)


def _solve(
    problem: dict[str, torch.Tensor | float],
    style_ranges: torch.Tensor,
    **kwargs: object,
):
    return solve_trade_style_qp(
        problem["score"],
        problem["max_sell"],
        problem["max_buy"],
        problem["spread"],
        problem["weights"],
        problem["market_weights"],
        problem["factor_exposure"],
        style_ranges,
        **kwargs,
    )


def _fused_prox_case(
    case: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if case == "random":
        generator = torch.Generator(device="cpu").manual_seed(20260813)
        value = 0.3 * torch.randn(257, dtype=DTYPE, generator=generator)
        lower = -(0.01 + 0.29 * torch.rand(257, dtype=DTYPE, generator=generator))
        upper = 0.01 + 0.24 * torch.rand(257, dtype=DTYPE, generator=generator)
        threshold = torch.tensor(0.027, dtype=DTYPE)
    elif case == "boundary":
        value = torch.tensor(
            [10.0, -10.0, 3.0, -3.0, 0.7, -0.7, 0.02, -0.02],
            dtype=DTYPE,
        )
        lower = torch.tensor(
            [-0.05, -0.40, -0.03, -0.22, -0.09, -0.17, -0.01, -0.08],
            dtype=DTYPE,
        )
        upper = torch.tensor(
            [0.40, 0.04, 0.20, 0.02, 0.13, 0.06, 0.07, 0.03],
            dtype=DTYPE,
        )
        threshold = torch.tensor(0.015, dtype=DTYPE)
    elif case == "zeros":
        value = torch.zeros(64, dtype=DTYPE)
        lower = torch.zeros(64, dtype=DTYPE)
        upper = torch.zeros(64, dtype=DTYPE)
        threshold = torch.tensor(0.0, dtype=DTYPE)
    elif case == "asymmetric_box":
        value = torch.tensor(
            [0.91, -0.37, 0.14, -0.82, 0.33, -0.06, 0.48, -0.25, 0.03],
            dtype=DTYPE,
        )
        lower = torch.tensor(
            [-0.01, -0.31, -0.17, -0.46, -0.04, -0.22, -0.08, -0.19, -0.11],
            dtype=DTYPE,
        )
        upper = torch.tensor(
            [0.29, 0.02, 0.05, 0.11, 0.27, 0.03, 0.16, 0.07, 0.01],
            dtype=DTYPE,
        )
        threshold = torch.tensor(0.041, dtype=DTYPE)
    else:
        raise AssertionError(f"unknown fused prox case: {case}")
    return value, lower, upper, threshold


def _prox_objective(
    dw: torch.Tensor, value: torch.Tensor, threshold: torch.Tensor
) -> torch.Tensor:
    return 0.5 * (dw - value).square().sum() + threshold * dw.abs().sum()


def _assert_neutral_box(
    dw: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor
) -> None:
    neutrality = dw.sum().abs().item()
    box_residual = torch.maximum(
        torch.clamp(lower - dw, min=0).amax(),
        torch.clamp(dw - upper, min=0).amax(),
    ).item()
    assert neutrality <= NEUTRALITY_ATOL
    assert box_residual <= BOX_ATOL


def test_trade_style_qp_without_active_style_penalty_matches_lp() -> None:
    problem = _problem()
    style_ranges = torch.tensor([[-10.0, 10.0], [-10.0, 10.0]], dtype=DTYPE)

    result = _solve(
        problem,
        style_ranges,
        max_iterations=4_000,
        tolerance=1e-12,
    )
    lp_result = solve_func_2_lp(
        problem["score"],
        problem["max_sell"],
        problem["max_buy"],
        problem["spread"],
    )

    assert result.style_utility.item() == pytest.approx(0.0, abs=1e-14)
    assert result.objective.item() == pytest.approx(lp_result.objective.item(), abs=2e-8)
    torch.testing.assert_close(result.dw, lp_result.dw, atol=2e-7, rtol=0.0)
    assert result.budget_residual.item() <= 1e-12
    assert result.box_residual.item() <= 1e-12


def test_trade_style_qp_matches_independent_osqp_reference() -> None:
    problem = _problem()
    style_ranges = torch.tensor([[-0.025, 0.020], [-0.018, 0.022]], dtype=DTYPE)
    style_penalty = 5.0

    result = _solve(
        problem,
        style_ranges,
        style_penalty=style_penalty,
        max_iterations=6_000,
        tolerance=1e-12,
    )
    reference_dw, reference_objective = _osqp_reference(
        score=problem["score"].numpy(),
        max_sell=problem["max_sell"].numpy(),
        max_buy=problem["max_buy"].numpy(),
        spread=float(problem["spread"]),
        weights=problem["weights"].numpy(),
        market_weights=problem["market_weights"].numpy(),
        factor_exposure=problem["factor_exposure"].numpy(),
        style_ranges=style_ranges.numpy(),
        style_penalty=style_penalty,
    )

    torch.testing.assert_close(
        result.dw,
        torch.from_numpy(reference_dw),
        atol=3e-6,
        rtol=0.0,
    )
    assert result.objective.item() == pytest.approx(reference_objective, abs=2e-8)
    assert result.budget_residual.item() <= 1e-12
    assert result.box_residual.item() <= 1e-12
    expected_exposure = problem["factor_exposure"].T @ (
        problem["weights"] - problem["market_weights"] + result.dw
    )
    expected_lower = torch.clamp(style_ranges[:, 0] - expected_exposure, min=0)
    expected_upper = torch.clamp(expected_exposure - style_ranges[:, 1], min=0)
    torch.testing.assert_close(result.exposure, expected_exposure, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(
        result.lower_style_violation, expected_lower, atol=1e-12, rtol=0.0
    )
    torch.testing.assert_close(
        result.upper_style_violation, expected_upper, atol=1e-12, rtol=0.0
    )
    assert result.style_utility.item() == pytest.approx(
        style_penalty
        * (expected_lower.square().sum() + expected_upper.square().sum()).item(),
        abs=1e-12,
    )


@pytest.mark.parametrize(
    "cuda_device",
    [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())],
)
def test_trade_style_qp_fp64_cuda_matches_cpu(cuda_device: torch.device) -> None:
    cpu_problem = _problem()
    cpu_ranges = torch.tensor([[-0.025, 0.020], [-0.018, 0.022]], dtype=DTYPE)
    cpu_result = _solve(
        cpu_problem,
        cpu_ranges,
        max_iterations=2_000,
        tolerance=1e-12,
    )
    cuda_problem = _problem(cuda_device)
    cuda_result = _solve(
        cuda_problem,
        cpu_ranges.to(cuda_device),
        max_iterations=2_000,
        tolerance=1e-12,
    )

    torch.testing.assert_close(
        cuda_result.dw.cpu(), cpu_result.dw, atol=2e-10, rtol=0.0
    )
    torch.testing.assert_close(
        cuda_result.objective.cpu(), cpu_result.objective, atol=2e-11, rtol=0.0
    )
    torch.testing.assert_close(
        cuda_result.exposure.cpu(), cpu_result.exposure, atol=2e-11, rtol=0.0
    )
    assert cuda_result.device == cuda_device
    assert cuda_result.budget_residual.item() <= 1e-11
    assert cuda_result.box_residual.item() <= 1e-12


@pytest.mark.parametrize(
    "cuda_device",
    [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())],
)
@pytest.mark.parametrize("case", ["random", "boundary", "zeros", "asymmetric_box"])
def test_cuda_fused_prox_matches_torch_reference(
    cuda_device: torch.device, case: str
) -> None:
    triton_projection = pytest.importorskip("openpo.triton_trade_projection")
    value, lower, upper, threshold = _fused_prox_case(case)
    reference_dw = trade_optimizer._prox_l1_box_neutral_torch(
        value, lower, upper, threshold, 80
    )
    cuda_dw = triton_projection.triton_prox_l1_box_neutral(
        value.to(cuda_device),
        lower.to(cuda_device),
        upper.to(cuda_device),
        threshold.to(cuda_device),
        80,
    ).cpu()

    torch.testing.assert_close(cuda_dw, reference_dw, atol=DW_ATOL, rtol=0.0)
    torch.testing.assert_close(
        _prox_objective(cuda_dw, value, threshold),
        _prox_objective(reference_dw, value, threshold),
        atol=OBJECTIVE_ATOL,
        rtol=0.0,
    )
    _assert_neutral_box(cuda_dw, lower, upper)


@pytest.mark.parametrize(
    "cuda_device",
    [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())],
)
def test_cuda_fused_prox_fixed_seed_is_reproducible(cuda_device: torch.device) -> None:
    triton_projection = pytest.importorskip("openpo.triton_trade_projection")
    first = _fused_prox_case("random")
    second = _fused_prox_case("random")
    for first_tensor, second_tensor in zip(first, second):
        torch.testing.assert_close(first_tensor, second_tensor, atol=0.0, rtol=0.0)
    first_dw = triton_projection.triton_prox_l1_box_neutral(
        *(tensor.to(cuda_device) for tensor in first), 80
    )
    second_dw = triton_projection.triton_prox_l1_box_neutral(
        *(tensor.to(cuda_device) for tensor in second), 80
    )
    torch.testing.assert_close(first_dw, second_dw, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    "cuda_device",
    [torch.device(f"cuda:{index}") for index in range(torch.cuda.device_count())],
)
def test_cuda_fused_prox_unavailable_falls_back(
    cuda_device: torch.device, monkeypatch: pytest.MonkeyPatch
) -> None:
    triton_projection = pytest.importorskip("openpo.triton_trade_projection")
    value, lower, upper, threshold = (
        tensor.to(cuda_device) for tensor in _fused_prox_case("asymmetric_box")
    )
    reference_dw = trade_optimizer._prox_l1_box_neutral_torch(
        value, lower, upper, threshold, 80
    )

    def unavailable(*args: object, **kwargs: object) -> torch.Tensor:
        raise triton_projection.TritonTradeProjectionUnavailable(
            "unsupported CUDA device"
        )

    monkeypatch.setattr(triton_projection, "triton_prox_l1_box_neutral", unavailable)
    monkeypatch.setattr(trade_optimizer, "_TRITON_TRADE_PROJECTION_DISABLED", set())
    fallback_dw = trade_optimizer._prox_l1_box_neutral(
        value, lower, upper, threshold, 80
    )

    torch.testing.assert_close(fallback_dw, reference_dw, atol=DW_ATOL, rtol=0.0)
    _assert_neutral_box(fallback_dw, lower, upper)


@pytest.mark.parametrize(
    ("field", "replacement", "error", "match"),
    [
        (
            "weights",
            torch.ones(5, dtype=DTYPE),
            ValueError,
            "weights must match score shape",
        ),
        (
            "market_weights",
            torch.tensor([math.nan, 0, 0, 0, 0, 0], dtype=DTYPE),
            ValueError,
            "market_weights must contain only finite values",
        ),
        (
            "factor_exposure",
            torch.ones((5, 2), dtype=DTYPE),
            ValueError,
            "factor_exposure must have shape",
        ),
        (
            "factor_exposure",
            torch.tensor(
                [[math.inf, 0.0]] * 6,
                dtype=DTYPE,
            ),
            ValueError,
            "factor_exposure must contain only finite values",
        ),
    ],
)
def test_trade_style_qp_rejects_invalid_portfolio_inputs(
    field: str,
    replacement: torch.Tensor,
    error: type[Exception],
    match: str,
) -> None:
    problem = _problem()
    problem[field] = replacement
    style_ranges = torch.tensor([[-0.1, 0.1], [-0.1, 0.1]], dtype=DTYPE)

    with pytest.raises(error, match=match):
        _solve(problem, style_ranges)


@pytest.mark.parametrize(
    ("style_ranges", "match"),
    [
        (torch.ones((3, 2), dtype=DTYPE), "style_ranges must have shape"),
        (
            torch.tensor([[0.2, -0.2], [-0.1, 0.1]], dtype=DTYPE),
            "lower <= upper",
        ),
        (
            torch.tensor([[math.nan, 0.2], [-0.1, 0.1]], dtype=DTYPE),
            "style_ranges must be finite",
        ),
    ],
)
def test_trade_style_qp_rejects_invalid_style_ranges(
    style_ranges: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _solve(_problem(), style_ranges)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({"style_penalty": math.inf}, ValueError, "style_penalty"),
        ({"style_penalty": True}, TypeError, "style_penalty"),
        ({"max_iterations": 0}, ValueError, "max_iterations"),
        ({"bisection_steps": 0}, ValueError, "bisection_steps"),
        ({"tolerance": math.nan}, ValueError, "tolerance"),
    ],
)
def test_trade_style_qp_rejects_invalid_solver_parameters(
    kwargs: dict[str, object],
    error: type[Exception],
    match: str,
) -> None:
    style_ranges = torch.tensor([[-0.1, 0.1], [-0.1, 0.1]], dtype=DTYPE)
    with pytest.raises(error, match=match):
        _solve(_problem(), style_ranges, **kwargs)


def test_trade_style_qp_fixed_iteration_result_is_reproducible() -> None:
    problem = _problem()
    style_ranges = torch.tensor([[-0.025, 0.020], [-0.018, 0.022]], dtype=DTYPE)
    kwargs = {
        "max_iterations": 37,
        "tolerance": 1e-30,
        "bisection_steps": 60,
    }

    first = _solve(problem, style_ranges, **kwargs)
    second = _solve(problem, style_ranges, **kwargs)

    assert first.status == second.status == "fixed_iterations"
    assert first.iterations == second.iterations == 37
    torch.testing.assert_close(first.dw, second.dw, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.objective, second.objective, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.exposure, second.exposure, atol=0.0, rtol=0.0)
