from __future__ import annotations

import math

import numpy as np
import osqp
import pytest
import scipy.sparse as sparse
import torch

from openpo import (
    AdmmConfig,
    AdmmSolveResult,
    FactorRiskModel,
    LinearConstraints,
    PortfolioProblem,
    solve_admm,
)


DTYPE = torch.float64


def make_problem(
    device: str = "cpu",
    *,
    batch: int = 1,
    trade_cost: float = 0.0,
    requires_grad: bool = False,
) -> PortfolioProblem:
    assets = 12
    base_alpha = torch.linspace(-0.015, 0.025, assets, dtype=DTYPE, device=device)
    if batch == 1:
        alpha = base_alpha.clone()
    else:
        perturbation = torch.linspace(-0.004, 0.006, assets, dtype=DTYPE, device=device)
        alpha = torch.stack(
            [base_alpha + 0.001 * row * torch.roll(perturbation, row) for row in range(batch)]
        )
    if requires_grad:
        alpha = alpha.clone().requires_grad_()

    grid = torch.linspace(0.1, 1.2, assets, dtype=DTYPE, device=device)
    risk = FactorRiskModel(
        exposures=torch.stack(
            (
                torch.linspace(-0.3, 0.4, assets, dtype=DTYPE, device=device),
                torch.cos(grid),
            ),
            dim=1,
        ),
        factor_variance=torch.tensor([0.4, 0.7], dtype=DTYPE, device=device),
        specific_variance=torch.linspace(0.6, 1.4, assets, dtype=DTYPE, device=device),
    )
    lower = torch.full((assets,), 0.01, dtype=DTYPE, device=device)
    upper = torch.full((assets,), 0.20, dtype=DTYPE, device=device)

    current = None
    if trade_cost:
        base_current = torch.linspace(0.03, 0.13, assets, dtype=DTYPE, device=device)
        base_current = base_current / base_current.sum()
        if batch == 1:
            current = base_current
        else:
            current = torch.stack([torch.roll(base_current, row) for row in range(batch)])

    return PortfolioProblem(
        alpha=alpha,
        risk=risk,
        lower=lower,
        upper=upper,
        risk_aversion=0.8,
        current=current,
        trade_cost=trade_cost,
    )


def make_constraints(device: str = "cpu") -> LinearConstraints:
    assets = 12
    equality = torch.zeros((1, assets), dtype=DTYPE, device=device)
    equality[0, :6] = 1.0
    equality[0, 6:] = -1.0

    interval = torch.zeros((2, assets), dtype=DTYPE, device=device)
    interval[0, :4] = 1.0
    interval[1, 4:8] = 1.0
    return LinearConstraints(
        A_eq=equality,
        b_eq=torch.tensor([0.0], dtype=DTYPE, device=device),
        A_ineq=interval,
        lower=torch.tensor([0.25, 0.20], dtype=DTYPE, device=device),
        upper=torch.tensor([0.42, 0.36], dtype=DTYPE, device=device),
    )


def make_ill_scaled_constraints(device: str = "cpu") -> LinearConstraints:
    constraints = make_constraints(device)
    assert constraints.A_eq is not None
    assert constraints.b_eq is not None
    assert constraints.A_ineq is not None
    assert constraints.lower is not None
    assert constraints.upper is not None

    equality_scale = torch.tensor([1e2], dtype=DTYPE, device=device)
    inequality_scale = torch.tensor([1e-2, 1e2], dtype=DTYPE, device=device)
    return LinearConstraints(
        A_eq=constraints.A_eq * equality_scale.unsqueeze(1),
        b_eq=constraints.b_eq * equality_scale,
        A_ineq=constraints.A_ineq * inequality_scale.unsqueeze(1),
        lower=constraints.lower * inequality_scale,
        upper=constraints.upper * inequality_scale,
    )


def make_zero_row_constraints(device: str = "cpu") -> LinearConstraints:
    constraints = make_constraints(device)
    assert constraints.A_eq is not None
    assert constraints.b_eq is not None
    return LinearConstraints(
        A_eq=torch.zeros_like(constraints.A_eq),
        b_eq=torch.zeros_like(constraints.b_eq),
        A_ineq=constraints.A_ineq,
        lower=constraints.lower,
        upper=constraints.upper,
    )


def solve_osqp_reference(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
) -> np.ndarray:
    cpu_problem = problem.to("cpu")
    cpu_constraints = constraints.to("cpu")
    assets = cpu_problem.risk.assets

    exposures = cpu_problem.risk.exposures.numpy()
    factor_variance = cpu_problem.risk.factor_variance.numpy()
    specific_variance = cpu_problem.risk.specific_variance.numpy()
    covariance = exposures @ np.diag(factor_variance) @ exposures.T + np.diag(
        specific_variance
    )
    quadratic = cpu_problem.risk_aversion * covariance

    rows = [np.ones((1, assets)), np.eye(assets)]
    lower = [np.array([cpu_problem.budget]), cpu_problem.lower.numpy()]
    upper = [np.array([cpu_problem.budget]), cpu_problem.upper.numpy()]
    rows.append(cpu_constraints.A_eq.numpy())
    lower.append(cpu_constraints.b_eq.numpy())
    upper.append(cpu_constraints.b_eq.numpy())
    rows.append(cpu_constraints.A_ineq.numpy())
    lower.append(cpu_constraints.lower.numpy())
    upper.append(cpu_constraints.upper.numpy())
    constraint_matrix = sparse.csc_matrix(np.vstack(rows))
    lower_vector = np.concatenate(lower)
    upper_vector = np.concatenate(upper)

    alpha = cpu_problem.batched_alpha.numpy()
    current = None
    if cpu_problem.current is not None:
        current_tensor = (
            cpu_problem.current.unsqueeze(0)
            if cpu_problem.current.ndim == 1
            else cpu_problem.current
        )
        current = current_tensor.numpy()

    solutions = []
    for row, alpha_row in enumerate(alpha):
        row_quadratic = quadratic.copy()
        linear = -alpha_row.copy()
        if current is not None and cpu_problem.trade_cost:
            current_row = current[row if current.shape[0] > 1 else 0]
            row_quadratic = row_quadratic + cpu_problem.trade_cost * np.eye(assets)
            linear = linear - cpu_problem.trade_cost * current_row

        solver = osqp.OSQP()
        solver.setup(
            P=sparse.csc_matrix((row_quadratic + row_quadratic.T) / 2.0),
            q=linear,
            A=constraint_matrix,
            l=lower_vector,
            u=upper_vector,
            eps_abs=1e-10,
            eps_rel=1e-10,
            polishing=True,
            verbose=False,
        )
        result = solver.solve(raise_error=True)
        assert result.info.status == "solved"
        solutions.append(result.x)

    return np.stack(solutions)


def objective_values(problem: PortfolioProblem, weights: torch.Tensor) -> torch.Tensor:
    alpha = problem.batched_alpha
    risk = 0.5 * problem.risk_aversion * (
        weights * problem.risk.matvec(weights)
    ).sum(dim=1)
    utility = (alpha * weights).sum(dim=1)
    if problem.current is None or problem.trade_cost == 0:
        trade = torch.zeros_like(risk)
    else:
        current = (
            problem.current.unsqueeze(0)
            if problem.current.ndim == 1
            else problem.current
        )
        trade = 0.5 * problem.trade_cost * ((weights - current) ** 2).sum(dim=1)
    return risk - utility + trade


def assert_final_block_rho_matches_scales(
    result: AdmmSolveResult,
    scales: tuple[float, float, float],
    *,
    atol: float = 1e-12,
) -> None:
    block_rho = block_rho_tensor(result)
    expected = result.final_rho * torch.tensor(scales, dtype=block_rho.dtype, device=block_rho.device)
    assert block_rho.shape == expected.shape == (3,)
    assert torch.allclose(block_rho, expected, atol=atol, rtol=0.0)


def block_rho_tensor(result: AdmmSolveResult) -> torch.Tensor:
    return torch.tensor(
        [
            result.final_block_rho["weights"],
            result.final_block_rho["equality"],
            result.final_block_rho["inequality"],
        ],
        dtype=DTYPE,
    )


def assert_successful_result(
    result: AdmmSolveResult,
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    *,
    max_violation: float = 2e-7,
) -> None:
    assert isinstance(result, AdmmSolveResult)
    assert result.converged
    assert result.status == "converged"
    assert result.iterations <= 5000
    assert result.device == str(result.weights.device)
    assert result.weights.shape == problem.batched_alpha.shape
    assert torch.isfinite(result.weights).all()
    assert torch.isfinite(result.objective).all()
    assert torch.isfinite(
        torch.as_tensor(list(result.final_block_rho.values()), dtype=DTYPE)
    ).all()
    assert torch.allclose(
        result.weights.sum(dim=1),
        torch.full(
            (result.weights.shape[0],),
            problem.budget,
            dtype=result.weights.dtype,
            device=result.weights.device,
        ),
        atol=max_violation,
        rtol=0.0,
    )
    assert result.max_constraint_violation <= max_violation
    assert float(constraints.residual(result.weights).max_violation) <= max_violation
    for residual in (
        result.primal_residual,
        result.dual_residual,
        result.stationarity_residual,
        result.max_constraint_violation,
        result.elapsed_seconds,
    ):
        assert math.isfinite(residual)
        assert residual >= 0.0


def test_admm_matches_osqp_for_equality_and_bilateral_interval() -> None:
    problem = make_problem()
    constraints = make_constraints()
    config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=10,
        rho=1.0,
        bisection_steps=48,
    )

    result = solve_admm(problem, constraints, config)
    reference = solve_osqp_reference(problem, constraints)
    reference_weights = torch.tensor(reference, dtype=DTYPE)
    reference_objective = objective_values(problem, reference_weights)

    assert_successful_result(result, problem, constraints)
    np.testing.assert_allclose(result.weights.numpy(), reference, atol=5e-6, rtol=0.0)
    assert torch.max(torch.abs(result.objective - reference_objective)) <= 5e-8


def test_admm_exposes_row_scaling_adaptive_rho_result_api() -> None:
    config_fields = set(AdmmConfig.__dataclass_fields__)
    result_fields = set(AdmmSolveResult.__dataclass_fields__)

    assert {
        "row_scaling",
        "adaptive_rho",
        "adaptive_rho_interval",
        "adaptive_rho_mu",
        "adaptive_rho_tau",
        "adaptive_rho_increase",
        "adaptive_rho_decrease",
        "rho_min",
        "rho_max",
        "rho_weight_scale",
        "rho_equality_scale",
        "rho_inequality_scale",
    }.issubset(config_fields)
    assert {"final_rho", "final_block_rho", "rho_update_count"}.issubset(result_fields)


@pytest.mark.parametrize("row_scaling", [False, True])
def test_admm_row_scaling_toggle_matches_osqp(row_scaling: bool) -> None:
    problem = make_problem()
    constraints = make_constraints()
    config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=10,
        rho=1.0,
        bisection_steps=48,
        row_scaling=row_scaling,
        adaptive_rho=False,
    )

    result = solve_admm(problem, constraints, config)
    reference = solve_osqp_reference(problem, constraints)
    reference_weights = torch.tensor(reference, dtype=DTYPE)

    assert_successful_result(result, problem, constraints)
    np.testing.assert_allclose(result.weights.numpy(), reference, atol=5e-6, rtol=0.0)
    assert torch.max(
        torch.abs(result.objective - objective_values(problem, reference_weights))
    ) <= 5e-8


def test_admm_row_scaling_recovers_ill_scaled_constraints() -> None:
    problem = make_problem()
    constraints = make_ill_scaled_constraints()
    reference = solve_osqp_reference(problem, constraints)
    reference_weights = torch.tensor(reference, dtype=DTYPE)

    unscaled = solve_admm(
        problem,
        constraints,
        AdmmConfig(
            iterations=800,
            tolerance=1e-8,
            check_interval=10,
            rho=1.0,
            bisection_steps=48,
            row_scaling=False,
            adaptive_rho=False,
        ),
    )
    assert not unscaled.converged
    assert unscaled.max_constraint_violation > 1e-5

    scaled = solve_admm(
        problem,
        constraints,
        AdmmConfig(
            iterations=5000,
            tolerance=1e-8,
            check_interval=10,
            rho=1e-4,
            bisection_steps=48,
            row_scaling=True,
            adaptive_rho=True,
            adaptive_rho_interval=5,
            adaptive_rho_mu=2.0,
            adaptive_rho_tau=2.0,
            rho_min=1e-6,
            rho_max=1.0,
        ),
    )

    assert_successful_result(scaled, problem, constraints, max_violation=2e-7)
    assert scaled.rho_update_count > 0
    np.testing.assert_allclose(scaled.weights.numpy(), reference, atol=5e-6, rtol=0.0)
    assert torch.max(
        torch.abs(scaled.objective - objective_values(problem, reference_weights))
    ) <= 5e-8


@pytest.mark.parametrize("field", ["rho_weight_scale", "rho_equality_scale", "rho_inequality_scale"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_admm_config_rejects_invalid_block_rho_scales(field: str, value: float) -> None:
    with pytest.raises(ValueError, match="(?i)(finite.*positive|positive.*finite|positive)"):
        AdmmConfig(**{field: value})


def test_admm_row_scaling_rejects_zero_constraint_rows() -> None:
    problem = make_problem()
    constraints = make_zero_row_constraints()
    config = AdmmConfig(
        iterations=100,
        tolerance=1e-8,
        check_interval=10,
        rho=1.0,
        bisection_steps=32,
        row_scaling=True,
    )

    with pytest.raises(ValueError, match="(?i)(zero.*row|row.*zero)"):
        solve_admm(problem, constraints, config)


def test_admm_row_scaling_rejects_overflowed_row_norms() -> None:
    assets = 12
    problem = PortfolioProblem(
        alpha=torch.zeros(assets),
        risk=FactorRiskModel(
            exposures=torch.zeros(assets, 1),
            factor_variance=torch.ones(1),
            specific_variance=torch.ones(assets),
        ),
        lower=torch.zeros(assets),
        upper=torch.full((assets,), 0.2),
    )
    huge_row = torch.full((1, assets), torch.finfo(torch.float32).max / 4)
    constraints = LinearConstraints(A_eq=huge_row, b_eq=torch.zeros(1))

    with pytest.raises(ValueError, match="finite general linear constraint row norms"):
        solve_admm(problem, constraints, AdmmConfig(row_scaling=True))


@pytest.mark.parametrize(
    ("rho", "rho_min", "rho_max", "direction"),
    [
        (1e-4, 1e-4, 8e-4, "increase"),
        (1e4, 1250.0, 1e4, "decrease"),
    ],
)
def test_adaptive_rho_updates_direction_and_bounds(
    rho: float,
    rho_min: float,
    rho_max: float,
    direction: str,
) -> None:
    problem = make_problem()
    constraints = make_ill_scaled_constraints()
    config = AdmmConfig(
        iterations=400,
        tolerance=1e-10,
        check_interval=1,
        rho=rho,
        bisection_steps=48,
        row_scaling=True,
        adaptive_rho=True,
        adaptive_rho_interval=1,
        adaptive_rho_mu=2.0,
        adaptive_rho_tau=2.0,
        rho_min=rho_min,
        rho_max=rho_max,
    )

    result = solve_admm(problem, constraints, config)

    assert result.rho_update_count > 0
    assert rho_min <= result.final_rho <= rho_max
    if direction == "increase":
        assert result.final_rho > rho
    else:
        assert result.final_rho < rho


def test_adaptive_rho_matches_osqp_after_dual_rescaling() -> None:
    problem = make_problem(batch=3, trade_cost=0.25)
    constraints = make_ill_scaled_constraints()
    config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=5,
        rho=1e-4,
        bisection_steps=48,
        row_scaling=True,
        adaptive_rho=True,
        adaptive_rho_interval=5,
        adaptive_rho_mu=2.0,
        adaptive_rho_tau=2.0,
        rho_min=1e-4,
        rho_max=1.0,
    )

    result = solve_admm(problem, constraints, config)
    reference = solve_osqp_reference(problem, constraints)

    assert_successful_result(result, problem, constraints, max_violation=2e-5)
    assert result.rho_update_count > 0
    assert result.final_rho != pytest.approx(config.rho)
    assert_final_block_rho_matches_scales(result, (1.0, 1.0, 1.0))
    np.testing.assert_allclose(result.weights.numpy(), reference, atol=7e-6, rtol=0.0)


def test_admm_solves_batch_with_trade_cost_against_osqp() -> None:
    problem = make_problem(batch=3, trade_cost=0.25)
    constraints = make_constraints()
    config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=10,
        rho=1.0,
        bisection_steps=48,
    )

    result = solve_admm(problem, constraints, config)
    reference = solve_osqp_reference(problem, constraints)
    reference_weights = torch.tensor(reference, dtype=DTYPE)

    assert_successful_result(result, problem, constraints)
    assert result.weights.shape == (3, problem.risk.assets)
    np.testing.assert_allclose(result.weights.numpy(), reference, atol=7e-6, rtol=0.0)
    assert torch.max(
        torch.abs(result.objective - objective_values(problem, reference_weights))
    ) <= 7e-8


def test_admm_block_rho_default_scales_match_legacy_behavior() -> None:
    problem = make_problem(batch=3, trade_cost=0.25)
    constraints = make_ill_scaled_constraints()
    legacy_config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=5,
        rho=1e-4,
        bisection_steps=48,
        row_scaling=True,
        adaptive_rho=True,
        adaptive_rho_interval=5,
        adaptive_rho_mu=2.0,
        adaptive_rho_tau=2.0,
        rho_min=1e-6,
        rho_max=1.0,
    )
    explicit_unit_scales_config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=5,
        rho=1e-4,
        bisection_steps=48,
        row_scaling=True,
        adaptive_rho=True,
        adaptive_rho_interval=5,
        adaptive_rho_mu=2.0,
        adaptive_rho_tau=2.0,
        rho_min=1e-6,
        rho_max=1.0,
        rho_weight_scale=1.0,
        rho_equality_scale=1.0,
        rho_inequality_scale=1.0,
    )

    legacy = solve_admm(problem, constraints, legacy_config)
    explicit = solve_admm(problem, constraints, explicit_unit_scales_config)

    assert_successful_result(legacy, problem, constraints, max_violation=2e-5)
    assert_successful_result(explicit, problem, constraints, max_violation=2e-5)
    assert torch.equal(legacy.weights, explicit.weights)
    assert torch.equal(legacy.objective, explicit.objective)
    assert legacy.iterations == explicit.iterations
    assert legacy.rho_update_count == explicit.rho_update_count
    assert legacy.final_rho == pytest.approx(explicit.final_rho)
    assert torch.allclose(
        block_rho_tensor(legacy),
        block_rho_tensor(explicit),
        atol=0.0,
        rtol=0.0,
    )
    assert_final_block_rho_matches_scales(legacy, (1.0, 1.0, 1.0))
    assert_final_block_rho_matches_scales(explicit, (1.0, 1.0, 1.0))


def test_admm_block_rho_scales_change_report_and_converge() -> None:
    problem = make_problem(batch=3, trade_cost=0.25)
    constraints = make_ill_scaled_constraints()
    baseline_config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=5,
        rho=1e-4,
        bisection_steps=48,
        row_scaling=True,
        adaptive_rho=True,
        adaptive_rho_interval=1,
        adaptive_rho_mu=2.0,
        adaptive_rho_tau=2.0,
        rho_min=1e-6,
        rho_max=1.0,
    )
    scaled_config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=5,
        rho=1e-4,
        bisection_steps=48,
        row_scaling=True,
        adaptive_rho=True,
        adaptive_rho_interval=1,
        adaptive_rho_mu=2.0,
        adaptive_rho_tau=2.0,
        rho_min=1e-6,
        rho_max=1.0,
        rho_weight_scale=0.25,
        rho_equality_scale=4.0,
        rho_inequality_scale=2.0,
    )

    baseline = solve_admm(problem, constraints, baseline_config)
    scaled = solve_admm(problem, constraints, scaled_config)

    assert_successful_result(baseline, problem, constraints, max_violation=2e-5)
    assert_successful_result(scaled, problem, constraints, max_violation=2e-5)
    assert baseline.rho_update_count > 0
    assert scaled.rho_update_count > 0
    assert baseline.weights.shape == scaled.weights.shape
    assert torch.allclose(baseline.weights, scaled.weights, atol=7e-6, rtol=1e-5)
    assert torch.allclose(baseline.objective, scaled.objective, atol=7e-8, rtol=1e-7)
    assert not torch.allclose(
        block_rho_tensor(baseline),
        block_rho_tensor(scaled),
    )
    assert (
        baseline.final_rho != pytest.approx(scaled.final_rho)
        or baseline.iterations != scaled.iterations
        or baseline.rho_update_count != scaled.rho_update_count
    )
    assert_final_block_rho_matches_scales(baseline, (1.0, 1.0, 1.0))
    assert_final_block_rho_matches_scales(scaled, (0.25, 4.0, 2.0))


def test_admm_row_scaling_and_adaptive_rho_disabled_preserve_legacy_behavior() -> None:
    problem = make_problem(batch=3, trade_cost=0.25)
    constraints = make_constraints()
    legacy_config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=10,
        rho=1.0,
        bisection_steps=48,
    )
    disabled_config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=10,
        rho=1.0,
        bisection_steps=48,
        row_scaling=False,
        adaptive_rho=False,
    )

    legacy = solve_admm(problem, constraints, legacy_config)
    disabled = solve_admm(problem, constraints, disabled_config)

    assert_successful_result(legacy, problem, constraints)
    assert_successful_result(disabled, problem, constraints)
    assert disabled.rho_update_count == 0
    assert disabled.final_rho == pytest.approx(disabled_config.rho)
    assert torch.equal(legacy.weights, disabled.weights)
    assert torch.equal(legacy.objective, disabled.objective)
    assert legacy.iterations == disabled.iterations
    assert legacy.primal_residual == pytest.approx(disabled.primal_residual)
    assert legacy.dual_residual == pytest.approx(disabled.dual_residual)
    assert legacy.stationarity_residual == pytest.approx(disabled.stationarity_residual)


def test_admm_infeasible_constraints_return_nonconverged_violation() -> None:
    problem = make_problem()
    impossible = LinearConstraints(
        A_eq=torch.ones((1, problem.risk.assets), dtype=DTYPE),
        b_eq=torch.tensor([2.0], dtype=DTYPE),
    )

    result = solve_admm(
        problem,
        impossible,
        AdmmConfig(
            iterations=100,
            tolerance=1e-12,
            check_interval=1,
            rho=1.0,
            bisection_steps=32,
        ),
    )

    assert isinstance(result, AdmmSolveResult)
    assert not result.converged
    assert result.status == "max_iterations"
    assert result.iterations == 100
    assert result.max_constraint_violation >= 0.9
    assert math.isfinite(result.primal_residual)
    assert math.isfinite(result.dual_residual)
    assert math.isfinite(result.stationarity_residual)
    assert torch.isfinite(result.weights).all()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"iterations": 0}, "positive"),
        ({"check_interval": 0}, "positive"),
        ({"bisection_steps": 0}, "positive"),
        ({"tolerance": -1e-9}, "nonnegative"),
        ({"tolerance": float("inf")}, "finite and nonnegative"),
        ({"rho": 0.0}, "finite and positive"),
        ({"rho": float("nan")}, "finite and positive"),
        ({"cholesky_jitter": -1e-12}, "finite and nonnegative"),
        ({"cholesky_jitter": float("inf")}, "finite and nonnegative"),
        ({"row_scaling": 1}, "row_scaling.*bool|bool.*row_scaling"),
        ({"adaptive_rho": 1}, "adaptive_rho.*bool|bool.*adaptive_rho"),
        (
            {"adaptive_rho": True, "adaptive_rho_interval": 0},
            "adaptive_rho_interval.*positive|positive.*adaptive_rho_interval",
        ),
        (
            {"adaptive_rho": True, "adaptive_rho_mu": 1.0},
            "adaptive_rho_mu.*greater than 1|greater than 1.*adaptive_rho_mu",
        ),
        (
            {
                "adaptive_rho": True,
                "adaptive_rho_interval": 10,
                "adaptive_rho_mu": 1.0,
            },
            "adaptive_rho_mu.*greater than 1|greater than 1.*adaptive_rho_mu",
        ),
        (
            {"adaptive_rho": True, "adaptive_rho_tau": 1.0},
            "adaptive_rho_tau.*greater than 1|greater than 1.*adaptive_rho_tau",
        ),
        (
            {"adaptive_rho": True, "adaptive_rho_increase": 1.0},
            "adaptive_rho_increase.*greater than 1|greater than 1.*adaptive_rho_increase",
        ),
        (
            {"adaptive_rho": True, "adaptive_rho_decrease": 1.0},
            "adaptive_rho_decrease.*greater than 1|greater than 1.*adaptive_rho_decrease",
        ),
        (
            {"adaptive_rho": True, "rho_min": 0.0},
            "rho_min.*rho_max|rho_max.*rho_min|finite positive",
        ),
        (
            {"adaptive_rho": True, "rho_max": 0.0},
            "rho_min.*rho_max|rho_max.*rho_min|finite positive",
        ),
        (
            {"adaptive_rho": True, "rho_min": 2.0, "rho_max": 1.0},
            "rho_min.*rho_max|rho_max.*rho_min",
        ),
    ],
)
def test_admm_config_rejects_invalid_settings(
    kwargs: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        AdmmConfig(**kwargs)


def test_portfolio_problem_rejects_nonfinite_budget() -> None:
    problem = make_problem()

    with pytest.raises(ValueError, match="budget must be finite"):
        PortfolioProblem(
            alpha=problem.alpha,
            risk=problem.risk,
            lower=problem.lower,
            upper=problem.upper,
            budget=float("nan"),
            risk_aversion=problem.risk_aversion,
        )


def test_admm_result_residuals_are_finite_on_success() -> None:
    problem = make_problem(batch=2, trade_cost=0.1)
    constraints = make_constraints()

    result = solve_admm(
        problem,
        constraints,
        AdmmConfig(iterations=5000, tolerance=1e-8, check_interval=10, bisection_steps=48),
    )

    assert_successful_result(result, problem, constraints)
    assert result.primal_residual <= 1e-6
    assert result.dual_residual <= 1e-6
    assert result.stationarity_residual <= 1e-6


def test_admm_solver_does_not_retain_autograd_graph() -> None:
    problem = make_problem(requires_grad=True)
    constraints = make_constraints()
    assert problem.alpha.requires_grad

    result = solve_admm(
        problem,
        constraints,
        AdmmConfig(iterations=5000, tolerance=1e-8, check_interval=10, bisection_steps=48),
    )

    assert_successful_result(result, problem, constraints)
    assert not result.weights.requires_grad
    assert result.weights.grad_fn is None
    assert not result.objective.requires_grad
    assert result.objective.grad_fn is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_admm_cpu_cuda_parity() -> None:
    config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=10,
        rho=1.0,
        bisection_steps=48,
        rho_weight_scale=1.0,
        rho_equality_scale=1.0,
        rho_inequality_scale=1.0,
    )

    cpu_problem = make_problem(batch=3, trade_cost=0.25)
    cpu_constraints = make_constraints()
    cpu = solve_admm(cpu_problem, cpu_constraints, config)
    gpu_problem = make_problem("cuda:0", batch=3, trade_cost=0.25)
    gpu_constraints = make_constraints("cuda:0")
    gpu = solve_admm(gpu_problem, gpu_constraints, config)

    assert_successful_result(cpu, cpu_problem, cpu_constraints)
    assert_successful_result(gpu, gpu_problem, gpu_constraints, max_violation=5e-7)
    assert gpu.device == "cuda:0"
    assert_final_block_rho_matches_scales(cpu, (1.0, 1.0, 1.0))
    assert_final_block_rho_matches_scales(gpu, (1.0, 1.0, 1.0))
    assert torch.allclose(
        block_rho_tensor(cpu),
        block_rho_tensor(gpu),
        atol=7e-8,
        rtol=1e-7,
    )
    assert torch.allclose(cpu.weights, gpu.weights.cpu(), atol=7e-6, rtol=1e-5)
    assert torch.allclose(cpu.objective, gpu.objective.cpu(), atol=7e-8, rtol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_adaptive_rho_dual_rescaling_cpu_cuda_parity() -> None:
    config = AdmmConfig(
        iterations=5000,
        tolerance=1e-8,
        check_interval=5,
        rho=1e-4,
        bisection_steps=48,
        row_scaling=True,
        adaptive_rho=True,
        adaptive_rho_interval=5,
        adaptive_rho_mu=2.0,
        adaptive_rho_tau=2.0,
        rho_min=1e-4,
        rho_max=1.0,
    )

    cpu_problem = make_problem(batch=3, trade_cost=0.25)
    cpu_constraints = make_ill_scaled_constraints()
    cpu = solve_admm(cpu_problem, cpu_constraints, config)
    gpu_problem = make_problem("cuda:0", batch=3, trade_cost=0.25)
    gpu_constraints = make_ill_scaled_constraints("cuda:0")
    gpu = solve_admm(gpu_problem, gpu_constraints, config)

    assert_successful_result(cpu, cpu_problem, cpu_constraints, max_violation=2e-5)
    assert_successful_result(gpu, gpu_problem, gpu_constraints, max_violation=2e-5)
    assert cpu.rho_update_count > 0
    assert gpu.rho_update_count > 0
    assert cpu.final_rho != pytest.approx(config.rho)
    assert gpu.final_rho != pytest.approx(config.rho)
    assert_final_block_rho_matches_scales(cpu, (1.0, 1.0, 1.0))
    assert_final_block_rho_matches_scales(gpu, (1.0, 1.0, 1.0))
    assert torch.allclose(
        block_rho_tensor(cpu),
        block_rho_tensor(gpu),
        atol=7e-8,
        rtol=1e-7,
    )
    assert gpu.device == "cuda:0"
    assert torch.allclose(cpu.weights, gpu.weights.cpu(), atol=7e-6, rtol=1e-5)
    assert torch.allclose(cpu.objective, gpu.objective.cpu(), atol=7e-8, rtol=1e-7)
