from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

CUDA_AVAILABLE = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())

from openpo.constrained_solver import PrimalDualConfig, PrimalDualState, solve_constrained
from openpo.constraints import LinearConstraints
from openpo.model import FactorRiskModel, PortfolioProblem


def make_problem(device: str = "cpu", batch: int = 2) -> PortfolioProblem:
    assets = 16
    alpha = torch.stack(
        [torch.linspace(-0.01 + offset, 0.02 + offset, assets) for offset in (0.0, 0.002)]
    )[:batch].to(device)
    return PortfolioProblem(
        alpha=alpha,
        risk=FactorRiskModel(
            exposures=torch.zeros(assets, 1, device=device),
            factor_variance=torch.ones(1, device=device),
            specific_variance=torch.ones(assets, device=device),
        ),
        lower=torch.zeros(assets, device=device),
        upper=torch.full((assets,), 0.2, device=device),
        risk_aversion=1.0,
    )


def make_constraints(device: str = "cpu") -> LinearConstraints:
    equality = torch.cat([torch.ones(8), -torch.ones(8)]).reshape(1, -1).to(device)
    inequality = torch.zeros(1, 16, device=device)
    inequality[0, :4] = 1.0
    return LinearConstraints(
        A_eq=equality,
        b_eq=torch.zeros(1, device=device),
        A_ineq=inequality,
        lower=torch.tensor([0.15], device=device),
        upper=torch.tensor([0.30], device=device),
    )


def test_constrained_solver_warm_start_matches_cold_and_does_not_increase_iterations() -> None:
    problem = make_problem(batch=2)
    constraints = make_constraints()
    config = PrimalDualConfig(
        iterations=5000,
        tolerance=1e-8,
        constraint_tolerance=1e-6,
        check_interval=1,
    )

    cold = solve_constrained(problem, constraints, config)
    warm = solve_constrained(
        problem,
        constraints,
        config,
        initial_state=cold.state,
    )

    assert hasattr(cold, "state")
    assert hasattr(warm, "state")
    assert cold.state.weights is not cold.weights
    assert cold.state.weights.data_ptr() != cold.weights.data_ptr()
    torch.testing.assert_close(cold.state.weights, cold.weights, atol=0.0, rtol=0.0)
    torch.testing.assert_close(warm.state.weights, warm.weights, atol=0.0, rtol=0.0)
    torch.testing.assert_close(warm.weights, cold.weights, atol=1e-6, rtol=0.0)
    assert warm.iterations <= cold.iterations
    assert cold.converged
    assert warm.converged


@pytest.mark.parametrize(
    ("case", "match"),
    [
        pytest.param("weights_shape", "shape", id="weights-shape"),
        pytest.param("dual_shape", "shape", id="dual-shape"),
        pytest.param("device", "same device", id="device"),
        pytest.param("dtype", "dtype", id="dtype"),
        pytest.param("nonfinite", "finite", id="nonfinite"),
    ],
)
def test_constrained_solver_rejects_invalid_initial_state(case: str, match: str) -> None:
    problem = make_problem(batch=1)
    constraints = make_constraints()
    weights = torch.full(
        problem.batched_alpha.shape,
        problem.budget / problem.risk.assets,
        device=problem.alpha.device,
        dtype=problem.alpha.dtype,
    )
    dual = torch.zeros(
        (problem.batched_alpha.shape[0], 2),
        device=problem.alpha.device,
        dtype=problem.alpha.dtype,
    )

    if case == "weights_shape":
        weights = weights[:, :-1]
    elif case == "dual_shape":
        dual = dual[:, :-1]
    elif case == "device":
        dual = dual.to("cuda" if CUDA_AVAILABLE else "meta")
    elif case == "dtype":
        weights = weights.double()
        dual = dual.double()
    elif case == "nonfinite":
        weights = weights.clone()
        weights[0, 0] = float("nan")
    else:  # pragma: no cover - defensive guard for future edits
        raise AssertionError(f"unexpected case: {case}")

    reference = solve_constrained(
        problem,
        constraints,
        PrimalDualConfig(iterations=10, tolerance=0.0, constraint_tolerance=0.0),
    ).state
    state = replace(reference, weights=weights, dual=dual)

    with pytest.raises((TypeError, ValueError), match=match):
        solve_constrained(
            problem,
            constraints,
            PrimalDualConfig(iterations=10, tolerance=0.0, constraint_tolerance=0.0),
            initial_state=state,
        )


def test_constrained_solver_rejects_state_from_different_constraints() -> None:
    problem = make_problem(batch=1)
    original = make_constraints()
    result = solve_constrained(problem, original)
    changed = LinearConstraints(
        A_eq=original.A_eq,
        b_eq=torch.tensor([0.01]),
        A_ineq=original.A_ineq,
        lower=original.lower,
        upper=original.upper,
    )

    with pytest.raises(ValueError, match="problem template does not match"):
        solve_constrained(problem, changed, initial_state=result.state)


def test_constrained_solver_enforces_alpha_drift_limit() -> None:
    problem = make_problem(batch=1)
    constraints = make_constraints()
    result = solve_constrained(problem, constraints)
    shifted = make_problem(batch=1)
    shifted.alpha.add_(0.01)

    with pytest.raises(ValueError, match="alpha drift"):
        solve_constrained(
            shifted,
            constraints,
            PrimalDualConfig(warm_start_max_alpha_delta=0.001),
            initial_state=result.state,
        )


@pytest.mark.parametrize("limit", [float("nan"), float("inf"), -float("inf")])
def test_primal_dual_config_rejects_nonfinite_alpha_drift_limit(limit: float) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        PrimalDualConfig(warm_start_max_alpha_delta=limit)


def test_constrained_solver_rejects_state_from_different_risk_template() -> None:
    problem = make_problem(batch=1)
    constraints = make_constraints()
    result = solve_constrained(problem, constraints)
    changed = make_problem(batch=1)
    changed.risk.specific_variance.add_(0.01)

    with pytest.raises(ValueError, match="problem template does not match"):
        solve_constrained(changed, constraints, initial_state=result.state)


def test_constrained_solver_does_not_retain_autograd_graph() -> None:
    problem = make_problem(batch=1)
    problem.alpha.requires_grad_(True)
    result = solve_constrained(problem, make_constraints())

    assert not result.weights.requires_grad
    assert not result.objective.requires_grad
    assert not result.state.weights.requires_grad
    assert not result.state.dual.requires_grad


@pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA unavailable")
def test_constrained_solver_cpu_cuda_parity() -> None:
    config = PrimalDualConfig(iterations=3000, tolerance=1e-7, constraint_tolerance=2e-6)
    cpu = solve_constrained(make_problem(), make_constraints(), config)
    gpu = solve_constrained(make_problem("cuda:0"), make_constraints("cuda:0"), config)

    assert cpu.converged
    assert gpu.converged
    assert gpu.max_constraint_violation <= 2e-6
    torch.testing.assert_close(cpu.weights, gpu.weights.cpu(), atol=3e-5, rtol=3e-4)
