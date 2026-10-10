import numpy as np
import osqp
import pytest
import scipy.sparse as sparse
import torch

from openpo.constrained_solver import PrimalDualConfig, solve_constrained
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


def solve_osqp_reference(problem: PortfolioProblem, constraints: LinearConstraints) -> np.ndarray:
    assets = problem.risk.assets
    matrix = sparse.eye(assets, format="csc") * problem.risk_aversion
    rows = [np.ones((1, assets)), np.eye(assets)]
    lower = [np.array([problem.budget]), problem.lower.numpy()]
    upper = [np.array([problem.budget]), problem.upper.numpy()]
    rows.extend([constraints.A_eq.numpy(), constraints.A_ineq.numpy()])
    lower.extend([constraints.b_eq.numpy(), constraints.lower.numpy()])
    upper.extend([constraints.b_eq.numpy(), constraints.upper.numpy()])
    constraint_matrix = sparse.csc_matrix(np.vstack(rows))
    solver = osqp.OSQP()
    solver.setup(
        P=matrix,
        q=-problem.alpha[0].numpy(),
        A=constraint_matrix,
        l=np.concatenate(lower),
        u=np.concatenate(upper),
        eps_abs=1e-9,
        eps_rel=1e-9,
        polishing=True,
        verbose=False,
    )
    result = solver.solve(raise_error=True)
    return result.x


def test_constrained_solver_matches_osqp_and_constraints() -> None:
    problem = make_problem(batch=1)
    constraints = make_constraints()
    result = solve_constrained(
        problem,
        constraints,
        PrimalDualConfig(iterations=5000, tolerance=1e-8, constraint_tolerance=1e-6),
    )
    reference = solve_osqp_reference(problem, constraints)
    residual = constraints.residual(result.weights)
    assert result.converged
    assert result.max_constraint_violation <= 1e-6
    assert float(residual.max_violation) <= 1e-6
    assert np.allclose(result.weights[0].numpy(), reference, atol=2e-5, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_constrained_solver_cpu_cuda_parity() -> None:
    config = PrimalDualConfig(iterations=3000, tolerance=1e-7, constraint_tolerance=2e-6)
    cpu = solve_constrained(make_problem(), make_constraints(), config)
    gpu = solve_constrained(make_problem("cuda:0"), make_constraints("cuda:0"), config)
    assert gpu.max_constraint_violation <= 2e-6
    assert torch.allclose(cpu.weights, gpu.weights.cpu(), atol=3e-5, rtol=3e-4)


def test_infeasible_constraints_return_nonconverged_residual() -> None:
    problem = make_problem(batch=1)
    impossible = LinearConstraints(
        A_eq=torch.ones(1, 16),
        b_eq=torch.tensor([2.0]),
    )
    result = solve_constrained(
        problem,
        impossible,
        PrimalDualConfig(iterations=200, tolerance=1e-8, constraint_tolerance=1e-6),
    )
    assert not result.converged
    assert result.max_constraint_violation >= 0.9
