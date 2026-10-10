from __future__ import annotations

import pytest
import torch

from openpo import FactorRiskModel, LinearConstraints, PortfolioProblem
from openpo.constrained_solver import PrimalDualConfig
from openpo.facade import (
    FallbackPolicy,
    ProductionRequest,
    SolveStatus,
    certify_exact,
    input_sha256,
    prepare_production_workspace,
    solve_fast,
    solve_prepared_production,
    solve_production,
)


def _request(
    *,
    device: str = "cpu",
    iterations: int = 400,
    exact_backend: str = "osqp",
) -> ProductionRequest:
    dtype = torch.float64
    problem = PortfolioProblem(
        alpha=torch.tensor([[0.10, 0.05, 0.02, 0.01]], dtype=dtype),
        risk=FactorRiskModel(
            exposures=torch.tensor([[1.0], [0.5], [-0.2], [0.1]], dtype=dtype),
            factor_variance=torch.tensor([0.03], dtype=dtype),
            specific_variance=torch.tensor([0.02, 0.03, 0.025, 0.04], dtype=dtype),
        ),
        lower=torch.zeros(4, dtype=dtype),
        upper=torch.full((4,), 0.8, dtype=dtype),
        current=torch.full((4,), 0.25, dtype=dtype),
        trade_cost=0.01,
        risk_aversion=2.0,
    )
    constraints = LinearConstraints(
        A_ineq=torch.tensor([[1.0, -1.0, 0.0, 0.0]], dtype=dtype),
        lower=torch.tensor([-0.4], dtype=dtype),
        upper=torch.tensor([0.4], dtype=dtype),
    )
    return ProductionRequest(
        problem=problem,
        constraints=constraints,
        fast_device=device,
        exact_backend=exact_backend,
        fast_config=PrimalDualConfig(iterations=iterations, tolerance=1e-8, check_interval=10),
    )


def _replace_problem(
    request: ProductionRequest,
    *,
    alpha: torch.Tensor | None = None,
    current: torch.Tensor | None = None,
    risk_aversion: float | None = None,
) -> PortfolioProblem:
    return PortfolioProblem(
        alpha=request.problem.alpha if alpha is None else alpha,
        risk=request.problem.risk,
        lower=request.problem.lower,
        upper=request.problem.upper,
        current=request.problem.current if current is None else current,
        trade_cost=request.problem.trade_cost,
        risk_aversion=request.problem.risk_aversion if risk_aversion is None else risk_aversion,
    )


def _replace_request(
    request: ProductionRequest,
    *,
    problem: PortfolioProblem | None = None,
    device: str | None = None,
    exact_backend: str | None = None,
    fallback_policy: FallbackPolicy | None = None,
) -> ProductionRequest:
    return ProductionRequest(
        problem=request.problem if problem is None else problem,
        constraints=request.constraints,
        fast_device=request.fast_device if device is None else device,
        exact_backend=request.exact_backend if exact_backend is None else exact_backend,
        fallback_policy=request.fallback_policy if fallback_policy is None else fallback_policy,
        fast_config=request.fast_config,
        residual_tolerance=request.residual_tolerance,
        objective_gap_tolerance=request.objective_gap_tolerance,
    )


def test_input_hash_is_stable_and_covers_scalars() -> None:
    request = _request()
    digest = input_sha256(request.problem, request.constraints)
    assert digest == input_sha256(request.problem, request.constraints)
    changed = PortfolioProblem(
        alpha=request.problem.alpha,
        risk=request.problem.risk,
        lower=request.problem.lower,
        upper=request.problem.upper,
        current=request.problem.current,
        trade_cost=0.02,
        risk_aversion=request.problem.risk_aversion,
    )
    assert digest != input_sha256(changed, request.constraints)


def test_solve_fast_returns_replayable_evidence() -> None:
    request = _request()
    first = solve_fast(request)
    second = solve_fast(request)
    assert first.status is SolveStatus.SOLVED_TO_TOLERANCE
    assert first.input_sha256 == second.input_sha256
    assert first.request_sha256 == second.request_sha256
    assert first.result_sha256 == second.result_sha256
    assert first.weights_sha256 == second.weights_sha256
    assert first.residuals.primal_residual <= request.residual_tolerance
    assert first.elapsed_seconds >= 0.0


def test_unavailable_cuda_never_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    request = _request(device="cuda:0")
    with pytest.raises(RuntimeError, match="unavailable"):
        solve_fast(request)


def test_exact_and_production_share_request_and_input_hashes() -> None:
    request = _request()
    exact = certify_exact(request)
    production = solve_production(request)
    replay = solve_production(request)

    assert exact.status is SolveStatus.OPTIMAL
    assert production.status is SolveStatus.SOLVED_TO_TOLERANCE
    assert production.fast.input_sha256 == production.exact.input_sha256
    assert production.fast.request_sha256 == production.exact.request_sha256
    assert production.request_sha256 == production.fast.request_sha256
    assert production.result_sha256 == replay.result_sha256
    assert production.objective_gap_sha256 == replay.objective_gap_sha256
    assert production.fast.result_sha256 == replay.fast.result_sha256
    assert production.exact.result_sha256 == replay.exact.result_sha256
    assert production.objective_gap is not None
    assert production.objective_gap <= request.objective_gap_tolerance


@pytest.mark.parametrize("exact_backend", ["osqp", "clarabel"])
def test_prepared_production_matches_cold_backend(exact_backend: str) -> None:
    request = _request(exact_backend=exact_backend)
    workspace = prepare_production_workspace(request)

    cold = solve_production(request)
    prepared = solve_prepared_production(workspace, request)

    assert prepared.prepared is True
    assert prepared.production_template_sha256 == workspace.production_template_sha256
    assert prepared.exact_template_sha256 == workspace.exact_template_sha256
    assert prepared.exact.prepared is True
    assert prepared.exact.template_sha256 == workspace.exact_template_sha256
    assert prepared.exact.setup_seconds is not None
    assert prepared.exact.cold_seconds is not None
    assert prepared.exact.build_seconds is not None
    assert prepared.exact.update_seconds is not None
    assert prepared.exact.solve_seconds is not None
    assert prepared.exact.elapsed_seconds >= prepared.exact.update_seconds + prepared.exact.solve_seconds
    assert prepared.status is cold.status
    assert prepared.selected_lane == cold.selected_lane
    assert prepared.fallback_used is cold.fallback_used
    assert prepared.fast.weights_sha256 == cold.fast.weights_sha256
    assert torch.allclose(prepared.exact.weights, cold.exact.weights, atol=1e-7, rtol=1e-7)
    assert torch.allclose(prepared.exact.objective, cold.exact.objective, atol=1e-8, rtol=1e-8)
    assert prepared.objective_gap == pytest.approx(cold.objective_gap, abs=1e-12)


def test_prepared_workspace_allows_dynamic_alpha_and_current() -> None:
    request = _request()
    workspace = prepare_production_workspace(request)
    dynamic_problem = _replace_problem(
        request,
        alpha=request.problem.alpha + torch.tensor([[0.01, -0.005, 0.002, 0.0]], dtype=request.problem.alpha.dtype),
        current=request.problem.current + torch.tensor([0.01, -0.01, 0.0, 0.0], dtype=request.problem.alpha.dtype),
    )
    dynamic_request = _replace_request(request, problem=dynamic_problem)

    cold = solve_production(dynamic_request)
    prepared = solve_prepared_production(workspace, dynamic_request)

    assert input_sha256(dynamic_request.problem, dynamic_request.constraints) != workspace.prepare_input_sha256
    assert prepared.production_template_sha256 == workspace.production_template_sha256
    assert prepared.exact_template_sha256 == workspace.exact_template_sha256
    assert torch.allclose(prepared.exact.weights, cold.exact.weights, atol=1e-7, rtol=1e-7)
    assert torch.allclose(prepared.exact.objective, cold.exact.objective, atol=1e-9, rtol=1e-9)
    assert torch.allclose(prepared.fast.weights, cold.fast.weights)
    assert prepared.objective_gap == pytest.approx(cold.objective_gap, abs=1e-12)


def test_prepared_workspace_rejects_static_template_drift() -> None:
    request = _request()
    workspace = prepare_production_workspace(request)
    drifted = _replace_request(
        request,
        problem=_replace_problem(request, risk_aversion=request.problem.risk_aversion * 1.5),
    )

    with pytest.raises(RuntimeError, match="static template drift"):
        solve_prepared_production(workspace, drifted)


def test_prepared_workspace_rejects_route_drift() -> None:
    request = _request()
    workspace = prepare_production_workspace(request)

    with pytest.raises(RuntimeError, match="exact_backend differs"):
        solve_prepared_production(workspace, _replace_request(request, exact_backend="clarabel"))

    with pytest.raises(RuntimeError, match="fallback_policy differs"):
        solve_prepared_production(
            workspace,
            _replace_request(request, fallback_policy=FallbackPolicy.ALLOW_CPU_EXACT),
        )


def test_prepared_workspace_cannot_cross_device() -> None:
    request = _request()
    workspace = prepare_production_workspace(request)

    with pytest.raises(RuntimeError, match="fast_device differs"):
        solve_prepared_production(workspace, _replace_request(request, device="cuda:0"))


def test_fallback_is_denied_by_default_and_explicit_when_allowed() -> None:
    denied_request = _request(iterations=1)
    denied = solve_production(denied_request)
    assert denied.status is SolveStatus.MAX_ITER
    assert denied.selected_lane == "fast"
    assert denied.fallback_used is False

    allowed_request = ProductionRequest(
        problem=denied_request.problem,
        constraints=denied_request.constraints,
        fast_device="cpu",
        fallback_policy=FallbackPolicy.ALLOW_CPU_EXACT,
        fast_config=denied_request.fast_config,
    )
    allowed = solve_production(allowed_request)
    assert allowed.status is SolveStatus.FALLBACK_USED
    assert allowed.selected_lane == "exact"
    assert allowed.fallback_used is True
    assert allowed.fallback_reason == "fast lane ended with max_iter"


def test_request_requires_explicit_device_and_policy() -> None:
    base = _request()
    with pytest.raises(ValueError, match="explicit"):
        ProductionRequest(base.problem, base.constraints, fast_device="cuda")
    with pytest.raises(TypeError, match="FallbackPolicy"):
        ProductionRequest(
            base.problem,
            base.constraints,
            fast_device="cpu",
            fallback_policy="deny",  # type: ignore[arg-type]
        )
    assert base.fallback_policy is FallbackPolicy.DENY
