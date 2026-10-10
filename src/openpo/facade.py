from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import importlib.metadata
import inspect
import json
import math
import re
import time
from typing import Any

import torch

from .constrained_solver import PrimalDualConfig, solve_constrained
from .constraints import LinearConstraints
from .model import PortfolioProblem


class SolveStatus(str, Enum):
    OPTIMAL = "optimal"
    SOLVED_TO_TOLERANCE = "solved_to_tolerance"
    MAX_ITER = "max_iter"
    NUMERICAL_FAILURE = "numerical_failure"
    INFEASIBLE_OR_UNBOUNDED_UNKNOWN = "infeasible_or_unbounded_unknown"
    FALLBACK_USED = "fallback_used"


class FallbackPolicy(str, Enum):
    DENY = "deny"
    ALLOW_CPU_EXACT = "allow_cpu_exact"


@dataclass(frozen=True)
class ResidualEvidence:
    budget_error: float
    box_lower_violation: float
    box_upper_violation: float
    box_violation: float
    linear_equality_error: float
    linear_lower_violation: float
    linear_upper_violation: float
    linear_violation: float
    primal_residual: float
    dual_residual: float | None


@dataclass(frozen=True)
class ProductionRequest:
    problem: PortfolioProblem
    constraints: LinearConstraints
    fast_device: str
    exact_backend: str = "osqp"
    fallback_policy: FallbackPolicy = FallbackPolicy.DENY
    fast_config: PrimalDualConfig = field(default_factory=PrimalDualConfig)
    residual_tolerance: float = 2e-5
    objective_gap_tolerance: float = 0.05

    def __post_init__(self) -> None:
        if self.fast_device != "cpu" and re.fullmatch(r"cuda:\d+", self.fast_device) is None:
            raise ValueError("fast_device must be 'cpu' or an explicit 'cuda:N' device")
        if self.exact_backend not in {"osqp", "clarabel"}:
            raise ValueError("exact_backend must be 'osqp' or 'clarabel'")
        if not isinstance(self.fallback_policy, FallbackPolicy):
            raise TypeError("fallback_policy must be a FallbackPolicy")
        if not math.isfinite(self.residual_tolerance) or self.residual_tolerance < 0:
            raise ValueError("residual_tolerance must be finite and nonnegative")
        if not math.isfinite(self.objective_gap_tolerance) or self.objective_gap_tolerance < 0:
            raise ValueError("objective_gap_tolerance must be finite and nonnegative")
        if self.constraints.num_variables != self.problem.risk.assets:
            raise ValueError("constraints and problem asset counts differ")


@dataclass(frozen=True)
class LaneResult:
    lane: str
    status: SolveStatus
    certificate: str
    requested_backend: str
    backend: str
    backend_version: str
    device: str
    input_sha256: str
    request_sha256: str
    weights_sha256: str
    result_sha256: str
    weights: torch.Tensor
    objective: torch.Tensor
    objective_gap: float | None
    residuals: ResidualEvidence
    iterations: int | None
    elapsed_seconds: float
    raw_statuses: tuple[str, ...]
    fallback_used: bool
    fallback_reason: str | None
    prepared: bool = False
    template_sha256: str | None = None
    setup_seconds: float | None = None
    cold_seconds: float | None = None
    build_seconds: float | None = None
    update_seconds: float | None = None
    solve_seconds: float | None = None


@dataclass(frozen=True)
class ProductionResult:
    status: SolveStatus
    input_sha256: str
    request_sha256: str
    result_sha256: str
    objective_gap_sha256: str
    objective_gap: float | None
    selected_lane: str
    fallback_used: bool
    fallback_reason: str | None
    fast: LaneResult
    exact: LaneResult
    prepared: bool = False
    production_template_sha256: str | None = None
    exact_template_sha256: str | None = None
    prepared_setup_seconds: float | None = None


@dataclass(frozen=True)
class PreparedProductionWorkspace:
    exact_workspace: Any
    fast_device: str
    exact_backend: str
    fallback_policy: FallbackPolicy
    production_template_sha256: str
    exact_template_sha256: str
    prepare_input_sha256: str
    setup_seconds: float
    native_exact_workspace: bool


def _tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(str(tuple(value.shape)).encode("utf-8"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def input_sha256(problem: PortfolioProblem, constraints: LinearConstraints) -> str:
    tensors: dict[str, torch.Tensor | None] = {
        "alpha": problem.alpha,
        "risk_exposures": problem.risk.exposures,
        "risk_factor_variance": problem.risk.factor_variance,
        "risk_specific_variance": problem.risk.specific_variance,
        "lower": problem.lower,
        "upper": problem.upper,
        "current": problem.current,
        "A_eq": constraints.A_eq,
        "b_eq": constraints.b_eq,
        "A_ineq": constraints.A_ineq,
        "constraint_lower": constraints.lower,
        "constraint_upper": constraints.upper,
    }
    return _canonical_sha256(
        {
            "tensors": {
                name: None if tensor is None else _tensor_sha256(tensor)
                for name, tensor in tensors.items()
            },
            "budget": float(problem.budget).hex(),
            "risk_aversion": float(problem.risk_aversion).hex(),
            "trade_cost": float(problem.trade_cost).hex(),
        }
    )


def _static_tensor_sha256(tensor: torch.Tensor | None) -> str | None:
    return None if tensor is None else _tensor_sha256(tensor)


def _static_input_template_sha256(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
) -> str:
    return _canonical_sha256(
        {
            "tensors": {
                "risk_exposures": _static_tensor_sha256(problem.risk.exposures),
                "risk_factor_variance": _static_tensor_sha256(problem.risk.factor_variance),
                "risk_specific_variance": _static_tensor_sha256(problem.risk.specific_variance),
                "lower": _static_tensor_sha256(problem.lower),
                "upper": _static_tensor_sha256(problem.upper),
                "A_eq": _static_tensor_sha256(constraints.A_eq),
                "b_eq": _static_tensor_sha256(constraints.b_eq),
                "A_ineq": _static_tensor_sha256(constraints.A_ineq),
                "constraint_lower": _static_tensor_sha256(constraints.lower),
                "constraint_upper": _static_tensor_sha256(constraints.upper),
            },
            "budget": float(problem.budget).hex(),
            "risk_aversion": float(problem.risk_aversion).hex(),
            "trade_cost": float(problem.trade_cost).hex(),
            "assets": problem.risk.assets,
            "alpha_dtype": str(problem.alpha.dtype),
            "constraint_dtype": str(constraints.dtype),
        }
    )


def _production_template_sha256(request: ProductionRequest) -> str:
    return _canonical_sha256(
        {
            "static_input_template_sha256": _static_input_template_sha256(request.problem, request.constraints),
            "fast_device": request.fast_device,
            "exact_backend": request.exact_backend,
            "fallback_policy": request.fallback_policy.value,
            "fast_config": asdict(request.fast_config),
            "residual_tolerance": float(request.residual_tolerance).hex(),
            "objective_gap_tolerance": float(request.objective_gap_tolerance).hex(),
        }
    )


def _exact_template_sha256(request: ProductionRequest) -> str:
    from . import exact_solver

    exact_template_sha256 = getattr(exact_solver, "exact_template_sha256", None)
    if exact_template_sha256 is not None:
        return exact_template_sha256(
            request.problem.to("cpu"),
            request.constraints.to("cpu"),
            request.exact_backend,
        )
    return _canonical_sha256(
        {
            "static_input_template_sha256": _static_input_template_sha256(request.problem, request.constraints),
            "exact_backend": request.exact_backend,
        }
    )


def _request_sha256(request: ProductionRequest) -> str:
    return _canonical_sha256(
        {
            "input_sha256": input_sha256(request.problem, request.constraints),
            "fast_device": request.fast_device,
            "exact_backend": request.exact_backend,
            "fallback_policy": request.fallback_policy.value,
            "fast_config": asdict(request.fast_config),
            "residual_tolerance": float(request.residual_tolerance).hex(),
            "objective_gap_tolerance": float(request.objective_gap_tolerance).hex(),
        }
    )


def _scalar_sha256(value: float | None) -> str:
    return _canonical_sha256({"value": None if value is None else float(value).hex()})


def _device_available(device: str) -> bool:
    if device == "cpu":
        return True
    index = int(device.split(":", 1)[1])
    return torch.cuda.is_available() and index < torch.cuda.device_count()


def _residuals(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    weights: torch.Tensor,
) -> ResidualEvidence:
    cpu_problem = problem.to("cpu")
    cpu_constraints = constraints.to("cpu")
    cpu_weights = weights.detach().cpu()
    if not torch.isfinite(cpu_weights).all().item():
        raise RuntimeError("solver returned non-finite weights")
    budget = float((cpu_weights.sum(dim=1) - cpu_problem.budget).abs().amax().item())
    lower = float(torch.clamp(cpu_problem.lower - cpu_weights, min=0).amax().item())
    upper = float(torch.clamp(cpu_weights - cpu_problem.upper, min=0).amax().item())
    linear = cpu_constraints.residual(cpu_weights)
    equality = 0.0 if linear.equality is None else float(linear.equality.abs().amax().item())
    linear_lower = 0.0 if linear.lower_violation is None else float(linear.lower_violation.amax().item())
    linear_upper = 0.0 if linear.upper_violation is None else float(linear.upper_violation.amax().item())
    box = max(lower, upper)
    linear_max = max(equality, linear_lower, linear_upper)
    return ResidualEvidence(
        budget_error=budget,
        box_lower_violation=lower,
        box_upper_violation=upper,
        box_violation=box,
        linear_equality_error=equality,
        linear_lower_violation=linear_lower,
        linear_upper_violation=linear_upper,
        linear_violation=linear_max,
        primal_residual=max(budget, box, linear_max),
        dual_residual=None,
    )


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _lane_result(
    *,
    lane: str,
    status: SolveStatus,
    certificate: str,
    requested_backend: str,
    backend: str,
    backend_version: str,
    device: str,
    input_hash: str,
    request_hash: str,
    weights: torch.Tensor,
    objective: torch.Tensor,
    objective_gap: float | None,
    residuals: ResidualEvidence,
    iterations: int | None,
    elapsed_seconds: float,
    raw_statuses: tuple[str, ...],
    fallback_used: bool = False,
    fallback_reason: str | None = None,
    prepared: bool = False,
    template_sha256: str | None = None,
    setup_seconds: float | None = None,
    cold_seconds: float | None = None,
    build_seconds: float | None = None,
    update_seconds: float | None = None,
    solve_seconds: float | None = None,
) -> LaneResult:
    cpu_weights = weights.detach().contiguous().cpu()
    cpu_objective = objective.detach().contiguous().cpu()
    weights_hash = _tensor_sha256(cpu_weights)
    result_hash = _canonical_sha256(
        {
            "lane": lane,
            "status": status.value,
            "certificate": certificate,
            "requested_backend": requested_backend,
            "backend": backend,
            "backend_version": backend_version,
            "device": device,
            "input_sha256": input_hash,
            "request_sha256": request_hash,
            "weights_sha256": weights_hash,
            "objective": [float(value).hex() for value in cpu_objective.reshape(-1)],
            "objective_gap_sha256": _scalar_sha256(objective_gap),
            "residuals": asdict(residuals),
            "iterations": iterations,
            "raw_statuses": list(raw_statuses),
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            **(
                {
                    "prepared": True,
                    "template_sha256": template_sha256,
                }
                if prepared
                else {}
            ),
        }
    )
    return LaneResult(
        lane=lane,
        status=status,
        certificate=certificate,
        requested_backend=requested_backend,
        backend=backend,
        backend_version=backend_version,
        device=device,
        input_sha256=input_hash,
        request_sha256=request_hash,
        weights_sha256=weights_hash,
        result_sha256=result_hash,
        weights=cpu_weights,
        objective=cpu_objective,
        objective_gap=objective_gap,
        residuals=residuals,
        iterations=iterations,
        elapsed_seconds=elapsed_seconds,
        raw_statuses=raw_statuses,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        prepared=prepared,
        template_sha256=template_sha256,
        setup_seconds=setup_seconds,
        cold_seconds=cold_seconds,
        build_seconds=build_seconds,
        update_seconds=update_seconds,
        solve_seconds=solve_seconds,
    )


def solve_fast(request: ProductionRequest) -> LaneResult:
    if not _device_available(request.fast_device):
        raise RuntimeError(f"requested fast device is unavailable: {request.fast_device}")
    frozen_input = input_sha256(request.problem, request.constraints)
    problem = request.problem.to(request.fast_device)
    constraints = request.constraints.to(request.fast_device)
    result = solve_constrained(problem, constraints, request.fast_config)
    if input_sha256(request.problem, request.constraints) != frozen_input:
        raise RuntimeError("production request changed during fast solve")
    residuals = _residuals(request.problem, request.constraints, result.weights)
    converged = result.converged and residuals.primal_residual <= request.residual_tolerance
    status = SolveStatus.SOLVED_TO_TOLERANCE if converged else SolveStatus.MAX_ITER
    backend = "openpo_cuda" if request.fast_device.startswith("cuda:") else "openpo_cpu"
    return _lane_result(
        lane="fast",
        status=status,
        certificate="HEURISTIC_ONLY",
        requested_backend=backend,
        backend=backend,
        backend_version=_package_version("openpo"),
        device=request.fast_device,
        input_hash=frozen_input,
        request_hash=_request_sha256(request),
        weights=result.weights,
        objective=result.objective,
        objective_gap=None,
        residuals=residuals,
        iterations=result.iterations,
        elapsed_seconds=result.elapsed_seconds,
        raw_statuses=("converged" if result.converged else "max_iter",),
    )


def _exact_status(raw_statuses: tuple[str, ...]) -> SolveStatus:
    lowered = [status.lower() for status in raw_statuses]
    if lowered and all("solved" in status or "optimal" in status for status in lowered):
        return SolveStatus.OPTIMAL
    if any("maximum" in status or "max_iter" in status for status in lowered):
        return SolveStatus.MAX_ITER
    if any("infeasible" in status or "unbounded" in status for status in lowered):
        return SolveStatus.INFEASIBLE_OR_UNBOUNDED_UNKNOWN
    return SolveStatus.NUMERICAL_FAILURE


def _call_prepare_exact_workspace(function: Any, problem: PortfolioProblem, constraints: LinearConstraints, backend: str) -> Any:
    signature = inspect.signature(function)
    parameters = signature.parameters
    if "backend" in parameters:
        return function(problem, constraints, backend=backend)
    if "exact_backend" in parameters:
        return function(problem, constraints, exact_backend=backend)
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        parameter
        for parameter in positional
        if parameter.default is inspect.Parameter.empty
    ]
    if len(required) >= 3:
        return function(problem, constraints, backend)
    if len(required) == 2:
        return function(problem, constraints)
    raise TypeError("prepare_exact_workspace has an unsupported signature")


def _call_solve_exact_prepared(
    function: Any,
    exact_workspace: Any,
    problem: PortfolioProblem,
    constraints: LinearConstraints,
) -> Any:
    signature = inspect.signature(function)
    parameters = signature.parameters
    if "problem" in parameters and "constraints" in parameters:
        return function(exact_workspace, problem, constraints)
    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    required = [
        parameter
        for parameter in positional
        if parameter.default is inspect.Parameter.empty
    ]
    if len(required) >= 3:
        return function(exact_workspace, problem, constraints)
    raise TypeError("solve_exact_prepared has an unsupported signature")


def _workspace_template_sha256(exact_workspace: Any) -> str | None:
    for name in ("template_sha256", "template_hash", "exact_template_sha256"):
        value = getattr(exact_workspace, name, None)
        if isinstance(value, str) and value:
            return value
    template = getattr(exact_workspace, "template", None)
    if isinstance(template, str) and template:
        return template
    return None


def _prepare_exact_workspace_for_request(request: ProductionRequest) -> tuple[Any, bool, str]:
    from . import exact_solver

    expected_template = _exact_template_sha256(request)
    prepare_exact_workspace = getattr(exact_solver, "prepare_exact_workspace", None)
    if prepare_exact_workspace is None:
        return None, False, expected_template
    exact_workspace = _call_prepare_exact_workspace(
        prepare_exact_workspace,
        request.problem.to("cpu"),
        request.constraints.to("cpu"),
        request.exact_backend,
    )
    workspace_template = _workspace_template_sha256(exact_workspace)
    return exact_workspace, True, workspace_template or expected_template


def _solve_exact_with_workspace(workspace: PreparedProductionWorkspace, request: ProductionRequest) -> Any:
    from . import exact_solver

    problem = request.problem.to("cpu")
    constraints = request.constraints.to("cpu")
    if workspace.native_exact_workspace:
        solve_exact_prepared = getattr(exact_solver, "solve_exact_prepared", None)
        if solve_exact_prepared is None:
            raise RuntimeError("exact prepared workspace exists but solve_exact_prepared is unavailable")
        workspace_template = _workspace_template_sha256(workspace.exact_workspace)
        if workspace_template is not None and workspace_template != workspace.exact_template_sha256:
            raise RuntimeError("exact workspace template drift detected")
        return _call_solve_exact_prepared(
            solve_exact_prepared,
            workspace.exact_workspace,
            problem,
            constraints,
        )
    return exact_solver.solve_exact(problem, constraints, backend=request.exact_backend)


def _validate_prepared_workspace(workspace: PreparedProductionWorkspace, request: ProductionRequest) -> None:
    if request.fast_device != workspace.fast_device:
        raise RuntimeError("prepared production route drift: fast_device differs")
    if request.exact_backend != workspace.exact_backend:
        raise RuntimeError("prepared production route drift: exact_backend differs")
    if request.fallback_policy is not workspace.fallback_policy:
        raise RuntimeError("prepared production route drift: fallback_policy differs")
    if _production_template_sha256(request) != workspace.production_template_sha256:
        raise RuntimeError("prepared production static template drift detected")
    if _exact_template_sha256(request) != workspace.exact_template_sha256:
        raise RuntimeError("prepared production exact template drift detected")


def certify_exact(request: ProductionRequest) -> LaneResult:
    from .exact_solver import solve_exact

    frozen_input = input_sha256(request.problem, request.constraints)
    result = solve_exact(
        request.problem.to("cpu"),
        request.constraints.to("cpu"),
        backend=request.exact_backend,
    )
    if input_sha256(request.problem, request.constraints) != frozen_input:
        raise RuntimeError("production request changed during exact certification")
    residuals = _residuals(request.problem, request.constraints, result.weights)
    status = _exact_status(result.raw_statuses)
    if status is SolveStatus.OPTIMAL and residuals.primal_residual > request.residual_tolerance:
        status = SolveStatus.NUMERICAL_FAILURE
    return _lane_result(
        lane="exact",
        status=status,
        certificate="EXACT_SOLVER",
        requested_backend=request.exact_backend,
        backend=result.backend,
        backend_version=result.backend_version,
        device=result.device,
        input_hash=frozen_input,
        request_hash=_request_sha256(request),
        weights=result.weights,
        objective=result.objective,
        objective_gap=0.0 if status is SolveStatus.OPTIMAL else None,
        residuals=residuals,
        iterations=None,
        elapsed_seconds=result.elapsed_seconds,
        raw_statuses=result.raw_statuses,
        setup_seconds=result.setup_seconds,
        cold_seconds=result.cold_seconds,
        build_seconds=result.build_seconds,
        update_seconds=result.update_seconds,
        solve_seconds=result.solve_seconds,
    )


def prepare_production_workspace(request: ProductionRequest) -> PreparedProductionWorkspace:
    if not _device_available(request.fast_device):
        raise RuntimeError(f"requested fast device is unavailable: {request.fast_device}")
    frozen_input = input_sha256(request.problem, request.constraints)
    started = time.perf_counter()
    exact_workspace, native_exact_workspace, exact_template = _prepare_exact_workspace_for_request(request)
    setup_seconds = time.perf_counter() - started
    if input_sha256(request.problem, request.constraints) != frozen_input:
        raise RuntimeError("production request changed during workspace preparation")
    production_template = _production_template_sha256(request)
    expected_exact_template = _exact_template_sha256(request)
    if exact_template != expected_exact_template:
        raise RuntimeError("exact workspace template differs from production request template")
    return PreparedProductionWorkspace(
        exact_workspace=exact_workspace,
        fast_device=request.fast_device,
        exact_backend=request.exact_backend,
        fallback_policy=request.fallback_policy,
        production_template_sha256=production_template,
        exact_template_sha256=exact_template,
        prepare_input_sha256=frozen_input,
        setup_seconds=setup_seconds,
        native_exact_workspace=native_exact_workspace,
    )


def _certify_exact_prepared(
    workspace: PreparedProductionWorkspace,
    request: ProductionRequest,
) -> LaneResult:
    _validate_prepared_workspace(workspace, request)
    frozen_input = input_sha256(request.problem, request.constraints)
    result = _solve_exact_with_workspace(workspace, request)
    if input_sha256(request.problem, request.constraints) != frozen_input:
        raise RuntimeError("production request changed during prepared exact certification")
    residuals = _residuals(request.problem, request.constraints, result.weights)
    status = _exact_status(result.raw_statuses)
    if status is SolveStatus.OPTIMAL and residuals.primal_residual > request.residual_tolerance:
        status = SolveStatus.NUMERICAL_FAILURE
    return _lane_result(
        lane="exact",
        status=status,
        certificate="EXACT_SOLVER",
        requested_backend=request.exact_backend,
        backend=result.backend,
        backend_version=result.backend_version,
        device=result.device,
        input_hash=frozen_input,
        request_hash=_request_sha256(request),
        weights=result.weights,
        objective=result.objective,
        objective_gap=0.0 if status is SolveStatus.OPTIMAL else None,
        residuals=residuals,
        iterations=None,
        elapsed_seconds=result.elapsed_seconds,
        raw_statuses=result.raw_statuses,
        prepared=True,
        template_sha256=workspace.exact_template_sha256,
        setup_seconds=result.setup_seconds,
        cold_seconds=result.cold_seconds,
        build_seconds=result.build_seconds,
        update_seconds=result.update_seconds,
        solve_seconds=result.solve_seconds,
    )


def _assemble_production_result(
    request: ProductionRequest,
    input_hash: str,
    fast: LaneResult,
    exact: LaneResult,
    *,
    prepared: bool = False,
    production_template_sha256: str | None = None,
    exact_template_sha256: str | None = None,
    prepared_setup_seconds: float | None = None,
) -> ProductionResult:
    if fast.input_sha256 != input_hash or exact.input_sha256 != input_hash:
        raise RuntimeError("fast and exact lanes used different frozen inputs")
    if fast.objective.shape != exact.objective.shape:
        raise RuntimeError("fast and exact objective shapes differ")
    denominator = exact.objective.abs().clamp_min(1e-12)
    objective_gap = float(((fast.objective - exact.objective).abs() / denominator).amax().item())
    fast = _lane_result(
        lane=fast.lane,
        status=fast.status,
        certificate=fast.certificate,
        requested_backend=fast.requested_backend,
        backend=fast.backend,
        backend_version=fast.backend_version,
        device=fast.device,
        input_hash=fast.input_sha256,
        request_hash=fast.request_sha256,
        weights=fast.weights,
        objective=fast.objective,
        objective_gap=objective_gap,
        residuals=fast.residuals,
        iterations=fast.iterations,
        elapsed_seconds=fast.elapsed_seconds,
        raw_statuses=fast.raw_statuses,
        fallback_used=fast.fallback_used,
        fallback_reason=fast.fallback_reason,
    )
    if fast.status is not SolveStatus.SOLVED_TO_TOLERANCE:
        if request.fallback_policy is FallbackPolicy.ALLOW_CPU_EXACT and exact.status is SolveStatus.OPTIMAL:
            status = SolveStatus.FALLBACK_USED
            selected_lane = "exact"
            fallback_used = True
            fallback_reason = f"fast lane ended with {fast.status.value}"
        else:
            status = fast.status
            selected_lane = "fast"
            fallback_used = False
            fallback_reason = None
    elif exact.status is not SolveStatus.OPTIMAL:
        status = exact.status
        selected_lane = "fast"
        fallback_used = False
        fallback_reason = None
    elif objective_gap <= request.objective_gap_tolerance:
        status = SolveStatus.SOLVED_TO_TOLERANCE
        selected_lane = "fast"
        fallback_used = False
        fallback_reason = None
    else:
        status = SolveStatus.MAX_ITER
        selected_lane = "fast"
        fallback_used = False
        fallback_reason = None
    request_hash = _request_sha256(request)
    objective_gap_hash = _scalar_sha256(objective_gap)
    result_hash = _canonical_sha256(
        {
            "status": status.value,
            "input_sha256": input_hash,
            "request_sha256": request_hash,
            "objective_gap_sha256": objective_gap_hash,
            "selected_lane": selected_lane,
            "fallback_used": fallback_used,
            "fallback_reason": fallback_reason,
            "fast_result_sha256": fast.result_sha256,
            "exact_result_sha256": exact.result_sha256,
            **(
                {
                    "prepared": True,
                    "production_template_sha256": production_template_sha256,
                    "exact_template_sha256": exact_template_sha256,
                }
                if prepared
                else {}
            ),
        }
    )
    return ProductionResult(
        status=status,
        input_sha256=input_hash,
        request_sha256=request_hash,
        result_sha256=result_hash,
        objective_gap_sha256=objective_gap_hash,
        objective_gap=objective_gap,
        selected_lane=selected_lane,
        fallback_used=fallback_used,
        fallback_reason=fallback_reason,
        fast=fast,
        exact=exact,
        prepared=prepared,
        production_template_sha256=production_template_sha256,
        exact_template_sha256=exact_template_sha256,
        prepared_setup_seconds=prepared_setup_seconds,
    )


def solve_production(request: ProductionRequest) -> ProductionResult:
    input_hash = input_sha256(request.problem, request.constraints)
    fast = solve_fast(request)
    exact = certify_exact(request)
    return _assemble_production_result(request, input_hash, fast, exact)


def solve_prepared_production(
    workspace: PreparedProductionWorkspace,
    request: ProductionRequest,
) -> ProductionResult:
    _validate_prepared_workspace(workspace, request)
    input_hash = input_sha256(request.problem, request.constraints)
    fast = solve_fast(request)
    exact = _certify_exact_prepared(workspace, request)
    return _assemble_production_result(
        request,
        input_hash,
        fast,
        exact,
        prepared=True,
        production_template_sha256=workspace.production_template_sha256,
        exact_template_sha256=workspace.exact_template_sha256,
        prepared_setup_seconds=workspace.setup_seconds,
    )
