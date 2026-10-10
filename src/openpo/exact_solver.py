from __future__ import annotations

from dataclasses import InitVar, dataclass, field
from enum import Enum
import hashlib
import importlib
import json
import threading
import time
from typing import Any

import numpy as np
import scipy.sparse as sparse
import torch

from .constraints import LinearConstraints
from .model import PortfolioProblem


class ExactBackend(str, Enum):
    OSQP = "osqp"
    CLARABEL = "clarabel"


EXACT_SOLVER_TOLERANCE = 1e-8
EXACT_SOLVER_TEMPLATE_SCHEMA_VERSION = 2
CLARABEL_DIRECT_SOLVE_METHOD = "qdldl"
CLARABEL_MAX_THREADS = 1
OSQP_SOLVER_SETTINGS: dict[str, Any] = {
    "verbose": False,
    "polish": True,
    "eps_abs": EXACT_SOLVER_TOLERANCE,
    "eps_rel": EXACT_SOLVER_TOLERANCE,
    "max_iter": 20_000,
}
CLARABEL_SOLVER_SETTINGS: dict[str, Any] = {
    "verbose": False,
    "max_iter": 500,
    "tol_feas": EXACT_SOLVER_TOLERANCE,
    "tol_gap_abs": EXACT_SOLVER_TOLERANCE,
    "tol_gap_rel": EXACT_SOLVER_TOLERANCE,
    "direct_solve_method": CLARABEL_DIRECT_SOLVE_METHOD,
    "max_threads": CLARABEL_MAX_THREADS,
}


def _canonical_template_evidence_json(evidence: dict[str, Any] | str | None) -> str | None:
    if evidence is None:
        return None
    if isinstance(evidence, str):
        evidence = json.loads(evidence)
    return json.dumps(evidence, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _template_evidence_from_json(evidence_json: str | None) -> dict[str, Any] | None:
    if evidence_json is None:
        return None
    loaded = json.loads(evidence_json)
    if not isinstance(loaded, dict):
        raise RuntimeError("template evidence JSON must decode to an object")
    return loaded


@dataclass(frozen=True)
class ExactSolveResult:
    weights: torch.Tensor
    objective: torch.Tensor
    raw_statuses: tuple[str, ...]
    setup_seconds: float
    cold_seconds: float
    elapsed_seconds: float
    backend: str
    backend_version: str
    device: str = "cpu"
    build_seconds: float = 0.0
    update_seconds: float = 0.0
    solve_seconds: float = 0.0
    prepared: bool = False
    template_sha256: str | None = None
    template_evidence: InitVar[dict[str, Any] | str | None] = None
    timing_evidence: dict[str, float] | None = None
    _template_evidence_json: str | None = field(init=False, repr=False, compare=False)

    def __post_init__(self, template_evidence: dict[str, Any] | str | None) -> None:
        object.__setattr__(
            self,
            "_template_evidence_json",
            _canonical_template_evidence_json(template_evidence),
        )

    @property
    def template_evidence(self) -> dict[str, Any] | None:
        return _template_evidence_from_json(self._template_evidence_json)


@dataclass(frozen=True)
class PreparedExactWorkspace:
    solver: object
    backend: str
    backend_version: str
    template_sha256: str
    template_evidence: InitVar[dict[str, Any] | str | None] = None
    setup_seconds: float = 0.0
    variable_count: int = 0
    assets: int = 0
    dtype: torch.dtype = torch.float64
    device: str = "cpu"
    lock: threading.Lock | None = None
    build_seconds: float = 0.0
    poisoned: bool = False
    poison_reason: str | None = None
    _template_evidence_json: str | None = field(init=False, repr=False, compare=False)

    def __post_init__(self, template_evidence: dict[str, Any] | str | None) -> None:
        if self.lock is None:
            object.__setattr__(self, "lock", threading.Lock())
        if template_evidence is None:
            raise ValueError("prepared exact workspace template evidence is required")
        object.__setattr__(
            self,
            "_template_evidence_json",
            _canonical_template_evidence_json(template_evidence),
        )

    @property
    def template_evidence(self) -> dict[str, Any]:
        evidence = _template_evidence_from_json(self._template_evidence_json)
        if evidence is None:
            raise RuntimeError("prepared exact workspace template evidence is missing")
        return evidence


def _normalize_backend(backend: ExactBackend | str) -> ExactBackend:
    if isinstance(backend, ExactBackend):
        return backend
    if isinstance(backend, str):
        normalized = backend.strip().lower()
        for member in ExactBackend:
            if normalized in (member.value, member.name.lower()):
                return member
    raise ValueError(f"unsupported exact backend: {backend!r}")


def _require_cpu_tensor(name: str, tensor: torch.Tensor | None) -> None:
    if tensor is not None and tensor.device.type != "cpu":
        raise ValueError(f"{name} must be on CPU for exact solving")


def _require_cpu_problem(problem: PortfolioProblem, constraints: LinearConstraints) -> None:
    tensors = [
        problem.alpha,
        problem.lower,
        problem.upper,
        problem.risk.exposures,
        problem.risk.factor_variance,
        problem.risk.specific_variance,
        constraints.A_eq,
        constraints.b_eq,
        constraints.A_ineq,
        constraints.lower,
        constraints.upper,
    ]
    if problem.current is not None:
        tensors.append(problem.current)
    for index, tensor in enumerate(tensors):
        _require_cpu_tensor(f"tensor[{index}]", tensor)


def _broadcast_rows(tensor: torch.Tensor, batch_size: int, assets: int, name: str) -> torch.Tensor:
    if tensor.ndim == 1:
        if tensor.shape[0] != assets:
            raise ValueError(f"{name} must have shape [assets]")
        return tensor.unsqueeze(0).expand(batch_size, -1)
    if tensor.ndim == 2:
        if tensor.shape[1] != assets:
            raise ValueError(f"{name} must have shape [batch, assets]")
        if tensor.shape[0] == batch_size:
            return tensor
        if tensor.shape[0] == 1:
            return tensor.expand(batch_size, -1)
    raise ValueError(f"{name} must have shape [assets] or [batch, assets]")


def _broadcast_optional_current(
    current: torch.Tensor | None,
    batch_size: int,
    assets: int,
) -> torch.Tensor | None:
    if current is None:
        return None
    return _broadcast_rows(current, batch_size, assets, "current")


def _align_batch(problem: PortfolioProblem) -> tuple[torch.Tensor, torch.Tensor | None]:
    alpha = problem.batched_alpha
    batch_size = alpha.shape[0]
    current = _broadcast_optional_current(problem.current, batch_size, problem.risk.assets)
    if current is not None and current.shape[0] != batch_size:
        raise ValueError("alpha and current batch dimensions are incompatible")
    return alpha, current


def _module_version(module: object, module_name: str) -> str:
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    try:
        import importlib.metadata as importlib_metadata

        return importlib_metadata.version(module_name)
    except Exception as error:  # pragma: no cover - defensive fallback
        raise RuntimeError(f"could not determine version for backend {module_name}") from error


def _validate_problem_constraints(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
) -> None:
    if constraints.num_variables != problem.risk.assets:
        raise ValueError("constraints and problem have inconsistent asset counts")
    if problem.alpha.device != problem.lower.device:
        raise ValueError("problem tensors must share the same device")
    if problem.alpha.dtype != problem.lower.dtype:
        raise ValueError("problem tensors must share the same dtype")
    if constraints.device != problem.alpha.device:
        raise ValueError("constraints and problem must be on the same device")
    if constraints.dtype != problem.alpha.dtype:
        raise ValueError("constraints and problem must use the same dtype")
    if problem.alpha.device.type != "cpu":
        raise ValueError("exact solver currently supports CPU tensors only")
    _require_cpu_problem(problem, constraints)


def _load_backend(backend: ExactBackend) -> tuple[object, str]:
    backend_module = importlib.import_module(backend.value)
    backend_version = _module_version(backend_module, backend.value)
    return backend_module, backend_version


def _tensor_evidence(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "sha256": hashlib.sha256(array.view(np.uint8)).hexdigest(),
    }


def _solver_settings_evidence(backend: ExactBackend) -> dict[str, Any]:
    if backend is ExactBackend.OSQP:
        return dict(OSQP_SOLVER_SETTINGS)
    return dict(CLARABEL_SOLVER_SETTINGS)


def configured_clarabel_settings(backend_module: object) -> object:
    settings = backend_module.DefaultSettings()
    for name, value in CLARABEL_SOLVER_SETTINGS.items():
        setattr(settings, name, value)
        if getattr(settings, name) != value:
            raise RuntimeError(f"clarabel setting {name!r} did not retain configured value")
    return settings


def _exact_template_evidence(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    backend: ExactBackend | str,
    *,
    backend_version: str | None = None,
) -> dict[str, Any]:
    normalized_backend = _normalize_backend(backend)
    if backend_version is None:
        backend_module, backend_version = _load_backend(normalized_backend)
        del backend_module
    return {
        "schema_version": EXACT_SOLVER_TEMPLATE_SCHEMA_VERSION,
        "risk": {
            "exposures": _tensor_evidence(problem.risk.exposures),
            "factor_variance": _tensor_evidence(problem.risk.factor_variance),
            "specific_variance": _tensor_evidence(problem.risk.specific_variance),
        },
        "bounds": {
            "lower": _tensor_evidence(problem.lower),
            "upper": _tensor_evidence(problem.upper),
        },
        "constraints": {
            "A_eq": _tensor_evidence(constraints.A_eq),
            "b_eq": _tensor_evidence(constraints.b_eq),
            "A_ineq": _tensor_evidence(constraints.A_ineq),
            "lower": _tensor_evidence(constraints.lower),
            "upper": _tensor_evidence(constraints.upper),
        },
        "scalars": {
            "assets": problem.risk.assets,
            "factors": int(problem.risk.factor_variance.shape[0]),
            "budget": float(problem.budget),
            "risk_aversion": float(problem.risk_aversion),
            "trade_cost": float(problem.trade_cost),
        },
        "backend": normalized_backend.value,
        "backend_version": backend_version,
        "solver_settings": _solver_settings_evidence(normalized_backend),
        "tolerance": EXACT_SOLVER_TOLERANCE,
        "dynamic_excluded": ["alpha", "current"],
    }


def _template_sha256_from_evidence(evidence: dict[str, Any]) -> str:
    payload = _canonical_template_evidence_json(evidence)
    if payload is None:
        raise RuntimeError("template evidence is missing")
    payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_template_sha256(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    backend: ExactBackend | str,
) -> str:
    _validate_problem_constraints(problem, constraints)
    alpha, current = _align_batch(problem)
    if alpha.device.type != "cpu":
        raise ValueError("exact solver currently supports CPU tensors only")
    if current is not None and current.shape[0] != alpha.shape[0]:
        raise ValueError("alpha and current batch dimensions are incompatible")
    return _template_sha256_from_evidence(
        _exact_template_evidence(problem, constraints, backend)
    )


def _build_qp(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
) -> tuple[
    sparse.csc_matrix,
    sparse.csc_matrix,
    np.ndarray,
    np.ndarray,
    sparse.csc_matrix,
    np.ndarray,
    tuple[tuple[str, int], ...],
    int,
]:
    assets = problem.risk.assets
    factors = problem.risk.factor_variance.shape[0]
    whitened_exposures = (
        problem.risk.exposures
        * torch.sqrt(problem.risk.factor_variance).unsqueeze(0)
        if factors
        else problem.risk.exposures.new_zeros((assets, 0))
    )

    diagonal = torch.cat(
        [
            problem.risk_aversion * problem.risk.specific_variance
            + problem.trade_cost * torch.ones_like(problem.risk.specific_variance),
            problem.risk_aversion * torch.ones(factors, dtype=problem.alpha.dtype, device=problem.alpha.device),
        ]
    )
    p = sparse.diags(
        diagonal.detach().cpu().numpy().astype(np.float64, copy=False),
        format="csc",
    )

    variable_count = assets + factors
    asset_identity = sparse.eye(assets, format="csc", dtype=np.float64)
    factor_identity = sparse.eye(factors, format="csc", dtype=np.float64) if factors else sparse.csc_matrix((0, 0), dtype=np.float64)
    zero_assets_factors = sparse.csc_matrix((assets, factors), dtype=np.float64)

    osqp_rows: list[sparse.csc_matrix] = []
    osqp_lower_bounds: list[np.ndarray] = []
    osqp_upper_bounds: list[np.ndarray] = []
    clarabel_rows: list[sparse.csc_matrix] = []
    clarabel_b: list[np.ndarray] = []
    clarabel_cones: list[tuple[str, int]] = []

    budget_row = sparse.hstack(
        [
            sparse.csc_matrix(np.ones((1, assets), dtype=np.float64)),
            sparse.csc_matrix((1, factors), dtype=np.float64),
        ],
        format="csc",
    )
    osqp_rows.append(budget_row)
    osqp_lower_bounds.append(np.array([problem.budget], dtype=np.float64))
    osqp_upper_bounds.append(np.array([problem.budget], dtype=np.float64))
    clarabel_rows.append(budget_row)
    clarabel_b.append(np.array([problem.budget], dtype=np.float64))
    clarabel_cones.append(("zero", 1))

    upper_box_row = sparse.hstack([asset_identity, zero_assets_factors], format="csc")
    osqp_rows.append(upper_box_row)
    osqp_lower_bounds.append(problem.lower.detach().cpu().numpy().astype(np.float64, copy=False))
    osqp_upper_bounds.append(problem.upper.detach().cpu().numpy().astype(np.float64, copy=False))
    clarabel_rows.append(upper_box_row)
    clarabel_b.append(problem.upper.detach().cpu().numpy().astype(np.float64, copy=False))
    clarabel_cones.append(("nonnegative", assets))

    lower_box_row = sparse.hstack([-asset_identity, zero_assets_factors], format="csc")
    osqp_rows.append(lower_box_row)
    osqp_lower_bounds.append((-problem.upper).detach().cpu().numpy().astype(np.float64, copy=False))
    osqp_upper_bounds.append((-problem.lower).detach().cpu().numpy().astype(np.float64, copy=False))
    clarabel_rows.append(lower_box_row)
    clarabel_b.append((-problem.lower).detach().cpu().numpy().astype(np.float64, copy=False))
    clarabel_cones.append(("nonnegative", assets))

    factor_row = sparse.hstack(
        [
            -sparse.csc_matrix(whitened_exposures.T.detach().cpu().numpy().astype(np.float64, copy=False)),
            factor_identity,
        ],
        format="csc",
    )
    if factors > 0:
        osqp_rows.append(factor_row)
        osqp_lower_bounds.append(np.zeros(factors, dtype=np.float64))
        osqp_upper_bounds.append(np.zeros(factors, dtype=np.float64))
        clarabel_rows.append(factor_row)
        clarabel_b.append(np.zeros(factors, dtype=np.float64))
        clarabel_cones.append(("zero", factors))

    if constraints.A_eq is not None and constraints.A_eq.shape[0] > 0:
        eq_row = sparse.hstack(
            [
                sparse.csc_matrix(constraints.A_eq.detach().cpu().numpy().astype(np.float64, copy=False)),
                sparse.csc_matrix((constraints.A_eq.shape[0], factors), dtype=np.float64),
            ],
            format="csc",
        )
        osqp_rows.append(eq_row)
        eq_values = constraints.b_eq.detach().cpu().numpy().astype(np.float64, copy=False)
        osqp_lower_bounds.append(eq_values)
        osqp_upper_bounds.append(eq_values)
        clarabel_rows.append(eq_row)
        clarabel_b.append(eq_values)
        clarabel_cones.append(("zero", constraints.A_eq.shape[0]))

    if constraints.A_ineq is not None and constraints.A_ineq.shape[0] > 0:
        ineq_matrix = constraints.A_ineq.detach().cpu().numpy().astype(np.float64, copy=False)
        upper_ineq_row = sparse.hstack(
            [
                sparse.csc_matrix(ineq_matrix),
                sparse.csc_matrix((constraints.A_ineq.shape[0], factors), dtype=np.float64),
            ],
            format="csc",
        )
        osqp_rows.append(upper_ineq_row)
        osqp_lower_bounds.append(constraints.lower.detach().cpu().numpy().astype(np.float64, copy=False))
        osqp_upper_bounds.append(constraints.upper.detach().cpu().numpy().astype(np.float64, copy=False))
        clarabel_rows.append(upper_ineq_row)
        clarabel_b.append(constraints.upper.detach().cpu().numpy().astype(np.float64, copy=False))
        clarabel_cones.append(("nonnegative", constraints.A_ineq.shape[0]))

        lower_ineq_row = sparse.hstack(
            [
                -sparse.csc_matrix(ineq_matrix),
                sparse.csc_matrix((constraints.A_ineq.shape[0], factors), dtype=np.float64),
            ],
            format="csc",
        )
        osqp_rows.append(lower_ineq_row)
        osqp_lower_bounds.append((-constraints.upper).detach().cpu().numpy().astype(np.float64, copy=False))
        osqp_upper_bounds.append((-constraints.lower).detach().cpu().numpy().astype(np.float64, copy=False))
        clarabel_rows.append(lower_ineq_row)
        clarabel_b.append((-constraints.lower).detach().cpu().numpy().astype(np.float64, copy=False))
        clarabel_cones.append(("nonnegative", constraints.A_ineq.shape[0]))

    osqp_a = sparse.vstack(osqp_rows, format="csc")
    osqp_l = np.concatenate(osqp_lower_bounds)
    osqp_u = np.concatenate(osqp_upper_bounds)
    clarabel_a = sparse.vstack(clarabel_rows, format="csc")
    clarabel_b_array = np.concatenate(clarabel_b)
    return p, osqp_a, osqp_l, osqp_u, clarabel_a, clarabel_b_array, tuple(clarabel_cones), variable_count


def _q_vector(
    problem: PortfolioProblem,
    alpha: torch.Tensor,
    current: torch.Tensor | None,
) -> np.ndarray:
    assets = problem.risk.assets
    factors = problem.risk.factor_variance.shape[0]
    q = (-alpha - (problem.trade_cost * current if current is not None else 0)).detach().cpu().numpy().astype(np.float64, copy=False)
    return np.concatenate([q, np.zeros(factors, dtype=np.float64)])


def _objective(
    problem: PortfolioProblem,
    alpha: torch.Tensor,
    current: torch.Tensor | None,
    weights: torch.Tensor,
) -> torch.Tensor:
    risk = 0.5 * problem.risk_aversion * (
        weights * problem.risk.matvec(weights)
    ).sum(dim=1)
    utility = (alpha * weights).sum(dim=1)
    if current is None or problem.trade_cost == 0:
        trade = torch.zeros_like(risk)
    else:
        trade = 0.5 * problem.trade_cost * ((weights - current) ** 2).sum(dim=1)
    return risk - utility + trade


def _status_text(result: object) -> str:
    status = getattr(result, "status", None)
    if status is None:
        info = getattr(result, "info", None)
        status = getattr(info, "status", None)
    if status is None:
        raise RuntimeError("backend returned a result without a status")
    return str(status)


def _is_success(backend: ExactBackend, status: str) -> bool:
    normalized = status.lower()
    if backend is ExactBackend.OSQP:
        return normalized in {"solved", "solved inaccurate"}
    return normalized in {"solved", "almost solved"}


def _clarabel_cones_from_specs(
    backend_module: object,
    cone_specs: tuple[tuple[str, int], ...],
) -> tuple[object, ...]:
    cones = []
    for kind, size in cone_specs:
        if kind == "zero":
            cones.append(backend_module.ZeroConeT(size))
        elif kind == "nonnegative":
            cones.append(backend_module.NonnegativeConeT(size))
        else:
            raise RuntimeError(f"unknown Clarabel cone kind: {kind!r}")
    return tuple(cones)


def _setup_solver(
    backend: ExactBackend,
    backend_module: object,
    qp_data: tuple[
        sparse.csc_matrix,
        sparse.csc_matrix,
        np.ndarray,
        np.ndarray,
        sparse.csc_matrix,
        np.ndarray,
        tuple[tuple[str, int], ...],
        int,
    ],
    q_template: np.ndarray,
) -> tuple[object, int]:
    (
        p,
        osqp_a,
        osqp_l,
        osqp_u,
        clarabel_a,
        clarabel_b,
        clarabel_cone_specs,
        variable_count,
    ) = qp_data
    if backend is ExactBackend.OSQP:
        solver = backend_module.OSQP()
        solver.setup(
            P=p,
            q=q_template,
            A=osqp_a,
            l=osqp_l,
            u=osqp_u,
            **OSQP_SOLVER_SETTINGS,
        )
        return solver, variable_count

    settings = configured_clarabel_settings(backend_module)
    solver = backend_module.DefaultSolver(
        p,
        q_template,
        clarabel_a,
        clarabel_b,
        _clarabel_cones_from_specs(backend_module, clarabel_cone_specs),
        settings,
    )
    return solver, variable_count


def _require_prepared_update_api(backend: ExactBackend, solver: object) -> None:
    update = getattr(solver, "update", None)
    if not callable(update):
        raise RuntimeError(f"{backend.value} prepared exact workspace does not support q updates")
    if backend is not ExactBackend.CLARABEL:
        return
    allowed_probe = getattr(solver, "is_data_update_allowed", None)
    if allowed_probe is None:
        return
    try:
        allowed = allowed_probe() if callable(allowed_probe) else bool(allowed_probe)
    except Exception as error:
        raise RuntimeError("clarabel prepared exact workspace data-update probe failed") from error
    if not allowed:
        raise RuntimeError("clarabel prepared exact workspace does not allow data updates")


def _workspace_lock(workspace: PreparedExactWorkspace) -> threading.Lock:
    lock = workspace.lock
    if lock is None:
        raise RuntimeError("prepared exact workspace lock is missing")
    return lock


def _poison_workspace_locked(workspace: PreparedExactWorkspace, reason: str) -> None:
    if not workspace.poisoned:
        object.__setattr__(workspace, "poisoned", True)
        object.__setattr__(workspace, "poison_reason", reason)


def _raise_if_poisoned_locked(workspace: PreparedExactWorkspace) -> None:
    if workspace.poisoned:
        reason = workspace.poison_reason or "unknown backend failure"
        raise RuntimeError(f"prepared exact workspace is poisoned after backend failure: {reason}")


def _poison_and_raise_locked(
    workspace: PreparedExactWorkspace | None,
    message: str,
    error: Exception | None = None,
) -> None:
    if workspace is not None:
        _poison_workspace_locked(workspace, message)
    if error is None:
        raise RuntimeError(message)
    raise RuntimeError(message) from error


def _solve_backend_batch(
    *,
    solver: object,
    normalized_backend: ExactBackend,
    problem: PortfolioProblem,
    alpha: torch.Tensor,
    current: torch.Tensor | None,
    variable_count: int,
    update_first: bool,
    workspace: PreparedExactWorkspace | None = None,
) -> tuple[list[torch.Tensor], list[str], float, float, float]:
    weights_rows: list[torch.Tensor] = []
    raw_statuses: list[str] = []
    cold_seconds = 0.0
    update_seconds = 0.0
    solve_seconds = 0.0
    batch_size = alpha.shape[0]
    for index in range(batch_size):
        q_template = _q_vector(
            problem,
            alpha[index],
            None if current is None else current[index],
        )
        if update_first or index > 0:
            update_started = time.perf_counter()
            try:
                solver.update(q=np.ascontiguousarray(q_template))
            except Exception as error:
                _poison_and_raise_locked(
                    workspace,
                    f"{normalized_backend.value} update failed at batch index {index}",
                    error,
                )
            update_seconds += time.perf_counter() - update_started

        solve_started = time.perf_counter()
        try:
            if normalized_backend is ExactBackend.OSQP:
                result = solver.solve(raise_error=True)
            else:
                result = solver.solve()
        except Exception as error:
            _poison_and_raise_locked(
                workspace,
                f"{normalized_backend.value} solve failed at batch index {index}",
                error,
            )
        elapsed = time.perf_counter() - solve_started
        solve_seconds += elapsed
        if index == 0:
            cold_seconds = elapsed

        try:
            status = _status_text(result)
        except Exception as error:
            _poison_and_raise_locked(
                workspace,
                f"{normalized_backend.value} returned invalid status at batch index {index}",
                error,
            )
        raw_statuses.append(status)
        if not _is_success(normalized_backend, status):
            _poison_and_raise_locked(
                workspace,
                f"{normalized_backend.value} solve failed with status {status!r}",
            )
        solution = getattr(result, "x", None)
        if solution is None:
            _poison_and_raise_locked(
                workspace,
                f"{normalized_backend.value} returned no primal solution",
            )
        solution_array = np.asarray(solution, dtype=np.float64)
        if solution_array.shape[0] != variable_count:
            _poison_and_raise_locked(
                workspace,
                f"{normalized_backend.value} returned solution with unexpected dimension "
                f"{solution_array.shape[0]} != {variable_count}",
            )
        weights_rows.append(
            torch.as_tensor(
                solution_array[: problem.risk.assets],
                dtype=problem.alpha.dtype,
                device="cpu",
            )
        )
    return weights_rows, raw_statuses, cold_seconds, update_seconds, solve_seconds


def prepare_exact_workspace(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    backend: ExactBackend | str,
) -> PreparedExactWorkspace:
    normalized_backend = _normalize_backend(backend)
    _validate_problem_constraints(problem, constraints)

    alpha, current = _align_batch(problem)
    if alpha.device.type != "cpu":
        raise ValueError("exact solver currently supports CPU tensors only")
    if current is not None and current.shape[0] != alpha.shape[0]:
        raise ValueError("alpha and current batch dimensions are incompatible")

    backend_module, backend_version = _load_backend(normalized_backend)
    template_evidence = _exact_template_evidence(
        problem,
        constraints,
        normalized_backend,
        backend_version=backend_version,
    )
    template_sha256 = _template_sha256_from_evidence(template_evidence)

    q_template = np.ascontiguousarray(
        _q_vector(problem, alpha[0], None if current is None else current[0])
    )
    build_started = time.perf_counter()
    qp_data = _build_qp(problem, constraints)
    build_seconds = time.perf_counter() - build_started
    setup_started = time.perf_counter()
    solver, variable_count = _setup_solver(
        normalized_backend,
        backend_module,
        qp_data,
        q_template,
    )
    setup_seconds = time.perf_counter() - setup_started
    _require_prepared_update_api(normalized_backend, solver)
    return PreparedExactWorkspace(
        solver=solver,
        backend=normalized_backend.value,
        backend_version=backend_version,
        template_sha256=template_sha256,
        template_evidence=template_evidence,
        setup_seconds=setup_seconds,
        variable_count=variable_count,
        assets=problem.risk.assets,
        dtype=problem.alpha.dtype,
        build_seconds=build_seconds,
    )


def _validate_workspace(
    workspace: PreparedExactWorkspace,
    problem: PortfolioProblem,
    constraints: LinearConstraints,
) -> ExactBackend:
    normalized_backend = _normalize_backend(workspace.backend)
    _validate_problem_constraints(problem, constraints)
    if workspace.device != "cpu":
        raise ValueError(f"prepared exact workspace device drifted: {workspace.device!r}")
    if problem.risk.assets != workspace.assets:
        raise ValueError(
            f"prepared exact workspace asset shape drifted: "
            f"{problem.risk.assets} != {workspace.assets}"
        )
    if problem.alpha.dtype != workspace.dtype:
        raise ValueError(
            f"prepared exact workspace dtype drifted: {problem.alpha.dtype} != {workspace.dtype}"
        )
    current_evidence = _exact_template_evidence(problem, constraints, normalized_backend)
    current_sha256 = _template_sha256_from_evidence(current_evidence)
    if current_sha256 != workspace.template_sha256:
        raise RuntimeError(
            "prepared exact workspace is stale: static template/backend/tolerance/version "
            f"drifted from {workspace.template_sha256} to {current_sha256}"
        )
    return normalized_backend


def solve_exact_prepared(
    workspace: PreparedExactWorkspace,
    problem: PortfolioProblem,
    constraints: LinearConstraints,
) -> ExactSolveResult:
    lock = _workspace_lock(workspace)
    with lock:
        _raise_if_poisoned_locked(workspace)
    normalized_backend = _validate_workspace(workspace, problem, constraints)
    alpha, current = _align_batch(problem)
    if current is not None and current.shape[0] != alpha.shape[0]:
        raise ValueError("alpha and current batch dimensions are incompatible")
    elapsed_started = time.perf_counter()
    with lock:
        _raise_if_poisoned_locked(workspace)
        (
            weights_rows,
            raw_statuses,
            cold_seconds,
            update_seconds,
            solve_seconds,
        ) = _solve_backend_batch(
            solver=workspace.solver,
            normalized_backend=normalized_backend,
            problem=problem,
            alpha=alpha,
            current=current,
            variable_count=workspace.variable_count,
            update_first=True,
            workspace=workspace,
        )
    elapsed_seconds = time.perf_counter() - elapsed_started
    weights = torch.stack(weights_rows, dim=0)
    objective = _objective(problem, alpha, current, weights)
    return ExactSolveResult(
        weights=weights,
        objective=objective,
        raw_statuses=tuple(raw_statuses),
        setup_seconds=0.0,
        cold_seconds=cold_seconds,
        elapsed_seconds=elapsed_seconds,
        backend=normalized_backend.value,
        backend_version=workspace.backend_version,
        build_seconds=0.0,
        update_seconds=update_seconds,
        solve_seconds=solve_seconds,
        prepared=True,
        template_sha256=workspace.template_sha256,
        template_evidence=workspace.template_evidence,
        timing_evidence={
            "build_seconds": 0.0,
            "setup_seconds": 0.0,
            "update_seconds": update_seconds,
            "solve_seconds": solve_seconds,
            "elapsed_seconds": elapsed_seconds,
            "workspace_build_seconds": workspace.build_seconds,
            "workspace_setup_seconds": workspace.setup_seconds,
        },
    )


def solve_exact(
    problem: PortfolioProblem,
    constraints: LinearConstraints,
    backend: ExactBackend | str,
) -> ExactSolveResult:
    elapsed_started = time.perf_counter()
    normalized_backend = _normalize_backend(backend)
    _validate_problem_constraints(problem, constraints)
    alpha, current = _align_batch(problem)
    if alpha.device.type != "cpu":
        raise ValueError("exact solver currently supports CPU tensors only")
    if current is not None and current.shape[0] != alpha.shape[0]:
        raise ValueError("alpha and current batch dimensions are incompatible")

    backend_module, backend_version = _load_backend(normalized_backend)
    template_evidence = _exact_template_evidence(
        problem,
        constraints,
        normalized_backend,
        backend_version=backend_version,
    )
    template_sha256 = _template_sha256_from_evidence(template_evidence)
    q_template = np.ascontiguousarray(
        _q_vector(problem, alpha[0], None if current is None else current[0])
    )

    build_started = time.perf_counter()
    qp_data = _build_qp(problem, constraints)
    build_seconds = time.perf_counter() - build_started
    setup_started = time.perf_counter()
    solver, variable_count = _setup_solver(
        normalized_backend,
        backend_module,
        qp_data,
        q_template,
    )
    setup_seconds = time.perf_counter() - setup_started
    (
        weights_rows,
        raw_statuses,
        cold_seconds,
        update_seconds,
        solve_seconds,
    ) = _solve_backend_batch(
        solver=solver,
        normalized_backend=normalized_backend,
        problem=problem,
        alpha=alpha,
        current=current,
        variable_count=variable_count,
        update_first=False,
    )
    elapsed_seconds = time.perf_counter() - elapsed_started
    weights = torch.stack(weights_rows, dim=0)
    objective = _objective(problem, alpha, current, weights)
    return ExactSolveResult(
        weights=weights,
        objective=objective,
        raw_statuses=tuple(raw_statuses),
        setup_seconds=setup_seconds,
        cold_seconds=cold_seconds,
        elapsed_seconds=elapsed_seconds,
        backend=normalized_backend.value,
        backend_version=backend_version,
        build_seconds=build_seconds,
        update_seconds=update_seconds,
        solve_seconds=solve_seconds,
        prepared=False,
        template_sha256=template_sha256,
        template_evidence=template_evidence,
        timing_evidence={
            "build_seconds": build_seconds,
            "setup_seconds": setup_seconds,
            "update_seconds": update_seconds,
            "solve_seconds": solve_seconds,
            "elapsed_seconds": elapsed_seconds,
        },
    )
