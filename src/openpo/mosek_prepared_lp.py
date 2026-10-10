from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
import threading
import time

import numpy as np


class MosekUnavailableError(RuntimeError):
    """Raised when the native MOSEK Python API cannot be imported."""


class MosekBackendError(RuntimeError):
    """Raised when a native MOSEK operation fails."""


@dataclass(frozen=True)
class MosekPreparedLPResult:
    dw: np.ndarray
    absolute_dw: np.ndarray
    objective_value: float
    mosek_primal_objective: float
    solution_status: str
    solution_type: str
    backend_version: str
    build_seconds: float
    update_objective_seconds: float
    update_bounds_seconds: float
    update_seconds: float
    optimize_seconds: float
    solution_seconds: float
    elapsed_seconds: float
    workspace_reused: bool
    solve_index: int
    env_identity: int
    task_identity: int
    neutrality_residual: float
    bound_violation: float
    absolute_value_violation: float
    timing_evidence: dict[str, float | bool | int | str]


@dataclass
class MosekPreparedLPWorkspace:
    env: object
    task: object
    mosek: object
    assets: int
    backend_version: str
    optimizer: str
    build_seconds: float
    env_create_seconds: float
    task_create_seconds: float
    matrix_build_seconds: float
    variable_indices: tuple[int, ...]
    dw_indices: tuple[int, ...]
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    solve_count: int = 0
    closed: bool = False
    poisoned: bool = False
    poison_reason: str | None = None

    def solve(
        self,
        score: object,
        spread: object,
        sell: object,
        buy: object,
    ) -> MosekPreparedLPResult:
        return solve_mosek_prepared_lp(self, score, spread, sell, buy)

    def close(self) -> None:
        with self.lock:
            if self.closed:
                return
            errors: list[Exception] = []
            for resource in (self.task, self.env):
                disposer = getattr(resource, "dispose", None)
                if disposer is None:
                    disposer = getattr(resource, "close", None)
                if disposer is None:
                    continue
                try:
                    disposer()
                except Exception as exc:
                    errors.append(exc)
            self.closed = True
            if errors:
                raise MosekBackendError(
                    f"failed to close native MOSEK resources: {errors[0]}"
                ) from errors[0]

    def __enter__(self) -> MosekPreparedLPWorkspace:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


PreparedMosekLP = MosekPreparedLPWorkspace


def _load_mosek(mosek_module: object | None) -> object:
    if mosek_module is not None:
        return mosek_module
    try:
        return importlib.import_module("mosek")
    except ImportError as exc:
        raise MosekUnavailableError(
            "native MOSEK Python API is required for prepared LP solving; "
            "install MOSEK and configure a valid license"
        ) from exc


def _require_mosek_api(mosek: object) -> None:
    missing = [
        name
        for name in (
            "Env",
            "boundkey",
            "iparam",
            "objsense",
            "optimizertype",
            "solsta",
            "soltype",
        )
        if not hasattr(mosek, name)
    ]
    if missing:
        raise MosekBackendError(
            "injected MOSEK module is missing required API members: "
            + ", ".join(missing)
        )


def _backend_version(mosek: object, env: object) -> str:
    version = getattr(mosek, "__version__", None)
    if version is not None:
        return str(version)
    getter = getattr(env, "getversion", None)
    if getter is None:
        getter = getattr(getattr(mosek, "Env", object), "getversion", None)
    if getter is None:
        return "unknown"
    try:
        value = getter()
    except Exception:
        return "unknown"
    if isinstance(value, tuple):
        return ".".join(str(part) for part in value)
    return str(value)


def _positive_assets(assets: int) -> int:
    if isinstance(assets, bool) or not isinstance(assets, (int, np.integer)):
        raise TypeError("assets must be a positive integer")
    normalized = int(assets)
    if normalized <= 0:
        raise ValueError("assets must be a positive integer")
    return normalized


def _as_vector(name: str, value: object, assets: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a float64 vector") from exc
    if name == "spread" and array.ndim == 0:
        array = np.full(assets, float(array), dtype=np.float64)
    if array.ndim != 1 or array.shape[0] != assets:
        raise ValueError(f"{name} must have shape ({assets},), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.ascontiguousarray(array)


def _validated_inputs(
    assets: int,
    score: object,
    spread: object,
    sell: object,
    buy: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    score_array = _as_vector("score", score, assets)
    spread_array = _as_vector("spread", spread, assets)
    sell_array = _as_vector("sell", sell, assets)
    buy_array = _as_vector("buy", buy, assets)
    for name, array in (
        ("spread", spread_array),
        ("sell", sell_array),
        ("buy", buy_array),
    ):
        if np.any(array < 0.0):
            raise ValueError(f"{name} must be nonnegative")
    return score_array, spread_array, sell_array, buy_array


def _static_matrix(assets: int) -> tuple[list[int], list[int], list[float]]:
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    positive_rows = 1
    negative_rows = 1 + assets
    absolute_offset = assets
    for asset in range(assets):
        row_indices.extend((0, positive_rows + asset, negative_rows + asset))
        column_indices.extend((asset, asset, asset))
        values.extend((1.0, 1.0, -1.0))
        row_indices.extend((positive_rows + asset, negative_rows + asset))
        column_indices.extend((absolute_offset + asset, absolute_offset + asset))
        values.extend((-1.0, -1.0))
    return row_indices, column_indices, values


def prepare_mosek_prepared_lp(
    assets: int,
    *,
    mosek_module: object | None = None,
    task_name: str = "openpo-prepared-trade-lp",
) -> MosekPreparedLPWorkspace:
    normalized_assets = _positive_assets(assets)
    mosek = _load_mosek(mosek_module)
    _require_mosek_api(mosek)
    build_start = time.perf_counter()
    env: object | None = None
    task: object | None = None
    try:
        env_start = time.perf_counter()
        env = mosek.Env()
        env_create_seconds = time.perf_counter() - env_start

        task_start = time.perf_counter()
        task = env.Task(0, 0)
        task_create_seconds = time.perf_counter() - task_start

        matrix_start = time.perf_counter()
        variable_count = 2 * normalized_assets
        constraint_count = 1 + 2 * normalized_assets
        variable_indices = tuple(range(variable_count))
        dw_indices = tuple(range(normalized_assets))
        absolute_indices = tuple(range(normalized_assets, variable_count))
        constraint_indices = tuple(range(constraint_count))

        if task_name:
            task.puttaskname(task_name)
        task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.primal_simplex)
        task.appendvars(variable_count)
        task.appendcons(constraint_count)
        task.putobjsense(mosek.objsense.maximize)
        task.putclist(variable_indices, [0.0] * variable_count)
        task.putvarboundlist(
            dw_indices,
            [mosek.boundkey.fr] * normalized_assets,
            [0.0] * normalized_assets,
            [0.0] * normalized_assets,
        )
        task.putvarboundlist(
            absolute_indices,
            [mosek.boundkey.lo] * normalized_assets,
            [0.0] * normalized_assets,
            [0.0] * normalized_assets,
        )

        constraint_keys = [mosek.boundkey.fx]
        constraint_keys.extend([mosek.boundkey.up] * (2 * normalized_assets))
        task.putconboundlist(
            constraint_indices,
            constraint_keys,
            [0.0] * constraint_count,
            [0.0] * constraint_count,
        )
        row_indices, column_indices, values = _static_matrix(normalized_assets)
        task.putaijlist(row_indices, column_indices, values)
        matrix_build_seconds = time.perf_counter() - matrix_start
    except Exception as exc:
        for resource in (task, env):
            disposer = getattr(resource, "dispose", None) if resource is not None else None
            if disposer is not None:
                try:
                    disposer()
                except Exception:
                    pass
        raise MosekBackendError(f"failed to build native MOSEK prepared LP: {exc}") from exc

    build_seconds = time.perf_counter() - build_start
    return MosekPreparedLPWorkspace(
        env=env,
        task=task,
        mosek=mosek,
        assets=normalized_assets,
        backend_version=_backend_version(mosek, env),
        optimizer="primal_simplex",
        build_seconds=build_seconds,
        env_create_seconds=env_create_seconds,
        task_create_seconds=task_create_seconds,
        matrix_build_seconds=matrix_build_seconds,
        variable_indices=variable_indices,
        dw_indices=dw_indices,
    )


def prepare_mosek_lp(
    assets: int,
    *,
    mosek_module: object | None = None,
    task_name: str = "openpo-prepared-trade-lp",
) -> MosekPreparedLPWorkspace:
    return prepare_mosek_prepared_lp(
        assets,
        mosek_module=mosek_module,
        task_name=task_name,
    )


def _accepted_solution_statuses(mosek: object) -> tuple[object, ...]:
    return (mosek.solsta.optimal,)


def _solution_types(mosek: object) -> tuple[object, ...]:
    candidates: list[object] = []
    for name in ("bas", "itr"):
        value = getattr(mosek.soltype, name, None)
        if value is not None and value not in candidates:
            candidates.append(value)
    if not candidates:
        raise MosekBackendError("MOSEK module exposes neither bas nor itr solution type")
    return tuple(candidates)


def _extract_solution(
    workspace: MosekPreparedLPWorkspace,
) -> tuple[np.ndarray, float, object, object]:
    accepted = _accepted_solution_statuses(workspace.mosek)
    observed: list[str] = []
    last_error: Exception | None = None
    for solution_type in _solution_types(workspace.mosek):
        try:
            status = workspace.task.getsolsta(solution_type)
            observed.append(f"{solution_type}={status}")
            if status not in accepted:
                continue
            values = [0.0] * (2 * workspace.assets)
            workspace.task.getxx(solution_type, values)
            primal_objective = float(workspace.task.getprimalobj(solution_type))
            solution = np.asarray(values, dtype=np.float64)
            if solution.shape != (2 * workspace.assets,) or not np.all(np.isfinite(solution)):
                raise MosekBackendError("MOSEK returned a nonfinite or malformed solution")
            if not math.isfinite(primal_objective):
                raise MosekBackendError("MOSEK returned a nonfinite primal objective")
            return solution, primal_objective, status, solution_type
        except MosekBackendError:
            raise
        except Exception as exc:
            last_error = exc
    detail = ", ".join(observed) if observed else "no solution status available"
    error = MosekBackendError(f"MOSEK LP did not return an optimal solution: {detail}")
    if last_error is not None:
        raise error from last_error
    raise error


def _poison(workspace: MosekPreparedLPWorkspace, phase: str, exc: Exception) -> None:
    workspace.poisoned = True
    workspace.poison_reason = f"{phase}: {exc}"


def solve_mosek_prepared_lp(
    workspace: MosekPreparedLPWorkspace,
    score: object,
    spread: object,
    sell: object,
    buy: object,
) -> MosekPreparedLPResult:
    if not isinstance(workspace, MosekPreparedLPWorkspace):
        raise TypeError("workspace must be a MosekPreparedLPWorkspace")
    score_array, spread_array, sell_array, buy_array = _validated_inputs(
        workspace.assets,
        score,
        spread,
        sell,
        buy,
    )
    with workspace.lock:
        if workspace.closed:
            raise RuntimeError("prepared MOSEK LP workspace is closed")
        if workspace.poisoned:
            raise RuntimeError(
                "prepared MOSEK LP workspace is poisoned after backend failure: "
                f"{workspace.poison_reason or 'unknown failure'}"
            )

        solve_index = workspace.solve_count + 1
        workspace_reused = workspace.solve_count > 0
        elapsed_start = time.perf_counter()
        objective_values = np.concatenate((score_array, -0.5 * spread_array))

        try:
            objective_start = time.perf_counter()
            workspace.task.putclist(
                workspace.variable_indices,
                objective_values.tolist(),
            )
            update_objective_seconds = time.perf_counter() - objective_start

            bounds_start = time.perf_counter()
            workspace.task.putvarboundlist(
                workspace.dw_indices,
                [workspace.mosek.boundkey.ra] * workspace.assets,
                (-sell_array).tolist(),
                buy_array.tolist(),
            )
            update_bounds_seconds = time.perf_counter() - bounds_start
            update_seconds = update_objective_seconds + update_bounds_seconds

            optimize_start = time.perf_counter()
            workspace.task.optimize()
            optimize_seconds = time.perf_counter() - optimize_start

            solution_start = time.perf_counter()
            solution, primal_objective, status, solution_type = _extract_solution(workspace)
            solution_seconds = time.perf_counter() - solution_start
        except Exception as exc:
            _poison(workspace, "update/optimize/extract", exc)
            if isinstance(exc, MosekBackendError):
                raise
            raise MosekBackendError(f"native MOSEK prepared LP solve failed: {exc}") from exc

        dw = solution[: workspace.assets].copy()
        absolute_dw = solution[workspace.assets :].copy()
        objective_value = float(score_array @ dw - 0.5 * spread_array @ np.abs(dw))
        neutrality_residual = float(abs(np.sum(dw)))
        lower_violation = np.maximum(-sell_array - dw, 0.0)
        upper_violation = np.maximum(dw - buy_array, 0.0)
        bound_violation = float(max(np.max(lower_violation), np.max(upper_violation)))
        absolute_value_violation = float(np.max(np.maximum(np.abs(dw) - absolute_dw, 0.0)))
        elapsed_seconds = time.perf_counter() - elapsed_start
        workspace.solve_count = solve_index
        timing_evidence: dict[str, float | bool | int | str] = {
            "build_seconds": workspace.build_seconds,
            "env_create_seconds": workspace.env_create_seconds,
            "task_create_seconds": workspace.task_create_seconds,
            "matrix_build_seconds": workspace.matrix_build_seconds,
            "update_objective_seconds": update_objective_seconds,
            "update_bounds_seconds": update_bounds_seconds,
            "update_seconds": update_seconds,
            "optimize_seconds": optimize_seconds,
            "solution_seconds": solution_seconds,
            "elapsed_seconds": elapsed_seconds,
            "workspace_reused": workspace_reused,
            "solve_index": solve_index,
            "env_identity": id(workspace.env),
            "task_identity": id(workspace.task),
            "optimizer": workspace.optimizer,
        }
        return MosekPreparedLPResult(
            dw=dw,
            absolute_dw=absolute_dw,
            objective_value=objective_value,
            mosek_primal_objective=primal_objective,
            solution_status=str(status),
            solution_type=str(solution_type),
            backend_version=workspace.backend_version,
            build_seconds=workspace.build_seconds,
            update_objective_seconds=update_objective_seconds,
            update_bounds_seconds=update_bounds_seconds,
            update_seconds=update_seconds,
            optimize_seconds=optimize_seconds,
            solution_seconds=solution_seconds,
            elapsed_seconds=elapsed_seconds,
            workspace_reused=workspace_reused,
            solve_index=solve_index,
            env_identity=id(workspace.env),
            task_identity=id(workspace.task),
            neutrality_residual=neutrality_residual,
            bound_violation=bound_violation,
            absolute_value_violation=absolute_value_violation,
            timing_evidence=timing_evidence,
        )


solve_prepared_mosek_lp = solve_mosek_prepared_lp


__all__ = [
    "MosekBackendError",
    "MosekPreparedLPResult",
    "MosekPreparedLPWorkspace",
    "MosekUnavailableError",
    "PreparedMosekLP",
    "prepare_mosek_lp",
    "prepare_mosek_prepared_lp",
    "solve_mosek_prepared_lp",
    "solve_prepared_mosek_lp",
]
