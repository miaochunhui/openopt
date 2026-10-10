from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

import openpo.mosek_prepared_lp as prepared_module
from openpo.mosek_prepared_lp import (
    MosekBackendError,
    MosekPreparedLPWorkspace,
    MosekUnavailableError,
    prepare_mosek_prepared_lp,
    solve_mosek_prepared_lp,
)


class FakeTask:
    def __init__(self, module: FakeMosekModule) -> None:
        self.module = module
        self.name = ""
        self.variable_count = 0
        self.constraint_count = 0
        self.objective: list[float] = []
        self.variable_bounds: dict[int, tuple[str, float, float]] = {}
        self.constraint_bounds: dict[int, tuple[str, float, float]] = {}
        self.matrix: dict[tuple[int, int], float] = {}
        self.objective_sense: str | None = None
        self.integer_params: dict[str, str] = {}
        self.appendvars_calls = 0
        self.appendcons_calls = 0
        self.optimize_calls = 0
        self.solution: list[float] = []
        self.status = module.solsta.optimal
        self.disposed = False
        self.fail_optimize = False
        self.active_optimizes = 0
        self.max_active_optimizes = 0

    def puttaskname(self, name: str) -> None:
        self.name = name

    def appendvars(self, count: int) -> None:
        self.appendvars_calls += 1
        self.variable_count += count
        self.objective.extend([0.0] * count)

    def appendcons(self, count: int) -> None:
        self.appendcons_calls += 1
        self.constraint_count += count

    def putobjsense(self, sense: str) -> None:
        self.objective_sense = sense

    def putintparam(self, parameter: str, value: str) -> None:
        self.integer_params[parameter] = value

    def putclist(self, indices, values) -> None:
        for index, value in zip(indices, values, strict=True):
            self.objective[index] = float(value)

    def putvarboundlist(self, indices, keys, lower, upper) -> None:
        for index, key, lower_value, upper_value in zip(
            indices, keys, lower, upper, strict=True
        ):
            self.variable_bounds[index] = (
                key,
                float(lower_value),
                float(upper_value),
            )

    def putconboundlist(self, indices, keys, lower, upper) -> None:
        for index, key, lower_value, upper_value in zip(
            indices, keys, lower, upper, strict=True
        ):
            self.constraint_bounds[index] = (
                key,
                float(lower_value),
                float(upper_value),
            )

    def putaijlist(self, rows, columns, values) -> None:
        for row, column, value in zip(rows, columns, values, strict=True):
            self.matrix[(row, column)] = float(value)

    def optimize(self) -> None:
        if self.fail_optimize:
            raise RuntimeError("synthetic optimize failure")
        self.active_optimizes += 1
        self.max_active_optimizes = max(self.max_active_optimizes, self.active_optimizes)
        try:
            time.sleep(self.module.optimize_delay)
            self.optimize_calls += 1
            assets = self.variable_count // 2
            dw = np.zeros(assets, dtype=np.float64)
            if assets == 2:
                lower_0 = self.variable_bounds[0][1]
                upper_0 = self.variable_bounds[0][2]
                lower_1 = self.variable_bounds[1][1]
                upper_1 = self.variable_bounds[1][2]
                lower_t = max(lower_0, -upper_1)
                upper_t = min(upper_0, -lower_1)
                candidates = [lower_t, upper_t]
                if lower_t <= 0.0 <= upper_t:
                    candidates.append(0.0)

                def objective(t: float) -> float:
                    candidate = np.array([t, -t], dtype=np.float64)
                    absolute = np.abs(candidate)
                    values = np.concatenate((candidate, absolute))
                    return float(np.asarray(self.objective) @ values)

                best_t = max(candidates, key=objective)
                dw[:] = (best_t, -best_t)
            absolute = np.abs(dw)
            self.solution = np.concatenate((dw, absolute)).tolist()
        finally:
            self.active_optimizes -= 1

    def getsolsta(self, solution_type: str) -> str:
        if solution_type == self.module.soltype.itr:
            raise RuntimeError("no interior solution in fake task")
        return self.status

    def getxx(self, solution_type: str, target: list[float]) -> None:
        target[:] = self.solution

    def getprimalobj(self, solution_type: str) -> float:
        return float(np.asarray(self.objective) @ np.asarray(self.solution))

    def dispose(self) -> None:
        self.disposed = True


class FakeEnv:
    def __init__(self, module: FakeMosekModule) -> None:
        self.module = module
        self.tasks: list[FakeTask] = []
        self.disposed = False

    def Task(self, constraints: int, variables: int) -> FakeTask:
        assert constraints == 0
        assert variables == 0
        task = FakeTask(self.module)
        self.tasks.append(task)
        return task

    def getversion(self) -> tuple[int, int, int]:
        return (11, 0, 7)

    def dispose(self) -> None:
        self.disposed = True


class FakeMosekModule:
    __version__ = "11.0.7-fake"
    boundkey = SimpleNamespace(fr="fr", lo="lo", up="up", fx="fx", ra="ra")
    objsense = SimpleNamespace(maximize="maximize")
    iparam = SimpleNamespace(optimizer="optimizer")
    optimizertype = SimpleNamespace(primal_simplex="primal_simplex")
    solsta = SimpleNamespace(optimal="optimal", near_optimal="near_optimal")
    soltype = SimpleNamespace(bas="bas", itr="itr")

    def __init__(self, *, optimize_delay: float = 0.0) -> None:
        self.envs: list[FakeEnv] = []
        self.optimize_delay = optimize_delay

    def Env(self) -> FakeEnv:
        env = FakeEnv(self)
        self.envs.append(env)
        return env


def test_builds_native_lp_matrix_once_for_6000_assets() -> None:
    fake = FakeMosekModule()
    workspace = prepare_mosek_prepared_lp(6000, mosek_module=fake)
    task = workspace.task

    assert isinstance(workspace, MosekPreparedLPWorkspace)
    assert workspace.assets == 6000
    assert len(fake.envs) == 1
    assert len(fake.envs[0].tasks) == 1
    assert task.variable_count == 12000
    assert task.constraint_count == 12001
    assert task.appendvars_calls == 1
    assert task.appendcons_calls == 1
    assert len(task.matrix) == 30000
    assert task.matrix[(0, 0)] == 1.0
    assert task.matrix[(1, 0)] == 1.0
    assert task.matrix[(6001, 0)] == -1.0
    assert task.matrix[(1, 6000)] == -1.0
    assert task.matrix[(6001, 6000)] == -1.0
    assert task.constraint_bounds[0] == ("fx", 0.0, 0.0)
    assert task.integer_params == {"optimizer": "primal_simplex"}
    assert workspace.optimizer == "primal_simplex"
    assert task.constraint_bounds[1] == ("up", 0.0, 0.0)
    assert task.objective_sense == "maximize"
    assert workspace.build_seconds >= 0.0
    assert workspace.matrix_build_seconds >= 0.0


def test_updates_objective_and_bounds_and_reuses_task_for_hot_optimize() -> None:
    fake = FakeMosekModule()
    workspace = prepare_mosek_prepared_lp(2, mosek_module=fake)
    task = workspace.task

    first = solve_mosek_prepared_lp(
        workspace,
        score=np.array([3.0, 1.0]),
        spread=np.array([0.2, 0.2]),
        sell=np.array([0.4, 0.5]),
        buy=np.array([0.3, 0.6]),
    )
    second = workspace.solve(
        score=np.array([0.0, 4.0]),
        spread=np.array([0.4, 0.2]),
        sell=np.array([0.2, 0.7]),
        buy=np.array([0.8, 0.25]),
    )

    np.testing.assert_allclose(first.dw, np.array([0.3, -0.3]))
    np.testing.assert_allclose(second.dw, np.array([-0.2, 0.2]))
    assert first.objective_value == pytest.approx(0.54)
    assert second.objective_value == pytest.approx(0.74)
    assert first.mosek_primal_objective == pytest.approx(first.objective_value)
    assert second.mosek_primal_objective == pytest.approx(second.objective_value)
    assert first.workspace_reused is False
    assert second.workspace_reused is True
    assert first.solve_index == 1
    assert second.solve_index == 2
    assert first.env_identity == second.env_identity == id(workspace.env)
    assert first.task_identity == second.task_identity == id(workspace.task)
    assert task.optimize_calls == 2
    assert task.appendvars_calls == 1
    assert task.appendcons_calls == 1
    assert task.objective == [0.0, 4.0, -0.2, -0.1]
    assert task.variable_bounds[0] == ("ra", -0.2, 0.8)
    assert task.variable_bounds[1] == ("ra", -0.7, 0.25)
    assert second.neutrality_residual == pytest.approx(0.0)
    assert second.bound_violation == pytest.approx(0.0)
    assert second.absolute_value_violation == pytest.approx(0.0)
    assert second.timing_evidence["workspace_reused"] is True
    assert second.timing_evidence["solve_index"] == 2
    for field in (
        "build_seconds",
        "update_objective_seconds",
        "update_bounds_seconds",
        "update_seconds",
        "optimize_seconds",
        "solution_seconds",
        "elapsed_seconds",
    ):
        assert getattr(second, field) >= 0.0


def test_accepts_scalar_spread_for_func2_contract() -> None:
    workspace = prepare_mosek_prepared_lp(2, mosek_module=FakeMosekModule())

    result = workspace.solve(
        score=np.array([3.0, 1.0]),
        spread=0.2,
        sell=np.array([0.4, 0.5]),
        buy=np.array([0.3, 0.6]),
    )

    np.testing.assert_allclose(result.absolute_dw, np.abs(result.dw))
    assert result.objective_value == pytest.approx(0.54)


def test_workspace_serializes_concurrent_update_and_optimize() -> None:
    fake = FakeMosekModule(optimize_delay=0.01)
    workspace = prepare_mosek_prepared_lp(2, mosek_module=fake)
    results = []

    def solve(score: np.ndarray) -> None:
        results.append(
            workspace.solve(
                score=score,
                spread=np.array([0.1, 0.1]),
                sell=np.array([0.5, 0.5]),
                buy=np.array([0.5, 0.5]),
            )
        )

    threads = [
        threading.Thread(target=solve, args=(np.array([2.0, 0.0]),)),
        threading.Thread(target=solve, args=(np.array([0.0, 2.0]),)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 2
    assert workspace.task.max_active_optimizes == 1
    assert sorted(result.solve_index for result in results) == [1, 2]
    assert sorted(result.workspace_reused for result in results) == [False, True]


def test_missing_mosek_fails_loud(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_import(name: str):
        assert name == "mosek"
        raise ImportError("mosek is not installed")

    monkeypatch.setattr(prepared_module.importlib, "import_module", missing_import)

    with pytest.raises(MosekUnavailableError, match="native MOSEK Python API") as exc_info:
        prepare_mosek_prepared_lp(2)

    assert isinstance(exc_info.value.__cause__, ImportError)


@pytest.mark.parametrize(
    ("argument", "value", "message"),
    [
        ("score", [1.0], "score must have shape"),
        ("spread", [0.1, -0.1], "spread must be nonnegative"),
        ("sell", [np.nan, 0.1], "sell must contain only finite"),
        ("buy", [0.1, -0.1], "buy must be nonnegative"),
    ],
)
def test_invalid_dynamic_input_fails_before_backend_update(
    argument: str,
    value: object,
    message: str,
) -> None:
    fake = FakeMosekModule()
    workspace = prepare_mosek_prepared_lp(2, mosek_module=fake)
    inputs = {
        "score": [1.0, 0.0],
        "spread": [0.1, 0.1],
        "sell": [0.2, 0.2],
        "buy": [0.2, 0.2],
    }
    inputs[argument] = value

    with pytest.raises(ValueError, match=message):
        workspace.solve(**inputs)

    assert workspace.task.optimize_calls == 0
    assert workspace.poisoned is False


def test_nonoptimal_or_backend_failure_poison_workspace() -> None:
    fake = FakeMosekModule()
    workspace = prepare_mosek_prepared_lp(2, mosek_module=fake)
    workspace.task.status = "unknown"
    inputs = {
        "score": [1.0, 0.0],
        "spread": [0.1, 0.1],
        "sell": [0.2, 0.2],
        "buy": [0.2, 0.2],
    }

    with pytest.raises(MosekBackendError, match="did not return an optimal solution"):
        workspace.solve(**inputs)
    assert workspace.poisoned is True
    assert workspace.solve_count == 0

    with pytest.raises(RuntimeError, match="workspace is poisoned"):
        workspace.solve(**inputs)


def test_close_disposes_persistent_task_and_env_and_prevents_reuse() -> None:
    fake = FakeMosekModule()
    workspace = prepare_mosek_prepared_lp(2, mosek_module=fake)
    task = workspace.task
    env = workspace.env

    workspace.close()
    workspace.close()

    assert task.disposed is True
    assert env.disposed is True
    with pytest.raises(RuntimeError, match="workspace is closed"):
        workspace.solve(
            score=[1.0, 0.0],
            spread=[0.1, 0.1],
            sell=[0.2, 0.2],
            buy=[0.2, 0.2],
        )
