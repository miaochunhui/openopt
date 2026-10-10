import pytest
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np

pytest.importorskip("scipy")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openpo.constraints import LinearConstraints
from openpo import exact_solver as exact_solver_module
from openpo.model import FactorRiskModel, PortfolioProblem
from openpo.projection import project_bounded_simplex
from openpo.solver import objective

torch = exact_solver_module.torch
ExactBackend = exact_solver_module.ExactBackend
ExactSolveResult = exact_solver_module.ExactSolveResult
PreparedExactWorkspace = exact_solver_module.PreparedExactWorkspace
exact_template_sha256 = exact_solver_module.exact_template_sha256
prepare_exact_workspace = exact_solver_module.prepare_exact_workspace
solve_exact = exact_solver_module.solve_exact
solve_exact_prepared = exact_solver_module.solve_exact_prepared
DTYPE = torch.float64


def make_problem(
    *,
    batch: int = 1,
    assets: int = 5,
    factors: int = 1,
    with_current: bool = False,
) -> PortfolioProblem:
    dtype = DTYPE
    exposures = torch.zeros(assets, factors, dtype=dtype)
    risk = FactorRiskModel(
        exposures=exposures,
        factor_variance=torch.ones(factors, dtype=dtype),
        specific_variance=torch.full((assets,), 0.2, dtype=dtype),
    )
    alpha = torch.tensor(
        [
            [0.05, -0.01, 0.02, 0.00, 0.03],
            [-0.02, 0.03, 0.04, -0.01, 0.01],
        ][:batch],
        dtype=dtype,
    )
    current = (
        torch.tensor([0.20, 0.10, 0.30, 0.15, 0.25], dtype=dtype)
        if with_current
        else None
    )
    return PortfolioProblem(
        alpha=alpha,
        risk=risk,
        lower=torch.zeros(assets, dtype=dtype),
        upper=torch.ones(assets, dtype=dtype),
        budget=1.0,
        risk_aversion=1.5,
        current=current,
        trade_cost=0.4 if with_current else 0.0,
    )


def make_constraints(assets: int) -> LinearConstraints:
    dtype = DTYPE
    return LinearConstraints(
        A_eq=torch.zeros((0, assets), dtype=dtype),
        b_eq=torch.zeros((0,), dtype=dtype),
    )


def with_dynamic_problem(
    problem: PortfolioProblem,
    *,
    alpha: torch.Tensor | None = None,
    current: torch.Tensor | None = None,
) -> PortfolioProblem:
    return PortfolioProblem(
        alpha=problem.alpha if alpha is None else alpha,
        risk=problem.risk,
        lower=problem.lower,
        upper=problem.upper,
        budget=problem.budget,
        risk_aversion=problem.risk_aversion,
        current=problem.current if current is None else current,
        trade_cost=problem.trade_cost,
    )


def test_exact_solver_matches_closed_form_with_trade_cost_and_broadcast() -> None:
    pytest.importorskip("osqp")
    pytest.importorskip("clarabel")

    problem = make_problem(batch=2, with_current=True)
    constraints = make_constraints(problem.risk.assets)
    expected = project_bounded_simplex(
        (
            problem.batched_alpha
            + problem.trade_cost * problem.current.unsqueeze(0)
        )
        / (problem.risk_aversion * problem.risk.specific_variance + problem.trade_cost),
        problem.lower,
        problem.upper,
    )

    osqp_result = solve_exact(problem, constraints, ExactBackend.OSQP)
    clarabel_result = solve_exact(problem, constraints, "clarabel")
    osqp_workspace = prepare_exact_workspace(problem, constraints, ExactBackend.OSQP)
    clarabel_workspace = prepare_exact_workspace(problem, constraints, ExactBackend.CLARABEL)
    osqp_prepared = solve_exact_prepared(osqp_workspace, problem, constraints)
    clarabel_prepared = solve_exact_prepared(clarabel_workspace, problem, constraints)

    for result, backend_name in (
        (osqp_result, "osqp"),
        (clarabel_result, "clarabel"),
        (osqp_prepared, "osqp"),
        (clarabel_prepared, "clarabel"),
    ):
        assert isinstance(result, ExactSolveResult)
        assert result.backend == backend_name
        assert result.device == "cpu"
        assert result.backend_version
        assert len(result.raw_statuses) == problem.batched_alpha.shape[0]
        assert all(isinstance(status, str) and status for status in result.raw_statuses)
        assert result.weights.dtype == DTYPE
        assert torch.allclose(result.weights, expected, atol=2e-8, rtol=2e-8)
        assert torch.allclose(result.objective, objective(problem, result.weights), atol=1e-12, rtol=1e-12)
        assert result.template_sha256
        assert result.template_evidence is not None
        assert result.timing_evidence is not None
        assert result.update_seconds >= 0.0
        assert result.solve_seconds > 0.0

    assert torch.allclose(osqp_result.weights, clarabel_result.weights, atol=2e-7, rtol=2e-7)
    assert torch.allclose(osqp_result.objective, clarabel_result.objective, atol=1e-10, rtol=1e-10)
    assert torch.allclose(osqp_prepared.weights, osqp_result.weights, atol=2e-8, rtol=2e-8)
    assert torch.allclose(clarabel_prepared.weights, clarabel_result.weights, atol=2e-8, rtol=2e-8)
    assert not osqp_result.prepared
    assert osqp_prepared.prepared
    assert isinstance(osqp_workspace, PreparedExactWorkspace)
    assert osqp_workspace.template_sha256 == osqp_prepared.template_sha256
    assert osqp_workspace.template_evidence["dynamic_excluded"] == ["alpha", "current"]
    assert clarabel_workspace.template_sha256 == clarabel_prepared.template_sha256
    assert clarabel_workspace.template_evidence["solver_settings"] == {
        "verbose": False,
        "max_iter": 500,
        "tol_feas": exact_solver_module.EXACT_SOLVER_TOLERANCE,
        "tol_gap_abs": exact_solver_module.EXACT_SOLVER_TOLERANCE,
        "tol_gap_rel": exact_solver_module.EXACT_SOLVER_TOLERANCE,
        "direct_solve_method": "qdldl",
        "max_threads": 1,
    }
    assert clarabel_prepared.template_evidence is not None
    assert (
        clarabel_prepared.template_evidence["solver_settings"]
        == clarabel_workspace.template_evidence["solver_settings"]
    )


def test_exact_solver_honors_linear_constraints_and_backend_consistency() -> None:
    pytest.importorskip("osqp")
    pytest.importorskip("clarabel")

    dtype = DTYPE
    assets = 4
    risk = FactorRiskModel(
        exposures=torch.tensor(
            [
                [0.10, -0.03],
                [0.02, 0.07],
                [-0.05, 0.04],
                [0.01, 0.02],
            ],
            dtype=dtype,
        ),
        factor_variance=torch.tensor([0.6, 1.3], dtype=dtype),
        specific_variance=torch.tensor([0.08, 0.09, 0.07, 0.05], dtype=dtype),
    )
    problem = PortfolioProblem(
        alpha=torch.tensor([0.04, -0.03, 0.02, 0.01], dtype=dtype),
        risk=risk,
        lower=torch.zeros(assets, dtype=dtype),
        upper=torch.full((assets,), 0.7, dtype=dtype),
        budget=1.0,
        risk_aversion=3.0,
    )
    constraints = LinearConstraints(
        A_eq=torch.tensor([[1.0, -1.0, 0.0, 0.0]], dtype=dtype),
        b_eq=torch.tensor([0.0], dtype=dtype),
        A_ineq=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=dtype),
        lower=torch.tensor([0.2], dtype=dtype),
        upper=torch.tensor([0.7], dtype=dtype),
    )

    osqp_result = solve_exact(problem, constraints, "OSQP")
    clarabel_result = solve_exact(problem, constraints, ExactBackend.CLARABEL)

    assert torch.allclose(osqp_result.weights, clarabel_result.weights, atol=2e-7, rtol=2e-7)
    assert torch.allclose(osqp_result.objective, clarabel_result.objective, atol=5e-10, rtol=5e-10)
    assert torch.allclose(osqp_result.objective, objective(problem, osqp_result.weights), atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        clarabel_result.objective,
        objective(problem, clarabel_result.weights),
        atol=1e-12,
        rtol=1e-12,
    )
    assert constraints.residual(osqp_result.weights).max_violation.item() <= 1e-8
    assert constraints.residual(clarabel_result.weights).max_violation.item() <= 1e-8


@pytest.mark.parametrize("backend", [ExactBackend.OSQP, ExactBackend.CLARABEL])
def test_prepared_exact_replay_and_dynamic_q_update_inputs(backend: ExactBackend) -> None:
    pytest.importorskip(backend.value)

    problem = make_problem(batch=2, with_current=True)
    constraints = make_constraints(problem.risk.assets)
    workspace = prepare_exact_workspace(problem, constraints, backend)
    first = solve_exact_prepared(workspace, problem, constraints)
    replay = solve_exact_prepared(workspace, problem, constraints)

    assert torch.allclose(first.weights, replay.weights, atol=0.0, rtol=0.0)
    assert torch.allclose(first.objective, replay.objective, atol=0.0, rtol=0.0)
    assert first.raw_statuses == replay.raw_statuses

    shifted = with_dynamic_problem(
        problem,
        alpha=problem.alpha + torch.tensor(
            [[0.005, 0.0, -0.003, 0.002, 0.001], [-0.004, 0.002, 0.001, 0.0, 0.003]],
            dtype=DTYPE,
        ),
        current=torch.tensor(
            [[0.18, 0.12, 0.27, 0.17, 0.26], [0.24, 0.07, 0.28, 0.18, 0.23]],
            dtype=DTYPE,
        ),
    )
    assert exact_template_sha256(problem, constraints, backend) == exact_template_sha256(
        shifted,
        constraints,
        backend,
    )

    prepared_shifted = solve_exact_prepared(workspace, shifted, constraints)
    cold_shifted = solve_exact(shifted, constraints, backend)
    assert torch.allclose(prepared_shifted.weights, cold_shifted.weights, atol=2e-8, rtol=2e-8)
    assert torch.allclose(prepared_shifted.objective, cold_shifted.objective, atol=1e-12, rtol=1e-12)
    assert prepared_shifted.template_sha256 == workspace.template_sha256
    assert prepared_shifted.template_evidence == workspace.template_evidence


def test_prepared_exact_rejects_static_template_and_dtype_drift() -> None:
    pytest.importorskip("osqp")

    problem = make_problem(batch=1)
    constraints = make_constraints(problem.risk.assets)
    workspace = prepare_exact_workspace(problem, constraints, ExactBackend.OSQP)
    original = problem.risk.specific_variance[0].clone()
    try:
        problem.risk.specific_variance[0] = original + 0.01
        with pytest.raises(RuntimeError, match="stale.*static template"):
            solve_exact_prepared(workspace, problem, constraints)
    finally:
        problem.risk.specific_variance[0] = original

    float32_problem = PortfolioProblem(
        alpha=problem.alpha.to(torch.float32),
        risk=FactorRiskModel(
            exposures=problem.risk.exposures.to(torch.float32),
            factor_variance=problem.risk.factor_variance.to(torch.float32),
            specific_variance=problem.risk.specific_variance.to(torch.float32),
        ),
        lower=problem.lower.to(torch.float32),
        upper=problem.upper.to(torch.float32),
        budget=problem.budget,
        risk_aversion=problem.risk_aversion,
        current=None,
        trade_cost=problem.trade_cost,
    )
    float32_constraints = LinearConstraints(
        A_eq=constraints.A_eq.to(torch.float32),
        b_eq=constraints.b_eq.to(torch.float32),
    )
    with pytest.raises(ValueError, match="dtype drifted"):
        solve_exact_prepared(workspace, float32_problem, float32_constraints)


def test_template_evidence_is_canonical_and_external_mutation_safe() -> None:
    pytest.importorskip("osqp")

    problem = make_problem(batch=1)
    constraints = make_constraints(problem.risk.assets)
    workspace = prepare_exact_workspace(problem, constraints, ExactBackend.OSQP)
    original_sha = workspace.template_sha256
    original_specific_sha = workspace.template_evidence["risk"]["specific_variance"]["sha256"]

    external_workspace_evidence = workspace.template_evidence
    external_workspace_evidence["risk"]["specific_variance"]["sha256"] = "tampered"
    external_workspace_evidence["dynamic_excluded"].append("risk")

    assert workspace.template_sha256 == original_sha
    assert workspace.template_evidence["risk"]["specific_variance"]["sha256"] == original_specific_sha
    assert workspace.template_evidence["dynamic_excluded"] == ["alpha", "current"]

    result = solve_exact_prepared(workspace, problem, constraints)
    external_result_evidence = result.template_evidence
    assert external_result_evidence is not None
    external_result_evidence["backend"] = "tampered"
    external_result_evidence["risk"]["specific_variance"]["sha256"] = "tampered"

    assert result.template_sha256 == original_sha
    assert result.template_evidence is not None
    assert result.template_evidence["backend"] == "osqp"
    assert result.template_evidence["risk"]["specific_variance"]["sha256"] == original_specific_sha


def test_template_hash_includes_backend_settings_and_is_stable() -> None:
    pytest.importorskip("osqp")
    pytest.importorskip("clarabel")

    problem = make_problem(batch=1)
    constraints = make_constraints(problem.risk.assets)

    osqp_sha = exact_template_sha256(problem, constraints, ExactBackend.OSQP)
    clarabel_sha = exact_template_sha256(problem, constraints, ExactBackend.CLARABEL)

    assert osqp_sha == exact_template_sha256(problem, constraints, "osqp")
    assert clarabel_sha == exact_template_sha256(problem, constraints, "clarabel")
    assert osqp_sha != clarabel_sha

    osqp_workspace = prepare_exact_workspace(problem, constraints, ExactBackend.OSQP)
    clarabel_workspace = prepare_exact_workspace(problem, constraints, ExactBackend.CLARABEL)
    assert osqp_workspace.template_sha256 == osqp_sha
    assert clarabel_workspace.template_sha256 == clarabel_sha
    assert osqp_workspace.template_evidence["schema_version"] == 2
    assert clarabel_workspace.template_evidence["schema_version"] == 2
    assert osqp_workspace.template_evidence["solver_settings"] == {
        "verbose": False,
        "polish": True,
        "eps_abs": exact_solver_module.EXACT_SOLVER_TOLERANCE,
        "eps_rel": exact_solver_module.EXACT_SOLVER_TOLERANCE,
        "max_iter": 20_000,
    }
    assert clarabel_workspace.template_evidence["solver_settings"] == {
        "verbose": False,
        "max_iter": 500,
        "tol_feas": exact_solver_module.EXACT_SOLVER_TOLERANCE,
        "tol_gap_abs": exact_solver_module.EXACT_SOLVER_TOLERANCE,
        "tol_gap_rel": exact_solver_module.EXACT_SOLVER_TOLERANCE,
        "direct_solve_method": "qdldl",
        "max_threads": 1,
    }


def test_clarabel_setting_drift_changes_hash_and_stales_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("clarabel")

    problem = make_problem(batch=1)
    constraints = make_constraints(problem.risk.assets)
    workspace = prepare_exact_workspace(problem, constraints, ExactBackend.CLARABEL)

    monkeypatch.setitem(
        exact_solver_module.CLARABEL_SOLVER_SETTINGS,
        "direct_solve_method",
        "faer",
    )

    changed_sha = exact_template_sha256(problem, constraints, ExactBackend.CLARABEL)
    assert changed_sha != workspace.template_sha256
    with pytest.raises(RuntimeError, match="stale.*static template"):
        solve_exact_prepared(workspace, problem, constraints)


def test_solve_exact_batch1_keeps_cold_no_update_path(monkeypatch) -> None:
    problem = make_problem(batch=1, factors=0)
    constraints = make_constraints(problem.risk.assets)
    captured: dict[str, object] = {}

    class FakeOSQPSolver:
        def __init__(self) -> None:
            self.q = np.zeros(problem.risk.assets, dtype=np.float64)
            captured["solver"] = self

        def setup(self, **kwargs) -> None:
            self.q = np.asarray(kwargs["q"], dtype=np.float64).copy()

        def solve(self, raise_error: bool = True) -> types.SimpleNamespace:
            return types.SimpleNamespace(status="solved", x=-self.q.copy())

    fake_osqp_module = types.SimpleNamespace(
        __version__="fake-osqp-no-update",
        OSQP=FakeOSQPSolver,
    )
    real_import = exact_solver_module.importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "osqp":
            return fake_osqp_module
        return real_import(name, package)

    perf_values = iter([0.0, 10.0, 12.0, 20.0, 23.0, 30.0, 35.0, 40.0, 50.0, 51.0, 60.0, 61.0])
    monkeypatch.setattr(exact_solver_module.importlib, "import_module", fake_import)
    monkeypatch.setattr(exact_solver_module.time, "perf_counter", lambda: next(perf_values))

    result = solve_exact(problem, constraints, ExactBackend.OSQP)

    assert torch.allclose(result.weights, problem.alpha, atol=0.0, rtol=0.0)
    assert result.update_seconds == 0.0
    assert result.build_seconds == pytest.approx(2.0)
    assert result.setup_seconds == pytest.approx(3.0)
    assert result.solve_seconds == pytest.approx(5.0)
    assert result.elapsed_seconds == pytest.approx(40.0)

    with pytest.raises(RuntimeError, match="does not support q updates"):
        prepare_exact_workspace(problem, constraints, ExactBackend.OSQP)


def test_prepare_exact_workspace_requires_clarabel_data_update_allowed(monkeypatch) -> None:
    problem = make_problem(batch=1, factors=0)
    constraints = make_constraints(problem.risk.assets)

    class FakeClarabelSettings:
        def __init__(self) -> None:
            self.verbose = True
            self.max_iter = 0
            self.tol_feas = None
            self.tol_gap_abs = None
            self.tol_gap_rel = None

    class FakeClarabelSolver:
        def __init__(self, p, q, a, b, cones, settings) -> None:
            pass

        def update(self, **kwargs) -> None:
            return None

        def is_data_update_allowed(self) -> bool:
            return False

    fake_clarabel_module = types.SimpleNamespace(
        __version__="fake-clarabel-no-update-allowed",
        DefaultSettings=FakeClarabelSettings,
        DefaultSolver=FakeClarabelSolver,
        ZeroConeT=lambda size: ("zero", size),
        NonnegativeConeT=lambda size: ("nonnegative", size),
    )
    real_import = exact_solver_module.importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "clarabel":
            return fake_clarabel_module
        return real_import(name, package)

    monkeypatch.setattr(exact_solver_module.importlib, "import_module", fake_import)

    with pytest.raises(RuntimeError, match="does not allow data updates"):
        prepare_exact_workspace(problem, constraints, ExactBackend.CLARABEL)


@pytest.mark.parametrize(
    ("failure_stage", "match"),
    [
        ("update", "update failed"),
        ("solve", "solve failed"),
        ("status", "status"),
        ("solution", "no primal solution"),
    ],
)
def test_prepared_exact_workspace_poison_rejects_after_backend_failure(
    monkeypatch,
    failure_stage: str,
    match: str,
) -> None:
    problem = make_problem(batch=1, factors=0)
    constraints = make_constraints(problem.risk.assets)
    captured: dict[str, object] = {}

    class FakeOSQPSolver:
        def __init__(self) -> None:
            self.q = np.zeros(problem.risk.assets, dtype=np.float64)
            self.update_calls = 0
            self.solve_calls = 0
            captured["solver"] = self

        def setup(self, **kwargs) -> None:
            self.q = np.asarray(kwargs["q"], dtype=np.float64).copy()

        def update(self, **kwargs) -> None:
            self.update_calls += 1
            self.q = np.asarray(kwargs["q"], dtype=np.float64).copy()
            if failure_stage == "update":
                raise RuntimeError("backend q buffer polluted")

        def solve(self, raise_error: bool = True) -> types.SimpleNamespace:
            self.solve_calls += 1
            if failure_stage == "solve":
                raise RuntimeError("backend solve polluted")
            if failure_stage == "status":
                return types.SimpleNamespace(status="primal infeasible", x=-self.q.copy())
            if failure_stage == "solution":
                return types.SimpleNamespace(status="solved", x=None)
            return types.SimpleNamespace(status="solved", x=-self.q.copy())

    fake_osqp_module = types.SimpleNamespace(
        __version__="fake-osqp-poison",
        OSQP=FakeOSQPSolver,
    )
    real_import = exact_solver_module.importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "osqp":
            return fake_osqp_module
        return real_import(name, package)

    monkeypatch.setattr(exact_solver_module.importlib, "import_module", fake_import)

    workspace = prepare_exact_workspace(problem, constraints, ExactBackend.OSQP)
    fake_solver = captured["solver"]

    with pytest.raises(RuntimeError, match=match):
        solve_exact_prepared(workspace, problem, constraints)

    assert workspace.poisoned
    assert workspace.poison_reason
    update_calls = fake_solver.update_calls
    solve_calls = fake_solver.solve_calls

    with pytest.raises(RuntimeError, match="poisoned"):
        solve_exact_prepared(workspace, problem, constraints)

    assert fake_solver.update_calls == update_calls
    assert fake_solver.solve_calls == solve_calls


def test_exact_solver_uses_shared_tolerance_for_both_backends(monkeypatch) -> None:
    problem = make_problem()
    constraints = make_constraints(problem.risk.assets)
    variable_count = problem.risk.assets + problem.risk.factor_variance.shape[0]
    captured: dict[str, object] = {}
    clarabel_settings_seen: list[object] = []

    class FakeOSQPSolver:
        def __init__(self) -> None:
            captured["osqp_solver"] = self
            self.setup_kwargs: dict[str, object] | None = None
            self.update_calls = 0

        def setup(self, **kwargs) -> None:
            self.setup_kwargs = kwargs

        def update(self, **kwargs) -> None:
            self.update_calls += 1

        def solve(self, raise_error: bool = True) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                status="solved",
                x=np.zeros(variable_count, dtype=np.float64),
            )

    class FakeClarabelSettings:
        def __init__(self) -> None:
            self.verbose = True
            self.max_iter = 0
            self.tol_feas = None
            self.tol_gap_abs = None
            self.tol_gap_rel = None

    class FakeClarabelSolver:
        def __init__(self, p, q, a, b, cones, settings) -> None:
            captured["clarabel_settings"] = settings
            clarabel_settings_seen.append(settings)

        def update(self, **kwargs) -> None:
            return None

        def is_data_update_allowed(self) -> bool:
            return True

        def solve(self) -> types.SimpleNamespace:
            return types.SimpleNamespace(
                status="Solved",
                x=np.zeros(variable_count, dtype=np.float64),
            )

    fake_osqp_module = types.SimpleNamespace(
        __version__="fake-osqp-1.0",
        OSQP=FakeOSQPSolver,
    )
    fake_clarabel_module = types.SimpleNamespace(
        __version__="fake-clarabel-1.0",
        DefaultSettings=FakeClarabelSettings,
        DefaultSolver=FakeClarabelSolver,
        ZeroConeT=lambda size: ("zero", size),
        NonnegativeConeT=lambda size: ("nonnegative", size),
    )

    real_import = exact_solver_module.importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "osqp":
            return fake_osqp_module
        if name == "clarabel":
            return fake_clarabel_module
        return real_import(name, package)

    monkeypatch.setattr(exact_solver_module.importlib, "import_module", fake_import)

    osqp_result = solve_exact(problem, constraints, ExactBackend.OSQP)
    clarabel_result = solve_exact(problem, constraints, ExactBackend.CLARABEL)
    clarabel_workspace = prepare_exact_workspace(problem, constraints, ExactBackend.CLARABEL)

    assert exact_solver_module.EXACT_SOLVER_TOLERANCE == pytest.approx(1e-8)
    assert captured["osqp_solver"].setup_kwargs["eps_abs"] == exact_solver_module.EXACT_SOLVER_TOLERANCE
    assert captured["osqp_solver"].setup_kwargs["eps_rel"] == exact_solver_module.EXACT_SOLVER_TOLERANCE
    assert captured["osqp_solver"].update_calls == 0
    assert len(clarabel_settings_seen) == 2
    for clarabel_settings in clarabel_settings_seen:
        assert clarabel_settings.verbose is False
        assert clarabel_settings.max_iter == 500
        assert clarabel_settings.direct_solve_method == "qdldl"
        assert clarabel_settings.max_threads == 1
        assert clarabel_settings.tol_feas == exact_solver_module.EXACT_SOLVER_TOLERANCE
        assert clarabel_settings.tol_gap_abs == exact_solver_module.EXACT_SOLVER_TOLERANCE
        assert clarabel_settings.tol_gap_rel == exact_solver_module.EXACT_SOLVER_TOLERANCE
    clarabel_settings = captured["clarabel_settings"]
    assert clarabel_settings.tol_feas == exact_solver_module.EXACT_SOLVER_TOLERANCE
    assert clarabel_settings.tol_gap_abs == exact_solver_module.EXACT_SOLVER_TOLERANCE
    assert clarabel_settings.tol_gap_rel == exact_solver_module.EXACT_SOLVER_TOLERANCE
    assert osqp_result.backend == "osqp"
    assert clarabel_result.backend == "clarabel"
    assert clarabel_result.template_evidence is not None
    assert clarabel_result.template_evidence["solver_settings"]["direct_solve_method"] == "qdldl"
    assert clarabel_result.template_evidence["solver_settings"]["max_threads"] == 1
    assert clarabel_workspace.template_sha256 == clarabel_result.template_sha256


def test_clarabel_setting_not_retained_fails_loud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = make_problem(batch=1, factors=0)
    constraints = make_constraints(problem.risk.assets)

    class RejectingSettings:
        def __setattr__(self, name, value) -> None:
            if name == "direct_solve_method":
                value = "auto"
            object.__setattr__(self, name, value)

    fake_clarabel_module = types.SimpleNamespace(
        __version__="fake-clarabel-reject-setting",
        DefaultSettings=RejectingSettings,
    )
    real_import = exact_solver_module.importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "clarabel":
            return fake_clarabel_module
        return real_import(name, package)

    monkeypatch.setattr(exact_solver_module.importlib, "import_module", fake_import)

    with pytest.raises(
        RuntimeError,
        match="clarabel setting 'direct_solve_method' did not retain configured value",
    ):
        solve_exact(problem, constraints, ExactBackend.CLARABEL)


def test_exact_solver_rejects_mixed_constraint_dtype() -> None:
    problem = make_problem()
    constraints = LinearConstraints(
        A_eq=torch.zeros((0, problem.risk.assets), dtype=torch.float32),
        b_eq=torch.zeros((0,), dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="same dtype"):
        solve_exact(problem, constraints, ExactBackend.OSQP)


def test_prepared_exact_api_probe_and_concurrent_update_solve_lock(monkeypatch) -> None:
    problem = make_problem(batch=1, factors=0)
    constraints = make_constraints(problem.risk.assets)
    captured: dict[str, object] = {}

    class FakeOSQPSolver:
        def __init__(self) -> None:
            self.q = np.zeros(problem.risk.assets, dtype=np.float64)
            self.update_calls = 0
            self.solve_calls = 0
            captured["solver"] = self

        def setup(self, **kwargs) -> None:
            self.q = np.asarray(kwargs["q"], dtype=np.float64).copy()

        def update(self, **kwargs) -> None:
            self.update_calls += 1
            self.q = np.asarray(kwargs["q"], dtype=np.float64).copy()

        def solve(self, raise_error: bool = True) -> types.SimpleNamespace:
            time.sleep(0.01)
            self.solve_calls += 1
            return types.SimpleNamespace(
                status="solved",
                x=-self.q.copy(),
            )

    fake_osqp_module = types.SimpleNamespace(
        __version__="fake-osqp-lock",
        OSQP=FakeOSQPSolver,
    )
    real_import = exact_solver_module.importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "osqp":
            return fake_osqp_module
        return real_import(name, package)

    monkeypatch.setattr(exact_solver_module.importlib, "import_module", fake_import)

    workspace = prepare_exact_workspace(problem, constraints, ExactBackend.OSQP)
    assert isinstance(workspace, PreparedExactWorkspace)
    assert workspace.template_sha256 == exact_template_sha256(problem, constraints, ExactBackend.OSQP)

    alpha_a = torch.tensor([[0.11, 0.12, 0.13, 0.14, 0.15]], dtype=DTYPE)
    alpha_b = torch.tensor([[0.21, 0.22, 0.23, 0.24, 0.25]], dtype=DTYPE)
    problem_a = with_dynamic_problem(problem, alpha=alpha_a)
    problem_b = with_dynamic_problem(problem, alpha=alpha_b)
    results: dict[str, ExactSolveResult] = {}

    def run(name: str, dynamic_problem: PortfolioProblem) -> None:
        results[name] = solve_exact_prepared(workspace, dynamic_problem, constraints)

    threads = [
        threading.Thread(target=run, args=("a", problem_a)),
        threading.Thread(target=run, args=("b", problem_b)),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert torch.allclose(results["a"].weights, alpha_a, atol=0.0, rtol=0.0)
    assert torch.allclose(results["b"].weights, alpha_b, atol=0.0, rtol=0.0)
    fake_solver = captured["solver"]
    assert fake_solver.update_calls == 2
    assert fake_solver.solve_calls == 2


def test_exact_solver_fails_loud_for_invalid_backend_and_missing_dependency(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unsupported exact backend"):
        solve_exact(None, None, "bogus")  # type: ignore[arg-type]

    class FakeTensor:
        def __init__(self, *, shape: tuple[int, ...], device: types.SimpleNamespace) -> None:
            self.shape = shape
            self.ndim = len(shape)
            self.device = device
            self.dtype = "float64"

        def is_floating_point(self) -> bool:
            return True

    fake_device = types.SimpleNamespace(type="cpu")
    fake_problem = types.SimpleNamespace(
        alpha=FakeTensor(shape=(1, 2), device=fake_device),
        batched_alpha=FakeTensor(shape=(1, 2), device=fake_device),
        lower=FakeTensor(shape=(2,), device=fake_device),
        upper=FakeTensor(shape=(2,), device=fake_device),
        risk=types.SimpleNamespace(
            assets=2,
            exposures=FakeTensor(shape=(2, 1), device=fake_device),
            factor_variance=FakeTensor(shape=(1,), device=fake_device),
            specific_variance=FakeTensor(shape=(2,), device=fake_device),
        ),
        current=None,
        risk_aversion=1.0,
        trade_cost=0.0,
    )
    fake_constraints = types.SimpleNamespace(
        num_variables=2,
        A_eq=None,
        b_eq=None,
        A_ineq=None,
        lower=None,
        upper=None,
        device=fake_device,
        dtype="float64",
    )

    import openpo.exact_solver as exact_solver

    real_import = exact_solver.importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name == "osqp":
            raise ModuleNotFoundError("No module named 'osqp'")
        return real_import(name, package)

    monkeypatch.setattr(exact_solver.importlib, "import_module", fake_import)
    with pytest.raises(ImportError, match="osqp"):
        solve_exact(fake_problem, fake_constraints, ExactBackend.OSQP)
