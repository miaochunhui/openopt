"""Offline adapters and mutated evidence are not licensed commercial proof."""

import builtins
import importlib.metadata
import json
import os
import socket
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_commercial as bc


PIN = "a" * 64  # Test-only synthetic approval digest, never a production approval.


def metadata_missing(name):
    raise importlib.metadata.PackageNotFoundError(name)


def fake_discovery(monkeypatch, available=True):
    monkeypatch.setattr(bc.importlib.util, "find_spec", lambda name: object() if available else None)
    monkeypatch.setattr(bc.importlib.metadata, "version", lambda name: "1.2.3" if available else metadata_missing(name))


def result_at(root):
    path = next(root.glob("*/result.json"))
    return path, json.loads(path.read_text())


class WitnessAdapter:
    def __init__(self, solver, arrays, config, mutation=None):
        self.solver, self.arrays, self.mutation = solver, arrays, mutation
        self.effective = {key: value if type(value) in (int, float) else 0
                          for key, value in config.items()}
        self.version = "1.2.3"
        self.calls = 0
        self.closed = False

    def solve(self):
        self.calls += 1
        if self.mutation == "exception":
            print("PRIVATE_SENTINEL license@private-host /private/license")
            os.write(2, b"PRIVATE_SENTINEL /private/license\n")
            error = RuntimeError("PRIVATE_SENTINEL license@private-host")
            error.error_code = 10009
            raise error
        x, y = self.arrays["expected"].copy(), self.arrays["witness_y"].copy()
        if self.mutation == "x":
            x[0] += 1
        if self.mutation == "y":
            y *= -1
        if self.mutation == "nan":
            x[0] = np.nan
        if self.mutation == "shape":
            y = y[:-1]
        if self.mutation == "problem":
            self.problem.q[0] += 10
        return dict(optimal=self.mutation != "status", status_code=1, x=x, y=y)

    def close(self):
        self.closed = True


def fake_authorized(monkeypatch, mutation=None, fail_solver=None):
    fake_discovery(monkeypatch)
    monkeypatch.setattr(bc, "import_sdk", lambda solver, args: object())
    arrays = bc.generated_problem(4, 20261010, "qp")
    workspaces = []

    def factory(solver, sdk, p, config):
        workspace = WitnessAdapter(solver, arrays, config,
                                   mutation if fail_solver in (None, solver) else None)
        workspace.problem = p
        workspaces.append(workspace)
        return workspace

    monkeypatch.setattr(bc, "create_adapter", factory)
    return workspaces


def run_fake(monkeypatch, tmp_path, mutation=None, fail_solver=None):
    workspaces = fake_authorized(monkeypatch, mutation, fail_solver)
    rc = bc.main(["--artifact-dir", str(tmp_path), "--size", "4", "--authorized-run",
                  "--authorization-sha256", PIN])
    path, report = result_at(tmp_path)
    return rc, path, report, workspaces


def test_missing_sdk_actual_structure_stays_unverified(monkeypatch, tmp_path):
    fake_discovery(monkeypatch, False)
    assert bc.main(["--artifact-dir", str(tmp_path)]) == 0
    path, report = result_at(tmp_path)
    assert report["verdict"] == report["qualification"] == "UNVERIFIED"
    assert report["authorization"]["legal_validity"] == "UNVERIFIED"
    assert report["authorization"]["record_is_legal_authorization"] is False
    assert report["counts"] == dict(planned=18, attempted=0, solve_attempted=0, passed=0, failed=0)
    assert all(a["reason"] == "SDK_NOT_DISCOVERED" for a in report["solvers"].values())
    assert bc.check_result(path)["qualification"] == "UNVERIFIED"
    assert path.with_name("problem.npz").is_file()
    assert "UNVERIFIED" in path.with_name("execution.log").read_text()


@pytest.mark.parametrize("available", [True, False])
def test_unauthorized_discovery_never_imports_sdk_or_opens_socket(monkeypatch, tmp_path, available):
    fake_discovery(monkeypatch, available)
    original_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert name.split(".")[0] not in [m for m, _ in bc.SDKS.values()]
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("forbidden SDK/license/network execution")

    monkeypatch.setattr(builtins, "__import__", guarded)
    monkeypatch.setattr(bc, "import_sdk", forbidden)
    monkeypatch.setattr(bc, "create_adapter", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    for name in ("GRB_LICENSE_FILE", "MOSEKLM_LICENSE_FILE", "ILOG_LICENSE_FILE"):
        monkeypatch.setenv(name, "PRIVATE_SENTINEL /private/license license@private-host")
    assert bc.main(["--artifact-dir", str(tmp_path)]) == 0
    path, report = result_at(tmp_path)
    assert report["verdict"] == "UNVERIFIED"
    assert all(not a["sdk_imported"] for a in report["solvers"].values())
    for file in path.parent.glob("*.json"):
        assert "PRIVATE_SENTINEL" not in file.read_text()
        assert "/private/license" not in file.read_text()


def test_poisoned_sdk_find_spec_does_not_execute(monkeypatch, tmp_path):
    sdk = tmp_path / "gurobipy"
    sdk.mkdir()
    (sdk / "__init__.py").write_text("raise AssertionError(\"SDK_IMPORTED\")\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(bc.importlib.metadata, "version", lambda name: "1.2.3")
    assert bc.discover("gurobi")["module_available"]
    assert bc.discover("gurobi")["licensed_run"] == "UNVERIFIED"


@pytest.mark.parametrize("options", [
    ["--authorized-run"], ["--authorization-sha256", PIN],
    ["--authorized-run", "--authorization-sha256", "PRIVATE_SENTINEL /private/license"],
    ["--repeats", "0"], ["--eps-abs", "nan"], ["--eps-rel", "inf"],
    ["--solvers", "gurobi", "gurobi"], ["--time-limit", "-1"],
])
def test_invalid_cli_fails_before_import_or_artifacts(monkeypatch, tmp_path, capsys, options):
    def forbidden(*args, **kwargs):
        raise AssertionError("import attempted")
    monkeypatch.setattr(bc, "import_sdk", forbidden)
    assert bc.main(["--artifact-dir", str(tmp_path), *options]) == 2
    assert not list(tmp_path.iterdir())
    assert "PRIVATE_SENTINEL" not in capsys.readouterr().out


def test_direct_import_gate_blocks(monkeypatch):
    args = SimpleNamespace(authorized_run=False, authorization_sha256=None)
    with pytest.raises(PermissionError):
        bc.import_sdk("gurobi", args)


@pytest.mark.parametrize("solver", tuple(bc.SDKS))
def test_direct_import_gate_allows_only_explicit_authorized(monkeypatch, solver):
    imported = []
    monkeypatch.setattr(bc.importlib, "import_module", lambda name: imported.append(name))
    bc.import_sdk(solver, SimpleNamespace(authorized_run=True, authorization_sha256=PIN))
    assert imported == [bc.SDKS[solver][0]]


def test_fake_licensed_full_cold_prepared_host_replay(monkeypatch, tmp_path):
    rc, path, report, workspaces = run_fake(monkeypatch, tmp_path)
    assert rc == 0
    assert report["verdict"] == "PASS"  # Offline fixture only, not a real commercial result.
    assert report["authorization"]["legal_validity"] == "UNVERIFIED"
    assert set(report["source"]["sha256"]) == set(bc.source_files())
    assert report["counts"] == dict(planned=18, attempted=18, solve_attempted=18, passed=18, failed=0)
    assert len(workspaces) == 12
    assert all(w.closed for w in workspaces)
    assert sum(w.calls == 3 for w in workspaces) == 3
    for arm in report["solvers"].values():
        assert arm["licensed_run"] == "EXECUTED"
        assert arm["parameters_sha256"] == bc.digest(arm["parameters"])
        for mode in arm["modes"].values():
            assert len(mode["samples"]) == 3
            assert all(s["host_x"] and s["host_y"] and s["certificate"]["passed"] for s in mode["samples"])
    assert bc.check_result(path)["acceptance"] == "PASS"


@pytest.mark.parametrize("mutation", ["x", "y", "nan", "shape", "status", "problem"])
def test_solver_output_mutants_rejected_independently(monkeypatch, tmp_path, mutation):
    rc, path, report, _ = run_fake(monkeypatch, tmp_path, mutation, "gurobi")
    assert rc == 1 and report["verdict"] == "FAIL"
    assert report["qualification"] == "UNVERIFIED"
    assert report["solvers"]["gurobi"]["status"] == "FAIL"
    assert report["solvers"]["gurobi"]["modes"]["cold"]["samples"][0]["verdict"] == "FAIL"
    assert not report["solvers"]["gurobi"]["modes"]["prepared"]["samples"]
    assert report["solvers"]["cplex"]["status"] == report["solvers"]["mosek"]["status"] == "PASS"
    assert report["counts"]["failed"] == 1
    assert bc.check_result(path)["verdict"] == "FAIL"


def test_license_error_keeps_first_cause_rc_coordinate_and_redacts_native_logs(monkeypatch, tmp_path, capfd):
    rc, path, report, _ = run_fake(monkeypatch, tmp_path, "exception", "gurobi")
    assert rc == 1
    arm = report["solvers"]["gurobi"]
    assert arm["licensed_run"] == arm["status"] == "UNVERIFIED"
    failure = json.loads(path.with_name("failure.json").read_text())
    cause = failure["first_cause"]
    assert cause["stage"] == "solve_host_result"
    assert cause["solver"] == "gurobi" and cause["mode"] == "cold" and cause["repeat"] == 0
    assert cause["causes"][0]["error_code"] == 10009
    for file in path.parent.glob("*.json"):
        assert "PRIVATE_SENTINEL" not in file.read_text()
        assert "private-host" not in file.read_text()
    captured = capfd.readouterr()
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err
    assert bc.check_result(path)["qualification"] == "UNVERIFIED"


@pytest.mark.parametrize("mutation", ["x", "y", "parameters", "hash", "npz", "counts", "authorization", "latency", "source", "mode"])
def test_disk_mutants_rejected(monkeypatch, tmp_path, mutation):
    _, path, report, _ = run_fake(monkeypatch, tmp_path)
    if mutation == "npz":
        arrays = bc.load_problem(path.with_name("problem.npz"))
        arrays["q"][0] += 1
        np.savez(path.with_name("problem.npz"), **arrays)
    elif mutation in ("x", "y"):
        report["solvers"]["gurobi"]["modes"]["cold"]["samples"][0]["host_" + mutation][0] += 1
    elif mutation == "parameters":
        report["solvers"]["gurobi"]["parameters"]["Threads"] = 99
    elif mutation == "hash":
        report["problem"]["content_sha256"] = "0" * 64
    elif mutation == "counts":
        report["counts"]["passed"] += 1
    elif mutation == "authorization":
        report["authorization"]["explicit_authorized_run"] = False
        report["authorization"]["record_sha256"] = None
    elif mutation == "latency":
        report["solvers"]["cplex"]["modes"]["cold"]["latency"]["p95_s"] = 0
    elif mutation == "source":
        report["source"]["sha256"]["scripts/benchmark_commercial.py"] = "0" * 64
    elif mutation == "mode":
        del report["solvers"]["gurobi"]["modes"]["prepared"]
    bc.write_json(path, report)
    with pytest.raises(ValueError):
        bc.check_result(path)


@pytest.mark.parametrize("solver", tuple(bc.SDKS))
def test_discovery_exception_fail_loud_without_message(monkeypatch, tmp_path, solver):
    def broken(name):
        raise OSError(5, "PRIVATE_SENTINEL /private/license")
    monkeypatch.setattr(bc.importlib.util, "find_spec", broken)
    monkeypatch.setattr(bc.importlib.metadata, "version", metadata_missing)
    assert bc.main(["--artifact-dir", str(tmp_path), "--solvers", solver]) == 1
    path, report = result_at(tmp_path)
    assert report["verdict"] == "FAIL"
    assert report["qualification"] == "UNVERIFIED"
    assert report["failures"][0]["causes"][0]["errno"] == 5
    assert "PRIVATE_SENTINEL" not in path.read_text()


@pytest.mark.parametrize("kind", ["lp", "qp"])
def test_generated_problem_and_npz_same_original_coordinates(tmp_path, kind):
    arrays = bc.generated_problem(4, 22, kind)
    p = bc.validate_problem(arrays)
    assert bc.check_qp_candidate(p, arrays["expected"], arrays["witness_y"]).passed
    path = tmp_path / "problem.npz"
    np.savez(path, **arrays)
    loaded = bc.load_problem(path)
    assert bc.problem_hash(loaded) == bc.problem_hash(arrays)


@pytest.mark.parametrize("mutation", ["indefinite", "float32", "nonfinite", "empty", "bounds"])
def test_problem_mutants_fail_before_sdk(mutation):
    arrays = bc.generated_problem(4, 22, "qp")
    if mutation == "indefinite":
        arrays["P"] = np.array([[1, 2, 0, 0], [2, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)
    elif mutation == "float32":
        arrays["q"] = arrays["q"].astype(np.float32)
    elif mutation == "nonfinite":
        arrays["A"][0, 0] = np.nan
    elif mutation == "empty":
        arrays["q"] = np.empty(0)
    else:
        arrays["l"][0] = arrays["u"][0] + 1
    with pytest.raises(ValueError):
        bc.validate_problem(arrays)


def test_row_mapping_two_sided_equality_and_free():
    p = SimpleNamespace(l=np.array([0., 1., -np.inf, -np.inf]),
                        u=np.array([2., 1., 3., np.inf]))
    rows = bc.split_rows(p)
    assert rows == [(0, "G", 0.), (0, "L", 2.), (1, "E", 1.), (2, "L", 3.)]
    assert np.array_equal(bc.mapped_dual(rows, [2, -3, 4, -5], 4), [1, -4, 5, 0])


def test_existing_artifact_preserved_and_in_tree_refused(monkeypatch, tmp_path):
    fake_discovery(monkeypatch, False)
    old = tmp_path / "retained.json"
    old.write_text("other-seat-asset")
    assert bc.main(["--artifact-dir", str(tmp_path)]) == 0
    assert bc.main(["--artifact-dir", str(tmp_path)]) == 0
    assert old.read_text() == "other-seat-asset"
    assert len(list(tmp_path.glob("discovery-*/result.json"))) == 2
    assert bc.main(["--artifact-dir", str(bc.ROOT / "forbidden-output")]) == 2


def test_nonfinite_version_metadata_cannot_leak(monkeypatch):
    monkeypatch.setattr(bc.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(bc.importlib.metadata, "version", lambda name: "PRIVATE_SENTINEL /private/license")
    row = bc.discover("gurobi")
    assert row["errors"] and row["version"] is None
    assert "PRIVATE_SENTINEL" not in json.dumps(row)


class ParamNode:
    def __init__(self):
        self.children = {}
        self.value = None

    def __getattr__(self, name):
        return self.children.setdefault(name, ParamNode())

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class RecordingTask:
    def __init__(self):
        self.calls = {}
        self.params = {}
        self.disposed = False

    def __getattr__(self, name):
        def record(*args, **kwargs):
            self.calls[name] = (args, kwargs)
        return record

    def putintparam(self, name, value):
        self.params[name] = value

    putdouparam = putintparam

    def getintparam(self, name):
        return self.params[name]

    getdouparam = getintparam

    def getsolsta(self, kind):
        return 1

    def getxx(self, kind, values):
        values[:] = [0.25, -0.25]

    def gety(self, kind, values):
        values[:] = [2., -3., 0.]

    def dispose(self):
        self.disposed = True


def adapter_problem():
    return SimpleNamespace(P=np.array([[2., 0.5], [0.5, 3.]]), q=np.array([1., -2.]),
                           A=np.array([[1., 2.], [3., 4.], [5., 6.]]),
                           l=np.array([-1., 0., -np.inf]), u=np.array([1., 0., np.inf]))


def test_native_mosek_model_quadratic_triangle_bounds_parameters_and_duals():
    task = RecordingTask()
    env = SimpleNamespace(Task=lambda *a: task, getversion=lambda: (11, 2, 6),
                          dispose=lambda: None)
    enum = SimpleNamespace(fr=0, lo=1, up=2, ra=3, fx=4)
    sdk = SimpleNamespace(Env=lambda: env, boundkey=enum, objsense=SimpleNamespace(minimize=0),
                          optimizertype=SimpleNamespace(intpnt=1), basindtype=SimpleNamespace(never=0),
                          startpointtype=SimpleNamespace(constant=0),
                          iparam=SimpleNamespace(optimizer="optimizer", num_threads="num_threads",
                                                intpnt_basis="intpnt_basis", intpnt_starting_point="start"),
                          dparam=SimpleNamespace(intpnt_tol_pfeas="pfeas"),
                          soltype=SimpleNamespace(itr=1), solsta=SimpleNamespace(optimal=1))
    config = dict(optimizer="intpnt", num_threads=1, intpnt_basis="never",
                  intpnt_starting_point="constant", intpnt_tol_pfeas=1e-7)
    adapter = bc.MosekAdapter(sdk, adapter_problem(), config)
    assert task.calls["putqobj"][0] == ([0, 1, 1], [0, 0, 1], [2., 0.5, 3.])
    assert task.calls["putconboundlist"][0] == ([0, 1, 2], [3, 4, 0], [-1., 0., 0.], [1., 0., 0.])
    assert task.calls["putvarboundlist"][0][1] == [0, 0]
    result = adapter.solve()
    assert result["optimal"] and np.array_equal(result["y"], [-2, 3, 0])
    assert adapter.effective["intpnt_starting_point"] == 0
    adapter.close()
    assert task.disposed


def test_native_cplex_model_full_half_quadratic_free_variables_row_duals():
    captured = {}
    objective = SimpleNamespace(sense=SimpleNamespace(minimize=1),
                                set_sense=lambda value: captured.update(sense=value),
                                set_quadratic=lambda value: captured.update(quadratic=value))
    model = SimpleNamespace(parameters=ParamNode(), objective=objective,
                            variables=SimpleNamespace(add=lambda **kw: captured.update(variables=kw)),
                            linear_constraints=SimpleNamespace(add=lambda **kw: captured.update(rows=kw)),
                            get_version=lambda: "22.1.2", solve=lambda: None, end=lambda: captured.update(closed=True),
                            set_log_stream=lambda value: None, set_error_stream=lambda value: None,
                            set_warning_stream=lambda value: None, set_results_stream=lambda value: None,
                            solution=SimpleNamespace(status=SimpleNamespace(optimal=1), get_status=lambda: 1,
                                                     get_values=lambda: [0.25, -0.25],
                                                     get_dual_values=lambda: [2, -3, 4]))
    sdk = SimpleNamespace(Cplex=lambda: model, infinity=1e20,
                          SparsePair=lambda **kw: kw)
    adapter = bc.CplexAdapter(sdk, adapter_problem(), dict(threads=1, advance=0))
    assert captured["quadratic"] == [dict(ind=[0, 1], val=[2., 0.5]), dict(ind=[0, 1], val=[0.5, 3.])]
    assert captured["variables"]["lb"] == [-1e20, -1e20]
    assert captured["rows"]["senses"] == "GLE"
    result = adapter.solve()
    assert np.array_equal(result["y"], [1., -4., 0.])
    assert model.parameters.advance.get() == 0
    adapter.close()
    assert captured["closed"]


class LinearExpression:
    def __init__(self, coefficients, variables):
        self.coefficients = coefficients

    def __eq__(self, bound):
        return ("E", self.coefficients, bound)

    def __ge__(self, bound):
        return ("G", self.coefficients, bound)

    def __le__(self, bound):
        return ("L", self.coefficients, bound)


class Symbol:
    def __init__(self, index):
        self.index = index
        self.X = 0.25 if index == 0 else -0.25

    def __rmul__(self, coefficient):
        return Term(coefficient, self.index)


class Term:
    def __init__(self, coefficient, index):
        self.coefficient, self.index = coefficient, index

    def __mul__(self, other):
        return self.coefficient, self.index, other.index


def test_native_gurobi_full_half_quadratic_reset_env_gate_row_duals():
    captured, params = {}, {}
    constraints = []
    terms = []

    def add_constraint(relation):
        constraint = SimpleNamespace(Pi=[2., -3., 4.][len(constraints)])
        constraints.append((relation, constraint))
        return constraint

    env = SimpleNamespace(setParam=lambda k, v: params.update({k: v}),
                          start=lambda: captured.update(started=True), dispose=lambda: None)
    model = SimpleNamespace(setParam=lambda k, v: params.update({k: v}),
                            getParamInfo=lambda k: (k, None, params[k]),
                            addVars=lambda n, **kw: {i: Symbol(i) for i in range(n)},
                            addConstr=add_constraint, setObjective=lambda *a: None,
                            update=lambda: None, reset=lambda v: captured.update(reset=v),
                            optimize=lambda: None, Status=1, dispose=lambda: captured.update(closed=True))
    sdk = SimpleNamespace(Env=lambda **kw: env, Model=lambda **kw: model,
                          GRB=SimpleNamespace(INFINITY=1e20, MINIMIZE=1, OPTIMAL=1),
                          LinExpr=LinearExpression, QuadExpr=lambda: SimpleNamespace(add=terms.append),
                          gurobi=SimpleNamespace(version=lambda: (13, 0, 3)))
    adapter = bc.GurobiAdapter(sdk, adapter_problem(), dict(Threads=1, Method=2))
    assert captured["started"] and params["OutputFlag"] == 0
    assert terms[1:] == [(1., 0, 0), (0.25, 0, 1), (0.25, 1, 0), (1.5, 1, 1)]
    assert [relation[0] for relation, _ in constraints] == ["G", "L", "E"]
    result = adapter.solve()
    assert captured["reset"] == 1 and np.array_equal(result["y"], [1., -4., 0.])
    adapter.close()
    assert captured["closed"]


@pytest.mark.parametrize("effective", [{}, {"Threads": "PRIVATE_SENTINEL"}, {"Threads": float("nan")}])
def test_effective_parameters_cannot_save_private_text(effective):
    with pytest.raises(ValueError):
        bc.checked_effective({"Threads": 1}, effective)


def test_argument_parser_never_echoes_private_unknown_values(capsys):
    with pytest.raises(SystemExit) as caught:
        bc.main(["--license", "PRIVATE_SENTINEL /private/license"])
    assert caught.value.code == 2
    captured = capsys.readouterr()
    assert "PRIVATE_SENTINEL" not in captured.out + captured.err
