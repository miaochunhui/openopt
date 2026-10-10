"""Portable synthetic controls; pinned local evidence is explicit opt-in only."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np
import pytest

from scripts import experiment_public_robustness as gate
from scripts import public_qp_corpus as corpus
from scripts import qualify_public_qp as runner

LP = b"""NAME ROBUSTNESS_FIXTURE
ROWS
 N COST
 L CAP
COLUMNS
 X COST -1 CAP 1
RHS
 RHS1 CAP 2
ENDATA
"""

REAL_V1 = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/real-v1/result.json")
REAL_V1_PIN = "aa584b62f97349f37afad438dfbd87320ace3408df4d7e574a0989370970e8e7"
BASELINE_PIN = "b10d49a32a13a9f34b8bf86ea6bd8a126dc779bf72a47db3f069420f653456ef"
EVIDENCE_ENV = "OPENOPT_RUN_PUBLIC_EVIDENCE_TESTS"
EVIDENCE_PATH_ENV = "OPENOPT_PUBLIC_ROBUSTNESS_V1_RESULT"
LOCAL_EVIDENCE = pytest.mark.skipif(
    os.environ.get(EVIDENCE_ENV) != "1",
    reason=f"local pinned evidence/setup probes require {EVIDENCE_ENV}=1",
)
P2_MUTATIONS = (
    ("osqp", "max_iter", 9999),
    ("osqp", "warm_starting", 1),
    ("clarabel", "max_iter", 199),
    ("clarabel", "max_threads", 2),
    ("clarabel", "presolve_enable", True),
)


def pinned_real_evidence() -> Path:
    """Opt-in missing/wrong bytes are errors, never skips or fabricated defaults."""
    if os.environ.get(EVIDENCE_ENV) != "1":
        raise ValueError(f"pinned local evidence requires {EVIDENCE_ENV}=1")
    result = Path(os.environ.get(EVIDENCE_PATH_ENV, str(REAL_V1)))
    if not result.is_file() or result.is_symlink():
        raise ValueError(f"opt-in pinned real-v1 result missing/unsafe: {result}")
    gate.exact(gate.digest(result), REAL_V1_PIN, "opt-in real-v1 SHA256")
    baseline = result.parent / "baseline/result.json"
    if not baseline.is_file() or baseline.is_symlink():
        raise ValueError(f"opt-in pinned baseline missing/unsafe: {baseline}")
    gate.exact(gate.digest(baseline), BASELINE_PIN, "opt-in baseline SHA256")
    return result


@pytest.fixture
def local_real_evidence():
    if os.environ.get(EVIDENCE_ENV) != "1":
        pytest.skip(f"local evidence requires {EVIDENCE_ENV}=1")
    return pinned_real_evidence()


@pytest.fixture(autouse=True)
def no_solver_calls_in_this_suite(monkeypatch):
    """All current P2/unit fixtures replay saved candidates or do setup only."""
    previous = gate.sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in ("solve_qp", "solve"):
            pytest.fail("no-solve unit contract violated")
        if previous is not None:
            previous(frame, event, arg)

    gate.sys.setprofile(profile)
    try:
        yield
    finally:
        gate.sys.setprofile(previous)


def reseal_rows(result, report):
    """Coordinate result, worker, row and progress edits; byte gates still agree."""
    for row in report["rows"]:
        folder = result.parent / "attempts" / row["case_id"]
        (folder / (row["arm"] + ".json")).write_text(json.dumps(row))
        worker = {key: value for key, value in row.items()
                  if key not in ("worker_exit_code", "worker_log")}
        (folder / (row["arm"] + "-worker-result.json")).write_text(json.dumps(worker))
    for path in result.parent.glob("progress-*.json"):
        progress = gate.read_json(path)
        count = int(path.stem.split("-")[1]) + 1
        progress["rows"][:count] = report["rows"][:count]
        path.write_text(json.dumps(progress))
    report["files"] = gate.inventory(result.parent)
    result.write_text(json.dumps(report))


def forbid_solves(monkeypatch):
    import osqp
    import clarabel
    import torch

    def forbidden(*args, **kwargs):
        pytest.fail("read-only P2 test called solve")

    monkeypatch.setattr(gate, "run_arm", forbidden)
    monkeypatch.setattr(osqp.OSQP, "solve", forbidden)
    monkeypatch.setattr(clarabel, "DefaultSolver", forbidden)
    # Frozen modules load their own solve_qp; profile guards that function too.
    previous = gate.sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in ("solve_qp", "solve"):
            pytest.fail("read-only P2 test entered a solver")
        if previous is not None:
            previous(frame, event, arg)

    monkeypatch.setattr(torch, "cholesky_solve", forbidden)
    gate.sys.setprofile(profile)
    return previous


@pytest.fixture
def real_metadata_test_copy(tmp_path, local_real_evidence):
    """Old vectors with fabricated v2 metadata ONLY for checker negative tests.

    This is not a v2 run, worker execution receipt or requalification of v1.
    """
    original = local_real_evidence
    result = tmp_path / "test-only-metadata-copy-not-a-run" / "result.json"
    shutil.copytree(original.parent, result.parent)
    report = gate.read_json(result)
    shutil.copyfile(Path(gate.__file__), result.parent / "code/producer.py")
    _, _, core = gate.load_dependencies(result.parent / "code")
    contracts = gate.settings_contract(core)
    runtime = gate.runtime_identity()
    report.update({"version": gate.VERSION, "producer_sha256": gate.digest(Path(gate.__file__)),
                   "settings_contract": contracts, "runtime": runtime,
                   "thread_contract": gate.THREAD_CONTRACT,
                   "test_fixture_kind": "test-only-metadata-copy-not-a-run",
                   "worker_execution_verified": False})
    for row in report["rows"]:
        settings = contracts[row["solver"]]
        row["raw"].update({
            "requested": deepcopy(settings["requested"]), "settings": deepcopy(settings["effective"]),
            "settings_before_solve": deepcopy(settings["effective"]),
            "settings_after_solve": deepcopy(settings["effective"]),
            "runtime": deepcopy(runtime), "threads": deepcopy(gate.THREAD_CONTRACT)})
    plan_path = result.parent / "plan.json"
    plan = gate.read_json(plan_path)
    plan.update({"version": gate.VERSION, "settings_contract": contracts,
                 "runtime": runtime, "thread_contract": gate.THREAD_CONTRACT})
    plan_path.write_text(json.dumps(plan))
    reseal_rows(result, report)
    return result


def coordinated_settings_reseal_rejected(result, monkeypatch, request, solver, key, value):
    report = gate.read_json(result)
    previous = forbid_solves(monkeypatch)
    try:
        assert gate.replay(result, gate.digest(result))["new_numerically_replayed"] == len(report["rows"])
    finally:
        gate.sys.setprofile(previous)
    for row in report["rows"]:
        if row["solver"] == solver:
            row["raw"]["requested"][key] = value
            row["raw"]["settings"][key] = value
            row["raw"]["settings_before_solve"][key] = value
            row["raw"]["settings_after_solve"][key] = value
    report["settings_contract"][solver]["requested"][key] = value
    report["settings_contract"][solver]["effective"][key] = value
    plan_path = result.parent / "plan.json"
    plan = gate.read_json(plan_path)
    plan["settings_contract"] = report["settings_contract"]
    plan_path.write_text(json.dumps(plan))
    reseal_rows(result, report)
    request.node.user_properties.append(("p2_mutation_result", str(result)))
    request.node.user_properties.append(("p2_mutation_pin", gate.digest(result)))
    previous = forbid_solves(monkeypatch)
    try:
        with pytest.raises(ValueError, match="settings"):
            gate.replay(result, gate.digest(result))
    finally:
        gate.sys.setprofile(previous)


@pytest.mark.parametrize("solver,key,value", P2_MUTATIONS)
def test_p2_coordinated_settings_reseal_rejected(
    experiment, monkeypatch, request, solver, key, value,
):
    coordinated_settings_reseal_rejected(experiment, monkeypatch, request, solver, key, value)


@LOCAL_EVIDENCE
@pytest.mark.parametrize("solver,key,value", P2_MUTATIONS)
def test_optin_real_metadata_only_coordinated_reseal_rejected(
    real_metadata_test_copy, local_real_evidence, monkeypatch, request, solver, key, value,
):
    report = gate.read_json(real_metadata_test_copy)
    assert report["test_fixture_kind"] == "test-only-metadata-copy-not-a-run"
    assert report["worker_execution_verified"] is False
    coordinated_settings_reseal_rejected(
        real_metadata_test_copy, monkeypatch, request, solver, key, value)
    assert gate.digest(local_real_evidence) == REAL_V1_PIN


@LOCAL_EVIDENCE
def test_p2_capture_default_fields_without_solve(tmp_path, request):
    import osqp
    import clarabel
    from scipy import sparse

    workspace = osqp.OSQP()
    workspace.setup(P=sparse.csc_matrix([[1.0]]), q=np.zeros(1),
                    A=sparse.csc_matrix([[1.0]]), l=np.zeros(1), u=np.ones(1), verbose=False)
    def fields(settings):
        values = {}
        for key in dir(settings):
            if key.startswith("_") or callable(getattr(settings, key)):
                continue
            value = getattr(settings, key)
            if hasattr(value, "value"):
                value = value.value
            if isinstance(value, float) and not np.isfinite(value):
                value = str(value)
            values[key] = value
        return values

    values = {"osqp": fields(workspace.settings), "clarabel": fields(clarabel.DefaultSettings()),
              "python": gate.sys.version.split()[0],
              "dependencies": {name: gate.importlib.metadata.version(name)
                               for name in ("numpy", "scipy", "torch", "osqp", "clarabel")}}
    path = tmp_path / "settings-defaults.json"
    corpus.write_json(path, values)
    assert gate.read_json(path) == values
    request.node.user_properties.append(("p2_defaults", str(path)))


@LOCAL_EVIDENCE
def test_p2_full_effective_settings_observed_at_setup_without_solve(tmp_path, request, monkeypatch):
    import osqp
    import clarabel
    from scipy import sparse

    def forbidden(*args, **kwargs):
        pytest.fail("setup-only contract probe attempted solve")

    monkeypatch.setattr(osqp.OSQP, "solve", forbidden)
    runtime = gate.runtime_identity()
    workspace = osqp.OSQP(algebra="builtin")
    workspace.setup(P=sparse.csc_matrix([[1.0]]), q=np.zeros(1),
                    A=sparse.csc_matrix([[1.0]]), l=np.zeros(1), u=np.ones(1),
                    **gate.OSQP_REQUESTED)
    osqp_settings = gate.settings_snapshot(workspace.settings)
    gate.exact(osqp_settings, gate.OSQP_EFFECTIVE, "OSQP setup settings")
    settings = clarabel.DefaultSettings()
    for key, value in gate.CLARABEL_EFFECTIVE.items():
        setattr(settings, key, float("inf") if value == "Infinity" else value)
    probe = clarabel.DefaultSolver(sparse.csc_matrix([[1.0]]), np.zeros(1),
                                  sparse.csc_matrix([[1.0]]), np.ones(1),
                                  [clarabel.NonnegativeConeT(1)], settings)
    observed = gate.settings_snapshot(probe.get_settings())
    gate.exact(observed, gate.CLARABEL_EFFECTIVE, "Clarabel setup settings")
    assert len(osqp_settings) == 28 and len(observed) == 45
    receipt = {"kind": "setup-only-not-a-solve", "runtime": runtime,
               "osqp": osqp_settings, "clarabel": observed, "solve_calls": 0}
    path = tmp_path / "p2-setup-contract.json"
    corpus.write_json(path, receipt)
    assert gate.read_json(path) == receipt
    request.node.user_properties.append(("p2_setup_evidence", str(path)))


@pytest.mark.parametrize("solver", ["osqp", "clarabel", "openopt"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "type", "requested-only", "effective-only",
                                     "before-only", "after-only"])
def test_p2_settings_exact_membership_types_and_each_stage(
    experiment, monkeypatch, solver, mutation,
):
    result = experiment
    report = gate.read_json(result)
    row = next(row for row in report["rows"] if row["solver"] == solver)
    raw = row["raw"]
    key = "solver" if solver == "openopt" else "max_iter"
    if mutation == "missing":
        raw["settings"].pop(key)
    elif mutation == "extra":
        raw["settings"]["unreviewed_default"] = 1
    elif mutation == "type":
        if solver == "openopt":
            raw["settings"]["solver"]["max_iterations"] = 10000.0
        else:
            raw["settings"]["max_iter"] = float(raw["settings"]["max_iter"])
    else:
        stage = {"requested-only": "requested", "effective-only": "settings",
                 "before-only": "settings_before_solve", "after-only": "settings_after_solve"}[mutation]
        if solver == "openopt":
            raw[stage]["solver"]["max_iterations"] -= 1
        else:
            raw[stage]["max_iter"] -= 1
    reseal_rows(result, report)
    previous = forbid_solves(monkeypatch)
    try:
        with pytest.raises(ValueError, match="settings"):
            gate.replay(result, gate.digest(result))
    finally:
        gate.sys.setprofile(previous)


@pytest.mark.parametrize("mutation", ["version", "module", "thread-env", "torch-thread"])
def test_p2_runtime_and_thread_coordinated_reseal_rejected(
    experiment, monkeypatch, mutation,
):
    result = experiment
    report = gate.read_json(result)
    if mutation == "version":
        report["runtime"]["packages"]["osqp"] = "1.0.5"
    elif mutation == "module":
        report["runtime"]["modules"]["osqp.ext_builtin"]["sha256"] = "0" * 64
    elif mutation == "thread-env":
        report["thread_contract"]["environment"]["OMP_NUM_THREADS"] = "2"
    else:
        report["thread_contract"]["torch_intraop"] = 2
    for row in report["rows"]:
        row["raw"]["runtime"] = deepcopy(report["runtime"])
        row["raw"]["threads"] = deepcopy(report["thread_contract"])
    plan_path = result.parent / "plan.json"
    plan = gate.read_json(plan_path)
    plan.update({"runtime": report["runtime"], "thread_contract": report["thread_contract"]})
    plan_path.write_text(json.dumps(plan))
    reseal_rows(result, report)
    previous = forbid_solves(monkeypatch)
    try:
        with pytest.raises(ValueError, match="runtime|thread"):
            gate.replay(result, gate.digest(result))
    finally:
        gate.sys.setprofile(previous)


@LOCAL_EVIDENCE
def test_p2_v1_is_not_promoted_or_relabelled(monkeypatch, local_real_evidence):
    previous = forbid_solves(monkeypatch)
    try:
        with pytest.raises(ValueError, match="unsupported experiment"):
            gate.replay(local_real_evidence, REAL_V1_PIN)
    finally:
        gate.sys.setprofile(previous)
    assert gate.read_json(local_real_evidence)["version"] == "public-robustness-v1"
    assert gate.digest(local_real_evidence) == REAL_V1_PIN


@pytest.mark.parametrize("package", tuple(gate.PACKAGE_VERSIONS))
def test_p2_wrong_installed_version_fails_before_worker_execution(monkeypatch, package):
    """All metadata are controlled; no dependence on installed exact versions."""
    monkeypatch.setattr(gate.importlib.metadata, "version",
                        lambda name: "unsupported-version" if name == package
                        else gate.PACKAGE_VERSIONS[name])
    with pytest.raises(ValueError, match=f"runtime versions.{package}"):
        gate.runtime_identity()


def test_p2_wrong_python_version_fails_before_module_probes(monkeypatch):
    monkeypatch.setattr(gate.importlib.metadata, "version", gate.PACKAGE_VERSIONS.__getitem__)
    monkeypatch.setattr(gate.sys, "version", "0.0.0 synthetic")
    with pytest.raises(ValueError, match="runtime Python version"):
        gate.runtime_identity()


def test_p2_wrong_worker_thread_environment_fails(monkeypatch):
    import torch

    for key, value in gate.THREAD_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 1)
    monkeypatch.setattr(torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with pytest.raises(ValueError, match="runtime threads.environment"):
        gate.worker_threads()


@LOCAL_EVIDENCE
def test_optin_observed_worker_thread_contract_without_solve(tmp_path, request):
    import torch

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    receipt = {"kind": "local-thread-probe-not-a-worker-execution",
               "runtime": gate.runtime_identity(), "threads": gate.worker_threads(),
               "solve_calls": 0}
    path = tmp_path / "local-thread-probe.json"
    corpus.write_json(path, receipt)
    gate.exact(gate.read_json(path), receipt, "local thread probe JSON readback")
    request.node.user_properties.append(("p2_thread_probe", str(path)))


@pytest.mark.parametrize("mutation", ["disabled", "missing", "wrong-bytes"])
def test_optin_evidence_preflight_fail_loud_without_real_files(tmp_path, monkeypatch, mutation):
    path = tmp_path / "absent-result.json"
    monkeypatch.setenv(EVIDENCE_ENV, "0" if mutation == "disabled" else "1")
    monkeypatch.setenv(EVIDENCE_PATH_ENV, str(path))
    if mutation == "wrong-bytes":
        path.write_bytes(b'{"kind":"synthetic-wrong-pin"}')
    with pytest.raises(ValueError, match="requires|missing/unsafe|SHA256"):
        pinned_real_evidence()


@pytest.mark.parametrize("solver,key,value", P2_MUTATIONS)
def test_p2_coordinated_worker_rows_without_contract_edit_rejected(
    experiment, solver, key, value,
):
    result = experiment
    report = gate.read_json(result)
    for row in report["rows"]:
        if row["solver"] == solver:
            for stage in ("requested", "settings", "settings_before_solve", "settings_after_solve"):
                row["raw"][stage][key] = value
    reseal_rows(result, report)
    with pytest.raises(ValueError, match="settings"):
        gate.replay(result, gate.digest(result))


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "SHARED_TMP", tmp_path.resolve())
    monkeypatch.setattr(runner, "SHARED_TMP", tmp_path.resolve())
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "cheap.mps").write_bytes(LP)
    manifest = tmp_path / "manifest.json"
    corpus.write_json(manifest, {"schema_version": 1, "cases": [{
        "case_id": "cheap", "path": "cheap.mps", "input_sha256": gate.digest(inputs / "cheap.mps"),
        "source_url": "https://example.invalid/synthetic-fixture.mps",
        "license": "synthetic unit fixture, not public coverage", "origin": "synthetic",
    }]})
    frozen = tmp_path / "frozen"
    corpus.freeze(manifest, inputs, frozen)

    def saved_bad_candidate(model, solver, values, core):
        # Deliberately negative, no solve in baseline fixture generation.
        return {"status": "Solved" if solver == "clarabel" else "solved", "iterations": 1,
                "settings": {"synthetic_saved_candidate": True}, "observed_device": "cpu",
                "vectors": {"x": np.asarray([1.0]), "y": np.asarray([0.0, 0.0]),
                            "reported_objective": np.asarray(-1.0)}}

    monkeypatch.setattr(runner, "run_solver", saved_bad_candidate)
    args = runner.parser().parse_args([
        "run", "--corpus", str(frozen), "--manifest-sha256", gate.digest(frozen / "corpus.json"),
        "--artifact-dir", str(tmp_path / "baseline"), "--source-root", str(gate.ROOT),
    ])
    result, code = runner.qualify(args)
    assert code == 1 and gate.read_json(result)["verdict"] == "UNQUALIFIED"
    assert gate.baseline_check(result, gate.digest(result))["rows"][0]["attempted"]
    return result


@pytest.fixture
def experiment(baseline, tmp_path, monkeypatch):
    import torch

    # This double exercises serialization/replay, NOT the real runtime gate.
    runtime = {"kind": "synthetic-runtime-double-not-an-execution-receipt",
               "python": "synthetic", "packages": {"synthetic": "not-installed"},
               "modules": {"osqp.ext_builtin": {"sha256": "synthetic-not-native-bytes"}}}
    monkeypatch.setattr(gate, "runtime_identity", lambda: deepcopy(runtime))
    monkeypatch.setattr(gate, "worker_threads", lambda: deepcopy(gate.THREAD_CONTRACT))
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)

    def saved_candidate(model, arm, core, saved_runner):
        """Planted unit KKT vectors, never a backend execution receipt."""
        import torch

        view = corpus.solver_arrays(model)
        x, y = np.asarray([2.0]), np.asarray([1.0, 0.0])
        evidence = core.checker.check_qp_candidate(
            saved_runner.problem(view), x, y, atol=1e-7, rtol=1e-7).to_dict()
        assert evidence["passed"]
        contract = gate.settings_contract(core)[arm["solver"]]
        raw = {"status": "Solved" if arm["solver"] == "clarabel" else "solved",
               "iterations": 1, "observed_device": "cpu",
               "requested": deepcopy(contract["requested"]),
               "settings": deepcopy(contract["effective"]),
               "settings_before_solve": deepcopy(contract["effective"]),
               "settings_after_solve": deepcopy(contract["effective"]),
               "vectors": {"x": x, "y": y, "reported_objective": np.asarray(-2.0)}}
        extra = {}
        if arm["solver"] == "openopt":
            config, scaling, _ = gate.openopt_configs(core)
            prepared = core.scaling.equilibrate_qp(gate.tensor_problem(view, core.qp), config, scaling)
            scaled_x, scaled_y = prepared.scaling.to_scaled(torch.tensor(x), torch.tensor(y))
            extra = {"scaled_x": gate.numpy_tensor(scaled_x), "scaled_y": gate.numpy_tensor(scaled_y),
                     "restored_x": x.copy(), "restored_y": y.copy(),
                     "variable_scale": gate.numpy_tensor(prepared.scaling.variable_scale),
                     "constraint_scale": gate.numpy_tensor(prepared.scaling.constraint_scale),
                     "objective_scale": np.asarray(prepared.scaling.objective_scale)}
            raw["refinement"] = {"status": "rejected", "reason": "synthetic_saved_control",
                                 "baseline_evidence": evidence, "returned_evidence": evidence,
                                 "attempted_evidence": None}
            raw["before_refinement_assessment"] = saved_runner.assess(
                model, "openopt", raw, gate.GATES, core.checker)
        elif arm["solver"] == "clarabel":
            import clarabel

            _, _, _, mapping = saved_runner.clarabel_form(view, clarabel)
            z = np.zeros(len(mapping))
            z[next(i for i, item in enumerate(mapping) if item == (0, 1.0))] = 1.0
            extra["cone_z"] = z
            raw["dual_mapping"] = [list(item) for item in mapping]
        return raw, extra

    monkeypatch.setattr(gate, "run_arm", saved_candidate)

    def inline_worker(command, stdout, stderr, timeout, check, env):
        assert command[3] == "worker" and timeout <= 180 and check is False
        for key, value in gate.THREAD_ENV.items():
            assert env[key] == value
            monkeypatch.setenv(key, value)
        index = int(command[-1])
        code = gate.worker(Path(command[-3]), index)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(gate.subprocess, "run", inline_worker)
    result, code = gate.run(baseline, gate.digest(baseline), tmp_path / "experiment", 180)
    assert code == 0
    return result


def test_fixed_arms_original_gates_and_one_attempt(baseline):
    report = gate.baseline_check(baseline, gate.digest(baseline))
    assert len(gate.ARMS) == 3
    assert gate.GATES == {"eps_abs": 1e-7, "eps_rel": 1e-7,
                          "direction_atol": 1e-8, "direction_rtol": 1e-8}
    assert all(arm["solve_eps"] < gate.GATES["eps_abs"] for arm in gate.ARMS)
    assert all(row["attempt"] == row["denominator"] == 1 for row in report["rows"])


def test_cheap_three_backend_saved_candidates_and_readonly_replay(experiment, monkeypatch, request):
    before = {key: gate.digest(gate.member(experiment.parent, key))
              for key in gate.read_json(experiment)["files"]}
    pin = gate.digest(experiment)
    report = gate.read_json(experiment)
    assert len(report["rows"]) == 3
    assert report["elapsed_seconds"] < 180
    assert all(row["input_model_before"] == row["input_model_after"] == row["model_sha256"]
               for row in report["rows"])
    assert report["baseline_rows"][0]["verdict"] == "UNQUALIFIED"

    def forbidden(*args, **kwargs):
        pytest.fail("read-only replay attempted solve")

    monkeypatch.setattr(gate, "run_arm", forbidden)
    replay = gate.replay(experiment, pin)
    assert replay["baseline_replayed"] == replay["new_numerically_replayed"] == 3
    assert replay["original_verdict"] == "UNQUALIFIED"
    assert before == {key: gate.digest(gate.member(experiment.parent, key)) for key in before}
    request.node.user_properties.append(("robustness_unit_json", str(experiment)))
    request.node.user_properties.append(("robustness_unit_sha256", pin))


@pytest.mark.parametrize("mutation", ["caller-pin", "vectors", "gate", "denominator",
                                     "baseline", "claim", "inventory", "restore"])
def test_negative_controls_reject_resealed_or_original_tamper(experiment, mutation):
    pin = gate.digest(experiment)
    report = gate.read_json(experiment)
    if mutation == "caller-pin":
        pin = "0" * 64
    elif mutation == "vectors":
        path = experiment.parent / report["rows"][0]["vectors_path"]
        path.write_bytes(path.read_bytes() + b"bad")
    elif mutation == "inventory":
        (experiment.parent / "unlisted.json").write_bytes(b"{}")
    elif mutation == "restore":
        path = experiment.parent / report["rows"][0]["transform_path"]
        extra = gate.read_extra(path)
        extra["restored_y"][0] += 0.1
        with path.open("wb") as stream:
            np.savez(stream, **extra)
        report["files"][str(path.relative_to(experiment.parent))] = {
            "bytes": path.stat().st_size, "sha256": gate.digest(path)}
        experiment.write_text(json.dumps(report))
        pin = gate.digest(experiment)
    else:
        if mutation == "gate":
            report["gates"]["eps_abs"] = 1e-3
        elif mutation == "denominator":
            report["planned_denominator"] -= 1
        elif mutation == "baseline":
            report["baseline_rows"][0]["verdict"] = "QUALIFIED"
        else:
            report["tiny_dual_clipping"] = True
        experiment.write_text(json.dumps(report))
        pin = gate.digest(experiment)
    with pytest.raises(ValueError):
        gate.replay(experiment, pin)


def test_empty_missing_or_wrong_baseline_refused(baseline):
    with pytest.raises(ValueError, match="caller"):
        gate.baseline_check(baseline, "0" * 64)
    report = gate.read_json(baseline)
    report["config"]["eps_abs"] = 1e-6
    baseline.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="original gate"):
        gate.baseline_check(baseline, gate.digest(baseline))


@pytest.mark.parametrize("budget", [0, -1, 181, float("nan")])
def test_invalid_cpu_budget_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="budget"):
        gate.run(tmp_path / "absent", "0" * 64, tmp_path / "out", budget)


def test_budget_exhaustion_has_full_failure_denominator(baseline, tmp_path, monkeypatch):
    def exhausted(*args, **kwargs):
        raise gate.subprocess.TimeoutExpired("synthetic budget control", 0.01)

    monkeypatch.setattr(gate.subprocess, "run", exhausted)
    monkeypatch.setattr(gate, "runtime_identity",
                        lambda: {"kind": "synthetic-budget-control-not-runtime-evidence"})
    result, code = gate.run(baseline, gate.digest(baseline), tmp_path / "exhausted", 180)
    report = gate.read_json(result)
    assert code == 2
    assert report["planned_denominator"] == 3
    assert all(row["verdict"] == "BUDGET_EXHAUSTED" for row in report["rows"])
    assert all(count["denominator"] == count["BUDGET_EXHAUSTED"] == 1
               for count in report["arm_counts"].values())
    assert gate.replay(result, gate.digest(result))["new_numerically_replayed"] == 0


def test_failed_worker_is_not_skipped_or_retried(baseline, tmp_path, monkeypatch):
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(gate.subprocess, "run", failed)
    monkeypatch.setattr(gate, "runtime_identity",
                        lambda: {"kind": "synthetic-failed-worker-control-not-runtime-evidence"})
    result, code = gate.run(baseline, gate.digest(baseline), tmp_path / "failed", 180)
    report = gate.read_json(result)
    assert code == 2 and len(calls) == 3
    assert all(row["verdict"] == "FAIL" and row["attempt"] == 1 for row in report["rows"])
    assert gate.replay(result, gate.digest(result))["planned_denominator"] == 3


def test_no_support_for_tiny_wrong_sign_or_looser_component_gate():
    core = runner.load_core(gate.ROOT / "src/openopt")
    problem = SimpleNamespace(P=np.zeros((1, 1)), q=np.zeros(1),
                              A=np.ones((1, 1)), l=np.zeros(1), u=np.full(1, np.inf))
    evidence = core.checker.check_qp_candidate(
        problem, np.zeros(1), np.asarray([1e-17]), atol=1e-7, rtol=1e-7).to_dict()
    assert not evidence["passed"] and not evidence["checks"]["bound_sign"]
    assert evidence["thresholds"]["bound_sign"] == 0
    assert evidence["metrics"]["dual_objective"] is None


def test_json_duplicate_and_artifact_symlink_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="duplicate"):
        gate.read_json(path)
    (tmp_path / "link").symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        gate.member(tmp_path.resolve(), "link")


def test_accepted_refinement_cannot_upgrade_iteration_exhaustion(baseline, monkeypatch):
    core = runner.load_core(gate.ROOT / "src/openopt")
    package = core.qp.__package__
    core.scaling = runner.load_module(gate.ROOT / "src/openopt/scaling.py", package + ".scaling")
    core.refinement = runner.load_module(gate.ROOT / "src/openopt/refinement.py", package + ".refinement")
    model = runner.load_model(baseline.parent / "frozen", "cheap")
    import torch

    refined = core.refinement.refine_qp_candidate(
        gate.tensor_problem(corpus.solver_arrays(model), core.qp),
        torch.tensor([2.0]), torch.tensor([1.0, 0.0]), gate.openopt_configs(core)[2])
    assert refined.status == "accepted"
    raw = {"status": "max_iterations", "iterations": 10000, "observed_device": "cpu",
           "settings": gate.settings_contract(core)["openopt"]["effective"],
           "vectors": {"x": gate.numpy_tensor(refined.x), "y": gate.numpy_tensor(refined.y),
                       "reported_objective": np.asarray(-2.0)}}
    assessment = runner.assess(model, "openopt", raw, gate.GATES, core.checker)
    assert assessment["candidate"]["passed"]
    assert raw["status"] == "max_iterations" and assessment["verdict"] == "UNQUALIFIED"


def test_source_membership_tamper_cannot_be_resealed(experiment):
    report = gate.read_json(experiment)
    report["source_hashes"].pop("src/openopt/scaling.py")
    experiment.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source inventory"):
        gate.replay(experiment, gate.digest(experiment))


def test_existing_output_is_never_overwritten(baseline, tmp_path):
    output = tmp_path / "owned"
    output.mkdir()
    marker = output / "other-owner.txt"
    marker.write_text("retain existing bytes")
    with pytest.raises(FileExistsError):
        gate.run(baseline, gate.digest(baseline), output)
    assert marker.read_text() == "retain existing bytes"


def replay_receipt(result: Path, pin: str, output: Path) -> dict:
    """Persist a read-only replay receipt outside the sealed experiment."""
    output = runner.external(output)
    if output.is_relative_to(result.resolve().parent):
        raise ValueError("receipt must not alter sealed experiment inventory")
    measured = gate.replay(result, pin)
    report = gate.read_json(result)
    receipt = {
        "schema_version": 1, "kind": "public-robustness-replay-receipt-v2",
        "result_path": str(result.resolve()), "result_sha256": pin,
        "baseline_sha256": report["baseline_sha256"], "corpus_sha256": report["corpus_sha256"],
        "public_acceptance_sha256": report["public_acceptance_sha256"],
        "source_hashes": report["source_hashes"], "replay": measured,
        "runtime": report["runtime"], "thread_contract": report["thread_contract"],
        "settings_contract": report["settings_contract"],
        "license_verified": False, "public_redistribution": False,
        "qualification_upgraded": False,
        "rows": [{
            "case_id": row["case_id"], "arm": row["arm"], "verdict": row["verdict"],
            "attempt": row["attempt"], "denominator": row["denominator"],
            "assessment": row.get("assessment"),
        } for row in report["rows"]],
    }
    corpus.write_json(output, receipt)
    if gate.read_json(output) != receipt:
        raise ValueError("replay receipt JSON roundtrip mismatch")
    return receipt


def test_external_replay_receipt_roundtrip_and_sealed_inventory(experiment, tmp_path):
    before = gate.inventory(experiment.parent)
    pin = gate.digest(experiment)
    output = tmp_path / "receipt.json"
    receipt = replay_receipt(experiment, pin, output)
    assert gate.read_json(output) == receipt
    assert before == gate.inventory(experiment.parent)
    assert not receipt["qualification_upgraded"] and receipt["replay"]["baseline_replayed"] == 3
    with pytest.raises(ValueError, match="sealed"):
        replay_receipt(experiment, pin, experiment.parent / "extra.json")
    with pytest.raises(FileExistsError):
        replay_receipt(experiment, pin, output)


def no_solve_units(artifact_root: Path, keyword: str | None = None) -> dict:
    """Run this suite directly: no general verifier's unrelated solve example."""
    root = runner.external(artifact_root)
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="p2-unit-", dir=root))
    paths = (Path(gate.__file__), Path(__file__), gate.ROOT / "docs/research/public-robustness.md")
    before = {str(path): gate.digest(path) for path in paths}
    evidence_enabled = os.environ.get(EVIDENCE_ENV) == "1"
    original = None
    report = {"kind": "p2-no-solve-unit-verification-v2", "started_at": gate.now(),
              "source_before": before, "public_evidence_opt_in": evidence_enabled,
              "original_result_sha256": None, "baseline_sha256": None,
              "public_evidence_status": "NOT_REQUESTED", "verdict": "RUNNING", "solver_calls": 0}
    old_profile = gate.sys.getprofile()
    calls = []

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in ("solve_qp", "solve"):
            calls.append(f"{frame.f_code.co_filename}:{frame.f_code.co_name}")
            raise RuntimeError("P2 unit entry attempted solve")

    gate.sys.setprofile(profile)
    try:
        if evidence_enabled:
            original = pinned_real_evidence()
            report.update({"original_result_sha256": gate.digest(original),
                           "baseline_sha256": gate.digest(original.parent / "baseline/result.json"),
                           "public_evidence_status": "METADATA_ONLY_NOT_EXECUTION"})
        with (output / "pytest.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
            command = [
                "-q", "-ra", "-o", f"cache_dir={output / 'pytest-cache'}",
                "--basetemp", str(output / "pytest-tmp"), f"--junitxml={output / 'pytest.xml'}",
                str(Path(__file__))]
            if keyword is not None:
                command.extend(["-k", keyword])
            code = pytest.main(command)
        report["exit_code"] = int(code)
        suites = ElementTree.parse(output / "pytest.xml").getroot().findall("testsuite")
        counts = {key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        report["tests"] = counts
        report["source_after"] = {str(path): gate.digest(path) for path in paths}
        assert report["source_after"] == before
        if original is not None:
            assert gate.digest(original) == REAL_V1_PIN == report["original_result_sha256"]
            assert gate.digest(original.parent / "baseline/result.json") == BASELINE_PIN
        assert not calls and code == 0 and counts["tests"] > counts["skipped"]
        assert counts["failures"] == counts["errors"] == 0
        if evidence_enabled:
            assert counts["skipped"] == 0
        report["passed"] = counts["tests"] - counts["skipped"]
        report["verdict"] = "PASS"
    except Exception as error:
        report.update({"verdict": "FAIL", "error": {"type": type(error).__name__,
                                                  "message": str(error)}})
        raise
    finally:
        gate.sys.setprofile(old_profile)
        report["solver_calls"] = len(calls)
        report["finished_at"] = gate.now()
        corpus.write_json(output / "verification.json", report)
        gate.exact(gate.read_json(output / "verification.json"), report, "verification JSON readback")
        print("P2_UNIT " + json.dumps({
            "verification": str(output / "verification.json"),
            "sha256": gate.digest(output / "verification.json"),
            "verdict": report["verdict"], "solver_calls": len(calls),
            "tests": report.get("tests")}, sort_keys=True))
    return report


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Read-only pinned robustness replay with external receipt")
    cli.add_argument("--check-result", type=Path)
    cli.add_argument("--sha256")
    cli.add_argument("--output", type=Path)
    cli.add_argument("--unit-artifact-dir", type=Path)
    cli.add_argument("--pytest-keyword", help="optional selected unit subset; actual counts retained")
    args = cli.parse_args()
    if args.unit_artifact_dir is not None:
        if any(value is not None for value in (args.check_result, args.sha256, args.output)):
            cli.error("unit and replay arguments are mutually exclusive")
        no_solve_units(args.unit_artifact_dir, args.pytest_keyword)
    else:
        if args.pytest_keyword is not None:
            cli.error("--pytest-keyword requires --unit-artifact-dir")
        if any(value is None for value in (args.check_result, args.sha256, args.output)):
            cli.error("replay requires --check-result, --sha256 and --output")
        receipt = replay_receipt(args.check_result, args.sha256, args.output)
        print("PUBLIC_ROBUSTNESS_RECEIPT " + json.dumps({
            "receipt": str(args.output), "receipt_sha256": gate.digest(args.output),
            "replay": receipt["replay"]}, sort_keys=True))
