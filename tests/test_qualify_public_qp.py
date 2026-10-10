"""Synthetic MPS format/function controls, never public or industrial evidence."""

import json
import copy
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

from scripts import public_qp_corpus as corpus
from scripts import qualify_public_qp as runner


MPS = """NAME FUNCTION
ROWS
 N OBJ
 E EQ
 L CAP
 G FLOOR
COLUMNS
 X OBJ -3 EQ 1
 X CAP 2 FLOOR 1
 Y OBJ 4 EQ -1
 Y CAP 1 FLOOR 2
RHS
 RHS1 OBJ -7 EQ 1
 RHS1 CAP 8 FLOOR -2
BOUNDS
 LO BND X -2
 UP BND X 3
 FR BND Y
QUADOBJ
 X X 4
 X Y 1
 Y Y 2
ENDATA
"""


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    assert tmp_path.resolve().is_relative_to(runner.SHARED_TMP)
    local = tmp_path / "local"
    local.mkdir()
    source = local / "function.mps"
    source.write_text(MPS, encoding="ascii")
    manifest = tmp_path / "input.json"
    corpus.write_json(manifest, {"schema_version": 1, "cases": [{
        "case_id": "function", "path": source.name,
        "source_url": "https://example.invalid/synthetic-function.mps",
        "license": "0BSD synthetic hand-authored function fixture",
        "input_sha256": corpus.sha256(source.read_bytes()), "origin": "synthetic",
    }]})
    output = tmp_path / "frozen"
    corpus.freeze(manifest, local, output)
    return tmp_path, output, runner.digest(output / "corpus.json")


def arguments(frozen, name="run", extra=()):
    root, source, pin = frozen
    return runner.parser().parse_args([
        "run", "--corpus", str(source), "--manifest-sha256", pin,
        "--artifact-dir", str(root / name), *extra,
    ])


def execute(frozen, name="run", extra=()):
    path, code = runner.qualify(arguments(frozen, name, extra))
    return path, code, corpus.read_json(path)


def test_real_cpu_backends_original_bounds_offset_sameoracle_and_replay(frozen, request):
    path, code, report = execute(frozen)
    assert code == 1 and report["verdict"] == "UNQUALIFIED"
    assert report["numeric_verdict"] == "PASS"
    assert report["public_qualified_count"] == report["industrial_qualified_count"] == 0
    assert report["planned_denominator"] == report["attempted_count"] == 3
    assert not report["automatic_retry"] and not report["license_claim"] and not report["speedup_claim"]
    assert all(report["config"][key] for key in
               ("dual_gap_termination", "direction_termination", "adaptive_rho"))
    for row in report["rows"]:
        assert row["assessment"]["strict_status"] and row["assessment"]["candidate"]["passed"]
        assert row["assessment"]["candidate"]["checks"]["duality_gap"]
        assert row["raw"]["observed_device"] == "cpu"
        objective = row["assessment"]["objective"]
        assert objective["sense"] == "minimize" and objective["offset"] == 7
        assert objective["primal_original"] == pytest.approx(3.75, abs=1e-6)
        assert objective["dual_original"] == pytest.approx(3.75, abs=1e-6)
        assert row["raw"]["settings"]["max_iterations" if row["solver"] == "openopt" else "max_iter"] == 5000
        assert report["backend_counts"][row["solver"]]["denominator"] == 1
        assert report["backend_counts"][row["solver"]]["QUALIFIED"] == 1
    source = frozen[1] / "function"
    saved = path.parent / "frozen" / "function"
    for filename in ("raw.mps", "arrays.npz", "metadata.json"):
        assert (saved / filename).read_bytes() == (source / filename).read_bytes()
    pin = runner.digest(path)
    replay = runner.replay(path, pin)
    assert replay["numerically_replayed"] == 3 and replay["qualification_verdict"] == "UNQUALIFIED"
    request.node.user_properties.extend([("result_path", str(path)), ("result_sha256", pin)])


def test_maximization_preserves_original_sense_offset_and_dual(frozen):
    root, _, _ = frozen
    text = MPS.replace("ROWS", "OBJSENSE\n MAX\nROWS", 1)
    text = text.replace("OBJ -3", "OBJ 3").replace("OBJ 4", "OBJ -4")
    text = text.replace("X X 4", "X X -4").replace("X Y 1", "X Y -1").replace("Y Y 2", "Y Y -2")
    local = root / "max-local"
    local.mkdir()
    raw = local / "max.mps"
    raw.write_text(text, encoding="ascii")
    entry = corpus.read_json(root / "input.json")["cases"][0]
    entry.update(path=raw.name, input_sha256=runner.digest(raw))
    manifest = root / "max-input.json"
    corpus.write_json(manifest, {"schema_version": 1, "cases": [entry]})
    directory = root / "max-frozen"
    corpus.freeze(manifest, local, directory)
    path, code, report = execute((root, directory, runner.digest(directory / "corpus.json")), "max-run")
    assert code == 1 and report["numeric_verdict"] == "PASS"
    for row in report["rows"]:
        obj = row["assessment"]["objective"]
        assert obj["sense"] == "maximize"
        assert obj["primal_original"] == pytest.approx(10.25, abs=1e-6)
        assert obj["dual_original"] == pytest.approx(10.25, abs=1e-6)
    assert runner.replay(path, runner.digest(path))["numerically_replayed"] == 3


@pytest.mark.parametrize("mutation", ["status", "candidate", "reported-objective", "missing-dual"])
def test_false_success_negative_controls(frozen, monkeypatch, mutation):
    original = runner.run_solver

    def corrupt(model, solver, values, core):
        raw = original(model, solver, values, core)
        if solver == "openopt":
            if mutation == "status":
                raw["status"] = "solved inaccurate"
            elif mutation == "candidate":
                raw["vectors"]["x"] = np.zeros_like(raw["vectors"]["x"])
            elif mutation == "reported-objective":
                raw["vectors"]["reported_objective"] = np.asarray(0.0)
            else:
                raw["vectors"]["y"] = np.zeros_like(raw["vectors"]["y"])
        return raw

    monkeypatch.setattr(runner, "run_solver", corrupt)
    path, code, report = execute(frozen, mutation)
    assert code == 1 and report["verdict"] == "UNQUALIFIED"
    counts = report["backend_counts"]["openopt"]
    assert counts["denominator"] == counts["attempted"] == counts["UNQUALIFIED"] == 1
    assert report["planned_denominator"] == report["attempted_count"] == 3
    assert len(report["comparisons"]) == 2
    assert all(pair["verdict"] == "UNQUALIFIED" for pair in report["comparisons"])
    assert runner.replay(path, runner.digest(path))["numerically_replayed"] == 3


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "gpu", "cpu:0"])
def test_explicit_gpu_rejected_before_backend_no_fallback(frozen, monkeypatch, device):
    def forbidden(*args):
        raise AssertionError("CPU/GPU backend must not be called")

    monkeypatch.setattr(runner, "run_solver", forbidden)
    path, code, report = execute(frozen, device.replace(":", "-"), ["--device", device])
    assert code == 2 and report["verdict"] == "FAIL"
    assert "CPU runner rejects" in report["error"]["message"]
    assert report["attempted_count"] == 0
    assert "no fallback" in (path.parent / "failure.log").read_text()
    assert runner.replay(path, runner.digest(path))["qualification_verdict"] == "FAIL"


@pytest.mark.parametrize("kind", ["unsupported", "exception"])
def test_each_attempt_error_denominator_firstcause_and_no_retry(frozen, monkeypatch, kind):
    original = runner.run_solver
    calls = []

    def injected(model, solver, values, core):
        calls.append(solver)
        if solver == "openopt":
            if kind == "unsupported":
                raise runner.UnsupportedAPI("missing required opt-in firstcause")
            try:
                raise ValueError("original numeric firstcause")
            except ValueError as error:
                raise RuntimeError("backend context") from error
        return original(model, solver, values, core)

    monkeypatch.setattr(runner, "run_solver", injected)
    path, code, report = execute(frozen, kind)
    assert calls == list(runner.SOLVERS)
    assert code == (1 if kind == "unsupported" else 2)
    row = report["rows"][0]
    assert row["verdict"] == ("UNSUPPORTED" if kind == "unsupported" else "FAIL")
    assert report["planned_denominator"] == report["attempted_count"] == 3
    assert "firstcause" in (path.parent / row["failure_log"]).read_text()
    assert runner.replay(path, runner.digest(path))["numerically_replayed"] == 2


@pytest.mark.parametrize("target", ["pin", "arrays", "host", "metadata", "code"])
def test_result_pin_and_sidecar_corruption_negative_controls(frozen, target):
    path, _, report = execute(frozen, target)
    pin = runner.digest(path)
    if target == "pin":
        pin = "0" * 64
    else:
        filename = {"arrays": "frozen/function/arrays.npz", "host": "attempts/function/openopt.npz",
                    "metadata": "frozen/function/metadata.json", "code": "code/certificates.py"}[target]
        sidecar = path.parent / filename
        sidecar.write_bytes(sidecar.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256"):
        runner.replay(path, pin)


def test_resealed_result_wrong_denominator_or_assessment_rejected(frozen):
    path, _, report = execute(frozen, "reseal")
    report["backend_counts"]["openopt"]["denominator"] = 0
    path.write_text(json.dumps(report, allow_nan=False))
    with pytest.raises(ValueError, match="aggregate/denominator"):
        runner.replay(path, runner.digest(path))
    report["backend_counts"]["openopt"]["denominator"] = 1
    report["rows"][0]["assessment"]["objective"]["primal_original"] = 0
    path.write_text(json.dumps(report, allow_nan=False))
    with pytest.raises(ValueError, match="numerical/status"):
        runner.replay(path, runner.digest(path))


def test_bad_frozen_pin_empty_and_no_overwrite(frozen):
    root, directory, _ = frozen
    path, code, report = execute((root, directory, "0" * 64), "bad-input")
    assert code == 2 and report["verdict"] == "FAIL"
    assert "manifest SHA256 mismatch" in (path.parent / "failure.log").read_text()
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        execute((root, directory, "0" * 64), "bad-input")
    assert path.read_bytes() == before
    manifest = corpus.read_json(directory / "corpus.json")
    manifest["cases"] = []
    manifest["case_count"] = 0
    (directory / "corpus.json").write_text(json.dumps(manifest))
    path, code, report = execute((root, directory, runner.digest(directory / "corpus.json")), "empty")
    assert code == 2 and "empty/incomplete" in report["error"]["message"]


def test_public_label_reserved_origin_is_not_public_evidence(frozen):
    root, directory, _ = frozen
    report = corpus.read_json(directory / "corpus.json")
    entry = corpus.read_json(directory / "function" / "metadata.json")
    entry["provenance"]["origin"] = "public"
    entry["qualification"] = "UNVERIFIED"
    (directory / "function" / "metadata.json").write_text(json.dumps(entry))
    manifest = corpus.read_json(directory / "input-manifest.json")
    manifest["cases"][0]["origin"] = "public"
    (directory / "input-manifest.json").write_text(json.dumps(manifest))
    report["input_manifest_sha256"] = runner.digest(directory / "input-manifest.json")
    report["cases"][0]["origin"] = "public"
    report["cases"][0]["metadata_sha256"] = runner.digest(directory / "function" / "metadata.json")
    report["public_declared_count"] = 1
    (directory / "corpus.json").write_text(json.dumps(report))
    path, code, outcome = execute((root, directory, runner.digest(directory / "corpus.json")), "fake-public")
    assert code == 1 and outcome["public_qualified_count"] == 0
    assert "declaration rejected" in outcome["source_eligibility"][0]["reason"]
    assert runner.replay(path, runner.digest(path))["qualification_verdict"] == "UNQUALIFIED"


def test_source_review_contract_is_only_external_trust_not_license(frozen):
    _, directory, pin = frozen
    report = corpus.read_json(directory / "corpus.json")
    # Only the eligibility function is tested here, not real public evidence.
    entry = corpus.read_json(directory / "function" / "metadata.json")
    entry["provenance"].update(origin="public", source_url="https://models.dataset.edu/problem.mps",
                               license="Caller declaration; rights not adjudicated by runner")
    (directory / "function" / "metadata.json").write_text(json.dumps(entry))
    report["cases"][0]["origin"] = "public"
    accepted = {"schema_version": 1, "verdict": "PUBLIC_SOURCE_ACCEPTED", "corpus_sha256": pin,
                "cases": [{"case_id": "function", "source_url": entry["provenance"]["source_url"],
                           "input_sha256": entry["raw_sha256"], "model_sha256": entry["model_sha256"],
                           "source_review": "Function-only attestation test, not public acquisition",
                           "evidence_sha256": "1" * 64}]}
    assert not runner.source_eligibility(report, directory, None, pin)[0]["eligible"]
    assert runner.source_eligibility(report, directory, accepted, pin)[0]["eligible"]
    accepted["cases"][0]["input_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source/model mismatch"):
        runner.source_eligibility(report, directory, accepted, pin)


def test_cli_run_and_inputreplay_actual_process_retains_unqualified(frozen, request):
    root, directory, pin = frozen
    command = [sys.executable, "-B", str(Path(runner.__file__).resolve())]
    process = subprocess.run(command + [
        "run", "--corpus", str(directory), "--manifest-sha256", pin,
        "--artifact-dir", str(root / "cli"), "--source-root", str(runner.ROOT),
    ], capture_output=True, text=True, check=False)
    (root / "cli-run.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 1, process.stderr
    assert "PUBLIC_QP_RESULT" in process.stdout and '"verdict": "UNQUALIFIED"' in process.stdout
    result = root / "cli" / "result.json"
    pin = runner.digest(result)
    process = subprocess.run(command + ["replay", "--result", str(result), "--result-sha256", pin],
                             capture_output=True, text=True, check=False)
    (root / "cli-replay.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 1 and '"numerically_replayed": 3' in process.stdout
    process = subprocess.run(command + ["replay", "--result", str(result),
                                       "--result-sha256", "0" * 64],
                             capture_output=True, text=True, check=False)
    (root / "cli-reject.log").write_text(process.stdout + process.stderr)
    assert process.returncode == 2 and "result SHA256 mismatch" in process.stderr
    request.node.user_properties.extend([("cli_result", str(result)), ("cli_result_pin", pin)])


@pytest.mark.parametrize("kind", ["infeasible", "unbounded", "unbounded-no-reference-base"])
def test_actual_generated_directions_no_planted_witness(frozen, kind):
    root, _, _ = frozen
    if kind == "infeasible":
        text = """NAME DIRECTION
ROWS
 N OBJ
 E FIRST
 E SECOND
COLUMNS
 X OBJ 0 FIRST 1
 X SECOND 1
RHS
 RHS FIRST 0 SECOND 2
BOUNDS
 FR BND X
ENDATA
"""
    else:
        text = """NAME DIRECTION
ROWS
 N OBJ
 G FLOOR
COLUMNS
 X OBJ -1 FLOOR 1
RHS
 RHS FLOOR 2
BOUNDS
 FR BND X
ENDATA
"""
        if kind == "unbounded":
            text = text.replace("FLOOR 2", "FLOOR 0")
    local = root / "direction-local"
    local.mkdir()
    raw = local / "direction.mps"
    raw.write_text(text, encoding="ascii")
    entry = corpus.read_json(root / "input.json")["cases"][0]
    entry.update(path=raw.name, input_sha256=runner.digest(raw))
    manifest = root / "direction-input.json"
    corpus.write_json(manifest, {"schema_version": 1, "cases": [entry]})
    directory = root / "direction-frozen"
    corpus.freeze(manifest, local, directory)
    path, code, report = execute((root, directory, runner.digest(directory / "corpus.json")), kind)
    assert code == 1 and report["verdict"] == "UNQUALIFIED"
    assert report["planned_denominator"] == report["attempted_count"] == 3
    expected = "infeasible" if kind == "infeasible" else "unbounded"
    first = report["rows"][0]
    assert first["assessment"]["property"] == expected
    assert first["assessment"]["direction"]["passed"] and first["verdict"] == "QUALIFIED"
    assert all(row["verdict"] != "FAIL" for row in report["rows"])
    if kind == "unbounded-no-reference-base":
        last = report["rows"][2]
        assert not last["assessment"]["direction"]["checks"]["feasible_base"]
        assert last["verdict"] == "UNQUALIFIED"
        assert all(pair["denominator"] == 1 for pair in report["comparisons"])
    assert runner.replay(path, runner.digest(path))["numerically_replayed"] == 3


def test_max_iterations_unknown_retained_not_joint_solved_filter(frozen):
    path, code, report = execute(frozen, "max-iterations", ["--max-iterations", "1"])
    assert code == 1 and report["numeric_verdict"] == "UNQUALIFIED"
    assert report["rows"][0]["raw"]["status"] == "max_iterations"
    assert report["rows"][0]["assessment"]["property"] == "unknown"
    assert report["planned_denominator"] == report["attempted_count"] == 3
    assert len(report["comparisons"]) == 2
    assert all(pair["verdict"] == "UNQUALIFIED" for pair in report["comparisons"])
    assert runner.replay(path, runner.digest(path))["numerically_replayed"] == 3


def test_explicit_source_root_and_absent_optional_api_are_not_fallback(frozen):
    root = frozen[0]
    source = root / "source-root" / "src" / "openopt"
    source.mkdir(parents=True)
    scripts = source.parents[1] / "scripts"
    scripts.mkdir()
    shutil.copyfile(Path(corpus.__file__), scripts / "public_qp_corpus.py")
    original = runner.ROOT / "src" / "openopt"
    for name in ("qp.py", "certificates.py"):
        text = (original / name).read_text()
        if name == "qp.py":
            text = text.replace("    dual_gap_termination: bool = False\n", "")
        (source / name).write_text(text)
    path, code, report = execute(frozen, "absent-api", ["--source-root", str(source.parents[1])])
    assert code == 1 and report["rows"][0]["verdict"] == "UNSUPPORTED"
    assert "lacks required" in report["rows"][0]["error"]["message"]
    assert report["rows"][1]["verdict"] == report["rows"][2]["verdict"] == "QUALIFIED"
    assert runner.replay(path, runner.digest(path))["numerically_replayed"] == 2


def test_nonfinite_host_failure_retains_vectors_and_original_firstcause(frozen, monkeypatch):
    original = runner.run_solver

    def malformed(model, solver, values, core):
        raw = original(model, solver, values, core)
        if solver == "openopt":
            raw["vectors"]["x"][:] = np.nan
        return raw

    monkeypatch.setattr(runner, "run_solver", malformed)
    path, code, report = execute(frozen, "nonfinite-host")
    assert code == 2 and report["rows"][0]["verdict"] == "FAIL"
    first = report["rows"][0]
    vectors = runner.read_vectors(path.parent / first["vectors_path"])
    assert np.isnan(vectors["x"]).all()
    assert "x must be finite" in (path.parent / first["failure_log"]).read_text()
    assert report["planned_denominator"] == report["attempted_count"] == 3
    assert runner.replay(path, runner.digest(path))["numerically_replayed"] == 2


def test_detached_result_only_replay_never_reads_original_input_or_resolves(frozen, monkeypatch):
    path, _, _ = execute(frozen, "original-run")
    pin = runner.digest(path)
    copied = frozen[0] / "detached-result"
    shutil.copytree(path.parent, copied)
    frozen[1].rename(frozen[0] / "original-corpus-moved")
    (frozen[0] / "local").rename(frozen[0] / "original-local-moved")

    def forbidden(*args, **kwargs):
        raise AssertionError("result replay must not solve or use current assessment logic")

    monkeypatch.setattr(runner, "run_solver", forbidden)
    monkeypatch.setattr(runner, "assess", forbidden)
    checked = runner.replay(copied / "result.json", pin)
    assert checked["numerically_replayed"] == 3
    assert checked["qualification_verdict"] == "UNQUALIFIED"


V3_LP = """NAME BOUNDEDLP
ROWS
 N OBJ
 E FIX
COLUMNS
 X OBJ 1 FIX 1
RHS
 RHS OBJ -7 FIX 2
BOUNDS
 LO BND X -4
 UP BND X 4
ENDATA
"""
V3_QP = V3_LP.replace("BOUNDEDLP", "BOUNDEDQP").replace(
    "ENDATA", "QUADOBJ\n X X 2\nENDATA")


@pytest.fixture(scope="module")
def v3_actual(tmp_path_factory):
    """Real bounded synthetic solves, not fabricated v3 execution metadata."""
    if os.environ.get("OPENOPT_RUN_PUBLIC_V3_WORKER_TESTS") != "1":
        pytest.skip("exact native v3 worker/runtime requires OPENOPT_RUN_PUBLIC_V3_WORKER_TESTS=1")
    root = tmp_path_factory.mktemp("v3-actual")
    assert root.resolve().is_relative_to(runner.SHARED_TMP)
    local = root / "local"
    local.mkdir()
    entries = []
    for case_id, text in (("boundedlp", V3_LP), ("boundedqp", V3_QP)):
        raw = local / f"{case_id}.mps"
        raw.write_text(text, encoding="ascii")
        entries.append({"case_id": case_id, "path": raw.name,
                        "source_url": f"https://example.invalid/synthetic-{case_id}.mps",
                        "license": "0BSD synthetic hand-authored test; not public coverage",
                        "origin": "synthetic", "input_sha256": runner.digest(raw)})
    manifest = root / "input.json"
    corpus.write_json(manifest, {"schema_version": 1, "cases": entries})
    directory = root / "frozen"
    corpus.freeze(manifest, local, directory)
    args = runner.parser().parse_args([
        "run", "--version", runner.V3, "--corpus", str(directory),
        "--manifest-sha256", runner.digest(directory / "corpus.json"),
        "--artifact-dir", str(root / "execution")])
    process = subprocess.run(
        [sys.executable, "-B", str(Path(runner.__file__).resolve()),
         "run", "--version", runner.V3, "--corpus", str(directory),
         "--manifest-sha256", args.manifest_sha256, "--artifact-dir", str(args.artifact_dir)],
        capture_output=True, text=True, check=False, env={**os.environ, **runner.V3_ENV})
    (root / "execution-process.log").write_text(process.stdout + process.stderr)
    path = args.artifact_dir / "result.json"
    report = corpus.read_json(path)
    assert process.returncode == 0, (report, process.stderr)
    assert report["worker_execution_verified"] is True
    return path, report


def v3_reseal(path, report):
    """Coordinated attacker changes all parent/worker/plan pins, not just labels."""
    root = path.parent
    plan = corpus.read_json(root / "plan.json")
    (root / "plan.json").write_text(json.dumps(plan, allow_nan=False))
    plan_pin = runner.digest(root / "plan.json")
    report["plan_sha256"] = plan_pin
    for index in range(len(report["rows"]) // 2):
        worker_path = root / "workers" / f"{index:03d}" / "result.json"
        worker = corpus.read_json(worker_path)
        for row in report["rows"][2 * index:2 * index + 2]:
            row["plan_sha256"] = plan_pin
        worker["plan_sha256"] = plan_pin
        worker["rows"] = report["rows"][2 * index:2 * index + 2]
        worker_path.write_text(json.dumps(worker, allow_nan=False))
    report["files"] = runner.inventory(root)
    path.write_text(json.dumps(report, allow_nan=False))


def v3_copy(v3_actual, tmp_path):
    original, report = v3_actual
    root = tmp_path / "copy"
    shutil.copytree(original.parent, root)
    return root / "result.json", copy.deepcopy(report)


@pytest.fixture(autouse=True)
def restore_profiler():
    previous = sys.getprofile()
    try:
        yield
    finally:
        sys.setprofile(previous)


def v3_no_solves(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("replay must not solve/refine a fresh candidate")

    monkeypatch.setattr(runner, "v3_solve", forbidden)
    monkeypatch.setattr(runner, "v3_setup", forbidden)
    import osqp
    import clarabel
    from openopt import qp, refinement
    monkeypatch.setattr(osqp.OSQP, "solve", forbidden)
    monkeypatch.setattr(clarabel.DefaultSolver, "solve", forbidden)
    monkeypatch.setattr(qp.PreparedQP, "solve", forbidden)
    monkeypatch.setattr(refinement, "refine_qp_candidate", forbidden)
    previous = sys.getprofile()

    def profile(frame, event, arg):
        if event == "call" and frame.f_code.co_name in (
            "solve", "solve_qp", "refine_qp_candidate",
        ):
            raise AssertionError("replay entered frozen solver/refinement")
        if event == "c_call" and getattr(arg, "__name__", None) == "solve":
            raise AssertionError("replay entered native solve")
        if previous is not None:
            previous(frame, event, arg)

    sys.setprofile(profile)


def test_v3_actual_lp_qp_cold_hot_complete_binding_and_replay(v3_actual, monkeypatch, request):
    path, report = v3_actual
    assert report["schema_version"] == 3 and report["runner_version"] == runner.V3
    assert report["planned_denominator"] == report["attempted_count"] == 12
    assert report["numeric_verdict"] == "PASS"
    assert report["public_qualified_count"] == report["industrial_qualified_count"] == 0
    assert report["contract"] == runner.v3_execution_contract()
    assert report["contract"]["settings"]["openopt"]["refinement"]["check_arithmetic"] == (
        "reproducible_binary64_v2")
    assert report["runtime"]["thread_libraries"]
    for index in range(6):
        worker = corpus.read_json(path.parent / "workers" / f"{index:03d}" / "result.json")
        assert worker["solve_calls"] == 2
        cold, hot = worker["rows"]
        assert [cold["mode"], hot["mode"]] == ["cold", "hot"]
        assert cold["raw"]["iterations"] == hot["raw"]["iterations"]
        for row in (cold, hot):
            assert row["attempt"] == row["denominator"] == 1
            assert row["input_model_before"] == row["input_model_after"] == row["model_sha256"]
            assert all(row["assessment"]["candidate"]["checks"].values())
            assert row["assessment"]["objective"]["offset"] == 7.
            expected = 9. if row["case_id"] == "boundedlp" else 13.
            assert row["assessment"]["objective"]["primal_original"] == pytest.approx(expected)
            assert row["assessment"]["candidate"]["thresholds"]["bound_sign"] == 0.
            runner.v3_check_settings(row["raw"], row["solver"])
            assert row["raw"]["actual_calls"] == {
                "workspace_solve": 1,
                "backend_native_solve": 1 if row["solver"] == "osqp" else 0,
                "other_solver": 0, "warm_start": 0,
                "refinement_proposal": 1 if row["solver"] == "openopt" else 0,
            }
        for key in ("x", "y"):
            left = runner.read_vectors(path.parent / cold["vectors_path"])[key]
            right = runner.read_vectors(path.parent / hot["vectors_path"])[key]
            assert np.array_equal(left, right), (cold["solver"], key)
    v3_no_solves(monkeypatch)
    pin = runner.digest(path)
    checked = runner.v3_replay(path, pin)
    assert checked["numerically_replayed"] == 12 and checked["solve_calls"] == 0
    assert checked["public_qualified_count"] == 0
    receipt = path.parent.parent / "replay.json"
    corpus.write_json(receipt, checked)
    assert corpus.read_json(receipt) == checked
    request.node.user_properties.extend([
        ("v3_actual_result", str(path)), ("v3_actual_pin", pin),
        ("v3_replay_path", str(receipt)), ("v3_replay_pin", runner.digest(receipt))])


@pytest.mark.parametrize("solver,key,value", [
    ("osqp", "max_iter", 1999), ("osqp", "warm_starting", 1),
    ("clarabel", "max_iter", 199), ("clarabel", "max_threads", 2),
    ("clarabel", "presolve_enable", True),
    ("openopt", "check_arithmetic", "numpy"),
])
def test_v3_coordinated_settings_reseal_must_fail_without_solve(
    v3_actual, tmp_path, monkeypatch, solver, key, value, request,
):
    path, report = v3_copy(v3_actual, tmp_path)
    plan_path = path.parent / "plan.json"
    plan = corpus.read_json(plan_path)
    for contract in (report["contract"], plan["contract"]):
        settings = contract["settings"][solver]
        if solver == "openopt":
            settings["refinement"][key] = value
        else:
            settings["requested"][key] = settings["effective"][key] = value
    plan_path.write_text(json.dumps(plan, allow_nan=False))
    for row in report["rows"]:
        if row["solver"] == solver:
            for stage in ("requested", "settings", "settings_before_solve", "settings_after_solve"):
                settings = row["raw"][stage]
                if solver == "openopt":
                    settings["refinement"][key] = value
                else:
                    settings[key] = value
    v3_reseal(path, report)
    v3_no_solves(monkeypatch)
    with pytest.raises(ValueError, match="predeclared contract"):
        runner.v3_replay(path, runner.digest(path))
    request.node.user_properties.append(("negative_result", str(path)))


@pytest.mark.parametrize("mutation", ["runtime", "module", "threads", "source",
                                     "mode", "model", "drop-failure", "schema",
                                     "callcounts", "loaded-source", "numeric-summary"])
def test_v3_rehashed_identity_and_denominator_mutations_fail(
    v3_actual, tmp_path, monkeypatch, mutation,
):
    path, report = v3_copy(v3_actual, tmp_path)
    plan_path = path.parent / "plan.json"
    plan = corpus.read_json(plan_path)
    if mutation == "runtime":
        plan["runtime"]["packages"]["osqp"] = report["runtime"]["packages"]["osqp"] = "1.0.5"
    elif mutation == "module":
        for runtime in (plan["runtime"], report["runtime"]):
            runtime["modules"]["osqp.ext_builtin"]["sha256"] = "0" * 64
    elif mutation == "threads":
        for runtime in (plan["runtime"], report["runtime"]):
            runtime["torch_interop"] = 2
    elif mutation == "source":
        sidecar = path.parent / "code/qp.py"
        sidecar.write_bytes(sidecar.read_bytes() + b"\n# coordinated source mutation\n")
        plan["sources"]["qp.py"] = report["sources"]["qp.py"] = runner.digest(sidecar)
    elif mutation == "mode":
        plan["rows"][0]["mode"] = report["rows"][0]["mode"] = "hot"
    elif mutation == "model":
        plan["rows"][0]["model_sha256"] = report["rows"][0]["model_sha256"] = "0" * 64
    elif mutation == "drop-failure":
        plan["rows"] = plan["rows"][2:]
        report["rows"] = report["rows"][2:]
        report["planned_denominator"] = report["attempted_count"] = 10
    elif mutation == "schema":
        report["schema_version"] = 2
    elif mutation == "callcounts":
        report["rows"][0]["raw"]["actual_calls"]["workspace_solve"] = 0
    elif mutation == "loaded-source":
        report["rows"][0]["loaded_sources"]["qp.py"]["sha256"] = "0" * 64
    else:
        report["numeric_verdict"] = "UNQUALIFIED"
    plan_path.write_text(json.dumps(plan, allow_nan=False))
    if mutation != "drop-failure":
        for row in report["rows"]:
            row["runtime"] = report["runtime"]
        v3_reseal(path, report)
    else:
        report["files"] = runner.inventory(path.parent)
        path.write_text(json.dumps(report, allow_nan=False))
    v3_no_solves(monkeypatch)
    with pytest.raises(ValueError):
        runner.v3_replay(path, runner.digest(path))


@pytest.mark.parametrize("stage", ["requested", "settings", "settings_before_solve",
                                  "settings_after_solve"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "bool-int"])
def test_v3_full_settings_fields_and_types_each_stage(stage, mutation):
    settings = runner.v3_settings()["osqp"]
    raw = {"requested": copy.deepcopy(settings["requested"]),
           **{key: copy.deepcopy(settings["effective"])
              for key in ("settings", "settings_before_solve", "settings_after_solve")}}
    if mutation == "missing":
        raw[stage].pop("max_iter")
    elif mutation == "extra":
        raw[stage]["unknown_field"] = 1
    else:
        raw[stage]["warm_starting"] = False
    with pytest.raises(ValueError):
        runner.v3_check_settings(raw, "osqp")


def test_v3_legacy_versions_and_preview_cannot_be_promoted(v3_actual):
    path, _ = v3_actual
    with pytest.raises(ValueError, match="unsupported result schema/version"):
        runner.replay(path, runner.digest(path))
    from scripts import experiment_public_robustness as legacy
    core = runner.v3_core(path.parent / "code")
    with pytest.raises(ValueError, match="field membership"):
        legacy.settings_contract(core)
    with pytest.raises(SystemExit):
        runner.parser().parse_args(["run", "--version", "public-robustness-v2"])


def test_v3_wrong_sign_one_sided_support_is_never_clipped():
    core = runner.load_core(runner.ROOT / "src/openopt")
    p = runner.problem({"P": np.zeros((1, 1)), "q": np.zeros(1), "A": np.ones((1, 1)),
                        "l": np.zeros(1), "u": np.asarray([np.inf])})
    measured = core.checker.check_qp_candidate(
        p, np.zeros(1), np.asarray([1e-17]), atol=1e-7, rtol=1e-7,
        arithmetic=runner.V3_ARITHMETIC)
    assert not measured.passed
    assert not measured.checks["bound_sign"] and not measured.checks["dual_objective_available"]
    assert measured.thresholds["bound_sign"] == 0.


def test_v3_worker_caller_pin_rejected_before_solve(v3_actual, monkeypatch):
    path, _ = v3_actual
    v3_no_solves(monkeypatch)
    with pytest.raises(ValueError, match="worker plan pin"):
        runner.v3_worker(path.parent, "0" * 64, 0)


def test_v3_runtime_environment_mismatch_fails_before_worker(monkeypatch):
    if os.environ.get("OPENOPT_RUN_PUBLIC_V3_WORKER_TESTS") != "1":
        pytest.skip("exact runtime opt-in required")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    runner.v3_initialize_threads()
    with pytest.raises(ValueError, match="thread environment"):
        runner.v3_runtime()


def test_v3_worker_failed_attempt_stops_without_retry_and_keeps_denominator(
    v3_actual, tmp_path, monkeypatch,
):
    path, _ = v3_copy(v3_actual, tmp_path)
    plan = corpus.read_json(path.parent / "plan.json")
    pin = runner.digest(path.parent / "plan.json")
    # A new output keeps the frozen inputs, not any original worker result.
    output = tmp_path / "failed-worker"
    output.mkdir()
    for name in ("code", "frozen"):
        shutil.copytree(path.parent / name, output / name)
    shutil.copyfile(path.parent / "plan.json", output / "plan.json")
    calls = []

    def fail(*args):
        calls.append(1)
        raise ValueError("independent injected backend firstcause")

    monkeypatch.setattr(runner, "v3_observe_call", fail)
    assert runner.v3_worker(output, pin, 0) == 2
    receipt = corpus.read_json(output / "workers/000/result.json")
    assert receipt["verdict"] == "FAIL" and len(calls) == 1
    assert receipt["rows"][0]["attempted"] and receipt["rows"][0]["verdict"] == "FAIL"
    assert "firstcause" in (output / "workers/000/failure.log").read_text()
    assert len(plan["rows"]) == 12  # unchanged complete producer plan
    assert runner.digest(output / "plan.json") == pin


def test_v3_accepted_refinement_does_not_upgrade_iteration_status(v3_actual):
    path, report = v3_actual
    core = runner.v3_core(path.parent / "code")
    original = report["rows"][0]
    assert original["raw"]["refinement"]["status"] == "accepted"
    raw = {**original["raw"], "status": "max_iterations",
           "vectors": runner.read_vectors(path.parent / original["vectors_path"])}
    model = runner.load_model(path.parent / "frozen", original["case_id"])
    measured = runner.v3_assess(model, "openopt", raw, core)
    assert measured["candidate"]["passed"]
    assert measured["verdict"] == "UNQUALIFIED" and measured["property"] == "unknown"


def test_v3_explicit_public_and_overrides_refused_before_execution(frozen, monkeypatch):
    _, directory, _ = frozen

    def forbidden(*args):
        raise AssertionError("v3 rejection must precede backend")

    monkeypatch.setattr(runner, "v3_setup", forbidden)
    args = arguments(frozen, "v3-bad-device", ["--version", runner.V3, "--device", "cuda"])
    path, code = runner.v3_run(args)
    assert code == 2 and corpus.read_json(path)["attempted_count"] == 0
    args = arguments(frozen, "v3-bad-override", ["--version", runner.V3, "--max-iterations", "1"])
    path, code = runner.v3_run(args)
    assert code == 2 and "setting overrides" in corpus.read_json(path)["error"]["message"]
    # Eligibility is not execution authorization; no legal status can grant it.
    manifest = corpus.read_json(directory / "corpus.json")
    manifest["cases"][0]["origin"] = "public"
    monkeypatch.setattr(corpus, "verify_frozen", lambda *args: None)
    (directory / "corpus.json").write_text(json.dumps(manifest))
    args = arguments((frozen[0], directory, runner.digest(directory / "corpus.json")),
                     "v3-bad-public", ["--version", runner.V3])
    path, code = runner.v3_run(args)
    assert code == 2 and "public execution not authorized" in corpus.read_json(path)["error"]["message"]


def test_v3_actual_call_observer_detects_missing_solve_without_faking_execution(monkeypatch):
    class Workspace:
        def solve(self):
            raise AssertionError("control must not call a solve")

    state = type("State", (), {"workspace": Workspace()})()
    monkeypatch.setattr(runner, "v3_solve", lambda *args: ({}, {}))
    with pytest.raises(ValueError, match="observed actual solver calls"):
        runner.v3_observe_call(state, {}, "openopt", None)


def test_v3_portable_defaults_do_not_observe_exact_native_runtime(monkeypatch):
    def forbidden():
        raise AssertionError("metadata-only contract must not probe local runtime")

    monkeypatch.setattr(runner, "v3_runtime", forbidden)
    contract = runner.v3_execution_contract()
    assert contract["gates"] == runner.V3_GATES
    assert contract["settings"]["openopt"]["refinement"]["check_arithmetic"] == runner.V3_ARITHMETIC
    assert not contract["public_execution_authorized"]


@pytest.mark.parametrize("package", ["numpy", "torch", "osqp", "clarabel", "highspy"])
def test_v3_version_difference_negative_controls_are_portable(monkeypatch, package):
    from scripts import experiment_public_robustness as contracts

    expected = {**contracts.PACKAGE_VERSIONS, "highspy": "1.15.1"}
    monkeypatch.setattr(runner.importlib.metadata, "version",
                        lambda name: "incompatible-version" if name == package else expected[name])
    monkeypatch.setattr(sys, "version", "3.10.20 controlled-test-version")
    with pytest.raises(ValueError, match="runtime (packages|highspy)"):
        runner.v3_runtime()


@pytest.fixture
def v3_loaded_core(tmp_path):
    """Copied reviewed core only; no runtime observation or solver execution."""
    output = tmp_path / "loaded-binding"
    code = output / "code"
    code.mkdir(parents=True)
    for name in ("certificates.py", "qp.py", "scaling.py", "refinement.py"):
        shutil.copyfile(runner.ROOT / runner.V3_SOURCES[name], code / name)
    return output, runner.v3_core(code)


def test_v3_loaded_binding_positive_without_solver(v3_loaded_core):
    output, core = v3_loaded_core
    loaded = runner.v3_loaded_sources(output, core)
    assert set(loaded) == {"certificates.py", "qp.py", "scaling.py", "refinement.py"}


@pytest.mark.parametrize("attack", [
    "restored-disk", "default", "code", "class-init", "class-default",
    "checker-alias", "wrapped-function", "global", "package-cache",
])
def test_v3_actual_loaded_binding_rejects_not_self_reported_hash(
    v3_loaded_core, monkeypatch, attack, request,
):
    output, core = v3_loaded_core
    checker = core.checker.check_qp_candidate
    if attack == "restored-disk":
        source = output / "code/certificates.py"
        trusted = source.read_bytes()
        old = b"y: object, *, atol: float = 1e-8, rtol: float = 1e-8,"
        assert trusted.count(old) == 1
        altered = trusted.replace(old, old.replace(b"atol: float = 1e-8", b"atol: float = 1e-3"))
        source.write_bytes(altered)
        try:
            core = runner.v3_core(output / "code")
            assert core.checker.check_qp_candidate.__kwdefaults__["atol"] == 1e-3
        finally:
            source.write_bytes(trusted)
        assert runner.digest(source) == runner.v3_sources()["certificates.py"]
        request.node.user_properties.append(("actually_loaded_sha256", corpus.sha256(altered)))
    elif attack == "default":
        monkeypatch.setattr(checker, "__kwdefaults__", {**checker.__kwdefaults__, "atol": 1e-3})
    elif attack == "code":
        monkeypatch.setattr(checker, "__code__", checker.__code__.replace(
            co_consts=(*checker.__code__.co_consts, "coordinated loaded substitution")))
    elif attack == "class-init":
        init = core.qp.QPConfig.__init__
        monkeypatch.setattr(init, "__code__", init.__code__.replace(
            co_consts=(*init.__code__.co_consts, "changed generated constructor")))
    elif attack == "class-default":
        monkeypatch.setattr(core.refinement.RefinementConfig, "check_arithmetic", "other")
    elif attack == "checker-alias":
        monkeypatch.setattr(core.refinement, "check_qp_candidate", lambda *args: None)
    elif attack == "wrapped-function":
        import functools

        original = core.scaling.equilibrate_qp

        @functools.wraps(original)
        def substituted(*args, **kwargs):
            raise AssertionError("substituted wrapper must never run")

        monkeypatch.setattr(core.scaling, "equilibrate_qp", substituted)
    elif attack == "global":
        monkeypatch.setattr(core.checker, "REPRODUCIBLE_ARITHMETIC", "other")
    else:
        monkeypatch.setitem(sys.modules, core.qp.__name__, core.checker)
    with pytest.raises(ValueError, match="loaded"):
        runner.v3_loaded_sources(output, core)


def test_v3_unchanged_nash_loaded_restored_counterexample(
    v3_actual, tmp_path, monkeypatch, request,
):
    """Invoke the reviewer-owned function without copying or editing its bytes."""
    if os.environ.get("OPENOPT_RUN_PUBLIC_V3_NASH_REVIEW") != "1":
        pytest.skip("independent source replay requires explicit Nash review opt-in")
    source = Path(os.environ["OPENOPT_PUBLIC_V3_NASH_TEST_SOURCE"]).resolve(strict=True)
    pin = "7d2262e7d1550800dddcb72b53341dc9b02efe44a8d62d58feaebe521da7995e"
    assert runner.digest(source) == pin, "unchanged independent test source required"
    audit = runner.load_module(source, "_public_v3_unchanged_nash_review")
    assert audit.producer is runner
    monkeypatch.setattr(audit, "ARTIFACT_ROOT", tmp_path.resolve())
    audit.test_restored_disk_does_not_hide_different_loaded_checker_bytes(
        v3_actual, tmp_path, monkeypatch, request)
    evidence = tmp_path / "loaded-restored-disk-counterexample.json"
    receipt = corpus.read_json(evidence)
    assert receipt["rejected"] and receipt["trusted_disk_restored"]
    assert receipt["loaded_observations"] and receipt["solver_calls"] == 0
    assert receipt["replay"] is None
    assert runner.digest(source) == pin
    request.node.user_properties.append(("unchanged_independent_test_sha256", pin))
