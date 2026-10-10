"""Synthetic MPS format/function controls, never public or industrial evidence."""

import json
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
