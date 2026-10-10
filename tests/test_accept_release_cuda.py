"""Independent CPU fixtures and semantic/hash mutations of CUDA acceptance."""

import json
from pathlib import Path

import pytest

from scripts import accept_release_cuda as audit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "ARTIFACT_ROOT", tmp_path.resolve())
    run = tmp_path / "run"
    source = run / "source"
    current = tmp_path / "current"
    for name in ("src/openopt/qp.py", "tests/test_general_qp_cuda.py",
                 "scripts/gpu_release_check.py"):
        for root in (source, current):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('"""Fixture, never executed."""\ndef test_explicit_cuda():\n    pass\n')
    hashes = audit.source_files(source)
    result = run / "case-0/cuda-check-frozen/result.json"
    report = {
        "verdict": "PASS", "returncode": 0,
        "targets": ["tests/test_general_qp_cuda.py"], "selection": "",
        "source_sha256": hashes,
        "tests": {"tests": 1, "failures": 0, "errors": 0, "skipped": 0},
        "hardware": {"gpu": "NVIDIA RTX PRO 4500 Blackwell", "node": "fixture",
                     "torch": "fixture", "cuda": "fixture", "cpu_quota": "max 100000"},
        "telemetry_samples": 1,
    }
    dump(result, report)
    dump(run / "run.json", {"stage": "release-regression", "state": "SUBMITTED",
                           "source_sha": "informational-not-worktree-head",
                           "source_hashes": hashes,
                           "jobs": [{"job": "fixture-job", "output": str(result.parent.parent),
                                     "manifest": str(run / "job-0.json")}]})
    dump(run / "job-0.json", {
        "kind": "Job", "metadata": {"name": "fixture-job"},
        "spec": {"backoffLimit": 0, "template": {"spec": {
            "restartPolicy": "Never", "containers": [{
                "resources": {"limits": {"nvidia.com/gpu": 1}},
                "command": ["python3", str(source / "scripts/gpu_release_check.py"),
                            "--artifact-dir", str(result.parent.parent),
                            "--target", "tests/test_general_qp_cuda.py"]}]}}}})
    dump(source / "source-hashes.json", hashes)
    xml = ('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
           '<testcase classname="tests.test_general_qp_cuda" '
           'name="test_explicit_cuda" time="0.1"/></testsuite></testsuites>')
    (result.parent / "pytest.xml").write_text(xml)
    (result.parent / "pytest.log").write_text("1 passed in 0.1s\n\nreturncode=0\n")
    (result.parent / "execution.log").write_text("OPENOPT_RELEASE_CUDA verdict=PASS tests=1\n")
    dump(result.parent / "telemetry.json", [{"rc": 0, "gpu": "GPU-fixture, 1, 20, 10",
                                           "cpu": {"usage_usec": 1}}])
    bundle = audit.seal(result, audit.sha(audit.read(result)), source_snapshot=source,
                        run_path=run / "run.json", current_root=current)
    # Fixture self-proof: the real public validator is exercised; source never runs.
    assert audit.validate(bundle, current_root=current)["tests"]["tests"] == 1
    return bundle, current, run


def changed_file(bundle, key, content):
    path = Path(bundle["files"][key]["path"])
    path.write_text(content)
    bundle["files"][key] = audit.descriptor(path)


def test_positive_execution_scope_is_not_full_certificate(evidence):
    bundle, current, _ = evidence
    result = audit.validate(bundle, current_root=current)
    assert result["scope"] == "correctness-suite execution"
    assert result["current_core"]["verdict"] == "PASS"
    assert result["proof"]["verdict"] == "UNVERIFIED"
    assert result["proof"]["qualified"] is False
    assert result["negative_verdict"] == "UNVERIFIED"


def test_source_scan_reads_only_explicit_document(tmp_path):
    for name in ("src/openopt/qp.py", "docs/research/refinement.md", "docs/private.md"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n")
    hashes = audit.source_files(tmp_path)
    assert set(hashes) == {"src/openopt/qp.py", "docs/research/refinement.md"}
    assert audit.relative("docs/research/refinement.md") == "docs/research/refinement.md"
    with pytest.raises(audit.AuditError, match="unknown or unsafe"):
        audit.relative("docs/private.md")


def test_caller_bundle_and_result_pins_are_required(evidence):
    bundle, current, run = evidence
    path = run / "bundle.json"
    dump(path, bundle)
    assert audit.accept(path, audit.sha(audit.read(path)), current_root=current)["verdict"] == "PASS"
    with pytest.raises(audit.AuditError, match="bundle pin"):
        audit.accept(path, "0" * 64, current_root=current)
    bundle["result_sha256"] = "0" * 64
    with pytest.raises(audit.AuditError, match="result pin"):
        audit.validate(bundle, current_root=current)


@pytest.mark.parametrize("mutation", ["skip", "empty", "unknown", "incomplete", "duplicate",
                                     "failure", "unknown_function"])
def test_semantic_junit_mutants_rejected_even_resealed(evidence, mutation):
    bundle, current, _ = evidence
    text = audit.bound(bundle["files"]["pytest.xml"]).decode()
    if mutation == "skip":
        text = text.replace('skipped="0"', 'skipped="1"')
    elif mutation == "empty":
        text = '<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0"/></testsuites>'
    elif mutation == "unknown":
        text = text.replace("tests.test_general_qp_cuda", "tests.test_unknown")
    elif mutation == "incomplete":
        text = text.replace('tests="1"', 'tests="2"')
    elif mutation == "duplicate":
        text = text.replace("</testsuite>", '<testcase classname="tests.test_general_qp_cuda" '
                            'name="test_explicit_cuda" time="0.1"/></testsuite>').replace(
                                'tests="1"', 'tests="2"')
    elif mutation == "failure":
        text = text.replace('time="0.1"/>', 'time="0.1"><failure/></testcase>')
    else:
        text = text.replace("test_explicit_cuda", "test_unknown_cuda")
    changed_file(bundle, "pytest.xml", text)
    with pytest.raises(audit.AuditError):
        audit.validate(bundle, current_root=current)


@pytest.mark.parametrize("mutation", ["unknown", "empty", "removed", "added", "run", "result"])
def test_source_membership_mutants_rejected(evidence, mutation):
    bundle, current, run = evidence
    if mutation == "unknown":
        bundle["sources"]["secrets/unknown.py"] = "0" * 64
    elif mutation == "empty":
        bundle["sources"] = {}
    elif mutation == "removed":
        bundle["sources"].pop("src/openopt/qp.py")
    elif mutation == "added":
        (run / "source/src/openopt/new.py").write_text('"""Unsealed."""\n')
    elif mutation == "run":
        path = Path(bundle["run"]["path"])
        record = audit.parse(audit.read(path), "run")
        record["source_hashes"].pop("src/openopt/qp.py")
        dump(path, record)
        bundle["run"] = audit.descriptor(path)
    else:
        path = Path(bundle["result"]["path"])
        report = audit.parse(audit.read(path), "result")
        report["source_sha256"]["scripts/unknown.py"] = "0" * 64
        dump(path, report)
        bundle["result"] = audit.descriptor(path)
        bundle["result_sha256"] = bundle["result"]["sha256"]
    with pytest.raises(audit.AuditError):
        audit.validate(bundle, current_root=current)


@pytest.mark.parametrize("file", ["pytest.xml", "pytest.log", "execution.log", "telemetry.json"])
def test_artifact_hash_mutation_rejected(evidence, file):
    bundle, current, _ = evidence
    path = Path(bundle["files"][file]["path"])
    with path.open("a") as stream:
        stream.write(" ")
    with pytest.raises(audit.AuditError, match="mutation"):
        audit.validate(bundle, current_root=current)


def test_current_core_bidirectional_differences_not_head_mismatch(evidence):
    bundle, current, _ = evidence
    (current / "src/openopt/new.py").write_text('"""New core."""\n')
    (current / "src/openopt/qp.py").write_text('"""Changed core."""\n')
    result = audit.validate(bundle, current_root=current)
    assert result["verdict"] == "PASS"
    assert result["current_core"]["verdict"] == "UNVERIFIED"
    assert result["current_core"]["missing"] == ["src/openopt/new.py"]
    assert result["current_core"]["changed"] == ["src/openopt/qp.py"]
    (current / "src/openopt/qp.py").unlink()
    assert audit.validate(bundle, current_root=current)["current_core"]["extra"] == ["src/openopt/qp.py"]


def attach_proof(bundle, run, saved, property_path=None):
    path = run / "case-0/cuda-check-frozen/pytest-tmp/proof.json"
    dump(path, saved)
    source = run / "source"
    old = "tests/test_general_qp_cuda.py"
    new = "tests/test_qp_scaling.py"
    (source / old).rename(source / new)
    (source / new).write_text(
        '"""Fixture, never executed."""\n'
        'def test_cuda_original_coordinate_restoration_kkt_only():\n    pass\n')
    hashes = audit.source_files(source)
    bundle["sources"] = hashes
    dump(source / "source-hashes.json", hashes)
    bundle["source_manifest"] = audit.descriptor(source / "source-hashes.json")
    record = audit.parse(audit.bound(bundle["run"]), "run")
    record["source_hashes"] = hashes
    dump(Path(bundle["run"]["path"]), record)
    bundle["run"] = audit.descriptor(Path(bundle["run"]["path"]))
    report = audit.parse(audit.bound(bundle["result"]), "result")
    report.update(targets=[new], source_sha256=hashes)
    dump(Path(bundle["result"]["path"]), report)
    bundle["result"] = audit.descriptor(Path(bundle["result"]["path"]))
    bundle["result_sha256"] = bundle["result"]["sha256"]
    job = audit.parse(audit.bound(bundle["job"]), "job")
    job["spec"]["template"]["spec"]["containers"][0]["command"][-1] = new
    dump(Path(bundle["job"]["path"]), job)
    bundle["job"] = audit.descriptor(Path(bundle["job"]["path"]))
    text = ('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase classname="tests.test_qp_scaling" '
            'name="test_cuda_original_coordinate_restoration_kkt_only" time="0.1">'
            '<properties><property name="cuda_scaling_acceptance_path" '
            f'value="{property_path or path}"/></properties></testcase></testsuite></testsuites>')
    changed_file(bundle, "pytest.xml", text)
    bundle["proofs"] = {str(path.resolve()): audit.descriptor(path)}
    return path


def test_missing_problem_is_unverified_not_reconstructed(evidence):
    bundle, current, run = evidence
    attach_proof(bundle, run, {"verdict": "PASS", "x": [1], "y": [0],
                              "device": "cuda:0", "gpu_name": "NVIDIA RTX PRO 4500 Blackwell"})
    result = audit.validate(bundle, current_root=current)
    assert result["proof"]["cases"][0]["reason"] == "frozen original problem missing"
    assert result["proof"]["qualified"] is False


@pytest.mark.parametrize("mutation", ["hash", "removed", "extra", "escape", "model", "device"])
def test_proof_mutations_rejected(evidence, mutation):
    bundle, current, run = evidence
    saved = {"verdict": "PASS", "x": [1], "y": [0],
             "device": "cuda:0", "gpu_name": "NVIDIA RTX PRO 4500 Blackwell"}
    if mutation in ("model", "device"):
        saved["gpu_name" if mutation == "model" else "device"] = "cpu"
    path = attach_proof(bundle, run, saved, "/etc/escape.json" if mutation == "escape" else None)
    if mutation == "hash":
        with path.open("a") as stream:
            stream.write(" ")
    elif mutation == "removed":
        bundle["proofs"] = {}
    elif mutation == "extra":
        bundle["proofs"][str(run / "extra.json")] = bundle["proofs"][str(path)]
    with pytest.raises(audit.AuditError):
        audit.validate(bundle, current_root=current)


def frozen_candidate():
    return {"verdict": "PASS", "problem": {"P": [[2]], "q": [-2], "A": [[1]],
                                         "l": [0], "u": [2]},
            "x": [1], "y": [0], "atol": 1e-8, "rtol": 1e-8}


def test_independent_saved_candidate_replay_positive():
    result = audit.recompute(frozen_candidate(), "scaling")
    assert result["verdict"] == "RECOMPUTED"
    assert result["metrics"]["stationarity"] == 0
    assert result["metrics"]["gap"] == 0


@pytest.mark.parametrize("mutation", ["x", "y", "problem", "tolerance"])
def test_saved_vector_replay_negative_mutants(mutation):
    saved = frozen_candidate()
    if mutation == "x":
        saved["x"] = [1.5]
    elif mutation == "y":
        saved["y"] = [2]
    elif mutation == "problem":
        saved["problem"]["q"] = [0]
    else:
        saved["atol"] = 10
    with pytest.raises(audit.AuditError):
        audit.recompute(saved, "gap")


def test_independent_pcg_true_residual_positive_and_negative():
    saved = frozen_candidate()
    saved.update(system=[[3]], rhs=[6], inner_x=[2], inner_tolerance=1e-8)
    assert audit.recompute(saved, "pcg")["metrics"]["inner_residual"] == 0
    saved["inner_x"] = [1]
    with pytest.raises(audit.AuditError, match="true residual"):
        audit.recompute(saved, "pcg")


@pytest.mark.parametrize("status", ["infeasible", "unbounded"])
def test_saved_ray_replay_positive_and_negative(status):
    saved = {"atol": 1e-8, "rtol": 1e-8, "status": status}
    if status == "infeasible":
        saved.update(problem={"P": [[0]], "q": [0], "A": [[1], [1]],
                              "l": [1, None], "u": [None, 0]}, direction=[-1, 1])
    else:
        saved.update(problem={"P": [[0]], "q": [-1], "A": [[1]],
                              "l": [0], "u": [None]}, direction=[1], feasible_base=[0])
    assert audit.recompute(saved, "ray")["verdict"] == "RECOMPUTED"
    saved["direction"] = [-v for v in saved["direction"]]
    with pytest.raises(audit.AuditError):
        audit.recompute(saved, "ray")


def test_unbounded_saved_base_and_positive_curvature_are_not_certified():
    saved = {"atol": 1e-8, "rtol": 1e-8, "status": "unbounded",
             "problem": {"P": [[0]], "q": [-1], "A": [[1]], "l": [0], "u": [None]},
             "direction": [1]}
    assert audit.recompute(saved, "ray")["verdict"] == "UNVERIFIED"
    saved["feasible_base"] = [-1]
    with pytest.raises(audit.AuditError, match="base rejected"):
        audit.recompute(saved, "ray")
    saved["feasible_base"] = [0]
    saved["problem"]["P"] = [[1e-12]]
    with pytest.raises(audit.AuditError, match="base rejected"):
        audit.recompute(saved, "ray")


@pytest.mark.parametrize("mutation", ["selection", "target", "cards", "retry", "identity"])
def test_job_semantic_mutants_rejected(evidence, mutation):
    bundle, current, _ = evidence
    path = Path(bundle["job"]["path"])
    job = audit.parse(audit.bound(bundle["job"]), "job")
    container = job["spec"]["template"]["spec"]["containers"][0]
    if mutation == "selection":
        container["command"] += ["--expression", "cuda"]
    elif mutation == "target":
        container["command"][-1] = "tests/test_unknown.py"
    elif mutation == "cards":
        container["resources"]["limits"]["nvidia.com/gpu"] = 0
    elif mutation == "retry":
        job["spec"]["backoffLimit"] = 1
    else:
        job["metadata"]["name"] = "other-job"
    dump(path, job)
    bundle["job"] = audit.descriptor(path)
    with pytest.raises(audit.AuditError):
        audit.validate(bundle, current_root=current)


def test_cli_cpu_end_to_end_and_external_pass_log(evidence):
    bundle, current, run = evidence
    path = run / "bundle.json"
    dump(path, bundle)
    artifact = run / "audit-output"
    assert audit.main(["accept", "--bundle", str(path), "--sha256", audit.sha(audit.read(path)),
                       "--current-root", str(current), "--artifact-dir", str(artifact)]) == 0
    outputs = list(artifact.glob("accept-*/acceptance.json"))
    assert len(outputs) == 1
    report = audit.parse(audit.read(outputs[0]), "acceptance")
    assert report["verdict"] == "PASS" and report["proof"]["qualified"] is False
    assert "verdict=PASS" in audit.read(outputs[0].parent / "execution.log").decode()
