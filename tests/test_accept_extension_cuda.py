"""CPU evidence fixtures: real execution validator, no producer/test imports.

Synthetic telemetry is explicitly a fixture, never actual CUDA evidence.
The command-line tests use real clean Git repositories and external outputs.
"""

import copy
import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from openopt.certificates import check_qp_candidate
from openopt.conic import ConicProblem, Nonnegative, check_conic_candidate
from scripts import accept_extension_cuda as audit

execution = audit.execution
ROOT = Path(__file__).resolve().parents[1]
GPU = "NVIDIA RTX PRO 4500 Blackwell fixture (not measured)"
NATIVE_TEST = """
raise AssertionError("frozen test module must never execute")
def test_explicit_cuda_never_falls_back(): pass
def test_reject_cuda_out_of_range(): pass
@pytest.mark.parametrize("name", ["soc_lp", "mixed", "nonnegative", "equality",
                                "empty", "singular", "soc_origin", "soc2"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_native_conic_frozen_original_model(): pass
@pytest.mark.parametrize("mutation", ["x", "s", "z"])
def test_cuda_native_conic_reject_numerical_tamper(): pass
"""
REFINEMENT_TEST = """
raise AssertionError("frozen test module must never execute")
@pytest.mark.parametrize("dtype,factor_dtype", [
    (torch.float64, torch.float64), (torch.float64, torch.float32),
    (torch.float32, torch.float32)])
def test_cuda_refinement_frozen_model_accuracy_ab(): pass
def test_cuda_refinement_actual_residual_correction(): pass
@pytest.mark.parametrize("kind", ["wrong_active_set", "singular_kkt", "wrong_sign"])
def test_cuda_refinement_negative_controls(): pass
"""


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    execution.write(path, value)


def overwrite(path, value):
    path.write_text(json.dumps(value, allow_nan=False) + "\n", encoding="utf-8")


def git(root, *arguments):
    result = subprocess.run(["git", "-C", str(root), *arguments], capture_output=True, check=False)
    assert result.returncode == 0, (result.returncode, result.stderr.decode())
    return result.stdout


def commit_fixture(root):
    git(root, "init", "-q")
    names = sorted(path.relative_to(root).as_posix() for prefix in ("src", "scripts", "tests", "docs")
                   for path in (root / prefix).rglob("*") if path.is_file())
    git(root, "add", "--", *names)
    git(root, "-c", "user.name=CPU acceptance fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-q", "-m", "Frozen acceptance fixture, not CUDA evidence")
    assert not git(root, "status", "--porcelain")


def source_fixture(root, suite):
    names = set(audit.SOURCES[suite]) | set(audit.AUDITORS) | {
        "src/openopt/certificates.py", "src/openopt/conic.py", execution.RUNNER}
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if name in audit.AUDITORS or name in ("src/openopt/certificates.py", "src/openopt/conic.py"):
            path.write_bytes((ROOT / name).read_bytes())
        elif name.startswith("tests/"):
            path.write_text(NATIVE_TEST if suite == "native_conic" else REFINEMENT_TEST)
        else:
            path.write_text('"""Fixture-only source, never imported."""\n')
    return names


def native_saved(hashes, dtype):
    problem = ConicProblem(np.array([[1.]]), np.array([-2.]),
                           np.array([[1.]]), np.array([1.]), (Nonnegative(1),))
    atol = 3e-5 if dtype == "torch.float32" else 1e-8
    evidence = check_conic_candidate(problem, [1.], [0.], [1.], atol=atol, rtol=0)
    assert evidence.passed
    return {
        "schema_version": 1, "verdict": "PASS", "family": "native_conic",
        "device": "cuda:0", "dtype": dtype, "gpu_name": GPU,
        "placements": dict.fromkeys(("x", "s", "z"), "cuda:0"),
        "problem": {"P": [[1.]], "q": [-2.], "A": [[1.]], "b": [1.],
                    "cones": [{"kind": "nonnegative", "dimension": 1}]},
        "x": [1.], "s": [0.], "z": [1.], "status": "solved",
        "iterations": 1, "objective": -1.5, "atol": atol, "rtol": 0,
        "config": {"enabled": True, "device": "cuda:0", "dtype": dtype,
                   "rho": 1., "sigma": 1., "max_iterations": 100, "check_interval": 1},
        "evidence": evidence.to_dict(), "source_sha256": hashes,
    }


def refinement_saved(hashes, definition):
    from types import SimpleNamespace

    parameters = definition["parameters"]
    negative = definition["function"] == "test_cuda_refinement_negative_controls"
    kind = parameters.get("kind")
    P, q, A, lo, hi = [[1.]], [-2.], [[1.]], [-np.inf], [1.]
    bx, by, xx, yy = [0.99999], [1.00001], [1.], [1.]
    status, reason = "accepted", "original_check_passed_nonworsening"
    if negative:
        if kind == "singular_kkt":
            q, A, lo, hi = [0.], [[1.], [1.]], [0., 0.], [0., 0.]
            bx, by = [0.], [0., 0.]
            status, reason = "unknown", "singular_or_unresolved_kkt"
        elif kind == "wrong_sign":
            q, bx, by = [-0.5], [1.], [0.]
            status, reason = "rejected", "original_check_failed"
        else:
            bx, by = [0.5], [0.5]
            status, reason = "rejected", "original_check_failed"
        xx, yy = bx[:], by[:]
    problem = SimpleNamespace(P=np.asarray(P), q=np.asarray(q), A=np.asarray(A),
                              l=np.asarray(lo), u=np.asarray(hi))
    correction = definition["function"] == "test_cuda_refinement_actual_residual_correction"
    threshold = 1e-10 if correction else 1e-8
    tol = {"atol": threshold, "rtol": threshold}
    before = check_qp_candidate(problem, bx, by, **tol)
    after = check_qp_candidate(problem, xx, yy, **tol)
    wrong_problem = SimpleNamespace(**{**vars(problem), "q": problem.q + 1})
    controls = {
        "wrong_x": check_qp_candidate(problem, np.asarray(xx) + 1, yy, **tol).to_dict(),
        "wrong_y": check_qp_candidate(problem, xx, -np.asarray(yy), **tol).to_dict(),
        "wrong_model": check_qp_candidate(wrong_problem, xx, yy, **tol).to_dict(),
    }
    dtype = parameters.get("dtype", {"dtype": "torch.float64"})["dtype"]
    factor = parameters.get("factor_dtype", {"dtype": "torch.float32" if correction
                                           else "torch.float64"})["dtype"]
    return {
        "schema_version": 1, "verdict": "PASS", "kind": "qp_refinement_original_coordinate_ab",
        "case": kind or "cuda_accuracy_fixture", "device": "cuda:0",
        "dtype": dtype, "gpu_name": GPU, "candidate_accepted": not negative,
        "problem": {"P": P, "q": q, "A": A,
                    "l": ["-Infinity" if v == -np.inf else v for v in lo], "u": hi},
        "config": {"device": "cuda:0", "dtype": dtype, "factor_dtype": factor,
                   "max_refinement_steps": 3, "active_atol": 1e-5, "active_rtol": 1e-7,
                   "check_atol": threshold, "check_rtol": threshold},
        "baseline": {"x": bx, "y": by, "evidence": before.to_dict()},
        "refined": {"x": xx, "y": yy, "status": status, "reason": reason,
                    "evidence": after.to_dict(), "diagnostics": {
                        "solver_fallback": False, "iterations": [
                            {"linear_residual": 1e-5, "original_check_passed": False},
                            {"linear_residual": 1e-15, "original_check_passed": True}]}},
        "negative_controls": controls, "source_sha256": hashes,
    }


@pytest.fixture(params=("native_conic", "refinement"))
def evidence(tmp_path, monkeypatch, request):
    monkeypatch.setattr(execution, "ARTIFACT_ROOT", tmp_path.resolve())
    suite = request.param
    run, current = tmp_path / "run", tmp_path / "current"
    source = run / "source"
    names = source_fixture(source, suite)
    for name in names:
        path = current / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((source / name).read_bytes())
    commit_fixture(current)
    monkeypatch.setattr(audit, "ROOT", current)
    hashes = execution.source_files(source)
    if suite == "refinement" and "docs/research/refinement.md" not in hashes:
        # Test the newly integrated narrow execution-source interface without
        # widening its source allowlist to every document.
        original_source_files, original_relative = execution.source_files, execution.relative
        def with_doc(root):
            result = original_source_files(root)
            doc = Path(root) / "docs/research/refinement.md"
            if doc.is_file():
                result["docs/research/refinement.md"] = execution.sha(doc.read_bytes())
            return result
        def relative_doc(name):
            return name if name == "docs/research/refinement.md" else original_relative(name)
        monkeypatch.setattr(execution, "source_files", with_doc)
        monkeypatch.setattr(execution, "relative", relative_doc)
        hashes = execution.source_files(source)
    definitions = audit.expected_cases(source, suite)
    result = run / "case-0/cuda-check-frozen/result.json"
    result.parent.mkdir(parents=True)
    root = ET.Element("testsuites")
    xml_suite = ET.SubElement(root, "testsuite", tests=str(len(definitions)),
                              failures="0", errors="0", skipped="0")
    for index, (identity, definition) in enumerate(definitions.items()):
        classname, name = identity.split("::")
        node = ET.SubElement(xml_suite, "testcase", classname=classname, name=name, time="0.1")
        if definition["cpu_only"]:
            continue
        sidecar = result.parent / f"pytest-tmp/case-{index}/candidate.json"
        pins = {name: execution.sha((source / name).read_bytes()) for name in audit.SOURCES[suite]}
        dtype = definition["parameters"].get("dtype", {"dtype": "torch.float64"})["dtype"]
        saved = (native_saved(pins, dtype) if suite == "native_conic"
                 else refinement_saved(pins, definition))
        dump(sidecar, saved)
        if "mutation" in definition["parameters"]:
            wrong = copy.deepcopy(saved)
            wrong[definition["parameters"]["mutation"]][0] += 1
            dump(sidecar.parent / "tampered.json", wrong)
        prop = ET.SubElement(node, "properties")
        prefix = audit.CONTRACTS[suite]["property"]
        ET.SubElement(prop, "property", name=prefix + "_path", value=str(sidecar))
        ET.SubElement(prop, "property", name=prefix + "_sha256",
                      value=execution.sha(sidecar.read_bytes()))
    (result.parent / "pytest.xml").write_bytes(ET.tostring(root))
    (result.parent / "pytest.log").write_text(f"{len(definitions)} passed in 1s\nreturncode=0\n")
    (result.parent / "execution.log").write_text(
        f"OPENOPT_RELEASE_CUDA verdict=PASS tests={len(definitions)}\n")
    dump(result.parent / "telemetry.json", [{"rc": 0, "gpu": "GPU-fixture, 1, 20, 10",
                                           "cpu": {"usage_usec": 1}}])
    target = next(name for name in names if name.startswith("tests/"))
    selected_hashes = {name: pin for name, pin in hashes.items() if name.startswith("src/openopt/")}
    selected_hashes.update({target: hashes[target], execution.RUNNER: hashes[execution.RUNNER]})
    dump(result, {"verdict": "PASS", "returncode": 0, "targets": [target],
                  "selection": audit.CONTRACTS[suite]["selection"],
                  "source_sha256": selected_hashes,
                  "tests": dict(tests=len(definitions), failures=0, errors=0, skipped=0),
                  "hardware": {"gpu": GPU, "node": "CPU fixture", "torch": "fixture",
                               "cuda": "fixture", "cpu_quota": "max 100000"},
                  "telemetry_samples": 1})
    dump(source / "source-hashes.json", hashes)
    dump(run / "run.json", {"stage": "release-regression", "state": "SUBMITTED",
                           "source_hashes": hashes, "jobs": [
                               {"job": "fixture-job", "output": str(result.parent.parent),
                                "manifest": str(run / "job-0.json")}]})
    dump(run / "job-0.json", {"kind": "Job", "metadata": {"name": "fixture-job"},
                             "spec": {"backoffLimit": 0, "template": {"spec": {
                                 "restartPolicy": "Never", "containers": [{
                                     "resources": {"limits": {"nvidia.com/gpu": 1}},
                                     "command": ["python3", str(source / execution.RUNNER),
                                                 "--artifact-dir", str(result.parent.parent),
                                                 "--target", target, "--expression",
                                                 audit.CONTRACTS[suite]["selection"]]}]}}}})
    arguments = dict(result_path=result, result_sha256=execution.sha(result.read_bytes()),
                     run_path=run / "run.json",
                     run_sha256=execution.sha((run / "run.json").read_bytes()),
                     junit_path=result.parent / "pytest.xml",
                     junit_sha256=execution.sha((result.parent / "pytest.xml").read_bytes()))
    bundle = audit.seal(suite=suite, current_root=current, **arguments)
    # Self-proof: real validator, complete frozen identity set, no test import.
    assert len(bundle["execution"]["proofs"]) == audit.CONTRACTS[suite]["cuda_count"]
    assert audit.validate(bundle, pins=audit.caller_pins(**arguments),
                          current_root=current)["verdict"] == "PASS"
    return suite, bundle, current, run, arguments


def reseal_xml(bundle, arguments, root):
    path = Path(bundle["execution"]["files"]["pytest.xml"]["path"])
    path.write_bytes(ET.tostring(root))
    descriptor = execution.descriptor(path)
    bundle["execution"]["files"]["pytest.xml"] = descriptor
    bundle["pins"]["junit"] = descriptor
    arguments["junit_sha256"] = descriptor["sha256"]


def reseal_proof(bundle, identity, saved, arguments):
    root = ET.fromstring(execution.bound(bundle["execution"]["files"]["pytest.xml"]))
    node = next(node for node in root.iter("testcase")
                if node.get("classname") + "::" + node.get("name") == identity)
    props = {p.get("name"): p for p in node.findall("./properties/property")}
    path = Path(next(p.get("value") for name, p in props.items() if name.endswith("_path")))
    overwrite(path, saved)
    item = execution.descriptor(path)
    bundle["execution"]["proofs"][str(path)] = item
    next(p for name, p in props.items() if name.endswith("_sha256")).set("value", item["sha256"])
    reseal_xml(bundle, arguments, root)


def validate(evidence):
    _, bundle, current, _, arguments = evidence
    return audit.validate(bundle, pins=audit.caller_pins(**arguments), current_root=current)


def test_full_execution_replay_and_legitimate_refinement_rejections(evidence):
    suite, _, _, _, _ = evidence
    report = validate(evidence)
    assert report["tests"]["tests"] == (21 if suite == "native_conic" else 7)
    assert report["cuda_testcases"] == (19 if suite == "native_conic" else 7)
    assert not report["certified"] and not report["qualified"]
    assert report["final_source"] is True and report["development_replay"] is False
    assert report["commercial"] == report["performance"] == "UNVERIFIED"
    if suite == "refinement":
        rejected = [case for case in report["cases"] if not case["candidate_accepted"]]
        assert len(rejected) == 3
        assert {case["candidate_status"] for case in rejected} == {"unknown", "rejected"}


@pytest.mark.parametrize("pin", ["result_sha256", "run_sha256", "junit_sha256"])
def test_caller_pins_required(evidence, pin):
    _, _, current, _, arguments = evidence
    arguments[pin] = "0" * 64
    with pytest.raises(execution.AuditError, match="caller artifact pin"):
        audit.seal(suite=evidence[0], current_root=current, **arguments)


@pytest.mark.parametrize("mutation", ["property_pin", "extra_property", "shared_path", "identity"])
def test_exact_properties_and_identities_reject_resealed_mutants(evidence, mutation):
    _, bundle, _, _, arguments = evidence
    root = ET.fromstring(execution.bound(bundle["execution"]["files"]["pytest.xml"]))
    nodes = [n for n in root.iter("testcase") if n.find("properties") is not None]
    first, second = nodes[:2]
    if mutation == "property_pin":
        first.find("./properties/property[@name='" + audit.CONTRACTS[evidence[0]]["property"]
                   + "_sha256']").set("value", "0" * 64)
    elif mutation == "extra_property":
        ET.SubElement(first.find("properties"), "property", name="unknown", value="value")
    elif mutation == "shared_path":
        second.remove(second.find("properties"))
        second.append(copy.deepcopy(first.find("properties")))
        path = first.find("./properties/property").get("value")
        # Execution's proof-path set must also be resealed for semantic rejection.
        bundle["execution"]["proofs"] = {
            str(Path(p.get("value"))): bundle["execution"]["proofs"][str(Path(p.get("value")))]
            for p in root.findall(".//property") if p.get("name").endswith("_path")}
        assert path
    else:
        first.set("name", first.get("name").replace("[", "[unknown-", 1))
    reseal_xml(bundle, arguments, root)
    with pytest.raises(execution.AuditError):
        validate(evidence)


def test_missing_negative_cannot_pass_after_counts_logs_and_all_pins_are_resealed(evidence):
    suite, bundle, _, _, arguments = evidence
    frozen = bundle["execution"]
    root = ET.fromstring(execution.bound(frozen["files"]["pytest.xml"]))
    xml_suite = root.find("testsuite")
    removed = next(node for node in xml_suite if "negative_controls[" in node.get("name")
                   or "reject_numerical_tamper[" in node.get("name"))
    identity = removed.get("classname") + "::" + removed.get("name")
    path = removed.find("./properties/property").get("value")
    xml_suite.remove(removed)
    count = audit.CONTRACTS[suite]["count"] - 1
    xml_suite.set("tests", str(count))
    bundle["negatives"].pop(identity, None)
    frozen["proofs"].pop(path)
    reseal_xml(bundle, arguments, root)
    result = execution.parse(execution.bound(frozen["result"]), "result")
    result["tests"]["tests"] = count
    overwrite(Path(frozen["result"]["path"]), result)
    frozen["result"] = execution.descriptor(Path(frozen["result"]["path"]))
    frozen["result_sha256"] = arguments["result_sha256"] = frozen["result"]["sha256"]
    bundle["pins"]["result"] = frozen["result"]
    for name, content in (("pytest.log", f"{count} passed in 1s\nreturncode=0\n"),
                          ("execution.log", f"OPENOPT_RELEASE_CUDA verdict=PASS tests={count}\n")):
        Path(frozen["files"][name]["path"]).write_text(content)
        frozen["files"][name] = execution.descriptor(Path(frozen["files"][name]["path"]))
    with pytest.raises(execution.AuditError, match="denominator"):
        validate(evidence)


@pytest.mark.parametrize("mutation", ["x", "multiplier", "model", "tolerance", "membership", "device"])
def test_numerical_and_source_mutants_rejected_even_with_new_pins(evidence, mutation):
    suite, bundle, _, _, arguments = evidence
    identity = next(identity for identity in audit.expected_cases(
        bundle["execution"]["source_snapshot"], suite)
                    if "frozen" in identity)
    definition = audit.expected_cases(bundle["execution"]["source_snapshot"], suite)[identity]
    item = next(item for item in bundle["execution"]["proofs"].values()
                if f"case-{list(audit.expected_cases(bundle['execution']['source_snapshot'], suite)).index(identity)}/"
                in item["path"])
    saved = execution.parse(execution.bound(item), "sidecar")
    if mutation == "x":
        (saved if suite == "native_conic" else saved["refined"])["x"][0] += 0.5
    elif mutation == "multiplier":
        (saved if suite == "native_conic" else saved["refined"])[
            "z" if suite == "native_conic" else "y"][0] *= -1
    elif mutation == "model":
        saved["problem"]["q"][0] += 1
    elif mutation == "tolerance":
        (saved if suite == "native_conic" else saved["config"])[
            "atol" if suite == "native_conic" else "check_atol"] = 10
    elif mutation == "membership":
        saved["source_sha256"].pop(next(iter(saved["source_sha256"])))
    else:
        saved["device"] = "cpu"
    assert definition
    reseal_proof(bundle, identity, saved, arguments)
    with pytest.raises((execution.AuditError, ValueError)):
        validate(evidence)


@pytest.mark.parametrize("mutation", ["dirty_current", "snapshot_doc", "loaded_checker", "negative_artifact"])
def test_source_and_negative_artifact_drift(evidence, mutation, monkeypatch):
    suite, bundle, current, _, _ = evidence
    if mutation == "dirty_current":
        (current / "unknown.txt").write_text("uncommitted")
    elif mutation == "snapshot_doc":
        if suite != "refinement":
            return
        (Path(bundle["execution"]["source_snapshot"]) / "docs/research/refinement.md").write_text("drift")
    elif mutation == "loaded_checker":
        monkeypatch.setattr(audit, "check_loaded", lambda *args: execution.require(
            False, "loaded independent checker", "mismatch control"))
    else:
        if suite != "native_conic":
            return
        item = next(iter(bundle["negatives"].values()))
        Path(item["path"]).write_text("{}")
    with pytest.raises(execution.AuditError):
        validate(evidence)


def test_dynamic_parametrization_and_same_count_domain_changes_rejected(evidence):
    suite, bundle, _, _, _ = evidence
    source = Path(bundle["execution"]["source_snapshot"])
    path = source / next(name for name in audit.SOURCES[suite] if name.startswith("tests/"))
    text = path.read_text()
    path.write_text(text.replace("torch.float32", "factory()", 1))
    with pytest.raises(execution.AuditError, match="dynamic"):
        audit.expected_cases(source, suite)
    path.write_text(text.replace("soc2", "different") if suite == "native_conic"
                    else text.replace("wrong_sign", "different"))
    with pytest.raises(execution.AuditError, match="domain"):
        audit.expected_cases(source, suite)


def cli_arguments(current, run, arguments):
    result = ["--current-root", str(current), "--artifact-dir", str(run / "audit-output")]
    for name in ("result", "run", "junit"):
        result += ["--" + name, str(arguments[name + "_path"]),
                   "--" + name + "-sha256", arguments[name + "_sha256"]]
    return result


def test_cli_seal_accept_and_failure_logs(evidence):
    suite, _, current, run, arguments = evidence
    options = cli_arguments(current, run, arguments)
    assert audit.main(["seal", "--suite", suite, *options]) == 0
    sealed = list((run / "audit-output").glob("seal-*/acceptance.json"))
    assert len(sealed) == 1
    seal_report = execution.parse(sealed[0].read_bytes(), "seal report")
    assert seal_report["verdict"] == "SEALED"
    accept_options = ["accept", "--bundle", seal_report["bundle"],
                      "--sha256", seal_report["bundle_sha256"], *options]
    assert audit.main(accept_options) == 0
    reports = list((run / "audit-output").glob("accept-*/acceptance.json"))
    assert len(reports) == 1
    report = execution.parse(reports[0].read_bytes(), "accept report")
    assert report["verdict"] == "PASS" and report["tests"]["tests"] == audit.CONTRACTS[suite]["count"]
    assert "returncode=0" in (reports[0].parent / "execution.log").read_text()
    accept_options[accept_options.index("--sha256") + 1] = "0" * 64
    assert audit.main(accept_options) == 1
    failed = [path for path in (run / "audit-output").glob("accept-*/acceptance.json")
              if execution.parse(path.read_bytes(), "report")["verdict"] == "FAIL"]
    assert len(failed) == 1
    assert "caller bundle pin" in (failed[0].parent / "failure.log").read_text()
    assert "returncode=1" in (failed[0].parent / "execution.log").read_text()


def test_clean_git_guard_rejects_untracked_source(evidence):
    _, _, current, _, _ = evidence
    audit.clean_sources(current, audit.AUDITORS)
    (current / "tests/untracked.py").write_text('raise AssertionError("not committed")')
    with pytest.raises(execution.AuditError, match="clean committed"):
        audit.clean_sources(current, audit.AUDITORS)


def test_retrospective_auditor_cannot_claim_final_source():
    sources = dict.fromkeys(audit.AUDITORS, "a" * 64)
    auditors = {name: {"sha256": "a" * 64, "bytes": 1} for name in audit.AUDITORS}
    audit.final_source_binding(sources, auditors, development_replay=False)
    sources.pop(audit.AUDITORS[0])
    with pytest.raises(execution.AuditError, match="no retrospective promotion"):
        audit.final_source_binding(sources, auditors, development_replay=False)
    audit.final_source_binding(sources, auditors, development_replay=True)
