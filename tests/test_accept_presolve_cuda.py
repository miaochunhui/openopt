"""Real file/JUnit/NPZ replay fixtures; source is parsed, never run or solved."""

import copy
import json
from pathlib import Path
import subprocess
from xml.etree import ElementTree as ET

import numpy as np
import pytest

from openopt.certificates import check_primal_infeasibility, check_qp_candidate
from scripts import accept_presolve_cuda as audit


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    return result.stdout.decode().strip()


def original_arrays(kind, dtype):
    if kind == "constant":
        P, q, A, lo, hi = [[2]], [0], [[1]], [2], [2]
        F, K, rows, kept, removed = [], [0], [0], [], [0]
        x, y, shifts, offset = [2], [-4], [2], 4
        Ph, qh, Ah, lh, uh = [], [], [], [], []
    elif kind == "infeasible":
        P, q, A, lo, hi = [[2]], [0], [[1], [2]], [2, 5], [2, 9]
        F, K, rows, kept, removed = [], [0], [0], [1], [0]
        x, y, shifts, offset = [2], [-4, 0], [2, 4], 4
        Ph, qh, Ah, lh, uh = [], [], [[]], [1], [5]
    elif kind == "active":
        P, q, A, lo, hi = [[2, 0], [0, 2]], [0, -4], [[1, 0], [1, 1]], [2, -np.inf], [2, 3]
        F, K, rows, kept, removed = [1], [0], [0], [1], [0]
        x, y, shifts, offset = [2, 1], [-6, 2], [2, 2], 4
        Ph, qh, Ah, lh, uh = [[2]], [-4], [[1]], [-np.inf], [1]
    else:
        P, q, A, lo, hi = [[2, 0], [0, 2]], [0, -2], [[1, 0], [0, 0], [0, 1]], [2, -1, 0], [2, 1, 2]
        F, K, rows, kept, removed = [1], [0], [0], [2], [0, 1]
        x, y, shifts, offset = [2, 1], [-4, 0, 0], [2, 0, 0], 4
        Ph, qh, Ah, lh, uh = [[2]], [-2], [[1]], [0], [2]
    original = dict(zip(audit.MODEL_NAMES, (P, q, A, lo, hi)))
    reduced = dict(zip(audit.MODEL_NAMES, (Ph, qh, Ah, lh, uh)))
    arrays = {}
    for prefix in ("original", "working"):
        for name, value in original.items():
            arrays[prefix + "_" + name] = np.asarray(value, dtype=np.float64 if prefix == "original" else dtype)
    for name, value in reduced.items():
        arrays["reduced_" + name] = np.asarray(value, dtype=dtype)
    arrays["reduced_P"] = arrays["reduced_P"].reshape(len(F), len(F))
    arrays["reduced_A"] = arrays["reduced_A"].reshape(len(kept), len(F))
    for name, value in zip(audit.MAP_NAMES, (F, K, [2], rows, kept, removed, shifts)):
        arrays[name] = np.asarray(value, dtype=dtype if name in ("fixed_values", "row_shifts") else np.int64)
    arrays.update(x=np.asarray(x, dtype=dtype), y=np.asarray(y, dtype=dtype),
                  objective_offset=np.array(offset, dtype=np.float64))
    if kind == "infeasible":
        arrays["infeasibility_direction"] = np.asarray([2, -1], dtype=dtype)
    return arrays


def sidecar(path, arrays, dtype, source_hashes):
    problem = audit.SimpleNamespace(**{name: arrays["original_" + name] for name in audit.MODEL_NAMES})
    tolerance = 2e-4 if dtype == "float32" else 1e-7
    evidence = check_qp_candidate(problem, arrays["x"], arrays["y"], atol=tolerance, rtol=tolerance)
    ray = check_primal_infeasibility(problem, arrays["infeasibility_direction"]) \
        if "infeasibility_direction" in arrays else None
    np.savez(path.parent / "full-model.npz", **arrays)
    saved = {
        "schema_version": 1, "kind": "presolve_original_coordinate_replay",
        "checked_at": "2026-10-10T09:00:05+00:00", "device": "cuda:0",
        "status": "infeasible_numerical" if ray is not None else "solved",
        "objective": evidence.metrics["primal_objective"],
        "presolve_config": dict(enabled=True, remove_zero_rows=True, fix_identity_equalities=True,
                                check_atol=tolerance, check_rtol=tolerance),
        "source_sha256": {name: source_hashes[name] for name in audit.SOURCES},
        "data": {"path": "full-model.npz", "sha256": audit.descriptor(path.parent / "full-model.npz")["sha256"]},
        "evidence": evidence.to_dict(), "infeasibility_evidence": ray.to_dict() if ray is not None else None,
        "qualification": "Synthetic acceptance-tool fixture, not CUDA execution evidence.",
    }
    dump(path, saved)
    return saved


@pytest.fixture
def evidence(tmp_path, monkeypatch):
    monkeypatch.setattr(audit.execution, "ARTIFACT_ROOT", tmp_path.resolve())
    run, current = tmp_path / "run", tmp_path / "current"
    snapshot = run / "source"
    producer = (
        'import pytest\nimport torch\n'
        '@pytest.mark.skipif(False, reason="fixture")\n'
        '@pytest.mark.parametrize("kind", ["coupled", "active", "constant", "infeasible"])\n'
        '@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])\n'
        'def test_cuda_presolve_original_dual_gap_full_model_sidecar(kind, dtype):\n'
        '    assert result.x.device == result.y.device == torch.device("cuda:0")\n'
        '    assert result.x.dtype == result.y.dtype == dtype\n'
        '    assert result.infeasibility_direction.device == torch.device("cuda:0")\n'
        '    assert getattr(result.presolved, name).device == torch.device("cuda:0")\n'
        '    raise RuntimeError("Fixture source must never be executed")\n'
    )
    sources = {
        "src/openopt/presolve.py": b'"""Fixture, never executed."""\n',
        "src/openopt/qp.py": (audit.ROOT / "src/openopt/qp.py").read_bytes(),
        "src/openopt/certificates.py": (audit.ROOT / "src/openopt/certificates.py").read_bytes(),
        audit.SOURCE: producer.encode(),
        audit.execution.RUNNER: (audit.ROOT / audit.execution.RUNNER).read_bytes(),
        "scripts/accept_release_cuda.py": Path(audit.execution.__file__).read_bytes(),
        audit.AUDITOR: Path(audit.__file__).read_bytes(),
    }
    sources.update({name: b"Explicit evidence dependency fixture.\n"
                    for name in getattr(audit.execution, "EXPLICIT_SOURCE_FILES", ())})
    for root in (current, snapshot):
        for name, data in sources.items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    git(current, "init", "-q")
    git(current, "add", "--", *sources)
    git(current, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "Acceptance fixture")
    hashes = audit.execution.source_files(snapshot)
    dump(snapshot / "source-hashes.json", hashes)
    result = run / "case-0/cuda-check-fixture/result.json"
    result.parent.mkdir(parents=True)
    expected = audit.expected_cases(snapshot)
    xml = ET.Element("testsuites")
    suite = ET.SubElement(xml, "testsuite", {name: str(value) for name, value in audit.COUNT.items()})
    for index, (identity, parameters) in enumerate(expected.items()):
        path = result.parent / "pytest-tmp" / ("case-" + str(index)) / "presolve-acceptance.json"
        path.parent.mkdir(parents=True)
        arrays = original_arrays(parameters["kind"], np.dtype(parameters["dtype"]))
        saved = sidecar(path, arrays, parameters["dtype"], hashes)
        classname, name = identity.split("::")
        case = ET.SubElement(suite, "testcase", classname=classname, name=name, time="0.1")
        properties = ET.SubElement(case, "properties")
        for prop, value in {
            "presolve_acceptance_path": str(path),
            "presolve_acceptance_sha256": audit.descriptor(path)["sha256"],
            "status": saved["status"],
        }.items():
            ET.SubElement(properties, "property", name=prop, value=value)
    (result.parent / "pytest.xml").write_bytes(ET.tostring(xml))
    (result.parent / "pytest.log").write_text("8 passed, 91 deselected in 0.8s\n\nreturncode=0\n")
    (result.parent / "execution.log").write_text("OPENOPT_RELEASE_CUDA verdict=PASS tests=8\n")
    dump(result.parent / "telemetry.json", [{"rc": 0, "gpu": "GPU-fixture, 1, 20, 10",
                                            "cpu": {"usage_usec": 1}}])
    report = {
        "verdict": "PASS", "returncode": 0, "targets": [audit.SOURCE], "selection": audit.SELECTION,
        "source_sha256": {name: hashes[name] for name in sources if name.startswith("src/openopt/")
                          or name in (audit.SOURCE, audit.execution.RUNNER)},
        "tests": audit.COUNT, "hardware": {
            "gpu": "NVIDIA RTX PRO 4500 Blackwell", "node": "fixture",
            "torch": "fixture", "cuda": "fixture", "cpu_quota": "max 100000",
        }, "telemetry_samples": 1,
        "started_at": "2026-10-10T09:00:01+00:00", "finished_at": "2026-10-10T09:00:06+00:00",
    }
    dump(result, report)
    record = {"stage": "release-regression", "state": "SUBMITTED", "namespace": "devtest",
              "source_sha": git(current, "rev-parse", "HEAD"), "source_hashes": hashes,
              "jobs": [{"job": "fixture-job", "output": str(result.parent.parent),
                        "manifest": str(run / "job-0.json")}]}
    record["source_identity"] = {"commit": record["source_sha"], "tree": git(current, "rev-parse", "HEAD^{tree}"),
                                 "clean": True, "sha256": hashes}
    dump(run / "run.json", record)
    command = ["python3", str(snapshot / audit.execution.RUNNER), "--artifact-dir",
               str(result.parent.parent), "--target", audit.SOURCE, "--expression", audit.SELECTION]
    pod_spec = {"restartPolicy": "Never", "runtimeClassName": "nvidia",
                "nodeSelector": {"nvidia.com/gpu.product": audit.PRODUCT},
                "containers": [{"name": "benchmark", "image": "fixture-image", "command": command,
                                "resources": {"limits": {"nvidia.com/gpu": 1},
                                              "requests": {"nvidia.com/gpu": 1}}}]}
    job = {"kind": "Job", "metadata": {"name": "fixture-job"},
           "spec": {"backoffLimit": 0, "template": {"spec": pod_spec}}}
    dump(run / "job-0.json", job)
    actual_job = copy.deepcopy(job)
    actual_job["metadata"].update(uid="fixture-job-uid", namespace="devtest")
    actual_job["status"] = {
        "succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}],
        "startTime": "2026-10-10T09:00:00Z", "completionTime": "2026-10-10T09:00:07Z",
    }
    actual_spec = copy.deepcopy(pod_spec)
    actual_spec["nodeName"] = "fixture"
    actual_spec["containers"][0]["resources"] = {
        "limits": {"nvidia.com/gpu": "1"}, "requests": {"nvidia.com/gpu": "1"}}
    pod = {"kind": "Pod", "metadata": {
        "name": "fixture-pod", "uid": "fixture-pod-uid", "namespace": "devtest",
        "ownerReferences": [{"kind": "Job", "name": "fixture-job", "uid": "fixture-job-uid", "controller": True}],
    }, "spec": actual_spec, "status": {"phase": "Succeeded", "containerStatuses": [{
        "name": "benchmark", "restartCount": 0, "containerID": "fixture-id", "imageID": "fixture-image-id",
        "lastState": {}, "state": {"terminated": {"exitCode": 0, "reason": "Completed",
                                                "startedAt": "2026-10-10T09:00:00Z",
                                                "finishedAt": "2026-10-10T09:00:07Z"}}}]}}
    dump(run / "fixture-job-status.json", {"job": actual_job, "pods": {"items": [pod]}})
    arguments = {}
    for name, path in (("result", result), ("run", run / "run.json"),
                       ("junit", result.parent / "pytest.xml"), ("status", run / "fixture-job-status.json")):
        arguments[name + "_path"] = path
        arguments[name + "_sha256"] = audit.descriptor(path)["sha256"]
    bundle = audit.seal(current_root=current, **arguments)
    # Actual validator and eight NPZ files were read; fake test source was not executed.
    assert audit.validate(bundle, pins=bundle["pins"], current_root=current)["cuda_testcases"] == 8
    return bundle, current, run, arguments


def repin(bundle, name, value=None):
    path = Path(bundle["pins"][name]["path"])
    if value is not None:
        dump(path, value)
    item = audit.descriptor(path)
    bundle["pins"][name] = item
    if name in ("result", "run"):
        bundle["execution"][name] = item
    if name == "result":
        bundle["execution"]["result_sha256"] = item["sha256"]
    if name == "junit":
        bundle["execution"]["files"]["pytest.xml"] = item


def mutate_sidecar(bundle, identity, *, change_json=None, change_arrays=None):
    item = bundle["npz"][identity]
    data_path = Path(item["path"])
    path = data_path.parent / "presolve-acceptance.json"
    saved = json.loads(path.read_text())
    if change_arrays is not None:
        arrays = audit.npz_arrays(data_path.read_bytes())
        change_arrays(arrays)
        np.savez(data_path, **arrays)
        bundle["npz"][identity] = audit.descriptor(data_path)
        saved["data"]["sha256"] = bundle["npz"][identity]["sha256"]
    if change_json is not None:
        change_json(saved)
    dump(path, saved)
    descriptor = audit.descriptor(path)
    bundle["execution"]["proofs"][str(path)] = descriptor
    xml_path = Path(bundle["pins"]["junit"]["path"])
    tree = ET.parse(xml_path)
    for case in tree.findall(".//testcase"):
        if case.get("classname") + "::" + case.get("name") == identity:
            for prop in case.findall("./properties/property"):
                if prop.get("name") == "presolve_acceptance_sha256":
                    prop.set("value", descriptor["sha256"])
    xml_path.write_bytes(ET.tostring(tree.getroot()))
    repin(bundle, "junit")


def test_positive_all_eight_original_replays_not_commercial(evidence):
    bundle, current, _, _ = evidence
    result = audit.validate(bundle, pins=bundle["pins"], current_root=current)
    assert result["tests"] == audit.COUNT
    assert len(result["cases"]) == 8
    assert sum(c["status"] == "solved" for c in result["cases"]) == 6
    assert sum(c["status"] == "infeasible_numerical" for c in result["cases"]) == 2
    assert result["qualified"] is result["certified"] is False
    assert result["commercial"] == result["performance"] == "UNVERIFIED"


@pytest.mark.parametrize("name", ["result", "run", "junit", "status"])
def test_caller_artifact_pin_mismatch(evidence, name):
    _, _, _, args = evidence
    args[name + "_sha256"] = "0" * 64
    with pytest.raises(audit.AuditError, match="pin"):
        audit.caller_pins(**args)


@pytest.mark.parametrize("mutation", ["x", "y", "fixed", "offset", "shift", "bounds", "q", "removed", "map", "extra", "nonfinite", "dtype"])
def test_resealed_npz_semantic_mutants_fail(evidence, mutation):
    bundle, current, _, _ = evidence
    identity = next(k for k in bundle["npz"] if "coupled" in k)

    def change(a):
        if mutation == "x":
            a["x"][1] += .5
        elif mutation == "y":
            a["y"][0] *= -1
        elif mutation == "fixed":
            a["fixed_values"][0] += 1
        elif mutation == "offset":
            a["objective_offset"] += 1
        elif mutation == "shift":
            a["row_shifts"][0] += 1
        elif mutation == "bounds":
            a["reduced_l"][0] += 1
        elif mutation == "q":
            a["reduced_q"][0] += 1
        elif mutation == "removed":
            a["working_A"][1, 1] = 1
        elif mutation == "map":
            a["free_variables"][0] = 0
        elif mutation == "extra":
            a["extra"] = np.array([1.])
        elif mutation == "nonfinite":
            a["original_q"][0] = np.nan
        else:
            a["x"] = a["x"].astype(np.float64)
    mutate_sidecar(bundle, identity, change_arrays=change)
    with pytest.raises((audit.AuditError, ValueError)):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


@pytest.mark.parametrize("mutation", ["device", "source", "status", "tolerance", "evidence", "data"])
def test_resealed_json_mutants_fail(evidence, mutation):
    bundle, current, _, _ = evidence
    identity = next(k for k in bundle["npz"] if "coupled" in k)

    def change(saved):
        if mutation == "device":
            saved["device"] = "cpu"
        elif mutation == "source":
            saved["source_sha256"]["src/openopt/presolve.py"] = "0" * 64
        elif mutation == "status":
            saved["status"] = "max_iterations"
        elif mutation == "tolerance":
            saved["presolve_config"]["check_atol"] = 1
        elif mutation == "data":
            saved["data"]["path"] = "../other.npz"
        else:
            saved["evidence"]["metrics"]["duality_gap"] = 100
    mutate_sidecar(bundle, identity, change_json=change)
    with pytest.raises(audit.AuditError):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


def test_wrong_original_ray_resealed_rejected(evidence):
    bundle, current, _, _ = evidence
    identity = next(k for k in bundle["npz"] if "infeasible" in k)
    mutate_sidecar(bundle, identity, change_arrays=lambda a: a["infeasibility_direction"].__imul__(-1))
    with pytest.raises(audit.AuditError, match="Farkas"):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


@pytest.mark.parametrize("mutation", ["skip", "missing", "duplicate", "wrong_domain", "property_hash"])
def test_resealed_junit_complete_denominator(evidence, mutation):
    bundle, current, _, _ = evidence
    path = Path(bundle["pins"]["junit"]["path"])
    tree = ET.parse(path)
    suite = tree.getroot().find("testsuite")
    cases = suite.findall("testcase")
    if mutation == "skip":
        suite.set("skipped", "1")
    elif mutation == "missing":
        suite.remove(cases[-1])
    elif mutation == "duplicate":
        cases[-1].set("name", cases[0].get("name"))
    elif mutation == "wrong_domain":
        cases[0].set("name", cases[0].get("name").replace("coupled", "unknown"))
    else:
        cases[0].find('./properties/property[@name="presolve_acceptance_sha256"]').set("value", "0" * 64)
    path.write_bytes(ET.tostring(tree.getroot()))
    repin(bundle, "junit")
    with pytest.raises(audit.AuditError):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


@pytest.mark.parametrize("mutation", ["phase", "exit", "restart", "owner", "node", "cards", "job_failed", "command"])
def test_resealed_real_terminal_status_mutations(evidence, mutation):
    bundle, current, _, _ = evidence
    status = json.loads(Path(bundle["pins"]["status"]["path"]).read_text())
    pod = status["pods"]["items"][0]
    if mutation == "phase":
        pod["status"]["phase"] = "Running"
    elif mutation == "exit":
        pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 1
    elif mutation == "restart":
        pod["status"]["containerStatuses"][0]["restartCount"] = 1
    elif mutation == "owner":
        pod["metadata"]["ownerReferences"][0]["uid"] = "wrong"
    elif mutation == "node":
        pod["spec"]["nodeName"] = "other"
    elif mutation == "cards":
        pod["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"] = "0"
    elif mutation == "job_failed":
        status["job"]["status"]["failed"] = 1
    else:
        pod["spec"]["containers"][0]["command"].append("--other")
    repin(bundle, "status", status)
    with pytest.raises(audit.AuditError):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


@pytest.mark.parametrize("mutation", ["core", "auditor", "test", "extra_core", "dirty"])
def test_no_current_source_drift_acceptance(evidence, mutation):
    bundle, current, _, _ = evidence
    target = current / {
        "core": "src/openopt/qp.py", "auditor": audit.AUDITOR, "test": audit.SOURCE,
        "extra_core": "src/openopt/new.py", "dirty": "unknown.txt",
    }[mutation]
    target.write_bytes(target.read_bytes() + b"\n" if target.exists() else b"new\n")
    with pytest.raises(audit.AuditError):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


@pytest.mark.parametrize("target", ["src/openopt/qp.py", audit.AUDITOR,
                                   "scripts/accept_release_cuda.py", audit.SOURCE])
def test_committed_drift_not_retroactively_accepted(evidence, target):
    bundle, current, _, _ = evidence
    path = current / target
    path.write_bytes(path.read_bytes() + b"\n# A different committed source.\n")
    git(current, "add", "--", target)
    git(current, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "-qm", "Changed source, old run remains unchanged", "--", target)
    assert not git(current, "status", "--porcelain")
    with pytest.raises(audit.AuditError, match="source|difference"):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


def test_current_auditor_missing_from_old_snapshot_rejected(evidence):
    bundle, current, run, _ = evidence
    (run / "source" / audit.AUDITOR).unlink()
    hashes = audit.execution.source_files(run / "source")
    dump(run / "source/source-hashes.json", hashes)
    bundle["execution"]["sources"] = hashes
    bundle["execution"]["source_manifest"] = audit.descriptor(run / "source/source-hashes.json")
    record = json.loads(Path(bundle["pins"]["run"]["path"]).read_text())
    record["source_hashes"] = record["source_identity"]["sha256"] = hashes
    repin(bundle, "run", record)
    with pytest.raises(audit.AuditError, match="frozen presolve auditor"):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


@pytest.mark.parametrize("mutation", ["primal", "stationarity", "sign", "complementarity"])
def test_failed_original_gates_even_matching_saved_evidence(evidence, mutation):
    bundle, current, _, _ = evidence
    identity = next(k for k in bundle["npz"] if "dtype1-active" in k)

    def change_arrays(a):
        if mutation == "primal":
            a["x"][1] = 1.5
            a["y"][:] = [-5, 1]
        elif mutation == "stationarity":
            a["x"][1] = 0
            a["y"][:] = [-4, 0]
        elif mutation == "sign":
            a["x"][1] = 1
            a["original_q"][1] = a["working_q"][1] = a["reduced_q"][0] = 0
            a["y"][:] = [-2, -2]
        else:
            a["x"][1] = 0
            a["y"][:] = [-8, 4]

    def change_json(saved):
        arrays = audit.npz_arrays(Path(bundle["npz"][identity]["path"]).read_bytes())
        problem = audit.SimpleNamespace(**{
            name: arrays["original_" + name] for name in audit.MODEL_NAMES})
        check = check_qp_candidate(problem, arrays["x"], arrays["y"], atol=1e-7, rtol=1e-7)
        assert not check.passed
        saved["evidence"] = check.to_dict()
        saved["objective"] = check.metrics["primal_objective"]

    mutate_sidecar(bundle, identity, change_arrays=change_arrays, change_json=change_json)
    with pytest.raises(audit.AuditError, match="all original six gates"):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


def test_mapping_original_constant_zero_dimension(evidence):
    bundle, current, _, _ = evidence
    identity = next(k for k in bundle["npz"] if "constant" in k)
    mutate_sidecar(bundle, identity, change_arrays=lambda a: a.__setitem__("reduced_P", np.ones((1, 1), dtype=a["x"].dtype)))
    with pytest.raises(audit.AuditError, match="reduced dimensions"):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


def test_missing_model_not_reconstructed_from_test(evidence):
    bundle, current, _, _ = evidence
    identity = next(iter(bundle["npz"]))
    mutate_sidecar(bundle, identity, change_arrays=lambda a: a.pop("original_P"))
    with pytest.raises(audit.AuditError, match="missing/extra frozen arrays"):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


def test_symlink_npz_rejected_even_correct_hash(evidence):
    bundle, current, run, _ = evidence
    item = next(iter(bundle["npz"].values()))
    path = Path(item["path"])
    outside = run / "aliased.npz"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(outside)
    with pytest.raises(audit.AuditError, match="symlink"):
        audit.validate(bundle, pins=bundle["pins"], current_root=current)


def test_bundle_pin_cli_pass_and_failure_logs(evidence):
    bundle, current, run, args = evidence
    bundle_path = run / "bundle.json"
    dump(bundle_path, bundle)
    pin = audit.descriptor(bundle_path)["sha256"]
    assert audit.accept(bundle_path, pin, current_root=current, **args)["verdict"] == "PASS"
    with pytest.raises(audit.AuditError, match="bundle pin"):
        audit.accept(bundle_path, "0" * 64, current_root=current, **args)
    command = ["accept", "--bundle", str(bundle_path), "--sha256", pin,
               "--current-root", str(current), "--artifact-dir", str(run.parent / "cli")]
    for name in ("result", "run", "junit", "status"):
        command.extend(["--" + name, str(args[name + "_path"]),
                        "--" + name + "-sha256", args[name + "_sha256"]])
    assert audit.main(command) == 0
    output = next((run.parent / "cli").glob("accept-*/acceptance.json"))
    assert json.loads(output.read_text())["verdict"] == "PASS"
    assert "returncode=0" in (output.parent / "execution.log").read_text()
    command[command.index("--sha256") + 1] = "0" * 64
    assert audit.main(command) == 1
    outputs = list((run.parent / "cli").glob("accept-*/acceptance.json"))
    failed = next(p for p in outputs if json.loads(p.read_text())["verdict"] == "FAIL")
    assert "caller bundle pin mismatch" in (failed.parent / "failure.log").read_text()


@pytest.mark.parametrize("mutation", ["object", "empty", "duplicate"])
def test_npz_archive_invalid_members(mutation):
    import io
    import zipfile

    stream = io.BytesIO()
    if mutation == "object":
        np.savez(stream, bad=np.array([object()], dtype=object))
    else:
        with zipfile.ZipFile(stream, "w") as archive:
            if mutation == "duplicate":
                archive.writestr("same.npy", b"invalid")
                with pytest.warns(UserWarning):
                    archive.writestr("same.npy", b"invalid")
    with pytest.raises((audit.AuditError, ValueError)):
        audit.npz_arrays(stream.getvalue())
