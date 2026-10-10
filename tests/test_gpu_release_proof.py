"""Real CPU producer/replay and adversarial evidence, never GPU qualification."""

from copy import deepcopy
import argparse
from collections import Counter
import cProfile
import importlib.util
import json
import os
from pathlib import Path
import pstats
import shutil
import subprocess
import sys
import time
from xml.etree import ElementTree

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpu_proof", ROOT / "scripts/gpu_release_proof.py")
proof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proof)


def freeze_expectations(root, cases):
    """Independent pre-run CPU model freeze, matching the dispatch conversion."""
    root.mkdir(parents=True, exist_ok=True)
    value = {"schema_version": 1, "source_sha256": proof.source_map(ROOT), "cases": {}}
    for case in cases:
        problem = proof.make_problem(case)
        tensors = {name: getattr(problem, name) for name in proof.FIELDS}
        if case.startswith("jacobi_"):
            for name in ("P", "A"):
                matrix = tensors[name].to_sparse()
                if case == "jacobi_csr":
                    matrix = matrix.to_sparse_csr()
                tensors[name] = matrix
        model = {name: proof.save_tensor(root, case + "-original-" + name, tensor)
                 for name, tensor in tensors.items()}
        value["cases"][case] = {"model": model, "model_sha256": proof.model_hash(model)}
    path = root / "proof-expectations.json"
    proof.write(path, value)
    return {"expectations": path, "expectations_sha256": proof.sha(path.read_bytes())}


def attach_expectations(saved, kwargs):
    path = kwargs["expectations"].resolve()
    saved["expectations"] = {
        "path": str(path), "sha256": kwargs["expectations_sha256"],
        "bytes": len(path.read_bytes()),
    }


def rebind_result(root, saved, *, gpu=False, expectations=None):
    """Coordinated pre-seal mutation: reissue all artifact hashes, not old pins."""
    path = root / "result.json"
    result = json.loads(path.read_text())
    result["cases"] = {saved["case"]: store_case(root, saved)}
    result["requested_cases"] = [saved["case"]]
    if expectations is not None:
        attach_expectations(result, expectations)
    if gpu:
        result.update(
            device="cuda:0", scope=proof.GPU_SCOPE, verdict=proof.NUMERICAL_PASS,
            numerical_verdict="PASS", execution_verdict="UNVERIFIED",
        )
        result["hardware"].update(
            requested_device="cuda:0", gpu_count=1, gpu="NVIDIA RTX PRO 4500 Blackwell",
            node="fabricated-audit-node", pod="fabricated-audit-pod", cuda="12.8",
            cpu_quota="200000 100000", cpu_scope="Pod cgroup v2", cuda_capability=[12, 0],
            gpu_uuid="GPU-00000000-0000-0000-0000-000000000001",
        )
        samples = [{
            "at": "2026-10-10T00:00:00+00:00", "cpu": {"usage_usec": 1}, "returncode": 0,
            "stdout": "GPU-00000000-0000-0000-0000-000000000001, NVIDIA RTX PRO 4500 Blackwell, 0, 0, 24576, 580.0",
            "stderr": "",
        }] * 2
        telemetry = root / "telemetry.json"
        telemetry.write_bytes(proof.encode(samples))
        result["telemetry"] = proof.descriptor(root, telemetry)
    path.write_bytes(proof.encode(result))
    digest = proof.sha(path.read_bytes())
    (root / "execution.log").write_text(
        f"OPENOPT_GPU_PROOF verdict={result['verdict']} result_sha256={digest}\nreturncode=0\n"
    )
    return path, digest


def recompute_summary(root, saved):
    model = {k: proof.load_tensor(root, v, k) for k, v in saved["model"].items()}
    vectors = {k: proof.load_tensor(root, v, k) for k, v in saved["vectors"].items()}
    actual = proof.original_checks(model, vectors, saved["status"], saved["atol"], saved["rtol"])
    assert actual["passed"]
    saved["numpy_original"] = actual
    saved["independent_original"] = {
        "passed": True, "checks": actual["checks"], "metrics": actual["metrics"],
    }
    if saved["status"] == "solved":
        saved["objective"] = actual["metrics"]["primal_objective"]
    return actual


def forbid_replay_execution(monkeypatch):
    import openopt.qp as qp
    import openopt.scaling as scaling
    import openopt.operators as operators

    def forbidden(*args, **kwargs):
        raise AssertionError("replay tried to solve/recreate evidence or access CUDA/network")

    for module, names in (
        (qp, ("prepare_qp", "solve_qp")),
        (operators, ("prepare_operator_qp", "conjugate_gradient")),
        (scaling, ("solve_scaled_qp",)),
        (proof, ("make_problem", "execute_case", "run_cases", "hardware", "freeze_sources")),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(proof.subprocess, "run", forbidden)


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    parent = tmp_path_factory.mktemp("gpu-proof-cpu")
    result = proof.run_cases(parent, list(proof.CASES), "cpu", cpu_smoke=True)
    result_hash = proof.sha(result.read_bytes())
    bundle = proof.seal(result, result_hash, ROOT, allow_cpu_smoke=True)
    bundle_path = parent / "bundle.json"
    proof.write(bundle_path, bundle)
    return result, bundle_path


def copy_evidence_file(source, destination):
    """Copy test bytes without sendfile/stat overhead; never alias mutable fixtures."""
    source, destination = Path(source), Path(destination)
    assert source.is_file() and not source.is_symlink()
    with destination.open("xb") as output:
        output.write(source.read_bytes())
    return destination


def copy_evidence_tree(source, destination):
    source, destination = Path(source), Path(destination)
    assert not destination.exists()
    files = []
    for path in sorted(source.rglob("*")):
        assert not path.is_symlink()
        if path.is_file():
            files.append(path.relative_to(source))
    assert files
    for directory in sorted({destination, *((destination / path).parent for path in files)}):
        directory.mkdir(parents=True, exist_ok=True)
    for path in files:
        copy_evidence_file(source / path, destination / path)
    return destination


def case_copy(produced, tmp_path, name):
    result, bundle_path = produced
    root = tmp_path / "evidence"
    sealed = json.loads(bundle_path.read_bytes())["files"]
    case_path = proof.bound(result.parent, sealed[name + ".json"])
    saved = json.loads(case_path.read_bytes())
    files = {name + ".json", "result.json", "execution.log", "telemetry.json"}

    def references(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256", "bytes"}:
                files.add(proof.relative(value["path"]))
            else:
                for item in value.values():
                    references(item)
        elif isinstance(value, list):
            for item in value:
                references(item)

    references(saved)
    files.update(path for path in sealed if path.startswith("source/"))
    expected = {name: sealed[name] for name in files}
    for directory in sorted({(root / name).parent for name in files}):
        directory.mkdir(parents=True, exist_ok=True)
    for name in sorted(files):
        target = root / name
        copy_evidence_file(result.parent / name, target)
    copied = proof.inventory(root)
    assert set(copied) == files, "case-copy file membership differs"
    assert copied == expected, "case-copy original byte descriptors differ"
    return root, saved


def store_case(root, saved):
    path = root / (saved["case"] + ".json")
    path.write_bytes(proof.encode(saved))
    return proof.descriptor(root, path)


def test_real_cpu_run_has_full_original_models_and_no_gpu_claim(produced):
    result, bundle = produced
    saved = json.loads(result.read_text())
    assert saved["verdict"] == "CPU_SMOKE_PASS"
    assert saved["scope"] == proof.CPU_SCOPE
    assert saved["hardware"]["gpu"] is None
    accepted = proof.replay(bundle, proof.sha(bundle.read_bytes()), ROOT, allow_cpu_smoke=True)
    assert accepted["verdict"] == "CPU_SMOKE_REPLAY_PASS"
    assert accepted["case_count"] == len(proof.CASES) == 9
    assert all(c["passed"] for c in accepted["cases"])
    for name, item in saved["cases"].items():
        case = json.loads(proof.bound(result.parent, item).read_text())
        assert set(case["model"]) == set(proof.FIELDS)
        assert case["model_sha256"] == proof.model_hash(case["model"])
        assert case["independent_original"]["passed"]
        if name == "unbounded":
            base = proof.load_tensor(result.parent, case["vectors"]["feasible_base"], "base")
            actual = proof.load_tensor(result.parent, case["vectors"]["x"], "x")
            assert np.array_equal(base, actual)
            assert base[1] > 2


def test_cpu_smoke_cannot_be_sealed_or_replayed_as_cuda(produced):
    result, bundle = produced
    with pytest.raises(proof.ProofError, match="CPU smoke cannot"):
        proof.seal(result, proof.sha(result.read_bytes()), ROOT)
    with pytest.raises(proof.ProofError, match="CPU smoke cannot"):
        proof.replay(bundle, proof.sha(bundle.read_bytes()), ROOT)


def test_explicit_cuda_never_runs_cpu_when_unavailable(tmp_path, monkeypatch):
    expected = freeze_expectations(tmp_path / "expected", ["gap_lp"])
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(proof.ProofError, match="exactly one visible CUDA"):
        proof.run_cases(tmp_path, ["gap_lp"], "cuda:0", **expected)
    reports = list(tmp_path.glob("cuda-proof-*/result.json"))
    assert len(reports) == 1
    saved = json.loads(reports[0].read_text())
    assert saved["verdict"] == "FAIL" and not saved["cases"]
    assert saved["returncode"] == 1
    assert (reports[0].parent / "failure.log").is_file()


@pytest.mark.parametrize("cases", [[], ["gap_lp", "gap_lp"], ["not-a-case"]])
def test_empty_duplicate_unknown_cases_rejected(cases, tmp_path):
    with pytest.raises(proof.ProofError, match="selection"):
        proof.run_cases(tmp_path, cases, "cpu", cpu_smoke=True)


@pytest.mark.parametrize("device,cpu", [("cpu", False), ("cuda:0", True)])
def test_scope_device_cannot_be_silently_changed(device, cpu, tmp_path):
    with pytest.raises(proof.ProofError, match="requires cuda:0"):
        proof.run_cases(tmp_path, ["gap_lp"], device, cpu_smoke=cpu)


def test_replay_never_calls_solvers_or_test_factories(produced, monkeypatch):
    import openopt.qp as qp
    import openopt.scaling as scaling
    import openopt.operators as operators

    def forbidden(*args, **kwargs):
        raise AssertionError("replay tried to solve/recreate evidence")

    monkeypatch.setattr(qp, "solve_qp", forbidden)
    monkeypatch.setattr(qp, "prepare_qp", forbidden)
    monkeypatch.setattr(scaling, "solve_scaled_qp", forbidden)
    monkeypatch.setattr(operators, "conjugate_gradient", forbidden)
    monkeypatch.setattr(operators, "prepare_operator_qp", forbidden)
    monkeypatch.setattr(proof, "make_problem", forbidden)
    _, bundle = produced
    assert proof.replay(bundle, proof.sha(bundle.read_bytes()), ROOT,
                        allow_cpu_smoke=True)["verdict"] == "CPU_SMOKE_REPLAY_PASS"


@pytest.mark.parametrize("mutation", ["bytes", "added", "removed", "source"])
def test_sealed_inventory_rejects_real_disk_tamper(produced, tmp_path, mutation):
    result, bundle_path = produced
    root = result.parent
    bundle = json.loads(bundle_path.read_text())
    before = proof.inventory(root)
    assert before == bundle["files"]
    if mutation == "bytes":
        path = next((root / "arrays").glob("*.npy"))
    elif mutation == "added":
        path = root / "unexpected.json"
    elif mutation == "removed":
        path = root / "gap_lp.json"
    else:
        path = root / "source/src/openopt/qp.py"
    original = None if mutation == "added" else path.read_bytes()
    assert mutation != "added" or not path.exists()
    try:
        if mutation == "removed":
            path.unlink()
        elif mutation == "added":
            path.write_bytes(b"{}")
        else:
            path.write_bytes(original + (b"tamper" if mutation == "bytes" else b"\n# tamper\n"))
        with pytest.raises(proof.ProofError, match="added, removed or changed"):
            proof.replay(bundle_path, proof.sha(bundle_path.read_bytes()), ROOT, allow_cpu_smoke=True)
    finally:
        if original is None:
            path.unlink()
        else:
            path.write_bytes(original)
        assert proof.inventory(root) == before, "shared sealed fixture not fully restored"
    (tmp_path / "sealed-mutation.json").write_bytes(proof.encode({
        "verdict": "PASS", "mutation": mutation, "restored_inventory": before,
        "qualification": "sequential reversible test mutation, no fixture sharing across workers",
    }))


@pytest.mark.parametrize("mutation", ["missing_base", "planted_base", "negated_ray", "checkpoint"])
def test_resealed_direction_tamper_is_rejected_by_original_semantics(produced, tmp_path, mutation):
    root, saved = case_copy(produced, tmp_path, "unbounded")
    if mutation == "missing_base":
        del saved["vectors"]["feasible_base"]
    elif mutation == "planted_base":
        saved["vectors"]["feasible_base"] = proof.save_tensor(
            root, "planted", torch.tensor([0., 2.], dtype=torch.float64))
    elif mutation == "negated_ray":
        old = proof.load_tensor(root, saved["vectors"]["unbounded_direction"], "ray")
        saved["vectors"]["unbounded_direction"] = proof.save_tensor(
            root, "negated", torch.from_numpy(-old))
    else:
        saved["extensions"]["checkpoint"]["iteration"] += 1
    with pytest.raises(proof.ProofError):
        proof.replay_case(root, store_case(root, saved), "cpu")


@pytest.mark.parametrize("mutation", [
    "tolerance", "config", "device", "status", "x", "model", "resident", "metric",
])
def test_resealed_candidate_tamper_is_rejected(produced, tmp_path, mutation):
    root, saved = case_copy(produced, tmp_path, "gap_spd")
    if mutation == "tolerance":
        saved["atol"] = 1e-2
    elif mutation == "config":
        saved["config"]["eps_abs"] = 1e-2
    elif mutation == "device":
        saved["observed"]["x"]["device"] = "cuda:0"
    elif mutation == "status":
        saved["status"] = "max_iterations"
    elif mutation == "x":
        x = proof.load_tensor(root, saved["vectors"]["x"], "x") + 10
        saved["vectors"]["x"] = proof.save_tensor(root, "bad-x", torch.from_numpy(x))
    elif mutation == "model":
        q = proof.load_tensor(root, saved["model"]["q"], "q") + 10
        saved["model"]["q"] = proof.save_tensor(root, "bad-q", torch.from_numpy(q))
        saved["model_sha256"] = proof.model_hash(saved["model"])
    elif mutation == "resident":
        q = proof.load_tensor(root, saved["resident_model"]["q"], "q") + 10
        saved["resident_model"]["q"] = proof.save_tensor(root, "bad-resident-q", torch.from_numpy(q))
    else:
        saved["independent_original"]["metrics"]["duality_gap"] = 10.
    with pytest.raises(proof.ProofError):
        proof.replay_case(root, store_case(root, saved), "cpu")


@pytest.mark.parametrize("mutation", [
    "rhs", "initial", "tolerance", "diagonal", "missing_call", "outer_y",
])
def test_resealed_pcg_tamper_rejects_false_true_residual(produced, tmp_path, mutation):
    root, saved = case_copy(produced, tmp_path, "jacobi_coo")
    extension = saved["extensions"]
    call = extension["inner_calls"][0]
    if mutation in ("rhs", "initial", "outer_y"):
        array = proof.load_tensor(root, call[mutation], mutation) + 10
        call[mutation] = proof.save_tensor(root, "bad-cg-" + mutation, torch.from_numpy(array))
    elif mutation == "tolerance":
        call["tolerance"] = 1e-2
    elif mutation == "diagonal":
        array = proof.load_tensor(root, extension["jacobi_diagonal"], "diagonal") * 2
        extension["jacobi_diagonal"] = proof.save_tensor(root, "bad-diagonal", torch.from_numpy(array))
    else:
        extension["inner_calls"].pop()
    with pytest.raises(proof.ProofError):
        proof.replay_case(root, store_case(root, saved), "cpu")


def test_resealed_scaling_restore_tamper_rejected(produced, tmp_path):
    root, saved = case_copy(produced, tmp_path, "scaling_basic")
    saved["extensions"]["scaling"]["c"] *= 2
    with pytest.raises(proof.ProofError, match="restoration"):
        proof.replay_case(root, store_case(root, saved), "cpu")


def test_current_source_membership_is_bidirectional(produced, tmp_path):
    result, _ = produced
    current = tmp_path / "current"
    shutil.copytree(result.parent / "source", current)
    (current / "src/openopt/unexpected.py").write_text("# unexpected\n")
    with pytest.raises(proof.ProofError, match="bidirectional source difference"):
        proof.validate_result(result, proof.sha(result.read_bytes()), current, allow_cpu_smoke=True)


@pytest.mark.parametrize("text", ['{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'])
def test_ambiguous_nonfinite_json_rejected(text):
    with pytest.raises(proof.ProofError):
        proof.parse(text, "json")


@pytest.mark.parametrize("value", ["../escape", "/absolute", "a/../b", "", "a//b"])
def test_unsafe_relative_paths_rejected(value):
    with pytest.raises(proof.ProofError):
        proof.relative(value)


def test_raw_array_hash_shape_dtype_and_trailing_bytes_are_verified(produced, tmp_path):
    root, saved = case_copy(produced, tmp_path, "gap_lp")
    record = deepcopy(saved["model"]["q"])
    record["shape"] = [4]
    with pytest.raises(proof.ProofError, match="shape"):
        proof.load_tensor(root, record, "q")
    record = deepcopy(saved["model"]["q"])
    path = root / record["components"]["values"]["path"]
    path.write_bytes(path.read_bytes() + b"extra")
    record["components"]["values"] = proof.descriptor(root, path)
    with pytest.raises(proof.ProofError, match="trailing"):
        proof.load_tensor(root, record, "q")


def test_sparse_execution_never_densifies_core(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("sparse GPU core must not densify")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    for case in ("jacobi_coo", "jacobi_csr"):
        assert proof.execute_case(case, tmp_path, "cpu")["path"] == case + ".json"


def test_caller_bundle_pin_required(produced):
    _, bundle = produced
    with pytest.raises(proof.ProofError, match="caller bundle pin"):
        proof.replay(bundle, "0" * 64, ROOT, allow_cpu_smoke=True)


def test_pre_pinned_nine_case_cpu_replay_without_execution(tmp_path, monkeypatch):
    expected = freeze_expectations(tmp_path / "expected", list(proof.CASES))
    result = proof.run_cases(tmp_path / "produced", list(proof.CASES), "cpu", cpu_smoke=True, **expected)
    digest = proof.sha(result.read_bytes())
    saved = json.loads(result.read_text())
    assert saved["numerical_verdict"] == "PASS" and saved["execution_verdict"] == "UNVERIFIED"
    source_before = proof.source_map(ROOT)
    original_before = proof.inventory(result.parent)
    forbid_replay_execution(monkeypatch)
    bundle = proof.seal(result, digest, ROOT, allow_cpu_smoke=True, **expected)
    bundle_path = tmp_path / "bundle.json"
    proof.write(bundle_path, bundle)
    outcome = proof.replay(bundle_path, proof.sha(bundle_path.read_bytes()), ROOT,
                           allow_cpu_smoke=True, **expected)
    assert outcome["verdict"] == "CPU_SMOKE_REPLAY_PASS"
    assert outcome["numerical_verdict"] == "PASS"
    assert outcome["execution_verdict"] == "UNVERIFIED"
    assert outcome["case_count"] == 9 and all(c["passed"] for c in outcome["cases"])
    assert source_before == proof.source_map(ROOT)
    assert original_before == proof.inventory(result.parent)


def test_coordinated_original_model_replacement_rejected(produced, tmp_path, monkeypatch):
    expected = freeze_expectations(tmp_path / "expected", ["gap_lp"])
    root, saved = case_copy(produced, tmp_path, "gap_lp")
    before = proof.model_hash(saved["model"])
    values = {"P": [[1.]], "q": [0.], "A": [[1.]], "l": [-1.], "u": [1.]}
    saved["model"] = {
        k: proof.save_tensor(root, "replacement-" + k, torch.tensor(v, dtype=torch.float64))
        for k, v in values.items()
    }
    saved["resident_model"] = deepcopy(saved["model"])
    saved["model_sha256"] = proof.model_hash(saved["model"])
    saved["vectors"] = {
        k: proof.save_tensor(root, "replacement-" + k, torch.zeros(1, dtype=torch.float64))
        for k in ("x", "y")
    }
    saved["iterations"] = 1
    recompute_summary(root, saved)
    assert before != saved["model_sha256"]
    result, digest = rebind_result(root, saved, expectations=expected)
    forbid_replay_execution(monkeypatch)
    with pytest.raises(proof.ProofError, match="original model differs from caller"):
        proof.seal(result, digest, ROOT, allow_cpu_smoke=True, **expected)


def test_coordinated_cpu_as_cuda_cannot_qualify_execution(produced, tmp_path, monkeypatch):
    expected = freeze_expectations(tmp_path / "expected", ["gap_lp"])
    root, saved = case_copy(produced, tmp_path, "gap_lp")
    saved["device"] = saved["config"]["device"] = "cuda:0"
    for group in ("resident_model", "vectors", "observed"):
        for record in saved[group].values():
            record["device"] = "cuda:0"
    result, digest = rebind_result(root, saved, gpu=True, expectations=expected)
    forbid_replay_execution(monkeypatch)
    with pytest.raises(proof.ProofError, match="independent caller pre-dispatch"):
        proof.seal(result, digest, ROOT)
    bundle = proof.seal(result, digest, ROOT, **expected)
    bundle_path = tmp_path / "forged-bundle.json"
    proof.write(bundle_path, bundle)
    accepted = proof.replay(bundle_path, proof.sha(bundle_path.read_bytes()), ROOT, **expected)
    assert accepted["verdict"] == proof.NUMERICAL_PASS != "PASS"
    assert accepted["numerical_verdict"] == "PASS"
    assert accepted["execution_verdict"] == "UNVERIFIED"
    assert not torch.cuda.is_available()
    value = json.loads(result.read_text())
    value.update(verdict="PASS", execution_verdict="PASS")
    result.write_bytes(proof.encode(value))
    with pytest.raises(proof.ProofError, match="execution or scope"):
        proof.validate_result(result, proof.sha(result.read_bytes()), ROOT, **expected)


def test_coordinated_planted_ray_base_checkpoint_rejected(produced, tmp_path, monkeypatch):
    root, saved = case_copy(produced, tmp_path, "unbounded")
    for name, values in (
        ("x", [0., 2.]), ("feasible_base", [0., 2.]), ("unbounded_direction", [0., 1.]),
    ):
        saved["vectors"][name] = proof.save_tensor(
            root, "planted-" + name, torch.tensor(values, dtype=torch.float64))
    trace = saved["extensions"]["checkpoint"]
    for name, values in (
        ("current_x", [0., 2.]), ("previous_x", [0., 0.]), ("dx", [0., 2.]),
    ):
        trace["tensors"][name] = proof.save_tensor(
            root, "planted-" + name, torch.tensor(values, dtype=torch.float64))
    trace["iteration"] = saved["iterations"] = 1
    trace["unbounded_evidence"] = recompute_summary(root, saved)
    result, digest = rebind_result(root, saved)
    forbid_replay_execution(monkeypatch)
    with pytest.raises(proof.ProofError, match="outside frozen check interval"):
        proof.seal(result, digest, ROOT, allow_cpu_smoke=True)


def test_coordinated_contradictory_terminal_summaries_rejected(produced, tmp_path, monkeypatch):
    root, saved = case_copy(produced, tmp_path, "unbounded")
    other = saved["extensions"]["checkpoint"]["infeasible_evidence"]
    other["passed"] = True
    other["checks"] = {k: True for k in other["checks"]}
    result, digest = rebind_result(root, saved)
    forbid_replay_execution(monkeypatch)
    with pytest.raises(proof.ProofError, match="full direction summary"):
        proof.seal(result, digest, ROOT, allow_cpu_smoke=True)


@pytest.mark.parametrize("mutation", ["rho", "selected_metric", "opposite_metric", "boolean_as_number",
                                     "selected_threshold", "opposite_unavailable", "initial"])
def test_complete_checkpoint_policy_and_both_summaries_checked(produced, tmp_path, mutation):
    root, saved = case_copy(produced, tmp_path, "unbounded")
    trace = saved["extensions"]["checkpoint"]
    if mutation == "rho":
        trace["rho"] *= 2
    elif mutation == "initial":
        tensor = trace["tensors"]["previous_y"]
        trace["tensors"]["previous_y"] = proof.save_tensor(
            root, "bad-initial", torch.from_numpy(proof.load_tensor(root, tensor, "previous-y") + 1))
        current = proof.load_tensor(root, trace["tensors"]["current_y"], "current-y")
        previous = proof.load_tensor(root, trace["tensors"]["previous_y"], "previous-y")
        trace["tensors"]["dy"] = proof.save_tensor(root, "bad-dy", torch.from_numpy(current - previous))
    elif mutation == "boolean_as_number":
        trace["unbounded_evidence"]["passed"] = 1
    else:
        kind = "unbounded" if mutation.startswith("selected") else "infeasible"
        summary = trace[kind + "_evidence"]
        if mutation.endswith("metric"):
            summary["metrics"]["input_direction_norm"] += 1
        elif mutation.endswith("threshold"):
            summary["thresholds"]["descent_margin"] *= 2
        else:
            summary["unavailable"]["forged"] = "forged"
    with pytest.raises(proof.ProofError):
        proof.replay_case(root, store_case(root, saved), "cpu")


def test_formal_gpu_requires_pre_dispatch_expectations_before_hardware(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("missing expectations must fail before hardware/solve")
    monkeypatch.setattr(proof, "hardware", forbidden)
    monkeypatch.setattr(proof, "execute_case", forbidden)
    with pytest.raises(proof.ProofError, match="formal CUDA run requires"):
        proof.run_cases(tmp_path, ["gap_lp"], "cuda:0")
    reports = list(tmp_path.glob("cuda-proof-*/result.json"))
    assert len(reports) == 1
    value = json.loads(reports[0].read_text())
    assert value["numerical_verdict"] == "FAIL"
    assert value["execution_verdict"] == "UNVERIFIED" and not value["cases"]


@pytest.mark.parametrize("mutation", ["schema", "missing_case", "source", "raw", "digest", "device",
                                     "boolean_schema", "nonmapping_model"])
def test_trusted_expectations_reject_mutation(tmp_path, mutation):
    expected = freeze_expectations(tmp_path, ["gap_lp"])
    path = expected["expectations"]
    saved = json.loads(path.read_text())
    if mutation == "schema":
        saved["scope"] = "unexpected"
    elif mutation == "missing_case":
        saved["cases"] = {}
    elif mutation == "source":
        saved["source_sha256"]["src/openopt/qp.py"] = "0" * 64
    elif mutation == "raw":
        item = saved["cases"]["gap_lp"]["model"]["q"]["components"]["values"]
        (tmp_path / item["path"]).write_bytes(b"wrong bytes")
    elif mutation == "digest":
        saved["cases"]["gap_lp"]["model_sha256"] = "0" * 64
    elif mutation == "device":
        saved["cases"]["gap_lp"]["model"]["q"]["device"] = "cuda:0"
    elif mutation == "boolean_schema":
        saved["schema_version"] = True
    else:
        saved["cases"]["gap_lp"]["model"] = []
    path.write_bytes(proof.encode(saved))
    with pytest.raises(proof.ProofError):
        proof.load_expectations(path, proof.sha(path.read_bytes()), ROOT)


def test_runtime_uuid_matches_cuda_properties_and_telemetry(monkeypatch):
    from types import SimpleNamespace

    uuid = "00000000-0000-0000-0000-000000000001"
    original_read = Path.read_text

    def pod_quota(path, *args, **kwargs):
        if path == Path("/sys/fs/cgroup/cpu.max"):
            return "200000 100000"
        return original_read(path, *args, **kwargs)

    # CUDA is simulated here; its mandatory Pod cgroup must be simulated too.
    monkeypatch.setattr(Path, "read_text", pod_quota)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "NVIDIA RTX PRO 4500 Blackwell")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(uuid=uuid))
    monkeypatch.setenv("OPENOPT_NODE", "fixture-node")
    monkeypatch.setenv("HOSTNAME", "fixture-pod")
    hw = proof.hardware("cuda:0")
    assert hw["gpu_uuid"] == "GPU-" + uuid
    sample = {
        "at": "2026-10-10T00:00:00+00:00", "cpu": {"usage_usec": 1}, "returncode": 0,
        "stdout": f"GPU-{uuid}, {hw['gpu']}, 0, 0, 24576, 580.0",
    }
    proof.telemetry_check([sample, sample], "cuda:0", gpu_uuid=hw["gpu_uuid"], gpu_name=hw["gpu"])
    with pytest.raises(proof.ProofError, match="card identity mismatch"):
        proof.telemetry_check([sample, sample], "cuda:0",
                              gpu_uuid="GPU-00000000-0000-0000-0000-000000000002",
                              gpu_name=hw["gpu"])


def test_expectations_descriptor_and_subset_are_caller_bound(produced, tmp_path, monkeypatch):
    expected = freeze_expectations(tmp_path / "expected", ["gap_lp", "gap_spd"])
    root, saved = case_copy(produced, tmp_path, "gap_lp")
    result, digest = rebind_result(root, saved, expectations=expected)
    forbid_replay_execution(monkeypatch)
    accepted = proof.validate_result(result, digest, ROOT, allow_cpu_smoke=True, **expected)
    assert accepted["case_count"] == 1 and accepted["execution_verdict"] == "UNVERIFIED"
    with pytest.raises(proof.ProofError, match="expectations differ"):
        proof.validate_result(result, digest, ROOT, allow_cpu_smoke=True)
    with pytest.raises(proof.ProofError, match="caller expectations pin"):
        proof.validate_result(result, digest, ROOT, allow_cpu_smoke=True,
                              expectations=expected["expectations"], expectations_sha256="0" * 64)


def test_execute_case_rejects_changed_model_before_solving(tmp_path, monkeypatch):
    expected = freeze_expectations(tmp_path / "expected", ["gap_lp"])
    manifest, _ = proof.load_expectations(**{
        "path": expected["expectations"], "expected_sha256": expected["expectations_sha256"],
        "current_root": ROOT,
    })
    original = proof.make_problem

    def changed(case):
        problem = original(case)
        problem.q.add_(1)
        return problem

    def forbidden(*args, **kwargs):
        raise AssertionError("changed model reached solver")

    import openopt.qp as qp
    monkeypatch.setattr(proof, "make_problem", changed)
    monkeypatch.setattr(qp, "prepare_qp", forbidden)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(proof.ProofError, match="original model differs"):
        proof.execute_case("gap_lp", output, "cpu", expected_model=manifest["cases"]["gap_lp"])


@pytest.mark.parametrize("module_name", [
    "openopt.qp", "openopt.operators", "openopt.scaling", "openopt.certificates",
])
@pytest.mark.parametrize("case", ["gap_lp", "infeasible", "unbounded"])
@pytest.mark.parametrize("entry", ["case", "validate", "seal", "replay", "production"])
def test_foreign_loaded_modules_reject_original_replay(produced, tmp_path, monkeypatch,
                                                      module_name, case, entry):
    """A foreign live module cannot be hidden by identical trusted disk bytes."""
    import importlib

    current = importlib.import_module(module_name)
    foreign_path = tmp_path / "foreign" / (module_name.rsplit(".", 1)[1] + ".py")
    foreign_path.parent.mkdir()
    foreign_path.write_bytes(Path(current.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location(module_name, foreign_path)
    foreign = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, foreign)
    spec.loader.exec_module(foreign)
    parent = sys.modules["openopt"]
    monkeypatch.setattr(parent, module_name.rsplit(".", 1)[1], foreign)
    assert foreign.__file__ == str(foreign_path)
    assert proof.sha(foreign_path.read_bytes()) == proof.sha(Path(current.__file__).read_bytes())
    result, bundle = produced
    with pytest.raises(proof.ProofError, match="loaded source"):
        if entry == "case":
            proof.replay_case(result.parent, proof.descriptor(
                result.parent, result.parent / (case + ".json")), "cpu")
        elif entry == "validate":
            proof.validate_result(result, proof.sha(result.read_bytes()), ROOT, allow_cpu_smoke=True)
        elif entry == "seal":
            proof.seal(result, proof.sha(result.read_bytes()), ROOT, allow_cpu_smoke=True)
        elif entry == "replay":
            proof.replay(bundle, proof.sha(bundle.read_bytes()), ROOT, allow_cpu_smoke=True)
        else:
            proof.run_cases(tmp_path / "production", [case], "cpu", cpu_smoke=True)
    assert sys.modules[module_name] is foreign
    if entry == "production":
        reports = list((tmp_path / "production").glob("cpu-smoke-*/result.json"))
        assert len(reports) == 1
        failed = json.loads(reports[0].read_bytes())
        assert failed["returncode"] == 1 and failed["numerical_verdict"] == "FAIL"
        assert failed["execution_verdict"] == "UNVERIFIED" and not failed["cases"]
        assert "loaded source" in (reports[0].parent / "failure.log").read_text()


@pytest.mark.parametrize("case", ["gap_lp", "infeasible", "unbounded"])
@pytest.mark.parametrize("mutation", [
    "checker", "checker_filename", "checker_helper", "config_default", "config_constructor",
    "captured_config",
])
def test_loaded_callable_substitution_rejected(produced, monkeypatch, case, mutation):
    import openopt.certificates as certificates
    import openopt.qp as qp
    import openopt.scaling as scaling

    def foreign(*args, **kwargs):
        raise AssertionError("substituted callable must never be called")

    if mutation in ("checker", "checker_filename", "checker_helper"):
        name = "_support" if mutation == "checker_helper" else "check_primal_infeasibility"
        if mutation == "checker_filename":
            foreign.__code__ = foreign.__code__.replace(co_filename=certificates.__file__)
        monkeypatch.setattr(certificates, name, foreign)
    elif mutation == "config_default":
        monkeypatch.setattr(qp.QPConfig, "rho", 42.)
    elif mutation == "config_constructor":
        monkeypatch.setattr(qp.QPConfig, "__init__", foreign)
    else:
        # Imported aliases must refer to the checked live definition, not an older class.
        foreign_config = type("QPConfig", (), {"__module__": "openopt.qp"})
        monkeypatch.setattr(scaling, "QPConfig", foreign_config)
    result, _ = produced
    with pytest.raises(proof.ProofError, match="loaded source"):
        proof.replay_case(result.parent, proof.descriptor(
            result.parent, result.parent / (case + ".json")), "cpu")


def test_foreign_module_metadata_cannot_hide_actual_code(produced, tmp_path, monkeypatch):
    import openopt.certificates as current
    import openopt.scaling as scaling

    path = tmp_path / "certificates.py"
    path.write_bytes(Path(current.__file__).read_bytes())
    spec = importlib.util.spec_from_file_location("openopt.certificates", path)
    foreign = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "openopt.certificates", foreign)
    spec.loader.exec_module(foreign)
    monkeypatch.setattr(sys.modules["openopt"], "certificates", foreign)
    foreign.__file__ = current.__file__
    foreign.__spec__.origin = current.__spec__.origin
    monkeypatch.setattr(scaling, "NumericalEvidence", foreign.NumericalEvidence)
    monkeypatch.setattr(scaling, "check_qp_candidate", foreign.check_qp_candidate)
    assert foreign.check_primal_infeasibility.__code__.co_filename == str(path)
    result, _ = produced
    with pytest.raises(proof.ProofError, match="loaded source callable"):
        proof.validate_result(result, proof.sha(result.read_bytes()), ROOT, allow_cpu_smoke=True)


@pytest.mark.parametrize("mode", [
    "fresh", "changed_file", "foreign_preload", "foreign_qp", "foreign_operators", "foreign_scaling",
    "checker_positional_defaults", "checker_keyword_defaults", "helper_keyword_defaults",
    "evidence_defaults", "evidence_constructor", "evidence_constructor_code",
    "sys_float_info",
])
def test_loaded_source_fresh_process(mode, tmp_path):
    """Execute this checked-in test file, not inline code or artifact snapshot code."""
    root = tmp_path / "trusted"
    copy_evidence_tree(ROOT / "src", root / "src")
    (root / "scripts").mkdir()
    copy_evidence_file(ROOT / proof.RUNNER, root / proof.RUNNER)
    if mode == "sys_float_info":
        path = root / "src/openopt/certificates.py"
        data = path.read_bytes()
        assert data.count(b"\nimport math\n") == 1
        path.write_bytes(data.replace(b"\nimport math\n", b"\nimport math\nimport sys\n") + (
            b"\n_FLOAT_EPS = sys.float_info.epsilon\n"
            b"\ndef _runtime_float_probe(*, floor=sys.float_info.min):\n"
            b"    return (_FLOAT_EPS, floor, sys.float_info.max)\n"
        ))
    output = tmp_path / "fresh-process.json"
    command = [
        sys.executable, "-B", str(Path(__file__).resolve()), "--loaded-source-child",
        mode, str(root), str(tmp_path / "evidence"), str(output),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    (tmp_path / "fresh-process.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(output.read_bytes())
    assert result["verdict"] == "PASS" and result["mode"] == mode
    if mode == "fresh":
        assert result["case_count"] == 3 and result["execution_verdict"] == "UNVERIFIED"
        identity = result["loaded_source_identity"]
        assert identity["root"] == str(root)
        assert all(Path(item["path"]).is_relative_to(root / "src/openopt")
                   for item in identity["modules"].values())
        assert {"openopt.certificates.check_primal_infeasibility",
                "openopt.certificates.check_unbounded_direction",
                "openopt.qp.QPConfig.validate"} <= set(identity["callables"])
    elif mode == "sys_float_info":
        assert result["negative_count"] == 4 and len(result["rejections"]) == 4
        assert result["unchanged_source_map"] is True
    else:
        assert "loaded source" in result["error"]


def test_loaded_source_cost_and_live_hashes(tmp_path):
    """Measured local cost only; no GPU/solver performance qualification."""
    proof._compiled_source.cache_clear()
    cold_start = time.perf_counter()
    identity = proof.loaded_source_check(ROOT)
    cold = time.perf_counter() - cold_start
    samples = []
    for _ in range(5):
        start = time.perf_counter()
        assert proof.loaded_source_check(ROOT) == identity
        samples.append(time.perf_counter() - start)
    assert proof._compiled_source.cache_info().hits > 0
    (tmp_path / "guard-cost.json").write_bytes(proof.encode({
        "verdict": "PASS", "root": str(ROOT), "cold_s": cold, "warm_s": samples,
        "warm_median_s": float(np.median(samples)),
        "cache": proof._compiled_source.cache_info()._asdict(),
        "modules": len(identity["modules"]), "callables": len(identity["callables"]),
        "qualification": "local CPU guard wall time; no GPU or optimizer speed claim",
    }))


def test_single_case_fixture_preserves_all_referenced_bytes(produced, tmp_path):
    result, _ = produced
    start = time.perf_counter()
    root, saved = case_copy(produced, tmp_path, "gap_lp")
    copied = proof.inventory(root)
    elapsed = time.perf_counter() - start
    original = proof.inventory(result.parent)
    assert len(copied) < len(original)
    assert all(value == original[name] for name, value in copied.items())
    proof.replay_case(root, proof.descriptor(root, root / "gap_lp.json"), "cpu")
    whole_start = time.perf_counter()
    whole = tmp_path / "whole-evidence"
    copy_evidence_tree(result.parent, whole)
    assert proof.inventory(whole) == original
    whole_elapsed = time.perf_counter() - whole_start
    (tmp_path / "fixture-cost.json").write_bytes(proof.encode({
        "verdict": "PASS", "original_files": len(original), "copied_files": len(copied),
        "copied_bytes": sum(item["bytes"] for item in copied.values()),
        "original_bytes": sum(item["bytes"] for item in original.values()),
        "copy_and_byte_check_s": elapsed, "whole_copy_and_byte_check_s": whole_elapsed,
        "copied_inventory": copied, "original_inventory": original, "case": saved["case"],
        "qualification": "single local fixture copy comparison only; not solver performance",
    }))


@pytest.mark.parametrize("mutation", ["missing", "extra", "same_length_hash"])
def test_single_case_fixture_rejects_file_set_mutations(produced, tmp_path, monkeypatch, mutation):
    """The narrowed fixture must prove exact membership AND bytes before returning."""
    result, _ = produced
    saved = json.loads((result.parent / "gap_lp.json").read_bytes())
    chosen = saved["model"]["q"]["components"]["values"]["path"]
    real_copy = copy_evidence_file
    calls = []

    def mutated_copy(source, destination):
        calls.append(Path(source).relative_to(result.parent).as_posix())
        if Path(source).relative_to(result.parent).as_posix() != chosen:
            return real_copy(source, destination)
        if mutation == "missing":
            return destination
        answer = real_copy(source, destination)
        if mutation == "extra":
            Path(destination).with_name("unreferenced.npy").write_bytes(b"extra bytes")
        else:
            original = Path(destination).read_bytes()
            changed = original[:-1] + bytes([original[-1] ^ 1])
            assert len(changed) == len(original) and proof.sha(changed) != proof.sha(original)
            Path(destination).write_bytes(changed)
        return answer

    monkeypatch.setitem(case_copy.__globals__, "copy_evidence_file", mutated_copy)
    with pytest.raises(AssertionError, match="case-copy"):
        case_copy(produced, tmp_path, "gap_lp")
    assert chosen in calls
    (tmp_path / "fixture-mutation.json").write_bytes(proof.encode({
        "verdict": "PASS", "mutation": mutation, "chosen": chosen, "copy_calls": calls,
        "qualification": "fixture membership/hash rejection, not numerical performance",
    }))


def test_fixture_copy_ab_preserves_nine_cases_and_original_bytes(produced, tmp_path):
    result, bundle = produced
    original = proof.inventory(result.parent)
    measurements = []
    for label, copier in (("shutil", shutil.copytree), ("direct", copy_evidence_tree)):
        destination = tmp_path / label
        start = time.perf_counter()
        copier(result.parent, destination)
        actual = proof.inventory(destination)
        elapsed = time.perf_counter() - start
        assert actual == original
        measurements.append({"method": label, "copy_and_byte_check_s": elapsed,
                             "files": len(actual), "bytes": sum(v["bytes"] for v in actual.values())})
    cases = {}
    for case in proof.CASES:
        root, saved = case_copy(produced, tmp_path / case, case)
        copied = proof.inventory(root)
        assert all(record == original[name] for name, record in copied.items())
        assert saved == json.loads((result.parent / (case + ".json")).read_bytes())
        cases[case] = {"files": copied, "status": saved["status"],
                       "passed": proof.replay_case(
                           root, proof.descriptor(root, root / (case + ".json")), "cpu")["passed"]}
        chosen = saved["model"]["q"]["components"]["values"]["path"]
        path = root / chosen
        data = path.read_bytes()
        path.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
        assert (result.parent / chosen).read_bytes() == data, "copies must not alias original files"
        with pytest.raises(proof.ProofError, match="byte hash mutation"):
            proof.bound(root, original[chosen])
    assert len(cases) == 9 and all(case["passed"] for case in cases.values())
    assert proof.inventory(result.parent) == original
    assert json.loads(bundle.read_bytes())["files"] == original
    (tmp_path / "nine-case-copy-ab.json").write_bytes(proof.encode({
        "verdict": "PASS", "methods": measurements, "cases": cases,
        "original_inventory": original,
        "qualification": "ordered local test-fixture copy comparison, not solver performance",
    }))


def pinned_reproducible_checker_compatibility(tmp_path, revision):
    """Fixed Git checker bytes, not a moving agent tree or an artifact code import."""
    command = ["git", "show", revision + ":src/openopt/certificates.py"]
    checkout = subprocess.run(command, cwd=ROOT, capture_output=True, check=False)
    assert checkout.returncode == 0, checkout.stderr.decode()
    root = tmp_path / "trusted"
    copy_evidence_tree(ROOT / "src", root / "src")
    (root / "scripts").mkdir()
    copy_evidence_file(ROOT / proof.RUNNER, root / proof.RUNNER)
    checker = root / "src/openopt/certificates.py"
    checker.write_bytes(checkout.stdout)
    output = tmp_path / "compatibility.json"
    completed = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--loaded-source-child",
         "reproducible_checker", str(root), str(tmp_path / "evidence"), str(output)],
        capture_output=True, text=True, check=False)
    (tmp_path / "compatibility.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    saved = json.loads(output.read_bytes())
    assert saved["verdict"] == "PASS" and saved["negative_count"] >= 3
    assert saved["checker_sha256"] == proof.sha(checkout.stdout)
    assert saved["original_candidate_checks"] == {
        name: True for name in ("primal_feasibility", "stationarity", "bound_sign",
                               "complementarity", "dual_objective_available", "duality_gap")}
    assert {"openopt.certificates._binary64_runtime",
            "openopt.certificates._reproducible_candidate",
            "openopt.certificates._jacobi"} <= set(saved["loaded_source_identity"]["callables"])


def test_function_local_imports_and_new_helper_defaults_rejected(tmp_path):
    root = tmp_path / "trusted"
    copy_evidence_tree(ROOT / "src", root / "src")
    (root / "scripts").mkdir()
    copy_evidence_file(ROOT / proof.RUNNER, root / proof.RUNNER)
    path = root / "src/openopt/certificates.py"
    path.write_bytes(path.read_bytes() + (
        b"\n_FIXTURE_ALGORITHM = 'fixture_v1'\n"
        b"\ndef _fixture_runtime_probe(*, algorithm='fixture_v1'):\n"
        b"    import ctypes\n"
        b"    import ctypes.util\n"
        b"    return (_FIXTURE_ALGORITHM, algorithm, ctypes.c_double(1.).value)\n"
        b"\ndef _fixture_new_helper(value):\n"
        b"    return value\n"
    ))
    output = tmp_path / "local-imports.json"
    completed = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--loaded-source-child",
         "local_imports", str(root), str(tmp_path / "evidence"), str(output)],
        capture_output=True, text=True, check=False)
    (tmp_path / "local-imports.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    saved = json.loads(output.read_bytes())
    assert saved["verdict"] == "PASS" and saved["negative_count"] == 4
    assert {"openopt.certificates._fixture_runtime_probe",
            "openopt.certificates._fixture_new_helper"} <= set(
                saved["loaded_source_identity"]["callables"])


def mutate_checker_runtime(monkeypatch, mutation):
    import openopt.certificates as certificates

    checker = certificates.check_primal_infeasibility
    evidence = certificates.NumericalEvidence
    if mutation == "checker_positional_defaults":
        monkeypatch.setattr(checker, "__defaults__", (None,))
    elif mutation == "checker_keyword_defaults":
        monkeypatch.setattr(checker, "__kwdefaults__", {"atol": .5, "rtol": .5})
    elif mutation == "helper_keyword_defaults":
        monkeypatch.setattr(certificates._array, "__kwdefaults__", {"bounds": True})
    elif mutation == "evidence_defaults":
        monkeypatch.setattr(evidence.__init__, "__defaults__", ("forged qualification",))
    elif mutation == "evidence_constructor":
        def replaced(self, *args, **kwargs):
            raise AssertionError("substituted evidence constructor must never run")
        monkeypatch.setattr(evidence, "__init__", replaced)
    elif mutation == "evidence_constructor_code":
        monkeypatch.setattr(evidence.__init__, "__code__",
                            evidence.__init__.__code__.replace(co_consts=(None, "forged")))
    elif mutation == "evidence_keyword_defaults":
        monkeypatch.setattr(evidence.__init__, "__kwdefaults__", {"kind": "forged"})
    elif mutation == "evidence_new":
        # Updating a live type's __new__ changes its CPython constructor slot even
        # after deleting the attribute. Mutate a detached type, not shared evidence.
        members = {key: value for key, value in vars(evidence).items()
                   if key not in ("__dict__", "__weakref__")}
        members["__new__"] = staticmethod(lambda *args: object())
        foreign = type(evidence.__name__, evidence.__bases__, members)
        monkeypatch.setattr(certificates, "NumericalEvidence", foreign)
    elif mutation == "problem_constructor":
        monkeypatch.setattr(certificates._Problem.__init__, "__defaults__", (1.,))
    elif mutation == "wrapper_defaults":
        monkeypatch.setattr(certificates._arithmetic, "__defaults__", (None,))
    else:
        assert mutation == "boolean_tolerance"
        monkeypatch.setattr(checker, "__kwdefaults__", {"atol": True, "rtol": 1e-8})


@pytest.mark.parametrize("mutation", ["numpy_alias", "math_alias", "parent_foreign"])
def test_checker_globals_and_parent_cache_rejected(tmp_path, monkeypatch, mutation):
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("openopt.certificates")
    proof.loaded_source_check(ROOT)
    before = proof.sha(Path(module.__file__).read_bytes())
    if mutation == "parent_foreign":
        path = tmp_path / "foreign-certificates.py"
        path.write_bytes(Path(module.__file__).read_bytes())
        spec = importlib.util.spec_from_file_location("openopt.certificates", path)
        foreign = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foreign)
        monkeypatch.setattr(sys.modules["openopt"], "certificates", foreign)
        assert sys.modules["openopt.certificates"] is module
        assert foreign is not module and proof.sha(path.read_bytes()) == before
    else:
        alias = "np" if mutation == "numpy_alias" else "math"
        original = getattr(module, alias)
        proxy = SimpleNamespace(**vars(original))
        monkeypatch.setattr(module, alias, proxy)
        assert getattr(module, alias) is not original
    assert proof.sha(Path(module.__file__).read_bytes()) == before
    with pytest.raises(proof.ProofError, match="loaded source"):
        proof.loaded_source_check(ROOT)


@pytest.mark.parametrize("mutation", [
    "checker_positional_defaults", "checker_keyword_defaults", "helper_keyword_defaults",
    "evidence_defaults", "evidence_constructor", "evidence_constructor_code",
    "evidence_keyword_defaults", "evidence_new", "problem_constructor", "wrapper_defaults",
    "boolean_tolerance",
])
def test_live_checker_defaults_and_evidence_constructor_rejected(monkeypatch, mutation):
    """Same code/file hash does not imply the same runtime defaults or constructor."""
    import openopt.certificates as certificates

    proof.loaded_source_check(ROOT)
    checker = certificates.check_primal_infeasibility
    code_before = checker.__code__
    hash_before = proof.sha(Path(certificates.__file__).read_bytes())
    with pytest.MonkeyPatch.context() as patch:
        mutate_checker_runtime(patch, mutation)
        assert checker.__code__ is code_before
        assert proof.sha(Path(certificates.__file__).read_bytes()) == hash_before
        with pytest.raises(proof.ProofError, match="loaded source"):
            proof.loaded_source_check(ROOT)
    proof.loaded_source_check(ROOT)
    assert certificates.NumericalEvidence(
        "fixture", True, {}, {}, {}, {}).qualification.startswith("approximate numerical")


@pytest.mark.parametrize("case", ["gap_lp", "infeasible", "unbounded"])
@pytest.mark.parametrize("entry", ["case", "validate", "seal", "replay", "production"])
@pytest.mark.parametrize("mutation", [
    "checker_positional_defaults", "checker_keyword_defaults", "helper_keyword_defaults",
    "evidence_defaults", "evidence_constructor", "evidence_constructor_code",
])
def test_runtime_defaults_reject_original_gap_and_rays(produced, tmp_path, monkeypatch,
                                                     case, entry, mutation):
    import openopt.certificates as certificates

    result, bundle = produced
    case_path = result.parent / (case + ".json")
    saved = json.loads(case_path.read_bytes())
    assert saved["numpy_original"]["passed"] and saved["independent_original"]["passed"]
    before = proof.sha(Path(certificates.__file__).read_bytes())
    mutate_checker_runtime(monkeypatch, mutation)
    if entry != "production":
        forbid_replay_execution(monkeypatch)
    with pytest.raises(proof.ProofError, match="loaded source"):
        if entry == "case":
            proof.replay_case(result.parent, proof.descriptor(result.parent, case_path), "cpu")
        elif entry == "validate":
            proof.validate_result(result, proof.sha(result.read_bytes()), ROOT, allow_cpu_smoke=True)
        elif entry == "seal":
            proof.seal(result, proof.sha(result.read_bytes()), ROOT, allow_cpu_smoke=True)
        elif entry == "replay":
            proof.replay(bundle, proof.sha(bundle.read_bytes()), ROOT, allow_cpu_smoke=True)
        else:
            proof.run_cases(tmp_path / "production", [case], "cpu", cpu_smoke=True)
    if entry == "production":
        reports = list((tmp_path / "production").glob("cpu-smoke-*/result.json"))
        assert len(reports) == 1
        failure = json.loads(reports[0].read_bytes())
        assert failure["returncode"] == 1 and failure["numerical_verdict"] == "FAIL"
        assert failure["execution_verdict"] == "UNVERIFIED" and not failure["cases"]
        assert "loaded source" in (reports[0].parent / "failure.log").read_text()
    assert proof.sha(Path(certificates.__file__).read_bytes()) == before


def loaded_source_child(mode, root, artifact, output):
    """Fresh-process fixture driver; source comes from caller-trusted test copies."""
    foreign_modules = {
        "foreign_preload": "openopt.certificates", "foreign_qp": "openopt.qp",
        "foreign_operators": "openopt.operators", "foreign_scaling": "openopt.scaling",
    }
    foreign = None
    if mode in foreign_modules:
        foreign = importlib.import_module(foreign_modules[mode])
        assert Path(foreign.__file__).is_relative_to(ROOT)
    spec = importlib.util.spec_from_file_location("fresh_trusted_producer", root / proof.RUNNER)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    if mode == "local_imports":
        expected = runner.source_map(root)
        identity = runner.loaded_source_check(root, expected)
        import openopt.certificates as certificates

        assert certificates._fixture_runtime_probe() == ("fixture_v1", "fixture_v1", 1.)
        errors = {}
        for mutation in ("internal_import_code", "new_helper_code", "new_keyword_default",
                         "new_global_constant"):
            with pytest.MonkeyPatch.context() as patch:
                if mutation in ("internal_import_code", "new_helper_code"):
                    helper = (certificates._fixture_runtime_probe
                              if mutation == "internal_import_code"
                              else certificates._fixture_new_helper)
                    patch.setattr(helper, "__code__", helper.__code__.replace(co_consts=(None, "forged")))
                elif mutation == "new_keyword_default":
                    patch.setattr(certificates._fixture_runtime_probe, "__kwdefaults__", {
                        "algorithm": "forged"})
                else:
                    patch.setattr(certificates, "_FIXTURE_ALGORITHM", "forged")
                assert runner.source_map(root) == expected
                try:
                    runner.loaded_source_check(root, expected)
                except runner.ProofError as error:
                    errors[mutation] = str(error)
                else:
                    raise AssertionError(f"new helper mutation accepted: {mutation}")
            assert runner.loaded_source_check(root, expected) == identity
            assert certificates._fixture_runtime_probe() == ("fixture_v1", "fixture_v1", 1.)
        value = {"verdict": "PASS", "negative_count": len(errors), "rejections": errors,
                 "loaded_source_identity": identity,
                 "qualification": "synthetic function-local-import fixture, not Nash integration"}
    elif mode == "reproducible_checker":
        expected = runner.source_map(root)
        identity = runner.loaded_source_check(root, expected)
        import openopt.certificates as certificates

        assert certificates.check_qp_candidate.__kwdefaults__["arithmetic"] == "numpy"
        certificates._binary64_runtime()
        problem = runner.make_problem("gap_lp")
        candidate = certificates.check_qp_candidate(
            problem, [0., 3., -1.], [-1., 2., -.5],
            arithmetic=certificates.REPRODUCIBLE_ARITHMETIC)
        assert candidate.passed, candidate.to_dict()
        errors = {}
        for mutation in ("internal_import_code", "new_helper_code", "new_keyword_default",
                         "new_global_constant"):
            with pytest.MonkeyPatch.context() as patch:
                if mutation in ("internal_import_code", "new_helper_code"):
                    helper = (certificates._binary64_runtime if mutation == "internal_import_code"
                              else certificates._jacobi)
                    patch.setattr(helper, "__code__", helper.__code__.replace(co_consts=(None, "forged")))
                elif mutation == "new_keyword_default":
                    patch.setattr(certificates.check_qp_candidate, "__kwdefaults__", {
                        **certificates.check_qp_candidate.__kwdefaults__, "arithmetic": "forged"})
                else:
                    patch.setattr(certificates, "REPRODUCIBLE_ARITHMETIC", "forged")
                assert runner.source_map(root) == expected
                try:
                    runner.loaded_source_check(root, expected)
                except runner.ProofError as error:
                    errors[mutation] = str(error)
                else:
                    raise AssertionError(f"new checker mutation accepted: {mutation}")
            assert runner.loaded_source_check(root, expected) == identity
        value = {"verdict": "PASS", "mode": mode, "negative_count": len(errors),
                 "rejections": errors, "loaded_source_identity": identity,
                 "original_candidate_checks": candidate.checks,
                 "checker_sha256": expected["src/openopt/certificates.py"],
                 "qualification": "pinned checker compatibility fixture; not integrated final source"}
    elif mode == "sys_float_info":
        expected = runner.source_map(root)
        identity = runner.loaded_source_check(root, expected)
        import openopt.certificates as certificates
        from types import SimpleNamespace

        probe = certificates._runtime_float_probe
        code = probe.__code__
        assert certificates.sys is sys and certificates._FLOAT_EPS == sys.float_info.epsilon
        assert probe() == (sys.float_info.epsilon, sys.float_info.min, sys.float_info.max)
        errors = {}
        for mutation in ("sys_alias", "global_epsilon", "float_default", "probe_code"):
            with pytest.MonkeyPatch.context() as patch:
                if mutation == "sys_alias":
                    patch.setattr(certificates, "sys", SimpleNamespace(float_info=sys.float_info))
                elif mutation == "global_epsilon":
                    patch.setattr(certificates, "_FLOAT_EPS", sys.float_info.epsilon * 2)
                elif mutation == "float_default":
                    patch.setattr(probe, "__kwdefaults__", {"floor": sys.float_info.min * 2})
                else:
                    patch.setattr(probe, "__code__", code.replace(co_consts=(None, "forged")))
                assert runner.source_map(root) == expected
                try:
                    runner.loaded_source_check(root, expected)
                except runner.ProofError as error:
                    errors[mutation] = str(error)
                else:
                    raise AssertionError(f"sys.float_info mutation accepted: {mutation}")
            assert runner.loaded_source_check(root, expected) == identity
            assert probe.__code__ is code
            assert probe() == (sys.float_info.epsilon, sys.float_info.min, sys.float_info.max)
        value = {"verdict": "PASS", "mode": mode, "negative_count": len(errors),
                 "rejections": errors, "unchanged_source_map": True,
                 "loaded_source_identity": identity,
                 "qualification": "synthetic trusted-source sys compatibility, not Nash integration"}
    elif mode == "fresh":
        assert not any(name == "openopt" or name.startswith("openopt.") for name in sys.modules)
        expected = runner.source_map(root)
        identity = runner.loaded_source_check(root, expected)
        result = runner.run_cases(artifact, ["gap_lp", "infeasible", "unbounded"],
                                  "cpu", cpu_smoke=True)
        before = runner.inventory(result.parent)
        with pytest.MonkeyPatch.context() as patch:
            def forbidden(*args, **kwargs):
                raise AssertionError("fresh replay attempted solve/factory/snapshot execution")
            import openopt.qp as qp
            import openopt.scaling as scaling
            import openopt.operators as operators
            for module, names in (
                (qp, ("prepare_qp", "solve_qp")),
                (scaling, ("solve_scaled_qp",)),
                (operators, ("prepare_operator_qp", "conjugate_gradient")),
                (runner, ("execute_case", "make_problem", "run_cases", "hardware", "freeze_sources")),
            ):
                for name in names:
                    patch.setattr(module, name, forbidden)
            digest = runner.sha(result.read_bytes())
            runner.validate_result(result, digest, root, allow_cpu_smoke=True)
            bundle = runner.seal(result, digest, root, allow_cpu_smoke=True)
            path = artifact / "bundle.json"
            runner.write(path, bundle)
            accepted = runner.replay(path, runner.sha(path.read_bytes()), root, allow_cpu_smoke=True)
        assert before == runner.inventory(result.parent)
        assert accepted["case_count"] == 3
        assert identity == accepted["loaded_source_identity"]
        value = {"verdict": "PASS", "mode": mode, **{
            key: accepted[key] for key in ("case_count", "execution_verdict", "loaded_source_identity")}}
    else:
        expected = runner.source_map(root)
        if mode == "changed_file":
            runner.loaded_source_check(root, expected)
            path = root / "src/openopt/qp.py"
            path.write_bytes(path.read_bytes() + b"\n# changed after trusted pin\n")
        elif mode not in foreign_modules:
            runner.loaded_source_check(root, expected)
            import openopt.certificates as certificates

            code = certificates.check_primal_infeasibility.__code__
            with pytest.MonkeyPatch.context() as patch:
                mutate_checker_runtime(patch, mode)
                assert certificates.check_primal_infeasibility.__code__ is code
                assert runner.source_map(root) == expected
                try:
                    runner.loaded_source_check(root, expected)
                except runner.ProofError as error:
                    value = {"verdict": "PASS", "mode": mode, "error": str(error),
                             "unchanged_checker_code": True, "unchanged_source_map": True}
                else:
                    raise AssertionError("unsafe runtime defaults/constructor was accepted")
            output.write_bytes(runner.encode(value))
            return 0
        try:
            runner.loaded_source_check(root, expected)
        except runner.ProofError as error:
            value = {"verdict": "PASS", "mode": mode, "error": str(error)}
        else:
            raise AssertionError("unsafe loaded source was accepted")
        if foreign is not None:
            assert sys.modules[foreign_modules[mode]] is foreign
            value["foreign_cache_retained"] = True
    output.write_bytes(runner.encode(value))
    return 0


def fixture_cost_main(argv):
    """Opt-in CPU diagnosis; explicit nodes, external logs and actual fixture closure."""
    parser = argparse.ArgumentParser(description=fixture_cost_main.__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--target", action="append", default=[])
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--sharder", type=Path)
    parser.add_argument("--history-junit", type=Path, action="append", default=[])
    parser.add_argument("--baseline-collection", type=Path)
    parser.add_argument("--group-collection", type=Path)
    parser.add_argument("--group-collection-sha256")
    parser.add_argument("--group-index", type=int)
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--checker-revision")
    args = parser.parse_args(argv)
    output = args.artifact_dir.resolve()
    assert output.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve())
    assert not output.is_relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from scripts.verify import source_snapshot, check_source_consistency

    before = source_snapshot(ROOT)
    selected_nodes = None
    if args.group_collection is not None:
        raw = args.group_collection.read_bytes()
        assert proof.sha(raw) == args.group_collection_sha256
        selection = json.loads(raw)
        assert selection["source"]["sha256"] == before["sha256"]
        selected_nodes = (
            [node for group in selection["groups"] if len(group["nodeids"]) == 1
             for node in group["nodeids"]] if args.group_index == -1 else
            max(selection["groups"], key=lambda group: len(group["nodeids"]))["nodeids"]
            if args.group_index == -2 else selection["groups"][args.group_index]["nodeids"])
        assert selected_nodes
    report = {"verdict": "RUNNING", "source": before, "targets": args.target,
              "qualification": "CPU test cost/coverage only, not solver or GPU performance"}
    plugins = []
    if args.collect:
        assert args.sharder is not None and args.sharder.is_file()
        spec = importlib.util.spec_from_file_location("scripts.verify_sharded", args.sharder)
        sharder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sharder)
        plugins.append(sharder)
        os.environ["OPENOPT_SHARD_COLLECTION"] = str(output / "collection.json")
        os.environ["OPENOPT_SHARD_GROUPING"] = sharder.FIXTURE_CONTRACT
        os.environ["OPENOPT_SHARD_FIXTURE_FILES"] = json.dumps(["tests/test_gpu_release_proof.py"])
        report["sharder"] = {"path": str(args.sharder.resolve()),
                             "sha256": proof.sha(args.sharder.read_bytes())}
    else:
        assert args.target or selected_nodes, "explicit targets required; no blind whole-file run"
    command = ["-q", "-ra", "-o", f"cache_dir={output / 'cache'}",
               "--basetemp", str(output / "pytest-tmp"),
               f"--junitxml={output / 'pytest.xml'}"]
    if args.collect:
        command += ["--collect-only", "tests/test_gpu_release_proof.py"]
    else:
        command += selected_nodes or [
            "tests/test_gpu_release_proof.py::" + target for target in args.target]
    profile = cProfile.Profile()
    started = time.perf_counter()
    try:
        if args.checker_revision:
            pinned_reproducible_checker_compatibility(output, args.checker_revision)
            report["checker_revision"] = args.checker_revision
        with (output / "pytest.log").open("x") as log:
            from contextlib import redirect_stdout, redirect_stderr
            with redirect_stdout(log), redirect_stderr(log):
                rc = (pytest.main(command, plugins=plugins) if args.no_profile else
                      profile.runcall(pytest.main, command, plugins=plugins))
        profile.dump_stats(output / "profile.pstats")
        report["wall_s"] = time.perf_counter() - started
        report["returncode"] = int(rc)
        report["costs"] = [
            {"path": path, "line": line, "function": function,
             "primitive_calls": counts[0], "calls": counts[1],
             "self_s": counts[2], "cumulative_s": counts[3]}
            for (path, line, function), counts in (
                {} if args.no_profile else pstats.Stats(profile).stats).items()
            if function in {"case_copy", "copyfile", "copy_evidence_file", "copy_evidence_tree",
                            "read", "descriptor", "inventory",
                            "loaded_source_check", "freeze_sources", "run_cases",
                            "replay", "replay_case", "validate_result"}
        ]
        assert rc == 0, f"pytest failed rc={rc}; see {output / 'pytest.log'}"
        if args.collect:
            collection = json.loads((output / "collection.json").read_bytes())
            report["groups"] = sharder.fixture_groups(
                collection["nodeids"], collection["fixture_audit"])
            historical = {}
            for path in args.history_junit:
                suites = ElementTree.parse(path).getroot()
                totals = Counter()
                for item in suites.iter("testcase"):
                    name = item.attrib["name"].split("[", 1)[0]
                    if item.attrib.get("classname") == "tests.test_gpu_release_proof":
                        totals[name] += float(item.attrib.get("time", 0))
                historical[str(path)] = dict(totals)
            report["historical_function_s"] = historical
            report["nodeids"] = collection["nodeids"]
            if args.baseline_collection is not None:
                baseline = json.loads(args.baseline_collection.read_bytes())["nodeids"]
                report["coverage_delta"] = {
                    "removed": sorted(set(baseline) - set(collection["nodeids"])),
                    "added": sorted(set(collection["nodeids"]) - set(baseline))}
                assert not report["coverage_delta"]["removed"]
                assert report["coverage_delta"]["added"] == sorted([
                    "tests/test_gpu_release_proof.py::"
                    "test_fixture_copy_ab_preserves_nine_cases_and_original_bytes",
                    "tests/test_gpu_release_proof.py::"
                    "test_function_local_imports_and_new_helper_defaults_rejected"])
                report["baseline_collection_sha256"] = proof.sha(
                    args.baseline_collection.read_bytes())
            report["node_count"] = len(collection["nodeids"])
            report["group_sizes"] = sorted(
                (len(group["nodeids"]) for group in report["groups"]), reverse=True)
        else:
            suites = ElementTree.parse(output / "pytest.xml").getroot()
            report["counts"] = {
                key: sum(int(s.attrib.get(key, 0)) for s in suites.findall("testsuite"))
                for key in ("tests", "failures", "errors", "skipped")}
            assert report["counts"]["tests"] > 0
            report["nodeids"] = selected_nodes
            report["under_180s"] = report["wall_s"] < 180
            report["under_300s"] = report["wall_s"] <= 300
        report["source_consistency"] = check_source_consistency(before, source_snapshot(ROOT))
        assert report["source_consistency"]["verdict"] == "PASS"
        report["verdict"] = "PASS"
    except Exception as error:
        report["verdict"] = "FAIL"
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "cost-report.json").write_bytes(proof.encode(report))
        print(f"FIXTURE_COST verdict={report['verdict']} report={output / 'cost-report.json'}")
        print(proof.encode({key: value for key, value in report.items()
                            if key in {"counts", "wall_s", "group_sizes", "node_count",
                                       "costs", "coverage_delta", "under_180s", "under_300s"}}).decode())
    return 0


if __name__ == "__main__" and sys.argv[1:2] == ["--fixture-cost"]:
    raise SystemExit(fixture_cost_main(sys.argv[2:]))
elif __name__ == "__main__":
    assert len(sys.argv) == 6 and sys.argv[1] == "--loaded-source-child"
    raise SystemExit(loaded_source_child(
        sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])))
