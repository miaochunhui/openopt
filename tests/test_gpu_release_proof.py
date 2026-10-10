"""Real CPU producer/replay and adversarial evidence, never GPU qualification."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("gpu_proof", ROOT / "scripts/gpu_release_proof.py")
proof = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(proof)


@pytest.fixture(scope="module")
def produced(tmp_path_factory):
    parent = tmp_path_factory.mktemp("gpu-proof-cpu")
    result = proof.run_cases(parent, list(proof.CASES), "cpu", cpu_smoke=True)
    result_hash = proof.sha(result.read_bytes())
    bundle = proof.seal(result, result_hash, ROOT, allow_cpu_smoke=True)
    bundle_path = parent / "bundle.json"
    proof.write(bundle_path, bundle)
    return result, bundle_path


def case_copy(produced, tmp_path, name):
    result, _ = produced
    root = tmp_path / "evidence"
    shutil.copytree(result.parent, root)
    saved = json.loads((root / (name + ".json")).read_text())
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
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(proof.ProofError, match="exactly one visible CUDA"):
        proof.run_cases(tmp_path, ["gap_lp"], "cuda:0")
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
    root = tmp_path / "frozen"
    shutil.copytree(result.parent, root)
    bundle = json.loads(bundle_path.read_text())
    bundle["evidence_root"] = str(root)
    new_bundle = tmp_path / "bundle.json"
    proof.write(new_bundle, bundle)
    if mutation == "bytes":
        path = next((root / "arrays").glob("*.npy"))
        path.write_bytes(path.read_bytes() + b"tamper")
    elif mutation == "added":
        (root / "unexpected.json").write_text("{}")
    elif mutation == "removed":
        (root / "gap_lp.json").unlink()
    else:
        path = root / "source/src/openopt/qp.py"
        path.write_bytes(path.read_bytes() + b"\n# tamper\n")
    with pytest.raises(proof.ProofError, match="added, removed or changed"):
        proof.replay(new_bundle, proof.sha(new_bundle.read_bytes()), ROOT, allow_cpu_smoke=True)


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
