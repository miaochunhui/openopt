"""Real CPU producer/replay and adversarial evidence, never GPU qualification."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

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


def case_copy(produced, tmp_path, name):
    result, _ = produced
    root = tmp_path / "evidence"
    saved = json.loads((result.parent / (name + ".json")).read_bytes())
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
    files.update(path.relative_to(result.parent).as_posix()
                 for path in (result.parent / "source").rglob("*") if path.is_file())
    expected = {name: proof.descriptor(result.parent, result.parent / name) for name in files}
    for name in sorted(files):
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(result.parent / name, target)
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
])
def test_loaded_source_fresh_process(mode, tmp_path):
    """Execute this checked-in test file, not inline code or artifact snapshot code."""
    root = tmp_path / "trusted"
    shutil.copytree(ROOT / "src", root / "src")
    (root / "scripts").mkdir()
    shutil.copyfile(ROOT / proof.RUNNER, root / proof.RUNNER)
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
    shutil.copytree(result.parent, whole)
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
    real_copy = shutil.copyfile
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

    monkeypatch.setattr(shutil, "copyfile", mutated_copy)
    with pytest.raises(AssertionError, match="case-copy"):
        case_copy(produced, tmp_path, "gap_lp")
    assert chosen in calls
    (tmp_path / "fixture-mutation.json").write_bytes(proof.encode({
        "verdict": "PASS", "mutation": mutation, "chosen": chosen, "copy_calls": calls,
        "qualification": "fixture membership/hash rejection, not numerical performance",
    }))


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
    if mode == "fresh":
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


if __name__ == "__main__":
    assert len(sys.argv) == 6 and sys.argv[1] == "--loaded-source-child"
    raise SystemExit(loaded_source_child(
        sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]), Path(sys.argv[5])))
