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
