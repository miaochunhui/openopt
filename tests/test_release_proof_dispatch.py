"""Exercise real one-card proof dispatch without a cluster write."""

import argparse
import json

import pytest

from scripts import cluster_benchmark as dispatch


@pytest.fixture
def args(monkeypatch):
    monkeypatch.setattr(dispatch, "proof_cases", lambda: ("gap_spd", "jacobi_coo", "unbounded"))
    return argparse.Namespace(
        stage="release-proof", proof_case=["gap_spd"], sizes=[64],
        target=[], expression="", repeats=5, campaign_mode="correctness",
    )


def test_proof_units_are_distinct_single_case_jobs(args):
    args.proof_case = ["gap_spd", "jacobi_coo", "unbounded"]
    assert dispatch.submission_units(args) == [
        (0, "gap_spd"), (0, "jacobi_coo"), (0, "unbounded"),
    ]


@pytest.mark.parametrize("cases", [[], ["gap_spd", "gap_spd"], ["unknown"], ["gap_spd"] * 4])
def test_proof_invalid_scope_rejected(args, cases):
    args.proof_case = cases
    with pytest.raises(ValueError):
        dispatch.submission_units(args)


@pytest.mark.parametrize("key,value", [("target", ["tests/test_x.py"]), ("expression", "cuda")])
def test_proof_cannot_mix_pytest_selection(args, key, value):
    setattr(args, key, value)
    with pytest.raises(ValueError, match="cannot mix"):
        dispatch.submission_units(args)


def test_proof_command_is_explicit_cuda_one_case(args, tmp_path):
    source, output = tmp_path / "source", tmp_path / "case-0"
    expectations = {"path": str(tmp_path / "proof-expectations.json"), "sha256": "a" * 64}
    assert dispatch.worker_command(args, source, output, 0, "gap_spd", expectations) == [
        "python3", str(source / "scripts/gpu_release_proof.py"), "run",
        "--artifact-dir", str(output), "--device", "cuda:0", "--case", "gap_spd",
        "--expectations", expectations["path"], "--expectations-sha256", expectations["sha256"],
    ]
    with pytest.raises(ValueError, match="explicit release proof"):
        dispatch.worker_command(args, source, output, 0, None)
    with pytest.raises(ValueError, match="pre-dispatch expectations"):
        dispatch.worker_command(args, source, output, 0, "gap_spd")


def test_regression_command_and_one_card_contract_unchanged(args, tmp_path):
    args.stage, args.proof_case = "release-regression", []
    args.target, args.expression = ["tests/test_native_conic.py"], "cuda"
    assert dispatch.submission_units(args) == [(64, None)]
    assert dispatch.worker_command(args, tmp_path, tmp_path, 64, None)[-4:] == [
        "--target", "tests/test_native_conic.py", "--expression", "cuda",
    ]
    args.sizes = [64, 64]
    with pytest.raises(ValueError, match="requires one card"):
        dispatch.submission_units(args)


def test_nonproof_cannot_hide_proof_scope(args):
    args.stage = "smoke"
    with pytest.raises(ValueError, match="require the release-proof"):
        dispatch.submission_units(args)


@pytest.fixture
def result(tmp_path):
    root = tmp_path / "case-0"
    path = root / "cuda-proof-fixture/result.json"
    path.parent.mkdir(parents=True)
    value = {"verdict": "SEMANTIC_NUMERICAL_PASS", "numerical_verdict": "PASS",
             "execution_verdict": "UNVERIFIED", "returncode": 0, "device": "cuda:0",
             "requested_cases": ["gap_spd"], "cases": {"gap_spd": {}},
             "finished_at": "2026-10-10T09:00:00+00:00",
             "expectations": {"path": str(tmp_path / "expected.json"), "sha256": "a" * 64, "bytes": 12}}
    path.write_text(json.dumps(value))
    return {"stage": "release-proof", "expectations": value["expectations"]}, {
        "output": str(root), "proof_case": "gap_spd",
    }, path, value


def test_proof_collect_uses_immutable_nested_result(result):
    record, item, path, _ = result
    assert dispatch.collected_result(record, item) == path


@pytest.mark.parametrize("key,value", [
    ("verdict", "FAIL"), ("verdict", "CPU_SMOKE_PASS"), ("returncode", 1),
    ("device", "cpu"), ("requested_cases", ["unbounded"]), ("cases", {}),
    ("expectations", None),
    ("verdict", "PASS"), ("numerical_verdict", "FAIL"), ("execution_verdict", "PASS"),
])
def test_collect_does_not_promote_wrong_scope_or_failure(result, key, value):
    record, item, path, report = result
    report[key] = value
    path.write_text(json.dumps(report))
    assert dispatch.collected_result(record, item) is None


def test_collect_rejects_duplicate_and_missing_results(result):
    record, item, path, _ = result
    duplicate = path.parent.parent / "cuda-proof-other/result.json"
    duplicate.parent.mkdir()
    duplicate.write_bytes(path.read_bytes())
    assert dispatch.collected_result(record, item) is None


def test_collect_does_not_accept_only_a_coordinator_mirror(result):
    record, item, path, _ = result
    path.rename(path.parent.parent / "result.json")
    assert dispatch.collected_result(record, item) is None


def test_completion_pin_must_come_from_exact_single_pod_log(result):
    _, _, path, report = result
    pin = "a" * 64
    line = (f"OPENOPT_GPU_PROOF verdict=SEMANTIC_NUMERICAL_PASS at={report['finished_at']} "
            f"result={path} result_sha256={pin}")
    assert dispatch.proof_completion_matches(line + "\n", path, pin)
    for log in ("", line.replace(pin, "b" * 64), line + "\n" + line,
                line.replace("PASS", "FAIL"), line.replace(str(path), "/other/result.json"),
                line + "\ntrailing unverified completion", line.replace("SEMANTIC_NUMERICAL_PASS", "PASS")):
        assert not dispatch.proof_completion_matches(log, path, pin)


def test_predispatch_manifest_freezes_original_models_without_solving(tmp_path, monkeypatch):
    proof = dispatch.proof_module()
    source = dispatch.ROOT
    monkeypatch.setattr(proof, "execute_case", lambda *args: pytest.fail("pre-dispatch solved"))
    run = tmp_path / "run"
    run.mkdir()
    item = dispatch.freeze_proof_expectations(run, source, ["gap_lp", "jacobi_coo", "jacobi_csr"])
    import hashlib

    path = run / "proof-expectations.json"
    assert item == {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "bytes": path.stat().st_size}
    saved = proof.parse(path.read_bytes(), path)
    assert set(saved) == {"schema_version", "source_sha256", "cases"}
    assert set(saved["cases"]) == {"gap_lp", "jacobi_coo", "jacobi_csr"}
    assert saved["source_sha256"] == proof.source_map(source)
    for name, model in saved["cases"].items():
        assert set(model) == {"model", "model_sha256"}
        assert model["model_sha256"] == proof.model_hash(model["model"])
        for field, tensor in model["model"].items():
            array = proof.load_tensor(run, tensor, name + "/" + field)
            assert tensor["device"] == "cpu"
            assert array.size > 0
    assert saved["cases"]["jacobi_coo"]["model"]["P"]["layout"] == "torch.sparse_coo"
    assert saved["cases"]["jacobi_csr"]["model"]["P"]["layout"] == "torch.sparse_csr"


def test_snapshot_includes_only_explicit_refinement_document(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "snapshot"
    names = ("tests/test_qp_refinement.py", "docs/research/refinement.md", "docs/private.md")
    for name in names:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n")
    monkeypatch.setattr(dispatch, "ROOT", source)
    monkeypatch.setattr(dispatch, "command", lambda argv: (
        "\0".join(names) + "\0" if "--others" not in argv else ""
    ))
    hashes = dispatch.snapshot(output)
    assert set(hashes) == set(names[:2])
    assert (output / names[1]).read_bytes() == (source / names[1]).read_bytes()
    assert not (output / "docs/private.md").exists()


def test_snapshot_rejects_refinement_missing_required_document(tmp_path, monkeypatch):
    path = tmp_path / "tests/test_qp_refinement.py"
    path.parent.mkdir()
    path.write_text("fixture source\n")
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    monkeypatch.setattr(dispatch, "command", lambda argv: (
        "tests/test_qp_refinement.py\0" if "--others" not in argv else ""
    ))
    with pytest.raises(ValueError, match="explicit research-document"):
        dispatch.snapshot(tmp_path / "snapshot")
