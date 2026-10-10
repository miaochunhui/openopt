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
    assert dispatch.worker_command(args, source, output, 0, "gap_spd") == [
        "python3", str(source / "scripts/gpu_release_proof.py"), "run",
        "--artifact-dir", str(output), "--device", "cuda:0", "--case", "gap_spd",
    ]
    with pytest.raises(ValueError, match="explicit release proof"):
        dispatch.worker_command(args, source, output, 0, None)


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
    value = {"verdict": "PASS", "returncode": 0, "device": "cuda:0",
             "requested_cases": ["gap_spd"], "cases": {"gap_spd": {}}}
    path.write_text(json.dumps(value))
    return {"stage": "release-proof"}, {"output": str(root), "proof_case": "gap_spd"}, path, value


def test_proof_collect_uses_immutable_nested_result(result):
    record, item, path, _ = result
    assert dispatch.collected_result(record, item) == path


@pytest.mark.parametrize("key,value", [
    ("verdict", "FAIL"), ("verdict", "CPU_SMOKE_PASS"), ("returncode", 1),
    ("device", "cpu"), ("requested_cases", ["unbounded"]), ("cases", {}),
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
