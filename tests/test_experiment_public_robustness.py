"""Cheap synthetic CPU controls, never public/industrial corpus coverage."""

from dataclasses import replace
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import experiment_public_robustness as gate
from scripts import public_qp_corpus as corpus
from scripts import qualify_public_qp as runner

LP = b"""NAME ROBUSTNESS_FIXTURE
ROWS
 N COST
 L CAP
COLUMNS
 X COST -1 CAP 1
RHS
 RHS1 CAP 2
ENDATA
"""


@pytest.fixture
def baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(corpus, "SHARED_TMP", tmp_path.resolve())
    monkeypatch.setattr(runner, "SHARED_TMP", tmp_path.resolve())
    inputs = tmp_path / "input"
    inputs.mkdir()
    (inputs / "cheap.mps").write_bytes(LP)
    manifest = tmp_path / "manifest.json"
    corpus.write_json(manifest, {"schema_version": 1, "cases": [{
        "case_id": "cheap", "path": "cheap.mps", "input_sha256": gate.digest(inputs / "cheap.mps"),
        "source_url": "https://example.invalid/synthetic-fixture.mps",
        "license": "synthetic unit fixture, not public coverage", "origin": "synthetic",
    }]})
    frozen = tmp_path / "frozen"
    corpus.freeze(manifest, inputs, frozen)

    def saved_bad_candidate(model, solver, values, core):
        # Deliberately negative, no solve in baseline fixture generation.
        return {"status": "Solved" if solver == "clarabel" else "solved", "iterations": 1,
                "settings": {"synthetic_saved_candidate": True}, "observed_device": "cpu",
                "vectors": {"x": np.asarray([1.0]), "y": np.asarray([0.0, 0.0]),
                            "reported_objective": np.asarray(-1.0)}}

    monkeypatch.setattr(runner, "run_solver", saved_bad_candidate)
    args = runner.parser().parse_args([
        "run", "--corpus", str(frozen), "--manifest-sha256", gate.digest(frozen / "corpus.json"),
        "--artifact-dir", str(tmp_path / "baseline"), "--source-root", str(gate.ROOT),
    ])
    result, code = runner.qualify(args)
    assert code == 1 and gate.read_json(result)["verdict"] == "UNQUALIFIED"
    assert gate.baseline_check(result, gate.digest(result))["rows"][0]["attempted"]
    return result


@pytest.fixture
def experiment(baseline, tmp_path, monkeypatch):
    def inline_worker(command, stdout, stderr, timeout, check):
        assert command[3] == "worker" and timeout <= 180 and check is False
        index = int(command[-1])
        code = gate.worker(Path(command[-3]), index)
        return SimpleNamespace(returncode=code)

    monkeypatch.setattr(gate.subprocess, "run", inline_worker)
    result, code = gate.run(baseline, gate.digest(baseline), tmp_path / "experiment", 180)
    assert code == 0
    return result


def test_fixed_arms_original_gates_and_one_attempt(baseline):
    report = gate.baseline_check(baseline, gate.digest(baseline))
    assert len(gate.ARMS) == 3
    assert gate.GATES == {"eps_abs": 1e-7, "eps_rel": 1e-7,
                          "direction_atol": 1e-8, "direction_rtol": 1e-8}
    assert all(arm["solve_eps"] < gate.GATES["eps_abs"] for arm in gate.ARMS)
    assert all(row["attempt"] == row["denominator"] == 1 for row in report["rows"])


def test_cheap_three_backend_real_calls_and_readonly_replay(experiment, monkeypatch, request):
    before = {key: gate.digest(gate.member(experiment.parent, key))
              for key in gate.read_json(experiment)["files"]}
    pin = gate.digest(experiment)
    report = gate.read_json(experiment)
    assert len(report["rows"]) == 3
    assert report["elapsed_seconds"] < 180
    assert all(row["input_model_before"] == row["input_model_after"] == row["model_sha256"]
               for row in report["rows"])
    assert report["baseline_rows"][0]["verdict"] == "UNQUALIFIED"

    def forbidden(*args, **kwargs):
        pytest.fail("read-only replay attempted solve")

    monkeypatch.setattr(gate, "run_arm", forbidden)
    replay = gate.replay(experiment, pin)
    assert replay["baseline_replayed"] == replay["new_numerically_replayed"] == 3
    assert replay["original_verdict"] == "UNQUALIFIED"
    assert before == {key: gate.digest(gate.member(experiment.parent, key)) for key in before}
    request.node.user_properties.append(("robustness_unit_json", str(experiment)))
    request.node.user_properties.append(("robustness_unit_sha256", pin))


@pytest.mark.parametrize("mutation", ["caller-pin", "vectors", "gate", "denominator",
                                     "baseline", "claim", "inventory", "restore"])
def test_negative_controls_reject_resealed_or_original_tamper(experiment, mutation):
    pin = gate.digest(experiment)
    report = gate.read_json(experiment)
    if mutation == "caller-pin":
        pin = "0" * 64
    elif mutation == "vectors":
        path = experiment.parent / report["rows"][0]["vectors_path"]
        path.write_bytes(path.read_bytes() + b"bad")
    elif mutation == "inventory":
        (experiment.parent / "unlisted.json").write_bytes(b"{}")
    elif mutation == "restore":
        path = experiment.parent / report["rows"][0]["transform_path"]
        extra = gate.read_extra(path)
        extra["restored_y"][0] += 0.1
        with path.open("wb") as stream:
            np.savez(stream, **extra)
        report["files"][str(path.relative_to(experiment.parent))] = {
            "bytes": path.stat().st_size, "sha256": gate.digest(path)}
        experiment.write_text(json.dumps(report))
        pin = gate.digest(experiment)
    else:
        if mutation == "gate":
            report["gates"]["eps_abs"] = 1e-3
        elif mutation == "denominator":
            report["planned_denominator"] -= 1
        elif mutation == "baseline":
            report["baseline_rows"][0]["verdict"] = "QUALIFIED"
        else:
            report["tiny_dual_clipping"] = True
        experiment.write_text(json.dumps(report))
        pin = gate.digest(experiment)
    with pytest.raises(ValueError):
        gate.replay(experiment, pin)


def test_empty_missing_or_wrong_baseline_refused(baseline):
    with pytest.raises(ValueError, match="caller"):
        gate.baseline_check(baseline, "0" * 64)
    report = gate.read_json(baseline)
    report["config"]["eps_abs"] = 1e-6
    baseline.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="original gate"):
        gate.baseline_check(baseline, gate.digest(baseline))


@pytest.mark.parametrize("budget", [0, -1, 181, float("nan")])
def test_invalid_cpu_budget_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="budget"):
        gate.run(tmp_path / "absent", "0" * 64, tmp_path / "out", budget)


def test_budget_exhaustion_has_full_failure_denominator(baseline, tmp_path, monkeypatch):
    def exhausted(*args, **kwargs):
        raise gate.subprocess.TimeoutExpired("synthetic budget control", 0.01)

    monkeypatch.setattr(gate.subprocess, "run", exhausted)
    result, code = gate.run(baseline, gate.digest(baseline), tmp_path / "exhausted", 180)
    report = gate.read_json(result)
    assert code == 2
    assert report["planned_denominator"] == 3
    assert all(row["verdict"] == "BUDGET_EXHAUSTED" for row in report["rows"])
    assert all(count["denominator"] == count["BUDGET_EXHAUSTED"] == 1
               for count in report["arm_counts"].values())
    assert gate.replay(result, gate.digest(result))["new_numerically_replayed"] == 0


def test_failed_worker_is_not_skipped_or_retried(baseline, tmp_path, monkeypatch):
    calls = []

    def failed(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(gate.subprocess, "run", failed)
    result, code = gate.run(baseline, gate.digest(baseline), tmp_path / "failed", 180)
    report = gate.read_json(result)
    assert code == 2 and len(calls) == 3
    assert all(row["verdict"] == "FAIL" and row["attempt"] == 1 for row in report["rows"])
    assert gate.replay(result, gate.digest(result))["planned_denominator"] == 3


def test_no_support_for_tiny_wrong_sign_or_looser_component_gate():
    core = runner.load_core(gate.ROOT / "src/openopt")
    problem = SimpleNamespace(P=np.zeros((1, 1)), q=np.zeros(1),
                              A=np.ones((1, 1)), l=np.zeros(1), u=np.full(1, np.inf))
    evidence = core.checker.check_qp_candidate(
        problem, np.zeros(1), np.asarray([1e-17]), atol=1e-7, rtol=1e-7).to_dict()
    assert not evidence["passed"] and not evidence["checks"]["bound_sign"]
    assert evidence["thresholds"]["bound_sign"] == 0
    assert evidence["metrics"]["dual_objective"] is None


def test_json_duplicate_and_artifact_symlink_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_bytes(b'{"a":1,"a":2}')
    with pytest.raises(ValueError, match="duplicate"):
        gate.read_json(path)
    (tmp_path / "link").symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        gate.member(tmp_path.resolve(), "link")


def test_accepted_refinement_cannot_upgrade_iteration_exhaustion(baseline, monkeypatch):
    core = runner.load_core(gate.ROOT / "src/openopt")
    package = core.qp.__package__
    core.scaling = runner.load_module(gate.ROOT / "src/openopt/scaling.py", package + ".scaling")
    core.refinement = runner.load_module(gate.ROOT / "src/openopt/refinement.py", package + ".refinement")
    original = core.qp.solve_qp

    def exhausted(*args, **kwargs):
        return replace(original(*args, **kwargs), status="max_iterations")

    monkeypatch.setattr(core.qp, "solve_qp", exhausted)
    model = runner.load_model(baseline.parent / "frozen", "cheap")
    raw, extra = gate.run_arm(model, gate.ARMS[0], core, runner)
    assessment = runner.assess(model, "openopt", raw, gate.GATES, core.checker)
    assert raw["refinement"]["status"] == "accepted"
    assert assessment["candidate"]["passed"]
    assert raw["status"] == "max_iterations" and assessment["verdict"] == "UNQUALIFIED"
    assert extra["restored_y"].shape == raw["vectors"]["y"].shape


def test_source_membership_tamper_cannot_be_resealed(experiment):
    report = gate.read_json(experiment)
    report["source_hashes"].pop("src/openopt/scaling.py")
    experiment.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="source inventory"):
        gate.replay(experiment, gate.digest(experiment))


def test_existing_output_is_never_overwritten(baseline, tmp_path):
    output = tmp_path / "owned"
    output.mkdir()
    marker = output / "other-owner.txt"
    marker.write_text("retain existing bytes")
    with pytest.raises(FileExistsError):
        gate.run(baseline, gate.digest(baseline), output)
    assert marker.read_text() == "retain existing bytes"


def replay_receipt(result: Path, pin: str, output: Path) -> dict:
    """Persist a read-only replay receipt outside the sealed experiment."""
    output = runner.external(output)
    if output.is_relative_to(result.resolve().parent):
        raise ValueError("receipt must not alter sealed experiment inventory")
    measured = gate.replay(result, pin)
    report = gate.read_json(result)
    receipt = {
        "schema_version": 1, "kind": "public-robustness-replay-receipt-v1",
        "result_path": str(result.resolve()), "result_sha256": pin,
        "baseline_sha256": report["baseline_sha256"], "corpus_sha256": report["corpus_sha256"],
        "public_acceptance_sha256": report["public_acceptance_sha256"],
        "source_hashes": report["source_hashes"], "replay": measured,
        "license_verified": False, "public_redistribution": False,
        "qualification_upgraded": False,
        "rows": [{
            "case_id": row["case_id"], "arm": row["arm"], "verdict": row["verdict"],
            "attempt": row["attempt"], "denominator": row["denominator"],
            "assessment": row.get("assessment"),
        } for row in report["rows"]],
    }
    corpus.write_json(output, receipt)
    if gate.read_json(output) != receipt:
        raise ValueError("replay receipt JSON roundtrip mismatch")
    return receipt


def test_external_replay_receipt_roundtrip_and_sealed_inventory(experiment, tmp_path):
    before = gate.inventory(experiment.parent)
    pin = gate.digest(experiment)
    output = tmp_path / "receipt.json"
    receipt = replay_receipt(experiment, pin, output)
    assert gate.read_json(output) == receipt
    assert before == gate.inventory(experiment.parent)
    assert not receipt["qualification_upgraded"] and receipt["replay"]["baseline_replayed"] == 3
    with pytest.raises(ValueError, match="sealed"):
        replay_receipt(experiment, pin, experiment.parent / "extra.json")
    with pytest.raises(FileExistsError):
        replay_receipt(experiment, pin, output)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description="Read-only pinned robustness replay with external receipt")
    cli.add_argument("--check-result", type=Path, required=True)
    cli.add_argument("--sha256", required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    receipt = replay_receipt(args.check_result, args.sha256, args.output)
    print("PUBLIC_ROBUSTNESS_RECEIPT " + json.dumps({
        "receipt": str(args.output), "receipt_sha256": gate.digest(args.output),
        "replay": receipt["replay"]}, sort_keys=True))
