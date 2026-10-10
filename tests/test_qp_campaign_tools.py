"""Independent negative controls for the eight-seat and GPU admission gates."""

import argparse
import hashlib
import json
from pathlib import Path

import pytest

from scripts import qp_campaign
from scripts import cluster_benchmark
from scripts import accept_qp_campaign
from scripts import render_qp_campaign
from scripts.gpu_qp_campaign import diagonal_problem
from scripts.benchmark_general_qp import quality


def test_seat_write_scopes_are_disjoint_and_eight():
    assert len(qp_campaign.SEATS) == 8
    seen = set()
    for paths in qp_campaign.SEATS.values():
        assert paths
        assert not seen.intersection(paths)
        seen.update(paths)


def test_sparse_benchmark_witness_and_three_mutations():
    problem, x, y = diagonal_problem(16)
    assert quality(problem, x, y, x)["verdict"] == "PASS"
    for bad_x, bad_y in ((x + 10, y), (x, y + 10), (x * 0, y * 0)):
        assert quality(problem, bad_x, bad_y, x)["verdict"] == "FAIL"


def test_campaign_refuses_nonshared_evidence():
    with pytest.raises(ValueError, match="shared 3FS"):
        qp_campaign.setup("test", Path("/var/openopt-rejected-test"))


@pytest.mark.parametrize("name", ["../escape", "", "x/y"])
def test_campaign_name_rejects_path_escape(name, tmp_path):
    with pytest.raises(ValueError, match="campaign name"):
        qp_campaign.setup(name, tmp_path)


def test_scope_audit_detects_foreign_file(tmp_path, monkeypatch):
    manifest = tmp_path / "campaign.json"
    manifest.write_text(json.dumps({
        "baseline": "baseline",
        "seats": {name: {"path": str(tmp_path), "scope": paths}
                  for name, paths in qp_campaign.SEATS.items()},
    }))

    def fake_git(*args, **kwargs):
        if args[0] == "diff":
            return "AGENTS.md"
        if args[0] == "ls-files":
            return ""
        return "fake-head"

    monkeypatch.setattr(qp_campaign, "git", fake_git)
    with pytest.raises(RuntimeError, match="scope audit failed"):
        qp_campaign.audit(manifest)
    reports = list(tmp_path.glob("audit-*.json"))
    assert len(reports) == 1
    report = json.loads(reports[0].read_text())
    assert all(seat["outside_scope"] == ["AGENTS.md"] for seat in report["seats"])


def test_gpu_submit_refuses_four_active_or_requested_cards(tmp_path, monkeypatch):
    args = argparse.Namespace(
        artifact_root=tmp_path, namespace="test", sizes=[1],
    )
    # This test exercises the actual admission code without a cluster write.
    monkeypatch.setattr(cluster_benchmark.Path, "resolve",
                        lambda path: cluster_benchmark.Path("/3fs/research3fs/data/tmp/test-gate"))
    monkeypatch.setattr(cluster_benchmark.Path, "mkdir", lambda *args, **kwargs: None)
    monkeypatch.setattr(cluster_benchmark, "command", lambda args: json.dumps({
        "items": [{"status": {"phase": "Running"}, "spec": {"containers": [{
            "resources": {"limits": {"nvidia.com/gpu": 3}}
        }]}}]
    }))
    with pytest.raises(RuntimeError, match="three-card cap exceeded"):
        cluster_benchmark.submit(args)


def prerequisite(tmp_path, monkeypatch):
    core = tmp_path / "src/openopt/qp.py"
    core.parent.mkdir(parents=True)
    core.write_text("frozen-core")
    monkeypatch.setattr(cluster_benchmark, "ROOT", tmp_path)
    output = tmp_path / "case-0"
    output.mkdir()
    (output / "result.json").write_text(json.dumps({"verdict": "PASS", "mode": "correctness"}))
    (tmp_path / "run.json").write_text(json.dumps({
        "stage": "campaign", "jobs": [{"output": str(output)}],
        "source_hashes": {"src/openopt/qp.py": hashlib.sha256(core.read_bytes()).hexdigest()},
    }))
    return argparse.Namespace(
        stage="campaign", campaign_mode="batch", prerequisite=tmp_path,
        acceptance_bundle=tmp_path / "bundle.json", acceptance_sha256="caller-pin",
    )


def test_performance_admission_requires_independent_pin(tmp_path, monkeypatch):
    args = prerequisite(tmp_path, monkeypatch)
    args.acceptance_sha256 = None
    with pytest.raises(ValueError, match="pinned independent"):
        cluster_benchmark.validate_performance_prerequisite(args)


def test_performance_admission_rejects_uncovered_and_changed_source(tmp_path, monkeypatch):
    args = prerequisite(tmp_path, monkeypatch)
    monkeypatch.setattr(accept_qp_campaign, "accept", lambda *args: {"entries": []})
    with pytest.raises(ValueError, match="does not cover"):
        cluster_benchmark.validate_performance_prerequisite(args)
    monkeypatch.setattr(accept_qp_campaign, "accept", lambda *args: {"entries": [{
        "result_path": str(tmp_path / "case-0/result.json"), "tests": {"tests": 165},
    }]})
    cluster_benchmark.validate_performance_prerequisite(args)
    (tmp_path / "src/openopt/qp.py").write_text("changed-core")
    with pytest.raises(ValueError, match="core changed"):
        cluster_benchmark.validate_performance_prerequisite(args)


def test_performance_admission_pin_is_passed_to_independent_checker(tmp_path, monkeypatch):
    args = prerequisite(tmp_path, monkeypatch)
    calls = []

    def rejected(bundle, pin):
        calls.append((bundle, pin))
        raise ValueError("independent pin rejected")

    monkeypatch.setattr(accept_qp_campaign, "accept", rejected)
    with pytest.raises(ValueError, match="independent pin rejected"):
        cluster_benchmark.validate_performance_prerequisite(args)
    assert calls == [(args.acceptance_bundle, args.acceptance_sha256)]


def test_report_input_changed_bytes_are_rejected(tmp_path):
    path = tmp_path / "input.json"
    path.write_text('{"verdict":"PASS"}')
    descriptor = render_qp_campaign.source(path)
    assert render_qp_campaign.load(descriptor)["verdict"] == "PASS"
    path.write_text('{"verdict":"FAIL"}')
    with pytest.raises(ValueError, match="input changed"):
        render_qp_campaign.load(descriptor)


def test_report_chart_escapes_labels_and_rejects_bad_values():
    chart = render_qp_campaign.chart("measured", ["<script>"], [1.0], "ms")
    assert "<script>" not in chart and "&lt;script&gt;" in chart
    assert 'viewBox="0 0 900' in chart
    for values in ([], [float("nan")], [1.0, 2.0]):
        with pytest.raises(ValueError, match="chart input"):
            render_qp_campaign.chart("bad", ["one"], values, "ms")
