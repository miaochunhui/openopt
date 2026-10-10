"""Self-check the measurement ruler and reject fake/empty green artifacts."""

import json

import numpy as np
import pytest
import torch

from scripts.benchmark_general_qp import generate, quality, latency
from scripts.summarize_qp_benchmark import summary, nsight_summary
from scripts.prepare_report_bundle import freeze_evidence


def test_known_kkt_witness_and_three_quality_mutations():
    p, x, y = generate(12)
    assert quality(p, x, y, x)["verdict"] == "PASS"
    for changed_x, changed_y in (
        (x + 10, y), (x, y + 10), (x * 0, y * 0),
    ):
        assert quality(p, changed_x, changed_y, x)["verdict"] == "FAIL"


def test_generator_is_reproducible():
    first, x1, y1 = generate(12)
    second, x2, y2 = generate(12)
    for a, b in zip(vars(first).values(), vars(second).values(), strict=True):
        assert torch.equal(a, b)
    assert np.array_equal(x1, x2) and np.array_equal(y1, y2)


def test_latency_quantiles_come_from_actual_samples():
    result = latency([1.0, 2.0, 3.0])
    assert result["count"] == 3
    assert result["p50_s"] == 2
    assert result["p95_s"] == pytest.approx(2.9)


@pytest.mark.parametrize("verdict", ["FAIL", "RUNNING"])
def test_summary_rejects_unaccepted_result(tmp_path, verdict):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"verdict": verdict}))
    with pytest.raises(ValueError, match="failed experiment"):
        summary(path)


def test_nsight_ruler_rejects_missing_tables(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text("nothing collected\n")
    with pytest.raises(ValueError, match="incomplete Nsight"):
        nsight_summary(path)


def test_nsight_ruler_uses_nonempty_event_counts(tmp_path):
    path = tmp_path / "stats.csv"
    path.write_text(
        "Time (%),Total Time (ns),Instances,Name\n100,1000000,2,kernel\n\n"
        "Time (%),Total Time (ns),Num Calls,Name\n100,2000000,3,cudaLaunchKernel\n\n"
        "Time (%),Total Time (ns),Count,Operation\n100,500000,4,memcpy\n\n"
    )
    result = nsight_summary(path)
    assert result["kernel_total_s"] == 0.001
    assert result["kernel_instances"] == 2
    assert result["top_api"][0]["calls"] == 3


def test_live_snapshot_refresh_cannot_change_report_evidence(tmp_path):
    live = tmp_path / "live.json"
    original = b'{"verdict":"PASS","read_at":"first"}\n'
    live.write_bytes(original)
    frozen = freeze_evidence(live, tmp_path)
    assert frozen.read_bytes() == original
    live.write_bytes(b'{"verdict":"PASS","read_at":"second"}\n')
    assert frozen.read_bytes() == original
    refreshed = freeze_evidence(live, tmp_path)
    assert refreshed != frozen
    assert refreshed.read_bytes() == live.read_bytes()


def test_corrupted_content_addressed_evidence_is_rejected(tmp_path):
    live = tmp_path / "live.json"
    live.write_text('{"verdict":"PASS"}')
    frozen = freeze_evidence(live, tmp_path)
    frozen.write_text("intentionally corrupted")
    with pytest.raises(ValueError, match="evidence mismatch"):
        freeze_evidence(live, tmp_path)
