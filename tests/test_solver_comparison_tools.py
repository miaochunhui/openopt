"""Independent negative controls for the same-host comparison runner."""

from dataclasses import dataclass
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts import benchmark_solver_comparison as bench


def test_frozen_witness_reproducible_and_hash_covers_bounds():
    first, x, y = bench.generate(12, 7)
    second, other_x, other_y = bench.generate(12, 7)
    assert bench.problem_hash(first) == bench.problem_hash(second)
    assert np.array_equal(x, other_x) and np.array_equal(y, other_y)
    assert bench.check_quality(first, x, y, x, 1e-7, 1e-7)["verdict"] == "PASS"
    first.u[0] += 1
    assert bench.problem_hash(first) != bench.problem_hash(second)


@pytest.mark.parametrize("mutation", ["primal", "dual", "nonfinite", "dimensions"])
def test_quality_negative_controls(mutation):
    problem, x, y = bench.generate(12, 7)
    expected = x.copy()
    if mutation == "primal":
        x = x + 10
    elif mutation == "dual":
        y = y + 10
    elif mutation == "nonfinite":
        x[0] = np.nan
    else:
        x = x[:-1]
    assert bench.check_quality(problem, x, y, expected, 1e-7, 1e-7)["verdict"] == "FAIL"


def test_full_psd_rejects_positive_diagonal_indefinite_matrix():
    problem, _, _ = bench.generate(3, 7)
    problem.P.copy_(torch.tensor([[1., 2., 0.], [2., 1., 0.], [0., 0., 1.]], dtype=torch.float64))
    assert bool((problem.P.diagonal() > 0).all())
    with pytest.raises(ValueError, match="full PSD"):
        bench.validate_problem(problem)


def test_adaptive_missing_field_fails_not_fallback():
    @dataclass
    class OldConfig:
        device: str = "cpu"

    with pytest.raises(ValueError, match="adaptive_rho is unavailable"):
        bench.openopt_config("cpu", "adaptive", 1e-7, 1e-7, 5000, OldConfig)


def test_adaptive_introspection_explicitly_disables_baseline():
    @dataclass
    class Config:
        device: str
        eps_abs: float
        eps_rel: float
        max_iterations: int
        adaptive_rho: bool = True

        def validate(self):
            assert self.device == "cpu"

    baseline = bench.openopt_config("cpu", "baseline", 1e-7, 1e-7, 5000, Config)
    adaptive = bench.openopt_config("cpu", "adaptive", 1e-7, 1e-7, 5000, Config)
    assert baseline.adaptive_rho is False
    assert adaptive.adaptive_rho is True


def test_clarabel_original_dual_mapping_with_equality_and_one_sided_bounds():
    problem, x, y = bench.generate(12, 7)
    matrix, b, indices, signs, count = bench.clarabel_form(problem)
    z = np.zeros(len(b))
    for index, (row, sign) in enumerate(zip(indices, signs, strict=True)):
        if index < count or y[row] * sign > 0:
            z[index] = y[row] * sign
    reconstructed = np.zeros_like(y)
    np.add.at(reconstructed, indices, z * signs)
    assert np.allclose(reconstructed, y)
    assert np.max(matrix @ x - b) < 1e-12
    problem.l[1] = -torch.inf
    problem.u[2] = torch.inf
    problem.l[4], problem.u[4] = -torch.inf, torch.inf
    matrix, b, indices, signs, _ = bench.clarabel_form(problem)
    assert np.isfinite(b).all() and 4 not in indices
    assert np.max(matrix @ x - b) < 1e-12


class WitnessRunner:
    def __init__(self, problem, x, y, fail=False):
        self.problem, self.x, self.y, self.fail = problem, x, y, fail
        self.config = {"eps_abs": 1e-7, "eps_rel": 1e-7}
        self.prepares, self.solves = 0, 0

    def prepare(self):
        self.prepares += 1
        bench.validate_problem(self.problem)
        return self

    def solve(self, workspace):
        self.solves += 1
        if self.fail:
            raise RuntimeError("planted first cause")
        return self.x, self.y, "solved", True, 1


def test_alternating_order_cold_setup_prepared_reuse_and_identical_control(tmp_path):
    problem, x, y = bench.generate(6, 7)
    first, control = WitnessRunner(problem, x, y), WitnessRunner(problem, x, y)
    rows = [bench.arm("a", "openopt", "cpu", first),
            bench.arm("b", "openopt", "cpu", control, control_of="a")]
    report = {"failures": [], "orders": [], "arms": rows, "problem_hash": bench.problem_hash(problem)}
    args = SimpleNamespace(repeats=3, eps_abs=1e-7, eps_rel=1e-7, artifact_dir=tmp_path)
    for mode in ("cold", "prepared"):
        bench.execute_mode(rows, {"a": first, "b": control}, mode, x, args, report)
    assert [r["arms"] for r in report["orders"][:3]] == [["a", "b"], ["b", "a"], ["a", "b"]]
    assert first.prepares == control.prepares == 4
    assert first.solves == control.solves == 6
    controls = bench.control_summary(rows, 3)
    assert controls[0]["identical_config_control"]
    assert controls[0]["modes"]["cold"]["empirical_noise_floor_s"] >= 0
    assert rows[0]["modes"]["prepared"]["setup_once_s"] > 0
    assert all(s["setup_s"] is None for s in rows[0]["modes"]["prepared"]["samples"])


def test_solver_failure_stops_arm_preserves_first_cause_but_other_arm_runs(tmp_path):
    problem, x, y = bench.generate(6, 7)
    bad, good = WitnessRunner(problem, x, y, True), WitnessRunner(problem, x, y)
    rows = [bench.arm("bad", "fake", "cpu", bad), bench.arm("good", "fake", "cpu", good)]
    report = {"failures": [], "orders": [], "arms": rows, "problem_hash": bench.problem_hash(problem)}
    args = SimpleNamespace(repeats=3, eps_abs=1e-7, eps_rel=1e-7, artifact_dir=tmp_path)
    bench.execute_mode(rows, {"bad": bad, "good": good}, "cold", x, args, report)
    assert bad.solves == 1 and good.solves == 3
    assert rows[0]["modes"]["cold"]["latency"] is None
    assert rows[1]["modes"]["cold"]["status"] == "PASS"
    assert report["failures"][0]["first_cause"] == "planted first cause"
    assert report["failures"][0]["coordinate"]["repeat"] == 0


def test_missing_references_are_unverified_not_required(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "available", lambda name: name in ("numpy", "torch"))
    output = tmp_path / "comparison"
    assert bench.main(["--artifact-dir", str(output), "--size", "6", "--repeats", "3"]) == 0
    report = json.loads((output / "result.json").read_text())
    refs = [r for r in report["arms"] if r["solver"] in ("osqp", "clarabel")]
    assert len(refs) == 2 and all(r["status"] == "UNVERIFIED" for r in refs)
    assert report["reference_coverage"]["status"] == "UNVERIFIED"
    assert report["comparison"]["ranking"] == "UNVERIFIED"
    assert all(v["status"] == "UNVERIFIED" for v in report["commercial"].values())
    assert (output / "problem.npz").stat().st_size > 0
    assert bench.check_result(output / "result.json")["checked_samples"] == 12


def test_explicit_cuda_unavailable_fails_and_never_cpu_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bench, "available", lambda name: name in ("numpy", "torch"))
    output = tmp_path / "failed-cuda"
    assert bench.main(["--artifact-dir", str(output), "--size", "6",
                       "--devices", "cpu", "cuda:0", "--repeats", "3"]) == 1
    report = json.loads((output / "result.json").read_text())
    cuda = [r for r in report["arms"] if r["device"] == "cuda:0"]
    assert len(cuda) == 1 and cuda[0]["status"] == "FAIL"
    assert "CUDA" in report["failures"][0]["first_cause"]
    failure = json.loads((output / "failure.json").read_text())
    assert failure["first_cause"]["coordinate"]["arm"].startswith("openopt:cuda:0")


@pytest.mark.parametrize("option", [
    ["--repeats", "0"], ["--eps-abs", "nan"], ["--devices", "cpu", "cpu"],
])
def test_cli_rejects_empty_or_invalid_experiments(tmp_path, option):
    with pytest.raises(SystemExit) as error:
        bench.main(["--artifact-dir", str(tmp_path / "invalid"), *option])
    assert error.value.code == 2
    assert not (tmp_path / "invalid").exists()


def test_nonempty_artifacts_are_never_overwritten(tmp_path):
    output = tmp_path / "old"
    output.mkdir()
    previous = output / "result.json"
    previous.write_text('{"first":"failure"}')
    with pytest.raises(SystemExit):
        bench.main(["--artifact-dir", str(output)])
    assert previous.read_text() == '{"first":"failure"}'


def test_real_cpu_references_matched_eps_and_original_quality(tmp_path):
    if not bench.available("osqp") or not bench.available("clarabel") or not bench.available("scipy"):
        pytest.skip("reference SDK unavailable; reference qualification UNVERIFIED")
    output = tmp_path / "cpu-references"
    assert bench.main(["--artifact-dir", str(output), "--size", "12", "--repeats", "3"]) == 0
    report = json.loads((output / "result.json").read_text())
    assert report["reference_coverage"]["status"] == "PASS"
    assert bench.check_result(output / "result.json")["checked_samples"] == 36
    for row in report["arms"]:
        config = row["config"]
        if row["solver"] == "clarabel":
            assert config["tol_gap_abs"] == config["tol_gap_rel"] == config["tol_feas"] == 1e-7
        else:
            assert config["eps_abs"] == config["eps_rel"] == 1e-7
        assert row["status"] == "PASS"
        assert all(m["status"] == "PASS" for m in row["modes"].values())
        prepared = row["modes"]["prepared"]["samples"]
        assert len({s["iterations"] for s in prepared}) == 1
        assert all(np.allclose(s["host_solution"]["x"], prepared[0]["host_solution"]["x"],
                               rtol=0, atol=1e-12) for s in prepared)


@pytest.mark.parametrize("mutation", [
    "samples", "control", "problem", "orders", "solution", "quality", "eps", "placement",
])
def test_disk_gate_rejects_tampered_acceptance(tmp_path, monkeypatch, mutation):
    monkeypatch.setattr(bench, "available", lambda name: name in ("numpy", "torch"))
    output = tmp_path / mutation
    assert bench.main(["--artifact-dir", str(output), "--size", "6", "--repeats", "3"]) == 0
    path = output / "result.json"
    assert bench.check_result(path)["verdict"] == "PASS"
    report = json.loads(path.read_text())
    if mutation == "samples":
        report["arms"][0]["modes"]["cold"]["samples"].pop()
    elif mutation == "control":
        report["controls"][0]["modes"]["cold"]["empirical_noise_floor_s"] = -1
    elif mutation == "problem":
        report["problem_hash"] = "corrupted"
    elif mutation == "orders":
        report["orders"][1]["arms"].reverse()
    elif mutation == "solution":
        report["arms"][0]["modes"]["cold"]["samples"][0]["host_solution"]["x"][0] += 1
    elif mutation == "quality":
        report["arms"][0]["modes"]["cold"]["samples"][0]["quality"]["objective"] += 1
    elif mutation == "eps":
        report["arms"][0]["config"]["eps_abs"] *= 100
    else:
        report["arms"][0]["modes"]["cold"]["samples"][0]["observed_tensor_device"] = "cuda:0"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError):
        bench.check_result(path)
