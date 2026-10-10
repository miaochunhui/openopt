"""CPU-only adversarial checks against the real producer schemas.

Schema fixtures use original-coordinate planted KKT solutions, never claim
measured performance. The separate documented CLI audit consumes real results.
Every mutation is resealed where appropriate so hashes cannot hide weak gates.
"""

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import accept_qp_campaign as gate


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False) + "\n")


def sources(root, gpu=False):
    files = ["scripts/gpu_qp_campaign.py", "src/openopt/qp.py",
             "scripts/benchmark_general_qp.py"] if gpu else [
        "scripts/benchmark_solver_comparison.py", "scripts/benchmark_general_qp.py",
        "src/openopt/qp.py",
    ]
    for name in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture source is never executed\n")
    empty = root / "scripts/__init__.py"
    empty.write_text("")
    return gate.source_files(root)


def stats(values):
    return {"count": len(values), "samples_s": values,
            "p50_s": float(np.percentile(values, 50)), "p95_s": float(np.percentile(values, 95))}


def comparison_fixture(root):
    source = root / "source"
    hashes = sources(source)
    out = root / "case"
    out.mkdir()
    p = dict(P=np.eye(3), q=np.array([-0.1, -0.2, -0.3]),
             A=np.eye(3), l=np.full(3, -1.0), u=np.ones(3),
             expected=np.array([0.1, 0.2, 0.3]), witness_y=np.zeros(3))
    np.savez(out / "problem.npz", **p)
    eps = 1e-7
    quality = {"verdict": "PASS", **gate.recompute(p, {"x": p["expected"], "y": p["witness_y"]},
                                                eps, eps, "fixture")}
    rows = []
    for solver, duration in (("openopt", 0.03), ("osqp", 0.01)):
        for is_control in (False, True):
            identifier = solver + (":control" if is_control else "")
            config = dict(eps_abs=eps, eps_rel=eps)
            if solver == "openopt":
                config.update(dtype="torch.float64", device="cpu")
            row = dict(id=identifier, solver=solver, device="cpu", status="PASS",
                       config=config, config_hash=gate.sha(json.dumps(config, sort_keys=True).encode()),
                       control_of=solver if is_control else None, modes={})
            for mode in ("cold", "prepared"):
                times = [duration + 1e-5 * (i + int(is_control)) for i in range(3)]
                samples = []
                for i, wall in enumerate(times):
                    position = len(rows) if i % 2 == 0 else 3 - len(rows)
                    sample = dict(arm=identifier, mode=mode, repeat=i, order_position=position,
                                  wall_s=wall, setup_s=0.001 if mode == "cold" else None,
                                  solve_host_result_s=wall - 0.001 if mode == "cold" else wall,
                                  verdict="PASS", solver_status="solved", observed_tensor_device="cpu",
                                  host_solution={"x": p["expected"].tolist(), "y": p["witness_y"].tolist()},
                                  quality=copy.deepcopy(quality))
                    samples.append(sample)
                row["modes"][mode] = dict(status="PASS", samples=samples, latency=stats(times),
                                         setup_once_s=0.001 if mode == "prepared" else None)
            rows.append(row)
    report = dict(verdict="PASS", exit_code=0, wall_s=1.0, source_hashes=hashes,
                  configuration=dict(repeats=3, cpu_threads=1, eps_abs=eps, eps_rel=eps),
                  hardware=dict(host="fixture-host", torch_threads=1, cpu_max="max 100000"),
                  thread_pools=[dict(status="MEASURED", threads=1)],
                  problem_hash=gate.problem_digest(p), arms=rows, failures=[],
                  orders=[dict(mode=mode, repeat=i, arms=[r["id"] for r in rows][::1 if i % 2 == 0 else -1])
                          for mode in ("cold", "prepared") for i in range(3)])
    report["controls"] = []
    for original, twin in ((rows[0], rows[1]), (rows[2], rows[3])):
        control = dict(arm=original["id"], control=twin["id"], identical_config_control=True, modes={})
        for mode in ("cold", "prepared"):
            deltas = [abs(a - b) for a, b in zip(
                original["modes"][mode]["latency"]["samples_s"],
                twin["modes"][mode]["latency"]["samples_s"], strict=True)]
            control["modes"][mode] = dict(status="PASS", paired_abs_delta_s=deltas,
                                          empirical_noise_floor_s=float(np.percentile(deltas, 95)))
        report["controls"].append(control)
    dump(out / "result.json", report)
    dump(out / "telemetry.json", [{"process_threads": 1}])
    (out / "execution.log").write_text("OPENOPT_SOLVER_COMPARISON verdict=PASS\n")
    return out / "result.json", source, report


def gpu_fixture(root, correctness=False):
    source = root / "source"
    hashes = sources(source, gpu=True)
    dump(source / "source-hashes.json", hashes)
    out = root / "case-0"
    out.mkdir()
    q = {**{key: 0.0 for key in gate.METRICS}, "verdict": "PASS", "solver_status": "solved"}
    result = {}
    if correctness:
        result.update(dtype="float64", eps_abs=1e-7, eps_rel=1e-7)
        p = dict(P=np.eye(3), q=np.array([-0.1, -0.2, -0.3]),
                 A=np.eye(3), l=np.full(3, -1.0), u=np.ones(3),
                 expected=np.array([0.1, 0.2, 0.3]), witness_y=np.zeros(3))
        for name in ("problem-dense.npz", "problem-sparse.npz"):
            np.savez(out / name, **p)
        actual = gate.recompute(p, {"x": p["expected"], "y": p["witness_y"]}, 1e-7, 1e-7, "fixture")
        q.update({key: actual[key] for key in (*gate.METRICS, "objective", "solution_error")},
                 host_x=p["expected"].tolist(), host_y=p["witness_y"].tolist())
        result["checks"] = {name: copy.deepcopy(q) for name in (
            "dense_adaptive_False", "dense_adaptive_True", "batch_0", "batch_1", "batch_2",
            "sparse_coo", "sparse_csr")}
        result["tests"] = dict(tests=7, errors=0, failures=0, skipped=0)
        (out / "pytest.xml").write_text(
            '<testsuites><testsuite tests="7" errors="0" failures="0" skipped="0"/></testsuites>')
        (out / "pytest.log").write_text("7 passed in 1.00s\n")
    else:
        p = dict(P=np.eye(3), q=np.array([-0.1, -0.2, -0.3]),
                 A=np.eye(3), l=np.full(3, -1.0), u=np.ones(3),
                 expected=np.array([0.1, 0.2, 0.3]), witness_y=np.zeros(3))
        np.savez(out / "problem.npz", **p)
        actual = gate.recompute(p, {"x": p["expected"], "y": p["witness_y"]}, 1e-7, 1e-7, "fixture")
        q.update({key: actual[key] for key in (*gate.METRICS, "objective", "solution_error")},
                 host_x=p["expected"].tolist(), host_y=p["witness_y"].tolist())
        result.update(problem_hash=gate.sha(b"".join(p[k].tobytes() for k in gate.KEYS)),
                      batch_size=1, validation_parity=True, dtype="float64", eps_abs=1e-7, eps_rel=1e-7,
                      input_generation="excluded; model assembly included",
                      control_order="forward/reverse alternating; one untimed warmup",
                      arm_config={
                          "baseline": dict(rho=1.0, adaptive_rho=False, linear_solver="cholesky"),
                          "control": dict(rho=1.0, adaptive_rho=False, linear_solver="cholesky"),
                          "candidate": dict(rho=1.0, adaptive_rho=True, linear_solver="cholesky")},
                      scope="same dense mathematical workload and validation", measurements={
                          mode: {name: dict(latency=stats([0.03 + i * 0.0001 for i in range(3)]),
                                            quality=[copy.deepcopy(q) for _ in range(3)])
                                 for name in ("baseline", "candidate", "control")}
                          for mode in ("cold", "prepared")})
        (out / "nsys-stats.csv").write_text(
            "Time (%),Total Time (ns),Instances\n100,1000000,10\n"
            "Time (%),Total Time (ns),Instances\n100,100000,10\n"
            "Time (%),Total Time (ns),Instances\n100,10000,10\n")
        (out / "profile.nsys-rep").write_bytes(b"fixture-not-a-real-profile")
        dump(out / "profile-wall.json", dict(profiled_prepared_e2e_s=0.02))
    report = dict(verdict="PASS", mode="correctness" if correctness else "adaptive", size=3, wall_s=1.0,
                  hardware=dict(node="fixture-node", gpu="RTX PRO 4500", torch="2.8.0",
                                cuda="12.8", cpu_threads=1, cpu_quota="100000 100000"),
                  result=result, nsight=dict(kernel_total_s=0.001, kernel_instances=10,
                                            profiled_prepared_e2e_s=0.02),
                  source_hash=hashes["scripts/gpu_qp_campaign.py"])
    dump(out / "result.json", report)
    dump(out / "telemetry.json", [dict(rc=0, gpu="fixture-uuid")])
    (out / "execution.log").write_text("QP_CAMPAIGN_GPU verdict=PASS\n")
    dump(root / "run.json", dict(source_hashes=hashes, jobs=[dict(output=str(out))],
                                 stage="campaign", state="SUBMITTED"))
    return out / "result.json", source, report


def pin(root, result, source):
    bundle = gate.seal([result], [source])
    path = root / "bundle.json"
    dump(path, bundle)
    return path, gate.sha(path.read_bytes())


def test_actual_comparison_schema_replays_kkt_and_ranks_only_scoped(tmp_path):
    result, source, _ = comparison_fixture(tmp_path)
    bundle, digest = pin(tmp_path, result, source)
    before = result.read_bytes()
    report = gate.accept(bundle, digest)
    assert report["verdict"] == "PASS" and report["checked_samples"] == 24
    assert len(report["ranking"]["pairs"]) == 2
    assert report["ranking"]["pairs"][0]["faster_arm"] == "osqp"
    assert report["ranking"]["commercial_win"] is False
    assert report["ranking"]["cross_host_ranking"] == "UNVERIFIED"
    assert result.read_bytes() == before


@pytest.mark.parametrize("mutation,cause", [
    ("status", "strictly solved"), ("primal", "independent threshold"),
    ("dual", "independent threshold"), ("quality", "independent quality mismatch"),
    ("zero", "positive finite"), ("empty", "at least 3 samples"),
    ("eps", "solver eps mismatch"), ("threads", "thread limit mismatch"),
    ("dtype", "dtype/device mismatch"), ("hash", "same-problem hash mismatch"),
    ("prepared", "includes cold setup"), ("order", "sample order position mismatch"),
])
def test_resealed_mutants_cannot_hide_semantic_failure(tmp_path, mutation, cause):
    result, source, report = comparison_fixture(tmp_path)
    row = report["arms"][0]
    sample = row["modes"]["cold"]["samples"][0]
    if mutation == "status":
        sample["solver_status"] = "max_iterations"
    elif mutation == "primal":
        sample["host_solution"]["x"][0] = 10.0
    elif mutation == "dual":
        sample["host_solution"]["y"][0] = 10.0
    elif mutation == "quality":
        sample["quality"]["objective"] = 1.0
    elif mutation == "zero":
        row["modes"]["cold"]["latency"]["samples_s"][0] = 0.0
    elif mutation == "empty":
        row["modes"]["cold"]["latency"]["samples_s"] = []
    elif mutation == "eps":
        row["config"]["eps_abs"] = 1e-4
        row["config_hash"] = gate.sha(json.dumps(row["config"], sort_keys=True).encode())
    elif mutation == "threads":
        report["thread_pools"][0]["threads"] = 8
    elif mutation == "dtype":
        row["config"]["dtype"] = "torch.float32"
        row["config_hash"] = gate.sha(json.dumps(row["config"], sort_keys=True).encode())
    elif mutation == "hash":
        report["problem_hash"] = "b" * 64
    elif mutation == "prepared":
        row["modes"]["prepared"]["samples"][0]["setup_s"] = 0.001
    else:
        sample["order_position"] = 3
    dump(result, report)
    bundle, digest = pin(tmp_path, result, source)
    with pytest.raises(gate.Rejected, match=cause):
        gate.accept(bundle, digest)


@pytest.mark.parametrize("target", ["source", "result", "problem", "bundle", "new-source"])
def test_pinned_bytes_reject_tampering(tmp_path, target):
    result, source, _ = comparison_fixture(tmp_path)
    bundle, digest = pin(tmp_path, result, source)
    path = {"source": source / "src/openopt/qp.py", "result": result,
            "problem": result.parent / "problem.npz", "bundle": bundle,
            "new-source": source / "src/new.py"}[target]
    path.write_bytes(b"changed")
    with pytest.raises(gate.Rejected, match="hash mismatch|source set changed"):
        gate.accept(bundle, digest)


def test_gpu_real_shape_replays_npz_and_host_solutions(tmp_path):
    result, source, _ = gpu_fixture(tmp_path)
    bundle, digest = pin(tmp_path, result, source)
    report = gate.accept(bundle, digest)
    assert report["checked_samples"] == 18
    assert len(report["ranking"]["pairs"]) == 2
    assert report["entries"][0]["independent_solution_replay"] == "PASS"
    assert all(row["status"] == "INCONCLUSIVE_NOISE" for row in report["ranking"]["pairs"])
    assert report["ranking"]["commercial_win"] is False


@pytest.mark.parametrize("mutation", ["status", "empty", "residual", "nan", "count", "profile"])
def test_gpu_resealed_mutants_are_red(tmp_path, mutation):
    result, source, report = gpu_fixture(tmp_path)
    arm = report["result"]["measurements"]["cold"]["candidate"]
    if mutation == "status":
        arm["quality"][0].pop("solver_status")
    elif mutation == "empty":
        arm["quality"] = []
    elif mutation == "residual":
        arm["quality"][0]["dual_residual"] = 10.0
    elif mutation == "nan":
        result.write_text('{"verdict":"PASS","wall_s":NaN,"mode":"adaptive"}')
        with pytest.raises(gate.Rejected, match="nonfinite JSON"):
            pin(tmp_path, result, source)
        return
    elif mutation == "count":
        arm["latency"]["count"] = 4
    else:
        report["nsight"]["profiled_prepared_e2e_s"] = 0.03
    dump(result, report)
    bundle, digest = pin(tmp_path, result, source)
    with pytest.raises(gate.Rejected):
        gate.accept(bundle, digest)


def test_correctness_checks_junit_counts_and_completed_summary(tmp_path):
    result, source, report = gpu_fixture(tmp_path, correctness=True)
    bundle, digest = pin(tmp_path, result, source)
    accepted = gate.accept(bundle, digest)
    assert accepted["entries"][0]["tests"]["tests"] == 7
    (result.parent / "pytest.log").write_text("....... [92%]\n")
    bundle, digest = pin(tmp_path, result, source)
    with pytest.raises(gate.Rejected, match="summary missing"):
        gate.accept(bundle, digest)


def test_empty_sets_and_duplicate_json_fail_loud(tmp_path):
    with pytest.raises(gate.Rejected, match="empty result"):
        gate.seal([])
    with pytest.raises(gate.Rejected, match="duplicate key"):
        gate.parse(b'{"verdict":"FAIL","verdict":"PASS"}', "fixture")
    with pytest.raises(gate.Rejected, match="real number"):
        gate.number(True, "fixture", positive=True)


def test_unknown_commercial_self_report_is_never_a_win(tmp_path):
    result, source, report = comparison_fixture(tmp_path)
    report["commercial"] = {"gurobipy": {"status": "PASS"}}
    dump(result, report)
    bundle, digest = pin(tmp_path, result, source)
    accepted = gate.accept(bundle, digest)
    assert accepted["entries"][0]["commercial"]["gurobi"] == "UNVERIFIED"
    assert not accepted["ranking"]["commercial_win"]


def test_cross_host_results_never_rank_across_entries(tmp_path):
    entries = []
    for name in ("host-a", "host-b"):
        root = tmp_path / name
        root.mkdir()
        result, source, report = comparison_fixture(root)
        report["hardware"]["host"] = name
        dump(result, report)
        entries.append(gate.seal_entry(result, source))
    bundle = tmp_path / "bundle.json"
    dump(bundle, dict(schema_version=1, entries=entries))
    accepted = gate.accept(bundle, gate.sha(bundle.read_bytes()))
    assert accepted["ranking"]["cross_host_ranking"] == "UNVERIFIED"
    assert len(accepted["ranking"]["pairs"]) == 4  # two internal modes per host only


def test_cli_check_preserves_first_cause_on_disk(tmp_path):
    result, source, report = comparison_fixture(tmp_path)
    report["arms"][0]["modes"]["cold"]["samples"][0]["solver_status"] = "max_iterations"
    dump(result, report)
    bundle, digest = pin(tmp_path, result, source)
    # pytest basetemp is external 3FS when using the documented verify entry.
    if not tmp_path.resolve().is_relative_to(Path("/3fs/research3fs/data/tmp")):
        pytest.skip("CLI artifact policy requires the documented external verify entry")
    completed = subprocess.run([
        sys.executable, "-B", str(Path(gate.__file__)), "check", "--bundle", str(bundle),
        "--bundle-sha256", digest, "--artifact-dir", str(tmp_path / "red-evidence"),
    ], capture_output=True, text=True)
    assert completed.returncode == 1 and "strictly solved" in completed.stdout
    evidence = list((tmp_path / "red-evidence").glob("check-*/acceptance.json"))
    assert len(evidence) == 1
    saved = json.loads(evidence[0].read_text())
    assert saved["verdict"] == "FAIL" and saved["exit_code"] == 1
    assert saved["coordinate"].endswith("repeat-0") and saved["traceback"]


def test_real_shape_mutant_entry_has_five_independent_rejections(tmp_path):
    result, source, _ = comparison_fixture(tmp_path)
    bundle, digest = pin(tmp_path, result, source)
    output = tmp_path / "mutants"
    output.mkdir()
    report = gate.mutants(bundle, digest, output)
    assert report["verdict"] == "PASS" and report["mutants_rejected"] == 5
    assert len(list(output.glob("*/rejection.json"))) == 5
    assert all(row["first_cause"] and row["coordinate"] for row in report["outcomes"])


@pytest.mark.parametrize("mutation", ["zero-floor", "missing-control", "unknown-threads"])
def test_control_and_unknown_threads_never_make_a_win(tmp_path, mutation):
    result, source, report = comparison_fixture(tmp_path)
    if mutation == "zero-floor":
        report["controls"][0]["modes"]["cold"]["empirical_noise_floor_s"] = 0.0
    elif mutation == "missing-control":
        report["controls"] = []
    else:
        report["thread_pools"] = [{"status": "UNVERIFIED", "threads": None}]
    dump(result, report)
    bundle, digest = pin(tmp_path, result, source)
    if mutation == "unknown-threads":
        accepted = gate.accept(bundle, digest)
        assert accepted["ranking"]["pairs"] == [] and accepted["ranking"]["excluded"]
    else:
        with pytest.raises(gate.Rejected, match="control"):
            gate.accept(bundle, digest)


@pytest.mark.parametrize("mutation", ["host-x", "host-y", "eps", "dtype", "arm-eps",
                                     "control-config", "algorithm", "problem", "parity", "tables"])
def test_gpu_new_schema_replay_and_parameter_mutants(tmp_path, mutation):
    result, source, report = gpu_fixture(tmp_path)
    new = report["result"]
    quality = new["measurements"]["prepared"]["candidate"]["quality"][0]
    if mutation == "host-x":
        quality["host_x"][0] = 100.0
    elif mutation == "host-y":
        quality["host_y"][0] = 100.0
    elif mutation == "eps":
        new["eps_abs"] = None
    elif mutation == "dtype":
        new["dtype"] = "float32"
    elif mutation == "arm-eps":
        new["arm_config"]["candidate"]["eps_abs"] = 1e-2
    elif mutation == "control-config":
        new["arm_config"]["control"]["rho"] = 2.0
    elif mutation == "algorithm":
        new["arm_config"]["candidate"]["linear_solver"] = "CG"
    elif mutation == "problem":
        new["problem_hash"] = "c" * 64
    elif mutation == "parity":
        new["validation_parity"] = False
    else:
        (result.parent / "nsys-stats.csv").write_text("no kernel table\n")
    dump(result, report)
    bundle, digest = pin(tmp_path, result, source)
    with pytest.raises(gate.Rejected):
        gate.accept(bundle, digest)


def test_check_alias_and_missing_correctness_replay_are_strict(tmp_path):
    result, source, report = gpu_fixture(tmp_path, correctness=True)
    report["result"]["checks"]["batch_0"].pop("host_x")
    dump(result, report)
    bundle, digest = pin(tmp_path, result, source)
    with pytest.raises(gate.Rejected, match="host solution evidence missing"):
        gate.accept(bundle, digest)
    completed = subprocess.run([sys.executable, "-B", str(Path(gate.__file__)),
                                "--check", "--help"], capture_output=True, text=True)
    assert completed.returncode == 0 and "--check" in completed.stdout


@pytest.mark.parametrize("field,value", [("dtype", "float32"), ("eps_abs", 1e-3)])
def test_correctness_config_mutants_rejected(tmp_path, field, value):
    result, source, report = gpu_fixture(tmp_path, correctness=True)
    report["result"][field] = value
    dump(result, report)
    bundle, digest = pin(tmp_path, result, source)
    with pytest.raises(gate.Rejected, match="dtype/eps parity"):
        gate.accept(bundle, digest)
