"""Bounded analytic CPU HSD tests; CUDA target requires explicit owner opt-in."""

from dataclasses import asdict, replace
import argparse
import ast
from fractions import Fraction
import hashlib
import json
import importlib.util
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from xml.etree import ElementTree

import numpy as np
import pytest
import torch

from openopt.conic import (
    ConicConfig, ConicProblem, NonnegativeCone, RotatedSecondOrderCone,
    SecondOrderCone, ZeroCone, check_conic_candidate, solve_conic,
)
import openopt.hsd_lp as hsd
from openopt.hsd_lp import (
    HSDLPConfig, check_hsd_candidate, check_hsd_primal_ray,
    check_hsd_recession_ray, prepare_hsd_lp, solve_hsd_lp,
)


def problem(q, A, b, cones):
    q = np.array(q, dtype=np.float64)
    return ConicProblem(np.zeros((len(q), len(q))), q, np.array(A, dtype=np.float64),
                        np.array(b, dtype=np.float64), tuple(cones))


def equality():
    return problem([1.0], [[1.0]], [2.0], [ZeroCone(1)])


def lower_bound():
    return problem([1.0], [[-1.0]], [-1.0], [NonnegativeCone(1)])


def infeasible():
    return problem([0.0], [[-1.0], [1.0]], [-1.0, 0.0], [NonnegativeCone(2)])


def soc():
    return problem([1.0], [[-1.0], [0.0]], [0.0, 1.0], [SecondOrderCone(2)])


def unbounded():
    return problem([-1.0], [[-1.0]], [0.0], [NonnegativeCone(1)])


def config(**kwargs):
    return HSDLPConfig(enabled=True, **kwargs)


def host(value):
    return value.detach().cpu().numpy()


def serial_result(result):
    record = asdict(result)
    for name in ("x", "s", "y", "primal_ray", "recession_ray", "feasible_base", "u", "v", "w"):
        record[name] = None if getattr(result, name) is None else host(getattr(result, name)).tolist()
    return record


def test_optin_and_preflight_before_factor(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.linalg, "lu_factor_ex", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="opt-in"):
        solve_hsd_lp(equality())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        solve_hsd_lp(equality(), config(device="cuda:0"))
    assert calls == []


@pytest.mark.parametrize("field,value", [
    ("enabled", 1), ("device", 1), ("dtype", torch.float32),
    ("max_iterations", True), ("max_iterations", 10001), ("check_interval", 0),
    ("max_embedding_dimension", 257), ("eps_abs", 2e-7),
    ("eps_rel", 2e-7), ("eps_ray", 2e-7), ("tau_min", 0),
    ("linear_residual_tolerance", 1e-8), ("eps_abs", float("nan")),
])
def test_typed_config_rejects(field, value):
    with pytest.raises((ValueError, RuntimeError, TypeError)):
        solve_hsd_lp(equality(), replace(config(), **{field: value}))


@pytest.mark.parametrize("mutation", ["qp", "tiny_qp", "nan", "empty", "shape", "cone", "cap"])
def test_model_preflight_rejects(mutation):
    p = equality()
    if mutation in ("qp", "tiny_qp"):
        p.P[0, 0] = 1.0 if mutation == "qp" else np.nextafter(0.0, 1.0)
    elif mutation == "nan":
        p.q[0] = float("nan")
    elif mutation == "empty":
        p = problem([], np.empty((0, 0)), [], [])
    elif mutation == "shape":
        p = replace(p, b=[1.0, 2.0])
    elif mutation == "cone":
        p = problem([1.0], [[0.0], [0.0], [0.0]], [1.0, 1.0, 0.0],
                    [RotatedSecondOrderCone(3)])
    else:
        p = problem(np.ones(255), np.empty((0, 255)), [], [])
    with pytest.raises((ValueError, TypeError)):
        solve_hsd_lp(p, config(max_embedding_dimension=255))


def test_exact_Q_resolvent_and_native_calls(monkeypatch):
    factors, solves = [], []
    factor, solve = torch.linalg.lu_factor_ex, torch.linalg.lu_solve

    def observe_factor(matrix, *args, **kwargs):
        factors.append(matrix.clone())
        assert matrix.device == torch.device("cpu")
        return factor(matrix, *args, **kwargs)

    def observe_solve(lu, pivots, rhs, *args, **kwargs):
        solves.append(tuple(str(v.device) for v in (lu, pivots, rhs)))
        return solve(lu, pivots, rhs, *args, **kwargs)

    monkeypatch.setattr(torch.linalg, "lu_factor_ex", observe_factor)
    monkeypatch.setattr(torch.linalg, "lu_solve", observe_solve)
    prepared = prepare_hsd_lp(equality(), config())
    expected_Q = torch.tensor([[0., 1., 1.], [-1., 0., 2.], [-1., -2., 0.]], dtype=torch.float64)
    assert torch.equal(prepared._Q, expected_Q)
    assert torch.equal(expected_Q, -expected_Q.T)
    assert torch.equal(factors[0], torch.eye(3, dtype=torch.float64) + expected_Q)
    result = prepared.solve()
    assert len(factors) == 1
    assert len(solves) == result.iterations == result.linear_solve_calls > 0
    assert set(solves) == {("cpu", "cpu", "cpu")}
    assert result.runtime["max_resolvent_backward_error"] <= 1e-12


@pytest.mark.parametrize("factory,expected", [(equality, 2.), (lower_bound, 1.), (soc, 1.)])
def test_analytic_feasible_and_clarabel(factory, expected, tmp_path):
    p = factory()
    before = [getattr(p, name).tobytes() for name in ("P", "q", "A", "b")]
    result = solve_hsd_lp(p, config())
    assert result.status == "solved_candidate", serial_result(result)
    assert result.candidate_check.passed
    assert len(result.candidate_check.checks) == 6
    assert abs(float(result.x[0]) - expected) <= 2e-7
    original = check_conic_candidate(p, host(result.x), host(result.s), host(result.y),
                                     atol=1e-7, rtol=1e-7)
    assert original.passed, original
    reference = solve_conic(p, ConicConfig(backend="clarabel_reference", eps_abs=1e-9, eps_rel=1e-9))
    assert reference.status == "reference_solved", reference
    assert abs(float(result.x[0]) - float(reference.x[0])) <= 3e-7
    assert before == [getattr(p, name).tobytes() for name in ("P", "q", "A", "b")]
    receipt = tmp_path / "actual.json"
    receipt.write_text(json.dumps({
        "synthetic": True, "commercial_qualification": False,
        "native": serial_result(result),
        "clarabel": {"status": reference.status, "iterations": reference.iterations,
                     "x": reference.x.tolist(), "evidence": asdict(reference.evidence)},
        "original_checker": asdict(original),
    }, indent=2, allow_nan=False) + "\n")
    reread = json.loads(receipt.read_text())
    assert reread["native"]["model_sha256"] == result.model_sha256
    assert reread["native"]["linear_solve_calls"] > 0


def test_analytic_farkas_and_reference(tmp_path):
    p = infeasible()
    assert check_hsd_primal_ray(p, [1., 1.]).passed
    result = solve_hsd_lp(p, config())
    assert result.status == "primal_infeasible_ray", serial_result(result)
    assert result.primal_ray_check.passed
    assert result.x is None and result.feasible_base is None
    assert result.trace[-1]["farkas"]["passed"] and result.trace[-2]["farkas"]["passed"]
    independent = check_hsd_primal_ray(p, host(result.primal_ray))
    assert independent.passed
    reference = solve_conic(p, ConicConfig(backend="clarabel_reference"))
    assert reference.backend_status == "PrimalInfeasible"
    path = tmp_path / "farkas.json"
    path.write_text(json.dumps(serial_result(result), indent=2, allow_nan=False) + "\n")
    assert json.loads(path.read_text())["status"] == result.status


def test_recession_without_base_is_not_primal_unbounded(tmp_path):
    p = unbounded()
    assert check_hsd_recession_ray(p, [1.]).passed
    prepared = prepare_hsd_lp(p, config())
    result = prepared.solve()
    assert result.status == "dual_infeasible_ray", serial_result(result)
    assert result.feasible_base is None and result.x is None
    assert result.recession_ray_check.passed
    with_base = prepared.solve(feasible_base=[0.])
    assert with_base.status == "unbounded_candidate", serial_result(with_base)
    assert with_base.base_check.passed
    assert torch.equal(with_base.feasible_base, torch.zeros(1, dtype=torch.float64))
    reference = solve_conic(p, ConicConfig(backend="clarabel_reference"))
    assert reference.backend_status == "DualInfeasible"
    path = tmp_path / "recession.json"
    path.write_text(json.dumps({"without_base": serial_result(result),
                               "with_explicit_base": serial_result(with_base)}, allow_nan=False) + "\n")
    assert json.loads(path.read_text())["without_base"]["feasible_base"] is None


def test_bad_feasible_base_rejects_before_solve(monkeypatch):
    prepared = prepare_hsd_lp(unbounded(), config())
    monkeypatch.setattr(torch.linalg, "lu_solve", lambda *args: pytest.fail("must reject before solve"))
    with pytest.raises(ValueError, match="base failed"):
        prepared.solve(feasible_base=[-1.])


def test_max_iterations_keeps_failed_candidate():
    result = solve_hsd_lp(equality(), config(max_iterations=1))
    assert result.status == "max_iterations"
    assert not (result.candidate_check and result.candidate_check.passed)
    assert result.iterations == 1 and result.trace
    assert not result.certified


@pytest.mark.parametrize("which", ["candidate", "farkas", "recession"])
def test_wrong_sign_and_wrong_ray_controls(which):
    if which == "candidate":
        evidence = check_hsd_candidate(lower_bound(), [1.], [0.], [-1e-17])
        assert not evidence.passed and not evidence.checks["dual_cone"]
    elif which == "farkas":
        evidence = check_hsd_primal_ray(infeasible(), [-1., -1.])
        assert not evidence.passed and not evidence.checks["strict_negative"]
        tiny = check_hsd_primal_ray(infeasible(), [1., -1e-17])
        assert not tiny.passed and not tiny.checks["dual_cone"]
    else:
        evidence = check_hsd_recession_ray(unbounded(), [-1.])
        assert not evidence.passed and not evidence.checks["recession_cone"]


def test_soc_wrong_ray_independent_of_projection():
    p = problem([0.], [[0.], [0.]], [-1., 0.], [SecondOrderCone(2)])
    assert check_hsd_primal_ray(p, [1., 0.]).passed
    bad = check_hsd_primal_ray(p, [1., 1. + 1e-12])
    assert not bad.passed and not bad.checks["dual_cone"]


def test_tiny_feasible_A_cannot_be_false_farkas():
    p = problem([0.], [[1e-20]], [-1.], [NonnegativeCone(1)])
    evidence = check_hsd_primal_ray(p, [1.])
    assert not evidence.passed and not evidence.checks["annihilation"]
    assert evidence.thresholds["annihilation_0"] <= 1e-27
    assert Fraction.from_float(p.A[0, 0]) * Fraction(1) != 0


def test_candidate_componentwise_not_global_threshold():
    p = problem([1., 1e8], [[1., 0.], [0., 1.]], [1., 1.], [ZeroCone(2)])
    good = check_hsd_candidate(p, [1., 1.], [0., 0.], [-1., -1e8])
    assert good.passed
    bad = check_hsd_candidate(p, [1., 1.], [0., 0.], [-1. + 1e-5, -1e8])
    assert not bad.passed and not bad.checks["stationarity"]
    assert bad.metrics["stationarity_0"] > bad.thresholds["stationarity_0"]


@pytest.mark.parametrize("name", ["_q", "_Q", "_lu", "_system"])
def test_coordinated_device_mutation_rejects(name):
    prepared = prepare_hsd_lp(equality(), config())
    tensor = getattr(prepared, name)
    before = tensor.clone()
    tensor[(0,) * tensor.ndim] += 0.125
    assert not torch.equal(tensor, before), "negative control must mutate the actual strided factor"
    with pytest.raises(RuntimeError, match="frozen device tensor changed"):
        prepared.solve()


def test_model_and_original_property_are_frozen():
    p = equality()
    prepared = prepare_hsd_lp(p, config())
    p.q[0] = 9.
    detached = prepared.original_problem
    detached.b[0] = 8.
    assert prepared.original_problem.q[0] == 1.
    assert prepared.original_problem.b[0] == 2.
    result = prepared.solve()
    assert result.status == "solved_candidate"
    prepared._original.q.setflags(write=True)
    prepared._original.q[0] = 7.
    with pytest.raises(RuntimeError, match="frozen model/config"):
        prepared.solve()


def load_hsd_external(source, name):
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def test_source_byte_tamper_rejects(tmp_path):
    source = tmp_path / "hsd_lp.py"
    source.write_bytes(Path(hsd.__file__).read_bytes())
    name = "openopt._hsd_owner_disk_tamper"
    try:
        module = load_hsd_external(source, name)
        prepared = module.prepare_hsd_lp(equality(), module.HSDLPConfig(enabled=True))
        source.write_bytes(source.read_bytes() + b"\n# tampered source\n")
        with pytest.raises(RuntimeError, match="binding.*source buffer changed"):
            prepared.solve()
    finally:
        sys.modules.pop(name, None)


def test_runtime_thread_change_rejects(monkeypatch):
    prepared = prepare_hsd_lp(equality(), config())
    actual = torch.get_num_threads()
    monkeypatch.setattr(torch, "get_num_threads", lambda: actual + 1)
    with pytest.raises(RuntimeError, match="runtime/thread"):
        prepared.solve()


def test_wrong_device_or_dtype_resolvent_rejects(monkeypatch):
    prepared = prepare_hsd_lp(equality(), config())
    monkeypatch.setattr(torch.linalg, "lu_solve",
                        lambda *args: torch.zeros((3, 1), device="meta", dtype=torch.float64))
    with pytest.raises(RuntimeError, match="residency mismatch"):
        prepared.solve()


def test_nonfinite_factor_fail_loud(monkeypatch):
    def bad(matrix):
        return matrix * float("nan"), torch.ones(3, dtype=torch.int32), torch.tensor(0)
    monkeypatch.setattr(torch.linalg, "lu_factor_ex", bad)
    with pytest.raises(RuntimeError, match="nonfinite LU"):
        prepare_hsd_lp(equality(), config())


def test_nonfinite_iterate_fail_loud(monkeypatch):
    prepared = prepare_hsd_lp(equality(), config())
    monkeypatch.setattr(torch.linalg, "lu_solve",
                        lambda *args: torch.full((3, 1), float("inf"), dtype=torch.float64))
    with pytest.raises(RuntimeError, match="nonfinite resolvent"):
        prepared.solve()


def test_wrong_resolvent_not_silently_accepted(monkeypatch):
    prepared = prepare_hsd_lp(equality(), config())
    monkeypatch.setattr(torch.linalg, "lu_solve", lambda *args: torch.zeros((3, 1), dtype=torch.float64))
    with pytest.raises(RuntimeError, match="resolvent residual failed"):
        prepared.solve()


def test_original_audit_nonrepresentable_fails():
    p = problem([1e308], [[1.]], [1.], [ZeroCone(1)])
    with pytest.raises(RuntimeError, match="overflow"):
        check_hsd_candidate(p, [1e308], [0.], [-1.])
    p = problem([1e-200], [[1.]], [1.], [ZeroCone(1)])
    with pytest.raises(RuntimeError, match="underflow"):
        check_hsd_candidate(p, [1e-200], [0.], [-1.])


def test_hot_factor_cold_state_is_reproducible():
    prepared = prepare_hsd_lp(equality(), config())
    first, second = prepared.solve(), prepared.solve()
    assert first.status == second.status == "solved_candidate"
    assert first.iterations == second.iterations
    assert torch.equal(first.u, second.u) and torch.equal(first.x, second.x)
    assert first.model_sha256 == second.model_sha256
    assert first.source_sha256["openopt.hsd_lp"] == hashlib.sha256(Path(hsd.__file__).read_bytes()).hexdigest()


def test_unconstrained_lp_has_ray_not_fabricated_base():
    p = problem([-1.], np.empty((0, 1)), [], [])
    result = solve_hsd_lp(p, config())
    assert result.status == "dual_infeasible_ray", serial_result(result)
    assert result.feasible_base is None
    with_base = solve_hsd_lp(p, config(), feasible_base=[0.])
    assert with_base.status == "unbounded_candidate"


@pytest.mark.parametrize("factory,expected", [
    (lambda: problem([0., 1.], [[0., 1.], [0., 2.]], [1., 2.], [ZeroCone(2)]), 1.),
    (lambda: problem([1.], [[-1.], [-1.]], [-1., -1.], [NonnegativeCone(2)]), 1.),
])
def test_degenerate_zero_column_and_dependent_rows(factory, expected):
    p = factory()
    result = solve_hsd_lp(p, config())
    assert result.status == "solved_candidate", serial_result(result)
    assert result.candidate_check.passed
    assert abs(result.candidate_check.metrics["primal_objective"] - expected) <= 2e-7
    assert check_conic_candidate(p, host(result.x), host(result.s), host(result.y),
                                 atol=1e-7, rtol=1e-7).passed


def test_soc_infeasible_analytic():
    p = problem([0.], [[0.], [0.]], [-1., 0.], [SecondOrderCone(2)])
    result = solve_hsd_lp(p, config())
    assert result.status == "primal_infeasible_ray", serial_result(result)
    assert result.primal_ray_check.passed
    assert result.feasible_base is None
    assert solve_conic(p, ConicConfig(backend="clarabel_reference")).backend_status == "PrimalInfeasible"


def test_both_primal_and_dual_infeasible_never_unbounded():
    p = problem([-1.], [[0.], [-1.]], [-1., 0.], [NonnegativeCone(2)])
    assert check_hsd_primal_ray(p, [1., 0.]).passed
    assert check_hsd_recession_ray(p, [1.]).passed
    result = solve_hsd_lp(p, config())
    assert result.status in ("primal_infeasible_ray", "dual_infeasible_ray"), serial_result(result)
    assert result.feasible_base is None
    with pytest.raises(ValueError, match="base failed"):
        solve_hsd_lp(p, config(), feasible_base=[0.])


def test_ray_normalization_underflow_fail_loud():
    prepared = prepare_hsd_lp(equality(), config())
    with pytest.raises(RuntimeError, match="normalization product underflow"):
        prepared._normalize(torch.tensor([1e-200], dtype=torch.float64),
                            torch.tensor([1e-200], dtype=torch.float64), "negative control")


def test_factor_exception_has_no_fallback(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("native factor rejected")
    monkeypatch.setattr(torch.linalg, "lu_factor_ex", fail)
    monkeypatch.setattr(np.linalg, "solve", lambda *args: pytest.fail("CPU numpy fallback forbidden"))
    with pytest.raises(RuntimeError, match="no fallback"):
        prepare_hsd_lp(equality(), config())


def test_scoped_source_security_and_membership():
    root = Path(__file__).resolve().parents[1]
    scope = ("src/openopt/hsd_lp.py", "tests/test_hsd_lp.py", "docs/research/hsd-lp.md")
    patterns = [
        re.compile(r"(?i)pass" + r"word[\"']?\s*[:=]\s*[\"'][^\"']{6,}"),
        re.compile(r"-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    ]
    for name in scope:
        path = root / name
        assert path.is_file() and not path.is_symlink()
        content = path.read_text()
        assert content and all(not pattern.search(content) for pattern in patterns)
    assert all(patterns[0].search(value) for value in (
        "pass" + "word = 'fixture-decoy'",
        "PASS" + "WORD: 'fixture-decoy'",
    )), "security scan must see its synthetic decoys"


def model_record(p):
    return {
        "P": np.asarray(p.P).tolist(), "q": np.asarray(p.q).tolist(),
        "A": np.asarray(p.A).tolist(), "b": np.asarray(p.b).tolist(),
        "A_shape": list(np.shape(p.A)),
        "cones": [{"type": type(cone).__name__, "dimension": cone.dimension} for cone in p.cones],
    }


def replay_record(record):
    """No solves: validate saved model/config/source and recompute all saved statuses."""
    if set(record) != {"schema", "cases", "calls", "qualification"} or record["schema"] != "hsd-lp-cpu-v1":
        raise ValueError("receipt schema mismatch")
    expected = {
        "equality": ("solved_candidate", False),
        "lower": ("solved_candidate", False), "soc": ("solved_candidate", False),
        "infeasible": ("primal_infeasible_ray", False),
        "soc_infeasible": ("primal_infeasible_ray", False),
        "recession_no_base": ("dual_infeasible_ray", False),
        "recession_explicit_base": ("unbounded_candidate", True),
        "budget_failure": ("max_iterations", False),
    }
    if len(record["cases"]) != len(expected) or {case["case_id"] for case in record["cases"]} != set(expected):
        raise ValueError("receipt denominator/case membership mismatch")
    if record["calls"]["native_solves"] != len(expected) or record["calls"]["cpu_fallbacks"] != 0:
        raise ValueError("receipt actual solve denominator mismatch")
    total_calls = 0
    cone_types = {"ZeroCone": ZeroCone, "NonnegativeCone": NonnegativeCone,
                  "SecondOrderCone": SecondOrderCone}
    for case in record["cases"]:
        model, result = case["model"], case["result"]
        cones = tuple(cone_types[item["type"]](item["dimension"]) for item in model["cones"])
        p = ConicProblem(np.array(model["P"]), np.array(model["q"]),
                         np.array(model["A"]).reshape(model["A_shape"]), np.array(model["b"]), cones)
        assert hsd._hash_model(hsd._model(p)) == result["model_sha256"], "original model byte hash"
        assert hsd._source_hashes() == result["source_sha256"], "source bytes"
        saved_config = dict(result["settings"])
        assert saved_config.pop("dtype") == "torch.float64"
        c = HSDLPConfig(**saved_config)
        c.validate()
        assert c == (config(max_iterations=1) if case["case_id"] == "budget_failure" else config())
        status, has_base = expected[case["case_id"]]
        assert result["status"] == status
        assert 1 <= result["iterations"] <= c.max_iterations
        assert result["linear_solve_calls"] == result["iterations"]
        assert result["runtime"]["linear_solve_calls"] == result["iterations"]
        assert result["runtime"]["factor_device"] == "cpu"
        assert result["runtime"]["pivots_device"] == "cpu"
        assert result["runtime"]["embedding_device"] == "cpu"
        assert result["runtime"]["factorization_calls"] == 1
        assert result["runtime"]["projection_calls"] == result["iterations"]
        assert result["runtime"]["max_resolvent_backward_error"] <= c.linear_residual_tolerance
        assert len(result["u"]) == len(p.q) + len(p.b) + 1
        assert np.isfinite(np.asarray(result["u"])).all()
        assert np.isfinite(np.asarray(result["v"])).all()
        assert np.isfinite(np.asarray(result["w"])).all()
        total_calls += result["iterations"]
        if status == "solved_candidate":
            evidence = check_hsd_candidate(p, result["x"], result["s"], result["y"])
            assert asdict(evidence) == result["candidate_check"] and evidence.passed
            assert check_conic_candidate(p, result["x"], result["s"], result["y"],
                                         atol=1e-7, rtol=1e-7).passed
        elif status == "primal_infeasible_ray":
            evidence = check_hsd_primal_ray(p, result["primal_ray"])
            assert asdict(evidence) == result["primal_ray_check"] and evidence.passed
            assert result["trace"][-1]["farkas"]["passed"] and result["trace"][-2]["farkas"]["passed"]
        elif status != "max_iterations":
            evidence = check_hsd_recession_ray(p, result["recession_ray"])
            assert asdict(evidence) == result["recession_ray_check"] and evidence.passed
            assert result["trace"][-1]["recession"]["passed"] and result["trace"][-2]["recession"]["passed"]
        else:
            assert result["candidate_check"] is None or not result["candidate_check"]["passed"]
        assert (result["feasible_base"] is not None) == has_base
        if has_base:
            base = np.array(result["feasible_base"])
            evidence = hsd._primal(hsd._model(p), base, p.b - p.A @ base, c.eps_abs, c.eps_rel)
            assert asdict(evidence) == result["base_check"] and evidence.passed
        assert result["certified"] is False
    assert record["calls"]["lu_solves"] == total_calls > 0
    return {"verdict": "HSD_CPU_REPLAY_PASS", "cases": len(expected), "replay_solve_calls": 0,
            "qualification": "bounded synthetic numerical evidence only"}


def test_actual_receipt_and_rehashed_tamper_controls(tmp_path, monkeypatch):
    cases = [
        ("equality", equality(), config(), None), ("lower", lower_bound(), config(), None),
        ("soc", soc(), config(), None), ("infeasible", infeasible(), config(), None),
        ("soc_infeasible", problem([0.], [[0.], [0.]], [-1., 0.], [SecondOrderCone(2)]), config(), None),
        ("recession_no_base", unbounded(), config(), None),
        ("recession_explicit_base", unbounded(), config(), [0.]),
        ("budget_failure", equality(), config(max_iterations=1), None),
    ]
    counts = {"native_solves": 0, "lu_solves": 0, "cpu_fallbacks": 0}
    actual_solve = torch.linalg.lu_solve

    def observe(*args, **kwargs):
        counts["lu_solves"] += 1
        assert all(value.device == torch.device("cpu") for value in args[:3])
        return actual_solve(*args, **kwargs)
    monkeypatch.setattr(torch.linalg, "lu_solve", observe)
    records = []
    for case_id, p, c, base in cases:
        result = solve_hsd_lp(p, c, feasible_base=base)
        counts["native_solves"] += 1
        records.append({"case_id": case_id, "model": model_record(p), "result": serial_result(result)})
    receipt = {
        "schema": "hsd-lp-cpu-v1", "cases": records, "calls": counts,
        "qualification": "synthetic; CPU only; not exact/commercial/public qualification",
    }
    path = tmp_path / "actual-cpu.json"
    path.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    pinned = hashlib.sha256(path.read_bytes()).hexdigest()
    assert pinned != hashlib.sha256(b"").hexdigest()
    reread = json.loads(path.read_text())
    monkeypatch.setattr(torch.linalg, "lu_solve", lambda *args: pytest.fail("replay must not solve"))
    assert replay_record(reread)["cases"] == 8
    for mutation in ("wrong_model", "wrong_ray", "dropped_failure", "wrong_settings", "fake_base"):
        bad = json.loads(json.dumps(reread))
        if mutation == "wrong_model":
            bad["cases"][0]["model"]["q"][0] += 1.
        elif mutation == "wrong_ray":
            bad["cases"][3]["result"]["primal_ray"] = [-1., -1.]
        elif mutation == "dropped_failure":
            bad["cases"].pop()
            bad["calls"]["native_solves"] -= 1
        elif mutation == "wrong_settings":
            bad["cases"][0]["result"]["settings"]["max_iterations"] += 1
        else:
            bad["cases"][5]["result"]["feasible_base"] = [0.]
        tamper_path = tmp_path / (mutation + ".json")
        tamper_path.write_text(json.dumps(bad, allow_nan=False) + "\n")
        resealed = hashlib.sha256(tamper_path.read_bytes()).hexdigest()
        assert resealed != pinned
        with pytest.raises((AssertionError, ValueError)):
            replay_record(json.loads(tamper_path.read_text()))


@pytest.mark.parametrize("function", ["check_hsd_candidate", "check_hsd_primal_ray"])
def test_loaded_checker_restored_disk_rejects(function, tmp_path):
    """Same actual load/restore control as the frozen independent red review."""
    trusted = Path(hsd.__file__).read_bytes()
    source = tmp_path / "hsd_lp.py"
    source.write_bytes(trusted)
    name = "openopt._hsd_owner_" + function

    def load():
        return load_hsd_external(source, name)

    try:
        baseline = load()
        budget = 10 if function == "check_hsd_candidate" else 2
        p = equality() if function == "check_hsd_candidate" else infeasible()
        original = baseline.solve_hsd_lp(
            p, baseline.HSDLPConfig(enabled=True, max_iterations=budget, check_interval=1),
        )
        assert original.status == "max_iterations"
        tree = ast.parse(trusted)
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        lines = trusted.decode().splitlines(keepends=True)
        malicious = (
            f"def {function}(*args, **kwargs):\n"
            "    return HSDLPCheck('altered', True, {'false_accept': True}, {}, {})\n"
        )
        source.write_text("".join(lines[:node.lineno - 1]) + malicious
                          + "".join(lines[node.end_lineno:]))
        altered_sha = hashlib.sha256(source.read_bytes()).hexdigest()
        module = load()
        source.write_bytes(trusted)
        assert source.read_bytes() == trusted
        rejected = False
        rejection_reason = None
        result = None
        try:
            result = module.solve_hsd_lp(
                p, module.HSDLPConfig(enabled=True, max_iterations=budget, check_interval=1),
            )
        except RuntimeError as error:
            assert "binding" in str(error).lower() or "source" in str(error).lower()
            rejected = True
            rejection_reason = str(error)
        (tmp_path / "loaded-restore.json").write_text(json.dumps({
            "checker": function, "rejected": rejected,
            "rejection_reason": rejection_reason, "original_model": model_record(p),
            "baseline_status": original.status,
            "altered_source_sha256": altered_sha,
            "restored_sha256": hashlib.sha256(trusted).hexdigest(),
            "result": None if result is None else serial_result(result),
            "qualification": "synthetic CPU negative control; no CUDA qualification",
        }, allow_nan=False) + "\n")
        assert rejected, "actual altered loaded checker accepted after trusted disk restoration"
    finally:
        source.write_bytes(trusted)
        sys.modules.pop(name, None)


@pytest.mark.parametrize("stage", ["prepare", "solve"])
@pytest.mark.parametrize("mutation", [
    "helper", "kwdefault", "posdefault", "config_default", "config_field", "config_init",
    "config_eq", "check_class", "class_getattribute", "class_extra",
    "projector_alias", "projector_wrapper", "projector_closure",
    "cone_alias", "dimension_helper", "fraction_alias", "numpy_alias", "global_cap",
    "binding_require", "binding_codes", "binding_entry",
])
def test_actual_loaded_closure_changes_reject(stage, mutation, monkeypatch, tmp_path):
    prepared = prepare_hsd_lp(equality(), config()) if stage == "solve" else None
    import openopt.conic as conic_module
    import openopt.native_conic as native_module

    def bad(*args, **kwargs):
        return None

    if mutation == "helper":
        monkeypatch.setattr(hsd, "_dot", bad)
    elif mutation == "kwdefault":
        monkeypatch.setattr(hsd.check_hsd_candidate, "__kwdefaults__",
                            {"eps_abs": 1e-3, "eps_rel": 1e-7})
    elif mutation == "posdefault":
        monkeypatch.setattr(hsd._model, "__defaults__", (255,))
    elif mutation == "config_default":
        monkeypatch.setattr(hsd.HSDLPConfig, "eps_abs", 1e-3)
    elif mutation == "config_field":
        monkeypatch.setattr(hsd.HSDLPConfig.__dataclass_fields__["eps_abs"], "default", 1e-3)
    elif mutation in ("config_init", "config_eq"):
        monkeypatch.setattr(hsd.HSDLPConfig, "__init__" if mutation == "config_init" else "__eq__", bad)
    elif mutation == "check_class":
        monkeypatch.setattr(hsd, "HSDLPCheck", type("HSDLPCheck", (), {}))
    elif mutation == "class_getattribute":
        monkeypatch.setattr(hsd.PreparedHSDLP, "__getattribute__", bad)
    elif mutation == "class_extra":
        monkeypatch.setattr(hsd.PreparedHSDLP, "__getattr__", bad, raising=False)
    elif mutation == "projector_alias":
        monkeypatch.setattr(hsd, "project_cones", bad)
    elif mutation == "projector_wrapper":
        trusted = native_module.project_cones

        def misleading(*args, **kwargs):
            return trusted(*args, **kwargs)
        misleading.__wrapped__ = trusted.__wrapped__
        monkeypatch.setattr(native_module, "project_cones", misleading)
        monkeypatch.setattr(hsd, "project_cones", misleading)
    elif mutation == "projector_closure":
        wrapper = native_module.project_cones
        cell = dict(zip(wrapper.__code__.co_freevars, wrapper.__closure__))["func"]
        original = cell.cell_contents
        cell.cell_contents = bad
        try:
            with pytest.raises(RuntimeError, match="binding"):
                (prepared.solve() if stage == "solve" else prepare_hsd_lp(equality(), config()))
        finally:
            cell.cell_contents = original
        return
    elif mutation == "cone_alias":
        monkeypatch.setattr(hsd, "SecondOrderCone", conic_module.NonnegativeCone)
    elif mutation == "dimension_helper":
        monkeypatch.setattr(conic_module, "_dimension", bad)
    elif mutation == "fraction_alias":
        monkeypatch.setattr(hsd, "Fraction", float)
    elif mutation == "numpy_alias":
        monkeypatch.setattr(hsd, "np", type("FakeNumpy", (), {})())
    elif mutation == "binding_require":
        monkeypatch.setattr(hsd, "_binding_require", bad)
    elif mutation == "binding_codes":
        monkeypatch.setattr(hsd, "_binding_codes", bad)
    elif mutation == "binding_entry":
        monkeypatch.setattr(hsd, "_loaded_binding", bad)
    else:
        monkeypatch.setattr(hsd, "MAX_ITERATIONS", 10001)
    rejected = False
    reason = None
    try:
        if stage == "solve":
            hsd.PreparedHSDLP.solve(prepared)
        else:
            prepare_hsd_lp(equality(), config())
    except (RuntimeError, TypeError) as error:
        rejected = True
        reason = str(error)
    (tmp_path / "closure-negative.json").write_text(json.dumps({
        "stage": stage, "mutation": mutation, "rejected": rejected, "reason": reason,
        "source_sha256": hashlib.sha256(Path(hsd.__file__).read_bytes()).hexdigest(),
    }, allow_nan=False) + "\n")
    assert rejected and ("binding" in reason or mutation == "config_init")


def test_subclass_and_extra_config_cannot_override_validate():
    class MaliciousConfig(HSDLPConfig):
        def validate(self):
            pytest.fail("subclass validation must never execute")
    with pytest.raises(TypeError, match="config must"):
        prepare_hsd_lp(equality(), MaliciousConfig(enabled=True))
    supplied = config()
    object.__setattr__(supplied, "extra_behavior", True)
    with pytest.raises(TypeError, match="config must"):
        prepare_hsd_lp(equality(), supplied)


def test_post_solve_loaded_helper_change_rejects(monkeypatch):
    prepared = prepare_hsd_lp(equality(), config(max_iterations=1))
    original = torch.linalg.lu_solve
    calls = []

    def mutate_during_solve(*args, **kwargs):
        calls.append(1)
        result = original(*args, **kwargs)
        monkeypatch.setattr(hsd.check_hsd_candidate, "__kwdefaults__",
                            {"eps_abs": 1e-7, "eps_rel": 0.0})
        return result
    monkeypatch.setattr(torch.linalg, "lu_solve", mutate_during_solve)
    with pytest.raises(RuntimeError, match="binding.*defaults"):
        prepared.solve()
    assert len(calls) == 1


@pytest.mark.skipif(os.environ.get("OPENOPT_RUN_HSD_CUDA_TESTS") != "1",
                    reason="owner-only explicit CUDA target; local CPU is not CUDA evidence")
def test_cuda_actual_residency_and_analytic_candidate():
    if not torch.cuda.is_available():
        pytest.fail("explicit HSD CUDA target lacks CUDA; no skip/fallback")
    prepared = prepare_hsd_lp(equality(), config(device="cuda:0"))
    result = prepared.solve()
    assert result.status == "solved_candidate"
    for value in (prepared._lu, prepared._pivots, prepared._Q, result.u, result.v, result.w, result.x):
        assert value.device == torch.device("cuda:0")
    assert result.runtime["factor_device"] == "cuda:0"
    assert result.linear_solve_calls > 0
    assert check_hsd_candidate(equality(), host(result.x), host(result.s), host(result.y)).passed


SCOPED_PATHS = ("src/openopt/hsd_lp.py", "tests/test_hsd_lp.py", "docs/research/hsd-lp.md")


def scan_scoped_sources(artifact_root, scan_tool, needle_file):
    """Reuse the existing literal scanner without its unsafe secret-tail output."""
    root = Path(__file__).resolve().parents[1]
    artifact_root = artifact_root.resolve()
    assert not artifact_root.is_relative_to(root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="scan-", dir=artifact_root))
    spec = importlib.util.spec_from_file_location("_hsd_scanner", scan_tool)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needle = needle_file.read_text().strip()
    assert needle, "empty scanner sensitivity anchor"
    paths = [str(root / p) for p in SCOPED_PATHS]
    hashes = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    hits, _ = scanner.scan_literal(paths, [needle])
    shape = re.compile(
        r"(?i)IDENTIFIED" + r" BY|pass" + r"""word["']?\s*[:=]\s*["'][^"']{6,}"""
        r"|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    )
    shape_hits = sum(bool(shape.search(Path(p).read_text())) for p in paths)
    canary = directory / "sensitivity.txt"
    anchor = "hsd-binding-scan-" + "not-a-credential"
    canary.write_text(anchor)
    seen, _ = scanner.scan_literal([str(canary)], [anchor])
    canary.write_text("cleared sensitivity fixture\n")
    cleared, _ = scanner.scan_literal([str(canary)], [anchor])
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}
    receipt = {"verdict": "PASS" if hits == shape_hits == cleared == 0 and seen == 1
               and hashes == after else "FAIL", "files": hashes,
               "literal_hits": hits, "shape_hits": shape_hits,
               "sensitivity_seen": seen, "sensitivity_removed": cleared == 0,
               "source_unchanged": hashes == after,
               "scan_tool_sha256": hashlib.sha256(scan_tool.read_bytes()).hexdigest()}
    destination = directory / "scan.json"
    destination.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdict": receipt["verdict"], "receipt": str(destination)}))
    return 0 if receipt["verdict"] == "PASS" else 1


def audit_unit_receipt(path, sha256, owner_commit):
    """Check clean current bytes and the exact actual collection/JUnit denominator."""
    from scripts.verify import source_snapshot
    root = Path(__file__).resolve().parents[1]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256, "caller receipt pin"
    receipt = json.loads(path.read_text())
    current = source_snapshot(root)
    assert current["clean"] and current["commit"] == owner_commit
    assert receipt["source"] == current and receipt["source_consistency"]["verdict"] == "PASS"
    assert receipt["targets"] == ["tests/test_hsd_lp.py"] and receipt["verdict"] == "PASS"
    directory = path.parent
    env = os.environ.copy()
    env.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=os.pathsep.join([str(root / "src"), str(root)]))
    collected = subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
         "-o", f"cache_dir={directory / 'collection-cache'}", "tests/test_hsd_lp.py"],
        cwd=root, env=env, text=True, capture_output=True, check=False,
    )
    (directory / "collection.log").write_text(
        collected.stdout + collected.stderr + f"\nreturncode={collected.returncode}\n")
    assert collected.returncode == 0
    selection = [v for v in collected.stdout.splitlines() if v.startswith("tests/test_hsd_lp.py::")]
    nodes = ElementTree.parse(directory / "pytest.xml").getroot().findall(".//testcase")
    membership = [v.attrib["classname"] + "::" + v.attrib["name"] for v in nodes]
    expected = ["tests.test_hsd_lp::" + v.split("::", 1)[1] for v in selection]
    assert len(selection) == len(set(selection)) == len(nodes) == len(set(membership)) == 104
    assert sorted(membership) == sorted(expected)
    assert not any(v.find("failure") is not None or v.find("error") is not None for v in nodes)
    skipped = [v for v in nodes if v.find("skipped") is not None]
    assert len(skipped) == 1 and skipped[0].attrib["name"] == \
        "test_cuda_actual_residency_and_analytic_candidate"
    negatives = [json.loads(p.read_text()) for p in directory.rglob("loaded-restore.json")]
    assert len(negatives) == 2 and all(v["rejected"] and v["rejection_reason"] for v in negatives)
    closures = [json.loads(p.read_text()) for p in directory.rglob("closure-negative.json")]
    assert len(closures) == 40 and all(v["rejected"] and v["reason"] for v in closures)
    assert source_snapshot(root) == current
    result = {"verdict": "HSD_LOADED_REPAIR_AUTHOR_PASS", "source": current,
              "verification_sha256": sha256, "selection": selection, "membership": membership,
              "tests": {"passed": 103, "skipped": 1, "failed": 0},
              "skip_reason": skipped[0].find("skipped").attrib,
              "loaded_restore_controls": negatives, "serialized_closure_controls": 40,
              "runtime": {"python": sys.version, "numpy": np.__version__,
                          "torch": str(torch.__version__), "cuda_available": torch.cuda.is_available(),
                          "torch_threads": torch.get_num_threads(),
                          "torch_interop_threads": torch.get_num_interop_threads()},
              "artifact_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (directory / "pytest.xml", directory / "pytest.log",
                                            directory / "collection.log")},
              "qualification": "author bounded CPU validation; unchanged independent review pending"}
    destination = directory / "author-receipt.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdict": result["verdict"], "receipt": str(destination),
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))
    return 0


def run_unchanged_review(artifact_root, reviewer_root, reviewer_commit, owner_commit):
    """Run byte-identical historical tests against caller-pinned new production.

    The historical in-module OWNER label is not rewritten or promoted. The
    enclosing receipt binds actual new owner and reviewer bytes independently.
    """
    from scripts.verify import source_snapshot
    root = Path(__file__).resolve().parents[1]
    reviewer_root = reviewer_root.resolve()
    artifact_root = artifact_root.resolve()
    assert not artifact_root.is_relative_to(root) and not artifact_root.is_relative_to(reviewer_root)
    before, review_before = source_snapshot(root), source_snapshot(reviewer_root)
    assert before["clean"] and before["commit"] == owner_commit
    assert review_before["clean"] and review_before["commit"] == reviewer_commit
    artifact_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="unchanged-", dir=artifact_root))
    mirror = directory / "source"
    for name in before["sha256"]:
        destination = mirror / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    review_paths = ("tests/test_hsd_lp_independent.py", "docs/research/hsd-lp-independent.md")
    for name in review_paths:
        destination = mirror / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(reviewer_root / name, destination)
        assert destination.read_bytes() == (reviewer_root / name).read_bytes()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", TMPDIR=str(directory),
               PYTHONPATH=os.pathsep.join([str(mirror / "src"), str(mirror)]))
    common = [sys.executable, "-B", "-m", "pytest", "-q", "-ra",
              "-o", f"cache_dir={directory / 'cache'}"]
    collected = subprocess.run(common + ["--collect-only", review_paths[0]], cwd=mirror,
                               env=env, text=True, capture_output=True, check=False)
    (directory / "collection.log").write_text(
        collected.stdout + collected.stderr + f"\nreturncode={collected.returncode}\n")
    assert collected.returncode == 0
    selection = [v for v in collected.stdout.splitlines() if v.startswith(review_paths[0] + "::")]
    assert len(selection) == len(set(selection)) == 45
    command = common + ["--basetemp", str(directory / "tmp"),
                        f"--junitxml={directory / 'pytest.xml'}", review_paths[0]]
    with (directory / "pytest.log").open("x") as log:
        log.write(f"command={command!r}\n")
        log.flush()
        execution = subprocess.run(command, cwd=mirror, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
        log.write(f"\nreturncode={execution.returncode}\n")
    nodes = ElementTree.parse(directory / "pytest.xml").getroot().findall(".//testcase")
    membership = [v.attrib["classname"] + "::" + v.attrib["name"] for v in nodes]
    assert sorted(membership) == sorted(
        "tests.test_hsd_lp_independent::" + v.split("::", 1)[1] for v in selection)
    counts = {key: sum(v.find(tag) is not None for v in nodes) for key, tag in
              (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))}
    counts.update(tests=len(nodes), passed=len(nodes) - sum(counts.values()))
    consistency = before == source_snapshot(root) and review_before == source_snapshot(reviewer_root)
    for name, wanted in before["sha256"].items():
        assert hashlib.sha256((mirror / name).read_bytes()).hexdigest() == wanted
    result = {"verdict": "PASS" if execution.returncode == 0 and consistency else "FAIL",
              "returncode": execution.returncode, "tests": counts,
              "owner": before, "reviewer": review_before, "source_consistency": consistency,
              "selection": selection, "membership": membership, "command": command,
              "unchanged_review_hashes": {p: review_before["sha256"][p] for p in review_paths},
              "runtime_env": {k: env[k] for k in ("CUDA_VISIBLE_DEVICES", "OPENBLAS_NUM_THREADS",
                                                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "PYTHONPATH")},
              "historical_internal_owner_label": "695df3b1f998ab4ede15c0172cd948cba1910173",
              "qualification": "author replay of unchanged independent controls; not reviewer signoff",
              "artifacts": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                            (directory / "pytest.log", directory / "pytest.xml",
                             directory / "collection.log")}}
    destination = directory / "receipt.json"
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"verdict": result["verdict"], "tests": counts, "receipt": str(destination),
                      "sha256": hashlib.sha256(destination.read_bytes()).hexdigest()}))
    return execution.returncode or (0 if consistency else 2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pinned HSD CPU replay and scoped validation receipts")
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--scan-tool", type=Path)
    parser.add_argument("--needle-file", type=Path)
    parser.add_argument("--audit-unit", type=Path)
    parser.add_argument("--owner-commit")
    parser.add_argument("--unchanged-review", type=Path)
    parser.add_argument("--reviewer-commit")
    args = parser.parse_args()
    if args.scan_tool:
        assert args.artifact_dir and args.needle_file
        raise SystemExit(scan_scoped_sources(args.artifact_dir, args.scan_tool, args.needle_file))
    if args.audit_unit:
        assert args.sha256 and args.owner_commit
        raise SystemExit(audit_unit_receipt(args.audit_unit, args.sha256, args.owner_commit))
    if args.unchanged_review:
        assert args.artifact_dir and args.owner_commit and args.reviewer_commit
        raise SystemExit(run_unchanged_review(args.artifact_dir, args.unchanged_review,
                                             args.reviewer_commit, args.owner_commit))
    assert args.replay and args.sha256
    if hashlib.sha256(args.replay.read_bytes()).hexdigest() != args.sha256:
        raise SystemExit("HSD replay caller byte pin mismatch")
    print(json.dumps(replay_record(json.loads(args.replay.read_text())), allow_nan=False))
