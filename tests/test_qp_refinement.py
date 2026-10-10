"""Same-model numerical A/B, explicit rejection and replayable CUDA sidecars."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from openopt.certificates import (
    REPRODUCIBLE_ARITHMETIC, check_qp_candidate, reproducible_checker_contract,
)
from openopt.qp import QPConfig, QPProblem, solve_qp
from openopt.refinement import RefinementConfig, refine_qp_candidate


def qp(P, q, A, l, u):
    return QPProblem(*[
        torch.tensor(value, dtype=torch.float64) for value in (P, q, A, l, u)
    ])


def mixed():
    # Known KKT pair: upper row 0, lower row 1, equality row 2,
    # inactive two-sided row 3 and a free structural zero row 4.
    P = np.diag([2., 4., 8.])
    A = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                  [1., 1., 1.], [0., 0., 0.]])
    expected_x, expected_y = np.array([1., -0.5, 0.25]), np.array([2., -1., 0.5, 0., 0.])
    p = qp(P, -P @ expected_x - A.T @ expected_y, A,
           [-np.inf, -0.5, 0.25, -10., -np.inf],
           [1., np.inf, 0.25, 10., np.inf])
    return p, torch.tensor(expected_x), torch.tensor(expected_y)


def perturbed(device="cpu"):
    p, x, y = mixed()
    return p, (x + torch.tensor([2e-6, -3e-6, 1e-6])).to(device), (
        y + torch.tensor([3e-6, -2e-6, 1e-6, 0., 0.])
    ).to(device)


def frozen_problem(p):
    def encode(value):
        if value == float("inf"):
            return "Infinity"
        if value == -float("inf"):
            return "-Infinity"
        return value

    return {
        name: [[encode(x) for x in row] for row in value.cpu().double().tolist()]
        if value.ndim == 2 else [encode(x) for x in value.cpu().double().tolist()]
        for name in ("P", "q", "A", "l", "u") for value in [getattr(p, name)]
    }


def thaw_problem(data):
    p = qp(*[
        [[float(x) for x in row] for row in data[name]]
        if name in ("P", "A") else [float(x) for x in data[name]]
        for name in ("P", "q", "A", "l", "u")
    ])
    return replace(p, A=p.A.reshape(-1, p.q.numel()))


def write_sidecar(p, x, y, config, result, tmp_path, request, name, require_mutations=True):
    root = Path(__file__).resolve().parents[1]
    p_data = frozen_problem(p)
    frozen = thaw_problem(p_data)
    baseline = check_qp_candidate(
        frozen, x.cpu().double(), y.cpu().double(),
        atol=config.check_atol, rtol=config.check_rtol,
        arithmetic=config.check_arithmetic,
    )
    final = check_qp_candidate(
        frozen, result.x.cpu().double(), result.y.cpu().double(),
        atol=config.check_atol, rtol=config.check_rtol,
        arithmetic=config.check_arithmetic,
    )
    assert baseline.to_dict() == result.baseline_evidence.to_dict()
    assert final.to_dict() == result.evidence.to_dict()
    wrong_x = check_qp_candidate(
        frozen, result.x.cpu().double() + 1., result.y.cpu().double(),
        atol=config.check_atol, rtol=config.check_rtol,
        arithmetic=config.check_arithmetic,
    )
    wrong_y = check_qp_candidate(
        frozen, result.x.cpu().double(), -result.y.cpu().double(),
        atol=config.check_atol, rtol=config.check_rtol,
        arithmetic=config.check_arithmetic,
    )
    wrong_model = check_qp_candidate(
        replace(frozen, q=frozen.q + 1.), result.x.cpu().double(), result.y.cpu().double(),
        atol=config.check_atol, rtol=config.check_rtol,
        arithmetic=config.check_arithmetic,
    )
    if require_mutations:
        assert not wrong_x.passed and not wrong_y.passed and not wrong_model.passed
    cfg = asdict(config)
    cfg["dtype"], cfg["factor_dtype"] = str(config.dtype), str(config.factor_dtype)
    if config.check_arithmetic == "numpy":
        cfg.pop("check_arithmetic")
    paths = [
        "src/openopt/refinement.py", "src/openopt/qp.py", "src/openopt/certificates.py",
        "tests/test_qp_refinement.py", "docs/research/refinement.md",
    ]
    report = {
        "schema_version": 1, "kind": "qp_refinement_original_coordinate_ab",
        "verdict": "PASS", "case": name, "problem": p_data, "config": cfg,
        "candidate_accepted": result.accepted,
        "scope": "numerical refinement test, including explicit negative outcomes",
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "device": str(result.x.device), "dtype": str(result.x.dtype),
        "gpu_name": torch.cuda.get_device_name(result.x.device) if result.x.is_cuda else None,
        "baseline": {"x": x.cpu().double().tolist(), "y": y.cpu().double().tolist(),
                     "evidence": baseline.to_dict()},
        "refined": {"x": result.x.cpu().double().tolist(), "y": result.y.cpu().double().tolist(),
                    "status": result.status, "reason": result.reason,
                    "evidence": final.to_dict(), "diagnostics": result.diagnostics},
        "negative_controls": {"wrong_x": wrong_x.to_dict(), "wrong_y": wrong_y.to_dict(),
                              "wrong_model": wrong_model.to_dict()},
        "source_sha256": {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                          for path in paths},
        "qualification": "same-model numerical accuracy A/B; no exact proof or speed claim",
    }
    if config.check_arithmetic == REPRODUCIBLE_ARITHMETIC:
        report["schema_version"] = 2
        report["checker_contract"] = reproducible_checker_contract()
        report["checker_runtime"] = checker_runtime()
    artifact = tmp_path / "refinement-acceptance.json"
    artifact.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(artifact.read_text()) == report
    request.node.user_properties.extend([
        ("refinement_acceptance_path", str(artifact)),
        ("refinement_acceptance_sha256", hashlib.sha256(artifact.read_bytes()).hexdigest()),
    ])


def checker_runtime():
    """Actual producer/worker runtime provenance, not a kernel-dispatch claim."""
    return {
        "python": platform.python_version(), "implementation": sys.implementation.name,
        "executable": sys.executable, "machine": platform.machine(),
        "numpy": np.__version__, "numpy_origin": np.__file__,
        "numpy_config": np.show_config(mode="dicts"),
        "torch": torch.__version__, "rounding": "nearest ties-even verified by checker",
        "environment": {name: os.environ.get(name) for name in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES", "CUDA_VISIBLE_DEVICES",
        )},
    }


def check_sidecar(path, sha256, source_root):
    """Independent disk replay, requiring caller-pinned bytes and current sources."""
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha256:
        raise ValueError("sidecar pin mismatch")
    data = json.loads(raw)
    if data["schema_version"] not in (1, 2) or data["kind"] != "qp_refinement_original_coordinate_ab":
        raise ValueError("unsupported refinement sidecar")
    expected_sources = {
        "src/openopt/refinement.py", "src/openopt/qp.py", "src/openopt/certificates.py",
        "tests/test_qp_refinement.py", "docs/research/refinement.md",
    }
    if set(data["source_sha256"]) != expected_sources:
        raise ValueError("source membership mismatch")
    for name, pin in data["source_sha256"].items():
        if hashlib.sha256((source_root / name).read_bytes()).hexdigest() != pin:
            raise ValueError(f"current source pin mismatch: {name}")
    loaded_checker = Path(sys.modules[check_qp_candidate.__module__].__file__)
    if hashlib.sha256(loaded_checker.read_bytes()).hexdigest() != data["source_sha256"]["src/openopt/certificates.py"]:
        raise ValueError("loaded independent checker source mismatch")
    p = thaw_problem(data["problem"])
    cfg = data["config"]
    tolerances = {"atol": cfg["check_atol"], "rtol": cfg["check_rtol"]}
    if data["schema_version"] == 2:
        if (cfg.get("check_arithmetic") != REPRODUCIBLE_ARITHMETIC
                or data.get("checker_contract") != reproducible_checker_contract()
                or not isinstance(data.get("checker_runtime"), dict)
                or data["refined"]["diagnostics"].get("check_arithmetic") != REPRODUCIBLE_ARITHMETIC):
            raise ValueError("versioned checker contract mismatch")
        tolerances["arithmetic"] = REPRODUCIBLE_ARITHMETIC
    elif "check_arithmetic" in cfg or "checker_contract" in data:
        raise ValueError("legacy checker schema cannot select new arithmetic")
    baseline, refined = data["baseline"], data["refined"]
    before = check_qp_candidate(p, baseline["x"], baseline["y"], **tolerances)
    after = check_qp_candidate(p, refined["x"], refined["y"], **tolerances)
    def exact(left, right):
        return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
            right, sort_keys=True, allow_nan=False)

    if not exact(before.to_dict(), baseline["evidence"]) or not exact(after.to_dict(), refined["evidence"]):
        raise ValueError("original checker replay mismatch")
    accepted = refined["status"] == "accepted"
    if data["candidate_accepted"] is not accepted:
        raise ValueError("acceptance status mismatch")
    if accepted:
        if not after.passed:
            raise ValueError("accepted candidate failed independent check")
        for name in ("primal_feasibility", "stationarity", "bound_sign", "complementarity", "duality_gap"):
            a, b = before.metrics[name], after.metrics[name]
            if b is None or (a is not None and abs(b) > abs(a)):
                raise ValueError(f"accepted candidate worsened {name}")
    elif refined["status"] not in ("rejected", "unknown"):
        raise ValueError("unrecognized status")
    elif not exact(refined["x"], baseline["x"]) or not exact(refined["y"], baseline["y"]):
        raise ValueError("rejection did not retain baseline")
    xx, yy = np.asarray(refined["x"]), np.asarray(refined["y"])
    negative = {
        "wrong_x": check_qp_candidate(p, xx + 1., yy, **tolerances),
        "wrong_y": check_qp_candidate(p, xx, -yy, **tolerances),
        "wrong_model": check_qp_candidate(replace(p, q=p.q + 1.), xx, yy, **tolerances),
    }
    if not exact({name: value.to_dict() for name, value in negative.items()}, data["negative_controls"]):
        raise ValueError("negative checker replay mismatch")
    if accepted and any(value.passed for value in negative.values()):
        raise ValueError("negative control unexpectedly passed")
    return {
        "verdict": "PASS", "candidate_accepted": accepted, "candidate_status": refined["status"],
        "source_membership_count": len(expected_sources), "sidecar_sha256": sha256,
        "baseline_evidence": before.to_dict(), "refined_evidence": after.to_dict(),
        "qualification": "current-source numerical sidecar replay; no speed or exact proof",
    }


def test_original_coordinate_accuracy_ab_and_three_negative_controls(tmp_path, request):
    p, x, y = perturbed()
    copies = [t.clone() for t in (p.P, p.q, p.A, p.l, p.u, x, y)]
    config = RefinementConfig()
    result = refine_qp_candidate(p, x, y, config)
    _, expected_x, expected_y = mixed()
    assert result.accepted and result.evidence.passed
    assert not result.baseline_evidence.passed
    assert torch.equal(result.x, expected_x) and torch.equal(result.y, expected_y)
    assert result.diagnostics["active_rows"] == [0, 1, 2]
    assert result.diagnostics["active_sides"] == ["upper", "lower", "equality"]
    assert all(torch.equal(a, b) for a, b in zip(copies, (p.P, p.q, p.A, p.l, p.u, x, y)))
    assert result.x.data_ptr() != x.data_ptr() and result.y.data_ptr() != y.data_ptr()
    write_sidecar(p, x, y, config, result, tmp_path, request, "mixed_perturbed")


@pytest.mark.parametrize("mutation", [
    "none", "pin", "x", "y", "model", "threshold", "source", "membership",
    "signed_zero", "integer_float",
])
def test_independent_disk_replay_sees_tampering(mutation, tmp_path, request):
    p, x, y = perturbed()
    config = RefinementConfig()
    result = refine_qp_candidate(p, x, y, config)
    write_sidecar(p, x, y, config, result, tmp_path, request, "replay_control")
    path = tmp_path / "refinement-acceptance.json"
    data = json.loads(path.read_text())
    if mutation == "x":
        data["refined"]["x"][0] += 1.
    elif mutation == "y":
        data["refined"]["y"][0] *= -1.
    elif mutation == "model":
        data["problem"]["q"][0] += 1.
    elif mutation == "threshold":
        data["config"]["check_atol"] *= 2.
    elif mutation == "source":
        data["source_sha256"]["src/openopt/refinement.py"] = "0" * 64
    elif mutation == "membership":
        data["source_sha256"].pop("src/openopt/qp.py")
    elif mutation == "signed_zero":
        assert data["refined"]["evidence"]["metrics"]["bound_sign"] == 0.
        data["refined"]["evidence"]["metrics"]["bound_sign"] = -0.
    elif mutation == "integer_float":
        data["refined"]["evidence"]["metrics"]["bound_sign"] = 0
    mutated = tmp_path / "replay-control.json"
    mutated.write_text(json.dumps(data, allow_nan=False), encoding="utf-8")
    pin = "0" * 64 if mutation == "pin" else hashlib.sha256(mutated.read_bytes()).hexdigest()
    root = Path(__file__).resolve().parents[1]
    if mutation == "none":
        report = check_sidecar(mutated, pin, root)
        assert report["verdict"] == "PASS" and report["candidate_accepted"]
    else:
        with pytest.raises(ValueError, match="mismatch|failed"):
            check_sidecar(mutated, pin, root)


def test_default_path_and_baseline_are_bitwise_unchanged():
    p, _, _ = mixed()
    config = QPConfig(eps_abs=1e-5, eps_rel=1e-5)
    before = solve_qp(p, config)
    x, y = before.x.clone(), before.y.clone()
    result = refine_qp_candidate(p, before.x, before.y, RefinementConfig(active_atol=1e-3))
    after = solve_qp(p, config)
    assert result.accepted
    assert torch.equal(before.x, x) and torch.equal(before.y, y)
    assert torch.equal(before.x, after.x) and torch.equal(before.y, after.y)
    assert before.status == after.status and before.iterations == after.iterations


def test_actual_admm_candidate_same_problem_polishing(tmp_path, request):
    p, expected_x, expected_y = mixed()
    baseline = solve_qp(p, QPConfig(eps_abs=1e-5, eps_rel=1e-5, check_interval=1))
    assert baseline.status == "solved"
    config = RefinementConfig(active_atol=1e-3)
    result = refine_qp_candidate(p, baseline.x, baseline.y, config)
    assert result.accepted and result.evidence.passed
    assert torch.equal(result.x, expected_x) and torch.equal(result.y, expected_y)
    assert result.evidence.metrics["stationarity"] < result.baseline_evidence.metrics["stationarity"]
    write_sidecar(p, baseline.x, baseline.y, config, result, tmp_path, request, "admm_mixed")


def test_no_third_party_or_qp_solver_fallback(monkeypatch):
    import openopt.qp as module

    def forbidden(*args, **kwargs):
        raise AssertionError("no ordinary/third-party solver is permitted")

    monkeypatch.setattr(module, "solve_qp", forbidden)
    monkeypatch.setattr(module, "prepare_qp", forbidden)
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(np.linalg, "lstsq", forbidden)
    p, x, y = perturbed()
    assert refine_qp_candidate(p, x, y).accepted


def test_wrong_active_set_is_explicitly_rejected_and_baseline_retained():
    p = qp([[1.]], [-2.], [[1.]], [-np.inf], [1.])
    x, y = torch.tensor([0.5], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64)
    result = refine_qp_candidate(p, x, y)
    assert result.status == "rejected" and result.reason == "original_check_failed"
    assert torch.equal(result.x, x) and torch.equal(result.y, y)
    assert result.attempted_x.item() == 2.
    assert not result.attempted_evidence.checks["primal_feasibility"]
    assert result.diagnostics["solver_fallback"] is False


@pytest.mark.parametrize("kind", ["duplicate_equalities", "null_curvature", "near_singular"])
def test_singular_or_unresolved_kkt_is_unknown_not_fake_success(kind):
    if kind == "duplicate_equalities":
        p = qp([[1.]], [0.], [[1.], [1.]], [0., 0.], [0., 0.])
    elif kind == "null_curvature":
        p = qp([[0.]], [0.], np.empty((0, 1)), [], [])
    else:
        p = qp(np.diag([1., 1e-16]), [0., 0.], np.empty((0, 2)), [], [])
    x, y = torch.zeros_like(p.q), torch.zeros_like(p.l)
    result = refine_qp_candidate(p, x, y)
    assert result.status == "unknown" and result.reason == "singular_or_unresolved_kkt"
    assert not result.accepted and result.attempted_x is None
    assert torch.equal(result.x, x) and torch.equal(result.y, y)


def test_ambiguous_two_sided_near_bounds_rejected():
    p = qp([[1.]], [0.], [[1.]], [-1e-6], [1e-6])
    x, y = torch.zeros(1, dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    result = refine_qp_candidate(p, x, y)
    assert result.status == "rejected" and result.reason == "ambiguous_active_bounds"
    assert result.diagnostics["ambiguous_rows"] == [0]
    assert torch.equal(result.x, x)


def test_wrong_active_multiplier_sign_not_promoted():
    p = qp([[1.]], [-0.5], [[1.]], [-np.inf], [1.])
    x, y = torch.tensor([1.], dtype=torch.float64), torch.zeros(1, dtype=torch.float64)
    result = refine_qp_candidate(p, x, y)
    assert result.status == "rejected"
    assert result.attempted_y.item() == -0.5
    assert not result.attempted_evidence.passed
    assert result.attempted_evidence.unavailable["dual_objective"]
    assert torch.equal(result.x, x) and torch.equal(result.y, y)


def test_passed_candidate_that_worsens_accuracy_is_rejected():
    p = qp([[1.]], [-1.00000003], np.empty((0, 1)), [], [])
    x, y = torch.tensor([1.00000003], dtype=torch.float64), torch.empty(0, dtype=torch.float64)
    config = RefinementConfig(dtype=torch.float32, factor_dtype=torch.float32, check_atol=1e-6)
    result = refine_qp_candidate(p, x, y, config)
    assert result.baseline_evidence.passed and result.attempted_evidence.passed
    assert result.status == "rejected" and result.reason == "accuracy_worsened"
    assert result.x.dtype == x.dtype and torch.equal(result.x, x)
    assert result.diagnostics["iterations"][0]["worsened_metrics"] == ["stationarity", "duality_gap"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unconstrained_and_lp_kkt_scope(dtype):
    config = RefinementConfig(dtype=dtype, factor_dtype=dtype, check_atol=1e-6)
    for p, x, y in [
        (qp([[2.]], [-4.], np.empty((0, 1)), [], []),
         torch.tensor([2.001], dtype=dtype), torch.empty(0, dtype=dtype)),
        (qp([[0.]], [-1.], [[1.]], [0.], [1.]),
         torch.tensor([1.000001], dtype=dtype), torch.tensor([1.000001], dtype=dtype)),
    ]:
        result = refine_qp_candidate(p, x, y, config)
        assert result.accepted and result.evidence.passed
        assert result.x.dtype == result.y.dtype == dtype


def test_residual_correction_float32_factor_float64_residual(tmp_path, request):
    # Dense non-dyadic equality system: coarse factor must be corrected using
    # the same original model, not a supplied optimal witness or new QP.
    p = qp([[2.1, 0.3], [0.3, 1.7]], [-0.6, 0.8],
           [[0.7, 1.2]], [0.35], [0.35])
    baseline = solve_qp(p, QPConfig(eps_abs=1e-5, eps_rel=1e-5, check_interval=1))
    config = RefinementConfig(factor_dtype=torch.float32, check_atol=1e-10, check_rtol=1e-10)
    result = refine_qp_candidate(p, baseline.x, baseline.y, config)
    assert result.accepted and result.evidence.passed
    trace = result.diagnostics["iterations"]
    assert len(trace) >= 2 and trace[1]["linear_residual"] < trace[0]["linear_residual"]
    assert trace[0]["original_check_passed"] is False
    assert not result.baseline_evidence.passed
    assert result.x.dtype == torch.float64 and config.factor_dtype == torch.float32
    # Nonzero multiplier and stationarity provide three meaningful controls.
    write_sidecar(p, baseline.x, baseline.y, config, result, tmp_path, request, "mixed_precision_correction")


def test_step_limit_without_accurate_correction_retains_original(tmp_path, request):
    p = qp([[2.1, 0.3], [0.3, 1.7]], [-0.6, 0.8],
           [[0.7, 1.2]], [0.35], [0.35])
    baseline = solve_qp(p, QPConfig(eps_abs=1e-5, eps_rel=1e-5, check_interval=1))
    config = RefinementConfig(
        factor_dtype=torch.float32, max_refinement_steps=0,
        check_atol=1e-10, check_rtol=1e-10,
    )
    result = refine_qp_candidate(p, baseline.x, baseline.y, config)
    assert not result.accepted and result.status == "rejected"
    assert result.diagnostics["stop_reason"] == "refinement_step_limit"
    assert len(result.diagnostics["iterations"]) == 1
    assert not result.attempted_evidence.passed
    assert torch.equal(result.x, baseline.x) and torch.equal(result.y, baseline.y)
    write_sidecar(p, baseline.x, baseline.y, config, result, tmp_path, request, "zero_correction_limit")


@pytest.mark.parametrize("kind", ["bad_rhs", "nonfinite_solution"])
def test_corrupted_linear_solve_cannot_fake_acceptance(monkeypatch, kind):
    real = torch.linalg.lu_solve

    def corrupt(*args, **kwargs):
        value = real(*args, **kwargs)
        return value + 1 if kind == "bad_rhs" else value * float("inf")

    monkeypatch.setattr(torch.linalg, "lu_solve", corrupt)
    p, x, y = perturbed()
    result = refine_qp_candidate(p, x, y)
    assert not result.accepted
    assert result.reason == ("original_check_failed" if kind == "bad_rhs" else "nonfinite_linear_solution")
    assert torch.equal(result.x, x) and torch.equal(result.y, y)


def test_failed_original_gap_is_unknown_without_regularized_inverse():
    p = qp(np.diag([1., 1e-16]), [0., 0.], np.eye(2), [0., 0.], [0., 0.])
    x, y = torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)
    result = refine_qp_candidate(p, x, y)
    # Equality constraints resolve the KKT system, but the independent original
    # dual-objective checker deliberately declines this tiny PSD eigenvalue.
    assert result.status == "unknown" and result.reason == "original_check_unavailable"
    assert result.attempted_x is not None and not result.attempted_evidence.passed
    assert result.attempted_evidence.unavailable["dual_objective"]
    assert torch.equal(result.x, x) and torch.equal(result.y, y)


@pytest.mark.parametrize("name", ["q", "P"])
@pytest.mark.parametrize("value", [1e100, 1e-100])
def test_unrepresentable_model_cast_raises(name, value):
    p = qp([[1.]], [-1.], np.empty((0, 1)), [], [])
    setattr_target = getattr(p, name)
    setattr_target.reshape(-1)[0] = value
    x, y = torch.zeros(1, dtype=torch.float64), torch.empty(0, dtype=torch.float64)
    with pytest.raises(ValueError, match="nonfinite|underflow"):
        refine_qp_candidate(p, x, y, RefinementConfig(dtype=torch.float32, factor_dtype=torch.float32))


@pytest.mark.parametrize("kwargs", [
    {"dtype": torch.float16}, {"factor_dtype": torch.float16},
    {"dtype": torch.float32}, {"device": "meta"},
    {"max_refinement_steps": -1}, {"max_refinement_steps": True},
    {"max_refinement_steps": 1.5}, {"active_atol": -1}, {"active_atol": float("nan")},
    {"active_rtol": 1}, {"active_rtol": True}, {"active_rtol": float("inf")},
    {"check_atol": 0}, {"check_atol": True}, {"check_rtol": 1}, {"check_rtol": -1},
])
def test_invalid_config_raises(kwargs):
    p, x, y = perturbed()
    with pytest.raises(ValueError):
        refine_qp_candidate(p, x, y, RefinementConfig(**kwargs))


@pytest.mark.parametrize("mutation", [
    "P_dims", "A_dims", "x_dims", "y_dims", "nan_x", "inf_y",
    "nan_P", "indefinite", "asymmetric", "bounds", "complex", "sparse", "non_tensor",
])
def test_malformed_inputs_raise_not_rollback_success(mutation):
    p, x, y = perturbed()
    if mutation == "P_dims":
        p = replace(p, P=p.P[:1])
    elif mutation == "A_dims":
        p = replace(p, A=p.A[:, :1])
    elif mutation == "x_dims":
        x = x[:1]
    elif mutation == "y_dims":
        y = y[None]
    elif mutation == "nan_x":
        x[0] = float("nan")
    elif mutation == "inf_y":
        y[0] = float("inf")
    elif mutation == "nan_P":
        p.P[0, 0] = float("nan")
    elif mutation == "indefinite":
        p = replace(p, P=-p.P)
    elif mutation == "asymmetric":
        p.P[0, 1] = 1.
    elif mutation == "bounds":
        p.l[0] = 2.
    elif mutation == "complex":
        x = x.to(torch.complex128)
    elif mutation == "sparse":
        p = replace(p, A=p.A.to_sparse())
    else:
        x = [1.]
    with pytest.raises((ValueError, TypeError)):
        refine_qp_candidate(p, x, y)


def test_explicit_cuda_unavailable_never_falls_back(monkeypatch):
    p, x, y = perturbed()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA"):
        refine_qp_candidate(p, x, y, RefinementConfig(device="cuda:0"))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; refinement GPU acceptance remains UNVERIFIED")
@pytest.mark.parametrize("dtype,factor_dtype", [
    (torch.float64, torch.float64), (torch.float64, torch.float32),
    (torch.float32, torch.float32),
])
def test_cuda_refinement_frozen_model_accuracy_ab(dtype, factor_dtype, tmp_path, request):
    p, x, y = perturbed("cuda:0")
    x, y = x.to(dtype), y.to(dtype)
    gpu_p = QPProblem(*[getattr(p, name).to("cuda:0") for name in ("P", "q", "A", "l", "u")])
    copies = [t.clone() for t in (gpu_p.P, gpu_p.q, gpu_p.A, gpu_p.l, gpu_p.u, x, y)]
    config = RefinementConfig(device="cuda:0", dtype=dtype, factor_dtype=factor_dtype,
                              check_arithmetic=REPRODUCIBLE_ARITHMETIC)
    result = refine_qp_candidate(gpu_p, x, y, config)
    torch.cuda.synchronize("cuda:0")
    assert result.accepted and result.evidence.passed
    assert result.x.device == result.y.device == torch.device("cuda:0")
    assert result.x.dtype == result.y.dtype == dtype
    assert all(torch.equal(a, b) for a, b in zip(
        copies, (gpu_p.P, gpu_p.q, gpu_p.A, gpu_p.l, gpu_p.u, x, y),
    ))
    write_sidecar(gpu_p, x, y, config, result, tmp_path, request, "cuda_mixed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; mixed-precision correction is UNVERIFIED")
def test_cuda_refinement_actual_residual_correction(tmp_path, request):
    p = qp([[2.1, 0.3], [0.3, 1.7]], [-0.6, 0.8],
           [[0.7, 1.2]], [0.35], [0.35])
    baseline = solve_qp(p, QPConfig(device="cuda:0", eps_abs=1e-5, eps_rel=1e-5, check_interval=1))
    config = RefinementConfig(
        device="cuda:0", factor_dtype=torch.float32, check_atol=1e-10, check_rtol=1e-10,
        check_arithmetic=REPRODUCIBLE_ARITHMETIC,
    )
    result = refine_qp_candidate(p, baseline.x, baseline.y, config)
    torch.cuda.synchronize("cuda:0")
    trace = result.diagnostics["iterations"]
    assert result.accepted and result.evidence.passed
    assert not result.baseline_evidence.passed
    assert len(trace) >= 2 and trace[1]["linear_residual"] < trace[0]["linear_residual"]
    assert not trace[0]["original_check_passed"]
    assert result.x.device == result.y.device == torch.device("cuda:0")
    assert result.x.dtype == result.y.dtype == torch.float64
    write_sidecar(p, baseline.x, baseline.y, config, result, tmp_path, request, "cuda_correction")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable; GPU rejection controls remain UNVERIFIED")
@pytest.mark.parametrize("kind", ["wrong_active_set", "singular_kkt", "wrong_sign"])
def test_cuda_refinement_negative_controls(kind, tmp_path, request):
    if kind == "wrong_active_set":
        p = qp([[1.]], [-2.], [[1.]], [-np.inf], [1.])
        x, y = torch.tensor([0.5]), torch.tensor([0.5])
    elif kind == "singular_kkt":
        p = qp([[1.]], [0.], [[1.], [1.]], [0., 0.], [0., 0.])
        x, y = torch.zeros(1), torch.zeros(2)
    else:
        p = qp([[1.]], [-0.5], [[1.]], [-np.inf], [1.])
        x, y = torch.ones(1), torch.zeros(1)
    x, y = x.cuda().double(), y.cuda().double()
    config = RefinementConfig(device="cuda:0", check_arithmetic=REPRODUCIBLE_ARITHMETIC)
    result = refine_qp_candidate(p, x, y, config)
    assert not result.accepted and result.status in ("rejected", "unknown")
    assert torch.equal(result.x, x) and torch.equal(result.y, y)
    assert result.x.is_cuda and result.y.is_cuda
    write_sidecar(p, x, y, config, result, tmp_path, request, kind, require_mutations=False)


@pytest.mark.parametrize("kind", ["mixed", "correction", "wrong_active_set", "singular_kkt", "wrong_sign"])
def test_reproducible_refinement_producer_cpu_exact_sidecar(kind, tmp_path, request):
    if kind == "mixed":
        p, x, y = perturbed()
        config = RefinementConfig(check_arithmetic=REPRODUCIBLE_ARITHMETIC)
    elif kind == "correction":
        p = qp([[2.1, .3], [.3, 1.7]], [-.6, .8], [[.7, 1.2]], [.35], [.35])
        baseline = solve_qp(p, QPConfig(eps_abs=1e-5, eps_rel=1e-5, check_interval=1))
        x, y = baseline.x, baseline.y
        config = RefinementConfig(factor_dtype=torch.float32, check_atol=1e-10,
                                  check_rtol=1e-10, check_arithmetic=REPRODUCIBLE_ARITHMETIC)
    else:
        if kind == "wrong_active_set":
            p, x, y = qp([[1.]], [-2.], [[1.]], [-np.inf], [1.]), torch.tensor([.5]), torch.tensor([.5])
        elif kind == "singular_kkt":
            p, x, y = qp([[1.]], [0.], [[1.], [1.]], [0., 0.], [0., 0.]), torch.zeros(1), torch.zeros(2)
        else:
            p, x, y = qp([[1.]], [-.5], [[1.]], [-np.inf], [1.]), torch.ones(1), torch.zeros(1)
        x, y = x.double(), y.double()
        config = RefinementConfig(check_arithmetic=REPRODUCIBLE_ARITHMETIC)
    copies = [v.clone() for v in (p.P, p.q, p.A, p.l, p.u, x, y)]
    result = refine_qp_candidate(p, x, y, config)
    positive = kind in ("mixed", "correction")
    assert result.accepted == positive and result.diagnostics["solver_fallback"] is False
    if positive:
        assert result.evidence.passed
        assert not any(result.diagnostics["iterations"][result.diagnostics["accepted_step"]]["worsened_metrics"])
    else:
        assert torch.equal(result.x, x) and torch.equal(result.y, y)
    assert all(torch.equal(a, b) for a, b in zip(copies, (p.P, p.q, p.A, p.l, p.u, x, y)))
    write_sidecar(p, x, y, config, result, tmp_path, request, kind, require_mutations=positive)
    path = tmp_path / "refinement-acceptance.json"
    saved = json.loads(path.read_text())
    assert saved["schema_version"] == 2 and saved["checker_runtime"]["numpy"] == np.__version__
    replay = check_sidecar(path, hashlib.sha256(path.read_bytes()).hexdigest(),
                           Path(__file__).resolve().parents[1])
    assert replay["candidate_accepted"] == positive
    assert replay["refined_evidence"] == result.evidence.to_dict()


def test_reproducible_refinement_default_unchanged_and_device_strict(monkeypatch):
    p, x, y = perturbed()
    before = refine_qp_candidate(p, x, y)
    refine_qp_candidate(p, x, y, RefinementConfig(check_arithmetic=REPRODUCIBLE_ARITHMETIC))
    after = refine_qp_candidate(p, x, y)
    assert torch.equal(before.x, after.x) and torch.equal(before.y, after.y)
    assert before.evidence.to_dict() == after.evidence.to_dict()
    assert before.diagnostics == after.diagnostics
    assert "check_arithmetic" not in before.diagnostics
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="explicit CUDA"):
        refine_qp_candidate(p, x, y, RefinementConfig(device="cuda:0",
                                                     check_arithmetic=REPRODUCIBLE_ARITHMETIC))
    with pytest.raises(ValueError, match="supports CPU or CUDA"):
        refine_qp_candidate(p, x, y, RefinementConfig(device="meta",
                                                     check_arithmetic=REPRODUCIBLE_ARITHMETIC))
    with pytest.raises(ValueError, match="arithmetic"):
        refine_qp_candidate(p, x, y, RefinementConfig(check_arithmetic="fallback"))


def test_reproducible_passed_but_worsened_candidate_retains_original():
    p = qp([[1.]], [-1.00000003], np.empty((0, 1)), [], [])
    x, y = torch.tensor([1.00000003], dtype=torch.float64), torch.empty(0, dtype=torch.float64)
    config = RefinementConfig(dtype=torch.float32, factor_dtype=torch.float32, check_atol=1e-6,
                              check_arithmetic=REPRODUCIBLE_ARITHMETIC)
    result = refine_qp_candidate(p, x, y, config)
    assert result.baseline_evidence.passed and result.attempted_evidence.passed
    assert not result.accepted and result.reason == "accuracy_worsened"
    assert torch.equal(result.x, x) and torch.equal(result.y, y)
    assert result.diagnostics["iterations"][0]["worsened_metrics"]


def test_reproducible_seven_cpu_producer_sidecars_cross_process_exact(tmp_path, request):
    import subprocess
    from scripts.accept_extension_cuda import expected_cases

    root = Path(__file__).resolve().parents[1]
    definitions = expected_cases(root, "refinement")
    paths = []
    for index, (identity, definition) in enumerate(definitions.items()):
        parameters = definition["parameters"]
        if "frozen_model" in definition["function"]:
            p, x, y = perturbed()
            dtype = getattr(torch, parameters["dtype"]["dtype"].split(".")[-1])
            factor = getattr(torch, parameters["factor_dtype"]["dtype"].split(".")[-1])
            x, y = x.to(dtype), y.to(dtype)
            config = RefinementConfig(dtype=dtype, factor_dtype=factor,
                                      check_arithmetic=REPRODUCIBLE_ARITHMETIC)
            positive = True
        elif "residual_correction" in definition["function"]:
            p = qp([[2.1, .3], [.3, 1.7]], [-.6, .8], [[.7, 1.2]], [.35], [.35])
            baseline = solve_qp(p, QPConfig(eps_abs=1e-5, eps_rel=1e-5, check_interval=1))
            x, y = baseline.x, baseline.y
            config = RefinementConfig(factor_dtype=torch.float32, check_atol=1e-10,
                                      check_rtol=1e-10, check_arithmetic=REPRODUCIBLE_ARITHMETIC)
            positive = True
        else:
            kind = parameters["kind"]
            if kind == "wrong_active_set":
                p, x, y = qp([[1.]], [-2.], [[1.]], [-np.inf], [1.]), torch.tensor([.5]), torch.tensor([.5])
            elif kind == "singular_kkt":
                p, x, y = qp([[1.]], [0.], [[1.], [1.]], [0., 0.], [0., 0.]), torch.zeros(1), torch.zeros(2)
            else:
                p, x, y = qp([[1.]], [-.5], [[1.]], [-np.inf], [1.]), torch.ones(1), torch.zeros(1)
            x, y = x.double(), y.double()
            config = RefinementConfig(check_arithmetic=REPRODUCIBLE_ARITHMETIC)
            positive = False
        result = refine_qp_candidate(p, x, y, config)
        assert result.accepted == positive
        directory = tmp_path / f"case-{index}"
        directory.mkdir()
        write_sidecar(p, x, y, config, result, directory, request,
                      parameters.get("kind", identity), require_mutations=positive)
        paths.append(directory / "refinement-acceptance.json")
    assert len(paths) == 7
    commands = []
    for path in paths:
        commands.extend(["--refinement-sidecar", str(path)])
    reports = []
    for index, arm in enumerate((
        {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
         "OPENBLAS_CORETYPE": "Haswell", "NPY_DISABLE_CPU_FEATURES": "AVX2,FMA3"},
        {"OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
         "OPENBLAS_CORETYPE": "Sandybridge", "NPY_DISABLE_CPU_FEATURES": ""},
    )):
        output = tmp_path / f"replay-{index}.json"
        completed = subprocess.run(
            [sys.executable, "-B", str(root / "tests/test_qp_certificates.py"),
             "--reproducible-worker-output", str(output), *commands],
            env={**os.environ, **arm, "CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1"},
            capture_output=True, check=False,
        )
        (tmp_path / f"replay-{index}.log").write_bytes(completed.stdout + completed.stderr)
        assert completed.returncode == 0, completed.stderr.decode()
        report = json.loads(output.read_text())
        assert len(report["evidence"]) == 15
        assert sum(item["candidate_accepted"] for item in report["evidence"][8:]) == 4
        assert all(report["runtime"]["environment"][key] == value for key, value in arm.items())
        reports.append(report)
    assert reports[0]["runtime"]["pid"] != reports[1]["runtime"]["pid"]
    assert reports[0]["evidence"] == reports[1]["evidence"]
    assert reports[0]["evidence_sha256"] == reports[1]["evidence_sha256"]
    receipt = {
        "verdict": "PASS", "testcases": list(definitions), "arms": reports,
        "sidecars": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "qualification": "seven CPU producer candidates, no GPU execution/final seal",
    }
    (tmp_path / "seven-sidecar-cross-process.json").write_text(json.dumps(receipt, allow_nan=False))


def replay_main():
    """CPU-only disk audit entry for the parent; no test or solver is invoked."""
    import argparse

    parser = argparse.ArgumentParser(description=check_sidecar.__doc__)
    parser.add_argument("--check-sidecar", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    source_root = args.source_root.resolve()
    if output.is_relative_to(source_root):
        raise ValueError("replay output must be outside source root")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "checked_at": datetime.now(timezone.utc).isoformat()}
    exit_code = 0
    try:
        report.update(check_sidecar(args.check_sidecar, args.sha256, source_root))
    except Exception as error:
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        exit_code = 1
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"REFINEMENT_REPLAY verdict={report['verdict']} report={output}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(replay_main())
