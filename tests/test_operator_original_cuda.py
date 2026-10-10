"""Opt-in actual CUDA operator regression with original-model JSON sidecars.

CPU exercises are development only. CUDA opt-in fails, never skips/falls back,
when unavailable. The coordinator alone submits scripts/gpu_release_check.py.
Small Fraction checks are host numerical validation, not GPU acceleration or
general PSD/gap certification. Input source/artifact code is never executed.
"""

from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

import openopt.operators as engine

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("dense", "coo", "csr", "matrix_free")
MODELS = ("interior_qp", "equality_qp", "bounded_lp", "empty_rows_qp")
LOSSES = ("reversed", "lower_overflow", "upper_overflow", "nonzero_underflow")
SOURCE_PATHS = ("src/openopt/operators.py", "tests/test_operator_original_cuda.py",
                "scripts/gpu_release_check.py")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def sources():
    assert Path(engine.__file__).resolve() == ROOT / SOURCE_PATHS[0], "foreign operator source"
    return {name: sha((ROOT / name).read_bytes()) for name in SOURCE_PATHS}


def strict_cuda():
    if os.environ.get("OPENOPT_RUN_OPERATOR_CUDA") != "1":
        pytest.skip("operator CUDA opt-in disabled; CUDA qualification UNVERIFIED")
    return required_cuda_device()


def required_cuda_device():
    """No environment/skip branch: selecting the formal entry requests CUDA."""
    if not torch.cuda.is_available():
        raise RuntimeError("explicit operator CUDA requested but unavailable; no CPU fallback")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("formal operator producer requires exactly one CUDA device")
    if "RTX PRO 4500" not in torch.cuda.get_device_name(0):
        raise RuntimeError("formal operator producer requires RTX PRO 4500")
    return torch.device("cuda:0")


class Placement(TorchDispatchMode):
    """Observe real dispatcher tensor operands/results, reject off-device ops."""

    def __init__(self, device):
        super().__init__()
        self.device, self.operations, self.tensors = torch.device(device), Counter(), 0

    def check(self, value):
        for item in tree_flatten(value)[0]:
            if isinstance(item, torch.Tensor):
                assert item.device == self.device, ("off-device operator tensor", item.device, self.device)
                self.tensors += 1

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        self.check((args, kwargs))
        output = function(*args, **kwargs)
        self.check(output)
        self.operations[str(function)] += 1
        return output

    def report(self):
        assert self.tensors > 0 and self.operations, "empty device trace"
        return {"device": str(self.device), "tensors": self.tensors,
                "operations": dict(self.operations), "scope": "dispatcher placement, not remote attestation"}


def original_model(name):
    P, q = [[2., 0.], [0., 4.]], [-1., 2.]
    A, l, u = [[1., 0.], [0., 1.], [0., 0.]], [-1., -2., -1.], [2., 1., 1.]
    expected_x, expected_y = [.5, -.5], [0., 0., 0.]
    if name == "equality_qp":
        q, l, u = [-1., 2.25], [.25, -.5, -1.], [.25, -.5, 1.]
        expected_x, expected_y = [.25, -.5], [.5, -.25, 0.]
    elif name == "bounded_lp":
        P, q, l, u = [[0., 0.], [0., 0.]], [-1., 1.], [0., -1., -1.], [1., 0., 1.]
        expected_x, expected_y = [1., -1.], [1., -1., 0.]
    elif name == "empty_rows_qp":
        A, l, u, expected_y = [], [], [], []
    else:
        assert name == "interior_qp"
    return {"name": name, "n": 2, "m": len(A), "P": P, "q": q, "A": A, "l": l, "u": u,
            "analytic_x": expected_x, "analytic_y": expected_y}


def negative_model(loss):
    model = {"name": loss, "n": 1, "m": 2, "P": [[1.]], "q": [-1e9],
             "A": [[1.], [0.]], "l": [1e9, 1.], "u": [1e9, 2.]}
    if loss == "reversed":
        model["l"][1], model["u"][1] = 1. + 2. ** -40, 1.
    elif loss == "lower_overflow":
        model["l"][1] = -(2. ** 400)
    elif loss == "upper_overflow":
        model["u"][1] = 2. ** 400
    elif loss == "nonzero_underflow":
        model["l"][1], model["u"][1] = 2. ** -400, 2. ** -399
    else:
        assert loss == "zero_row"
    return model


def fraction(value):
    assert type(value) in (float, int) and math.isfinite(value), "nonfinite Fraction input"
    return Fraction.from_float(float(value))


def matvec(matrix, vector):
    return [sum((fraction(a) * v for a, v in zip(row, vector)), Fraction(0)) for row in matrix]


def objective(model, x):
    Px = matvec(model["P"], x)
    return sum((v * p / 2 + fraction(q) * v for v, p, q in zip(x, Px, model["q"])), Fraction(0))


def original_check(model, x, y, *, tolerance, require_certificate=True):
    """Exact represented-rational arithmetic; tolerance acceptance is approximate.

    Diagonal nonnegative P is analytic here, not inferred from solver probes.
    The independent dual objective uses each diagonal range and signed bound
    support. A negative signed gap is rejected, never absolute-valued/clipped.
    Numerical KKT-only mode reports unavailable/negative certificates rather
    than certifying a nearly feasible point or silently projecting duals.
    """
    assert len(x) == model["n"] and len(y) == model["m"]
    xf, yf = [fraction(v) for v in x], [fraction(v) for v in y]
    P, A = model["P"], model["A"]
    assert all(P[i][j] == 0 for i in range(len(P)) for j in range(len(P)) if i != j)
    assert all(fraction(P[i][i]) >= 0 for i in range(len(P)))
    Px, Ax = matvec(P, xf), matvec(A, xf)
    ATy = [sum((fraction(A[j][i]) * yf[j] for j in range(model["m"])), Fraction(0))
           for i in range(model["n"])]
    stationarity = [p + fraction(q) + a for p, q, a in zip(Px, model["q"], ATy)]
    feasibility, complementarity, support = [], [], Fraction(0)
    for value, dual, lo, hi in zip(Ax, yf, model["l"], model["u"]):
        lo, hi = fraction(lo), fraction(hi)
        assert lo <= hi, "original reversed bounds"
        feasibility.append(max(lo - value, value - hi, Fraction(0)))
        bound = hi if dual > 0 else lo if dual < 0 else Fraction(0)
        support += dual * bound
        complementarity.append(abs(dual * (bound - value)))
    curvature, range_available = Fraction(0), True
    for i, a in enumerate(ATy):
        z, diagonal = fraction(model["q"][i]) + a, fraction(P[i][i])
        if diagonal == 0:
            range_available = range_available and z == 0
        else:
            curvature += z * z / (2 * diagonal)
    primal, limit = objective(model, xf), fraction(tolerance)
    dual = -curvature - support if range_available else None
    gap = primal - dual if dual is not None else None
    assert max(feasibility, default=Fraction(0)) <= limit, "original primal feasibility"
    assert max(map(abs, stationarity), default=Fraction(0)) <= limit, "original stationarity"
    assert max(complementarity, default=Fraction(0)) <= limit, "original complementarity"
    if require_certificate:
        assert range_available, "dual outside exact diagonal range"
        assert gap >= 0, "negative signed gap; candidate not exactly primal feasible"
        assert gap <= limit, "original signed gap"
    raw = {"primal": primal,
           "feasibility": max(feasibility, default=Fraction(0)),
           "stationarity": max(map(abs, stationarity), default=Fraction(0)),
           "complementarity": max(complementarity, default=Fraction(0))}
    if dual is not None:
        raw.update(dual=dual, signed_gap=gap)
    metrics = {name: {"numerator": str(value.numerator), "denominator": str(value.denominator)}
               for name, value in raw.items()}
    metrics["candidate_certificate"] = ("UNAVAILABLE_DUAL_RANGE" if not range_available else
                                        "REJECTED_NEGATIVE_SIGNED_GAP" if gap < 0 else
                                        "APPROXIMATE_NONNEGATIVE_GAP" if gap <= limit else "REJECTED_LARGE_GAP")
    metrics["qualification"] = "host original approximate KKT only; raw signed gap is not repaired"
    return metrics


def represent(matrix, kind):
    if kind == "coo":
        return matrix.to_sparse_coo()
    if kind == "csr":
        return matrix.to_sparse_csr()
    if kind == "matrix_free":
        return engine.MatrixFreeOperator(tuple(matrix.shape), matrix.device, matrix.dtype,
                                         lambda v: matrix @ v, lambda v: matrix.T @ v)
    assert kind == "dense"
    return matrix


def build(model, kind, device, dtype, *, vector_double=False):
    vector_dtype = torch.float64 if vector_double else dtype
    P = torch.tensor(model["P"], device=device, dtype=dtype).reshape(model["n"], model["n"])
    A = torch.tensor(model["A"], device=device, dtype=dtype).reshape(model["m"], model["n"])
    vectors = [torch.tensor(model[name], device=device, dtype=vector_dtype) for name in ("q", "l", "u")]
    problem = engine.OperatorQPProblem(represent(P, kind), vectors[0], represent(A, kind),
                                       vectors[1], vectors[2], assume_psd=True)
    return problem, [P, A, *vectors]


def config(device, dtype, *, max_iterations=300):
    small = dtype == torch.float64
    return engine.OperatorQPConfig(device=str(device), dtype=dtype, max_iterations=max_iterations,
                                   check_interval=5, eps_abs=1e-8 if small else 2e-5,
                                   eps_rel=1e-8 if small else 2e-5,
                                   cg_eps_abs=1e-12 if small else 1e-7,
                                   cg_eps_rel=1e-12 if small else 1e-7)


def write(path, value):
    assert not path.resolve().is_relative_to(ROOT), "proof must be external"
    payload = encoded(value)
    with path.open("xb") as stream:
        stream.write(payload)
    return {"path": str(path), "sha256": sha(payload), "bytes": len(payload)}


def exercise(model, kind, device, dtype, monkeypatch, tmp_path, request, *, loss=None):
    before = sources()
    original = write(tmp_path / "original.json", model)
    settings = config(device, dtype, max_iterations=100 if loss else 300)
    settings_json = asdict(settings)
    settings_json["dtype"] = str(settings.dtype)
    problem, inputs = build(model, kind, device, dtype, vector_double=loss in LOSSES)
    frozen_inputs = [t.detach().clone() for t in inputs]
    trace = Placement(device)
    cuda = device.type == "cuda"
    profile = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA]) if cuda else nullcontext()
    result, rejection, workspace = None, None, None

    def no_densify(*args, **kwargs):
        raise AssertionError("sparse/matrix-free execution attempted densification")

    with monkeypatch.context() as patch:
        if kind != "dense":
            patch.setattr(torch.Tensor, "to_dense", no_densify)
            patch.setattr(torch.linalg, "cholesky_ex", no_densify)
        with profile as profiler, trace:
            if loss in LOSSES:
                with pytest.raises(ValueError, match="original bounds|conversion") as caught:
                    engine.prepare_operator_qp(problem, settings)
                rejection = {"type": type(caught.value).__name__, "reason": loss}
            else:
                workspace = engine.prepare_operator_qp(problem, settings)
                result = workspace.solve()
            if cuda:
                torch.cuda.synchronize(device)
    assert all(torch.equal(t, saved) for t, saved in zip(inputs, frozen_inputs)), "original inputs mutated"
    assert sources() == before, "source changed during original case"
    profiler_receipt = {"kernel_events": 0, "qualification": "CUDA_UNVERIFIED_CPU_DEVELOPMENT"}
    if cuda:
        events = [event for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA]
        assert events, "CUDA profiler recorded no actual kernels"
        profiler.export_chrome_trace(str(tmp_path / "kernel-trace.json"))
        payload = (tmp_path / "kernel-trace.json").read_bytes()
        profiler_receipt = {"kernel_events": len(events), "trace_sha256": sha(payload), "trace_bytes": len(payload),
                            "trace_path": str(tmp_path / "kernel-trace.json"), "scope": "local profiler, not remote attestation"}
    saved_result = {"status": "rejected_before_solve", "rejection": rejection}
    validation = {"qualification": "conversion refusal, no solved candidate"}
    if result is not None:
        assert result.x.device == result.y.device == device and result.x.dtype == result.y.dtype == dtype
        assert workspace.P.device == workspace.A.device == workspace.system.device == device
        assert all(t.device == device for t in (workspace.q, workspace.l, workspace.u,
                                                workspace.original_q, workspace.original_l, workspace.original_u))
        if kind in ("coo", "csr"):
            expected_layout = torch.sparse_coo if kind == "coo" else torch.sparse_csr
            assert workspace.P.operator.matrix.layout == workspace.A.operator.matrix.layout == expected_layout
            assert workspace.original_P.operator.matrix.layout == workspace.original_A.operator.matrix.layout == expected_layout
            assert any("_sparse" in name or "addmm" in name for name in trace.operations), "no sparse operation trace"
        x, y = result.x.detach().cpu().tolist(), result.y.detach().cpu().tolist()
        saved_result = {"status": result.status, "x": x, "y": y, "x_hex": [float(v).hex() for v in x],
                        "y_hex": [float(v).hex() for v in y], "iterations": result.iterations,
                        "objective": result.objective, "inner_residual": result.inner_residual,
                        "inner_tolerance": result.inner_tolerance, "observed_device": str(result.x.device)}
        # Retain the real candidate even if the independent check below fails.
        write(tmp_path / "raw-solution.json", saved_result)
        if loss == "zero_row":
            assert result.status == "max_iterations", "zero-row contradiction falsely solved"
            assert fraction(model["l"][1]) > 0 and all(v == 0 for v in model["A"][1])
            validation = {"original_zero_row_infeasible": "ANALYTIC", "qualification": "max_iterations is not certificate"}
        else:
            assert result.status == "solved" and result.inner_residual <= result.inner_tolerance
            limit = 2e-6 if dtype == torch.float64 else 2e-3
            validation = original_check(model, x, y, tolerance=limit, require_certificate=False)
            analytic = original_check(model, model["analytic_x"], model["analytic_y"], tolerance=0)
            assert analytic["signed_gap"]["numerator"] == "0"
            validation["analytic_original_witness"] = analytic
            assert max(abs(fraction(v) - fraction(e)) for v, e in zip(x, model["analytic_x"])) <= fraction(limit)
    result_file = write(tmp_path / "solution.json", saved_result)
    proof = {"schema_version": 1, "scope": "operator-original-device-regression-v1", "kind": kind,
             "case": model["name"], "dtype": str(dtype), "requested_device": str(device),
             "settings": settings_json, "test_nodeid": request.node.nodeid,
             "source_sha256": before, "original": original, "solution": result_file,
             "placement": trace.report(), "profiler": profiler_receipt,
             "original_validation": validation, "execution_attestation": "UNVERIFIED",
             "commercial": "UNVERIFIED", "performance": "UNVERIFIED"}
    proof_file = write(tmp_path / "proof.json", proof)
    request.node.user_properties.extend(("operator_original_" + key, str(value))
                                        for key, value in proof_file.items())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("dtype", (torch.float64, torch.float32), ids=("f64", "f32"))
def test_cpu_original_models_fraction(kind, name, dtype, monkeypatch, tmp_path, request):
    exercise(original_model(name), kind, torch.device("cpu"), dtype, monkeypatch, tmp_path, request)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("dtype", (torch.float64, torch.float32), ids=("f64", "f32"))
def test_cuda_original_models_fraction(kind, name, dtype, monkeypatch, tmp_path, request):
    exercise(original_model(name), kind, strict_cuda(), dtype, monkeypatch, tmp_path, request)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("loss", LOSSES)
def test_cpu_original_conversion_refused(kind, loss, monkeypatch, tmp_path, request):
    exercise(negative_model(loss), kind, torch.device("cpu"), torch.float32,
             monkeypatch, tmp_path, request, loss=loss)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("loss", LOSSES)
def test_cuda_original_conversion_refused(kind, loss, monkeypatch, tmp_path, request):
    exercise(negative_model(loss), kind, strict_cuda(), torch.float32,
             monkeypatch, tmp_path, request, loss=loss)


@pytest.mark.parametrize("kind", KINDS)
def test_cpu_zero_row_cannot_borrow_other_row_scale(kind, monkeypatch, tmp_path, request):
    exercise(negative_model("zero_row"), kind, torch.device("cpu"), torch.float64,
             monkeypatch, tmp_path, request, loss="zero_row")


@pytest.mark.parametrize("kind", KINDS)
def test_cuda_zero_row_cannot_borrow_other_row_scale(kind, monkeypatch, tmp_path, request):
    exercise(negative_model("zero_row"), kind, strict_cuda(), torch.float64,
             monkeypatch, tmp_path, request, loss="zero_row")


def test_cpu_explicit_unavailable_cuda_never_falls_back(monkeypatch):
    monkeypatch.setenv("OPENOPT_RUN_OPERATOR_CUDA", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable; no CPU fallback"):
        strict_cuda()
    with pytest.raises(RuntimeError, match="explicit CUDA.*no CPU fallback"):
        engine.OperatorQPConfig(device="cuda:0").validate()


def test_cpu_fraction_oracle_mutations_not_summary_pass():
    model = original_model("equality_qp")
    assert original_check(model, model["analytic_x"], model["analytic_y"], tolerance=0)
    for x, y in (([.25, -.49], model["analytic_y"]),
                 (model["analytic_x"], [-.5, .25, 0.])):
        with pytest.raises(AssertionError):
            original_check(model, x, y, tolerance=1e-8)
    changed = original_model("equality_qp")
    changed["l"][0] = .5
    with pytest.raises(AssertionError):
        original_check(changed, model["analytic_x"], model["analytic_y"], tolerance=1e-8)


def test_cpu_negative_signed_gap_never_certified_even_nearly_feasible():
    model = original_model("equality_qp")
    x, y = [.25, -.5 - 2. ** -30], model["analytic_y"]
    numerical = original_check(model, x, y, tolerance=1e-6, require_certificate=False)
    assert numerical["candidate_certificate"] == "REJECTED_NEGATIVE_SIGNED_GAP"
    assert int(numerical["signed_gap"]["numerator"]) < 0
    with pytest.raises(AssertionError, match="negative signed gap"):
        original_check(model, x, y, tolerance=1e-6)


def test_cpu_lp_tiny_nonzero_range_never_dual_certificate():
    model = original_model("bounded_lp")
    y = [1. - 2. ** -30, -1., 0.]
    numerical = original_check(model, model["analytic_x"], y, tolerance=1e-6, require_certificate=False)
    assert numerical["candidate_certificate"] == "UNAVAILABLE_DUAL_RANGE" and "dual" not in numerical
    with pytest.raises(AssertionError, match="dual outside exact diagonal range"):
        original_check(model, model["analytic_x"], y, tolerance=1e-6)


def test_cuda_callback_returning_cpu_tensor_rejected(monkeypatch):
    device = strict_cuda()
    model = original_model("interior_qp")
    problem, _ = build(model, "matrix_free", device, torch.float64)
    original = problem.A
    broken = engine.MatrixFreeOperator(original.shape, device, original.dtype,
                                       lambda v: original.matvec(v).cpu(), original.rmatvec)
    bad = engine.OperatorQPProblem(problem.P, problem.q, broken, problem.l, problem.u, assume_psd=True)
    with pytest.raises(ValueError, match="device/dtype"):
        engine.prepare_operator_qp(bad, config(device, torch.float64))


FORMAL_CASES = [
    pytest.param("model", kind, name, dtype, id=f"{precision}-{name}-{kind}")
    for dtype, precision in ((torch.float64, "f64"), (torch.float32, "f32"))
    for name in MODELS for kind in KINDS
] + [
    pytest.param("conversion", kind, loss, torch.float32, id=f"f32-{loss}-{kind}")
    for loss in LOSSES for kind in KINDS
] + [
    pytest.param("zero_row", kind, "zero_row", torch.float64, id=f"f64-zero_row-{kind}")
    for kind in KINDS
] + [pytest.param("callback", "matrix_free", "callback_cpu", torch.float64,
                  id="f64-callback_cpu-matrix_free")]


def formal_entry_selected(pytest_config):
    entry = "test_cuda_collection_entry"
    return (any("::" + entry in str(target) for target in pytest_config.args)
            or entry in pytest_config.option.keyword)


@pytest.mark.parametrize("role,kind,name,dtype", FORMAL_CASES)
def test_cuda_collection_entry(role, kind, name, dtype, monkeypatch, tmp_path, request):
    """Explicit target/keyword authorizes strict CUDA without worker env changes."""
    if not formal_entry_selected(request.config):
        pytest.skip("formal CUDA collection entry not explicitly selected; UNVERIFIED")
    device = required_cuda_device()
    # Check real device first; opt-in only lives inside this selected node.
    with monkeypatch.context() as patch:
        patch.setenv("OPENOPT_RUN_OPERATOR_CUDA", "1")
        assert strict_cuda() == device
        if role == "model":
            test_cuda_original_models_fraction(kind, name, dtype, patch, tmp_path, request)
        elif role == "conversion":
            test_cuda_original_conversion_refused(kind, name, patch, tmp_path, request)
        elif role == "zero_row":
            test_cuda_zero_row_cannot_borrow_other_row_scale(kind, patch, tmp_path, request)
        else:
            assert role == "callback"
            test_cuda_callback_returning_cpu_tensor_rejected(patch)


@pytest.mark.parametrize("failure", ("unavailable", "count", "hardware"))
def test_cpu_formal_entry_requires_device_before_optin(failure, monkeypatch, tmp_path, request):
    from types import SimpleNamespace

    monkeypatch.delenv("OPENOPT_RUN_OPERATOR_CUDA", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: failure != "unavailable")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if failure == "count" else 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "wrong hardware")
    fake_request = SimpleNamespace(config=SimpleNamespace(
        args=["tests/test_operator_original_cuda.py::test_cuda_collection_entry"],
        option=SimpleNamespace(keyword="")))
    message = {"unavailable": "unavailable; no CPU fallback", "count": "exactly one",
               "hardware": "RTX PRO 4500"}[failure]
    with pytest.raises(RuntimeError, match=message):
        test_cuda_collection_entry("model", "dense", "interior_qp", torch.float64,
                                   monkeypatch, tmp_path, fake_request)
    assert "OPENOPT_RUN_OPERATOR_CUDA" not in os.environ


def test_cpu_formal_entry_selection_is_explicit():
    from types import SimpleNamespace

    for targets, keyword, expected in (
        (["tests/test_operator_original_cuda.py"], "", False),
        (["tests/test_operator_original_cuda.py"], "test_cuda", False),
        (["tests/test_operator_original_cuda.py::test_cuda_collection_entry[f64-interior_qp-dense]"], "", True),
        (["tests/test_operator_original_cuda.py"], "test_cuda_collection_entry", True),
    ):
        assert formal_entry_selected(SimpleNamespace(
            args=targets, option=SimpleNamespace(keyword=keyword))) is expected
    ids = [case.id for case in FORMAL_CASES]
    assert len(ids) == len(set(ids)) == 53


def test_cpu_formal_cli_collection_membership(tmp_path):
    entry = "tests/test_operator_original_cuda.py::test_cuda_collection_entry"
    expected = {f"{entry}[{case.id}]" for case in FORMAL_CASES}
    env = os.environ.copy()
    env.pop("OPENOPT_RUN_OPERATOR_CUDA", None)
    env.update(PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))),
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTHONDONTWRITEBYTECODE="1")
    for name, selection in (("target", [entry]),
                            ("expression", ["tests/test_operator_original_cuda.py",
                                            "-k", "test_cuda_collection_entry"])):
        command = [sys.executable, "-m", "pytest", "--collect-only", "-q",
                   "-o", f"cache_dir={tmp_path / name}", *selection]
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        write(tmp_path / f"{name}-collection.json", {
            "command": command, "returncode": result.returncode,
            "stdout": result.stdout, "stderr": result.stderr,
            "scope": "CPU collection only; not CUDA execution"})
        assert result.returncode == 0, "formal collection CLI failed"
        nodes = [line for line in result.stdout.splitlines() if line.startswith(entry + "[")]
        assert len(nodes) == len(set(nodes)) == 53 and set(nodes) == expected


def scan(argv=None):
    parser = argparse.ArgumentParser(description="Scoped safe source credential scan")
    parser.add_argument("--scan-only", action="store_true", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.artifact_dir.resolve()
    assert not output.is_relative_to(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("safe_scanner", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(secret_hex=None, secrets_file=None, no_needle=False,
                            needle_from=str(Path.home() / ".config/dsh-n600-token")))
    assert needles, "credential scan needle unavailable"
    paths = [Path(__file__).resolve(), ROOT / "docs/research/operator-original-cuda.md"]
    literal, _ = scanner.scan_literal([str(p) for p in paths], needles)
    pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                         r"(?i:password|token|secret)\s*[\"\x27]?\s*[:=]\s*[\"\x27][^\"\x27\s]{6,}[\"\x27]")
    shape = sum(bool(pattern.search(p.read_text())) for p in paths)
    write(output / "source-security-scan.json", {"verdict": "PASS" if literal == shape == 0 else "FAIL",
                                               "literal_hits": literal, "shape_hits": shape,
                                               "files": {str(p): sha(p.read_bytes()) for p in paths}})
    print(f"OPERATOR_SOURCE_SCAN literal_hits={literal} shape_hits={shape} files={len(paths)}")
    return 0 if literal == shape == 0 else 1


if __name__ == "__main__":
    raise SystemExit(scan())
