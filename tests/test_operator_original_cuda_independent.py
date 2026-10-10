"""CPU-only independent review of the frozen original-operator CUDA producer.

The trusted repository producer may execute CPU solves. Artifact files are
data only; their mathematical replay uses this file's own Fraction oracle.
No CUDA submission, artifact code execution or performance claim is made.
Use --review for external logs, exact collection/JUnit membership and pins.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
SUBJECT = "f986c9404b807f453b6d3dfefaaad061970c5148"
BASE = "73613fce9faafdd622baf69dcf1ecb64b4cfa6f7"
PRODUCER = "tests/test_operator_original_cuda.py"
SELF = "tests/test_operator_original_cuda_independent.py"
DOC = "docs/research/operator-original-cuda-independent.md"
ENTRY = PRODUCER + "::test_cuda_collection_entry"
PIN_PATHS = ("src/openopt/operators.py", PRODUCER,
             "docs/research/operator-original-cuda.md", "scripts/gpu_release_check.py")
KINDS = ("dense", "coo", "csr", "matrix_free")
MODELS = ("interior_qp", "equality_qp", "bounded_lp", "empty_rows_qp")
LOSSES = ("reversed", "lower_overflow", "upper_overflow", "nonzero_underflow")


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def save(path, value):
    path = Path(path)
    assert not path.resolve().is_relative_to(ROOT), "external evidence required"
    with path.open("xb") as stream:
        stream.write(encoded(value))


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"Git identity query failed rc={result.returncode}")
    return result.stdout


def pinned_source():
    pins = {}
    for name in PIN_PATHS:
        path = ROOT / name
        assert path.is_file() and not path.is_symlink(), ("source absent/symlink", name)
        frozen = git("show", f"{SUBJECT}:{name}")
        assert path.read_bytes() == frozen, ("producer source drift", name)
        pins[name] = sha(frozen)
    assert (ROOT / PIN_PATHS[0]).read_bytes() == git("show", f"{BASE}:{PIN_PATHS[0]}")
    return pins


@pytest.fixture(scope="session")
def producer():
    pinned_source()
    spec = importlib.util.spec_from_file_location("trusted_operator_producer", ROOT / PRODUCER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.engine.__file__).resolve() == ROOT / "src/openopt/operators.py"
    return module


def rational(value):
    assert type(value) in (int, float), "nonfinite/bool scalar"
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    assert finite, "scalar outside finite binary64 range"
    return Fraction(value) if type(value) is int else Fraction.from_float(value)


def own_model(name):
    P, q = [[2.0, 0.0], [0.0, 4.0]], [-1.0, 2.0]
    A, lower, upper = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], [-1.0, -2.0, -1.0], [2.0, 1.0, 1.0]
    x, y = [0.5, -0.5], [0.0, 0.0, 0.0]
    if name == "equality_qp":
        q, lower, upper = [-1.0, 2.25], [0.25, -0.5, -1.0], [0.25, -0.5, 1.0]
        x, y = [0.25, -0.5], [0.5, -0.25, 0.0]
    elif name == "bounded_lp":
        P, q = [[0.0, 0.0], [0.0, 0.0]], [-1.0, 1.0]
        lower, upper, x, y = [0.0, -1.0, -1.0], [1.0, 0.0, 1.0], [1.0, -1.0], [1.0, -1.0, 0.0]
    elif name == "empty_rows_qp":
        A, lower, upper, y = [], [], [], []
    else:
        assert name == "interior_qp"
    return dict(name=name, n=2, m=len(A), P=P, q=q, A=A, l=lower, u=upper, analytic_x=x, analytic_y=y)


def own_negative(name):
    lower, upper = [1e9, 1.0], [1e9, 2.0]
    if name == "reversed":
        lower[1], upper[1] = 1.0 + 2.0 ** -40, 1.0
    elif name == "lower_overflow":
        lower[1] = -(2.0 ** 400)
    elif name == "upper_overflow":
        upper[1] = 2.0 ** 400
    elif name == "nonzero_underflow":
        lower[1], upper[1] = 2.0 ** -400, 2.0 ** -399
    else:
        assert name == "zero_row"
    return dict(name=name, n=1, m=2, P=[[1.0]], q=[-1e9], A=[[1.0], [0.0]], l=lower, u=upper)


def dot(a, b):
    assert len(a) == len(b), "vector dimension"
    return sum((rational(v) * rational(w) for v, w in zip(a, b)), Fraction())


def oracle(model, x, y, tolerance, *, certificate=False):
    """Independent diagonal dual derivation, with exact signed gap/range.

    Feasibility, stationary and complementarity tests are approximate, with
    an explicit absolute limit. A certificate additionally needs exact range
    and a nonnegative gap. No solver or producer numerical function is used.
    """
    n, m = model["n"], model["m"]
    assert type(n) is int and type(m) is int and n > 0 and m >= 0
    assert len(model["P"]) == len(model["q"]) == len(x) == n
    assert len(model["A"]) == len(model["l"]) == len(model["u"]) == len(y) == m
    assert all(len(row) == n for row in model["P"] + model["A"])
    assert all(rational(model["P"][i][j]) == 0 for i in range(n) for j in range(n) if i != j)
    diagonal = [rational(model["P"][i][i]) for i in range(n)]
    assert min(diagonal) >= 0
    limit = rational(tolerance)
    assert limit >= 0
    px = [dot(row, x) for row in model["P"]]
    ax = [dot(row, x) for row in model["A"]]
    aty = [dot([row[i] for row in model["A"]], y) for i in range(n)]
    stationarity = [px[i] + rational(model["q"][i]) + aty[i] for i in range(n)]
    support, violation, complementarity = Fraction(), Fraction(), Fraction()
    for i in range(m):
        lower, upper, multiplier = rational(model["l"][i]), rational(model["u"][i]), rational(y[i])
        assert lower <= upper
        violation = max(violation, lower - ax[i], ax[i] - upper)
        selected = upper if multiplier > 0 else lower if multiplier < 0 else Fraction()
        support += multiplier * selected
        complementarity = max(complementarity, abs(multiplier * (selected - ax[i])))
    primal = sum((rational(x[i]) * (px[i] / 2 + rational(model["q"][i])) for i in range(n)), Fraction())
    linear = [rational(model["q"][i]) + aty[i] for i in range(n)]
    range_ok = all(linear[i] == 0 for i in range(n) if diagonal[i] == 0)
    dual = (-sum((linear[i] ** 2 / (2 * diagonal[i]) for i in range(n) if diagonal[i]), Fraction())
            - support) if range_ok else None
    gap = primal - dual if range_ok else None
    assert violation <= limit, "independent primal"
    assert max(map(abs, stationarity)) <= limit, "independent stationarity"
    assert complementarity <= limit, "independent complementarity"
    if certificate:
        assert range_ok, "independent exact range"
        assert 0 <= gap <= limit, "independent signed gap"
    status = ("UNAVAILABLE_DUAL_RANGE" if not range_ok else
              "REJECTED_NEGATIVE_SIGNED_GAP" if gap < 0 else
              "APPROXIMATE_NONNEGATIVE_GAP" if gap <= limit else "REJECTED_LARGE_GAP")
    return dict(primal=primal, feasibility=violation, stationarity=max(map(abs, stationarity)),
                complementarity=complementarity, dual=dual, signed_gap=gap, candidate_certificate=status)


def metrics_json(metrics):
    return {key: {"numerator": str(value.numerator), "denominator": str(value.denominator)}
            if isinstance(value, Fraction) else value for key, value in metrics.items() if value is not None}


def expected_nodes(device):
    identities = set()
    for dtype in ("f64", "f32"):
        for model in MODELS:
            for kind in KINDS:
                identities.add(f"{PRODUCER}::test_{device}_original_models_fraction[{dtype}-{model}-{kind}]")
    for loss in LOSSES:
        for kind in KINDS:
            identities.add(f"{PRODUCER}::test_{device}_original_conversion_refused[{loss}-{kind}]")
    for kind in KINDS:
        identities.add(f"{PRODUCER}::test_{device}_zero_row_cannot_borrow_other_row_scale[{kind}]")
    extras = (
        ("explicit_unavailable_cuda_never_falls_back", "fraction_oracle_mutations_not_summary_pass",
         "negative_signed_gap_never_certified_even_nearly_feasible", "lp_tiny_nonzero_range_never_dual_certificate",
         "formal_entry_requires_device_before_optin[unavailable]", "formal_entry_requires_device_before_optin[count]",
         "formal_entry_requires_device_before_optin[hardware]", "formal_entry_selection_is_explicit",
         "formal_cli_collection_membership")
        if device == "cpu" else ("callback_returning_cpu_tensor_rejected",)
    )
    identities.update(f"{PRODUCER}::test_{device}_{name}" for name in extras)
    return identities


def formal_cases():
    cases = [(f"{precision}-{name}-{kind}", "model", kind, name, dtype)
             for dtype, precision in ((torch.float64, "f64"), (torch.float32, "f32"))
             for name in MODELS for kind in KINDS]
    cases.extend((f"f32-{loss}-{kind}", "conversion", kind, loss, torch.float32)
                 for loss in LOSSES for kind in KINDS)
    cases.extend((f"f64-zero_row-{kind}", "zero_row", kind, "zero_row", torch.float64) for kind in KINDS)
    cases.append(("f64-callback_cpu-matrix_free", "callback", "matrix_free", "callback_cpu", torch.float64))
    return cases


def formal_nodes():
    return {f"{ENTRY}[{case[0]}]" for case in formal_cases()}


def environment():
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=os.pathsep.join((str(ROOT / "src"), str(ROOT))),
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("OPENOPT_RUN_OPERATOR_CUDA", None)
    return env


def logged(command, output, name, env=None):
    with (output / f"{name}.log").open("x") as stream:
        stream.write(f"command={command!r}\n")
        stream.flush()
        result = subprocess.run(command, cwd=ROOT, env=env or environment(), stdout=stream,
                                stderr=subprocess.STDOUT, timeout=120, check=False)
        stream.write(f"\nreturncode={result.returncode}\n")
    return result.returncode


def collect(target, output, expression=None):
    command = [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts=",
               "-o", f"cache_dir={output / 'cache'}", target]
    if expression:
        command.extend(("-k", expression))
    rc = logged(command, output, "collection")
    assert rc == 0, ("collection failed", output)
    prefix = target + "[" if "::" in target else target + "::"
    nodes = [line.strip() for line in (output / "collection.log").read_text().splitlines()
             if line.startswith(prefix)]
    assert nodes and len(set(nodes)) == len(nodes), "empty/duplicate collection"
    return set(nodes)


def junit(path):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    identities = [case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"] for case in cases]
    assert len(identities) == len(set(identities)), "duplicate JUnit"
    totals = {key: sum(int(s.attrib.get(key, 0)) for s in root.iter("testsuite"))
              for key in ("tests", "failures", "errors", "skipped")}
    actual = dict(tests=len(cases), failures=sum(case.find("failure") is not None for case in cases),
                  errors=sum(case.find("error") is not None for case in cases),
                  skipped=sum(case.find("skipped") is not None for case in cases))
    assert totals == actual, ("JUnit totals inconsistent", totals, actual)
    return cases, set(identities), actual


def pytest_run(target, output, expression=None, env=None):
    command = [sys.executable, "-m", "pytest", "-q", "-ra", "-o", "addopts=",
               "-o", "junit_family=xunit1", "-o", f"cache_dir={output / 'cache'}",
               "--basetemp", str(output / "pytest-tmp"), f"--junitxml={output / 'pytest.xml'}", target]
    if expression:
        command.extend(("-k", expression))
    return logged(command, output, "pytest", env)


@pytest.fixture(scope="session")
def cpu_run(tmp_path_factory):
    output = tmp_path_factory.mktemp("actual-frozen-cpu-producer")
    before = pinned_source()
    collection = collect(PRODUCER, output, "test_cpu")
    assert collection == expected_nodes("cpu")
    assert pytest_run(PRODUCER, output, "test_cpu") == 0
    cases, executed, counts = junit(output / "pytest.xml")
    assert executed == collection and counts == dict(tests=61, failures=0, errors=0, skipped=0)
    assert pinned_source() == before
    properties = {}
    for case in cases:
        node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
        pairs = [(p.attrib["name"], p.attrib["value"]) for p in case.findall("properties/property")]
        assert len(pairs) == len(dict(pairs)), "duplicate property"
        if pairs:
            properties[node] = dict(pairs)
    assert len(properties) == 52
    return output, properties


def descriptor(desc, directory):
    assert set(desc) == {"path", "sha256", "bytes"}
    path = Path(desc["path"])
    assert path.is_absolute() and path.resolve().parent == directory.resolve()
    assert path.is_file() and not path.is_symlink()
    payload = path.read_bytes()
    assert type(desc["bytes"]) is int and desc["bytes"] == len(payload) > 0
    assert desc["sha256"] == sha(payload), "sidecar bytes"
    def unique(pairs):
        result = {}
        for key, value in pairs:
            assert key not in result, "duplicate JSON key"
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=unique)


def check_proof(proof, directory, node, pins):
    assert proof["schema_version"] == 1 and type(proof["schema_version"]) is int
    assert proof["scope"] == "operator-original-device-regression-v1"
    assert proof["test_nodeid"] == node and proof["source_sha256"] == {
        name: pins[name] for name in (PIN_PATHS[0], PRODUCER, PIN_PATHS[3])
    }
    assert proof["kind"] in KINDS
    suffix = ("[" if proof["case"] == "zero_row" else "-") + proof["kind"] + "]"
    assert node.endswith(suffix)
    assert proof["requested_device"] == "cpu" and proof["dtype"] in ("torch.float64", "torch.float32")
    assert all(proof[name] == "UNVERIFIED" for name in ("execution_attestation", "performance", "commercial"))
    trace = proof["placement"]
    assert trace["device"] == "cpu" and type(trace["tensors"]) is int and trace["tensors"] > 0
    assert trace["operations"] and all(type(count) is int and count > 0 for count in trace["operations"].values())
    assert proof["profiler"]["kernel_events"] == 0
    assert proof["profiler"]["qualification"] == "CUDA_UNVERIFIED_CPU_DEVELOPMENT"
    original, solution = descriptor(proof["original"], directory), descriptor(proof["solution"], directory)
    name = proof["case"]
    expected = own_model(name) if name in MODELS else own_negative(name)
    assert original == expected, "original model anchor"
    if name in MODELS:
        dtype = "f64" if proof["dtype"] == "torch.float64" else "f32"
        assert f"[{dtype}-{name}-" in node
    elif name in LOSSES:
        assert f"[{name}-" in node and proof["dtype"] == "torch.float32"
    else:
        assert name == "zero_row" and proof["dtype"] == "torch.float64"
    settings = proof["settings"]
    assert settings == dict(device="cpu", dtype=proof["dtype"], rho=1.0, sigma=1e-6, relaxation=1.6,
                            max_iterations=300 if name in MODELS else 100, check_interval=5,
                            eps_abs=1e-8 if proof["dtype"] == "torch.float64" else 2e-5,
                            eps_rel=1e-8 if proof["dtype"] == "torch.float64" else 2e-5,
                            cg_max_iterations=1000,
                            cg_eps_abs=1e-12 if proof["dtype"] == "torch.float64" else 1e-7,
                            cg_eps_rel=1e-12 if proof["dtype"] == "torch.float64" else 1e-7,
                            cg_preconditioner="none")
    if name in LOSSES:
        assert solution == dict(status="rejected_before_solve", rejection=dict(type="ValueError", reason=name))
        assert proof["original_validation"] == {"qualification": "conversion refusal, no solved candidate"}
        return None
    assert solution["observed_device"] == "cpu"
    assert len(solution["x"]) == original["n"] and len(solution["y"]) == original["m"]
    for value in solution["x"] + solution["y"]:
        rational(value)
    rational(solution["objective"])
    assert solution["x_hex"] == [float(value).hex() for value in solution["x"]]
    assert solution["y_hex"] == [float(value).hex() for value in solution["y"]]
    assert type(solution["iterations"]) is int and 0 < solution["iterations"] <= settings["max_iterations"]
    assert rational(solution["inner_residual"]) <= rational(solution["inner_tolerance"])
    if name == "zero_row":
        assert solution["status"] == "max_iterations" and solution["iterations"] == 100
        assert dot(original["A"][1], solution["x"]) == 0 < rational(original["l"][1])
        assert proof["original_validation"] == {
            "original_zero_row_infeasible": "ANALYTIC", "qualification": "max_iterations is not certificate"
        }
        return None
    assert solution["status"] == "solved"
    limit = 2e-6 if proof["dtype"] == "torch.float64" else 2e-3
    metrics = oracle(original, solution["x"], solution["y"], limit)
    assert abs(rational(solution["objective"]) - metrics["primal"]) <= rational(limit), "saved original objective"
    validation = proof["original_validation"]
    for key, value in metrics_json(metrics).items():
        assert validation[key] == value, ("original numerical metric", key)
    analytic = oracle(original, original["analytic_x"], original["analytic_y"], 0, certificate=True)
    assert analytic["signed_gap"] == 0
    for key, value in metrics_json(analytic).items():
        assert validation["analytic_original_witness"][key] == value
    assert max(abs(rational(a) - rational(b)) for a, b in zip(solution["x"], original["analytic_x"])) <= rational(limit)
    return metrics


@pytest.mark.parametrize("name", MODELS)
def test_original_analytic_oracle_is_independent_and_exact(producer, name):
    model = own_model(name)
    assert producer.original_model(name) == model
    metrics = oracle(model, model["analytic_x"], model["analytic_y"], 0, certificate=True)
    assert metrics["signed_gap"] == metrics["feasibility"] == metrics["stationarity"] == 0


@pytest.mark.parametrize("mutation", ("primal", "dual_sign", "original_bound", "nonfinite", "bool", "shape", "negative_psd"))
def test_actual_producer_oracle_rejects_independent_bad_candidates(producer, mutation):
    model = own_model("equality_qp")
    x, y = model["analytic_x"][:], model["analytic_y"][:]
    if mutation == "primal":
        x[1] = -0.49
    elif mutation == "dual_sign":
        y = [-0.5, 0.25, 0.0]
    elif mutation == "original_bound":
        model["l"][0] = 0.5
    elif mutation == "nonfinite":
        x[0] = float("nan")
    elif mutation == "bool":
        x[0] = True
    elif mutation == "shape":
        x.pop()
    else:
        model["P"][0][0] = -2.0
    with pytest.raises(AssertionError):
        oracle(model, x, y, 1e-8, certificate=True)
    with pytest.raises(AssertionError):
        producer.original_check(model, x, y, tolerance=1e-8)


@pytest.mark.parametrize("name", ("negative_gap", "null_range"))
def test_nearly_kkt_candidate_cannot_be_promoted_to_certificate(producer, name):
    model = own_model("equality_qp" if name == "negative_gap" else "bounded_lp")
    x, y = model["analytic_x"][:], model["analytic_y"][:]
    if name == "negative_gap":
        x[1] -= 2.0 ** -30
    else:
        y[0] -= 2.0 ** -30
    metrics = oracle(model, x, y, 1e-6)
    assert metrics["candidate_certificate"] in ("REJECTED_NEGATIVE_SIGNED_GAP", "UNAVAILABLE_DUAL_RANGE")
    assert producer.original_check(model, x, y, tolerance=1e-6, require_certificate=False)["candidate_certificate"] == metrics["candidate_certificate"]
    with pytest.raises(AssertionError):
        oracle(model, x, y, 1e-6, certificate=True)
    with pytest.raises(AssertionError):
        producer.original_check(model, x, y, tolerance=1e-6)


@pytest.mark.parametrize("dtype", ("f64", "f32"))
@pytest.mark.parametrize("name", MODELS)
@pytest.mark.parametrize("kind", KINDS)
def test_actual_original_candidate_sidecars_not_producer_summary(cpu_run, kind, name, dtype):
    output, properties = cpu_run
    node = f"{PRODUCER}::test_cpu_original_models_fraction[{dtype}-{name}-{kind}]"
    prop = properties[node]
    desc = {key: prop["operator_original_" + key] for key in ("path", "sha256", "bytes")}
    desc["bytes"] = int(desc["bytes"])
    directory = Path(desc["path"]).parent
    assert directory.resolve().is_relative_to((output / "pytest-tmp").resolve())
    proof = descriptor(desc, directory)
    assert check_proof(proof, directory, node, pinned_source())


@pytest.mark.parametrize("loss", (*LOSSES, "zero_row"))
@pytest.mark.parametrize("kind", KINDS)
def test_actual_negative_original_sidecars_preserve_no_certificate(cpu_run, kind, loss):
    _, properties = cpu_run
    function = "original_conversion_refused" if loss in LOSSES else "zero_row_cannot_borrow_other_row_scale"
    suffix = f"[{loss}-{kind}]" if loss in LOSSES else f"[{kind}]"
    node = f"{PRODUCER}::test_cpu_{function}{suffix}"
    prop = properties[node]
    desc = {key: prop["operator_original_" + key] for key in ("path", "sha256", "bytes")}
    desc["bytes"] = int(desc["bytes"])
    directory = Path(desc["path"]).parent
    assert check_proof(descriptor(desc, directory), directory, node, pinned_source()) is None


@pytest.mark.parametrize("mutation", ("original_bound", "primal", "dual_sign", "hex", "source", "node",
                                    "config", "placement", "empty_trace", "fake_kernel", "certificate",
                                    "objective", "nonfinite_objective"))
def test_coordinated_rehash_mutations_rejected_by_independent_replay(cpu_run, tmp_path, mutation):
    _, properties = cpu_run
    node = f"{PRODUCER}::test_cpu_original_models_fraction[f64-equality_qp-coo]"
    original_path = Path(properties[node]["operator_original_path"])
    proof = json.loads(original_path.read_bytes())
    for key in ("original", "solution"):
        payload = json.loads(Path(proof[key]["path"]).read_bytes())
        if mutation == "original_bound" and key == "original":
            payload["l"][0] = 0.5
        if key == "solution":
            if mutation == "primal":
                payload["x"][1] = -0.49
                payload["x_hex"] = [float(v).hex() for v in payload["x"]]
            elif mutation == "dual_sign":
                payload["y"] = [-0.5, 0.25, 0.0]
                payload["y_hex"] = [float(v).hex() for v in payload["y"]]
            elif mutation == "hex":
                payload["x_hex"][0] = "0x0.0p+0"
            elif mutation == "objective":
                payload["objective"] += 1.0
            elif mutation == "nonfinite_objective":
                # JSON finite values can overflow binary64; no NaN JSON is needed.
                payload["objective"] = 10 ** 400
        path = tmp_path / (key + ".json")
        save(path, payload)
        proof[key] = dict(path=str(path), sha256=sha(path.read_bytes()), bytes=path.stat().st_size)
    if mutation == "source":
        proof["source_sha256"]["src/openopt/operators.py"] = "0" * 64
    elif mutation == "node":
        proof["test_nodeid"] += "-foreign"
    elif mutation == "config":
        proof["settings"]["eps_abs"] = 1.0
    elif mutation == "placement":
        proof["placement"]["device"] = "cuda:0"
    elif mutation == "empty_trace":
        proof["placement"]["operations"] = {}
    elif mutation == "fake_kernel":
        proof["profiler"]["kernel_events"] = 1
    elif mutation == "certificate":
        proof["original_validation"]["candidate_certificate"] = "EXACT_CERTIFIED"
    path = tmp_path / "proof.json"
    save(path, proof)
    outer = dict(path=str(path), sha256=sha(path.read_bytes()), bytes=path.stat().st_size)
    with pytest.raises(AssertionError):
        check_proof(descriptor(outer, tmp_path), tmp_path, node, pinned_source())


def test_actual_dispatcher_observes_operands_outputs_and_rejects_wrong_device(producer):
    trace = producer.Placement(torch.device("cpu"))
    with trace:
        result = torch.ones(2) + torch.ones(2)
    assert torch.equal(result, torch.tensor([2.0, 2.0]))
    assert trace.report()["tensors"] > 0 and len(trace.operations) >= 2
    with pytest.raises(AssertionError, match="off-device"):
        producer.Placement(torch.device("cuda:0")).check(torch.ones(2))
    with pytest.raises(AssertionError, match="off-device"):
        producer.Placement(torch.device("cpu")).check(torch.empty(2, device="meta"))
    with pytest.raises(AssertionError, match="empty device trace"):
        producer.Placement(torch.device("cpu")).report()


@pytest.mark.parametrize("stage", ("operand", "keyword", "output"))
def test_dispatcher_wrong_tensor_is_rejected_on_both_sides(producer, stage):
    trace = producer.Placement(torch.device("cpu"))
    calls = []
    def action(*args, **kwargs):
        calls.append((args, kwargs))
        return torch.empty(1, device="meta") if stage == "output" else torch.ones(1)
    args = (torch.empty(1, device="meta"),) if stage == "operand" else (torch.ones(1),)
    kwargs = {"bad": torch.empty(1, device="meta")} if stage == "keyword" else {}
    with pytest.raises(AssertionError, match="off-device"):
        trace.__torch_dispatch__(action, (), args, kwargs)
    assert len(calls) == (1 if stage == "output" else 0)
    assert not trace.operations, "rejected operation cannot be a successful trace"


@pytest.mark.parametrize("reason", ("unavailable", "count", "hardware"))
def test_cuda_optin_strict_before_any_solve(producer, monkeypatch, reason):
    monkeypatch.setenv("OPENOPT_RUN_OPERATOR_CUDA", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: reason != "unavailable")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2 if reason == "count" else 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "unsupported hardware")
    def forbidden(*args, **kwargs):
        raise AssertionError("must not solve on admission failure")
    monkeypatch.setattr(producer.engine, "prepare_operator_qp", forbidden)
    with pytest.raises(RuntimeError, match={"unavailable": "no CPU fallback", "count": "exactly one", "hardware": "RTX PRO 4500"}[reason]):
        producer.strict_cuda()


def test_all_53_actual_optin_cuda_nodes_fail_without_cuda_zero_skip(tmp_path):
    before = pinned_source()
    selection = "test_cuda and not test_cuda_collection_entry"
    collection = collect(PRODUCER, tmp_path, selection)
    assert collection == expected_nodes("cuda") and len(collection) == 53
    env = environment()
    env["OPENOPT_RUN_OPERATOR_CUDA"] = "1"
    assert pytest_run(PRODUCER, tmp_path, selection, env) == 1
    cases, executed, counts = junit(tmp_path / "pytest.xml")
    assert executed == collection and counts == dict(tests=53, failures=53, errors=0, skipped=0)
    for case in cases:
        assert "explicit operator CUDA requested but unavailable; no CPU fallback" in case.find("failure").text
    assert not list((tmp_path / "pytest-tmp").rglob("proof.json")), "unavailable CUDA ran exercise"
    assert pinned_source() == before


@pytest.mark.parametrize("identity,role,kind,name,dtype", formal_cases(),
                         ids=[case[0] for case in formal_cases()])
def test_formal_exact_identity_routes_without_solver(producer, monkeypatch, tmp_path,
                                                    identity, role, kind, name, dtype):
    from types import SimpleNamespace

    frozen = {case.id: case.values for case in producer.FORMAL_CASES}
    assert len(frozen) == 53 and set(frozen) == {case[0] for case in formal_cases()}
    assert frozen[identity] == (role, kind, name, dtype)
    monkeypatch.delenv("OPENOPT_RUN_OPERATOR_CUDA", raising=False)
    calls = []
    device = torch.device("cuda:0")
    monkeypatch.setattr(producer, "required_cuda_device", lambda: device)
    def strict():
        assert os.environ["OPENOPT_RUN_OPERATOR_CUDA"] == "1"
        return device
    monkeypatch.setattr(producer, "strict_cuda", strict)
    for label in ("test_cuda_original_models_fraction", "test_cuda_original_conversion_refused",
                  "test_cuda_zero_row_cannot_borrow_other_row_scale", "test_cuda_callback_returning_cpu_tensor_rejected"):
        def observe(*args, _label=label):
            assert os.environ["OPENOPT_RUN_OPERATOR_CUDA"] == "1"
            calls.append((_label, args))
        monkeypatch.setattr(producer, label, observe)
    request = SimpleNamespace(config=SimpleNamespace(args=[ENTRY], option=SimpleNamespace(keyword="test_cuda_collection_entry")))
    producer.test_cuda_collection_entry(role, kind, name, dtype, monkeypatch, tmp_path, request)
    target = {"model": "test_cuda_original_models_fraction", "conversion": "test_cuda_original_conversion_refused",
              "zero_row": "test_cuda_zero_row_cannot_borrow_other_row_scale",
              "callback": "test_cuda_callback_returning_cpu_tensor_rejected"}[role]
    assert len(calls) == 1 and calls[0][0] == target
    if role == "model":
        assert calls[0][1][:3] == (kind, name, dtype)
        assert calls[0][1][-2:] == (tmp_path, request)
    elif role == "conversion":
        assert calls[0][1][:2] == (kind, name)
        assert calls[0][1][-2:] == (tmp_path, request)
    elif role == "zero_row":
        assert calls[0][1][0] == kind and calls[0][1][-2:] == (tmp_path, request)
    else:
        assert len(calls[0][1]) == 1
    assert "OPENOPT_RUN_OPERATOR_CUDA" not in os.environ


def test_actual_formal_entry_target_expression_53_fail_no_env_no_skip(tmp_path):
    pins = pinned_source()
    target_dir, expression_dir = tmp_path / "target", tmp_path / "expression"
    target_dir.mkdir()
    expression_dir.mkdir()
    assert collect(ENTRY, target_dir, "test_cuda_collection_entry") == formal_nodes()
    assert collect(PRODUCER, expression_dir, "test_cuda_collection_entry") == formal_nodes()
    assert pytest_run(ENTRY, target_dir, "test_cuda_collection_entry", environment()) == 1
    cases, executed, counts = junit(target_dir / "pytest.xml")
    assert executed == formal_nodes() and counts == dict(tests=53, failures=53, errors=0, skipped=0)
    assert all("explicit operator CUDA requested but unavailable; no CPU fallback" in case.find("failure").text for case in cases)
    assert not list(target_dir.rglob("proof.json"))
    assert pinned_source() == pins


def scan_sources(output):
    spec = importlib.util.spec_from_file_location("safe_scanner", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(secret_hex=None, secrets_file=None, no_needle=False,
                            needle_from=str(Path.home() / ".config/dsh-n600-token")))
    assert needles, "credential scan unavailable"
    paths = [ROOT / SELF, ROOT / DOC]
    literal, _ = scanner.scan_literal([str(path) for path in paths], needles)
    pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                         r"(?i:password|token|secret)\s*[\"\x27]?\s*[:=]\s*[\"\x27][^\"\x27\s]{6,}[\"\x27]")
    shape = sum(bool(pattern.search(path.read_text())) for path in paths)
    report = dict(verdict="PASS" if literal == shape == 0 else "FAIL", literal_hits=literal, shape_hits=shape,
                  files={str(path): sha(path.read_bytes()) for path in paths})
    save(output / "source-security-scan.json", report)
    assert literal == shape == 0


def child_receipts(output, pins):
    """Read only the two actual subprocess groups, selected by exact filenames."""
    groups = {}
    for path in (output / "pytest-tmp").rglob("collection.log"):
        parent = path.parent
        if not (parent / "pytest.xml").is_file():
            # The independent expression-only collection has no execution.
            nodes = [line.strip() for line in path.read_text().splitlines() if line.startswith(ENTRY + "[")]
            assert len(nodes) == 53 and set(nodes) == formal_nodes()
            continue
        _, executed, counts = junit(parent / "pytest.xml")
        nodes = [line.strip() for line in path.read_text().splitlines() if line.startswith(PRODUCER + "::")]
        assert len(nodes) == len(set(nodes))
        collected = set(nodes)
        assert collected == executed and collected
        device = "cpu" if collected == expected_nodes("cpu") else "formal" if collected == formal_nodes() else "cuda"
        assert collected == (formal_nodes() if device == "formal" else expected_nodes(device)) and device not in groups
        assert counts == dict(tests=61 if device == "cpu" else 53, failures=0 if device == "cpu" else 53,
                              errors=0, skipped=0)
        files = {str(file.relative_to(parent)): sha(file.read_bytes())
                 for file in parent.rglob("*") if file.is_file() and not file.is_symlink()
                 and file.name in ("collection.log", "pytest.log", "pytest.xml", "proof.json",
                                   "original.json", "solution.json", "raw-solution.json")}
        proofs = sorted(parent.rglob("proof.json"))
        assert len(proofs) == (52 if device == "cpu" else 0)
        classifications = {}
        if device == "cpu":
            for case in ET.parse(parent / "pytest.xml").getroot().iter("testcase"):
                props = {p.attrib["name"]: p.attrib["value"] for p in case.findall("properties/property")}
                if not props:
                    continue
                desc = {key: props["operator_original_" + key] for key in ("path", "sha256", "bytes")}
                desc["bytes"] = int(desc["bytes"])
                directory = Path(desc["path"]).parent
                assert directory.resolve().is_relative_to(parent.resolve())
                proof = descriptor(desc, directory)
                node = case.attrib["classname"].replace(".", "/") + ".py::" + case.attrib["name"]
                metrics = check_proof(proof, directory, node, pins)
                classification = metrics["candidate_certificate"] if metrics else proof["case"]
                classifications[classification] = classifications.get(classification, 0) + 1
        groups[device] = dict(directory=str(parent), counts=counts, collection=sorted(collected),
                              missing=[], extra=[], proofs=len(proofs), files_sha256=files,
                              candidate_classifications=classifications,
                              interpretation="actual CPU original replay" if device == "cpu" else
                              "expected unavailable CUDA failure, not CUDA execution qualification")
    assert set(groups) == {"cpu", "cuda", "formal"}
    return groups


def review(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    assert args.review != args.scan_only, "select exactly one operation"
    parent = args.artifact_dir.resolve()
    assert not parent.is_relative_to(ROOT)
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="review-", dir=parent))
    start = time.perf_counter()
    receipt = dict(verdict="RUNNING", subject_commit=SUBJECT, base_commit=BASE,
                   started_at=datetime.now(timezone.utc).isoformat(), artifact_dir=str(output),
                   scope="bounded CPU producer code review, not CUDA attestation/performance/commercial signoff")
    try:
        if args.scan_only:
            scan_sources(output)
        else:
            from scripts.verify import source_snapshot, check_source_consistency
            before = source_snapshot(ROOT)
            receipt["source"] = before
            receipt["producer_pins"] = pinned_source()
            collection = collect(SELF, output)
            rc = pytest_run(SELF, output)
            receipt["returncode"] = rc
            _, executed, counts = junit(output / "pytest.xml")
            receipt["counts"] = counts
            receipt["membership"] = dict(missing=sorted(collection - executed), extra=sorted(executed - collection))
            receipt["collection"] = sorted(collection)
            assert rc == 0 and counts["tests"] > 0 and counts["failures"] == counts["errors"] == counts["skipped"] == 0
            assert collection == executed
            receipt["child_executions"] = child_receipts(output, receipt["producer_pins"])
            consistency = check_source_consistency(before, source_snapshot(ROOT))
            receipt["source_consistency"] = consistency
            assert consistency["verdict"] == "PASS"
            receipt["cuda_attestation"] = receipt["performance"] = receipt["commercial"] = "UNVERIFIED"
        receipt["verdict"] = "PASS"
        return 0
    except Exception as error:
        receipt["verdict"], receipt["first_cause"] = "FAIL", f"{type(error).__name__}: {error}"
        return 1
    finally:
        receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
        receipt["wall_s"] = time.perf_counter() - start
        receipt["files"] = {path.name: sha(path.read_bytes()) for path in output.iterdir() if path.is_file()}
        save(output / "independent-review.json", receipt)
        print(f"OPERATOR_INDEPENDENT verdict={receipt['verdict']} receipt={output / 'independent-review.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(review())
