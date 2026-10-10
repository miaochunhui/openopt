"""Independent repaired conic input/math review; all evidence stays on 3FS.

Run this file with --run --artifact-dir DIR, then --audit-artifacts RUN_DIR.
The old 09751dc review is neither imported, changed nor retrospectively passed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

import numpy as np
import pytest
import torch

import openopt.conic as conic
import openopt.conic_certificates as directions
from openopt.conic import ConicProblem, Nonnegative, RSOC, SOC, Zero

TEST = "tests/test_conic_types_repair_independent.py"
DOC = "docs/research/conic-types-repair-independent.md"
OWNER = "966a2b3520f21aa0fe9e9156b960756d9f56d3d7"
PINS = {
    "src/openopt/conic_certificates.py": "63b1d278f26d536903904fdf502c49f3fca914b52cf60249778219f08be3a517",
    "src/openopt/conic.py": "c56dff1923c61af55f992023f64051db5a2e3b3b5bbeec5593731b8b569032cf",
    "src/openopt/certificates.py": "594d1ab59d4d9acd6bd0b6ac0f374b93cce5834b3f27508825a1de3aa0ed639a",
    "src/openopt/native_conic.py": "a98d77c55766f12c0e339da54258554b3b1b6ab897b7ae5a168ca77bf8fd95c7",
    "scripts/verify.py": "4b90e4962ebc4e96697744f04e0f91dc1ff9b50a5634a8f24e323cfe5ebc03b5",
    "tests/test_conic_certificates.py": "6fb024094690c8babe81b2219c90478feb1d5b52482b3d77170cc5fdfbf464f4",
    "docs/research/conic-certificates.md": "15078ff04836144714bf517f3609442b0663d3a22932a9d47e94943111548dfe",
}
SCOPE = (*PINS, TEST, DOC)
THREADS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
OLD_REPLAY = Path(
    "/hf3fs-jd/data/tmp/openopt-release-20261010/conic-types-repair/"
    "unchanged-independent-v1/replay-auvgoot7/replay.json"
)
OLD_REPLAY_PIN = "1c1f6b73a8c19dd902bae71f9275f743f34168e557689d83ee75d2918972f0ba"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    assert json.loads(Path(path).read_bytes()) == value


def f(value):
    return value if isinstance(value, Fraction) else Fraction.from_float(float(value))


def dot(left, right):
    assert len(left) == len(right)
    return sum((f(a) * f(b) for a, b in zip(left, right)), Fraction(0))


def membership(cones, vector, dual=False):
    """Original Euclidean cone formulas, not producer helpers or projections."""
    assert sum(c.dimension for c in cones) == len(vector)
    offset = 0
    for cone in cones:
        part = [f(v) for v in vector[offset:offset + cone.dimension]]
        offset += cone.dimension
        if type(cone) is Zero:
            valid = dual or all(v == 0 for v in part)
        elif type(cone) is Nonnegative:
            valid = all(v >= 0 for v in part)
        elif type(cone) is SOC:
            valid = part[0] >= 0 and part[0] ** 2 >= sum(v ** 2 for v in part[1:])
        elif type(cone) is RSOC:
            valid = (part[0] >= 0 and part[1] >= 0
                     and 2 * part[0] * part[1] >= sum(v ** 2 for v in part[2:]))
        else:
            raise AssertionError("unsupported independent cone")
        if not valid:
            return False
    return True


def oracle(p, kind, witness, atol=1e-8, rtol=1e-8):
    """Exact identities on represented original data and a strict signed margin."""
    P, q, A, b = [np.asarray(getattr(p, key)) for key in ("P", "q", "A", "b")]
    if kind == "farkas":
        vector = witness["y"]
        checks = {"dual_cone": membership(p.cones, vector, True)}
        checks.update({f"Aty_{j}": dot(column, vector) == 0
                       for j, column in enumerate(A.T)})
        products = [f(a) * f(v) for a, v in zip(b, vector)]
    else:
        vector = witness["d"]
        checks = {f"Pd_{i}": dot(row, vector) == 0 for i, row in enumerate(P)}
        checks["recession_cone"] = membership(p.cones, [-dot(row, vector) for row in A])
        checks["base_present"] = witness.get("x0") is not None and witness.get("s0") is not None
        if checks["base_present"]:
            point, slack = witness["x0"], witness["s0"]
            checks["base_cone"] = membership(p.cones, slack)
            checks.update({f"base_{i}": dot(row, point) + f(s) == f(rhs)
                           for i, (row, s, rhs) in enumerate(zip(A, slack, b))})
        products = [f(a) * f(v) for a, v in zip(q, vector)]
    signed = sum(products, Fraction(0))
    margin = f(atol) + f(rtol) * sum(map(abs, products), Fraction(0))
    checks["strict_margin"] = signed < -margin
    return {"passed": all(checks.values()), "checks": checks,
            "signed": str(signed), "margin": str(margin)}


def model(P, q, A, b, cones):
    """Only numeric mathematical fixtures use this explicitly converting builder."""
    n = len(q)
    return ConicProblem(np.array(P, dtype=np.float64).reshape(n, n),
                        np.array(q, dtype=np.float64),
                        np.array(A, dtype=np.float64).reshape(len(b), n),
                        np.array(b, dtype=np.float64), tuple(cones))


def examples():
    cones = (Zero(1), Nonnegative(1), SOC(3), RSOC(3))
    product_ray = [0., 1., 5., 3., 4., 1., 2., 2.]
    return {
        "farkas-coupled-zero": (
            model([[0.]], [0.], [[2.], [-4.]], [1., -3.], [Zero(2)]),
            "farkas", {"y": [2., 1.]}),
        "farkas-nonnegative": (
            model([[0.]], [0.], [[0.], [0.]], [-2., 0.], [Nonnegative(2)]),
            "farkas", {"y": [1., 0.]}),
        "farkas-soc": (
            model([[0.]], [0.], [[0.]] * 3, [4., 0., 5.], [SOC(3)]),
            "farkas", {"y": [1., 0., -1.]}),
        "farkas-rsoc": (
            model([[0.]], [0.], [[0.]] * 3, [1., 1., 2.], [RSOC(3)]),
            "farkas", {"y": [1., 2., -2.]}),
        "farkas-product": (
            model([[0.]], [0.], [[0.]] * 8, [0., -2.] + [0.] * 6, cones),
            "farkas", {"y": [0., 1.] + [0.] * 6}),
        "ray-empty-lp": (
            model([[0.]], [-2.], [], [], []),
            "ray", {"d": [1.], "x0": [0.], "s0": []}),
        "ray-nonnegative": (
            model([[0.]], [-2.], [[-1.]], [0.], [Nonnegative(1)]),
            "ray", {"d": [1.], "x0": [3.], "s0": [3.]}),
        "ray-equality-qp": (
            model([[0., 0.], [0., 2.]], [-3., 2.], [[0., 1.]], [3.], [Zero(1)]),
            "ray", {"d": [1., 0.], "x0": [2., 3.], "s0": [0.]}),
        "ray-soc": (
            model(np.zeros((3, 3)), [-1., 0., 0.], -np.eye(3), [0.] * 3, [SOC(3)]),
            "ray", {"d": [5., 3., 4.], "x0": [0.] * 3, "s0": [0.] * 3}),
        "ray-rsoc": (
            model(np.zeros((3, 3)), [-1., 0., 0.], -np.eye(3), [0.] * 3, [RSOC(3)]),
            "ray", {"d": [1., 2., 2.], "x0": [0.] * 3, "s0": [0.] * 3}),
        "ray-rank-one-qp": (
            model([[4., 6.], [6., 9.]], [-2., 0.], [], [], []),
            "ray", {"d": [3., -2.], "x0": [0., 1.], "s0": []}),
        "ray-product": (
            model(np.zeros((8, 8)), [0., -1.] + [0.] * 6, -np.eye(8), [0.] * 8, cones),
            "ray", {"d": product_ray, "x0": product_ray[:], "s0": product_ray[:]}),
    }


def invoke(p, kind, witness, **settings):
    function = (directions.check_conic_infeasibility_direction if kind == "farkas"
                else directions.check_conic_unbounded_direction)
    return function(p, **witness, **settings)


def typed(value):
    """Preserve raw scalar types separately from materialized numeric bytes."""
    if isinstance(value, torch.Tensor):
        result = {"type": "torch.Tensor", "dtype": str(value.dtype),
                  "device": str(value.device), "layout": str(value.layout),
                  "shape": list(value.shape), "requires_grad": value.requires_grad}
        if value.device.type == "cpu" and value.layout == torch.strided:
            result["bytes"] = value.detach().numpy().tobytes().hex()
        return result
    if isinstance(value, np.ndarray):
        result = {"type": "numpy.ndarray", "dtype": value.dtype.str, "shape": list(value.shape)}
        if value.dtype.hasobject:
            result["values"] = [typed(v) for v in value.flat]
        else:
            result["bytes"] = value.tobytes().hex()
        return result
    if isinstance(value, (bool, np.bool_)):
        return {"type": type(value).__module__ + "." + type(value).__name__, "value": bool(value)}
    if isinstance(value, (list, tuple)):
        return {"type": type(value).__name__, "values": [typed(v) for v in value]}
    if isinstance(value, dict):
        return {key: typed(v) for key, v in value.items()}
    if isinstance(value, (float, np.floating)):
        return {"type": type(value).__name__, "hex": float(value).hex()}
    if isinstance(value, (int, np.integer, complex, np.complexfloating)):
        return {"type": type(value).__name__, "value": str(value)}
    if value is None:
        return None
    raise TypeError(f"unhandled review snapshot type: {type(value).__name__}")


def snapshot(p, witness):
    return {"model": {key: typed(getattr(p, key)) for key in ("P", "q", "A", "b")},
            "cones": [{"type": type(c).__name__, "dimension": c.dimension} for c in p.cones],
            "witness": typed(witness)}


def observe(p, kind, witness, request, tmp_path, expected, category, **settings):
    before = snapshot(p, witness)
    try:
        evidence = invoke(p, kind, witness, **settings)
    except (ValueError, RuntimeError, TypeError) as error:
        result = {"outcome": "REJECTED", "error_type": type(error).__name__, "reason": str(error)}
    else:
        result = {"outcome": "RETURNED", "evidence": evidence.to_dict()}
    after = snapshot(p, witness)
    value = {"nodeid": request.node.nodeid, "category": category, "expected": expected,
             "kind": kind, "before": before, "after": after, "settings": typed(settings),
             "result": result}
    path = tmp_path / "observation.json"
    write_json(path, value)
    request.node.user_properties.extend([
        ("review_observation_path", str(path)), ("review_observation_sha256", sha(path)),
    ])
    assert before == after, "checker mutated original caller bytes/types"
    if expected == "REJECTED":
        assert result["outcome"] == "REJECTED", value
    else:
        assert result["outcome"] == "RETURNED", value
        assert result["evidence"]["passed"] is (expected == "PASS"), value
        assert result["evidence"]["checks"]
        assert "host CPU verification" in result["evidence"]["qualification"]
    return value


@pytest.mark.parametrize("name", tuple(examples()))
def test_original_fraction_positive_and_prepared_bytes(name, request, tmp_path):
    p, kind, witness = examples()[name]
    exact = oracle(p, kind, witness)
    assert exact["passed"] is True and len(examples()) == 12
    value = observe(p, kind, witness, request, tmp_path, "PASS", "fraction-positive")
    assert all(value["result"]["evidence"]["checks"].values())


@pytest.mark.parametrize("name", tuple(examples()))
def test_each_original_wrong_sign_is_not_evidence(name, request, tmp_path):
    p, kind, good = examples()[name]
    key = "y" if kind == "farkas" else "d"
    witness = {**good, key: [-v for v in good[key]]}
    assert oracle(p, kind, witness)["passed"] is False
    observe(p, kind, witness, request, tmp_path, "FAIL", "fraction-negative")


@pytest.mark.parametrize("field", ["P", "q", "A", "b", "y", "d", "x0", "s0"])
@pytest.mark.parametrize("boolean", [True, False, np.bool_(True), np.bool_(False)],
                         ids=["py-true", "py-false", "np-true", "np-false"])
def test_raw_mixed_boolean_reject_before_conversion(field, boolean, request, tmp_path):
    # Do not call model() on the raw mutated field: that was the old q harness bug.
    p = model(np.zeros((2, 2)), [-1., 0.], np.zeros((2, 2)), [0., 0.], [Zero(2)])
    kind, witness = "ray", {"d": [1., 0.], "x0": [0., 0.], "s0": [0., 0.]}
    if field in ("y", "b"):
        p = replace(p, b=np.array([-2., 0.]))
        kind, witness = "farkas", {"y": [1., 0.]}
    if field in ("P", "A"):
        p = replace(p, **{field: [[0., boolean], [0., 0.]]})
    elif field in ("q", "b"):
        p = replace(p, **{field: [-1., boolean]})
    else:
        witness[field] = [1., boolean] if field in ("y", "d") else [0., boolean]
    value = observe(p, kind, witness, request, tmp_path, "REJECTED", "raw-bool")
    assert "boolean" in value["result"]["reason"]


@pytest.mark.parametrize("form", ["tuple", "nested-list", "np-array", "torch-tensor", "nested-tuple"])
def test_nested_boolean_is_not_hidden_by_outer_numeric_sequence(form, request, tmp_path):
    p = model(np.zeros((2, 2)), [-1., 0.], [], [], [])
    values = {
        "tuple": (1., False),
        "nested-list": [1., [False]],
        "np-array": [1., np.array([False])],
        "torch-tensor": [1., torch.tensor([False])],
        "nested-tuple": [1., ((np.bool_(False),),)],
    }
    value = observe(p, "ray", {"d": values[form], "x0": [0., 0.], "s0": []},
                    request, tmp_path, "REJECTED", "nested-bool")
    assert "boolean" in value["result"]["reason"]


@pytest.mark.parametrize("field", ["P", "A"])
@pytest.mark.parametrize("form", ["list", "tuple", "np-array", "torch-tensor"])
def test_nested_model_boolean_rejects_at_original_field(field, form, request, tmp_path):
    p = model(np.zeros((2, 2)), [-1., 0.], np.zeros((2, 2)), [0., 0.], [Zero(2)])
    row = {"list": [0., False], "tuple": (0., False),
           "np-array": np.array([False, False]), "torch-tensor": torch.tensor([False, False])}[form]
    p = replace(p, **{field: [[0., 0.], row]})
    value = observe(p, "ray", {"d": [1., 0.], "x0": [0., 0.], "s0": [0., 0.]},
                    request, tmp_path, "REJECTED", "nested-model-bool")
    assert value["result"]["reason"].startswith(field)
    assert "boolean" in value["result"]["reason"]


@pytest.mark.parametrize("field", ["atol", "rtol"])
@pytest.mark.parametrize("boolean", [True, False, np.bool_(True), np.bool_(False)],
                         ids=["py-true", "py-false", "np-true", "np-false"])
def test_boolean_tolerances_fail_loud(field, boolean, request, tmp_path):
    p, kind, witness = examples()["ray-empty-lp"]
    value = observe(p, kind, witness, request, tmp_path, "REJECTED", "bool-tolerance",
                    **{field: boolean})
    assert "tolerances cannot be booleans" in value["result"]["reason"]


@pytest.mark.parametrize("form", ["list", "tuple", "numpy-f64", "torch-f64", "exact-int"])
def test_numeric_original_model_and_cpu_double_remain_accepted(form, request, tmp_path):
    p, kind, witness = examples()["ray-equality-qp"]
    def convert(value):
        if form == "numpy-f64":
            return np.array(value, dtype=np.float64)
        if form == "torch-f64":
            return torch.tensor(value, dtype=torch.float64, requires_grad=True)
        if form == "exact-int":
            return np.array(value, dtype=np.int64)
        values = np.asarray(value).tolist()
        return tuple(tuple(v) if isinstance(v, list) else v for v in values) if form == "tuple" else values
    p = replace(p, **{key: convert(getattr(p, key)) for key in ("P", "q", "A", "b")})
    witness = {key: convert(value) for key, value in witness.items()}
    observe(p, kind, witness, request, tmp_path, "PASS", "numeric-compatibility")
    if form == "torch-f64":
        assert all(getattr(p, key).grad is None for key in ("P", "q", "A", "b"))
        assert all(v.grad is None and v.device == torch.device("cpu") for v in witness.values())


@pytest.mark.parametrize("field", ["q", "b", "y"])
def test_precast_bool_provenance_is_lost_but_raw_input_is_rejected(field, request, tmp_path):
    raw = [-1., False] if field in ("q", "b") else [True, 0.]
    materialized = np.array(raw, dtype=np.float64)
    intended = np.array([-1., 0.] if field in ("q", "b") else [1., 0.], dtype=np.float64)
    assert typed(raw) != typed(intended)
    assert materialized.dtype == np.float64 and materialized.tobytes() == intended.tobytes()
    # Identical represented bytes cannot encode which of these two callers cast them.
    assert typed(materialized) == typed(intended)
    if field == "q":
        p = model(np.zeros((2, 2)), [-1., 0.], [], [], [])
        kind, witness = "ray", {"d": [1., 0.], "x0": [0., 0.], "s0": []}
    else:
        p = model([[0.]], [0.], [[0.], [0.]], [-1., 0.], [Nonnegative(2)])
        kind, witness = "farkas", {"y": [1., 0.]}
    numeric_p = replace(p, **{field: materialized}) if field != "y" else p
    numeric_witness = {**witness, "y": materialized} if field == "y" else witness
    assert oracle(numeric_p, kind, numeric_witness)["passed"]
    numeric_before = snapshot(numeric_p, numeric_witness)
    accepted = invoke(numeric_p, kind, numeric_witness).to_dict()
    assert accepted["passed"] is True
    assert snapshot(numeric_p, numeric_witness) == numeric_before
    raw_p = replace(p, **{field: raw}) if field != "y" else p
    raw_witness = {**witness, "y": raw} if field == "y" else witness
    observed = observe(raw_p, kind, raw_witness, request, tmp_path, "REJECTED", "cast-boundary")
    path = tmp_path / "cast-boundary.json"
    write_json(path, {"field": field, "raw": typed(raw), "materialized": typed(materialized),
                      "independent_numeric": typed(intended), "numeric_evidence": accepted,
                      "numeric_before": numeric_before,
                      "numeric_after": snapshot(numeric_p, numeric_witness),
                      "raw_result": observed["result"], "historical_fail_requalified": False})
    request.node.user_properties.extend([
        ("cast_boundary_path", str(path)), ("cast_boundary_sha256", sha(path)),
    ])


@pytest.mark.parametrize("kind", ["farkas", "ray"])
def test_tiny_identity_residual_never_replaces_original_exact_equation(kind, request, tmp_path):
    if kind == "farkas":
        p, witness = model([[0.]], [0.], [[1e-20]], [-1.], [Zero(1)]), {"y": [1.]}
        key = "Aty_0"
    else:
        p, witness = model([[1e-20]], [-1.], [], [], []), {"d": [1.], "x0": [0.], "s0": []}
        key = "Pd_0"
    assert oracle(p, kind, witness)["passed"] is False
    value = observe(p, kind, witness, request, tmp_path, "FAIL", "exact-identity")
    checks = value["result"]["evidence"]["checks"]
    assert checks[key + "_residual"] is True and checks[key + "_identity"] is False


@pytest.mark.parametrize("case", ["missing-x", "missing-s", "base-equation", "base-cone", "one-ulp-soc",
                                 "one-ulp-rsoc", "tiny-negative-head"])
def test_base_and_original_cone_math_negatives(case, request, tmp_path):
    p, kind, good = examples()["ray-nonnegative"]
    witness = {key: list(value) for key, value in good.items()}
    if case.startswith("missing"):
        witness.pop("x0" if case == "missing-x" else "s0")
    elif case == "base-equation":
        witness["s0"] = [3.25]
    elif case == "base-cone":
        witness.update(x0=[-1.], s0=[-1.])
    else:
        p, kind, good = examples()["ray-soc" if case == "one-ulp-soc" else "ray-rsoc"]
        witness = {key: list(value) for key, value in good.items()}
        witness["d"] = ([np.nextafter(5., 0.), 3., 4.] if case == "one-ulp-soc"
                        else [np.nextafter(1., 0.), 2., 2.] if case == "one-ulp-rsoc"
                        else [-1e-300, 1e300, 0.])
    assert oracle(p, kind, witness)["passed"] is False
    observe(p, kind, witness, request, tmp_path, "FAIL", "base-cone-negative")


@pytest.mark.parametrize("kind", ["farkas", "ray"])
@pytest.mark.parametrize("slope", [-1e-8, np.nextafter(-1e-8, -math.inf), -1e-20, -1.],
                         ids=["equal-margin", "past-margin", "tiny-negative", "negative"])
def test_signed_strict_margin_matches_fraction_formula(kind, slope, request, tmp_path):
    p = (model([[0.]], [0.], [[0.]], [slope], [Zero(1)]) if kind == "farkas"
         else model([[0.]], [slope], [], [], []))
    witness = {"y": [1.]} if kind == "farkas" else {"d": [1.], "x0": [0.], "s0": []}
    expected = oracle(p, kind, witness, rtol=0.)["passed"]
    value = observe(p, kind, witness, request, tmp_path, "PASS" if expected else "FAIL",
                    "strict-margin", rtol=0.)
    metrics = value["result"]["evidence"]["metrics"]
    key = "bty" if kind == "farkas" else "qtd"
    assert f(metrics[key + "_lower"]) <= f(slope) <= f(metrics[key + "_upper"])


@pytest.mark.parametrize("case", ["nonfinite", "float32", "complex", "inexact-int", "overflow",
                                 "cancelled-overflow", "underflow", "n-budget", "nonsymmetric",
                                 "indefinite", "zero-pivot-indefinite"])
def test_unavailable_data_and_arithmetic_never_return_green(case, request, tmp_path):
    p, kind, witness = examples()["ray-empty-lp"]
    if case in ("nonfinite", "float32", "complex", "inexact-int"):
        bad = {"nonfinite": np.array([-np.inf]), "float32": np.array([-1.], dtype=np.float32),
               "complex": np.array([-1.+0j]), "inexact-int": np.array([2**53 + 1])}[case]
        p = replace(p, q=bad)
    elif case == "overflow":
        p, witness = replace(p, q=[-1e308]), {**witness, "d": [2.]}
    elif case == "underflow":
        p, witness = replace(p, q=[-1e-300]), {**witness, "d": [1e-300]}
    elif case == "cancelled-overflow":
        p = model(np.zeros((2, 2)), [1e308, -1e308], [], [], [])
        witness = {"d": [2., 2.], "x0": [0., 0.], "s0": []}
    elif case == "n-budget":
        p = model(np.zeros((17, 17)), [-1.] * 17, [], [], [])
        witness = {"d": [1.] * 17, "x0": [0.] * 17, "s0": []}
    else:
        P = {"nonsymmetric": [[1., 1.], [0., 1.]],
             "indefinite": [[1., 1.], [1., np.nextafter(1., 0.)]],
             "zero-pivot-indefinite": [[0., 1e-200], [1e-200, 0.]]}[case]
        if case != "nonsymmetric":
            assert f(P[0][0]) * f(P[1][1]) - f(P[0][1]) * f(P[1][0]) < 0
        p = model(P, [-1., 0.], [], [], [])
        witness = {"d": [1., 0.], "x0": [0., 0.], "s0": []}
    observe(p, kind, witness, request, tmp_path, "REJECTED", "unavailable")


@pytest.mark.parametrize("kind", ["farkas", "ray"])
@pytest.mark.parametrize("device", ["cuda", "cuda:0", "meta", "cpu:1"])
def test_unsupported_device_rejects_before_any_input_access(kind, device):
    class Unreadable:
        def __array__(self, *args, **kwargs):
            raise AssertionError("input read before strict device rejection")
    p = ConicProblem(Unreadable(), Unreadable(), Unreadable(), Unreadable(), ())
    witness = {"y": Unreadable()} if kind == "farkas" else {"d": Unreadable()}
    with pytest.raises((ValueError, RuntimeError), match="cpu|CUDA|host-only"):
        invoke(p, kind, witness, device=device)


@pytest.mark.parametrize("field", ["P", "q", "A", "b", "d", "x0", "s0"])
def test_meta_tensor_no_implicit_cpu_transfer(field, request, tmp_path):
    p, kind, witness = examples()["ray-nonnegative"]
    shape = getattr(p, field).shape if hasattr(p, field) else np.asarray(witness[field]).shape
    value = torch.empty(shape, dtype=torch.float64, device="meta")
    if hasattr(p, field):
        p = replace(p, **{field: value})
    else:
        witness = {**witness, field: value}
    result = observe(p, kind, witness, request, tmp_path, "REJECTED", "no-transfer")
    assert "implicit device transfer" in result["result"]["reason"]


def test_independent_checks_never_consult_solver_default_or_eigensolver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("independent direction audit consulted shared numerical oracle/solver")
    for name in ("solve_conic", "check_conic_candidate", "check_cone_feasibility"):
        monkeypatch.setattr(conic, name, forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    for p, kind, witness in examples().values():
        assert invoke(p, kind, witness).passed and oracle(p, kind, witness)["passed"]


def test_native_prepared_original_bytes_copies_are_isolated_without_solve(request, tmp_path, monkeypatch):
    from openopt.native_conic import NativeConicConfig, PreparedNativeConic, prepare_native_conic
    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-only byte probe executed a native solve")
    monkeypatch.setattr(PreparedNativeConic, "solve", forbidden)
    p, _, _ = examples()["ray-equality-qp"]
    before = snapshot(p, {})
    prepared = prepare_native_conic(p, NativeConicConfig(enabled=True, device="cpu"))
    frozen = snapshot(prepared.original_problem, {})
    # Preparing a factor is not a solve or a new native type-contract claim.
    assert all(torch.equal(getattr(prepared.original_problem, k),
                           torch.tensor(getattr(p, k), dtype=torch.float64))
               for k in ("P", "q", "A", "b"))
    for key in ("P", "q", "A", "b"):
        getattr(p, key).flat[0] += 7.
        getattr(prepared.original_problem, key).fill_(19.)
    after = snapshot(prepared.original_problem, {})
    assert frozen == after and snapshot(p, {}) != before
    assert prepared._factor.device == torch.device("cpu")
    path = tmp_path / "prepared-bytes.json"
    write_json(path, {"caller_before": before, "frozen": frozen, "after": after,
                      "solver_calls": 0, "native_boolean_contract_reviewed": False})
    request.node.user_properties.extend([
        ("prepared_bytes_path", str(path)), ("prepared_bytes_sha256", sha(path)),
    ])


def loaded_binding():
    result = {}
    for module, relative in ((directions, "src/openopt/conic_certificates.py"),
                             (conic, "src/openopt/conic.py")):
        path = Path(module.__file__).resolve(strict=True)
        assert path == ROOT / relative and sha(path) == PINS[relative]
        result[relative] = {"path": str(path), "sha256": sha(path)}
    for function in (directions.check_conic_infeasibility_direction,
                     directions.check_conic_unbounded_direction):
        assert Path(function.__code__.co_filename).resolve() == ROOT / "src/openopt/conic_certificates.py"
        assert function.__module__ == directions.__name__
    return result


def validate_cast_record(value):
    field = value["field"]
    if field not in ("q", "b", "y"):
        raise ValueError("cast field outside declared original scope")
    raw = [-1., False] if field in ("q", "b") else [True, 0.]
    numeric = np.array([-1., 0.] if field in ("q", "b") else [1., 0.], dtype=np.float64)
    if (value["raw"] != typed(raw) or value["materialized"] != typed(numeric)
            or value["independent_numeric"] != typed(numeric)
            or value["numeric_before"] != value["numeric_after"]
            or value["numeric_evidence"]["passed"] is not True
            or value["raw_result"]["outcome"] != "REJECTED"
            or value["historical_fail_requalified"] is not False):
        raise ValueError("raw/materialized cast provenance boundary mismatch")


@pytest.mark.parametrize("mutation", ["raw-type", "numeric-bytes", "raw-returned", "requalify", "mutated"])
def test_cast_boundary_metadata_negative_controls(mutation):
    # Metadata-only validator fixtures, never actual checker execution evidence.
    raw, numeric = [-1., False], np.array([-1., 0.], dtype=np.float64)
    value = {"field": "q", "raw": typed(raw), "materialized": typed(numeric),
             "independent_numeric": typed(numeric), "numeric_before": {}, "numeric_after": {},
             "numeric_evidence": {"passed": True}, "raw_result": {"outcome": "REJECTED"},
             "historical_fail_requalified": False}
    validate_cast_record(value)
    if mutation == "raw-type":
        value["raw"] = typed([-1., 0.])
    elif mutation == "numeric-bytes":
        value["materialized"] = typed(np.array([-1., 1.], dtype=np.float64))
    elif mutation == "raw-returned":
        value["raw_result"]["outcome"] = "RETURNED"
    elif mutation == "requalify":
        value["historical_fail_requalified"] = True
    else:
        value["numeric_after"] = {"changed": True}
    with pytest.raises(ValueError, match="provenance boundary"):
        validate_cast_record(value)


def test_new_frozen_producer_binding_and_original_12_case_receipt(request, tmp_path):
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert all(os.environ[key] == "1" for key in THREADS)
    assert all(sha(ROOT / key) == pin for key, pin in PINS.items())
    cases = {}
    for name, (p, kind, witness) in examples().items():
        before = snapshot(p, witness)
        value = invoke(p, kind, witness).to_dict()
        assert value["passed"] is True and oracle(p, kind, witness)["passed"] is True
        assert snapshot(p, witness) == before
        cases[name] = {"input": before, "oracle": oracle(p, kind, witness), "evidence": value}
    path = tmp_path / "original-math.json"
    write_json(path, {"owner_commit": OWNER, "source_sha256": {k: sha(ROOT / k) for k in SCOPE},
                      "loaded": loaded_binding(), "cases": cases, "solver_calls": 0,
                      "old_09751dc_fail_requalified": False, "gpu_execution": "UNVERIFIED",
                      "commercial": "UNVERIFIED", "certified": False})
    request.node.user_properties.extend([
        ("original_math_path", str(path)), ("original_math_sha256", sha(path)),
    ])


def membership_check(selected, actual):
    if not selected or len(selected) != len(set(selected)):
        raise ValueError("empty/duplicate collection")
    if len(actual) != len(set(actual)) or set(selected) != set(actual):
        raise ValueError("collection/JUnit bidirectional membership mismatch")
    return {"missing": [], "extra": [], "duplicates": []}


@pytest.mark.parametrize("mutation", ["empty", "drop", "extra", "duplicate", "coordinated-drop"])
def test_membership_controls_reject_missing_duplicate_and_coordinated_drop(mutation):
    expected = [TEST + "::a", TEST + "::b"]
    selected, actual = expected[:], expected[:]
    if mutation == "empty":
        selected = []
    elif mutation == "drop":
        actual.pop()
    elif mutation == "extra":
        actual.append(TEST + "::c")
    elif mutation == "duplicate":
        actual.append(actual[0])
    else:
        selected.pop()
        actual.pop()
    if mutation == "coordinated-drop":
        # Trust the independent live collection, not two mutually resealed truncated lists.
        with pytest.raises(ValueError):
            membership_check(expected, actual)
    else:
        with pytest.raises(ValueError):
            membership_check(selected, actual)


def environment(directory):
    env = os.environ.copy()
    env.update(PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]),
               PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
               CUDA_VISIBLE_DEVICES="-1", TMPDIR=str(directory))
    env.update({key: "1" for key in THREADS})
    return env


def audit_artifacts(directory):
    """Full live collection, typed observations, original math and clean source replay."""
    from scripts.verify import check_source_consistency, source_snapshot
    directory = Path(directory).resolve(strict=True)
    assert directory.is_relative_to(Path("/3fs/research3fs/data/tmp"))
    assert not directory.is_relative_to(ROOT)
    output = directory / "independent-review.json"
    assert not output.exists(), "immutable receipt: use a new run directory"
    before = source_snapshot(ROOT)
    receipt = {"schema_version": 1, "verdict": "FAIL", "owner_commit": OWNER,
               "review_commit": before["commit"], "checked_at": datetime.now(timezone.utc).isoformat(),
               "source": before, "solver_calls": 0, "gpu_execution": "UNVERIFIED",
               "commercial": "UNVERIFIED", "certified": False,
               "old_09751dc_fail_requalified": False, "scope": [TEST, DOC]}
    try:
        assert before["clean"], "independent acceptance requires clean frozen source"
        assert all(before["sha256"][key] == pin for key, pin in PINS.items())
        receipt["loaded"] = loaded_binding()
        verification = json.loads((directory / "verification.json").read_bytes())
        tree = ET.parse(directory / "pytest.xml").getroot()
        cases = list(tree.iter("testcase"))
        actual = [c.get("classname", "").replace(".", "/") + ".py::" + c.get("name", "")
                  for c in cases]
        command = [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
                   "-o", f"cache_dir={directory / 'collection-cache'}", TEST]
        result = subprocess.run(command, cwd=ROOT, env=environment(directory),
                                capture_output=True, text=True, timeout=45, check=False)
        with (directory / "collection.log").open("x", encoding="utf-8") as stream:
            stream.write(result.stdout + result.stderr)
        assert result.returncode == 0, "live independent collection failed"
        selected = [line.strip() for line in result.stdout.splitlines() if line.startswith(TEST + "::")]
        differences = membership_check(selected, actual)
        write_json(directory / "collection.json", {"nodeids": selected, "returncode": result.returncode})
        counts = {"tests": len(cases), **{
            name: sum(c.find(tag) is not None for c in cases)
            for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
        }}
        receipt.update(tests=counts, collection_nodeids=selected, junit_nodeids=actual,
                       membership=differences)
        for suite in tree.iter("testsuite"):
            leaves = suite.findall("testcase")
            if not suite.findall("testsuite"):
                assert int(suite.get("tests")) == len(leaves)
                assert all(int(suite.get(name)) == sum(c.find(tag) is not None for c in leaves)
                           for name, tag in (("failures", "failure"), ("errors", "error"),
                                             ("skipped", "skipped")))
        assert verification["targets"] == [TEST] and verification["coverage"] == "explicit_targets"
        assert verification["source"]["clean"]
        receipt["source_consistency"] = check_source_consistency(verification["source"], before)
        assert receipt["source_consistency"]["verdict"] == "PASS"
        if "tests" in verification:
            assert verification["tests"] == counts
        start, finish = [datetime.fromisoformat(verification[key]) for key in ("started_at", "finished_at")]
        elapsed = (finish - start).total_seconds()
        assert start.tzinfo and finish.tzinfo and 0 < elapsed <= 170
        receipt["verification_wall_seconds"] = elapsed
        observations, extras = [], []
        for c in cases:
            props = {p.get("name"): p.get("value") for p in c.findall("properties/property")}
            for prefix in ("review_observation", "cast_boundary", "prepared_bytes", "original_math"):
                if prefix + "_path" not in props:
                    continue
                path = Path(props[prefix + "_path"]).resolve(strict=True)
                assert path.is_relative_to(directory) and sha(path) == props[prefix + "_sha256"]
                value = json.loads(path.read_bytes())
                item = {"path": str(path), "sha256": sha(path), "value": value}
                (observations if prefix == "review_observation" else extras).append(item)
        category_counts = dict(Counter(o["value"]["category"] for o in observations))
        receipt.update(observation_categories=category_counts, observations=observations, extras=extras)
        assert category_counts == {
            "fraction-positive": 12, "fraction-negative": 12, "raw-bool": 32, "nested-bool": 5,
            "nested-model-bool": 8, "bool-tolerance": 8, "numeric-compatibility": 5,
            "cast-boundary": 3, "exact-identity": 2, "base-cone-negative": 7,
            "strict-margin": 8, "unavailable": 11, "no-transfer": 7,
        }
        false_accepts = []
        for item in observations:
            value = item["value"]
            assert value["before"] == value["after"]
            result = value["result"]
            expected = value["expected"]
            matches = (result["outcome"] == "REJECTED" if expected == "REJECTED"
                       else result["outcome"] == "RETURNED"
                       and result["evidence"]["passed"] is (expected == "PASS"))
            if not matches:
                false_accepts.append(value["nodeid"])
        receipt["false_accepts"] = false_accepts
        assert not false_accepts
        math_items = [item for item in extras if "cases" in item["value"]]
        assert len(math_items) == 1
        math_value = math_items[0]["value"]
        assert math_value["source_sha256"] == {k: sha(ROOT / k) for k in SCOPE}
        assert math_value["loaded"] == loaded_binding()
        expected_math = {}
        for name, (p, kind, witness) in examples().items():
            expected_math[name] = {"input": snapshot(p, witness), "oracle": oracle(p, kind, witness),
                                   "evidence": invoke(p, kind, witness).to_dict()}
        assert math_value["owner_commit"] == OWNER and math_value["cases"] == expected_math
        assert math_value["old_09751dc_fail_requalified"] is False
        assert len(extras) == 5
        casts = [item["value"] for item in extras if "materialized" in item["value"]]
        assert len(casts) == 3 and {value["field"] for value in casts} == {"q", "b", "y"}
        for value in casts:
            validate_cast_record(value)
        prepared = [item["value"] for item in extras if "frozen" in item["value"]]
        assert len(prepared) == 1 and prepared[0]["frozen"] == prepared[0]["after"]
        assert prepared[0]["solver_calls"] == 0 and prepared[0]["native_boolean_contract_reviewed"] is False
        assert math_value["solver_calls"] == 0 and math_value["certified"] is False
        receipt["original_math_cases_replayed"] = len(expected_math)
        receipt["historical_replay"] = {"path": str(OLD_REPLAY), "sha256": sha(OLD_REPLAY)}
        assert sha(OLD_REPLAY) == OLD_REPLAY_PIN
        old = json.loads(OLD_REPLAY.read_bytes())
        assert old["verdict"] == "FAIL" and old["tests"] == {
            "tests": 120, "failures": 1, "errors": 0, "skipped": 0,
        }
        assert next(o for o in old["input_observations"] if o["label"] == "mixed-bool-q")["outcome"] == "RETURNED"
        receipt["historical_replay"]["verdict"] = "FAIL_RETAINED_119_PASS_1_FAIL"
        assert verification["verdict"] == "PASS" and counts["tests"] > 0
        assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
        assert verification["source_consistency"]["verdict"] == "PASS"
        receipt["verdict"] = "PASS"
    except Exception as error:
        receipt["first_failure"] = f"{type(error).__name__}: {error}"
    finally:
        after = source_snapshot(ROOT)
        receipt["replay_source_consistency"] = check_source_consistency(before, after)
        if not after["clean"] or receipt["replay_source_consistency"]["verdict"] != "PASS":
            receipt.update(verdict="FAIL", first_failure=receipt.get("first_failure", "replay source drift"))
        receipt["source_sha256"] = {key: sha(ROOT / key) for key in SCOPE}
        receipt["artifact_sha256"] = {
            name: sha(directory / name) for name in ("verification.json", "pytest.xml", "pytest.log",
                                                    "collection.log", "collection.json")
            if (directory / name).is_file()
        }
        write_json(output, receipt)
    print(f"CONIC_TYPES_INDEPENDENT verdict={receipt['verdict']} "
          f"tests={receipt.get('tests')} source={receipt['replay_source_consistency']['verdict']} "
          f"receipt={output}", flush=True)
    return 0 if receipt["verdict"] == "PASS" else 1


def run(artifact_root):
    """Logged subprocess, then strict independent replay; never hide an old FAIL."""
    root = artifact_root.resolve()
    assert root.is_relative_to(Path("/3fs/research3fs/data/tmp")) and not root.is_relative_to(ROOT)
    root.mkdir(parents=True, exist_ok=True)
    batch = Path(tempfile.mkdtemp(prefix="review-", dir=root))
    started = time.monotonic()
    with (batch / "run.log").open("x", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                [sys.executable, "-B", "scripts/verify.py", "tests", "--target", TEST,
                 "--artifact-dir", str(batch)], cwd=ROOT, env=environment(batch), stdout=log,
                stderr=subprocess.STDOUT, timeout=160, check=False,
            )
        except subprocess.TimeoutExpired as error:
            write_json(batch / "run-receipt.json", {
                "verdict": "FAIL", "first_failure": f"TimeoutExpired: {error}",
                "wall_seconds": time.monotonic() - started,
                "run_log_sha256": sha(batch / "run.log"),
            })
            raise
    directories = [path.parent for path in batch.glob("tests-*/verification.json")]
    assert len(directories) == 1, "missing/ambiguous actual verification"
    rc = audit_artifacts(directories[0])
    write_json(batch / "run-receipt.json", {
        "verdict": "PASS" if result.returncode == rc == 0 else "FAIL",
        "returncode": result.returncode, "audit_returncode": rc,
        "wall_seconds": time.monotonic() - started, "verification": str(directories[0]),
        "run_log_sha256": sha(batch / "run.log"),
        "independent_receipt_sha256": sha(directories[0] / "independent-review.json"),
    })
    return rc if rc else result.returncode


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit-artifacts", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args()
    if args.run and args.artifact_dir is None:
        parser.error("--run requires --artifact-dir")
    raise SystemExit(run(args.artifact_dir) if args.run else audit_artifacts(args.audit_artifacts))
