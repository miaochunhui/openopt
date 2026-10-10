"""Self-contained original-coordinate direction checks and negative controls.

No solver is called. Pytest freezes original models, directions, dependencies
and evidence in its external tmp directory. The CLI re-reads real receipts.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import sys
from xml.etree import ElementTree

import numpy as np
import pytest
import torch

from openopt.conic import ConicProblem, Nonnegative, RSOC, SOC, Zero
from openopt.conic_certificates import (
    check_conic_infeasibility_direction as infeasible,
    check_conic_unbounded_direction as unbounded,
    conic_direction_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PINS = [
    "src/openopt/conic_certificates.py", "tests/test_conic_certificates.py",
    "docs/research/conic-certificates.md", "src/openopt/conic.py",
    "src/openopt/certificates.py",
]


def constant(cone, b):
    return ConicProblem(np.zeros((1, 1)), np.zeros(1),
                        np.zeros((len(b), 1)), np.array(b, dtype=np.float64), (cone,))


def in_cone(cone, q):
    n = cone.dimension
    return ConicProblem(np.zeros((n, n)), np.array(q, dtype=np.float64),
                        -np.eye(n), np.zeros(n), (cone,))


def empty(P=None, q=None):
    q = [-1.] if q is None else q
    P = np.zeros((len(q), len(q))) if P is None else np.array(P, dtype=np.float64)
    return ConicProblem(P, np.array(q, dtype=np.float64),
                        np.empty((0, len(q))), np.empty(0), ())


def analytic_cases():
    nn = ConicProblem(np.zeros((1, 1)), np.zeros(1),
                      np.array([[1.], [-1.]]), np.array([0., -1.]), (Nonnegative(2),))
    equality = ConicProblem(np.diag([0., 1.]), np.array([-1., 0.]),
                            np.array([[0., 1.]]), np.ones(1), (Zero(1),))
    return {
        "farkas_equality": (constant(Zero(1), [1.]), "farkas", {"y": [-1.]}),
        "farkas_nonnegative": (nn, "farkas", {"y": [1., 1.]}),
        "farkas_soc": (constant(SOC(2), [0., 1.]), "farkas", {"y": [1., -1.]}),
        "farkas_rsoc": (constant(RSOC(3), [-1., 0., 0.]), "farkas", {"y": [1., 0., 0.]}),
        "unbounded_empty_lp": (empty(), "unbounded", {"d": [1.], "x0": [0.], "s0": []}),
        "unbounded_nonnegative": (in_cone(Nonnegative(1), [-1.]), "unbounded",
                                  {"d": [1.], "x0": [2.], "s0": [2.]}),
        "unbounded_soc_boundary": (in_cone(SOC(3), [-1., 0., 0.]), "unbounded",
                                   {"d": [5., 3., 4.], "x0": [5., 3., 4.], "s0": [5., 3., 4.]}),
        "unbounded_rsoc_boundary": (in_cone(RSOC(3), [-1., 0., 0.]), "unbounded",
                                    {"d": [1., 2., 2.], "x0": [1., 2., 2.], "s0": [1., 2., 2.]}),
        "unbounded_equality_qp": (equality, "unbounded", {"d": [1., 0.], "x0": [0., 1.], "s0": [0.]}),
        "unbounded_nondiagonal_singular": (empty([[1., 1.], [1., 1.]], [-1., 0.]), "unbounded",
                                           {"d": [1., -1.], "x0": [0., 0.], "s0": []}),
    }


@pytest.mark.parametrize("name", list(analytic_cases()))
def test_original_analytic_directions(name):
    p, kind, witness = analytic_cases()[name]
    evidence = (infeasible if kind == "farkas" else unbounded)(p, **witness)
    assert evidence.passed, evidence.to_dict()
    assert evidence.unavailable == {}
    assert "not an exact certificate" in evidence.qualification
    assert "host CPU" in evidence.qualification
    # Independently recompute the negative linear expression using original data.
    values, direction = (p.b, witness["y"]) if kind == "farkas" else (p.q, witness["d"])
    oracle = sum((Fraction.from_float(float(a)) * Fraction.from_float(float(b))
                  for a, b in zip(values, direction)), Fraction(0))
    assert oracle < 0
    key = "bty" if kind == "farkas" else "qtd"
    assert Fraction.from_float(evidence.metrics[f"{key}_lower"]) <= oracle
    assert Fraction.from_float(evidence.metrics[f"{key}_upper"]) >= oracle
    assert evidence.metrics[f"{key}_upper"] < -evidence.thresholds[key]


@pytest.mark.parametrize("y", [[-1., -1.], [0., 0.], [1., 2.], [1., -1.]])
def test_farkas_mutants_rejected(y):
    p, _, _ = analytic_cases()["farkas_nonnegative"]
    assert not infeasible(p, y).passed


def test_tiny_nonzero_At_y_cannot_be_accepted_by_residual_tolerance():
    p = constant(Zero(1), [-1.])
    p = replace(p, A=np.array([[1e-20]]))
    evidence = infeasible(p, [1.])
    assert evidence.checks["Aty_0_residual"]
    assert not evidence.checks["Aty_0_identity"] and not evidence.passed


@pytest.mark.parametrize("change", ["ray_sign", "base_x", "base_s", "base_outside", "cone_ray"])
def test_unbounded_mutants_rejected(change):
    p, _, w = analytic_cases()["unbounded_soc_boundary"]
    w = {k: list(v) for k, v in w.items()}
    if change == "ray_sign":
        w["d"] = [-v for v in w["d"]]
    elif change == "base_x":
        w["x0"][0] += 1.
    elif change == "base_s":
        w["s0"][0] += 1.
    elif change == "base_outside":
        w["x0"][0] = w["s0"][0] = 4.
    else:
        w["d"][0] = 4.
    assert not unbounded(p, **w).passed


@pytest.mark.parametrize("x0,s0", [(None, None), ([0.], None), (None, [])])
def test_dual_infeasibility_without_both_base_parts_is_not_unbounded(x0, s0):
    evidence = unbounded(empty(), [1.], x0=x0, s0=s0)
    assert not evidence.passed and not evidence.checks["base_point_present"]
    assert "base_point" in evidence.unavailable
    assert "not primal unboundedness" in evidence.unavailable["base_point"]


@pytest.mark.parametrize("curvature", [1e-20, 1e-300, np.nextafter(0., 1.)])
def test_tiny_positive_curvature_is_not_kernel(curvature):
    evidence = unbounded(empty([[curvature]]), [1.], x0=[0.], s0=[])
    assert evidence.checks["Pd_0_residual"]
    assert not evidence.checks["Pd_0_identity"] and not evidence.passed


@pytest.mark.parametrize("P", [
    [[-1e-300]], [[0., 1e-200], [1e-200, 1.]],
    [[1., 1.], [1., np.nextafter(1., 0.)]],
])
def test_indefinite_original_P_never_rounded_to_psd(P):
    n = len(P)
    with pytest.raises(ValueError, match="positive semidefinite"):
        unbounded(empty(P, [-1.] * n), [1.] * n, x0=[0.] * n, s0=[])


def test_exact_psd_singular_non_dd_supported_without_spectral_tolerance():
    v = np.array([1., 2., 3.])
    p = empty(np.outer(v, v), [-1., 0., 0.])
    assert unbounded(p, [2., -1., 0.], x0=[0.] * 3, s0=[]).passed


def test_nonpaired_exact_nullspace_and_positive_definite_boundaries():
    p = empty(np.ones((3, 3)), [-1., 0., 0.])
    assert unbounded(p, [3., -1., -2.], x0=[0.] * 3, s0=[]).passed
    p = empty([[2., 1.], [1., 2.]], [-1., 0.])
    assert not unbounded(p, [1., -1.], x0=[0., 0.], s0=[]).passed


def test_mixed_original_cone_order_and_base_rows():
    cones = (Zero(1), Nonnegative(1), SOC(2), RSOC(3))
    d = [0., 1., 1., 1., 1., 2., 2.]
    p = ConicProblem(np.zeros((7, 7)), np.array([0., -1., 0., 0., 0., 0., 0.]),
                     -np.eye(7), np.zeros(7), cones)
    assert unbounded(p, d, x0=[0.] * 7, s0=[0.] * 7).passed
    bad = list(d)
    bad[0] = 1e-20
    assert not unbounded(p, bad, x0=[0.] * 7, s0=[0.] * 7).passed


def test_infeasible_base_cannot_upgrade_valid_recession_to_unbounded():
    p = ConicProblem(np.zeros((1, 1)), np.array([-1.]),
                     np.zeros((1, 1)), np.array([-1.]), (Nonnegative(1),))
    e = unbounded(p, [1.], x0=[0.], s0=[-1.])
    assert e.checks["base_equation_0_identity"] and e.checks["qtd_strict_margin"]
    assert not e.checks["base_slack_cone_block_0_component_0_nonnegative"] and not e.passed


@pytest.mark.parametrize("seed", range(8))
def test_random_small_fraction_oracle_and_outward_margin(seed):
    rng = np.random.default_rng(seed)
    q = -rng.uniform(.1, 1., 4)
    d = rng.uniform(.1, 1., 4)
    p = empty(q=q)
    e = unbounded(p, d, x0=[0.] * 4, s0=[])
    assert e.passed
    products = [Fraction.from_float(float(a)) * Fraction.from_float(float(b)) for a, b in zip(q, d)]
    exact = sum(products, Fraction(0))
    margin = Fraction.from_float(1e-8) * (1 + sum(map(abs, products), Fraction(0)))
    assert Fraction.from_float(e.metrics["qtd_lower"]) <= exact <= Fraction.from_float(e.metrics["qtd_upper"])
    assert Fraction.from_float(e.thresholds["qtd"]) >= margin
    # Reverse ONLY the original objective, not the claimed evidence.
    assert not unbounded(replace(p, q=-q), d, x0=[0.] * 4, s0=[]).passed


def test_nonrepresentable_dot_has_distinct_outward_endpoints():
    e = unbounded(empty(q=[-.1]), [.3], x0=[0.], s0=[])
    exact = Fraction.from_float(-.1) * Fraction.from_float(.3)
    assert e.passed and e.metrics["qtd_lower"] < e.metrics["qtd_upper"]
    assert Fraction.from_float(e.metrics["qtd_lower"]) < exact < Fraction.from_float(e.metrics["qtd_upper"])


def test_full_symmetry_no_averaging_even_tiny_asymmetry():
    P = np.eye(2)
    P[0, 1] = 1e-300
    with pytest.raises(ValueError, match="full symmetric"):
        unbounded(empty(P, [-1., 0.]), [1., 0.])


@pytest.mark.parametrize("direction", [[1e-300, -1e300, 0.], [-1e-300, 1e300, 0.], [0., 1e300, 1.]])
def test_rsoc_tiny_or_negative_head_cannot_hide_behind_large_head(direction):
    p = constant(RSOC(3), [0., -1e-300, 0.])
    assert not infeasible(p, direction).passed


def test_rsoc_original_required_tiny_head_boundary_and_mutant():
    # 2 * 2^-100 * 2^100 == 2, and ||(1,1)||^2 == 2 exactly.
    a, b = 2. ** -100, 2. ** 100
    p = in_cone(RSOC(4), [-1., 0., 0., 0.])
    kwargs = {"x0": [0.] * 4, "s0": [0.] * 4, "atol": 1e-40, "rtol": 0.}
    good = unbounded(p, [a, b, 1., 1.], **kwargs)
    bad = unbounded(p, [np.nextafter(a, 0.), b, 1., 1.], **kwargs)
    assert good.passed and not bad.passed
    assert not bad.checks["recession_cone_block_0_original_rsoc"]


def test_soc_boundary_one_ulp_mutant_and_wrong_axis():
    p = in_cone(SOC(3), [-1., 0., 0.])
    for d in ([np.nextafter(5., 0.), 3., 4.], [4., 3., 5.]):
        assert not unbounded(p, d, x0=[0.] * 3, s0=[0.] * 3).passed


def test_component_zero_and_nonnegative_not_global_scale():
    p = constant(Nonnegative(2), [-1., 0.])
    e = infeasible(p, [1e100, -1e-20])
    assert not e.passed and not e.checks["dual_cone_block_0_component_1_nonnegative"]
    p = ConicProblem(np.zeros((2, 2)), np.array([-1., 0.]),
                     np.eye(2), np.array([1e100, 0.]), (Zero(2),))
    e = unbounded(p, [0., 0.], x0=[1e100, 1e-20], s0=[0., 0.])
    assert not e.checks["base_equation_1_identity"]


@pytest.mark.parametrize("slope", [0., 1., -1e-12, -1e-8])
def test_strict_negative_margin_not_nonpositive_or_atol_equality(slope):
    e = unbounded(empty(q=[slope]), [1.], x0=[0.], s0=[], rtol=0.)
    assert not e.passed
    assert not e.checks["qtd_strict_margin"]
    assert e.checks["qtd_strict_sign"] is (slope < 0)


def test_cancellation_does_not_erase_signed_margin_or_absolute_scale():
    p = empty(q=[1e16, -1e16, -1.])
    e = unbounded(p, [1., 1., 1.], x0=[0.] * 3, s0=[])
    assert e.metrics["qtd_upper"] == -1.
    assert not e.checks["qtd_strict_margin"] and not e.passed
    e = unbounded(p, [1., 1., 1.], x0=[0.] * 3, s0=[], rtol=0.)
    assert e.passed


@pytest.mark.parametrize("field", ["P", "q", "A", "b"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_original_model_fail_loud(field, value):
    p = in_cone(Nonnegative(1), [-1.])
    array = getattr(p, field).copy()
    array.flat[0] = value
    with pytest.raises(ValueError, match="finite"):
        unbounded(replace(p, **{field: array}), [1.], x0=[0.], s0=[0.])


@pytest.mark.parametrize("field", ["d", "x0", "s0", "y"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_witness_fail_loud(field, value):
    p = in_cone(Nonnegative(1), [-1.])
    if field == "y":
        with pytest.raises(ValueError, match="finite"):
            infeasible(p, [value])
    else:
        kwargs = {"d": [1.], "x0": [0.], "s0": [0.]}
        kwargs[field] = [value]
        with pytest.raises(ValueError, match="finite"):
            unbounded(p, **kwargs)


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:99"])
@pytest.mark.parametrize("available", [True, False])
def test_explicit_cuda_never_falls_back(device, available, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)
    with pytest.raises(RuntimeError, match="host-only; no fallback"):
        unbounded(empty(), [1.], device=device)
    with pytest.raises(RuntimeError, match="host-only; no fallback"):
        infeasible(constant(Zero(1), [1.]), [-1.], device=device)


def test_non_cpu_tensor_never_transferred_and_cpu_input_not_mutated():
    with pytest.raises(ValueError, match="implicit device transfer"):
        unbounded(empty(), torch.empty(1, dtype=torch.float64, device="meta"))
    tensor = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
    snapshot = tensor.detach().clone()
    assert unbounded(empty(), tensor, x0=[0.], s0=[]).passed
    assert torch.equal(tensor.detach(), snapshot) and tensor.grad is None


@pytest.mark.parametrize("value", [np.array([1.], dtype=np.float32), [True], [1+0j], [2**53+1]])
def test_nonbinary64_or_inexact_conversion_rejected(value):
    with pytest.raises(ValueError, match="binary64"):
        unbounded(empty(), value)


@pytest.mark.parametrize("change", ["n_limit", "m_limit", "P_shape", "b_shape", "cone_rows", "cone_type", "empty_n", "device"])
def test_unsupported_and_malformed_fail_loud(change):
    p = empty()
    kwargs = {}
    if change == "n_limit":
        p = empty(np.zeros((17, 17)), [-1.] * 17)
    elif change == "m_limit":
        p = constant(Nonnegative(129), [0.] * 129)
    elif change == "P_shape":
        p = replace(p, P=np.ones((1, 2)))
    elif change == "b_shape":
        p = replace(p, b=np.ones(1))
    elif change == "cone_rows":
        p = replace(p, cones=(Zero(1),))
    elif change == "cone_type":
        p = replace(p, cones=("SOC",))
    elif change == "empty_n":
        p = empty(np.empty((0, 0)), [])
    else:
        kwargs["device"] = "meta"
    with pytest.raises((ValueError, TypeError)):
        unbounded(p, [1.], **kwargs)


@pytest.mark.parametrize("kwargs", [
    {"atol": 0.}, {"atol": True}, {"atol": np.inf}, {"atol": np.nan},
    {"rtol": -1.}, {"rtol": 1.}, {"rtol": True}, {"rtol": np.nan},
])
def test_invalid_tolerances_fail_loud(kwargs):
    with pytest.raises(ValueError):
        unbounded(empty(), [1.], **kwargs)


@pytest.mark.parametrize("mode", ["dot_product", "cancelled_products", "absolute_sum", "underflow", "soc_square", "rsoc_product"])
def test_overflow_underflow_and_unavailable_bounds_fail_loud(mode):
    p, d = empty(q=[-1e308]), [2.]
    if mode == "cancelled_products":
        p, d = empty(q=[1e308, -1e308]), [2., 2.]
    elif mode == "absolute_sum":
        p, d = empty(q=[1e308, -1e308]), [1., 1.]
    elif mode == "underflow":
        p, d = empty(q=[-1e-300]), [1e-300]
    elif mode == "soc_square":
        p, d = in_cone(SOC(2), [-1., 0.]), [1e300, 1e300]
    elif mode == "rsoc_product":
        p, d = in_cone(RSOC(3), [-1., 0., 0.]), [1e300, 1e300, 0.]
    with pytest.raises(ValueError, match="unavailable.*bound"):
        unbounded(p, d, x0=[0.] * len(d), s0=[0.] * len(p.b))


def test_rational_budget_missing_fail_loud(monkeypatch):
    import openopt.conic_certificates as module
    monkeypatch.setattr(module, "MAX_RATIONAL_BITS", 1)
    with pytest.raises(ValueError, match="bit budget"):
        unbounded(empty([[.5]]), [1.])


@pytest.mark.parametrize("kind", ["linear", "quadratic"])
def test_nonfinite_base_objective_bound_rejected(kind):
    if kind == "linear":
        p, d, x0 = empty(q=[-1e308]), [.01], [2.]
    else:
        p, d, x0 = empty([[1e308, 0.], [0., 0.]], [0., -1.]), [0., 1.], [2., 0.]
    with pytest.raises(ValueError, match="unavailable.*bound"):
        unbounded(p, d, x0=x0, s0=[])


def test_direction_checker_does_not_consult_solver_candidate_or_eigensolver(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("direction checker called forbidden solver/producer/default checker")
    monkeypatch.setattr("openopt.conic.solve_conic", forbidden)
    monkeypatch.setattr("openopt.conic.check_conic_candidate", forbidden)
    monkeypatch.setattr("openopt.conic.check_cone_feasibility", forbidden)
    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    p, kind, w = analytic_cases()["unbounded_nondiagonal_singular"]
    assert kind == "unbounded" and unbounded(p, **w).passed
    p, _, w = analytic_cases()["farkas_nonnegative"]
    assert infeasible(p, **w).passed


def test_defaults_and_exports_unchanged():
    import openopt
    from openopt.conic import ConicConfig
    assert ConicConfig().backend is None and ConicConfig().device == "cpu"
    assert not hasattr(openopt, "check_conic_infeasibility_direction")
    assert not hasattr(openopt, "check_conic_unbounded_direction")
    assert conic_direction_contract()["certified"] is False


def _model_json(p):
    return {**{key: getattr(p, key).tolist() for key in ("P", "q", "A", "b")},
            "cones": [{"type": type(c).__name__, "dimension": c.dimension} for c in p.cones]}


def test_original_model_receipt_and_mutant_selftest(tmp_path, request):
    cases = {}
    for name, (p, kind, witness) in analytic_cases().items():
        evidence = (infeasible if kind == "farkas" else unbounded)(p, **witness)
        assert evidence.passed
        cases[name] = {"original_model": _model_json(p), "kind": kind,
                       "original_witness": witness, "evidence": evidence.to_dict()}
    negatives = {
        "wrong_farkas_sign": infeasible(constant(Zero(1), [1.]), [1.]),
        "missing_base": unbounded(empty(), [1.]),
        "tiny_curvature": unbounded(empty([[1e-20]]), [1.], x0=[0.], s0=[]),
        "rsoc_bad_tiny_head": unbounded(in_cone(RSOC(3), [-1., 0., 0.]),
                                        [0., 1e300, 1.], x0=[0.]*3, s0=[0.]*3),
    }
    assert all(not e.passed for e in negatives.values())
    report = {
        "schema_version": 1, "verdict": "HOST_NUMERICAL_CHECKS_PASS",
        "checked_at": datetime.now(timezone.utc).isoformat(), "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "solver_calls": 0, "gpu_execution_verdict": "UNVERIFIED", "certified": False,
        "contract": conic_direction_contract(),
        "source_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in PINS},
        "cases": cases, "negative_controls": {n: e.to_dict() for n, e in negatives.items()},
    }
    path = tmp_path / "conic-directions.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(path.read_text()) == report
    request.node.user_properties.append(("conic_directions_receipt", str(path)))


def audit_artifacts(directory: Path):
    directory = directory.resolve(strict=True)
    if directory.is_relative_to(ROOT):
        raise ValueError("evidence must be outside worktree")
    verification = json.loads((directory / "verification.json").read_text())
    assert verification["verdict"] == verification["source_consistency"]["verdict"] == "PASS"
    assert verification["source"]["clean"]
    suites = ElementTree.parse(directory / "pytest.xml").getroot()
    counts = {n: sum(int(s.attrib.get(n, 0)) for s in suites.findall("testsuite"))
              for n in ("tests", "failures", "errors", "skipped")}
    assert counts == verification["tests"] and counts["tests"] > 0
    assert counts["failures"] == counts["errors"] == counts["skipped"] == 0
    paths = [p.attrib["value"] for p in suites.findall(".//property")
             if p.attrib["name"] == "conic_directions_receipt"]
    assert len(paths) == 1
    path = Path(paths[0]).resolve(strict=True)
    assert path.is_relative_to(directory)
    receipt = json.loads(path.read_text())
    assert receipt["verdict"] == "HOST_NUMERICAL_CHECKS_PASS" and receipt["device"] == "cpu"
    assert receipt["cuda_visible_devices"] == "-1" and receipt["solver_calls"] == 0
    assert not receipt["certified"] and receipt["gpu_execution_verdict"] == "UNVERIFIED"
    assert set(receipt["source_sha256"]) == set(PINS)
    for p, digest in receipt["source_sha256"].items():
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == digest
        assert verification["source"]["sha256"][p] == digest
    assert len(receipt["cases"]) == 10 and len(receipt["negative_controls"]) == 4
    for name, case in receipt["cases"].items():
        model = case["original_model"]
        cone_types = {c.__name__: c for c in (Zero, Nonnegative, SOC, RSOC)}
        p = ConicProblem(
            np.array(model["P"], dtype=np.float64), np.array(model["q"], dtype=np.float64),
            np.array(model["A"], dtype=np.float64).reshape(len(model["b"]), len(model["q"])),
            np.array(model["b"], dtype=np.float64),
            tuple(cone_types[c["type"]](c["dimension"]) for c in model["cones"]),
        )
        recomputed = (infeasible if case["kind"] == "farkas" else unbounded)(p, **case["original_witness"])
        assert recomputed.to_dict() == case["evidence"] and recomputed.passed, name
    assert all(not e["passed"] for e in receipt["negative_controls"].values())
    total = counts["tests"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    checked_at = datetime.now(timezone.utc).isoformat()
    audit = {
        "schema_version": 1, "verdict": "PASS", "checked_at": checked_at,
        "source_commit": verification["source"]["commit"], "source_clean": True,
        "source_consistency": verification["source_consistency"], "tests": counts,
        "cases_recomputed": 10, "negative_controls": 4, "solver_calls": 0,
        "gpu_execution_verdict": "UNVERIFIED", "certified": False,
        "receipt": str(path), "receipt_sha256": digest,
        "source_sha256": receipt["source_sha256"],
    }
    audit_path = directory / "conic-directions-audit.json"
    with audit_path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    assert json.loads(audit_path.read_text()) == audit
    print(f"CONIC_DIRECTIONS_AUDIT verdict=PASS checked_at={checked_at} "
          f"tests={total} skipped=0 cases=10 negatives=4 solver_calls=0 "
          f"gpu=UNVERIFIED receipt={path} sha256={digest} audit={audit_path}")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--audit-artifacts":
        audit_artifacts(Path(sys.argv[2]))
    else:
        raise SystemExit("use pytest or --audit-artifacts DIRECTORY")
