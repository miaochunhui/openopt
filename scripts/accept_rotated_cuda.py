"""Independent fixed-13 archive audit. No producer, Torch, or solver imports.

Original finite binary coordinates use exact rational gates; only the six
nearest-point oracles use bounded Decimal arithmetic. Old proof flags are not
authority. See docs/research/rotated-cuda-acceptance.md for caller pin format.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import struct
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PRODUCER = "tests/test_rotated_conic_cuda.py"
ALGORITHM = "fixed13_original_rational_decimal_v1"
MODEL_IDS = ("rotated_lp-float32", "rotated_lp-float64",
             "rotated_qp-float32", "rotated_qp-float64")
INPUTS = {
    "interior-asymmetric-float64": [1., 1e20, 0.],
    "interior-overflow-float64": [1.3e308, 1.3e308, 0.],
    "near-polar-float64": [-1e20, 0., 1e8],
    "small-head-float64": [-1e300, 0., 1e210],
    "small-head-float32": [-1e30, 0., 1e17],
    "tail-overflow-float64": [0., 0., 1.3e308, 1.3e308],
}
NEGATIVES = {
    "wrong-equality-dual": ["stationarity"],
    "planted-slack": ["primal_equation", "original_slack_cone"],
    "wrong-dual-cone": ["dual_cone"],
}
TARGETS = {
    "original_model": ("test_cuda_rotated_original_model", MODEL_IDS),
    "projection": ("test_cuda_rotated_projection_extremes", tuple(INPUTS)),
    "original_rejection": ("test_cuda_rotated_original_rejections", tuple(NEGATIVES)),
}
IDENTITIES = {
    f"tests.test_rotated_conic_cuda::{function}[{case}]": (kind, case)
    for kind, (function, cases) in TARGETS.items() for case in cases
}
BASE_FIELDS = set(
    "schema_version family verdict checked_at test_identity case_id case_kind device dtype "
    "config runtime loaded_sources source_sha256 tensor_metadata problem x s z status "
    "iterations evidence cone_evidence objective certified qualified commercial performance".split()
)
PIN_NAMES = {
    "run", "result", "junit", "submitted_job", "status", "pod_log", "pytest_log",
    "execution_log", "telemetry", "source_manifest", "outer_result",
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(a, b):
    return json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
        b, sort_keys=True, allow_nan=False)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def invalid_constant(value):
    raise ValueError(f"nonfinite JSON constant: {value}")


def load(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), f"unsafe/missing file: {path}")
    return json.loads(path.read_bytes(), object_pairs_hook=pairs,
                      parse_constant=invalid_constant)


def number(value):
    require(type(value) in (float, int), "finite represented coordinate required")
    require(math.isfinite(float(value)), "nonfinite/overflow coordinate")
    return Fraction.from_float(float(value))


def vector(values, size):
    require(isinstance(values, list) and len(values) == size, "vector dimension mismatch")
    return [number(value) for value in values]


def dot(a, b):
    require(len(a) == len(b), "dot dimension mismatch")
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def maximum(values):
    return max(map(abs, values), default=Fraction(0))


def cone_checks(values, cones, atol, rtol, dual=False):
    require(sum(c["dimension"] for c in cones) == len(values), "cone dimension mismatch")
    result = {}
    offset = 0
    for i, cone in enumerate(cones):
        require(set(cone) == {"kind", "dimension"} and type(cone["dimension"]) is int,
                "exact cone schema required")
        size = cone["dimension"]
        block = values[offset:offset + size]
        offset += size
        if cone["kind"] == "zero":
            require(size == 1, "unsupported zero block")
            passed = dual or maximum(block) <= atol
        else:
            require(cone["kind"] == "rsoc" and size in (3, 4), "unsupported cone")
            a, b, *tail = block
            small, large = min(a, b), max(a, b)
            radial = dot(tail, tail)
            signs = -a <= atol + rtol * abs(a) and -b <= atol + rtol * abs(b)
            if large > 0:
                required = radial / (2 * large)
                passed = signs and required - small <= atol + rtol * max(abs(small), required)
            else:
                # Squared form of sqrt(radial) <= (small + atol)/(1-rtol).
                bound = (small + atol) / (1 - rtol)
                passed = signs and bound >= 0 and radial <= bound * bound
        result[f"block_{i}"] = bool(passed)
    return result


def candidate(problem, vectors, atol, rtol):
    """Full original-coordinate gate, only diagonal {0,1} Hessians, n<=3,m<=4."""
    require(set(problem) == {"P", "q", "A", "b", "cones"}, "missing original model")
    n, m = len(problem["q"]), len(problem["b"])
    require(n in (1, 3) and m in (3, 4), "unsupported original dimensions")
    q, b = vector(problem["q"], n), vector(problem["b"], m)
    require(len(problem["P"]) == n and len(problem["A"]) == m, "matrix dimensions")
    P = [vector(row, n) for row in problem["P"]]
    A = [vector(row, n) for row in problem["A"]]
    require(all(P[i][j] == (P[i][i] if i == j else 0)
                for i in range(n) for j in range(n))
            and all(P[i][i] in (0, 1) for i in range(n)), "unsupported Hessian")
    x, s, z = vector(vectors["x"], n), vector(vectors["s"], m), vector(vectors["z"], m)
    atol, rtol = number(atol), number(rtol)
    require(atol >= 0 and 0 <= rtol < 1, "invalid tolerances")
    Ax, Px = [dot(row, x) for row in A], [dot(row, x) for row in P]
    ATz = [dot([row[j] for row in A], z) for j in range(n)]
    equation = [a + si - bi for a, si, bi in zip(Ax, s, b)]
    eq_limits = [atol + rtol * max(abs(a), abs(si), abs(bi)) for a, si, bi in zip(Ax, s, b)]
    stationarity = [p + qi + a for p, qi, a in zip(Px, q, ATz)]
    st_limits = [atol + rtol * max(abs(p), abs(qi), abs(a)) for p, qi, a in zip(Px, q, ATz)]
    cones = {
        "supplied_slack": cone_checks(s, problem["cones"], atol, rtol),
        "original_slack": cone_checks([bi - a for bi, a in zip(b, Ax)], problem["cones"], atol, rtol),
        "dual": cone_checks(z, problem["cones"], atol, rtol, True),
    }
    checks = {
        "primal_equation": all(abs(v) <= t for v, t in zip(equation, eq_limits)),
        "primal_cone": all(cones["supplied_slack"].values()),
        "original_slack_cone": all(cones["original_slack"].values()),
        "dual_cone": all(cones["dual"].values()),
        "stationarity": all(abs(v) <= t for v, t in zip(stationarity, st_limits)),
    }
    offset = 0
    metrics = {"primal_equation": str(maximum(equation)),
               "stationarity": str(maximum(stationarity))}
    for i, cone in enumerate(problem["cones"]):
        size = cone["dimension"]
        products = [si * zi for si, zi in zip(s[offset:offset + size], z[offset:offset + size])]
        offset += size
        name = f"complementarity_block_{i}"
        residual = abs(sum(products, Fraction(0)))
        checks[name] = residual <= atol + rtol * sum(map(abs, products), Fraction(0))
        metrics[name] = str(residual)
    h = [qi + a for qi, a in zip(q, ATz)]
    in_range = all(P[i][i] == 1 or abs(h[i]) <= atol + rtol * max(abs(q[i]), abs(ATz[i]))
                   for i in range(n))
    primal = dot(x, Px) / 2 + dot(q, x)
    available = checks["dual_cone"] and in_range
    dual = -sum((h[i] ** 2 / 2 for i in range(n) if P[i][i] == 1), Fraction(0)) - dot(b, z)
    gap = primal - dual
    checks["dual_objective_available"] = available
    checks["duality_gap"] = available and abs(gap) <= atol + rtol * max(abs(primal), abs(dual))
    metrics.update(primal_objective=str(primal), dual_objective=str(dual) if available else None,
                   signed_duality_gap=str(gap) if available else None)
    return {"checks": checks, "cone_checks": cones, "passed": all(checks.values()), "metrics": metrics}


def represented(values, dtype):
    if dtype == "torch.float32":
        return [struct.unpack("!f", struct.pack("!f", float(v)))[0] for v in values]
    require(dtype == "torch.float64", "unsupported dtype")
    return [float(v) for v in values]


def projection_oracle(values, precision=800):
    """Independent direct RSOC nearest point; no transformed model or factory."""
    require(precision in (800, 1000) and len(values) in (3, 4), "unsupported oracle bound")
    with localcontext() as context:
        context.prec = precision
        a, b, *tail = [Decimal.from_float(float(v)) for v in values]
        squared = sum((v * v for v in tail), Decimal(0))
        if a >= 0 and b >= 0 and 2 * a * b >= squared:
            out = [a, b, *tail]
        elif a <= 0 and b <= 0 and 2 * a * b >= squared:
            out = [Decimal(0)] * len(values)
        else:
            # Direct orthogonal RSOC coordinate elimination, distinct from producer SOC mapping.
            radius = ((a - b) ** 2 + 2 * squared).sqrt()
            total = (a + b + radius) / 2
            weight = total / radius
            out = [(total + weight * (a - b)) / 2,
                   (total - weight * (a - b)) / 2, *(weight * v for v in tail)]
        result = [float(v) for v in out]
        require(all(math.isfinite(v) for v in result), "unrepresentable oracle")
        return result


def projection_gate(values, output, dtype, case):
    expected = projection_oracle(values)
    require(exact(expected, projection_oracle(values, 1000)), "bounded oracle precision disagreement")
    u, p = vector(values, len(values)), vector(output, len(values))
    reference = vector(expected, len(values))
    tolerance = number(3e-6 if dtype == "torch.float32" else 3e-14)
    component = [abs(a - e) <= tolerance * abs(e) for a, e in zip(p, reference)]
    require(all(component), "independent projection component oracle failed")
    if case.startswith("interior-"):
        require(all(struct.pack("!d", float(a)) == struct.pack("!d", float(b))
                    for a, b in zip(values, output)), "interior input not bitwise preserved")
    cones = [{"kind": "rsoc", "dimension": len(values)}]
    # Moreau residual -r belongs to the self-dual cone. Relative gate is homogeneous.
    residual_dual = [a - b for a, b in zip(p, u)]
    positive = cone_checks(p, cones, Fraction(0), tolerance)
    polar = cone_checks(residual_dual, cones, Fraction(0), tolerance)
    orthogonal = abs(dot(p, residual_dual)) <= tolerance * sum(
        (abs(a * b) for a, b in zip(p, residual_dual)), Fraction(0))
    require(all(positive.values()) and all(polar.values()) and orthogonal,
            "original cone/Moreau gate failed")
    return {"component_checks": component, "cone": positive, "negative_residual_dual": polar,
            "moreau_orthogonal": orthogonal, "oracle": expected, "precision_crosscheck": [800, 1000]}


def check_recorded(evidence, replay, label):
    fields = {"schema_version", "kind", "passed", "checks", "metrics", "thresholds",
              "unavailable", "qualification"}
    require(isinstance(evidence, dict) and set(evidence) == fields
            and evidence["schema_version"] == 1
            and evidence["kind"] == "conic_candidate_optimality", f"{label}: evidence schema")
    require(exact(evidence["checks"], replay["checks"])
            and evidence["passed"] is replay["passed"], f"{label}: original check verdict mismatch")
    metrics = set(replay["checks"]) - {"dual_objective_available", "duality_gap"}
    metrics |= {"primal_objective", "dual_objective", "duality_gap", "range_residual"}
    require(set(evidence["metrics"]) == metrics, f"{label}: complete metric membership")
    thresholds = {"primal_equation_max", "stationarity_max", "spectral", "range_max"}
    thresholds |= {name for name in replay["checks"] if name.startswith("complementarity_")}
    if replay["checks"]["dual_objective_available"]:
        thresholds.add("duality_gap")
    require(set(evidence["thresholds"]) == thresholds, f"{label}: threshold membership")
    for name, value in evidence["metrics"].items():
        require(value is None or type(value) in (int, float) and math.isfinite(float(value)),
                f"{label}: finite recorded metric required: {name}")
    for name, value in evidence["thresholds"].items():
        require(type(value) in (int, float) and math.isfinite(float(value)) and value >= 0,
                f"{label}: finite nonnegative threshold: {name}")
    require(isinstance(evidence["unavailable"], dict)
            and set(evidence["unavailable"]) == (
                set() if replay["checks"]["dual_objective_available"] else {"dual_objective"}),
            f"{label}: dual availability reasons")
    if not replay["checks"]["dual_objective_available"]:
        require(evidence["metrics"]["dual_objective"] is None
                and evidence["metrics"]["duality_gap"] is None, f"{label}: unavailable dual fabricated")
    for name in replay["checks"]:
        if name.startswith("complementarity_"):
            require(evidence["checks"][name] is (
                evidence["metrics"][name] <= evidence["thresholds"][name]),
                f"{label}: recorded complementarity contradicts its gate")
    if replay["checks"]["dual_objective_available"]:
        require(evidence["checks"]["duality_gap"] is (
            abs(evidence["metrics"]["duality_gap"]) <= evidence["thresholds"]["duality_gap"]),
            f"{label}: recorded gap contradicts its gate")
    require(evidence["qualification"] == "approximate numerical evidence; not an exact proof",
            f"{label}: unsupported qualification")


def recorded_thresholds(evidence, problem, vectors, atol, rtol):
    """Legacy binary64 thresholds for these sparse fixed matrices, without BLAS."""
    x, s, z = vectors["x"], vectors["s"], vectors["z"]
    P, A, q, b = (problem[name] for name in ("P", "A", "q", "b"))
    Ax = [sum(a * v for a, v in zip(row, x)) for row in A]
    Px = [sum(a * v for a, v in zip(row, x)) for row in P]
    ATz = [sum(row[i] * v for row, v in zip(A, z)) for i in range(len(x))]
    expected = {
        "primal_equation_max": max(atol + rtol * max(abs(a), abs(si), abs(bi))
                                   for a, si, bi in zip(Ax, s, b)),
        "stationarity_max": max(atol + rtol * max(abs(p), abs(qi), abs(a))
                                for p, qi, a in zip(Px, q, ATz)),
        "spectral": 64 * (2. ** -52) * len(x) * max(P[i][i] for i in range(len(x))),
        "range_max": max(atol + rtol * max(abs(qi), abs(a)) for qi, a in zip(q, ATz)),
    }
    offset = 0
    for i, cone in enumerate(problem["cones"]):
        size = cone["dimension"]
        products = [a * v for a, v in zip(s[offset:offset + size], z[offset:offset + size])]
        offset += size
        expected[f"complementarity_block_{i}"] = atol + rtol * sum(map(abs, products))
    if evidence["checks"]["dual_objective_available"]:
        expected["duality_gap"] = atol + rtol * max(abs(evidence["metrics"]["primal_objective"]),
                                                   abs(evidence["metrics"]["dual_objective"]))
    return expected


def threshold_contract(evidence, problem, vectors, atol, rtol):
    require(exact(evidence["thresholds"], recorded_thresholds(evidence, problem, vectors, atol, rtol)),
            "exact original threshold contract mismatch")


def replay_sidecar(saved):
    require(saved.get("test_identity") in IDENTITIES, "unknown exact identity")
    kind, case = IDENTITIES[saved["test_identity"]]
    extra = {"original_model": "analytic", "projection": "projection", "original_rejection": "negative"}[kind]
    require(set(saved) == BASE_FIELDS | {extra} and saved["schema_version"] == 1
            and saved["family"] == "rotated_conic_cuda" and saved["verdict"] == "PASS",
            "exact schema1 required")
    require(saved["case_kind"] == kind and saved["case_id"] == case, "identity/case mismatch")
    dtype = "torch.float64" if kind == "original_rejection" else "torch." + case.rsplit("-", 1)[1]
    eps = (1e-12 if kind == "projection" else 3e-5 if dtype == "torch.float32" else 1e-8)
    rel = (3e-6 if dtype == "torch.float32" else 3e-14) if kind == "projection" else 0
    expected_config = {"enabled": True, "device": "cuda:0", "dtype": dtype, "rho": 1.,
                       "sigma": 1e-6, "max_iterations": 10000, "check_interval": 25,
                       "eps_abs": eps, "eps_rel": rel}
    require(saved["device"] == "cuda:0" and saved["dtype"] == dtype
            and exact(saved["config"], expected_config), "fixed explicit device/dtype/full config")
    require(saved["certified"] is False and saved["qualified"] is False
            and saved["commercial"] == saved["performance"] == "UNVERIFIED", "qualification inflation")
    problem = saved["problem"]
    require(set(problem) == {"P", "q", "A", "b", "cones"}, "complete original model")
    if kind != "projection":
        diagonal = 1. if case.startswith("rotated_qp") else 0.
        expected_model = {
            "P": [[diagonal, 0., 0.], [0., diagonal, 0.], [0., 0., 0.]],
            "q": [0., 0., 0.] if diagonal else [1., 1., 0.],
            "A": [[0., 0., 1.], [-1., -0., -0.], [-0., -1., -0.], [-0., -0., -1.]],
            "b": [2., 0., 0., 0.], "cones": [{"kind": "zero", "dimension": 1},
                                           {"kind": "rsoc", "dimension": 3}],
        }
        require(exact(problem, expected_model), "fixed original LP/QP model changed")
    else:
        dim = len(INPUTS[case])
        require(exact(problem, {"P": [[0.]], "q": [0.], "A": [[0.]] * dim,
                                "b": saved["s"], "cones": [{"kind": "rsoc", "dimension": dim}]})
                and exact(saved["x"], [0.]) and exact(saved["z"], [0.] * dim),
                "complete original projection feasibility model changed")
    n, m = len(problem["q"]), len(problem["b"])
    shapes = {"x": [n], "s": [m], "z": [m], "original.P": [n, n],
              "original.q": [n], "original.A": [m, n], "original.b": [m]}
    if kind == "projection":
        shapes.update({"projection.input": [m], "projection.output": [m]})
    else:
        shapes.update({"prepared." + key: shape for key, shape in
                       (("P", [n, n]), ("q", [n]), ("A", [m, n]), ("b", [m]), ("factor", [n, n]))})
    require(set(saved["tensor_metadata"]) == set(shapes), "actual placement complete membership")
    for key, shape in shapes.items():
        require(exact(saved["tensor_metadata"][key], {
            "shape": shape, "device": "cuda:0", "layout": "torch.strided",
            "dtype": "torch.float64" if key.startswith("original.") else dtype,
        }), f"actual placement mismatch: {key}")
    for key in ("x", "s", "z"):
        require(exact(represented(saved[key], dtype), saved[key]), f"{key}: dtype representation mismatch")
    replay = candidate(problem, saved, eps, rel)
    check_recorded(saved["evidence"], replay, case)
    threshold_contract(saved["evidence"], problem, saved, eps, rel)
    cone_saved = saved["cone_evidence"]
    require(set(cone_saved) == set(replay["cone_checks"]), "complete original cone evidence")
    for name, checks in replay["cone_checks"].items():
        item = cone_saved[name]
        require(set(item) == {"schema_version", "kind", "passed", "checks", "metrics",
                             "thresholds", "unavailable", "qualification"}
                and item["schema_version"] == 1
                and item["kind"] == ("dual_cone_feasibility" if name == "dual" else "cone_feasibility")
                and exact(item["checks"], checks) and item["passed"] is all(checks.values())
                and set(item["metrics"]) == set(item["thresholds"]) == set(checks),
                f"original cone verdict/schema mismatch: {name}")
    require(exact(saved["objective"], saved["evidence"]["metrics"]["primal_objective"]),
            "objective/evidence mismatch")
    require(type(saved["iterations"]) is int and 0 <= saved["iterations"] <= 10000,
            "iteration bound")
    if kind == "original_rejection":
        negative = saved["negative"]
        require(set(negative) == {"mutation", "expected_failed_checks", "baseline"}
                and negative["mutation"] == case
                and negative["expected_failed_checks"] == NEGATIVES[case], "negative exact contract")
        baseline = negative["baseline"]
        require(set(baseline) == {"x", "s", "z", "evidence", "status", "tensor_metadata"}
                and baseline["status"] == "solved"
                and exact(baseline["tensor_metadata"], {key: saved["tensor_metadata"][key]
                                                        for key in ("x", "s", "z")}), "baseline contract")
        original = candidate(problem, baseline, eps, rel)
        check_recorded(baseline["evidence"], original, "negative baseline")
        threshold_contract(baseline["evidence"], problem, baseline, eps, rel)
        require(original["passed"], "negative baseline rejected")
        changed = {key: list(baseline[key]) for key in ("x", "s", "z")}
        if case == "wrong-equality-dual":
            changed["z"][0] *= -1
        elif case == "planted-slack":
            changed["x"] = [0., 0., 2.]
        else:
            changed["z"][1:] = [-v for v in changed["z"][1:]]
        require(exact(changed, {key: saved[key] for key in changed}), "negative mutation mismatch")
        require(not replay["passed"] and all(not replay["checks"][key] for key in NEGATIVES[case])
                and saved["status"] == "candidate_rejected", "expected rejection not observed")
        replay["baseline_passed"] = True
    else:
        require(replay["passed"], "independent original candidate rejected")
        if kind == "original_model":
            require(saved["status"] == "solved" and saved["iterations"] > 0, "solve status")
            with localcontext() as context:
                context.prec = 800
                root = float(Decimal(2).sqrt())
            analytic = saved["analytic"]
            dual = [-2., root, root, -2.] if diagonal else [-root, 1., 1., -root]
            expected = {"x": [root, root, 2.], "s": [0., root, root, 2.], "z": dual}
            require(set(analytic) == {"vectors", "evidence"} and exact(analytic["vectors"], expected),
                    "analytic original coordinates changed")
            analytic_replay = candidate(problem, analytic["vectors"], 1e-12, 0)
            check_recorded(analytic["evidence"], analytic_replay, "analytic")
            threshold_contract(analytic["evidence"], problem, analytic["vectors"], 1e-12, 0)
            require(analytic_replay["passed"]
                    and all(abs(number(a) - number(b)) <= 20 * number(eps)
                            for a, b in zip(saved["x"], expected["x"])), "analytic original gate")
        else:
            p = saved["projection"]
            require(set(p) == {"input", "output", "oracle", "input_unchanged"}
                    and exact(p["input"], represented(INPUTS[case], dtype))
                    and exact(p["output"], saved["s"]) and p["input_unchanged"] is True
                    and saved["status"] == "projected" and saved["iterations"] == 0,
                    "original projection coordinates/contract")
            oracle = p["oracle"]
            require(set(oracle) == {"algorithm", "precision", "expected", "atol", "rtol", "passed"}
                    and oracle["algorithm"] == "bounded_decimal_analytic_rsoc_v1"
                    and oracle["precision"] == 800 and exact(oracle["atol"], 0.)
                    and exact(oracle["rtol"], rel) and oracle["passed"] is True,
                    "recorded oracle contract")
            replay["projection"] = projection_gate(p["input"], p["output"], dtype, case)
            # Recorded producer coordinates remain pinned, but are not used as oracle.
            reference = vector(replay["projection"]["oracle"], m)
            recorded = vector(oracle["expected"], m)
            require(all(abs(a - b) <= number(rel) * abs(b) for a, b in zip(recorded, reference)),
                    "recorded oracle coordinates inconsistent")
    return {"identity": saved["test_identity"], "kind": kind, "case": case, **replay}


def safe_file(root, path):
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink(), "absolute non-symlink path required")
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root.resolve()) and resolved.is_file(), "file outside pinned root")
    return resolved


def pinned(root, binding):
    require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}, "exact caller file pin")
    require(isinstance(binding["sha256"], str) and re.fullmatch("[0-9a-f]{64}", binding["sha256"]),
            "caller SHA256 required")
    path = safe_file(root, binding["path"])
    require(sha(path) == binding["sha256"], f"caller byte pin mismatch: {path.name}")
    return path


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    require(result.returncode == 0, f"git evidence failure: {args[0]}")
    return result.stdout


def source_binding(run_root, run, source_root, sidecars):
    identity = run["source_identity"]
    require(set(identity) == {"commit", "tree", "clean", "sha256"} and identity["clean"] is True
            and identity["commit"] == run["source_sha"], "clean frozen source identity")
    require(git(source_root, "rev-parse", identity["commit"] + "^{tree}").decode().strip()
            == identity["tree"], "run commit/tree mismatch")
    manifest = load(run_root / "source/source-hashes.json")
    require(exact(manifest, run["source_hashes"]) and bool(manifest), "source manifest/run mismatch")
    actual = {p.relative_to(run_root / "source").as_posix()
              for p in (run_root / "source").rglob("*") if p.is_file()}
    require(actual == set(manifest) | {"source-hashes.json"}, "frozen source exact membership")
    for name, digest in manifest.items():
        path = safe_file(run_root / "source", (run_root / "source" / name).absolute())
        require(sha(path) == digest and identity["sha256"].get(name) == digest
                and hashlib.sha256(git(source_root, "show", identity["commit"] + ":" + name)).hexdigest()
                == digest, f"frozen/commit source byte mismatch: {name}")
    scope = {p.relative_to(source_root).as_posix() for p in (source_root / "src").rglob("*.py")}
    scope.add(PRODUCER)
    require(scope == {name for name in manifest if name.startswith("src/") and name.endswith(".py")}
            | {PRODUCER}, "current/snapshot source membership drift")
    expected = {name: manifest[name] for name in sorted(scope)}
    for name in scope:
        require(sha(safe_file(source_root, (source_root / name).absolute())) == expected[name],
                f"scope-current byte mismatch: {name}")
    modules = {name: {"source": "src/" + name.replace(".", "/") + ".py",
                      "sha256": expected["src/" + name.replace(".", "/") + ".py"]}
               for name in ("openopt.conic", "openopt.native_conic", "openopt.certificates")}
    for saved in sidecars:
        require(exact(saved["source_sha256"], expected) and exact(saved["loaded_sources"], modules),
                "sidecar full src/loaded byte membership mismatch")
    return {"execution_commit": identity["commit"], "execution_tree": identity["tree"],
            "audit_commit": git(source_root, "rev-parse", "HEAD").decode().strip(),
            "binding_mode": "scope-current-bytes; auditor was not in historical GPU snapshot",
            "scope_sha256": expected}


def expected_environment(snapshot, case_root):
    return [
        {"name": "PYTHONPATH", "value": str(snapshot / "src")},
        {"name": "PYTHONDONTWRITEBYTECODE", "value": "1"},
        {"name": "PYTHONUNBUFFERED", "value": "1"},
        *({"name": name, "value": "4"} for name in
          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")),
        {"name": "OPENOPT_NODE", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "TMPDIR", "value": str(case_root)},
    ]


def container_contract(container, command, environment):
    require("args" not in container and "envFrom" not in container, "extra args/envFrom rejected")
    require(container.get("command") == command, "exact frozen command mismatch")
    observed = deepcopy(container.get("env"))
    require(isinstance(observed, list), "explicit complete environment")
    for item in observed:
        if item.get("name") == "OPENOPT_NODE":
            ref = item.get("valueFrom", {}).get("fieldRef", {})
            if "apiVersion" in ref:
                require(ref.pop("apiVersion") == "v1", "unsupported fieldRef apiVersion")
    require(exact(observed, environment), "exact frozen environment mismatch")
    require(container["name"] == "benchmark" and container["imagePullPolicy"] == "Never",
            "container identity/pull contract")
    resources = deepcopy(container["resources"])
    for name in ("requests", "limits"):
        require(set(resources[name]) == {"cpu", "memory", "nvidia.com/gpu"}, "resource membership")
        gpu = resources[name]["nvidia.com/gpu"]
        require(type(gpu) is int and gpu == 1 or type(gpu) is str and gpu == "1", "one GPU required")
        resources[name]["nvidia.com/gpu"] = "1"
        require(exact(resources[name], {"cpu": "4", "memory": "12Gi", "nvidia.com/gpu": "1"}),
                "resource contract mismatch")


def canonical_spec(spec, *, pod=False):
    """Strip only enumerated API defaults, not arbitrary injected execution."""
    result = deepcopy(spec)
    for key, expected in (("dnsPolicy", "ClusterFirst"), ("schedulerName", "default-scheduler"),
                          ("terminationGracePeriodSeconds", 30)):
        if key in result:
            require(result.pop(key) == expected, f"unexpected API default: {key}")
    if "serviceAccount" in result:
        require(result.pop("serviceAccount") == result.get("serviceAccountName"), "serviceAccount alias")
    if pod:
        result.pop("nodeName")
        for key, value in (("enableServiceLinks", True), ("preemptionPolicy", "Never"), ("priority", 500)):
            if key in result:
                require(result.pop(key) == value, f"unexpected Pod default: {key}")
        if "tolerations" in result:
            require(exact(result.pop("tolerations"), [
                {"effect": "NoExecute", "key": "node.kubernetes.io/not-ready",
                 "operator": "Exists", "tolerationSeconds": 300},
                {"effect": "NoExecute", "key": "node.kubernetes.io/unreachable",
                 "operator": "Exists", "tolerationSeconds": 300}]), "unexpected Pod tolerations")
    container = result["containers"][0]
    for key, expected in (("terminationMessagePath", "/dev/termination-log"),
                          ("terminationMessagePolicy", "File")):
        if key in container:
            require(container.pop(key) == expected, f"unexpected container default: {key}")
    for item in container["env"]:
        if item["name"] == "OPENOPT_NODE":
            item["valueFrom"]["fieldRef"].pop("apiVersion", None)
    for resource in container["resources"].values():
        resource["nvidia.com/gpu"] = str(resource["nvidia.com/gpu"])
    if pod and "volumeMounts" in container:
        extras = [item for item in container["volumeMounts"] if
                  item["mountPath"] == "/var/run/secrets/kubernetes.io/serviceaccount"]
        require(len(extras) <= 1, "duplicate default serviceAccount mount")
        if extras:
            mount = extras[0]
            name = mount["name"]
            require(re.fullmatch(r"kube-api-access-[a-z0-9]{5}", name) is not None
                    and exact(mount, {"mountPath": "/var/run/secrets/kubernetes.io/serviceaccount",
                                      "name": name, "readOnly": True}), "unrecognized API mount")
            injected = [v for v in result.get("volumes", []) if v["name"] == name]
            require(len(injected) == 1 and set(injected[0]) == {"name", "projected"},
                    "default API projected volume")
            projected = injected[0]["projected"]
            require(set(projected) == {"defaultMode", "sources"} and projected["defaultMode"] == 420,
                    "default API volume mode")
            sources = projected["sources"]
            require(isinstance(sources, list) and len(sources) == 3
                    and set(sources[0]) == {"serviceAccountToken"}
                    and set(sources[0]["serviceAccountToken"]) == {"expirationSeconds", "path"}
                    and type(sources[0]["serviceAccountToken"]["expirationSeconds"]) is int
                    and 3600 <= sources[0]["serviceAccountToken"]["expirationSeconds"] <= 3607
                    and sources[0]["serviceAccountToken"]["path"] == "token"
                    and exact(sources[1], {"configMap": {"items": [{"key": "ca.crt", "path": "ca.crt"}],
                                                        "name": "kube-root-ca.crt"}})
                    and exact(sources[2], {"downwardAPI": {"items": [{
                        "fieldRef": {"apiVersion": "v1", "fieldPath": "metadata.namespace"},
                        "path": "namespace"}]}}), "unrecognized API projected sources")
            container["volumeMounts"].remove(mount)
            result["volumes"].remove(injected[0])
    return result


def archive_execution(run_root, run, result_path, result, submitted, status, pod_log):
    require(len(run["jobs"]) == 1, "sole submitted job required")
    descriptor = run["jobs"][0]
    archived_case = Path(descriptor["output"])
    require(archived_case.resolve() == (run_root / "case-0").resolve()
            and Path(descriptor["manifest"]).resolve() == (run_root / "job-0.json").resolve(),
            "run job/case path mismatch")
    snapshot = archived_case.parent / "source"
    command = ["python3", str(snapshot / "scripts/gpu_release_check.py"), "--artifact-dir",
               str(archived_case), "--target", PRODUCER, "--expression", "cuda_rotated"]
    environment = expected_environment(snapshot, archived_case)
    job, pods = status["job"], status["pods"]["items"]
    require(len(pods) == 1 and job["metadata"]["name"] == descriptor["job"]
            and submitted["metadata"]["name"] == descriptor["job"]
            and job["metadata"]["namespace"] == submitted["metadata"]["namespace"] == run["namespace"],
            "sole Job/Pod identity")
    pod = pods[0]
    owners = pod["metadata"]["ownerReferences"]
    require(len(owners) == 1 and owners[0]["kind"] == "Job"
            and owners[0]["uid"] == job["metadata"]["uid"]
            and owners[0]["name"] == descriptor["job"] and owners[0]["controller"] is True
            and pod["metadata"]["namespace"] == run["namespace"] and bool(pod["metadata"]["uid"]),
            "Pod owner UID/namespace mismatch")
    specs = [submitted["spec"]["template"]["spec"], job["spec"]["template"]["spec"], pod["spec"]]
    for spec in specs:
        require(len(spec["containers"]) == 1 and not spec.get("initContainers")
                and not spec.get("ephemeralContainers"), "unexpected container membership")
        container_contract(spec["containers"][0], command, environment)
        require(spec["restartPolicy"] == "Never" and spec["runtimeClassName"] == "nvidia"
                and spec["nodeSelector"] == {"nvidia.com/gpu.product": "NVIDIA-RTX-PRO-4500-Blackwell"},
                "GPU pod policy mismatch")
    baseline = specs[0]["containers"][0]
    expected_spec = canonical_spec(specs[0])
    require(exact(canonical_spec(specs[1]), expected_spec)
            and exact(canonical_spec(specs[2], pod=True), expected_spec),
            "complete actual/submitted Pod execution spec mismatch")
    for spec in specs[1:]:
        for key in ("image", "securityContext"):
            require(exact(spec["containers"][0][key], baseline[key]), f"actual container {key} mismatch")
    require(submitted["spec"]["backoffLimit"] == job["spec"]["backoffLimit"] == 0
            and job["spec"]["parallelism"] == job["spec"]["completions"] == 1
            and job["status"]["succeeded"] == 1 and not job["status"].get("failed")
            and not job["status"].get("active") and any(
                c["type"] == "Complete" and c["status"] == "True" for c in job["status"]["conditions"]),
            "Job not exactly completed once")
    states = pod["status"]["containerStatuses"]
    require(pod["status"]["phase"] == "Succeeded" and len(states) == 1, "Pod not succeeded")
    state = states[0]
    terminated = state["state"]["terminated"]
    require(state["name"] == "benchmark" and state["restartCount"] == 0 and state["lastState"] == {}
            and terminated["exitCode"] == 0 and terminated["reason"] == "Completed"
            and "@sha256:" in state["imageID"], "container failure/restart/image digest")
    require(result["hardware"]["node"] == pod["spec"]["nodeName"], "result/Pod node mismatch")
    start, finish = timestamp(result["started_at"]), timestamp(result["finished_at"])
    require(timestamp(terminated["startedAt"]) <= start < finish
            and finish <= timestamp(terminated["finishedAt"]) + 1
            and timestamp(job["status"]["startTime"]) <= start
            and finish <= timestamp(job["status"]["completionTime"]), "execution interval mismatch")
    match = re.fullmatch(r"OPENOPT_RELEASE_CUDA verdict=PASS tests=13 evidence=(\S+)\s*", pod_log)
    require(match is not None and Path(match[1]).resolve() == result_path.parent.resolve(),
            "actual Pod completion/evidence log mismatch")
    return {"job": descriptor["job"], "job_uid": job["metadata"]["uid"],
            "pod": pod["metadata"]["name"], "pod_uid": pod["metadata"]["uid"],
            "node": pod["spec"]["nodeName"], "image_id": state["imageID"]}


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(result.tzinfo is not None, "timezone required")
    return result.timestamp()


def junit_sidecars(path, result_root):
    document = ET.fromstring(path.read_bytes())
    require(document.tag == "testsuites" and len(document) == 1 and document[0].tag == "testsuite",
            "exact single JUnit suite")
    suite = document[0]
    require({key: suite.get(key) for key in ("tests", "failures", "errors", "skipped")}
            == {"tests": "13", "failures": "0", "errors": "0", "skipped": "0"},
            "exact complete13 JUnit denominator")
    selected, bindings = {}, {}
    for case in suite:
        require(case.tag == "testcase" and all(item.tag == "properties" for item in case),
                "JUnit failure/skip/extra child")
        identity = case.get("classname", "") + "::" + case.get("name", "")
        require(identity in IDENTITIES and identity not in selected, "JUnit exact identity duplicate/extra")
        require(len(case) == 1 and len(case[0]) == 2, "exact two sidecar properties")
        props = {}
        for prop in case[0]:
            require(prop.tag == "property" and set(prop.attrib) == {"name", "value"}
                    and prop.get("name") not in props, "duplicate/invalid JUnit property")
            props[prop.get("name")] = prop.get("value")
        require(set(props) == {"rotated_conic_acceptance_path", "rotated_conic_acceptance_sha256"},
                "exact sidecar property names")
        binding = {"path": props["rotated_conic_acceptance_path"],
                   "sha256": props["rotated_conic_acceptance_sha256"]}
        file = pinned(result_root / "pytest-tmp", binding)
        require(file.name == "rotated-conic.json" and file not in bindings, "sidecar reused")
        bindings[file] = binding
        selected[identity] = file
    require(set(selected) == set(IDENTITIES), "incomplete exact thirteen identities")
    actual = {p.resolve() for p in (result_root / "pytest-tmp").rglob("rotated-conic.json")}
    require(actual == set(bindings), "sidecar selection bidirectional membership")
    return selected, bindings


def audit(manifest, source_root=ROOT):
    require(set(manifest) == {"schema_version", "run_root", "files"} and manifest["schema_version"] == 1
            and set(manifest["files"]) == PIN_NAMES, "complete caller pin manifest")
    root = Path(manifest["run_root"]).resolve(strict=True)
    files = {name: pinned(root, binding) for name, binding in manifest["files"].items()}
    require(files["run"] == root / "run.json" and files["submitted_job"] == root / "job-0.json"
            and files["source_manifest"] == root / "source/source-hashes.json", "exact archive pin paths")
    result_path = files["result"]
    require(result_path.parent.parent == root / "case-0"
            and result_path.parent.name.startswith("cuda-check-") and result_path.name == "result.json",
            "single pinned result path")
    for name, filename in (("junit", "pytest.xml"), ("telemetry", "telemetry.json"),
                           ("pytest_log", "pytest.log"), ("execution_log", "execution.log")):
        require(files[name] == result_path.parent / filename, f"exact {name} result path")
    require(files["outer_result"] == root / "case-0/result.json"
            and files["outer_result"].read_bytes() == result_path.read_bytes(), "inner/outer result bytes")
    run, result = load(files["run"]), load(result_path)
    require(result["verdict"] == "PASS" and result["returncode"] == 0
            and result["targets"] == [PRODUCER] and result["selection"] == "cuda_rotated"
            and result["tests"] == {"tests": 13, "failures": 0, "errors": 0, "skipped": 0},
            "exact fixed13 result execution")
    selected, bindings = junit_sidecars(files["junit"], result_path.parent)
    sidecars = [load(path) for path in selected.values()]
    numeric, errors = [], []
    for identity, path in selected.items():
        saved = load(path)
        require(saved["test_identity"] == identity, "sidecar/JUnit identity mismatch")
        try:
            numeric.append(replay_sidecar(saved))
        except (ValueError, KeyError, TypeError, OverflowError) as exc:
            errors.append({"identity": identity, "reason": str(exc)})
    require(not errors, "independent numeric failures: " + json.dumps(errors, sort_keys=True))
    source = source_binding(root, run, source_root, sidecars)
    result_sources = {name: digest for name, digest in run["source_hashes"].items()
                      if name.startswith("src/openopt/") and name.endswith(".py")}
    result_sources.update({PRODUCER: run["source_hashes"][PRODUCER],
                           "scripts/gpu_release_check.py": run["source_hashes"]["scripts/gpu_release_check.py"]})
    require(exact(result["source_sha256"], result_sources), "result full source membership/hash")
    job = load(files["submitted_job"])
    status = load(files["status"])
    require(files["status"].name == run["jobs"][0]["job"] + "-status.json",
            "exact Job status archive path")
    require(files["status"].parent == root and files["pod_log"].parent == root,
            "status/log outside exact run root")
    execution = archive_execution(root, run, result_path, result, job, status, files["pod_log"].read_text())
    require(files["pod_log"].name == execution["pod"] + ".log", "exact actual Pod log path")
    require(files["execution_log"].read_bytes() == files["pod_log"].read_bytes(), "inner/actual log bytes")
    pytest_log = files["pytest_log"].read_text()
    require(re.search(r"\b13 passed\b", pytest_log) is not None
            and "returncode=0" in pytest_log, "pytest log final count/exit")
    runtime = sidecars[0]["runtime"]
    runtime_keys = set("python implementation executable numpy torch compiled_cuda cuda_available "
                       "cuda_execution gpu_name capability environment".split())
    require(set(runtime) == runtime_keys
            and all(type(runtime[key]) is str and runtime[key] for key in
                    ("python", "implementation", "executable", "numpy", "torch", "compiled_cuda"))
            and runtime["cuda_available"] is True and runtime["cuda_execution"] is True
            and runtime["capability"] == [12, 0]
            and runtime["gpu_name"] == result["hardware"]["gpu"] == "NVIDIA RTX PRO 4500 Blackwell"
            and runtime["torch"] == result["hardware"]["torch"]
            and runtime["compiled_cuda"] == result["hardware"]["cuda"]
            and exact(runtime["environment"], {"OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
                                               "MKL_NUM_THREADS": "4", "CUDA_VISIBLE_DEVICES": None}),
            "complete runtime/placement execution binding")
    require(all(exact(saved["runtime"], runtime) and
                timestamp(result["started_at"]) <= timestamp(saved["checked_at"])
                <= timestamp(result["finished_at"]) for saved in sidecars), "runtime/time per case mismatch")
    telemetry = load(files["telemetry"])
    require(type(telemetry) is list and len(telemetry) == result["telemetry_samples"] > 0,
            "actual telemetry count mismatch")
    previous_time, previous_cpu, gpu_uuid = -1., -1, None
    for sample in telemetry:
        require(sample["rc"] == 0 and sample["error"] is None
                and type(sample["cpu"]["usage_usec"]) is int and sample["cpu"]["usage_usec"] >= previous_cpu
                and sample["monotonic_s"] > previous_time
                and timestamp(result["started_at"]) <= timestamp(sample["at"])
                <= timestamp(result["finished_at"]), "telemetry failure/order/interval")
        parts = sample["gpu"].strip().split(",")
        require(len(parts) == 4 and re.fullmatch(
            r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", parts[0]),
            "telemetry sole GPU UUID")
        usage, memory, power = [float(value.strip()) for value in parts[1:]]
        require(all(math.isfinite(v) for v in (usage, memory, power))
                and 0 <= usage <= 100 and memory >= 0 and power >= 0,
                "telemetry utilization/memory/power invalid")
        require(gpu_uuid is None or gpu_uuid == parts[0], "telemetry GPU UUID changed")
        gpu_uuid = parts[0]
        previous_time, previous_cpu = sample["monotonic_s"], sample["cpu"]["usage_usec"]
    for name, binding in manifest["files"].items():
        require(sha(files[name]) == binding["sha256"], f"archive changed during audit: {name}")
    for path, binding in bindings.items():
        require(sha(path) == binding["sha256"], "sidecar changed during audit")
    for name, digest in run["source_hashes"].items():
        require(sha(root / "source" / name) == digest, f"snapshot changed during audit: {name}")
    for name, digest in source["scope_sha256"].items():
        require(sha(source_root / name) == digest, f"current source changed during audit: {name}")
    return {"schema_version": 1, "algorithm": ALGORITHM, "verdict": "PASS",
            "scope": "fixed13 original-model numeric/archive audit only",
            "numeric_cases": numeric, "denominator": {"positive": 10, "negative": 3, "total": 13},
            "source": source, "execution": execution, "runtime": runtime,
            "sidecars": list(bindings.values()), "caller_pins": manifest,
            "telemetry_samples": len(telemetry), "gpu_uuid": gpu_uuid,
            "certified": False, "commercial": "UNVERIFIED", "performance": "UNVERIFIED",
            "old_proof_authority": False}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "accept", "numeric-replay"))
    parser.add_argument("--pins", type=Path, required=True)
    parser.add_argument("--pins-sha256", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--source-root", type=Path, default=ROOT)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    require(args.artifact_dir.is_absolute() and not args.artifact_dir.exists()
            and not args.artifact_dir.resolve().is_relative_to(ROOT), "fresh external artifact dir required")
    inventory = load(args.pins)
    evidence_root = inventory.get("run_root", inventory.get("result_root"))
    require(isinstance(evidence_root, str) and
            not args.artifact_dir.resolve().is_relative_to(Path(evidence_root).resolve()),
            "audit output cannot modify evidence root")
    args.artifact_dir.mkdir(parents=True)
    try:
        require(re.fullmatch("[0-9a-f]{64}", args.pins_sha256) is not None
                and sha(args.pins) == args.pins_sha256, "caller manifest SHA mismatch")
        manifest = load(args.pins)
        if args.mode == "numeric-replay":
            require(set(manifest) == {"schema_version", "result_root", "junit"}
                    and manifest["schema_version"] == 1, "numeric-only pin schema")
            root = Path(manifest["result_root"]).resolve(strict=True)
            selected, _ = junit_sidecars(pinned(root, manifest["junit"]), root)
            results, failures = [], []
            for identity, file in selected.items():
                try:
                    saved = load(file)
                    require(saved["test_identity"] == identity, "JUnit identity mismatch")
                    results.append(replay_sidecar(saved))
                except (ValueError, KeyError, TypeError, OverflowError) as exc:
                    failures.append({"identity": identity, "reason": str(exc)})
            report = {"verdict": "PASS" if not failures else "FAIL", "algorithm": ALGORITHM,
                      "mode": "numeric-replay-only; NOT seal/archive/source/device qualification",
                      "passed": len(results), "total": 13, "cases": results, "failures": failures,
                      "commercial": "UNVERIFIED", "performance": "UNVERIFIED"}
            write_json(args.artifact_dir / "numeric-replay.json", report)
            require(not failures, "numeric replay failed: " + json.dumps(failures))
        else:
            require(args.source_root.resolve() == ROOT, "must audit with explicit current tool root")
            require(not git(ROOT, "status", "--porcelain").strip(), "clean audit source required")
            report = audit(manifest, ROOT)
            if args.mode == "accept":
                require(args.bundle is not None and args.bundle_sha256 is not None
                        and sha(args.bundle) == args.bundle_sha256, "caller bundle byte pin required")
                require(exact(load(args.bundle), report), "bundle/current complete replay mismatch")
            require(not git(ROOT, "status", "--porcelain").strip(), "audit source changed during replay")
            write_json(args.artifact_dir / "bundle.json", report)
        receipt = {"verdict": "PASS", "mode": args.mode,
                   "checked_at": datetime.now(timezone.utc).isoformat(),
                   "pins_sha256": args.pins_sha256, "auditor_sha256": sha(__file__),
                   "audit_runtime": {"python": platform.python_version(), "implementation": platform.python_implementation()},
                   "commercial": "UNVERIFIED", "performance": "UNVERIFIED"}
        write_json(args.artifact_dir / "acceptance.json", receipt)
        (args.artifact_dir / "execution.log").write_text(f"FIXED13_AUDIT mode={args.mode} verdict=PASS\n")
        print(f"FIXED13_AUDIT mode={args.mode} verdict=PASS artifact={args.artifact_dir}")
        return 0
    except Exception as exc:
        (args.artifact_dir / "failure.log").write_text(traceback.format_exc())
        write_json(args.artifact_dir / "acceptance.json", {
            "verdict": "FAIL", "mode": args.mode, "reason": str(exc),
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "commercial": "UNVERIFIED", "performance": "UNVERIFIED"})
        (args.artifact_dir / "execution.log").write_text(f"FIXED13_AUDIT mode={args.mode} verdict=FAIL\n")
        print(f"FIXED13_AUDIT mode={args.mode} verdict=FAIL reason={exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
