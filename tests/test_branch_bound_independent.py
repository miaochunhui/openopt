"""Independent exact tiny MIQP oracles and explicit clean-source review.

Default tests exercise only self-authored mathematics. Producer tests require a
full caller-pinned clean source commit; no moving producer is imported.
Enumeration is restricted to tiny test oracles, not a production algorithm.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib
import importlib
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODEL_FIELDS = ("P", "q", "A", "l", "u", "lower", "upper", "integrality")
SUBJECT_ENV = "OPENOPT_BRANCH_BOUND_REVIEW_ROOT"
COMMIT_ENV = "OPENOPT_BRANCH_BOUND_REVIEW_COMMIT"
ARTIFACT_ROOT = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/branch-bound-independent")


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False,
                            timeout=30)
    if result.returncode:
        raise RuntimeError(f"git_failed_rc={result.returncode}: {result.stderr.decode()}")
    return result.stdout


def clean_pin(root, commit):
    root = Path(root).resolve()
    if (not isinstance(commit, str) or len(commit) != 40
            or any(char not in "0123456789abcdef" for char in commit)):
        raise ValueError("require full forty-character frozen source commit")
    actual = git(root, "rev-parse", "HEAD").decode().strip()
    if actual != commit or git(root, "status", "--porcelain"):
        raise ValueError("subject must be the selected current clean committed tree")
    tracked = git(root, "ls-files", "-z").decode().split("\0")
    for name in tracked:
        path = root / name
        if name and (path.is_symlink() or not path.resolve().is_relative_to(root)
                     or not path.is_file()):
            raise ValueError(f"untrusted tracked source path: {name}")
    return {
        "commit": actual, "tree": git(root, "rev-parse", "HEAD^{tree}").decode().strip(),
        "clean": True, "sha256": {name: sha(root / name) for name in tracked if name},
    }


def rational(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("independent oracle requires finite represented real scalars")
    return Fraction(value)


def unpack(value):
    if not isinstance(value, str) or len(value) > 5000:
        raise ValueError("invalid rational")
    number = Fraction(value)
    if max(abs(number.numerator).bit_length(), number.denominator.bit_length()) > 8192:
        raise ValueError("oversized proof rational")
    if value != f"{number.numerator}/{number.denominator}":
        raise ValueError("noncanonical rational")
    return number


def pack(value):
    return f"{value.numerator}/{value.denominator}"


def dot(a, b):
    if len(a) != len(b):
        raise ValueError("independent vector dimension mismatch")
    return sum((x * y for x, y in zip(a, b)), Fraction(0))


def model(P, q, lower, upper, integer, A=(), l=(), u=()):
    n = len(q)
    if not 1 <= n <= 8 or len(A) > 64:
        raise ValueError("tiny original model dimension scope")
    if any(type(v) is not int or v not in (0, 1) for v in integer) or not any(integer):
        raise ValueError("explicit integer mask scope")
    matrix = tuple(tuple(rational(v) for v in row) for row in P)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("Hessian dimensions")
    if any(matrix[i][j] != matrix[j][i] for i in range(n) for j in range(n)):
        raise ValueError("Hessian symmetry")
    margins = tuple(matrix[i][i] - sum(abs(matrix[i][j]) for j in range(n) if j != i)
                    for i in range(n))
    if any(value < 0 for value in margins):
        raise ValueError("exact DD PSD subset not proved")
    if len(lower) != n or len(upper) != n or len(integer) != n:
        raise ValueError("box dimensions")
    box = []
    for low, high, discrete in zip(lower, upper, integer):
        low, high = rational(low), rational(high)
        if low > high:
            raise ValueError("reversed original variable bounds")
        box.append((Fraction(math.ceil(low)), Fraction(math.floor(high))) if discrete
                   else (low, high))
    if len(A) != len(l) or len(A) != len(u) or any(len(row) != n for row in A):
        raise ValueError("row dimensions")
    original = {
        "P": matrix, "q": tuple(rational(v) for v in q),
        "A": tuple(tuple(rational(v) for v in row) for row in A),
        "l": tuple(None if v == -math.inf else rational(v) for v in l),
        "u": tuple(None if v == math.inf else rational(v) for v in u),
        "box": tuple(box), "integer": tuple(bool(v) for v in integer),
        "margins": margins, "n": n,
    }
    if any(low is not None and high is not None and low > high
           for low, high in zip(original["l"], original["u"])):
        raise ValueError("reversed original row bounds")
    return original


def objective(m, x):
    return dot(m["q"], x) + sum(
        (x[i] * m["P"][i][j] * x[j] / 2
         for i in range(m["n"]) for j in range(m["n"])), Fraction(0))


def feasible(m, box, x):
    if len(x) != m["n"]:
        return False
    x = tuple(rational(v) for v in x)
    if any(not low <= x[i] <= high or m["integer"][i] and x[i].denominator != 1
           for i, (low, high) in enumerate(box)):
        return False
    for row, low, high in zip(m["A"], m["l"], m["u"]):
        activity = dot(row, x)
        if low is not None and activity < low or high is not None and activity > high:
            return False
    return True


def lower_bound(m, box, z, y):
    """Derive the convex tangent and signed row support independently."""
    z, y = tuple(rational(v) for v in z), tuple(rational(v) for v in y)
    if len(z) != m["n"] or len(y) != len(m["A"]):
        raise ValueError("witness dimensions")
    support = Fraction(0)
    for dual, low, high in zip(y, m["l"], m["u"]):
        if dual == 0:
            continue
        endpoint = high if dual > 0 else low
        if endpoint is None:
            raise ValueError("wrong infinite signed support")
        support += dual * endpoint
    constant = -sum((z[i] * m["P"][i][j] * z[j] / 2
                     for i in range(m["n"]) for j in range(m["n"])), Fraction(0)) - support
    gradient = tuple(dot(m["P"][i], z) + m["q"][i]
                     + sum((row[i] * dual for row, dual in zip(m["A"], y)), Fraction(0))
                     for i in range(m["n"]))
    return constant + sum((coefficient * (low if coefficient >= 0 else high)
                           for coefficient, (low, high) in zip(gradient, box)), Fraction(0))


def row_excludes(m, box):
    if any(low > high for low, high in box):
        return True
    for row, low, high in zip(m["A"], m["l"], m["u"]):
        minimum = sum((a * (left if a >= 0 else right)
                       for a, (left, right) in zip(row, box)), Fraction(0))
        maximum = sum((a * (right if a >= 0 else left)
                       for a, (left, right) in zip(row, box)), Fraction(0))
        if low is not None and maximum < low or high is not None and minimum > high:
            return True
    return False


def original_model(problem):
    arrays = {}
    for name in MODEL_FIELDS:
        value = getattr(problem, name)
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.dtype not in ((torch.int64,) if name == "integrality"
                                       else (torch.float32, torch.float64))):
            raise ValueError("unsupported original tensor type")
        arrays[name] = value.detach().cpu().tolist()
    return model(**{name: arrays[name] for name in ("P", "q", "lower", "upper")},
                 integer=arrays["integrality"], A=arrays["A"], l=arrays["l"], u=arrays["u"])


def descriptor(problem):
    arrays = {}
    for name in MODEL_FIELDS:
        tensor = getattr(problem, name)
        host = tensor.detach().cpu().contiguous()
        raw = host.numpy().tobytes()
        arrays[name] = {
            "dtype": str(tensor.dtype), "shape": list(tensor.shape), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "values": [float(v).hex() for v in host.flatten().tolist()],
        }
    return {"arrays": arrays, "sha256": hashlib.sha256(
        json.dumps(arrays, sort_keys=True, separators=(",", ":")).encode()).hexdigest()}


def boxes(value, n):
    if not isinstance(value, list) or len(value) != n or any(
            not isinstance(pair, list) or len(pair) != 2 for pair in value):
        raise ValueError("invalid box")
    return tuple((unpack(low), unpack(high)) for low, high in value)


def independent_review(problem, config, report, *, trusted_source=None):
    """Use caller original tensors; never call a producer mathematical helper."""
    m = original_model(problem)
    if trusted_source is not None and encode(report["source_sha256"]) != encode(trusted_source):
        raise ValueError("caller source binding differs")
    if type(report["schema_version"]) is not int or report["schema_version"] != 1:
        raise ValueError("unreviewed report version")
    if encode(report["model"]) != encode(descriptor(problem)):
        raise ValueError("original arrays differ")
    config_values = {
        "enabled": config.enabled, "device": str(torch.device(config.device)),
        "dtype": str(config.dtype), "max_nodes": config.max_nodes,
        "qp_max_iterations": config.qp_max_iterations,
        "qp_check_interval": config.qp_check_interval, "abs_gap": float(config.abs_gap),
        "rel_gap": float(config.rel_gap), "wall_seconds": config.wall_seconds,
    }
    if encode(report["config"]) != encode(config_values):
        raise ValueError("caller configuration differs")
    if report["psd_dd_margins"] != [pack(value) for value in m["margins"]]:
        raise ValueError("original exact PSD margins differ")
    if report["execution_verdict"] != "UNVERIFIED" or report["performance_verdict"] != "UNVERIFIED":
        raise ValueError("host proof cannot claim GPU or performance")
    if report["proof_arithmetic"] != "host_bounded_fraction_8192_v1":
        raise ValueError("unreviewed host proof arithmetic")
    nodes, reachable, candidates, computed = report["nodes"], {0}, [], []
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 2 * config.max_nodes + 1:
        raise ValueError("missing original region coverage")
    calls = visits = limits = 0
    for index, node in enumerate(nodes):
        if node["id"] != index or type(node["id"]) is not int:
            raise ValueError("wrong node identity")
        box = boxes(node["box"], m["n"])
        if index == 0 and (node["parent"] is not None or box != m["box"]):
            raise ValueError("original root region changed")
        if index and (type(node["parent"]) is not int or not 0 <= node["parent"] < index):
            raise ValueError("invalid parent region")
        if type(node["visited"]) is not bool:
            raise ValueError("wrong visit type")
        visits += node["visited"]
        if not node["visited"] and (node["proposal"] is not None or node["candidates"]):
            raise ValueError("unvisited region contains executed evidence")
        empty = any(low > high for low, high in box)
        if empty and (index or node["witnesses"] or node["bound"] is not None):
            raise ValueError("invalid empty region")
        values = [lower_bound(m, box, witness["z"], witness["y"])
                  for witness in node["witnesses"]]
        if index:
            values.append(computed[node["parent"]])
        if not empty and not node["witnesses"]:
            raise ValueError("missing lower bound witness")
        bound = None if empty else max(values)
        for candidate in node["candidates"]:
            point = candidate["x"]
            valid = feasible(m, box, point)
            value = objective(m, tuple(rational(v) for v in point)) if valid else None
            if candidate["objective"] != (pack(value) if value is not None else None):
                raise ValueError("strict represented candidate mismatch")
            if valid:
                candidates.append((value, point))
        proposal = node["proposal"]
        if proposal is not None:
            calls += 1
            if proposal["actual_device"] != config_values["device"] or proposal["dtype"] != str(config.dtype):
                raise ValueError("proposal placement metadata mismatch")
            if proposal["status"] not in ("solved", "max_iterations"):
                raise ValueError("unreviewed proposal status")
            limits += proposal["status"] == "max_iterations"
            if (type(proposal["iterations"]) is not int
                    or not 1 <= proposal["iterations"] <= config.qp_max_iterations):
                raise ValueError("proposal iteration budget mismatch")
            if ({"kind": "qp_tangent", "z": proposal["z"], "y": proposal["y"]}
                    not in node["witnesses"]):
                raise ValueError("proposal not linked to mathematical witness")
            lower_bound(m, box, proposal["z"], proposal["y"])
        disposition = node["disposition"]
        if disposition == "singleton":
            if not all(m["integer"]) or any(low != high for low, high in box):
                raise ValueError("unclosed mixed leaf")
            point = [float(low) for low, _ in box]
            if not feasible(m, box, point):
                raise ValueError("invalid singleton")
            bound = objective(m, tuple(rational(v) for v in point))
        if node["bound"] != (pack(bound) if bound is not None else None):
            raise ValueError("inflated original lower bound")
        computed.append(bound)
        if disposition == "infeasible":
            if not row_excludes(m, box):
                raise ValueError("unproved infeasible region")
        elif disposition == "branch":
            split = node["split"]
            if not isinstance(split, dict) or len(split["children"]) != 2:
                raise ValueError("deleted branch")
            coordinate, cut = split["index"], split["cut"]
            if type(coordinate) is not int or not 0 <= coordinate < m["n"] or not m["integer"][coordinate]:
                raise ValueError("invalid branch dimension")
            low, high = box[coordinate]
            if type(cut) is not int or not low <= cut < high:
                raise ValueError("invalid branch cut")
            children = split["children"]
            for child, interval in zip(children, ((low, Fraction(cut)), (Fraction(cut + 1), high))):
                if type(child) is not int or not index < child < len(nodes) or child in reachable:
                    raise ValueError("deleted or duplicate region")
                child_box = list(box)
                child_box[coordinate] = interval
                if nodes[child]["parent"] != index or boxes(nodes[child]["box"], m["n"]) != tuple(child_box):
                    raise ValueError("branch gap or changed continuous region")
                reachable.add(child)
        elif disposition not in ("frontier", "unresolved", "pruned", "singleton"):
            raise ValueError("unknown region disposition")
        if disposition != "branch" and node["split"] is not None:
            raise ValueError("nonbranch split")
        if not empty and (disposition == "frontier") == node["visited"]:
            raise ValueError("frontier visit partition mismatch")
        if disposition == "unresolved" and any(
                m["integer"][i] and low < high for i, (low, high) in enumerate(box)):
            raise ValueError("unresolved nonleaf")
    if reachable != set(range(len(nodes))):
        raise ValueError("missing or orphan coverage")
    if (any(type(report[key]) is not int for key in (
            "actual_qp_calls", "visited_nodes", "qp_iteration_limit_count"))
            or calls != report["actual_qp_calls"] or visits != report["visited_nodes"]
            or limits != report["qp_iteration_limit_count"] or visits > config.max_nodes):
        raise ValueError("call/visit budget mismatch")
    incumbent = report["incumbent"]
    summary = report["summary"]
    regions = [node for node in nodes if node["disposition"] != "branch"]
    lows = [computed[node["id"]] for node in regions if node["disposition"] != "infeasible"]
    lower = min(lows) if lows else None
    if summary["lower_bound"] != (pack(lower) if lower is not None else None):
        raise ValueError("whole region lower bound mismatch")
    if incumbent is None:
        if (candidates or summary["objective"] is not None or summary["gap_abs"] is not None
                or summary["gap_rel"] is not None or summary["enclosures"] is not None
                or any(node["disposition"] == "pruned" for node in nodes)):
            raise ValueError("lost incumbent")
        expected_status = "unknown" if lows else "infeasible_bounded"
    else:
        if not feasible(m, m["box"], incumbent):
            raise ValueError("fractional or equality-violating incumbent")
        value = objective(m, tuple(rational(v) for v in incumbent))
        if (value, incumbent) not in candidates or value != min(item[0] for item in candidates):
            raise ValueError("incumbent not supported by strict candidate")
        if lower is None:
            raise ValueError("incumbent without original covered lower bound")
        gap = value - lower
        if gap < 0:
            raise ValueError("negative signed gap")
        scale = max(Fraction(1), abs(value))
        if summary["objective"] != pack(value) or summary["gap_abs"] != pack(gap) or summary["gap_rel"] != pack(gap / scale):
            raise ValueError("signed gap/objective mismatch")
        if any(node["disposition"] == "pruned" and computed[node["id"]] < value for node in nodes):
            raise ValueError("unproved pruning")
        unresolved = any(
            node["disposition"] != "infeasible" and computed[node["id"]] < value
            and (node["disposition"] == "unresolved" or not all(m["integer"])
                 and all(not m["integer"][i] or low == high
                         for i, (low, high) in enumerate(boxes(node["box"], m["n"]))))
            for node in regions)
        threshold = rational(config.abs_gap) + rational(config.rel_gap) * scale
        expected_status = "unknown" if unresolved else (
            "solved_bounded" if gap <= threshold else "feasible")
        for key, exact in (("objective", value), ("lower_bound", lower),
                           ("gap_abs", gap), ("gap_rel", gap / scale)):
            low, high = summary["enclosures"][key]
            if not all(math.isfinite(v) for v in (low, high)) or not Fraction(low) <= exact <= Fraction(high):
                raise ValueError("nonenclosing binary64 proof")
    if summary["status"] != expected_status:
        raise ValueError("unknown/solved status mismatch")
    frontier = any(node["disposition"] == "frontier" for node in nodes)
    reason = report["stop_reason"]
    if (reason not in ("tree_exhausted", "node_limit", "wall_budget")
            or reason == "tree_exhausted" and frontier
            or reason == "node_limit" and (not frontier or visits != config.max_nodes)
            or reason == "wall_budget" and (not frontier or config.wall_seconds is None)):
        raise ValueError("stop reason original partition mismatch")
    cost_fields = {"total_before_replay", "qp_proposal", "host_coordination_and_scalar_proof"}
    if set(report["cost_seconds"]) not in (cost_fields, cost_fields | {"independent_replay"}):
        raise ValueError("missing disclosed host proof cost")
    for value in report["cost_seconds"].values():
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("undisclosed/nonfinite host proof cost")
    return {"numerical": "PASS", "coverage": "PASS", "status": expected_status,
            "region_count": len(regions), "gpu": "UNVERIFIED"}


def tiny_enumeration(m):
    if not all(m["integer"]):
        raise ValueError("tiny enumeration oracle is pure-integer only")
    count = math.prod(max(0, int(high - low + 1)) for low, high in m["box"])
    if count > 256:
        raise ValueError("test-only enumeration cap")
    values = [objective(m, tuple(map(Fraction, point))) for point in itertools.product(
        *(range(int(low), int(high) + 1) for low, high in m["box"]))
        if feasible(m, m["box"], point)]
    return min(values) if values else None


def specs():
    return {
        "pure": dict(P=[[2.]], q=[-1.], lower=[0.], upper=[2.], integer=[1]),
        "mixed": dict(P=[[4., -2.], [-2., 2.]], q=[-1., 0.],
                      lower=[0., 0.], upper=[1., 1.], integer=[1, 0]),
        "singular": dict(P=[[1., 1.], [1., 1.]], q=[-1., -1.],
                         lower=[0., 0.], upper=[1., 1.], integer=[1, 0]),
        "lp": dict(P=[[0.]], q=[1.], lower=[0.], upper=[2.], integer=[1],
                   A=[[1.]], l=[1.], u=[math.inf]),
        "mixed_open": dict(P=[[0., 0.], [0., 6.]], q=[0., -2.],
                           lower=[0., 0.], upper=[0., 1.], integer=[1, 0]),
        "equality": dict(P=[[0., 0.], [0., 2.]], q=[0., 0.],
                         lower=[0., 0.], upper=[0., 1.], integer=[1, 0],
                         A=[[0., 3.]], l=[1.], u=[1.]),
    }


@pytest.mark.parametrize("kind,expected", (("pure", 0), ("lp", 1)))
def test_independent_pure_polynomial_reference(kind, expected):
    assert tiny_enumeration(model(**specs()[kind])) == expected


def test_exact_dd_energy_identity_and_rounding_hidden_negative_margin():
    m = model([[2., -1.], [-1., 2.]], [0., 0.], [-1., -1.], [2., 2.], [1, 1])
    x = (Fraction(1, 3), Fraction(-7, 5))
    exact = 2 * objective(m, x)
    energy = sum((margin * v * v for margin, v in zip(m["margins"], x)), Fraction(0))
    energy += (x[0] - x[1]) ** 2
    assert energy == exact >= 0
    tiny = math.ldexp(1., -54)
    assert 1. + tiny == 1.
    with pytest.raises(ValueError, match="exact DD"):
        model([[1., 1., tiny], [1., 1., 0.], [tiny, 0., 1.]],
              [0.] * 3, [0.] * 3, [1.] * 3, [1, 0, 0])


def test_signed_support_subnormal_and_tangent_lower_bound():
    m = model(**specs()["lp"])
    assert lower_bound(m, m["box"], [0.], [-1.]) == 1
    assert lower_bound(m, m["box"], [0.], [0.]) == 0
    with pytest.raises(ValueError, match="infinite signed"):
        lower_bound(m, m["box"], [0.], [math.nextafter(0., 1.)])
    pure = model(**specs()["pure"])
    assert lower_bound(pure, pure["box"], [.5], []) == Fraction(-1, 4)
    assert lower_bound(pure, pure["box"], [0.], []) <= tiny_enumeration(pure)


def test_strict_fractional_equality_and_nonrepresentable_mixed_optimum():
    pure = model(**specs()["pure"])
    assert not feasible(pure, pure["box"], [.5])
    equality = model(**specs()["equality"])
    assert not feasible(equality, equality["box"], [0., 1. / 3.])
    mixed = model(**specs()["mixed_open"])
    exact = Fraction(-1, 3)
    point = (Fraction(0), Fraction(1. / 3.))
    assert objective(mixed, point) > exact
    assert objective(mixed, (Fraction(0), Fraction(1, 3))) == exact


def test_independent_complete_square_mixed_and_singular_reference():
    mixed, singular = model(**specs()["mixed"]), model(**specs()["singular"])
    for integer, continuous in itertools.product((0, 1), (Fraction(0), Fraction(1, 2), Fraction(1))):
        point = (Fraction(integer), continuous)
        assert objective(mixed, point) == (continuous - integer) ** 2 + integer * (integer - 1)
        assert objective(singular, point) == (sum(point) - 1) ** 2 / 2 - Fraction(1, 2)
    assert objective(mixed, (Fraction(0), Fraction(0))) == 0
    assert objective(singular, (Fraction(0), Fraction(1))) == -Fraction(1, 2)


def test_independent_bound_survives_float_overflow_without_fake_finite_result():
    m = model([[1e308]], [0.], [2.], [2.], [1])
    exact = objective(m, (Fraction(2),))
    with pytest.raises(OverflowError):
        float(exact)
    assert exact > Fraction(sys.float_info.max)
    for value in (math.inf, -math.inf, math.nan):
        with pytest.raises(ValueError, match="finite"):
            rational(value)


def handcrafted_report(kind):
    """Self-authored transcript for testing the oracle, without producer code."""
    problem = make_problem(SimpleNamespace(BoundedMIQPProblem=SimpleNamespace), specs()[kind])
    config = SimpleNamespace(
        enabled=True, device="cpu", dtype=torch.float64, max_nodes=128,
        qp_max_iterations=2000, qp_check_interval=25, abs_gap=0., rel_gap=0.,
        wall_seconds=None)
    m = original_model(problem)

    def node(identity, parent, box, disposition, z, candidates=()):
        witness = {"kind": "zero_tangent", "z": z, "y": [0.] * len(m["A"])}
        bound = lower_bound(m, box, z, witness["y"])
        points = []
        for point in candidates:
            value = objective(m, tuple(map(rational, point))) if feasible(m, box, point) else None
            points.append({"kind": "lower_endpoint", "x": point,
                           "objective": None if value is None else pack(value)})
        if disposition == "singleton":
            bound = objective(m, tuple(left for left, _ in box))
        return {
            "id": identity, "parent": parent,
            "box": [[pack(left), pack(right)] for left, right in box],
            "disposition": disposition, "bound": pack(bound), "witnesses": [witness],
            "candidates": points, "split": None, "proposal": None, "visited": True,
        }

    if kind == "pure":
        nodes = [
            node(0, None, m["box"], "branch", [0.], ([0.], [2.])),
            node(1, 0, ((Fraction(0), Fraction(0)),), "singleton", [0.]),
            node(2, 0, ((Fraction(1), Fraction(2)),), "pruned", [1.]),
        ]
        nodes[0]["split"] = {"index": 0, "cut": 0, "children": [1, 2]}
        incumbent, bound, value, status = [0.], Fraction(0), Fraction(0), "solved_bounded"
    elif kind == "lp":
        nodes = [node(0, None, m["box"], "frontier", [0.])]
        nodes[0]["visited"] = False
        incumbent, bound, value, status = None, Fraction(0), None, "unknown"
        config.max_nodes = 0
    elif kind == "equality":
        nodes = [node(0, None, m["box"], "unresolved", [0., 0.], ([0., 1. / 3.],))]
        incumbent, bound, value, status = None, Fraction(0), None, "unknown"
    else:
        raise ValueError("unimplemented handcrafted reference")
    summary = {
        "status": status, "lower_bound": pack(bound),
        "objective": None if value is None else pack(value),
        "gap_abs": None if value is None else pack(value - bound),
        "gap_rel": None if value is None else pack((value - bound) / max(Fraction(1), abs(value))),
        "enclosures": None if value is None else {
            key: [0., 0.] for key in ("objective", "lower_bound", "gap_abs", "gap_rel")},
    }
    report = {
        "schema_version": 1, "model": descriptor(problem), "config": vars(config).copy(),
        "psd_dd_margins": [pack(v) for v in m["margins"]], "source_sha256": {},
        "nodes": nodes, "incumbent": incumbent, "summary": summary,
        "actual_qp_calls": 0, "visited_nodes": sum(n["visited"] for n in nodes),
        "qp_iteration_limit_count": 0, "stop_reason": "node_limit" if kind == "lp" else "tree_exhausted",
        "cost_seconds": {
            "total_before_replay": 0., "qp_proposal": 0., "host_coordination_and_scalar_proof": 0.},
        "execution_verdict": "UNVERIFIED", "performance_verdict": "UNVERIFIED",
        "proof_arithmetic": "host_bounded_fraction_8192_v1",
    }
    report["config"]["dtype"] = str(config.dtype)
    return problem, config, report


def mutate_report(problem, original, mutation):
    changed = deepcopy(original)
    if mutation == "inflatedLB":
        changed["nodes"][0]["bound"] = pack(unpack(changed["nodes"][0]["bound"]) + 1)
    elif mutation == "fractionalequalityincumbent":
        changed["incumbent"] = [0., 1. / 3.]
        value = objective(original_model(problem), tuple(map(rational, changed["incumbent"])))
        changed["nodes"][0]["candidates"][0].update(
            x=changed["incumbent"], objective=pack(value))
        changed["summary"].update(objective=pack(value), status="solved_bounded",
                                  gap_abs="0/1", gap_rel="0/1")
    elif mutation == "deletedbranch":
        if not any(node["disposition"] == "branch" for node in changed["nodes"]):
            raise ValueError("deletedbranch counterexample requires genuine branching")
        changed["nodes"].pop()
    elif mutation == "wronginfinite_support":
        changed["nodes"][0]["witnesses"][0]["y"] = [math.nextafter(0., 1.)]
    else:
        raise ValueError("unreviewed mutation")
    return changed


def retain_rehashed_mutation(tmp_path, request, original, changed, source, mutation):
    baseline, forged = tmp_path / "baseline.json", tmp_path / "forged.json"
    baseline.write_bytes(encode(original))
    baseline_pin = sha(baseline)
    forged.write_bytes(encode(changed))
    envelope = tmp_path / "envelope.json"
    envelope.write_bytes(encode({"path": str(forged), "sha256": sha(forged),
                                 "model_sha256": changed["model"]["sha256"], "source": source}))
    outer_pin = sha(envelope)
    assert sha(forged) == json.loads(envelope.read_bytes())["sha256"]
    assert changed["model"] == original["model"] and changed["config"] == original["config"]
    assert changed["source_sha256"] == original["source_sha256"]
    assert outer_pin != baseline_pin and encode(changed) != encode(original)
    request.node.user_properties.extend([
        ("baseline_sha256", baseline_pin), ("forged_sha256", sha(forged)),
        ("envelope_sha256", outer_pin), ("mutation", mutation)])
    return baseline, baseline_pin


@pytest.mark.parametrize("mutation", (
    "inflatedLB", "fractionalequalityincumbent", "deletedbranch", "wronginfinite_support",
))
def test_handcrafted_coordinated_rehash_oracle_rejects(tmp_path, request, mutation):
    kind = "equality" if mutation == "fractionalequalityincumbent" else (
        "lp" if mutation == "wronginfinite_support" else "pure")
    problem, config, original = handcrafted_report(kind)
    assert independent_review(problem, config, original)["numerical"] == "PASS"
    changed = mutate_report(problem, original, mutation)
    baseline, baseline_pin = retain_rehashed_mutation(
        tmp_path, request, original, changed, "self_authored_no_producer", mutation)
    with pytest.raises(ValueError):
        independent_review(problem, config, changed)
    assert sha(baseline) == baseline_pin


@pytest.mark.parametrize("mutation", ("signed_gap", "cuda_label", "original_array", "config"))
def test_handcrafted_metadata_and_gap_fail_closed(mutation):
    problem, config, original = handcrafted_report("pure")
    assert independent_review(problem, config, original)["coverage"] == "PASS"
    changed = deepcopy(original)
    if mutation == "signed_gap":
        changed["summary"]["gap_abs"] = "-1/1"
    elif mutation == "cuda_label":
        changed["execution_verdict"] = "PASS"
    elif mutation == "original_array":
        changed["model"]["arrays"]["q"]["values"] = [float(1.).hex()]
    else:
        changed["config"]["max_nodes"] += 1
    with pytest.raises(ValueError):
        independent_review(problem, config, changed)


def complete_cases(nodes, records, counts, key_function, allow_frozen_skip=False):
    if not nodes or len(nodes) != len(set(nodes)):
        raise ValueError("empty or duplicate original collection")
    expected = Counter(key_function(node) for node in nodes)
    actual = Counter(record["key"] for record in records)
    if any(value != 1 for value in expected.values()) or expected != actual:
        raise ValueError("collection/JUnit original case denominator mismatch")
    observed = {"tests": len(records)}
    observed.update({key: sum(record[key] for record in records)
                     for key in ("failures", "errors", "skipped")})
    if counts != observed or counts["failures"] or counts["errors"]:
        raise ValueError("failed or inconsistent complete case counts")
    if allow_frozen_skip:
        for record in records:
            frozen = record["key"][1].startswith("test_frozen_")
            if bool(record["skipped"]) != frozen:
                raise ValueError("only unselected frozen-subject cases may skip")
    elif counts["skipped"]:
        raise ValueError("frozen review skipped cases")


@pytest.mark.parametrize("mutation", ("deleted", "duplicate", "unknown", "skip"))
def test_complete_cases_counter_controls(mutation):
    names = ["test_one", "test_two"]
    nodes = [f"tests/test_branch_bound_independent.py::{name}" for name in names]
    key = lambda value: ("tests.test_branch_bound_independent", value.split("::")[-1])
    records = [{"key": key(node), "failures": 0, "errors": 0, "skipped": 0} for node in nodes]
    counts = {"tests": 2, "failures": 0, "errors": 0, "skipped": 0}
    complete_cases(nodes, records, counts, key)
    if mutation == "deleted":
        records.pop()
    elif mutation == "duplicate":
        records[1]["key"] = records[0]["key"]
    elif mutation == "unknown":
        records[1]["key"] = ("tests.test_branch_bound_independent", "test_unknown")
    else:
        records[1]["skipped"] = counts["skipped"] = 1
    with pytest.raises(ValueError):
        complete_cases(nodes, records, counts, key)


@pytest.fixture(scope="module")
def frozen_subject():
    path, commit = os.environ.get(SUBJECT_ENV), os.environ.get(COMMIT_ENV)
    if not path or not commit:
        pytest.skip("clean producer freeze not selected; mechanism tests are not producer acceptance")
    root = Path(path).resolve()
    before = clean_pin(root, commit)
    required = "src/openopt/branch_bound.py"
    if required not in before["sha256"]:
        raise ValueError("selected commit has no bounded MIQP producer")
    sys.path.insert(0, str(root / "src"))
    for name in tuple(sys.modules):
        if name == "openopt" or name.startswith("openopt."):
            del sys.modules[name]
    bb = importlib.import_module("openopt.branch_bound")
    for name, module in tuple(sys.modules.items()):
        if name == "openopt" or name.startswith("openopt."):
            if not Path(module.__file__).resolve().is_relative_to(root / "src/openopt"):
                raise ValueError(f"foreign loaded subject module: {name}")
    assert sha(Path(bb.__file__)) == before["sha256"][required]
    yield bb, root, before
    assert clean_pin(root, commit) == before, "selected source changed during review"


def make_problem(bb, values):
    values = deepcopy(values)
    n = len(values["q"])
    integer = values.pop("integer")
    for name in ("A", "l", "u"):
        values.setdefault(name, [])
    tensors = {name: torch.tensor(values[name], dtype=torch.float64)
               for name in ("P", "q", "l", "u", "lower", "upper")}
    tensors["A"] = (torch.tensor(values["A"], dtype=torch.float64) if values["A"]
                    else torch.empty((0, n), dtype=torch.float64))
    tensors["integrality"] = torch.tensor(integer, dtype=torch.int64)
    return bb.BoundedMIQPProblem(**tensors)


@pytest.fixture(scope="module")
def produced(frozen_subject, tmp_path_factory):
    bb, _, pin = frozen_subject
    artifacts = tmp_path_factory.mktemp("frozen-original-cases")
    result = {}
    for name, values in specs().items():
        problem = make_problem(bb, values)
        config = bb.BranchBoundConfig(enabled=True, abs_gap=1. if name == "mixed_open" else 0.)
        before = descriptor(problem)
        started = time.monotonic()
        output = bb.solve_bounded_miqp(problem, config)
        assert descriptor(problem) == before
        trusted = {name: pin["sha256"][name] for name in (
            "src/openopt/branch_bound.py", "src/openopt/qp.py")}
        independent = independent_review(problem, config, output.report, trusted_source=trusted)
        result[name] = (problem, config, output, independent, time.monotonic() - started)
        assert output.report["source_sha256"] == trusted
        (artifacts / f"{name}.json").write_bytes(encode({
            "report": output.report, "independent": independent, "subject_commit": pin["commit"],
            "subject_tree": pin["tree"], "wall_s": result[name][-1],
        }))
    return result


@pytest.mark.parametrize("kind,expected", (
    ("pure", "0/1"), ("mixed", "0/1"), ("singular", "-1/2"), ("lp", "1/1"),
))
def test_frozen_analytic_pure_mixed_singular_lp(frozen_subject, produced, request, kind, expected):
    _, _, output, independent, wall = produced[kind]
    assert output.status == independent["status"] == "solved_bounded"
    assert output.report["summary"]["objective"] == expected
    assert output.report["summary"]["gap_abs"] == "0/1"
    assert output.x.device.type == "cpu" and output.x.dtype == torch.float64
    assert output.x.tolist() == output.report["incumbent"]
    request.node.user_properties.extend([
        ("subject_commit", frozen_subject[2]["commit"]),
        ("original_model_sha256", output.report["model"]["sha256"]),
        ("report_sha256", hashlib.sha256(encode(output.report)).hexdigest()),
        ("solve_and_exact_review_wall_s", wall),
        ("cost_seconds", json.dumps(output.report["cost_seconds"], sort_keys=True)),
    ])


@pytest.mark.parametrize("kind", ("mixed_open", "equality"))
def test_frozen_unclosed_mixed_regions_remain_unknown(produced, kind):
    _, _, output, independent, _ = produced[kind]
    assert output.status == independent["status"] == "unknown"
    assert any(node["disposition"] == "unresolved" for node in output.report["nodes"])
    if kind == "equality":
        assert output.x is None
    else:
        assert unpack(output.report["summary"]["gap_abs"]) > 0


def replay(bb, pin, problem, config, report):
    return bb.check_bounded_miqp_result(problem, config, report, trusted_source={
        name: pin["sha256"][name] for name in ("src/openopt/branch_bound.py", "src/openopt/qp.py")})


@pytest.mark.parametrize("mutation", (
    "inflatedLB", "fractionalequalityincumbent", "deletedbranch", "wronginfinite_support",
))
def test_frozen_coordinated_rehash_rejects_semantic_forgery(
        frozen_subject, produced, tmp_path, request, mutation):
    bb, _, pin = frozen_subject
    kind = "equality" if mutation == "fractionalequalityincumbent" else (
        "lp" if mutation == "wronginfinite_support" else "pure")
    problem, config, output, _, _ = produced[kind]
    original = deepcopy(output.report)
    assert independent_review(problem, config, original)["numerical"] == "PASS"
    assert replay(bb, pin, problem, config, original)["numerical_verdict"] == "PASS"
    changed = mutate_report(problem, original, mutation)
    baseline, baseline_pin = retain_rehashed_mutation(
        tmp_path, request, original, changed, pin["commit"], mutation)
    with pytest.raises(ValueError):
        independent_review(problem, config, changed)
    with pytest.raises((ValueError, ArithmeticError)):
        replay(bb, pin, problem, config, changed)
    assert sha(baseline) == baseline_pin


@pytest.mark.parametrize("budget", ("node", "wall"))
def test_frozen_budget_zero_never_calls_qp(frozen_subject, monkeypatch, budget):
    bb, _, pin = frozen_subject
    problem = make_problem(bb, specs()["pure"])
    kwargs = {"max_nodes": 0} if budget == "node" else {"wall_seconds": 0.}
    config = bb.BranchBoundConfig(enabled=True, **kwargs)
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("budget zero attempted solve"))
    output = bb.solve_bounded_miqp(problem, config)
    assert output.status == "unknown" and output.x is None
    assert output.report["actual_qp_calls"] == output.report["visited_nodes"] == 0
    assert independent_review(problem, config, output.report)["status"] == "unknown"
    assert replay(bb, pin, problem, config, output.report)["coverage_verdict"] == "PASS"


def test_frozen_explicit_cuda_missing_no_cpu_fallback(frozen_subject, monkeypatch):
    bb, _, _ = frozen_subject
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("CUDA request called CPU"))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        bb.solve_bounded_miqp(make_problem(bb, specs()["pure"]),
                              bb.BranchBoundConfig(enabled=True, device="cuda:0"))


def test_frozen_exact_dd_not_rounded_sum_or_proposal_cast(frozen_subject, monkeypatch):
    bb, _, _ = frozen_subject
    tiny = math.ldexp(1., -54)
    values = dict(P=[[1., 1., tiny], [1., 1., 0.], [tiny, 0., 1.]],
                  q=[0.] * 3, lower=[0.] * 3, upper=[1.] * 3, integer=[1, 0, 0])
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("unproved PSD called solve"))
    with pytest.raises(NotImplementedError, match="psd"):
        bb.solve_bounded_miqp(make_problem(bb, values),
                              bb.BranchBoundConfig(enabled=True, dtype=torch.float32))


def test_frozen_replay_no_solver_factory_and_cost_not_gpu(produced, frozen_subject, monkeypatch):
    bb, _, pin = frozen_subject
    problem, config, output, _, wall = produced["pure"]
    for name in ("solve_qp", "prepare_bounded_miqp", "solve_bounded_miqp"):
        monkeypatch.setattr(bb, name, lambda *a, **k: pytest.fail("replay called solve/factory"))
    assert replay(bb, pin, problem, config, output.report)["execution_verdict"] == "UNVERIFIED"
    assert independent_review(problem, config, output.report)["gpu"] == "UNVERIFIED"
    assert wall > 0 and output.report["cost_seconds"]["independent_replay"] > 0
    forged = deepcopy(output.report)
    forged["execution_verdict"] = "PASS"
    with pytest.raises(ValueError):
        independent_review(problem, config, forged)
    with pytest.raises(ValueError):
        replay(bb, pin, problem, config, forged)


def test_frozen_oracle_calls_no_producer_mathematical_function(frozen_subject, produced, monkeypatch):
    bb, _, _ = frozen_subject
    for name in ("_candidate", "_bound", "_summary", "_objective", "_row_infeasible", "_Model",
                 "check_bounded_miqp_result", "solve_qp", "prepare_bounded_miqp", "solve_bounded_miqp"):
        monkeypatch.setattr(bb, name, lambda *a, **k: pytest.fail("oracle called producer mathematics"))
    for problem, config, output, _, _ in produced.values():
        assert independent_review(problem, config, output.report)["numerical"] == "PASS"


@pytest.mark.parametrize("original_dtype", (torch.float32, torch.float64))
def test_frozen_original_precision_retained_before_proposal_dtype(
        frozen_subject, monkeypatch, original_dtype):
    bb, _, pin = frozen_subject
    problem = make_problem(bb, specs()["pure"])
    arrays = {name: getattr(problem, name).to(original_dtype) if name != "integrality"
              else getattr(problem, name) for name in MODEL_FIELDS}
    arrays["q"] = torch.tensor([-1.00000001], dtype=original_dtype)
    problem = bb.BoundedMIQPProblem(**arrays)
    before = descriptor(problem)
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("zero budget attempted dtype proposal"))
    config = bb.BranchBoundConfig(enabled=True, max_nodes=0, dtype=torch.float32)
    output = bb.solve_bounded_miqp(problem, config)
    assert output.report["model"] == before == descriptor(problem)
    expected = Fraction(float(arrays["q"][0]))
    assert unpack(output.report["summary"]["lower_bound"]) == 2 * expected
    assert independent_review(problem, config, output.report)["status"] == "unknown"
    assert replay(bb, pin, problem, config, output.report)["numerical_verdict"] == "PASS"


def test_frozen_negative_signed_gap_and_host_overflow_fail(frozen_subject, produced):
    bb, _, pin = frozen_subject
    problem, config, output, _, _ = produced["pure"]
    forged = deepcopy(output.report)
    forged["summary"]["gap_abs"] = "-1/1"
    with pytest.raises(ValueError, match="gap"):
        independent_review(problem, config, forged)
    with pytest.raises((ValueError, ArithmeticError)):
        replay(bb, pin, problem, config, forged)
    problem = make_problem(bb, dict(P=[[1e308]], q=[0.], lower=[2.], upper=[2.], integer=[1]))
    with pytest.raises(ArithmeticError, match="overflow"):
        bb.solve_bounded_miqp(problem, bb.BranchBoundConfig(enabled=True))


def test_frozen_mixed_frontier_unknown_under_loose_declared_gap(frozen_subject, tmp_path, request):
    bb, _, pin = frozen_subject
    problem = make_problem(bb, specs()["mixed"])
    config = bb.BranchBoundConfig(enabled=True, max_nodes=1, abs_gap=1.)
    output = bb.solve_bounded_miqp(problem, config)
    assert output.report["stop_reason"] == "node_limit"
    assert any(node["disposition"] == "frontier" for node in output.report["nodes"])
    assert output.status == independent_review(problem, config, output.report)["status"] == "unknown"
    assert replay(bb, pin, problem, config, output.report)["numerical_verdict"] == "PASS"
    forged = deepcopy(output.report)
    forged["summary"]["status"] = "solved_bounded"
    baseline, baseline_pin = retain_rehashed_mutation(
        tmp_path, request, output.report, forged, pin["commit"], "mixed_frontier_upgrade")
    with pytest.raises(ValueError, match="status"):
        independent_review(problem, config, forged)
    with pytest.raises(ValueError):
        replay(bb, pin, problem, config, forged)
    assert sha(baseline) == baseline_pin


@pytest.mark.parametrize("mutation", (
    "null_incumbent", "null_bound", "null_model", "bool_schema", "bool_count",
    "bool_config", "bool_integrality_model", "source_claim", "trusted_source",
    "continuous_box_shrink", "wrong_enclosure",
))
def test_frozen_original_null_bool_source_and_partition_binding(frozen_subject, produced, mutation):
    bb, _, pin = frozen_subject
    problem, config, output, _, _ = produced["mixed" if mutation == "continuous_box_shrink" else "pure"]
    original, forged = deepcopy(output.report), deepcopy(output.report)
    trusted = {name: pin["sha256"][name] for name in (
        "src/openopt/branch_bound.py", "src/openopt/qp.py")}
    assert independent_review(problem, config, original, trusted_source=trusted)["numerical"] == "PASS"
    assert replay(bb, pin, problem, config, original)["numerical_verdict"] == "PASS"
    if mutation == "null_incumbent":
        forged["incumbent"] = None
    elif mutation == "null_bound":
        forged["nodes"][0]["bound"] = None
    elif mutation == "null_model":
        forged["model"] = None
    elif mutation == "bool_schema":
        forged["schema_version"] = True
    elif mutation == "bool_count":
        forged["visited_nodes"] = True
    elif mutation == "bool_config":
        forged["config"]["enabled"] = 1
    elif mutation == "bool_integrality_model":
        forged["model"]["arrays"]["integrality"]["shape"] = [True]
    elif mutation == "source_claim":
        forged["source_sha256"]["src/openopt/qp.py"] = "0" * 64
    elif mutation == "trusted_source":
        trusted["src/openopt/qp.py"] = "0" * 64
    elif mutation == "continuous_box_shrink":
        child = next(node for node in forged["nodes"] if node["parent"] is not None)
        child["box"][1][1] = "1/2"
    else:
        forged["summary"]["enclosures"]["gap_abs"] = [1., 1.]
    assert encode(forged) != encode(original) or trusted != original["source_sha256"]
    with pytest.raises(ValueError):
        independent_review(problem, config, forged, trusted_source=trusted)
    with pytest.raises((ValueError, ArithmeticError)):
        bb.check_bounded_miqp_result(problem, config, forged, trusted_source=trusted)


@pytest.mark.parametrize("kind,expected", (("empty", "infeasible_bounded"), ("row", "infeasible_bounded")))
def test_frozen_empty_integer_and_row_proofs(frozen_subject, monkeypatch, kind, expected):
    bb, _, pin = frozen_subject
    values = dict(P=[[0.]], q=[0.], lower=[.2], upper=[.8], integer=[1]) if kind == "empty" else dict(
        P=[[0.]], q=[0.], lower=[0.], upper=[1.], integer=[1], A=[[1.]], l=[2.], u=[3.])
    problem = make_problem(bb, values)
    config = bb.BranchBoundConfig(enabled=True)
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("proved empty region called QP"))
    output = bb.solve_bounded_miqp(problem, config)
    assert output.status == independent_review(problem, config, output.report)["status"] == expected
    assert output.x is None and output.report["actual_qp_calls"] == 0
    assert replay(bb, pin, problem, config, output.report)["coverage_verdict"] == "PASS"


@pytest.mark.parametrize("mutation", ("dtype", "nan", "status"))
def test_frozen_actual_proposal_failure_never_silently_retries(frozen_subject, monkeypatch, mutation):
    bb, _, _ = frozen_subject
    calls = []
    def bad_proposal(problem, config):
        calls.append(str(problem.P.device))
        dtype = torch.float32 if mutation == "dtype" else config.dtype
        x = torch.zeros_like(problem.q, dtype=dtype)
        if mutation == "nan":
            x[0] = math.nan
        return SimpleNamespace(x=x, y=torch.zeros_like(problem.l, dtype=dtype),
                               status="infeasible" if mutation == "status" else "solved", iterations=1)
    monkeypatch.setattr(bb, "solve_qp", bad_proposal)
    with pytest.raises(RuntimeError, match="QP proposal"):
        bb.solve_bounded_miqp(make_problem(bb, specs()["pure"]), bb.BranchBoundConfig(enabled=True))
    assert calls == ["cpu"]


@pytest.mark.parametrize("mutation", ("n0", "nonfinite", "bad_integrality", "nonsymmetric", "infinite_box"))
def test_frozen_unsupported_inputs_never_solve(frozen_subject, monkeypatch, mutation):
    bb, _, _ = frozen_subject
    values = deepcopy(specs()["pure"])
    if mutation == "n0":
        values = dict(P=[], q=[], lower=[], upper=[], integer=[])
    elif mutation == "nonfinite":
        values["q"] = [math.inf]
    elif mutation == "bad_integrality":
        values["integer"] = [2]
    elif mutation == "nonsymmetric":
        values = dict(P=[[2., 1.], [0., 2.]], q=[0., 0.],
                      lower=[0., 0.], upper=[1., 1.], integer=[1, 0])
    else:
        values["upper"] = [math.inf]
    monkeypatch.setattr(bb, "solve_qp", lambda *a, **k: pytest.fail("invalid model attempted solve"))
    with pytest.raises((ValueError, NotImplementedError)):
        bb.solve_bounded_miqp(make_problem(bb, values), bb.BranchBoundConfig(enabled=True))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-repo", type=Path)
    parser.add_argument("--subject-commit")
    parser.add_argument("--mechanism-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_ROOT)
    args = parser.parse_args(argv)
    if args.mechanism_only:
        if args.subject_repo is not None or args.subject_commit is not None:
            parser.error("mechanism-only cannot select a producer")
        root, before = None, None
    else:
        if args.subject_repo is None or args.subject_commit is None:
            parser.error("explicit clean producer root and full commit required")
        root = args.subject_repo.resolve()
        before = clean_pin(root, args.subject_commit)
        if "src/openopt/branch_bound.py" not in before["sha256"]:
            parser.error("selected clean commit does not contain the producer")
    artifact_root = args.artifact_dir.resolve()
    if (not artifact_root.is_relative_to(Path("/3fs/research3fs/data/tmp"))
            or root is not None and artifact_root.is_relative_to(root)
            or artifact_root.is_relative_to(ROOT)):
        parser.error("external shared 3FS temporary artifact root required")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        parser.error("explicit CUDA_VISIBLE_DEVICES=-1 required")
    artifact_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="review-", dir=artifact_root))
    reviewer = clean_pin(ROOT, git(ROOT, "rev-parse", "HEAD").decode().strip())
    receipt = {"subject": before, "reviewer": reviewer, "verdict": "RUNNING",
               "qualification": "mechanism only; producer UNVERIFIED" if args.mechanism_only
               else "tiny exact CPU review, not commercial/GPU", "first_failure": None}
    started = time.monotonic()
    try:
        from scripts import verify_sharded as shard
        env = shard.cpu_env(ROOT, output)
        if args.mechanism_only:
            env.pop(SUBJECT_ENV, None)
            env.pop(COMMIT_ENV, None)
        else:
            env.update({SUBJECT_ENV: str(root), COMMIT_ENV: args.subject_commit})
        collection = output / "collection"
        collection.mkdir()
        env["OPENOPT_SHARD_COLLECTION"] = str(collection / "collection.json")
        target = "tests/test_branch_bound_independent.py"
        receipt["collection_process"] = shard.command_logged(
            [sys.executable, "-B", "-m", "pytest", "--collect-only", "-q",
             "-p", "scripts.verify_sharded", "-o", "addopts=",
             "-o", f"cache_dir={collection / 'pytest-cache'}", target],
            ROOT, collection, 45, env)
        collected = json.loads((collection / "collection.json").read_bytes())["nodeids"]
        env.pop("OPENOPT_SHARD_COLLECTION")
        receipt["collection"] = {
            "nodeids": collected, "sha256": sha(collection / "collection.json")}
        process = output / "process"
        process.mkdir()
        command = [sys.executable, "-B", "scripts/verify.py", "tests", "--target",
                   target, "--artifact-dir", str(output / "verification")]
        receipt["process"] = shard.command_logged(command, ROOT, process, 179, env)
        reports = list((output / "verification").glob("tests-*/verification.json"))
        if len(reports) != 1:
            raise ValueError("review receipt membership")
        report = json.loads(reports[0].read_bytes())
        records, counts = shard.parse_junit(reports[0].parent / "pytest.xml")
        if report["verdict"] != "PASS":
            raise ValueError("frozen review did not fully execute/pass")
        complete_cases(collected, records, counts, shard.junit_key, args.mechanism_only)
        if (report["source"] != reviewer
                or report["source_consistency"]["verdict"] != "PASS"
                or clean_pin(ROOT, reviewer["commit"]) != reviewer):
            raise ValueError("reviewer source drift")
        if root is not None and clean_pin(root, args.subject_commit) != before:
            raise ValueError("subject source drift")
        receipt.update(verdict="PASS", tests=counts, verification=str(reports[0]),
                       verification_sha256=sha(reports[0]), junit_sha256=sha(reports[0].parent / "pytest.xml"),
                       source_consistency=True, producer_verified=not args.mechanism_only,
                       leafsum_s=sum(record["time_s"] for record in records))
    except Exception as error:
        receipt.update(verdict="FAIL", first_failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt["wall_s"] = time.monotonic() - started
        (output / "receipt.json").write_bytes(encode(receipt))
        print(f"BRANCH_BOUND_REVIEW verdict={receipt['verdict']} receipt={output / 'receipt.json'}",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
