"""Opt-in tiny finite-box convex MIQP, native QP proposals and host proof.

Only exactly symmetric diagonally dominant PSD Hessians are supported.
Bounded Fraction scalar verification is not an arbitrary-precision solver.
Replay proves numerical bounds/coverage, never CPU/CUDA execution attestation.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .qp import QPConfig, QPProblem, solve_qp
from . import qp as _qp_module

BIT_LIMIT = 8192
INTEGER_LIMIT = 2**24 - 1
MODEL_FIELDS = ("P", "q", "A", "l", "u", "lower", "upper", "integrality")


@dataclass(frozen=True)
class BoundedMIQPProblem:
    P: torch.Tensor
    q: torch.Tensor
    A: torch.Tensor
    l: torch.Tensor
    u: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    integrality: torch.Tensor


@dataclass(frozen=True)
class BranchBoundConfig:
    enabled: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    max_nodes: int = 128
    qp_max_iterations: int = 2000
    qp_check_interval: int = 25
    abs_gap: float = 1e-7
    rel_gap: float = 0.0
    wall_seconds: float | None = None

    def validate(self) -> torch.device:
        if type(self.enabled) is not bool or not self.enabled:
            raise ValueError("bounded MIQP is disabled; enabled=True is required")
        for name, low, high in (("max_nodes", 0, 4096), ("qp_max_iterations", 1, 100000),
                                ("qp_check_interval", 1, 100000)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} outside bounded integer range")
        for name in ("abs_gap", "rel_gap", "wall_seconds"):
            value = getattr(self, name)
            if name == "wall_seconds" and value is None:
                continue
            if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
            raise ValueError("only explicit cpu or cuda devices are supported")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("proposal dtype must be float32/64")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("explicit CUDA requested but unavailable; no CPU fallback")
            index = torch.cuda.current_device() if device.index is None else device.index
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError("CUDA device index out of range")
            device = torch.device("cuda", index)
        return device


@dataclass(frozen=True)
class BranchBoundResult:
    x: torch.Tensor | None
    status: str
    report: dict


def _checked(value: Fraction) -> Fraction:
    if max(abs(value.numerator).bit_length(), value.denominator.bit_length()) > BIT_LIMIT:
        raise ArithmeticError("bounded Fraction scalar exceeds 8192-bit limit")
    return value


def _add(a, b):
    return _checked(a + b)


def _mul(a, b):
    return _checked(a * b)


def _sum(values):
    total = Fraction(0)
    for value in values:
        total = _add(total, value)
    return total


def _fraction(value) -> Fraction:
    if type(value) not in (float, int) or not math.isfinite(value):
        raise ValueError("proof scalar must be finite and real")
    return _checked(Fraction(value))


def _pack(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _unpack(value) -> Fraction:
    if not isinstance(value, str) or len(value) > 5000:
        raise ValueError("invalid bounded rational encoding")
    try:
        result = _checked(Fraction(value))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("invalid bounded rational encoding") from error
    if _pack(result) != value:
        raise ValueError("noncanonical bounded rational encoding")
    return result


def _enclose(value: Fraction) -> list[float]:
    try:
        rounded = float(_checked(value))
    except OverflowError as error:
        raise ArithmeticError("proof scalar binary64 overflow") from error
    if not math.isfinite(rounded):
        raise ArithmeticError("proof scalar binary64 overflow")
    lo = math.nextafter(rounded, -math.inf) if Fraction(rounded) > value else rounded
    hi = math.nextafter(rounded, math.inf) if Fraction(rounded) < value else rounded
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise ArithmeticError("proof enclosure binary64 overflow")
    return [lo, hi]


def source_sha256() -> dict:
    root = Path(__file__).resolve().parents[2]
    if Path(_qp_module.__file__).resolve() != root / "src/openopt/qp.py":
        raise ValueError("QP dependency loaded outside frozen source root")
    names = ("src/openopt/branch_bound.py", "src/openopt/qp.py")
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _descriptor(problem: BoundedMIQPProblem) -> dict:
    arrays = {}
    for name in MODEL_FIELDS:
        tensor = getattr(problem, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("MIQP inputs must be tensors")
        if tensor.layout != torch.strided or tensor.is_complex():
            raise ValueError("MIQP inputs must be real dense tensors")
        host = tensor.detach().cpu().contiguous()
        if name == "integrality":
            if tensor.dtype != torch.int64:
                raise ValueError("integrality must have int64 dtype")
        elif tensor.dtype not in (torch.float32, torch.float64):
            raise ValueError("original model tensors must be float32/64")
        raw = host.numpy().tobytes()
        arrays[name] = {"dtype": str(tensor.dtype), "shape": list(tensor.shape),
                        "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                        "values": [float(v).hex() for v in host.flatten().tolist()]}
    digest = hashlib.sha256(json.dumps(arrays, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"sha256": digest, "arrays": arrays}


class _Model:
    def __init__(self, problem):
        if not isinstance(problem, BoundedMIQPProblem):
            raise TypeError("expected BoundedMIQPProblem")
        self.descriptor = _descriptor(problem)
        host = {name: getattr(problem, name).detach().cpu().to(torch.float64).clone()
                for name in MODEL_FIELDS}
        q, P, A = host["q"], host["P"], host["A"]
        if q.ndim != 1 or not q.numel():
            raise ValueError("q must be a nonempty vector")
        self.n, self.m = q.numel(), A.shape[0] if A.ndim == 2 else -1
        if P.shape != (self.n, self.n) or A.ndim != 2 or A.shape[1] != self.n:
            raise ValueError("P/A shape mismatch")
        if any(host[k].shape != (self.m,) for k in ("l", "u")) or any(
                host[k].shape != (self.n,) for k in ("lower", "upper", "integrality")):
            raise ValueError("bounds/integrality shape mismatch")
        if any(not bool(torch.isfinite(host[k]).all()) for k in ("P", "q", "A")):
            raise ValueError("P/q/A must be finite")
        for low, high in (("l", "u"), ("lower", "upper")):
            L, U = host[low], host[high]
            if bool(torch.isnan(L).any() | torch.isnan(U).any() | torch.isposinf(L).any() | torch.isneginf(U).any()):
                raise ValueError("invalid signed bounds/NaN")
            if bool((L > U).any()):
                raise ValueError("reversed bounds")
        if not bool(((host["integrality"] == 0) | (host["integrality"] == 1)).all()):
            raise ValueError("integrality must be exactly 0/1")
        if not torch.equal(P, P.T):
            raise ValueError("P must be exactly symmetric before proposal casting")
        if self.n > 8 or self.m > 64 or not bool(host["integrality"].any()):
            raise NotImplementedError("unsupported tiny MIQP dimension/integrality scope")
        if not bool(torch.isfinite(host["lower"]).all() & torch.isfinite(host["upper"]).all()):
            raise NotImplementedError("unsupported infinite variable boxes")
        self.P = tuple(tuple(_fraction(v) for v in row) for row in P.tolist())
        self.q = tuple(_fraction(v) for v in q.tolist())
        self.A = tuple(tuple(_fraction(v) for v in row) for row in A.tolist())
        self.l = tuple(None if v == -math.inf else _fraction(v) for v in host["l"].tolist())
        self.u = tuple(None if v == math.inf else _fraction(v) for v in host["u"].tolist())
        self.integer = tuple(bool(v) for v in host["integrality"].tolist())
        margins = [_add(self.P[i][i], -_sum(abs(self.P[i][j]) for j in range(self.n) if i != j))
                   for i in range(self.n)]
        if any(v < 0 for v in margins):
            raise NotImplementedError("unsupported psd_scope_unproved: exact diagonal dominance required")
        self.psd = [_pack(v) for v in margins]
        self.box = []
        for i, (L, U) in enumerate(zip(host["lower"].tolist(), host["upper"].tolist())):
            a, b = _fraction(L), _fraction(U)
            if self.integer[i]:
                a, b = Fraction(math.ceil(a)), Fraction(math.floor(b))
                if abs(a) > INTEGER_LIMIT or abs(b) > INTEGER_LIMIT:
                    raise NotImplementedError("unsupported integer endpoint range")
            self.box.append((a, b))
        self.host = host


def _dot(a, b):
    return _sum(_mul(x, y) for x, y in zip(a, b))


def _objective(model, x):
    Px = [_dot(row, x) for row in model.P]
    return _add(_mul(Fraction(1, 2), _dot(x, Px)), _dot(model.q, x))


def _candidate(model, box, x):
    if len(x) != model.n:
        raise ValueError("candidate shape mismatch")
    x = tuple(_fraction(v) for v in x)
    for i, (L, U) in enumerate(box):
        if x[i] < L or x[i] > U or (model.integer[i] and x[i].denominator != 1):
            return None
    for row, L, U in zip(model.A, model.l, model.u):
        activity = _dot(row, x)
        if (L is not None and activity < L) or (U is not None and activity > U):
            return None
    f = _objective(model, x)
    _enclose(f)
    return f


def _bound(model, box, z, y):
    if len(z) != model.n or len(y) != model.m:
        raise ValueError("bound witness shape mismatch")
    z, y = tuple(_fraction(v) for v in z), tuple(_fraction(v) for v in y)
    support = []
    for v, L, U in zip(y, model.l, model.u):
        endpoint = U if v > 0 else L
        if v != 0 and endpoint is None:
            raise ArithmeticError("nonzero dual selects infinite signed support")
        support.append(Fraction(0) if v == 0 else _mul(v, endpoint))
    a = [_add(_add(_dot(model.P[i], z), model.q[i]),
              _sum(_mul(model.A[j][i], y[j]) for j in range(model.m))) for i in range(model.n)]
    c = _add(-_mul(Fraction(1, 2), _sum(_mul(z[i], _dot(model.P[i], z)) for i in range(model.n))),
             -_sum(support))
    B = _add(c, _sum(min(_mul(a[i], L), _mul(a[i], U)) for i, (L, U) in enumerate(box)))
    _enclose(B)
    return B


def _row_infeasible(model, box):
    if any(L > U for L, U in box):
        return True
    for row, L, U in zip(model.A, model.l, model.u):
        low = _sum(min(_mul(a, x), _mul(a, y)) for a, (x, y) in zip(row, box))
        high = _sum(max(_mul(a, x), _mul(a, y)) for a, (x, y) in zip(row, box))
        _enclose(low)
        _enclose(high)
        if (L is not None and high < L) or (U is not None and low > U):
            return True
    return False


def _box_pack(box):
    return [[_pack(L), _pack(U)] for L, U in box]


def _box_unpack(value, n):
    if not isinstance(value, list) or len(value) != n or any(not isinstance(v, list) or len(v) != 2 for v in value):
        raise ValueError("invalid node box schema")
    return [(_unpack(L), _unpack(U)) for L, U in value]


def _config_descriptor(config, device):
    return {"enabled": config.enabled, "device": str(device), "dtype": str(config.dtype),
            "max_nodes": config.max_nodes, "qp_max_iterations": config.qp_max_iterations,
            "qp_check_interval": config.qp_check_interval, "abs_gap": float(config.abs_gap),
            "rel_gap": float(config.rel_gap), "wall_seconds": config.wall_seconds}


def _json_equal(a, b):
    # Ordinary dict equality treats True as 1, weakening pinned typed fields.
    return json.dumps(a, sort_keys=True, allow_nan=False, separators=(",", ":")) == json.dumps(
        b, sort_keys=True, allow_nan=False, separators=(",", ":"))


def _summary(model, nodes, incumbent, config):
    regions = [node for node in nodes if node["disposition"] != "branch"]
    bounds = [_unpack(node["bound"]) for node in regions if node["disposition"] != "infeasible"]
    lower = min(bounds) if bounds else None
    if incumbent is None:
        status = "unknown" if bounds else "infeasible_bounded"
        return {"status": status, "lower_bound": _pack(lower) if lower is not None else None,
                "objective": None, "gap_abs": None, "gap_rel": None, "enclosures": None}
    objective = _candidate(model, model.box, incumbent)
    if objective is None or lower is None:
        raise ValueError("incumbent missing feasibility/covered bound")
    gap = _add(objective, -lower)
    if gap < 0:
        raise ArithmeticError("negative signed global gap")
    scale = max(Fraction(1), abs(objective))
    threshold = _add(_fraction(config.abs_gap), _mul(_fraction(config.rel_gap), scale))
    def unclosed_mixed_leaf(node):
        box = _box_unpack(node["box"], model.n)
        return (not all(model.integer)
                and all(not model.integer[i] or L == U for i, (L, U) in enumerate(box)))

    unresolved = any(node["disposition"] != "infeasible"
                     and (node["disposition"] == "unresolved" or unclosed_mixed_leaf(node))
                     and _unpack(node["bound"]) < objective for node in regions)
    status = "unknown" if unresolved else ("solved_bounded" if gap <= threshold else "feasible")
    return {"status": status, "lower_bound": _pack(lower), "objective": _pack(objective),
            "gap_abs": _pack(gap), "gap_rel": _pack(_checked(gap / scale)),
            "enclosures": {"objective": _enclose(objective), "lower_bound": _enclose(lower),
                           "gap_abs": _enclose(gap), "gap_rel": _enclose(_checked(gap / scale))}}


class PreparedBoundedMIQP:
    def __init__(self, problem, config=BranchBoundConfig()):
        self.device = config.validate()
        self.config, self.original = config, problem
        self.model = _Model(problem)
        self.source = source_sha256()

    def _unchanged(self):
        if _descriptor(self.original) != self.model.descriptor:
            raise ValueError("original MIQP model changed after freeze")
        if source_sha256() != self.source:
            raise ValueError("MIQP source changed after freeze")

    def solve(self) -> BranchBoundResult:
        self._unchanged()
        start = time.monotonic()
        model, config = self.model, self.config
        nodes, frontier = [], []
        incumbent, best = None, None
        visited, calls, qp_seconds = 0, 0, 0.0

        def add_node(box, parent=None):
            empty = any(L > U for L, U in box)
            seed = {"kind": "zero_tangent", "z": [0.0] * model.n, "y": [0.0] * model.m}
            B = None if empty else _bound(model, box, seed["z"], seed["y"])
            if parent is not None:
                B = max(B, _unpack(nodes[parent]["bound"]))
            node = {"id": len(nodes), "parent": parent, "box": _box_pack(box),
                    "bound": _pack(B) if B is not None else None,
                    "witnesses": [] if empty else [seed], "disposition": "infeasible" if empty else "frontier",
                    "split": None, "proposal": None, "candidates": [], "visited": False}
            nodes.append(node)
            if not empty:
                frontier.append(node["id"])

        def consider(node, box, values, kind):
            nonlocal incumbent, best
            values = [float(v) for v in values]
            f = _candidate(model, box, values)
            node["candidates"].append({"kind": kind, "x": values, "objective": _pack(f) if f is not None else None})
            if f is not None and (best is None or f < best):
                incumbent, best = values, f
            # This explicit zero-dual witness is independent of QP convergence.
            witness = {"kind": "candidate_tangent", "z": values, "y": [0.0] * model.m}
            B = _bound(model, box, values, witness["y"])
            node["witnesses"].append(witness)
            node["bound"] = _pack(max(_unpack(node["bound"]), B))

        add_node(model.box)
        reason = "tree_exhausted"
        while frontier:
            if visited >= config.max_nodes:
                reason = "node_limit"
                break
            if config.wall_seconds is not None and time.monotonic() - start >= config.wall_seconds:
                reason = "wall_budget"
                break
            index = frontier.pop()
            node = nodes[index]
            box = _box_unpack(node["box"], model.n)
            node["visited"] = True
            visited += 1
            if _row_infeasible(model, box):
                node["disposition"] = "infeasible"
                continue
            if best is not None and _unpack(node["bound"]) >= best:
                node["disposition"] = "pruned"
                continue
            consider(node, box, [L for L, U in box], "lower_endpoint")
            consider(node, box, [U for L, U in box], "upper_endpoint")
            pure = all(model.integer) and all(L == U for L, U in box)
            if pure:
                if best is None or node["candidates"][0]["objective"] is None:
                    raise ValueError("singleton feasibility inconsistent with row proof")
                node["bound"] = node["candidates"][0]["objective"]
                node["disposition"] = "singleton"
                continue
            qp_start = time.monotonic()
            h = model.host
            tensors = [h[k].to(device=self.device, dtype=config.dtype) for k in ("P", "q", "A", "l", "u")]
            P, q, A, l, u = tensors
            for name, cast in (("l", l), ("u", u)):
                finite_original = torch.isfinite(h[name]).to(self.device)
                if bool((finite_original & ~torch.isfinite(cast)).any()):
                    raise ArithmeticError("proposal row-bound cast overflow; finite bounds cannot become infinite")
            L = torch.tensor([float(v[0]) for v in box], device=self.device, dtype=config.dtype)
            U = torch.tensor([float(v[1]) for v in box], device=self.device, dtype=config.dtype)
            if any(not bool(torch.isfinite(t).all()) for t in (P, q, A, L, U)):
                raise ArithmeticError("proposal cast overflow; no alternate-device/dtype retry")
            qp = QPProblem(P, q, torch.cat((A, torch.eye(model.n, device=self.device, dtype=config.dtype))),
                           torch.cat((l, L)), torch.cat((u, U)))
            proposed = solve_qp(qp, QPConfig(device=str(self.device), dtype=config.dtype,
                                            max_iterations=config.qp_max_iterations,
                                            check_interval=config.qp_check_interval))
            calls += 1
            for tensor, shape in ((proposed.x, (model.n,)), (proposed.y, (model.m + model.n,))):
                if (not isinstance(tensor, torch.Tensor) or tensor.device != self.device
                        or tensor.dtype != config.dtype or tensor.shape != shape or not bool(torch.isfinite(tensor).all())):
                    raise RuntimeError("QP proposal actual device/dtype/shape/nonfinite failure")
            if proposed.status not in ("solved", "max_iterations"):
                raise RuntimeError(f"QP proposal failed status={proposed.status}")
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            z = proposed.x.detach().cpu().to(torch.float64).tolist()
            y = proposed.y[:model.m].detach().cpu().to(torch.float64).tolist()
            qp_seconds += time.monotonic() - qp_start
            node["proposal"] = {"status": proposed.status, "iterations": proposed.iterations,
                                "actual_device": str(proposed.x.device), "dtype": str(proposed.x.dtype),
                                "z": z, "y": y}
            witness = {"kind": "qp_tangent", "z": z, "y": y}
            B = _bound(model, box, z, y)
            node["witnesses"].append(witness)
            node["bound"] = _pack(max(_unpack(node["bound"]), B))
            candidate = [float(round(v)) if model.integer[i] else v for i, v in enumerate(z)]
            consider(node, box, candidate, "rounded_integer_proposal")
            if best is not None and _unpack(node["bound"]) >= best:
                node["disposition"] = "pruned"
                continue
            split_index = next((i for i, (L, U) in enumerate(box) if model.integer[i] and L < U), None)
            if split_index is None:
                node["disposition"] = "unresolved"
                continue
            L, U = box[split_index]
            cut = math.floor(_fraction(z[split_index]))
            if not L <= cut < U:
                cut = math.floor(_checked((L + U) / 2))
            left, right = list(box), list(box)
            left[split_index], right[split_index] = (L, Fraction(cut)), (Fraction(cut + 1), U)
            first = len(nodes)
            add_node(left, index)
            add_node(right, index)
            node["split"] = {"index": split_index, "cut": cut, "children": [first, first + 1]}
            node["disposition"] = "branch"
        self._unchanged()
        summary = _summary(model, nodes, incumbent, config)
        elapsed = time.monotonic() - start
        report = {"schema_version": 1, "model": model.descriptor, "source_sha256": self.source,
                  "config": _config_descriptor(config, self.device), "psd_dd_margins": model.psd,
                  "nodes": nodes, "incumbent": incumbent, "summary": summary, "stop_reason": reason,
                  "visited_nodes": visited, "actual_qp_calls": calls,
                  "qp_iteration_limit_count": sum(n["proposal"] is not None and
                      n["proposal"]["status"] == "max_iterations" for n in nodes),
                  "cost_seconds": {"total_before_replay": elapsed, "qp_proposal": qp_seconds,
                                   "host_coordination_and_scalar_proof": max(0.0, elapsed - qp_seconds)},
                  "proof_arithmetic": "host_bounded_fraction_8192_v1",
                  "execution_verdict": "UNVERIFIED", "performance_verdict": "UNVERIFIED"}
        replay_start = time.monotonic()
        report["acceptance"] = check_bounded_miqp_result(self.original, config, report, trusted_source=self.source)
        report["cost_seconds"]["independent_replay"] = time.monotonic() - replay_start
        self._unchanged()
        # Keep original binary64 incumbent, never recast a proof point to float32.
        x = None if incumbent is None else torch.tensor(incumbent, device=self.device, dtype=torch.float64)
        return BranchBoundResult(x, summary["status"], report)


def check_bounded_miqp_result(problem, config, report, *, trusted_source: dict) -> dict:
    """Caller-model/config/source-pinned numerical replay; no solve or factory."""
    device = config.validate()
    model = _Model(problem)
    required = {"schema_version", "model", "source_sha256", "config", "psd_dd_margins", "nodes", "incumbent",
                "summary", "stop_reason", "visited_nodes", "actual_qp_calls", "cost_seconds", "proof_arithmetic",
                "execution_verdict", "performance_verdict", "qp_iteration_limit_count"}
    if not isinstance(report, dict) or set(report) not in (required, required | {"acceptance"}):
        raise ValueError("invalid MIQP report schema")
    if (type(report["schema_version"]) is not int or report["schema_version"] != 1
            or not _json_equal(report["model"], model.descriptor)
            or not _json_equal(report["config"], _config_descriptor(config, device))
            or not _json_equal(report["source_sha256"], trusted_source) or trusted_source != source_sha256()
            or report["psd_dd_margins"] != model.psd
            or report["proof_arithmetic"] != "host_bounded_fraction_8192_v1"
            or report["execution_verdict"] != "UNVERIFIED" or report["performance_verdict"] != "UNVERIFIED"):
        raise ValueError("MIQP model/source/config/arithmetic binding mismatch")
    nodes = report["nodes"]
    if not isinstance(nodes, list) or not 1 <= len(nodes) <= 2 * config.max_nodes + 1:
        raise ValueError("missing/oversized coverage tree")
    reachable, calls, visited, iteration_limits = {0}, 0, 0, 0
    accepted_candidates = []
    node_fields = {"id", "parent", "box", "bound", "witnesses", "disposition", "split", "proposal", "candidates", "visited"}
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or set(node) != node_fields or type(node["id"]) is not int or node["id"] != index:
            raise ValueError("invalid node schema/id")
        box = _box_unpack(node["box"], model.n)
        if index == 0:
            if node["parent"] is not None or box != model.box:
                raise ValueError("root domain differs from original integer ceil/floor")
        elif type(node["parent"]) is not int or not 0 <= node["parent"] < index:
            raise ValueError("invalid node parent")
        if type(node["visited"]) is not bool:
            raise ValueError("invalid node visit flag")
        visited += int(node["visited"])
        if not isinstance(node["witnesses"], list) or not isinstance(node["candidates"], list):
            raise ValueError("missing witness/candidate arrays")
        if len(node["witnesses"]) > 5 or len(node["candidates"]) > 3:
            raise ValueError("oversized bounded witness/candidate arrays")
        if not node["visited"] and (node["proposal"] is not None or node["candidates"]):
            raise ValueError("unvisited region cannot contain executed proposals/candidates")
        empty = any(L > U for L, U in box)
        if empty:
            if index != 0 or node["bound"] is not None or node["witnesses"]:
                raise ValueError("invalid empty integer region")
            bound = None
        else:
            if not node["witnesses"]:
                raise ValueError("covered region missing bound witness")
            values = []
            for witness in node["witnesses"]:
                if not isinstance(witness, dict) or set(witness) != {"kind", "z", "y"} or witness["kind"] not in {
                        "zero_tangent", "candidate_tangent", "qp_tangent"}:
                    raise ValueError("invalid bound witness schema")
                values.append(_bound(model, box, witness["z"], witness["y"]))
            if index:
                values.append(_unpack(nodes[node["parent"]]["bound"]))
            bound = max(values)
        for candidate in node["candidates"]:
            if (not isinstance(candidate, dict) or set(candidate) != {"kind", "x", "objective"}
                    or candidate["kind"] not in {"lower_endpoint", "upper_endpoint", "rounded_integer_proposal"}):
                raise ValueError("invalid candidate schema")
            f = _candidate(model, box, candidate["x"])
            if candidate["objective"] != (_pack(f) if f is not None else None):
                raise ValueError("candidate exact original feasibility/objective mismatch")
            if f is not None:
                accepted_candidates.append((f, candidate["x"]))
        proposal = node["proposal"]
        if proposal is not None:
            fields = {"status", "iterations", "actual_device", "dtype", "z", "y"}
            if (not isinstance(proposal, dict) or set(proposal) != fields
                    or proposal["status"] not in {"solved", "max_iterations"}
                    or proposal["actual_device"] != str(device) or proposal["dtype"] != str(config.dtype)
                    or type(proposal["iterations"]) is not int or not 1 <= proposal["iterations"] <= config.qp_max_iterations):
                raise ValueError("invalid QP proposal device/status/iteration contract")
            _bound(model, box, proposal["z"], proposal["y"])
            witness = {"kind": "qp_tangent", "z": proposal["z"], "y": proposal["y"]}
            if witness not in node["witnesses"] or not node["visited"]:
                raise ValueError("QP proposal witness/count binding missing")
            calls += 1
            iteration_limits += int(proposal["status"] == "max_iterations")
        disposition = node["disposition"]
        if disposition == "singleton":
            if not all(model.integer) or any(L != U for L, U in box):
                raise ValueError("mixed continuous leaf cannot claim singleton closure")
            f = _candidate(model, box, [float(L) for L, U in box])
            if f is None:
                raise ValueError("infeasible singleton marked feasible")
            bound = f
        if node["bound"] != (_pack(bound) if bound is not None else None):
            raise ValueError("inflated/changed original-model bound")
        if disposition == "infeasible":
            if not _row_infeasible(model, box):
                raise ValueError("unproved infeasible region")
        elif disposition == "branch":
            split = node["split"]
            if not isinstance(split, dict) or set(split) != {"index", "cut", "children"}:
                raise ValueError("missing split coverage")
            i, cut, children = split["index"], split["cut"], split["children"]
            if (type(i) is not int or not 0 <= i < model.n or not model.integer[i] or type(cut) is not int
                    or not box[i][0] <= cut < box[i][1] or not isinstance(children, list) or len(children) != 2
                    or any(type(c) is not int or not index < c < len(nodes) for c in children) or children[0] == children[1]):
                raise ValueError("invalid integer split/nonshrinking/lost branch")
            expected = [list(box), list(box)]
            expected[0][i], expected[1][i] = (box[i][0], Fraction(cut)), (Fraction(cut + 1), box[i][1])
            for child, child_box in zip(children, expected):
                if child in reachable or nodes[child]["parent"] != index or _box_unpack(nodes[child]["box"], model.n) != child_box:
                    raise ValueError("duplicate/lost/mismatched child coverage")
                reachable.add(child)
        elif disposition not in {"frontier", "unresolved", "pruned", "singleton"}:
            raise ValueError("unknown region disposition")
        if disposition != "branch" and node["split"] is not None:
            raise ValueError("nonbranch has children")
        if (disposition == "frontier") == node["visited"] and not empty:
            raise ValueError("visited/frontier partition mismatch")
        if disposition == "unresolved" and any(model.integer[i] and L < U for i, (L, U) in enumerate(box)):
            raise ValueError("nonleaf incorrectly marked unresolved")
    if reachable != set(range(len(nodes))):
        raise ValueError("orphan/lost coverage region")
    if type(report["visited_nodes"]) is not int or report["visited_nodes"] != visited or visited > config.max_nodes:
        raise ValueError("visited budget/count mismatch")
    if type(report["actual_qp_calls"]) is not int or report["actual_qp_calls"] != calls:
        raise ValueError("QP proposal count mismatch")
    if type(report["qp_iteration_limit_count"]) is not int or report["qp_iteration_limit_count"] != iteration_limits:
        raise ValueError("QP iteration limit count mismatch")
    incumbent = report["incumbent"]
    if incumbent is not None:
        f = _candidate(model, model.box, incumbent)
        if f is None or not accepted_candidates or (f, incumbent) not in accepted_candidates or f != min(v[0] for v in accepted_candidates):
            raise ValueError("incumbent not the best strictly accepted original-model candidate")
        if any(node["disposition"] == "pruned" and _unpack(node["bound"]) < f for node in nodes):
            raise ValueError("unproved objective pruning")
    elif accepted_candidates or any(node["disposition"] == "pruned" for node in nodes):
        raise ValueError("lost incumbent or pruning without incumbent")
    if not _json_equal(report["summary"], _summary(model, nodes, incumbent, config)):
        raise ValueError("status/lower/objective/signed gap summary mismatch")
    frontier = any(node["disposition"] == "frontier" for node in nodes)
    reason = report["stop_reason"]
    if (reason not in {"tree_exhausted", "node_limit", "wall_budget"}
            or (reason == "tree_exhausted" and frontier)
            or (reason == "node_limit" and (not frontier or visited != config.max_nodes))
            or (reason == "wall_budget" and (not frontier or config.wall_seconds is None))):
        raise ValueError("stop reason/budget partition mismatch")
    cost = report["cost_seconds"]
    keys = {"total_before_replay", "qp_proposal", "host_coordination_and_scalar_proof"}
    if not isinstance(cost, dict) or set(cost) not in (keys, keys | {"independent_replay"}) or any(
            type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in cost.values()):
        raise ValueError("invalid disclosed cost metrics")
    acceptance = {"numerical_verdict": "PASS", "coverage_verdict": "PASS", "execution_verdict": "UNVERIFIED",
                  "scope": "tiny finite-box exact-DD convex MIQP scalar bound/candidate/coverage only"}
    if "acceptance" in report and not _json_equal(report["acceptance"], acceptance):
        raise ValueError("embedded acceptance cannot claim stronger execution/qualification")
    return acceptance


def prepare_bounded_miqp(problem, config=BranchBoundConfig()):
    return PreparedBoundedMIQP(problem, config)


def solve_bounded_miqp(problem, config=BranchBoundConfig()):
    return prepare_bounded_miqp(problem, config).solve()
