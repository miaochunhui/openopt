"""Seal and independently accept real GPU-campaign / solver-comparison evidence.

No producer code is imported or executed. A caller-pinned bundle hash binds
result bytes, sidecars and the full source set. Sealing is not acceptance and
does not authenticate evidence against a party that can replace the caller's
pin. CLI artifacts must be external 3FS directories; previous runs are retained.
"""

from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import traceback
from xml.etree import ElementTree

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("P", "q", "A", "l", "u")
METRICS = ("primal_residual", "dual_residual", "normal_cone_residual",
           "relative_objective_gap")
SOLVED = {"openopt": "solved", "osqp": "solved", "clarabel": "Solved"}
GPU_LIMITS = dict(zip(METRICS, (5e-6, 5e-6, 5e-6, 1e-6), strict=True))


class Rejected(ValueError):
    """A failed criterion with its first cause and disk/sample coordinate."""

    def __init__(self, coordinate: str, cause: str):
        super().__init__(f"{coordinate}: {cause}")
        self.coordinate, self.cause = coordinate, cause


def require(condition: bool, coordinate: str, cause: str) -> None:
    if not condition:
        raise Rejected(coordinate, cause)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def number(value, coordinate: str, *, positive=False) -> float:
    require(type(value) in (int, float), coordinate, "real number required")
    require(math.isfinite(value) and (value > 0 if positive else value >= 0),
            coordinate, "positive finite required" if positive else "nonnegative finite required")
    return float(value)


def integer(value, coordinate: str, minimum=0) -> int:
    require(type(value) is int and value >= minimum, coordinate, f"integer >= {minimum} required")
    return value


def text(value, coordinate: str) -> str:
    require(isinstance(value, str) and bool(value.strip()), coordinate, "nonempty string required")
    return value


def digest(value, coordinate: str) -> str:
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
            coordinate, "SHA256 required")
    return value


def pairs(items):
    output = {}
    for key, value in items:
        require(key not in output, "json", f"duplicate key {key}")
        output[key] = value
    return output


def parse(data: bytes, coordinate: str):
    def nonfinite(value):
        raise Rejected(coordinate, f"nonfinite JSON constant {value}")
    try:
        return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise Rejected(coordinate, f"invalid JSON: {error}") from error


def read(path: Path, *, allow_empty=False) -> bytes:
    require(not path.is_symlink() and path.is_file(), str(path), "regular non-symlink file required")
    data = path.read_bytes()
    require(bool(data) or allow_empty, str(path), "empty evidence")
    return data


def descriptor(path: Path) -> dict:
    return {"path": str(path.absolute()), "sha256": sha(read(path))}


def bound(item: dict) -> bytes:
    path = Path(item["path"])
    require(path.is_absolute(), str(path), "absolute evidence path required")
    data = read(path)
    require(sha(data) == digest(item["sha256"], str(path)), str(path), "evidence hash mismatch")
    return data


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and "." not in path.parts
            and str(path) == value, value, "unsafe source path")
    require(value.startswith(("src/", "scripts/", "tests/", "examples/")),
            value, "source outside public snapshot scope")
    return value


def source_files(root: Path) -> dict:
    files = {}
    for directory in ("src", "scripts", "tests", "examples"):
        base = root / directory
        if not base.exists():
            continue
        for path in sorted(base.rglob("*")):
            require(not path.is_symlink(), str(path), "source symlink forbidden")
            if path.is_file() and "__pycache__" not in path.parts:
                files[path.relative_to(root).as_posix()] = sha(read(path, allow_empty=True))
    require(bool(files), str(root), "empty source snapshot")
    return files


def check_sources(root: Path, mapping: dict) -> None:
    require(isinstance(mapping, dict) and bool(mapping), str(root), "nonempty source hashes required")
    for relative, expected in mapping.items():
        path = root / relative_path(relative)
        require(sha(read(path, allow_empty=True)) == digest(expected, str(path)), str(path), "source hash mismatch")


def kind_of(report: dict) -> str:
    if "arms" in report and "configuration" in report:
        return "comparison"
    require(report.get("mode") in ("correctness", "adaptive", "batch", "operators"),
            "result", "unsupported result schema")
    return "gpu"


def seal_entry(result: Path, source_root: Path | None = None, run_path: Path | None = None) -> dict:
    report = parse(read(result), str(result))
    kind = kind_of(report)
    sidecars = {}
    run = None
    if kind == "gpu":
        run_path = run_path or result.parent.parent / "run.json"
        record = parse(read(run_path), str(run_path))
        source_root = source_root or run_path.parent / "source"
        claimed = record["source_hashes"]
        source_manifest = source_root / "source-hashes.json"
        require(parse(read(source_manifest), str(source_manifest)) == claimed,
                str(source_manifest), "run/snapshot source manifest mismatch")
        require(claimed.get("scripts/gpu_qp_campaign.py") == report.get("source_hash"),
                str(result), "GPU runner source hash mismatch")
        require(any(Path(job["output"]).resolve() == result.parent.resolve()
                    for job in record["jobs"]), str(result), "result absent from run jobs")
        run = descriptor(run_path)
        sidecars["source-hashes.json"] = descriptor(source_manifest)
        names = ("telemetry.json", "execution.log", "pytest.log", "pytest.xml") \
            if report["mode"] == "correctness" else (
                "telemetry.json", "execution.log", "nsys-stats.csv",
                "profile.nsys-rep", "profile-wall.json", "problem.npz",
            )
        if report["mode"] == "correctness":
            for label, default in (("dense", "problem-dense.npz"), ("sparse", "problem-sparse.npz")):
                name = report["result"].get("problem_files", {}).get(label, default)
                require(isinstance(name, str) and Path(name).name == name,
                        "problem_files", "problem sidecar must be local basename")
                sidecars[f"problem-{label}.npz"] = descriptor(result.parent / name)
    else:
        require(source_root is not None, str(result), "comparison needs explicit source root")
        claimed = report.get("source_hashes")
        require(isinstance(claimed, dict) and set(claimed) >= {
            "scripts/benchmark_solver_comparison.py", "scripts/benchmark_general_qp.py",
            "src/openopt/qp.py",
        }, str(result), "comparison producer source hashes missing")
        names = ("problem.npz", "telemetry.json", "execution.log")
    source_root = source_root.absolute()
    check_sources(source_root, claimed)
    sources = source_files(source_root)
    if kind == "gpu":
        require(sources == claimed, str(source_root), "snapshot source set mismatch")
    for name in names:
        sidecars[name] = descriptor(result.parent / name)
    return {"kind": kind, "result": descriptor(result), "source_root": str(source_root),
            "sources": sources, "sidecars": sidecars, "run": run}


def seal(results: list[Path], roots: list[Path] | None = None,
         runs: list[Path] | None = None) -> dict:
    require(bool(results), "bundle", "empty result set")
    require(len(set(p.resolve() for p in results)) == len(results), "bundle", "duplicate result")
    require(not roots or len(roots) == len(results), "bundle", "one source root per result required")
    require(not runs or len(runs) == len(results), "bundle", "one run record per result required")
    return {"schema_version": 1, "sealed_at": now(), "entries": [
        seal_entry(path, roots[i] if roots else None, runs[i] if runs else None)
        for i, path in enumerate(results)
    ]}


def latency(value: dict, coordinate: str) -> list[float]:
    samples = value["samples_s"]
    require(isinstance(samples, list) and len(samples) >= 3, coordinate, "at least 3 samples required")
    for i, sample in enumerate(samples):
        number(sample, f"{coordinate}/{i}", positive=True)
    require(integer(value["count"], coordinate, 3) == len(samples), coordinate, "sample count mismatch")
    for key, quantile in (("p50_s", 50), ("p95_s", 95)):
        actual = number(value[key], coordinate + "/" + key, positive=True)
        require(math.isclose(actual, float(np.percentile(samples, quantile)), rel_tol=1e-12, abs_tol=0),
                coordinate, "latency statistics mismatch")
    return samples


def profile_tables(data: bytes) -> dict:
    tables, current = [], None
    for line in data.decode().splitlines():
        if line.startswith("Time (%),"):
            current = []
            tables.append(current)
        elif current is not None and line[:1].isdigit():
            row = next(csv.reader([line]))
            require(len(row) >= 3, "nsys-stats.csv", "malformed profiler row")
            current.append(row)
    require(len(tables) == 3 and all(tables), "nsys-stats.csv", "three nonempty profiler tables required")
    total_ns, instances = 0.0, 0
    for row in tables[0]:
        total_ns += number(float(row[1]), "nsys/kernel_time", positive=True)
        instances += integer(int(row[2]), "nsys/kernel_instances", 1)
    return {"kernel_total_s": total_ns * 1e-9, "kernel_instances": instances}


def check_quality(value: dict, coordinate: str, limits: dict) -> None:
    require(value.get("verdict") == "PASS", coordinate, "independent quality verdict not PASS")
    for key in METRICS:
        actual = number(value[key], coordinate + "/" + key)
        require(actual <= limits[key], coordinate, f"{key} exceeds independent threshold")
    if "solution_error" in value:
        number(value["solution_error"], coordinate + "/solution_error")


def norm(value) -> float:
    return float(np.max(np.abs(value))) if value.size else 0.0


def load_problem(data: bytes, coordinate: str) -> dict:
    with np.load(io.BytesIO(data), allow_pickle=False) as saved:
        require(set(saved.files) >= set(KEYS) | {"expected", "witness_y"},
                coordinate, "problem/witness arrays missing")
        arrays = {key: np.array(saved[key], copy=True) for key in (*KEYS, "expected", "witness_y")}
    P, q, A, l, u = (arrays[k] for k in KEYS)
    require(q.ndim == 1 and q.size > 0 and P.shape == (q.size, q.size), coordinate, "invalid P/q shape")
    require(A.ndim == 2 and A.shape[1] == q.size and l.shape == u.shape == (A.shape[0],),
            coordinate, "invalid A/l/u shape")
    require(all(a.dtype == P.dtype for a in arrays.values()) and P.dtype in (np.float32, np.float64),
            coordinate, "matched real float32/float64 arrays required")
    require(all(np.isfinite(arrays[k]).all() for k in ("P", "q", "A", "expected", "witness_y")),
            coordinate, "nonfinite problem/witness")
    require(not np.isnan(l).any() and not np.isnan(u).any() and not np.isposinf(l).any()
            and not np.isneginf(u).any() and bool(np.all(l <= u)), coordinate, "invalid bounds")
    tolerance = 20 * np.finfo(P.dtype).eps * max(1.0, norm(P))
    require(norm(P - P.T) <= tolerance
            and np.linalg.eigvalsh((P + P.T) * 0.5).min() >= -tolerance,
            coordinate, "problem is not symmetric PSD")
    return arrays


def problem_digest(arrays: dict) -> str:
    hasher = hashlib.sha256()
    for key in KEYS:
        a = arrays[key]
        hasher.update(key.encode())
        hasher.update(str(a.shape).encode())
        hasher.update(a.dtype.str.encode())
        hasher.update(a.tobytes())
    return hasher.hexdigest()


def recompute(arrays: dict, solution: dict, eps_abs: float, eps_rel: float, coordinate: str) -> dict:
    P, q, A, l, u = (arrays[k] for k in KEYS)
    x, y = np.asarray(solution["x"]), np.asarray(solution["y"])
    require(x.shape == q.shape and y.shape == l.shape and np.isfinite(x).all()
            and np.isfinite(y).all(), coordinate, "invalid/nonfinite host solution")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        Ax, Px, ATy = A @ x, P @ x, A.T @ y
        projected = np.clip(Ax, l, u)
        optimum = float(0.5 * arrays["expected"] @ P @ arrays["expected"]
                        + q @ arrays["expected"])
        objective = float(0.5 * x @ P @ x + q @ x)
        result = {
            "primal_residual": norm(Ax - projected), "dual_residual": norm(Px + q + ATy),
            "normal_cone_residual": norm(projected - np.clip(projected + y, l, u)),
            "relative_objective_gap": abs(objective - optimum) / max(1.0, abs(optimum)),
            "primal_tolerance": eps_abs + eps_rel * max(norm(Ax), norm(projected)),
            "dual_tolerance": eps_abs + eps_rel * max(norm(Px), norm(q), norm(ATy)),
            "objective_gap_tolerance": eps_abs + eps_rel,
            "objective": objective, "witness_objective": optimum,
            "solution_error": norm(x - arrays["expected"]),
        }
    limits = {METRICS[0]: result["primal_tolerance"], METRICS[1]: result["dual_tolerance"],
              METRICS[2]: result["primal_tolerance"], METRICS[3]: result["objective_gap_tolerance"]}
    check_quality({"verdict": "PASS", **result}, coordinate, limits)
    return result


def context(report: dict, *, gpu=False) -> tuple[dict, list[str]]:
    hardware = report["hardware"]
    host = text(hardware.get("node") or hardware.get("host"), "hardware/host")
    threads = integer(hardware["cpu_threads" if gpu else "torch_threads"], "hardware/threads", 1)
    result = {"host": host, "threads": threads,
              "quota": hardware.get("cpu_quota" if gpu else "cpu_max"),
              "problem_hash": report["result"]["problem_hash"] if gpu else report["problem_hash"]}
    reasons = []
    if gpu:
        config = report["result"]
        result.update(dtype=config.get("dtype"), eps_abs=config.get("eps_abs"),
                      eps_rel=config.get("eps_rel"), gpu=hardware.get("gpu"))
        if result["quota"] is None:
            reasons.append("CPU quota unverified")
        if result["dtype"] not in ("float32", "float64", "torch.float32", "torch.float64"):
            reasons.append("dtype unverified")
        for name in ("eps_abs", "eps_rel"):
            if result[name] is None:
                reasons.append(name + " unverified")
            else:
                number(result[name], "configuration/" + name, positive=True)
    else:
        config = report["configuration"]
        require(threads == config["cpu_threads"], "hardware/threads", "requested/observed threads mismatch")
        pools = report.get("thread_pools")
        if not pools:
            reasons.append("BLAS thread pools unverified")
        for pool in pools or []:
            if pool.get("status") != "MEASURED" or pool.get("threads") is None:
                reasons.append("BLAS threads unverified")
            else:
                require(pool["threads"] == threads, "thread_pools", "observed thread limit mismatch")
        result.update(eps_abs=number(config["eps_abs"], "eps_abs", positive=True),
                      eps_rel=number(config["eps_rel"], "eps_rel", positive=True))
    return result, reasons


def commercial(report: dict) -> dict:
    rows = report.get("commercial", {})
    # Discovery/status declarations are not matched benchmark acceptance.
    for name, row in rows.items():
        require(row.get("status") in ("UNVERIFIED", "UNAVAILABLE", "PASS", "FAIL"),
                "commercial/" + name, "invalid commercial status")
    return {name: "UNVERIFIED" for name in ("gurobi", "cplex", "mosek")}


def validate_comparison(report: dict, sidecars: dict) -> dict:
    require(integer(report["exit_code"], "exit_code") == 0 and not report.get("failures"),
            "comparison", "nonzero exit or retained execution failure")
    arrays = load_problem(sidecars["problem.npz"], "problem.npz")
    require(problem_digest(arrays) == digest(report["problem_hash"], "problem_hash"),
            "problem.npz", "same-problem hash mismatch")
    ctx, reasons = context(report)
    ctx["dtype"] = str(arrays["P"].dtype)
    count = integer(report["configuration"]["repeats"], "configuration/repeats", 3)
    recompute(arrays, {"x": arrays["expected"], "y": arrays["witness_y"]},
              ctx["eps_abs"], ctx["eps_rel"], "witness")
    rows = report["arms"]
    require(isinstance(rows, list) and bool(rows), "arms", "empty arms")
    ids = [text(row["id"], "arm/id") for row in rows]
    require(len(set(ids)) == len(ids), "arms", "duplicate arm IDs")
    accepted, checked = [], 0
    for row in rows:
        coordinate = "arm/" + row["id"]
        require(row["status"] in ("PASS", "UNVERIFIED"), coordinate, "arm not terminal PASS/UNVERIFIED")
        require(set(row["modes"]) == {"cold", "prepared"}, coordinate, "cold/prepared must stay separate")
        if row["status"] == "UNVERIFIED":
            for mode in row["modes"].values():
                require(mode["status"] == "UNVERIFIED" and not mode["samples"]
                        and mode["latency"] is None, coordinate, "UNVERIFIED arm contains timings")
            continue
        require(row["solver"] in SOLVED, coordinate, "unsupported measured solver")
        config = row["config"]
        require(sha(json.dumps(config, sort_keys=True).encode()) == row["config_hash"],
                coordinate, "solver configuration hash mismatch")
        abs_key, rel_key = ("tol_gap_abs", "tol_gap_rel") if row["solver"] == "clarabel" \
            else ("eps_abs", "eps_rel")
        require(config[abs_key] == ctx["eps_abs"] and config[rel_key] == ctx["eps_rel"],
                coordinate, "solver eps mismatch")
        if row["solver"] == "openopt":
            require(config["dtype"] == "torch." + ctx["dtype"] and config["device"] == row["device"],
                    coordinate, "solver dtype/device mismatch")
        if row["solver"] == "clarabel":
            require(config["max_threads"] == ctx["threads"] and config["tol_feas"] == ctx["eps_abs"],
                    coordinate, "Clarabel threads/feas eps mismatch")
        for mode, measured in row["modes"].items():
            where = coordinate + "/" + mode
            require(measured["status"] == "PASS", where, "mode not PASS")
            times = latency(measured["latency"], where)
            samples = measured["samples"]
            require(len(samples) == len(times) == count, where, "incomplete sample set")
            require([s["repeat"] for s in samples] == list(range(count)), where, "repeat mismatch")
            if mode == "prepared":
                number(measured["setup_once_s"], where + "/setup_once_s", positive=True)
            else:
                require(measured["setup_once_s"] is None, where, "cold contains prepared setup")
            for i, sample in enumerate(samples):
                point = where + f"/repeat-{i}"
                require(sample["solver_status"] == SOLVED[row["solver"]], point,
                        "solver status is not strictly solved")
                require(sample["verdict"] == "PASS" and sample["arm"] == row["id"]
                        and sample["mode"] == mode, point, "sample verdict/arm/mode mismatch")
                require(sample["wall_s"] == times[i], point, "wall/latency sample mismatch")
                solve_time = number(sample["solve_host_result_s"], point + "/solve_host_result_s",
                                    positive=True)
                require(solve_time <= times[i], point, "solve exceeds timed wall")
                if mode == "cold":
                    setup = number(sample["setup_s"], point + "/setup_s", positive=True)
                    require(math.isclose(setup + solve_time, times[i], rel_tol=1e-9),
                            point, "cold setup/solve wall mismatch")
                else:
                    require(sample["setup_s"] is None, point, "prepared sample includes cold setup")
                if row["solver"] == "openopt":
                    require(sample.get("observed_tensor_device") == row["device"], point,
                            "actual tensor placement missing/mismatched")
                actual = recompute(arrays, sample["host_solution"], ctx["eps_abs"], ctx["eps_rel"], point)
                require(sample["quality"]["verdict"] == "PASS", point, "quality verdict not PASS")
                for key, value in actual.items():
                    saved = sample["quality"][key]
                    require(type(saved) in (int, float) and math.isfinite(saved)
                            and math.isclose(saved, value, rel_tol=1e-8, abs_tol=1e-12),
                            point, f"independent quality mismatch: {key}")
                checked += 1
            accepted.append({"arm": row["id"], "solver": row["solver"], "mode": mode,
                             "context": ctx, "p50_s": measured["latency"]["p50_s"],
                             "p95_s": measured["latency"]["p95_s"], "control_of": row.get("control_of")})
    require(checked > 0 and any(row["solver"] == "openopt" for row in accepted),
            "comparison", "no accepted openopt samples")
    orders = report["orders"]
    require(len(orders) == 2 * count, "orders", "incomplete order schedule")
    for mode in ("cold", "prepared"):
        selected = [o for o in orders if o["mode"] == mode]
        require(len(selected) == count, "orders", "missing mode orders")
        for i, order in enumerate(selected):
            expected = ids if i % 2 == 0 else ids[::-1]
            require(order["repeat"] == i and order["arms"] == expected, "orders", "order mismatch")
            for row in rows:
                if row["status"] == "PASS":
                    require(row["modes"][mode]["samples"][i]["order_position"] == expected.index(row["id"]),
                            "orders", "sample order position mismatch")
    controls = {}
    by_id = {r["id"]: r for r in rows}
    for row in rows:
        if not row.get("control_of") or row["status"] != "PASS":
            continue
        baseline = by_id[row["control_of"]]
        require(baseline["status"] == "PASS" and baseline["config"] == row["config"]
                and baseline["solver"] == row["solver"] and baseline["device"] == row["device"],
                row["id"], "identical control mismatch")
        for mode in ("cold", "prepared"):
            a = baseline["modes"][mode]["latency"]["samples_s"]
            b = row["modes"][mode]["latency"]["samples_s"]
            controls[(baseline["id"], mode)] = float(np.percentile(
                [abs(x - y) for x, y in zip(a, b, strict=True)], 95))
    for row in accepted:
        if row["control_of"] is None and (row["arm"], row["mode"]) not in controls:
            reasons.append("identical control unavailable: " + row["arm"])
    saved_controls = report.get("controls")
    require(isinstance(saved_controls, list), "controls", "control evidence missing")
    saved = {}
    for item in saved_controls:
        require(item["identical_config_control"] is True, "controls", "control not identical")
        twin, original = by_id[item["control"]], by_id[item["arm"]]
        require(twin["control_of"] == original["id"], "controls", "control linkage mismatch")
        for mode, value in item["modes"].items():
            key = (original["id"], mode)
            require(key not in saved and key in controls, "controls", "duplicate/unknown control")
            require(value["status"] == "PASS", "controls", "control not PASS")
            deltas = [abs(a - b) for a, b in zip(
                original["modes"][mode]["latency"]["samples_s"],
                twin["modes"][mode]["latency"]["samples_s"], strict=True)]
            require(value["paired_abs_delta_s"] == deltas
                    and math.isclose(value["empirical_noise_floor_s"], controls[key], rel_tol=1e-12),
                    "controls", "saved paired control/noise floor mismatch")
            saved[key] = True
    require(set(saved) == set(controls), "controls", "incomplete control evidence")
    return {"schema": "benchmark_solver_comparison", "checked_samples": checked, "rows": accepted,
            "ranking_reasons": sorted(set(reasons)), "controls": [
                {"arm": arm, "mode": mode, "noise_floor_s": value}
                for (arm, mode), value in controls.items()], "commercial": commercial(report)}


def validate_gpu(report: dict, sidecars: dict) -> dict:
    hardware = report["hardware"]
    for key in ("node", "gpu", "torch", "cuda", "cpu_quota"):
        text(hardware.get(key), "hardware/" + key)
    integer(hardware["cpu_threads"], "hardware/cpu_threads", 1)
    require("RTX PRO 4500" in hardware["gpu"], "hardware/gpu", "unqualified GPU")
    telemetry = parse(sidecars["telemetry.json"], "telemetry.json")
    require(isinstance(telemetry, list) and bool(telemetry)
            and all(s.get("rc") == 0 and s.get("gpu") for s in telemetry),
            "telemetry.json", "missing/failed GPU telemetry")
    result = report["result"]
    if report["mode"] == "correctness":
        checks = result["checks"]
        require(isinstance(checks, dict) and bool(checks), "correctness/checks", "empty checks")
        required = {"dense_adaptive_False", "dense_adaptive_True", "batch_0", "batch_1", "batch_2",
                    "sparse_coo", "sparse_csr"}
        require(set(checks) >= required, "correctness/checks", "new-path coverage missing")
        problems = {label: load_problem(sidecars[f"problem-{label}.npz"], f"problem-{label}.npz")
                    for label in ("dense", "sparse")}
        for label, arrays in problems.items():
            recompute(arrays, {"x": arrays["expected"], "y": arrays["witness_y"]},
                      1e-7, 1e-7, "correctness/" + label + "/witness")
        for name, measured in checks.items():
            check_quality(measured, "correctness/" + name, GPU_LIMITS)
            require(measured.get("solver_status", measured.get("status")) == "solved",
                    "correctness/" + name, "solver status evidence missing/not solved")
            arrays = problems["sparse" if name.startswith("sparse_") else "dense"]
            require("host_x" in measured and "host_y" in measured,
                    "correctness/" + name, "host solution evidence missing")
            actual = recompute(arrays, {"x": measured["host_x"], "y": measured["host_y"]},
                               1e-7, 1e-7, "correctness/" + name)
            for key in (*METRICS, "objective", "solution_error"):
                require(math.isclose(measured[key], actual[key], rel_tol=1e-8, abs_tol=1e-12),
                        "correctness/" + name, "GPU independent quality mismatch: " + key)
        tree = ElementTree.fromstring(sidecars["pytest.xml"])
        suites = [tree] if tree.tag == "testsuite" else tree.findall(".//testsuite")
        counts = {key: sum(int(s.attrib.get(key, 0)) for s in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        for key, value in result["tests"].items():
            integer(value, "correctness/tests/" + key)
        require(counts == result["tests"] and counts["tests"] > 0
                and not any(counts[k] for k in ("failures", "errors", "skipped")),
                "pytest.xml", "zero/incomplete/failing/skipped CUDA regression")
        log = sidecars["pytest.log"].decode()
        require(re.search(rf"\b{counts['tests']} passed\b", log) is not None,
                "pytest.log", "completed pytest summary missing")
        return {"schema": "gpu_qp_campaign", "checked_samples": len(checks), "rows": [],
                "tests": counts, "ranking_reasons": ["correctness is not a performance ranking"],
                "independent_solution_replay": "PASS",
                "controls": [], "commercial": commercial(report)}
    ctx, reasons = context(report, gpu=True)
    digest(ctx["problem_hash"], "problem_hash")
    arrays = load_problem(sidecars["problem.npz"], "problem.npz")
    raw_hash = sha(b"".join(arrays[key].tobytes() for key in KEYS))
    require(raw_hash == ctx["problem_hash"], "problem.npz", "GPU same-problem hash mismatch")
    require(str(arrays["P"].dtype) == ctx["dtype"], "problem.npz", "GPU dtype mismatch")
    require(arrays["q"].size == integer(report["size"], "size", 3),
            "problem.npz", "GPU problem size mismatch")
    require(ctx["eps_abs"] is not None and ctx["eps_rel"] is not None,
            "result", "GPU eps missing")
    recompute(arrays, {"x": arrays["expected"], "y": arrays["witness_y"]},
              ctx["eps_abs"], ctx["eps_rel"], "GPU/witness")
    configs = result["arm_config"]
    require(set(configs) == {"baseline", "candidate", "control"}, "arm_config", "three configs required")
    require(configs["baseline"] == configs["control"], "arm_config", "baseline/control config mismatch")
    expected_base = {"rho": 1.0, "adaptive_rho": False, "linear_solver": "cholesky"}
    expected_candidate = {**expected_base, "adaptive_rho": report["mode"] == "adaptive",
                          "linear_solver": "CG" if report["mode"] == "operators" else "cholesky"}
    for name, expected in (("baseline", expected_base), ("control", expected_base),
                           ("candidate", expected_candidate)):
        require(all(configs[name].get(k) == v for k, v in expected.items()), "arm_config/" + name,
                "unexpected algorithm/configuration difference")
        for key in ("dtype", "eps_abs", "eps_rel"):
            require(configs[name].get(key, result[key]) == result[key],
                    "arm_config/" + name, "arm dtype/eps parity mismatch")
        for key in set(configs[name]) - set(expected) - {"dtype", "eps_abs", "eps_rel"}:
            require(configs[name][key] == configs["baseline"].get(key), "arm_config/" + name,
                    "undeclared arm configuration difference")
    require(result["input_generation"].startswith("excluded")
            and result["control_order"] == "forward/reverse alternating; one untimed warmup",
            "timing", "model-cold/alternating timing contract missing")
    batch = integer(result["batch_size"], "batch_size", 1)
    require(batch == (8 if report["mode"] == "batch" else 1), "batch_size", "workload mismatch")
    require(type(result["validation_parity"]) is bool, "validation_parity", "boolean required")
    require(result["validation_parity"] == (report["mode"] != "operators"),
            "validation_parity", "unexpected validation claim")
    if not result["validation_parity"]:
        reasons.append("validation parity not established; sparse caller PSD assumption")
    text(result["scope"], "scope")
    require(set(result["measurements"]) == {"cold", "prepared"}, "measurements",
            "cold/prepared must stay separate")
    rows, controls, checked = [], [], 0
    for mode, arms in result["measurements"].items():
        require(set(arms) == {"baseline", "candidate", "control"}, mode, "three complete arms required")
        times = {}
        for name, arm in arms.items():
            coordinate = mode + "/" + name
            values = latency(arm["latency"], coordinate)
            times[name] = values
            require(len(arm["quality"]) == len(values) * batch, coordinate, "quality/sample count mismatch")
            for i, measured in enumerate(arm["quality"]):
                check_quality(measured, coordinate + f"/quality-{i}", GPU_LIMITS)
                require(measured.get("solver_status", measured.get("status")) == "solved",
                        coordinate + f"/quality-{i}", "solver status evidence missing/not solved")
                point = coordinate + f"/quality-{i}"
                require("host_x" in measured and "host_y" in measured,
                        point, "host solution evidence missing")
                actual = recompute(arrays, {"x": measured["host_x"], "y": measured["host_y"]},
                                   ctx["eps_abs"], ctx["eps_rel"], point)
                for key in (*METRICS, "objective", "solution_error"):
                    value = measured[key]
                    require(type(value) in (int, float) and math.isfinite(value)
                            and math.isclose(value, actual[key], rel_tol=1e-8, abs_tol=1e-12),
                            point, "GPU independent quality mismatch: " + key)
                if "solution_error" in measured:
                    require(measured["solution_error"] <= 1e-5, coordinate, "solution error too high")
                checked += 1
            rows.append({"arm": name, "solver": "openopt", "mode": mode, "context": ctx,
                         "p50_s": arm["latency"]["p50_s"], "p95_s": arm["latency"]["p95_s"],
                         "control_of": "baseline" if name == "control" else None})
        require(len({len(v) for v in times.values()}) == 1, mode, "unmatched repeat counts")
        controls.append({"arm": "baseline", "mode": mode, "noise_floor_s": float(np.percentile(
            [abs(x - y) for x, y in zip(times["baseline"], times["control"], strict=True)], 95))})
    nsight = report["nsight"]
    for key in ("kernel_total_s", "profiled_prepared_e2e_s"):
        number(nsight[key], "nsight/" + key, positive=True)
    integer(nsight["kernel_instances"], "nsight/kernel_instances", 1)
    actual_profile = profile_tables(sidecars["nsys-stats.csv"])
    require(actual_profile["kernel_instances"] == nsight["kernel_instances"]
            and math.isclose(actual_profile["kernel_total_s"], nsight["kernel_total_s"], rel_tol=1e-12),
            "nsight", "raw kernel profiler metrics mismatch")
    profile = parse(sidecars["profile-wall.json"], "profile-wall.json")
    require(profile["profiled_prepared_e2e_s"] == nsight["profiled_prepared_e2e_s"],
            "nsight", "profile wall sidecar mismatch")
    return {"schema": "gpu_qp_campaign", "checked_samples": checked, "rows": rows,
            "independent_solution_replay": "PASS", "control_kind": "baseline-candidate-control",
            "ranking_reasons": sorted(set(reasons)), "controls": controls,
            "commercial": commercial(report)}


def rank_pairs(results: list[dict]) -> dict:
    comparisons, excluded = [], []
    # Only within-result sequential pairs share the measurement clock/order.
    # Cross-result and cross-host aggregation is deliberately not attempted.
    for index, result in enumerate(results):
        rows = [r for r in result["rows"] if r["control_of"] is None]
        for i, left in enumerate(rows):
            for right in rows[i + 1:]:
                if left["mode"] != right["mode"]:
                    continue
                coordinate = f"result-{index}/{left['mode']}/{left['arm']}:{right['arm']}"
                reasons = list(result["ranking_reasons"])
                if left["context"] != right["context"]:
                    reasons.append("same problem/host/thread/dtype/eps context mismatch")
                if reasons:
                    excluded.append({"coordinate": coordinate, "status": "UNVERIFIED",
                                     "reasons": sorted(set(reasons))})
                    continue
                floors = [c["noise_floor_s"] for c in result["controls"]
                          if c["mode"] == left["mode"] and c["arm"] in (left["arm"], right["arm"])]
                require(len(floors) == (1 if result.get("control_kind") == "baseline-candidate-control"
                                       else 2), coordinate, "paired controls missing")
                delta = abs(left["p50_s"] - right["p50_s"])
                comparisons.append({"coordinate": coordinate,
                                    "status": "MEASURED_DIFFERENCE" if delta > max(floors)
                                    else "INCONCLUSIVE_NOISE",
                                    "faster_arm": (left if left["p50_s"] < right["p50_s"] else right)["arm"]
                                    if delta > max(floors) else None,
                                    "p50_delta_s": delta, "noise_floor_s": max(floors),
                                    "scope": "one frozen synthetic problem, same-host model-" + left["mode"]})
    return {"status": "SCOPED" if comparisons else "UNVERIFIED", "pairs": comparisons,
            "excluded": excluded, "cross_result_ranking": "UNVERIFIED",
            "cross_host_ranking": "UNVERIFIED", "commercial_ranking": "UNVERIFIED",
            "commercial_win": False}


def accept(bundle_path: Path, expected_sha256: str) -> dict:
    raw = read(bundle_path)
    require(sha(raw) == digest(expected_sha256, "bundle_sha256"), str(bundle_path), "bundle hash mismatch")
    bundle = parse(raw, str(bundle_path))
    require(bundle.get("schema_version") == 1 and isinstance(bundle.get("entries"), list)
            and bool(bundle["entries"]), "bundle", "empty/unsupported bundle")
    checked = []
    seen = set()
    for index, entry in enumerate(bundle["entries"]):
        coordinate = f"entry-{index}"
        data = bound(entry["result"])
        path = entry["result"]["path"]
        require(path not in seen, coordinate, "duplicate result")
        seen.add(path)
        root = Path(entry["source_root"])
        require(root.is_absolute(), coordinate, "absolute source root required")
        check_sources(root, entry["sources"])
        require(source_files(root) == entry["sources"], coordinate, "source set changed since seal")
        report = parse(data, path)
        require(report.get("verdict") == "PASS", path,
                "producer not PASS; first cause=" + str(report.get("error")))
        number(report["wall_s"], path + "/wall_s", positive=True)
        require(kind_of(report) == entry["kind"], coordinate, "schema kind mismatch")
        sidecars = {name: bound(item) for name, item in entry["sidecars"].items()}
        marker = sidecars["execution.log"].decode()
        expected_marker = "QP_CAMPAIGN_GPU" if entry["kind"] == "gpu" else "OPENOPT_SOLVER_COMPARISON"
        require(expected_marker in marker and "verdict=PASS" in marker,
                path, "execution completion marker missing")
        telemetry = parse(sidecars["telemetry.json"], "telemetry.json")
        require(isinstance(telemetry, list) and bool(telemetry), path, "empty telemetry")
        if entry["kind"] == "gpu":
            record = parse(bound(entry["run"]), "run.json")
            require(record["source_hashes"] == entry["sources"]
                    == parse(sidecars["source-hashes.json"], "source-hashes.json"),
                    coordinate, "run/source manifest mismatch")
            require(entry["sources"]["scripts/gpu_qp_campaign.py"] == report["source_hash"],
                    coordinate, "GPU source_hash mismatch")
            require(record["stage"] == "campaign" and record["state"] == "SUBMITTED"
                    and any(Path(j["output"]).resolve() == Path(path).parent.resolve()
                            for j in record["jobs"]), coordinate, "run stage/state/result mismatch")
            result = validate_gpu(report, sidecars)
        else:
            claimed = report["source_hashes"]
            require(set(claimed) >= {"scripts/benchmark_solver_comparison.py",
                                    "scripts/benchmark_general_qp.py", "src/openopt/qp.py"}
                    and all(entry["sources"].get(k) == v for k, v in claimed.items()),
                    coordinate, "comparison source hashes mismatch")
            result = validate_comparison(report, sidecars)
        result["result_path"] = path
        result["result_sha256"] = sha(data)
        checked.append(result)
    return {"verdict": "PASS", "read_at": now(), "bundle_sha256": sha(raw), "entries": checked,
            "checked_samples": sum(r["checked_samples"] for r in checked), "ranking": rank_pairs(checked),
            "qualification": "evidence acceptance only; not exact certification or commercial parity"}


def output_dir(root: Path, prefix: str) -> Path:
    root = root.expanduser().resolve()
    require(root.is_relative_to(Path("/3fs/research3fs/data/tmp")), str(root),
            "artifacts must be external shared 3FS tmp")
    require(not root.is_relative_to(ROOT), str(root), "artifacts inside worktree")
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix + "-", dir=root))


def write(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def mutants(bundle_path: Path, expected_sha256: str, output: Path) -> dict:
    """Run semantic negative controls on caller-pinned, real comparison evidence."""
    baseline = accept(bundle_path, expected_sha256)
    bundle = parse(read(bundle_path), str(bundle_path))
    selected = [e for e in bundle["entries"] if e["kind"] == "comparison"]
    require(bool(selected), "mutants", "real accepted comparison entry required")
    entry = selected[0]
    original = parse(bound(entry["result"]), entry["result"]["path"])
    outcomes = []
    for name in ("solver-status", "primal-corruption", "dual-corruption", "zero-timing",
                 "sourcehash-forgery"):
        report = copy.deepcopy(original)
        row = next(r for r in report["arms"] if r["status"] == "PASS" and r["solver"] == "openopt")
        sample = row["modes"]["cold"]["samples"][0]
        if name == "solver-status":
            sample["solver_status"] = "max_iterations"
        elif name == "primal-corruption":
            sample["host_solution"]["x"][0] += 100.0
        elif name == "dual-corruption":
            sample["host_solution"]["y"][0] += 100.0
        elif name == "zero-timing":
            row["modes"]["cold"]["latency"]["samples_s"][0] = 0.0
        else:
            report["source_hashes"]["src/openopt/qp.py"] = "0" * 64
        folder = output / name
        folder.mkdir()
        write(folder / "result.json", report)
        changed = copy.deepcopy(entry)
        changed["result"] = descriptor(folder / "result.json")
        write(folder / "bundle.json", {"schema_version": 1, "entries": [changed]})
        try:
            accept(folder / "bundle.json", sha(read(folder / "bundle.json")))
        except Rejected as error:
            outcome = {"mutant": name, "verdict": "REJECTED", "first_cause": error.cause,
                       "coordinate": error.coordinate, "bundle": str(folder / "bundle.json")}
            write(folder / "rejection.json", outcome)
            outcomes.append(outcome)
        else:
            raise Rejected(name, "negative control escaped acceptance")
    return {"verdict": "PASS", "baseline_checked_samples": baseline["checked_samples"],
            "mutants_rejected": len(outcomes), "outcomes": outcomes,
            "qualification": "negative controls, not new performance measurements"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("seal", "check", "mutants"))
    parser.add_argument("--check", action="store_true", help="sealed bundle admission alias for action check")
    parser.add_argument("--result", type=Path, action="append")
    parser.add_argument("--source-root", type=Path, action="append")
    parser.add_argument("--run", type=Path, action="append")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.check:
        if args.action not in (None, "check"):
            parser.error("--check conflicts with action")
        args.action = "check"
    if args.action is None:
        parser.error("action or --check required")
    if args.action == "seal" and not args.result:
        parser.error("seal requires explicit --result")
    if args.action != "seal" and (args.bundle is None or not args.bundle_sha256):
        parser.error("check/mutants requires --bundle and caller-pinned --bundle-sha256")
    output = output_dir(args.artifact_dir, args.action)
    report = {"verdict": "RUNNING", "action": args.action, "started_at": now()}
    try:
        if args.action == "seal":
            bundle = seal(args.result, args.source_root, args.run)
            write(output / "bundle.json", bundle)
            report.update(verdict="SEALED", bundle=str(output / "bundle.json"),
                          bundle_sha256=sha(read(output / "bundle.json")), entries=len(bundle["entries"]))
        elif args.action == "check":
            report.update(accept(args.bundle, args.bundle_sha256))
        else:
            report.update(mutants(args.bundle, args.bundle_sha256, output))
    except Exception as error:
        report.update(verdict="FAIL", first_cause=str(error),
                      coordinate=getattr(error, "coordinate", "acceptance"),
                      exception_type=type(error).__name__, traceback=traceback.format_exc())
    report.update(finished_at=now(), exit_code=1 if report["verdict"] == "FAIL" else 0)
    write(output / "acceptance.json", report)
    line = (f"QP_CAMPAIGN_ACCEPT verdict={report['verdict']} action={args.action} "
            f"at={report['finished_at']} evidence={output / 'acceptance.json'}")
    if report["verdict"] == "SEALED":
        line += f" bundle={report['bundle']} sha256={report['bundle_sha256']}"
    if report["verdict"] == "FAIL":
        line += f" first_cause={report['first_cause']}"
    with (output / "execution.log").open("x") as log:
        log.write(line + "\n")
    print(line, flush=True)
    return report["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
