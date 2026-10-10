"""CPU-only, caller-pinned acceptance of frozen CUDA correctness execution.

Seal binds result, JUnit, logs, telemetry, run and every snapshot/source byte.
Accept never runs producer code or reconstructs missing problems from tests.
This is not an exact certificate, performance or commercial release gate.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import re
import tempfile
import traceback
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/hf3fs-jd/data/tmp").resolve()
PREFIXES = ("src/", "scripts/", "tests/", "examples/")
COUNT_KEYS = ("tests", "failures", "errors", "skipped")
RUNNER = "scripts/gpu_release_check.py"
SCOPE = "correctness-suite execution"


class AuditError(ValueError):
    """Failure with its first evidence coordinate preserved."""


def require(condition, coordinate, cause):
    if not condition:
        raise AuditError(f"{coordinate}: {cause}")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def digest(value):
    require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value),
            "sha256", "caller-pinned SHA256 required")
    return value


def parse(data, coordinate):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, coordinate, f"duplicate JSON key {key}")
            result[key] = value
        return result

    def constant(value):
        raise AuditError(f"{coordinate}: nonfinite JSON {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), path, "regular file required")
    data = path.read_bytes()
    require(data, path, "empty evidence")
    return data


def external(path, tree=None):
    path = Path(path)
    require(path.is_absolute(), path, "absolute artifact path required")
    resolved = path.resolve()
    require(resolved.is_relative_to(ARTIFACT_ROOT), path, "outside shared 3FS artifact tree")
    if tree is not None:
        require(resolved.is_relative_to(Path(tree).resolve()), path, "artifact escapes run tree")
    require(not path.is_symlink(), path, "symlink artifact rejected")
    return resolved


def descriptor(path):
    path = external(path)
    data = read(path)
    return {"path": str(path), "sha256": sha(data), "bytes": len(data)}


def bound(item, tree=None):
    require(isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"},
            "descriptor", "exact path/hash/bytes descriptor required")
    path = external(item["path"], tree)
    data = read(path)
    require(type(item["bytes"]) is int and len(data) == item["bytes"],
            path, "artifact byte count mutation")
    require(sha(data) == digest(item["sha256"]), path, "artifact hash mutation")
    return data


def relative(value):
    require(isinstance(value, str) and value, "source", "nonempty relative path required")
    path = PurePosixPath(value)
    require(not path.is_absolute() and ".." not in path.parts and str(path) == value
            and value.startswith(PREFIXES), value, "unknown or unsafe source")
    return value


def source_files(root):
    root = Path(root).resolve()
    result = {}
    for prefix in PREFIXES:
        directory = root / prefix
        if not directory.exists():
            continue
        for path in sorted(directory.rglob("*")):
            require(not path.is_symlink(), path, "source symlink rejected")
            if path.is_file():
                name = relative(path.relative_to(root).as_posix())
                result[name] = sha(path.read_bytes())
    require(result, root, "empty source set")
    return result


def core_files(root):
    directory = Path(root).resolve() / "src/openopt"
    result = {}
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), path, "core symlink rejected")
        if path.is_file():
            result[path.relative_to(Path(root).resolve()).as_posix()] = sha(path.read_bytes())
    require(result, directory, "empty current core")
    return result


def source_map(value, coordinate):
    require(isinstance(value, dict) and value, coordinate, "empty source set")
    for name, value_hash in value.items():
        relative(name)
        digest(value_hash)
    return value


def difference(expected, actual):
    return {"missing": sorted(set(expected) - set(actual)),
            "extra": sorted(set(actual) - set(expected)),
            "changed": sorted(k for k in set(expected) & set(actual)
                              if expected[k] != actual[k])}


def equal_sources(expected, actual, coordinate):
    delta = difference(expected, actual)
    require(not any(delta.values()), coordinate, f"source bidirectional difference {delta}")


def targets(report):
    values = report.get("targets")
    require(isinstance(values, list) and values and len(set(values)) == len(values),
            "targets", "empty or duplicate targets")
    for value in values:
        name = relative(value.split("::", 1)[0])
        require(name.startswith("tests/test_") and name.endswith(".py"),
                name, "explicit test source required")
    require(isinstance(report.get("selection"), str), "selection", "explicit selection required")
    return values


def junit(data, report, snapshot):
    require(b"<!DOCTYPE" not in data and b"<!ENTITY" not in data,
            "JUnit", "DTD/entity forbidden")
    root = ET.fromstring(data)
    require(root.tag in ("testsuites", "testsuite"), "JUnit", "unknown root")
    suites = [root] if root.tag == "testsuite" else list(root)
    require(suites and all(s.tag == "testsuite" for s in suites),
            "JUnit", "empty or unknown suite")
    total = dict.fromkeys(COUNT_KEYS, 0)
    cases, seen = [], set()
    allowed = targets(report)
    for suite in suites:
        counts = {}
        for key in COUNT_KEYS:
            value = suite.get(key)
            require(value is not None and re.fullmatch(r"\d+", value),
                    "JUnit/" + key, "explicit nonnegative count required")
            counts[key] = int(value)
            total[key] += counts[key]
        children = suite.findall("testcase")
        require(len(children) == counts["tests"] and counts["tests"] > 0,
                "JUnit", "empty/incomplete testcase completion")
        require(not any(counts[k] for k in COUNT_KEYS[1:]),
                "JUnit", "failure/error/skip rejected")
        for case in children:
            require(not any(c.tag in ("failure", "error", "skipped") for c in case),
                    "JUnit", "case failure/error/skip contradicts counts")
            classname, name = case.get("classname", ""), case.get("name", "")
            key = classname + "::" + name
            require(classname and name.startswith("test_") and key not in seen,
                    key, "missing or duplicate testcase identity")
            seen.add(key)
            duration = float(case.get("time", "nan"))
            require(math.isfinite(duration) and duration >= 0, key, "completion time absent")
            matched = []
            for target in allowed:
                file, *selector = target.split("::")
                module = file[:-3].replace("/", ".")
                if classname == module or classname.startswith(module + "."):
                    if not selector or selector[-1] == name or name.startswith(selector[-1] + "["):
                        matched.append(file)
            require(len(matched) == 1 and matched[0] in snapshot,
                    key, "testcase outside explicit frozen targets")
            properties = {}
            for prop in case.findall("./properties/property"):
                prop_name = prop.get("name")
                require(prop_name and prop_name not in properties and prop.get("value") is not None,
                        key, "duplicate/incomplete property")
                properties[prop_name] = prop.get("value")
            cases.append({"id": key, "source": matched[0], "name": name,
                          "properties": properties})
    require(total == report.get("tests"), "JUnit", "result counts mismatch")
    require(total["tests"] == len(cases) > 0, "JUnit", "empty execution")
    return total, cases


def proof_paths(cases, tree):
    paths = {}
    for case in cases:
        for name, value in case["properties"].items():
            if name.endswith("_path"):
                require(value.endswith(".json"), case["id"], "only machine-read JSON proof paths")
                path = external(value, tree)
                paths[str(path)] = path
    return paths


def proof_family(case):
    source, name = case["source"], case["name"]
    if any(term in name for term in ("never_falls_back", "cannot_accept", "reject")):
        return None
    if source.endswith("test_qp_termination.py"):
        return "gap" if "_gap_" in name else "ray" if "_direction_" in name else None
    if source.endswith("test_qp_scaling.py") and "restoration_kkt" in name:
        return "scaling"
    if source.endswith("test_qp_precondition.py"):
        return "pcg"
    return None


def check_test_definitions(cases, snapshot):
    functions = {}
    for case in cases:
        name = case["source"]
        if name not in functions:
            tree = ast.parse(read(Path(snapshot) / name), filename=name)
            functions[name] = {node.name for node in ast.walk(tree)
                               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        require(case["name"].split("[", 1)[0] in functions[name], case["id"],
                "testcase function absent from frozen source")


def check_job(job, run, snapshot, report, case_dir):
    item = run["jobs"][0]
    require(job.get("kind") == "Job" and job.get("metadata", {}).get("name") == item.get("job"),
            "job", "run Job identity mismatch")
    spec = job.get("spec", {})
    pod = spec.get("template", {}).get("spec", {})
    containers = pod.get("containers", [])
    require(spec.get("backoffLimit") == 0 and pod.get("restartPolicy") == "Never"
            and len(containers) == 1, "job", "one-card non-retrying execution required")
    container = containers[0]
    require(container.get("resources", {}).get("limits", {}).get("nvidia.com/gpu") == 1,
            "job", "actual one-card request missing")
    command = ["python3", str(Path(snapshot) / RUNNER), "--artifact-dir", str(case_dir)]
    for target in targets(report):
        command += ["--target", target]
    if report["selection"]:
        command += ["--expression", report["selection"]]
    require(container.get("command") == command, "job", "result target/selection command mismatch")


def recompute(saved, family):
    """Replay only saved original data, using NumPy, never a solver or test factory."""
    import numpy as np

    problem = saved.get("problem")
    if not isinstance(problem, dict) or set(problem) != {"P", "q", "A", "l", "u"}:
        return {"verdict": "UNVERIFIED", "reason": "frozen original problem missing"}
    if not {"atol", "rtol"} <= set(saved):
        return {"verdict": "UNVERIFIED", "reason": "frozen replay tolerances missing"}
    atol, rtol = saved["atol"], saved["rtol"]
    require(type(atol) in (int, float) and type(rtol) in (int, float)
            and 0 < atol <= 1e-3 and 0 <= rtol <= 1e-3,
            "proof", "invalid/overbroad replay tolerances")
    P, q, A = (np.asarray(problem[k], dtype=np.float64) for k in ("P", "q", "A"))
    # JSON null is allowed only for an open bound, never for a vector/matrix.
    lo = np.asarray([-np.inf if v is None else v for v in problem["l"]], dtype=np.float64)
    hi = np.asarray([np.inf if v is None else v for v in problem["u"]], dtype=np.float64)
    require(q.ndim == 1 and q.size > 0 and P.shape == (q.size, q.size)
            and A.ndim == 2 and A.shape[1] == q.size
            and lo.shape == hi.shape == (A.shape[0],), "problem", "dimensions invalid")
    require(all(np.isfinite(v).all() for v in (P, q, A))
            and not np.isnan(lo).any() and not np.isnan(hi).any()
            and not np.isposinf(lo).any() and not np.isneginf(hi).any()
            and np.all(lo <= hi), "problem", "invalid/nonfinite frozen data")

    def norm(v):
        return float(np.max(np.abs(v))) if v.size else 0.0

    with np.errstate(all="raise"):
        require(norm(P - P.T) <= 1e-12 * max(1, norm(P)),
                "problem", "asymmetric curvature")
        eig, V = np.linalg.eigh(P)
        require(eig.min() >= 0, "problem", "negative/unresolved curvature")
        if family == "ray":
            if "direction" not in saved or saved.get("status") not in ("infeasible", "unbounded"):
                return {"verdict": "UNVERIFIED", "reason": "frozen ray direction/status missing"}
            direction = np.asarray(saved["direction"], dtype=np.float64)
            size = lo.size if saved["status"] == "infeasible" else q.size
            require(direction.shape == (size,) and np.isfinite(direction).all()
                    and norm(direction) > 0, "ray", "empty/invalid direction")
            direction = direction / norm(direction)
            if saved["status"] == "infeasible":
                pos, neg = direction > 0, direction < 0
                require(np.isfinite(hi[pos]).all() and np.isfinite(lo[neg]).all(),
                        "ray", "infinite support")
                support = float(hi[pos] @ direction[pos] + lo[neg] @ direction[neg])
                residual = norm(A.T @ direction)
                require(residual <= atol + rtol * norm(A)
                        and support < -atol, "ray", "infeasibility direction rejected")
                return {"verdict": "RECOMPUTED",
                        "metrics": {"transpose_residual": residual, "support": support}}
            if "feasible_base" not in saved:
                return {"verdict": "UNVERIFIED", "reason": "frozen feasible base missing"}
            base = np.asarray(saved["feasible_base"], dtype=np.float64)
            require(base.shape == q.shape and np.isfinite(base).all(),
                    "ray", "invalid feasible base")
            Ax, Ad = A @ base, A @ direction
            bounded_lo, bounded_hi = np.isfinite(lo), np.isfinite(hi)
            feasible = norm(Ax - np.clip(Ax, lo, hi))
            recession = max(norm(np.maximum(-Ad[bounded_lo], 0)),
                            norm(np.maximum(Ad[bounded_hi], 0)))
            curvature = norm(P @ direction)
            descent = float(q @ direction)
            require(feasible <= atol + rtol * norm(Ax) and recession <= atol
                    and curvature <= atol and float(direction @ P @ direction) == 0
                    and descent < -atol, "ray", "unbounded direction/base rejected")
            return {"verdict": "RECOMPUTED", "metrics": {
                "feasibility": feasible, "recession": recession,
                "curvature": curvature, "descent": descent}}
        if family in ("gap", "scaling", "pcg"):
            if not {"x", "y"} <= set(saved):
                return {"verdict": "UNVERIFIED", "reason": "saved x/y vectors missing"}
            x, y = (np.asarray(saved[k], dtype=np.float64) for k in ("x", "y"))
            require(x.shape == q.shape and y.shape == lo.shape
                    and np.isfinite(x).all() and np.isfinite(y).all(),
                    "proof", "invalid host vectors")
            Ax, Px, ATy = A @ x, P @ x, A.T @ y
            primal = norm(Ax - np.clip(Ax, lo, hi))
            stationarity = norm(Px + q + ATy)
            pos, neg = y > 0, y < 0
            require(np.isfinite(hi[pos]).all() and np.isfinite(lo[neg]).all(),
                    "proof", "multiplier has infinite box support")
            complementarity = norm(np.concatenate(
                (y[pos] * (hi[pos] - Ax[pos]), y[neg] * (lo[neg] - Ax[neg]))))
            b = q + ATy
            coordinates = V.T @ b
            positive = eig > 0
            if np.any((eig > 0) & (eig <= 64 * np.finfo(float).eps * q.size * norm(eig))):
                return {"verdict": "UNVERIFIED", "reason": "unresolved PSD spectrum"}
            inverse_coordinates = np.zeros_like(coordinates)
            inverse_coordinates[positive] = coordinates[positive] / eig[positive]
            inverse = V @ inverse_coordinates
            require(norm(P @ inverse - b) <= atol + rtol * norm(b),
                    "proof", "dual objective outside range(P)")
            objective = float(.5 * x @ Px + q @ x)
            dual = float(-.5 * b @ inverse - hi[pos] @ y[pos] - lo[neg] @ y[neg])
            gap = abs(objective - dual)
            metrics = {"primal": primal, "stationarity": stationarity,
                       "complementarity": complementarity, "objective": objective,
                       "dual_objective": dual, "gap": gap}
            require(primal <= atol + rtol * max(norm(Ax), norm(lo[np.isfinite(lo)]),
                                               norm(hi[np.isfinite(hi)]))
                    and stationarity <= atol + rtol * max(norm(Px), norm(q), norm(ATy))
                    and complementarity <= atol + rtol * max(1, norm(y) * norm(Ax))
                    and gap <= atol + rtol * max(abs(objective), abs(dual)),
                    "proof", f"saved candidate rejected {metrics}")
            if family == "pcg":
                if not {"system", "rhs", "inner_x", "inner_tolerance"} <= set(saved):
                    return {"verdict": "UNVERIFIED", "reason": "frozen PCG system/vectors missing"}
                K, rhs, inner = (np.asarray(saved[k], dtype=np.float64)
                                 for k in ("system", "rhs", "inner_x"))
                require(K.shape == P.shape and rhs.shape == inner.shape == q.shape
                        and all(np.isfinite(v).all() for v in (K, rhs, inner)),
                        "pcg", "invalid frozen system")
                tolerance = saved["inner_tolerance"]
                require(type(tolerance) in (int, float) and 0 < tolerance <= 1e-3,
                        "pcg", "invalid inner tolerance")
                residual = float(np.linalg.norm(rhs - K @ inner))
                require(residual <= tolerance, "pcg", "true residual rejected")
                metrics["inner_residual"] = residual
            return {"verdict": "RECOMPUTED", "metrics": metrics}
    return {"verdict": "UNVERIFIED", "reason": "unknown replay family; no promotion"}


def proof_audit(cases, proofs, tree, hardware):
    outcomes = []
    for case in cases:
        family = proof_family(case)
        paths = [str(external(v, tree)) for k, v in case["properties"].items()
                 if k.endswith("_path")]
        if family and not paths:
            outcomes.append({"case": case["id"], "family": family, "verdict": "UNVERIFIED",
                             "reason": "JUnit has no frozen machine-read proof sidecar"})
        for path in paths:
            saved = parse(bound(proofs[path], tree), path)
            require(isinstance(saved, dict) and saved.get("verdict") == "PASS",
                    path, "proof sidecar not successful")
            if "device" in saved:
                require(saved["device"] == "cuda:0", path, "actual proof device mismatch")
            if "gpu_name" in saved:
                require(saved["gpu_name"] == hardware["gpu"], path, "proof model mismatch")
            outcome = recompute(saved, family) if family else {
                "verdict": "UNVERIFIED", "reason": "unrecognized proof family"}
            outcomes.append({"case": case["id"], "family": family, "path": path, **outcome})
    return {"verdict": "UNVERIFIED", "qualified": False, "cases": outcomes,
            "reason": "scoped numerical replay is not a comprehensive GPU certificate"}


def seal(result_path, expected_result_sha256, *, source_snapshot, run_path, current_root):
    result_path = external(result_path)
    data = read(result_path)
    require(sha(data) == digest(expected_result_sha256), result_path, "caller result pin mismatch")
    report = parse(data, result_path)
    snapshot = external(source_snapshot)
    run_path = external(run_path)
    require(snapshot == run_path.parent / "source", snapshot, "snapshot must belong to explicit run")
    tree = run_path.parent
    external(result_path, tree)
    require(result_path.parent.parent.name.startswith("case-"), result_path,
            "immutable cuda-check result required, not coordinator mirror")
    files = {name: descriptor(result_path.parent / name)
             for name in ("pytest.xml", "pytest.log", "execution.log", "telemetry.json")}
    sources = source_files(snapshot)
    _, cases = junit(bound(files["pytest.xml"], tree), report, sources)
    proofs = {name: descriptor(path) for name, path in proof_paths(cases, tree).items()}
    bundle = {
        "schema_version": 1, "scope": SCOPE, "result": descriptor(result_path),
        "result_sha256": expected_result_sha256, "run": descriptor(run_path),
        "job": descriptor(tree / "job-0.json"),
        "source_snapshot": str(snapshot),
        "source_manifest": descriptor(snapshot / "source-hashes.json"),
        "sources": sources, "files": files, "proofs": proofs,
    }
    # Validate before publishing a seal; no producer functions or worktree HEAD used.
    validate(bundle, current_root=current_root)
    return bundle


def validate(bundle, *, current_root):
    require(set(bundle) == {"schema_version", "scope", "result", "result_sha256", "run", "job",
                           "source_snapshot", "source_manifest", "sources", "files", "proofs"}
            and bundle["schema_version"] == 1 and bundle["scope"] == SCOPE,
            "bundle", "unknown or incomplete acceptance scope")
    data = bound(bundle["result"])
    require(sha(data) == digest(bundle["result_sha256"]), "result", "caller result pin mismatch")
    report = parse(data, "result")
    require(report.get("verdict") == "PASS" and type(report.get("returncode")) is int
            and report["returncode"] == 0, "result", "execution not successful")
    tree = external(bundle["run"]["path"]).parent
    external(bundle["result"]["path"], tree)
    snapshot = external(bundle["source_snapshot"], tree)
    require(snapshot == tree / "source", snapshot, "snapshot/run mismatch")
    require(external(bundle["source_manifest"]["path"], tree) == snapshot / "source-hashes.json",
            "manifest", "source manifest substitution")
    run = parse(bound(bundle["run"], tree), "run")
    require(run.get("stage") == "release-regression" and run.get("state") == "SUBMITTED",
            "run", "not submitted release regression")
    manifest = source_map(parse(bound(bundle["source_manifest"], tree), "source manifest"), "manifest")
    sources = source_map(bundle["sources"], "bundle sources")
    equal_sources(sources, source_files(snapshot), "frozen snapshot")
    equal_sources(sources, source_map(run.get("source_hashes"), "run sources"), "run sources")
    equal_sources(sources, manifest, "source manifest")
    chosen = targets(report)
    expected = {k: v for k, v in sources.items()
                if k.startswith("src/openopt/") and "/" not in k[len("src/openopt/"):]}
    expected.update({t.split("::", 1)[0]: sources[t.split("::", 1)[0]] for t in chosen})
    require(RUNNER in sources, RUNNER, "runner missing")
    expected[RUNNER] = sources[RUNNER]
    equal_sources(expected, source_map(report.get("source_sha256"), "result sources"), "result sources")
    case_dir = external(bundle["result"]["path"]).parent.parent
    require(case_dir.name.startswith("case-") and len(run.get("jobs", [])) == 1
            and external(run["jobs"][0]["output"], tree) == case_dir,
            "run", "result is not the explicit one-card run output")
    job_path = external(bundle["job"]["path"], tree)
    require(job_path == tree / "job-0.json"
            and external(run["jobs"][0]["manifest"], tree) == job_path,
            "job", "run manifest substitution")
    check_job(parse(bound(bundle["job"], tree), "job"), run, snapshot, report, case_dir)
    require(set(bundle["files"]) == {"pytest.xml", "pytest.log", "execution.log", "telemetry.json"},
            "files", "execution sidecars silently added/removed")
    files = {}
    for name, item in bundle["files"].items():
        require(external(item["path"], tree) == external(bundle["result"]["path"]).parent / name,
                name, "execution sidecar substitution")
        files[name] = bound(item, tree)
    counts, cases = junit(files["pytest.xml"], report, sources)
    check_test_definitions(cases, snapshot)
    log = files["pytest.log"].decode("utf-8")
    require(re.search(rf"(?m)^{counts['tests']} passed(?:,| in )", log)
            and log.rstrip().endswith("returncode=0"), "pytest.log", "completion summary/rc absent")
    require("OPENOPT_RELEASE_CUDA verdict=PASS" in files["execution.log"].decode()
            and f"tests={counts['tests']}" in files["execution.log"].decode(),
            "execution.log", "completion contradicts result")
    hardware = report.get("hardware", {})
    require(all(isinstance(hardware.get(k), str) and hardware[k]
                for k in ("gpu", "node", "torch", "cuda", "cpu_quota"))
            and "RTX PRO 4500" in hardware["gpu"], "hardware", "actual model/runtime evidence missing")
    telemetry = parse(files["telemetry.json"], "telemetry")
    require(isinstance(telemetry, list) and telemetry
            and type(report.get("telemetry_samples")) is int
            and len(telemetry) == report["telemetry_samples"],
            "telemetry", "empty or inconsistent samples")
    for sample in telemetry:
        require(type(sample.get("rc")) is int and sample["rc"] == 0
                and isinstance(sample.get("gpu"), str) and sample["gpu"].startswith("GPU-")
                and type(sample.get("cpu", {}).get("usage_usec")) is int,
                "telemetry", "failed/absent actual GPU/CPU sample")
    paths = proof_paths(cases, tree)
    require(set(paths) == set(bundle["proofs"]), "proofs", "proof sources silently added/removed")
    for name, item in bundle["proofs"].items():
        require(str(external(item["path"], tree)) == name, "proofs", "proof path substitution")
        bound(item, tree)
    core = {k: v for k, v in sources.items() if k.startswith("src/openopt/")}
    require(core, "core", "empty frozen core")
    current = core_files(current_root)
    delta = difference(current, core)
    negative = [c["id"] for c in cases if any(
        term in c["name"] for term in ("reject", "never_falls_back", "cannot_accept", "false_recurrence"))]
    device_cases = [c["id"] for c in cases if (
        "cuda" in c["name"] or "[cuda:0]" in c["name"]) and "[cpu]" not in c["name"]
        and c["id"] not in negative]
    return {
        "verdict": "PASS", "scope": SCOPE, "result_sha256": bundle["result_sha256"],
        "tests": counts, "testcases": [c["id"] for c in cases], "hardware": hardware,
        "device_assertion_cases": device_cases, "negative_execution_cases": negative,
        "placement_scope": "frozen test assertions only; no saved per-vector placement proof",
        "negative_verdict": "PRESENT" if negative else "UNVERIFIED",
        "source_count": len(sources), "source_sha_informational": run.get("source_sha"),
        "junit_sha256": bundle["files"]["pytest.xml"]["sha256"],
        "proof_sidecar_count": len(bundle["proofs"]),
        "current_root": str(Path(current_root).resolve()),
        "current_core": {"verdict": "PASS" if not any(delta.values()) else "UNVERIFIED",
                         **delta, "expected_count": len(current), "frozen_count": len(core)},
        "proof": proof_audit(cases, bundle["proofs"], tree, hardware),
        "qualification": "execution only; not comprehensive GPU proof, performance or commercial parity",
    }


def accept(bundle_path, expected_sha256, *, current_root):
    data = read(external(bundle_path))
    require(sha(data) == digest(expected_sha256), bundle_path, "caller bundle pin mismatch")
    return validate(parse(data, bundle_path), current_root=current_root)


def write(path, value):
    with Path(path).open("x", encoding="utf-8") as output:
        output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "accept"))
    parser.add_argument("--result", type=Path)
    parser.add_argument("--result-sha256")
    parser.add_argument("--source-snapshot", type=Path)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = external(args.artifact_dir)
    artifact.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=artifact, prefix="write-check-") as probe:
        probe.write(b"gpu audit write check")
        probe.flush()
        require(Path(probe.name).read_bytes() == b"gpu audit write check", artifact, "write check failed")
    output = Path(tempfile.mkdtemp(prefix=args.action + "-", dir=artifact))
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "verdict": "RUNNING"}
    rc = 0
    try:
        if args.action == "seal":
            require(all((args.result, args.result_sha256, args.source_snapshot, args.run)),
                    "seal", "explicit result pin, snapshot and run required")
            bundle = seal(args.result, args.result_sha256, source_snapshot=args.source_snapshot,
                          run_path=args.run, current_root=args.current_root)
            write(output / "bundle.json", bundle)
            report.update(verdict="SEALED", bundle=str(output / "bundle.json"),
                          bundle_sha256=sha(read(output / "bundle.json")),
                          result_sha256=args.result_sha256, scope=SCOPE)
        else:
            require(args.bundle and args.sha256, "accept", "caller bundle pin required")
            report.update(accept(args.bundle, args.sha256, current_root=args.current_root))
    except Exception as error:
        rc = 1
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        with (output / "failure.log").open("x") as log:
            log.write(traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    write(output / "acceptance.json", report)
    line = (f"OPENOPT_RELEASE_CUDA_ACCEPT verdict={report['verdict']} "
            f"at={report['finished_at']} evidence={output / 'acceptance.json'}")
    with (output / "execution.log").open("x") as log:
        log.write(line + f"\nreturncode={rc}\n")
    print(line, flush=True)
    if rc:
        print(report["error"], flush=True)
    elif args.action == "seal":
        print(f"bundle_sha256={report['bundle_sha256']}", flush=True)
    else:
        print(f"scope={SCOPE} tests={report['tests']['tests']} "
              f"current_core={report['current_core']['verdict']} proof=UNVERIFIED", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
