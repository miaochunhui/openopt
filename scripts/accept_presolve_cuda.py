"""Caller-pinned CPU replay of eight presolve CUDA cases, without any solve.

Accept original NPZ arrays directly, verify saved reductions/mappings, and bind
JUnit, terminal Job/Pod, producer source and actual execution assertions.
Never execute frozen sources, import tests or reconstruct a missing problem.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
from fractions import Fraction
import io
import itertools
import math
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
from scripts import accept_release_cuda as execution

require = execution.require
AuditError = execution.AuditError
SCOPE = "presolve CUDA execution and original-model numerical replay"
SOURCE = "tests/test_qp_presolve.py"
FUNCTION = "test_cuda_presolve_original_dual_gap_full_model_sidecar"
SELECTION = "cuda_presolve"
SOURCES = ("src/openopt/presolve.py", "src/openopt/qp.py",
           "src/openopt/certificates.py", SOURCE)
AUDITOR = "scripts/accept_presolve_cuda.py"
PRODUCT = "NVIDIA-RTX-PRO-4500-Blackwell"
COUNT = {"tests": 8, "failures": 0, "errors": 0, "skipped": 0}
MODEL_NAMES = ("P", "q", "A", "l", "u")
MAP_NAMES = ("free_variables", "fixed_variables", "fixed_values", "fixed_rows",
             "kept_rows", "removed_rows", "row_shifts")
GATES = {"primal_feasibility", "stationarity", "bound_sign", "complementarity",
         "dual_objective_available", "duality_gap"}


def external(path, tree=None):
    """Allow the shared mount alias, but no symlinks below the artifact root."""
    path = Path(path)
    require(path.is_absolute() and ".." not in path.parts, path, "absolute safe artifact path required")
    roots = (execution.ARTIFACT_ROOT, Path("/hf3fs-jd/data/tmp"))
    root = next((p for p in roots if path.is_relative_to(p)), None)
    require(root is not None, path, "outside shared 3FS tmp")
    current = root
    for part in path.relative_to(root).parts:
        current /= part
        require(not current.is_symlink(), current, "symlink artifact component rejected")
    return execution.external(path, tree)


def descriptor(path):
    return execution.descriptor(external(path))


def bound(item, tree=None):
    external(item["path"], tree)
    return execution.bound(item, tree)


def pinned(path, pin):
    item = descriptor(path)
    require(item["sha256"] == execution.digest(pin), path, "caller artifact pin mismatch")
    return item


def caller_pins(**arguments):
    require(set(arguments) == {name + suffix for name in ("result", "run", "junit", "status")
                              for suffix in ("_path", "_sha256")},
            "caller pins", "explicit result/run/JUnit/status paths and hashes required")
    return {name: pinned(arguments[name + "_path"], arguments[name + "_sha256"])
            for name in ("result", "run", "junit", "status")}


def clean_sources(root, names):
    root = Path(root).resolve()
    status = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "-z",
                             "--untracked-files=all"],
                            capture_output=True, check=False)
    require(status.returncode == 0 and not status.stdout,
            root, "current source must be clean and committed")
    for name in names:
        path = root / name
        require(path.is_file() and not path.is_symlink()
                and path.resolve().is_relative_to(root), name, "current source escape/missing")
        result = subprocess.run(["git", "-C", str(root), "show", "HEAD:" + name],
                                capture_output=True, check=False)
        require(result.returncode == 0 and result.stdout == execution.read(path),
                name, "current source differs from committed HEAD bytes")


def expected_cases(snapshot):
    tree = ast.parse(execution.read(Path(snapshot) / SOURCE), filename=SOURCE)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name.startswith("test_") and SELECTION in node.name]
    require(len(selected) == 1 and selected[0].name == FUNCTION,
            SOURCE, "versioned presolve CUDA function set changed")
    node, groups = selected[0], []
    for decorator in reversed(node.decorator_list):
        require(isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Attribute)
                and isinstance(decorator.func.value.value, ast.Name)
                and decorator.func.value.value.id == "pytest"
                and decorator.func.value.attr == "mark", SOURCE, "dynamic selected decorator rejected")
        if decorator.func.attr == "skipif":
            continue
        require(decorator.func.attr == "parametrize" and len(decorator.args) == 2
                and not decorator.keywords and isinstance(decorator.args[0], ast.Constant),
                SOURCE, "static parametrization required")
        name, values = decorator.args[0].value, decorator.args[1]
        require(name in ("dtype", "kind") and isinstance(values, (ast.List, ast.Tuple)),
                SOURCE, "unknown presolve CUDA parameter")
        parsed = []
        for index, value in enumerate(values.elts):
            if name == "dtype":
                require(isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                        and value.value.id == "torch" and value.attr in ("float32", "float64"),
                        SOURCE, "dynamic dtype rejected")
                parsed.append(("dtype" + str(index), value.attr))
            else:
                require(isinstance(value, ast.Constant) and isinstance(value.value, str),
                        SOURCE, "dynamic case kind rejected")
                parsed.append((value.value, value.value))
        groups.append((name, parsed))
    require([name for name, _ in groups] == ["dtype", "kind"]
            and [value for _, value in groups[0][1]] == ["float32", "float64"]
            and [value for _, value in groups[1][1]] == ["coupled", "active", "constant", "infeasible"],
            SOURCE, "full eight-case domain/order changed")
    # This checks the pinned execution assertions without running producer code.
    body = ast.unparse(node)
    required_assertions = (
        'assert result.x.device == result.y.device == torch.device("cuda:0")',
        "assert result.x.dtype == result.y.dtype == dtype",
        "assert result.infeasibility_direction.device == torch.device",
        "assert getattr(result.presolved, name).device == torch.device",
    )
    normalized = body.replace("'", '"')
    require(all(text in normalized for text in required_assertions),
            SOURCE, "frozen actual vector/mapping placement assertions missing")
    return {
        "tests.test_qp_presolve::" + FUNCTION + "[" + dtype_id + "-" + kind_id + "]":
        {"dtype": dtype, "kind": kind}
        for (dtype_id, dtype), (kind_id, kind) in itertools.product(groups[0][1], groups[1][1])
    }


def timestamp(value, coordinate):
    require(isinstance(value, str), coordinate, "explicit timestamp required")
    try:
        date = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise AuditError(f"{coordinate}: invalid timestamp") from error
    require(date.tzinfo is not None, coordinate, "timezone required")
    return date


def check_terminal(status, submitted, run, report):
    require(isinstance(status, dict) and set(status) == {"job", "pods"},
            "status", "collected Job/Pod record required")
    job = status["job"]
    meta, spec, done = job.get("metadata", {}), job.get("spec", {}), job.get("status", {})
    expected = run["jobs"][0]["job"]
    require(job.get("kind") == "Job" and meta.get("name") == expected
            and meta.get("uid") and meta.get("namespace") == run["namespace"],
            "status Job", "actual Job identity/namespace/UID mismatch")
    require(done.get("succeeded") == 1 and done.get("failed", 0) == 0
            and done.get("active", 0) == 0
            and any(c.get("type") == "Complete" and c.get("status") == "True"
                    for c in done.get("conditions", []))
            and not any(c.get("type") == "Failed" and c.get("status") == "True"
                        for c in done.get("conditions", [])),
            "status Job", "not one successful completed Job")
    require(timestamp(done.get("startTime"), "Job start")
            <= timestamp(done.get("completionTime"), "Job completion"),
            "status Job", "inverted terminal timestamps")
    require(spec.get("backoffLimit") == 0 and spec.get("completions", 1) == 1
            and spec.get("parallelism", 1) == 1, "status Job", "retry/parallel count changed")
    intended = submitted["spec"]["template"]["spec"]
    actual = spec.get("template", {}).get("spec", {})
    for p in (intended, actual):
        require(p.get("runtimeClassName") == "nvidia" and p.get("restartPolicy") == "Never"
                and p.get("nodeSelector", {}).get("nvidia.com/gpu.product") == PRODUCT
                and not p.get("initContainers") and not p.get("ephemeralContainers"),
                "Job runtime", "strict single-card NVIDIA runtime/product required")
    require(actual.get("containers", [])[0]["command"] == intended["containers"][0]["command"],
            "status Job", "actual/submitted command mismatch")
    pods = status["pods"].get("items")
    require(isinstance(pods, list) and len(pods) == 1, "status Pods", "exact one Pod required")
    pod = pods[0]
    pmeta, pspec, pstatus = pod.get("metadata", {}), pod.get("spec", {}), pod.get("status", {})
    require(pod.get("kind") == "Pod" and pmeta.get("namespace") == meta["namespace"]
            and pmeta.get("uid") and pmeta.get("name")
            and any(o.get("kind") == "Job" and o.get("name") == expected
                    and o.get("uid") == meta["uid"] and o.get("controller") is True
                    for o in pmeta.get("ownerReferences", [])),
            "status Pod", "owner/identity mismatch")
    require(pstatus.get("phase") == "Succeeded"
            and pspec.get("nodeName") == report["hardware"]["node"],
            "status Pod", "actual node/phase mismatch")
    containers = pspec.get("containers", [])
    require(len(containers) == 1 and pspec.get("restartPolicy") == "Never"
            and pspec.get("runtimeClassName") == "nvidia"
            and containers[0]["command"] == intended["containers"][0]["command"]
            and containers[0].get("image") == intended["containers"][0].get("image")
            and containers[0].get("resources", {}).get("limits", {}).get("nvidia.com/gpu") == "1"
            and containers[0].get("resources", {}).get("requests", {}).get("nvidia.com/gpu") == "1",
            "status Pod", "actual one-card command/image/resources mismatch")
    states = pstatus.get("containerStatuses")
    require(isinstance(states, list) and len(states) == 1, "status container", "exact termination required")
    state = states[0]
    terminated = state.get("state", {}).get("terminated", {})
    require(state.get("name") == containers[0]["name"] and state.get("restartCount") == 0
            and state.get("containerID") and state.get("imageID")
            and not state.get("lastState")
            and type(terminated.get("exitCode")) is int and terminated["exitCode"] == 0
            and terminated.get("reason") == "Completed"
            and terminated.get("signal", 0) == 0
            and timestamp(terminated.get("startedAt"), "container start")
            <= timestamp(terminated.get("finishedAt"), "container finish"),
            "status container", "failed/retried/missing actual terminal evidence")


def execution_audit(frozen, pins, current_root):
    require(set(frozen) == {"schema_version", "scope", "result", "result_sha256", "run", "job",
                          "source_snapshot", "source_manifest", "sources", "files", "proofs"}
            and frozen["schema_version"] == 1 and frozen["scope"] == execution.SCOPE,
            "execution", "exact release execution schema required")
    require(frozen["result"] == pins["result"] and frozen["run"] == pins["run"]
            and frozen["files"]["pytest.xml"] == pins["junit"]
            and frozen["result_sha256"] == pins["result"]["sha256"],
            "caller pins", "execution/caller result/run/JUnit mismatch")
    tree = external(pins["run"]["path"]).parent
    result = external(pins["result"]["path"], tree)
    require(result.parent.name.startswith("cuda-check-") and result.parent.parent.name == "case-0",
            "result", "immutable cuda-check result required")
    snapshot = external(frozen["source_snapshot"], tree)
    require(snapshot == tree / "source"
            and external(frozen["source_manifest"]["path"], tree) == snapshot / "source-hashes.json",
            "snapshot", "explicit run snapshot/manifest substitution")
    run = execution.parse(bound(pins["run"], tree), "run")
    report = execution.parse(bound(pins["result"], tree), "result")
    require(run.get("stage") == "release-regression" and run.get("state") == "SUBMITTED"
            and len(run.get("jobs", [])) == 1, "run", "one submitted release Job required")
    require(report.get("verdict") == "PASS" and type(report.get("returncode")) is int
            and report["returncode"] == 0 and report.get("tests") == COUNT
            and report.get("targets") == [SOURCE] and report.get("selection") == SELECTION,
            "result", "exact successful full presolve selection required")
    sources = execution.source_map(frozen["sources"], "sources")
    identity = run.get("source_identity")
    require(isinstance(identity, dict) and set(identity) == {"commit", "tree", "clean", "sha256"}
            and identity["clean"] is True and identity["commit"] == run.get("source_sha")
            and isinstance(identity["commit"], str) and re.fullmatch(r"[0-9a-f]{40}", identity["commit"])
            and isinstance(identity["tree"], str) and re.fullmatch(r"[0-9a-f]{40}", identity["tree"])
            and isinstance(identity["sha256"], dict) and identity["sha256"],
            "run identity", "clean committed producer identity required")
    execution.equal_sources(
        sources, {k: v for k, v in identity["sha256"].items()
                  if k.startswith(execution.PREFIXES)
                  or k in getattr(execution, "EXPLICIT_SOURCE_FILES", ())},
        "committed run snapshot selection",
    )
    execution.equal_sources(sources, execution.source_files(snapshot), "snapshot bytes")
    execution.equal_sources(sources, execution.source_map(run.get("source_hashes"), "run sources"),
                            "run sources")
    execution.equal_sources(sources, execution.source_map(
        execution.parse(bound(frozen["source_manifest"], tree), "manifest"), "manifest"), "manifest")
    expected = {k: v for k, v in sources.items()
                if k.startswith("src/openopt/") and "/" not in k[len("src/openopt/"):]}
    expected.update({name: sources[name] for name in (SOURCE, execution.RUNNER)})
    execution.equal_sources(expected, execution.source_map(report.get("source_sha256"), "result sources"),
                            "result sources")
    execution.equal_sources(execution.core_files(current_root),
                            {k: v for k, v in sources.items() if k.startswith("src/openopt/")},
                            "current core")
    for name in SOURCES + (execution.RUNNER, "scripts/accept_release_cuda.py"):
        require(sources.get(name) == execution.sha(execution.read(Path(current_root) / name)),
                name, "producer/current/run source drift")
    require(execution.sha(execution.read(Path(execution.__file__)))
            == sources["scripts/accept_release_cuda.py"],
            "loaded execution auditor", "loaded/current/frozen source mismatch")
    require(external(run["jobs"][0]["output"], tree) == result.parent.parent
            and external(frozen["job"]["path"], tree) == tree / "job-0.json"
            and external(run["jobs"][0]["manifest"], tree) == tree / "job-0.json",
            "run", "output/Job manifest substitution")
    job = execution.parse(bound(frozen["job"], tree), "job")
    execution.check_job(job, run, snapshot, report, result.parent.parent)
    require(external(pins["status"]["path"], tree)
            == tree / (run["jobs"][0]["job"] + "-status.json"),
            "status", "collected status substitution")
    check_terminal(execution.parse(bound(pins["status"], tree), "status"), job, run, report)
    require(set(frozen["files"]) == {"pytest.xml", "pytest.log", "execution.log", "telemetry.json"},
            "execution files", "missing/extra completion artifact")
    files = {}
    for name, item in frozen["files"].items():
        require(external(item["path"], tree) == result.parent / name,
                name, "completion artifact substitution")
        files[name] = bound(item, tree)
    counts, cases = execution.junit(files["pytest.xml"], report, sources)
    require(counts == COUNT, "JUnit", "full eight-case completion required")
    execution.check_test_definitions(cases, snapshot)
    require(re.search(r"(?m)^8 passed(?:, \d+ deselected)? in \d", files["pytest.log"].decode())
            and files["pytest.log"].decode().rstrip().endswith("returncode=0")
            and "OPENOPT_RELEASE_CUDA verdict=PASS tests=8" in files["execution.log"].decode(),
            "completion", "successful counts/returncode marker missing")
    hardware = report.get("hardware", {})
    require(all(isinstance(hardware.get(k), str) and hardware[k]
                for k in ("gpu", "node", "torch", "cuda", "cpu_quota"))
            and "RTX PRO 4500" in hardware["gpu"], "hardware", "actual GPU/runtime missing")
    samples = execution.parse(files["telemetry.json"], "telemetry")
    require(isinstance(samples, list) and samples and type(report.get("telemetry_samples")) is int
            and len(samples) == report["telemetry_samples"], "telemetry", "sample count mismatch/empty")
    for sample in samples:
        require(type(sample.get("rc")) is int and sample["rc"] == 0
                and sample.get("gpu", "").startswith("GPU-")
                and type(sample.get("cpu", {}).get("usage_usec")) is int,
                "telemetry", "failed/missing GPU/cgroup sample")
    require(timestamp(report.get("started_at"), "result start")
            <= timestamp(report.get("finished_at"), "result finish"),
            "result", "completed timestamp ordering required")
    return report, cases, tree, snapshot


def npz_arrays(data):
    import numpy as np

    require(len(data) <= 16 * 1024 * 1024, "NPZ", "archive exceeds bounded replay scope")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        records = archive.infolist()
        names = [item.filename for item in records]
        require(names and len(set(names)) == len(names) and all(
            name.endswith(".npy") and "/" not in name and "\\" not in name for name in names),
            "NPZ", "duplicate/unsafe/empty members")
        require(sum(item.file_size for item in records) <= 64 * 1024 * 1024,
                "NPZ", "expanded archive exceeds bounded replay scope")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    require(all(a.dtype in (np.dtype("float32"), np.dtype("float64"), np.dtype("int64"))
                for a in arrays.values()), "NPZ", "object/complex/unknown dtype")
    return arrays


def fractions(values):
    return [Fraction.from_float(float(value)) for value in values]


def check_maps(arrays, dtype):
    """Verify saved map equations; do not rerun reduction or build a replacement model."""
    import numpy as np

    original = {name: arrays["original_" + name] for name in MODEL_NAMES}
    working = {name: arrays["working_" + name] for name in MODEL_NAMES}
    reduced = {name: arrays["reduced_" + name] for name in MODEL_NAMES}
    n, m = original["q"].size, original["l"].size
    require(original["q"].ndim == 1 and n > 0 and original["P"].shape == (n, n)
            and original["A"].shape == (m, n) and original["l"].shape == original["u"].shape == (m,),
            "original model", "complete nonempty original dimensions required")
    for name in MODEL_NAMES:
        require(original[name].dtype == np.dtype("float64")
                and working[name].dtype == dtype and working[name].shape == original[name].shape
                and np.array_equal(working[name], original[name].astype(dtype)),
                name, "precast original/working cast mismatch")
        for value in (original[name], working[name]):
            require(not np.isnan(value).any() and (
                name in ("l", "u") or np.isfinite(value).all()), name, "nonfinite model")
    require(np.array_equal(original["P"], original["P"].T)
            and not np.isposinf(original["l"]).any() and not np.isneginf(original["u"]).any()
            and (original["l"] <= original["u"]).all(), "original model", "invalid P/bounds")
    for name, size in (("free_variables", n), ("fixed_variables", n), ("fixed_rows", m),
                       ("kept_rows", m), ("removed_rows", m)):
        a = arrays[name]
        require(a.dtype == np.dtype("int64") and a.ndim == 1
                and len(set(a.tolist())) == a.size and ((0 <= a) & (a < size)).all(),
                name, "invalid/duplicate mapping index")
    F, K, fixed_rows, kept, removed = [
        arrays[name] for name in ("free_variables", "fixed_variables", "fixed_rows", "kept_rows", "removed_rows")
    ]
    require(sorted(np.concatenate((F, K)).tolist()) == list(range(n))
            and sorted(np.concatenate((kept, removed)).tolist()) == list(range(m))
            and all(a.tolist() == sorted(a.tolist()) for a in (F, K, kept, removed))
            and fixed_rows.size == K.size and set(fixed_rows) <= set(removed),
            "mapping", "full disjoint variable/row partitions required")
    c, shifts = arrays["fixed_values"], arrays["row_shifts"]
    require(c.dtype == shifts.dtype == dtype and c.shape == K.shape and shifts.shape == (m,)
            and np.isfinite(c).all() and np.isfinite(shifts).all(),
            "mapping", "finite fixed values/shifts required")
    require(all(reduced[name].dtype == dtype for name in MODEL_NAMES)
            and reduced["q"].shape == (F.size,) and reduced["P"].shape == (F.size, F.size)
            and reduced["A"].shape == (kept.size, F.size)
            and reduced["l"].shape == reduced["u"].shape == (kept.size,),
            "reduced model", "complete reduced dimensions/dtype required")
    require(arrays["objective_offset"].shape == ()
            and arrays["objective_offset"].dtype == np.dtype("float64")
            and np.isfinite(arrays["objective_offset"]), "offset", "finite scalar objective offset required")
    if K.size:
        require(all(np.array_equal(original[name], working[name].astype(np.float64))
                    for name in MODEL_NAMES), "fixed substitution", "lossy input cast cannot be exact")
    for j, row, value in zip(K, fixed_rows, c):
        identity = np.zeros(n)
        identity[j] = 1
        require(np.array_equal(original["A"][row], identity)
                and math.isfinite(original["l"][row])
                and original["l"][row] == original["u"][row] == value,
                "fixed identity", "not an original exact +1 finite equality")
    require(np.array_equal(reduced["P"], working["P"][np.ix_(F, F)])
            and np.array_equal(reduced["A"], working["A"][np.ix_(kept, F)]),
            "reduced model", "P/A saved submatrix mismatch")
    c_exact = fractions(c)
    products = [
        sum((a * b for a, b in zip(fractions(working["P"][j, K]), c_exact)), Fraction())
        for j in range(n)
    ]
    for j, qhat in zip(F, reduced["q"]):
        require(Fraction.from_float(float(qhat)) == Fraction.from_float(float(working["q"][j])) + products[j],
                "reduced q", "saved exact linear-term substitution mismatch")
    offset = sum(
        (v * (products[j] / 2 + Fraction.from_float(float(working["q"][j])))
         for j, v in zip(K, c_exact)), Fraction(),
    )
    require(Fraction.from_float(float(arrays["objective_offset"])) == offset,
            "offset", "saved exact objective constant mismatch")
    for i in range(m):
        shift = sum((a * b for a, b in zip(fractions(working["A"][i, K]), c_exact)), Fraction())
        require(Fraction.from_float(float(shifts[i])) == shift, "row shift", "exact shift mismatch")
        if i in removed:
            require((working["A"][i, F] == 0).all()
                    and (not math.isfinite(working["l"][i]) or Fraction.from_float(float(working["l"][i])) <= shift)
                    and (not math.isfinite(working["u"][i]) or Fraction.from_float(float(working["u"][i])) >= shift),
                    "removed row", "not an exactly zero feasible row")
        else:
            reduced_index = int(np.flatnonzero(kept == i)[0])
            for name in ("l", "u"):
                before, after = float(working[name][i]), float(reduced[name][reduced_index])
                require((math.isinf(before) and after == before)
                        or (math.isfinite(before) and math.isfinite(after)
                            and Fraction.from_float(after) == Fraction.from_float(before) - shift),
                        "reduced bounds", "saved exact bound substitution mismatch")
    x, y = arrays["x"], arrays["y"]
    require(x.dtype == y.dtype == dtype and x.shape == (n,) and y.shape == (m,)
            and np.isfinite(x).all() and np.isfinite(y).all()
            and np.array_equal(x[K], c), "restored vectors", "original x/y dtype/dimension/fixed value mismatch")
    zero_rows = sorted(set(removed) - set(fixed_rows))
    require((y[zero_rows] == 0).all(), "restored y", "removed non-selected rows must have zero multiplier")
    return SimpleNamespace(**original)


def replay(saved, arrays, parameters):
    import numpy as np
    from openopt.certificates import check_primal_infeasibility, check_qp_candidate

    require(saved.get("schema_version") == 1
            and saved.get("kind") == "presolve_original_coordinate_replay"
            and saved.get("device") == "cuda:0",
            "sidecar", "presolve CUDA original replay schema required")
    dtype = np.dtype(parameters["dtype"])
    cfg = saved.get("presolve_config")
    tolerance = 2e-4 if parameters["dtype"] == "float32" else 1e-7
    require(isinstance(cfg, dict) and set(cfg) == {
        "enabled", "remove_zero_rows", "fix_identity_equalities", "check_atol", "check_rtol"}
        and all(cfg[name] is True for name in ("enabled", "remove_zero_rows", "fix_identity_equalities"))
        and type(cfg["check_atol"]) in (int, float) and type(cfg["check_rtol"]) in (int, float)
        and cfg["check_atol"] == cfg["check_rtol"] == tolerance,
        "presolve config", "full opt-in/versioned original gap thresholds required")
    loaded = Path(sys.modules[check_qp_candidate.__module__].__file__)
    require(execution.sha(execution.read(loaded)) == saved["source_sha256"]["src/openopt/certificates.py"],
            "checker", "loaded independent checker source mismatch")
    fields = {prefix + "_" + name for prefix in ("original", "working", "reduced") for name in MODEL_NAMES}
    fields.update(MAP_NAMES + ("x", "y", "objective_offset"))
    if parameters["kind"] == "infeasible":
        fields.add("infeasibility_direction")
    require(set(arrays) == fields, "NPZ", "missing/extra frozen arrays")
    with np.errstate(all="raise"):
        problem = check_maps(arrays, dtype)
        evidence = check_qp_candidate(problem, arrays["x"], arrays["y"], atol=tolerance, rtol=tolerance)
    require(evidence.to_dict() == saved.get("evidence")
            and type(saved.get("objective")) in (int, float)
            and evidence.metrics["primal_objective"] == saved["objective"],
            "original evidence", "independent original six-gate replay mismatch")
    # Verify the postsolve stationarity equation, without repairing saved y.
    F, K, rows, kept = [arrays[n] for n in ("free_variables", "fixed_variables", "fixed_rows", "kept_rows")]
    work = {n: arrays["working_" + n] for n in MODEL_NAMES}
    other_y = np.zeros_like(arrays["y"])
    other_y[kept] = arrays["y"][kept]
    with np.errstate(all="raise"):
        gradient = work["P"] @ arrays["x"] + work["q"] + work["A"].T @ other_y
    require(np.all(np.abs(arrays["y"][rows].astype(float) + gradient[K].astype(float))
                   <= tolerance + tolerance * np.maximum(np.abs(arrays["y"][rows]), np.abs(gradient[K]))),
            "restored fixed dual", "original signed stationarity restoration mismatch")
    status = saved.get("status")
    if parameters["kind"] == "infeasible":
        require(status == "infeasible_numerical", "status", "infeasible case substituted")
        ray = arrays["infeasibility_direction"]
        require(ray.dtype == dtype and ray.shape == problem.l.shape and np.isfinite(ray).all(),
                "ray", "saved original direction invalid")
        ray_evidence = check_primal_infeasibility(problem, ray)
        require(ray_evidence.passed and ray_evidence.to_dict() == saved.get("infeasibility_evidence"),
                "ray", "independent original Farkas replay rejected/disagrees")
    else:
        require(status == "solved" and evidence.passed and set(evidence.checks) == GATES
                and all(evidence.checks.values()) and saved.get("infeasibility_evidence") is None,
                "original gap", "all original six gates required, not reduced success")
        ray_evidence = None
    return {"verdict": "PASS", "status": status, "evidence": evidence.to_dict(),
            "infeasibility_evidence": ray_evidence.to_dict() if ray_evidence is not None else None,
            "mapping": "PASS", "free_variables": int(F.size), "fixed_variables": int(K.size),
            "objective_offset": float(arrays["objective_offset"])}


def validate(bundle, *, pins, current_root):
    require(isinstance(bundle, dict) and set(bundle) == {
        "schema_version", "scope", "suite", "execution", "pins", "npz"}
        and bundle["schema_version"] == 1 and bundle["scope"] == SCOPE and bundle["suite"] == "presolve",
        "bundle", "exact versioned presolve bundle required")
    require(bundle["pins"] == pins, "caller pins", "exact independent caller pins required")
    for item in pins.values():
        bound(item)
    clean_sources(current_root, SOURCES + (execution.RUNNER, "scripts/accept_release_cuda.py", AUDITOR))
    frozen = bundle["execution"]
    report, cases, tree, snapshot = execution_audit(frozen, pins, current_root)
    require(execution.sha(execution.read(Path(__file__))) == frozen["sources"].get(AUDITOR)
            == execution.sha(execution.read(Path(current_root) / AUDITOR)),
            AUDITOR, "loaded/current/frozen presolve auditor source mismatch")
    expected = expected_cases(snapshot)
    actual = {case["id"]: case for case in cases}
    require(set(actual) == set(expected), "JUnit denominator", "missing/extra full eight-case identity")
    outcomes, json_paths, npz_paths = [], set(), set()
    for identity, parameters in expected.items():
        props = actual[identity]["properties"]
        require(set(props) == {"presolve_acceptance_path", "presolve_acceptance_sha256", "status"},
                identity, "exact JSON path/hash/status properties required")
        path = external(props["presolve_acceptance_path"], tree)
        require(str(path) == props["presolve_acceptance_path"]
                and path.is_relative_to(Path(pins["result"]["path"]).parent / "pytest-tmp")
                and str(path) not in json_paths, identity, "sidecar alias/escape/shared path")
        json_paths.add(str(path))
        item = frozen["proofs"].get(str(path))
        require(item is not None and item["path"] == str(path)
                and item["sha256"] == execution.digest(props["presolve_acceptance_sha256"]),
                identity, "JUnit JSON property/hash descriptor mismatch")
        saved = execution.parse(bound(item, tree), str(path))
        require(saved.get("status") == props["status"], identity, "JUnit/sidecar status mismatch")
        hashes = saved.get("source_sha256")
        require(isinstance(hashes, dict) and set(hashes) == set(SOURCES),
                identity, "exact producer/checker/test source membership required")
        for name, pin in hashes.items():
            require(execution.digest(pin) == frozen["sources"].get(name)
                    == execution.sha(execution.read(Path(current_root) / name)),
                    name, "sidecar/current/run source mismatch")
        data_info = saved.get("data")
        require(isinstance(data_info, dict) and set(data_info) == {"path", "sha256"}
                and data_info["path"] == "full-model.npz", identity, "exact relative full NPZ required")
        npz = external(path.parent / data_info["path"], tree)
        require(str(npz) not in npz_paths and identity in bundle["npz"],
                identity, "shared/missing NPZ descriptor")
        npz_paths.add(str(npz))
        npz_item = bundle["npz"][identity]
        require(npz_item["path"] == str(npz)
                and npz_item["sha256"] == execution.digest(data_info["sha256"]),
                identity, "saved NPZ pin/path mismatch")
        replayed = replay(saved, npz_arrays(bound(npz_item, tree)), parameters)
        outcomes.append({"case": identity, "json_sha256": item["sha256"],
                         "npz_sha256": npz_item["sha256"], **replayed})
    require(json_paths == set(frozen["proofs"]) and set(bundle["npz"]) == set(expected),
            "sidecar denominator", "unlinked/missing/extra JSON or NPZ")
    # Prevent binding a PASS to evidence changed during replay.
    for item in list(pins.values()) + list(frozen["files"].values()) + list(frozen["proofs"].values()) \
            + list(bundle["npz"].values()) + [frozen["job"], frozen["source_manifest"]]:
        bound(item, tree)
    execution.equal_sources(execution.core_files(current_root),
                            {k: v for k, v in frozen["sources"].items() if k.startswith("src/openopt/")},
                            "final current core")
    clean_sources(current_root, SOURCES + (execution.RUNNER, "scripts/accept_release_cuda.py", AUDITOR))
    execution.equal_sources(frozen["sources"], execution.source_files(snapshot), "final snapshot bytes")
    for name in SOURCES + (execution.RUNNER, "scripts/accept_release_cuda.py", AUDITOR):
        require(frozen["sources"].get(name) == execution.sha(execution.read(Path(current_root) / name)),
                name, "final current source drift")
    return {"verdict": "PASS", "scope": SCOPE, "suite": "presolve", "tests": COUNT,
            "expected_testcases": sorted(expected), "cuda_testcases": 8, "cases": outcomes,
            "pins": pins, "hardware": report["hardware"], "current_core": "PASS",
            "placement_scope": "pinned successful test assertions; NPZ stores explicit host copies",
            "qualified": False, "certified": False, "commercial": "UNVERIFIED",
            "performance": "UNVERIFIED",
            "qualification": "eight synthetic CUDA cases and scoped numerical replay; not hardware attestation"}


def seal(*, current_root, **pin_arguments):
    pins = caller_pins(**pin_arguments)
    tree = external(pins["run"]["path"]).parent
    result = external(pins["result"]["path"], tree)
    snapshot = tree / "source"
    report = execution.parse(bound(pins["result"], tree), "result")
    sources = execution.source_files(snapshot)
    files = {name: descriptor(result.parent / name)
             for name in ("pytest.xml", "pytest.log", "execution.log", "telemetry.json")}
    _, cases = execution.junit(bound(files["pytest.xml"], tree), report, sources)
    proofs, npz = {}, {}
    for case in cases:
        props = case["properties"]
        require("presolve_acceptance_path" in props, case["id"], "sidecar missing")
        path = external(props["presolve_acceptance_path"], tree)
        proofs[str(path)] = descriptor(path)
        npz[case["id"]] = descriptor(path.parent / "full-model.npz")
    frozen = {
        "schema_version": 1, "scope": execution.SCOPE,
        "result": pins["result"], "result_sha256": pins["result"]["sha256"], "run": pins["run"],
        "job": descriptor(tree / "job-0.json"), "source_snapshot": str(snapshot),
        "source_manifest": descriptor(snapshot / "source-hashes.json"),
        "sources": sources, "files": files, "proofs": proofs,
    }
    bundle = {"schema_version": 1, "scope": SCOPE, "suite": "presolve",
              "execution": frozen, "pins": pins, "npz": npz}
    validate(bundle, pins=pins, current_root=current_root)
    return bundle


def accept(bundle_path, bundle_sha256, *, current_root, **pin_arguments):
    data = execution.read(external(bundle_path))
    require(execution.sha(data) == execution.digest(bundle_sha256),
            bundle_path, "caller bundle pin mismatch")
    return validate(execution.parse(data, bundle_path),
                    pins=caller_pins(**pin_arguments), current_root=current_root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "accept"))
    parser.add_argument("--suite", choices=("presolve",), default="presolve")
    for name in ("result", "run", "junit", "status"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", "--log-dir", dest="artifact_dir", type=Path, required=True)
    args = parser.parse_args(argv)
    artifact = external(args.artifact_dir)
    require(not artifact.is_relative_to(args.current_root.resolve())
            and not artifact.is_relative_to(external(args.run).parent),
            artifact, "output must be outside current source and immutable run")
    artifact.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.action + "-", dir=artifact))
    report = {"started_at": datetime.now(timezone.utc).isoformat(), "verdict": "RUNNING"}
    rc = 0
    arguments = {name + "_path": getattr(args, name) for name in ("result", "run", "junit", "status")}
    arguments.update({name + "_sha256": getattr(args, name + "_sha256")
                      for name in ("result", "run", "junit", "status")})
    try:
        if args.action == "seal":
            bundle = seal(current_root=args.current_root, **arguments)
            execution.write(output / "bundle.json", bundle)
            report.update(verdict="SEALED", scope=SCOPE, bundle=str(output / "bundle.json"),
                          bundle_sha256=execution.sha(execution.read(output / "bundle.json")))
        else:
            require(args.bundle and args.sha256, "accept", "caller bundle path/hash required")
            report.update(accept(args.bundle, args.sha256, current_root=args.current_root, **arguments))
    except Exception as error:
        rc = 1
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        with (output / "failure.log").open("x", encoding="utf-8") as log:
            log.write(traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    execution.write(output / "acceptance.json", report)
    line = (f"OPENOPT_PRESOLVE_CUDA_ACCEPT verdict={report['verdict']} "
            f"at={report['finished_at']} evidence={output / 'acceptance.json'}")
    with (output / "execution.log").open("x", encoding="utf-8") as log:
        log.write(line + f"\nreturncode={rc}\n")
    print(line, flush=True)
    if rc:
        print(report["error"], flush=True)
    elif args.action == "seal":
        print(f"bundle_sha256={report['bundle_sha256']}", flush=True)
    else:
        print("suite=presolve tests=8 cuda_tests=8 commercial=UNVERIFIED certified=false", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
