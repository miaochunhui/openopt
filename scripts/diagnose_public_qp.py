"""Read-only LP candidate diagnosis against pinned original qualification rules.

No solve, download, dual clipping, tolerance change, qualification upgrade or
model rewrite. Every component is reported with its own threshold, not judged
by unrelated maxima. Runtime evidence and failure logs must be external.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path, PurePosixPath
import sys
import traceback

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import public_qp_corpus as local_importer

VERSION = "public-qp-diagnosis-v1"
SOLVERS = ("openopt", "osqp", "clarabel")
CHECKS = ("primal_feasibility", "stationarity", "bound_sign", "complementarity",
          "dual_objective_available", "duality_gap")
LIMIT = 128 * 1024 * 1024


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_bytes(root: Path, relative: str) -> bytes:
    parts = PurePosixPath(relative).parts
    require(bool(parts) and not relative.startswith("/") and ".." not in parts
            and "\\" not in relative and relative == "/".join(parts), "unsafe evidence path")
    path = root
    for part in parts:
        path /= part
        require(not path.is_symlink(), "symlink evidence refused")
    require(path.resolve(strict=True).is_relative_to(root) and path.is_file(),
            "evidence escaped root")
    require(0 < path.stat().st_size <= LIMIT, "empty/oversized evidence")
    with path.open("rb") as stream:
        raw = stream.read(LIMIT + 1)
    require(0 < len(raw) <= LIMIT, "empty/oversized evidence read")
    return raw


def object_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"nonfinite JSON token: {value}")

    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    require(isinstance(result, dict), "expected JSON object")
    return result


def load_code(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load pinned rules")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def maximum(value) -> float:
    value = np.asarray(value)
    return float(np.max(np.abs(value))) if value.size else 0.0


def finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def independent_lp(model: dict, x: np.ndarray, y: np.ndarray, atol: float, rtol: float) -> dict:
    """Independently reconstruct the unchanged LP checks and component limits."""
    require(np.isfinite(atol) and atol > 0 and np.isfinite(rtol) and 0 <= rtol < 1,
            "invalid original tolerance")
    a = model["arrays"]
    require(np.count_nonzero(a["P"]) == 0, "diagnosis is limited to original LPs, not general QPs")
    sign = 1 if model["objective_sense"] == "minimize" else -1
    q = sign * a["q"]
    A = np.concatenate((a["A"], np.eye(len(q))))
    l = np.concatenate((a["l"], a["variable_lower"]))
    u = np.concatenate((a["u"], a["variable_upper"]))
    require(x.shape == q.shape and y.shape == l.shape and np.isfinite(x).all()
            and np.isfinite(y).all(), "invalid saved candidate dimensions/values")
    with np.errstate(over="raise", under="raise", invalid="raise", divide="raise"):
        Ax, ATy = A @ x, A.T @ y
        b = q + ATy
        stationarity_limits = atol + rtol * np.maximum.reduce(
            [np.zeros_like(q), np.abs(q), np.abs(ATy)])
        lower, upper = np.isfinite(l), np.isfinite(u)
        violations = np.concatenate((np.maximum(l[lower] - Ax[lower], 0),
                                     np.maximum(Ax[upper] - u[upper], 0)))
        feasibility_limits = atol + rtol * np.concatenate((
            np.maximum(np.abs(l[lower]), np.abs(Ax[lower])),
            np.maximum(np.abs(u[upper]), np.abs(Ax[upper]))))
        positive, negative = y > 0, y < 0
        bad = (positive & ~upper) | (negative & ~lower)
        chosen_upper, chosen_lower = positive & upper, negative & lower
        comp = np.concatenate((y[chosen_upper] * (u[chosen_upper] - Ax[chosen_upper]),
                               y[chosen_lower] * (l[chosen_lower] - Ax[chosen_lower])))
        comp_limits = atol + rtol * np.concatenate((
            np.maximum(np.abs(y[chosen_upper] * u[chosen_upper]),
                       np.abs(y[chosen_upper] * Ax[chosen_upper])),
            np.maximum(np.abs(y[chosen_lower] * l[chosen_lower]),
                       np.abs(y[chosen_lower] * Ax[chosen_lower]))))
        primal = float(q @ x)
        range_limit = float(atol + rtol * maximum(b))
        support = dual = gap = gap_limit = range_residual = None
        unavailable = {}
        if np.any(bad):
            unavailable["dual_objective"] = "multiplier has an infinite box support"
        else:
            support = float(np.sum(np.concatenate((u[positive] * y[positive],
                                                  l[negative] * y[negative]))))
            range_residual = maximum(b)
            if range_residual > range_limit:
                unavailable["dual_objective"] = "q + A.T @ y is outside numerical range(P)"
            else:
                dual = -support
                gap = float(primal - dual)
                gap_limit = float(atol + rtol * max(abs(primal), abs(dual)))
        checks = {
            "primal_feasibility": bool(np.all(violations <= feasibility_limits)),
            "stationarity": bool(np.all(np.abs(b) <= stationarity_limits)),
            "bound_sign": not bool(np.any(bad)),
            "complementarity": bool(np.all(np.abs(comp) <= comp_limits)) and not bool(np.any(bad)),
            "dual_objective_available": dual is not None,
            "duality_gap": gap is not None and abs(gap) <= gap_limit,
        }
        thresholds = {"primal_feasibility_max": maximum(feasibility_limits),
                      "stationarity_max": maximum(stationarity_limits),
                      "complementarity_max": maximum(comp_limits), "bound_sign": 0.0,
                      "range": range_limit, "spectral": 0.0}
        if gap_limit is not None:
            thresholds["duality_gap"] = gap_limit
        candidate = {
            "schema_version": 1, "kind": "candidate_optimality", "passed": all(checks.values()),
            "checks": checks,
            "metrics": {"primal_feasibility": maximum(violations), "stationarity": maximum(b),
                        "bound_sign": maximum(y[bad]), "complementarity": maximum(comp),
                        "primal_objective": primal, "dual_objective": dual, "duality_gap": gap,
                        "range_residual": range_residual, "spectral_rank": 0.0},
            "thresholds": thresholds, "unavailable": unavailable,
            "qualification": "approximate numerical evidence; not an exact proof",
        }
        components = []
        for i in range(len(l)):
            sides = {}
            for side, bound in (("lower", l[i]), ("upper", u[i])):
                if np.isfinite(bound):
                    residual = float(max(bound - Ax[i], 0) if side == "lower"
                                     else max(Ax[i] - bound, 0))
                    threshold = float(atol + rtol * max(abs(bound), abs(Ax[i])))
                    sides[side] = {"residual": residual, "threshold": threshold,
                                   "ratio": residual / threshold, "passed": residual <= threshold}
            selected = "upper" if y[i] > 0 else "lower" if y[i] < 0 else None
            complement = None
            if selected is not None and selected in sides:
                bound = u[i] if selected == "upper" else l[i]
                residual = float(y[i] * (bound - Ax[i]))
                threshold = float(atol + rtol * max(abs(y[i] * bound), abs(y[i] * Ax[i])))
                complement = {"signed_residual": residual, "residual": abs(residual),
                              "threshold": threshold, "ratio": abs(residual) / threshold,
                              "passed": abs(residual) <= threshold}
            j = i - len(a["l"])
            components.append({
                "view_row": i, "coordinate_kind": "constraint" if j < 0 else "variable_bound",
                "coordinate_name": model["row_names"][i] if j < 0 else model["variable_names"][j],
                "l": finite(l[i]), "u": finite(u[i]), "lower_infinite": not bool(lower[i]),
                "upper_infinite": not bool(upper[i]), "Ax": float(Ax[i]), "y": float(y[i]),
                "selected_side": selected, "bound_sign_passed": not bool(bad[i]),
                "feasibility": sides, "complementarity": complement,
            })
        variables = [{
            "variable_index": i, "variable_name": model["variable_names"][i],
            "q": float(q[i]), "ATy": float(ATy[i]), "b": float(b[i]),
            "stationarity_residual": abs(float(b[i])),
            "stationarity_threshold": float(stationarity_limits[i]),
            "stationarity_ratio": abs(float(b[i])) / float(stationarity_limits[i]),
            "stationarity_passed": bool(abs(b[i]) <= stationarity_limits[i]),
            "lp_range_residual": abs(float(b[i])), "lp_range_threshold": range_limit,
            "lp_range_ratio": abs(float(b[i])) / range_limit,
            "lp_range_component_passed": bool(abs(b[i]) <= range_limit),
        } for i in range(len(q))]
    failures = {
        "bound_sign": [row["view_row"] for row in components if not row["bound_sign_passed"]],
        "complementarity": [row["view_row"] for row in components
                            if row["complementarity"] and not row["complementarity"]["passed"]],
        "primal_feasibility": [row["view_row"] for row in components
                               if any(not side["passed"] for side in row["feasibility"].values())],
        "stationarity": [row["variable_index"] for row in variables if not row["stationarity_passed"]],
        "lp_range": [row["variable_index"] for row in variables if not row["lp_range_component_passed"]],
    }
    misleading = {}
    for name in ("primal_feasibility", "stationarity", "complementarity"):
        misleading[name] = (candidate["metrics"][name] <= thresholds[name + "_max"]
                            and not checks[name])
    return {"candidate": candidate, "constraint_and_bound_rows": components, "variables": variables,
            "failed_components": failures, "maxima_can_mask_failure": misleading,
            "lp_range_evaluated_by_checker": not bool(np.any(bad)),
            "lp_range_diagnostic_residual": maximum(b), "lp_range_diagnostic_threshold": range_limit,
            "lp_range_diagnostic_note": "unmodified b; diagnostic only when support is unavailable"}


def assert_candidate_equal(measured: dict, expected: dict) -> None:
    require(measured.keys() == expected.keys(), "independent candidate schema mismatch")
    for key in ("checks", "thresholds", "metrics", "unavailable", "passed", "kind",
                "schema_version", "qualification"):
        if key not in ("metrics", "thresholds"):
            require(measured[key] == expected[key], f"independent checker mismatch: {key}")
            continue
        require(measured[key].keys() == expected[key].keys(), f"checker field mismatch: {key}")
        for name, value in measured[key].items():
            other = expected[key][name]
            if value is None or other is None:
                require(value is other, f"checker unavailable mismatch: {name}")
            else:
                # Independent scalar products may group zero terms differently.
                limit = 16 * np.finfo(float).eps * max(abs(value), abs(other), 1.0)
                require(abs(value - other) <= limit, f"independent checker metric mismatch: {name}")


def magnitude_summary(arrays: dict) -> dict:
    result = {}
    for name in ("A", "q"):
        nonzero = np.abs(arrays[name][arrays[name] != 0])
        result[name] = {"nonzero_min": float(nonzero.min()) if nonzero.size else None,
                        "nonzero_max": float(nonzero.max()) if nonzero.size else None,
                        "magnitude_span": float(nonzero.max() / nonzero.min()) if nonzero.size else None}
    return result


def compact_attempt(row: dict) -> dict:
    detail = row["independent_lp"]
    components = detail["constraint_and_bound_rows"]
    variables = detail["variables"]
    bad_signs = [{key: item[key] for key in
                  ("view_row", "coordinate_kind", "coordinate_name", "selected_side", "y")}
                 for item in components if not item["bound_sign_passed"]]
    comp = [{**{key: item[key] for key in ("view_row", "coordinate_kind", "coordinate_name")},
             **item["complementarity"]}
            for item in components if item["complementarity"] is not None]
    feasibility = [
        {**{key: item[key] for key in ("view_row", "coordinate_kind", "coordinate_name")},
         "side": side, **values}
        for item in components for side, values in item["feasibility"].items()
    ]
    return {
        "case_id": row["case_id"], "solver": row["solver"], "status": row["status"],
        "iterations": row["iterations"], "original_verdict": row["original_verdict"],
        "first_failure": row["first_failure"], "failed_checks": row["failed_checks"],
        "bound_sign_failures": bad_signs,
        "failed_component_counts": {key: len(values)
                                    for key, values in detail["failed_components"].items()},
        "worst_feasibility": max(feasibility, key=lambda item: item["ratio"]) if feasibility else None,
        "worst_complementarity": max(comp, key=lambda item: item["ratio"]) if comp else None,
        "worst_stationarity": max(variables, key=lambda item: item["stationarity_ratio"]),
        "worst_lp_range": max(variables, key=lambda item: item["lp_range_ratio"]),
        "maxima_can_mask_failure": detail["maxima_can_mask_failure"],
        "lp_range_evaluated_by_checker": detail["lp_range_evaluated_by_checker"],
        "convergence_interpretation": row["convergence_interpretation"],
        "matrix_magnitude_facts": row["matrix_magnitude_facts"],
    }


def build_diagnosis(result_path: Path, result_pin: str) -> dict:
    local_importer._sha(result_pin, "caller result pin")
    require(not result_path.is_symlink(), "symlink result refused")
    result_path = result_path.expanduser().resolve(strict=True)
    root = result_path.parent
    require(not root.is_relative_to(ROOT), "qualification evidence must be external")
    raw_result = read_bytes(root, result_path.name)
    require(sha(raw_result) == result_pin, "caller result SHA256 mismatch")
    report = object_json(raw_result)
    require(report.get("schema_version") == 1
            and report.get("runner_version") == "public-qp-qualification-v1"
            and report.get("verdict") in ("PASS", "UNQUALIFIED")
            and "error" not in report, "unsupported/incomplete qualification result")
    inventory = report.get("files")
    require(isinstance(inventory, dict) and bool(inventory), "empty qualification inventory")
    for relative, pin in inventory.items():
        raw = read_bytes(root, relative)
        require(pin == {"bytes": len(raw), "sha256": sha(raw)},
                f"qualification sidecar hash/size mismatch: {relative}")
    actual_files = {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()}
    require(actual_files == set(inventory) | {result_path.name}, "qualification file membership mismatch")
    suffix = "_" + result_pin
    importer = load_code(root / "code/importer.py", "_diagnosis_importer" + suffix)
    runner = load_code(root / "code/runner.py", "_diagnosis_runner" + suffix)
    checker = load_code(root / "code/certificates.py", "_diagnosis_checker" + suffix)
    runner.corpus = importer
    importer.verify_frozen(root / "frozen", report["corpus_sha256"], report["max_dense_bytes"])
    frozen = object_json(read_bytes(root, "frozen/corpus.json"))
    expected = [(row["case_id"], solver, row["model_sha256"])
                for row in frozen["cases"] for solver in SOLVERS]
    require(expected and [(row["case_id"], row["solver"], row["model_sha256"])
                          for row in report["rows"]] == expected, "candidate membership/order mismatch")
    acceptance = object_json(read_bytes(root, "public-acceptance.json")) if (
        "public_acceptance_sha256" in report) else None
    if acceptance is not None:
        require(sha(read_bytes(root, "public-acceptance.json")) == report["public_acceptance_sha256"],
                "source acceptance pin mismatch")
    eligibility = runner.source_eligibility(frozen, root / "frozen", acceptance, report["corpus_sha256"])
    require(eligibility == report["source_eligibility"], "saved source eligibility mismatch")
    rules = {}
    checker_text = read_bytes(root, "code/certificates.py").decode()
    for function in ("_feasibility", "_support", "check_qp_candidate"):
        lines = [i for i, line in enumerate(checker_text.splitlines(), 1)
                 if line.startswith("def " + function + "(")]
        require(len(lines) == 1, "ambiguous pinned checker rule")
        rules[function] = {"path": str(root / "code/certificates.py"), "line": lines[0]}
    results = []
    for row in report["rows"]:
        require(row["attempted"] is True and row["attempt"] == 1 and row["denominator"] == 1
                and row["verdict"] in ("QUALIFIED", "UNQUALIFIED"), "unsupported/missing attempt")
        row_path = f"attempts/{row['case_id']}/{row['solver']}.json"
        require(object_json(read_bytes(root, row_path)) == row, "attempt/result mismatch")
        vectors = runner.read_vectors(root / row["vectors_path"])
        require("x" in vectors and "y" in vectors, "missing original candidate")
        model = runner.load_model(root / "frozen", row["case_id"])
        raw = {**row["raw"], "vectors": vectors}
        assessed = runner.assess(model, row["solver"], raw, report["config"], checker)
        require(assessed == row["assessment"] and assessed["verdict"] == row["verdict"],
                "original checker/status replay mismatch")
        independent = independent_lp(model, vectors["x"], vectors["y"],
                                     report["config"]["eps_abs"], report["config"]["eps_rel"])
        assert_candidate_equal(independent["candidate"], assessed["candidate"])
        failing = [name for name in CHECKS if not assessed["candidate"]["checks"][name]]
        status_cause = None if assessed["strict_status"] else "status is not strict solved/infeasible/unbounded"
        if status_cause:
            first = status_cause
        elif independent["failed_components"]["bound_sign"]:
            first = "nonzero multiplier on absent infinite bound; infinite support"
        else:
            first = failing[0] if failing else None
        results.append({
            "case_id": row["case_id"], "solver": row["solver"], "status": row["raw"]["status"],
            "iterations": row["raw"]["iterations"], "original_verdict": row["verdict"],
            "original_assessment": assessed, "first_failure": first, "failed_checks": failing,
            "original_checker_agrees": True, "independent_lp": independent,
            "settings": row["raw"]["settings"], "matrix_magnitude_facts": magnitude_summary(model["arrays"]),
            "convergence_interpretation": (
                "Iteration budget exhausted with unchanged original residual failures. "
                "No trajectory or scaling ablation was captured; scaling causality is UNVERIFIED."
                if not assessed["strict_status"] else "No convergence-causality claim"),
        })
    summary = runner.summarize(report["rows"], eligibility, report["config"])
    require(all(report.get(key) == value for key, value in summary.items()),
            "qualification aggregate/denominator mismatch")
    doc = ROOT / "docs/research/certificates.md"
    return {
        "schema_version": 1, "version": VERSION, "verdict": "DIAGNOSED_NOT_REQUALIFIED",
        "result_path": str(result_path), "result_sha256": result_pin,
        "original_verdict": report["verdict"], "original_exit_code": report["exit_code"],
        "candidate_count": len(results), "diagnoses": results,
        "compact_summary": [compact_attempt(row) for row in results],
        "frozen_code": {name: inventory["code/" + name] for name in
                        ("runner.py", "importer.py", "certificates.py", "qp.py")},
        "rule_locations": rules,
        "rule_document": {"path": str(doc), "sha256": sha(doc.read_bytes()),
                          "scope": "local explanation; pinned checker source governs original result"},
        "original_config": report["config"], "producer_sha256": sha(Path(__file__).read_bytes()),
        "scope": "LP saved-candidate diagnostics only; zero solves, no candidate or tolerance changes",
        "qualification_upgraded": False, "license_verified": False, "public_redistribution": False,
    }


def verify(path: Path, pin: str) -> dict:
    local_importer._sha(pin, "caller diagnosis pin")
    path = path.expanduser().resolve(strict=True)
    raw = read_bytes(path.parent, path.name)
    require(sha(raw) == pin, "diagnosis caller pin mismatch")
    report = object_json(raw)
    require(report == build_diagnosis(Path(report["result_path"]), report["result_sha256"]),
            "diagnosis differs from saved-candidate replay")
    summary = object_json(read_bytes(path.parent, "summary.json"))
    require(summary == {"schema_version": 1, "diagnosis_sha256": pin,
                        "original_verdict": report["original_verdict"],
                        "qualification_upgraded": False, "attempts": report["compact_summary"]},
            "compact summary differs from full diagnosis")
    return {"verdict": "DIAGNOSIS_REPLAYED", "candidate_count": report["candidate_count"],
            "checked_at": now(), "diagnosis_sha256": pin, "original_verdict": report["original_verdict"],
            "qualification_upgraded": False}


def produce(result: Path, pin: str, output: Path) -> dict:
    output = local_importer.external(output)
    output.mkdir(parents=True, exist_ok=False)
    status = {"schema_version": 1, "verdict": "RUNNING", "started_at": now()}
    try:
        report = build_diagnosis(result, pin)
        local_importer.write_json(output / "diagnosis.json", report)
        diagnosis_pin = sha((output / "diagnosis.json").read_bytes())
        local_importer.write_json(
            output / "summary.json",
            {"schema_version": 1, "diagnosis_sha256": diagnosis_pin,
             "original_verdict": report["original_verdict"], "qualification_upgraded": False,
             "attempts": report["compact_summary"]},
        )
        status.update(verify(output / "diagnosis.json", diagnosis_pin))
        status["verdict"] = report["verdict"]
    except Exception as error:
        status.update({"verdict": "ERROR", "exit_code": 2,
                       "error": {"type": type(error).__name__, "message": str(error)}})
        (output / "failure.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        status["finished_at"] = now()
        local_importer.write_json(output / "status.json", status)
        (output / "diagnosis.log").write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
    return status


def main(argv: list[str] | None = None) -> int:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("--result", type=Path, required=True)
    diagnose.add_argument("--result-sha256", required=True)
    diagnose.add_argument("--artifact-dir", type=Path, required=True)
    replay = commands.add_parser("verify")
    replay.add_argument("--diagnosis", type=Path, required=True)
    replay.add_argument("--diagnosis-sha256", required=True)
    args = cli.parse_args(argv)
    try:
        result = produce(args.result, args.result_sha256, args.artifact_dir) if args.action == "diagnose" else (
            verify(args.diagnosis, args.diagnosis_sha256))
        print("PUBLIC_QP_DIAGNOSIS " + json.dumps(result, sort_keys=True), flush=True)
        return 0
    except Exception:
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
