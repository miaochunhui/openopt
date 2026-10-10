"""Caller-pinned, CPU-only extension CUDA execution and candidate replay.

Only the versioned single-target native_conic/refinement selections are
supported. Frozen tests are parsed, never imported; only existing independent
checkers run. PASS is not a certificate, commercial qualification or speedup.
"""

from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import itertools
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
from scripts import accept_release_cuda as execution

require = execution.require
SCOPE = "extension CUDA execution and original-model numerical replay"
AUDITORS = ("scripts/accept_extension_cuda.py", "scripts/accept_release_cuda.py")
SOURCES = {
    "native_conic": (
        "src/openopt/native_conic.py", "src/openopt/conic.py",
        "src/openopt/certificates.py", "tests/test_native_conic.py",
    ),
    "refinement": (
        "src/openopt/refinement.py", "src/openopt/qp.py", "src/openopt/certificates.py",
        "tests/test_qp_refinement.py", "docs/research/refinement.md",
    ),
}
CONTRACTS = {
    "native_conic": {
        "selection": "cuda", "count": 21, "cuda_count": 19,
        "cpu": {"test_explicit_cuda_never_falls_back", "test_reject_cuda_out_of_range"},
        "functions": {"test_explicit_cuda_never_falls_back", "test_reject_cuda_out_of_range",
                      "test_cuda_native_conic_frozen_original_model",
                      "test_cuda_native_conic_reject_numerical_tamper"},
        "property": "native_conic_acceptance",
    },
    "refinement": {
        "selection": "cuda_refinement", "count": 7, "cuda_count": 7, "cpu": set(),
        "functions": {"test_cuda_refinement_frozen_model_accuracy_ab",
                      "test_cuda_refinement_actual_residual_correction",
                      "test_cuda_refinement_negative_controls"},
        "property": "refinement_acceptance",
    },
}


def static_value(node):
    """Restricted pytest parameter values; no eval, imports or factory calls."""
    if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        return [static_value(value) for value in node.elts]
    if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "torch" and node.attr in ("float32", "float64")):
        return {"dtype": "torch." + node.attr}
    raise execution.AuditError("frozen parametrization: unsupported dynamic value")


def expected_cases(snapshot, suite):
    require(suite in CONTRACTS, "suite", "unknown extension suite")
    contract = CONTRACTS[suite]
    source = next(name for name in SOURCES[suite] if name.startswith("tests/"))
    tree = ast.parse(execution.read(Path(snapshot) / source), filename=source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name.startswith("test_") and contract["selection"] in node.name.lower()]
    require({node.name for node in functions} == contract["functions"], source,
            "selected frozen functions changed; update versioned acceptance contract")
    cases = {}
    for node in functions:
        groups = []
        for decorator in reversed(node.decorator_list):
            require(isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and isinstance(decorator.func.value, ast.Attribute)
                    and isinstance(decorator.func.value.value, ast.Name)
                    and decorator.func.value.value.id == "pytest"
                    and decorator.func.value.attr == "mark",
                    node.name, "unsupported selected-test decorator")
            if decorator.func.attr == "skipif":
                continue
            require(decorator.func.attr == "parametrize" and len(decorator.args) == 2
                    and not decorator.keywords, node.name, "dynamic pytest IDs/marks rejected")
            names = static_value(decorator.args[0]).split(",")
            names = [name.strip() for name in names]
            values = static_value(decorator.args[1])
            require(values and len(set(names)) == len(names), node.name, "empty/duplicate parameters")
            group = []
            for index, row in enumerate(values):
                row = [row] if len(names) == 1 else row
                require(isinstance(row, list) and len(row) == len(names),
                        node.name, "parameter dimensions mismatch")
                ids = [name + str(index) if isinstance(value, dict) else str(value)
                       for name, value in zip(names, row)]
                group.append(("-".join(ids), dict(zip(names, row))))
            groups.append(group)
        for product in itertools.product(*groups):
            parameters = {}
            for _, values in product:
                require(not set(parameters) & set(values), node.name, "repeated parameter names")
                parameters.update(values)
            name = node.name + ("[" + "-".join(item[0] for item in product) + "]" if groups else "")
            identity = source[:-3].replace("/", ".") + "::" + name
            require(identity not in cases, identity, "duplicate expected pytest identity")
            cases[identity] = {"source": source, "name": name, "function": node.name,
                               "parameters": parameters, "cpu_only": node.name in contract["cpu"]}
    require(len(cases) == contract["count"]
            and sum(not case["cpu_only"] for case in cases.values()) == contract["cuda_count"],
            source, "full selected positive/negative denominator changed")
    # Count alone cannot detect a replaced parameter at the same cardinality.
    if suite == "native_conic":
        required_names = {"soc_lp", "mixed", "nonnegative", "equality",
                          "empty", "singular", "soc_origin", "soc2"}
        positives = [case["parameters"] for case in cases.values()
                     if case["function"] == "test_cuda_native_conic_frozen_original_model"]
        require({(p["name"], p["dtype"]["dtype"]) for p in positives}
                == set(itertools.product(required_names, ("torch.float32", "torch.float64"))),
                source, "positive parameter domain changed")
        require({case["parameters"]["mutation"] for case in cases.values()
                 if case["function"] == "test_cuda_native_conic_reject_numerical_tamper"}
                == {"x", "s", "z"}, source, "negative parameter domain changed")
    else:
        positives = [case["parameters"] for case in cases.values()
                     if case["function"] == "test_cuda_refinement_frozen_model_accuracy_ab"]
        require({(p["dtype"]["dtype"], p["factor_dtype"]["dtype"]) for p in positives}
                == {("torch.float64", "torch.float64"), ("torch.float64", "torch.float32"),
                    ("torch.float32", "torch.float32")}, source, "accuracy A/B domain changed")
        require({case["parameters"]["kind"] for case in cases.values()
                 if case["function"] == "test_cuda_refinement_negative_controls"}
                == {"wrong_active_set", "singular_kkt", "wrong_sign"},
                source, "negative parameter domain changed")
    return cases


def clean_sources(root, names):
    root = Path(root).resolve()
    result = subprocess.run(["git", "-C", str(root), "status", "--porcelain=v1", "-z",
                             "--untracked-files=all"], capture_output=True, check=False)
    require(result.returncode == 0, root, f"git status failed rc={result.returncode}")
    require(not result.stdout, root, "clean committed current source required")
    for name in names:
        committed = subprocess.run(["git", "-C", str(root), "show", f"HEAD:{name}"],
                                   capture_output=True, check=False)
        require(committed.returncode == 0, name, "current source not tracked at HEAD")
        require(committed.stdout == execution.read(root / name), name, "HEAD source byte drift")


def auditor_sources():
    clean_sources(ROOT, AUDITORS)
    result = {}
    for name, module in zip(AUDITORS, (sys.modules[__name__], execution)):
        data = execution.read(Path(module.__file__))
        require(data == execution.read(ROOT / name), name, "loaded/current auditor byte mismatch")
        result[name] = {"sha256": execution.sha(data), "bytes": len(data)}
    return result


def final_source_binding(frozen_sources, auditors, *, development_replay):
    if not development_replay:
        for name, item in auditors.items():
            require(frozen_sources.get(name) == item["sha256"], name,
                    "final-source requires identical auditor in run snapshot; no retrospective promotion")


def check_loaded(function, name, sources):
    path = Path(sys.modules[function.__module__].__file__)
    require(execution.sha(execution.read(path)) == sources[name],
            path, "loaded independent checker source mismatch")


def source_audit(saved, suite, snapshot, current_root, sources, extras):
    hashes = saved.get("source_sha256")
    require(isinstance(hashes, dict) and set(hashes) == set(SOURCES[suite]),
            "sidecar sources", "exact sidecar source membership required")
    for name, pin in hashes.items():
        execution.digest(pin)
        current = Path(current_root) / name
        frozen = Path(snapshot) / name
        require(not current.is_symlink() and not frozen.is_symlink(),
                name, "source symlink rejected")
        require(execution.sha(execution.read(current)) == pin
                and execution.sha(execution.read(frozen)) == pin,
                name, "sidecar source differs from current/run snapshot")
        if name in sources:
            require(sources.get(name) == pin, name, "sidecar source differs from run manifest")
        else:
            require(name in extras and extras[name]["sha256"] == pin
                    and extras[name]["path"] == str(frozen),
                    name, "supplemental snapshot source not sealed")


def tolerance(atol, rtol):
    require(type(atol) in (int, float) and math.isfinite(atol) and 0 < atol <= 1e-3
            and type(rtol) in (int, float) and math.isfinite(rtol) and 0 <= rtol <= 1e-3,
            "sidecar tolerance", "finite bounded original-check thresholds required")
    return {"atol": atol, "rtol": rtol}


def numeric(value, coordinate, *, bounds=False):
    import numpy as np

    def checked(item):
        if isinstance(item, list):
            return [checked(v) for v in item]
        if bounds and item in ("Infinity", "-Infinity"):
            return float(item)
        require(type(item) in (int, float) and math.isfinite(item),
                coordinate, "real finite frozen data required")
        return item

    return np.asarray(checked(value), dtype=np.float64)


def evidence_differences(saved, replayed, pointer=""):
    """Exact, typed evidence comparison. No ULP/relative-error acceptance."""
    if isinstance(saved, dict) and isinstance(replayed, dict):
        result = []
        for key in sorted(set(saved) | set(replayed)):
            child = pointer + "/" + key
            if key not in saved or key not in replayed:
                result.append({"pointer": child, "reason": "missing evidence member"})
            else:
                result.extend(evidence_differences(saved[key], replayed[key], child))
        return result
    numeric_types = (int, float)
    same_type = type(saved) is type(replayed) or (
        type(saved) in numeric_types and type(replayed) in numeric_types)
    if same_type and saved == replayed:
        return []
    return [{"pointer": pointer, "saved": saved, "replayed": replayed,
             "saved_hex": float(saved).hex() if type(saved) in numeric_types else None,
             "replayed_hex": float(replayed).hex() if type(replayed) in numeric_types else None}]


def exact_evidence(saved, replayed, coordinate):
    differences = evidence_differences(saved, replayed)
    require(not differences, coordinate,
            "independent original checker replay mismatch at "
            + ", ".join(item["pointer"] for item in differences))


def native_replay(saved, parameters, *, negative=None):
    from openopt.certificates import NumericalEvidence
    from openopt.conic import ConicProblem, Nonnegative, SOC, Zero, check_conic_candidate

    check_loaded(check_conic_candidate, "src/openopt/conic.py", saved["source_sha256"])
    check_loaded(NumericalEvidence.to_dict, "src/openopt/certificates.py", saved["source_sha256"])
    require(saved.get("schema_version") == 1 and saved.get("family") == "native_conic"
            and saved.get("verdict") == "PASS" and saved.get("status") == "solved",
            "native sidecar", "successful frozen candidate required")
    cfg = saved["config"]
    require(set(cfg) == {"enabled", "device", "dtype", "rho", "sigma",
                         "max_iterations", "check_interval"},
            "native config", "complete frozen config required")
    require(cfg["enabled"] is True and cfg["device"] == saved["device"] == "cuda:0"
            and cfg["dtype"] == saved["dtype"] in ("torch.float32", "torch.float64")
            and saved["placements"] == dict.fromkeys(("x", "s", "z"), "cuda:0"),
            "native placement", "explicit config/vector device/dtype mismatch")
    if "dtype" in parameters:
        require(saved["dtype"] == parameters["dtype"]["dtype"],
                "native dtype", "JUnit parameter/sidecar substitution")
    expected_atol = 3e-5 if saved["dtype"] == "torch.float32" else 1e-8
    require(saved["atol"] == expected_atol and saved["rtol"] == 0,
            "native tolerance", "frozen-test check threshold mismatch")
    if "mutation" in parameters:
        require(saved["dtype"] == "torch.float64", "native negative dtype", "expected float64")
    require(type(saved["iterations"]) is int and saved["iterations"] > 0
            and type(cfg["max_iterations"]) is int and cfg["max_iterations"] >= saved["iterations"]
            and type(cfg["check_interval"]) is int and cfg["check_interval"] > 0,
            "native iterations", "actual iteration/config evidence missing")
    require(all(type(cfg[name]) in (int, float) and math.isfinite(cfg[name]) and cfg[name] > 0
                for name in ("rho", "sigma")), "native config", "invalid proximal parameters")
    model = saved["problem"]
    require(set(model) == {"P", "q", "A", "b", "cones"}, "native problem", "incomplete model")
    q, b = (numeric(model[name], name) for name in ("q", "b"))
    A = numeric(model["A"], "A")
    if model["A"] == []:
        A = A.reshape(0, q.size)
    cone_types = {"zero": Zero, "nonnegative": Nonnegative, "soc": SOC}
    cones = []
    for block in model["cones"]:
        require(set(block) == {"kind", "dimension"} and block["kind"] in cone_types,
                "native cone", "unknown frozen cone")
        cones.append(cone_types[block["kind"]](block["dimension"]))
    problem = ConicProblem(numeric(model["P"], "P"), q, A, b, tuple(cones))
    vectors = {name: numeric(saved[name], name) for name in ("x", "s", "z")}
    tol = tolerance(saved["atol"], saved["rtol"])
    evidence = check_conic_candidate(problem, **vectors, **tol)
    require(evidence.passed and evidence.to_dict() == saved["evidence"]
            and evidence.metrics["primal_objective"] == saved["objective"],
            "native candidate", "independent original-model replay rejected/disagrees")
    if negative is not None:
        mutation = parameters["mutation"]
        expected = copy.deepcopy(saved)
        expected[mutation][0] += 1
        require(negative == expected, "native negative artifact",
                "actual tampered.json differs from frozen test mutation")
        mutated = {name: numeric(negative[name], name) for name in vectors}
        require(not check_conic_candidate(problem, **mutated, **tol).passed,
                "native negative replay", "numerical tamper unexpectedly passed")
    return {"verdict": "PASS", "candidate_status": "solved",
            "candidate_accepted": True, "evidence": evidence.to_dict(),
            "negative_replayed": negative is not None}


def refinement_replay(saved, parameters, function, *, diagnostic=False):
    import numpy as np
    from openopt.certificates import check_qp_candidate

    check_loaded(check_qp_candidate, "src/openopt/certificates.py", saved["source_sha256"])
    require(saved.get("schema_version") == 1
            and saved.get("kind") == "qp_refinement_original_coordinate_ab"
            and saved.get("verdict") == "PASS", "refinement sidecar", "unknown schema/verdict")
    cfg = saved["config"]
    require(set(cfg) == {"device", "dtype", "factor_dtype", "max_refinement_steps",
                         "active_atol", "active_rtol", "check_atol", "check_rtol"},
            "refinement config", "complete frozen config required")
    require(cfg["device"] == saved["device"] == "cuda:0"
            and cfg["dtype"] == saved["dtype"] in ("torch.float32", "torch.float64")
            and cfg["factor_dtype"] in ("torch.float32", "torch.float64"),
            "refinement config", "explicit device/dtype contradiction")
    for name in ("dtype", "factor_dtype"):
        if name in parameters:
            require(cfg[name] == parameters[name]["dtype"], name, "JUnit/config substitution")
    tol = tolerance(cfg["check_atol"], cfg["check_rtol"])
    correction = function == "test_cuda_refinement_actual_residual_correction"
    require(cfg["max_refinement_steps"] == 3 and type(cfg["max_refinement_steps"]) is int
            and cfg["active_atol"] == 1e-5 and type(cfg["active_atol"]) in (int, float)
            and cfg["active_rtol"] == 1e-7 and type(cfg["active_rtol"]) in (int, float)
            and cfg["check_atol"] == cfg["check_rtol"] == (1e-10 if correction else 1e-8),
            "refinement config", "frozen-test thresholds/step limit mismatch")
    model = saved["problem"]
    require(set(model) == {"P", "q", "A", "l", "u"},
            "refinement problem", "complete original model required")
    values = {name: numeric(model[name], name, bounds=name in ("l", "u")) for name in model}
    if model["A"] == []:
        values["A"] = values["A"].reshape(0, values["q"].size)
    problem = SimpleNamespace(**values)
    baseline, refined = saved["baseline"], saved["refined"]
    bx, by = (numeric(baseline[name], "baseline/" + name) for name in ("x", "y"))
    xx, yy = (numeric(refined[name], "refined/" + name) for name in ("x", "y"))
    before = check_qp_candidate(problem, bx, by, **tol)
    after = check_qp_candidate(problem, xx, yy, **tol)
    differences = evidence_differences(baseline["evidence"], before.to_dict(), "/baseline")
    differences += evidence_differences(refined["evidence"], after.to_dict(), "/refined")
    if not diagnostic:
        exact_evidence(baseline["evidence"], before.to_dict(), "refinement baseline evidence")
        exact_evidence(refined["evidence"], after.to_dict(), "refinement refined evidence")
    accepted = refined["status"] == "accepted"
    require(type(saved["candidate_accepted"]) is bool and saved["candidate_accepted"] == accepted,
            "refinement status", "candidate acceptance contradiction")
    diagnostics = refined["diagnostics"]
    require(diagnostics.get("solver_fallback") is False, "refinement diagnostics",
            "explicit no-solver-fallback diagnostic required")
    require(isinstance(refined["reason"], str) and refined["reason"],
            "refinement reason", "explicit outcome reason required")
    if accepted:
        require(after.passed and refined["reason"] == "original_check_passed_nonworsening",
                "refinement accepted", "original KKT/gap or accepted reason failed")
        for name in ("primal_feasibility", "stationarity", "bound_sign",
                     "complementarity", "duality_gap"):
            a, b = before.metrics[name], after.metrics[name]
            require(b is not None and (a is None or abs(b) <= abs(a)),
                    name, "accepted refinement worsened original metric")
    else:
        require(refined["status"] in ("rejected", "unknown")
                and np.array_equal(bx, xx) and np.array_equal(by, yy),
                "refinement rejection", "explicit rejection must retain baseline")
    if function == "test_cuda_refinement_negative_controls":
        kind = parameters["kind"]
        require(not accepted and saved["case"] == kind, "refinement negative",
                "negative testcase incorrectly promoted/substituted")
        reason = "singular_or_unresolved_kkt" if kind == "singular_kkt" else "original_check_failed"
        require(refined["reason"] == reason
                and refined["status"] == ("unknown" if kind == "singular_kkt" else "rejected"),
                "refinement negative", "wrong active-set/singular/sign first cause mismatch")
        require(cfg["dtype"] == cfg["factor_dtype"] == "torch.float64",
                "refinement negative config", "expected float64 residual/factor")
    else:
        require(accepted, "refinement positive", "positive CUDA candidate was not accepted")
    wrong_problem = SimpleNamespace(**{**values, "q": values["q"] + 1})
    controls = {
        "wrong_x": check_qp_candidate(problem, xx + 1, yy, **tol),
        "wrong_y": check_qp_candidate(problem, xx, -yy, **tol),
        "wrong_model": check_qp_candidate(wrong_problem, xx, yy, **tol),
    }
    controls_dict = {name: value.to_dict() for name, value in controls.items()}
    differences += evidence_differences(saved["negative_controls"], controls_dict, "/negative_controls")
    if not diagnostic:
        exact_evidence(saved["negative_controls"], controls_dict, "refinement controls")
    require(not accepted or not any(value.passed for value in controls.values()),
            "refinement controls", "accepted candidate negative control unexpectedly passed")
    if function == "test_cuda_refinement_actual_residual_correction":
        trace = diagnostics["iterations"]
        require(cfg["dtype"] == "torch.float64" and cfg["factor_dtype"] == "torch.float32"
                and not before.passed and len(trace) >= 2
                and trace[0]["original_check_passed"] is False
                and trace[1]["linear_residual"] < trace[0]["linear_residual"],
                "refinement correction", "actual mixed-precision correction evidence missing")
    if diagnostic:
        # Independent current-runtime checks remain complete and unchanged.
        # Diagnostic evidence never grants seal/accept, even when numerics pass.
        return {"verdict": "DIAGNOSTIC_ONLY", "numerical_gate": "PASS",
                "recorded_evidence_exact": not differences, "differences": differences,
                "candidate_accepted": accepted, "candidate_status": refined["status"],
                "baseline_evidence": before.to_dict(), "refined_evidence": after.to_dict(),
                "negative_controls": controls_dict, "reason": refined["reason"],
                "qualified": False, "certified": False}
    return {"verdict": "PASS", "candidate_accepted": accepted,
            "candidate_status": refined["status"], "reason": refined["reason"],
            "baseline_evidence": before.to_dict(), "refined_evidence": after.to_dict(),
            "negative_replayed": True}


def pinned(path, pin):
    item = execution.descriptor(path)
    require(item["sha256"] == execution.digest(pin), path, "caller artifact pin mismatch")
    return item


def caller_pins(*, result_path, result_sha256, run_path, run_sha256, junit_path, junit_sha256):
    return {"result": pinned(result_path, result_sha256),
            "run": pinned(run_path, run_sha256), "junit": pinned(junit_path, junit_sha256)}


def validate(bundle, *, pins, current_root, development_replay=False):
    require(isinstance(bundle, dict) and set(bundle) == {
        "schema_version", "scope", "suite", "execution", "pins", "extra_sources", "negatives",
        "auditors", "development_replay"},
        "extension bundle", "exact extension schema required")
    suite = bundle["suite"]
    require(bundle["schema_version"] == 1 and bundle["scope"] == SCOPE and suite in CONTRACTS,
            "extension bundle", "unknown version/scope/suite")
    require(type(bundle["development_replay"]) is bool
            and bundle["development_replay"] == development_replay,
            "replay mode", "caller/bundle final-source mode mismatch")
    frozen = bundle["execution"]
    require(bundle["pins"] == pins
            and frozen["result"] == pins["result"] and frozen["run"] == pins["run"]
            and frozen["files"]["pytest.xml"] == pins["junit"],
            "caller pins", "exact caller run/result/JUnit path+hash+bytes mismatch")
    for item in pins.values():
        execution.bound(item)
    clean_sources(current_root, SOURCES[suite])
    require(bundle["auditors"] == auditor_sources(), "auditors", "sealed/loaded auditor byte drift")
    final_source_binding(frozen["sources"], bundle["auditors"],
                         development_replay=development_replay)
    observed = execution.validate(frozen, current_root=current_root)
    tree, snapshot = Path(pins["run"]["path"]).parent, Path(frozen["source_snapshot"])
    expected_extras = {name for name in SOURCES[suite] if name not in frozen["sources"]}
    require(development_replay or not expected_extras, "final-source",
            "all sidecar sources including explicit refinement doc must be in run manifest")
    require(set(bundle["extra_sources"]) == expected_extras,
            "supplemental sources", "exact supplemental source set required")
    for name, item in bundle["extra_sources"].items():
        require(item["path"] == str(snapshot / name), name, "supplemental path substitution")
        execution.bound(item, snapshot)
    report = execution.parse(execution.bound(frozen["result"], tree), "result")
    source = next(name for name in SOURCES[suite] if name.startswith("tests/"))
    require(report["targets"] == [source] and report["selection"] == CONTRACTS[suite]["selection"],
            "selection", "exact single-target full extension selection required")
    _, cases = execution.junit(execution.bound(frozen["files"]["pytest.xml"], tree),
                               report, frozen["sources"])
    expected = expected_cases(snapshot, suite)
    actual = {case["id"]: case for case in cases}
    require(set(actual) == set(expected), "JUnit denominator",
            f"missing={sorted(set(expected) - set(actual))} extra={sorted(set(actual) - set(expected))}")
    outcomes, paths, negative_ids = [], set(), set()
    prop = CONTRACTS[suite]["property"]
    for identity, definition in expected.items():
        case = actual[identity]
        if definition["cpu_only"]:
            require(not case["properties"], identity, "CPU rejection test cannot claim CUDA sidecar")
            outcomes.append({"case": identity, "execution": "PASS", "device_scope": "CPU rejection"})
            continue
        require(set(case["properties"]) == {prop + "_path", prop + "_sha256"},
                identity, "exact sidecar path/SHA properties required")
        path = execution.external(case["properties"][prop + "_path"], tree)
        require(str(path) == case["properties"][prop + "_path"]
                and path.is_relative_to(Path(pins["result"]["path"]).parent / "pytest-tmp")
                and str(path) not in paths, identity, "sidecar path alias/escape/shared substitution")
        paths.add(str(path))
        item = frozen["proofs"][str(path)]
        require(item["path"] == str(path)
                and item["sha256"] == execution.digest(case["properties"][prop + "_sha256"]),
                identity, "JUnit property/sidecar descriptor pin mismatch")
        saved = execution.parse(execution.bound(item, tree), str(path))
        source_audit(saved, suite, snapshot, current_root, frozen["sources"], bundle["extra_sources"])
        require(saved.get("device") == "cuda:0" and saved.get("gpu_name") == observed["hardware"]["gpu"],
                identity, "saved CUDA device/GPU differs from actual run")
        negative = None
        if definition["function"] == "test_cuda_native_conic_reject_numerical_tamper":
            negative_ids.add(identity)
            require(identity in bundle["negatives"], identity, "actual negative artifact absent")
            negative_item = bundle["negatives"][identity]
            require(negative_item["path"] == str(path.parent / "tampered.json"),
                    identity, "negative artifact substitution")
            negative = execution.parse(execution.bound(negative_item, tree), identity)
        replay = (native_replay(saved, definition["parameters"], negative=negative)
                  if suite == "native_conic" else refinement_replay(
                      saved, definition["parameters"], definition["function"]))
        outcomes.append({"case": identity, "path": str(path), "sha256": item["sha256"],
                         "device_scope": "CUDA test assertions", **replay})
    require(paths == set(frozen["proofs"]), "sidecars", "unlinked/extra/missing sidecar")
    require(set(bundle["negatives"]) == negative_ids,
            "negative artifacts", "unlinked/extra/missing negative artifact")
    return {"verdict": "PASS", "scope": SCOPE, "suite": suite, "tests": observed["tests"],
            "expected_testcases": sorted(expected), "cuda_testcases": CONTRACTS[suite]["cuda_count"],
            "cpu_rejection_testcases": len(CONTRACTS[suite]["cpu"]), "cases": outcomes,
            "pins": pins, "hardware": observed["hardware"], "current_scope": "PASS",
            "current_core": observed["current_core"], "auditors": bundle["auditors"],
            "source_sha_informational": observed["source_sha_informational"],
            "final_source": not development_replay, "development_replay": development_replay,
            "qualified": False, "commercial": "UNVERIFIED", "certified": False,
            "performance": "UNVERIFIED",
            "qualification": "execution plus scoped numerical replay only; no commercial/exact/speed claim"}


def seal(*, suite, current_root, development_replay=False, **pin_arguments):
    require(suite in CONTRACTS, "suite", "unknown extension suite")
    pins = caller_pins(**pin_arguments)
    snapshot = Path(pins["run"]["path"]).parent / "source"
    frozen = execution.seal(pins["result"]["path"], pins["result"]["sha256"],
                            source_snapshot=snapshot, run_path=pins["run"]["path"],
                            current_root=current_root)
    extras = {name: execution.descriptor(snapshot / name) for name in SOURCES[suite]
              if name not in frozen["sources"]}
    report = execution.parse(execution.bound(frozen["result"]), "result")
    _, cases = execution.junit(execution.bound(frozen["files"]["pytest.xml"]), report, frozen["sources"])
    negatives = {}
    for case in cases:
        if case["name"].startswith("test_cuda_native_conic_reject_numerical_tamper["):
            path = Path(case["properties"]["native_conic_acceptance_path"]).parent / "tampered.json"
            negatives[case["id"]] = execution.descriptor(path)
    bundle = {"schema_version": 1, "scope": SCOPE, "suite": suite,
              "execution": frozen, "pins": pins, "extra_sources": extras, "negatives": negatives,
              "auditors": auditor_sources(), "development_replay": development_replay}
    validate(bundle, pins=pins, current_root=current_root, development_replay=development_replay)
    return bundle


def accept(bundle_path, bundle_sha256, *, current_root, development_replay=False, **pin_arguments):
    data = execution.read(execution.external(bundle_path))
    require(execution.sha(data) == execution.digest(bundle_sha256),
            bundle_path, "caller bundle pin mismatch")
    return validate(execution.parse(data, bundle_path),
                    pins=caller_pins(**pin_arguments), current_root=current_root,
                    development_replay=development_replay)


def diagnose_refinement(*, current_root, **pin_arguments):
    """Complete pinned execution/source check, all seven diagnostic-only replays."""
    import numpy as np
    import torch

    pins = caller_pins(**pin_arguments)
    snapshot = Path(pins["run"]["path"]).parent / "source"
    frozen = execution.seal(pins["result"]["path"], pins["result"]["sha256"],
                            source_snapshot=snapshot, run_path=pins["run"]["path"],
                            current_root=current_root)
    require(frozen["result"] == pins["result"] and frozen["run"] == pins["run"]
            and frozen["files"]["pytest.xml"] == pins["junit"],
            "diagnostic caller pins", "exact caller run/result/JUnit path+hash+bytes mismatch")
    clean_sources(current_root, SOURCES["refinement"])
    auditor = auditor_sources()
    observed = execution.validate(frozen, current_root=current_root)
    require(not any(name not in frozen["sources"] for name in SOURCES["refinement"]),
            "diagnostic sources", "complete frozen refinement sources required")
    report = execution.parse(execution.bound(frozen["result"]), "result")
    require(report["targets"] == ["tests/test_qp_refinement.py"]
            and report["selection"] == "cuda_refinement", "diagnostic selection",
            "exact isolated refinement selection required")
    _, cases = execution.junit(execution.bound(frozen["files"]["pytest.xml"]), report, frozen["sources"])
    expected = expected_cases(snapshot, "refinement")
    actual = {case["id"]: case for case in cases}
    require(set(expected) == set(actual), "diagnostic denominator", "exact seven testcases required")
    outcomes, paths = [], set()
    for identity, definition in expected.items():
        props = actual[identity]["properties"]
        require(set(props) == {"refinement_acceptance_path", "refinement_acceptance_sha256"},
                identity, "exact sidecar path/hash properties required")
        path = execution.external(props["refinement_acceptance_path"], Path(pins["run"]["path"]).parent)
        require(str(path) == props["refinement_acceptance_path"]
                and path.is_relative_to(Path(pins["result"]["path"]).parent / "pytest-tmp")
                and str(path) not in paths, identity, "sidecar alias/escape/shared path")
        paths.add(str(path))
        item = frozen["proofs"][str(path)]
        require(item["sha256"] == execution.digest(props["refinement_acceptance_sha256"]),
                identity, "sidecar property pin mismatch")
        saved = execution.parse(execution.bound(item), identity)
        source_audit(saved, "refinement", snapshot, current_root, frozen["sources"], {})
        require(saved["device"] == "cuda:0" and saved["gpu_name"] == observed["hardware"]["gpu"],
                identity, "sidecar execution placement mismatch")
        try:
            outcome = refinement_replay(saved, definition["parameters"], definition["function"],
                                        diagnostic=True)
        except Exception as exc:
            outcome = {"verdict": "DIAGNOSTIC_ONLY", "numerical_gate": "FAIL",
                       "first_cause": f"{type(exc).__name__}: {exc}"}
        outcomes.append({"case": identity, "sidecar": item, **outcome})
    require(paths == set(frozen["proofs"]), "diagnostic sidecars", "unlinked/extra/missing sidecar")
    for item in pins.values():
        execution.bound(item)
    return {"verdict": "DIAGNOSTIC_ONLY", "scope": "no seal/accept qualification granted",
            "tests": observed["tests"], "cases": outcomes, "pins": pins,
            "auditors": auditor, "source_sha_informational": observed["source_sha_informational"],
            "numpy": np.__version__, "torch": torch.__version__,
            "numpy_config": np.show_config(mode="dicts"),
            "producer_checker_runtime": "NOT_FROZEN_IN_SIDECAR",
            "exact_mismatch_cases": sum(not item.get("recorded_evidence_exact", False) for item in outcomes),
            "numerical_gate_failures": sum(item["numerical_gate"] != "PASS" for item in outcomes),
            "final_source": False, "qualified": False, "certified": False,
            "commercial": "UNVERIFIED", "performance": "UNVERIFIED"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("seal", "accept", "diagnose-refinement"))
    parser.add_argument("--suite", choices=tuple(CONTRACTS))
    for name in ("result", "run", "junit"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sha256")
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--development-replay", action="store_true",
                        help="explicit non-final retrospective numerical replay only")
    args = parser.parse_args(argv)
    artifact = execution.external(args.artifact_dir)
    require(not artifact.is_relative_to(args.current_root.resolve()),
            artifact, "audit output must be external to current source")
    artifact.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=args.action + "-", dir=artifact))
    report = {"started_at": datetime.now(timezone.utc).isoformat()}
    rc = 0
    pin_arguments = {name + "_path": getattr(args, name) for name in ("result", "run", "junit")}
    pin_arguments.update({name + "_sha256": getattr(args, name + "_sha256")
                          for name in ("result", "run", "junit")})
    try:
        if args.action == "diagnose-refinement":
            require(not args.bundle and not args.sha256 and not args.development_replay
                    and args.suite in (None, "refinement"), "diagnostic mode",
                    "diagnostic-only entry cannot take acceptance options")
            report.update(diagnose_refinement(current_root=args.current_root, **pin_arguments))
            rc = 1 if report["exact_mismatch_cases"] or report["numerical_gate_failures"] else 0
            if rc:
                with (output / "failure.log").open("x", encoding="utf-8") as log:
                    for case in report["cases"]:
                        log.write(case["case"] + "\n")
                        for item in case.get("differences", []):
                            log.write(item["pointer"] + ": "
                                      + str(item.get("saved_hex")) + " != "
                                      + str(item.get("replayed_hex")) + "\n")
                        if "first_cause" in case:
                            log.write(case["first_cause"] + "\n")
        elif args.action == "seal":
            require(args.suite, "seal", "explicit suite required")
            bundle = seal(suite=args.suite, current_root=args.current_root,
                          development_replay=args.development_replay, **pin_arguments)
            execution.write(output / "bundle.json", bundle)
            report.update(verdict="SEALED", scope=SCOPE, bundle=str(output / "bundle.json"),
                          bundle_sha256=execution.sha(execution.read(output / "bundle.json")),
                          final_source=not args.development_replay)
        else:
            require(args.bundle and args.sha256, "accept", "caller bundle path/SHA required")
            report.update(accept(args.bundle, args.sha256, current_root=args.current_root,
                                 development_replay=args.development_replay, **pin_arguments))
            require(not args.suite or args.suite == report["suite"], "suite", "caller suite mismatch")
    except Exception as error:
        rc = 1
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        with (output / "failure.log").open("x", encoding="utf-8") as log:
            log.write(traceback.format_exc())
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    execution.write(output / "acceptance.json", report)
    line = (f"OPENOPT_EXTENSION_CUDA_ACCEPT verdict={report['verdict']} "
            f"at={report['finished_at']} evidence={output / 'acceptance.json'}")
    with (output / "execution.log").open("x", encoding="utf-8") as log:
        log.write(line + f"\nreturncode={rc}\n")
    print(line, flush=True)
    if rc:
        print(report.get("error", "diagnostic mismatch; seal/accept NOT_GRANTED"), flush=True)
    elif args.action == "diagnose-refinement":
        print("diagnostic only; seal/accept NOT_GRANTED", flush=True)
    elif args.action == "seal":
        print(f"bundle_sha256={report['bundle_sha256']}", flush=True)
    else:
        print(f"suite={report['suite']} tests={report['tests']['tests']} "
              f"cuda_tests={report['cuda_testcases']} final_source={report['final_source']} "
              "commercial=UNVERIFIED certified=false", flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
