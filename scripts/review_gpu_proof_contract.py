"""CPU-only adversarial review of caller-pinned producer/dispatch contracts.

Only run produces CPU evidence. Replay reads immutable evidence and a clean,
caller-pinned producer checkout; it never solves, recreates models or contacts
the cluster. This scoped review is not CUDA execution acceptance.
"""

from __future__ import annotations

import argparse
import ast
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = Path("/hf3fs-jd/data/tmp").resolve()
PRODUCER = "scripts/gpu_release_proof.py"
DISPATCH = "scripts/cluster_benchmark.py"
AUDITOR = "scripts/accept_gpu_proof_run.py"
CASES = (
    "scaling_basic", "scaling_ill_conditioned", "jacobi_coo", "jacobi_csr",
    "gap_spd", "gap_lp", "gap_singular", "infeasible", "unbounded",
)
MUTATIONS = ("cpu-as-cuda", "replace-model", "planted-ray", "contradictory-ray")
SELECTED_FUNCTIONS = (
    "freeze_proof_expectations", "submission_units", "worker_command",
    "collected_result", "proof_completion_matches",
)


class ReviewError(ValueError):
    """Fail-loud review rejection with the first coordinate."""


def require(condition, coordinate, cause):
    if not condition:
        raise ReviewError(f"{coordinate}: {cause}")


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def parse(data):
    def pairs(items):
        value = {}
        for key, item in items:
            require(key not in value, "JSON", f"duplicate key {key}")
            value[key] = item
        return value

    def constant(value):
        raise ReviewError(f"JSON: nonfinite {value}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encode(value))


def read(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), path, "regular file required")
    return path.read_bytes()


def external(path):
    path = Path(path)
    require(path.is_absolute(), path, "absolute artifact path required")
    require(not path.is_symlink(), path, "symlink rejected")
    path = path.resolve()
    require(path.is_relative_to(ARTIFACT_ROOT), path, "outside shared 3FS tmp")
    return path


def at(root, name):
    require(isinstance(name, str) and name and not Path(name).is_absolute()
            and ".." not in Path(name).parts and Path(name).as_posix() == name,
            "path", "unsafe relative path")
    root = Path(root).resolve()
    path = root / name
    require(path.resolve().is_relative_to(root), path, "path escape")
    require(not any(p.is_symlink() for p in (path, *path.parents)), path, "symlink component")
    return path


def descriptor(root, path):
    data = read(path)
    return {"path": Path(path).relative_to(root).as_posix(), "sha256": sha(data), "bytes": len(data)}


def bound(root, item):
    require(isinstance(item, dict) and set(item) == {"path", "sha256", "bytes"},
            "descriptor", "exact descriptor required")
    path = at(root, item["path"])
    data = read(path)
    require(type(item["bytes"]) is int and item["bytes"] == len(data)
            and item["sha256"] == sha(data), path, "pinned bytes changed")
    return path


def inventory(root):
    files = {}
    for path in sorted(Path(root).rglob("*")):
        require(not path.is_symlink(), path, "inventory symlink")
        if path.is_file():
            files[path.relative_to(root).as_posix()] = descriptor(root, path)
    require(files, root, "empty inventory")
    return files


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    require(result.returncode == 0, "git", f"rc={result.returncode}: {result.stderr.decode()[:500]}")
    return result.stdout


def commit_id(root, commit):
    require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
            "commit", "explicit full commit hash required")
    require(git(root, "rev-parse", f"{commit}^{{commit}}").decode().strip() == commit,
            "commit", "commit identity mismatch")
    return commit


def pinned_producer(root, commit):
    root = Path(root).resolve()
    commit_id(root, commit)
    require(git(root, "rev-parse", "HEAD").decode().strip() == commit
            and not git(root, "status", "--porcelain"), root, "producer must be clean at pinned HEAD")
    names = git(root, "ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")
    required = {PRODUCER, "tests/test_gpu_release_proof.py", "docs/research/gpu-release-proof.md"}
    selected = sorted(name for name in names if name in required
                      or (name.startswith("src/") and name.endswith(".py")))
    require(required <= set(selected) and any(n.startswith("src/openopt/") for n in selected),
            root, "reviewed producer sources missing")
    hashes = {}
    for name in selected:
        data = git(root, "show", f"{commit}:{name}")
        require(read(at(root, name)) == data, name, "current bytes differ from pinned Git object")
        hashes[name] = sha(data)
    return {"commit": commit, "tree": git(root, "rev-parse", f"{commit}^{{tree}}").decode().strip(),
            "clean": True, "sha256": hashes}


def dispatch_identity(repo, commit):
    commit_id(repo, commit)
    names = (DISPATCH, "tests/test_release_proof_dispatch.py", "docs/qp-design.md")
    sources = {name: git(repo, "show", f"{commit}:{name}") for name in names}
    return {"commit": commit, "tree": git(repo, "rev-parse", f"{commit}^{{tree}}").decode().strip(),
            "sha256": {name: sha(data) for name, data in sources.items()}}, sources


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def dispatch_functions(data, proof):
    """Execute only reviewed pure functions, never module/submission entrypoints."""
    tree = ast.parse(data, filename=DISPATCH)
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in SELECTED_FUNCTIONS]
    require({node.name for node in selected} == set(SELECTED_FUNCTIONS),
            DISPATCH, "missing or duplicate selected contract functions")
    require(len(selected) == len(SELECTED_FUNCTIONS), DISPATCH, "duplicate contract functions")
    namespace = {
        "__name__": "_reviewed_dispatch_contract", "Path": Path, "json": json,
        "hashlib": hashlib, "argparse": argparse, "SimpleNamespace": SimpleNamespace,
        "ROOT": proof.ROOT, "proof_module": lambda: proof, "proof_cases": lambda: proof.CASES,
    }
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), DISPATCH, "exec", dont_inherit=True), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in SELECTED_FUNCTIONS})


@contextmanager
def no_execution(proof):
    """Forbid solver, factory, CUDA and subprocess entrypoints during replay."""
    import torch
    import openopt.qp as qp
    import openopt.operators as operators
    import openopt.scaling as scaling

    calls = []

    def forbidden(label):
        def reject(*args, **kwargs):
            calls.append(label)
            raise AssertionError(f"forbidden replay execution: {label}")
        return reject

    targets = (
        (qp, ("prepare_qp", "solve_qp")),
        (qp.PreparedQP, ("solve",)),
        (operators, ("prepare_operator_qp", "conjugate_gradient")),
        (operators.PreparedOperatorQP, ("solve",)),
        (scaling, ("solve_scaled_qp",)),
        (proof, ("make_problem", "execute_case", "run_cases", "hardware", "freeze_sources")),
        (torch.cuda, ("synchronize", "get_device_name", "get_device_properties", "device_count")),
        (subprocess, ("run", "Popen")),
    )
    with ExitStack() as stack:
        for module, names in targets:
            for name in names:
                stack.enter_context(patch.object(module, name, forbidden(name)))
        stack.enter_context(patch.object(torch.cuda, "is_available", forbidden("cuda.is_available")))
        yield calls
    require(not calls, "replay", f"attempted forbidden operations {calls}")


def rejected(function, error_type, reason):
    try:
        function()
    except error_type as error:
        require(reason in str(error), "mutation", f"wrong first cause: {error}")
        return {"verdict": "REJECTED", "exception": type(error).__name__, "reason": str(error)}
    raise ReviewError(f"mutation: accepted forbidden evidence; expected {reason}")


def summary(proof, root, saved):
    import openopt.certificates as certificates
    model = {k: proof.load_tensor(root, v, k) for k, v in saved["model"].items()}
    vectors = {k: proof.load_tensor(root, v, k) for k, v in saved["vectors"].items()}
    actual = proof.original_checks(model, vectors, saved["status"], saved["atol"], saved["rtol"])
    require(actual["passed"], saved["case"], "mutation fixture is not numerically valid")
    saved["numpy_original"] = actual
    host = SimpleNamespace(**model)
    if saved["status"] == "unbounded":
        independent = certificates.check_unbounded_direction(
            host, vectors["unbounded_direction"], feasible_x=vectors["feasible_base"],
            atol=saved["atol"], rtol=saved["rtol"]).to_dict()
    else:
        independent = certificates.check_qp_candidate(
            host, vectors["x"], vectors["y"], atol=saved["atol"], rtol=saved["rtol"]).to_dict()
        saved["objective"] = actual["metrics"]["primal_objective"]
    require(independent["passed"], "fixture", "independent original check did not pass")
    saved["independent_original"] = independent
    return actual


def refresh(proof, root, saved, expected, *, gpu=False):
    """Reissue every modified sidecar/result/log pin before sealing."""
    sidecar = root / (saved["case"] + ".json")
    sidecar.write_bytes(proof.encode(saved))
    path = root / "result.json"
    result = parse(read(path))
    result["requested_cases"] = [saved["case"]]
    result["cases"] = {saved["case"]: proof.descriptor(root, sidecar)}
    result["expectations"] = expected
    if gpu:
        result.update(device="cuda:0", scope=proof.GPU_SCOPE, verdict=proof.NUMERICAL_PASS,
                      numerical_verdict="PASS", execution_verdict="UNVERIFIED")
        uuid = "GPU-00000000-0000-0000-0000-000000000001"
        result["hardware"].update(
            requested_device="cuda:0", gpu_count=1, gpu="NVIDIA RTX PRO 4500 Blackwell",
            gpu_uuid=uuid, node="fabricated-review-node", pod="fabricated-review-pod",
            cuda="12.8", cpu_quota="200000 100000", cpu_scope="Pod cgroup v2",
            cuda_capability=[12, 0],
        )
        samples = [
            {"at": result["started_at"], "cpu": {"usage_usec": index}, "returncode": 0,
             "stdout": f"{uuid}, NVIDIA RTX PRO 4500 Blackwell, 0, 0, 24576, 580.0", "stderr": ""}
            for index in (1, 2)
        ]
        (root / "telemetry.json").write_bytes(proof.encode(samples))
        result["telemetry"] = proof.descriptor(root, root / "telemetry.json")
    path.write_bytes(proof.encode(result))
    digest = sha(read(path))
    marker = (f"OPENOPT_GPU_PROOF verdict={result['verdict']} at={result['finished_at']} "
              f"result={path} result_sha256={digest}")
    (root / "execution.log").write_text(marker + "\nreturncode=0\n")
    return path


def build_mutations(proof, output, original, expected):
    import numpy as np
    import torch
    records = {}
    for mutation in MUTATIONS:
        case = "gap_lp" if mutation in ("cpu-as-cuda", "replace-model") else "unbounded"
        root = output / "mutations" / mutation / "cuda-proof-evidence"
        shutil.copytree(original.parent, root)
        saved = parse(read(root / (case + ".json")))
        before_model = saved["model_sha256"]
        if mutation == "cpu-as-cuda":
            saved["device"] = saved["config"]["device"] = "cuda:0"
            for group in ("resident_model", "vectors", "observed"):
                for item in saved[group].values():
                    item["device"] = "cuda:0"
        elif mutation == "replace-model":
            values = {"P": [[1.]], "q": [0.], "A": [[1.]], "l": [-1.], "u": [1.]}
            saved["model"] = {
                key: proof.save_tensor(root, "replacement-" + key,
                                       torch.tensor(value, dtype=torch.float64))
                for key, value in values.items()
            }
            saved["resident_model"] = deepcopy(saved["model"])
            saved["model_sha256"] = proof.model_hash(saved["model"])
            saved["vectors"] = {
                key: proof.save_tensor(root, "replacement-" + key, torch.zeros(1, dtype=torch.float64))
                for key in ("x", "y")
            }
            saved["iterations"] = 1
            summary(proof, root, saved)
            require(before_model != saved["model_sha256"], mutation, "replacement did not change model")
        elif mutation == "planted-ray":
            for key, value in (("x", [0., 2.]), ("feasible_base", [0., 2.]),
                               ("unbounded_direction", [0., 1.])):
                saved["vectors"][key] = proof.save_tensor(
                    root, "planted-" + key, torch.tensor(value, dtype=torch.float64))
            trace = saved["extensions"]["checkpoint"]
            for key, value in (("current_x", [0., 2.]), ("previous_x", [0., 0.]), ("dx", [0., 2.])):
                trace["tensors"][key] = proof.save_tensor(
                    root, "planted-" + key, torch.tensor(value, dtype=torch.float64))
            trace["iteration"] = saved["iterations"] = 1
            summary(proof, root, saved)
            from openopt.certificates import check_unbounded_direction
            host = SimpleNamespace(**{
                k: proof.load_tensor(root, v, k) for k, v in saved["model"].items()})
            trace["unbounded_evidence"] = check_unbounded_direction(
                host, np.array([0., 2.]), feasible_x=np.array([0., 2.]),
                atol=saved["atol"], rtol=saved["rtol"]).to_dict()
        else:
            opposite = saved["extensions"]["checkpoint"]["infeasible_evidence"]
            require(opposite["passed"] is False, mutation, "opposite direction already passes")
            opposite["passed"] = True
            opposite["checks"] = {key: True for key in opposite["checks"]}
        result = refresh(proof, root, saved, expected, gpu=mutation == "cpu-as-cuda")
        records[mutation] = {
            "result": descriptor(output, result), "case": case,
            "original_model_sha256": before_model, "mutated_model_sha256": saved["model_sha256"],
        }
    return records


def evaluate_mutations(proof, output, records, kwargs, *, create):
    require(set(records) == set(MUTATIONS), "mutations", "exact four denominator required")
    outcomes = {}
    reasons = {
        "replace-model": "original model differs from caller",
        "planted-ray": "outside frozen check interval",
        "contradictory-ray": "full direction summary",
    }
    for name in MUTATIONS:
        result = bound(output, records[name]["result"])
        digest = sha(read(result))
        if name == "cpu-as-cuda":
            missing = rejected(
                lambda: proof.seal(result, digest, proof.ROOT), proof.ProofError,
                "independent caller pre-dispatch")
            bundle_path = output / "bundles" / (name + ".json")
            if create:
                write(bundle_path, proof.seal(result, digest, proof.ROOT, **kwargs))
            numerical = proof.replay(bundle_path, sha(read(bundle_path)), proof.ROOT, **kwargs)
            require(numerical["verdict"] == proof.NUMERICAL_PASS != "PASS"
                    and numerical["numerical_verdict"] == "PASS"
                    and numerical["execution_verdict"] == "UNVERIFIED",
                    name, "bare numerical replay promoted execution")
            auditor = load_module(ROOT / AUDITOR, "_contract_origin_audit")
            missing_origin = rejected(
                lambda: auditor.control_audit(
                    output / "missing-origin" / "collection.json", "0" * 64,
                    proof.ROOT, kwargs["expectations_sha256"]),
                auditor.AuditError, "regular non-symlink file",
            )
            outcomes[name] = {
                "verdict": "EXECUTION_NOT_QUALIFIED", "missing_expectations": missing,
                "missing_origin": missing_origin,
                "numerical_verdict": numerical["verdict"], "execution_verdict": "UNVERIFIED",
            }
        else:
            outcomes[name] = rejected(
                lambda: proof.seal(result, digest, proof.ROOT, allow_cpu_smoke=True, **kwargs),
                proof.ProofError, reasons[name],
            )
    return outcomes


def build_dispatch_fixtures(output, gpu_result, expected):
    result = parse(read(gpu_result))
    collectors = []
    changes = (
        ("semantic", {}, True), ("legacy-pass", {"verdict": "PASS"}, False),
        ("cpu-smoke", {"verdict": "CPU_SMOKE_PASS", "device": "cpu"}, False),
        ("overclaim", {"execution_verdict": "PASS"}, False),
        ("bad-numerical", {"numerical_verdict": "FAIL"}, False),
        ("bad-device", {"device": "cpu"}, False),
        ("nonzero-exit", {"returncode": 1}, False),
        ("wrong-case", {"requested_cases": ["gap_spd"]}, False),
        ("no-expectations", {"expectations": None}, False),
    )
    for name, change, accepted in changes:
        value = deepcopy(result)
        value.update(change)
        path = output / "dispatch" / name / "cuda-proof-fixture" / "result.json"
        write(path, value)
        collectors.append({"name": name, "result": descriptor(output, path), "accepted": accepted})
    legacy = []
    for name, verdict, accepted in (("legacy", "PASS", True),
                                    ("semantic", "SEMANTIC_NUMERICAL_PASS", False)):
        path = output / "dispatch" / ("non-proof-" + name) / "result.json"
        write(path, {"verdict": verdict})
        legacy.append({"name": name, "result": descriptor(output, path), "accepted": accepted})
    path = bound(output, collectors[0]["result"])
    value = parse(read(path))
    marker = (f"OPENOPT_GPU_PROOF verdict=SEMANTIC_NUMERICAL_PASS at={value['finished_at']} "
              f"result={path} result_sha256={sha(read(path))}")
    logs = [
        {"name": "exact-last", "log": "worker progress\n" + marker + "\n", "accepted": True},
        {"name": "legacy-marker", "log": marker.replace("SEMANTIC_NUMERICAL_PASS", "PASS"),
         "accepted": False},
        {"name": "bad-pin", "log": marker.replace(sha(read(path)), "0" * 64), "accepted": False},
        {"name": "duplicate", "log": marker + "\n" + marker, "accepted": False},
        {"name": "not-last", "log": marker + "\nafter-completion", "accepted": False},
        {"name": "empty", "log": "", "accepted": False},
    ]
    value = {"collectors": collectors, "logs": logs, "legacy": legacy, "expectations": expected}
    write(output / "dispatch" / "fixtures.json", value)
    return value


def evaluate_dispatch(dispatch, proof, output, fixtures):
    rows = fixtures["collectors"]
    require(len(rows) == 9 and len(fixtures["logs"]) == 6 and len(fixtures["legacy"]) == 2,
            "dispatch", "fixture denominator")
    checks = []
    for row in rows:
        path = bound(output, row["result"])
        record = {"stage": "release-proof", "expectations": fixtures["expectations"]}
        item = {"output": str(path.parent.parent), "proof_case": "gap_lp"}
        actual = dispatch.collected_result(record, item)
        passed = (actual == path) if row["accepted"] else actual is None
        checks.append({"name": "collector/" + row["name"], "passed": passed})
    for row in fixtures["legacy"]:
        path = bound(output, row["result"])
        actual = dispatch.collected_result({"stage": "release-regression"}, {"output": str(path.parent)})
        passed = (actual == path) if row["accepted"] else actual is None
        checks.append({"name": "non-proof/" + row["name"], "passed": passed})
    path = bound(output, rows[0]["result"])
    for row in fixtures["logs"]:
        actual = dispatch.proof_completion_matches(row["log"], path, sha(read(path)))
        checks.append({"name": "completion/" + row["name"], "passed": actual is row["accepted"]})
    for cases in ([CASES[0]], list(CASES[:2]), list(CASES[:3])):
        args = SimpleNamespace(stage="release-proof", proof_case=cases, target=[], expression="")
        checks.append({"name": f"denominator/{len(cases)}",
                       "passed": dispatch.submission_units(args) == [(0, case) for case in cases]})
    for index, cases in enumerate(([], [CASES[0]] * 2, list(CASES[:4]), ["unknown"])):
        args = SimpleNamespace(stage="release-proof", proof_case=cases, target=[], expression="")
        try:
            dispatch.submission_units(args)
        except ValueError:
            passed = True
        else:
            passed = False
        checks.append({"name": f"invalid-denominator/{index}", "passed": passed})
    args = SimpleNamespace(stage="release-proof")
    expected = fixtures["expectations"]
    command = dispatch.worker_command(args, proof.ROOT, path.parent.parent, 0, "gap_lp", expected)
    exact = ["python3", str(proof.ROOT / PRODUCER), "run", "--artifact-dir", str(path.parent.parent),
             "--device", "cuda:0", "--case", "gap_lp", "--expectations", expected["path"],
             "--expectations-sha256", expected["sha256"]]
    checks.append({"name": "exact-command", "passed": command == exact})
    return {"verdict": "PASS" if all(row["passed"] for row in checks) else "FAIL",
            "checks": checks, "failed": [row["name"] for row in checks if not row["passed"]]}


def cpu_replay(proof, output, result, kwargs, *, create):
    import numpy as np
    bundle_path = output / "bundles" / "original-cpu.json"
    if create:
        write(bundle_path, proof.seal(result, sha(read(result)), proof.ROOT,
                                      allow_cpu_smoke=True, **kwargs))
    outcome = proof.replay(bundle_path, sha(read(bundle_path)), proof.ROOT,
                           allow_cpu_smoke=True, **kwargs)
    require(outcome["case_count"] == 9 and {row["case"] for row in outcome["cases"]} == set(CASES)
            and all(row["passed"] for row in outcome["cases"])
            and outcome["verdict"] == "CPU_SMOKE_REPLAY_PASS"
            and outcome["execution_verdict"] == "UNVERIFIED",
            "CPU nine", "incomplete numerical denominator or GPU claim")
    saved = parse(read(result.parent / "unbounded.json"))
    base = proof.load_tensor(result.parent, saved["vectors"]["feasible_base"], "base")
    x = proof.load_tensor(result.parent, saved["vectors"]["x"], "x")
    require(np.array_equal(base, x) and base[1] > 2, "CPU nine", "base is not actual terminal iterate")
    return {"verdict": outcome["verdict"], "case_count": 9, "execution_verdict": "UNVERIFIED",
            "cases": outcome["cases"], "actual_unbounded_base": base.tolist()}


def loaded_module_probe(proof, result, kwargs):
    """A negative control for cached foreign policy and direction modules."""
    import openopt
    import openopt.qp as qp
    import openopt.certificates as certificates

    original_qp = qp.QPConfig
    original_infeasible = certificates.check_primal_infeasibility
    original_unbounded = certificates.check_unbounded_direction
    calls = {"foreign_policy": 0, "foreign_infeasible": 0, "foreign_unbounded": 0}

    def foreign_policy(*args, **options):
        calls["foreign_policy"] += 1
        return original_qp(*args, **options)

    def foreign_infeasible(*args, **options):
        calls["foreign_infeasible"] += 1
        return original_infeasible(*args, **options)

    def foreign_unbounded(*args, **options):
        calls["foreign_unbounded"] += 1
        return original_unbounded(*args, **options)

    foreign_qp = ModuleType("openopt.qp")
    foreign_qp.__dict__.update(qp.__dict__)
    foreign_qp.__file__ = str(result.parent / "foreign-loaded-qp.py")
    foreign_qp.QPConfig = foreign_policy
    foreign_certificates = ModuleType("openopt.certificates")
    foreign_certificates.__dict__.update(certificates.__dict__)
    foreign_certificates.__file__ = str(result.parent / "foreign-loaded-certificates.py")
    foreign_certificates.check_primal_infeasibility = foreign_infeasible
    foreign_certificates.check_unbounded_direction = foreign_unbounded
    with patch.dict(sys.modules, {"openopt.qp": foreign_qp,
                                 "openopt.certificates": foreign_certificates}), \
            patch.object(openopt, "qp", foreign_qp), \
            patch.object(openopt, "certificates", foreign_certificates):
        try:
            numerical = proof.validate_result(
                result, sha(read(result)), proof.ROOT, allow_cpu_smoke=True, **kwargs)
        except proof.ProofError as error:
            return {"verdict": "REJECTED", "reason": str(error), "calls": calls}
    require(numerical["case_count"] == 9 and all(value > 0 for value in calls.values()),
            "loaded-module fixture", "foreign modules were not actually consulted")
    return {"verdict": "ACCEPTED_FOREIGN_MODULES", "calls": calls,
            "numerical_verdict": numerical["verdict"], "execution_verdict": numerical["execution_verdict"],
            "foreign_paths": [foreign_qp.__file__, foreign_certificates.__file__]}


def run(producer_root, producer_commit, repo, dispatch_commit, artifact_dir):
    parent = external(artifact_dir)
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="review-", dir=parent))
    report = {"schema_version": 1, "started_at": now(), "verdict": "RUNNING",
              "scope": "CPU numerical and producer/dispatch contract review; not CUDA execution",
              "reviewer_sha256": sha(read(Path(__file__))),
              "auditor_sha256": sha(read(ROOT / AUDITOR)), "execution_verdict": "UNVERIFIED"}
    try:
        import torch
        require(os.environ.get("CUDA_VISIBLE_DEVICES") == "-1" and not torch.cuda.is_available(),
                "CPU scope", "explicit CUDA_VISIBLE_DEVICES=-1 and unavailable CUDA required")
        producer_identity = pinned_producer(producer_root, producer_commit)
        coordinator_identity, dispatch_sources = dispatch_identity(repo, dispatch_commit)
        report.update(producer=producer_identity, dispatch=coordinator_identity)
        proof = load_module(Path(producer_root) / PRODUCER, "_contract_producer")
        require(tuple(proof.CASES) == CASES and proof.NUMERICAL_PASS == "SEMANTIC_NUMERICAL_PASS",
                "producer", "new semantic verdict and exact nine cases required")
        dispatch = dispatch_functions(dispatch_sources[DISPATCH], proof)
        auditor_copy = output / "reviewed-auditor" / AUDITOR
        auditor_copy.parent.mkdir(parents=True, exist_ok=True)
        with auditor_copy.open("xb") as stream:
            stream.write(read(ROOT / AUDITOR))
        for name in producer_identity["sha256"]:
            path = output / "reviewed-producer" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(read(Path(producer_root) / name))
        for name, data in dispatch_sources.items():
            path = output / "reviewed-dispatch" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
        expected_root = output / "expected"
        expected_root.mkdir()
        with (output / "cpu-production.log").open("x") as log, redirect_stdout(log), redirect_stderr(log):
            expected = dispatch.freeze_proof_expectations(expected_root, proof.ROOT, list(CASES))
            kwargs = {"expectations": Path(expected["path"]), "expectations_sha256": expected["sha256"]}
            result = proof.run_cases(output / "cpu-evidence", list(CASES), "cpu",
                                     cpu_smoke=True, **kwargs)
        expected_before = inventory(expected_root)
        original_before = inventory(result.parent)
        mutations = build_mutations(proof, output, result, expected)
        fixtures = build_dispatch_fixtures(
            output, bound(output, mutations["cpu-as-cuda"]["result"]), expected)
        with no_execution(proof) as calls:
            cpu = cpu_replay(proof, output, result, kwargs, create=True)
            mutated = evaluate_mutations(proof, output, mutations, kwargs, create=True)
            contract = evaluate_dispatch(dispatch, proof, output, fixtures)
            loaded = loaded_module_probe(proof, result, kwargs)
        require(expected_before == inventory(expected_root), "expectations", "review modified frozen models")
        require(original_before == inventory(result.parent), "original evidence", "review changed CPU evidence")
        require(producer_identity == pinned_producer(producer_root, producer_commit),
                "source", "producer changed during review")
        require(coordinator_identity == dispatch_identity(repo, dispatch_commit)[0],
                "source", "dispatch Git object changed")
        require(report["auditor_sha256"] == sha(read(ROOT / AUDITOR))
                == sha(read(auditor_copy)), "source", "read-only auditor changed")
        report.update(cpu=cpu, mutations=mutated, mutation_inputs=mutations, dispatch_contract=contract,
                      loaded_module_negative=loaded,
                      expectations=expected, expectations_files=expected_before,
                      original_result=descriptor(output, result), original_files=original_before,
                      replay_forbidden_calls=calls,
                      verdict="CONTRACT_REVIEW_PASS" if contract["verdict"] == "PASS"
                      and loaded["verdict"] == "REJECTED" else "FAIL",
                      numerical_verdict="PASS")
        if contract["verdict"] != "PASS":
            report["error"] = f"dispatch contract incompatibility: {contract['failed']}"
        elif loaded["verdict"] != "REJECTED":
            report["error"] = "P1 loaded foreign policy/direction modules accepted; GPU proof stop-gate"
    except Exception as error:
        report.update(verdict="FAIL", error=f"{type(error).__name__}: {error}")
        with (output / "failure.log").open("x") as log:
            log.write(traceback.format_exc())
    report["finished_at"] = now()
    line = (f"GPU_PROOF_CONTRACT_REVIEW verdict={report['verdict']} at={report['finished_at']} "
            f"report={output / 'review.json'}")
    with (output / "execution.log").open("x") as log:
        log.write(line + f"\nreturncode={0 if report['verdict'] == 'CONTRACT_REVIEW_PASS' else 1}\n")
    write(output / "review.json", report)
    files = inventory(output)
    bundle = {"schema_version": 1, "evidence_root": str(output), "files": files,
              "report": descriptor(output, output / "review.json")}
    bundle_path = parent / (output.name + "-bundle.json")
    write(bundle_path, bundle)
    print(f"{line} bundle={bundle_path} sha256={sha(read(bundle_path))}", flush=True)
    if report["verdict"] == "FAIL":
        print(report["error"], flush=True)
    return bundle_path, report


def replay(bundle_path, digest, producer_root, producer_commit, repo, dispatch_commit):
    data = read(external(bundle_path))
    require(re.fullmatch(r"[0-9a-f]{64}", digest or "") and sha(data) == digest,
            "bundle", "independent caller pin mismatch")
    bundle = parse(data)
    require(set(bundle) == {"schema_version", "evidence_root", "files", "report"}
            and type(bundle["schema_version"]) is int and bundle["schema_version"] == 1,
            "bundle", "exact schema required")
    output = external(bundle["evidence_root"])
    require(inventory(output) == bundle["files"], "inventory", "added, removed or changed evidence")
    report = parse(read(bound(output, bundle["report"])))
    require(report["verdict"] in ("CONTRACT_REVIEW_PASS", "FAIL") and "cpu" in report
            and report["execution_verdict"] == "UNVERIFIED"
            and report["reviewer_sha256"] == sha(read(Path(__file__)))
            and report["auditor_sha256"] == sha(read(ROOT / AUDITOR))
            == sha(read(output / "reviewed-auditor" / AUDITOR)),
            "review", "failed report or changed reviewer")
    require(report["producer"] == pinned_producer(producer_root, producer_commit),
            "producer", "source identity mismatch")
    identity, sources = dispatch_identity(repo, dispatch_commit)
    require(report["dispatch"] == identity, "dispatch", "source identity mismatch")
    for name, expected_hash in report["producer"]["sha256"].items():
        require(sha(read(at(output / "reviewed-producer", name))) == expected_hash,
                name, "producer snapshot mismatch")
    for name, expected_hash in identity["sha256"].items():
        require(sha(read(at(output / "reviewed-dispatch", name))) == expected_hash,
                name, "dispatch snapshot mismatch")
    proof = load_module(Path(producer_root) / PRODUCER, "_contract_replay_producer")
    dispatch = dispatch_functions(sources[DISPATCH], proof)
    expected = report["expectations"]
    require(Path(expected["path"]).parent == output / "expected"
            and len(read(expected["path"])) == expected["bytes"]
            and sha(read(expected["path"])) == expected["sha256"]
            and inventory(output / "expected") == report["expectations_files"],
            "expectations", "independently frozen model bytes changed")
    kwargs = {"expectations": Path(expected["path"]), "expectations_sha256": expected["sha256"]}
    with no_execution(proof) as calls:
        cpu = cpu_replay(proof, output, bound(output, report["original_result"]), kwargs, create=False)
        mutated = evaluate_mutations(proof, output, report["mutation_inputs"], kwargs, create=False)
        fixtures = parse(read(output / "dispatch" / "fixtures.json"))
        contract = evaluate_dispatch(dispatch, proof, output, fixtures)
        loaded = loaded_module_probe(proof, bound(output, report["original_result"]), kwargs)
    require(cpu == report["cpu"] and mutated == report["mutations"]
            and contract == report["dispatch_contract"] and loaded == report["loaded_module_negative"],
            "replay", "machine outcomes changed")
    require(inventory(output) == bundle["files"], "inventory", "replay changed evidence")
    require(report["producer"] == pinned_producer(producer_root, producer_commit),
            "source", "producer changed during replay")
    passed = contract["verdict"] == "PASS" and loaded["verdict"] == "REJECTED"
    require((report["verdict"] == "CONTRACT_REVIEW_PASS") is passed,
            "review", "saved verdict contradicts independently recomputed gates")
    return {"verdict": "CONTRACT_REPLAY_PASS" if passed else "REPRODUCED_REVIEW_FAIL",
            "reviewed_gate_verdict": "PASS" if passed else "FAIL",
            "case_count": 9, "mutation_count": 4, "loaded_module_negative": loaded,
            "dispatch_check_count": len(contract["checks"]), "replay_forbidden_calls": calls,
            "execution_verdict": "UNVERIFIED", "at": now()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("run", "replay"))
    parser.add_argument("--producer-root", type=Path, required=True)
    parser.add_argument("--producer-commit", required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--dispatch-commit", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--sha256")
    args = parser.parse_args(argv)
    if args.action == "run":
        _, report = run(args.producer_root, args.producer_commit, args.repo_root,
                        args.dispatch_commit, args.artifact_dir)
        return 0 if report["verdict"] == "CONTRACT_REVIEW_PASS" else 1
    parent = external(args.artifact_dir)
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="replay-", dir=parent))
    try:
        require(args.bundle and args.sha256, "replay", "bundle and independent caller pin required")
        result = replay(args.bundle, args.sha256, args.producer_root, args.producer_commit,
                        args.repo_root, args.dispatch_commit)
        code = 0 if result["verdict"] == "CONTRACT_REPLAY_PASS" else 1
    except Exception as error:
        result = {"verdict": "FAIL", "execution_verdict": "UNVERIFIED", "at": now(),
                  "error": f"{type(error).__name__}: {error}"}
        with (output / "failure.log").open("x") as log:
            log.write(traceback.format_exc())
        code = 1
    write(output / "acceptance.json", result)
    line = (f"GPU_PROOF_CONTRACT_REPLAY verdict={result['verdict']} at={result['at']} "
            f"evidence={output / 'acceptance.json'}")
    with (output / "execution.log").open("x") as log:
        log.write(line + f"\nreturncode={code}\n")
    print(line, flush=True)
    if code and "error" in result:
        print(result["error"], flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
