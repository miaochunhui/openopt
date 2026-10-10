"""Caller-pinned HSD loaded-repair review; all evidence stays outside the tree.

The historical 45 test functions execute from their byte-identical Git object.
Only their counting fixture is adapted to read-only Python profiling. Neither
the numerical oracle, assertions nor production aliases/guards are relaxed.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from fractions import Fraction
import argparse
import ast
import hashlib
import json
import marshal
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import ModuleType
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import torch

import openopt.conic as conic
import openopt.hsd_lp as hsd
import openopt.native_conic as native


ROOT = Path(__file__).resolve().parents[1]
OWNER = "6cefdcf0b1f2a05c76c26dceafbf10430bc67169"
HISTORICAL_REVIEW = "9c6b69bda8e86ad8cd3b198ebb72afa155186bff"
HISTORICAL_TEST = "tests/test_hsd_lp_independent.py"
HISTORICAL_TEST_SHA = "74a7996bcb60d96b19cc6935736b18aaa62341f1d3f48c53384b2ddf5726aedd"
TEST = "tests/test_hsd_loaded_repair_independent.py"
WRITE_PATHS = (TEST, "docs/research/hsd-loaded-repair-independent.md")
OWNER_PINS = {
    "src/openopt/hsd_lp.py": "5d6fd0c96adcf773a379838e7dc14fe3f3d9fcbe3404e5c2516133c33a9b313d",
    "src/openopt/conic.py": "c56dff1923c61af55f992023f64051db5a2e3b3b5bbeec5593731b8b569032cf",
    "src/openopt/native_conic.py": "a98d77c55766f12c0e339da54258554b3b1b6ab897b7ae5a168ca77bf8fd95c7",
    "tests/test_hsd_lp.py": "a8774fc77a1df7a6611bed1c4c7b09efe6a9d283982f4cee83d64f2106b53edd",
    "docs/research/hsd-lp.md": "d7ddfe9f88f4923a26290cbb3568d63eb86000c33bc656357506f232a2ba1c36",
}
EXTERNAL = Path("/hf3fs-jd/data/tmp/openopt-release-20261010").resolve()
HISTORY = {
    "original_false_accept": (
        EXTERNAL / "hsd-independent/clean-full-v1/review-j3id7xaj/receipt.json",
        "438aa68a33aa98d18b81b2c6e2ae7e4a32723ced2c1c1c54eb6c7f0dcabd55c8",
    ),
    "unchanged_observer_incompatible": (
        EXTERNAL / "hsd-loaded-repair/unchanged45-clean-v1/unchanged-ma249lmb/receipt.json",
        "95f193844f1824c62236e0ebc37bb6fad978492379c9bd0cc3323691934a51df",
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    return digest(Path(path).read_bytes())


def git_bytes(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                          check=True, timeout=15).stdout


def json_write(path, value):
    serialized = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with Path(path).open("x") as output:
        output.write(serialized)
    assert Path(path).read_text() == serialized, "receipt byte readback"
    assert json.loads(Path(path).read_text()) == json.loads(serialized), "receipt JSON readback"


def owner_binding():
    assert hsd.__file__ == str(ROOT / "src/openopt/hsd_lp.py"), "loaded HSD root"
    for name, wanted in OWNER_PINS.items():
        assert sha(ROOT / name) == wanted, "caller source pin: " + name
        assert digest(git_bytes("show", OWNER + ":" + name)) == wanted, "owner Git pin"
    assert conic.__file__ == str(ROOT / "src/openopt/conic.py")
    assert native.__file__ == str(ROOT / "src/openopt/native_conic.py")
    return dict(OWNER_PINS)


ORIGINAL_BYTES = git_bytes("show", HISTORICAL_REVIEW + ":" + HISTORICAL_TEST)
assert digest(ORIGINAL_BYTES) == HISTORICAL_TEST_SHA, "historical test byte pin"
legacy = ModuleType("tests._hsd_original45_pinned")
legacy.__file__ = str(Path(__file__).resolve())
sys.modules[legacy.__name__] = legacy
exec(compile(ORIGINAL_BYTES, legacy.__file__, "exec"), vars(legacy))
# These are explicit harness context bindings, not edits to historical bytes.
legacy.ROOT, legacy.OWNER, legacy.WRITE_PATHS = ROOT, OWNER, WRITE_PATHS
assert legacy.hsd is hsd
ORIGINAL_FUNCTION_PINS = {}
for declaration in ast.parse(ORIGINAL_BYTES).body:
    if isinstance(declaration, ast.FunctionDef) and declaration.name.startswith("test_"):
        start = min([declaration.lineno] + [d.lineno for d in declaration.decorator_list])
        fragment = b"".join(ORIGINAL_BYTES.splitlines(keepends=True)[start - 1:declaration.end_lineno])
        ORIGINAL_FUNCTION_PINS[declaration.name] = digest(fragment)
        globals()[declaration.name] = getattr(legacy, declaration.name)


@pytest.fixture(scope="module")
def actual_receipt(tmp_path_factory):
    """Count actual calls without altering any production executable or alias."""
    owner_binding()
    destination = tmp_path_factory.mktemp("hsd-profile-actual")
    counts = {"native_solves": 0, "factorizations": 0, "lu_solves": 0,
              "projections": 0, "cpu_fallbacks": 0}
    projector = hsd.project_cones
    project_code = projector.__wrapped__.__code__
    solve_code = hsd.PreparedHSDLP.solve.__wrapped__.__code__
    factor, linear = torch.linalg.lu_factor_ex, torch.linalg.lu_solve
    previous = sys.getprofile()
    assert previous is None, "profile collision"
    before = legacy.source_hashes()
    rows = []

    def observe(frame, event, arg):
        if event == "call":
            if frame.f_code is project_code:
                value = frame.f_locals["value"]
                assert value.device == torch.device("cpu") and value.dtype == torch.float64
                counts["projections"] += 1
            elif frame.f_code is solve_code:
                counts["native_solves"] += 1
        elif event == "c_call":
            if arg is factor:
                tensor = frame.f_locals["self"]._system
                assert tensor.device == torch.device("cpu") and tensor.dtype == torch.float64
                counts["factorizations"] += 1
            elif arg is linear:
                prepared = frame.f_locals["self"]
                assert all(v.device == torch.device("cpu") for v in
                           (prepared._lu, prepared._pivots, frame.f_locals["old_w"]))
                counts["lu_solves"] += 1

    sys.setprofile(observe)
    try:
        for identity, (m, status, optimum, base, budget) in legacy.cases().items():
            p = legacy.problem(m)
            original = [getattr(p, k).tobytes() for k in ("P", "q", "A", "b")]
            result = hsd.solve_hsd_lp(
                p, hsd.HSDLPConfig(enabled=True, max_iterations=budget), feasible_base=base,
            )
            assert original == [getattr(p, k).tobytes() for k in ("P", "q", "A", "b")]
            rows.append({"case_id": identity, "model": m, "result": legacy.saved(result)})
    finally:
        sys.setprofile(previous)
    assert sys.getprofile() is previous and hsd.project_cones is projector
    assert legacy.source_hashes() == before
    assert counts["native_solves"] == counts["factorizations"] == 12
    assert counts["projections"] == counts["lu_solves"] > 0
    receipt = {
        "schema": "hsd-independent-cpu-v1", "owner": OWNER,
        "source_hashes": before, "calls": counts, "cases": rows,
        "qualification": "actual CPU synthetic; no exact/public/CUDA/commercial qualification",
    }
    path = destination / "actual-cpu.json"
    legacy.write_json(path, receipt)
    legacy.replay(json.loads(path.read_text()))
    json_write(destination / "profile-binding.json", {
        "owner": OWNER, "owner_sha256": owner_binding(),
        "historical_review": HISTORICAL_REVIEW, "historical_test_sha256": HISTORICAL_TEST_SHA,
        "historical_function_sha256": ORIGINAL_FUNCTION_PINS, "calls": counts,
        "observer": "sys.setprofile Python call/C-call; no production wrapper or alias replacement",
        "projector_identity_unchanged": True, "profile_restored": True,
        "actual_receipt_sha256": sha(path),
    })
    return json.loads(path.read_text()), path


def record(request, tmp_path, data):
    path = tmp_path / "independent-observation.json"
    json_write(path, data)
    request.node.user_properties.extend([
        ("independent_observation_path", str(path)),
        ("independent_observation_sha256", sha(path)),
    ])


def test_original45_byte_pins_and_current_owner(actual_receipt):
    assert len(ORIGINAL_FUNCTION_PINS) == 10
    assert owner_binding() == OWNER_PINS
    for name in ORIGINAL_FUNCTION_PINS:
        assert globals()[name] is getattr(legacy, name)
        assert globals()[name].__globals__ is vars(legacy)
    receipt, path = actual_receipt
    assert receipt["owner"] == OWNER and receipt["calls"]["native_solves"] == 12
    assert legacy.replay(json.loads(path.read_text()))["replay_solve_calls"] == 0


def test_exact_half_farkas_and_legal_candidate(request, tmp_path):
    owner_binding()
    m = legacy.cases()["farkas"][0]
    ray = [1.0, 1.5]
    annihilation = sum((Fraction.from_float(row[0]) * Fraction.from_float(v)
                        for row, v in zip(m["A"], ray)), Fraction())
    assert annihilation == Fraction(1, 2)
    expected = legacy.farkas_oracle(m, ray)
    actual = hsd.check_hsd_primal_ray(legacy.problem(m), ray)
    good = legacy.cases()["equality"][0]
    legal = hsd.check_hsd_candidate(legacy.problem(good), [2.0], [0.0], [-1.0])
    record(request, tmp_path, {
        "case": "exact-half-and-legal", "annihilation_fraction": str(annihilation),
        "oracle": expected, "actual_checks": actual.checks, "actual_passed": actual.passed,
        "legal_candidate_passed": legal.passed, "owner_sha256": owner_binding(),
    })
    assert not expected["annihilation"] and not actual.checks["annihilation"] and not actual.passed
    assert legal.passed and all(legacy.candidate_oracle(good, [2.0], [0.0], [-1.0]).values())


def hostile_check(*args, **kwargs):
    # This code executes only after transplantation into the actual HSD globals.
    return HSDLPCheck("independent_hostile", True, {"false_accept": True}, {}, {})


def disabled_binding(buffers=None):
    return buffers


def replacement(*args, **kwargs):
    return None


MUTATIONS = (
    "candidate_code", "farkas_code", "recession_code", "dot_helper", "base_helper",
    "checker_kwdefaults", "constructor_default", "config_field", "config_class",
    "prepared_extra_method", "dimension_helper", "projector_alias", "projector_wrapper",
    "projector_closure", "cone_alias", "fraction_alias", "native_helper",
    "guard_helper", "module_cache", "source_origin",
)


@contextmanager
def mutation_context(mutation):
    with pytest.MonkeyPatch.context() as patch:
        if mutation in ("candidate_code", "farkas_code", "recession_code"):
            target = {"candidate_code": "check_hsd_candidate", "farkas_code": "check_hsd_primal_ray",
                      "recession_code": "check_hsd_recession_ray"}[mutation]
            patch.setattr(getattr(hsd, target), "__code__", hostile_check.__code__)
        elif mutation == "dot_helper":
            patch.setattr(hsd, "_dot", replacement)
        elif mutation == "base_helper":
            patch.setattr(hsd, "_primal", replacement)
        elif mutation == "checker_kwdefaults":
            patch.setattr(hsd.check_hsd_candidate, "__kwdefaults__",
                          {"eps_abs": 1e-3, "eps_rel": 1e-7})
        elif mutation == "constructor_default":
            config = hsd.prepare_hsd_lp.__defaults__[0]
            patch.setitem(vars(config), "eps_abs", 1e-3)
        elif mutation == "config_field":
            patch.setattr(hsd.HSDLPConfig.__dataclass_fields__["eps_abs"], "default", 1e-3)
        elif mutation == "config_class":
            patch.setattr(hsd.HSDLPCheck, "__init__", replacement)
        elif mutation == "prepared_extra_method":
            patch.setattr(hsd.PreparedHSDLP, "__getattr__", replacement, raising=False)
        elif mutation == "dimension_helper":
            patch.setattr(conic, "_dimension", replacement)
        elif mutation == "projector_alias":
            patch.setattr(hsd, "project_cones", replacement)
        elif mutation == "projector_wrapper":
            trusted = native.project_cones

            def misleading(*args, **kwargs):
                return trusted(*args, **kwargs)

            misleading.__wrapped__ = trusted.__wrapped__
            patch.setattr(native, "project_cones", misleading)
            patch.setattr(hsd, "project_cones", misleading)
        elif mutation == "projector_closure":
            cell = dict(zip(native.project_cones.__code__.co_freevars,
                            native.project_cones.__closure__))["func"]
            original = cell.cell_contents
            cell.cell_contents = replacement
            try:
                yield
            finally:
                cell.cell_contents = original
            return
        elif mutation == "cone_alias":
            patch.setattr(hsd, "SecondOrderCone", conic.NonnegativeCone)
        elif mutation == "fraction_alias":
            patch.setattr(hsd, "Fraction", float)
        elif mutation == "native_helper":
            patch.setattr(native, "_finite", replacement)
        elif mutation == "guard_helper":
            patch.setattr(hsd, "_binding_codes", replacement)
        elif mutation == "module_cache":
            patch.setitem(sys.modules, "openopt.hsd_lp", ModuleType("openopt.hsd_lp"))
        else:
            assert mutation == "source_origin"
            patch.setattr(hsd.__spec__, "origin", "/not-the-caller-source.py")
        yield


@pytest.mark.parametrize("stage", ["prepare", "solve"])
@pytest.mark.parametrize("mutation", MUTATIONS)
def test_loaded_closure_rejects(stage, mutation, request, tmp_path):
    owner_binding()
    p = legacy.problem(legacy.cases()["equality"][0])
    config = hsd.HSDLPConfig(enabled=True, max_iterations=10, check_interval=1)
    prepared = hsd.prepare_hsd_lp(p, config) if stage == "solve" else None
    outcome, reason, returned = "RETURNED", None, None
    with mutation_context(mutation):
        try:
            returned = (prepared.solve() if stage == "solve" else hsd.prepare_hsd_lp(p, config))
        except (RuntimeError, TypeError, ValueError) as error:
            outcome, reason = "REJECTED", str(error)
    record(request, tmp_path, {
        "case": mutation, "stage": stage, "outcome": outcome, "reason": reason,
        "returned_status": getattr(returned, "status", None), "owner_sha256": owner_binding(),
    })
    assert outcome == "REJECTED" and reason, f"{stage}/{mutation}: {outcome}"


@pytest.mark.parametrize("stage", ["prepare", "solve"])
def test_guard_inplace_code_rejects(stage, request, tmp_path):
    owner_binding()
    p = legacy.problem(legacy.cases()["equality"][0])
    config = hsd.HSDLPConfig(enabled=True, max_iterations=1)
    prepared = hsd.prepare_hsd_lp(p, config) if stage == "solve" else None
    entry = hsd._loaded_binding
    original_code = entry.__code__
    outcome, reason = "RETURNED", None
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(entry, "__code__", disabled_binding.__code__)
        assert hsd._loaded_binding is entry and entry.__code__ is not original_code
        try:
            if stage == "solve":
                assert prepared._binding_entry is entry
                prepared.solve()
            else:
                hsd.prepare_hsd_lp(p, config)
        except RuntimeError as error:
            outcome, reason = "REJECTED", str(error)
    record(request, tmp_path, {
        "case": "guard-inplace", "stage": stage, "outcome": outcome, "reason": reason,
        "object_identity_preserved": True, "source_sha256": sha(ROOT / "src/openopt/hsd_lp.py"),
    })
    assert outcome == "REJECTED", f"in-place guard {stage}: {outcome}"


@pytest.mark.parametrize("checker", ["check_hsd_candidate", "check_hsd_primal_ray"])
@pytest.mark.parametrize("stage", ["after_prepare", "during_solve"])
def test_guard_inplace_cannot_admit_invalid_math(checker, stage, request, tmp_path):
    owner_binding()
    identity = "equality" if checker == "check_hsd_candidate" else "farkas"
    m = legacy.cases()[identity][0]
    budget = 10 if identity == "equality" else 2
    config = hsd.HSDLPConfig(enabled=True, max_iterations=budget, check_interval=1)
    baseline = hsd.solve_hsd_lp(legacy.problem(m), config)
    assert baseline.status == "max_iterations", "hostile-control baseline"
    prepared = hsd.prepare_hsd_lp(legacy.problem(m), config)
    entry, function = hsd._loaded_binding, getattr(hsd, checker)
    entry_code, checker_code = entry.__code__, function.__code__
    original_linear = torch.linalg.lu_solve
    altered_calls = []
    outcome, reason, result = "RETURNED", None, None
    with pytest.MonkeyPatch.context() as patch:
        def alter():
            patch.setattr(entry, "__code__", disabled_binding.__code__)
            patch.setattr(function, "__code__", hostile_check.__code__)
            assert entry is prepared._binding_entry and entry is hsd._loaded_binding

        if stage == "after_prepare":
            alter()
        else:
            def mutate_during_linear(*args, **kwargs):
                value = original_linear(*args, **kwargs)
                if not altered_calls:
                    alter()
                altered_calls.append(1)
                return value
            patch.setattr(torch.linalg, "lu_solve", mutate_during_linear)
        try:
            result = prepared.solve()
        except RuntimeError as error:
            outcome, reason = "REJECTED", str(error)
        changed_entry = digest(marshal.dumps(entry.__code__))
        changed_checker = digest(marshal.dumps(function.__code__))
    data = {
        "case": "inplace-guard-with-" + checker, "stage": stage, "outcome": outcome,
        "reason": reason, "baseline_status": baseline.status, "original_model": m,
        "owner_sha256": owner_binding(), "object_identity_preserved": True,
        "original_guard_code_sha256": digest(marshal.dumps(entry_code)),
        "altered_guard_code_sha256": changed_entry,
        "original_checker_code_sha256": digest(marshal.dumps(checker_code)),
        "altered_checker_code_sha256": changed_checker,
        "linear_calls_after_injection": len(altered_calls),
    }
    if result is not None:
        saved = legacy.saved(result)
        data["result"] = saved
        if identity == "equality":
            data["original_oracle"] = legacy.candidate_oracle(m, saved["x"], saved["s"], saved["y"])
        else:
            ray = saved["primal_ray"]
            data["original_oracle"] = legacy.farkas_oracle(m, ray)
            exact = sum((Fraction.from_float(row[0]) * Fraction.from_float(v)
                         for row, v in zip(m["A"], ray)), Fraction())
            data["annihilation_fraction"] = str(exact)
            assert exact == Fraction(1, 2), "actual wrong Farkas must be the exact half control"
        assert not all(data["original_oracle"].values()), "control must actually be invalid"
    record(request, tmp_path, data)
    assert outcome == "REJECTED", f"{checker}/{stage} false accept; evidence={tmp_path}"


@pytest.mark.parametrize("mutation", ["checker_default", "dot_helper", "projector_closure"])
def test_post_iteration_closure_rejects(mutation, request, tmp_path):
    owner_binding()
    prepared = hsd.prepare_hsd_lp(legacy.problem(legacy.cases()["equality"][0]),
                                  hsd.HSDLPConfig(enabled=True, max_iterations=1))
    linear = torch.linalg.lu_solve
    cell = dict(zip(native.project_cones.__code__.co_freevars,
                    native.project_cones.__closure__))["func"]
    original_project, original_dot = cell.cell_contents, hsd._dot
    calls = []
    outcome, reason = "RETURNED", None
    with pytest.MonkeyPatch.context() as patch:
        def inject(*args, **kwargs):
            result = linear(*args, **kwargs)
            calls.append(1)
            if mutation == "checker_default":
                patch.setattr(hsd.check_hsd_candidate, "__kwdefaults__",
                              {"eps_abs": 1e-3, "eps_rel": 1e-7})
            elif mutation == "dot_helper":
                def altered_dot(*args, **kwargs):
                    return original_dot(*args, **kwargs)
                patch.setattr(hsd, "_dot", altered_dot)
            else:
                def altered_project(*args, **kwargs):
                    return original_project(*args, **kwargs)
                patch.setattr(cell, "cell_contents", altered_project)
            return result

        patch.setattr(torch.linalg, "lu_solve", inject)
        try:
            prepared.solve()
        except RuntimeError as error:
            outcome, reason = "REJECTED", str(error)
    record(request, tmp_path, {
        "case": "post-iteration-" + mutation, "outcome": outcome,
        "reason": reason, "linear_calls": len(calls), "owner_sha256": owner_binding(),
    })
    assert len(calls) == 1 and outcome == "REJECTED"


def source_snapshot():
    from scripts.verify import source_snapshot as snapshot
    return snapshot(ROOT)


def run_logged(command, path, env, timeout):
    started = time.monotonic()
    with path.open("x") as log:
        log.write(f"command={command!r}\n")
        log.flush()
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=timeout, check=False)
            rc = result.returncode
        except subprocess.TimeoutExpired:
            rc = 124
            log.write(f"\nTIMEOUT budget_seconds={timeout}; FAIL\n")
        log.write(f"\nreturncode={rc}\n")
    return rc, time.monotonic() - started


def review_cli(root, development=False):
    root = root.resolve()
    assert root.is_relative_to(EXTERNAL) and not root.is_relative_to(ROOT), "external artifacts"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="review-", dir=root))
    started = datetime.now(timezone.utc).isoformat()
    before = source_snapshot()
    first_failure = None
    selection, membership, observations, failures = [], [], [], []
    counts = {"tests": 0, "passed": 0, "failures": 0, "errors": 0, "skipped": 0}
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", TMPDIR=str(directory),
               PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(ROOT)]))
    rc, collect_rc = 2, None
    timings = {}
    try:
        assert development or before["clean"], "clean reviewer source required"
        owner_binding()
        base = [sys.executable, "-B", "-m", "pytest", "-q", "-ra",
                "-o", f"cache_dir={directory / 'cache'}"]
        collect_rc, timings["collection"] = run_logged(
            base + ["--collect-only", TEST], directory / "collection.log", env, 25,
        )
        assert collect_rc == 0, "collection failed"
        selection = [s for s in (directory / "collection.log").read_text().splitlines()
                     if s.startswith(TEST + "::")]
        assert selection and len(selection) == len(set(selection)), "empty/duplicate collection"
        rc, timings["pytest"] = run_logged(
            base + ["--basetemp", str(directory / "tmp"),
                    f"--junitxml={directory / 'pytest.xml'}", TEST],
            directory / "pytest.log", env, 120,
        )
        assert rc in (0, 1), f"pytest execution failed rc={rc}"
        xml = ET.parse(directory / "pytest.xml").getroot()
        cases = list(xml.iter("testcase"))
        for case in cases:
            assert case.attrib["classname"] == "tests.test_hsd_loaded_repair_independent", \
                "JUnit module identity"
        membership = [TEST + "::" + c.attrib["name"] for c in cases]
        assert membership and len(membership) == len(set(membership)), "empty/duplicate JUnit"
        assert set(membership) == set(selection), "collection/JUnit bidirectional mismatch"
        counts = {"tests": len(cases), **{
            name: sum(c.find(tag) is not None for c in cases)
            for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
        }}
        counts["passed"] = counts["tests"] - sum(counts[k] for k in ("failures", "errors", "skipped"))
        for suite in xml.iter("testsuite"):
            if suite.findall("testsuite"):
                continue
            leaves = suite.findall("testcase")
            assert int(suite.get("tests")) == len(leaves), "JUnit suite denominator"
            for key, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
                assert int(suite.get(key)) == sum(c.find(tag) is not None for c in leaves)
        original = [n for n in selection if n.split("::", 1)[1].split("[", 1)[0]
                    in ORIGINAL_FUNCTION_PINS]
        assert len(original) == 45, "original45 complete denominator"
        assert len(selection) == 96, "complete version-one denominator"
        for nodeid, case in zip(membership, cases):
            for tag in ("failure", "error"):
                if case.find(tag) is not None:
                    failures.append({"nodeid": nodeid, "reason": case.find(tag).get("message", "")})
            props = {p.get("name"): p.get("value") for p in case.findall("properties/property")}
            if "independent_observation_path" in props:
                path = Path(props["independent_observation_path"]).resolve(strict=True)
                assert path.is_relative_to(directory) and sha(path) == \
                    props["independent_observation_sha256"], "observation identity"
                observations.append({"nodeid": nodeid, "path": str(path), "sha256": sha(path),
                                     "actual": json.loads(path.read_text())})
        assert len(observations) == 50, "complete actual observation denominator"
        profile_paths = list(directory.rglob("profile-binding.json"))
        actual_paths = list(directory.rglob("actual-cpu.json"))
        assert len(profile_paths) == len(actual_paths) == 1, "actual positive receipt membership"
        profile = json.loads(profile_paths[0].read_text())
        assert profile["historical_function_sha256"] == ORIGINAL_FUNCTION_PINS
        assert profile["actual_receipt_sha256"] == sha(actual_paths[0])
        assert profile["projector_identity_unchanged"] and profile["profile_restored"]
        assert legacy.replay(json.loads(actual_paths[0].read_text()))["cases"] == 12
        loaded_paths = list(directory.rglob("loaded-restored-disk-counterexample.json"))
        assert len(loaded_paths) == 2, "original loaded controls complete"
        assert all(json.loads(p.read_text())["rejected"] for p in loaded_paths), \
            "original loaded control false accept"
        assert (rc == 1) == bool(counts["failures"] + counts["errors"]), "exit/JUnit mismatch"
        first_failure = failures[0] if failures else None
    except (AssertionError, OSError, ValueError, ET.ParseError) as error:
        first_failure = first_failure or {"stage": "harness", "reason": str(error)}
    after = source_snapshot()
    consistent = before == after
    if not consistent:
        first_failure = first_failure or {"stage": "source", "reason": "source drift"}
    accepted = rc == 0 and not first_failure and consistent and after["clean"] \
        and counts["tests"] > 0 and counts["errors"] == counts["skipped"] == counts["failures"] == 0
    history = {}
    for name, (path, pin) in HISTORY.items():
        assert sha(path) == pin, "historical evidence drift"
        value = json.loads(path.read_text())
        history[name] = {"path": str(path), "sha256": pin, "verdict": value["verdict"],
                         "tests": value["tests"], "not_upgraded": True}
    original_failures = [f for f in failures if f["nodeid"].split("::", 1)[1].split("[", 1)[0]
                         in ORIGINAL_FUNCTION_PINS]
    original_nodes = [n for n in membership if n.split("::", 1)[1].split("[", 1)[0]
                      in ORIGINAL_FUNCTION_PINS]
    receipt = {
        "schema": "hsd-loaded-repair-independent-v1", "verdict": "PASS" if accepted else "FAIL",
        "accept_hsd_loaded_repair": accepted, "returncode": rc, "collection_returncode": collect_rc,
        "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat(),
        "source_before": before, "source_after": after, "source_consistency": consistent,
        "owner": OWNER, "owner_sha256": OWNER_PINS, "historical_review": HISTORICAL_REVIEW,
        "historical_test_sha256": HISTORICAL_TEST_SHA,
        "original_function_sha256": ORIGINAL_FUNCTION_PINS, "history": history,
        "fixture_adaptation": "read-only sys.setprofile; no production alias/code replacement",
        "tests": counts, "selection": selection, "junit_membership": membership,
        "membership_missing": sorted(set(selection) - set(membership)),
        "membership_extra": sorted(set(membership) - set(selection)),
        "duplicates": sorted(n for n in set(membership) if membership.count(n) > 1),
        "original45": {"tests": len([n for n in selection if n.split("::", 1)[1].split("[", 1)[0]
                                     in ORIGINAL_FUNCTION_PINS]),
                       "actual_nodes": original_nodes, "failures_or_errors": original_failures,
                       "passed": len(original_nodes) - len(original_failures)},
        "first_failure": first_failure, "failures": failures, "observations": observations,
        "timings_seconds": timings,
        "runtime": {"python": sys.version, "numpy": np.__version__, "torch": str(torch.__version__),
                    "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads()},
        "artifacts": {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob("*"))
                      if p.is_file() and p.suffix in (".json", ".log", ".xml")},
        "gpu_jobs_submitted": 0, "cuda_verdict": "UNVERIFIED",
        "commercial_verdict": "UNVERIFIED", "remote_execution_verdict": "UNVERIFIED",
    }
    path = directory / "receipt.json"
    json_write(path, receipt)
    print(f"HSD_LOADED_REPAIR_INDEPENDENT verdict={receipt['verdict']} tests={counts} "
          f"source={'PASS' if consistent else 'FAIL'} receipt={path} sha256={sha(path)}", flush=True)
    return 0 if accepted else 1


def check_receipt(path):
    """Read back frozen FAIL/PASS evidence without executing any solver."""
    path = path.resolve(strict=True)
    assert path.is_relative_to(EXTERNAL), "external receipt required"
    value = json.loads(path.read_text())
    assert value["schema"] == "hsd-loaded-repair-independent-v1"
    assert value["owner"] == OWNER and value["owner_sha256"] == owner_binding()
    current = source_snapshot()
    assert current["clean"] and value["source_before"] == value["source_after"] == current
    assert value["source_consistency"] is True
    assert value["historical_test_sha256"] == HISTORICAL_TEST_SHA
    assert value["original_function_sha256"] == ORIGINAL_FUNCTION_PINS
    directory = path.parent
    for name, pin in value["artifacts"].items():
        target = (directory / name).resolve(strict=True)
        assert target.is_relative_to(directory) and sha(target) == pin, "artifact pin: " + name
    selected = [s for s in (directory / "collection.log").read_text().splitlines()
                if s.startswith(TEST + "::")]
    xml = ET.parse(directory / "pytest.xml").getroot()
    cases = list(xml.iter("testcase"))
    actual = [TEST + "::" + c.attrib["name"] for c in cases]
    assert len(selected) == len(set(selected)) == len(actual) == len(set(actual)) == 96
    assert set(selected) == set(actual)
    assert selected == value["selection"] and actual == value["junit_membership"]
    assert value["membership_missing"] == value["membership_extra"] == value["duplicates"] == []
    counts = {"tests": len(cases), **{
        name: sum(c.find(tag) is not None for c in cases)
        for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped"))
    }}
    counts["passed"] = counts["tests"] - sum(counts[k] for k in ("failures", "errors", "skipped"))
    assert counts == value["tests"]
    assert len(value["observations"]) == 50
    observed = {}
    for item in value["observations"]:
        target = Path(item["path"]).resolve(strict=True)
        assert target.is_relative_to(directory) and sha(target) == item["sha256"]
        assert json.loads(target.read_text()) == item["actual"]
        assert item["nodeid"] not in observed
        observed[item["nodeid"]] = item
    props_members = set()
    for nodeid, case in zip(actual, cases):
        assert case.attrib["classname"] == "tests.test_hsd_loaded_repair_independent"
        props = {p.get("name"): p.get("value") for p in case.findall("properties/property")}
        if "independent_observation_path" in props:
            item = observed[nodeid]
            assert str(Path(props["independent_observation_path"]).resolve()) == item["path"]
            assert props["independent_observation_sha256"] == item["sha256"]
            props_members.add(nodeid)
    assert props_members == set(observed)
    original = [n for n in actual if n.split("::", 1)[1].split("[", 1)[0]
                in ORIGINAL_FUNCTION_PINS]
    assert len(original) == 45 and original == value["original45"]["actual_nodes"]
    for name, (target, pin) in HISTORY.items():
        assert sha(target) == pin and value["history"][name]["sha256"] == pin
        assert value["history"][name]["verdict"] == "FAIL"
    expected_pass = counts["failures"] == counts["errors"] == counts["skipped"] == 0 \
        and value["first_failure"] is None and value["returncode"] == 0
    assert value["accept_hsd_loaded_repair"] is expected_pass
    assert value["verdict"] == ("PASS" if expected_pass else "FAIL")
    assert source_snapshot() == current
    print(f"HSD_INDEPENDENT_READBACK integrity=PASS acceptance={value['verdict']} "
          f"tests={counts} original45={len(original)} source=PASS "
          f"receipt={path} sha256={sha(path)}", flush=True)
    return 0 if expected_pass else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--development", action="store_true")
    parser.add_argument("--scan-tool", type=Path)
    parser.add_argument("--needle-file", type=Path)
    parser.add_argument("--check-receipt", type=Path)
    args = parser.parse_args()
    if args.check_receipt:
        raise SystemExit(check_receipt(args.check_receipt))
    assert args.artifact_dir
    if args.scan_tool:
        assert args.needle_file
        raise SystemExit(legacy.scan_cli(args.artifact_dir, args.scan_tool, args.needle_file))
    raise SystemExit(review_cli(args.artifact_dir, args.development))
