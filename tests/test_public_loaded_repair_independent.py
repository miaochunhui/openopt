"""Independent c18300a review; unchanged Nash assertions, read-only profiling.

Synthetic LP/QP workers are real CPU solves, not public/GPU qualification.
No numerical entry is wrapped by the acceptance observer. Interface probes
and loaded-object mutations are separate controls, never the positive run.
"""

from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import CodeType

import pytest

from scripts import public_qp_corpus as corpus
from scripts import qualify_public_qp as producer

ROOT = Path(__file__).resolve().parents[1]
FREEZE = "c18300a53a255f1be2d887430490ddf9d9fbd9b9"
RUNNER_PIN = "e0b65c9f0fd3070a90ac2082d059c85e98e8ee6fff21371732f1fc25a51fb25b"
NASH = Path(
    "/data/dev/openopt-wt-release4-20261010-public-v3-independent-nash/"
    "tests/test_public_v3_binding_independent.py"
)
NASH_PIN = "7d2262e7d1550800dddcb72b53341dc9b02efe44a8d62d58feaebe521da7995e"
ARTIFACT_ROOT = Path(
    "/hf3fs-jd/data/tmp/openopt-release-20261010/public-loaded-repair-independent"
).resolve()
OPT_IN = "OPENOPT_RUN_PUBLIC_V3_INDEPENDENT_TESTS"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    corpus.write_json(Path(path), value)


assert sha(NASH) == NASH_PIN, "unchanged reviewer source pin required"
assert producer.ROOT.resolve() == ROOT.resolve(), "cached foreign producer is not the target"
assert sha(ROOT / "scripts/qualify_public_qp.py") == RUNNER_PIN
spec = importlib.util.spec_from_file_location("unchanged_nash_public_review", NASH)
nash = importlib.util.module_from_spec(spec)
spec.loader.exec_module(nash)
assert nash.producer is producer
LEGACY_PINS = deepcopy(nash.EXECUTION_SOURCES)
LEGACY_ROOT = nash.ROOT
# Explicit reviewed target selection, not relaxed source/wrapper validation.
nash.ROOT = ROOT
nash.FREEZE = FREEZE
nash.ARTIFACT_ROOT = ARTIFACT_ROOT
nash.EXECUTION_SOURCES = {
    **LEGACY_PINS,
    "runner.py": ("scripts/qualify_public_qp.py", RUNNER_PIN),
}
for name, value in vars(nash).copy().items():
    if name.startswith("test_") and callable(value):
        globals()[name] = value


@pytest.fixture(scope="module", autouse=True)
def collected_contract(request, tmp_path_factory):
    nodes = [item.nodeid for item in request.session.items
             if Path(str(item.path)).resolve() == Path(__file__).resolve()]
    assert nodes and len(nodes) == len(set(nodes)), "nonempty exact review collection required"
    path = tmp_path_factory.getbasetemp() / "public-loaded-collection.json"
    write(path, {
        "schema_version": 1, "nodeids": nodes, "collected": len(nodes),
        "original_suite_sha256": NASH_PIN, "review_sha256": sha(__file__),
        "runner_sha256": RUNNER_PIN, "production_freeze": FREEZE,
        "actual_opt_in": os.environ.get(OPT_IN) == "1",
        "scope": "test collection inventory, not a terminal verdict or qualification",
    })


@contextmanager
def readonly_observer():
    """Count bound entries/returns from frames; never edit numerical callables."""
    previous = sys.getprofile()
    events, active, states = [], {}, {}
    workspace_codes, refinement_codes = {}, set()
    bridges = [value for value in producer.v3_observe_call.__code__.co_consts
               if isinstance(value, CodeType) and value.co_name == "native_call"]
    assert len(bridges) == 1, "exact reviewed production bridge required"
    native_bridge = bridges[0]

    def observe(frame, event, argument):
        if event == "return" and frame.f_code is producer.v3_setup.__code__:
            state = argument
            backend = frame.f_locals["solver"]
            core = frame.f_locals["core"]
            if state is not None:
                states[id(state.workspace)] = backend
                method = getattr(state.workspace.solve, "__func__", state.workspace.solve)
                while hasattr(method, "__wrapped__"):
                    method = method.__wrapped__
                code = getattr(method, "__code__", None)
                if code is not None:
                    workspace_codes[code] = id(state.workspace)
                if backend == "osqp":
                    states[id(state.workspace._solver)] = "native_osqp"
                if backend == "openopt":
                    function = core.refinement.refine_qp_candidate
                    while hasattr(function, "__wrapped__"):
                        function = function.__wrapped__
                    refinement_codes.add(function.__code__)
                    placements = {
                        key: {"device": str(getattr(state.p, key).device),
                              "dtype": str(getattr(state.p, key).dtype)}
                        for key in ("P", "q", "A", "l", "u")
                    }
                    assert all(value == {"device": "cpu", "dtype": "torch.float64"}
                               for value in placements.values())
                    events.append({"entry": "placement", "actual_original_tensors": placements})
        elif event == "call":
            label = None
            if frame.f_code is native_bridge:
                receiver = frame.f_locals.get("self")
                if (states.get(id(receiver)) == "native_osqp"
                        and receiver is frame.f_locals.get("native")
                        and frame.f_locals.get("solver") == "osqp"):
                    label = "native_osqp"
            elif frame.f_code in workspace_codes:
                identity = workspace_codes[frame.f_code]
                if id(frame.f_locals.get("self")) == identity:
                    label = "workspace"
            elif frame.f_code in refinement_codes:
                label = "refinement"
            if label is not None:
                active[id(frame)] = label
                events.append({"entry": label, "phase": "enter"})
        elif event == "return" and id(frame) in active:
            label = active.pop(id(frame))
            if label != "native_osqp":
                assert argument is not None, "count successful return separately from entry"
            events.append({"entry": label, "phase": "return"})
        elif event in ("c_call", "c_return"):
            if getattr(argument, "__name__", "") == "solve":
                backend = states.get(id(getattr(argument, "__self__", None)))
                label = "native_osqp" if backend == "native_osqp" else (
                    "workspace" if backend == "clarabel" else None)
                if label is not None:
                    events.append({"entry": label, "phase": "enter" if event == "c_call" else "return"})
        if previous is not None:
            previous(frame, event, argument)

    sys.setprofile(observe)
    try:
        yield events
    finally:
        sys.setprofile(previous)


def observe_worker(output, pin, index):
    output = nash.external_root(output)
    with readonly_observer() as events:
        code = producer.v3_worker(output, pin, index)
    folder = output.parent / "observations"
    folder.mkdir(exist_ok=True)
    write(folder / f"worker-{index:03d}.json", {
        "worker_index": index, "exit_code": code, "plan_sha256": pin,
        "events": events,
        "test_source_sha256": NASH_PIN,
        "observer_source_sha256": sha(__file__),
        "observer": "sys.setprofile read-only bound entry/return; no numerical wrapper",
        "runner_sha256": RUNNER_PIN,
    })
    return code


def execute(frozen, output):
    """Only redirect orchestration to this observer; worker logic stays unchanged."""
    nash.external_root(output)
    original = subprocess.run

    def dispatch(command, *args, **kwargs):
        if isinstance(command, list) and "worker" in command and "--worker-index" in command:
            index = command[command.index("--worker-index") + 1]
            pin = command[command.index("--plan-sha256") + 1]
            command = [sys.executable, "-B", str(Path(__file__).resolve()),
                       "--observe-worker", str(output), pin, index]
        return original(command, *args, **kwargs)

    args = producer.parser().parse_args([
        "run", "--version", producer.V3, "--corpus", str(frozen),
        "--manifest-sha256", sha(frozen / "corpus.json"), "--artifact-dir", str(output),
    ])
    subprocess.run = dispatch
    try:
        _, code = producer.v3_run(args)
    finally:
        subprocess.run = original
    return code


@pytest.fixture(scope="module")
def own_execution(tmp_path_factory):
    if os.environ.get(OPT_IN) != "1":
        pytest.skip(f"actual CPU workers require {OPT_IN}=1; not acceptance")
    root = nash.external_root(tmp_path_factory.mktemp("independent-own-execution"))
    local = root / "input"
    local.mkdir()
    entries = []
    for case_id, text in (("nashlp", nash.LP), ("nashqp", nash.QP)):
        raw = local / f"{case_id}.mps"
        raw.write_text(text, encoding="ascii")
        entries.append({
            "case_id": case_id, "path": raw.name, "origin": "synthetic",
            "source_url": f"https://example.invalid/nash-synthetic-{case_id}.mps",
            "license": "0BSD independent hand-authored synthetic; no public qualification",
            "input_sha256": sha(raw),
        })
    manifest = root / "input.json"
    write(manifest, {"schema_version": 1, "cases": entries})
    frozen = root / "frozen"
    corpus.freeze(manifest, local, frozen)
    process = subprocess.run(
        [sys.executable, "-B", str(Path(__file__).resolve()), "--execute",
         str(frozen), str(root / "execution")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False,
        env={**os.environ, **producer.V3_ENV}, cwd=ROOT,
    )
    (root / "execute.log").write_text(process.stdout)
    path = root / "execution/result.json"
    assert process.returncode == 0, (process.returncode, str(root / "execute.log"))
    report = corpus.read_json(path)
    assert report["verdict"] == "EXECUTION_COMPLETE_NOT_QUALIFICATION", report.get("error")
    return path, report


def test_frozen_selection_and_original_assertions_are_unchanged():
    assert sha(NASH) == NASH_PIN
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, check=True)
    assert subprocess.run(
        ["git", "merge-base", "--is-ancestor", FREEZE, process.stdout.strip()],
        cwd=ROOT, capture_output=True, check=False,
    ).returncode == 0
    assert sha(producer.__file__) == RUNNER_PIN
    assert set(nash.EXECUTION_SOURCES) == set(LEGACY_PINS)
    assert {key: value for key, value in nash.EXECUTION_SOURCES.items() if key != "runner.py"} == {
        key: value for key, value in LEGACY_PINS.items() if key != "runner.py"}
    assert all(globals()[key] is value for key, value in vars(nash).items()
               if key.startswith("test_") and callable(value))


def test_old_caller_pin_conflict_is_observed_not_silently_relaxed(tmp_path, monkeypatch):
    with monkeypatch.context() as context:
        context.setattr(nash, "EXECUTION_SOURCES", LEGACY_PINS)
        with pytest.raises(AssertionError) as rejected:
            nash.test_default_optin_contract_is_explicit_and_source_pinned()
    write(tmp_path / "legacy-pin-conflict.json", {
        "verdict": "HARNESS_PIN_CONFLICT_REPRODUCED",
        "old_runner_sha256": LEGACY_PINS["runner.py"][1], "new_runner_sha256": RUNNER_PIN,
        "reason": str(rejected.value), "negative_assertions_changed": False,
        "scope": "new reviewed caller selection only; no wrapped-function whitelist",
    })


def test_old_wrapper_observer_conflict_reproduced_without_solving(
    own_execution, tmp_path, monkeypatch,
):
    path, report = own_execution
    core = producer.v3_core(path.parent / "code")
    model = producer.load_model(path.parent / "frozen", "nashlp")
    output = nash.external_root(tmp_path / "legacy-wrapper-probe")
    output.mkdir()
    calls = []
    previous = sys.getprofile()

    def probe(frame, event, argument):
        if event == "call" and frame.f_code.co_name in ("solve", "solve_qp", "refine_qp_candidate"):
            calls.append(frame.f_code.co_name)
            raise AssertionError("interface probe must not solve")
        if previous is not None:
            previous(frame, event, argument)

    def only_setup_then_guard(*args):
        # Invoke the unchanged observer's actual install path, before any solve.
        producer.v3_setup(model, "openopt", core)
        producer.v3_loaded_sources(path.parent, core)
        raise AssertionError("wrapped entry must not be allowed")

    with monkeypatch.context() as context:
        context.setattr(producer, "v3_worker", only_setup_then_guard)
        sys.setprofile(probe)
        try:
            with pytest.raises(ValueError, match="loaded") as rejected:
                nash._observe_worker(output, report["plan_sha256"], 0)
        finally:
            sys.setprofile(previous)
    assert calls == []
    write(tmp_path / "legacy-wrapper-conflict.json", {
        "verdict": "HARNESS_WRAPPER_CONFLICT_REPRODUCED", "reason": str(rejected.value),
        "solve_calls": 0, "scope": "unchanged observer setup/install then real guard, not worker acceptance",
        "old_source_sha256": NASH_PIN, "runner_sha256": RUNNER_PIN,
        "numerical_wrappers_whitelisted": False,
    })


@pytest.fixture
def loaded_core(tmp_path):
    code = tmp_path / "loaded-object/code"
    code.mkdir(parents=True)
    for name in ("certificates.py", "qp.py", "scaling.py", "refinement.py"):
        shutil.copyfile(ROOT / producer.V3_SOURCES[name], code / name)
    core = producer.v3_core(code)
    with nash.replay_trap() as hits:
        assert set(producer.v3_loaded_sources(code.parent, core)) == {
            "certificates.py", "qp.py", "scaling.py", "refinement.py"}
    assert hits == []
    return code.parent, core


@pytest.mark.parametrize("attack", [
    "restored-disk", "default", "code", "class-init", "class-default",
    "checker-alias", "wrapped-function", "global", "package-cache",
])
def test_nine_live_loaded_mutations_reject_without_solving(
    loaded_core, monkeypatch, tmp_path, attack, request,
):
    import functools

    output, core = loaded_core
    checker = core.checker.check_qp_candidate
    loaded_pin = None
    if attack == "restored-disk":
        source = output / "code/certificates.py"
        trusted = source.read_bytes()
        original = b"y: object, *, atol: float = 1e-8, rtol: float = 1e-8,"
        assert trusted.count(original) == 1
        altered = trusted.replace(original, original.replace(b"1e-8", b"1e-3", 1))
        loaded_pin = hashlib.sha256(altered).hexdigest()
        source.write_bytes(altered)
        try:
            core = producer.v3_core(output / "code")
            assert core.checker.check_qp_candidate.__kwdefaults__["atol"] == 1e-3
        finally:
            source.write_bytes(trusted)
        assert sha(source) == LEGACY_PINS["certificates.py"][1]
    elif attack == "default":
        monkeypatch.setattr(checker, "__kwdefaults__", {**checker.__kwdefaults__, "atol": 1e-3})
    elif attack == "code":
        monkeypatch.setattr(checker, "__code__", checker.__code__.replace(
            co_consts=(*checker.__code__.co_consts, "independent loaded code mutation")))
    elif attack == "class-init":
        init = core.qp.QPConfig.__init__
        monkeypatch.setattr(init, "__code__", init.__code__.replace(
            co_consts=(*init.__code__.co_consts, "independent generated constructor mutation")))
    elif attack == "class-default":
        monkeypatch.setattr(core.refinement.RefinementConfig, "check_arithmetic", "other")
    elif attack == "checker-alias":
        monkeypatch.setattr(core.refinement, "check_qp_candidate", lambda *args: None)
    elif attack == "wrapped-function":
        original = core.scaling.equilibrate_qp

        @functools.wraps(original)
        def substituted(*args, **kwargs):
            raise AssertionError("arbitrary wrapped entry must never run")

        monkeypatch.setattr(core.scaling, "equilibrate_qp", substituted)
    elif attack == "global":
        monkeypatch.setattr(core.checker, "REPRODUCIBLE_ARITHMETIC", "other")
    else:
        monkeypatch.setitem(sys.modules, core.qp.__name__, core.checker)
    with nash.replay_trap() as hits:
        with pytest.raises(ValueError, match="loaded") as rejected:
            producer.v3_loaded_sources(output, core)
    assert hits == []
    receipt = tmp_path / "loaded-rejection.json"
    write(receipt, {
        "attack": attack, "rejected": True, "error": str(rejected.value),
        "solve_calls": 0, "actually_loaded_sha256": loaded_pin,
        "runner_sha256": RUNNER_PIN, "review_sha256": sha(__file__),
    })
    request.node.user_properties.extend([
        ("loaded_rejection_path", str(receipt)), ("loaded_rejection_sha256", sha(receipt))])


def scan(argv):
    """Only the two authorized files; credential needle values never leave memory."""
    import argparse
    import re

    parser = argparse.ArgumentParser()
    parser.add_argument("--scan-only", action="store_true", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    assert args.artifact_dir.is_absolute() and not args.artifact_dir.exists()
    location = "/data/dev/infra/ops/dsh/scan-secrets-precommit.py"
    module_spec = importlib.util.spec_from_file_location("public_loaded_safe_scan", location)
    scanner = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(
        secret_hex=None, secrets_file=None, no_needle=False,
        needle_from=str(Path.home() / ".config/dsh-n600-token")))
    assert needles
    paths = [Path(__file__).resolve(), ROOT / "docs/research/public-loaded-repair-independent.md"]
    literal, _ = scanner.scan_literal([str(path) for path in paths], needles)
    pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                         r"""(?i:password|token|secret)\s*["']?\s*[:=]\s*["'][^"'\s]{6,}["']""")
    shape = sum(bool(pattern.search(path.read_text())) for path in paths)
    args.artifact_dir.mkdir(parents=True)
    receipt = {"verdict": "PASS" if literal == shape == 0 else "FAIL",
               "literal_hits": literal, "shape_hits": shape, "files": [str(path) for path in paths]}
    write(args.artifact_dir / "scan.json", receipt)
    (args.artifact_dir / "execution.log").write_text(json.dumps(receipt) + "\n")
    print(json.dumps(receipt, sort_keys=True))
    return 0 if literal == shape == 0 else 1


if __name__ == "__main__":
    if "--scan-only" in sys.argv:
        raise SystemExit(scan(sys.argv[1:]))
    if len(sys.argv) == 5 and sys.argv[1] == "--observe-worker":
        raise SystemExit(observe_worker(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])))
    if len(sys.argv) == 4 and sys.argv[1] == "--execute":
        raise SystemExit(execute(Path(sys.argv[2]), Path(sys.argv[3])))
    raise SystemExit("only --execute FROZEN OUTPUT or --observe-worker OUTPUT PLAN_PIN INDEX")
