"""Independent CPU review of a caller-selected RSOC gate; never solve or submit.

Run this file directly with --artifact-dir PATH. Default review source bytes and
the original GPU artifacts are independently pinned here. Mutations touch only
an external relocated fixture; those copies are not fresh CUDA/origin evidence.
"""

from __future__ import annotations

import argparse
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
import types
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path("/data/dev/openopt-wt-release4-20261010-extremes-acceptance")
TARGET_COMMIT = "6ab87e96b869506039af0b9e6f4c09066d5b8363"
GATE = "scripts/accept_rsoc_extremes.py"
HELPER = "scripts/accept_release_cuda.py"
GATE_PIN = "217652d2b58675d349c380a326430a05103331f6c3743a747280036a8abe1c62"
HELPER_PIN = "9980fe81a4d5e8c85546f846c7f4a1a3e75dbf4205a12e515721fb111f3f84bc"
RUN = Path("/hf3fs-jd/data/tmp/openopt-release-20261010/"
           "owner-acc1993-extremes-cuda-v1/release-regression-o4xde6y1").resolve()
REPORT_DIR = "case-0/cuda-check-ys_jck1h"
STATUS = "openopt-release-regression-o4xde6y1-0-status.json"
PINS = {
    "run.json": "6e312f40fc6c4b3b8fb15c01d95f197702d9b58d091077889e70113faef94e3e",
    REPORT_DIR + "/result.json": "a721c1c4968ab1cd9c6a490928cdf9cdbac0e08c3a320bbb87d65585ea86ce58",
    REPORT_DIR + "/pytest.xml": "82979ed3a25c711d2042871209a4cd4940d110136f4b903cc26d1bc28740b1ce",
    STATUS: "bd48f0c8f2fec2c1d56dc17435a961db28c953b4cc731ea6ac903def68b20f85",
    "collection.json": "ab9d81d8578f2b6378cb89ebbdd84930a315cb3c1ff9657264dc202963c523b7",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def put(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def selected():
    root = Path(os.environ.get("RSOC_REVIEW_ROOT", str(TARGET))).resolve()
    pin = os.environ.get("RSOC_REVIEW_SHA256", GATE_PIN)
    return root, {GATE: pin, HELPER: HELPER_PIN}


@pytest.fixture(scope="module")
def gate():
    root, pins = selected()
    raw = {}
    for name, pin in pins.items():
        path = root / name
        assert path.is_file() and not path.is_symlink()
        raw[name] = path.read_bytes()
        assert sha(raw[name]) == pin, ("caller-selected source pin", name)
    # Explicit isolation chooses reviewed bytes, not a claim that a cached
    # foreign production module was rejected by any loaded-source guard.
    helper = types.ModuleType("rsoc_review_execution")
    helper.__file__ = str(root / HELPER)
    helper.__package__ = "rsoc_review"
    package = types.ModuleType("scripts")
    package.__path__ = [str(root / "scripts")]
    old = {name: sys.modules.get(name) for name in ("scripts", "scripts.accept_release_cuda")}
    module = types.ModuleType("rsoc_review_gate")
    module.__file__ = str(root / GATE)
    module.__package__ = "rsoc_review"
    try:
        exec(compile(raw[HELPER], helper.__file__, "exec"), helper.__dict__)
        package.accept_release_cuda = helper
        sys.modules["scripts"] = package
        sys.modules["scripts.accept_release_cuda"] = helper
        exec(compile(raw[GATE], module.__file__, "exec"), module.__dict__)
    finally:
        for name, previous in old.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    assert module.execution is helper
    for name, data in raw.items():
        assert (root / name).read_bytes() == data, "target changed while loading"
    yield module
    for name, data in raw.items():
        assert (root / name).read_bytes() == data, "target changed during independent review"


def arguments(root, pinned=False):
    paths = {
        "result": root / REPORT_DIR / "result.json", "run": root / "run.json",
        "junit": root / REPORT_DIR / "pytest.xml", "status": root / STATUS,
    }
    result = {"current_root": ROOT}
    for name, path in paths.items():
        result[name] = path
        result[name + "_sha256"] = PINS[path.relative_to(root).as_posix()] if pinned else sha(path.read_bytes())
    return result


@pytest.fixture(scope="module")
def relocated(tmp_path_factory, gate):
    for name, pin in PINS.items():
        assert sha((RUN / name).read_bytes()) == pin, ("original caller pin", name)
    root = tmp_path_factory.mktemp("relocated-no-new-origin") / RUN.name
    shutil.copytree(RUN, root, copy_function=shutil.copyfile)
    for path in root.rglob("*"):
        if not path.is_file() or path.is_relative_to(root / "source"):
            continue
        raw = path.read_bytes()
        replaced = raw.replace(str(RUN).encode(), str(root).encode())
        if raw != replaced:
            path.write_bytes(replaced)
    assert gate.accept(**arguments(root))["tests"]["tests"] == 76
    return root


def outcome(gate, root):
    try:
        return {"accepted": True, "report": gate.accept(**arguments(root))}
    except Exception as error:
        return {"accepted": False, "error": f"{type(error).__name__}: {error}"}


def record(tmp_path, name, result):
    put(tmp_path / "review-observation.json", {
        "case": name, "checked_at": datetime.now(timezone.utc).isoformat(),
        "fixture": "relocated real frozen GPU bytes; mutation is synthetic, not CUDA origin",
        **result,
    })


def test_actual_76_frozen_baseline(gate, tmp_path):
    report = gate.accept(**arguments(RUN, pinned=True))
    assert report["verdict"] == "RSOC_EXTREMES_REPLAY_PASS"
    assert report["tests"] == {"tests": 76, "failures": 0, "errors": 0, "skipped": 0}
    assert len(report["members"]) == len({x["case"] for x in report["members"]}) == 76
    assert report["qualified"] is False and report["remote_attestation"] == "NOT_PROVIDED"
    record(tmp_path, "actual-baseline", {"accepted": True, "report": report})


def test_replay_never_imports_producer_solver_or_torch(gate, monkeypatch):
    import builtins

    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert not name.startswith(("torch", "openopt", "tests.", "scripts.gpu_"))
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    assert gate.accept(**arguments(RUN, pinned=True))["tests"]["tests"] == 76


STATUS_MUTATIONS = (
    "job-envFrom", "pod-envFrom", "pod-workingDir", "pod-source-mount", "pod-source-volume",
    "pod-placement", "pod-namespace", "owner-kind", "image-invalid-digest",
    "exit-bool", "restart-bool",
)


def mutate_status(value, name):
    pod = value["pods"]["items"][0]
    spec = pod["spec"]
    container = spec["containers"][0]
    if name == "job-envFrom":
        job_container = value["job"]["spec"]["template"]["spec"]["containers"][0]
        job_container["envFrom"] = [{"configMapRef": {"name": "unsubmitted-python-environment"}}]
    elif name == "pod-envFrom":
        container["envFrom"] = [{"configMapRef": {"name": "unsubmitted-python-environment"}}]
    elif name == "pod-workingDir":
        container["workingDir"] = "/unsubmitted-python-shadow"
    elif name == "pod-source-mount":
        container["volumeMounts"][0]["mountPath"] = "/unsubmitted"
    elif name == "pod-source-volume":
        spec["volumes"][0] = {"name": "threefs", "emptyDir": {}}
    elif name == "pod-placement":
        spec["nodeSelector"] = {"nvidia.com/gpu.product": "NVIDIA-A100"}
    elif name == "pod-namespace":
        pod["metadata"]["namespace"] = "foreign-namespace"
    elif name == "owner-kind":
        pod["metadata"]["ownerReferences"][0]["kind"] = "ConfigMap"
    elif name == "image-invalid-digest":
        pod["status"]["containerStatuses"][0]["imageID"] = "foreign@sha256:x"
    elif name == "exit-bool":
        pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = False
    elif name == "restart-bool":
        pod["status"]["containerStatuses"][0]["restartCount"] = False
    else:
        raise AssertionError(name)


@pytest.mark.parametrize("mutation", STATUS_MUTATIONS)
def test_rejects_observed_execution_drift(gate, relocated, tmp_path, mutation):
    path = relocated / STATUS
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
        mutate_status(value, mutation)
        put(path, value)
        put(tmp_path / "mutated-status.json", value)
        result = outcome(gate, relocated)
        record(tmp_path, mutation, result)
        assert not result["accepted"], f"accepted observed execution drift: {mutation}"
    finally:
        path.write_bytes(raw)


@pytest.mark.parametrize("mutation", ("restart-one", "exit-one", "wrong-owner-uid",
                                     "pod-args", "job-args", "pod-env", "job-env", "pod-resource"))
def test_execution_negative_controls(gate, relocated, tmp_path, mutation):
    path = relocated / STATUS
    raw = path.read_bytes()
    value = json.loads(raw)
    pod = value["pods"]["items"][0]
    if mutation == "restart-one":
        pod["status"]["containerStatuses"][0]["restartCount"] = 1
    elif mutation == "exit-one":
        pod["status"]["containerStatuses"][0]["state"]["terminated"]["exitCode"] = 1
    elif mutation == "wrong-owner-uid":
        pod["metadata"]["ownerReferences"][0]["uid"] = "00000000-0000-0000-0000-000000000000"
    else:
        spec = value["job"]["spec"]["template"]["spec"] if mutation.startswith("job-") else pod["spec"]
        container = spec["containers"][0]
        if mutation.endswith("args"):
            container["args"] = ["--unsubmitted"]
        elif mutation.endswith("env"):
            container["env"].append({"name": "PYTHONSTARTUP", "value": "/foreign.py"})
        else:
            container["resources"]["requests"]["nvidia.com/gpu"] = "2"
    try:
        put(path, value)
        result = outcome(gate, relocated)
        record(tmp_path, mutation, result)
        assert not result["accepted"]
    finally:
        path.write_bytes(raw)


@pytest.mark.parametrize("mutation", ("lost-head", "input-substitution", "cpu-label", "opposite-ray"))
def test_numerical_negative_controls(gate, tmp_path, mutation):
    path = RUN / REPORT_DIR / "pytest-tmp/test_cuda_rotated_extreme_coor0/rotated-conic-extreme.json"
    saved = json.loads(path.read_bytes())
    identity = "tests.test_rotated_conic_cuda_extremes::" + saved["nodeid"].split("::")[1]
    if mutation == "lost-head":
        saved["positive_hex"][0] = "0x0.0p+0"
    elif mutation == "input-substitution":
        saved["input_hex"][-1] = "0x0.0p+0"
    elif mutation == "cpu-label":
        saved["observed_device"] = "cpu"
    else:
        saved["negative_hex"] = deepcopy(saved["positive_hex"])
    with pytest.raises(Exception) as failure:
        gate.replay(saved, identity, saved["source_sha256"], saved["gpu_name"])
    record(tmp_path, mutation, {"accepted": False, "error": str(failure.value)})


def test_caller_status_pin_rejects_unrepinned_mutation(gate, relocated):
    args = arguments(relocated)
    path = relocated / STATUS
    raw = path.read_bytes()
    try:
        path.write_bytes(raw + b" ")
        with pytest.raises(Exception, match="caller pin mismatch"):
            gate.accept(**args)
    finally:
        path.write_bytes(raw)


@pytest.mark.parametrize("location", ("snapshot", "current"))
def test_changed_source_is_rejected(gate, relocated, tmp_path, location):
    if location == "snapshot":
        path = relocated / "source/src/openopt/native_conic.py"
    else:
        current = tmp_path / "trusted-current-copy"
        shutil.copytree(ROOT / "src/openopt", current / "src/openopt", copy_function=shutil.copyfile)
        path = current / "src/openopt/native_conic.py"
    raw = path.read_bytes()
    try:
        path.write_bytes(raw + b"\n# independent source mutation\n")
        args = arguments(relocated)
        if location == "current":
            args["current_root"] = current
        with pytest.raises(Exception, match="source bidirectional difference"):
            gate.accept(**args)
    finally:
        path.write_bytes(raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=TARGET)
    parser.add_argument("--target-sha256", default=GATE_PIN)
    parser.add_argument("--target-commit", default=TARGET_COMMIT)
    parser.add_argument("--selector")
    parser.add_argument("--scan-only", action="store_true")
    args = parser.parse_args(argv)
    artifact = args.artifact_dir.resolve()
    assert artifact.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve())
    artifact.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="review-", dir=artifact))
    if args.scan_only:
        scanner_path = Path("/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
        spec = importlib.util.spec_from_file_location("independent_safe_scan", scanner_path)
        scanner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(scanner)
        scan_args = argparse.Namespace(
            secret_hex=None, secrets_file=None, no_needle=False,
            needle_from=str(Path.home() / ".config/dsh-n600-token"),
        )
        needles = scanner.read_needles(scan_args)
        assert needles, "credential scan requires nonempty in-memory needles"
        paths = [str(Path(__file__).resolve()), str(ROOT / "docs/research/rsoc-extremes-independent.md")]
        literal, _ = scanner.scan_literal(paths, needles)
        grep, _, _ = scanner.scan_grep(paths)
        pattern = re.compile(
            r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|" r"IDENTIFIED" r" BY|"
            r"""(?i:password|token|secret)\s*[\"']?\s*[:=]\s*[\"'][^\"'\s]{6,}[\"']"""
        )
        shape = sum(bool(pattern.search(Path(path).read_text())) for path in paths)
        put(output / "scan.json", {"verdict": "PASS" if literal == shape == 0 else "FAIL",
                                  "literal_hits": literal, "shape_hits": shape,
                                  "semantic_name_hits": grep,
                                  "semantic_review": "scanner pattern/file-path references; no values",
                                  "paths": paths})
        print(f"INDEPENDENT_SAFE_SCAN literal_hits={literal} shape_hits={shape} report={output / 'scan.json'}")
        return 0 if literal == shape == 0 else 1
    target_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.target_root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    target_status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=args.target_root, capture_output=True, text=True, check=True,
    ).stdout
    assert target_commit == args.target_commit and not target_status, "caller-selected clean commit required"
    snapshot = output / "reviewed-source"
    (snapshot / "scripts").mkdir(parents=True)
    pins = {GATE: args.target_sha256, HELPER: HELPER_PIN}
    for name, pin in pins.items():
        original = args.target_root.resolve() / name
        assert not original.is_symlink() and original.is_file()
        data = original.read_bytes()
        assert sha(data) == pin, ("review target pin", name)
        (snapshot / name).write_bytes(data)
        assert original.read_bytes() == data, ("target changed during capture", name)
    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": "-1", "PYTHONDONTWRITEBYTECODE": "1",
                "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                "PYTHONPATH": str(ROOT), "RSOC_REVIEW_ROOT": str(snapshot),
                "RSOC_REVIEW_SHA256": args.target_sha256, "TMPDIR": str(output)})
    command = [sys.executable, "-m", "pytest", "-q", "-ra", str(Path(__file__).resolve()),
               "--basetemp", str(output / "pytest-tmp"), "--junitxml", str(output / "pytest.xml"),
               "-o", f"cache_dir={output / 'cache'}"]
    if args.selector:
        command += ["-k", args.selector]
    start = datetime.now(timezone.utc)
    source_before = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()
    test_before = sha(Path(__file__).read_bytes())
    with (output / "pytest.log").open("x", encoding="utf-8") as log:
        log.write(f"command={command!r}\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        log.write(f"\nreturncode={result.returncode}\n")
    counts = None
    if (output / "pytest.xml").exists():
        suites = ET.parse(output / "pytest.xml").getroot()
        counts = {name: sum(int(s.get(name, "0")) for s in suites.findall("testsuite"))
                  for name in ("tests", "failures", "errors", "skipped")}
    receipt = {
        "verdict": "PASS" if result.returncode == 0 and counts and counts["tests"] > 0 else "FAIL",
        "returncode": result.returncode, "counts": counts, "started_at": start.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "reviewed_source": {GATE: args.target_sha256, HELPER: HELPER_PIN},
        "reviewed_identity": {"commit": target_commit, "clean": True,
                              "root": str(args.target_root.resolve())},
        "independent_test_sha256": sha(Path(__file__).read_bytes()),
        "independent_source": {"commit": commit, "clean": not bool(source_before),
                               "status": source_before},
        "source_consistency": sha(Path(__file__).read_bytes()) == test_before
        and subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True, check=True).stdout == source_before,
        "original_pins": PINS, "qualification": "read-only CPU audit; not fresh CUDA/origin evidence",
        "command": command,
        "artifacts": {str(p.relative_to(output)): sha(p.read_bytes())
                      for p in sorted(output.rglob("*")) if p.is_file()},
    }
    if not receipt["source_consistency"]:
        receipt["verdict"] = "FAIL"
    put(output / "receipt.json", receipt)
    print(f"RSOC_INDEPENDENT_REVIEW verdict={receipt['verdict']} receipt={output / 'receipt.json'}", flush=True)
    return result.returncode if result.returncode else (0 if receipt["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
