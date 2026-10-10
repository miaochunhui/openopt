"""Independent frozen renderer binding review; never issue numeric acceptance.

The original 32 tests execute unchanged in a separate compatibility process.
New envelopes are test-only contract data, even when carrying real byte-pinned
historical bundles. Artifact Python is read, never imported or executed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile
import time
from xml.etree import ElementTree as ET

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import render_release_report as renderer
from scripts.verify import check_source_consistency, source_snapshot

SUBJECT = "9a21d79d07d82b6df4f3f6946e6c214245bce24a"
OWNER_PINS = {
    "scripts/render_release_report.py": "01978efd35db8aa9a96ea532f4b135e6874fa51a1f4d193d515659f5866faf6f",
    "tests/test_render_release_report.py": "1576d9d262fa79856e887ea4e70aa33b9112ac93421b3b3029c765fda0d52699",
    "docs/research/release-report.md": "d43351007cdf91e943aefa5ec79d5b3e681a476732fa00f26692ff78e17898a1",
}
ORIGINAL_ROOT = Path("/data/dev/openopt-wt-release4-20261010-rsoc-report-independent")
ORIGINAL_COMMIT = "1ad740a741a31c19171f60ea7760e2a74ffc9254"
ORIGINAL_PATH = ORIGINAL_ROOT / "tests/test_rsoc_report_independent.py"
ORIGINAL_PIN = "4b0049e5dfe76bff581fc4c9b2a81d727553862c12d908e556cb19ab8cb457b9"
REAL_ROOT = Path("/hf3fs-jd/data/tmp/openopt-release-20261010")
OLD_REVIEW = REAL_ROOT / "rsoc-report-independent/clean-real/review-sv9xq1hj/review.json"
OLD_REVIEW_PIN = "5068a9ff4058ffcc32e180211fd605ee70291ab2610aa126f91f6f700dec848b"
AUDITOR_COMMIT = "75b54762e9fd37e334f9f34064bd7436c55c1250"
AUDITOR_PIN = "b249e9a3f13e422c5d992c78411aaeb070735750b73900eee323dd8c5b6809c0"
SELF = "tests/test_report_binding_repair_independent.py"
ORIGINAL_TARGET = "tests/test_rsoc_report_independent.py"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = value if isinstance(value, bytes) else encoded(value)
    path.write_bytes(payload)
    return {"path": str(path), "sha256": digest(payload)}


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, timeout=20)
    assert result.returncode == 0, f"trusted Git query failed rc={result.returncode}"
    return result.stdout


def owner_pins():
    for name, pin in OWNER_PINS.items():
        assert digest((ROOT / name).read_bytes()) == pin, f"frozen owner drift: {name}"
        assert digest(git("show", SUBJECT + ":" + name)) == pin
    assert Path(renderer.__file__).resolve() == ROOT / "scripts/render_release_report.py"


def original_bytes():
    payload = ORIGINAL_PATH.read_bytes()
    assert digest(payload) == ORIGINAL_PIN
    assert git("show", ORIGINAL_COMMIT + ":" + ORIGINAL_TARGET) == payload
    return payload


def original_module():
    original_bytes()
    name = "_report_binding_original_review"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, ORIGINAL_PATH)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        assert module.renderer is renderer
    return sys.modules[name]


def cpu_env(output):
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="-1", PYTHONDONTWRITEBYTECODE="1",
               PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", PYTEST_ADDOPTS="", PYTEST_PLUGINS="",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               TMPDIR=str(output),
               PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"))))
    for key in ("OPENOPT_SHARD_COLLECTION", "OPENOPT_SHARD_GROUPING", "OPENOPT_SHARD_FIXTURE_FILES"):
        env.pop(key, None)
    return env


class Case:
    """Independent caller selection; only explicit trust methods repin authority."""

    def __init__(self, root, template=None):
        old = original_module()
        if template is None:
            self.data = old.Fixture(root)
            self.authority = None
            self.authority_value = None
        else:
            root.mkdir()
            self.data = object.__new__(old.Fixture)
            self.data.root = root
            self.data.records = deepcopy(template.data.records)
            self.data.descriptors = deepcopy(template.data.descriptors)
            self.data.manifest = deepcopy(template.data.manifest)
            self.data.path = root / "input-manifest.json"
            self.data.seal()
            self.authority_value = deepcopy(template.authority_value)
            self.authority = None
            if self.authority_value is not None:
                self.authority = write(root / "test-only-authority.json", self.authority_value)

    @property
    def records(self):
        return self.data.records

    def put(self, role, value):
        return self.data.put(role, value)

    def render(self):
        return renderer.generate(self.data.path, self.data.pin, rsoc_authority=self.authority)

    def fresh_v2_envelope(self):
        """New fixture record, not an edit/backfill of any historical receipt."""
        bundle = self.records["rotated_cuda_bundle"]
        assert bundle["source"]["audit_commit"] == AUDITOR_COMMIT
        source = git("show", AUDITOR_COMMIT + ":scripts/accept_rotated_cuda.py")
        assert digest(source) == AUDITOR_PIN
        source_descriptor = write(self.data.root / "nonexecuted-auditor-source.py", source)
        evidence = {
            "schema_version": 2, "verdict": "PASS", "mode": "accept",
            "checked_at": "2026-10-10T14:00:00+00:00",
            "pins_sha256": self.data.descriptors["rotated_cuda_pins"]["sha256"],
            "bundle_sha256": self.data.descriptors["rotated_cuda_bundle"]["sha256"],
            "auditor_sha256": AUDITOR_PIN, "audit_commit": AUDITOR_COMMIT,
            "audit_runtime": {"python": platform.python_version(),
                              "implementation": platform.python_implementation()},
            "certified": False, "qualified": False, "false_accept_count": 0,
            "commercial": "UNVERIFIED", "performance": "UNVERIFIED",
        }
        self.put("rotated_cuda_acceptance", evidence)
        self.authority_value = {
            "schema_version": 1, "receipts": {"rotated_cuda": {
                "acceptance_sha256": self.data.descriptors["rotated_cuda_acceptance"]["sha256"],
                "bundle_sha256": evidence["bundle_sha256"],
                "auditor": {"commit": AUDITOR_COMMIT, **source_descriptor, "bytes": len(source)},
            }},
        }
        self.authority = write(self.data.root / "test-only-authority.json", self.authority_value)
        write(self.data.root / "NOT-ISSUED-NUMERICAL-ACCEPTANCE.json", {
            "qualification": "TEST-ONLY V2 ENVELOPE; NOT PRODUCER ISSUED",
            "historical_receipt_modified": False, "solver_calls": 0,
            "artifact_source_executed": False, "actual_new_numeric_accept": False,
            "bundle_sha256": evidence["bundle_sha256"],
        })

    def trust_acceptance_only(self):
        self.authority_value["receipts"]["rotated_cuda"]["acceptance_sha256"] = (
            self.data.descriptors["rotated_cuda_acceptance"]["sha256"])
        self.authority = write(self.data.root / "test-only-authority.json", self.authority_value)

    def trust_internal_bundle_for_gate_probe(self):
        pin = self.data.descriptors["rotated_cuda_bundle"]["sha256"]
        self.records["rotated_cuda_acceptance"]["bundle_sha256"] = pin
        self.put("rotated_cuda_acceptance", self.records["rotated_cuda_acceptance"])
        self.authority_value["receipts"]["rotated_cuda"]["bundle_sha256"] = pin
        self.trust_acceptance_only()

    def copy_fixed13_archive_for_junit_probe(self):
        """Relocate byte-pinned originals; never mutate historical artifacts."""
        pins = deepcopy(self.records["rotated_cuda_pins"])
        bundle = self.records["rotated_cuda_bundle"]
        original = Path(pins["run_root"]).resolve()
        target = self.data.root / "copied-real-fixed13"

        def relocated(path):
            return target / Path(path).resolve().relative_to(original)

        for descriptor in [*pins["files"].values(), *bundle["sidecars"]]:
            payload = Path(descriptor["path"]).read_bytes()
            assert digest(payload) == descriptor["sha256"]
            write(relocated(descriptor["path"]), payload)
        for name, pin in self.records["rotated_cuda_run"]["source_hashes"].items():
            payload = (original / "source" / name).read_bytes()
            assert digest(payload) == pin
            write(target / "source" / name, payload)
        run = deepcopy(self.records["rotated_cuda_run"])
        for job in run["jobs"]:
            for key in ("output", "manifest"):
                job[key] = str(relocated(job[key]))
        write(relocated(pins["files"]["run"]["path"]), run)
        tree = ET.fromstring(self.records["rotated_cuda_junit"])
        for prop in tree.findall(".//property"):
            if prop.attrib["name"].endswith("_path"):
                prop.attrib["value"] = str(relocated(prop.attrib["value"]))
        write(relocated(pins["files"]["junit"]["path"]), ET.tostring(tree))
        for descriptor in [*pins["files"].values(), *bundle["sidecars"]]:
            descriptor["path"] = str(relocated(descriptor["path"]))
            payload = Path(descriptor["path"]).read_bytes()
            descriptor["sha256"] = digest(payload)
            if "bytes" in descriptor:
                descriptor["bytes"] = len(payload)
        pins["run_root"] = str(target)
        for role in ("run", "result", "junit"):
            descriptor = pins["files"][role]
            self.data.link("rotated_cuda_" + role, Path(descriptor["path"]), descriptor["sha256"])
        self.put("rotated_cuda_pins", pins)
        bundle["caller_pins"] = deepcopy(pins)
        self.put("rotated_cuda_bundle", bundle)
        evidence = self.records["rotated_cuda_acceptance"]
        evidence["pins_sha256"] = self.data.descriptors["rotated_cuda_pins"]["sha256"]
        self.put("rotated_cuda_acceptance", evidence)
        self.trust_internal_bundle_for_gate_probe()

    def cli(self, output, check=False, authority=True):
        command = [sys.executable, "-B", str(ROOT / "scripts/render_release_report.py"),
                   "--manifest", str(self.data.path), "--manifest-sha256", self.data.pin,
                   "--output-dir", str(output)]
        if authority and self.authority is not None:
            command += ["--rsoc-authority", self.authority["path"],
                        "--rsoc-authority-sha256", self.authority["sha256"]]
        if check:
            command.append("--check")
        result = subprocess.run(command, env=cpu_env(self.data.root), cwd=ROOT,
                                capture_output=True, text=True, timeout=60)
        write(self.data.root / (output.name + ("-check" if check else "-write") + ".log"),
              (result.stdout + result.stderr + f"\nreturncode={result.returncode}\n").encode())
        return result


def real_selection(root):
    case = Case(root)
    old = original_module()
    for role, (relative, pin) in old.REAL_PINS.items():
        target = REAL_ROOT / relative
        assert digest(target.read_bytes()) == pin, f"historical drift: {role}"
        case.data.link(role, target, pin)
    fixed = case.records["rotated_cuda_pins"]["files"]
    extremes = case.records["rsoc_extremes_acceptance"]["inputs"]
    for prefix, originals in (("rotated_cuda", list(fixed.values())), ("rsoc_extremes", extremes)):
        selected = {role: fixed[role] for role in ("run", "result", "junit")} if prefix == "rotated_cuda" else {
            "run": next(pin for pin in originals if Path(pin["path"]).name == "run.json"),
            "junit": next(pin for pin in originals if Path(pin["path"]).name == "pytest.xml"),
            "result": next(pin for pin in originals if Path(pin["path"]).name == "result.json"
                           and Path(pin["path"]).parent.name.startswith("cuda-check-")),
        }
        for role, pin in selected.items():
            case.data.link(prefix + "_" + role, Path(pin["path"]), pin["sha256"])
    return case


def inspect_real_originals(case):
    old = original_module()
    result = {}
    for prefix, total in (("rotated_cuda", 13), ("rsoc_extremes", 76)):
        originals = (list(case.records["rotated_cuda_pins"]["files"].values())
                     + case.records["rotated_cuda_bundle"]["sidecars"]) if total == 13 else (
                         case.records["rsoc_extremes_acceptance"]["inputs"])
        assert len(originals) == 11 + total
        for pin in originals:
            payload = Path(pin["path"]).read_bytes()
            assert digest(payload) == pin["sha256"]
            if "bytes" in pin:
                assert len(payload) == pin["bytes"]
        cases = list(ET.fromstring(case.records[prefix + "_junit"]).iter("testcase"))
        names = [c.attrib["classname"] + "::" + c.attrib["name"] for c in cases]
        assert len(names) == len(set(names)) == total
        assert set(names) == set(old.identities(total))
        assert not any(c.find(tag) is not None for c in cases for tag in ("failure", "error", "skipped"))
        result[prefix] = {"tests": total, "original_pins": len(originals),
                          "source": case.records[prefix + "_run"]["source_sha"]}
    bundle = case.records["rotated_cuda_bundle"]
    assert bundle["runtime"]["cuda_execution"] is True
    for member in bundle["numeric_cases"]:
        if member["kind"] == "original_model":
            metrics = member["metrics"]
            assert Fraction(metrics["primal_objective"]) - Fraction(metrics["dual_objective"]) == (
                Fraction(metrics["signed_duality_gap"]))
    return result


@pytest.fixture(scope="module")
def real_template(tmp_path_factory):
    owner_pins()
    case = real_selection(tmp_path_factory.mktemp("real-parent") / "evidence")
    inspect_real_originals(case)
    original_bundle = case.data.descriptors["rotated_cuda_bundle"]["sha256"]
    case.fresh_v2_envelope()
    assert case.data.descriptors["rotated_cuda_bundle"]["sha256"] == original_bundle
    assert case.render()
    yield case
    inspect_real_originals(case)
    owner_pins()
    original_bytes()


@pytest.fixture
def selected(tmp_path, real_template):
    return Case(tmp_path / "isolated-inputs", real_template)


def reject(case, request, code, *, cli=False):
    try:
        files = case.render()
    except renderer.Rejected as error:
        observed = {"accepted": False, "code": error.code, "coordinate": error.coordinate}
    else:
        observed = {"accepted": True, "summary": json.loads(files["summary.json"])}
    path = case.data.root / "negative.json"
    write(path, {"subject": SUBJECT, "manifest_sha256": case.data.pin,
                 "authority": case.authority, "expected_code": code, **observed})
    request.node.user_properties.append(("report_binding_negative", str(path)))
    assert observed["accepted"] is False, "actual false accept on a valid v2 starting point"
    assert observed["code"] == code, observed
    if cli:
        output = case.data.root / "must-not-exist"
        result = case.cli(output)
        assert result.returncode == 1 and "code=" + code in result.stderr
        assert not output.exists()


def test_frozen_subject_original_test_and_retained_six_red_controls():
    owner_pins()
    original_bytes()
    assert digest(OLD_REVIEW.read_bytes()) == OLD_REVIEW_PIN
    record = json.loads(OLD_REVIEW.read_bytes())
    assert record["review_source"]["clean"] is True
    assert record["review_source"]["commit"] == ORIGINAL_COMMIT
    assert record["tests"] == {"tests": 32, "failures": 6, "errors": 0, "skipped": 0}
    assert record["passed"] == 26 and record["verdict"] == "BLOCKED"


def test_valid_v2_real_bundle_and_real76_cli_readback_without_backfill(selected, request):
    observations = inspect_real_originals(selected)
    files = selected.render()
    assert selected.render() == files
    scalars = json.loads(files["summary.json"])["scalars"]
    for prefix, total in (("rotated_cuda", 13), ("rsoc_extremes", 76)):
        assert scalars[prefix + "_status"] == "PASS" and scalars[prefix + "_tests"] == total
        assert scalars[prefix + "_source_relation"] == "historical"
    assert scalars["commercial_status"] == scalars["online_status"] == "UNVERIFIED"
    private = json.loads(files["report-bundle.json"])
    assert private["rsoc_authority"] == selected.authority
    public = b"\n".join(value for key, value in files.items() if key != "report-bundle.json")
    assert b"nonexecuted-auditor-source" not in public and b"/3fs/" not in public
    output = selected.data.root / "actual-cli"
    result = selected.cli(output)
    assert result.returncode == 0, result.stderr
    assert {path.name: path.read_bytes() for path in output.iterdir()} == files
    assert selected.cli(output, check=True).returncode == 0
    path = selected.data.root / "positive.json"
    write(path, {"verdict": "TEST_ONLY_V2_BINDING_PASS", "observations": observations,
                 "real_bundle_unchanged": True, "new_numeric_acceptance_issued": False,
                 "solver_calls": 0, "historical_receipt_modified": False,
                 "authority": selected.authority, "actual_cli_returncode": result.returncode})
    request.node.user_properties.append(("report_binding_positive", str(path)))


@pytest.mark.parametrize("total,field,value", (
    (13, "qualified", True), (13, "certified", True),
    (76, "certified", True), (76, "false_accept_count", 1)))
def test_original_four_conflicting_flags_rehashed_reject_valid_baseline(selected, request, total, field, value):
    role = ("rotated_cuda" if total == 13 else "rsoc_extremes") + "_acceptance"
    selected.records[role][field] = value
    selected.put(role, selected.records[role])
    if total == 13:
        selected.trust_acceptance_only()
    reject(selected, request, "rsoc_qualification_overclaim" if total == 13 else (
        "rsoc_extremes_success_schema"), cli=field == "qualified")


def test_original_auditor_zero_resealed_acceptance_is_not_source_authority(selected, request):
    selected.records["rotated_cuda_acceptance"]["auditor_sha256"] = "0" * 64
    selected.put("rotated_cuda_acceptance", selected.records["rotated_cuda_acceptance"])
    selected.trust_acceptance_only()
    reject(selected, request, "rsoc_fixed13_auditor_authority_mismatch", cli=True)


@pytest.mark.parametrize("producer_rehash", [False, True])
def test_real_signedgap_bundle_rehash_rejected_without_trusted_bundle_repin(selected, request, producer_rehash):
    original_accept = selected.data.descriptors["rotated_cuda_acceptance"]["sha256"]
    original_bundle = selected.data.descriptors["rotated_cuda_bundle"]["sha256"]
    selected.records["rotated_cuda_bundle"]["numeric_cases"][0]["metrics"]["signed_duality_gap"] = "999999999"
    selected.put("rotated_cuda_bundle", selected.records["rotated_cuda_bundle"])
    assert selected.data.descriptors["rotated_cuda_bundle"]["sha256"] != original_bundle
    if producer_rehash:
        selected.records["rotated_cuda_acceptance"]["bundle_sha256"] = (
            selected.data.descriptors["rotated_cuda_bundle"]["sha256"])
        selected.put("rotated_cuda_acceptance", selected.records["rotated_cuda_acceptance"])
        selected.trust_acceptance_only()
    else:
        assert selected.data.descriptors["rotated_cuda_acceptance"]["sha256"] == original_accept
    reject(selected, request, "rsoc_fixed13_bundle_binding", cli=not producer_rehash)


@pytest.mark.parametrize("mutation", ("count", "drop", "duplicate", "hidden", "baseline", "moreau"))
def test_original_numeric_gate_controls_survive_explicit_internal_byte_authority(selected, request, mutation):
    bundle = selected.records["rotated_cuda_bundle"]
    if mutation == "count":
        bundle["denominator"]["total"] = 12
        code = "rsoc_fixed13_bundle_contract"
    elif mutation == "drop":
        bundle["numeric_cases"].pop()
        code = "rsoc_numeric_membership"
    elif mutation == "duplicate":
        bundle["numeric_cases"][-1] = deepcopy(bundle["numeric_cases"][0])
        code = "rsoc_numeric_membership"
    elif mutation == "hidden":
        bundle["numeric_cases"][0]["checks"]["stationarity"] = False
        code = "rsoc_hidden_numeric_failure"
    elif mutation == "baseline":
        bundle["numeric_cases"][-1]["baseline_passed"] = False
        code = "rsoc_expected_rejection_missing"
    else:
        bundle["numeric_cases"][4]["projection"]["moreau_orthogonal"] = False
        code = "rsoc_projection_checks"
    selected.put("rotated_cuda_bundle", bundle)
    selected.trust_internal_bundle_for_gate_probe()
    reject(selected, request, code)


@pytest.mark.parametrize("mutation", ("source", "junit", "sidecar"))
def test_original_archive_gates_reject_coordinated_v2_rehash(selected, request, mutation):
    bundle = selected.records["rotated_cuda_bundle"]
    if mutation == "source":
        bundle["source"]["execution_commit"] = "0" * 40
        code = "rsoc_audit_source_mismatch"
    elif mutation == "sidecar":
        bundle["sidecars"][0]["sha256"] = "0" * 64
        code = "changed_hash"
    else:
        selected.copy_fixed13_archive_for_junit_probe()
        assert selected.render(), "relocated real archive must be valid before the mutation"
        bundle = selected.records["rotated_cuda_bundle"]
        pins = selected.records["rotated_cuda_pins"]
        tree = ET.fromstring(selected.records["rotated_cuda_junit"])
        cases = list(tree.iter("testcase"))
        assert len(cases) == 13
        cases[1].attrib.update(classname=cases[0].attrib["classname"],
                               name=cases[0].attrib["name"])
        payload = ET.tostring(tree)
        descriptor = pins["files"]["junit"]
        replacement = write(Path(descriptor["path"]), payload)
        descriptor.update(replacement)
        if "bytes" in descriptor:
            descriptor["bytes"] = len(payload)
        selected.data.link("rotated_cuda_junit", Path(descriptor["path"]), descriptor["sha256"])
        selected.put("rotated_cuda_pins", pins)
        bundle["caller_pins"] = deepcopy(pins)
        evidence = selected.records["rotated_cuda_acceptance"]
        evidence["pins_sha256"] = selected.data.descriptors["rotated_cuda_pins"]["sha256"]
        selected.put("rotated_cuda_acceptance", evidence)
        code = "rsoc_junit_membership"
    selected.put("rotated_cuda_bundle", bundle)
    selected.trust_internal_bundle_for_gate_probe()
    reject(selected, request, code)


@pytest.mark.parametrize("mutation", ("schema_bool", "schema_float", "old", "unknown",
                                     "counter_bool", "counter_float", "runtime_empty", "runtime_type"))
def test_strict_version_runtime_and_counter_types_not_relaxed_by_rehash(selected, request, mutation):
    evidence = selected.records["rotated_cuda_acceptance"]
    code = "rsoc_fixed13_success_schema_unsupported"
    if mutation.startswith("schema_"):
        evidence["schema_version"] = True if mutation == "schema_bool" else 2.0
    elif mutation == "old":
        evidence.pop("schema_version")
    elif mutation == "unknown":
        evidence["new_trusted_flag"] = True
    elif mutation.startswith("counter_"):
        evidence["false_accept_count"] = False if mutation == "counter_bool" else 0.0
        code = "rsoc_qualification_overclaim"
    else:
        evidence["audit_runtime"]["python"] = "" if mutation == "runtime_empty" else 310
        code = "rsoc_audit_runtime_schema"
    selected.put("rotated_cuda_acceptance", evidence)
    selected.trust_acceptance_only()
    reject(selected, request, code)


@pytest.mark.parametrize("mutation", ("missing", "stale", "resealed_not_repinned", "source_size", "foreign_source"))
def test_independent_authority_and_nonexecuted_committed_source_negative(selected, request, mutation):
    entry = selected.authority_value["receipts"]["rotated_cuda"]
    code = "rsoc_fixed13_authority_required"
    if mutation == "missing":
        selected.authority = None
    elif mutation == "stale":
        selected.authority["sha256"] = "0" * 64
        code = "changed_hash"
    elif mutation == "resealed_not_repinned":
        entry["auditor"]["bytes"] += 1
        write(Path(selected.authority["path"]), selected.authority_value)
        code = "changed_hash"
    elif mutation == "source_size":
        entry["auditor"]["bytes"] += 1
        selected.authority = write(Path(selected.authority["path"]), selected.authority_value)
        code = "rsoc_fixed13_auditor_source_bytes"
    else:
        marker = selected.data.root / "ARTIFACT-MUST-NOT-EXECUTE"
        payload = f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n".encode()
        source = write(selected.data.root / "foreign-source.py", payload)
        entry["auditor"].update(source, bytes=len(payload))
        selected.records["rotated_cuda_acceptance"]["auditor_sha256"] = source["sha256"]
        selected.put("rotated_cuda_acceptance", selected.records["rotated_cuda_acceptance"])
        selected.trust_acceptance_only()
        code = "rsoc_fixed13_committed_auditor_source"
    reject(selected, request, code, cli=mutation == "foreign_source")
    if mutation == "foreign_source":
        assert not marker.exists()


def test_historical_unbound_receipt_remains_unsupported_never_backfilled(selected, request):
    old = original_module()
    relative, pin = old.REAL_PINS["rotated_cuda_acceptance"]
    payload = (REAL_ROOT / relative).read_bytes()
    assert digest(payload) == pin
    selected.data.link("rotated_cuda_acceptance", REAL_ROOT / relative, pin)
    selected.authority = None
    reject(selected, request, "rsoc_fixed13_success_schema_unsupported")
    assert (REAL_ROOT / relative).read_bytes() == payload


def test_optional_fail_and_omission_preserved_without_unused_authority(tmp_path, request):
    case = Case(tmp_path / "legacy")
    scalars = json.loads(case.render()["summary.json"])["scalars"]
    assert not any(name.startswith(("rotated_cuda_", "rsoc_extremes_")) for name in scalars)
    for prefix in ("rotated_cuda", "rsoc_extremes"):
        case.put(prefix + "_acceptance", {"verdict": "FAIL", "checked_at": "2026-10-10T14:00:00+00:00"})
    scalars = json.loads(case.render()["summary.json"])["scalars"]
    assert scalars["rotated_cuda_status"] == scalars["rsoc_extremes_status"] == "FAIL"
    assert "rotated_cuda_tests" not in scalars and "rsoc_extremes_tests" not in scalars
    case.authority = write(case.data.root / "unused-authority.json", {
        "schema_version": 1, "receipts": {"rotated_cuda": {}}})
    reject(case, request, "rsoc_unused_authority")


def xml_result(path, collected, classname):
    cases = list(ET.parse(path).getroot().iter("testcase"))
    expected = Counter((classname, name) for name in collected)
    actual = Counter((case.attrib["classname"], case.attrib["name"]) for case in cases)
    membership = {
        "expected": ["::".join(item) for item in sorted(expected.elements())],
        "observed": ["::".join(item) for item in sorted(actual.elements())],
        "missing": ["::".join(item) for item in sorted((expected - actual).elements())],
        "extra": ["::".join(item) for item in sorted((actual - expected).elements())],
        "duplicates": ["::".join(item) for item, count in sorted(actual.items()) if count != 1],
    }
    assert expected and expected == actual and all(count == 1 for count in actual.values()), membership
    counts = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    failures = {}
    for case in cases:
        tags = [tag for tag in ("failure", "error", "skipped") if case.find(tag) is not None]
        assert len(tags) <= 1
        for tag in tags:
            counts[{"failure": "failures", "error": "errors", "skipped": "skipped"}[tag]] += 1
            if tag != "skipped":
                failures[case.attrib["name"]] = case.find(tag).attrib.get("message", "")
    leaves = [suite for suite in ET.parse(path).getroot().iter("testsuite")
              if not suite.findall("testsuite")]
    assert {key: sum(int(suite.attrib[key]) for suite in leaves) for key in counts} == counts
    return counts, failures, sum(float(suite.attrib["time"]) for suite in leaves), membership


@pytest.mark.parametrize("mutation", ("classname", "duplicate", "missing", "extra"))
def test_bidirectional_junit_oracle_rejects_identity_forgeries(tmp_path, mutation):
    path = tmp_path / "oracle.xml"
    tree = ET.Element("testsuites")
    suite = ET.SubElement(tree, "testsuite", tests="1", failures="0", errors="0",
                          skipped="0", time="0")
    leaf = ET.SubElement(suite, "testcase", classname="independent", name="positive")
    write(path, ET.tostring(tree))
    counts, _, _, membership = xml_result(path, ["positive"], "independent")
    assert counts["tests"] == 1 and membership["missing"] == membership["extra"] == []
    if mutation == "classname":
        leaf.attrib["classname"] = "foreign"
    elif mutation == "missing":
        suite.remove(leaf)
        suite.attrib["tests"] = "0"
    else:
        ET.SubElement(suite, "testcase", classname="independent",
                      name="positive" if mutation == "duplicate" else "unselected")
        suite.attrib["tests"] = "2"
    write(path, ET.tostring(tree))
    with pytest.raises(AssertionError):
        xml_result(path, ["positive"], "independent")


def original_worker(output):
    """Execute unchanged original functions; ROOT routes only actual CLI source."""
    owner_pins()
    original_bytes()
    before = source_snapshot(ROOT)
    collected = []

    class RouteOnly:
        def pytest_collection_modifyitems(self, items):
            for item in items:
                assert Path(item.module.__file__).resolve() == ORIGINAL_PATH
                assert item.module.renderer is renderer
                item.module.ROOT = ROOT
                collected.append(item.name)

    os.environ.update(cpu_env(output), OPENOPT_RUN_RSOC_REPORT_EVIDENCE_TESTS="1")
    junit = output / "pytest.xml"
    rc = int(pytest.main(["--import-mode=importlib", "--rootdir", str(ORIGINAL_ROOT),
                         "-q", "-ra", "-o", "addopts=", "-o", f"cache_dir={output / 'cache'}",
                         "--basetemp", str(output / "pytest-tmp"), f"--junitxml={junit}",
                         str(ORIGINAL_PATH)], plugins=[RouteOnly()]))
    counts, failures, seconds, membership = xml_result(
        junit, collected, "tests.test_rsoc_report_independent")
    assert len(collected) == 32 and counts["errors"] == counts["skipped"] == 0
    incompatible = {
        "test_subject_exact_frozen_bytes_and_independent_domain",
        "test_selfcontained_complete_archive_is_only_historical_projection[13]",
        "test_fixed13_wrong_auditor_pin_must_not_receive_pass",
        "test_false_current_requires_clean_same_source_and_does_not_upgrade[13]",
        "test_actual_cli_rehashed_membership_negative_no_output[13]",
        "test_real_historical13_76_exact_receipt_source_junit_projection",
        "test_real_fixed13_bundle_can_drift_without_acceptance_pin_change",
        *{f"test_rehashed_false_counts_source_and_junit_rejected[{kind}-13]"
          for kind in ("count", "drop", "duplicate", "source", "junit", "sidecar")},
        *{f"test_invalid_fixed13_negative_and_projection_rejected[{kind}]"
          for kind in ("baseline", "required_false", "hidden_check", "moreau", "mode")},
        "test_unknown_success_overclaim_must_not_receive_pass[13-qualified-True]",
        "test_unknown_success_overclaim_must_not_receive_pass[13-certified-True]",
    }
    assert set(failures) == incompatible and counts["failures"] == 20 and rc == 1
    for name, message in failures.items():
        if name != "test_subject_exact_frozen_bytes_and_independent_domain":
            assert "rsoc_fixed13_success_schema_unsupported" in message, (name, message)
    consistency = check_source_consistency(before, source_snapshot(ROOT))
    assert consistency["verdict"] == "PASS"
    original_bytes()
    sys.stdout.flush()
    sys.stderr.flush()
    write(output / "original32.json", {
        "verdict": "CONTRACT_INCOMPATIBLE", "subject": SUBJECT,
        "original_commit": ORIGINAL_COMMIT, "original_sha256": ORIGINAL_PIN,
        "assertions_changed": False, "root_routing_only": True, "pytest_returncode": rc,
        "tests": counts, "passed": 12, "failures_by_first_cause": failures,
        "collection": collected, "membership": "PASS", "identity_membership": membership,
        "junit": {"path": str(junit), "sha256": digest(junit.read_bytes()), "bytes": junit.stat().st_size},
        "log": {"path": str(output / "process.log"),
                "sha256": digest((output / "process.log").read_bytes()),
                "bytes": (output / "process.log").stat().st_size},
        "source_consistency": consistency,
        "source": before, "pytest_seconds": seconds, "finished_at": datetime.now(timezone.utc).isoformat(),
    })
    return 0


def test_original32_byte_unchanged_execution_has_explicit_contract_incompatibility(tmp_path, request):
    output = tmp_path / "original32"
    output.mkdir()
    command = [sys.executable, "-B", str(ROOT / SELF), "--original32-worker",
               "--artifact-dir", str(output)]
    with (output / "process.log").open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=cpu_env(output), stdout=log,
                                stderr=subprocess.STDOUT, timeout=150)
    assert result.returncode == 0, (output / "process.log").read_text()[-4000:]
    receipt = json.loads((output / "original32.json").read_bytes())
    assert receipt["verdict"] == "CONTRACT_INCOMPATIBLE"
    assert receipt["tests"] == {"tests": 32, "failures": 20, "errors": 0, "skipped": 0}
    for key in ("junit", "log"):
        artifact = receipt[key]
        payload = Path(artifact["path"]).read_bytes()
        assert digest(payload) == artifact["sha256"] and len(payload) == artifact["bytes"]
    request.node.user_properties.append(("report_binding_original32", str(output / "original32.json")))


def review(output_root):
    owner_pins()
    original_bytes()
    assert not output_root.resolve().is_relative_to(ROOT)
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="review-", dir=output_root))
    before = source_snapshot(ROOT)
    started = time.monotonic()
    receipt = {"verdict": "RUNNING", "subject_commit": SUBJECT, "owner_sha256": OWNER_PINS,
               "reviewer": before, "started_at": datetime.now(timezone.utc).isoformat(),
               "qualification": "bounded renderer binding only; no issued v2 numeric acceptance"}
    env = cpu_env(output)
    common = [sys.executable, "-B", "-m", "pytest", "-o", "addopts=",
              "-o", f"cache_dir={output / 'cache'}"]
    try:
        collection = subprocess.run([*common, "--collect-only", "-q", SELF], cwd=ROOT,
                                    env=env, capture_output=True, text=True, timeout=30)
        write(output / "collection.log", (collection.stdout + collection.stderr).encode())
        assert collection.returncode == 0
        nodes = [line[len(SELF) + 2:] for line in collection.stdout.splitlines()
                 if line.startswith(SELF + "::")]
        assert nodes and len(nodes) == len(set(nodes))
        junit = output / "pytest.xml"
        with (output / "pytest.log").open("x") as log:
            result = subprocess.run([*common, "-q", "-ra", "--basetemp", str(output / "pytest-tmp"),
                                     f"--junitxml={junit}", SELF], cwd=ROOT, env=env,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=240)
        counts, failures, seconds, membership = xml_result(
            junit, nodes, "tests.test_report_binding_repair_independent")
        assert counts["skipped"] == 0 and result.returncode == (1 if failures else 0)
        consistency = check_source_consistency(before, source_snapshot(ROOT))
        assert consistency["verdict"] == "PASS"
        owner_pins()
        original_bytes()
        artifacts = []
        for case in ET.parse(junit).getroot().iter("testcase"):
            for prop in case.findall("properties/property"):
                if prop.attrib.get("name", "").startswith("report_binding_"):
                    path = Path(prop.attrib["value"]).resolve(strict=True)
                    assert path.is_relative_to(output.resolve())
                    payload = path.read_bytes()
                    artifacts.append({"kind": prop.attrib["name"],
                        "identity": case.attrib["classname"] + "::" + case.attrib["name"],
                        "path": str(path), "sha256": digest(payload), "bytes": len(payload)})
        assert len({item["path"] for item in artifacts}) == len(artifacts)
        negatives = [json.loads(Path(item["path"]).read_bytes()) for item in artifacts
                     if item["kind"] == "report_binding_negative"]
        if not failures:
            neutral = {
                "test_frozen_subject_original_test_and_retained_six_red_controls",
                "test_valid_v2_real_bundle_and_real76_cli_readback_without_backfill",
                "test_original32_byte_unchanged_execution_has_explicit_contract_incompatibility",
            }
            expected_negative = {
                "tests.test_report_binding_repair_independent::" + name for name in nodes
                if name not in neutral and not name.startswith("test_bidirectional_junit_oracle_")
            }
            actual_negative = [item["identity"] for item in artifacts
                               if item["kind"] == "report_binding_negative"]
            assert len(actual_negative) == len(set(actual_negative))
            assert set(actual_negative) == expected_negative
            assert all(value["accepted"] is False and value["code"] == value["expected_code"]
                       for value in negatives)
            assert sum(item["kind"] == "report_binding_positive" for item in artifacts) == 1
            assert sum(item["kind"] == "report_binding_original32" for item in artifacts) == 1
        receipt.update(verdict="BLOCKED" if failures else "PASS", tests=counts,
                       passed=counts["tests"] - counts["failures"] - counts["errors"],
                       pytest_returncode=result.returncode, failed_nodes=failures,
                       false_accept_count=sum(value["accepted"] is True for value in negatives),
                       negative_cases=len(negatives), collection=nodes, membership="PASS",
                       identity_membership=membership,
                       source_consistency=consistency, artifacts=artifacts, pytest_seconds=seconds,
                       junit_sha256=digest(junit.read_bytes()),
                       collection_sha256=digest((output / "collection.log").read_bytes()),
                       log_sha256=digest((output / "pytest.log").read_bytes()))
    except Exception as error:
        receipt.update(verdict="ERROR", first_failure=f"{type(error).__name__}: {error}")
        raise
    finally:
        receipt.update(finished_at=datetime.now(timezone.utc).isoformat(),
                       wall_seconds=time.monotonic() - started)
        path = output / "review.json"
        write(path, receipt)
        print(f"REPORT_BINDING_REVIEW verdict={receipt['verdict']} "
              f"receipt={path} sha256={digest(path.read_bytes())}", flush=True)
    return 0 if receipt["verdict"] == "PASS" else 1


def inspect_receipt(path):
    """Read terminal evidence independently; never rerun tests or issue acceptance."""
    value = json.loads(path.read_bytes())
    assert value["verdict"] == "PASS" and value["tests"]["tests"] > 0
    assert value["tests"]["failures"] == value["tests"]["errors"] == value["tests"]["skipped"] == 0
    assert value["subject_commit"] == SUBJECT and value["owner_sha256"] == OWNER_PINS
    assert value["reviewer"]["clean"] is True
    assert value["reviewer"]["commit"] == git("rev-parse", "HEAD").decode().strip()
    assert value["source_consistency"]["verdict"] == "PASS"
    assert check_source_consistency(value["reviewer"], source_snapshot(ROOT))["verdict"] == "PASS"
    owner_pins()
    original_bytes()
    counts, failures, _, membership = xml_result(
        path.parent / "pytest.xml", value["collection"], "tests.test_report_binding_repair_independent")
    assert counts == value["tests"] and failures == value["failed_nodes"] == {}
    assert membership == value["identity_membership"]
    for filename, key in (("pytest.xml", "junit_sha256"), ("pytest.log", "log_sha256"),
                          ("collection.log", "collection_sha256")):
        assert digest((path.parent / filename).read_bytes()) == value[key]
    artifacts = value["artifacts"]
    for item in artifacts:
        artifact = Path(item["path"]).resolve(strict=True)
        assert artifact.is_relative_to(path.parent.resolve())
        payload = artifact.read_bytes()
        assert digest(payload) == item["sha256"] and len(payload) == item["bytes"]
        assert item["identity"] in membership["observed"]
        if item["kind"] == "report_binding_original32":
            original = json.loads(payload)
            assert original["pytest_returncode"] == 1 and original["passed"] == 12
            assert original["tests"] == {"tests": 32, "failures": 20, "errors": 0, "skipped": 0}
            assert original["original_sha256"] == ORIGINAL_PIN and original["assertions_changed"] is False
            original_counts, _, _, original_membership = xml_result(
                Path(original["junit"]["path"]), original["collection"], "tests.test_rsoc_report_independent")
            assert original_counts == original["tests"]
            assert original_membership == original["identity_membership"]
            for kind in ("junit", "log"):
                nested = original[kind]
                actual = Path(nested["path"]).read_bytes()
                assert digest(actual) == nested["sha256"] and len(actual) == nested["bytes"]
    negative = [json.loads(Path(item["path"]).read_bytes()) for item in artifacts
                if item["kind"] == "report_binding_negative"]
    assert len(negative) == value["negative_cases"] > 0 and value["false_accept_count"] == 0
    assert all(item["accepted"] is False and item["code"] == item["expected_code"] for item in negative)
    print("REPORT_BINDING_TERMINAL_READ verdict=PASS "
          f"read_at={datetime.now(timezone.utc).isoformat()} reviewer={value['reviewer']['commit']} "
          f"tests={counts['tests']} negatives={len(negative)} false_accept=0 skip=0 "
          f"receipt={path} sha256={digest(path.read_bytes())}")
    return 0


def scan(output):
    """Reuse only the safe in-memory legs of the existing credential scanner."""
    spec = importlib.util.spec_from_file_location(
        "_report_binding_safe_scan", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(secret_hex=None, secrets_file=None,
        no_needle=False, needle_from=str(Path.home() / ".config/dsh-n600-token")))
    assert needles
    paths = [ROOT / SELF, ROOT / "docs/research/report-binding-repair-independent.md"]
    literal, _ = scanner.scan_literal([str(path) for path in paths], needles)
    pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                         r"""(?i:password|token|secret)\s*["']?\s*[:=]\s*["'][^"'\s]{6,}["']""")
    shape = sum(bool(pattern.search(path.read_text())) for path in paths)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "scan.json"
    write(path, {"verdict": "PASS" if literal == shape == 0 else "FAIL",
                 "literal_hits": literal, "shape_hits": shape,
                 "sha256": {str(path): digest(path.read_bytes()) for path in paths}})
    print(f"REPORT_BINDING_SCAN receipt={path} sha256={digest(path.read_bytes())}")
    return 0 if literal == shape == 0 else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--original32-worker", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--inspect-receipt", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    assert sum((args.review, args.original32_worker, args.scan_only, args.inspect_receipt is not None)) == 1
    if args.inspect_receipt is not None:
        raise SystemExit(inspect_receipt(args.inspect_receipt))
    if args.original32_worker:
        raise SystemExit(original_worker(args.artifact_dir))
    if args.scan_only:
        raise SystemExit(scan(args.artifact_dir))
    raise SystemExit(review(args.artifact_dir))
