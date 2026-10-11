"""Portable production regressions; no SDK import, network or publication."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import benchmark_commercial as commercial
from scripts import prepare_publication as publication

ROOT = Path(__file__).resolve().parents[1]
OWNED = (
    "scripts/benchmark_commercial.py",
    "scripts/prepare_publication.py",
    "tests/test_commercial_publication_portable.py",
    "docs/research/commercial-publication-portable.md",
)
RECORD_PIN = hashlib.sha256(b"offline record index, not authorization").hexdigest()


@pytest.mark.parametrize("flag", [
    1, 0, 1.0, 0.0, "False", "True", "", None, [], [True],
    np.bool_(True), np.bool_(False),
])
def test_exact_boolean_required_before_import(monkeypatch, flag):
    imported = []
    monkeypatch.setattr(commercial.importlib, "import_module", imported.append)
    args = SimpleNamespace(authorized_run=flag, authorization_sha256=RECORD_PIN)
    with pytest.raises(ValueError, match="boolean"):
        commercial.authorization(args)
    with pytest.raises(ValueError, match="boolean"):
        commercial.import_sdk("gurobi", args)
    assert imported == []


def test_custom_truth_conversion_is_never_used():
    class Trap:
        def __bool__(self):
            raise AssertionError("truth conversion must not authorize")

    with pytest.raises(ValueError, match="boolean"):
        commercial.authorization(SimpleNamespace(
            authorized_run=Trap(), authorization_sha256=RECORD_PIN))


@pytest.mark.parametrize("solver", tuple(commercial.SDKS))
def test_exact_true_reaches_recording_importer_only(monkeypatch, solver):
    imported = []
    marker = object()

    def recording(name):
        imported.append(name)
        return marker

    monkeypatch.setattr(commercial.importlib, "import_module", recording)
    args = SimpleNamespace(authorized_run=True, authorization_sha256=RECORD_PIN)
    assert commercial.import_sdk(solver, args) is marker
    assert imported == [commercial.SDKS[solver][0]]
    record = commercial.authorization(args)
    assert record["legal_validity"] == "UNVERIFIED"
    assert record["record_is_legal_authorization"] is False
    assert record["operator_external_approval_attested"] is True


def test_exact_false_cannot_import_or_attach_record(monkeypatch):
    imported = []
    monkeypatch.setattr(commercial.importlib, "import_module", imported.append)
    args = SimpleNamespace(authorized_run=False, authorization_sha256=None)
    assert commercial.authorization(args)["explicit_authorized_run"] is False
    with pytest.raises(PermissionError):
        commercial.import_sdk("gurobi", args)
    args.authorization_sha256 = RECORD_PIN
    with pytest.raises(ValueError):
        commercial.authorization(args)
    assert imported == []


@pytest.mark.parametrize("payload", [
    "23:15:45", "2026-10-11T00:19:23+08:00",
    "<p>Measured at 23&#58;15&#58;45.</p>",
])
def test_clock_is_not_network_address(payload):
    publication.scan(payload.encode(), "clock")


@pytest.mark.parametrize("payload", [
    "192.168.1.172026-10-11T23:15:45",
    "23:15:45fe80::1", "23:15:45::1",
    "2001:db8:2026-10-11T23:15:45",
    "2001:db8:1:2:3:4:5:6", "[fe80::1]", "&#91;fe80&#58;&#58;1&#93;",
    "192%2e168%2e1%2e17", "https" + "://example.invalid/service",
])
def test_joined_and_encoded_network_addresses_remain_rejected(payload):
    with pytest.raises(publication.Rejected, match="network_entry"):
        publication.scan(payload.encode(), "network_control")


def test_current_committed_catalog_is_scannable():
    publication.scan((ROOT / "docs/site/index.html").read_bytes(), "current_catalog")


@pytest.mark.parametrize("payload,code", [
    ("/data/private/file", "absolute_path"),
    ("hostname = fixture", "host_identity"),
    ("Bearer " + "synthetic-fixture", "credential"),
    ("license " + "server fixture", "license_material"),
    ("<script>run()</script>", "active_content"),
])
def test_other_publication_safety_arms_stay_active(payload, code):
    with pytest.raises(publication.Rejected, match=code):
        publication.scan(payload.encode(), "other_control")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--scan-output", type=Path, required=True)
    args = parser.parse_args()
    output = args.scan_output.resolve()
    assert output.is_relative_to(Path("/hf3fs-jd/data/tmp").resolve())
    helper = Path("/data/dev/openopt-wt-resume-safe-scan-error-repair-20261011/"
                  "scripts/safe_scan_owned_sources.py")
    assert hashlib.sha256(helper.read_bytes()).hexdigest() == (
        "656cdd1c57b45e84f5ed5efc0e63d6652c9c6fd5559dc0884ca6374bbe903795")
    spec = importlib.util.spec_from_file_location("_portable_safe_scanner", helper)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    result = scanner.scan([ROOT / name for name in OWNED], root=ROOT)
    report = {name: result[name] for name in
              ("verdict", "files", "literal_hits", "shape_hits", "source_sha256")}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(f"PORTABLE_SCAN verdict={report['verdict']} files={report['files']} "
          f"literal_hits={report['literal_hits']} shape_hits={report['shape_hits']}")
    return int(report["verdict"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
