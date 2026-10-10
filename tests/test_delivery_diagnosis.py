"""Coordinates-only controls; sample strings are synthetic, not credentials."""

import json
from pathlib import Path
import subprocess

import pytest

from scripts import diagnose_delivery_findings as diagnosis
from scripts.diagnose_delivery_findings import locations


def test_shape_is_located_without_value_disclosure():
    fake = b"not-a-real-credential-0000"
    source = b"def test_fixture():\n    " + b"pass" + b"word = '" + fake + b"'\n"
    result = locations(source)
    assert result == [{
        "pattern_index": 3, "line": 2, "node_kind": "FunctionDef",
        "enclosing_test": True, "classification": "UNADJUDICATED",
    }]
    assert fake.decode() not in json.dumps(result)


def test_nonpython_remains_unadjudicated():
    result = locations(b"invalid syntax\n" + b"tok" + b"en: 'synthetic-only-0000'\n")
    assert len(result) == 1
    assert result[0]["line"] == 2
    assert not result[0]["enclosing_test"]
    assert result[0]["classification"] == "UNADJUDICATED"


def test_empty_shape_set_is_not_a_release_pass():
    assert locations(b"ordinary text") == []


@pytest.mark.parametrize("marker", [
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"AK" + b"IA" + b"A" * 16,
    b"gh" + b"p_" + b"a" * 24,
])
def test_other_shapes_never_echo_bytes(marker):
    result = locations(marker)
    assert len(result) == 1
    assert marker.decode() not in json.dumps(result)
    assert result[0]["classification"] == "UNADJUDICATED"


def command(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args],
                            capture_output=True, check=False)
    assert result.returncode == 0
    return result.stdout.decode().strip()


@pytest.fixture
def actual_history(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    command(root, "init", "-q")
    command(root, "config", "user.name", "Synthetic test")
    command(root, "config", "user.email", "synthetic@example.invalid")
    source = root / "sample.py"
    source.write_bytes(b"ordinary = True\n")
    command(root, "add", "--", "sample.py")
    command(root, "commit", "-qm", "base")
    base = command(root, "rev-parse", "HEAD")
    source.write_bytes(b"def test_fake():\n    tok" + b"en = 'synthetic-only-0000'\n")
    command(root, "add", "--", "sample.py")
    command(root, "commit", "-qm", "synthetic negative")
    head = command(root, "rev-parse", "HEAD")
    identity = command(root, "rev-parse", head + ":sample.py")
    report = {
        "base": base, "head": head,
        "inventory": {identity: {"sha256": diagnosis.digest(source.read_bytes())}},
        "findings": [{"object": identity, "type": "blob", "literal": 0, "shape": 1}],
    }
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps(report))
    return root, audit, report


def test_actual_git_locations_never_grant_release(actual_history):
    root, audit, _ = actual_history
    result = diagnosis.inspect(root, audit, diagnosis.digest(audit.read_bytes()))
    assert result["verdict"] == "LOCATED" and result["release_gate"] == "BLOCKED"
    assert result["push_performed"] is False
    assert result["findings"][0]["paths"] == ["sample.py"]
    assert result["findings"][0]["shape_locations"][0]["enclosing_test"] is True
    assert "synthetic-only-0000" not in json.dumps(result)


@pytest.mark.parametrize("mutation", ["pin", "hash", "count", "empty"])
def test_actual_git_tamper_is_rejected(actual_history, mutation):
    root, audit, report = actual_history
    if mutation == "hash":
        next(iter(report["inventory"].values()))["sha256"] = "0" * 64
    elif mutation == "count":
        report["findings"][0]["shape"] = 2
    elif mutation == "empty":
        report["findings"] = []
    audit.write_text(json.dumps(report))
    pin = "0" * 64 if mutation == "pin" else diagnosis.digest(audit.read_bytes())
    with pytest.raises(diagnosis.DiagnosisError):
        diagnosis.inspect(root, audit, pin)


def test_raw_git_error_retains_code_without_content(tmp_path):
    with pytest.raises(diagnosis.DiagnosisError) as found:
        diagnosis.git(tmp_path / "missing", "rev-parse", "HEAD")
    assert found.value.reason == "raw_git_query_failed"
    assert type(found.value.returncode) is int and found.value.returncode != 0
    assert str(tmp_path) not in str(found.value)


def test_named_submission_scan():
    from tests.test_render_release_report import scan_submission_with_existing_gate
    scan_submission_with_existing_gate(
        ["scripts/diagnose_delivery_findings.py", "tests/test_delivery_diagnosis.py"],
        Path(__file__).resolve().parents[1],
    )
