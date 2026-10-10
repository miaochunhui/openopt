"""CPU-only real-file fixtures and fail-closed public report negative controls."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import render_release_report as report


def save(path, value):
    payload = report.encoded(value) if isinstance(value, dict) else value
    path.write_bytes(payload)
    return {"id": path.stem, "path": str(path), "sha256": report.sha(payload)}


def fixture_inputs(root):
    root.mkdir()
    counts = {"tests": 4, "failures": 0, "errors": 0, "skipped": 1}
    xml = (b'<testsuites><testsuite tests="4" failures="0" errors="0" skipped="1">'
           b'<testcase/><testcase/><testcase/><testcase><skipped/></testcase>'
           b'</testsuite></testsuites>')
    cpu = {"verdict": "PASS", "tests": counts, "coverage": "explicit_targets",
           "finished_at": "2026-10-10T09:00:00+00:00", "internal_details": "/private/storage"}
    rows = []
    for index in range(24):
        certificate = index >= 18
        rows.append({
            "case_id": f"synthetic-{index}", "solver": "openopt", "verdict": "QUALIFIED",
            "solver_status": "infeasible" if certificate else "solved",
            "status_matches_expected": True,
            "expected_property": "infeasible" if certificate else "optimal",
            "candidate_quality": {"verdict": "FAIL" if certificate else "PASS"},
            "certificate_quality": {"verdict": "PASS"} if certificate else None,
        })
    meta = {"version": 4, "modified_at": "2026-10-10T17:00:00+08:00",
            "source_sha": "a" * 40, "regression_scope": "intermediate_snapshot",
            "cpu_binding": {"kind": "scope_only"},
            "online_publication": "UNVERIFIED", "commercial_qualification": "UNVERIFIED"}
    records = {
        "cpu": cpu, "junit": xml,
        "gpu": {"verdict": "PASS", "tests": counts, "scope": "one-card numerical regression only",
                "finished_at": cpu["finished_at"], "source_sha256": {"src/core.py": "b" * 64},
                "returncode": 0, "hardware": {"node": "private-identity"}},
        "gpu_junit": xml, "gpu_run": {"source_sha": "c" * 40},
        "wheel": {"verdict": "PASS", "modules": 56, "license_expression": "0BSD"},
        "commercial": {"verdict": "UNVERIFIED", "qualification": "UNVERIFIED",
                       "counts": {"planned": 18, "attempted": 0, "solve_attempted": 0,
                                  "passed": 0, "failed": 0},
                       "solvers": {"gurobi": {"status": "UNVERIFIED"}}},
        "corpus": {"verdict": "PASS", "scope": "Synthetic dense convex QP/LP only", "rows": rows},
        "campaign": {"seats": {"report": {}}},
        "audit": {"seats": [{"seat": "report", "verdict": "PASS", "outside_scope": []}]},
        "live": {"verdict": "PASS", "read_at": cpu["finished_at"],
                 "ready": {"private-identity": True}, "active": [], "cordoned": [],
                 "retained_failures": [{"error": "/private/storage"}]},
        "metadata": meta,
    }
    descriptors = [save(root / f"{key}.{'xml' if key in {'junit', 'gpu_junit'} else 'json'}", value)
                   for key, value in records.items()]
    manifest = {"schema_version": 1, "source_sha": meta["source_sha"],
                "modified_at": meta["modified_at"], "inputs": descriptors}
    path = root / "input-manifest.json"
    pin = save(path, manifest)["sha256"]
    return path, pin, records


@pytest.fixture
def inputs(tmp_path):
    assert report.shared_path(tmp_path, "fixture").is_relative_to(report.SHARED)
    return fixture_inputs(tmp_path / "evidence")


def reseal(path, records, role):
    manifest = json.loads(path.read_bytes())
    source = next(row for row in manifest["inputs"] if row["id"] == role)
    source["sha256"] = save(Path(source["path"]), records[role])["sha256"]
    return save(path, manifest)["sha256"]


def test_deterministic_sanitized_scalar_provenance(inputs):
    path, pin, _ = inputs
    before = {p: report.read(p, "fixture") for p in path.parent.iterdir()}
    first = report.generate(path, pin)
    assert first == report.generate(path, pin)
    assert before == {p: report.read(p, "fixture") for p in path.parent.iterdir()}
    summary = json.loads(first["summary.json"])
    assert summary["scalars"]["cpu_passed"] == 3
    assert summary["scalars"]["corpus_rows"] == 24
    assert summary["scalars"]["corpus_qualified"] == 24
    assert summary["scalars"]["live_retained_failures"] == 1
    private = json.loads(first["report-bundle.json"])
    for key, origin in private["scalar_provenance"].items():
        if origin["operation"] == "identity":
            item = next(row for row in private["inputs"] if row["id"] == origin["evidence_id"])
            assert summary["scalars"][key] == report.pointer(json.loads(Path(item["path"]).read_bytes()),
                                                           origin["pointer"])
    for name, payload in first.items():
        if name != "report-bundle.json":
            assert b"/private/storage" not in payload
            assert b"private-identity" not in payload
            if name.endswith(".json"):
                report.public_json(payload, name)
            else:
                report.public_text(payload.decode(), name)
    html = first["benchmark-report-v4.html"].decode()
    assert "FAIL" in html and "UNVERIFIED" in html and "intermediate_snapshot" in html
    assert "最后修改：2026-10-10 17:00:00 +08:00" in html
    assert "onchange" not in html and "<script" not in html
    assert 'href="benchmark-report-v1.html"' in html
    assert 'href="benchmark-report-v3.html"' in html


def test_claims_are_exact_summary_scalars(inputs):
    files = report.generate(*inputs[:2])
    summary = json.loads(files["summary.json"])
    claims = json.loads(files["claims.json"])["claims"]
    assert claims
    assert len({(c["document"], c["id"]) for c in claims}) == len(claims)
    for claim in claims:
        assert claim["text"] == str(report.pointer(summary, claim["pointer"]))
        assert claim["text"] in files[Path(claim["document"]).name].decode()


def test_changed_input_hash_rejected(inputs):
    path, pin, records = inputs
    records["cpu"]["tests"]["tests"] = 5
    save(path.parent / "cpu.json", records["cpu"])
    with pytest.raises(report.Rejected, match="changed_hash"):
        report.generate(path, pin)


@pytest.mark.parametrize("payload", [b"", b"{}", b"[]", b'{"tests":NaN}', b'{"x":1,"x":2}'])
def test_empty_invalid_or_duplicate_input_rejected(inputs, payload):
    path, _, _ = inputs
    manifest = json.loads(path.read_bytes())
    item = next(row for row in manifest["inputs"] if row["id"] == "cpu")
    item["sha256"] = save(Path(item["path"]), payload)["sha256"]
    pin = save(path, manifest)["sha256"]
    with pytest.raises(report.Rejected):
        report.generate(path, pin)


def test_false_count_resealed_json_rejected(inputs):
    path, _, records = inputs
    records["cpu"]["tests"] = {**records["cpu"]["tests"], "tests": 100}
    with pytest.raises(report.Rejected, match="junit_json_mismatch"):
        report.generate(path, reseal(path, records, "cpu"))


def test_failed_junit_cannot_hide_behind_pass(inputs):
    path, _, records = inputs
    records["junit"] = records["junit"].replace(b'failures="0"', b'failures="1"').replace(
        b"<testcase/>", b"<testcase><failure/></testcase>", 1)
    with pytest.raises(report.Rejected, match="junit_json_mismatch"):
        report.generate(path, reseal(path, records, "junit"))


def test_junit_attribute_falsecount_rejected(inputs):
    path, _, records = inputs
    records["junit"] = records["junit"].replace(b'tests="4"', b'tests="400"')
    with pytest.raises(report.Rejected, match="junit_false_count"):
        report.generate(path, reseal(path, records, "junit"))


def test_legitimate_failed_regression_retained(inputs):
    path, _, records = inputs
    records["cpu"]["verdict"] = "FAIL"
    records["cpu"]["tests"] = {**records["cpu"]["tests"], "failures": 1}
    records["junit"] = records["junit"].replace(b'failures="0"', b'failures="1"').replace(
        b"<testcase/>", b"<testcase><failure/></testcase>", 1)
    reseal(path, records, "cpu")
    files = report.generate(path, reseal(path, records, "junit"))
    summary = json.loads(files["summary.json"])["scalars"]
    assert summary["cpu_status"] == "FAIL" and summary["cpu_passed"] == 2
    assert b'data-scalar="cpu_status">FAIL' in files["design-v4.html"]


@pytest.mark.parametrize("unsafe", [
    "/private/storage/result.json", "https" + "://hidden.internal",
    "h3c" + "2", "<img on" + "error='alert(1)'>",
    "password" + "=not-a-real-credential", "%2Fhome%2Fprivate%2Ffile",
    "&lt;script&gt;bad&lt;/script&gt;", "wide\u200bhidden",
])
def test_unsafe_selected_content_rejected_without_echo(inputs, unsafe):
    path, _, records = inputs
    records["corpus"]["rows"][0]["case_id"] = unsafe
    with pytest.raises(report.Rejected) as error:
        report.generate(path, reseal(path, records, "corpus"))
    assert unsafe not in str(error.value)


def test_false_qualification_rejected(inputs):
    path, _, records = inputs
    records["corpus"]["rows"][0]["candidate_quality"]["verdict"] = "FAIL"
    with pytest.raises(report.Rejected, match="false_corpus_qualification"):
        report.generate(path, reseal(path, records, "corpus"))


def test_nonqualified_rows_and_scope_failure_retained(inputs):
    path, _, records = inputs
    records["corpus"]["rows"][0]["verdict"] = "REJECTED"
    records["corpus"]["rows"][0]["candidate_quality"]["verdict"] = "FAIL"
    records["audit"]["seats"][0]["verdict"] = "FAIL"
    records["audit"]["seats"][0]["outside_scope"] = ["/private/storage"]
    reseal(path, records, "corpus")
    files = report.generate(path, reseal(path, records, "audit"))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["corpus_rejected"] == 1
    assert scalars["audit_failed"] == 1 and scalars["seat_0_outside"] == 1
    assert b"/private/storage" not in files["benchmark-report-v4.html"]


def test_frozen_release_needs_version_binding(inputs):
    path, _, records = inputs
    records["metadata"]["regression_scope"] = "frozen_release"
    with pytest.raises(report.Rejected, match="unbound_frozen_release"):
        report.generate(path, reseal(path, records, "metadata"))


def test_explicit_cpu_code_hash_binding(inputs):
    path, _, records = inputs
    code_hash = report.sha(b"synthetic core source")
    records["metadata"]["cpu_binding"] = {"kind": "code_hash", "value": code_hash}
    manifest = json.loads(path.read_bytes())
    manifest["inputs"].append(save(path.parent / "cpu_version.json",
                                  {"source_sha": records["metadata"]["source_sha"],
                                   "code_hash": code_hash}))
    save(path, manifest)
    files = report.generate(path, reseal(path, records, "metadata"))
    assert json.loads(files["summary.json"])["scalars"]["cpu_binding_value"] == code_hash


def with_source_gate(records):
    source = {"commit": records["metadata"]["source_sha"], "tree": "d" * 40,
              "clean": True, "sha256": {"src/core.py": "b" * 64}}
    records["cpu"]["source"] = source
    records["cpu"]["source_consistency"] = {
        "verdict": "PASS", "same_commit": True, "changed": [], "missing": [],
        "extra": [], "files": len(source["sha256"]),
    }


def test_actual_source_gate_and_distinct_document_commit(inputs):
    path, _, records = inputs
    with_source_gate(records)
    records["metadata"]["doc_source_sha"] = "e" * 40
    reseal(path, records, "cpu")
    files = report.generate(path, reseal(path, records, "metadata"))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["cpu_source_consistency"] == "PASS"
    assert scalars["source_sha"] != scalars["doc_source_sha"]
    assert scalars["cpu_source_hash"] == report.sha(report.encoded(records["cpu"]["source"]["sha256"]))


@pytest.mark.parametrize("mutation", ["changed", "count", "commit", "hidden_fail", "empty"])
def test_source_consistency_false_pass_rejected(inputs, mutation):
    path, _, records = inputs
    with_source_gate(records)
    gate = records["cpu"]["source_consistency"]
    if mutation == "changed":
        gate["changed"] = ["src/core.py"]
    elif mutation == "count":
        gate["files"] = 999
    elif mutation == "commit":
        records["cpu"]["source"]["commit"] = "f" * 40
    elif mutation == "hidden_fail":
        gate["verdict"] = "FAIL"
    else:
        records["cpu"]["source"]["sha256"] = {}
    with pytest.raises(report.Rejected):
        report.generate(path, reseal(path, records, "cpu"))


def test_legacy_gate_missing_explicit_unverified(inputs):
    files = report.generate(*inputs[:2])
    assert json.loads(files["summary.json"])["scalars"]["cpu_source_consistency"] == "UNVERIFIED"
    assert "历史证据未冻结" in files["benchmark-report-v4.html"].decode()


def test_nested_failure_statuses_not_ignored(inputs):
    path, _, records = inputs
    records["commercial"]["solvers"]["gurobi"]["modes"] = {"cold": {"status": "FAIL"}}
    files = report.generate(path, reseal(path, records, "commercial"))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["evidence_failure_statuses"] == 7
    private = json.loads(files["report-bundle.json"])
    assert any(item["pointer"] == "/solvers/gurobi/modes/cold/status"
               for item in private["scalar_provenance"].values())


def test_prepare_exact_descriptors_and_no_overwrite(inputs, tmp_path):
    path, _, records = inputs
    descriptors = json.loads(path.read_bytes())["inputs"]
    selected = [(item["id"], item["path"], item["sha256"])
                for item in descriptors if item["id"] != "metadata"]
    output = tmp_path / "prepared"
    manifest, pin = report.prepare_inputs(output, selected, records["metadata"]["source_sha"],
                                         records["metadata"]["modified_at"])
    files = report.generate(manifest, pin)
    assert json.loads(files["summary.json"])["scalars"]["cpu_source_consistency"] == "UNVERIFIED"
    with pytest.raises(report.Rejected, match="output_already_exists"):
        report.prepare_inputs(output, selected, records["metadata"]["source_sha"],
                              records["metadata"]["modified_at"])


def with_wheel_gate(records):
    with_source_gate(records)
    records["wheel"]["source"] = records["cpu"]["source"]
    records["wheel"]["source_consistency"] = records["cpu"]["source_consistency"]
    records["wheel"]["wheel_sha256"] = report.sha(b"fixture wheel bytes")


@pytest.mark.parametrize("mutation", ["changed", "hidden_fail", "wrong_commit", "hash"])
def test_wheel_source_gate_negative_controls(inputs, mutation):
    path, _, records = inputs
    with_wheel_gate(records)
    if mutation == "changed":
        records["wheel"]["source_consistency"]["changed"] = ["src/core.py"]
    elif mutation == "hidden_fail":
        records["wheel"]["source_consistency"]["verdict"] = "FAIL"
    elif mutation == "wrong_commit":
        records["wheel"]["source"] = {**records["wheel"]["source"], "commit": "f" * 40}
    else:
        records["wheel"]["wheel_sha256"] = ""
    with pytest.raises(report.Rejected):
        report.generate(path, reseal(path, records, "wheel"))


def frozen_inputs(path, records):
    with_wheel_gate(records)
    records["cpu"]["coverage"] = "tests"
    records["metadata"]["regression_scope"] = "frozen_release"
    code_hash = report.sha(report.encoded(records["cpu"]["source"]["sha256"]))
    records["metadata"]["cpu_binding"] = {"kind": "code_hash", "value": code_hash}
    manifest = json.loads(path.read_bytes())
    manifest["inputs"].append(save(path.parent / "cpu_version.json",
                                  {"source_sha": records["metadata"]["source_sha"],
                                   "code_hash": code_hash}))
    save(path, manifest)
    reseal(path, records, "cpu")
    reseal(path, records, "wheel")
    return reseal(path, records, "metadata")


def test_frozen_scope_binds_clean_full_cpu_and_wheel(inputs):
    path, _, records = inputs
    files = report.generate(path, frozen_inputs(path, records))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["cpu_source_consistency"] == scalars["wheel_source_consistency"] == "PASS"
    assert scalars["regression_scope"] == "frozen_release"
    assert scalars["commercial_status"] == scalars["online_status"] == "UNVERIFIED"
    assert scalars["wheel_hash"] == report.sha(b"fixture wheel bytes")


@pytest.mark.parametrize("mutation", ["dirty", "targeted", "missing_gate", "false_hash", "legacy_wheel"])
def test_frozen_scope_negative_controls(inputs, mutation):
    path, _, records = inputs
    frozen_inputs(path, records)
    if mutation == "dirty":
        records["cpu"]["source"]["clean"] = False
        pin = reseal(path, records, "cpu")
    elif mutation == "targeted":
        records["cpu"]["coverage"] = "explicit_targets"
        pin = reseal(path, records, "cpu")
    elif mutation == "missing_gate":
        del records["cpu"]["source_consistency"]
        pin = reseal(path, records, "cpu")
    elif mutation == "legacy_wheel":
        del records["wheel"]["source_consistency"]
        pin = reseal(path, records, "wheel")
    else:
        records["cpu"]["source"]["sha256"]["src/core.py"] = "f" * 64
        pin = reseal(path, records, "cpu")
    with pytest.raises(report.Rejected):
        report.generate(path, pin)


def test_symlink_rejected(inputs):
    path, _, _ = inputs
    manifest = json.loads(path.read_bytes())
    symlink = path.parent / "linked.json"
    symlink.symlink_to(path.parent / "cpu.json")
    next(row for row in manifest["inputs"] if row["id"] == "cpu")["path"] = str(symlink)
    with pytest.raises(report.Rejected, match="missing_or_unsafe_input"):
        report.generate(path, save(path, manifest)["sha256"])


def test_snapshot_race_rejected(inputs, monkeypatch):
    path, pin, _ = inputs
    original = report.read
    calls = {}

    def changed_identity(source, coordinate):
        payload, identity = original(source, coordinate)
        calls[source] = calls.get(source, 0) + 1
        if calls[source] > 1:
            identity = (*identity[:-1], identity[-1] + 1)
        return payload, identity

    monkeypatch.setattr(report, "read", changed_identity)
    with pytest.raises(report.Rejected, match="snapshot_race"):
        report.generate(path, pin)


def test_cli_external_readback_and_no_overwrite(inputs, tmp_path):
    path, pin, _ = inputs
    output = tmp_path / "rendered"
    command = [sys.executable, "-B", str(report.ROOT / "scripts/render_release_report.py"),
               "--manifest", str(path), "--manifest-sha256", pin, "--output-dir", str(output)]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "RELEASE_REPORT verdict=PASS" in result.stdout
    assert (output / "manifest.json").stat().st_size > 0
    assert subprocess.run(command + ["--check"], capture_output=True, check=False).returncode == 0
    failed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert failed.returncode == 1 and "output_already_exists" in failed.stderr
    before = (output / "summary.json").read_bytes()
    assert json.loads(before)["scalars"]["cpu_passed"] == 3


def test_static_parser_rejects_event_and_nonrelative_links():
    for content in ('<a href="../escape.html">x</a>', '<a onclick="x">x</a>',
                    '<svg><animate/></svg>', '<img src="//hidden.internal">'):
        with pytest.raises(report.Rejected):
            report.StaticPage().feed(content)


def test_timestamp_is_not_ipv6_but_real_addresses_rejected():
    for value in ("2026-10-10T09:00:00+00:00", "2026-10-10 17:00:00 +08:00"):
        assert report.public_text(value, "time") == value
    for value in ("::1", "2001:db8::1", "2001:db8:1234:0000:0000:0000:0000:0001"):
        with pytest.raises(report.Rejected, match="unsafe_public_content"):
            report.public_text(value, "address")


@pytest.mark.parametrize("false_pass", [False, True])
def test_optional_public_qualification_no_invented_run(inputs, false_pass):
    path, _, _ = inputs
    manifest = json.loads(path.read_bytes())
    value = {"verdict": "PASS" if false_pass else "UNVERIFIED",
             "scope": "public runner not executed",
             "counts": {"planned": 2, "attempted": 0, "qualified": 0, "rejected": 0}}
    manifest["inputs"].append(save(path.parent / "public_qualification.json", value))
    pin = save(path, manifest)["sha256"]
    if false_pass:
        with pytest.raises(report.Rejected, match="public_empty_false_pass"):
            report.generate(path, pin)
    else:
        files = report.generate(path, pin)
        assert json.loads(files["summary.json"])["scalars"]["public_status"] == "UNVERIFIED"
        assert b'data-scalar="public_attempted">0' in files["benchmark-report-v4.html"]


def scan_submission_with_existing_gate():
    scanner = Path("/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    assert scanner.is_file(), "required local precommit scanner missing"
    check = subprocess.run([sys.executable, "-B", str(scanner), "--selftest"],
                           capture_output=True, text=True, check=False)
    assert check.returncode == 0 and "verdict=PASS" in check.stdout, "secret scanner selftest failed"
    command = [sys.executable, "-B", str(scanner), "--no-grep"]
    for name in ("scripts/render_release_report.py", "tests/test_render_release_report.py",
                 "docs/research/release-report.md"):
        command.extend(["--path", str(report.ROOT / name)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, "local credential-literal gate failed; output withheld"
    verdict = next(line for line in result.stdout.splitlines() if line.startswith("SECRET_SCAN:"))
    assert "literal_hits=0" in verdict and "needles=1" in verdict
    shape = subprocess.run([sys.executable, "-B", str(scanner), "--no-needle", *command[4:]],
                           capture_output=True, text=True, check=False)
    assert shape.returncode == 0, "redaction-safe shape scan failed"
    print("REPORT_SUBMISSION_SCAN verdict=PASS files=3 literal_hits=0 scanner_selftest=PASS")


if __name__ == "__main__":
    assert sys.argv[1:] == ["--scan-submission"], "use --scan-submission"
    scan_submission_with_existing_gate()
