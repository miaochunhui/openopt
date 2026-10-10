"""CPU-only real-file fixtures and fail-closed public report negative controls."""

from __future__ import annotations

from copy import deepcopy
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import pytest
from xml.etree import ElementTree as ET

from scripts import render_release_report as report

RSOC_AUTHORITIES = {}
AUDITOR_FIXTURE_COMMIT = "0160ba1d48558b33d33c54009ccfdf1fff98ae29"


def rsoc_generate(path, pin):
    return report.generate(path, pin, rsoc_authority=RSOC_AUTHORITIES.get(path))


def rsoc_cli_authority(path):
    authority = RSOC_AUTHORITIES.get(path)
    return [] if authority is None else ["--rsoc-authority", authority["path"],
                                        "--rsoc-authority-sha256", authority["sha256"]]


def bind_synthetic_authority(path, records):
    """Synthetic caller fixture, not an issued numerical qualification.

    Only the caller fixture explicitly repins; ordinary mutations never do.
    Auditor bytes come from committed repository Git, never artifact execution.
    """
    selected = {row["id"]: row for row in json.loads(path.read_bytes())["inputs"]}
    process = subprocess.run(["git", "show", AUDITOR_FIXTURE_COMMIT + ":scripts/accept_rotated_cuda.py"],
                             cwd=report.ROOT, capture_output=True, check=False)
    assert process.returncode == 0, "trusted committed fixture source unavailable"
    source = path.parent / "caller-auditor-source.py"
    source.write_bytes(process.stdout)
    value = {"schema_version": 1, "receipts": {"rotated_cuda": {
        "acceptance_sha256": selected["rotated_cuda_acceptance"]["sha256"],
        "bundle_sha256": selected["rotated_cuda_bundle"]["sha256"],
        "auditor": {"commit": AUDITOR_FIXTURE_COMMIT, "path": str(source),
                    "sha256": report.sha(process.stdout), "bytes": len(process.stdout)},
    }}}
    descriptor = save(path.parent / "rsoc-authority.json", value)
    RSOC_AUTHORITIES[path] = {key: descriptor[key] for key in ("path", "sha256")}
    return value


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


def optional_record(path, role, value):
    manifest = json.loads(path.read_bytes())
    manifest["inputs"].append(save(path.parent / (role + ".json"), value))
    return save(path, manifest)["sha256"]


def extension_record(suite):
    return {
        "verdict": "PASS", "suite": suite, "final_source": True,
        "qualified": False, "certified": False,
        "commercial": "UNVERIFIED", "performance": "UNVERIFIED",
        "source_sha_informational": "e" * 40,
        "finished_at": "2026-10-10T10:00:00+00:00",
        "tests": {"tests": {"native_conic": 21, "presolve": 8, "refinement": 7}[suite],
                  "failures": 0, "errors": 0, "skipped": 0},
    }


def test_independent_extension_fail_cannot_hide_behind_pytest_pass(inputs):
    path, _, _ = inputs
    failed = {"verdict": "FAIL", "error": "/private/not-published",
              "finished_at": "2026-10-10T10:00:00+00:00"}
    files = report.generate(path, optional_record(path, "extension_refinement", failed))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["gpu_status"] == "PASS"
    assert scalars["extension_refinement_status"] == "FAIL"
    assert scalars["evidence_failure_statuses"] == 7
    assert b"/private/not-published" not in files["benchmark-report-v4.html"]
    assert "FAIL 不因 pytest 通过而升级" in files["design-v4.html"].decode()


@pytest.mark.parametrize("mutation", ["suite", "retrospective", "qualified", "certified",
                                     "commercial", "performance", "count", "source",
                                     "skip", "denominator"])
def test_extension_acceptance_overclaim_rejected(inputs, mutation):
    path, _, _ = inputs
    value = extension_record("native_conic")
    if mutation == "suite":
        value["suite"] = "refinement"
    elif mutation == "retrospective":
        value["final_source"] = False
    elif mutation in {"qualified", "certified"}:
        value[mutation] = True
    elif mutation in {"commercial", "performance"}:
        value[mutation] = "PASS"
    elif mutation == "count":
        value["tests"]["failures"] = 1
    elif mutation == "skip":
        value["tests"]["skipped"] = 1
    elif mutation == "denominator":
        value["tests"]["tests"] = 1
    else:
        value["source_sha_informational"] = "HEAD"
    with pytest.raises(report.Rejected):
        report.generate(path, optional_record(path, "extension_native", value))


def test_proof_contract_review_is_not_cuda_qualification(inputs):
    path, _, _ = inputs
    value = {"verdict": "FAIL", "producer": {"commit": "e" * 40},
             "finished_at": "2026-10-10T10:00:00+00:00",
             "private": "/home/hidden"}
    files = report.generate(path, optional_record(path, "proof_contract_review", value))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["proof_review_status"] == "FAIL"
    assert scalars["proof_review_source"] == "e" * 40
    assert b"/home/hidden" not in files["design-v4.html"]


def test_presolve_source_comes_from_explicit_acceptance_bound_run(inputs):
    path, _, _ = inputs
    optional_record(path, "extension_presolve_run", {"source_sha": "e" * 40})
    descriptor = next(item for item in json.loads(path.read_bytes())["inputs"]
                      if item["id"] == "extension_presolve_run")
    value = extension_record("presolve")
    del value["source_sha_informational"]
    del value["final_source"]
    value["current_core"] = "PASS"
    value["pins"] = {"run": {"path": descriptor["path"], "sha256": descriptor["sha256"]}}
    files = report.generate(path, optional_record(path, "extension_presolve", value))
    assert json.loads(files["summary.json"])["scalars"]["extension_presolve_source"] == "e" * 40
    value["pins"]["run"]["sha256"] = "0" * 64
    with pytest.raises(report.Rejected, match="presolve_acceptance_run_mismatch"):
        report.generate(path, reseal(path, {"extension_presolve": value}, "extension_presolve"))


def robustness_record():
    return {
        "version": "public-robustness-v1",
        "arms": [{"arm": "strict", "solver": "openopt"}], "gates": {"eps_abs": 1e-7},
        "numeric_verdict": "UNQUALIFIED", "original_verdict": "UNQUALIFIED",
        "finished_at": "2026-10-10T10:00:00+00:00",
        "planned_denominator": 2, "attempted_count": 2,
        "speedup_claim": False, "tiny_dual_clipping": False, "license_verified": False,
        "rows": [{"case_id": "case-a", "arm": "strict", "solver": "openopt",
                  "model_sha256": "a" * 64, "attempt": 1, "denominator": 1,
                  "attempted": True, "verdict": "QUALIFIED"},
                 {"case_id": "case-b", "arm": "strict", "solver": "openopt",
                  "model_sha256": "b" * 64, "attempt": 1, "denominator": 1,
                  "attempted": True, "verdict": "UNQUALIFIED"}],
        "arm_counts": {"strict": {"denominator": 2, "QUALIFIED": 1, "UNQUALIFIED": 1,
                                  "FAIL": 0, "BUDGET_EXHAUSTED": 0}},
    }


def robustness_inputs(path, value):
    original = robustness_record()
    optional_record(path, "public_robustness_plan", {
        "arms": original["arms"], "gates": original["gates"], "rows": original["rows"],
    })
    optional_record(path, "public_robustness_corpus", {
        "cases": [{"case_id": "case-a", "model_sha256": "a" * 64},
                  {"case_id": "case-b", "model_sha256": "b" * 64}],
    })
    corpus = next(item for item in json.loads(path.read_bytes())["inputs"]
                  if item["id"] == "public_robustness_corpus")
    value["corpus_sha256"] = corpus["sha256"]
    return optional_record(path, "public_robustness", value)


def test_robustness_denominator_and_unqualified_status_preserved(inputs):
    path, _, _ = inputs
    files = report.generate(path, robustness_inputs(path, robustness_record()))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["robustness_0_denominator"] == 2
    assert scalars["robustness_0_qualified"] == scalars["robustness_0_unqualified"] == 1
    assert scalars["robustness_status"] == "UNQUALIFIED"
    assert scalars["robustness_version"] == "public-robustness-v1"
    assert scalars["robustness_settings_binding"] == scalars["robustness_runtime_binding"] == "UNVERIFIED"
    assert scalars["evidence_failure_statuses"] == 10


@pytest.mark.parametrize("mutation", ["planned", "attempted", "qualified_count", "drop_arm",
                                     "duplicate", "clip", "speedup", "rights",
                                     "false_pass", "unattempted_candidate"])
def test_robustness_coordinated_false_denominator_rejected(inputs, mutation):
    path, _, _ = inputs
    value = robustness_record()
    if mutation == "planned":
        value["planned_denominator"] = 1
    elif mutation == "attempted":
        value["attempted_count"] = 1
    elif mutation == "qualified_count":
        value["arm_counts"]["strict"]["QUALIFIED"] = 2
    elif mutation == "drop_arm":
        value["rows"][1]["arm"] = "unaccounted"
    elif mutation == "duplicate":
        value["rows"][1]["case_id"] = "case-a"
    elif mutation == "false_pass":
        value["numeric_verdict"] = "PASS"
    elif mutation == "unattempted_candidate":
        value["rows"][0]["attempted"] = False
        value["attempted_count"] = 1
    else:
        value[{"clip": "tiny_dual_clipping", "speedup": "speedup_claim",
               "rights": "license_verified"}[mutation]] = True
    with pytest.raises(report.Rejected):
        report.generate(path, robustness_inputs(path, value))


def test_robustness_success_uses_producer_pass_contract(inputs):
    path, _, _ = inputs
    value = robustness_record()
    value["rows"][1]["verdict"] = "QUALIFIED"
    value["arm_counts"]["strict"]["QUALIFIED"] = 2
    value["arm_counts"]["strict"]["UNQUALIFIED"] = 0
    value["numeric_verdict"] = "PASS"
    files = report.generate(path, robustness_inputs(path, value))
    assert json.loads(files["summary.json"])["scalars"]["robustness_status"] == "PASS"
    value["numeric_verdict"] = "UNQUALIFIED"
    with pytest.raises(report.Rejected, match="robustness_false_overall_status"):
        report.generate(path, reseal(path, {"public_robustness": value}, "public_robustness"))


def test_robustness_resealed_failed_row_drop_rejected_by_independent_plan(inputs):
    path, _, _ = inputs
    value = robustness_record()
    value["rows"] = value["rows"][:1]
    value["planned_denominator"] = value["attempted_count"] = 1
    value["numeric_verdict"] = "PASS"
    value["arm_counts"]["strict"].update(denominator=1, UNQUALIFIED=0)
    with pytest.raises(report.Rejected, match="robustness_result_incomplete"):
        report.generate(path, robustness_inputs(path, value))


def test_robustness_projection_without_independent_contract_rejected(inputs):
    path, _, _ = inputs
    with pytest.raises(report.Rejected, match="robustness_plan_corpus_required"):
        report.generate(path, optional_record(path, "public_robustness", robustness_record()))


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


def rsoc_fixture(path, total):
    """Self-described synthetic archives, never actual CUDA/numerical evidence."""
    prefix = "rotated_cuda" if total == 13 else "rsoc_extremes"
    role = prefix + "_acceptance"
    root = path.parent / (prefix + "-archive")
    output = root / "case-0/cuda-check-fixture"
    output.mkdir(parents=True)
    test = ("tests/test_rotated_conic_cuda.py" if total == 13
            else "tests/test_rotated_conic_cuda_extremes.py")
    source_names = ["src/openopt/conic.py", "src/openopt/native_conic.py", test,
                    "scripts/gpu_release_check.py"]
    if total == 76:
        source_names.append("tests/test_rotated_conic_independent.py")
    source = {}
    for name in source_names:
        target = root / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = ("# synthetic source " + name + "\n").encode()
        target.write_bytes(payload)
        source[name] = report.sha(payload)
    execution = {"job": "fixture-job", "pod": "fixture-pod"}
    run = {"stage": "release-regression", "state": "SUBMITTED", "source_sha": "c" * 40,
           "source_hashes": source,
           "source_identity": {"commit": "c" * 40, "tree": "d" * 40, "clean": True, "sha256": source},
           "jobs": [{"job": execution["job"], "output": str(output.parent),
                     "manifest": str(root / "job-0.json")}]}
    counts = {"tests": total, "failures": 0, "errors": 0, "skipped": 0}
    result = {"verdict": "PASS", "returncode": 0, "tests": counts,
              "targets": [test] if total == 13 else [test + "::test_cuda_rotated_extreme_coordinates"],
              "selection": "cuda_rotated" if total == 13 else "",
              "source_sha256": {key: digest for key, digest in source.items()
                                if key.startswith("src/openopt/") or key in (test, "scripts/gpu_release_check.py")}}
    files = {
        "run": (root / "run.json", run), "result": (output / "result.json", result),
        "submitted_job": (root / "job-0.json", {"kind": "fixture"}),
        "status": (root / "fixture-job-status.json", {"kind": "fixture"}),
        "pod_log": (root / "fixture-pod.log", b"synthetic\n"),
        "pytest_log": (output / "pytest.log", b"synthetic\n"),
        "execution_log": (output / "execution.log", b"synthetic\n"),
        "telemetry": (output / "telemetry.json", {"synthetic": True}),
        "source_manifest": (root / "source/source-hashes.json", source),
        "outer_result": (output.parent / "result.json", result),
    }
    junit = ET.Element("testsuites")
    suite = ET.SubElement(junit, "testsuite", {key: str(value) for key, value in counts.items()})
    scope = {name: digest for name, digest in source.items()
             if name not in ("scripts/gpu_release_check.py",)}
    sidecars, numeric = [], []
    for index, (identity, details) in enumerate(report.rsoc_identities(total).items()):
        classname, name = identity.split("::")
        case = ET.SubElement(suite, "testcase", classname=classname, name=name)
        properties = ET.SubElement(case, "properties")
        sidecar = output / "pytest-tmp" / str(index) / (
            "rotated-conic.json" if total == 13 else "rotated-conic-extreme.json")
        sidecar.parent.mkdir(parents=True)
        saved = {"source_sha256": scope, "synthetic": True}
        if total == 13:
            saved["test_identity"] = identity
            kind, name = details
            checks = {key: True for key in (
                "primal_equation", "primal_cone", "original_slack_cone", "dual_cone",
                "stationarity", "complementarity_block_0", "dual_objective_available", "duality_gap")}
            if kind != "projection":
                checks["complementarity_block_1"] = True
            cones = {key: {"block_0": True, **({"block_1": True} if kind != "projection" else {})}
                     for key in ("supplied_slack", "original_slack", "dual")}
            if kind == "original_rejection":
                checks[{"wrong-equality-dual": "stationarity", "planted-slack": "primal_equation",
                        "wrong-dual-cone": "dual_cone"}[name]] = False
                if name == "planted-slack":
                    checks["original_slack_cone"] = False
                    cones["original_slack"]["block_1"] = False
                if name == "wrong-dual-cone":
                    cones["dual"]["block_1"] = False
            member = {"identity": identity, "kind": kind, "case": name, "checks": checks,
                      "cone_checks": cones, "passed": all(checks.values())}
            if kind == "original_rejection":
                member["baseline_passed"] = True
            elif kind == "projection":
                member["projection"] = {
                    "component_checks": [True] * (4 if name == "tail-overflow-float64" else 3),
                    "moreau_orthogonal": True, "precision_crosscheck": [800, 1000],
                    "cone": {"block_0": True}, "negative_residual_dual": {"block_0": True},
                }
            numeric.append(member)
        else:
            name, dual = details
            saved.update(case=name, dual=dual,
                         nodeid=test + "::" + identity.split("::", 1)[1])
        descriptor = save(sidecar, saved)
        descriptor.pop("id")
        sidecars.append(descriptor)
        prop = "rotated_conic_acceptance" if total == 13 else "rsoc_cuda_extremes"
        for key, value in ((prop + "_path", descriptor["path"]), (prop + "_sha256", descriptor["sha256"])):
            ET.SubElement(properties, "property", name=key, value=value)
    files["junit"] = (output / "pytest.xml", ET.tostring(junit))
    originals = {}
    for name, (target, value) in files.items():
        descriptor = save(target, value)
        descriptor.pop("id")
        originals[name] = descriptor
    for name in ("run", "result", "junit"):
        manifest = json.loads(path.read_bytes())
        manifest["inputs"].append({"id": prefix + "_" + name, **originals[name]})
        save(path, manifest)
    if total == 13:
        pins = {"schema_version": 1, "run_root": str(root), "files": originals}
        pin = optional_record(path, prefix + "_pins", pins)
        bundle = {"schema_version": 1, "algorithm": "fixed13_original_rational_decimal_v1",
                  "verdict": "PASS", "denominator": {"positive": 10, "negative": 3, "total": 13},
                  "certified": False, "old_proof_authority": False, "commercial": "UNVERIFIED",
                  "performance": "UNVERIFIED", "caller_pins": pins, "sidecars": sidecars,
                  "numeric_cases": numeric, "execution": execution,
                  "source": {"execution_commit": run["source_sha"], "execution_tree": "d" * 40,
                             "audit_commit": AUDITOR_FIXTURE_COMMIT, "scope_sha256": scope}}
        optional_record(path, prefix + "_bundle", bundle)
        selected = next(row for row in json.loads(path.read_bytes())["inputs"]
                        if row["id"] == prefix + "_pins")
        committed = subprocess.run(["git", "show", AUDITOR_FIXTURE_COMMIT + ":scripts/accept_rotated_cuda.py"],
                                   cwd=report.ROOT, capture_output=True, check=False)
        assert committed.returncode == 0
        bundle_descriptor = next(row for row in json.loads(path.read_bytes())["inputs"]
                                 if row["id"] == prefix + "_bundle")
        evidence = {"schema_version": 2, "verdict": "PASS", "mode": "accept", "pins_sha256": selected["sha256"],
                    "bundle_sha256": bundle_descriptor["sha256"], "audit_commit": AUDITOR_FIXTURE_COMMIT,
                    "auditor_sha256": report.sha(committed.stdout), "certified": False, "qualified": False,
                    "false_accept_count": 0, "audit_runtime": {"python": "fixture", "implementation": "fixture"}}
    else:
        evidence = {"schema_version": 1, "verdict": "RSOC_EXTREMES_REPLAY_PASS", "tests": counts,
                    "source_commit": run["source_sha"], "qualified": False,
                    "remote_attestation": "NOT_PROVIDED", "execution": execution,
                    "inputs": [{**item, "bytes": Path(item["path"]).stat().st_size}
                               for item in [*originals.values(), *sidecars]],
                    "members": [{"case": name, "verdict": "ORIGINAL_COORDINATES_REPLAYED"}
                                for name in report.rsoc_identities(total)]}
    evidence.update(checked_at="2026-10-10T10:00:00+00:00",
                    commercial="UNVERIFIED", performance="UNVERIFIED")
    pin = optional_record(path, role, evidence)
    if total == 13:
        bind_synthetic_authority(path, {role: evidence})
    return pin, {role: evidence, prefix + "_run": run, prefix + "_result": result,
                 prefix + "_junit": ET.tostring(junit),
                 **({prefix + "_bundle": bundle, prefix + "_pins": pins} if total == 13 else {})}


@pytest.mark.parametrize("total", [13, 76])
def test_rsoc_complete_receipt_projection_deterministic_and_historical(inputs, total):
    path, _, _ = inputs
    pin, _ = rsoc_fixture(path, total)
    before = {item["path"]: Path(item["path"]).read_bytes()
              for item in json.loads(path.read_bytes())["inputs"]}
    files = rsoc_generate(path, pin)
    assert files == rsoc_generate(path, pin)
    assert before == {name: Path(name).read_bytes() for name in before}
    prefix = "rotated_cuda" if total == 13 else "rsoc_extremes"
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars[prefix + "_tests"] == total
    assert scalars[prefix + "_status"] == "PASS"
    assert scalars[prefix + "_source_relation"] == "historical"
    assert scalars[prefix + "_source"] != scalars["source_sha"]
    assert scalars["online_status"] == scalars["commercial_status"] == "UNVERIFIED"
    assert len(report.rsoc_identities(total)) == total
    assert b"synthetic source" not in files["benchmark-report-v4.html"]
    assert b"cuda-check-fixture" not in files["benchmark-report-v4.html"]
    assert "owner PASS 不代替独审签收" in files["design-v4.html"].decode()


@pytest.mark.parametrize("total", [13, 76])
@pytest.mark.parametrize("mutation", [
    "denominator", "drop_member", "duplicate_member", "unknown_member", "selected_pin",
    "source", "sidecar_hash", "junit_drop", "junit_duplicate", "snapshot_drift",
])
def test_rsoc_rehashed_false_receipt_rejected(inputs, total, mutation):
    path, _, _ = inputs
    _, records = rsoc_fixture(path, total)
    prefix = "rotated_cuda" if total == 13 else "rsoc_extremes"
    role = prefix + "_acceptance"
    record_role = prefix + "_bundle" if total == 13 else role
    value = records[record_role]
    members = value["numeric_cases" if total == 13 else "members"]
    if mutation == "denominator":
        if total == 13:
            value["denominator"]["total"] = 1
        else:
            value["tests"]["tests"] = 1
    elif mutation == "drop_member":
        members.pop()
    elif mutation == "duplicate_member":
        members[-1] = members[0]
    elif mutation == "unknown_member":
        members[0]["identity" if total == 13 else "case"] += "-unknown"
    elif mutation in ("selected_pin", "sidecar_hash"):
        original = (value["caller_pins"]["files"]["result"] if mutation == "selected_pin"
                    else value["sidecars"][0]) if total == 13 else value["inputs"][0 if mutation == "selected_pin" else 11]
        original["sha256"] = "0" * 64
    elif mutation == "source":
        if total == 13:
            value["source"]["execution_commit"] = "f" * 40
        else:
            value["source_commit"] = "f" * 40
    elif mutation == "snapshot_drift":
        run = records[prefix + "_run"]
        run_path = next(row["path"] for row in json.loads(path.read_bytes())["inputs"]
                        if row["id"] == prefix + "_run")
        (Path(run_path).parent / "source" / next(iter(run["source_hashes"]))).write_bytes(b"# changed\n")
    else:
        tree = ET.fromstring(records[prefix + "_junit"])
        suite = tree.find("testsuite")
        cases = suite.findall("testcase")
        if mutation == "junit_drop":
            suite.remove(cases[-1])
            suite.set("tests", str(total - 1))
        else:
            cases[-1].set("name", cases[0].get("name"))
        records[prefix + "_junit"] = ET.tostring(tree)
        digest = report.sha(records[prefix + "_junit"])
        if total == 13:
            value["caller_pins"]["files"]["junit"]["sha256"] = digest
            records[prefix + "_pins"]["files"]["junit"]["sha256"] = digest
        else:
            original = next(item for item in value["inputs"] if item["path"].endswith("/pytest.xml"))
            original.update(sha256=digest, bytes=len(records[prefix + "_junit"]))
        reseal(path, records, prefix + "_junit")
    if total == 13:
        reseal(path, records, prefix + "_pins")
        descriptor = next(row for row in json.loads(path.read_bytes())["inputs"]
                          if row["id"] == prefix + "_pins")
        records[role]["pins_sha256"] = descriptor["sha256"]
        reseal(path, records, role)
        reseal(path, records, record_role)
        records[role]["bundle_sha256"] = next(row["sha256"] for row in json.loads(path.read_bytes())["inputs"]
                                              if row["id"] == record_role)
        reseal(path, records, role)
        bind_synthetic_authority(path, records)
    with pytest.raises(report.Rejected):
        rsoc_generate(path, reseal(path, records, record_role))


@pytest.mark.parametrize("mutation", ["top_only", "seal_only", "numeric_replay", "hidden_check",
                                     "baseline", "projection", "pins_mismatch"])
def test_fixed13_top_pass_and_hidden_math_do_not_qualify(inputs, mutation):
    path, _, _ = inputs
    _, records = rsoc_fixture(path, 13)
    role = "rotated_cuda_acceptance"
    bundle = records["rotated_cuda_bundle"]
    if mutation == "top_only":
        manifest = json.loads(path.read_bytes())
        manifest["inputs"] = [item for item in manifest["inputs"] if item["id"] != "rotated_cuda_bundle"]
        save(path, manifest)
    elif mutation in ("seal_only", "numeric_replay"):
        records[role]["mode"] = "seal" if mutation == "seal_only" else "numeric-replay"
    elif mutation == "hidden_check":
        bundle["numeric_cases"][0]["checks"]["stationarity"] = False
    elif mutation == "baseline":
        bundle["numeric_cases"][-1]["baseline_passed"] = False
    elif mutation == "projection":
        bundle["numeric_cases"][4]["projection"]["component_checks"][0] = False
    else:
        records[role]["pins_sha256"] = "0" * 64
    if mutation != "top_only":
        reseal(path, records, "rotated_cuda_bundle")
        records[role]["bundle_sha256"] = next(row["sha256"] for row in json.loads(path.read_bytes())["inputs"]
                                              if row["id"] == "rotated_cuda_bundle")
    reseal(path, records, role)
    if mutation != "top_only":
        bind_synthetic_authority(path, records)
    with pytest.raises(report.Rejected):
        rsoc_generate(path, reseal(path, records, role))


def test_rsoc_omitted_and_failed_roles_never_imply_full_release(inputs):
    path, pin, _ = inputs
    assert not any(key.startswith(("rotated_cuda_", "rsoc_extremes_"))
                   for key in json.loads(report.generate(path, pin)["summary.json"])["scalars"])
    value = {"verdict": "FAIL", "checked_at": "2026-10-10T10:00:00+00:00",
             "commercial": "UNVERIFIED", "performance": "UNVERIFIED",
             "error": "/private/hidden"}
    files = report.generate(path, optional_record(path, "rsoc_extremes_acceptance", value))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["rsoc_extremes_status"] == "FAIL"
    assert "rsoc_extremes_tests" not in scalars
    assert b"/private/hidden" not in files["design-v4.html"]


@pytest.mark.parametrize("total", [13, 76])
def test_rsoc_false_current_source_rejected(inputs, total):
    path, _, records = inputs
    _, added = rsoc_fixture(path, total)
    records.update(added)
    records["metadata"]["source_sha"] = "c" * 40
    manifest = json.loads(path.read_bytes())
    manifest["source_sha"] = "c" * 40
    save(path, manifest)
    with_source_gate(records)
    reseal(path, records, "cpu")
    with pytest.raises(report.Rejected, match="rsoc_current_source_mismatch"):
        rsoc_generate(path, reseal(path, records, "metadata"))


@pytest.mark.parametrize("total", [13, 76])
def test_rsoc_current_source_requires_matching_clean_cpu_snapshot(inputs, total):
    path, _, records = inputs
    _, added = rsoc_fixture(path, total)
    records.update(added)
    prefix = "rotated_cuda" if total == 13 else "rsoc_extremes"
    records["metadata"]["source_sha"] = records[prefix + "_run"]["source_sha"]
    manifest = json.loads(path.read_bytes())
    manifest["source_sha"] = records["metadata"]["source_sha"]
    save(path, manifest)
    with_source_gate(records)
    records["cpu"]["source"]["sha256"] = records[prefix + "_run"]["source_hashes"]
    records["cpu"]["source_consistency"]["files"] = len(records["cpu"]["source"]["sha256"])
    reseal(path, records, "cpu")
    files = rsoc_generate(path, reseal(path, records, "metadata"))
    assert files == rsoc_generate(path, report.sha(path.read_bytes()))
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars[prefix + "_source_relation"] == "current"
    assert scalars["commercial_status"] == scalars["online_status"] == "UNVERIFIED"
    assert scalars["regression_scope"] == "intermediate_snapshot"


@pytest.mark.parametrize("total", [13, 76])
@pytest.mark.parametrize("mutation", ["denominator", "drop_member"])
def test_rsoc_rehashed_invalid_receipt_cli_rejects_with_evidence(inputs, total, mutation):
    path, _, _ = inputs
    _, records = rsoc_fixture(path, total)
    prefix = "rotated_cuda" if total == 13 else "rsoc_extremes"
    role = prefix + ("_bundle" if total == 13 else "_acceptance")
    value = records[role]
    if mutation == "denominator":
        if total == 13:
            value["denominator"]["total"] -= 1
        else:
            value["tests"]["tests"] -= 1
    else:
        value["numeric_cases" if total == 13 else "members"].pop()
    pin = reseal(path, records, role)
    if total == 13:
        records["rotated_cuda_acceptance"]["bundle_sha256"] = next(
            row["sha256"] for row in json.loads(path.read_bytes())["inputs"] if row["id"] == role)
        pin = reseal(path, records, "rotated_cuda_acceptance")
        bind_synthetic_authority(path, records)
    with pytest.raises(report.Rejected) as error:
        rsoc_generate(path, pin)
    expected = ("rsoc_fixed13_bundle_contract" if total == 13 else "rsoc_extremes_contract"
                ) if mutation == "denominator" else "rsoc_numeric_membership"
    assert error.value.code == expected
    output = path.parent / "rejected-render"
    command = [sys.executable, "-B", str(report.ROOT / "scripts/render_release_report.py"),
               "--manifest", str(path), "--manifest-sha256", pin, "--output-dir", str(output),
               *rsoc_cli_authority(path)]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    (path.parent / "rsoc-negative-cli.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 1
    assert f"RELEASE_REPORT verdict=FAIL code={expected}" in result.stderr
    assert not output.exists(), "rejected receipt must not write a successful output bundle"
    (path.parent / "rsoc-negative-receipt.json").write_bytes(report.encoded({
        "verdict": "REJECTED", "total": total, "mutation": mutation,
        "returncode": result.returncode, "first_failure": expected,
        "rehashed_manifest_sha256": pin, "output_absent": not output.exists(),
        "cuda_execution": False, "qualified": False,
    }))


def test_real_pinned_rsoc_archives_project_only(inputs):
    """Read two explicit historical receipts; no producer or GPU invocation."""
    path, _, _ = inputs
    base = Path("/hf3fs-jd/data/tmp/openopt-release-20261010")
    selections = {
        "rsoc_extremes_acceptance": (
            base / "owner-main-c0565b0-rsoc-replay-v1/rsoc-audit-tgd87z6i/acceptance.json",
            "0c0175fd1b6166106cdb34c248c8c16cdd0636ff356e1f8f2117a65f901da68f"),
        "rotated_cuda_acceptance": (base / "fixed13-numeric-audit/historical-accept/acceptance.json",
                                    "dbc3a1e5699c5d3feb904465344eaf11972c6a7e80bd1516904de3b972d5b7f3"),
        "rotated_cuda_bundle": (base / "fixed13-numeric-audit/historical-accept/bundle.json",
                               "ee47f512e6790f6a7044c1690548236021d2fd898ebcea33949b3494198f7a3e"),
        "rotated_cuda_pins": (base / "fixed13-numeric-audit/historical-inventory-v1/pins.json",
                              "a0d802d9e496df48c6ad81cf016758047f8d07860c93af75d72b1e7784128b51"),
    }
    if not all(target.is_file() for target, _ in selections.values()):
        pytest.skip("explicit local historical receipts absent; fixtures do not qualify them")
    for role, (target, pin) in selections.items():
        payload = target.read_bytes()
        assert report.sha(payload) == pin, "explicit historical caller pin changed"
    extremes = json.loads(selections["rsoc_extremes_acceptance"][0].read_bytes())
    fixed = json.loads(selections["rotated_cuda_pins"][0].read_bytes())
    for prefix, originals in (("rsoc_extremes", extremes["inputs"]),
                              ("rotated_cuda", list(fixed["files"].values()))):
        for name in ("run", "result", "junit"):
            target = (Path(fixed["files"][name]["path"]) if prefix == "rotated_cuda" else
                      Path(extremes["inputs"][("result", "run", "junit").index(name)]["path"]))
            original = next(item for item in originals if Path(item["path"]) == target)
            selections[prefix + "_" + name] = (target, original["sha256"])
    manifest = json.loads(path.read_bytes())
    manifest["inputs"].extend({"id": role, "path": str(target), "sha256": pin}
                              for role, (target, pin) in selections.items())
    pin = save(path, manifest)["sha256"]
    with pytest.raises(report.Rejected, match="rsoc_fixed13_success_schema_unsupported"):
        report.generate(path, pin)
    old_output = path.parent / "unsupported-old-fixed13"
    old_command = [sys.executable, "-B", str(report.ROOT / "scripts/render_release_report.py"),
                   "--manifest", str(path), "--manifest-sha256", pin, "--output-dir", str(old_output)]
    old_result = subprocess.run(old_command, capture_output=True, text=True, check=False)
    assert old_result.returncode == 1 and not old_output.exists()
    assert "code=rsoc_fixed13_success_schema_unsupported" in old_result.stderr
    save(path.parent / "historical-fixed13-unsupported.json", {
        "verdict": "UNSUPPORTED_OLD_SUCCESS", "acceptance_sha256": selections["rotated_cuda_acceptance"][1],
        "bundle_sha256": selections["rotated_cuda_bundle"][1], "cli_returncode": old_result.returncode,
        "first_failure": "rsoc_fixed13_success_schema_unsupported", "backfilled": False,
        "cuda_execution": False, "qualified": False,
    })
    mutated = json.loads(selections["rotated_cuda_bundle"][0].read_bytes())
    mutated["numeric_cases"][0]["metrics"]["signed_duality_gap"] = "999999999"
    copy = save(path.parent / "old-fixed13-mutated-bundle.json", mutated)
    bundle_row = next(item for item in manifest["inputs"] if item["id"] == "rotated_cuda_bundle")
    bundle_row.update(path=copy["path"], sha256=copy["sha256"])
    mutated_pin = save(path, manifest)["sha256"]
    with pytest.raises(report.Rejected, match="rsoc_fixed13_success_schema_unsupported"):
        report.generate(path, mutated_pin)
    mutated_command = [sys.executable, "-B", str(report.ROOT / "scripts/render_release_report.py"),
                       "--manifest", str(path), "--manifest-sha256", mutated_pin,
                       "--output-dir", str(path.parent / "rejected-real-metric-drift")]
    rejected = subprocess.run(mutated_command, capture_output=True, text=True, check=False)
    assert rejected.returncode == 1 and "rsoc_fixed13_success_schema_unsupported" in rejected.stderr
    assert not (path.parent / "rejected-real-metric-drift").exists()
    save(path.parent / "real-bundle-drift-rejected.json", {
        "verdict": "REJECTED", "acceptance_sha256_unchanged": selections["rotated_cuda_acceptance"][1],
        "original_bundle_sha256": selections["rotated_cuda_bundle"][1], "modified_bundle_sha256": copy["sha256"],
        "api_rejected": True, "cli_returncode": rejected.returncode,
        "first_failure": "rsoc_fixed13_success_schema_unsupported", "backfilled": False,
        "cuda_execution": False, "qualified": False,
    })
    manifest["inputs"] = [item for item in manifest["inputs"] if not item["id"].startswith("rotated_cuda_")]
    pin = save(path, manifest)["sha256"]
    files = report.generate(path, pin)
    assert files == report.generate(path, pin)
    scalars = json.loads(files["summary.json"])["scalars"]
    assert scalars["rsoc_extremes_tests"] == 76 and "rotated_cuda_tests" not in scalars
    assert scalars["rsoc_extremes_status"] == "PASS"
    assert scalars["rsoc_extremes_source_relation"] == "historical"
    assert scalars["online_status"] == scalars["commercial_status"] == "UNVERIFIED"
    output = path.parent / "historical-render"
    report.write_bundle(output, files)
    report.write_bundle(output, files, check=True)
    command = [sys.executable, "-B", str(report.ROOT / "scripts/render_release_report.py"),
               "--manifest", str(path), "--manifest-sha256", pin,
               "--output-dir", str(output), "--check"]
    result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
    (path.parent / "real-rsoc-cli.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    assert "RELEASE_REPORT verdict=PASS" in result.stdout
    (path.parent / "real-rsoc-projection.json").write_bytes(report.encoded({
        "verdict": "PASS", "scope": "caller-pinned historical receipt projection only",
        "cuda_execution": False, "qualified": False,
        "cli_returncode": result.returncode,
        "counts": {"rsoc_extremes": 76}, "fixed13": "UNSUPPORTED_OLD_SUCCESS",
        "receipts": {role: {"path": str(target), "sha256": pin}
                     for role, (target, pin) in selections.items()},
        "public_manifest_sha256": report.sha(files["manifest.json"]),
    }))


@pytest.mark.parametrize("role", ["rotated_cuda_acceptance", "rsoc_extremes_acceptance"])
def test_real_minimal_auditor_fail_is_visible_without_success_fields(inputs, role):
    path, _, _ = inputs
    evidence = {"verdict": "FAIL", "checked_at": "2026-10-10T10:00:00+00:00",
                "error": "/private/not-public"}
    files = report.generate(path, optional_record(path, role, evidence))
    prefix = role.removesuffix("_acceptance")
    assert json.loads(files["summary.json"])["scalars"][prefix + "_status"] == "FAIL"
    assert b"/private/not-public" not in files["benchmark-report-v4.html"]


@pytest.mark.parametrize("total,field,value", [
    (13, "qualified", True), (13, "certified", True), (13, "false_accept_count", 1),
    (13, "false_accept_count", False), (76, "certified", True), (76, "false_accept_count", 1),
    (76, "qualified", True), (13, "extra", False), (76, "extra", False),
    (13, "schema_version", 2.0), (76, "schema_version", True),
])
def test_rsoc_strict_success_schema_conflicting_flags_resealed(inputs, total, field, value):
    path, _, _ = inputs
    pin, records = rsoc_fixture(path, total)
    assert rsoc_generate(path, pin)
    role = ("rotated_cuda" if total == 13 else "rsoc_extremes") + "_acceptance"
    records[role][field] = value
    pin = reseal(path, records, role)
    if total == 13:
        bind_synthetic_authority(path, records)
    with pytest.raises(report.Rejected):
        rsoc_generate(path, pin)


@pytest.mark.parametrize("mutation", ["missing", "old_schema", "bad_descriptor", "stale_authority",
    "acceptance", "bundle", "auditor_zero", "auditor_bytes", "auditor_commit", "source_rehashed",
    "uncommitted_source", "authority_rehashed_not_repinned", "source_missing"])
def test_fixed13_caller_authority_resealed_mutations_fail_closed(inputs, mutation, request):
    path, _, _ = inputs
    pin, records = rsoc_fixture(path, 13)
    assert rsoc_generate(path, pin)
    descriptor = deepcopy(RSOC_AUTHORITIES[path])
    authority_path = Path(descriptor["path"])
    authority = json.loads(authority_path.read_bytes())
    entry = authority["receipts"]["rotated_cuda"]
    acceptance, bundle = records["rotated_cuda_acceptance"], records["rotated_cuda_bundle"]
    if mutation == "missing":
        descriptor = None
    elif mutation == "old_schema":
        acceptance.pop("schema_version")
        pin = reseal(path, records, "rotated_cuda_acceptance")
    elif mutation == "bad_descriptor":
        descriptor["extra"] = True
    elif mutation == "stale_authority":
        descriptor["sha256"] = "0" * 64
    elif mutation == "acceptance":
        acceptance["checked_at"] = "2026-10-10T11:00:00+00:00"
        pin = reseal(path, records, "rotated_cuda_acceptance")
    elif mutation == "bundle":
        bundle["numeric_cases"][0]["metrics"] = {"signed_duality_gap": "999999999"}
        pin = reseal(path, records, "rotated_cuda_bundle")
    elif mutation == "auditor_zero":
        acceptance["auditor_sha256"] = "0" * 64
        pin = reseal(path, records, "rotated_cuda_acceptance")
        entry["acceptance_sha256"] = next(row["sha256"] for row in json.loads(path.read_bytes())["inputs"]
                                            if row["id"] == "rotated_cuda_acceptance")
        descriptor["sha256"] = save(authority_path, authority)["sha256"]
    elif mutation in ("auditor_bytes", "auditor_commit"):
        entry["auditor"]["bytes" if mutation == "auditor_bytes" else "commit"] = (
            entry["auditor"]["bytes"] + 1 if mutation == "auditor_bytes" else "f" * 40)
        descriptor["sha256"] = save(authority_path, authority)["sha256"]
    elif mutation in ("source_rehashed", "uncommitted_source"):
        payload = b"raise RuntimeError(\"MUST NEVER EXECUTE ARTIFACT SOURCE\")\n"
        Path(entry["auditor"]["path"]).write_bytes(payload)
        if mutation == "uncommitted_source":
            entry["auditor"].update(sha256=report.sha(payload), bytes=len(payload))
            acceptance["auditor_sha256"] = report.sha(payload)
            pin = reseal(path, records, "rotated_cuda_acceptance")
            entry["acceptance_sha256"] = next(row["sha256"] for row in json.loads(path.read_bytes())["inputs"]
                                                if row["id"] == "rotated_cuda_acceptance")
            descriptor["sha256"] = save(authority_path, authority)["sha256"]
    elif mutation == "source_missing":
        entry["auditor"]["path"] = str(path.parent / "never-created.py")
        descriptor["sha256"] = save(authority_path, authority)["sha256"]
    else:
        entry["auditor"]["bytes"] += 1
        save(authority_path, authority)
    with pytest.raises(report.Rejected) as error:
        report.generate(path, pin, rsoc_authority=descriptor)
    save(path.parent / "binding-negative.json", {"verdict": "REJECTED", "mutation": mutation,
        "first_failure": error.value.code, "manifest_sha256": pin, "authority": descriptor,
        "artifact_source_executed": False, "cuda_execution": False, "qualified": False})
    request.node.user_properties.append(("report_binding_negative", str(path.parent / "binding-negative.json")))
    if mutation == "bundle":
        assert error.value.code == "rsoc_fixed13_bundle_binding"
    if mutation == "uncommitted_source":
        assert error.value.code == "rsoc_fixed13_committed_auditor_source"


def test_fixed13_v2_explicit_caller_cli_write_check_and_no_authority_rejection(inputs):
    path, _, _ = inputs
    pin, _ = rsoc_fixture(path, 13)
    output = path.parent / "v2-caller-render"
    command = [sys.executable, "-B", str(report.ROOT / "scripts/render_release_report.py"),
               "--manifest", str(path), "--manifest-sha256", pin, "--output-dir", str(output)]
    absent = subprocess.run(command, capture_output=True, text=True, check=False)
    assert absent.returncode == 1 and not output.exists()
    assert "rsoc_fixed13_authority_required" in absent.stderr
    written = subprocess.run([*command, *rsoc_cli_authority(path)], capture_output=True, text=True, check=False)
    assert written.returncode == 0
    checked = subprocess.run([*command, *rsoc_cli_authority(path), "--check"],
                             capture_output=True, text=True, check=False)
    assert checked.returncode == 0
    files = rsoc_generate(path, pin)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == files
    save(path.parent / "synthetic-v2-cli.json", {"verdict": "SYNTHETIC_CONTRACT_PASS",
        "without_authority_returncode": absent.returncode, "write_returncode": written.returncode,
        "check_returncode": checked.returncode, "manifest_sha256": pin,
        "authority": RSOC_AUTHORITIES[path], "cuda_execution": False, "qualified": False})


@pytest.mark.parametrize("field,value", [("qualified", True), ("false_accept_count", 1),
                                       ("unknown", False), ("schema_version", True)])
def test_fixed13_bundle_schema_rejects_coordinated_caller_repins(inputs, field, value):
    path, _, _ = inputs
    _, records = rsoc_fixture(path, 13)
    records["rotated_cuda_bundle"][field] = value
    reseal(path, records, "rotated_cuda_bundle")
    records["rotated_cuda_acceptance"]["bundle_sha256"] = next(
        row["sha256"] for row in json.loads(path.read_bytes())["inputs"] if row["id"] == "rotated_cuda_bundle")
    pin = reseal(path, records, "rotated_cuda_acceptance")
    bind_synthetic_authority(path, records)
    with pytest.raises(report.Rejected, match="rsoc_fixed13_bundle_schema"):
        rsoc_generate(path, pin)


def test_fixed13_unused_authority_rejected_without_upgrade(inputs):
    path, _, _ = inputs
    _, records = rsoc_fixture(path, 13)
    records["rotated_cuda_acceptance"] = {"verdict": "FAIL", "checked_at": "2026-10-10T10:00:00+00:00"}
    pin = reseal(path, records, "rotated_cuda_acceptance")
    with pytest.raises(report.Rejected, match="rsoc_unused_authority"):
        rsoc_generate(path, pin)
    files = report.generate(path, pin)
    assert json.loads(files["summary.json"])["scalars"]["rotated_cuda_status"] == "FAIL"


def scan_submission_with_existing_gate(paths=None, source_root=None):
    scanner = Path("/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    assert scanner.is_file(), "required local precommit scanner missing"
    check = subprocess.run([sys.executable, "-B", str(scanner), "--selftest"],
                           capture_output=True, text=True, check=False)
    assert check.returncode == 0 and "verdict=PASS" in check.stdout, "secret scanner selftest failed"
    command = [sys.executable, "-B", str(scanner), "--no-grep"]
    source_root = report.ROOT if source_root is None else Path(source_root).resolve()
    assert source_root.is_dir(), "explicit source root must exist"
    paths = paths or (
        "scripts/render_release_report.py", "tests/test_render_release_report.py",
        "docs/research/release-report.md",
    )
    assert len(paths) == len(set(paths)) and paths, "require unique explicit source paths"
    for name in paths:
        relative = Path(name)
        target = source_root / relative
        assert (not relative.is_absolute() and ".." not in relative.parts
                and target.is_file() and not target.is_symlink()
                and target.resolve().is_relative_to(source_root.resolve()))
        command.extend(["--path", str(target)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, "local credential-literal gate failed; output withheld"
    verdict = next(line for line in result.stdout.splitlines() if line.startswith("SECRET_SCAN:"))
    assert "literal_hits=0" in verdict and "needles=1" in verdict
    shape = subprocess.run([sys.executable, "-B", str(scanner), "--no-needle", *command[4:]],
                           capture_output=True, text=True, check=False)
    assert shape.returncode == 0, "redaction-safe shape scan failed"
    print(f"REPORT_SUBMISSION_SCAN verdict=PASS files={len(paths)} "
          "literal_hits=0 scanner_selftest=PASS")


def replay_original_review(artifact_root):
    """Run exact submitted reviewer bytes against this repaired renderer.

    Only test ROOT is routed to the repair tree for its CLI subprocesses. No
    assertion or subject pin is changed. Expected contract-incompatibility
    failures remain real failures, never used as repaired qualification.
    """
    from scripts.verify import source_snapshot, check_source_consistency

    frozen_root = Path("/data/dev/openopt-wt-release4-20261010-rsoc-report-independent")
    target = frozen_root / "tests/test_rsoc_report_independent.py"
    commit = "1ad740a741a31c19171f60ea7760e2a74ffc9254"
    committed = subprocess.run(["git", "show", commit + ":tests/test_rsoc_report_independent.py"],
                               cwd=report.ROOT, capture_output=True, check=False)
    assert committed.returncode == 0 and committed.stdout == target.read_bytes(), "original reviewer byte drift"
    artifact_root = report.shared_path(artifact_root, "review")
    artifact_root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="original32-", dir=artifact_root))
    before, started = source_snapshot(report.ROOT), time.perf_counter()
    receipt = {"schema_version": 1, "verdict": "RUNNING", "reviewer_commit": commit,
               "reviewer_sha256": report.sha(committed.stdout), "repair_source": before,
               "started_at": datetime.now(timezone.utc).isoformat(), "cuda_execution": False,
               "assertions_changed": False, "test_root_routed_for_current_cli": True}
    expected = []

    class FrozenReview:
        def pytest_collection_modifyitems(self, items):
            for item in items:
                assert Path(item.module.__file__).resolve() == target.resolve()
                assert item.module.renderer is report, "original review imported old renderer"
                item.module.ROOT = report.ROOT
                expected.append(item.nodeid)

    os.environ.update(PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", OPENOPT_RUN_RSOC_REPORT_EVIDENCE_TESTS="1",
                      PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1")
    junit = directory / "pytest.xml"
    try:
        with (directory / "pytest.log").open("x", encoding="utf-8") as stream:
            previous_out, previous_err = sys.stdout, sys.stderr
            try:
                sys.stdout = sys.stderr = stream
                rc = int(pytest.main(["--import-mode=importlib", "-q", "-ra", "--rootdir", str(frozen_root),
                    "-o", "cache_dir=" + str(directory / "pytest-cache"), "--basetemp", str(directory / "pytest-tmp"),
                    "--junitxml=" + str(junit), str(target)], plugins=[FrozenReview()]))
            finally:
                sys.stdout, sys.stderr = previous_out, previous_err
        cases = list(ET.parse(junit).getroot().iter("testcase"))
        observed = ["tests/test_rsoc_report_independent.py::" + case.get("name", "") for case in cases]
        normalized = [item[item.index("tests/test_rsoc_report_independent.py::"):] for item in expected]
        assert len(normalized) == len(set(normalized)) == len(observed) == len(set(observed)) == 32
        assert set(normalized) == set(observed), "original32 collection/JUnit membership"
        counts = {"tests": len(cases), "failures": sum(c.find("failure") is not None for c in cases),
                  "errors": sum(c.find("error") is not None for c in cases),
                  "skipped": sum(c.find("skipped") is not None for c in cases)}
        consistency = check_source_consistency(before, source_snapshot(report.ROOT))
        assert consistency["verdict"] == "PASS"
        receipt.update(verdict="CONTRACT_INCOMPATIBLE" if rc else "PASS", pytest_returncode=rc, tests=counts,
            membership={"verdict": "PASS", "expected": normalized, "observed": observed},
            source_consistency=consistency, failed_nodeids=[name for name, case in zip(observed, cases)
                if case.find("failure") is not None or case.find("error") is not None],
            junit={"path": str(junit), "sha256": report.sha(junit.read_bytes())})
    except Exception as error:
        receipt.update(verdict="ERROR", error=type(error).__name__)
        raise
    finally:
        receipt.update(finished_at=datetime.now(timezone.utc).isoformat(), wall_seconds=time.perf_counter() - started)
        output = directory / "original-review.json"
        save(output, receipt)
        print("ORIGINAL_RSOC_REVIEW verdict=" + receipt["verdict"] + " receipt=" + str(output)
              + " sha256=" + report.sha(output.read_bytes()), flush=True)
    return rc


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Safe explicit-file submission scan")
    parser.add_argument("--scan-submission", action="store_true")
    parser.add_argument("--replay-original-review", action="store_true")
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--path", action="append")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    if args.replay_original_review:
        assert not args.scan_submission and not args.path and args.source_root is None and args.artifact_dir is not None
        raise SystemExit(replay_original_review(args.artifact_dir))
    assert args.scan_submission and args.artifact_dir is None
    scan_submission_with_existing_gate(args.path, args.source_root)
