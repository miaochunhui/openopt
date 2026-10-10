"""CPU-only publication fixtures and resealed leakage/tampering controls.

All fixtures, bundles and CLI logs stay in pytest's external artifact directory.
No repository source is executed from the synthetic committed tree.
"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from scripts import prepare_publication as gate


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gate.encoded(value))


def command(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    assert result.returncode == 0, "synthetic git fixture failed"
    return result.stdout.decode().strip()


@pytest.fixture
def fixture(tmp_path):
    assert tmp_path.resolve().is_relative_to(gate.SHARED), "use the external verify entry"
    repo, evidence = tmp_path / "repo", tmp_path / "evidence"
    repo.mkdir()
    evidence.mkdir()
    document = repo / "docs/report-v4.html"
    document.parent.mkdir()
    document.write_text(
        "<h1>Research, not commercial qualification</h1>"
        "<p>Latency 1.25 ms; status PASS.</p>\n")
    producer = repo / "scripts/measure.py"
    producer.parent.mkdir()
    producer.write_text('"""Synthetic provenance declaration; never executed."""\n')
    command(repo, "init", "-q")
    command(repo, "-c", "user.name=Publication Fixture", "-c", "user.email=fixture@example.invalid",
            "add", "--", "docs/report-v4.html", "scripts/measure.py")
    command(repo, "-c", "user.name=Publication Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-q", "-m", "synthetic documentation")
    # Private fields are intentionally omitted, not copied then regex-redacted.
    record = {"measurement": {"latency_ms": 1.25, "status": "PASS"},
              "private": {"host": "private-fixture", "license_material": "private-fixture"}}
    dump(evidence / "result.json", record)
    plan = {
        "schema_version": 1, "source_commit": command(repo, "rev-parse", "HEAD"),
        "docs_version": "v4",
        "documents": [{"path": "docs/report-v4.html", "sha256": gate.sha(document.read_bytes())}],
        "evidence": [{"id": "run", "path": "result.json",
                      "sha256": gate.sha((evidence / "result.json").read_bytes()),
                      "producer": "scripts/measure.py"}],
        "claims": [
            {"id": "latency", "evidence": "run", "pointer": "/measurement/latency_ms",
             "kind": "number", "unit": "ms", "document": "docs/report-v4.html", "text": "1.25"},
            {"id": "status", "evidence": "run", "pointer": "/measurement/status",
             "kind": "status", "unit": "", "document": "docs/report-v4.html", "text": "PASS"},
        ],
    }
    dump(evidence / "plan.json", plan)
    assert command(repo, "status", "--porcelain") == ""
    return tmp_path, repo, evidence, plan


def preparation(fixture, plan=None):
    root, repo, evidence, original = fixture
    dump(evidence / "plan.json", original if plan is None else plan)
    return gate.Preparation(repo, evidence, evidence / "plan.json",
                            gate.sha((evidence / "plan.json").read_bytes()))


def test_deterministic_archive_readback_and_scalar_provenance(fixture):
    root, repo, evidence, _ = fixture
    tool = preparation(fixture)
    first = tool.prepare(root / "first")
    second = tool.prepare(root / "second")
    assert first == second and first["verdict"] == "PREPARED"
    assert first["online_publication"] == first["commercial_qualification"] == "UNVERIFIED"
    assert (root / "first/bundle.zip").read_bytes() == (root / "second/bundle.zip").read_bytes()
    with zipfile.ZipFile(root / "first/bundle.zip") as archive:
        assert set(archive.namelist()) == {
            "docs/report-v4.html", "public-evidence.json", "manifest.json", "version.json"}
        for item in archive.infolist():
            assert item.date_time == (1980, 1, 1, 0, 0, 0)
        public = json.loads(archive.read("public-evidence.json"))
        assert len(public["claims"]) == 2 and public["raw_evidence_included"] is False
        latency = public["claims"][0]
        assert latency["value"] == 1.25
        assert latency["origin"]["json_pointer_tokens"] == ["measurement", "latency_ms"]
        assert latency["origin"]["evidence_sha256"] == gate.sha((evidence / "result.json").read_bytes())
        assert b"1.25 ms; status PASS" in archive.read("docs/report-v4.html")
        assert archive.read("docs/report-v4.html") == (repo / "docs/report-v4.html").read_bytes()
        for name in archive.namelist():
            assert b"private-fixture" not in archive.read(name)
            assert str(repo).encode() not in archive.read(name)
            assert str(evidence).encode() not in archive.read(name)
    assert tool.verify(root / "first", first["bundle_sha256"]) == first
    before = (root / "first/bundle.zip").read_bytes()
    with pytest.raises(gate.Rejected, match="output_already_exists"):
        tool.prepare(root / "first")
    assert (root / "first/bundle.zip").read_bytes() == before


@pytest.mark.parametrize("payload,code", [
    ("/data/internal/report.json", "absolute_path"),
    ("/private-file", "absolute_path"),
    ("&#47;home&#47;internal&#47;file", "absolute_path"),
    ("%2fdata%2finternal%2ffile", "absolute_path"),
    ("\\u002fdata\\u002finternal\\u002ffile", "absolute_path"),
    ("C:\\private\\file", "absolute_path"),
    ("h3c2", "host_identity"),
    ("hostname = fixture", "host_identity"),
    ("https" + "://example.invalid/service", "network_entry"),
    ("192.168.1.17", "network_entry"),
    ("[fe80::1]", "network_entry"),
    ("operator@fixture", "network_entry"),
    ("pass" + "word = synthetic-fixture", "credential"),
    ("Bearer " + "synthetic-fixture", "credential"),
    ("-----" + "BEGIN RSA PRIVATE KEY" + "-----", "credential"),
    ("license " + "server 123", "license_material"),
    ("SERVER " + "fixture 123", "license_material"),
    ("<script>run()</script>", "active_content"),
    ("<p onclick='run()'>x</p>", "active_content"),
    ("private\u200bfixture", "hidden_character"),
    ("", "empty_content"),
    ("\x00", "hidden_character"),
])
def test_leakage_rules_have_real_positive_controls(payload, code):
    with pytest.raises(gate.Rejected, match=code):
        gate.scan(payload.encode(), "synthetic_control")


@pytest.mark.parametrize("payload", [
    "<p>Research v4, 0BSD project code; commercial qualification is UNVERIFIED.</p>",
    "docs/report-v4.html", "Latency 1.25 ms.", "PASS", "# Mathematical QP / LP\n",
])
def test_scanner_negative_controls(payload):
    gate.scan(payload.encode(), "safe_control")


@pytest.mark.parametrize("mutation", [
    "dirty_document", "staged_document", "untracked_source", "changed_evidence",
    "changed_plan", "touch_evidence", "source_commit",
])
def test_changed_inputs_rejected(fixture, mutation):
    root, repo, evidence, _ = fixture
    tool = preparation(fixture)
    tool.build()
    if mutation in ("dirty_document", "staged_document"):
        (repo / "docs/report-v4.html").write_text("<p>changed</p>")
        if mutation == "staged_document":
            command(repo, "add", "--", "docs/report-v4.html")
    elif mutation == "untracked_source":
        (repo / "unknown.py").write_text("# not committed\n")
    elif mutation == "changed_evidence":
        dump(evidence / "result.json", {"measurement": {"latency_ms": 2.0, "status": "PASS"}})
    elif mutation == "changed_plan":
        (evidence / "plan.json").write_bytes(b"{}\n")
    elif mutation == "touch_evidence":
        target = evidence / "result.json"
        current = target.stat()
        replacement = evidence / "replacement.json"
        replacement.write_bytes(target.read_bytes())
        replacement.replace(target)
        assert target.stat().st_ino != current.st_ino, "metadata control must change identity"
    else:
        command(repo, "-c", "user.name=Publication Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "--allow-empty", "-q", "-m", "changed commit")
    with pytest.raises(gate.Rejected, match="uncommitted_source|changed_evidence|source_commit_mismatch"):
        tool.prepare(root / "rejected")
    assert not (root / "rejected").exists()


@pytest.mark.parametrize("mutation", [
    "empty_documents", "empty_claims", "empty_evidence", "unused_claim",
    "duplicate_claim", "missing_field", "sensitive_field", "object_not_scalar",
    "boolean_number", "status_published", "version", "path_escape", "absolute_path",
    "wrong_text", "missing_text", "wrong_unit",
])
def test_resealed_plan_semantic_controls(fixture, mutation):
    root, _, evidence, original = fixture
    plan = copy.deepcopy(original)
    if mutation.startswith("empty_"):
        plan[mutation.removeprefix("empty_")] = []
    elif mutation == "unused_claim":
        plan["claims"][0]["document"] = "docs/missing-v4.html"
    elif mutation == "duplicate_claim":
        plan["claims"].append(plan["claims"][0])
    elif mutation == "missing_field":
        plan["claims"][0]["pointer"] = "/measurement/missing"
    elif mutation == "sensitive_field":
        plan["claims"][0]["pointer"] = "/private/host"
    elif mutation == "object_not_scalar":
        plan["claims"][0]["pointer"] = "/measurement"
    elif mutation == "wrong_text":
        plan["claims"][0]["text"] = "1.26"
    elif mutation == "missing_text":
        plan["claims"][0]["text"] = "1.250"
    elif mutation == "wrong_unit":
        plan["claims"][0]["unit"] = "host"
    elif mutation in ("boolean_number", "status_published"):
        record = {"measurement": {"latency_ms": True if mutation == "boolean_number" else 1.25,
                                  "status": "PUBLISHED" if mutation == "status_published" else "PASS"}}
        dump(evidence / "result.json", record)
        plan["evidence"][0]["sha256"] = gate.sha((evidence / "result.json").read_bytes())
    elif mutation == "version":
        plan["docs_version"] = "v5"
    else:
        plan["evidence"][0]["path"] = "../result.json" if mutation == "path_escape" else "/data/result.json"
    with pytest.raises(gate.Rejected):
        preparation(fixture, plan).prepare(root / "rejected")
    assert not (root / "rejected").exists()


@pytest.mark.parametrize("target", ["evidence_file", "evidence_parent", "document", "output_parent"])
def test_symlink_rejected_without_following(fixture, target):
    root, repo, evidence, original = fixture
    plan = copy.deepcopy(original)
    if target == "evidence_file":
        (evidence / "linked.json").symlink_to(evidence / "result.json")
        plan["evidence"][0]["path"] = "linked.json"
    elif target == "evidence_parent":
        (evidence / "linked").symlink_to(evidence, target_is_directory=True)
        plan["evidence"][0]["path"] = "linked/result.json"
    elif target == "document":
        linked = repo / "docs/linked-v4.html"
        linked.symlink_to(repo / "docs/report-v4.html")
        command(repo, "add", "--", "docs/linked-v4.html")
        command(repo, "-c", "user.name=Publication Fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "-q", "-m", "synthetic symlink")
        plan["source_commit"] = command(repo, "rev-parse", "HEAD")
        plan["documents"][0]["path"] = "docs/linked-v4.html"
    else:
        (root / "linked").symlink_to(root, target_is_directory=True)
    with pytest.raises(gate.Rejected, match="unsafe|symlink"):
        preparation(fixture, plan).prepare(root / "linked/output" if target == "output_parent"
                                          else root / "output")


@pytest.mark.parametrize("target", ["bundle.zip", "manifest.json", "version.json",
                                   "SHA256SUMS", "local-verification.json", "extra"])
def test_public_output_tampering_rejected_even_with_rehashed_archive(fixture, target):
    root, _, _, _ = fixture
    tool = preparation(fixture)
    report = tool.prepare(root / "output")
    path = root / "output" / target
    if target == "bundle.zip":
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("leak.txt", "/data/internal/private\n")
        repin = gate.sha(path.read_bytes())
    else:
        path.write_bytes(b"tampered\n")
        repin = report["bundle_sha256"]
    with pytest.raises(gate.Rejected, match="mismatch|unexpected_output_members"):
        tool.verify(root / "output", repin)


def test_resealed_committed_leak_rejected_not_silently_redacted(fixture):
    root, repo, _, original = fixture
    plan = copy.deepcopy(original)
    target = repo / "docs/report-v4.html"
    target.write_text(target.read_text() + "<p>/data/internal/private</p>\n")
    command(repo, "add", "--", "docs/report-v4.html")
    command(repo, "-c", "user.name=Publication Fixture", "-c", "user.email=fixture@example.invalid",
            "commit", "-q", "-m", "synthetic leak")
    plan["source_commit"] = command(repo, "rev-parse", "HEAD")
    plan["documents"][0]["sha256"] = gate.sha(target.read_bytes())
    with pytest.raises(gate.Rejected, match="absolute_path"):
        preparation(fixture, plan).prepare(root / "rejected")


def test_cli_real_flow_and_failure_logs_no_rejected_values(fixture):
    root, repo, evidence, _ = fixture
    preparation(fixture)
    common = [
        sys.executable, "-B", str(Path(gate.__file__)), "--repo", str(repo),
        "--evidence-root", str(evidence), "--plan", str(evidence / "plan.json"),
        "--plan-sha256", gate.sha((evidence / "plan.json").read_bytes()),
        "--output", str(root / "cli-output"),
        "--log-dir", str(root / "logs"),
    ]
    result = subprocess.run(common[:3] + ["prepare"] + common[3:], capture_output=True, text=True)
    (root / "cli-prepare.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0 and "verdict=PREPARED" in result.stdout
    digest = gate.sha((root / "cli-output/bundle.zip").read_bytes())
    check = subprocess.run(common[:3] + ["verify"] + common[3:] + ["--bundle-sha256", digest],
                           capture_output=True, text=True)
    (root / "cli-verify.log").write_text(check.stdout + check.stderr)
    assert check.returncode == 0 and "byte_readback=PASS" in check.stdout
    (evidence / "result.json").write_text("private synthetic invalid content")
    failed = subprocess.run(common[:3] + ["verify"] + common[3:] + ["--bundle-sha256", digest],
                            capture_output=True, text=True)
    (root / "cli-negative-control.log").write_text(failed.stdout + failed.stderr)
    assert failed.returncode == 1 and "code=changed_evidence" in failed.stderr
    assert "private synthetic" not in failed.stdout + failed.stderr
    logs = [json.loads(p.read_text()) for p in (root / "logs").glob("audit-*/audit.json")]
    assert len(logs) == 3
    assert sum(row["verdict"] == "PREPARED" for row in logs) == 2
    failure = [row for row in logs if row["verdict"] == "FAIL"]
    assert failure[0]["exit_code"] == 1 and failure[0]["first_cause"] == "changed_evidence"


@pytest.mark.parametrize("rule,payload", [
    ("absolute_path", "/data/internal/private"),
    ("host_identity", "h3c2"),
    ("credential", "pass" + "word = synthetic-fixture"),
    ("license_material", "license " + "server 123"),
    ("network_entry", "192.168.1.17"),
])
def test_real_mutant_source_disables_exactly_one_detection_arm(tmp_path, rule, payload):
    import importlib.util

    original = Path(gate.__file__).read_text()
    anchor = 'for code in ("network_entry", *[c for c in LEAK_RULES if c != "network_entry"]):'
    assert original.count(anchor) == 1
    mutant = original.replace(anchor, anchor + f'\n        if code == "{rule}":\n            continue')
    path = tmp_path / "publication_mutant.py"
    path.write_text(mutant)
    spec = importlib.util.spec_from_file_location("publication_mutant", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        controls = {
            "absolute_path": "/data/internal/private",
            "host_identity": "h3c2",
            "credential": "pass" + "word = synthetic-fixture",
            "license_material": "license " + "server 123",
            "network_entry": "192.168.1.17",
        }
        detected = set()
        for name, text in controls.items():
            with pytest.raises(gate.Rejected):
                gate.scan(text.encode(), "original_control")
            try:
                module.scan(text.encode(), "mutant_control")
            except module.Rejected:
                detected.add(name)
        assert detected == set(controls) - {rule}
        module.scan(payload.encode(), "disabled_arm")
    finally:
        del sys.modules[spec.name]


@pytest.mark.parametrize("payload", [b"{}", b"[]", b'{"x":1,"x":2}', b'{"x":NaN}', b" \n"])
def test_bad_plan_and_json_fail_loud(fixture, payload):
    _, repo, evidence, _ = fixture
    target = evidence / "plan.json"
    target.write_bytes(payload)
    with pytest.raises(gate.Rejected):
        gate.Preparation(repo, evidence, target, gate.sha(payload))


def test_submission_files_use_existing_secret_engine_when_available(tmp_path):
    import importlib.util

    engine_path = Path("/data/dev/memory/tools/redact_secrets.py")
    if not engine_path.is_file():
        pytest.skip("optional local submission audit engine is unavailable")
    spec = importlib.util.spec_from_file_location("submission_secret_engine", engine_path)
    engine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(engine)
    positive = "-----" + "BEGIN RSA PRIVATE KEY" + "-----"
    assert engine.sensitive_value(positive) is not None
    assert engine.sensitive_value("ordinary research prose") is None
    root = Path(gate.__file__).parents[1]
    paths = ["scripts/prepare_publication.py", "tests/test_prepare_publication.py",
             "docs/research/publication.md"]
    findings = []
    for path in paths:
        lines = (root / path).read_text().splitlines()
        assert lines, "submission audit must not be empty"
        for number, line in enumerate(lines, 1):
            reason = engine.sensitive_value(line)
            if reason:
                findings.append({"path": path, "line": number, "reason": reason})
    dump(tmp_path / "submission-secret-audit.json", {
        "files": paths, "findings": findings, "positive_control": True,
        "negative_control": True, "verdict": "PASS" if not findings else "FAIL",
    })
    assert not findings, "submission secret audit findings; see external coordinate-only report"
