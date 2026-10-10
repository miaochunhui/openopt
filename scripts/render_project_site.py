"""Generate the Openopt-owned research landing page; never publish it.

Collect caller-pinned owner CPU receipts into an external scalar input, then
render/check exactly two project-owned files. Reported checks are not numerical
requalification, GPU execution, licensed comparisons or online readback.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from xml.etree import ElementTree

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import prepare_publication as safe

ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ("docs/site/index.html", "docs.publish.json")
OWNED = ("scripts/render_project_site.py", "tests/test_render_project_site.py",
         "docs/research/project-site.md", *OUTPUTS)
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
STAMP = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+08:00\Z")
CLAIMS = {
    "commercial": "PENDING", "general_optimizer": "PENDING",
    "current_source_cuda": "UNVERIFIED", "fair_speed_comparison": "UNVERIFIED",
}
EVIDENCE_LABELS = {
    "cpu-smoke": "Historical CPU smoke",
    "psd-owner-cpu": "Separate PSD primitive owner CPU checks",
    "operator-cuda-failure": "Failed operator CUDA owner regression",
}
CUDA_FAILURE_SOURCE_FILES = {
    *(f"src/openopt/{name}.py" for name in (
        "__init__", "batch", "branch_bound", "certificates", "conic",
        "conic_certificates", "corpus", "hsd_lp", "integer", "native_conic",
        "operators", "presolve", "qp", "refinement", "scaling")),
    "tests/test_operator_original_cuda.py", "scripts/gpu_release_check.py",
}
BOUNDARY_PATHS = ("AGENTS.md", "LICENSE")
DEFAULT_ARTIFACTS = (
    "/hf3fs-jd/data/tmp/openopt-release-20261010/project-site"
)


def require(condition, code, coordinate="input"):
    safe.require(condition, code, coordinate)


def digest(value, coordinate="input", commit=False):
    pattern = COMMIT if commit else safe.HASH
    require(isinstance(value, str) and pattern.fullmatch(value) is not None,
            "invalid_digest", coordinate)
    return value


def timestamp(value, coordinate="modified_at"):
    require(isinstance(value, str) and STAMP.fullmatch(value) is not None,
            "timestamp_requires_seconds_and_offset", coordinate)
    try:
        result = datetime.fromisoformat(value)
    except ValueError as error:
        raise safe.Rejected("invalid_timestamp", coordinate) from error
    require(result.utcoffset() == timedelta(hours=8), "timestamp_offset", coordinate)
    return result


def count(value, coordinate):
    require(type(value) is int and 0 <= value <= 1_000_000,
            "invalid_count", coordinate)
    return value


def scan_content(payload, record, coordinate):
    # The legacy network detector treats hh:mm:ss as IPv6. Exempt only
    # complete, already-validated timestamp tokens, never arbitrary payload.
    times = [record["modified_at"]]
    times.extend(item["finished_at"] for item in record["evidence"].values())
    for value in times:
        moment = timestamp(value)
        for spelling in (value, moment.strftime("%Y-%m-%d %H:%M:%S +08:00")):
            payload = payload.replace(spelling.encode(), b"VALIDATED_TIMESTAMP")
    safe.scan(payload, coordinate)


def validate_input(record):
    safe.keys(record, {
        "schema_version", "project", "content_source_commit", "modified_at",
        "license", "boundary_sha256", "access", "publication", "qualification",
        "evidence",
    }, "input")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1,
            "unsupported_schema")
    require(record["project"] == "openopt", "wrong_project")
    digest(record["content_source_commit"], commit=True)
    timestamp(record["modified_at"])
    require(record["license"] == "0BSD", "license_scope")
    require(record["access"] == "existing-super-only", "access_scope")
    require(record["publication"] == "NOT_PUBLISHED", "publication_promotion")
    safe.keys(record["qualification"], CLAIMS, "qualification")
    require(record["qualification"] == CLAIMS, "qualification_promotion")
    safe.keys(record["boundary_sha256"], BOUNDARY_PATHS, "boundary")
    for value in record["boundary_sha256"].values():
        digest(value, "boundary")
    evidence = record["evidence"]
    require(isinstance(evidence, dict) and 1 <= len(evidence) <= 3
            and set(evidence) <= set(EVIDENCE_LABELS), "evidence_scope")
    for name, item in evidence.items():
        fields = {"source_commit", "receipt_sha256", "finished_at",
                  "tests", "passed", "failures", "errors", "skipped",
                  "relation", "kind"}
        failed_cuda = name == "operator-cuda-failure"
        if failed_cuda:
            fields |= {"junit_sha256", "pytest_log_sha256"}
        safe.keys(item, fields, name)
        digest(item["source_commit"], name, commit=True)
        digest(item["receipt_sha256"], name)
        timestamp(item["finished_at"], name)
        for field in ("tests", "passed", "failures", "errors", "skipped"):
            count(item[field], name)
        require(item["tests"] > 0 and item["passed"] > 0
                and item["tests"] == sum(item[field] for field in
                                        ("passed", "failures", "errors", "skipped")),
                "incomplete_count_denominator", name)
        if failed_cuda:
            digest(item["junit_sha256"], name)
            digest(item["pytest_log_sha256"], name)
            require(item["failures"] > 0, "cuda_failure_not_red", name)
        require(item["kind"] == ("reported-failed-cuda-owner-checks" if failed_cuda
                                 else "reported-owner-cpu-checks"),
                "evidence_qualification_promotion", name)
        expected = ("same-content-source" if item["source_commit"] ==
                    record["content_source_commit"] else
                    ("separate-freeze" if name == "psd-owner-cpu" else "historical"))
        require(item["relation"] == expected, "evidence_source_relation", name)
        require(timestamp(item["finished_at"], name) <=
                timestamp(record["modified_at"]), "evidence_after_modified", name)
    # No caller-controlled prose, vectors, paths or unbounded strings survive.
    scan_content(safe.encoded(record), record, "scalar_input")
    return record


def git(repo, *arguments, input_bytes=None):
    result = subprocess.run(["git", "-C", str(repo), *arguments], input=input_bytes,
                            capture_output=True, check=False)
    require(result.returncode == 0, "git_query_failed", "source")
    return result.stdout


def tree_hashes(repo, commit):
    digest(commit, "source", commit=True)
    listing = git(repo, "ls-tree", "-rz", commit)
    entries = []
    for entry in listing.split(b"\0"):
        if not entry:
            continue
        metadata, raw_name = entry.split(b"\t", 1)
        mode, kind, oid = metadata.split()
        require(mode in (b"100644", b"100755") and kind == b"blob",
                "non_regular_source", "source")
        entries.append((raw_name.decode("utf-8"), oid))
    require(bool(entries), "empty_source", "source")
    payload = git(repo, "cat-file", "--batch",
                  input_bytes=b"".join(oid + b"\n" for _, oid in entries))
    offset, result = 0, {}
    for name, oid in entries:
        end = payload.find(b"\n", offset)
        header = payload[offset:end].split()
        require(len(header) == 3 and header[:2] == [oid, b"blob"],
                "source_batch_header", "source")
        size = int(header[2])
        data = payload[end + 1:end + 1 + size]
        require(len(data) == size and payload[end + 1 + size:end + 2 + size] == b"\n",
                "source_batch_size", "source")
        result[name] = safe.sha(data)
        offset = end + 2 + size
    require(offset == len(payload), "source_batch_extra", "source")
    return result


def check_boundary(record, repo):
    hashes = tree_hashes(repo, record["content_source_commit"])
    require(record["boundary_sha256"] == {name: hashes.get(name)
                                         for name in BOUNDARY_PATHS},
            "boundary_source_mismatch", "source")
    license_bytes = git(repo, "cat-file", "blob",
                        f"{record['content_source_commit']}:LICENSE")
    require(b"Zero Clause BSD" in license_bytes or b"BSD Zero Clause" in license_bytes,
            "license_not_zero_clause", "source")


def normalized_time(value):
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError) as error:
        raise safe.Rejected("invalid_receipt_time", "receipt") from error
    require(moment.tzinfo is not None, "receipt_time_without_offset", "receipt")
    return moment.astimezone(timezone(timedelta(hours=8))).replace(
        microsecond=0).isoformat(timespec="seconds")


def extract_receipt(repo, name, path, expected_hash, expected_commit):
    require(name in {"cpu-smoke", "psd-owner-cpu"}, "evidence_scope", "receipt")
    digest(expected_hash, "receipt")
    digest(expected_commit, "receipt", commit=True)
    payload = safe.read(safe.absolute(path, "receipt"), "receipt").payload
    require(safe.sha(payload) == expected_hash, "receipt_pin_mismatch", "receipt")
    raw = safe.parse(payload, "receipt")
    require(isinstance(raw, dict) and type(raw.get("schema_version")) is int
            and raw["schema_version"] == 1 and raw.get("verdict") == "PASS",
            "receipt_not_owner_pass", "receipt")
    source = raw.get("source")
    require(isinstance(source, dict) and source.get("clean") is True
            and source.get("commit") == expected_commit,
            "receipt_source_pin", "receipt")
    require(source.get("sha256") == tree_hashes(repo, expected_commit)
            and source.get("tree") == git(repo, "rev-parse",
                                          f"{expected_commit}^{{tree}}").decode().strip(),
            "receipt_source_bytes", "receipt")
    consistency = raw.get("source_consistency", {})
    require(isinstance(consistency, dict), "receipt_source_consistency", "receipt")
    require(consistency.get("verdict") == "PASS" and
            consistency.get("same_commit") is True and
            all(consistency.get(key) == [] for key in ("changed", "missing", "extra"))
            and type(consistency.get("files")) is int
            and consistency["files"] == len(source["sha256"]),
            "receipt_source_consistency", "receipt")
    if name == "cpu-smoke":
        require(raw.get("mode") == "smoke" and raw.get("cuda_available") is False,
                "receipt_not_cpu_smoke", "receipt")
        counts = raw.get("tests")
    else:
        runtime = raw.get("runtime", {})
        require(isinstance(runtime, dict) and
                isinstance(runtime.get("environment"), dict),
                "receipt_not_psd_cpu", "receipt")
        require(runtime["environment"].get("CUDA_VISIBLE_DEVICES") == "-1"
                and runtime.get("gpu_qualification") == "UNVERIFIED"
                and raw.get("pytest_exit_code") == 0
                and type(raw.get("pytest_exit_code")) is int,
                "receipt_not_psd_cpu", "receipt")
        counts = raw.get("counts")
        members = raw.get("membership")
        require(raw.get("exact_membership") is True and isinstance(members, list)
                and all(isinstance(value, str) for value in members)
                and len(set(members)) == len(members)
                and members == raw.get("collection_membership"),
                "receipt_membership", "receipt")
        require(isinstance(counts, dict) and len(members) == counts.get("tests"),
                "receipt_denominator", "receipt")
    safe.keys(counts, {"tests", "failures", "errors", "skipped"}, "receipt_counts")
    for key, value in counts.items():
        count(value, key)
    require(counts["tests"] > 0 and counts["failures"] == counts["errors"] == 0
            and counts["tests"] > counts["skipped"], "receipt_count_failure", "receipt")
    return {
        "source_commit": expected_commit, "receipt_sha256": expected_hash,
        "finished_at": normalized_time(raw.get("finished_at")),
        **counts, "passed": counts["tests"] - counts["skipped"],
        "kind": "reported-owner-cpu-checks",
    }


def extract_cuda_failure(repo, arguments):
    """Read an explicitly failed result, never treat it as execution acceptance."""
    path, result_hash, junit_path, junit_hash, log_path, log_hash, commit = arguments
    digest(commit, "cuda_failure", commit=True)
    snapshots = []
    for file_path, expected in ((path, result_hash), (junit_path, junit_hash),
                                (log_path, log_hash)):
        digest(expected, "cuda_failure")
        data = safe.read(safe.absolute(file_path, "cuda_failure"), "cuda_failure").payload
        require(safe.sha(data) == expected, "cuda_failure_pin_mismatch", "cuda_failure")
        snapshots.append(data)
    raw = safe.parse(snapshots[0], "cuda_failure")
    require(isinstance(raw, dict) and raw.get("verdict") == "FAIL"
            and type(raw.get("returncode")) is int and raw["returncode"] != 0,
            "cuda_failure_not_red", "cuda_failure")
    hashes = tree_hashes(repo, commit)
    require(raw.get("source_sha256") == {name: hashes.get(name)
                                        for name in CUDA_FAILURE_SOURCE_FILES},
            "cuda_failure_source_bytes", "cuda_failure")
    require(raw.get("targets") ==
            ["tests/test_operator_original_cuda.py::test_cuda_collection_entry"]
            and raw.get("selection") == "test_cuda_collection_entry",
            "cuda_failure_target", "cuda_failure")
    try:
        require(b"<!DOCTYPE" not in snapshots[1] and b"<!ENTITY" not in snapshots[1],
                "unsafe_junit", "cuda_failure")
        xml = ElementTree.fromstring(snapshots[1])
        suites = list(xml.iter("testsuite"))
        counts = {key: sum(int(suite.attrib[key]) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        cases = list(xml.iter("testcase"))
        names = [case.attrib["name"] for case in cases]
        log = snapshots[2].decode("utf-8")
    except (ElementTree.ParseError, KeyError, ValueError, UnicodeError) as error:
        raise safe.Rejected("invalid_failed_cuda_junit_or_log", "cuda_failure") from error
    for key, value in counts.items():
        count(value, key)
    actual = {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }
    require(bool(suites) and actual == counts and counts["failures"] > 0
            and len(names) == len(set(names)) and all(
                name.startswith("test_cuda_collection_entry[") for name in names),
            "cuda_failure_membership", "cuda_failure")
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    require(passed > 0 and f"{counts['failures']} failed, {passed} passed" in log
            and "\nreturncode=1" in log, "cuda_failure_log_summary", "cuda_failure")
    require(str(counts) in raw.get("error", ""), "cuda_failure_result_counts", "cuda_failure")
    return {
        "source_commit": commit, "receipt_sha256": result_hash,
        "junit_sha256": junit_hash, "pytest_log_sha256": log_hash,
        "finished_at": normalized_time(raw.get("finished_at")), **counts,
        "passed": passed, "kind": "reported-failed-cuda-owner-checks",
    }


def collect(repo, commit, modified_at, receipt_arguments, cuda_failure=None):
    hashes = tree_hashes(repo, commit)
    record = {
        "schema_version": 1, "project": "openopt", "content_source_commit": commit,
        "modified_at": modified_at, "license": "0BSD",
        "boundary_sha256": {name: hashes.get(name) for name in BOUNDARY_PATHS},
        "access": "existing-super-only", "publication": "NOT_PUBLISHED",
        "qualification": dict(CLAIMS), "evidence": {},
    }
    require(bool(receipt_arguments) or cuda_failure is not None, "empty_evidence", "receipt")
    for name, path, receipt_hash, source_commit in receipt_arguments:
        require(name not in record["evidence"], "duplicate_evidence", "receipt")
        item = extract_receipt(repo, name, path, receipt_hash, source_commit)
        item["relation"] = ("same-content-source" if source_commit == commit else
                            ("historical" if name == "cpu-smoke" else "separate-freeze"))
        record["evidence"][name] = item
    if cuda_failure is not None:
        item = extract_cuda_failure(repo, cuda_failure)
        item["relation"] = ("same-content-source" if item["source_commit"] == commit
                            else "historical")
        record["evidence"]["operator-cuda-failure"] = item
    validate_input(record)
    check_boundary(record, repo)
    return safe.encoded(record)


def render(payload):
    record = validate_input(safe.parse(payload, "input"))
    input_hash = safe.sha(payload)
    rows = []
    for name, item in sorted(record["evidence"].items()):
        failed = item["failures"] or item["errors"]
        scope = ("FAILED; no CUDA acceptance." if failed else
                 "Reported owner CPU checks only.")
        extra_hashes = (
            f'<dt>JUnit SHA-256</dt><dd><code>{item["junit_sha256"]}</code></dd>'
            f'<dt>Pytest log SHA-256</dt><dd><code>{item["pytest_log_sha256"]}</code></dd>'
            if name == "operator-cuda-failure" else ""
        )
        rows.append(
            f'<article class="evidence{" failed" if failed else ""}">'
            f'<p class="eyebrow">{EVIDENCE_LABELS[name]}</p>'
            f'<h3>{item["passed"]}<span> passed / {item["tests"]} selected</span></h3>'
            f'<p>{item["skipped"]} skipped; {item["failures"]} failed; '
            f'{item["errors"]} errors. {scope}</p>'
            f'<dl><dt>Source relation</dt><dd>{item["relation"]}</dd>'
            f'<dt>Checked at</dt><dd>{item["finished_at"]}</dd>'
            f'<dt>Source commit</dt><dd><code>{item["source_commit"]}</code></dd>'
            f'<dt>Receipt SHA-256</dt><dd><code>{item["receipt_sha256"]}</code></dd>'
            f'{extra_hashes}</dl></article>'
        )
    claims = "".join(
        f'<li><span>{label}</span><strong>{record["qualification"][key]}</strong></li>'
        for key, label in (
            ("commercial", "Commercial qualification"),
            ("general_optimizer", "Complete general optimizer"),
            ("current_source_cuda", "Current-source CUDA qualification"),
            ("fair_speed_comparison", "Fair comparative speed"),
        )
    )
    stamp = timestamp(record["modified_at"]).strftime("%Y-%m-%d %H:%M:%S +08:00")
    page = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Openopt: a 0BSD optimization research project. Scoped evidence, explicit limits, no commercial ranking.">
<meta name="generator" content="scripts/render_project_site.py schema 1">
<title>Openopt | Optimization, with evidence</title>
<style>
:root{--paper:#f4f1e7;--ink:#15382d;--muted:#4f6155;--line:#c7cec0;--accent:#a94724;--card:#fffef7}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;color:var(--ink);background:radial-gradient(ellipse at 85% 8%,#e2e8cb,transparent 48%),var(--paper);font-family:Georgia,"Times New Roman",serif}
a{color:inherit;text-decoration-thickness:1px;text-underline-offset:4px}main,header,footer{max-width:1120px;margin:auto;padding:24px 36px}header{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid var(--line)}header b{font-size:24px;letter-spacing:-1px}nav{display:flex;gap:22px;font-size:15px}
.hero{padding:64px 0 54px;max-width:880px}.eyebrow{font:12px/1.5 "Courier New",monospace;text-transform:uppercase;letter-spacing:2px;color:var(--accent)}h1{font-size:clamp(48px,7vw,88px);font-weight:normal;letter-spacing:-3px;line-height:1.03;margin:18px 0 26px}h1 em{color:var(--accent);font-weight:normal}.lede{font-size:22px;line-height:1.55;max-width:700px}.tag{display:inline-block;border:1px solid var(--ink);padding:8px 12px;font:12px "Courier New",monospace;margin:4px 8px 4px 0}
section{border-top:1px solid var(--line);padding:32px 0}h2{font-size:34px;letter-spacing:-1px;font-weight:normal;margin:0 0 20px}p{line-height:1.65}section>p{max-width:780px}.split{display:grid;grid-template-columns:1fr 1fr;gap:36px}.status{list-style:none;padding:0;margin:0}.status li{display:flex;gap:12px;justify-content:space-between;padding:15px 0;border-bottom:1px solid var(--line);line-height:1.5}.status strong{font:12px/2 "Courier New",monospace;color:var(--accent)}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.evidence{background:var(--card);border:1px solid var(--line);padding:24px}.evidence.failed{border:2px solid var(--accent);background:#fff2e9}.evidence h3{font-size:45px;font-weight:normal;margin:16px 0}.evidence h3 span{font-size:17px}.evidence p{font-size:15px}dl{font-size:13px}dt{color:var(--muted);margin-top:12px}dd{margin:4px 0}code{font:12px/1.6 "Courier New",monospace;overflow-wrap:anywhere}.note{border-left:3px solid var(--accent);padding-left:18px}.provenance{font-size:14px;overflow-wrap:anywhere}footer{border-top:1px solid var(--line);font:12px/1.7 "Courier New",monospace;padding-bottom:36px}.hero{animation:reveal .6s ease-out}@keyframes reveal{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}@media(prefers-reduced-motion:reduce){.hero{animation:none}html{scroll-behavior:auto}}@media(max-width:700px){main,header,footer{padding-left:20px;padding-right:20px}header{align-items:flex-start;gap:18px}nav{flex-wrap:wrap;gap:10px 18px;justify-content:flex-end}.hero{padding:38px 0}h1{letter-spacing:-2px}.lede{font-size:19px}.split,.grid{grid-template-columns:1fr}.status li{font-size:14px}.evidence{padding:20px}}
</style>
</head>
<body>
<header><b>openopt.</b><nav aria-label="Project sections"><a href="#scope">Scope</a><a href="#evidence">Evidence</a><a href="#access">Access</a></nav></header>
<main>
<div class="hero"><p class="eyebrow">Optimization research / 0BSD</p>
<h1>Build the optimizer.<br><em>Keep the evidence.</em></h1>
<p class="lede">A GPU-first optimization project with bounded, opt-in research components.
The commercial-grade general optimizer is the destination, not an achieved qualification.</p>
<span class="tag">RESEARCH ONLY</span><span class="tag">0BSD PROJECT CODE</span>
<span class="tag">__PUBLICATION__</span></div>
<section id="scope" class="split"><div><h2>Useful foundations.<br>Explicit limits.</h2>
<p>Experimental convex quadratic and linear programming (QP/LP), native cone prototypes,
original-model checks and finite-precision refinement form the research foundation.
Defaults and supported dimensions are component-specific.</p>
<p>Project code uses BSD Zero Clause (0BSD). Dependency licenses are separate.
The project license is not authorization for commercial benchmark execution or disclosure.</p>
<p class="note">Not complete semidefinite, exponential or power-cone optimization;
not general native mixed-integer, nonlinear or global optimization. A cone primitive is not
a full solver. No arbitrary-precision optimizer or universal GPU speedup is claimed.</p></div>
<div><h2>Qualification ledger</h2><ul class="status">__CLAIMS__</ul>
<p>Historical and separate-source checks below do not qualify this content baseline.
No comparative timing, commercial ranking, or current-source GPU claim is inferred.</p></div></section>
<section id="evidence"><p class="eyebrow">Pinned scalar readback</p><h2>What was actually selected</h2>
<p>These are counts extracted from caller-pinned owner receipts and failed CUDA result/JUnit/log bytes.
Skipped tests are not passes. Extraction checks receipt/source consistency; it does not
rerun the solver, independently accept the numerical results, or attest remote execution.</p>
<p class="note">Failed CUDA checks remain red. No fixed13 evidence is included or
treated as effective acceptance. Failure readback is not a GPU execution seal.</p>
<div class="grid">__EVIDENCE__</div></section>
<section id="access"><h2>Access has not changed.</h2>
<p class="note">Existing super-only access is the intended onboarding default. This generated
project catalog does not grant public access, alter authentication, or establish a live URL.
Shared registration, publication and exact online byte readback remain the chair's responsibility.</p>
<p>No raw model arrays, proprietary SDK details, private evidence locations or credentials
are included here. Only selected scalars, public code hashes and research boundaries leave
the evidence input.</p></section>
<section class="provenance"><h2>Reproducible content</h2>
<p>Content baseline commit: <code>__SOURCE__</code><br>
Scalar input SHA-256: <code>__INPUT_HASH__</code><br>
0BSD license bytes SHA-256: <code>__LICENSE_HASH__</code><br>
Research boundary bytes SHA-256: <code>__BOUNDARY_HASH__</code></p></section>
</main><footer><span lang="zh">最后修改：</span>__STAMP__<br>
Generated deterministically from a pinned scalar input. Local content, not online acceptance.</footer>
</body></html>
"""
    substitutions = {
        "__PUBLICATION__": record["publication"], "__CLAIMS__": claims,
        "__EVIDENCE__": "".join(rows), "__SOURCE__": record["content_source_commit"],
        "__INPUT_HASH__": input_hash,
        "__LICENSE_HASH__": record["boundary_sha256"]["LICENSE"],
        "__BOUNDARY_HASH__": record["boundary_sha256"]["AGENTS.md"],
        "__STAMP__": stamp,
    }
    for key, value in substitutions.items():
        page = page.replace(key, value)
    catalog = {
        "version": 1, "project": "openopt",
        "audiences": {"admin": {"path_prefix": ""}},
        "pages": [{
            "slug": "overview", "title": "Openopt research",
            "source": OUTPUTS[0], "section": "Research",
            "order": 1, "audience": "admin", "status": "research",
            "owner": "openopt", "aliases": [],
        }],
    }
    result = {OUTPUTS[0]: page.encode("utf-8"), OUTPUTS[1]: safe.encoded(catalog)}
    for name, content in result.items():
        scan_content(content, record, name)
    return result


def external(path, repo):
    target = safe.absolute(path, "external")
    require(target.is_relative_to(safe.SHARED) and
            not target.is_relative_to(repo), "external_storage_required", "filesystem")
    return target


def create_once(path, payload):
    fd = safe.directory(path.parent, create=True)
    try:
        with os.fdopen(os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL |
                              os.O_NOFOLLOW, 0o644, dir_fd=fd), "wb") as stream:
            stream.write(payload)
    except OSError as error:
        raise safe.Rejected("existing_or_unsafe_output", "filesystem") from error
    finally:
        os.close(fd)


def write_outputs(repo, files):
    # Preflight both files before creating either; never overwrite even one.
    for name in OUTPUTS:
        parent = safe.directory((repo / name).parent, create=True)
        try:
            try:
                os.stat((repo / name).name, dir_fd=parent, follow_symlinks=False)
            except FileNotFoundError:
                continue
            raise safe.Rejected("output_already_exists", name)
        finally:
            os.close(parent)
    for name in OUTPUTS:
        create_once(repo / name, files[name])


def check_outputs(repo, files):
    for name in OUTPUTS:
        require(safe.read(repo / name, name).payload == files[name],
                "generated_output_mismatch", name)


def source_state(repo):
    names = git(repo, "ls-files", "--cached", "--others", "--exclude-standard",
                "-z").decode().split("\0")
    hashes = {name: safe.sha(safe.read(repo / name, "source").payload)
              for name in sorted(set(names) - {""})}
    require(bool(hashes), "empty_source", "source")
    return {
        "commit": git(repo, "rev-parse", "HEAD").decode().strip(),
        "tree": git(repo, "rev-parse", "HEAD^{tree}").decode().strip(),
        "clean": not bool(git(repo, "status", "--porcelain=v1", "-z")),
        "sha256": hashes,
    }


def execute(args, artifacts):
    repo = safe.absolute(args.repository, "repository")
    before = source_state(repo)
    report = {
        "schema_version": 1, "operation": args.operation, "source_before": before,
        "qualification": dict(CLAIMS), "publication": "NOT_PUBLISHED",
        "access": "existing-super-only", "verdict": "FAIL", "exit_code": 1,
    }
    try:
        if args.operation == "collect":
            require(args.content_source_commit is not None and args.modified_at is not None,
                    "collect_arguments", "cli")
            for _, receipt_path, _, _ in args.evidence:
                external(receipt_path, repo)
            if getattr(args, "cuda_failure", None):
                for position in (0, 2, 4):
                    external(args.cuda_failure[position], repo)
            payload = collect(repo, args.content_source_commit, args.modified_at,
                              args.evidence, getattr(args, "cuda_failure", None))
            target = external(args.input, repo)
            create_once(target, payload)
            report["input_sha256"] = safe.sha(payload)
        else:
            require(args.input_sha256 is not None, "input_pin_required", "cli")
            digest(args.input_sha256, "cli")
            target = external(args.input, repo)
            payload = safe.read(target, "input").payload
            require(safe.sha(payload) == args.input_sha256, "input_pin_mismatch")
            record = validate_input(safe.parse(payload, "input"))
            check_boundary(record, repo)
            files = render(payload)
            if args.operation == "render":
                write_outputs(repo, files)
            else:
                require(args.owner_commit is not None, "owner_pin_required", "cli")
                digest(args.owner_commit, "owner", commit=True)
                require(before["commit"] == args.owner_commit and before["clean"],
                        "owner_not_pinned_clean", "source")
                expected = tree_hashes(repo, args.owner_commit)
                require(before["sha256"] == expected, "owner_bytes_not_committed", "source")
            check_outputs(repo, files)
            report["input_sha256"] = safe.sha(payload)
            report["outputs_sha256"] = {name: safe.sha(value) for name, value in files.items()}
        after = source_state(repo)
        report["source_after"] = after
        changed = sorted(name for name in before["sha256"].keys() & after["sha256"].keys()
                         if before["sha256"][name] != after["sha256"][name])
        extra = sorted(after["sha256"].keys() - before["sha256"].keys())
        missing = sorted(before["sha256"].keys() - after["sha256"].keys())
        require(before["commit"] == after["commit"] and not (changed or missing)
                and (extra == sorted(OUTPUTS) if args.operation == "render" else not extra),
                "source_changed_outside_generated_outputs", "source")
        report["source_consistency"] = {
            "verdict": "PASS", "changed": changed, "missing": missing, "extra": extra,
        }
        report["owned_sha256"] = {name: after["sha256"].get(name) for name in OWNED}
        report["verdict"], report["exit_code"] = "PASS", 0
    except (safe.Rejected, OSError, ValueError) as error:
        report["first_cause"] = (
            {"code": error.code, "coordinate": error.coordinate}
            if isinstance(error, safe.Rejected) else {"code": type(error).__name__}
        )
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    create_once(artifacts / "receipt.json", safe.encoded(report))
    line = f"PROJECT_SITE verdict={report['verdict']} operation={args.operation}"
    create_once(artifacts / "execution.log", (line + "\n").encode())
    print(f"{line} receipt={artifacts / 'receipt.json'}", flush=True)
    if report.get("first_cause"):
        print("PROJECT_SITE first_cause=" + json.dumps(report["first_cause"],
                                                      sort_keys=True), flush=True)
    if "input_sha256" in report:
        print(f"PROJECT_SITE input_sha256={report['input_sha256']}", flush=True)
    return report["exit_code"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("operation", choices=("collect", "render", "check"))
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256")
    parser.add_argument("--content-source-commit")
    parser.add_argument("--owner-commit")
    parser.add_argument("--modified-at")
    parser.add_argument("--evidence", action="append", nargs=4, default=[],
                        metavar=("ID", "PATH", "SHA256", "SOURCE_COMMIT"))
    parser.add_argument("--cuda-failure", nargs=7,
                        metavar=("RESULT", "RESULT_SHA", "JUNIT", "JUNIT_SHA",
                                 "LOG", "LOG_SHA", "SOURCE_COMMIT"))
    parser.add_argument("--artifact-dir", type=Path, default=Path(DEFAULT_ARTIFACTS))
    args = parser.parse_args(argv)
    root = external(args.artifact_dir, safe.absolute(args.repository, "repository"))
    fd = safe.directory(root, create=True)
    os.close(fd)
    artifacts = Path(tempfile.mkdtemp(prefix=f"{args.operation}-", dir=root))
    return execute(args, artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
