"""Generate the Openopt-owned research landing page; never publish it.

Collect caller-pinned owner CPU receipts into an external scalar input, then
render/check exactly two project-owned files. Reported checks are not numerical
requalification, GPU execution, licensed comparisons or online readback.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timedelta, timezone
import hashlib
import json
import marshal
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
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
RUNTIME_PATHS = ("scripts/render_project_site.py", "scripts/prepare_publication.py")
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
            boundary = rb"[A-Za-z0-9_:./%+@-]"
            pattern = rb"(?<!" + boundary + rb")" + re.escape(spelling.encode()) + rb"(?!" + boundary + rb")"
            payload = re.sub(pattern, b"VALIDATED_TIMESTAMP", payload)
    safe.scan(payload, coordinate)


def validate_input(record):
    safe.keys(record, {
        "schema_version", "project", "content_source_commit", "modified_at",
        "license", "boundary_sha256", "access", "publication", "qualification",
        "evidence",
    }, "input")
    require(type(record["schema_version"]) is int and record["schema_version"] in (1, 2),
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
        elif record["schema_version"] == 2:
            fields |= {"junit_sha256", "pytest_log_sha256", "collection_sha256", "authority_sha256"}
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
            require(item["errors"] == item["skipped"] == 0, "cuda_failure_incomplete", name)
        if record["schema_version"] == 2:
            for field in fields & {"junit_sha256", "pytest_log_sha256", "collection_sha256", "authority_sha256"}:
                digest(item[field], name)
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
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1", GIT_OPTIONAL_LOCKS="0",
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null")
    result = subprocess.run(["git", "--no-replace-objects", "-c", "core.fsmonitor=false", "-C", str(repo), *arguments],
                            input=input_bytes, env=env, capture_output=True, check=False)
    require(result.returncode == 0, "git_query_failed", "source")
    return result.stdout


def git_object(repo, kind, oid):
    digest(oid, "git_object", commit=True)
    data = git(repo, "cat-file", kind, oid)
    prefix = (kind + " " + str(len(data))).encode() + b"\0"
    require(hashlib.sha1(prefix + data).hexdigest() == oid, "raw_git_object_digest", "source")
    return data


def tree_hashes(repo, commit):
    digest(commit, "source", commit=True)
    body = git_object(repo, "commit", commit)
    tree = body.split(b"\n", 1)[0]
    require(tree.startswith(b"tree "), "raw_commit_tree", "source")
    pending, result = [("", tree[5:].decode("ascii"))], {}
    while pending:
        parent, oid = pending.pop()
        data = git_object(repo, "tree", oid)
        offset = 0
        while offset < len(data):
            space, end = data.find(b" ", offset), data.find(b"\0", offset)
            require(offset < space < end and end + 21 <= len(data), "raw_tree_entry", "source")
            mode, leaf = data[offset:space], data[space + 1:end].decode("utf-8")
            name = parent + leaf
            safe.relative(name, "source_name")
            require("/" not in leaf, "raw_tree_leaf", "source")
            child = data[end + 1:end + 21].hex()
            offset = end + 21
            if mode == b"40000":
                pending.append((name + "/", child))
            else:
                require(mode in (b"100644", b"100755"), "non_regular_source", "source")
                require(name not in result, "duplicate_source_path", "source")
                result[name] = safe.sha(git_object(repo, "blob", child))
    require(bool(result), "empty_source", "source")
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


def load_authority(repo, authority):
    require(isinstance(authority, (tuple, list)) and len(authority) == 2,
            "caller_authority_required", "receipt")
    path, pin = authority
    digest(pin, "authority")
    payload = safe.read(external(path, repo), "authority").payload
    require(safe.sha(payload) == pin, "authority_pin_mismatch", "receipt")
    record = safe.parse(payload, "authority")
    safe.keys(record, {"schema_version", "evidence"}, "authority")
    require(type(record["schema_version"]) is int and record["schema_version"] == 1,
            "authority_schema", "receipt")
    require(isinstance(record["evidence"], dict) and bool(record["evidence"])
            and set(record["evidence"]) <= set(EVIDENCE_LABELS), "authority_scope", "receipt")
    return record["evidence"], pin


def authority_files(repo, entry, roles):
    safe.keys(entry["files"], roles, "authority_files")
    payloads = {}
    paths = set()
    for role, descriptor in entry["files"].items():
        safe.keys(descriptor, {"path", "sha256", "bytes"}, "authority_file")
        digest(descriptor["sha256"], "authority_file")
        count(descriptor["bytes"], "authority_bytes")
        target = external(descriptor["path"], repo)
        require(target not in paths, "duplicate_authority_path", "receipt")
        paths.add(target)
        data = safe.read(target, "authority_file").payload
        require(len(data) == descriptor["bytes"] and safe.sha(data) == descriptor["sha256"],
                "authority_file_pin_mismatch", "receipt")
        payloads[role] = data
    return payloads


def junit_record(payload, coordinate):
    require(b"<!DOCTYPE" not in payload and b"<!ENTITY" not in payload, "unsafe_junit", coordinate)
    try:
        xml = ElementTree.fromstring(payload)
        require(xml.tag in {"testsuites", "testsuite"}, "junit_root", coordinate)
        suites = [suite for suite in xml.iter("testsuite") if not suite.findall("testsuite")]
        counts = {key: sum(int(suite.attrib[key]) for suite in suites)
                  for key in ("tests", "failures", "errors", "skipped")}
        cases = list(xml.iter("testcase"))
        members = [case.attrib["classname"] + "::" + case.attrib["name"] for case in cases]
    except (ElementTree.ParseError, KeyError, ValueError, UnicodeError) as error:
        raise safe.Rejected("invalid_junit", coordinate) from error
    for key, value in counts.items():
        count(value, key)
    actual = {"tests": len(cases), "failures": 0, "errors": 0, "skipped": 0}
    for case in cases:
        outcomes = [tag for tag in ("failure", "error", "skipped") if case.find(tag) is not None]
        require(len(outcomes) <= 1 and all(len(case.findall(tag)) <= 1 for tag in outcomes),
                "ambiguous_junit_outcome", coordinate)
        for tag, key in (("failure", "failures"), ("error", "errors"), ("skipped", "skipped")):
            actual[key] += tag in outcomes
    require(bool(suites) and bool(cases) and actual == counts and len(members) == len(set(members)),
            "junit_membership_or_counts", coordinate)
    return counts, members


def pytest_summary(log, counts, exit_code, coordinate):
    """Accept one terminal summary, with every actual outcome represented."""
    require(log.endswith(f"\nreturncode={exit_code}\n"), "log_exit", coordinate)
    lines = [line for line in log.splitlines() if line.strip()]
    require(len(lines) >= 2, "missing_summary", coordinate)
    summary = lines[-2].strip("= ")
    matched = re.fullmatch(r"(.+) in [0-9]+(?:\.[0-9]+)?s(?: \([^\n]+\))?", summary)
    require(matched is not None, "invalid_summary", coordinate)
    summaries = [line for line in lines[:-1] if re.fullmatch(
        r"[0-9]+ (?:passed|failed|errors?|skipped)(?:, [0-9]+ [a-z]+)* in [0-9]+(?:\.[0-9]+)?s(?: \([^\n]+\))?",
        line.strip("= "))]
    require(len(summaries) == 1, "multiple_or_missing_summaries", coordinate)
    outcomes = {}
    for item in matched[1].split(", "):
        field = re.fullmatch(r"([0-9]+) (passed|failed|error|errors|skipped|warning|warnings)", item)
        require(field is not None, "unknown_summary_outcome", coordinate)
        name = field[2].removesuffix("s") if field[2] in {"errors", "warnings"} else field[2]
        require(name not in outcomes, "duplicate_summary_outcome", coordinate)
        outcomes[name] = int(field[1])
    expected = {"failed": counts["failures"], "error": counts["errors"], "skipped": counts["skipped"],
                "passed": counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]}
    require(all(outcomes.get(key, 0) == value for key, value in expected.items()),
            "summary_count_mismatch", coordinate)


def cpu_bytes(repo, name, raw, path, expected_hash, expected_commit, authority):
    entries, pin = load_authority(repo, authority)
    require(name in entries, "authority_missing_evidence", "receipt")
    entry = entries[name]
    safe.keys(entry, {"source_commit", "files", "membership", "runtime", "pytest_exit_code"}, "cpu_authority")
    require(entry["source_commit"] == expected_commit and type(entry["pytest_exit_code"]) is int
            and entry["pytest_exit_code"] == 0, "authority_source_or_exit", "receipt")
    files = authority_files(repo, entry, {"receipt", "junit", "pytest_log", "collection"})
    descriptor = entry["files"]["receipt"]
    require(external(path, repo) == external(descriptor["path"], repo)
            and expected_hash == descriptor["sha256"], "authority_receipt_mismatch", "receipt")
    counts, members = junit_record(files["junit"], "cpu_junit")
    expected = entry["membership"]
    require(isinstance(expected, list) and bool(expected)
            and all(isinstance(node, str) and "::" in node for node in expected)
            and expected == members, "cpu_original_membership", "receipt")
    original = tree_hashes(repo, expected_commit)
    require(all(node.split("::", 1)[0].replace(".", "/") + ".py" in original for node in expected),
            "cpu_membership_source_path", "receipt")
    try:
        collection = [line.strip() for line in files["collection"].decode().splitlines() if "::" in line]
        log = files["pytest_log"].decode()
    except UnicodeError as error:
        raise safe.Rejected("invalid_cpu_log", "receipt") from error
    collection = [node.split("::", 1)[0].removesuffix(".py").replace("/", ".")
                  + "::" + node.split("::", 1)[1] for node in collection]
    require(collection == expected, "cpu_collection_membership", "receipt")
    pytest_summary(log, counts, 0, "cpu_log")
    runtime = entry["runtime"]
    safe.keys(runtime, {"environment", "gpu_qualification"}, "cpu_runtime")
    require(runtime["gpu_qualification"] == "UNVERIFIED" and isinstance(runtime["environment"], dict)
            and runtime["environment"].get("CUDA_VISIBLE_DEVICES") == "-1", "cpu_runtime", "receipt")
    for field in ("exit_code", "pytest_exit_code", "returncode"):
        if field in raw:
            require(type(raw[field]) is int and raw[field] == 0, "cpu_conflicting_exit", "receipt")
    if "runtime" in raw:
        reported = raw["runtime"]
        require(isinstance(reported, dict) and reported.get("environment") == runtime["environment"],
                "cpu_conflicting_runtime", "receipt")
        if "gpu_qualification" in reported:
            require(reported["gpu_qualification"] == "UNVERIFIED", "cpu_conflicting_runtime", "receipt")
    if "junit" in raw:
        safe.keys(raw["junit"], counts, "cpu_junit")
        for key, value in raw["junit"].items():
            count(value, key)
        require(raw["junit"] == counts, "cpu_conflicting_junit", "receipt")
    if "exact_membership" in raw:
        require(raw["exact_membership"] is True, "cpu_conflicting_membership", "receipt")
    for key in ("membership", "collection_membership"):
        if key in raw:
            require(raw[key] == [node.split("::", 1)[1] for node in members], "cpu_conflicting_membership", "receipt")
    return counts, {"authority_sha256": pin, "junit_sha256": entry["files"]["junit"]["sha256"],
                    "pytest_log_sha256": entry["files"]["pytest_log"]["sha256"],
                    "collection_sha256": entry["files"]["collection"]["sha256"]}


def extract_receipt(repo, name, path, expected_hash, expected_commit, authority=None):
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
    actual, binding = cpu_bytes(repo, name, raw, path, expected_hash, expected_commit, authority)
    require(actual == counts, "cpu_actual_count_mismatch", "receipt")
    return {
        "source_commit": expected_commit, "receipt_sha256": expected_hash,
        "finished_at": normalized_time(raw.get("finished_at")),
        **counts, "passed": counts["tests"] - counts["skipped"],
        "kind": "reported-owner-cpu-checks", **binding,
    }


def operator_membership():
    kinds = ("dense", "coo", "csr", "matrix_free")
    models = ("interior_qp", "equality_qp", "bounded_lp", "empty_rows_qp")
    losses = ("reversed", "lower_overflow", "upper_overflow", "nonzero_underflow")
    ids = [f"{dtype}-{name}-{kind}" for dtype in ("f64", "f32") for name in models for kind in kinds]
    ids += [f"f32-{loss}-{kind}" for loss in losses for kind in kinds]
    ids += [f"f64-zero_row-{kind}" for kind in kinds] + ["f64-callback_cpu-matrix_free"]
    return ["tests.test_operator_original_cuda::test_cuda_collection_entry[" + label + "]" for label in ids]


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
            and type(raw.get("returncode")) is int and raw["returncode"] == 1,
            "cuda_failure_not_red", "cuda_failure")
    hashes = tree_hashes(repo, commit)
    require(raw.get("source_sha256") == {name: hashes.get(name)
                                        for name in CUDA_FAILURE_SOURCE_FILES},
            "cuda_failure_source_bytes", "cuda_failure")
    require(raw.get("targets") ==
            ["tests/test_operator_original_cuda.py::test_cuda_collection_entry"]
            and raw.get("selection") == "test_cuda_collection_entry",
            "cuda_failure_target", "cuda_failure")
    counts, members = junit_record(snapshots[1], "cuda_failure")
    try:
        log = snapshots[2].decode("utf-8")
    except UnicodeError as error:
        raise safe.Rejected("invalid_failed_cuda_junit_or_log", "cuda_failure") from error
    require(members == operator_membership() and counts["tests"] == 53
            and counts["failures"] > 0 and counts["errors"] == counts["skipped"] == 0,
            "cuda_failure_membership", "cuda_failure")
    passed = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    require(passed > 0, "cuda_failure_no_passes", "cuda_failure")
    pytest_summary(log, counts, 1, "cuda_failure")
    require(isinstance(raw.get("error"), str) and
            raw["error"] == f"ValueError: CUDA tests failed or skipped: {counts}",
            "cuda_failure_result_counts", "cuda_failure")
    if "tests" in raw:
        safe.keys(raw["tests"], counts, "cuda_result_counts")
        require(raw["tests"] == counts and all(type(v) is int for v in raw["tests"].values()),
                "cuda_failure_result_counts", "cuda_failure")
    return {
        "source_commit": commit, "receipt_sha256": result_hash,
        "junit_sha256": junit_hash, "pytest_log_sha256": log_hash,
        "finished_at": normalized_time(raw.get("finished_at")), **counts,
        "passed": passed, "kind": "reported-failed-cuda-owner-checks",
    }


def collect(repo, commit, modified_at, receipt_arguments, cuda_failure=None, authority=None):
    hashes = tree_hashes(repo, commit)
    record = {
        "schema_version": 2, "project": "openopt", "content_source_commit": commit,
        "modified_at": modified_at, "license": "0BSD",
        "boundary_sha256": {name: hashes.get(name) for name in BOUNDARY_PATHS},
        "access": "existing-super-only", "publication": "NOT_PUBLISHED",
        "qualification": dict(CLAIMS), "evidence": {},
    }
    require(bool(receipt_arguments) or cuda_failure is not None, "empty_evidence", "receipt")
    for name, path, receipt_hash, source_commit in receipt_arguments:
        require(name not in record["evidence"], "duplicate_evidence", "receipt")
        item = extract_receipt(repo, name, path, receipt_hash, source_commit, authority)
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
            if "junit_sha256" in item else ""
        )
        if "authority_sha256" in item:
            extra_hashes += (
                f"<dt>Caller manifest SHA-256</dt><dd><code>{item['authority_sha256']}</code></dd>"
                f"<dt>Collection SHA-256</dt><dd><code>{item['collection_sha256']}</code></dd>"
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
<meta name="generator" content="scripts/render_project_site.py schema 1/2">
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
<p>Schema 2 CPU counts also bind caller-pinned JUnit, collection and exit-log bytes.
Legacy schema 1 scalars are unverified receipt projections, never upgraded proof.</p>
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


def refresh_outputs(repo, files, previous):
    """Explicit pinned regeneration only; normal render remains create-once."""
    require(isinstance(previous, list) and len(previous) == len(OUTPUTS), "previous_output_pins", "cli")
    pins = dict(previous)
    safe.keys(pins, OUTPUTS, "previous_outputs")
    for name, pin in pins.items():
        digest(pin, "previous_output")
        require(safe.sha(safe.read(repo / name, name).payload) == pin,
                "previous_output_mismatch", name)
    # All inputs and both old files are checked before either output is replaced.
    for name in OUTPUTS:
        parent = safe.directory((repo / name).parent)
        stage = ".project-site-" + safe.sha(files[name])
        try:
            with os.fdopen(os.open(stage, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                                  0o644, dir_fd=parent), "wb") as stream:
                stream.write(files[name])
            require(safe.sha(safe.read(repo / name, name).payload) == pins[name],
                    "previous_output_changed", name)
            os.replace(stage, (repo / name).name, src_dir_fd=parent, dst_dir_fd=parent)
        finally:
            os.close(parent)
    check_outputs(repo, files)


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


def normalized_code(code):
    return code.replace(co_filename="<source>", co_consts=tuple(
        normalized_code(value) if isinstance(value, types.CodeType) else value
        for value in code.co_consts))


def loaded_source(repo):
    """Compare actual live functions with compile-only caller repository bytes."""
    modules = ((RUNTIME_PATHS[0], sys.modules[__name__]), (RUNTIME_PATHS[1], safe))
    result = {}
    for name, module in modules:
        original = safe.read(ROOT / name, "runtime_source").payload
        require(safe.read(repo / name, "runtime_source").payload == original,
                "runtime_repository_bytes", "source")
        compiled = compile(original, str(ROOT / name), "exec", dont_inherit=True)
        expected = {value.co_name: normalized_code(value) for value in compiled.co_consts
                    if isinstance(value, types.CodeType)}
        functions = {key: value for key, value in vars(module).items()
                     if isinstance(value, types.FunctionType) and value.__module__ == module.__name__}
        syntax = ast.parse(original)
        require(set(functions) == {node.name for node in syntax.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))},
                "loaded_function_membership", "source")
        for key, function in functions.items():
            require(key in expected and normalized_code(function.__code__) == expected[key],
                    "loaded_code_source_mismatch", "source")
        require(bool(functions), "empty_loaded_code", "source")
        # Literal policy values are not code objects; bind them separately.
        for node in syntax.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    literal = ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
                key = node.targets[0].id
                require(vars(module).get(key) == literal, "loaded_policy_mismatch", "source")
        for key, cls in vars(module).items():
            if isinstance(cls, type) and cls.__module__ == module.__name__:
                require(key in expected, "loaded_class_membership", "source")
                methods = {value.co_name: normalized_code(value) for value in expected[key].co_consts
                           if isinstance(value, types.CodeType)}
                for method, code in methods.items():
                    actual = vars(cls).get(method)
                    require(isinstance(actual, types.FunctionType) and normalized_code(actual.__code__) == code,
                            "loaded_method_source_mismatch", "source")
        result[name] = {"source_sha256": safe.sha(original), "functions": len(functions),
                        "code_sha256": safe.sha(marshal.dumps(tuple(
                            normalized_code(value.__code__) for _, value in sorted(functions.items()))))}
    return result


def execute(args, artifacts):
    repo = safe.absolute(args.repository, "repository")
    before = source_state(repo)
    report = {
        "schema_version": 1, "operation": args.operation, "source_before": before,
        "qualification": dict(CLAIMS), "publication": "NOT_PUBLISHED",
        "access": "existing-super-only", "verdict": "FAIL", "exit_code": 1,
        "owned_sha256": {name: before["sha256"].get(name) for name in OWNED},
    }
    try:
        if args.operation == "check":
            report["runtime_binding"] = loaded_source(repo)
        if args.operation == "collect":
            require(args.content_source_commit is not None and args.modified_at is not None,
                    "collect_arguments", "cli")
            for _, receipt_path, _, _ in args.evidence:
                external(receipt_path, repo)
            if getattr(args, "cuda_failure", None):
                for position in (0, 2, 4):
                    external(args.cuda_failure[position], repo)
            payload = collect(repo, args.content_source_commit, args.modified_at,
                              args.evidence, getattr(args, "cuda_failure", None),
                              getattr(args, "authority", None))
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
            elif args.operation == "refresh":
                report["runtime_binding"] = loaded_source(repo)
                refresh_outputs(repo, files, getattr(args, "previous_output", None))
            else:
                require(args.owner_commit is not None, "owner_pin_required", "cli")
                digest(args.owner_commit, "owner", commit=True)
                require(before["commit"] == args.owner_commit and before["clean"],
                        "owner_not_pinned_clean", "source")
                expected = tree_hashes(repo, args.owner_commit)
                require(before["sha256"] == expected, "owner_bytes_not_committed", "source")
            check_outputs(repo, files)
            if args.operation == "check":
                require(report["runtime_binding"] == loaded_source(repo),
                        "loaded_runtime_changed", "source")
            report["input_sha256"] = safe.sha(payload)
            report["outputs_sha256"] = {name: safe.sha(value) for name, value in files.items()}
        after = source_state(repo)
        report["source_after"] = after
        changed = sorted(name for name in before["sha256"].keys() & after["sha256"].keys()
                         if before["sha256"][name] != after["sha256"][name])
        extra = sorted(after["sha256"].keys() - before["sha256"].keys())
        missing = sorted(before["sha256"].keys() - after["sha256"].keys())
        require(before["commit"] == after["commit"] and not missing
                and (set(changed) <= set(OUTPUTS) if args.operation == "refresh" else not changed)
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
    parser.add_argument("operation", choices=("collect", "render", "refresh", "check"))
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--input-sha256")
    parser.add_argument("--content-source-commit")
    parser.add_argument("--owner-commit")
    parser.add_argument("--previous-output", action="append", nargs=2,
                        metavar=("NAME", "SHA256"), help="refresh only: exact old generated output pins")
    parser.add_argument("--modified-at")
    parser.add_argument("--authority", nargs=2, metavar=("PATH", "SHA256"),
                        help="external caller-owned CPU evidence manifest and trusted SHA pin")
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
