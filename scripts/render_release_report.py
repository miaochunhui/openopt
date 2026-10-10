"""Render only public-safe v4 pages from explicit caller-pinned 3FS inputs.

No solver, network, historical generator or publication entry is invoked.
The private report-bundle records input descriptors; never publish that file.
CLI and schemas are documented in docs/research/release-report.md.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import unicodedata
from urllib.parse import unquote
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[1]
SHARED = Path("/3fs/research3fs/data/tmp")
HASH = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
ID = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
REQUIRED = frozenset({
    "cpu", "junit", "wheel", "gpu", "gpu_run", "commercial", "corpus",
    "campaign", "audit", "live", "metadata",
})
EXTENSIONS = {
    "extension_native": ("native_conic", "原生锥 CUDA"),
    "extension_presolve": ("presolve", "保守预处理 CUDA"),
    "extension_refinement": ("refinement", "精度修正 CUDA"),
}
OPTIONAL = frozenset({
    "gpu_junit", "cpu_version", "public_qualification", "proof_contract_review",
    "public_robustness", "public_robustness_plan", "public_robustness_corpus",
    "extension_presolve_run", *EXTENSIONS,
})
STATUS = frozenset({
    "PASS", "FAIL", "UNVERIFIED", "QUALIFIED", "REJECTED", "INCONCLUSIVE",
    "RUNNING", "PENDING", "COMPLETE", "solved", "max_iterations",
    "infeasible", "unbounded", "NOT_QUALIFIED", "UNQUALIFIED",
})
MAX_BYTES = 16 * 1024 * 1024
LEAKS = (
    re.compile(r"[a-z][a-z0-9+.-]*://|(?<!:)//[a-z0-9]|"
               r"\b(?:\d{1,3}\.){3}\d{1,3}\b|"
               r"\b(?:[0-9a-f]{1,4}:){3,}[0-9a-f:]*\b|"
               r"(?<![\w:])[0-9a-f:]*::[0-9a-f:]+(?![\w:])|"
               r"\b(?:localhost|[\w-]+(?:\.[\w-]+)*\."
               r"(?:com|org|net|io|cn|local|internal|lan))\b|[\w.+-]+@[\w.-]+", re.I),
    re.compile(r"(?:^|[\s\"'>(=:])/(?:[A-Za-z0-9_.-]+/)+|"
               r"/(?:data|home|root|tmp|etc|3fs|hf3fs-jd)\b|[A-Za-z]:[\\/]|\\\\", re.I),
    re.compile(r"\b(?:h3c|hw|hdp|mch|gyf|r640)\d[\w.-]*\b|"
               r"\b(?:hostname|host_name|node_name|machine_id)\s*[:=]", re.I),
    re.compile(r"\b(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|"
               r"client[_-]?secret|authorization)\b[\"']?\s*[:=]|"
               r"\bBearer\s+\S+|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
               r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
               r"\b(?:gh[pousr]_|sk-)[A-Za-z0-9_-]{16,}|"
               r"\beyJ[\w-]+\.[\w-]+\.[\w-]+\b", re.I),
    re.compile(r"\b(?:license|licence)[ _-]*(?:server|key|file)\b|"
               r"\b(?:LM_LICENSE_FILE|GRB_LICENSE_FILE|MOSEKLM_LICENSE_FILE)\b|"
               r"^\s*(?:FEATURE|INCREMENT|SERVER|VENDOR|DAEMON)\s+\S+", re.I | re.M),
    re.compile(r"<\s*(?:script|iframe|object|embed|base|foreignObject)\b|"
               r"\bon[a-z]+\s*=|javascript\s*:|data\s*:|@import\b", re.I),
)


class Rejected(ValueError):
    """Fail-loud code and safe coordinate, without echoing rejected content."""

    def __init__(self, code: str, coordinate: str):
        self.code, self.coordinate = code, coordinate
        super().__init__(f"{code} at {coordinate}")


def require(condition, code, coordinate):
    if not condition:
        raise Rejected(code, coordinate)


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, "duplicate_json_key", "json")
        result[key] = value
    return result


def nonfinite(_):
    raise Rejected("nonfinite_json", "json")


def parse(payload: bytes, coordinate: str):
    require(bool(payload.strip()), "empty_input", coordinate)
    try:
        value = json.loads(payload.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=nonfinite)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise Rejected("invalid_json", coordinate) from error
    require(isinstance(value, dict) and bool(value), "empty_or_nonobject_json", coordinate)
    return value


def shared_path(value, coordinate):
    raw = os.fspath(value)
    require(isinstance(raw, str) and raw.startswith("/") and "\\" not in raw
            and ".." not in raw.split("/") and not any(ord(c) < 32 for c in raw),
            "invalid_shared_path", coordinate)
    if raw.startswith("/hf3fs-jd/"):
        raw = "/3fs/research3fs/" + raw[len("/hf3fs-jd/"):]
    path = Path(raw)
    require(path.is_relative_to(SHARED) and path != SHARED,
            "outside_shared_storage", coordinate)
    return path


def directory(path: Path, create=False):
    fd = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            if create:
                try:
                    os.mkdir(part, dir_fd=fd)
                except FileExistsError:
                    pass
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=fd)
            os.close(fd)
            fd = child
        return fd
    except OSError as error:
        os.close(fd)
        raise Rejected("unsafe_directory", "filesystem") from error


def read(path: Path, coordinate):
    fd = directory(path.parent)
    try:
        target = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=fd)
        with os.fdopen(target, "rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_BYTES,
                    "nonregular_or_large_input", coordinate)
            payload = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after), "snapshot_race", coordinate)
        require(bool(payload.strip()) and len(payload) <= MAX_BYTES, "empty_or_large_input", coordinate)
        return payload, identity(after)
    except OSError as error:
        raise Rejected("missing_or_unsafe_input", coordinate) from error
    finally:
        os.close(fd)


def public_text(value, coordinate):
    require(isinstance(value, str) and bool(value.strip()), "invalid_text", coordinate)
    require(not any((ord(c) < 32 and c not in "\n\r\t") or
                    unicodedata.category(c) == "Cf" for c in value),
            "hidden_character", coordinate)
    decoded = value
    for _ in range(4):
        next_value = html.unescape(unquote(decoded))
        next_value = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m[1], 16)), next_value)
        next_value = next_value.replace("\\/", "/")
        if next_value == decoded:
            break
        decoded = next_value
    else:
        raise Rejected("excessive_encoding", coordinate)
    decoded = unicodedata.normalize("NFKC", decoded)
    require(not any(rule.search(decoded) for rule in LEAKS), "unsafe_public_content", coordinate)
    return value


def public_json(payload, coordinate):
    """JSON pointers are typed references, not arbitrary absolute paths."""
    record = parse(payload, coordinate)

    def visit(value, key=None):
        if isinstance(value, dict):
            for name, child in value.items():
                public_text(name, coordinate)
                visit(child, name)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            if key == "pointer":
                require(bool(re.fullmatch(r"/scalars/[a-z][a-z0-9_-]{0,63}", value)),
                        "unsafe_claim_pointer", coordinate)
            elif key == "unit" and value == "":
                return
            else:
                public_text(value, coordinate)

    visit(record)


def integer(value, coordinate, positive=False):
    require(type(value) is int and value >= int(positive), "invalid_count", coordinate)
    return value


def status(value, coordinate):
    require(isinstance(value, str) and value in STATUS, "invalid_status", coordinate)
    return value


def timestamp(value, coordinate):
    require(isinstance(value, str), "invalid_timestamp", coordinate)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise Rejected("invalid_timestamp", coordinate) from error
    require(parsed.utcoffset() is not None, "timestamp_without_zone", coordinate)
    return parsed


def pointer(record, value):
    require(isinstance(value, str) and value.startswith("/"), "invalid_pointer", "pointer")
    for token in value[1:].split("/"):
        require(not re.search(r"~(?![01])", token), "invalid_pointer", "pointer")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(record, dict):
            require(token in record, "missing_field", "pointer")
            record = record[token]
        elif isinstance(record, list):
            require(bool(re.fullmatch(r"0|[1-9][0-9]*", token)) and int(token) < len(record),
                    "missing_index", "pointer")
            record = record[int(token)]
        else:
            raise Rejected("nonscalar_pointer", "pointer")
    return record


def junit_counts(payload, coordinate):
    require(b"<!DOCTYPE" not in payload.upper() and b"<!ENTITY" not in payload.upper(),
            "unsafe_junit", coordinate)
    try:
        root = ElementTree.fromstring(payload)
    except ElementTree.ParseError as error:
        raise Rejected("invalid_junit", coordinate) from error
    require(root.tag in ("testsuites", "testsuite"), "invalid_junit_root", coordinate)
    suites = [suite for suite in root.iter("testsuite") if not suite.findall("testsuite")]
    require(bool(suites), "empty_junit", coordinate)
    total = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
    for suite in suites:
        cases = suite.findall("testcase")
        counts = {"tests": len(cases)}
        for name, tag in (("failures", "failure"), ("errors", "error"), ("skipped", "skipped")):
            counts[name] = sum(case.find(tag) is not None for case in cases)
        require(all(sum(case.find(tag) is not None for tag in ("failure", "error", "skipped")) <= 1
                    for case in cases), "ambiguous_junit_case", coordinate)
        for name, actual in counts.items():
            raw = suite.get(name)
            require(raw is not None and bool(re.fullmatch(r"0|[1-9][0-9]*", raw))
                    and int(raw) == actual, "junit_false_count", coordinate)
            total[name] += actual
    require(total["tests"] > 0, "empty_junit", coordinate)
    return total


def test_counts(record, coordinate):
    require(isinstance(record.get("tests"), dict), "missing_counts", coordinate)
    counts = {key: integer(record["tests"].get(key), coordinate)
              for key in ("tests", "failures", "errors", "skipped")}
    require(counts["tests"] > 0 and sum(counts[key] for key in ("failures", "errors", "skipped"))
            <= counts["tests"], "false_count", coordinate)
    verdict = status(record.get("verdict"), coordinate)
    require(verdict in ("PASS", "FAIL"), "incomplete_regression", coordinate)
    require(verdict != "PASS" or counts["failures"] + counts["errors"] == 0,
            "false_pass_junit", coordinate)
    if "returncode" in record:
        require(type(record["returncode"]) is int and
                (verdict != "PASS" or record["returncode"] == 0),
                "false_pass_exit_code", coordinate)
    return counts


class StaticPage(HTMLParser):
    def handle_starttag(self, tag, attrs):
        require(tag not in {"script", "iframe", "object", "embed", "base", "foreignobject",
                            "form", "input", "button", "animate", "set"},
                "active_html", "html")
        for name, value in attrs:
            require(not name.startswith("on"), "event_handler", "html")
            if name in {"href", "src", "xlink:href", "action", "formaction", "srcset"}:
                require(name == "href" and isinstance(value, str) and
                        (bool(re.fullmatch(r"#[a-z][a-z0-9_-]*", value)) or
                         bool(re.fullmatch(r"[a-z][a-z0-9_-]*\.html", value))),
                        "nonstatic_link", "html")


class Report:
    def __init__(self, manifest: Path, manifest_sha256: str):
        require(isinstance(manifest_sha256, str) and HASH.fullmatch(manifest_sha256),
                "invalid_manifest_pin", "manifest")
        self.snapshots = {}
        self.manifest_path = shared_path(manifest, "manifest")
        self.manifest_sha256 = manifest_sha256
        self.manifest = parse(self.input(self.manifest_path, manifest_sha256, "manifest"), "manifest")
        m = self.manifest
        require(set(m) == {"schema_version", "modified_at", "source_sha", "inputs"} and
                type(m["schema_version"]) is int and m["schema_version"] == 1,
                "manifest_schema", "manifest")
        timestamp(m["modified_at"], "manifest")
        require(isinstance(m["source_sha"], str) and COMMIT.fullmatch(m["source_sha"]),
                "invalid_source_sha", "manifest")
        require(isinstance(m["inputs"], list) and bool(m["inputs"]), "empty_inputs", "manifest")
        self.descriptors, self.records = {}, {}
        paths = set()
        for item in m["inputs"]:
            require(isinstance(item, dict) and set(item) == {"id", "path", "sha256"},
                    "descriptor_schema", "inputs")
            name = item["id"]
            require(isinstance(name, str) and ID.fullmatch(name) and name not in self.descriptors
                    and name in REQUIRED | OPTIONAL, "invalid_or_duplicate_id", "inputs")
            path = shared_path(item["path"], name)
            require(path not in paths and path != self.manifest_path, "duplicate_input_path", name)
            paths.add(path)
            payload = self.input(path, item["sha256"], name)
            self.descriptors[name] = item
            self.records[name] = payload if name in {"junit", "gpu_junit"} else parse(payload, name)
        require(REQUIRED <= self.records.keys(), "missing_required_input", "inputs")
        self.scalars, self.provenance, self.claims = {}, {}, []

    def input(self, path, digest, coordinate):
        require(isinstance(digest, str) and HASH.fullmatch(digest), "invalid_input_pin", coordinate)
        snapshot = read(path, coordinate)
        require(sha(snapshot[0]) == digest, "changed_hash", coordinate)
        require(path not in self.snapshots or self.snapshots[path] == snapshot,
                "snapshot_race", coordinate)
        self.snapshots[path] = snapshot
        return snapshot[0]

    def unchanged(self):
        for path, snapshot in self.snapshots.items():
            require(read(path, "recheck") == snapshot, "snapshot_race", "recheck")

    def scalar(self, key, value, role, source_pointer, operation="identity"):
        require(key not in self.scalars and ID.fullmatch(key), "duplicate_scalar", "scalar")
        if isinstance(value, str):
            public_text(value, "scalar")
        else:
            require(type(value) in (int, float, bool) and
                    (type(value) is not float or math.isfinite(value)), "invalid_scalar", "scalar")
        self.scalars[key] = value
        self.provenance[key] = {
            "evidence_id": role, "evidence_sha256": self.descriptors[role]["sha256"],
            "pointer": source_pointer, "operation": operation,
        }
        return key

    def direct(self, key, role, source_pointer):
        return self.scalar(key, pointer(self.records[role], source_pointer), role, source_pointer)

    def show(self, key, document, tag="span"):
        value = self.scalars[key]
        kind = "number" if type(value) in (int, float) else "status" if value in STATUS else "text"
        text = str(value)
        claim = {"id": f"{document.split('-')[0]}_{key}", "evidence": "summary",
                 "pointer": f"/scalars/{key}", "kind": kind,
                 "unit": "count" if kind == "number" else "",
                 "document": f"docs/{document}", "text": text}
        if claim not in self.claims:
            self.claims.append(claim)
        return f'<{tag} data-scalar="{key}">{html.escape(text)}</{tag}>'

    def project(self):
        r, meta = self.records, self.records["metadata"]
        require(meta.get("version") == 4 and type(meta.get("version")) is int,
                "not_v4_metadata", "metadata")
        require(meta.get("source_sha") == self.manifest["source_sha"] and
                meta.get("modified_at") == self.manifest["modified_at"],
                "metadata_binding_mismatch", "metadata")
        require(meta.get("regression_scope") in {"intermediate_snapshot", "frozen_release"},
                "missing_regression_scope", "metadata")
        require(meta.get("online_publication") == "UNVERIFIED" and
                meta.get("commercial_qualification") == "UNVERIFIED",
                "unsupported_qualification", "metadata")
        self.direct("version", "metadata", "/version")
        self.direct("source_sha", "metadata", "/source_sha")
        doc_sha = meta.get("doc_source_sha")
        if doc_sha is not None:
            require(isinstance(doc_sha, str) and COMMIT.fullmatch(doc_sha),
                    "invalid_doc_source_sha", "metadata")
            self.direct("doc_source_sha", "metadata", "/doc_source_sha")
        self.direct("regression_scope", "metadata", "/regression_scope")
        self.direct("online_status", "metadata", "/online_publication")
        self.direct("commercial_status", "metadata", "/commercial_qualification")
        binding = meta.get("cpu_binding")
        require(isinstance(binding, dict) and binding.get("kind") in
                {"scope_only", "code_hash", "version_metadata"}, "cpu_binding_required", "metadata")
        self.direct("cpu_binding_kind", "metadata", "/cpu_binding/kind")
        if binding["kind"] != "scope_only":
            require("cpu_version" in r, "cpu_version_required", "cpu_version")
            version = r["cpu_version"]
            require(version.get("source_sha") == meta["source_sha"], "cpu_source_mismatch", "cpu_version")
            expected = version.get("code_hash") if binding["kind"] == "code_hash" else version.get("version")
            require(expected is not None and binding.get("value") == expected,
                    "cpu_binding_mismatch", "cpu_version")
            if binding["kind"] == "code_hash":
                require(isinstance(expected, str) and HASH.fullmatch(expected),
                        "invalid_code_hash", "cpu_version")
            self.scalar("cpu_binding_value", expected, "cpu_version",
                        "/code_hash" if binding["kind"] == "code_hash" else "/version")
        require(meta["regression_scope"] != "frozen_release" or binding["kind"] != "scope_only",
                "unbound_frozen_release", "metadata")
        consistency = r["cpu"].get("source_consistency")
        if consistency is None:
            self.scalar("cpu_source_consistency", "UNVERIFIED", "cpu", "/source_consistency",
                        "missing-field-unverified")
            require(meta["regression_scope"] != "frozen_release",
                    "legacy_regression_not_frozen", "cpu")
        else:
            require(isinstance(consistency, dict) and consistency.get("verdict") in {"PASS", "FAIL"},
                    "invalid_source_consistency", "cpu")
            self.direct("cpu_source_consistency", "cpu", "/source_consistency/verdict")
            source = r["cpu"].get("source")
            require(isinstance(source, dict) and isinstance(source.get("sha256"), dict)
                    and bool(source["sha256"]) and isinstance(source.get("commit"), str)
                    and COMMIT.fullmatch(source["commit"]) and isinstance(source.get("tree"), str)
                    and COMMIT.fullmatch(source["tree"]) and type(source.get("clean")) is bool
                    and all(isinstance(v, str) and HASH.fullmatch(v) for v in source["sha256"].values()),
                    "invalid_cpu_source_snapshot", "cpu")
            require(source["commit"] == meta["source_sha"], "cpu_source_mismatch", "cpu")
            self.direct("cpu_source_commit", "cpu", "/source/commit")
            self.scalar("cpu_source_hash", sha(encoded(source["sha256"])), "cpu",
                        "/source/sha256", "sha256-canonical-sorted-json-with-newline")
            self.direct("cpu_source_files", "cpu", "/source_consistency/files")
            require(type(consistency.get("same_commit")) is bool
                    and integer(consistency.get("files"), "cpu", True) == len(source["sha256"]),
                    "source_consistency_false_count", "cpu")
            for name in ("changed", "missing", "extra"):
                require(isinstance(consistency.get(name), list), "source_consistency_missing_detail", "cpu")
                self.scalar(f"cpu_source_{name}", len(consistency[name]), "cpu",
                            f"/source_consistency/{name}", "length")
            if consistency["verdict"] == "PASS":
                require(consistency["same_commit"] and not any(
                    consistency[name] for name in ("changed", "missing", "extra")),
                    "false_source_consistency_pass", "cpu")
            else:
                require(r["cpu"].get("verdict") == "FAIL", "source_failure_hidden_by_pass", "cpu")
            if binding["kind"] == "code_hash":
                require(binding["value"] == self.scalars["cpu_source_hash"],
                        "cpu_code_hash_not_source_snapshot", "cpu")
            if meta["regression_scope"] == "frozen_release":
                require(consistency["verdict"] == "PASS" and source["clean"]
                        and r["cpu"].get("coverage") == "tests",
                        "source_not_clean_frozen_full_regression", "cpu")
        for role in ("cpu", "gpu"):
            counts = test_counts(r[role], role)
            xml_role = "junit" if role == "cpu" else "gpu_junit"
            if xml_role in r:
                require(counts == junit_counts(r[xml_role], xml_role), "junit_json_mismatch", role)
            self.direct(f"{role}_status", role, "/verdict")
            for key, value in counts.items():
                self.direct(f"{role}_{key}", role, f"/tests/{key}")
            self.scalar(f"{role}_passed", counts["tests"] - counts["skipped"] -
                        counts["failures"] - counts["errors"], role, "/tests",
                        "tests-minus-skipped-minus-failures-minus-errors")
        self.direct("cpu_coverage", "cpu", "/coverage")
        self.direct("cpu_finished", "cpu", "/finished_at")
        self.direct("gpu_scope", "gpu", "/scope")
        self.direct("gpu_finished", "gpu", "/finished_at")
        timestamp(r["cpu"]["finished_at"], "cpu")
        timestamp(r["gpu"]["finished_at"], "gpu")
        require(isinstance(r["gpu"].get("source_sha256"), dict) and bool(r["gpu"]["source_sha256"])
                and all(isinstance(v, str) and HASH.fullmatch(v)
                        for v in r["gpu"]["source_sha256"].values()), "gpu_source_hashes_missing", "gpu")
        gpu_sha = r["gpu_run"].get("source_sha")
        require(isinstance(gpu_sha, str) and COMMIT.fullmatch(gpu_sha), "gpu_run_source_missing", "gpu_run")
        self.direct("gpu_source_sha", "gpu_run", "/source_sha")
        self.scalar("gpu_result_hash", self.descriptors["gpu"]["sha256"], "gpu", "", "descriptor-sha256")
        self.scalar("gpu_run_hash", self.descriptors["gpu_run"]["sha256"], "gpu_run", "", "descriptor-sha256")
        wheel = r["wheel"]
        status(wheel.get("verdict"), "wheel")
        require(wheel.get("license_expression") == "0BSD", "wheel_not_0bsd", "wheel")
        integer(wheel.get("modules"), "wheel", True)
        self.direct("wheel_status", "wheel", "/verdict")
        self.direct("wheel_modules", "wheel", "/modules")
        self.direct("wheel_license", "wheel", "/license_expression")
        wheel_gate = wheel.get("source_consistency")
        if wheel_gate is None:
            self.scalar("wheel_source_consistency", "UNVERIFIED", "wheel", "/source_consistency",
                        "missing-field-unverified")
            require(meta["regression_scope"] != "frozen_release",
                    "legacy_wheel_not_frozen", "wheel")
        else:
            source = wheel.get("source")
            require(isinstance(wheel_gate, dict) and wheel_gate.get("verdict") in {"PASS", "FAIL"}
                    and isinstance(source, dict) and isinstance(source.get("sha256"), dict)
                    and bool(source["sha256"]) and source.get("commit") == meta["source_sha"]
                    and type(source.get("clean")) is bool
                    and all(isinstance(v, str) and HASH.fullmatch(v) for v in source["sha256"].values()),
                    "invalid_wheel_source", "wheel")
            require(type(wheel_gate.get("same_commit")) is bool
                    and integer(wheel_gate.get("files"), "wheel", True) == len(source["sha256"])
                    and all(isinstance(wheel_gate.get(key), list)
                            for key in ("changed", "missing", "extra")),
                    "invalid_wheel_consistency", "wheel")
            if wheel_gate["verdict"] == "PASS":
                require(wheel_gate["same_commit"] and not any(wheel_gate[key] for key in
                        ("changed", "missing", "extra")), "false_wheel_consistency_pass", "wheel")
            else:
                require(wheel["verdict"] == "FAIL", "wheel_source_failure_hidden_by_pass", "wheel")
            self.direct("wheel_source_consistency", "wheel", "/source_consistency/verdict")
            self.direct("wheel_source_commit", "wheel", "/source/commit")
            self.scalar("wheel_source_hash", sha(encoded(source["sha256"])), "wheel",
                        "/source/sha256", "sha256-canonical-sorted-json-with-newline")
            wheel_hash = wheel.get("wheel_sha256")
            require(isinstance(wheel_hash, str) and HASH.fullmatch(wheel_hash),
                    "wheel_hash_missing", "wheel")
            self.direct("wheel_hash", "wheel", "/wheel_sha256")
            require(meta["regression_scope"] != "frozen_release" or
                    (wheel_gate["verdict"] == "PASS" and source["clean"]),
                    "wheel_not_clean_frozen", "wheel")
        commercial = r["commercial"]
        require(commercial.get("qualification") == "UNVERIFIED", "commercial_not_unverified", "commercial")
        status(commercial.get("verdict"), "commercial")
        self.direct("commercial_result", "commercial", "/verdict")
        counts = commercial.get("counts")
        require(isinstance(counts, dict), "commercial_counts_missing", "commercial")
        for key in ("planned", "attempted", "solve_attempted", "passed", "failed"):
            integer(counts.get(key), "commercial")
            self.direct(f"commercial_{key}", "commercial", f"/counts/{key}")
        require(counts["passed"] + counts["failed"] <= counts["solve_attempted"]
                <= counts["attempted"] <= counts["planned"], "commercial_false_count", "commercial")
        solvers = commercial.get("solvers")
        require(isinstance(solvers, dict) and bool(solvers), "empty_commercial_solvers", "commercial")
        self.commercial_rows = []
        for index, (name, entry) in enumerate(sorted(solvers.items())):
            public_text(name, "commercial_solver")
            require(isinstance(entry, dict), "invalid_commercial_solver", "commercial")
            status(entry.get("status"), "commercial")
            token = name.replace("~", "~0").replace("/", "~1")
            key = self.direct(f"sdk_{index}_status", "commercial", f"/solvers/{token}/status")
            reason_key = None
            if entry.get("reason") is not None:
                reason_key = self.direct(f"sdk_{index}_reason", "commercial", f"/solvers/{token}/reason")
            self.commercial_rows.append((name, key, reason_key))
        corpus = r["corpus"]
        status(corpus.get("verdict"), "corpus")
        rows = corpus.get("rows")
        require(isinstance(rows, list) and len(rows) >= 24, "incomplete_corpus", "corpus")
        self.direct("corpus_status", "corpus", "/verdict")
        self.direct("corpus_scope", "corpus", "/scope")
        seen, self.corpus_rows = set(), []
        qualified = 0
        for index, row in enumerate(rows):
            require(isinstance(row, dict), "invalid_corpus_row", "corpus")
            name, solver = row.get("case_id"), row.get("solver")
            public_text(name, "corpus_case")
            public_text(solver, "corpus_solver")
            require((name, solver) not in seen, "duplicate_corpus_row", "corpus")
            seen.add((name, solver))
            verdict = status(row.get("verdict"), "corpus")
            solver_status = status(row.get("solver_status"), "corpus")
            require(type(row.get("status_matches_expected")) is bool, "corpus_match_missing", "corpus")
            check_keys = []
            for field in ("candidate_quality", "certificate_quality"):
                if row.get(field) is not None:
                    require(isinstance(row[field], dict), "invalid_corpus_check", "corpus")
                    status(row[field].get("verdict"), "corpus")
                    check_keys.append(self.direct(f"corpus_{index}_{field}", "corpus",
                                                  f"/rows/{index}/{field}/verdict"))
            require(bool(check_keys), "missing_corpus_checks", "corpus")
            gate = "candidate_quality" if row.get("expected_property") == "optimal" else "certificate_quality"
            require(verdict != "QUALIFIED" or (row["status_matches_expected"] and
                    isinstance(row.get(gate), dict) and row[gate].get("verdict") == "PASS"),
                    "false_corpus_qualification", "corpus")
            qualified += verdict == "QUALIFIED"
            keys = [self.direct(f"corpus_{index}_{field}", "corpus", f"/rows/{index}/{field}")
                    for field in ("case_id", "solver", "verdict", "solver_status")]
            self.corpus_rows.append((keys, check_keys))
        require(len({name for name, _ in seen}) == 24, "synthetic_case_scope_mismatch", "corpus")
        self.scalar("corpus_rows", len(rows), "corpus", "/rows", "length")
        self.scalar("corpus_qualified", qualified, "corpus", "/rows", "count-verdict-QUALIFIED")
        self.scalar("corpus_rejected", len(rows) - qualified, "corpus", "/rows", "count-not-QUALIFIED")
        campaign, audit = r["campaign"], r["audit"]
        seats, audits = campaign.get("seats"), audit.get("seats")
        require(isinstance(seats, dict) and bool(seats) and isinstance(audits, list)
                and bool(audits), "empty_campaign_audit", "audit")
        require(len(audits) == len(seats) and {row.get("seat") for row in audits} == set(seats),
                "campaign_audit_membership", "audit")
        self.audit_rows = []
        for index, row in enumerate(audits):
            public_text(row["seat"], "audit_seat")
            verdict = status(row.get("verdict"), "audit")
            outside = row.get("outside_scope")
            require(isinstance(outside, list) and (verdict != "PASS" or not outside),
                    "false_scope_pass", "audit")
            self.audit_rows.append((
                self.direct(f"seat_{index}", "audit", f"/seats/{index}/seat"),
                self.direct(f"seat_{index}_status", "audit", f"/seats/{index}/verdict"),
                self.scalar(f"seat_{index}_outside", len(outside), "audit",
                            f"/seats/{index}/outside_scope", "length"),
            ))
        self.scalar("audit_seats", len(audits), "audit", "/seats", "length")
        self.scalar("audit_failed", sum(row["verdict"] != "PASS" for row in audits),
                    "audit", "/seats", "count-not-PASS")
        live = r["live"]
        status(live.get("verdict"), "live")
        timestamp(live.get("read_at"), "live")
        self.direct("live_status", "live", "/verdict")
        self.direct("live_read_at", "live", "/read_at")
        for name in ("ready", "active", "cordoned", "retained_failures"):
            require(isinstance(live.get(name), (list, dict)), "live_collection_missing", "live")
            self.scalar(f"live_{name}", len(live[name]), "live", f"/{name}", "length")
        require(bool(live["ready"]), "empty_live_nodes", "live")
        if "public_qualification" in r:
            public = r["public_qualification"]
            status(public.get("verdict"), "public_qualification")
            public_text(public.get("scope"), "public_qualification")
            counts = public.get("counts")
            require(isinstance(counts, dict), "public_counts_missing", "public_qualification")
            for key in ("planned", "attempted", "qualified", "rejected"):
                integer(counts.get(key), "public_qualification")
                self.direct(f"public_{key}", "public_qualification", f"/counts/{key}")
            require(counts["qualified"] + counts["rejected"] <= counts["attempted"]
                    <= counts["planned"], "public_false_count", "public_qualification")
            require(public["verdict"] != "PASS" or counts["attempted"] > 0,
                    "public_empty_false_pass", "public_qualification")
            self.direct("public_status", "public_qualification", "/verdict")
            self.direct("public_scope", "public_qualification", "/scope")
        self.extension_rows = []
        for role, (suite, label) in EXTENSIONS.items():
            if role not in r:
                continue
            evidence = r[role]
            require(evidence.get("verdict") in {"PASS", "FAIL"},
                    "incomplete_extension_acceptance", role)
            timestamp(evidence.get("finished_at"), role)
            keys = [self.direct(f"{role}_status", role, "/verdict"),
                    self.direct(f"{role}_finished", role, "/finished_at")]
            if evidence["verdict"] == "PASS":
                require(evidence.get("suite") == suite
                        and (suite == "presolve" or evidence.get("final_source") is True)
                        and evidence.get("qualified") is False
                        and evidence.get("certified") is False
                        and evidence.get("commercial") == "UNVERIFIED"
                        and evidence.get("performance") == "UNVERIFIED",
                        "extension_acceptance_overclaim", role)
                counts = test_counts(evidence, role)
                require(counts == {
                    "tests": {"native_conic": 21, "presolve": 8, "refinement": 7}[suite],
                    "failures": 0, "errors": 0, "skipped": 0,
                }, "extension_independent_denominator", role)
                source = evidence.get("source_sha_informational")
                source_role, source_pointer = role, "/source_sha_informational"
                if suite == "presolve":
                    source_role, source_pointer = "extension_presolve_run", "/source_sha"
                    require(source_role in r and evidence.get("current_core") == "PASS",
                            "presolve_run_pin_required", role)
                    original = evidence.get("pins", {}).get("run", {})
                    selected = self.descriptors[source_role]
                    require(original.get("sha256") == selected["sha256"]
                            and shared_path(original.get("path", ""), role) ==
                            shared_path(selected["path"], role),
                            "presolve_acceptance_run_mismatch", role)
                    source = r[source_role].get("source_sha")
                require(isinstance(source, str) and COMMIT.fullmatch(source),
                        "extension_source_missing", role)
                self.scalar(f"{role}_source", source, source_role, source_pointer)
                self.direct(f"{role}_tests", role, "/tests/tests")
            self.extension_rows.append((role, label, keys))
        require("extension_presolve_run" not in r or "extension_presolve" in r,
                "unused_presolve_run", "extension_presolve_run")
        if "proof_contract_review" in r:
            role = "proof_contract_review"
            evidence = r[role]
            require(evidence.get("verdict") in {"PASS", "FAIL"},
                    "incomplete_proof_contract_review", role)
            timestamp(evidence.get("finished_at"), role)
            source = evidence.get("producer", {}).get("commit")
            require(isinstance(source, str) and COMMIT.fullmatch(source),
                    "proof_review_source_missing", role)
            self.direct("proof_review_status", role, "/verdict")
            self.direct("proof_review_finished", role, "/finished_at")
            self.direct("proof_review_source", role, "/producer/commit")
        if "public_robustness" in r:
            self.robustness()
        require(not ({"public_robustness_plan", "public_robustness_corpus"} & set(r))
                or "public_robustness" in r, "unused_robustness_contract", "public_robustness")
        self.failure_rows = []

        def failures(value, role, tokens):
            if isinstance(value, dict):
                for key, child in value.items():
                    escaped = key.replace("~", "~0").replace("/", "~1")
                    failures(child, role, tokens + [escaped])
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    failures(child, role, tokens + [str(index)])
            elif isinstance(value, str) and value in {
                "FAIL", "REJECTED", "NOT_QUALIFIED", "UNQUALIFIED",
            }:
                key = self.scalar(f"failure_{len(self.failure_rows)}", value, role,
                                  "/" + "/".join(tokens))
                self.failure_rows.append((role, key))

        for role, record in sorted(r.items()):
            if isinstance(record, dict):
                failures(record, role, [])
        self.scalar("evidence_failure_statuses", len(self.failure_rows), "metadata", "",
                    "count-failure-or-unqualified-status-in-all-pinned-json-inputs")

    def robustness(self):
        role = "public_robustness"
        evidence = self.records[role]
        require(evidence.get("numeric_verdict") in {"PASS", "UNQUALIFIED"}
                and evidence.get("original_verdict") in {"PASS", "UNQUALIFIED"}
                and evidence.get("speedup_claim") is False
                and evidence.get("tiny_dual_clipping") is False,
                "robustness_qualification_overclaim", role)
        timestamp(evidence.get("finished_at"), role)
        require(evidence.get("version") in {"public-robustness-v1", "public-robustness-v2"},
                "unknown_robustness_version", role)
        require({"public_robustness_plan", "public_robustness_corpus"} <= set(self.records),
                "robustness_plan_corpus_required", role)
        plan = self.records["public_robustness_plan"]
        corpus = self.records["public_robustness_corpus"]
        require(evidence.get("corpus_sha256") ==
                self.descriptors["public_robustness_corpus"]["sha256"],
                "robustness_corpus_pin_mismatch", role)
        arms, cases = plan.get("arms"), corpus.get("cases")
        require(isinstance(arms, list) and bool(arms) and isinstance(cases, list)
                and bool(cases) and evidence.get("arms") == arms
                and evidence.get("gates") == plan.get("gates")
                and isinstance(plan.get("gates"), dict) and bool(plan["gates"]),
                "robustness_contract_mismatch", role)
        arm_map, case_map = {}, {}
        for arm in arms:
            require(isinstance(arm, dict) and isinstance(arm.get("arm"), str)
                    and isinstance(arm.get("solver"), str) and arm["arm"] not in arm_map,
                    "robustness_plan_arm_identity", role)
            public_text(arm["arm"], role)
            arm_map[arm["arm"]] = arm["solver"]
        for case in cases:
            require(isinstance(case, dict) and isinstance(case.get("case_id"), str)
                    and case["case_id"] not in case_map
                    and isinstance(case.get("model_sha256"), str)
                    and HASH.fullmatch(case["model_sha256"]),
                    "robustness_corpus_identity", role)
            case_map[case["case_id"]] = case["model_sha256"]
        expected = {(case, arm, model, solver)
                    for case, model in case_map.items() for arm, solver in arm_map.items()}

        def membership(selected, coordinate):
            require(isinstance(selected, list) and bool(selected), coordinate, role)
            found = set()
            for row in selected:
                require(isinstance(row, dict), coordinate, role)
                identity = tuple(row.get(key) for key in
                                 ("case_id", "arm", "model_sha256", "solver"))
                require(all(isinstance(value, str) for value in identity)
                        and identity not in found and type(row.get("attempt")) is int
                        and row["attempt"] == 1 and type(row.get("denominator")) is int
                        and row["denominator"] == 1, coordinate, role)
                found.add(identity)
            require(found == expected, coordinate, role)

        membership(plan.get("rows"), "robustness_plan_incomplete")
        membership(evidence.get("rows"), "robustness_result_incomplete")
        rows, counts = evidence.get("rows"), evidence.get("arm_counts")
        require(isinstance(rows, list) and bool(rows) and isinstance(counts, dict)
                and set(counts) == set(arm_map), "empty_robustness_attempts", role)
        require(all(isinstance(row, dict) for row in rows),
                "invalid_robustness_attempt", role)
        require(integer(evidence.get("planned_denominator"), role, True) == len(rows)
                and integer(evidence.get("attempted_count"), role) ==
                sum(row.get("attempted") is True for row in rows),
                "robustness_denominator_mismatch", role)
        allowed = {"QUALIFIED", "UNQUALIFIED", "FAIL", "BUDGET_EXHAUSTED"}
        identities = set()
        self.robustness_rows = []
        for arm, entry in sorted(counts.items()):
            public_text(arm, role)
            selected = [row for row in rows if row.get("arm") == arm]
            require(selected and isinstance(entry, dict)
                    and set(entry) == allowed | {"denominator"}
                    and integer(entry["denominator"], role, True) == len(selected),
                    "robustness_arm_denominator", role)
            for row in selected:
                identity = (row.get("case_id"), arm)
                require(isinstance(identity[0], str) and identity not in identities
                        and row.get("verdict") in allowed
                        and type(row.get("attempted")) is bool
                        and (row["verdict"] not in {"QUALIFIED", "UNQUALIFIED"}
                             or row["attempted"] is True),
                        "robustness_attempt_identity", role)
                identities.add(identity)
            for value in allowed:
                require(integer(entry[value], role) ==
                        sum(row["verdict"] == value for row in selected),
                        "robustness_arm_false_count", role)
            token = arm.replace("~", "~0").replace("/", "~1")
            index = len(self.robustness_rows)
            keys = [self.direct(f"robustness_{index}_{name.lower()}", role,
                                f"/arm_counts/{token}/{name}")
                    for name in ("denominator", "QUALIFIED", "UNQUALIFIED", "FAIL",
                                 "BUDGET_EXHAUSTED")]
            self.robustness_rows.append((arm, keys))
        require(len(identities) == len(rows), "robustness_unaccounted_attempts", role)
        expected = "PASS" if all(row["verdict"] == "QUALIFIED" for row in rows) else "UNQUALIFIED"
        require(evidence["numeric_verdict"] == expected,
                "robustness_false_overall_status", role)
        self.direct("robustness_status", role, "/numeric_verdict")
        self.direct("robustness_original_status", role, "/original_verdict")
        self.direct("robustness_planned", role, "/planned_denominator")
        self.direct("robustness_attempted", role, "/attempted_count")
        self.direct("robustness_finished", role, "/finished_at")
        self.direct("robustness_version", role, "/version")
        self.scalar("robustness_settings_binding", "UNVERIFIED", role, "",
                    "renderer-does-not-execute-independent-full-settings-review")
        self.scalar("robustness_runtime_binding", "UNVERIFIED", role, "",
                    "renderer-does-not-execute-independent-runtime-review")
        require(evidence.get("license_verified") is False,
                "robustness_dataset_rights_not_reviewed", role)

    def table(self, caption, headings, rows):
        return (f'<div class="table" role="region" aria-label="{html.escape(caption)}" tabindex="0">'
                f'<table><caption>{html.escape(caption)}</caption><thead><tr>' +
                "".join(f'<th scope="col">{html.escape(h)}</th>' for h in headings) +
                "</tr></thead><tbody>" +
                "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows) +
                "</tbody></table></div>")

    def chart(self, document):
        keys = ["corpus_qualified", "corpus_rejected"]
        total = self.scalars["corpus_rows"]
        bars = []
        for index, (key, label, color) in enumerate(zip(keys, ("合格", "非合格"), ("#14564b", "#a73522"))):
            y = 48 + index * 62
            width = self.scalars[key] / total * 430
            bars.append(f'<text x="16" y="{y + 22}">{label}</text>'
                        f'<rect x="120" y="{y}" width="{width:.4f}" height="30" fill="{color}"/>'
                        f'<text x="580" y="{y + 22}">{self.show(key, document, "tspan")}</text>')
        return ('<figure><svg viewBox="0 0 720 200" role="img" aria-label="合成题资格，不是商用排名">'
                '<text x="16" y="25">同一输入逐行资格汇总</text>' + "".join(bars) +
                '</svg><figcaption>合格与非合格均来自逐行判定；普通候选 FAIL 不等于方向证据 FAIL。'
                '仅合成题已测范围，不能外推公开工业题库或商业排名。</figcaption></figure>')

    def page(self, document, design=False):
        s = lambda key: self.show(key, document)
        version = self.scalars["version"]
        family = "design" if design else "benchmark-report"
        menu = "".join(f'<a href="{family}-v{i}.html"' +
                       (' aria-current="page"' if i == version else "") +
                       f'>v{i}</a>' for i in range(1, version + 1))
        heading = "设计与已测边界" if design else "实测证据与交付边界"
        body = (
            f'<header><p class="eyebrow">OPENOPT / RELEASE EVIDENCE / v{s("version")}</p>'
            f'<h1>{heading}</h1><p class="lead">证据只支持已测 scope，不代表全面胜过商用优化器。</p></header>'
            f'<section id="conclusion"><h2>先看结论</h2><p>线上发布：{s("online_status")}；'
            f'商用资格：{s("commercial_status")}。不把接口存在、合成测试或 SDK 缺席当作商业通过。</p>'
            f'<p>CPU 回归类型：{s("regression_scope")}；源码绑定方式：{s("cpu_binding_kind")}。'
            f'CPU 实测判定 {s("cpu_status")}，CUDA 已测判定 {s("gpu_status")}；任何 FAIL 都保留，'
            '没有全域正确性或生产可靠性承诺。</p></section>'
        )
        body += (f'<p>CPU 源码一致性门：{s("cpu_source_consistency")}。缺字段的历史证据未冻结；'
                 '一致性 PASS 也不把定向 scope 扩大为整仓或商业资格。</p>')
        if "cpu_source_hash" in self.scalars:
            body += (f'<p>CPU 数值源码 commit：<code>{s("cpu_source_commit")}</code>；'
                     f'源码快照 SHA256：<code>{s("cpu_source_hash")}</code>。</p>'
                     f'<p>快照文件 {s("cpu_source_files")}；变化 {s("cpu_source_changed")}，'
                     f'缺失 {s("cpu_source_missing")}，新增 {s("cpu_source_extra")}。</p>')
        if "cpu_binding_value" in self.scalars:
            body += f'<p>CPU 版本绑定值：<code>{s("cpu_binding_value")}</code>。</p>'
        body += '<section id="verification"><h2>回归与安装包</h2>'
        body += self.table("同源回归计数，不相加去重", ["范围", "判定", "总数", "通过", "失败", "错误", "跳过"], [
            [role.upper()] + [s(f"{role}_{key}") for key in
                             ("status", "tests", "passed", "failures", "errors", "skipped")]
            for role in ("cpu", "gpu")])
        body += (f'<p>CPU coverage：{s("cpu_coverage")}；完成时间：{s("cpu_finished")}。</p>'
                 f'<p>CUDA 范围：{s("gpu_scope")}；完成时间：{s("gpu_finished")}。'
                 '此项是数值正确性检查，不是性能资格或多卡扩展实测。</p>'
                 f'<p>隔离 wheel：{s("wheel_status")}，{s("wheel_modules")} 个模块；'
                 f'项目许可 {s("wheel_license")} 不替依赖或商业 SDK 重新许可。</p>'
                 f'<p>wheel 源码一致性门：{s("wheel_source_consistency")}；缺字段不称冻结版本。</p>')
        if self.extension_rows:
            body += ("<h3>独立原模型回放门</h3><p>下表不是顶层 CUDA 测试计数，"
                     "而是执行证据与原始问题数值的独立验收。FAIL 不因 pytest 通过而升级。</p>")
            body += self.table("独立 CUDA 验收，逐项绑定不同源码",
                               ["范围", "验收", "验收时间", "测试数", "源码坐标"], [
                [label, s(keys[0]), s(keys[1]),
                 s(f"{role}_tests") if f"{role}_tests" in self.scalars else "失败验收不授予计数",
                 s(f"{role}_source") if f"{role}_source" in self.scalars else "失败记录未提供"]
                for role, label, keys in self.extension_rows])
        if "proof_review_status" in self.scalars:
            body += (f'<p>数值证明生产/回放合同审查：{s("proof_review_status")}；'
                     f'时间 {s("proof_review_finished")}；'
                     f'审查源码 <code>{s("proof_review_source")}</code>。'
                     '此审查只覆盖 CPU 生产与负例，PASS 也不授予 GPU 来源、商业或精确证明资格。</p>')
        if "wheel_hash" in self.scalars:
            body += (f'<p>wheel SHA256：<code>{s("wheel_hash")}</code>；'
                     f'安装包数值 source commit：<code>{s("wheel_source_commit")}</code>；'
                     f'源码快照 hash：<code>{s("wheel_source_hash")}</code>。</p>')
        body += ('</section>'
                 '<section id="corpus"><h2>合成题资格与反证</h2>'
                 f'<p>执行判定：{s("corpus_status")}；逐行记录：{s("corpus_rows")}。'
                 f'输入声明范围：{s("corpus_scope")}。</p>' + self.chart(document))
        rows = [[s(key) for key in keys] + [" / ".join(s(key) for key in checks)]
                for keys, checks in self.corpus_rows]
        body += '<details><summary>展开全部逐题状态和子检查，含 FAIL</summary>'
        body += self.table("逐题 qualification", ["题目", "求解器", "资格", "求解状态", "候选 / 方向检查"], rows)
        body += "</details>"
        if "public_status" in self.scalars:
            body += (f'<h3>可选公开题资格摘要</h3><p>输入判定 {s("public_status")}；'
                     f'范围 {s("public_scope")}；计划 {s("public_planned")}，'
                     f'尝试 {s("public_attempted")}，合格 {s("public_qualified")}，'
                     f'非合格 {s("public_rejected")}。仅绑定此输入声明，不扩成商业资格。</p>')
        if "robustness_status" in self.scalars:
            body += (f'<h3>公开题组合臂实验</h3><p>新臂数值资格 {s("robustness_status")}；'
                     f'实验版本 {s("robustness_version")}；'
                     f'原实验资格 {s("robustness_original_status")}；'
                     f'计划 {s("robustness_planned")}，实际尝试 {s("robustness_attempted")}；'
                     f'完成时间 {s("robustness_finished")}。'
                     '这是组合配置观察，不是单变量消融、商业排名或公平延迟对比。'
                     '数据再分发权未经确认，不发布原始或派生数组。</p>')
            body += (f'<p>完整 settings 配置绑定 {s("robustness_settings_binding")}；'
                     f'runtime 运行环境绑定 {s("robustness_runtime_binding")}。'
                     '旧 v1 缺少完整配置与环境校验，不追认为 v2 执行。'
                     '分母只验调用者独立固定的计划、题集与组合臂全集，不代替数值执行验收。</p>')
            body += self.table("公开题失败不削分母",
                               ["组合臂", "分母", "合格", "非合格", "失败", "未开始预算耗尽"],
                               [[html.escape(arm), *[s(key) for key in keys]]
                                for arm, keys in self.robustness_rows])
        body += '</section><section id="commercial"><h2>商用缺口</h2>'
        body += (f'<p>探测结果 {s("commercial_result")}；计划 {s("commercial_planned")}，'
                 f'尝试 {s("commercial_attempted")}，实际求解尝试 {s("commercial_solve_attempted")}，'
                 f'通过 {s("commercial_passed")}，失败 {s("commercial_failed")}。'
                 '没有实测不是零误差，缺 SDK 不能换算为胜出。</p>')
        body += self.table("SDK 状态原值", ["求解器", "状态", "输入原因"],
                           [[html.escape(name), s(key), s(reason) if reason else "未提供原因字段"]
                            for name, key, reason in self.commercial_rows])
        body += '</section><section id="scope"><h2>写集审计与现场快照</h2>'
        body += (f'<p>审计席数 {s("audit_seats")}，非 PASS 席数 {s("audit_failed")}。'
                 '审计只覆盖输入 campaign，不是仓库全局并发证明。</p>')
        body += self.table("职责写集核对", ["职责", "判定", "越界数量"],
                           [[s(key) for key in row] for row in self.audit_rows])
        body += (f'<p>现场判定 {s("live_status")}；采集时间 {s("live_read_at")}；'
                 f'Ready {s("live_ready")}，活动任务 {s("live_active")}，'
                 f'cordon {s("live_cordoned")}，保留失败 {s("live_retained_failures")}。'
                 '仅展示数量，不展示主机、入口或失败日志中的内部信息。</p>'
                 f'<p>全部固定 JSON 中失败状态标量数量：{s("evidence_failure_statuses")}。'
                 '这是状态出现次数，不是独立失败题数；包含预期反证，原始错误详情不外泄。</p>')
        if self.failure_rows:
            body += '<details><summary>失败状态逐项保留，完整指针留在私有回读账</summary>'
            body += self.table("所有失败状态，不按顶层 PASS 丢弃", ["证据角色", "状态"],
                               [[html.escape(role), s(key)] for role, key in self.failure_rows])
            body += "</details>"
        body += ('</section>'
                 '<section id="limits"><h2>软件限制</h2><p>实验性数值接口的支持范围不等于'
                 '成熟通用优化器。锥与整数接口必须分别按自己的问题范围验收；'
                 '近似数值方向不是精确证明，单卡 GPU 回归不是多 GPU 算法缩放。</p>'
                 '<p>公开工业题库代表性、授权商业同题对照、持续吞吐、硬件下界及部署级长期'
                 '可靠性不由本报告推导。旧版本对应历史输入，不用本轮源码重写旧测量。</p></section>'
                 '<section id="provenance"><h2>版本与证据绑定</h2>'
                 f'<p>输入声明的数值 source SHA：<code>{s("source_sha")}</code>。'
                 '缺一致性门时不能证明中间 CPU 与此版本相同；它不是文档发布 commit，'
                 '不能声称该整仓包含后来生成的文档。</p>'
                 f'<p>GPU run source SHA：<code>{s("gpu_source_sha")}</code>。'
                 '两个源码坐标分别绑定，不强行当作同一冻结版本。</p>'
                 f'<p>GPU result SHA256：<code>{s("gpu_result_hash")}</code>。</p>'
                 f'<p>GPU run SHA256：<code>{s("gpu_run_hash")}</code>。</p>'
                 '<p>每个显示值带 data-scalar 标识；外置 summary 的标量与 private report-bundle'
                 '可逐项回读。不在公共页输出私有路径、原始日志或网络入口。</p></section>')
        if "doc_source_sha" in self.scalars:
            body += f'<p>独立文档 source SHA：<code>{s("doc_source_sha")}</code>，不等同数值源码。</p>'
        else:
            body += '<p>文档 commit 由后续 publication 计划在文档实际提交后独立绑定，本页不虚构同仓冻结。</p>'
        modified = timestamp(self.manifest["modified_at"], "modified_at").astimezone(
            timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S +08:00")
        side = "".join(f'<a href="#{key}">{label}</a>' for key, label in (
            ("conclusion", "结论"), ("verification", "回归"), ("corpus", "合成题"),
            ("commercial", "商用缺口"), ("scope", "审计与快照"),
            ("limits", "限制"), ("provenance", "证据")))
        page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; base-uri 'none'; form-action 'none'">
<title>Openopt v{version} / {heading}</title><style>
:root {{ --ink:#183830; --muted:#455a51; --paper:#faf7ef; --line:#c9d1c4; --accent:#9c3824; }}
html {{ box-sizing:border-box; }} *, *::before, *::after {{ box-sizing:inherit; }}
body {{ margin:0; color:var(--ink); background:radial-gradient(ellipse at top left,#e0eadd,transparent 60%),var(--paper);
font:1.0625rem/1.7 "Noto Serif CJK SC","Source Han Serif SC",Georgia,serif; }}
a {{ color:var(--ink); overflow-wrap:anywhere; }} a:focus-visible,summary:focus-visible,.table:focus-visible {{ outline:3px solid var(--accent); outline-offset:3px; }}
.shell {{ max-inline-size:82rem; margin:auto; padding:clamp(1rem,3vw,2.5rem); }}
nav {{ display:flex; flex-wrap:wrap; gap:.5rem; }} nav a,aside a {{ display:inline-flex; align-items:center; min-height:44px; padding:.5rem .8rem; }}
nav a[aria-current] {{ background:var(--ink); color:var(--paper); }}
.layout {{ display:grid; gap:2rem; }} main,aside {{ min-inline-size:0; }}
main {{ max-inline-size:75ch; }} p {{ max-inline-size:42em; overflow-wrap:anywhere; }}
h1,h2 {{ line-height:1.2; letter-spacing:-.02em; }} h1 {{ font-size:clamp(2rem,4vw,3.8rem); }} h2 {{ font-size:1.6rem; }}
header {{ padding:2rem 0; border-bottom:3px solid var(--ink); }} .lead {{ font-size:1.3rem; }}
.eyebrow {{ font: .875rem/1.6 monospace; letter-spacing:.08em; }}
section {{ padding:1.5rem 0; border-bottom:1px solid var(--line); }} aside {{ align-self:start; }} aside a {{ display:flex; }}
.table {{ overflow:auto; margin:1rem 0; }} table {{ border-collapse:collapse; width:100%; font-size:.875rem; }}
caption {{ text-align:left; font-weight:bold; padding:.6rem; }} th,td {{ text-align:left; border-bottom:1px solid var(--line); padding:.7rem; vertical-align:top; overflow-wrap:anywhere; }}
svg {{ width:100%; height:auto; font-size:18px; }} figure {{ margin:1.5rem 0; }} figcaption,footer {{ font-size:.875rem; color:var(--muted); }}
summary {{ padding:.7rem 0; cursor:pointer; min-height:44px; }} code {{ overflow-wrap:anywhere; font-size:.875rem; }}
[data-scalar] {{ font-variant-numeric:tabular-nums; }} footer {{ padding:2rem 0; }}
@media (min-width:64rem) {{ .layout {{ grid-template-columns:minmax(0,1fr) 14rem; }} aside {{ position:sticky; top:1rem; }} }}
@media (prefers-reduced-motion:reduce) {{ * {{ scroll-behavior:auto; }} }}
</style></head><body><div class="shell">
<a href="#main">跳到主内容</a><nav aria-label="历史版本">{menu}
<a href="{'benchmark-report' if design else 'design'}-v{version}.html">{'实测报告' if design else '设计边界'}</a></nav>
<div class="layout"><main id="main" data-profile="long-form">{body}</main>
<aside aria-label="章节导航">{side}</aside></div>
<footer>最后修改：{modified}。由 scripts/render_release_report.py 从固定输入生成。</footer>
</div></body></html>
'''
        public_text(page, "html")
        StaticPage().feed(page)
        return page.encode("utf-8")

    def build(self):
        self.project()
        pages = {name: self.page(name, name.startswith("design")) for name in
                 ("benchmark-report-v4.html", "design-v4.html")}
        summary = {"schema_version": 1, "source_sha": self.manifest["source_sha"],
                   "modified_at": self.manifest["modified_at"], "scalars": self.scalars}
        pages["summary.json"] = encoded(summary)
        pages["claims.json"] = encoded({"claims": self.claims,
                                       "publication_kinds": ["number", "status"],
                                       "text_claims_require_separate_legal_or_scope_review": True})
        manifest = {"schema_version": 1, "source_sha": self.manifest["source_sha"],
                    "modified_at": self.manifest["modified_at"],
                    "files": [{"path": name, "sha256": sha(payload), "bytes": len(payload)}
                              for name, payload in sorted(pages.items())]}
        pages["manifest.json"] = encoded(manifest)
        for name, payload in pages.items():
            if name.endswith(".json"):
                public_json(payload, name)
            else:
                public_text(payload.decode("utf-8"), name)
        private = {
            "schema_version": 1, "source_sha": self.manifest["source_sha"],
            "modified_at": self.manifest["modified_at"], "public": False,
            "manifest_sha256": self.manifest_sha256,
            "inputs": sorted(self.descriptors.values(), key=lambda item: item["id"]),
            "scalar_provenance": self.provenance,
            "public_manifest_sha256": sha(pages["manifest.json"]),
            "scope": "caller-declared metadata, not independently proved release freeze",
            "numerical_source_sha": self.manifest["source_sha"],
            "doc_source_sha": self.records["metadata"].get("doc_source_sha"),
        }
        pages["report-bundle.json"] = encoded(private)
        self.unchanged()
        return pages


def generate(manifest, manifest_sha256):
    """Return a small byte bundle; callers decide any authorized repo writes."""
    return Report(manifest, manifest_sha256).build()


def prepare_inputs(output, inputs, source_sha, modified_at, doc_source_sha=None,
                   regression_scope="intermediate_snapshot", cpu_binding_kind="scope_only",
                   cpu_binding_value=None):
    """Prepare explicit pinned inputs, never infer a latest run."""
    require(isinstance(source_sha, str) and COMMIT.fullmatch(source_sha),
            "invalid_source_sha", "prepare")
    timestamp(modified_at, "prepare")
    require(regression_scope in {"intermediate_snapshot", "frozen_release"},
            "invalid_regression_scope", "prepare")
    require(cpu_binding_kind in {"scope_only", "code_hash", "version_metadata"} and
            (cpu_binding_kind == "scope_only") == (cpu_binding_value is None),
            "invalid_cpu_binding", "prepare")
    descriptors, ids, paths = [], set(), set()
    output = shared_path(output, "prepare_output")
    for name, raw, digest in inputs:
        require(name in (REQUIRED | OPTIONAL) - {"metadata"} and name not in ids,
                "invalid_or_duplicate_id", "prepare")
        path = shared_path(raw, name)
        require(path not in paths and not path.is_relative_to(output),
                "unsafe_prepare_input_location", name)
        require(isinstance(digest, str) and HASH.fullmatch(digest),
                "invalid_input_pin", name)
        require(sha(read(path, name)[0]) == digest, "changed_hash", name)
        descriptors.append({"id": name, "path": str(path), "sha256": digest})
        ids.add(name)
        paths.add(path)
    require(REQUIRED - {"metadata"} <= ids, "missing_required_input", "prepare")
    metadata = {"version": 4, "source_sha": source_sha, "modified_at": modified_at,
                "regression_scope": regression_scope,
                "cpu_binding": {"kind": cpu_binding_kind},
                "online_publication": "UNVERIFIED", "commercial_qualification": "UNVERIFIED"}
    if cpu_binding_value is not None:
        metadata["cpu_binding"]["value"] = cpu_binding_value
    if doc_source_sha is not None:
        require(isinstance(doc_source_sha, str) and COMMIT.fullmatch(doc_source_sha),
                "invalid_doc_source_sha", "prepare")
        metadata["doc_source_sha"] = doc_source_sha
    meta_payload = encoded(metadata)
    descriptors.append({"id": "metadata", "path": str(output / "metadata.json"),
                        "sha256": sha(meta_payload)})
    manifest = {"schema_version": 1, "source_sha": source_sha, "modified_at": modified_at,
                "inputs": descriptors}
    write_bundle(output, {"metadata.json": meta_payload, "input-manifest.json": encoded(manifest)})
    pin = sha(encoded(manifest))
    generate(output / "input-manifest.json", pin)
    return output / "input-manifest.json", pin


def write_bundle(output, files, check=False):
    output = shared_path(output, "output")
    require(not output.is_relative_to(ROOT.resolve()), "output_in_worktree", "output")
    if check:
        fd = directory(output)
        try:
            require(set(os.listdir(fd)) == set(files), "output_members_mismatch", "output")
        finally:
            os.close(fd)
        for name, payload in files.items():
            require(read(output / name, "readback")[0] == payload, "stale_output", "readback")
        return
    parent = directory(output.parent, create=True)
    try:
        os.mkdir(output.name, dir_fd=parent)
    except FileExistsError as error:
        raise Rejected("output_already_exists", "output") from error
    finally:
        os.close(parent)
    fd = directory(output)
    try:
        for name, payload in files.items():
            target = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o644, dir_fd=fd)
            with os.fdopen(target, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            require(read(output / name, "readback")[0] == payload, "output_readback_mismatch", "readback")
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--manifest-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prepare-inputs", action="store_true")
    parser.add_argument("--input", action="append", nargs=3, metavar=("ID", "PATH", "SHA256"))
    parser.add_argument("--source-sha")
    parser.add_argument("--doc-source-sha")
    parser.add_argument("--modified-at")
    parser.add_argument("--regression-scope", choices=("intermediate_snapshot", "frozen_release"),
                        default="intermediate_snapshot")
    parser.add_argument("--cpu-binding-kind", choices=("scope_only", "code_hash", "version_metadata"),
                        default="scope_only")
    parser.add_argument("--cpu-binding-value")
    args = parser.parse_args(argv)
    try:
        if args.prepare_inputs:
            require(not args.check and args.manifest is None and args.manifest_sha256 is None
                    and args.input and args.source_sha and args.modified_at,
                    "invalid_prepare_cli", "cli")
            path, pin = prepare_inputs(
                args.output_dir, args.input, args.source_sha, args.modified_at,
                args.doc_source_sha, args.regression_scope,
                args.cpu_binding_kind, args.cpu_binding_value)
            print(f"RELEASE_REPORT_INPUT verdict=PREPARED manifest={path} sha256={pin}")
            return 0
        require(args.manifest is not None and args.manifest_sha256 is not None
                and not args.input and args.source_sha is None and args.modified_at is None
                and args.doc_source_sha is None and args.cpu_binding_value is None,
                "invalid_render_cli", "cli")
        files = generate(args.manifest, args.manifest_sha256)
        require(files == generate(args.manifest, args.manifest_sha256), "nondeterministic_output", "render")
        output = shared_path(args.output_dir, "output")
        inputs = {shared_path(item["path"], "input")
                  for item in parse(read(shared_path(args.manifest, "manifest"), "manifest")[0],
                                    "manifest")["inputs"]}
        inputs.add(shared_path(args.manifest, "manifest"))
        require(not any(path.is_relative_to(output) for path in inputs), "output_contains_input", "output")
        write_bundle(output, files, args.check)
        require(files == generate(args.manifest, args.manifest_sha256), "snapshot_race", "final_recheck")
        print(f"RELEASE_REPORT verdict=PASS files={len(files)} deterministic=true "
              f"source_sha={parse(files['summary.json'], 'summary')['source_sha']} "
              "online=UNVERIFIED commercial=UNVERIFIED")
        return 0
    except (Rejected, OSError) as error:
        code = error.code if isinstance(error, Rejected) else "filesystem_error"
        coordinate = error.coordinate if isinstance(error, Rejected) else "filesystem"
        print(f"RELEASE_REPORT verdict=FAIL code={code} coordinate={coordinate}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
