"""Prepare and byte-verify a static public bundle, never publish it.

The caller pins a private plan, a committed documentation tree and external
JSON evidence. Only selected numeric/status scalars leave the evidence root.
See docs/research/publication.md for the schema and explicit non-claims.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tempfile
import unicodedata
from urllib.parse import unquote
import zipfile

SHARED = Path("/3fs/research3fs/data/tmp")
MAX_BYTES = 8 * 1024 * 1024
NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}\Z")
HASH = re.compile(r"[0-9a-f]{64}\Z")
STATUSES = frozenset({
    "PASS", "FAIL", "UNVERIFIED", "QUALIFIED", "REJECTED", "INCONCLUSIVE",
    "INCONCLUSIVE_NOISE", "solved", "max_iterations", "infeasible", "unbounded",
})
SENSITIVE_FIELD = re.compile(
    r"password|credential|secret|token|api.?key|private.?key|license|licence|"
    r"host|node|machine|user|endpoint|network|address|kube|url|port", re.I)
LEAK_RULES = {
    "absolute_path": re.compile(
        r"(?:^|[\s\"'>(=:])/(?!/)[A-Za-z0-9_.-]+|"
        r"[A-Za-z]:[\\/]|\\\\[A-Za-z]|/(?:data|home|root|tmp|etc|3fs|hf3fs-jd)\b"),
    "network_entry": re.compile(
        r"[a-z][a-z0-9+.-]*://|(?<!:)//[a-z0-9]|"
        r"(?:\d{1,3}\.){3}\d{1,3}|"
        # Uncompressed IPv6 has eight fields; compressed forms contain "::".
        # Three clock fields are neither. Keep conservative compression matches.
        r"(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}|"
        r"(?:[0-9a-f]{1,4}:)*[0-9a-f]{0,4}::"
        r"(?:[0-9a-f]{1,4}(?::[0-9a-f]{1,4})*)?|"
        # Also reject non-clock IPv6-like prefixes joined to other tokens.
        r"(?=[0-9a-f:]*[a-f])(?:[0-9a-f]{1,4}:){2}|"
        r"(?:[0-9a-f]{3,4}:[0-9a-f]{1,4}|"
        r"[0-9a-f]{1,4}:[0-9a-f]{3,4}):|"
        r"\b(?:localhost|[a-z0-9_-]+(?:\.[a-z0-9_-]+)*\."
        r"(?:com|org|net|io|cn|local|internal|lan))\b|[\w.+-]+@[\w.-]+", re.I),
    "host_identity": re.compile(
        r"\b(?:h3c|hw|hdp|mch|gyf|r640)\d[\w.-]*\b|"
        r"\b(?:hostname|host_name|node_name|machine_id)\s*[:=]", re.I),
    "credential": re.compile(
        r"\b(?:password|passwd|secret|token|api[_-]?key|access[_-]?key|"
        r"client[_-]?secret|authorization)\b[\"']?\s*[:=]|"
        r"\bBearer\s+\S+|-----BEGIN [A-Z ]*PRIVATE KEY-----|"
        r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|"
        r"\b(?:gh[pousr]_|sk-)[A-Za-z0-9_-]{16,}|"
        r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", re.I),
    "license_material": re.compile(
        r"\b(?:license|licence)[ _-]*(?:server|key|file)\b|"
        r"\b(?:LM_LICENSE_FILE|GRB_LICENSE_FILE|MOSEKLM_LICENSE_FILE)\b|"
        r"^\s*(?:FEATURE|INCREMENT|SERVER|VENDOR|DAEMON)\s+\S+", re.I | re.M),
    "active_content": re.compile(
        r"<\s*(?:script|iframe|object|embed|base)\b|"
        r"\bon[a-z]+\s*=|javascript\s*:|data\s*:|@import\b", re.I),
}


class Rejected(ValueError):
    """Safe first-cause code and coordinate; never echo rejected payloads."""

    def __init__(self, code: str, coordinate: str):
        self.code, self.coordinate = code, coordinate
        super().__init__(f"{code} at {coordinate}")


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def encoded(value) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=True,
                       allow_nan=False) + "\n").encode()


def require(condition, code, coordinate):
    if not condition:
        raise Rejected(code, coordinate)


def object_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate_json_key", "json")
        result[key] = value
    return result


def parse(payload, coordinate):
    require(bool(payload.strip()), "empty_content", coordinate)
    try:
        return json.loads(payload.decode("utf-8"), object_pairs_hook=object_pairs,
                          parse_constant=lambda _: reject_nonfinite(coordinate))
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise Rejected("invalid_json", coordinate) from error


def reject_nonfinite(coordinate):
    raise Rejected("nonfinite_json", coordinate)


def keys(value, expected, coordinate):
    require(isinstance(value, dict) and set(value) == set(expected),
            "schema_keys", coordinate)


def relative(value, coordinate):
    require(isinstance(value, str) and bool(value) and "\\" not in value
            and not any(c.isspace() or ord(c) < 32 for c in value),
            "invalid_path", coordinate)
    path = PurePosixPath(value)
    require(not path.is_absolute() and all(p not in ("", ".", "..") for p in value.split("/"))
            and ":" not in value, "path_escape", coordinate)
    return path


def absolute(value, coordinate):
    raw = os.fspath(value)
    require(raw.startswith("/") and ".." not in raw.split("/")
            and "\\" not in raw and not any(ord(c) < 32 for c in raw),
            "invalid_absolute_path", coordinate)
    # This is the established storage mount alias, not a caller-chosen symlink.
    if raw == "/hf3fs-jd" or raw.startswith("/hf3fs-jd/"):
        raw = "/3fs/research3fs" + raw[len("/hf3fs-jd"):]
    return Path(raw)


def directory(path: Path, create=False):
    """Anchor every component with O_NOFOLLOW, including existing parents."""
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


@dataclass(frozen=True)
class Snapshot:
    payload: bytes
    identity: tuple


def read(path: Path, coordinate) -> Snapshot:
    parent = directory(path.parent)
    try:
        fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                     dir_fd=parent)
        with os.fdopen(fd, "rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode), "not_regular_file", coordinate)
            require(before.st_size <= MAX_BYTES, "input_too_large", coordinate)
            payload = stream.read(MAX_BYTES + 1)
            after = os.fstat(stream.fileno())
        identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        require(identity(before) == identity(after), "changed_input_during_read", coordinate)
        require(len(payload) <= MAX_BYTES, "input_too_large", coordinate)
        require(bool(payload.strip()), "empty_content", coordinate)
        return Snapshot(payload, identity(after))
    except OSError as error:
        raise Rejected("unsafe_or_missing_file", coordinate) from error
    finally:
        os.close(parent)


def scan(payload: bytes, coordinate):
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise Rejected("non_utf8_content", coordinate) from error
    require(bool(text.strip()), "empty_content", coordinate)
    require(not any((ord(c) < 32 and c not in "\n\r\t") or
                    unicodedata.category(c) == "Cf" for c in text),
            "hidden_character", coordinate)
    for _ in range(4):
        decoded = html.unescape(unquote(text))
        decoded = re.sub(r"\\u([0-9a-fA-F]{4})",
                         lambda m: chr(int(m[1], 16)), decoded)
        decoded = decoded.replace("\\/", "/")
        if decoded == text:
            break
        text = decoded
    else:
        raise Rejected("excessive_encoding", coordinate)
    text = unicodedata.normalize("NFKC", text)
    for code in ("network_entry", *[c for c in LEAK_RULES if c != "network_entry"]):
        pattern = LEAK_RULES[code]
        require(not pattern.search(text), code, coordinate)


def git(repo, *arguments):
    result = subprocess.run(["git", "-C", str(repo), *arguments],
                            capture_output=True, check=False)
    require(result.returncode == 0, f"git_failed_rc_{result.returncode}", "source")
    return result.stdout


def committed(repo, commit, path):
    item = relative(path, "source_path").as_posix()
    listing = git(repo, "ls-tree", "-z", commit, "--", item)
    require(bool(listing) and listing.count(b"\0") == 1, "uncommitted_source", "source")
    require(listing.split(b" ", 1)[0] in (b"100644", b"100755"),
            "source_symlink_or_nonfile", "source")
    snapshot = read(repo / item, "source")
    require(snapshot.payload == git(repo, "cat-file", "blob", f"{commit}:{item}"),
            "uncommitted_source", "source")
    return snapshot.payload


def pointer(record, value):
    require(isinstance(value, str) and value.startswith("/") and len(value) <= 256,
            "invalid_pointer", "claim")
    for part in value[1:].split("/"):
        require(not re.search(r"~(?![01])", part), "invalid_pointer", "claim")
        part = part.replace("~1", "/").replace("~0", "~")
        scan(part.encode(), "claim_pointer_token")
        require(not SENSITIVE_FIELD.search(part), "sensitive_field", "claim")
        if isinstance(record, dict):
            require(part in record, "missing_field", "claim")
            record = record[part]
        elif isinstance(record, list):
            require(bool(re.fullmatch(r"0|[1-9][0-9]*", part)) and int(part) < len(record),
                    "missing_field", "claim")
            record = record[int(part)]
        else:
            raise Rejected("missing_field", "claim")
    return record


def source_clean(repo, commit):
    fd = directory(repo)
    os.close(fd)
    require(git(repo, "rev-parse", "HEAD").decode().strip() == commit,
            "source_commit_mismatch", "source")
    require(not git(repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"),
            "uncommitted_source", "source")


def zip_bytes(files):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(files.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
    return output.getvalue()


class Preparation:
    def __init__(self, repo, evidence_root, plan, plan_sha256):
        self.repo = absolute(repo, "repo")
        self.evidence_root = absolute(evidence_root, "evidence_root")
        self.plan_path = absolute(plan, "plan")
        require(self.evidence_root.is_relative_to(SHARED)
                and self.evidence_root != SHARED
                and not self.evidence_root.is_relative_to(self.repo),
                "evidence_must_be_external_shared_storage", "evidence_root")
        require(self.plan_path.is_relative_to(self.evidence_root),
                "plan_outside_evidence_root", "plan")
        require(isinstance(plan_sha256, str) and HASH.fullmatch(plan_sha256),
                "invalid_plan_pin", "plan")
        self.snapshots = {}
        payload = self.input(self.plan_path, plan_sha256, "plan")
        self.plan = parse(payload, "plan")
        keys(self.plan, ("schema_version", "source_commit", "docs_version",
                         "documents", "evidence", "claims"), "plan")
        require(type(self.plan["schema_version"]) is int and self.plan["schema_version"] == 1,
                "schema_version", "plan")
        commit, version = self.plan["source_commit"], self.plan["docs_version"]
        require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit),
                "invalid_source_commit", "plan")
        require(isinstance(version, str) and re.fullmatch(r"v[1-9][0-9]*", version),
                "invalid_docs_version", "plan")
        for name in ("documents", "evidence", "claims"):
            require(isinstance(self.plan[name], list) and bool(self.plan[name]),
                    "empty_selection", name)
        source_clean(self.repo, commit)

    def input(self, path, digest, coordinate):
        require(isinstance(digest, str) and HASH.fullmatch(digest),
                "invalid_input_pin", coordinate)
        current = read(path, coordinate)
        require(sha(current.payload) == digest, "changed_evidence", coordinate)
        if path in self.snapshots:
            require(self.snapshots[path] == current, "changed_evidence", coordinate)
        self.snapshots[path] = current
        return current.payload

    def unchanged(self):
        source_clean(self.repo, self.plan["source_commit"])
        for path, snapshot in self.snapshots.items():
            require(read(path, "input_recheck") == snapshot,
                    "changed_evidence", "input_recheck")

    def build(self):
        plan, evidence, source_files = self.plan, {}, {}
        for row in plan["evidence"]:
            keys(row, ("id", "path", "sha256", "producer"), "evidence")
            name = row["id"]
            require(isinstance(name, str) and NAME.fullmatch(name) and name not in evidence,
                    "invalid_or_duplicate_id", "evidence")
            path = relative(row["path"], "evidence_path")
            producer = relative(row["producer"], "producer").as_posix()
            require(producer.startswith("scripts/") and producer.endswith(".py"),
                    "invalid_producer", "evidence")
            scan(producer.encode(), "producer")
            source_files[producer] = sha(committed(self.repo, plan["source_commit"], producer))
            evidence[name] = (
                parse(self.input(self.evidence_root / path, row["sha256"], "evidence"), "evidence"),
                row,
            )
        values, claims, references = {}, [], []
        for row in plan["claims"]:
            keys(row, ("id", "evidence", "pointer", "kind", "unit",
                       "document", "text"), "claim")
            name = row["id"]
            require(isinstance(name, str) and NAME.fullmatch(name) and name not in values,
                    "invalid_or_duplicate_id", "claim")
            require(isinstance(row["evidence"], str) and row["evidence"] in evidence,
                    "missing_evidence", "claim")
            record, origin = evidence[row["evidence"]]
            value = pointer(record, row["pointer"])
            if row["kind"] == "number":
                require(type(value) in (int, float)
                        and (type(value) is int or math.isfinite(value)),
                        "not_finite_number", "claim")
                require(row["unit"] in ("s", "ms", "count", "ratio", "percent", "bytes", "MiB"),
                        "invalid_unit", "claim")
                require(isinstance(row["text"], str) and bool(re.fullmatch(
                    r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?", row["text"])),
                    "invalid_numeric_text", "claim")
                try:
                    require(Decimal(row["text"]) == Decimal(str(value)),
                            "claim_text_value_mismatch", "claim")
                except InvalidOperation as error:
                    raise Rejected("invalid_numeric_text", "claim") from error
            else:
                require(row["kind"] == "status" and isinstance(value, str)
                        and value in STATUSES and row["unit"] == "",
                        "invalid_status", "claim")
                require(row["text"] == value, "claim_text_value_mismatch", "claim")
            document = relative(row["document"], "claim_document").as_posix()
            references.append((document, row["text"], row["kind"]))
            values[name] = str(value)
            claims.append({
                "id": name, "kind": row["kind"], "unit": row["unit"], "value": value,
                "document": document, "text": row["text"],
                "origin": {"evidence_id": row["evidence"], "evidence_sha256": origin["sha256"],
                           "json_pointer_tokens": [p.replace("~1", "/").replace("~0", "~")
                                                   for p in row["pointer"][1:].split("/")],
                           "declared_producer": origin["producer"],
                           "producer_sha256": source_files[origin["producer"]]},
            })
        files = {}
        for row in plan["documents"]:
            keys(row, ("path", "sha256"), "document")
            path = relative(row["path"], "document_path")
            name = path.as_posix()
            require(name.startswith("docs/") and path.suffix in (".md", ".html", ".css", ".svg"),
                    "invalid_document_type", "document")
            if path.suffix == ".html":
                require(path.stem.endswith("-" + plan["docs_version"]),
                        "document_version_mismatch", "document")
            require(name not in files, "duplicate_document", "document")
            scan(name.encode(), "document_name")
            payload = committed(self.repo, plan["source_commit"], name)
            require(isinstance(row["sha256"], str) and sha(payload) == row["sha256"],
                    "document_hash_mismatch", "document")
            source_files[name] = sha(payload)
            scan(payload, "document_source")
            files[name] = payload
        for document, text, kind in references:
            require(document in files, "claim_document_not_selected", "claim")
            boundary = r"(?!\w|\.[0-9])" if kind == "number" else r"(?!\w)"
            require(bool(re.search(r"(?<![\w.])" + re.escape(text) + boundary,
                                   html.unescape(files[document].decode()))),
                    "claim_text_not_in_document", "claim")
        require(set(files) == {document for document, _, _ in references},
                "document_without_provenance", "documents")
        require({c["origin"]["evidence_id"] for c in claims} == set(evidence),
                "unused_evidence", "claims")
        files["public-evidence.json"] = encoded({
            "schema_version": 1, "claims": sorted(claims, key=lambda c: c["id"]),
            "raw_evidence_included": False,
        })
        version = {
            "schema_version": 1, "docs_version": plan["docs_version"],
            "source_commit": plan["source_commit"], "state": "PREPARED",
            "online_publication": "UNVERIFIED", "commercial_qualification": "UNVERIFIED",
        }
        files["version.json"] = encoded(version)
        for name, payload in files.items():
            scan(payload, "public_content")
        manifest = {
            **version, "raw_evidence_included": False,
            "source_files": dict(sorted(source_files.items())),
            "files": [{"path": name, "sha256": sha(payload), "bytes": len(payload)}
                      for name, payload in sorted(files.items())],
        }
        files["manifest.json"] = encoded(manifest)
        scan(files["manifest.json"], "manifest")
        self.unchanged()
        require(sum(map(len, files.values())) <= 64 * MAX_BYTES, "bundle_too_large", "bundle")
        archive = zip_bytes(files)
        return {
            "bundle.zip": archive, "manifest.json": files["manifest.json"],
            "version.json": files["version.json"],
            "SHA256SUMS": "".join(f"{sha(payload)}  {name}\n" for name, payload in sorted({
                "bundle.zip": archive, "manifest.json": files["manifest.json"],
                "version.json": files["version.json"],
            }.items())).encode(),
        }

    def verify(self, output, bundle_sha256):
        output = absolute(output, "output")
        self.output_location(output)
        require(isinstance(bundle_sha256, str) and HASH.fullmatch(bundle_sha256),
                "invalid_bundle_pin", "bundle")
        expected = self.build()
        require(sha(expected["bundle.zip"]) == bundle_sha256, "bundle_pin_mismatch", "bundle")
        fd = directory(output)
        try:
            require(set(os.listdir(fd)) == set(expected) | {"local-verification.json"},
                    "unexpected_output_members", "output")
        finally:
            os.close(fd)
        for name, payload in expected.items():
            require(read(output / name, "bundle_readback").payload == payload,
                    "bundle_bytes_mismatch", "bundle_readback")
        acceptance = self.acceptance(bundle_sha256)
        require(read(output / "local-verification.json", "local_verification").payload == encoded(acceptance),
                "local_verification_mismatch", "output")
        self.unchanged()
        return acceptance

    def acceptance(self, digest):
        return {"schema_version": 1, "verdict": "PREPARED", "bundle_sha256": digest,
                "byte_readback": "PASS", "deterministic": True,
                "online_publication": "UNVERIFIED", "commercial_qualification": "UNVERIFIED"}

    def prepare(self, output):
        output = absolute(output, "output")
        self.output_location(output)
        files = self.build()
        require(files == self.build(), "nondeterministic_bundle", "bundle")
        parent = directory(output.parent, create=True)
        try:
            os.mkdir(output.name, dir_fd=parent)
        except FileExistsError as error:
            raise Rejected("output_already_exists", "output") from error
        except OSError as error:
            raise Rejected("output_create_failed", "output") from error
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
                require(read(output / name, "bundle_readback").payload == payload,
                        "bundle_bytes_mismatch", "bundle_readback")
            require(files == self.build(), "changed_evidence", "final_recheck")
            digest = sha(files["bundle.zip"])
            target = os.open("local-verification.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                             0o644, dir_fd=fd)
            with os.fdopen(target, "wb") as stream:
                stream.write(encoded(self.acceptance(digest)))
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)
        return self.verify(output, digest)

    def output_location(self, output):
        require(output.is_relative_to(SHARED) and output != SHARED
                and not output.is_relative_to(self.repo)
                and not output.is_relative_to(self.evidence_root)
                and not self.evidence_root.is_relative_to(output),
                "unsafe_output_location", "output")


def audit_log(root, args, report):
    root = absolute(root, "log_dir")
    require(root.is_relative_to(SHARED) and root != SHARED,
            "logs_must_be_external_shared_storage", "log_dir")
    repo, output = absolute(args.repo, "repo"), absolute(args.output, "output")
    require(not root.is_relative_to(repo) and not root.is_relative_to(output),
            "unsafe_log_location", "log_dir")
    fd = directory(root, create=True)
    os.close(fd)
    path = Path(tempfile.mkdtemp(prefix="audit-", dir=root)) / "audit.json"
    saved = {**report, "action": args.action,
             "read_at": datetime.now(timezone.utc).isoformat()}
    with path.open("xb") as stream:
        stream.write(encoded(saved))
    require(read(path, "audit_log").payload == encoded(saved),
            "audit_readback_mismatch", "log_dir")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--log-dir", type=Path,
                        default=SHARED / "openopt-release-20261010/publication/logs")
    args = parser.parse_args(argv)
    try:
        preparation = Preparation(args.repo, args.evidence_root, args.plan, args.plan_sha256)
        if args.action == "prepare":
            require(args.bundle_sha256 is None, "prepare_does_not_take_bundle_pin", "cli")
            report = preparation.prepare(args.output)
        else:
            report = preparation.verify(args.output, args.bundle_sha256)
        path = audit_log(args.log_dir, args, {**report, "exit_code": 0})
        print(f"PUBLICATION verdict=PREPARED byte_readback=PASS "
              f"bundle_sha256={report['bundle_sha256']} online=UNVERIFIED commercial=UNVERIFIED "
              f"audit={path}")
        return 0
    except Rejected as error:
        audit_log(args.log_dir, args, {"verdict": "FAIL", "exit_code": 1,
                                      "first_cause": error.code, "coordinate": error.coordinate})
        print(f"PUBLICATION verdict=FAIL code={error.code} coordinate={error.coordinate}", file=sys.stderr)
        return 1
    except OSError as error:
        audit_log(args.log_dir, args, {"verdict": "FAIL", "exit_code": 1,
                                      "first_cause": "filesystem_error", "errno": error.errno})
        print(f"PUBLICATION verdict=FAIL code=filesystem_error errno={error.errno}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
