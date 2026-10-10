"""Locate pinned outgoing shape findings without revealing matched bytes.

This is read-only diagnosis, not credential adjudication, push permission or a
replacement for the complete outgoing-object scanner.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

from scripts.audit_git_delivery import SHAPES


class DiagnosisError(ValueError):
    """Controlled, non-content failure coordinates safe for the receipt."""

    def __init__(self, reason, returncode=None):
        super().__init__(reason)
        self.reason = reason
        self.returncode = returncode


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def git(root, *args):
    env = {name: value for name, value in os.environ.items()
           if not name.startswith("GIT_")}
    env.update(GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1",
               GIT_OPTIONAL_LOCKS="0")
    result = subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        env=env, capture_output=True, check=False,
    )
    if result.returncode:
        raise DiagnosisError("raw_git_query_failed", result.returncode)
    return result.stdout


def locations(payload):
    """Only coordinates leave this function; never include matched text."""
    try:
        tree = ast.parse(payload)
    except (SyntaxError, UnicodeError, ValueError):
        tree = None
    records = []
    for number, pattern in enumerate(SHAPES):
        for match in pattern.finditer(payload):
            line = payload[:match.start()].count(b"\n") + 1
            enclosing = []
            if tree is not None:
                enclosing = [
                    node for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.lineno <= line <= node.end_lineno
                ]
            smallest = min(enclosing, key=lambda n: n.end_lineno - n.lineno,
                           default=None)
            records.append({
                "pattern_index": number, "line": line,
                "node_kind": type(smallest).__name__ if smallest else "module_or_nonpython",
                "enclosing_test": bool(smallest and smallest.name.startswith("test_")),
                "classification": "UNADJUDICATED",
            })
    return records


def inspect(root, audit, pin):
    payload = audit.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", pin) or digest(payload) != pin:
        raise DiagnosisError("audit_pin_mismatch")
    report = json.loads(payload)
    for name in ("base", "head"):
        if not re.fullmatch(r"[0-9a-f]{40}", report[name]):
            raise DiagnosisError("invalid_commit")
    paths = {}
    commits = git(root, "rev-list", report["head"], "^" + report["base"]).splitlines()
    if not commits or not report["findings"]:
        raise DiagnosisError("empty_diagnosis")
    for commit in commits:
        for entry in git(root, "ls-tree", "-rz", commit.decode()).split(b"\0"):
            if entry:
                metadata, name = entry.split(b"\t", 1)
                mode, kind, identity = metadata.split()
                if kind == b"blob" and mode in (b"100644", b"100755"):
                    paths.setdefault(identity.decode(), set()).add(name.decode())
    findings = []
    for finding in report["findings"]:
        identity = finding["object"]
        if not re.fullmatch(r"[0-9a-f]{40}", identity) or finding["type"] != "blob":
            raise DiagnosisError("unsupported_finding")
        blob = git(root, "cat-file", "blob", identity)
        raw_id = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
        if raw_id != identity or digest(blob) != report["inventory"][identity]["sha256"]:
            raise DiagnosisError("raw_blob_mismatch")
        hits = locations(blob)
        if len(hits) != finding["shape"] or identity not in paths:
            raise DiagnosisError("finding_count_or_path_mismatch")
        findings.append({
            "object": identity, "paths": sorted(paths[identity]),
            "literal_count": finding["literal"], "shape_locations": hits,
        })
    return {
        "verdict": "LOCATED", "release_gate": "BLOCKED",
        "audit_sha256": pin, "base": report["base"], "head": report["head"],
        "findings": findings, "read_at": datetime.now(timezone.utc).isoformat(),
        "push_performed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--audit-sha256", required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    if not root.is_relative_to(Path("/3fs/research3fs/data/tmp")):
        raise ValueError("external_shared_artifacts_required")
    root.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="delivery-diagnosis-", dir=root))
    report, code = {"verdict": "FAIL", "release_gate": "BLOCKED"}, 1
    try:
        report = inspect(args.repo, args.audit, args.audit_sha256)
        code = 0
    except Exception as error:
        report["first_cause_type"] = type(error).__name__
        report["first_cause"] = (
            error.reason if isinstance(error, DiagnosisError)
            else "unexpected_diagnosis_error_content_withheld"
        )
        report["returncode"] = (
            error.returncode if isinstance(error, DiagnosisError) else None
        )
    path = out / "diagnosis.json"
    with path.open("x", encoding="ascii") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(f"DELIVERY_DIAGNOSIS verdict={report['verdict']} release_gate=BLOCKED report={path}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
