"""Read every outgoing Git object before a separately authorized research push.

No fetch, push, ref mutation, credential echo or publication is performed.
Known local credential material stays in memory. This is a bounded content
scan, not proof of all redistribution rights or absence of unknown secrets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCANNER = Path("/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SHAPES = (
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"\b(?:gh[pousr]_|sk-)[A-Za-z0-9_-]{24,}\b"),
    re.compile(rb"(?i)\b(?:password|passwd|secret|token|api_key)\s*[\"']?\s*[:=]"
               rb"\s*[\"'][^\"'\s]{8,}[\"']"),
)


def git(repo, *args):
    result = subprocess.run(
        ["git", "--no-replace-objects", *args], cwd=repo,
        capture_output=True, check=False,
    )
    if result.returncode:
        raise ValueError(f"trusted Git query failed: {args[0]} rc={result.returncode}")
    return result.stdout


def content_hits(payload, needles):
    """Return counts only, including binary content without decoding loss."""
    if not needles or any(not isinstance(value, bytes) or not value for value in needles):
        raise ValueError("nonempty byte needles required")
    return {
        "literal": sum(payload.count(value) for value in needles),
        "shape": sum(len(pattern.findall(payload)) for pattern in SHAPES),
    }


def inspect_history(repo, base, head, needles):
    if git(repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise ValueError("complete non-shallow history required")
    grafts = Path(git(repo, "rev-parse", "--git-path", "info/grafts").decode().strip())
    if not grafts.is_absolute():
        grafts = Path(repo) / grafts
    if grafts.exists():
        raise ValueError("legacy grafted history is not a raw object graph")
    for value in (base, head):
        if not isinstance(value, str) or not COMMIT.fullmatch(value):
            raise ValueError("full commit identity required")
        if git(repo, "cat-file", "-t", value).strip() != b"commit":
            raise ValueError("commit object required")
    git(repo, "merge-base", "--is-ancestor", base, head)
    commits = git(repo, "rev-list", base + ".." + head).decode("ascii").splitlines()
    if not commits:
        raise ValueError("empty outgoing history")
    object_lines = git(repo, "rev-list", "--objects", base + ".." + head).splitlines()
    identifiers = sorted({line.split(b" ", 1)[0].decode("ascii") for line in object_lines})
    if not identifiers:
        raise ValueError("empty outgoing objects")
    inventory, findings = {}, []
    for identity in identifiers:
        kind = git(repo, "cat-file", "-t", identity).decode("ascii").strip()
        if kind not in ("blob", "commit", "tree"):
            raise ValueError("unexpected outgoing object type")
        payload = git(repo, "cat-file", kind, identity)
        if len(payload) > 16 * 1024 * 1024:
            raise ValueError("outgoing object exceeds bounded scan size")
        inventory[identity] = {
            "type": kind, "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        hits = content_hits(payload, needles)
        if any(hits.values()):
            findings.append({"object": identity, "type": kind, **hits})
    return {
        "base": base, "head": head, "commits": len(commits),
        "objects": len(inventory), "blobs": sum(v["type"] == "blob" for v in inventory.values()),
        "bytes": sum(v["bytes"] for v in inventory.values()),
        "inventory": inventory, "findings": findings,
        "verdict": "PASS" if not findings else "FAIL",
    }


def source_bytes(repo, head):
    """Compare files with committed blobs even when index flags hide changes."""
    names = git(repo, "ls-tree", "-rz", "--full-tree", head).split(b"\0")
    hashes = {}
    for record in names:
        if not record:
            continue
        metadata, raw_name = record.split(b"\t", 1)
        mode, kind, identity = metadata.split()
        if kind != b"blob" or mode not in (b"100644", b"100755"):
            raise ValueError("ordinary source blobs required")
        name = raw_name.decode("utf-8")
        path = Path(repo) / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("current source file unavailable or linked")
        if not path.resolve().is_relative_to(Path(repo).resolve()):
            raise ValueError("current source file escaped repository")
        committed = git(repo, "cat-file", "blob", identity.decode("ascii"))
        actual = path.read_bytes()
        if actual != committed:
            raise ValueError("current source bytes differ from committed blobs")
        hashes[name] = hashlib.sha256(actual).hexdigest()
    if not hashes:
        raise ValueError("empty source inventory")
    return hashes


def load_needles():
    check = subprocess.run(["python3", "-B", str(SCANNER), "--selftest"],
                           capture_output=True, check=False)
    if check.returncode or b"verdict=PASS" not in check.stdout:
        raise ValueError("existing credential scanner selftest failed; output withheld")
    spec = importlib.util.spec_from_file_location("_delivery_scanner", SCANNER)
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    values = scanner.read_needles(argparse.Namespace(
        secret_hex=None, secrets_file=None, no_needle=False,
        needle_from=str(Path.home() / ".config/dsh-n600-token")))
    if not values:
        raise ValueError("existing local credential scan material unavailable")
    return [value.encode("utf-8") for value in values]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--base")
    parser.add_argument("--head")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    parent = args.artifact_dir.resolve()
    if not parent.is_relative_to(Path("/3fs/research3fs/data/tmp")) or parent.is_relative_to(repo):
        raise ValueError("external shared artifact directory required")
    parent.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="git-audit-", dir=parent))
    receipt = {
        "verdict": "RUNNING", "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "outgoing-object credential/content scan only",
        "push_performed": False, "online": "UNVERIFIED",
        "unknown_secrets": "UNVERIFIED", "redistribution_rights": "UNVERIFIED",
    }
    try:
        if args.scan_only:
            paths = (
                ROOT / "scripts/audit_git_delivery.py",
                ROOT / "tests/test_audit_git_delivery.py",
                ROOT / "docs/research/git-delivery.md",
            )
            needles = load_needles()
            counts = [content_hits(path.read_bytes(), needles) for path in paths]
            receipt.update(
                verdict="FAIL" if any(any(item.values()) for item in counts) else "PASS",
                scope="three explicitly owned source files only",
                objects=len(paths),
                findings=[
                    {"path": str(path), **count}
                    for path, count in zip(paths, counts) if any(count.values())
                ],
                current_source_sha256={
                    str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in paths
                },
            )
        else:
            receipt.update(audit_source(repo, args.base, args.head))
    except Exception as error:
        receipt.update(verdict="FAIL", first_cause=type(error).__name__,
                       reason=str(error))
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    path = output / "audit.json"
    path.write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + "\n")
    line = (f"GIT_DELIVERY_AUDIT verdict={receipt['verdict']} "
            f"objects={receipt.get('objects', 'NA')} "
            f"findings={len(receipt.get('findings', []))} receipt={path}")
    (output / "execution.log").write_text(line + "\n")
    print(line)
    return 0 if receipt["verdict"] == "PASS" else 1


def audit_source(repo, base, head):
    if not isinstance(base, str) or not isinstance(head, str):
        raise ValueError("explicit base and head required")
    repo = repo.resolve(strict=True)
    before = git(repo, "rev-parse", "HEAD").decode("ascii").strip()
    if before != head or git(repo, "status", "--porcelain").strip():
        raise ValueError("clean caller-pinned source HEAD required")
    source_before = source_bytes(repo, head)
    result = inspect_history(repo, base, head, load_needles())
    if git(repo, "rev-parse", "HEAD").decode("ascii").strip() != before:
        raise ValueError("source HEAD changed during audit")
    if git(repo, "status", "--porcelain").strip():
        raise ValueError("source changed during audit")
    if source_bytes(repo, head) != source_before:
        raise ValueError("current source bytes changed during audit")
    result["current_source_sha256"] = source_before
    result["source_consistency"] = "PASS"
    return result


if __name__ == "__main__":
    raise SystemExit(main())
