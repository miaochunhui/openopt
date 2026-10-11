"""Read-only local/GitLab/Gitee ref receipt, never a three-host completion claim."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REMOTES = {
    "origin": "ssh://git@192.168.1.54/infra/openopt.git",
    "gitee": "git@gitee.com:sdfwefqaa/openopt.git",
    "github": "https://github.com/miaochunhui/openopt.git",
}
GITLAB_SSH = (
    "ssh -i /home/infra/.ssh/id_ed25519_gitlab -o IdentitiesOnly=yes "
    "-o BatchMode=yes -o ConnectTimeout=10 -o HostKeyAlias=192.168.1.181"
)
OID = re.compile(r"[0-9a-f]{40}\Z")
OWNED = (
    "scripts/inspect_repository_refs.py",
    "tests/test_inspect_repository_refs.py",
    "docs/repository-maintenance.md",
    "docs/handoff/release4-20261011-1923.md",
    "docs/handoff/release4-20261011.md",
)


class InspectionError(ValueError):
    """Only fixed non-content error codes leave the subprocess boundary."""


def require(ok, code):
    if not ok:
        raise InspectionError(code)


def query(repo, args, *, remote=None, runner=subprocess.run):
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", GIT_OPTIONAL_LOCKS="0",
               GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1")
    if remote == "origin":
        env["GIT_SSH_COMMAND"] = GITLAB_SSH
    if remote == "gitee":
        env["GIT_SSH_COMMAND"] = "ssh -o BatchMode=yes -o ConnectTimeout=10"
    try:
        result = runner(
            ["git", "--no-replace-objects", "-C", str(repo), *args],
            env=env, capture_output=True, check=False, timeout=30,
        )
    except subprocess.TimeoutExpired:
        raise InspectionError("git_query_timeout") from None
    except OSError:
        raise InspectionError("git_query_unavailable") from None
    require(result.returncode == 0, "git_query_failed")
    require(type(result.stdout) is bytes and len(result.stdout) <= 65536,
            "git_response_invalid")
    return result.stdout


def head_value(payload):
    require(type(payload) is bytes, "head_response_invalid")
    try:
        value = payload.decode("ascii").strip()
    except UnicodeError:
        raise InspectionError("head_response_invalid") from None
    require(bool(OID.fullmatch(value)), "head_response_invalid")
    return value


def remote_head(payload):
    require(type(payload) is bytes, "remote_response_invalid")
    rows = payload.splitlines()
    require(len(rows) == 1, "remote_branch_missing_or_ambiguous")
    fields = rows[0].split(b"\t")
    require(len(fields) == 2 and fields[1] == b"refs/heads/master",
            "remote_branch_invalid")
    return head_value(fields[0])


def inspect(repo, expected, *, runner=subprocess.run):
    require(type(expected) is str and bool(OID.fullmatch(expected)),
            "explicit_full_head_required")
    repo = Path(repo).resolve(strict=True)
    before = head_value(query(repo, ["rev-parse", "HEAD"], runner=runner))
    require(before == expected, "local_head_mismatch")
    require(not query(repo, ["status", "--porcelain=v1", "--untracked-files=all"],
                      runner=runner), "dirty_source")
    require(query(repo, ["config", "--get", "branch.master.remote"],
                  runner=runner).strip() == b"origin", "tracking_remote_mismatch")
    require(query(repo, ["config", "--get", "branch.master.merge"],
                  runner=runner).strip() == b"refs/heads/master",
            "tracking_branch_mismatch")
    for name, url in REMOTES.items():
        for direction in ([], ["--push"]):
            actual = query(repo, ["remote", "get-url", *direction, "--all", name],
                           runner=runner).splitlines()
            require(actual == [url.encode("ascii")], "remote_configuration_mismatch")
    refs = {}
    for name in ("origin", "gitee"):
        identity = remote_head(query(
            repo, ["ls-remote", name, "refs/heads/master"], remote=name, runner=runner,
        ))
        refs[name] = {"head": identity, "matches_expected": identity == expected}
    require(head_value(query(repo, ["rev-parse", "HEAD"], runner=runner)) == before,
            "local_head_changed")
    require(not query(repo, ["status", "--porcelain=v1", "--untracked-files=all"],
                      runner=runner), "source_changed")
    matched = all(row["matches_expected"] for row in refs.values())
    return {
        "verdict": "PRIMARY_AND_GITEE_PASS" if matched else "REF_MISMATCH",
        "expected_head": expected,
        "local_head": before,
        "remotes": refs,
        "github": {
            "repository": "miaochunhui/openopt",
            "state": "NOT_VERIFIED",
            "credential_read": False,
            "network_attempted": False,
        },
        "three_host_complete": False,
        "ha_failover": "NOT_TESTED",
        "automatic_mirror": "NOT_INSTALLED_BY_THIS_TOOL",
    }


def scan_owned(repo):
    """Reuse the existing content scanner without emitting needles or raw output."""
    from scripts.audit_git_delivery import content_hits, load_needles

    needles = load_needles()
    rows = {}
    for name in OWNED:
        path = repo / name
        require(path.is_file() and not path.is_symlink(), "owned_file_unavailable")
        payload = path.read_bytes()
        hits = content_hits(payload, needles)
        rows[name] = {**hits, "sha256": hashlib.sha256(payload).hexdigest()}
    passed = all(not row["literal"] and not row["shape"] for row in rows.values())
    return {"verdict": "OWNED_SCAN_PASS" if passed else "OWNED_SCAN_FAIL",
            "files": rows, "three_host_complete": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--expected-head")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--scan-owned", action="store_true")
    args = parser.parse_args(argv)
    parent = args.artifact_dir.resolve()
    shared = Path("/hf3fs-jd/data/tmp").resolve(strict=True)
    require(parent.is_relative_to(shared) and parent != shared,
            "external_shared_output_required")
    require(not parent.is_relative_to(args.repo.resolve()), "output_inside_repository")
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="refs-", dir=parent))
    report = {
        "schema_version": 1,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "read_only_local_primary_and_gitee_refs",
        "push_performed": False,
        "repository_created": False,
        "credentials_changed": False,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    try:
        if args.scan_owned:
            require(args.expected_head is None, "scan_scope_conflict")
            report.update(scan_owned(args.repo.resolve(strict=True)))
            report["scope"] = "explicit_owned_known_content_scan"
        else:
            report.update(inspect(args.repo, args.expected_head))
    except InspectionError as error:
        report.update(verdict="FAIL", first_cause=str(error), three_host_complete=False)
    except Exception as error:
        report.update(verdict="FAIL", first_cause=type(error).__name__,
                      three_host_complete=False)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    receipt = out / "receipt.json"
    with receipt.open("x", encoding="ascii") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    line = f"REPOSITORY_REFS verdict={report['verdict']} three_host_complete=false receipt={receipt}"
    with (out / "execution.log").open("x", encoding="ascii") as stream:
        stream.write(line + "\n")
    print(line, flush=True)
    return 0 if report["verdict"] in ("PRIMARY_AND_GITEE_PASS", "OWNED_SCAN_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
