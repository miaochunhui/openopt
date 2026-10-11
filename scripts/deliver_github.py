"""Explicit audited master delivery, using the existing infra@astrill channel.

Reads only the user-selected GitHub account credential. Never force, mirror,
copy source-repository history, persist credentials or accept unbound replies.
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

if __package__:
    from .inspect_github_target import NoRedirect
else:
    from inspect_github_target import NoRedirect

TARGET = "miaochunhui/openopt"
URL = "https://github.com/" + TARGET + ".git"
TOKEN = Path("/home/mch2/.passwd_github_miaochunhui")
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "infra@astrill")
OID = re.compile(r"[0-9a-f]{40}\Z")
SHA = re.compile(r"[0-9a-f]{64}\Z")


class DeliveryError(ValueError):
    """A fixed cause, never external error or credential content."""


def require(ok, cause):
    if not ok:
        raise DeliveryError(cause)


def digest(path):
    require(path.is_file() and not path.is_symlink(), "regular_file_required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def call(args, *, data=None, env=None, cwd=None, timeout=90):
    try:
        result = subprocess.run(args, input=data, env=env, cwd=cwd,
                                capture_output=True, check=False, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise DeliveryError("command_timeout") from None
    except OSError:
        raise DeliveryError("command_unavailable") from None
    require(result.returncode == 0, "command_failed_rc_" + str(result.returncode))
    require(type(result.stdout) is bytes and len(result.stdout) <= 1048576,
            "command_response_invalid")
    return result.stdout


def git_env(token=None):
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG") if key in os.environ}
    env.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL="/dev/null",
               GIT_CONFIG_COUNT="0", GIT_TERMINAL_PROMPT="0",
               GIT_NO_REPLACE_OBJECTS="1", GIT_NO_LAZY_FETCH="1")
    if token is not None:
        env.update(
            GIT_CONFIG_COUNT="1",
            GIT_CONFIG_KEY_0="http.https://github.com/.extraHeader",
            GIT_CONFIG_VALUE_0="Authorization: Basic " + base64.b64encode(
                ("miaochunhui:" + token).encode("ascii")).decode("ascii"),
        )
    return env


def git(repo, args, token=None):
    return call(
        ["/usr/bin/git", "--no-replace-objects", "-c", "safe.directory=" + str(repo),
         "-c", "http.followRedirects=false", "-c", "http.sslVerify=true",
         "-c", "push.followTags=false", "-C", str(repo), *args],
        env=git_env(token), timeout=150,
    )


def api(token, path, method="GET", body=None):
    require((path, method) in {
        ("/user", "GET"), ("/repos/" + TARGET, "GET"), ("/user/repos", "POST"),
    }, "api_scope_refused")
    if method == "POST":
        require(body == {
            "name": "openopt", "private": False, "auto_init": False,
            "description": "GPU-accelerated optimization research; 0BSD.",
        }, "creation_payload_refused")
    req = urllib.request.Request(
        "https://api.github.com" + path, method=method,
        data=None if body is None else json.dumps(body).encode("ascii"),
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "openopt-explicit-audited-delivery"},
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=25) as response:
            raw = response.read(1048577)
            require(len(raw) <= 1048576, "api_response_too_large")
            return response.status, json.loads(raw)
    except urllib.error.HTTPError as error:
        return error.code, None
    except DeliveryError:
        raise
    except Exception:
        raise DeliveryError("api_transport_failure") from None


def validate_repo(data):
    require(type(data) is dict and data.get("full_name") == TARGET
            and data.get("clone_url") == URL and type(data.get("owner")) is dict
            and data["owner"].get("login") == "miaochunhui", "repository_identity_mismatch")
    permissions = data.get("permissions")
    require(type(permissions) is dict and permissions.get("push") is True,
            "repository_push_permission_missing")


def remote(repo, token):
    raw = git(repo, ["ls-remote", URL, "refs/heads/master"], token)
    rows = raw.decode("ascii").splitlines()
    require(len(rows) <= 1, "remote_ref_ambiguous")
    if not rows:
        return None
    fields = rows[0].split("\t")
    require(len(fields) == 2 and OID.fullmatch(fields[0])
            and fields[1] == "refs/heads/master", "remote_ref_invalid")
    return fields[0]


def deliver(token, bundle, pin, head, directory):
    require(type(token) is str and 16 <= len(token) <= 4096
            and not any(c.isspace() for c in token), "credential_format_invalid")
    require(bool(OID.fullmatch(head)) and bool(SHA.fullmatch(pin)), "delivery_pins_invalid")
    require(digest(bundle) == pin, "bundle_changed")
    status, user = api(token, "/user")
    require(status == 200 and type(user) is dict and user.get("login") == "miaochunhui",
            "credential_account_mismatch")
    repo = directory / "transport.git"
    git(directory, ["init", "--bare", str(repo)])
    git(repo, ["fetch", str(bundle), "refs/heads/master:refs/heads/master"])
    require(git(repo, ["rev-parse", "refs/heads/master"]).strip().decode() == head,
            "bundle_head_mismatch")
    git(repo, ["fsck", "--full", "--no-reflogs"])
    status, data = api(token, "/repos/" + TARGET)
    created = False
    if status == 404:
        status, data = api(token, "/user/repos", "POST", {
            "name": "openopt", "private": False, "auto_init": False,
            "description": "GPU-accelerated optimization research; 0BSD.",
        })
        require(status == 201, "repository_creation_http_" + str(status))
        created = True
    else:
        require(status == 200, "repository_discovery_http_" + str(status))
    validate_repo(data)
    before = remote(repo, token)
    if before and before != head:
        git(repo, ["fetch", URL, "refs/heads/master:refs/remotes/github/master"], token)
        require(git(repo, ["rev-parse", "refs/remotes/github/master"]).strip().decode()
                == before, "remote_changed_before_push")
        git(repo, ["merge-base", "--is-ancestor", before, head])
    git(repo, ["push", URL, head + ":refs/heads/master"], token)
    require(remote(repo, token) == head, "post_push_ref_mismatch")
    require(digest(bundle) == pin, "bundle_changed_after_push")
    return {
        "verdict": "GITHUB_MASTER_PASS", "repository": TARGET, "identity": "miaochunhui",
        "head": head, "bundle_sha256": pin, "created": created,
        "push_performed": True, "readback": "PASS", "before": before,
    }


def audit_guard(path, pin, head):
    require(bool(SHA.fullmatch(pin)) and digest(path) == pin, "audit_pin_mismatch")
    data = json.loads(path.read_bytes())
    audit = data.get("audit", {})
    require(data.get("verdict") == "READER_INTEGRITY_PASS"
            and data.get("content_verdict") == "BOUNDED_CONTENT_PASS"
            and audit.get("head") == head, "audit_scope_mismatch")
    counts = audit.get("counts", {})
    require(type(counts.get("commits")) is int and counts["commits"] > 0
            and type(counts.get("objects")) is int and counts["objects"] > 0,
            "audit_empty")
    require(audit.get("verdict") == "BOUNDED_CONTENT_PASS", "audit_not_cleared")
    require(audit.get("scope") == "full HEAD reachable raw commit/tree/blob graph"
            and audit.get("source_consistency") == "PASS"
            and type(audit.get("unadjudicated")) is int and audit["unadjudicated"] == 0,
            "audit_scope_mismatch")
    membership = audit.get("membership", {})
    require(membership.get("missing") == [] and membership.get("extra") == []
            and membership.get("objects") == counts["objects"], "audit_membership_invalid")
    return audit["source"]["files"]


def local_guard(repo, head):
    require(bool(OID.fullmatch(head)), "full_head_required")
    require(git(repo, ["rev-parse", "HEAD"]).strip().decode() == head,
            "local_head_changed")
    require(git(repo, ["rev-parse", "refs/heads/master"]).strip().decode() == head,
            "local_master_changed")
    require(not git(repo, ["status", "--porcelain=v1", "--untracked-files=all"]),
            "local_source_dirty")
    config = git(repo, ["config", "--local", "--name-only", "--list"]).decode().splitlines()
    allowed = {
        "core.repositoryformatversion", "core.filemode", "core.bare", "core.logallrefupdates",
        "branch.master.remote", "branch.master.merge",
        *(f"remote.{name}.{key}" for name in ("origin", "gitee", "github")
          for key in ("url", "fetch")),
    }
    require(set(config).issubset(allowed), "unsafe_local_git_configuration")


def source_guard(repo, files):
    require(type(files) is dict and bool(files), "audit_source_empty")
    for name, pin in files.items():
        require(type(name) is str and type(pin) is str and bool(SHA.fullmatch(pin)),
                "audit_source_invalid")
        path = repo / name
        require(path.resolve().is_relative_to(repo.resolve()), "audit_source_escape")
        require(digest(path) == pin, "audit_source_changed")


def checked_worker(result, head, pin):
    require(type(result) is dict, "worker_fields_invalid")
    if set(result) == {"verdict", "first_cause"} and result["verdict"] == "FAIL":
        cause = result["first_cause"]
        fixed = {
            "credential_format_invalid", "delivery_pins_invalid", "bundle_changed",
            "credential_account_mismatch", "bundle_head_mismatch", "api_scope_refused",
            "creation_payload_refused", "api_response_too_large", "api_transport_failure",
            "repository_identity_mismatch", "repository_push_permission_missing",
            "remote_ref_ambiguous", "remote_ref_invalid", "remote_changed_before_push",
            "post_push_ref_mismatch", "bundle_changed_after_push", "worker_source_changed",
            "worker_input_too_large", "worker_input_invalid", "worker_internal_failure",
            "command_timeout", "command_unavailable", "command_response_invalid",
            "regular_file_required",
        }
        require(type(cause) is str and (
            cause in fixed or re.fullmatch(
                r"(command_failed_rc_-?[0-9]{1,3}|repository_(creation|discovery)_http_[0-9]{3})",
                cause)), "worker_error_invalid")
        raise DeliveryError(cause)
    fields = {"verdict", "repository", "identity", "head", "bundle_sha256",
              "created", "push_performed", "readback", "before"}
    require(set(result) == fields, "worker_fields_invalid")
    require(result["verdict"] == "GITHUB_MASTER_PASS" and result["repository"] == TARGET
            and result["identity"] == "miaochunhui" and result["head"] == head
            and result["bundle_sha256"] == pin and type(result["created"]) is bool
            and result["push_performed"] is True and result["readback"] == "PASS",
            "worker_response_binding_invalid")
    require(result["before"] is None or (
        type(result["before"]) is str and bool(OID.fullmatch(result["before"]))),
        "worker_before_invalid")
    return result


def coordinated(repo, head, audit, audit_pin, out):
    local_guard(repo, head)
    files = audit_guard(audit, audit_pin, head)
    source_guard(repo, files)
    bundle = out / "master.bundle"
    git(repo, ["bundle", "create", str(bundle), "refs/heads/master"])
    pin = digest(bundle)
    source = Path(__file__).resolve()
    probe = source.with_name("inspect_github_target.py")
    source_pin, probe_pin = digest(source), digest(probe)
    remote_dir = "/var/tmp/openopt-github-delivery-" + head + "-" + out.name
    require(bool(re.fullmatch(r"/var/tmp/openopt-github-delivery-[a-z0-9-]+", remote_dir)),
            "remote_path_invalid")
    call([*SSH, "mkdir", "-m", "700", remote_dir])
    call(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
          str(source), str(probe), str(bundle), "infra@astrill:" + remote_dir + "/"])
    actual = call([*SSH, "sha256sum", remote_dir + "/deliver_github.py",
                   remote_dir + "/inspect_github_target.py", remote_dir + "/master.bundle"])
    require(actual.split() == [
        source_pin.encode(), (remote_dir + "/deliver_github.py").encode(),
        probe_pin.encode(), (remote_dir + "/inspect_github_target.py").encode(),
        pin.encode(), (remote_dir + "/master.bundle").encode(),
    ], "remote_transport_pin_mismatch")
    require(TOKEN.is_file() and not TOKEN.is_symlink(), "credential_file_unavailable")
    token = TOKEN.read_text(encoding="ascii").strip()
    try:
        execution = subprocess.run(
            [*SSH, "python3", remote_dir + "/deliver_github.py", "--worker",
             "--source-sha256", source_pin, "--probe-sha256", probe_pin,
             "--head", head, "--bundle-sha256", pin],
            input=json.dumps({"token": token}).encode("ascii"),
            capture_output=True, check=False, timeout=210,
        )
    except subprocess.TimeoutExpired:
        raise DeliveryError("command_timeout") from None
    except OSError:
        raise DeliveryError("command_unavailable") from None
    raw = execution.stdout
    require(type(raw) is bytes and len(raw) <= 8192, "worker_response_invalid")
    require(token.encode() not in raw, "credential_echo_refused")
    result = checked_worker(json.loads(raw), head, pin)
    require(execution.returncode == 0, "worker_returncode_invalid")
    require(digest(source) == source_pin and digest(probe) == probe_pin, "tool_source_changed")
    local_guard(repo, head)
    source_guard(repo, files)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--probe-sha256")
    parser.add_argument("--head")
    parser.add_argument("--scan-owned", action="store_true")
    parser.add_argument("--bundle-sha256")
    parser.add_argument("--repo", type=Path)
    parser.add_argument("--audit-reader", type=Path)
    parser.add_argument("--audit-reader-sha256")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    if args.scan_owned:
        require(not args.worker and args.artifact_dir is not None, "scan_scope_invalid")
        if __package__:
            from .audit_git_delivery import content_hits, load_needles
        else:
            from audit_git_delivery import content_hits, load_needles
        needles = load_needles()
        for credential in (TOKEN, Path("/home/mch2/.passwd_github")):
            require(credential.is_file() and not credential.is_symlink(),
                    "scan_credential_unavailable")
            value = credential.read_bytes().strip()
            require(bool(value), "scan_credential_empty")
            needles.append(value)
        root = Path(__file__).resolve().parents[1]
        names = (
            "scripts/deliver_github.py", "scripts/inspect_github_target.py",
            "tests/test_deliver_github.py", "tests/test_inspect_github_target.py",
            "docs/research/github-target-permission.md",
        )
        rows = {
            name: dict(content_hits((root / name).read_bytes(), needles),
                       sha256=digest(root / name))
            for name in names
        }
        passed = all(row["literal"] == 0 and row["shape"] == 0 for row in rows.values())
        parent = args.artifact_dir.resolve()
        shared = Path("/hf3fs-jd/data/tmp").resolve(strict=True)
        require(parent.is_relative_to(shared) and parent != shared, "external_output_required")
        parent.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix="scan-", dir=parent))
        receipt = out / "receipt.json"
        with receipt.open("x", encoding="ascii") as stream:
            json.dump(dict(verdict="OWNED_SCAN_PASS" if passed else "FAIL", files=rows,
                           credential_values_recorded=False, scan_credentials_read=True,
                           observed_at=datetime.now(timezone.utc).isoformat()), stream,
                      sort_keys=True, indent=2)
            stream.write("\n")
        line = f"GITHUB_SOURCE_SCAN verdict={'PASS' if passed else 'FAIL'} receipt={receipt}"
        with (out / "execution.log").open("x", encoding="ascii") as stream:
            stream.write(line + "\n")
        print(line, flush=True)
        return 0 if passed else 1
    if args.worker:
        try:
            source = Path(__file__).resolve()
            require(digest(source) == args.source_sha256
                    and digest(source.with_name("inspect_github_target.py")) == args.probe_sha256,
                    "worker_source_changed")
            payload = sys.stdin.buffer.read(8193)
            require(len(payload) <= 8192, "worker_input_too_large")
            data = json.loads(payload)
            require(type(data) is dict and set(data) == {"token"}, "worker_input_invalid")
            result = deliver(data["token"], source.with_name("master.bundle"),
                             args.bundle_sha256, args.head, source.parent)
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        except Exception as error:
            cause = str(error) if isinstance(error, DeliveryError) else "worker_internal_failure"
            print(json.dumps({"verdict": "FAIL", "first_cause": cause}), flush=True)
            return 1
    require(all((args.head, args.repo, args.audit_reader, args.audit_reader_sha256, args.artifact_dir)),
            "explicit_delivery_inputs_required")
    parent = args.artifact_dir.resolve()
    shared = Path("/hf3fs-jd/data/tmp").resolve(strict=True)
    require(parent.is_relative_to(shared) and parent != shared, "external_output_required")
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="delivery-", dir=parent))
    result = dict(started_at=datetime.now(timezone.utc).isoformat(),
                  credentials_changed=False, credential_values_recorded=False,
                  source_sha256=digest(Path(__file__)), three_host_complete=False)
    try:
        result.update(coordinated(args.repo.resolve(), args.head, args.audit_reader,
                                  args.audit_reader_sha256, out))
    except Exception as error:
        result.update(verdict="FAIL", first_cause=(
            str(error) if isinstance(error, DeliveryError) else type(error).__name__))
    result["finished_at"] = datetime.now(timezone.utc).isoformat()
    receipt = out / "receipt.json"
    with receipt.open("x", encoding="ascii") as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write("\n")
    line = f"GITHUB_DELIVERY verdict={result['verdict']} receipt={receipt}"
    with (out / "execution.log").open("x", encoding="ascii") as stream:
        stream.write(line + "\n")
    print(line, flush=True)
    return 0 if result["verdict"] == "GITHUB_MASTER_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
