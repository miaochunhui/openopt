"""Inspect only miaochunhui/openopt permissions through the existing SSH channel.

No Git operations, API writes, credential changes or account substitutions.
The source-addressed remote probe is code, not a service installation.
Reference: https://docs.github.com/en/rest/repos/repos#get-a-repository
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request

TARGET = "miaochunhui/openopt"
TOKEN_FILE = Path("/home/mch2/.passwd_github_miaochunhui")
SSH = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "infra@astrill")
FIELDS = {"verdict", "state", "repository", "http_status", "push_permission",
          "first_cause"}
STATES = {
    "WRITE_PERMISSION_VERIFIED", "NO_WRITE_PERMISSION",
    "MISSING_OR_INACCESSIBLE", "AUTHENTICATION_REJECTED",
    "HTTP_FAILURE", "PROBE_FAILURE",
}


class ProbeError(ValueError):
    """Only fixed non-content errors may cross the transport boundary."""


class SafeParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(2, "invalid_arguments\n")


def require(ok, cause):
    if not ok:
        raise ProbeError(cause)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProbeError("redirect_refused")


def project(status, data):
    require(type(status) is int and 100 <= status <= 599, "http_status_invalid")
    result = dict(repository=TARGET, http_status=status, first_cause=None,
                  push_permission=None, verdict="BLOCKED")
    if status == 404:
        return dict(result, state="MISSING_OR_INACCESSIBLE")
    if status in (401, 403):
        return dict(result, state="AUTHENTICATION_REJECTED")
    if status != 200:
        return dict(result, state="HTTP_FAILURE", verdict="FAIL")
    require(type(data) is dict, "repository_schema_invalid")
    require(data.get("full_name") == TARGET, "repository_identity_invalid")
    require(type(data.get("owner")) is dict
            and data["owner"].get("login") == "miaochunhui",
            "repository_owner_invalid")
    require(data.get("clone_url") == "https://github.com/" + TARGET + ".git",
            "repository_url_invalid")
    permissions = data.get("permissions")
    require(type(permissions) is dict
            and type(permissions.get("push")) is bool, "permissions_unavailable")
    push = permissions["push"]
    return dict(result, push_permission=push,
                verdict="PASS" if push else "BLOCKED",
                state="WRITE_PERMISSION_VERIFIED" if push else "NO_WRITE_PERMISSION")


def api_probe(token, *, opener=None):
    require(type(token) is str and 16 <= len(token) <= 4096
            and not any(c.isspace() for c in token), "credential_format_invalid")
    request = urllib.request.Request(
        "https://api.github.com/repos/" + TARGET,
        headers={"Authorization": "Bearer " + token,
                 "Accept": "application/vnd.github+json",
                 "User-Agent": "openopt-target-permission-inspection"},
        method="GET",
    )
    if opener is None:
        opener = urllib.request.build_opener(NoRedirect())
    try:
        identity = urllib.request.Request(
            "https://api.github.com/user",
            headers=dict(request.header_items()), method="GET",
        )
        with opener.open(identity, timeout=25) as response:
            raw = response.read(1048577)
            require(len(raw) <= 1048576, "response_too_large")
            user = json.loads(raw)
            require(response.status == 200 and type(user) is dict
                    and user.get("login") == "miaochunhui", "credential_account_mismatch")
        with opener.open(request, timeout=25) as response:
            raw = response.read(1048577)
            require(len(raw) <= 1048576, "response_too_large")
            data = json.loads(raw)
            return project(response.status, data)
    except urllib.error.HTTPError as error:
        return project(error.code, None)
    except ProbeError:
        raise
    except Exception as error:
        raise ProbeError("api_transport_" + type(error).__name__) from None


def validate_result(result):
    require(type(result) is dict and set(result) == FIELDS, "response_fields_invalid")
    require(result["repository"] == TARGET, "response_target_invalid")
    require(type(result["state"]) is str and result["state"] in STATES,
            "response_state_invalid")
    state = result["state"]
    cause = result["first_cause"]
    if state == "PROBE_FAILURE":
        safe_causes = {
            "source_pin_required", "source_pin_mismatch", "input_too_large",
            "input_schema_invalid", "credential_format_invalid", "response_too_large",
            "redirect_refused", "repository_schema_invalid", "repository_identity_invalid",
            "repository_owner_invalid", "repository_url_invalid", "permissions_unavailable",
            "worker_internal_failure", "credential_account_mismatch",
        }
        require(type(cause) is str and (
            cause in safe_causes or cause in {
                "api_transport_URLError", "api_transport_TimeoutError",
                "api_transport_JSONDecodeError", "api_transport_UnicodeDecodeError",
                "api_transport_OSError",
            }), "response_error_invalid")
        require(result["verdict"] == "FAIL" and result["http_status"] is None
                and result["push_permission"] is None, "response_binding_invalid")
        raise ProbeError(cause)
    require(cause is None, "response_error_invalid")
    status = result["http_status"]
    require(type(status) is int and 100 <= status <= 599, "response_status_invalid")
    push = result["push_permission"]
    require(push is None or type(push) is bool, "response_permission_invalid")
    expected = {
        "WRITE_PERMISSION_VERIFIED": ("PASS", 200, True),
        "NO_WRITE_PERMISSION": ("BLOCKED", 200, False),
        "MISSING_OR_INACCESSIBLE": ("BLOCKED", 404, None),
    }
    if state in expected:
        require((result["verdict"], status, push) == expected[state],
                "response_binding_invalid")
    elif state == "AUTHENTICATION_REJECTED":
        require(result["verdict"] == "BLOCKED" and status in (401, 403)
                and push is None, "response_binding_invalid")
    elif state == "HTTP_FAILURE":
        require(result["verdict"] == "FAIL" and status not in (200, 401, 403, 404)
                and push is None, "response_binding_invalid")
    return result


def command(args, *, payload=None, timeout=30, runner=subprocess.run):
    try:
        child = runner(args, input=payload, capture_output=True, check=False,
                       timeout=timeout)
    except subprocess.TimeoutExpired:
        raise ProbeError("transport_timeout") from None
    except OSError:
        raise ProbeError("transport_unavailable") from None
    require(child.returncode == 0, "transport_failed_rc_" + str(child.returncode))
    require(type(child.stdout) is bytes and len(child.stdout) <= 8192,
            "transport_response_invalid")
    return child.stdout


def inspect(*, runner=subprocess.run, token_path=TOKEN_FILE):
    source_path = Path(__file__).resolve()
    source = source_path.read_bytes()
    pin = hashlib.sha256(source).hexdigest()
    remote = "/var/tmp/openopt-target-permission-" + pin + ".py"
    command(["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             str(source_path), "infra@astrill:" + remote], runner=runner)
    fingerprint = command([*SSH, "sha256sum", remote], runner=runner)
    require(fingerprint.split() == [pin.encode(), remote.encode()],
            "remote_source_mismatch")
    require(token_path.is_file() and not token_path.is_symlink(),
            "credential_file_unavailable")
    before = token_path.stat()
    token = token_path.read_text(encoding="ascii").strip()
    require(16 <= len(token) <= 4096 and not any(c.isspace() for c in token),
            "credential_format_invalid")
    try:
        child = runner(
            [*SSH, "python3", remote, "--worker", "--source-sha256", pin],
            input=json.dumps({"token": token}).encode("ascii"),
            capture_output=True, check=False, timeout=40,
        )
    except subprocess.TimeoutExpired:
        raise ProbeError("inspect_timeout") from None
    except OSError:
        raise ProbeError("inspect_unavailable") from None
    require(type(child.stdout) is bytes and len(child.stdout) <= 8192,
            "worker_response_invalid")
    require(token.encode() not in child.stdout, "credential_echo_refused")
    require(source_path.read_bytes() == source, "source_changed")
    after = token_path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_ino, after.st_size, after.st_mtime_ns), "credential_file_changed")
    try:
        result = validate_result(json.loads(child.stdout))
    except (UnicodeError, json.JSONDecodeError):
        raise ProbeError("worker_json_invalid") from None
    require(child.returncode == (0 if result["verdict"] == "PASS" else 1),
            "worker_returncode_inconsistent")
    return result


def worker(pin):
    try:
        require(type(pin) is str and bool(re.fullmatch("[0-9a-f]{64}", pin)),
                "source_pin_required")
        require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == pin,
                "source_pin_mismatch")
        payload = sys.stdin.buffer.read(8193)
        require(len(payload) <= 8192, "input_too_large")
        data = json.loads(payload)
        require(type(data) is dict and set(data) == {"token"}, "input_schema_invalid")
        result = validate_result(api_probe(data["token"]))
    except Exception as error:
        # Fixed output deliberately excludes arbitrary exceptions and HTTP bodies.
        result = dict(verdict="FAIL", state="PROBE_FAILURE", repository=TARGET,
                      http_status=None, push_permission=None, first_cause=(
                          str(error) if isinstance(error, ProbeError)
                          else "worker_internal_failure"))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["verdict"] == "PASS" else 1


def main(argv=None):
    parser = SafeParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--source-sha256")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args.source_sha256)
    require(args.source_sha256 is None, "worker_arguments_refused")
    require(args.artifact_dir is not None, "artifact_directory_required")
    shared = Path("/hf3fs-jd/data/tmp").resolve(strict=True)
    parent = args.artifact_dir.resolve()
    require(parent.is_relative_to(shared) and parent != shared,
            "external_output_required")
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="target-", dir=parent))
    report = dict(
        schema_version=1, started_at=datetime.now(timezone.utc).isoformat(),
        scope="read_only_exact_target_push_permission", repository=TARGET,
        repository_created=False, push_performed=False, credentials_changed=False,
        credential_values_recorded=False,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        three_host_complete=False,
    )
    try:
        report.update(inspect())
    except Exception as error:
        report.update(verdict="FAIL", first_cause=(
            str(error) if isinstance(error, ProbeError) else type(error).__name__))
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    receipt = out / "receipt.json"
    with receipt.open("x", encoding="ascii") as stream:
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    line = f"GITHUB_TARGET verdict={report['verdict']} receipt={receipt}"
    with (out / "execution.log").open("x", encoding="ascii") as stream:
        stream.write(line + "\n")
    print(line, flush=True)
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
