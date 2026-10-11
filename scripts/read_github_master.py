"""Anonymous fixed GitHub master readback through the existing astrill SSH."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/miaochunhui/openopt.git"
REF = "refs/heads/master"
OID = re.compile(r"[0-9a-f]{40}\Z")
OWNED = ("scripts/read_github_master.py", "tests/test_read_github_master.py",
         "docs/research/github-master-readback.md")
ENV = {
    "PATH": "/usr/bin:/bin", "HOME": "/nonexistent",
    "XDG_CONFIG_HOME": "/nonexistent", "LC_ALL": "C",
    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_REPLACE_OBJECTS": "1", "GIT_NO_LAZY_FETCH": "1",
    "GIT_ALLOW_PROTOCOL": "https",
}
GIT = ("/usr/bin/git", "--no-replace-objects", "-c", "credential.helper=",
       "-c", "http.extraHeader=", "-c", "http.followRedirects=false",
       "-c", "http.sslVerify=true", "-c", "core.askPass=")
REMOTE_COMMAND = "cd / && exec " + shlex.join(
    ["env", "-i", *[f"{key}={value}" for key, value in ENV.items()],
     *GIT, "ls-remote", "--exit-code", "--refs", URL, REF])
COMMAND = ("/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
           "infra@astrill", REMOTE_COMMAND)


class ReadbackError(ValueError):
    """Only fixed error codes cross the captured subprocess boundary."""


def require(ok, code):
    if not ok:
        raise ReadbackError(code)


def parse_head(payload):
    require(type(payload) is bytes, "invalid_stdout_type")
    match = re.fullmatch(rb"([0-9a-f]{40})\trefs/heads/master\n", payload)
    require(match is not None, "missing_or_invalid_master")
    head = match[1].decode("ascii")
    require(head != "0" * 40, "null_master")
    return head


def source():
    def git(*args):
        result = subprocess.run(
            [*GIT, "-c", f"safe.directory={ROOT.resolve()}", "-C", str(ROOT), *args], env=ENV,
            capture_output=True, check=False, timeout=10)
        require(result.returncode == 0, "source_query_failed")
        return result.stdout
    commit = git("rev-parse", "HEAD").decode("ascii").strip()
    require(bool(OID.fullmatch(commit)), "source_commit_invalid")
    require(not git("status", "--porcelain=v1", "--untracked-files=all"), "source_dirty")
    hashes = {}
    for name in OWNED:
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), "source_file_invalid")
        payload = path.read_bytes()
        require(payload == git("show", f"{commit}:{name}"), "source_blob_mismatch")
        hashes[name] = hashlib.sha256(payload).hexdigest()
    return {"commit": commit, "sha256": hashes}


def readback(expected, receipt, *, runner=subprocess.run, snapshot=source):
    require(type(expected) is str and bool(OID.fullmatch(expected))
            and expected != "0" * 40, "expected_full_head_required")
    receipt = Path(receipt)
    shared = Path("/hf3fs-jd/data/tmp").resolve(strict=True)
    parent = receipt.parent.resolve(strict=True)
    require(parent.is_relative_to(shared) and not parent.is_relative_to(ROOT)
            and receipt.suffix == ".json", "external_receipt_required")
    report = {
        "verdict": "FAIL", "url": URL, "ref": REF, "expected_head": expected,
        "head": None, "matches_expected": False, "network_attempted": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "scope": "github_master_only", "three_host_complete": False,
        "push_performed": False, "credentials_read": False,
    }
    # Reserve before querying: an existing receipt must never trigger a retry.
    with receipt.open("x", encoding="ascii") as stream:
        try:
            report["source_before"] = snapshot()
            report["network_attempted"] = True
            result = runner(COMMAND, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                            capture_output=True, check=False, timeout=30)
            require(type(result.returncode) is int, "invalid_returncode")
            report["returncode"] = result.returncode
            require(result.returncode == 0, "remote_query_failed")
            report["head"] = parse_head(result.stdout)
            report["matches_expected"] = report["head"] == expected
            require(report["matches_expected"], "master_mismatch")
            report["verdict"] = "PASS"
        except subprocess.TimeoutExpired:
            report["first_cause"] = "remote_query_timeout"
        except ReadbackError as error:
            report["first_cause"] = str(error)
        except Exception:
            report["first_cause"] = "readback_execution_failed"
        try:
            report["source_after"] = snapshot()
            require(report.get("source_before") == report["source_after"], "source_changed")
        except Exception:
            report.setdefault("first_cause", "source_changed_or_unavailable")
            report["verdict"] = "FAIL"
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        json.dump(report, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = readback(args.expected_head, args.receipt)
    except Exception:
        print("GITHUB_MASTER verdict=FAIL first_cause=receipt_or_input_rejected")
        return 1
    print(f"GITHUB_MASTER verdict={report['verdict']} expected={report['expected_head']} "
          f"head={report['head']} matches={report['matches_expected']} receipt={args.receipt}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
