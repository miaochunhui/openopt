"""Offline readback controls; no SSH, GitHub request or credential access."""

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import read_github_master as tool

HEAD = "aed5f343dd235ae01196fe83090c593c89131d63"
SNAPSHOT = {"commit": HEAD, "sha256": {"scripts/read_github_master.py": "a" * 64}}
VALID = (HEAD + "\trefs/heads/master\n").encode("ascii")


def fake(stdout=VALID, rc=0):
    def run(command, **kwargs):
        assert command == tool.COMMAND
        assert kwargs == {"env": {"PATH": "/usr/bin:/bin", "LC_ALL": "C"},
                          "capture_output": True, "check": False, "timeout": 30}
        return subprocess.CompletedProcess(command, rc, stdout, b"untrusted diagnostic")
    return run


def execute(tmp_path, runner=None, snapshot=None, expected=HEAD):
    return tool.readback(expected, tmp_path / "receipt.json", runner=runner or fake(),
                         snapshot=snapshot or (lambda: deepcopy(SNAPSHOT)))


def test_exact_output_and_create_once_receipt(tmp_path):
    result = execute(tmp_path)
    assert result["verdict"] == "PASS" and result["head"] == HEAD
    assert result["matches_expected"] and result["source_before"] == result["source_after"]
    assert result["started_at"] <= result["finished_at"]
    assert json.loads((tmp_path / "receipt.json").read_bytes()) == result
    original = (tmp_path / "receipt.json").read_bytes()
    with pytest.raises(FileExistsError):
        execute(tmp_path, runner=lambda *a, **kw: pytest.fail("must not retry"))
    assert (tmp_path / "receipt.json").read_bytes() == original


@pytest.mark.parametrize("payload", [
    b"", VALID + VALID, b"banner\n" + VALID, VALID.replace(b"master", b"main"),
    VALID.replace(b"\t", b" "), VALID.rstrip(b"\n"), VALID + b"\n",
    VALID.upper(), VALID.replace(HEAD.encode(), b"a" * 64),
    VALID.replace(HEAD.encode(), b"0" * 40), b"\xff\n", "not bytes",
])
def test_fake_output_rejects_without_content_recording(tmp_path, payload):
    result = execute(tmp_path, runner=fake(payload))
    assert result["verdict"] == "FAIL" and result["head"] is None
    assert not result["matches_expected"] and result["first_cause"]
    assert "untrusted diagnostic" not in (tmp_path / "receipt.json").read_text()


def test_head_mismatch_retains_actual_whitelisted_head(tmp_path):
    actual = "b" * 40
    result = execute(tmp_path, runner=fake((actual + "\trefs/heads/master\n").encode()))
    assert result["verdict"] == "FAIL" and result["head"] == actual
    assert result["first_cause"] == "master_mismatch"


@pytest.mark.parametrize("rc", [2, 128, 255, -9])
def test_missing_public_repo_or_transport_failure_stays_fail(tmp_path, rc):
    result = execute(tmp_path, runner=fake(rc=rc))
    assert result["verdict"] == "FAIL" and result["returncode"] == rc
    assert result["first_cause"] == "remote_query_failed"


def test_timeout_has_fixed_noncontent_cause(tmp_path):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], 30, output=b"untrusted timeout output")
    result = execute(tmp_path, runner=timed_out)
    assert result["verdict"] == "FAIL"
    assert result["first_cause"] == "remote_query_timeout"
    assert "untrusted timeout output" not in (tmp_path / "receipt.json").read_text()


def test_source_drift_cannot_pass(tmp_path):
    snapshots = iter([SNAPSHOT, {**SNAPSHOT, "commit": "c" * 40}])
    result = execute(tmp_path, snapshot=lambda: next(snapshots))
    assert result["verdict"] == "FAIL"
    assert result["first_cause"] == "source_changed_or_unavailable"


def test_dirty_source_rejects_before_network(tmp_path):
    def unavailable():
        raise tool.ReadbackError("source_dirty")
    result = execute(tmp_path, snapshot=unavailable,
                     runner=lambda *a, **kw: pytest.fail("no network before clean source"))
    assert result["verdict"] == "FAIL" and not result["network_attempted"]
    assert result["first_cause"] == "source_dirty"


def test_cli_bad_input_does_not_query_network(tmp_path):
    result = subprocess.run(
        ["/hf3fs-jd/data/tmp/openopt-validation-env/bin/python", "-B",
         str(tool.ROOT / tool.OWNED[0]), "--expected-head", "short",
         "--receipt", str(tmp_path / "receipt.json")],
        capture_output=True, check=False, timeout=10)
    assert result.returncode == 1
    assert result.stdout == b"GITHUB_MASTER verdict=FAIL first_cause=receipt_or_input_rejected\n"
    assert not result.stderr and not (tmp_path / "receipt.json").exists()


@pytest.mark.parametrize("expected", [None, True, "a" * 7, "a" * 64, "A" * 40, "0" * 40])
def test_expected_must_be_full_nonzero_sha1(tmp_path, expected):
    with pytest.raises(tool.ReadbackError, match="expected_full_head_required"):
        execute(tmp_path, expected=expected)
    assert not (tmp_path / "receipt.json").exists()


def test_fixed_readonly_command_and_isolation():
    assert tool.COMMAND[:7] == (
        "/usr/bin/ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "infra@astrill",
        tool.REMOTE_COMMAND)
    assert tool.URL == "https://github.com/miaochunhui/openopt.git"
    assert tool.REF == "refs/heads/master"
    assert "env -i" in tool.REMOTE_COMMAND and "http.followRedirects=false" in tool.REMOTE_COMMAND
    assert tool.ENV["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert tool.ENV["GIT_CONFIG_NOSYSTEM"] == "1"
    assert not any("TRACE" in key or "TOKEN" in key for key in tool.ENV)
    assert "credential.helper=" in tool.GIT and "http.extraHeader=" in tool.GIT
    assert "ls-remote --exit-code --refs" in tool.REMOTE_COMMAND
    assert not any(word in tool.COMMAND for word in ("push", "fetch", "clone"))


def test_source_pins_actual_owned_blobs(monkeypatch):
    monkeypatch.setattr(tool, "ROOT", Path(__file__).resolve().parents[1])
    if subprocess.run(["git", "status", "--porcelain"], cwd=tool.ROOT,
                      capture_output=True, check=True).stdout:
        with pytest.raises(tool.ReadbackError, match="source_dirty"):
            tool.source()
    else:
        current = tool.source()
        assert set(current["sha256"]) == set(tool.OWNED)
        assert current["sha256"][tool.OWNED[0]] == hashlib.sha256(
            (tool.ROOT / tool.OWNED[0]).read_bytes()).hexdigest()


def shape_scan(output):
    """Existing shape scanner only; no credential files or known needles read."""
    from scripts.audit_git_delivery import content_hits
    output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    rows = {}
    for name in tool.OWNED:
        payload = (root / name).read_bytes()
        sentinel = b"offline-scan-sentinel-" + b"not-a-real-credential"
        rows[name] = content_hits(payload, [sentinel])
    result = {"files": rows, "known_credential_scan": "NOT_RUN", "credentials_read": False}
    (output / "scan.json").write_text(json.dumps(result, sort_keys=True, indent=2))
    assert all(not row["literal"] and not row["shape"] for row in rows.values())
    print("GITHUB_READBACK_SCAN shape=0 synthetic_literal=0 known_credentials=NOT_RUN files=3")


def check_receipt(args):
    """Reuse existing source/collection/JUnit helpers; never contact a remote."""
    from scripts import verify_sharded as sharded
    from scripts.verify import source_snapshot
    args.output.mkdir(parents=True, exist_ok=False)
    raw = args.receipt.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == args.receipt_sha256
    report = json.loads(raw)
    before = source_snapshot(tool.ROOT)
    assert before["clean"] and before == report["source"]
    assert report["verdict"] == report["source_consistency"]["verdict"] == "PASS"
    env = sharded.cpu_env(tool.ROOT, args.output)
    env["OPENOPT_SHARD_COLLECTION"] = str(args.output / "collection.json")
    sharded.command_logged(
        ["/hf3fs-jd/data/tmp/openopt-validation-env/bin/python", "-B", "-m", "pytest",
         "--collect-only", "-q", "-p", "scripts.verify_sharded", "-o", "addopts=",
         "-o", f"cache_dir={args.output / 'cache'}", *report["targets"]],
        tool.ROOT, args.output, 60, env)
    nodes = json.loads((args.output / "collection.json").read_bytes())["nodeids"]
    records, counts = sharded.parse_junit(args.receipt.parent / "pytest.xml")
    assert nodes and len(nodes) == len(set(nodes))
    assert sorted(sharded.junit_key(node) for node in nodes) == sorted(item["key"] for item in records)
    assert counts == report["tests"] and not any(counts[key] for key in ("failures", "errors", "skipped"))
    assert before == source_snapshot(tool.ROOT)
    terminal = {"verdict": "PASS", "source_commit": before["commit"], "counts": counts,
                "receipt_sha256": args.receipt_sha256, "network_executed": False,
                "collection_minus_junit": [], "junit_minus_collection": [], "duplicates": [],
                "source_sha256": {name: before["sha256"][name] for name in tool.OWNED}}
    with (args.output / "terminal.json").open("x") as stream:
        json.dump(terminal, stream, sort_keys=True, indent=2)
    print(f"GITHUB_READBACK_TEST_READER verdict=PASS counts={counts} network=false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-output", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--receipt-sha256")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    selected = args.scan_output or args.output
    assert selected and selected.resolve().is_relative_to(Path("/hf3fs-jd/data/tmp").resolve())
    if args.scan_output:
        shape_scan(args.scan_output)
    else:
        assert args.receipt and args.receipt_sha256
        check_receipt(args)
