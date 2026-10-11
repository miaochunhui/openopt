"""Offline real-Git ownership controls; no SSH, network or permission changes."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import read_github_master as tool

ROOT = Path(__file__).resolve().parents[1]
BASE = "adb955828689e4f2933890b3a00f7d50af11ea7e"
NEW_FILES = (
    "scripts/read_github_master.py",
    "tests/test_read_github_master_source_ownership.py",
    "docs/research/github-master-source-ownership-repair.md",
)
ORIGINAL_TEST = "tests/test_read_github_master.py"
REAL_RUN = subprocess.run


def local_git(root, env, *args, safe=None):
    command = [*tool.GIT]
    if safe is not None:
        command.extend(["-c", f"safe.directory={safe}"])
    return REAL_RUN([*command, "-C", str(root), *args], env=env,
                    capture_output=True, check=False, timeout=10)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "source"
    root.mkdir()
    env = dict(tool.ENV)
    result = local_git(root, env, "init", "--template=", "-b", "master")
    assert result.returncode == 0
    for name in tool.OWNED:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT / name).read_bytes())
    assert local_git(root, env, "add", "--", *tool.OWNED).returncode == 0
    result = local_git(root, env, "-c", "user.name=offline-test",
                       "-c", "user.email=offline@example.invalid",
                       "-c", "core.hooksPath=/nonexistent",
                       "commit", "-m", "offline source fixture")
    assert result.returncode == 0
    # Git's own ownership test switch exercises its real protected-config check.
    env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"
    monkeypatch.setattr(tool, "ROOT", root)
    monkeypatch.setattr(tool, "ENV", env)
    return root, env


def test_real_git_rejects_untrusted_owner_and_accepts_exact_path(repository):
    root, env = repository
    untrusted = local_git(root, env, "rev-parse", "HEAD")
    assert untrusted.returncode == 128
    assert b"detected dubious ownership" in untrusted.stderr
    accepted = local_git(root, env, "rev-parse", "HEAD", safe=root.resolve())
    assert accepted.returncode == 0
    source = tool.source()
    assert source["commit"] == accepted.stdout.decode("ascii").strip()
    assert set(source["sha256"]) == set(tool.OWNED)


def test_wrong_path_does_not_authorize_repository(repository, tmp_path):
    root, env = repository
    wrong = local_git(root, env, "rev-parse", "HEAD", safe=tmp_path / "other")
    assert wrong.returncode == 128
    assert b"detected dubious ownership" in wrong.stderr


def test_exact_trust_is_process_only(repository):
    root, env = repository
    before = (root / ".git/config").read_bytes()
    tool.source()
    assert (root / ".git/config").read_bytes() == before
    later = local_git(root, env, "rev-parse", "HEAD")
    assert later.returncode == 128
    assert b"detected dubious ownership" in later.stderr


def test_dirty_source_still_rejects_before_remote(repository, tmp_path):
    root, _ = repository
    (root / "unexpected.md").write_text("offline dirty control\n", encoding="ascii")
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("network prohibited")

    result = tool.readback(BASE, tmp_path / "dirty.json", runner=forbidden)
    assert result["verdict"] == "FAIL"
    assert result["first_cause"] == "source_dirty"
    assert not result["network_attempted"] and not calls


def test_hidden_blob_drift_still_rejects(repository, tmp_path):
    root, env = repository
    name = tool.OWNED[0]
    assert local_git(root, env, "update-index", "--assume-unchanged", "--", name,
                     safe=root.resolve()).returncode == 0
    with (root / name).open("ab") as stream:
        stream.write(b"\n# offline drift control\n")
    assert local_git(root, env, "status", "--porcelain", safe=root.resolve()).stdout == b""
    result = tool.readback(BASE, tmp_path / "drift.json",
                          runner=lambda *args, **kwargs: pytest.fail("network prohibited"))
    assert result["verdict"] == "FAIL"
    assert result["first_cause"] == "source_blob_mismatch"
    assert not result["network_attempted"]


def test_local_commands_have_one_exact_trust_remote_unchanged(repository, monkeypatch):
    root, env = repository
    calls = []
    remote = tool.COMMAND

    def observed(command, **kwargs):
        assert command[0] == "/usr/bin/git"
        assert kwargs["env"] == env
        trusts = [word for word in command if word.startswith("safe.directory=")]
        assert trusts == [f"safe.directory={root.resolve()}"]
        calls.append(command)
        return REAL_RUN(command, **kwargs)

    monkeypatch.setattr(tool.subprocess, "run", observed)
    tool.source()
    assert len(calls) == 2 + len(tool.OWNED)
    assert tool.COMMAND == remote and "safe.directory" not in tool.REMOTE_COMMAND
    assert tool.ENV["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert tool.ENV["GIT_CONFIG_NOSYSTEM"] == "1"


def test_original_author_file_byte_parity():
    original = local_git(ROOT, tool.ENV, "show", f"{BASE}:{ORIGINAL_TEST}")
    assert original.returncode == 0
    assert (ROOT / ORIGINAL_TEST).read_bytes() == original.stdout


def scan_additions(output):
    """Reuse the fixed helper API; raw scanner output never leaves RAM."""
    from contextlib import redirect_stderr, redirect_stdout
    import importlib.util
    import io

    fixed = "5a6b6ff88b9f7ff7830421e3723ec18669e2e3ce"
    helper_root = Path("/data/dev/openopt-wt-repository-delivery-docs-scan-20261011")
    for name in ("scripts/safe_scan_owned_sources.py", "scripts/audit_git_delivery.py"):
        expected = local_git(ROOT, tool.ENV, "show", f"{fixed}:{name}")
        assert expected.returncode == 0
        assert (helper_root / name).read_bytes() == expected.stdout
    path = helper_root / "scripts/safe_scan_owned_sources.py"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="ascii") as stream:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            spec = importlib.util.spec_from_file_location("fixed_owned_scan", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            report = module.scan([ROOT / name for name in NEW_FILES], root=ROOT)
        report["helper_commit"] = fixed
        json.dump(report, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(f"OWNERSHIP_REPAIR_SCAN verdict={report['verdict']} files={report['files']} "
          f"literal_hits={report['literal_hits']} shape_hits={report['shape_hits']} "
          f"receipt_sha256={hashlib.sha256(output.read_bytes()).hexdigest()}")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan-receipt", type=Path, required=True)
    args = parser.parse_args()
    assert args.scan_receipt.resolve().is_relative_to(Path("/hf3fs-jd/data/tmp").resolve())
    raise SystemExit(scan_additions(args.scan_receipt))
