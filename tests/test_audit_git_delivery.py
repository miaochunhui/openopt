"""Disposable real-Git tests; no remote, local credentials or GPU access."""

from pathlib import Path
import subprocess

import pytest

from scripts.audit_git_delivery import content_hits, inspect_history, source_bytes


def command(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True, check=False)
    assert result.returncode == 0
    return result.stdout.decode().strip()


@pytest.fixture
def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    command(root, "init")
    command(root, "config", "user.name", "test")
    command(root, "config", "user.email", "test@example.invalid")
    (root / "sample.txt").write_bytes(b"baseline\n")
    command(root, "add", "--", "sample.txt")
    command(root, "commit", "-m", "baseline")
    return root, command(root, "rev-parse", "HEAD")


def add_version(root, value):
    (root / "sample.txt").write_bytes(value)
    command(root, "add", "--", "sample.txt")
    command(root, "commit", "-m", "version")
    return command(root, "rev-parse", "HEAD")


def test_positive_real_history_membership(repository):
    root, base = repository
    head = add_version(root, b"ordinary research code\n")
    result = inspect_history(root, base, head, [b"synthetic-needle-for-test"])
    assert result["verdict"] == "PASS" and result["commits"] == 1
    assert result["blobs"] == 1 and result["objects"] == 3
    assert result["findings"] == [] and all(v["bytes"] > 0 for v in result["inventory"].values())


def test_removed_historical_literal_still_rejects(repository):
    root, base = repository
    add_version(root, b"synthetic-needle-for-test\n")
    head = add_version(root, b"clean current content\n")
    result = inspect_history(root, base, head, [b"synthetic-needle-for-test"])
    assert result["verdict"] == "FAIL" and result["commits"] == 2
    assert result["blobs"] == 2 and result["findings"][0]["literal"] == 1
    assert "synthetic-needle-for-test" not in repr(result)


def test_binary_literal_cannot_hide_behind_decoding():
    assert content_hits(b"\xffhidden\x00needle\xfe", [b"needle"])["literal"] == 1


@pytest.mark.parametrize("payload", [
    b"-----BEGIN " + b"PRIVATE KEY-----",
    b"-----BEGIN " + b"OPENSSH PRIVATE KEY-----",
    b"AKIA" + b"A" * 16,
    b"sk-" + b"a" * 24,
    b"pass" + b"word = 'synthetic-value-only'",
])
def test_obvious_shapes_detected(payload):
    assert content_hits(payload, [b"unused-synthetic-needle"])["shape"] > 0


@pytest.mark.parametrize("needles", [[], [b""], ["string-not-bytes"]])
def test_empty_or_invalid_needles_fail_loud(needles):
    with pytest.raises(ValueError, match="byte needles"):
        content_hits(b"clean", needles)


@pytest.mark.parametrize("bad", ["abc", "0" * 40, None])
def test_bad_commit_rejected(repository, bad):
    root, base = repository
    add_version(root, b"clean\n")
    with pytest.raises(ValueError):
        inspect_history(root, base, bad, [b"test-only"])


def test_empty_history_rejected(repository):
    root, base = repository
    with pytest.raises(ValueError, match="empty outgoing"):
        inspect_history(root, base, base, [b"test-only"])


def test_reversed_history_rejected(repository):
    root, base = repository
    head = add_version(root, b"clean\n")
    with pytest.raises(ValueError, match="Git query failed"):
        inspect_history(root, head, base, [b"test-only"])


def test_raw_blob_not_hidden_by_local_replace(repository):
    root, base = repository
    bad = add_version(root, b"test-only-replacement-needle\n")
    bad_blob = command(root, "rev-parse", bad + ":sample.txt")
    head = add_version(root, b"ordinary content\n")
    replacement = command(root, "rev-parse", head + ":sample.txt")
    command(root, "replace", bad_blob, replacement)
    result = inspect_history(root, base, head, [b"test-only-replacement-needle"])
    assert result["verdict"] == "FAIL"
    assert any(item["object"] == bad_blob for item in result["findings"])


def test_shallow_history_rejected(repository):
    root, base = repository
    head = add_version(root, b"ordinary content\n")
    shallow = Path(command(root, "rev-parse", "--git-path", "shallow"))
    if not shallow.is_absolute():
        shallow = root / shallow
    shallow.write_text(base + "\n")
    with pytest.raises(ValueError, match="non-shallow"):
        inspect_history(root, base, head, [b"test-only"])


def test_legacy_graft_history_rejected(repository):
    root, base = repository
    head = add_version(root, b"ordinary content\n")
    grafts = Path(command(root, "rev-parse", "--git-path", "info/grafts"))
    if not grafts.is_absolute():
        grafts = root / grafts
    grafts.write_text(head + "\n")
    with pytest.raises(ValueError, match="grafted"):
        inspect_history(root, base, head, [b"test-only"])


@pytest.mark.parametrize("flag", ("--assume-unchanged", "--skip-worktree"))
def test_hidden_source_bytes_rejected(repository, flag):
    root, _ = repository
    head = add_version(root, b"ordinary content\n")
    command(root, "update-index", flag, "--", "sample.txt")
    (root / "sample.txt").write_bytes(b"changed bytes\n")
    assert command(root, "status", "--porcelain") == ""
    with pytest.raises(ValueError, match="committed blobs"):
        source_bytes(root, head)
