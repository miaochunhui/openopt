"""Frozen-source acceptance must reject mutations, not only pytest failures."""

import subprocess

import pytest

from scripts.verify import check_source_consistency, source_snapshot


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True,
    ).stdout


def repository(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.email", "fixture@example.invalid")
    git(root, "config", "user.name", "Fixture")
    (root / "solver.py").write_text("VALUE = 1\n")
    git(root, "add", "--", "solver.py")
    git(root, "commit", "-q", "-m", "Fixture")
    return root


def test_frozen_source_preserves_committed_and_untracked_membership(tmp_path):
    root = repository(tmp_path)
    (root / "new_test.py").write_text("VALUE = 2\n")
    snapshot = source_snapshot(root)
    assert set(snapshot["sha256"]) == {"solver.py", "new_test.py"}
    assert not snapshot["clean"]
    assert check_source_consistency(snapshot, source_snapshot(root))["verdict"] == "PASS"


@pytest.mark.parametrize("mutation", ["changed", "missing", "extra", "commit"])
def test_source_consistency_rejects_each_mismatch(tmp_path, mutation):
    root = repository(tmp_path)
    before = source_snapshot(root)
    if mutation == "changed":
        (root / "solver.py").write_text("VALUE = 3\n")
    elif mutation == "missing":
        git(root, "rm", "--", "solver.py")
        (root / "replacement.py").write_text("VALUE = 1\n")
    elif mutation == "extra":
        (root / "unexpected.py").write_text("VALUE = 2\n")
    else:
        git(root, "commit", "--allow-empty", "-q", "-m", "New identity")
    check = check_source_consistency(before, source_snapshot(root))
    assert check["verdict"] == "FAIL"
    if mutation == "commit":
        assert not check["same_commit"]
    else:
        assert check[mutation]


def test_source_snapshot_rejects_symlink_and_empty_repo(tmp_path):
    root = repository(tmp_path)
    (root / "link.py").symlink_to(root / "solver.py")
    with pytest.raises(ValueError, match="symlink"):
        source_snapshot(root)
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init", "-q")
    with pytest.raises(ValueError, match="empty"):
        source_snapshot(empty)
