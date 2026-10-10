"""Provenance must protect the initial import without freezing development."""

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts import migrate_openpo as migration


def git(root, *args):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def snapshot(tmp_path, monkeypatch):
    source = tmp_path / "src/openpo"
    source.mkdir(parents=True)
    module = source / "__init__.py"
    content = b"ORIGINAL = True\n"
    module.write_bytes(content)
    docs = tmp_path / "docs"
    docs.mkdir()
    manifest = docs / "migration-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 1,
        "source_commit": migration.SOURCE_COMMIT,
        "files": [{"path": "src/openpo/__init__.py", "bytes": len(content),
                   "sha256": hashlib.sha256(content).hexdigest()}],
    }))
    git(tmp_path, "init")
    git(tmp_path, "config", "user.name", "Migration Test")
    git(tmp_path, "config", "user.email", "migration-test@example.invalid")
    monkeypatch.setattr(migration, "ROOT", tmp_path)
    monkeypatch.setattr(migration, "MANIFEST", manifest)
    return tmp_path, module, manifest


def test_uncommitted_snapshot_checks_real_worktree(snapshot):
    root, module, _ = snapshot
    migration.verify_snapshot(exact_worktree=False)
    module.write_text("ORIGINAL = False\n")
    with pytest.raises(ValueError, match="differs from manifest"):
        migration.verify_snapshot(exact_worktree=False)


def test_provenance_survives_independent_source_edits(snapshot):
    root, module, _ = snapshot
    git(root, "add", "--", "src/openpo/__init__.py", "docs/migration-manifest.json")
    git(root, "commit", "-m", "Initial import")
    module.write_text("ORIGINAL = False\n")
    (module.parent / "new_algorithm.py").write_text("NEW = True\n")
    migration.verify_snapshot(exact_worktree=False)
    with pytest.raises(ValueError, match="differs from manifest"):
        migration.verify_snapshot()
    git(root, "add", "--", "src/openpo/__init__.py", "src/openpo/new_algorithm.py")
    git(root, "commit", "-m", "Independent algorithm edit")
    migration.verify_snapshot(exact_worktree=False)


def test_provenance_rejects_tampered_manifest(snapshot):
    root, _, manifest = snapshot
    git(root, "add", "--", "src/openpo/__init__.py", "docs/migration-manifest.json")
    git(root, "commit", "-m", "Initial import")
    data = json.loads(manifest.read_text())
    data["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="differs from manifest"):
        migration.verify_snapshot(exact_worktree=False)


def test_snapshot_rejects_unlisted_python_file(snapshot):
    _, module, _ = snapshot
    (module.parent / "unlisted.py").write_text("EXTRA = True\n")
    with pytest.raises(ValueError, match="source set mismatch"):
        migration.verify_snapshot()


def test_snapshot_rejects_path_escape(snapshot):
    with pytest.raises(ValueError, match="unsafe manifest path"):
        migration.local_path("../outside.py")
