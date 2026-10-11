"""No-network ref controls, including actual local Git repositories."""

import os
import subprocess

import pytest

from scripts import inspect_repository_refs as refs


def git(repo, *args):
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, check=False,
        env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null"},
    )
    assert result.returncode == 0, "isolated Git fixture failed"
    return result.stdout


@pytest.fixture
def repository(tmp_path):
    repo = tmp_path / "isolated"
    repo.mkdir()
    git(repo, "init", "-b", "master")
    git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
        "commit", "--allow-empty", "-m", "isolated ref fixture")
    for name, url in refs.REMOTES.items():
        git(repo, "remote", "add", name, url)
    git(repo, "config", "branch.master.remote", "origin")
    git(repo, "config", "branch.master.merge", "refs/heads/master")
    return repo, git(repo, "rev-parse", "HEAD").decode().strip()


def transport(head, calls, *, mismatch=False, fail=False):
    def run(command, **kwargs):
        if "ls-remote" not in command:
            return subprocess.run(command, **kwargs)
        name = command[-2]
        calls.append((name, kwargs["env"]))
        assert name in ("origin", "gitee"), "GitHub must not be contacted"
        assert "push" not in command
        value = "0" * 40 if mismatch and name == "gitee" else head
        return subprocess.CompletedProcess(
            command, 1 if fail else 0,
            (value + "\trefs/heads/master\n").encode(),
            b"non-content fixture",
        )
    return run


def test_actual_local_repository_and_two_readonly_transports(repository):
    repo, head = repository
    calls = []
    result = refs.inspect(repo, head, runner=transport(head, calls))
    assert result["verdict"] == "PRIMARY_AND_GITEE_PASS"
    assert result["three_host_complete"] is False
    assert result["github"]["state"] == "NOT_VERIFIED"
    assert result["github"]["credential_read"] is False
    assert result["github"]["network_attempted"] is False
    assert [name for name, _ in calls] == ["origin", "gitee"]
    assert calls[0][1]["GIT_SSH_COMMAND"] == refs.GITLAB_SSH
    assert all(env["GIT_TERMINAL_PROMPT"] == "0" for _, env in calls)


def test_remote_mismatch_remains_failure(repository):
    repo, head = repository
    result = refs.inspect(repo, head, runner=transport(head, [], mismatch=True))
    assert result["verdict"] == "REF_MISMATCH"
    assert result["remotes"]["gitee"]["matches_expected"] is False
    assert result["three_host_complete"] is False


def test_remote_failure_does_not_leak_child_content(repository):
    repo, head = repository
    with pytest.raises(refs.InspectionError, match="^git_query_failed$"):
        refs.inspect(repo, head, runner=transport(head, [], fail=True))


@pytest.mark.parametrize("payload", [
    b"", b"0" * 40 + b"\trefs/heads/main\n",
    b"0" * 40 + b"\trefs/heads/master\n" + b"1" * 40 + b"\trefs/heads/master\n",
    b"short\trefs/heads/master\n",
])
def test_missing_wrong_ambiguous_and_short_refs_reject(payload):
    with pytest.raises(refs.InspectionError):
        refs.remote_head(payload)


@pytest.mark.parametrize("expected", [None, "short", True])
def test_unpinned_source_never_queries_network(repository, expected):
    repo, _ = repository
    calls = []
    with pytest.raises(refs.InspectionError, match="explicit_full_head_required"):
        refs.inspect(repo, expected, runner=transport("0" * 40, calls))
    assert calls == []


def test_wrong_head_rejects_before_network(repository):
    repo, head = repository
    calls = []
    assert head != "0" * 40
    with pytest.raises(refs.InspectionError, match="local_head_mismatch"):
        refs.inspect(repo, "0" * 40, runner=transport(head, calls))
    assert calls == []


def test_dirty_source_rejects_before_network(repository):
    repo, head = repository
    (repo / "untracked").mkdir()
    calls = []
    # Empty directories are not Git changes; staging a gitlink would be a change.
    git(repo, "-c", "core.bare=false", "update-index", "--add", "--cacheinfo",
        "160000," + head + ",fixture")
    with pytest.raises(refs.InspectionError, match="dirty_source"):
        refs.inspect(repo, head, runner=transport(head, calls))
    assert calls == []


def test_multiple_push_urls_reject_before_network(repository):
    repo, head = repository
    git(repo, "config", "--add", "remote.origin.pushurl", refs.REMOTES["origin"])
    git(repo, "config", "--add", "remote.origin.pushurl", "ssh://fixture.invalid/other")
    calls = []
    with pytest.raises(refs.InspectionError, match="remote_configuration_mismatch"):
        refs.inspect(repo, head, runner=transport(head, calls))
    assert calls == []


def test_source_change_during_queries_rejects(repository):
    repo, head = repository
    original = transport(head, [])

    def changed(command, **kwargs):
        result = original(command, **kwargs)
        if "ls-remote" in command and command[-2] == "gitee":
            git(repo, "-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid",
                "commit", "--allow-empty", "-m", "isolated drift")
        return result

    with pytest.raises(refs.InspectionError, match="local_head_changed"):
        refs.inspect(repo, head, runner=changed)


def test_timeout_is_not_success(repository):
    repo, _ = repository

    def timeout(command, **kwargs):
        raise subprocess.TimeoutExpired(command, 30, output=b"withheld")

    with pytest.raises(refs.InspectionError, match="^git_query_timeout$"):
        refs.query(repo, ["rev-parse", "HEAD"], runner=timeout)
