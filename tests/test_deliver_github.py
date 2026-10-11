"""Isolated delivery controls with real local Git and synthetic API operations."""

import hashlib
import json
import subprocess

import pytest

from scripts import deliver_github as delivery

HEAD = "a" * 40
PIN = "b" * 64


def result():
    return dict(verdict="GITHUB_MASTER_PASS", repository=delivery.TARGET,
                identity="miaochunhui", head=HEAD, bundle_sha256=PIN,
                created=True, push_performed=True, readback="PASS", before=None)


def test_exact_bound_worker_positive():
    assert delivery.checked_worker(result(), HEAD, PIN) == result()


@pytest.mark.parametrize("change", [
    {"head": "c" * 40}, {"identity": "other"}, {"repository": "other/openopt"},
    {"bundle_sha256": "c" * 64}, {"created": 1}, {"push_performed": False},
    {"readback": "FAIL"}, {"before": ""}, {"unexpected": "private_value"},
])
def test_unbound_or_arbitrary_response_refused(change):
    data = result()
    data.update(change)
    with pytest.raises(delivery.DeliveryError):
        delivery.checked_worker(data, HEAD, PIN)


def test_fixed_first_cause_retained():
    with pytest.raises(delivery.DeliveryError, match="command_failed_rc_128"):
        delivery.checked_worker(
            dict(verdict="FAIL", first_cause="command_failed_rc_128"), HEAD, PIN)
    with pytest.raises(delivery.DeliveryError, match="worker_error_invalid"):
        delivery.checked_worker(
            dict(verdict="FAIL", first_cause="untrusted_private_text"), HEAD, PIN)


def test_git_environment_discards_inherited_redirects_and_trace(monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_COUNT", "2")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "url.ssh://other/.insteadOf")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "https://github.com/")
    monkeypatch.setenv("GIT_TRACE_PACKET", "1")
    monkeypatch.setenv("GIT_TRACE_CURL", "1")
    env = delivery.git_env()
    assert env["GIT_CONFIG_COUNT"] == "0"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert "GIT_CONFIG_KEY_0" not in env
    assert "GIT_TRACE_PACKET" not in env and "GIT_TRACE_CURL" not in env


@pytest.mark.parametrize("field", ["full_name", "owner", "clone_url", "permissions"])
def test_wrong_repository_or_missing_permission_rejected(field):
    data = {"full_name": delivery.TARGET, "clone_url": delivery.URL,
            "owner": {"login": "miaochunhui"}, "permissions": {"push": True}}
    delivery.validate_repo(data)
    data[field] = None
    with pytest.raises(delivery.DeliveryError):
        delivery.validate_repo(data)


def test_identity_refusal_before_any_git_or_creation(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.write_bytes(b"synthetic_bundle")
    pin = hashlib.sha256(bundle.read_bytes()).hexdigest()
    calls = []

    def api(token, path, *args):
        calls.append(path)
        return 200, {"login": "wrong"}

    monkeypatch.setattr(delivery, "api", api)
    monkeypatch.setattr(delivery, "git", lambda *args: pytest.fail("unexpected Git call"))
    with pytest.raises(delivery.DeliveryError, match="credential_account_mismatch"):
        delivery.deliver("synthetic_test_value_only", bundle, pin, HEAD, tmp_path)
    assert calls == ["/user"]


def test_wrong_bundle_pin_before_identity(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.write_bytes(b"synthetic_bundle")
    monkeypatch.setattr(delivery, "api", lambda *args: pytest.fail("unexpected API call"))
    with pytest.raises(delivery.DeliveryError, match="bundle_changed"):
        delivery.deliver("synthetic_test_value_only", bundle, PIN, HEAD, tmp_path)


def local_git(repo, *args):
    child = subprocess.run(["/usr/bin/git", "-C", str(repo), *args], capture_output=True)
    assert child.returncode == 0
    return child.stdout.decode().strip()


def test_real_git_guard_and_url_rewrite_rejection(tmp_path):
    local_git(tmp_path, "init", "-b", "master")
    local_git(tmp_path, "-c", "user.name=synthetic", "-c", "user.email=synthetic@example.invalid",
              "commit", "--allow-empty", "-m", "isolated")
    head = local_git(tmp_path, "rev-parse", "HEAD")
    delivery.local_guard(tmp_path, head)
    local_git(tmp_path, "config", "url.ssh://unwanted/.insteadOf", "https://github.com/")
    with pytest.raises(delivery.DeliveryError, match="unsafe_local_git_configuration"):
        delivery.local_guard(tmp_path, head)


def test_api_write_scope_rejected_before_network():
    with pytest.raises(delivery.DeliveryError, match="api_scope_refused"):
        delivery.api("synthetic_test_value_only", "/user", "POST", {})
    with pytest.raises(delivery.DeliveryError, match="creation_payload_refused"):
        delivery.api("synthetic_test_value_only", "/user/repos", "POST", {"name": "wrong"})


def test_source_guard_actual_bytes_and_escape(tmp_path):
    path = tmp_path / "source"
    path.write_bytes(b"original")
    files = {"source": hashlib.sha256(b"original").hexdigest()}
    delivery.source_guard(tmp_path, files)
    path.write_bytes(b"changed")
    with pytest.raises(delivery.DeliveryError, match="audit_source_changed"):
        delivery.source_guard(tmp_path, files)
    with pytest.raises(delivery.DeliveryError, match="audit_source_empty"):
        delivery.source_guard(tmp_path, {})
    with pytest.raises(delivery.DeliveryError, match="audit_source_escape"):
        delivery.source_guard(tmp_path, {"../escape": PIN})


def test_audit_false_empty_cannot_satisfy_integer_count(tmp_path):
    payload = dict(verdict="READER_INTEGRITY_PASS", content_verdict="BOUNDED_CONTENT_PASS",
                   audit=dict(head=HEAD, counts={"commits": False, "objects": False}))
    path = tmp_path / "audit"
    path.write_text(json.dumps(payload))
    with pytest.raises(delivery.DeliveryError, match="audit_empty"):
        delivery.audit_guard(path, delivery.digest(path), HEAD)
