"""Offline positive and negative controls, never credential or network reads."""

import json
from types import SimpleNamespace
import urllib.error

import pytest

from scripts import inspect_github_target as probe


def repository(push):
    return {
        "full_name": "miaochunhui/openopt", "owner": {"login": "miaochunhui"},
        "clone_url": "https://github.com/miaochunhui/openopt.git",
        "permissions": {"push": push},
    }


def test_collaborator_permissions_do_not_require_matching_token_owner():
    data = repository(True)
    data["arbitrary_field"] = "must_not_be_retained"
    result = probe.project(200, data)
    assert probe.validate_result(result)["state"] == "WRITE_PERMISSION_VERIFIED"
    assert set(result) == probe.FIELDS
    assert "arbitrary_field" not in result


@pytest.mark.parametrize("status,state", [
    (404, "MISSING_OR_INACCESSIBLE"), (401, "AUTHENTICATION_REJECTED"),
    (403, "AUTHENTICATION_REJECTED"), (500, "HTTP_FAILURE"),
])
def test_http_status_is_preserved_without_body(status, state):
    result = probe.validate_result(probe.project(status, None))
    assert result["http_status"] == status
    assert result["state"] == state
    assert result["push_permission"] is None
    assert result["verdict"] != "PASS"


def test_read_access_not_push_access():
    result = probe.validate_result(probe.project(200, repository(False)))
    assert result["verdict"] == "BLOCKED"
    assert result["push_permission"] is False


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_permission_must_be_boolean(value):
    with pytest.raises(probe.ProbeError, match="permissions_unavailable"):
        probe.project(200, repository(value))


@pytest.mark.parametrize("field", ["full_name", "clone_url", "owner"])
def test_wrong_target_refused(field):
    data = repository(True)
    data[field] = "other"
    with pytest.raises(probe.ProbeError):
        probe.project(200, data)


@pytest.mark.parametrize("mutation", [
    {"unexpected": "value"}, {"verdict": "PASS"}, {"http_status": 200},
    {"push_permission": True}, {"repository": "other/openopt"},
])
def test_blocked_response_cannot_be_promoted(mutation):
    result = probe.project(404, None)
    result.update(mutation)
    with pytest.raises(probe.ProbeError):
        probe.validate_result(result)


def test_exact_read_only_api_and_projected_response():
    class Response:
        status = 200
        def __init__(self, identity):
            self.identity = identity

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, limit):
            assert limit == 1048577
            return json.dumps({"login": "miaochunhui"} if self.identity
                              else repository(True)).encode()

    class Opener:
        def open(self, request, timeout):
            assert request.full_url in ("https://api.github.com/user",
                                       "https://api.github.com/repos/miaochunhui/openopt")
            assert request.method == "GET"
            assert request.data is None
            assert timeout == 25
            return Response(request.full_url.endswith("/user"))

    assert probe.api_probe("synthetic_test_value_only", opener=Opener())["verdict"] == "PASS"


def test_redirect_is_never_followed():
    with pytest.raises(probe.ProbeError, match="redirect_refused"):
        probe.NoRedirect().redirect_request(None, None, 302, "", {}, "https://other/")


def test_http_error_keeps_original_status():
    class Opener:
        def open(self, request, timeout):
            raise urllib.error.HTTPError(request.full_url, 404, "", {}, None)

    assert probe.api_probe("synthetic_test_value_only", opener=Opener())["http_status"] == 404


def test_wrong_token_account_stops_before_repository_discovery():
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, limit):
            return b'{"login":"wrong_account"}'
    class Opener:
        calls = 0
        def open(self, request, timeout):
            self.calls += 1
            assert request.full_url == "https://api.github.com/user"
            return Response()
    opener = Opener()
    with pytest.raises(probe.ProbeError, match="credential_account_mismatch"):
        probe.api_probe("synthetic_test_value_only", opener=opener)
    assert opener.calls == 1


def test_invalid_cli_never_echoes_parameter_value(capsys):
    marker = "not" + "_a_credential_example"
    with pytest.raises(SystemExit) as error:
        probe.main(["--untrusted-input=" + marker])
    assert error.value.code == 2
    output = capsys.readouterr()
    assert output.out == "" and output.err == "invalid_arguments\n"
    assert marker not in output.err


def test_transport_failure_never_records_stderr():
    def runner(*args, **kwargs):
        return SimpleNamespace(returncode=17, stdout=b"", stderr=b"synthetic_private_text")

    with pytest.raises(probe.ProbeError) as error:
        probe.command(["ssh"], runner=runner)
    assert str(error.value) == "transport_failed_rc_17"


def test_fixed_worker_first_cause_is_preserved():
    result = dict(verdict="FAIL", state="PROBE_FAILURE", repository=probe.TARGET,
                  http_status=None, push_permission=None,
                  first_cause="redirect_refused")
    with pytest.raises(probe.ProbeError, match="redirect_refused"):
        probe.validate_result(result)
    result["first_cause"] = "untrusted arbitrary failure body"
    with pytest.raises(probe.ProbeError, match="response_error_invalid"):
        probe.validate_result(result)


def test_source_failure_occurs_before_token_read():
    class ForbiddenCredential:
        def is_file(self):
            pytest.fail("must not inspect credentials after source failure")

    calls = []

    def runner(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=0, stdout=b"wrong fingerprint", stderr=b"")

    with pytest.raises(probe.ProbeError, match="remote_source_mismatch"):
        probe.inspect(runner=runner, token_path=ForbiddenCredential())
    assert len(calls) == 2
    assert calls[0][0] == "scp"
    assert calls[1][-2] == "sha256sum"


def test_stdin_transport_and_response_binding(tmp_path):
    path = tmp_path / "synthetic-credential"
    synthetic_value = "synthetic_test_value_only"
    path.write_text(synthetic_value)
    calls = []

    def runner(args, **kwargs):
        calls.append((args, kwargs))
        if args[0] == "scp":
            return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")
        if args[-2] == "sha256sum":
            pin = args[-1].split("permission-")[1][:-3]
            raw = (pin + "  " + args[-1] + "\n").encode()
            return SimpleNamespace(returncode=0, stdout=raw, stderr=b"")
        result = probe.project(404, None)
        return SimpleNamespace(returncode=1, stdout=json.dumps(result).encode(), stderr=b"")

    assert probe.inspect(runner=runner, token_path=path)["state"] == "MISSING_OR_INACCESSIBLE"
    assert len(calls) == 3
    assert all(synthetic_value not in " ".join(args) for args, _ in calls)
    assert json.loads(calls[2][1]["input"]) == {"token": synthetic_value}
    assert all(item[0][0] in ("ssh", "scp") for item in calls)
