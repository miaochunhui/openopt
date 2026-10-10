"""Discovery is not license qualification; fail-loud and privacy controls."""

import builtins
import importlib.metadata
import json
import socket
from urllib.error import HTTPError

import pytest

from scripts import commercial_probe as cp


def metadata_missing(name):
    raise importlib.metadata.PackageNotFoundError(name)


def test_missing_sdks_are_unverified(monkeypatch):
    monkeypatch.setattr(cp.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cp.importlib.metadata, "version", metadata_missing)
    report = cp.probe()
    assert report["verdict"] == "UNVERIFIED"
    assert report["collection_status"] == "COMPLETE"
    assert len(report["sdks"]) == 4
    for result in report["sdks"].values():
        assert result["sdk_status"] == "UNVERIFIED"
        assert result["reason"] == "SDK_NOT_DISCOVERED"
        assert result["licensed_run"] == "UNVERIFIED"


def test_installed_sdk_never_proves_licensed_run(monkeypatch):
    monkeypatch.setattr(cp.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cp.importlib.metadata, "version", lambda name: "13.0.3")
    report = cp.probe()
    assert report["verdict"] == report["qualification"] == "UNVERIFIED"
    for result in report["sdks"].values():
        assert result["sdk_status"] == "AVAILABLE"
        assert result["licensed_run"] == "UNVERIFIED"
        assert result["solve_attempted"] is False
        assert result["license_connection_attempted"] is False


def test_spec_without_metadata_is_unverified(monkeypatch):
    monkeypatch.setattr(cp.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cp.importlib.metadata, "version", metadata_missing)
    result = cp.probe_sdk("mosek")
    assert result["module_available"] is True
    assert result["sdk_status"] == "UNVERIFIED"
    assert result["reason"] == "VERSION_METADATA_NOT_DISCOVERED"


@pytest.mark.parametrize("stage", ["find_spec", "distribution_version"])
def test_discovery_error_fails_without_leaking_message(monkeypatch, stage):
    private_message = "PRIVATE_SENTINEL /private/license/file private-license-host"

    def broken(name):
        raise RuntimeError(private_message)

    monkeypatch.setattr(cp.importlib.util, "find_spec", broken if stage == "find_spec" else lambda name: None)
    monkeypatch.setattr(cp.importlib.metadata, "version", broken if stage == "distribution_version" else metadata_missing)
    report = cp.probe(("cplex",))
    assert report["collection_status"] == report["verdict"] == "FAIL"
    assert report["sdks"]["cplex"]["errors"][0]["stage"] == stage
    assert private_message not in json.dumps(report)
    assert "PRIVATE_SENTINEL" not in json.dumps(report)


@pytest.mark.parametrize("version", ["", "/private/path", "license@host", "1.0\nPRIVATE", None])
def test_invalid_version_cannot_leak_into_report(monkeypatch, version):
    monkeypatch.setattr(cp.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(cp.importlib.metadata, "version", lambda name: version)
    assert cp.probe(("gurobipy",))["verdict"] == "FAIL"


@pytest.mark.parametrize("selection", [(), ("mosek", "mosek"), ("unknown",)])
def test_empty_duplicate_unknown_selection_rejected(selection):
    with pytest.raises(ValueError):
        cp.probe(selection)


def test_offline_probe_never_imports_solver_or_opens_socket(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert name.split(".")[0] not in cp.SDKS
        return original_import(name, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("network or solver import was attempted")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(cp, "urlopen", forbidden)
    monkeypatch.setenv("GRB_LICENSE_FILE", "PRIVATE_SENTINEL")
    monkeypatch.setenv("MOSEKLM_LICENSE_FILE", "/private/PRIVATE_SENTINEL")
    monkeypatch.setenv("ILOG_LICENSE_FILE", "PRIVATE_SENTINEL@private-host")
    report = cp.probe()
    assert "PRIVATE_SENTINEL" not in json.dumps(report)
    assert report["license_checked"] is False
    assert report["network_requested"] is False


def test_cli_persists_actual_report_and_does_not_clobber(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(cp.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(cp.importlib.metadata, "version", metadata_missing)
    assert cp.main(["--artifact-dir", str(tmp_path)]) == 0
    assert cp.main(["--artifact-dir", str(tmp_path)]) == 0
    files = list(tmp_path.glob("sdk-*/result.json"))
    assert len(files) == 2
    for path in files:
        report = json.loads(path.read_text())
        assert report["verdict"] == "UNVERIFIED"
        assert len(report["sdks"]) == 4
        assert "verdict=UNVERIFIED" in path.with_name("execution.log").read_text()
    assert "COMMERCIAL_PROBE verdict=UNVERIFIED" in capsys.readouterr().out


def test_cli_failure_has_nonzero_exit_and_external_first_cause(monkeypatch, tmp_path):
    def broken(name):
        raise OSError("PRIVATE_SENTINEL")

    monkeypatch.setattr(cp.importlib.util, "find_spec", broken)
    monkeypatch.setattr(cp.importlib.metadata, "version", metadata_missing)
    assert cp.main(["--artifact-dir", str(tmp_path)]) == 1
    path = next(tmp_path.glob("sdk-*/result.json"))
    report = json.loads(path.read_text())
    assert report["sdks"]["mosek"]["errors"][0]["type"] == "OSError"
    assert "PRIVATE_SENTINEL" not in path.read_text()


def test_artifacts_cannot_be_written_in_worktree():
    with pytest.raises(ValueError, match="outside"):
        cp.main(["--artifact-dir", str(cp.ROOT / "forbidden-probe-output")])


@pytest.mark.parametrize("url", [
    "http://docs.gurobi.com/page", "https://example.org/page",
    "https://user:private@docs.mosek.com/page", "https://docs.mosek.com:8000/page",
])
def test_document_mode_rejects_nonpublic_targets_before_network(monkeypatch, url):
    def forbidden(*args, **kwargs):
        raise AssertionError("network attempted")

    monkeypatch.setattr(cp, "urlopen", forbidden)
    with pytest.raises(ValueError, match="public"):
        cp.read_document(url, ["GPU"])


def test_document_text_excludes_script_and_style():
    parser = cp._DocumentText()
    parser.feed("<p>GPU capability</p><script>PRIVATE</script><style>HIDDEN</style>")
    assert "GPU capability" in "".join(parser.parts)
    assert "PRIVATE" not in "".join(parser.parts)
    assert "HIDDEN" not in "".join(parser.parts)


def test_find_spec_does_not_execute_poisoned_sdk(monkeypatch, tmp_path):
    package = tmp_path / "gurobipy"
    package.mkdir()
    (package / "__init__.py").write_text('raise AssertionError("SDK_IMPORTED")\n')
    metadata = tmp_path / "gurobipy-13.0.3.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Name: gurobipy\nVersion: 13.0.3\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    result = cp.probe_sdk("gurobipy")
    assert result["sdk_status"] == "AVAILABLE"
    assert result["versions"]["gurobipy"] == "13.0.3"
    assert result["licensed_run"] == "UNVERIFIED"


def test_http_failure_retains_code_not_private_message():
    error = HTTPError("https://docs.mosek.com/", 404, "PRIVATE_SENTINEL", {}, None)
    result = cp._safe_error(error, "public_document")
    assert result["http_status"] == 404
    assert "PRIVATE_SENTINEL" not in json.dumps(result)


def test_document_redirect_rejected_before_opening_new_target():
    with pytest.raises(ValueError, match="public"):
        cp._PublicRedirect().redirect_request(
            None, None, 302, "Found", {}, "https://private-license-host/",
        )
