from __future__ import annotations

from dataclasses import replace
from http import HTTPStatus
from pathlib import Path
import json
import subprocess
import sys
from threading import Event, Thread
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from openpo.certification_policy import CertificationDisposition, CertificationMode, CertificationPolicy
from openpo.durable_shadow import DurableShadowCertificationService
from openpo.facade import solve_production
from openpo.shadow_codec import request_to_payload
from openpo.shadow_server import (
    ShadowHTTPApplication,
    ShadowHTTPServer,
    build_parser,
    validate_args,
    _admission_to_payload,
)
from openpo.durable_shadow import ShadowAdmissionSnapshot
from openpo.shadow_journal import ShadowCapacitySnapshot
from openpo.service_time_calibration import ServiceTimeCalibrationSnapshot
from tests.test_facade import _request


def _post(
    base_url: str,
    path: str,
    payload: dict[str, object],
    *,
    bearer_token: str | None = None,
) -> tuple[int, dict[str, object]]:
    headers = {"Content-Type": "application/json"}
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=20.0) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def _post_with_headers(
    base_url: str,
    path: str,
    payload: dict[str, object],
) -> tuple[int, dict[str, object], dict[str, str]]:
    request = Request(
        f"{base_url}{path}",
        data=json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=20.0) as response:
        body = json.loads(response.read().decode("utf-8"))
        return response.status, body, dict(response.headers.items())


def _wait_ready(base_url: str, process: subprocess.Popen[bytes] | None = None) -> None:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            stderr = b"" if process.stderr is None else process.stderr.read()
            raise RuntimeError(
                f"shadow server exited early with {process.returncode}: "
                f"{stderr.decode('utf-8', errors='replace')}"
            )
        try:
            with urlopen(f"{base_url}/readyz", timeout=0.5) as response:
                if response.status == HTTPStatus.OK:
                    return
        except (HTTPError, URLError, TimeoutError):
            pass
        time.sleep(0.02)
    raise TimeoutError(f"shadow server did not become ready at {base_url}")


def _start_process(journal: Path, ready_file: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "openpo.shadow_server",
            "--journal",
            str(journal),
            "--port",
            "0",
            "--ready-file",
            str(ready_file),
            "--lease-seconds",
            "2",
            "--heartbeat-seconds",
            "0.2",
        ],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )


def _wait_process_ready(
    ready_file: Path,
    process: subprocess.Popen[bytes],
) -> str:
    deadline = time.monotonic() + 20.0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = b"" if process.stderr is None else process.stderr.read()
            raise RuntimeError(
                f"shadow server exited early with {process.returncode}: "
                f"{stderr.decode('utf-8', errors='replace')}"
            )
        if ready_file.exists():
            payload = json.loads(ready_file.read_text(encoding="utf-8"))
            assert payload["schema"] == "shadow-server-ready-v1"
            assert payload["pid"] == process.pid
            base_url = f"http://127.0.0.1:{int(payload['port'])}"
            _wait_ready(base_url, process)
            return base_url
        time.sleep(0.02)
    raise TimeoutError("shadow server did not publish its bound address")


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        return_code = process.wait(timeout=20.0)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10.0)
        raise
    assert return_code == 0


def test_application_rejects_submit_after_drain(tmp_path) -> None:
    service = DurableShadowCertificationService(tmp_path / "shadow.sqlite3")
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)

    assert application.begin_drain() is True
    assert application.begin_drain() is False
    status, response = application.submit(
        {
            "submission_id": "order-1",
            "request": request_to_payload(_request(exact_backend="osqp")),
        }
    )

    assert status == HTTPStatus.SERVICE_UNAVAILABLE
    assert response["error"] == "DRAINING"
    assert application.health()["status"] == "DRAINING"
    service.shutdown()


def test_http_submit_collect_ack_roundtrip(tmp_path) -> None:
    service = DurableShadowCertificationService(tmp_path / "shadow.sqlite3")
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    server = ShadowHTTPServer(
        ("127.0.0.1", 0),
        application,
        max_body_bytes=8 * 1024 * 1024,
        request_timeout_seconds=10.0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_ready(base_url)
        request = _request(exact_backend="osqp")
        status, submission, response_headers = _post_with_headers(
            base_url,
            "/v1/submit",
            {
                "submission_id": "order-1",
                "request": request_to_payload(request),
            },
        )
        assert status == HTTPStatus.OK
        server_timing = response_headers["Server-Timing"]
        assert "request_decode;dur=" in server_timing
        assert "application;dur=" in server_timing
        assert "response_encode;dur=" in server_timing
        ticket = submission["ticket"]
        changed = replace(
            request,
            fast_config=replace(
                request.fast_config,
                iterations=request.fast_config.iterations + 20,
            ),
        )
        status, conflict = _post(
            base_url,
            "/v1/submit",
            {
                "submission_id": "order-1",
                "request": request_to_payload(changed),
            },
        )
        assert status == HTTPStatus.CONFLICT
        assert conflict["error"] == "SUBMISSION_CONFLICT"
        status, collected = _post(
            base_url,
            "/v1/collect",
            {"ticket": ticket, "timeout_seconds": 10.0},
        )
        assert status == HTTPStatus.OK
        assert collected["status"] == "SUCCEEDED"
        assert collected["result"]["durable"] is True
        status, acknowledged = _post(base_url, "/v1/ack", {"ticket": ticket})
        assert status == HTTPStatus.OK
        assert acknowledged["acknowledged"] is True
    finally:
        application.begin_drain()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10.0)
        service.shutdown()


def test_policy_http_v2_full_returns_ticket_and_certification(tmp_path) -> None:
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        certification_policy=CertificationPolicy(),
    )
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    try:
        status, response = application.submit(
            {
                "submission_id": "full-order",
                "request": request_to_payload(_request(exact_backend="osqp")),
            }
        )
        assert status == HTTPStatus.OK
        assert response["schema"] == "shadow-submit-v2"
        assert response["ticket"] is not None
        assert response["certification"]["disposition"] == "queued"
        assert response["certification"]["reasons"] == ["full"]
        application.collect(
            {"ticket": response["ticket"], "timeout_seconds": 10.0}
        )
        application.ack({"ticket": response["ticket"]})
    finally:
        service.shutdown()


def test_policy_http_v2_skip_is_explicit_and_has_no_ticket(tmp_path) -> None:
    policy = CertificationPolicy(
        mode=CertificationMode.SAMPLED,
        sample_numerator=1,
        sample_denominator=4,
    )
    request = _request(exact_backend="osqp")
    for iterations in range(
        request.fast_config.iterations,
        request.fast_config.iterations + 100,
    ):
        candidate = replace(
            request,
            fast_config=replace(request.fast_config, iterations=iterations),
        )
        result = solve_production(candidate)
        if policy.sample_bucket(result.request_sha256) >= policy.sample_numerator:
            request = candidate
            break
    else:
        raise AssertionError("failed to construct a deterministic skipped request")

    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        certification_policy=policy,
    )
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    try:
        status, response = application.submit(
            {
                "submission_id": "skipped-order",
                "request": request_to_payload(request),
            }
        )
        assert status == HTTPStatus.OK
        assert response["schema"] == "shadow-submit-v2"
        assert response["ticket"] is None
        assert response["certification"]["disposition"] == (
            CertificationDisposition.SKIPPED.value
        )
        assert response["certification"]["reasons"] == ["hash_not_selected"]
        assert service.outstanding_count == 0
    finally:
        service.shutdown()


def test_policy_http_route_circuit_open_returns_503(tmp_path) -> None:
    policy = CertificationPolicy()
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        certification_policy=policy,
    )
    assert service._policy_store is not None
    assert service._policy_store.open_circuit(
        route_key="cpu",
        policy_sha256=policy.policy_sha256,
        evidence_sha256="0" * 64,
        reason="test_divergence",
    )
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    server = ShadowHTTPServer(
        ("127.0.0.1", 0),
        application,
        max_body_bytes=8 * 1024 * 1024,
        request_timeout_seconds=10.0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        _wait_ready(base_url)
        status, response = _post(
            base_url,
            "/v1/submit",
            {
                "submission_id": "blocked-order",
                "request": request_to_payload(_request(exact_backend="osqp")),
            },
        )
        assert status == HTTPStatus.SERVICE_UNAVAILABLE
        assert response["error"] == "ROUTE_CIRCUIT_OPEN"
        assert "test_divergence" in response["message"]
    finally:
        application.begin_drain()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10.0)
        service.shutdown()


def test_http_backpressure_returns_capacity_snapshot(tmp_path) -> None:
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        max_pending=1,
    )
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    server = ShadowHTTPServer(
        ("127.0.0.1", 0),
        application,
        max_body_bytes=8 * 1024 * 1024,
        request_timeout_seconds=10.0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload = request_to_payload(_request(exact_backend="osqp"))
    try:
        _wait_ready(base_url)
        status, first = _post(
            base_url,
            "/v1/submit",
            {"submission_id": "order-1", "request": payload},
        )
        assert status == HTTPStatus.OK
        status, rejected = _post(
            base_url,
            "/v1/submit",
            {"submission_id": "order-2", "request": payload},
        )
        assert status == HTTPStatus.TOO_MANY_REQUESTS
        assert rejected["error"] == "CERTIFICATION_BACKPRESSURE"
        assert rejected["reasons"] == ["capacity_exhausted"]
        assert rejected["capacity"]["outstanding"] == 1
        assert rejected["capacity"]["available"] == 0
        assert rejected["capacity"]["schema"] == "shadow-capacity-v2"
        assert rejected["capacity"]["worker_count"] == 1
        assert rejected["capacity"]["active_workers"] in {0, 1}
        application.collect({"ticket": first["ticket"], "timeout_seconds": 10.0})
        application.ack({"ticket": first["ticket"]})
    finally:
        application.begin_drain()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10.0)
        service.shutdown()


def test_http_drain_backpressure_returns_frozen_worker_snapshot(
    tmp_path, monkeypatch
) -> None:
    from openpo import durable_shadow as durable_module

    original = durable_module.certify_exact
    release = Event()

    def blocked_certify(request):
        if not release.wait(timeout=10.0):
            raise TimeoutError("blocked certification was not released")
        return original(request)

    monkeypatch.setattr(durable_module, "certify_exact", blocked_certify)
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        max_pending=4,
        worker_count=1,
        certification_service_time_budget_seconds=10.0,
        max_drain_seconds=10.0,
    )
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    server = ShadowHTTPServer(
        ("127.0.0.1", 0),
        application,
        max_body_bytes=8 * 1024 * 1024,
        request_timeout_seconds=10.0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload = request_to_payload(_request(exact_backend="osqp"))
    first = None
    try:
        _wait_ready(base_url)
        status, first = _post(
            base_url,
            "/v1/submit",
            {"submission_id": "drain-1", "request": payload},
        )
        assert status == HTTPStatus.OK
        status, rejected = _post(
            base_url,
            "/v1/submit",
            {"submission_id": "drain-2", "request": payload},
        )
        assert status == HTTPStatus.TOO_MANY_REQUESTS
        assert rejected["schema"] == "shadow-error-v2"
        assert rejected["error"] == "CERTIFICATION_BACKPRESSURE"
        assert rejected["reasons"] == ["drain_time_exceeded"]
        capacity = rejected["capacity"]
        assert capacity["schema"] == "shadow-capacity-v2"
        assert capacity["worker_count"] == 1
        assert capacity["active_workers"] == 1
        assert capacity["outstanding"] == 1
        assert capacity["available"] == 3
        assert float.fromhex(capacity["service_time_budget_seconds"]) == 10.0
        assert float.fromhex(capacity["max_drain_seconds"]) == 10.0
        assert float.fromhex(capacity["estimated_drain_seconds"]) == 10.0
    finally:
        release.set()
        if first is not None:
            application.collect({"ticket": first["ticket"], "timeout_seconds": 10.0})
            application.ack({"ticket": first["ticket"]})
        application.begin_drain()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10.0)
        service.shutdown()


def test_http_circuit_reset_requires_enablement_and_preserves_audit(tmp_path) -> None:
    policy = CertificationPolicy()
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        certification_policy=policy,
        circuit_reset_cooldown_seconds=0.0,
    )
    assert service._policy_store is not None
    assert service._policy_store.open_circuit(
        route_key="cpu",
        policy_sha256=policy.policy_sha256,
        evidence_sha256="1" * 64,
        reason="test_divergence",
    )
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    server = ShadowHTTPServer(
        ("127.0.0.1", 0),
        application,
        max_body_bytes=8 * 1024 * 1024,
        request_timeout_seconds=10.0,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    reset_payload = {
        "route_key": "cpu",
        "expected_evidence_sha256": "1" * 64,
        "approval_sha256": "2" * 64,
        "actor": "risk-oncall",
        "reason": "incident reviewed",
    }
    try:
        _wait_ready(base_url)
        status, response = _post(base_url, "/v1/reset-circuit", reset_payload)
        assert status == HTTPStatus.OK
        assert response["changed"] is True
        assert response["reset"]["approval_sha256"] == "2" * 64
        status, replay = _post(base_url, "/v1/reset-circuit", reset_payload)
        assert status == HTTPStatus.OK
        assert replay["changed"] is False
        assert replay["reset"] == response["reset"]
    finally:
        application.begin_drain()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10.0)
        service.shutdown()


def test_sigterm_restart_preserves_collect_and_ack(tmp_path) -> None:
    journal = tmp_path / "shadow.sqlite3"
    first_ready = tmp_path / "first-ready.json"
    first = _start_process(journal, first_ready)
    try:
        first_url = _wait_process_ready(first_ready, first)
        status, submission = _post(
            first_url,
            "/v1/submit",
            {
                "submission_id": "order-1",
                "request": request_to_payload(_request(exact_backend="osqp")),
            },
        )
        assert status == HTTPStatus.OK
        ticket = submission["ticket"]
        _stop_process(first)
    finally:
        if first.poll() is None:
            first.kill()
            first.wait(timeout=10.0)

    second_ready = tmp_path / "second-ready.json"
    second = _start_process(journal, second_ready)
    try:
        second_url = _wait_process_ready(second_ready, second)
        status, collected = _post(
            second_url,
            "/v1/collect",
            {"ticket": ticket, "timeout_seconds": 10.0},
        )
        assert status == HTTPStatus.OK
        assert collected["status"] == "SUCCEEDED"
        assert collected["result"]["ticket"] == ticket
        status, acknowledged = _post(second_url, "/v1/ack", {"ticket": ticket})
        assert status == HTTPStatus.OK
        assert acknowledged["acknowledged"] is True
        _stop_process(second)
    finally:
        if second.poll() is None:
            second.kill()
            second.wait(timeout=10.0)


def test_http_rejects_unknown_fields(tmp_path) -> None:
    service = DurableShadowCertificationService(tmp_path / "shadow.sqlite3")
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)

    with pytest.raises(ValueError, match="fields must be exactly"):
        application.submit(
            {
                "submission_id": "order-1",
                "request": request_to_payload(_request(exact_backend="osqp")),
                "unexpected": True,
            }
        )
    service.shutdown()


def test_http_business_routes_require_configured_bearer_token(tmp_path) -> None:
    service = DurableShadowCertificationService(tmp_path / "shadow.sqlite3")
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    token = "a" * 32
    server = ShadowHTTPServer(
        ("127.0.0.1", 0),
        application,
        max_body_bytes=8 * 1024 * 1024,
        request_timeout_seconds=10.0,
        bearer_token=token,
    )
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    payload = {
        "submission_id": "order-1",
        "request": request_to_payload(_request(exact_backend="osqp")),
    }
    try:
        _wait_ready(base_url)
        status, error = _post(base_url, "/v1/submit", payload)
        assert status == HTTPStatus.UNAUTHORIZED
        assert error["error"] == "UNAUTHORIZED"
        status, submission = _post(
            base_url,
            "/v1/submit",
            payload,
            bearer_token=token,
        )
        assert status == HTTPStatus.OK
        ticket = submission["ticket"]
        application.collect({"ticket": ticket, "timeout_seconds": 10.0})
        application.ack({"ticket": ticket})
    finally:
        application.begin_drain()
        server.shutdown()
        server.server_close()
        thread.join(timeout=10.0)
        service.shutdown()


def test_non_loopback_listener_requires_existing_token_file(tmp_path) -> None:
    parser = build_parser()
    without_token = parser.parse_args(
        ["--journal", str(tmp_path / "shadow.sqlite3"), "--host", "0.0.0.0"]
    )
    with pytest.raises(SystemExit):
        validate_args(without_token, parser)

    token_file = tmp_path / "token"
    token_file.write_text("a" * 32 + "\n", encoding="utf-8")
    with_token = parser.parse_args(
        [
            "--journal",
            str(tmp_path / "shadow.sqlite3"),
            "--host",
            "0.0.0.0",
            "--bearer-token-file",
            str(token_file),
        ]
    )
    validate_args(with_token, parser)


def test_certification_cli_rejects_invalid_sampling_contract(tmp_path) -> None:
    parser = build_parser()
    invalid_full = parser.parse_args(
        [
            "--journal",
            str(tmp_path / "shadow.sqlite3"),
            "--certification-mode",
            "full",
            "--sample-denominator",
            "2",
        ]
    )
    with pytest.raises(SystemExit):
        validate_args(invalid_full, parser)

    valid_anomaly = parser.parse_args(
        [
            "--journal",
            str(tmp_path / "shadow.sqlite3"),
            "--certification-mode",
            "anomaly",
            "--sample-numerator",
            "1",
            "--sample-denominator",
            "8",
        ]
    )
    validate_args(valid_anomaly, parser)


def test_worker_capacity_health_and_cli_contract(tmp_path) -> None:
    service = DurableShadowCertificationService(
        tmp_path / "health.sqlite3",
        max_pending=4,
        worker_count=2,
        certification_service_time_budget_seconds=10.0,
        max_drain_seconds=20.0,
    )
    application = ShadowHTTPApplication(service, max_collect_timeout_seconds=10.0)
    health = application.health()
    assert health["schema"] == "shadow-health-v3"
    assert health["capacity"]["schema"] == "shadow-capacity-v2"
    assert health["capacity"]["worker_count"] == 2
    assert health["capacity"]["active_workers"] == 0
    assert float.fromhex(health["capacity"]["estimated_drain_seconds"]) == 0.0
    service.shutdown()

    parser = build_parser()
    valid = parser.parse_args(
        [
            "--journal",
            str(tmp_path / "server.sqlite3"),
            "--max-pending",
            "4",
            "--worker-count",
            "2",
            "--certification-service-time-budget-seconds",
            "10",
            "--service-time-calibration",
            "--service-time-calibration-minimum-samples",
            "1",
            "--max-drain-seconds",
            "20",
        ]
    )
    validate_args(valid, parser)

    for arguments in (
        ["--worker-count", "5", "--max-pending", "4"],
        ["--certification-service-time-budget-seconds", "10"],
        ["--max-drain-seconds", "20"],
        ["--service-time-calibration"],
        [
            "--certification-service-time-budget-seconds",
            "10",
            "--max-drain-seconds",
            "20",
            "--service-time-calibration",
            "--service-time-calibration-minimum-samples",
            "33",
        ],
        ["--certification-service-time-budget-seconds", "nan", "--max-drain-seconds", "20"],
    ):
        invalid = parser.parse_args(
            ["--journal", str(tmp_path / "invalid.sqlite3"), *arguments]
        )
        with pytest.raises(SystemExit):
            validate_args(invalid, parser)


def test_admission_payload_exposes_service_time_calibration() -> None:
    calibration = ServiceTimeCalibrationSnapshot(
        source="adaptive_quantile",
        static_budget_seconds=10.0,
        effective_budget_seconds=8.0,
        total_sample_count=6,
        retained_sample_count=4,
        window_size=4,
        minimum_samples=4,
        ewma_alpha=0.25,
        ewma_seconds=7.5,
        quantile_probability=0.9,
        quantile_seconds=8.0,
        safety_multiplier=1.0,
        minimum_budget_seconds=5.0,
        maximum_budget_seconds=20.0,
        route_key="service-time-route-v1:" + "a" * 64,
    )
    payload = _admission_to_payload(
        ShadowAdmissionSnapshot(
            capacity=ShadowCapacitySnapshot(
                max_outstanding=4,
                outstanding=2,
                online_ready=1,
                leased=1,
                terminal_unacked=0,
                oldest_outstanding_age_seconds=2.0,
                oldest_inflight_age_seconds=2.0,
                oldest_ready_age_seconds=1.0,
            ),
            worker_count=2,
            active_workers=1,
            service_time_budget_seconds=8.0,
            service_time_budget_source="adaptive_quantile",
            service_time_calibration=calibration,
            max_drain_seconds=20.0,
            estimated_drain_seconds=8.0,
        )
    )

    assert payload["service_time_budget_source"] == "adaptive_quantile"
    encoded = payload["service_time_calibration"]
    assert encoded["source"] == "adaptive_quantile"
    assert encoded["route_key"] == "service-time-route-v1:" + "a" * 64
    assert encoded["total_sample_count"] == 6
    assert encoded["retained_sample_count"] == 4
    assert float.fromhex(encoded["ewma_seconds"]) == 7.5
    assert float.fromhex(encoded["quantile_seconds"]) == 8.0
    assert float.fromhex(encoded["effective_budget_seconds"]) == 8.0
