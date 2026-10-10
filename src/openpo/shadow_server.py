from __future__ import annotations

import argparse
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import logging
import math
import os
from pathlib import Path
import signal
from threading import Event, Lock, Thread
import time
from typing import Any, Callable

from .certification_policy import (
    CertificationCircuitReset,
    CertificationCircuitResetCooldownError,
    CertificationMode,
    CertificationPolicy,
    CertificationPolicyConflictError,
    certification_decision_to_payload,
)
from .durable_shadow import (
    DurableShadowCertificationService,
    ShadowAdmissionSnapshot,
    ShadowBackpressureError,
    ShadowCircuitResetDisabledError,
    ShadowExecutionError,
    ShadowRouteCircuitOpenError,
    ShadowSubmissionConflictError,
    _result_to_payload,
    _ticket_from_payload,
    _ticket_to_payload,
)
from .shadow import ShadowQueueFullError
from .shadow_codec import production_to_payload, request_from_payload
from .shadow_journal import ShadowCapacitySnapshot
from .service_time_calibration import (
    ServiceTimeCalibrationConfig,
    ServiceTimeCalibrationSnapshot,
)


LOGGER = logging.getLogger("openpo.shadow_server")


class ShadowHTTPApplication:
    def __init__(
        self,
        service: DurableShadowCertificationService,
        *,
        max_collect_timeout_seconds: float,
    ) -> None:
        if max_collect_timeout_seconds <= 0:
            raise ValueError("max_collect_timeout_seconds must be positive")
        self.service = service
        self.max_collect_timeout_seconds = max_collect_timeout_seconds
        self._lock = Lock()
        self._draining = False

    @property
    def draining(self) -> bool:
        with self._lock:
            return self._draining

    def begin_drain(self) -> bool:
        with self._lock:
            if self._draining:
                return False
            self._draining = True
            return True

    def health(self) -> dict[str, object]:
        self.service.health_check()
        return {
            "schema": "shadow-health-v3",
            "status": "DRAINING" if self.draining else "SERVING",
            "capacity": _admission_to_payload(self.service.admission_snapshot()),
        }

    def submit(self, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        _require_exact_keys(payload, {"submission_id", "request"}, "submit")
        if self.draining:
            return HTTPStatus.SERVICE_UNAVAILABLE, {
                "schema": "shadow-error-v1",
                "error": "DRAINING",
                "message": "durable shadow service is draining",
            }
        submission_id = _require_string(payload.get("submission_id"), "submission_id")
        request = request_from_payload(_require_mapping(payload.get("request"), "request"))
        submission = self.service.submit(request, submission_id=submission_id)
        if submission.policy_decision is None:
            if submission.ticket is None:
                raise RuntimeError("legacy shadow submission is missing its ticket")
            response: dict[str, object] = {
                "schema": "shadow-submit-v1",
                "ticket": _ticket_to_payload(submission.ticket),
                "online": production_to_payload(submission.online),
                "online_elapsed_seconds": float(
                    submission.online_elapsed_seconds
                ).hex(),
            }
        else:
            response = {
                "schema": "shadow-submit-v2",
                "ticket": (
                    None
                    if submission.ticket is None
                    else _ticket_to_payload(submission.ticket)
                ),
                "online": production_to_payload(submission.online),
                "online_elapsed_seconds": float(
                    submission.online_elapsed_seconds
                ).hex(),
                "certification": certification_decision_to_payload(
                    submission.policy_decision
                ),
            }
        return HTTPStatus.OK, response

    def collect(self, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        allowed = {"ticket", "timeout_seconds"}
        if not set(payload).issubset(allowed) or "ticket" not in payload:
            raise ValueError("collect must contain ticket and optional timeout_seconds only")
        ticket = _ticket_from_payload(dict(_require_mapping(payload.get("ticket"), "ticket")))
        timeout = _require_timeout(
            payload.get("timeout_seconds", 0.0),
            maximum=self.max_collect_timeout_seconds,
        )
        try:
            result = self.service.collect(ticket, timeout=timeout)
        except TimeoutError:
            return HTTPStatus.ACCEPTED, {
                "schema": "shadow-collect-v1",
                "status": "PENDING",
                "ticket_sha256": ticket.ticket_sha256,
            }
        except ShadowExecutionError as error:
            return HTTPStatus.INTERNAL_SERVER_ERROR, {
                "schema": "shadow-error-v1",
                "error": error.error_type,
                "message": error.worker_message,
            }
        return HTTPStatus.OK, {
            "schema": "shadow-collect-v1",
            "status": "SUCCEEDED",
            "result": _result_to_payload(result),
        }

    def ack(self, payload: dict[str, object]) -> tuple[int, dict[str, object]]:
        _require_exact_keys(payload, {"ticket"}, "ack")
        ticket = _ticket_from_payload(dict(_require_mapping(payload.get("ticket"), "ticket")))
        acknowledged = self.service.ack(ticket)
        return HTTPStatus.OK, {
            "schema": "shadow-ack-v1",
            "ticket_sha256": ticket.ticket_sha256,
            "acknowledged": acknowledged,
        }

    def reset_circuit(
        self, payload: dict[str, object]
    ) -> tuple[int, dict[str, object]]:
        _require_exact_keys(
            payload,
            {
                "route_key",
                "expected_evidence_sha256",
                "approval_sha256",
                "actor",
                "reason",
            },
            "reset-circuit",
        )
        reset, changed = self.service.reset_route_circuit(
            route_key=_require_string(payload.get("route_key"), "route_key"),
            expected_evidence_sha256=_require_string(
                payload.get("expected_evidence_sha256"),
                "expected_evidence_sha256",
            ),
            approval_sha256=_require_string(
                payload.get("approval_sha256"), "approval_sha256"
            ),
            actor=_require_string(payload.get("actor"), "actor"),
            reason=_require_string(payload.get("reason"), "reason"),
        )
        return HTTPStatus.OK, {
            "schema": "shadow-circuit-reset-v1",
            "changed": changed,
            "reset": _circuit_reset_to_payload(reset),
        }


class ShadowHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    allow_reuse_address = True

    def __init__(
        self,
        server_address: tuple[str, int],
        application: ShadowHTTPApplication,
        *,
        max_body_bytes: int,
        request_timeout_seconds: float,
        bearer_token: str | None = None,
    ) -> None:
        if max_body_bytes <= 0:
            raise ValueError("max_body_bytes must be positive")
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self.application = application
        self.max_body_bytes = max_body_bytes
        self.request_timeout_seconds = request_timeout_seconds
        self.bearer_token = bearer_token
        super().__init__(server_address, ShadowHTTPRequestHandler)


class ShadowHTTPRequestHandler(BaseHTTPRequestHandler):
    server: ShadowHTTPServer
    server_version = "OpenPOShadow/1"
    sys_version = ""
    protocol_version = "HTTP/1.1"

    def setup(self) -> None:
        super().setup()
        self.connection.settimeout(self.server.request_timeout_seconds)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._dispatch_health(readiness=False)
            return
        if self.path == "/readyz":
            self._dispatch_health(readiness=True)
            return
        self._write_json(HTTPStatus.NOT_FOUND, _error("NOT_FOUND", "unknown endpoint"))

    def do_POST(self) -> None:
        routes: dict[str, Callable[[dict[str, object]], tuple[int, dict[str, object]]]] = {
            "/v1/submit": self.server.application.submit,
            "/v1/collect": self.server.application.collect,
            "/v1/ack": self.server.application.ack,
            "/v1/reset-circuit": self.server.application.reset_circuit,
        }
        handler = routes.get(self.path)
        if handler is None:
            self._write_json(HTTPStatus.NOT_FOUND, _error("NOT_FOUND", "unknown endpoint"))
            return
        if not self._authorized():
            self._write_json(
                HTTPStatus.UNAUTHORIZED,
                _error("UNAUTHORIZED", "valid bearer token is required"),
                extra_headers={"WWW-Authenticate": "Bearer"},
            )
            return
        decode_started = time.perf_counter()
        decode_seconds = 0.0
        application_started: float | None = None
        try:
            payload = self._read_json()
            decode_seconds = time.perf_counter() - decode_started
            application_started = time.perf_counter()
            status, response = handler(payload)
        except ShadowBackpressureError as error:
            status, response = HTTPStatus.TOO_MANY_REQUESTS, {
                "schema": "shadow-error-v2",
                "error": "CERTIFICATION_BACKPRESSURE",
                "message": str(error),
                "reasons": list(error.reasons),
                "capacity": _admission_to_payload(error.admission),
            }
        except ShadowQueueFullError as error:
            status, response = HTTPStatus.TOO_MANY_REQUESTS, _error("QUEUE_FULL", str(error))
        except ShadowSubmissionConflictError as error:
            status, response = HTTPStatus.CONFLICT, _error("SUBMISSION_CONFLICT", str(error))
        except ShadowRouteCircuitOpenError as error:
            status, response = HTTPStatus.SERVICE_UNAVAILABLE, _error(
                "ROUTE_CIRCUIT_OPEN", str(error)
            )
        except ShadowCircuitResetDisabledError as error:
            status, response = HTTPStatus.FORBIDDEN, _error(
                "CIRCUIT_RESET_DISABLED", str(error)
            )
        except CertificationCircuitResetCooldownError as error:
            status, response = HTTPStatus.CONFLICT, {
                **_error("CIRCUIT_RESET_COOLDOWN", str(error)),
                "remaining_seconds": float(error.remaining_seconds).hex(),
            }
        except CertificationPolicyConflictError as error:
            status, response = HTTPStatus.CONFLICT, _error(
                "CIRCUIT_RESET_CONFLICT", str(error)
            )
        except KeyError as error:
            status, response = HTTPStatus.NOT_FOUND, _error("UNKNOWN_TICKET", str(error))
        except (TypeError, ValueError) as error:
            status, response = HTTPStatus.BAD_REQUEST, _error("INVALID_REQUEST", str(error))
        except Exception:
            LOGGER.exception("unhandled durable shadow HTTP request failure")
            status, response = HTTPStatus.INTERNAL_SERVER_ERROR, _error(
                "INTERNAL_ERROR", "durable shadow request failed"
            )
        finally:
            timing_finished = time.perf_counter()
            if application_started is None:
                decode_seconds = timing_finished - decode_started
                application_seconds = 0.0
            else:
                application_seconds = timing_finished - application_started
        self._write_json(
            status,
            response,
            server_timing_seconds={
                "request_decode": decode_seconds,
                "application": application_seconds,
            },
        )

    def _dispatch_health(self, *, readiness: bool) -> None:
        try:
            payload = self.server.application.health()
        except Exception:
            LOGGER.exception("durable shadow health check failed")
            self._write_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                _error("UNHEALTHY", "journal health check failed"),
            )
            return
        status = (
            HTTPStatus.SERVICE_UNAVAILABLE
            if readiness and self.server.application.draining
            else HTTPStatus.OK
        )
        self._write_json(status, payload)

    def _read_json(self) -> dict[str, object]:
        if self.headers.get("Transfer-Encoding") is not None:
            raise ValueError("Transfer-Encoding is not supported")
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
        if content_type != "application/json":
            raise ValueError("Content-Type must be application/json")
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ValueError("Content-Length is required")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("Content-Length must be an integer") from error
        if length <= 0 or length > self.server.max_body_bytes:
            raise ValueError(
                f"Content-Length must be in [1, {self.server.max_body_bytes}]"
            )
        encoded = self.rfile.read(length)
        if len(encoded) != length:
            raise ValueError("request body ended before Content-Length bytes were read")
        try:
            payload = json.loads(encoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid UTF-8 JSON") from error
        return _require_mapping(payload, "request body")

    def _authorized(self) -> bool:
        expected = self.server.bearer_token
        if expected is None:
            return True
        authorization = self.headers.get("Authorization", "")
        prefix = "Bearer "
        if not authorization.startswith(prefix):
            return False
        return hmac.compare_digest(authorization[len(prefix) :], expected)

    def _write_json(
        self,
        status: int,
        payload: dict[str, object],
        *,
        extra_headers: dict[str, str] | None = None,
        server_timing_seconds: dict[str, float] | None = None,
    ) -> None:
        encode_started = time.perf_counter()
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
        ).encode("utf-8")
        response_encode_seconds = time.perf_counter() - encode_started
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        if server_timing_seconds is not None:
            timings = {**server_timing_seconds, "response_encode": response_encode_seconds}
            self.send_header(
                "Server-Timing",
                ",".join(
                    f"{name};dur={seconds * 1000.0:.9f}"
                    for name, seconds in timings.items()
                ),
            )
        for name, value in (extra_headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format_string: str, *args: object) -> None:
        LOGGER.info("%s - %s", self.address_string(), format_string % args)


def serve(args: argparse.Namespace) -> int:
    certification_policy = (
        None
        if args.certification_mode is None
        else CertificationPolicy(
            mode=args.certification_mode,
            sample_numerator=args.sample_numerator,
            sample_denominator=args.sample_denominator,
            anomaly_objective_gap=args.anomaly_objective_gap,
            anomaly_primal_residual=args.anomaly_primal_residual,
        )
    )
    service = DurableShadowCertificationService(
        args.journal,
        max_pending=args.max_pending,
        worker_count=args.worker_count,
        objective_tolerance=args.objective_tolerance,
        weights_tolerance=args.weights_tolerance,
        lease_seconds=args.lease_seconds,
        heartbeat_seconds=args.heartbeat_seconds,
        certification_policy=certification_policy,
        max_queue_age_seconds=args.max_queue_age_seconds,
        certification_service_time_budget_seconds=(
            args.certification_service_time_budget_seconds
        ),
        service_time_calibration=(
            ServiceTimeCalibrationConfig(
                window_size=args.service_time_calibration_window_size,
                minimum_samples=args.service_time_calibration_minimum_samples,
                ewma_alpha=args.service_time_calibration_ewma_alpha,
                quantile_probability=args.service_time_calibration_quantile_probability,
                safety_multiplier=args.service_time_calibration_safety_multiplier,
                minimum_budget_ratio=args.service_time_calibration_minimum_budget_ratio,
                maximum_budget_ratio=args.service_time_calibration_maximum_budget_ratio,
            )
            if args.service_time_calibration
            else None
        ),
        max_drain_seconds=args.max_drain_seconds,
        circuit_reset_cooldown_seconds=args.circuit_reset_cooldown_seconds,
    )
    application = ShadowHTTPApplication(
        service,
        max_collect_timeout_seconds=args.max_collect_timeout_seconds,
    )
    try:
        server = ShadowHTTPServer(
            (args.host, args.port),
            application,
            max_body_bytes=args.max_body_bytes,
            request_timeout_seconds=args.request_timeout_seconds,
            bearer_token=_load_bearer_token(args.bearer_token_file),
        )
    except BaseException:
        service.shutdown(wait=True, require_collected=False)
        raise
    shutdown_requested = Event()

    def request_shutdown(signum: int, frame: object) -> None:
        del frame
        if shutdown_requested.is_set():
            return
        shutdown_requested.set()
        application.begin_drain()
        LOGGER.info("received signal %s; draining durable shadow service", signum)
        Thread(target=server.shutdown, name="openpo-shadow-http-shutdown", daemon=True).start()

    previous_handlers = {
        signum: signal.signal(signum, request_shutdown)
        for signum in (signal.SIGTERM, signal.SIGINT)
    }
    try:
        bound_host, bound_port = server.server_address[:2]
        if args.ready_file is not None:
            _write_ready_file(args.ready_file, host=str(bound_host), port=int(bound_port))
        LOGGER.info("serving durable shadow HTTP API on %s:%s", bound_host, bound_port)
        server.serve_forever(poll_interval=0.1)
    finally:
        application.begin_drain()
        server.server_close()
        service.shutdown(wait=True, require_collected=False)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        if args.ready_file is not None:
            args.ready_file.unlink(missing_ok=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve the durable OpenPO shadow API")
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--max-pending", type=int, default=1)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--objective-tolerance", type=float, default=1e-6)
    parser.add_argument("--weights-tolerance", type=float, default=1e-5)
    parser.add_argument("--lease-seconds", type=float, default=60.0)
    parser.add_argument("--heartbeat-seconds", type=float, default=10.0)
    parser.add_argument("--max-queue-age-seconds", type=float)
    parser.add_argument("--certification-service-time-budget-seconds", type=float)
    parser.add_argument("--service-time-calibration", action="store_true")
    parser.add_argument("--service-time-calibration-window-size", type=int, default=32)
    parser.add_argument("--service-time-calibration-minimum-samples", type=int, default=4)
    parser.add_argument("--service-time-calibration-ewma-alpha", type=float, default=0.25)
    parser.add_argument(
        "--service-time-calibration-quantile-probability", type=float, default=0.90
    )
    parser.add_argument(
        "--service-time-calibration-safety-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--service-time-calibration-minimum-budget-ratio", type=float, default=0.50
    )
    parser.add_argument(
        "--service-time-calibration-maximum-budget-ratio", type=float, default=2.0
    )
    parser.add_argument("--max-drain-seconds", type=float)
    parser.add_argument("--circuit-reset-cooldown-seconds", type=float)
    parser.add_argument(
        "--certification-mode",
        type=CertificationMode,
        choices=list(CertificationMode),
    )
    parser.add_argument("--sample-numerator", type=int, default=1)
    parser.add_argument("--sample-denominator", type=int, default=1)
    parser.add_argument("--anomaly-objective-gap", type=float, default=0.01)
    parser.add_argument("--anomaly-primal-residual", type=float, default=1e-5)
    parser.add_argument("--max-body-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--request-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-collect-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--bearer-token-file", type=Path)
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not 0 <= args.port <= 65535:
        parser.error("--port must be in [0, 65535]")
    if args.max_pending <= 0 or args.worker_count <= 0 or args.max_body_bytes <= 0:
        parser.error("--max-pending, --worker-count and --max-body-bytes must be positive")
    if args.worker_count > args.max_pending:
        parser.error("--worker-count must not exceed --max-pending")
    if args.sample_denominator <= 0:
        parser.error("--sample-denominator must be positive")
    if not 0 < args.sample_numerator <= args.sample_denominator:
        parser.error("--sample-numerator must be in [1, --sample-denominator]")
    if args.certification_mode is CertificationMode.FULL and (
        args.sample_numerator != 1 or args.sample_denominator != 1
    ):
        parser.error("full certification mode requires a 1/1 sample ratio")
    for field in (
        "objective_tolerance",
        "weights_tolerance",
        "lease_seconds",
        "heartbeat_seconds",
        "anomaly_objective_gap",
        "anomaly_primal_residual",
        "request_timeout_seconds",
        "max_collect_timeout_seconds",
    ):
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if args.heartbeat_seconds >= args.lease_seconds:
        parser.error("--heartbeat-seconds must be less than --lease-seconds")
    if args.max_queue_age_seconds is not None and (
        not math.isfinite(args.max_queue_age_seconds)
        or args.max_queue_age_seconds <= 0.0
    ):
        parser.error("--max-queue-age-seconds must be positive")
    if (args.certification_service_time_budget_seconds is None) != (
        args.max_drain_seconds is None
    ):
        parser.error(
            "--certification-service-time-budget-seconds and --max-drain-seconds "
            "must be configured together"
        )
    if (
        args.service_time_calibration
        and args.certification_service_time_budget_seconds is None
    ):
        parser.error(
            "--service-time-calibration requires "
            "--certification-service-time-budget-seconds and --max-drain-seconds"
        )
    if args.service_time_calibration:
        try:
            ServiceTimeCalibrationConfig(
                window_size=args.service_time_calibration_window_size,
                minimum_samples=args.service_time_calibration_minimum_samples,
                ewma_alpha=args.service_time_calibration_ewma_alpha,
                quantile_probability=args.service_time_calibration_quantile_probability,
                safety_multiplier=args.service_time_calibration_safety_multiplier,
                minimum_budget_ratio=args.service_time_calibration_minimum_budget_ratio,
                maximum_budget_ratio=args.service_time_calibration_maximum_budget_ratio,
            )
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    for field in (
        "certification_service_time_budget_seconds",
        "max_drain_seconds",
    ):
        value = getattr(args, field)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            parser.error(f"--{field.replace('_', '-')} must be positive")
    if (
        args.circuit_reset_cooldown_seconds is not None
        and args.circuit_reset_cooldown_seconds < 0.0
    ):
        parser.error("--circuit-reset-cooldown-seconds must be nonnegative")
    if (
        args.circuit_reset_cooldown_seconds is not None
        and args.certification_mode is None
    ):
        parser.error(
            "--circuit-reset-cooldown-seconds requires --certification-mode"
        )
    if args.ready_file is not None and not args.ready_file.parent.is_dir():
        parser.error("--ready-file parent directory must exist")
    if args.host not in {"127.0.0.1", "::1", "localhost"} and args.bearer_token_file is None:
        parser.error("--bearer-token-file is required for non-loopback listeners")
    if args.bearer_token_file is not None and not args.bearer_token_file.is_file():
        parser.error("--bearer-token-file must exist and be a regular file")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args, parser)
    raise SystemExit(serve(args))


def _capacity_to_payload(snapshot: ShadowCapacitySnapshot) -> dict[str, object]:
    def optional_hex(value: float | None) -> str | None:
        return None if value is None else float(value).hex()

    return {
        "schema": "shadow-capacity-v1",
        "max_outstanding": snapshot.max_outstanding,
        "outstanding": snapshot.outstanding,
        "available": snapshot.available,
        "utilization": float(snapshot.utilization).hex(),
        "online_ready": snapshot.online_ready,
        "leased": snapshot.leased,
        "terminal_unacked": snapshot.terminal_unacked,
        "oldest_outstanding_age_seconds": optional_hex(
            snapshot.oldest_outstanding_age_seconds
        ),
        "oldest_inflight_age_seconds": optional_hex(
            snapshot.oldest_inflight_age_seconds
        ),
        "oldest_ready_age_seconds": optional_hex(
            snapshot.oldest_ready_age_seconds
        ),
    }


def _admission_to_payload(snapshot: ShadowAdmissionSnapshot) -> dict[str, object]:
    payload = _capacity_to_payload(snapshot.capacity)
    payload["schema"] = "shadow-capacity-v2"
    payload.update(
        {
            "worker_count": snapshot.worker_count,
            "active_workers": snapshot.active_workers,
            "service_time_budget_seconds": (
                None
                if snapshot.service_time_budget_seconds is None
                else float(snapshot.service_time_budget_seconds).hex()
            ),
            "service_time_budget_source": snapshot.service_time_budget_source,
            "service_time_calibration": _service_time_calibration_to_payload(
                snapshot.service_time_calibration
            ),
            "max_drain_seconds": (
                None
                if snapshot.max_drain_seconds is None
                else float(snapshot.max_drain_seconds).hex()
            ),
            "estimated_drain_seconds": (
                None
                if snapshot.estimated_drain_seconds is None
                else float(snapshot.estimated_drain_seconds).hex()
            ),
        }
    )
    return payload


def _service_time_calibration_to_payload(
    snapshot: ServiceTimeCalibrationSnapshot | None,
) -> dict[str, object] | None:
    if snapshot is None:
        return None
    return {
        "source": snapshot.source,
        "route_key": snapshot.route_key,
        "static_budget_seconds": float(snapshot.static_budget_seconds).hex(),
        "effective_budget_seconds": float(snapshot.effective_budget_seconds).hex(),
        "total_sample_count": snapshot.total_sample_count,
        "retained_sample_count": snapshot.retained_sample_count,
        "window_size": snapshot.window_size,
        "minimum_samples": snapshot.minimum_samples,
        "ewma_alpha": float(snapshot.ewma_alpha).hex(),
        "ewma_seconds": (
            None if snapshot.ewma_seconds is None else float(snapshot.ewma_seconds).hex()
        ),
        "quantile_probability": float(snapshot.quantile_probability).hex(),
        "quantile_seconds": (
            None
            if snapshot.quantile_seconds is None
            else float(snapshot.quantile_seconds).hex()
        ),
        "safety_multiplier": float(snapshot.safety_multiplier).hex(),
        "minimum_budget_seconds": float(snapshot.minimum_budget_seconds).hex(),
        "maximum_budget_seconds": float(snapshot.maximum_budget_seconds).hex(),
    }


def _circuit_reset_to_payload(
    reset: CertificationCircuitReset,
) -> dict[str, object]:
    return {
        "reset_id": reset.reset_id,
        "route_key": reset.route_key,
        "policy_sha256": reset.policy_sha256,
        "circuit_evidence_sha256": reset.circuit_evidence_sha256,
        "circuit_reason": reset.circuit_reason,
        "circuit_opened_at": float(reset.circuit_opened_at).hex(),
        "approval_sha256": reset.approval_sha256,
        "actor": reset.actor,
        "reason": reset.reason,
        "reset_at": float(reset.reset_at).hex(),
        "reset_evidence_sha256": reset.reset_evidence_sha256,
    }


def _require_mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _require_exact_keys(
    payload: dict[str, object], expected: set[str], field: str
) -> None:
    if set(payload) != expected:
        raise ValueError(f"{field} fields must be exactly {sorted(expected)}")


def _require_timeout(value: object, *, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("timeout_seconds must be numeric")
    timeout = float(value)
    if timeout < 0 or timeout > maximum:
        raise ValueError(f"timeout_seconds must be in [0, {maximum}]")
    return timeout


def _error(code: str, message: str) -> dict[str, object]:
    return {"schema": "shadow-error-v1", "error": code, "message": message}


def _write_ready_file(path: Path, *, host: str, port: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = {
        "schema": "shadow-server-ready-v1",
        "host": host,
        "port": port,
        "pid": os.getpid(),
    }
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_bearer_token(path: Path | None) -> str | None:
    if path is None:
        return None
    token = path.read_text(encoding="utf-8").strip()
    if len(token) < 32 or any(character.isspace() for character in token):
        raise ValueError("bearer token must contain at least 32 non-whitespace characters")
    return token


if __name__ == "__main__":
    main()
