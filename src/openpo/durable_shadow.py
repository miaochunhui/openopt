from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import platform
from threading import Event, Lock, Thread
import time
import traceback
import uuid

import torch

from .facade import FallbackPolicy, LaneResult, ProductionRequest, ProductionResult, SolveStatus, certify_exact, solve_production
from .certification_policy import (
    CertificationCircuitReset,
    CertificationDisposition,
    CertificationObservation,
    CertificationPolicy,
    CertificationPolicyDecision,
    SQLiteCertificationPolicyStore,
    validate_certification_decision,
)
from .shadow import (
    OnlineShadowSubmission,
    ShadowCertificationResult,
    ShadowDecision,
    ShadowQueueFullError,
    ShadowTicket,
    _canonical_sha256,
    _freeze_request,
    _request_hash,
)
from .shadow_codec import (
    decode_payload,
    encode_payload,
    lane_from_payload,
    lane_to_payload,
    production_from_payload,
    production_to_payload,
    request_from_payload,
    request_to_payload,
)
from .shadow_journal import (
    DurableShadowJob,
    ShadowCapacitySnapshot,
    ShadowJournalFullError,
    SQLiteShadowJournal,
)
from .service_time_calibration import (
    ServiceTimeCalibrationConfig,
    ServiceTimeCalibrationSnapshot,
    calibrate_service_time_budget,
)


class ShadowExecutionError(RuntimeError):
    def __init__(self, *, error_type: str, message: str, traceback_text: str) -> None:
        super().__init__(f"durable shadow worker failed with {error_type}: {message}")
        self.error_type = error_type
        self.worker_message = message
        self.traceback_text = traceback_text


class ShadowSubmissionConflictError(RuntimeError):
    pass


class ShadowRouteCircuitOpenError(RuntimeError):
    pass


class ShadowCircuitResetDisabledError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShadowAdmissionSnapshot:
    capacity: ShadowCapacitySnapshot
    worker_count: int
    active_workers: int
    service_time_budget_seconds: float | None
    service_time_budget_source: str | None
    service_time_calibration: ServiceTimeCalibrationSnapshot | None
    max_drain_seconds: float | None
    estimated_drain_seconds: float | None


class ShadowBackpressureError(ShadowQueueFullError):
    def __init__(
        self,
        *,
        reasons: tuple[str, ...],
        admission: ShadowAdmissionSnapshot,
    ) -> None:
        self.reasons = reasons
        self.admission = admission
        self.capacity = admission.capacity
        detail = ", ".join(reasons)
        super().__init__(f"durable Clarabel shadow queue is full or stale: {detail}")


def certification_backpressure_reasons(
    snapshot: ShadowCapacitySnapshot,
    *,
    max_queue_age_seconds: float | None,
    worker_count: int = 1,
    service_time_budget_seconds: float | None = None,
    max_drain_seconds: float | None = None,
    additional_jobs: int = 0,
) -> tuple[str, ...]:
    if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count <= 0:
        raise ValueError("worker_count must be a positive integer")
    if max_queue_age_seconds is not None and (
        not math.isfinite(max_queue_age_seconds) or max_queue_age_seconds <= 0.0
    ):
        raise ValueError("max_queue_age_seconds must be finite and positive or null")
    if (service_time_budget_seconds is None) != (max_drain_seconds is None):
        raise ValueError(
            "service_time_budget_seconds and max_drain_seconds must be configured together"
        )
    for name, value in (
        ("service_time_budget_seconds", service_time_budget_seconds),
        ("max_drain_seconds", max_drain_seconds),
    ):
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise ValueError(f"{name} must be finite and positive or null")
    reasons: list[str] = []
    required_capacity = max(1, additional_jobs)
    if snapshot.available < required_capacity:
        reasons.append("capacity_exhausted")
    if (
        max_queue_age_seconds is not None
        and snapshot.oldest_inflight_age_seconds is not None
        and snapshot.oldest_inflight_age_seconds > max_queue_age_seconds
    ):
        reasons.append("queue_age_exceeded")
    estimated_drain = certification_estimated_drain_seconds(
        snapshot,
        worker_count=worker_count,
        service_time_budget_seconds=service_time_budget_seconds,
        additional_jobs=additional_jobs,
    )
    if (
        estimated_drain is not None
        and max_drain_seconds is not None
        and estimated_drain > max_drain_seconds
    ):
        reasons.append("drain_time_exceeded")
    return tuple(reasons)


def certification_estimated_drain_seconds(
    snapshot: ShadowCapacitySnapshot,
    *,
    worker_count: int,
    service_time_budget_seconds: float | None,
    additional_jobs: int = 0,
) -> float | None:
    if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count <= 0:
        raise ValueError("worker_count must be a positive integer")
    if (
        not isinstance(additional_jobs, int)
        or isinstance(additional_jobs, bool)
        or additional_jobs < 0
    ):
        raise ValueError("additional_jobs must be a nonnegative integer")
    if service_time_budget_seconds is None:
        return None
    if not math.isfinite(service_time_budget_seconds) or service_time_budget_seconds <= 0.0:
        raise ValueError("service_time_budget_seconds must be finite and positive or null")
    inflight = snapshot.online_ready + snapshot.leased + additional_jobs
    waves = math.ceil(inflight / worker_count)
    return waves * service_time_budget_seconds


class DurableShadowCertificationService:
    def __init__(
        self,
        journal_path: str | Path,
        *,
        max_pending: int = 1,
        worker_count: int = 1,
        objective_tolerance: float = 1e-6,
        weights_tolerance: float = 1e-5,
        lease_seconds: float = 60.0,
        heartbeat_seconds: float = 10.0,
        service_epoch: str | None = None,
        certification_policy: CertificationPolicy | None = None,
        policy_store_path: str | Path | None = None,
        max_queue_age_seconds: float | None = None,
        certification_service_time_budget_seconds: float | None = None,
        service_time_calibration: ServiceTimeCalibrationConfig | None = None,
        max_drain_seconds: float | None = None,
        circuit_reset_cooldown_seconds: float | None = None,
    ) -> None:
        if not isinstance(max_pending, int) or isinstance(max_pending, bool) or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        if not isinstance(worker_count, int) or isinstance(worker_count, bool) or worker_count <= 0:
            raise ValueError("worker_count must be a positive integer")
        if worker_count > max_pending:
            raise ValueError("worker_count must not exceed max_pending")
        for name, value in (
            ("objective_tolerance", objective_tolerance),
            ("weights_tolerance", weights_tolerance),
            ("lease_seconds", lease_seconds),
            ("heartbeat_seconds", heartbeat_seconds),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if heartbeat_seconds >= lease_seconds:
            raise ValueError("heartbeat_seconds must be less than lease_seconds")
        if max_queue_age_seconds is not None and (
            not math.isfinite(max_queue_age_seconds) or max_queue_age_seconds <= 0.0
        ):
            raise ValueError("max_queue_age_seconds must be finite and positive or null")
        if (certification_service_time_budget_seconds is None) != (
            max_drain_seconds is None
        ):
            raise ValueError(
                "certification_service_time_budget_seconds and max_drain_seconds "
                "must be configured together"
            )
        if (
            service_time_calibration is not None
            and certification_service_time_budget_seconds is None
        ):
            raise ValueError(
                "service_time_calibration requires a static certification service-time budget"
            )
        if service_time_calibration is not None and not isinstance(
            service_time_calibration, ServiceTimeCalibrationConfig
        ):
            raise TypeError(
                "service_time_calibration must be a ServiceTimeCalibrationConfig or null"
            )
        for name, value in (
            (
                "certification_service_time_budget_seconds",
                certification_service_time_budget_seconds,
            ),
            ("max_drain_seconds", max_drain_seconds),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive or null")
        if circuit_reset_cooldown_seconds is not None and (
            not math.isfinite(circuit_reset_cooldown_seconds)
            or circuit_reset_cooldown_seconds < 0.0
        ):
            raise ValueError(
                "circuit_reset_cooldown_seconds must be finite and nonnegative or null"
            )
        self._objective_tolerance = objective_tolerance
        self._weights_tolerance = weights_tolerance
        self._lease_seconds = lease_seconds
        self._heartbeat_seconds = heartbeat_seconds
        self._max_queue_age_seconds = max_queue_age_seconds
        self._worker_count = worker_count
        self._certification_service_time_budget_seconds = (
            certification_service_time_budget_seconds
        )
        self._service_time_calibration = service_time_calibration
        self._max_drain_seconds = max_drain_seconds
        self._circuit_reset_cooldown_seconds = circuit_reset_cooldown_seconds
        self._certification_policy = certification_policy
        if certification_policy is None and policy_store_path is not None:
            raise ValueError(
                "policy_store_path requires an explicit certification_policy"
            )
        self._policy_store = (
            None
            if certification_policy is None
            else SQLiteCertificationPolicyStore(
                journal_path if policy_store_path is None else policy_store_path
            )
        )
        self._service_epoch = uuid.uuid4().hex if service_epoch is None else service_epoch
        if (
            len(self._service_epoch) != 32
            or any(character not in "0123456789abcdef" for character in self._service_epoch)
        ):
            raise ValueError("service_epoch must be 32 lowercase hexadecimal characters")
        policy_payload = {
            "policy_version": "durable-shadow-certification-v1",
            "comparison_rule": "exact-objective-relative-and-weights-max-abs-v1",
            "primary_backend": "osqp",
            "shadow_backend": "clarabel",
            "objective_tolerance": float(objective_tolerance).hex(),
            "weights_tolerance": float(weights_tolerance).hex(),
        }
        if certification_policy is not None:
            policy_payload["async_certification_policy_sha256"] = (
                certification_policy.policy_sha256
            )
        self._policy_sha256 = _canonical_sha256(policy_payload)
        self._owner = uuid.uuid4().hex
        self._journal = SQLiteShadowJournal(journal_path, max_outstanding=max_pending)
        self._lock = Lock()
        self._admission_lock = Lock()
        self._closed = False
        self._active: set[Future[None]] = set()
        self._executor = ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix="openpo-durable-clarabel-shadow",
        )
        self._scheduler_stop = Event()
        for job in self._journal.outstanding_jobs():
            ticket = _ticket_from_payload(decode_payload(job.ticket_payload))
            if ticket.policy_sha256 != self._policy_sha256:
                self._executor.shutdown(wait=False, cancel_futures=False)
                raise RuntimeError(
                    "durable shadow journal contains an unsupported frozen policy"
                )
            request = request_from_payload(decode_payload(job.request_payload))
            online = production_from_payload(decode_payload(job.online_payload))
            if _request_hash(request) != ticket.shadow_request_sha256:
                self._executor.shutdown(wait=False, cancel_futures=False)
                raise RuntimeError("durable shadow journal request hash mismatch")
            if (
                online.input_sha256 != ticket.input_sha256
                or online.request_sha256 != ticket.primary_request_sha256
            ):
                self._executor.shutdown(wait=False, cancel_futures=False)
                raise RuntimeError("durable shadow journal online evidence hash mismatch")
            if job.state == "SUCCEEDED":
                if job.terminal_payload is None:
                    self._executor.shutdown(wait=False, cancel_futures=False)
                    raise RuntimeError("durable shadow journal success is missing payload")
                try:
                    _validated_terminal_result(job, ticket, online)
                except (TypeError, ValueError, RuntimeError):
                    self._executor.shutdown(wait=False, cancel_futures=False)
                    raise
            if self._policy_store is not None:
                self._submission_from_job(job)
        self._scheduler = Thread(
            target=self._scheduler_loop,
            name="openpo-durable-shadow-scheduler",
            daemon=True,
        )
        self._scheduler.start()
        self._schedule_next()

    def __enter__(self) -> DurableShadowCertificationService:
        return self

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        self.shutdown(wait=True, require_collected=exc_type is None)

    @property
    def service_epoch(self) -> str:
        return self._service_epoch

    @property
    def policy_sha256(self) -> str:
        return self._policy_sha256

    @property
    def outstanding_count(self) -> int:
        return self._journal.outstanding_count

    @property
    def uncollected_count(self) -> int:
        return self._journal.unobserved_count

    @property
    def inflight_count(self) -> int:
        return self._journal.inflight_count

    def capacity_snapshot(self) -> ShadowCapacitySnapshot:
        return self._journal.capacity_snapshot()

    def admission_snapshot(
        self,
        *,
        service_time_route_key: str | None = None,
    ) -> ShadowAdmissionSnapshot:
        capacity = self._journal.capacity_snapshot()
        calibration = self._service_time_calibration_snapshot(
            service_time_route_key=service_time_route_key
        )
        effective_budget = (
            self._certification_service_time_budget_seconds
            if calibration is None
            else calibration.effective_budget_seconds
        )
        with self._lock:
            active_workers = len(self._active)
        return ShadowAdmissionSnapshot(
            capacity=capacity,
            worker_count=self._worker_count,
            active_workers=active_workers,
            service_time_budget_seconds=effective_budget,
            service_time_budget_source=(
                None
                if effective_budget is None
                else "static_configured"
                if calibration is None
                else calibration.source
            ),
            service_time_calibration=calibration,
            max_drain_seconds=self._max_drain_seconds,
            estimated_drain_seconds=certification_estimated_drain_seconds(
                capacity,
                worker_count=self._worker_count,
                service_time_budget_seconds=(
                    effective_budget
                ),
            ),
        )

    def _service_time_calibration_snapshot(
        self,
        *,
        service_time_route_key: str | None = None,
    ) -> ServiceTimeCalibrationSnapshot | None:
        config = self._service_time_calibration
        static_budget = self._certification_service_time_budget_seconds
        if config is None:
            return None
        if static_budget is None:
            raise RuntimeError("service-time calibration lost its static fallback budget")
        samples = self._journal.service_time_samples(
            window_size=config.window_size,
            route_key=service_time_route_key,
        )
        return calibrate_service_time_budget(
            static_budget_seconds=static_budget,
            samples=samples.elapsed_seconds,
            total_sample_count=samples.total_sample_count,
            config=config,
            route_key=service_time_route_key,
        )

    def _service_time_route_key(self, fast_device: str) -> str:
        hardware: dict[str, object] = {
            "cpu_machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        }
        if fast_device.startswith("cuda:"):
            index = int(fast_device.split(":", 1)[1])
            properties = torch.cuda.get_device_properties(index)
            hardware.update(
                {
                    "gpu_name": properties.name,
                    "gpu_capability": list(torch.cuda.get_device_capability(index)),
                    "gpu_total_memory": int(properties.total_memory),
                }
            )
        payload = {
            "schema": "service-time-route-v1",
            "fast_device": fast_device,
            "worker_count": self._worker_count,
            "hardware": hardware,
        }
        return f"service-time-route-v1:{_canonical_sha256(payload)}"

    def reset_route_circuit(
        self,
        *,
        route_key: str,
        expected_evidence_sha256: str,
        approval_sha256: str,
        actor: str,
        reason: str,
    ) -> tuple[CertificationCircuitReset, bool]:
        if self._policy_store is None:
            raise ShadowCircuitResetDisabledError(
                "circuit reset requires an explicit certification policy"
            )
        if self._circuit_reset_cooldown_seconds is None:
            raise ShadowCircuitResetDisabledError("circuit reset is disabled")
        return self._policy_store.reset_circuit(
            route_key=route_key,
            expected_evidence_sha256=expected_evidence_sha256,
            approval_sha256=approval_sha256,
            actor=actor,
            reason=reason,
            cooldown_seconds=self._circuit_reset_cooldown_seconds,
        )

    def health_check(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("durable shadow service is closed")
        self._journal.quick_check()
        if self._policy_store is not None:
            self._policy_store.quick_check()

    def submit(
        self,
        request: ProductionRequest,
        *,
        submission_id: str,
    ) -> OnlineShadowSubmission:
        if not isinstance(submission_id, str) or not submission_id or len(submission_id) > 256:
            raise ValueError("submission_id must be a nonempty string of at most 256 characters")
        if request.exact_backend != "osqp":
            raise ValueError("durable shadow service requires exact_backend='osqp'")
        if request.fallback_policy is not FallbackPolicy.DENY:
            raise ValueError("durable shadow service requires fallback_policy=DENY")
        with self._lock:
            if self._closed:
                raise RuntimeError("durable shadow service is closed")
        primary_request = _freeze_request(request, exact_backend="osqp")
        shadow_request = _freeze_request(request, exact_backend="clarabel")
        primary_request_sha256 = _request_hash(primary_request)
        shadow_request_sha256 = _request_hash(shadow_request)
        existing = self._journal.get_by_submission_id(submission_id)
        if existing is not None:
            return self._submission_from_job(
                existing,
                expected_primary_request_sha256=primary_request_sha256,
                expected_shadow_request_sha256=shadow_request_sha256,
            )
        if self._policy_store is not None:
            stored_decision = self._policy_store.decision_record(submission_id)
            if stored_decision is not None:
                if (
                    stored_decision.route_key != request.fast_device
                    or stored_decision.decision.policy_sha256
                    != self._certification_policy.policy_sha256
                    or stored_decision.decision.request_sha256
                    != primary_request_sha256
                ):
                    raise ShadowSubmissionConflictError(
                        f"submission_id {submission_id!r} is already bound to different policy evidence"
                    )
                if stored_decision.ticket_sha256 is not None:
                    raise RuntimeError(
                        "queued policy decision is missing its durable shadow job"
                    )
                online = production_from_payload(
                    decode_payload(stored_decision.online_payload)
                )
                if (
                    online.request_sha256 != primary_request_sha256
                    or online.result_sha256
                    != stored_decision.decision.result_sha256
                ):
                    raise RuntimeError("stored skipped policy evidence is inconsistent")
                validate_certification_decision(
                    stored_decision.decision,
                    CertificationObservation.from_production_result(online),
                )
                return OnlineShadowSubmission(
                    ticket=None,
                    online=online,
                    online_elapsed_seconds=stored_decision.online_elapsed_seconds,
                    policy_decision=stored_decision.decision,
                )
            circuit = self._policy_store.circuit_state(request.fast_device)
            if circuit is not None:
                raise ShadowRouteCircuitOpenError(
                    f"shadow route {request.fast_device!r} is circuit-open: {circuit.reason}"
                )

        submitted_at = time.time()
        started = time.perf_counter()
        online = solve_production(primary_request)
        online_elapsed = time.perf_counter() - started
        online_completed_at = time.time()
        if online.status is not SolveStatus.SOLVED_TO_TOLERANCE:
            raise RuntimeError(
                f"online OSQP production certification failed with {online.status.value}"
            )
        if online.exact.status is not SolveStatus.OPTIMAL:
            raise RuntimeError(f"online OSQP exact lane failed with {online.exact.status.value}")
        policy_decision: CertificationPolicyDecision | None = None
        online_payload = encode_payload(production_to_payload(online))
        if self._certification_policy is not None:
            policy_decision = self._certification_policy.evaluate(
                online.request_sha256,
                CertificationObservation.from_production_result(online),
            )
            if policy_decision.disposition is CertificationDisposition.SKIPPED:
                self._policy_store.record_decision(
                    submission_id=submission_id,
                    route_key=request.fast_device,
                    decision=policy_decision,
                    online_payload=online_payload,
                    online_elapsed_seconds=online_elapsed,
                )
                return OnlineShadowSubmission(
                    ticket=None,
                    online=online,
                    online_elapsed_seconds=online_elapsed,
                    policy_decision=policy_decision,
                )
            if policy_decision.disposition is not CertificationDisposition.QUEUED:
                raise RuntimeError(
                    f"unexpected certification disposition {policy_decision.disposition.value}"
                )
        with self._admission_lock:
            with self._lock:
                if self._closed:
                    raise RuntimeError("durable shadow service is closed")
            existing = self._journal.get_by_submission_id(submission_id)
            if existing is not None:
                return self._submission_from_job(
                    existing,
                    expected_primary_request_sha256=primary_request_sha256,
                    expected_shadow_request_sha256=shadow_request_sha256,
                )
            service_time_route_key = self._service_time_route_key(request.fast_device)
            self._enforce_backpressure(
                additional_jobs=1,
                service_time_route_key=service_time_route_key,
            )
            sequence = self._journal.allocate_sequence(self._service_epoch)
            ticket_nonce = uuid.uuid4().hex
            ticket_fields = {
                "ticket_nonce": ticket_nonce,
                "service_epoch": self._service_epoch,
                "sequence": sequence,
                "attempt": 1,
                "policy_sha256": self._policy_sha256,
                "input_sha256": online.input_sha256,
                "primary_request_sha256": online.request_sha256,
                "shadow_request_sha256": shadow_request_sha256,
            }
            ticket = ShadowTicket(
                ticket_sha256=_canonical_sha256(ticket_fields),
                **ticket_fields,
            )
            try:
                job, inserted = self._journal.insert_online_ready(
                    submission_id=submission_id,
                    ticket_sha256=ticket.ticket_sha256,
                    ticket_payload=encode_payload(_ticket_to_payload(ticket)),
                    request_payload=encode_payload(request_to_payload(shadow_request)),
                    online_payload=online_payload,
                    submitted_at=submitted_at,
                    online_completed_at=online_completed_at,
                )
            except ShadowJournalFullError as error:
                admission = self.admission_snapshot()
                raise ShadowBackpressureError(
                    reasons=("capacity_exhausted",),
                    admission=admission,
                ) from error
        if not inserted:
            return self._submission_from_job(
                job,
                expected_primary_request_sha256=primary_request_sha256,
                expected_shadow_request_sha256=shadow_request_sha256,
            )
        if self._policy_store is not None:
            self._policy_store.record_decision(
                submission_id=submission_id,
                route_key=request.fast_device,
                decision=policy_decision,
                online_payload=online_payload,
                online_elapsed_seconds=online_elapsed,
                ticket_sha256=ticket.ticket_sha256,
            )
        self._schedule_next()
        return OnlineShadowSubmission(
            ticket=ticket,
            online=online,
            online_elapsed_seconds=online_elapsed,
            policy_decision=policy_decision,
        )

    def _enforce_backpressure(
        self,
        *,
        additional_jobs: int = 0,
        service_time_route_key: str | None = None,
    ) -> None:
        admission = self.admission_snapshot(
            service_time_route_key=service_time_route_key
        )
        reasons = certification_backpressure_reasons(
            admission.capacity,
            max_queue_age_seconds=self._max_queue_age_seconds,
            worker_count=self._worker_count,
            service_time_budget_seconds=admission.service_time_budget_seconds,
            max_drain_seconds=self._max_drain_seconds,
            additional_jobs=additional_jobs,
        )
        if reasons:
            raise ShadowBackpressureError(
                reasons=reasons,
                admission=admission,
            )

    def collect(
        self,
        ticket: ShadowTicket,
        *,
        timeout: float | None = None,
    ) -> ShadowCertificationResult:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            job = self._validated_job(ticket)
            if job.state == "SUCCEEDED":
                if job.terminal_payload is None:
                    raise RuntimeError("durable shadow success is missing terminal payload")
                online = production_from_payload(decode_payload(job.online_payload))
                result = _validated_terminal_result(job, ticket, online)
                if (
                    self._policy_store is not None
                    and result.decision is ShadowDecision.DIVERGED
                ):
                    policy_record = self._policy_store.decision_record_by_ticket(
                        ticket.ticket_sha256
                    )
                    if policy_record is None:
                        raise RuntimeError(
                            "diverged shadow result is missing policy decision evidence"
                        )
                    self._policy_store.open_circuit(
                        route_key=policy_record.route_key,
                        policy_sha256=policy_record.decision.policy_sha256,
                        evidence_sha256=result.result_sha256,
                        reason="shadow_diverged",
                    )
                self._journal.mark_observed(ticket.ticket_sha256)
                return result
            if job.state == "FAILED":
                self._journal.mark_observed(ticket.ticket_sha256)
                raise ShadowExecutionError(
                    error_type=job.error_type or "UnknownError",
                    message=job.error_message or "durable shadow worker failed",
                    traceback_text=job.error_traceback or "",
                )
            if job.state == "ACKED":
                raise RuntimeError("durable shadow ticket has already been acknowledged")
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"durable shadow certification timed out for ticket {ticket.ticket_sha256}"
                    )
                time.sleep(min(0.01, remaining))
            else:
                time.sleep(0.01)

    def is_ready(self, ticket: ShadowTicket) -> bool:
        return self._validated_job(ticket).state in {"SUCCEEDED", "FAILED"}

    def ack(self, ticket: ShadowTicket) -> bool:
        self._validated_job(ticket)
        return self._journal.ack(ticket.ticket_sha256)

    def shutdown(self, *, wait: bool = True, require_collected: bool = True) -> None:
        with self._admission_lock:
            with self._lock:
                self._closed = True
        self._scheduler_stop.set()
        self._scheduler.join(timeout=max(1.0, self._heartbeat_seconds * 2.0))
        self._executor.shutdown(wait=wait, cancel_futures=False)
        if require_collected and self._journal.outstanding_count:
            raise RuntimeError(
                "durable shadow service closed with "
                f"{self._journal.outstanding_count} unacknowledged ticket(s)"
            )

    def _submission_from_job(
        self,
        job: DurableShadowJob,
        *,
        expected_primary_request_sha256: str | None = None,
        expected_shadow_request_sha256: str | None = None,
    ) -> OnlineShadowSubmission:
        ticket = _ticket_from_payload(decode_payload(job.ticket_payload))
        if ticket.policy_sha256 != self._policy_sha256:
            raise RuntimeError("durable shadow submission uses an unsupported frozen policy")
        if (
            expected_primary_request_sha256 is not None
            and ticket.primary_request_sha256 != expected_primary_request_sha256
        ) or (
            expected_shadow_request_sha256 is not None
            and ticket.shadow_request_sha256 != expected_shadow_request_sha256
        ):
            raise ShadowSubmissionConflictError(
                f"submission_id {job.submission_id!r} is already bound to a different request"
            )
        online = production_from_payload(decode_payload(job.online_payload))
        if (
            online.input_sha256 != ticket.input_sha256
            or online.request_sha256 != ticket.primary_request_sha256
        ):
            raise RuntimeError("durable shadow stored online evidence mismatch")
        policy_decision: CertificationPolicyDecision | None = None
        online_elapsed_seconds = max(
            0.0, job.online_completed_at - job.submitted_at
        )
        if self._policy_store is not None:
            policy_record = self._policy_store.decision_record(job.submission_id)
            if policy_record is None:
                policy_decision = self._certification_policy.evaluate(
                    online.request_sha256,
                    CertificationObservation.from_production_result(online),
                )
                if policy_decision.disposition is not CertificationDisposition.QUEUED:
                    raise RuntimeError(
                        "durable shadow job contradicts reconstructed policy decision"
                    )
                self._policy_store.record_decision(
                    submission_id=job.submission_id,
                    route_key=online.fast.device,
                    decision=policy_decision,
                    online_payload=job.online_payload,
                    online_elapsed_seconds=online_elapsed_seconds,
                    ticket_sha256=ticket.ticket_sha256,
                )
            else:
                if (
                    policy_record.ticket_sha256 != ticket.ticket_sha256
                    or policy_record.decision.request_sha256
                    != online.request_sha256
                    or policy_record.decision.result_sha256 != online.result_sha256
                    or policy_record.decision.disposition
                    is not CertificationDisposition.QUEUED
                ):
                    raise RuntimeError(
                        "durable shadow job policy evidence is inconsistent"
                    )
                policy_decision = policy_record.decision
                validate_certification_decision(
                    policy_decision,
                    CertificationObservation.from_production_result(online),
                )
        return OnlineShadowSubmission(
            ticket=ticket,
            online=online,
            online_elapsed_seconds=online_elapsed_seconds,
            policy_decision=policy_decision,
        )

    def _validated_job(self, ticket: ShadowTicket) -> DurableShadowJob:
        job = self._journal.get_by_ticket(ticket.ticket_sha256)
        if job is None:
            raise KeyError(f"unknown or forged durable shadow ticket {ticket.ticket_sha256}")
        stored = _ticket_from_payload(decode_payload(job.ticket_payload))
        if stored != ticket:
            raise KeyError(f"unknown or forged durable shadow ticket {ticket.ticket_sha256}")
        return job

    def _scheduler_loop(self) -> None:
        interval = min(1.0, self._heartbeat_seconds)
        while not self._scheduler_stop.wait(interval):
            self._schedule_next()

    def _schedule_next(self) -> None:
        futures: list[Future[None]] = []
        with self._lock:
            if self._closed:
                return
            while len(self._active) < self._worker_count:
                job = self._journal.claim_next(
                    owner=self._owner,
                    lease_seconds=self._lease_seconds,
                )
                if job is None:
                    break
                future = self._executor.submit(self._run_job, job)
                self._active.add(future)
                futures.append(future)
        for future in futures:
            future.add_done_callback(self._job_finished)

    def _job_finished(self, future: Future[None]) -> None:
        try:
            future.result()
        finally:
            with self._lock:
                self._active.discard(future)
            self._schedule_next()

    def _run_job(self, job: DurableShadowJob) -> None:
        heartbeat_stop = Event()
        heartbeat = Thread(
            target=self._heartbeat_loop,
            args=(job, heartbeat_stop),
            name=f"openpo-shadow-heartbeat-{job.job_id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            ticket = _ticket_from_payload(decode_payload(job.ticket_payload))
            request = request_from_payload(decode_payload(job.request_payload))
            online = production_from_payload(decode_payload(job.online_payload))
            result = self._execute_shadow(job, ticket, request, online.exact)
            encoded = encode_payload(_result_to_payload(result))
            accepted = self._journal.complete_success(
                job_id=job.job_id,
                owner=self._owner,
                fence_token=job.fence_token,
                terminal_payload=encoded,
                service_time_seconds=result.shadow_elapsed_seconds,
                service_time_route_key=self._service_time_route_key(
                    request.fast_device
                ),
            )
            if not accepted:
                return
            if self._policy_store is not None and result.decision is ShadowDecision.DIVERGED:
                policy_record = self._policy_store.decision_record_by_ticket(
                    ticket.ticket_sha256
                )
                if policy_record is None:
                    raise RuntimeError(
                        "diverged shadow result is missing policy decision evidence"
                    )
                self._policy_store.open_circuit(
                    route_key=policy_record.route_key,
                    policy_sha256=policy_record.decision.policy_sha256,
                    evidence_sha256=result.result_sha256,
                    reason="shadow_diverged",
                )
        except BaseException as error:
            self._journal.complete_failure(
                job_id=job.job_id,
                owner=self._owner,
                fence_token=job.fence_token,
                error_type=type(error).__name__,
                error_message=str(error),
                error_traceback=traceback.format_exc(),
            )
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=max(1.0, self._heartbeat_seconds * 2.0))

    def _heartbeat_loop(self, job: DurableShadowJob, stop: Event) -> None:
        while not stop.wait(self._heartbeat_seconds):
            if not self._journal.heartbeat(
                job_id=job.job_id,
                owner=self._owner,
                fence_token=job.fence_token,
                lease_seconds=self._lease_seconds,
            ):
                return

    def _execute_shadow(
        self,
        job: DurableShadowJob,
        ticket: ShadowTicket,
        request: ProductionRequest,
        primary_exact: LaneResult,
    ) -> ShadowCertificationResult:
        started_wall = time.time()
        started = time.perf_counter()
        shadow = certify_exact(request)
        shadow_elapsed = time.perf_counter() - started
        completed_wall = time.time()
        if shadow.input_sha256 != ticket.input_sha256:
            raise RuntimeError("Clarabel shadow used a different frozen input")
        if shadow.request_sha256 != ticket.shadow_request_sha256:
            raise RuntimeError("Clarabel shadow request hash does not match its ticket")
        denominator = primary_exact.objective.abs().clamp_min(1e-12)
        objective_delta = float(
            ((shadow.objective - primary_exact.objective).abs() / denominator).amax().item()
        )
        weights_delta = float((shadow.weights - primary_exact.weights).abs().amax().item())
        passed = (
            shadow.status is SolveStatus.OPTIMAL
            and shadow.backend == "clarabel"
            and shadow.residuals.primal_residual <= request.residual_tolerance
            and objective_delta <= self._objective_tolerance
            and weights_delta <= self._weights_tolerance
        )
        decision = ShadowDecision.MATCH if passed else ShadowDecision.DIVERGED
        result_hash = _canonical_sha256(
            {
                "ticket_sha256": ticket.ticket_sha256,
                "policy_sha256": ticket.policy_sha256,
                "execution_attempt": job.execution_attempt,
                "fence_token": job.fence_token,
                "decision": decision.value,
                "primary_exact_result_sha256": primary_exact.result_sha256,
                "shadow_exact_result_sha256": shadow.result_sha256,
                "objective_relative_delta": float(objective_delta).hex(),
                "weights_max_abs_diff": float(weights_delta).hex(),
            }
        )
        return ShadowCertificationResult(
            ticket=ticket,
            decision=decision,
            passed=passed,
            primary_exact_result_sha256=primary_exact.result_sha256,
            shadow_exact_result_sha256=shadow.result_sha256,
            shadow=shadow,
            objective_relative_delta=objective_delta,
            weights_max_abs_diff=weights_delta,
            shadow_queue_wait_seconds=max(0.0, started_wall - job.online_completed_at),
            shadow_elapsed_seconds=shadow_elapsed,
            certification_age_seconds=max(0.0, completed_wall - job.online_completed_at),
            total_elapsed_seconds=max(0.0, completed_wall - job.submitted_at),
            result_sha256=result_hash,
            execution_attempt=job.execution_attempt,
            fence_token=job.fence_token,
            durable=True,
        )


def _ticket_to_payload(ticket: ShadowTicket) -> dict[str, object]:
    return {"schema": "shadow-ticket-v2", **asdict(ticket)}


def _ticket_from_payload(payload: dict[str, object]) -> ShadowTicket:
    if payload.pop("schema", None) != "shadow-ticket-v2":
        raise ValueError("unsupported durable shadow ticket schema")
    ticket = ShadowTicket(**payload)
    expected = _canonical_sha256(
        {
            "ticket_nonce": ticket.ticket_nonce,
            "service_epoch": ticket.service_epoch,
            "sequence": ticket.sequence,
            "attempt": ticket.attempt,
            "policy_sha256": ticket.policy_sha256,
            "input_sha256": ticket.input_sha256,
            "primary_request_sha256": ticket.primary_request_sha256,
            "shadow_request_sha256": ticket.shadow_request_sha256,
        }
    )
    if ticket.ticket_sha256 != expected:
        raise ValueError("durable shadow ticket hash mismatch")
    return ticket


def _result_to_payload(result: ShadowCertificationResult) -> dict[str, object]:
    return {
        "schema": "shadow-result-v2",
        "ticket": _ticket_to_payload(result.ticket),
        "decision": result.decision.value,
        "passed": result.passed,
        "primary_exact_result_sha256": result.primary_exact_result_sha256,
        "shadow_exact_result_sha256": result.shadow_exact_result_sha256,
        "shadow": lane_to_payload(result.shadow),
        "objective_relative_delta": float(result.objective_relative_delta).hex(),
        "weights_max_abs_diff": float(result.weights_max_abs_diff).hex(),
        "shadow_queue_wait_seconds": float(result.shadow_queue_wait_seconds).hex(),
        "shadow_elapsed_seconds": float(result.shadow_elapsed_seconds).hex(),
        "certification_age_seconds": float(result.certification_age_seconds).hex(),
        "total_elapsed_seconds": float(result.total_elapsed_seconds).hex(),
        "result_sha256": result.result_sha256,
        "execution_attempt": result.execution_attempt,
        "fence_token": result.fence_token,
        "durable": result.durable,
    }


def _result_from_payload(payload: dict[str, object]) -> ShadowCertificationResult:
    if payload.get("schema") != "shadow-result-v2":
        raise ValueError("unsupported durable shadow result schema")
    passed = _payload_bool(payload.get("passed"), "passed")
    decision = ShadowDecision(_payload_string(payload.get("decision"), "decision"))
    execution_attempt = _payload_positive_int(
        payload.get("execution_attempt"), "execution_attempt"
    )
    fence_token = _payload_positive_int(payload.get("fence_token"), "fence_token")
    durable = _payload_bool(payload.get("durable"), "durable")
    result = ShadowCertificationResult(
        ticket=_ticket_from_payload(dict(payload["ticket"])),
        decision=decision,
        passed=passed,
        primary_exact_result_sha256=_payload_string(
            payload.get("primary_exact_result_sha256"), "primary_exact_result_sha256"
        ),
        shadow_exact_result_sha256=_payload_string(
            payload.get("shadow_exact_result_sha256"), "shadow_exact_result_sha256"
        ),
        shadow=lane_from_payload(dict(payload["shadow"])),
        objective_relative_delta=_payload_hex_float(
            payload.get("objective_relative_delta"), "objective_relative_delta"
        ),
        weights_max_abs_diff=_payload_hex_float(
            payload.get("weights_max_abs_diff"), "weights_max_abs_diff"
        ),
        shadow_queue_wait_seconds=_payload_hex_float(
            payload.get("shadow_queue_wait_seconds"), "shadow_queue_wait_seconds"
        ),
        shadow_elapsed_seconds=_payload_hex_float(
            payload.get("shadow_elapsed_seconds"), "shadow_elapsed_seconds"
        ),
        certification_age_seconds=_payload_hex_float(
            payload.get("certification_age_seconds"), "certification_age_seconds"
        ),
        total_elapsed_seconds=_payload_hex_float(
            payload.get("total_elapsed_seconds"), "total_elapsed_seconds"
        ),
        result_sha256=_payload_string(payload.get("result_sha256"), "result_sha256"),
        execution_attempt=execution_attempt,
        fence_token=fence_token,
        durable=durable,
    )
    expected_hash = _canonical_sha256(
        {
            "ticket_sha256": result.ticket.ticket_sha256,
            "policy_sha256": result.ticket.policy_sha256,
            "execution_attempt": result.execution_attempt,
            "fence_token": result.fence_token,
            "decision": result.decision.value,
            "primary_exact_result_sha256": result.primary_exact_result_sha256,
            "shadow_exact_result_sha256": result.shadow_exact_result_sha256,
            "objective_relative_delta": float(result.objective_relative_delta).hex(),
            "weights_max_abs_diff": float(result.weights_max_abs_diff).hex(),
        }
    )
    if result.result_sha256 != expected_hash:
        raise ValueError("durable shadow terminal result hash mismatch")
    if result.passed is not (result.decision is ShadowDecision.MATCH):
        raise ValueError("durable shadow terminal decision/pass mismatch")
    if not result.durable:
        raise ValueError("durable shadow terminal result must be durable")
    return result


def _validated_terminal_result(
    job: DurableShadowJob,
    ticket: ShadowTicket,
    online: ProductionResult,
) -> ShadowCertificationResult:
    if job.terminal_payload is None:
        raise RuntimeError("durable shadow success is missing terminal payload")
    result = _result_from_payload(decode_payload(job.terminal_payload))
    if result.ticket != ticket:
        raise RuntimeError("durable shadow terminal ticket mismatch")
    if (
        result.execution_attempt != job.execution_attempt
        or result.fence_token != job.fence_token
    ):
        raise RuntimeError("durable shadow terminal fence evidence mismatch")
    if result.primary_exact_result_sha256 != online.exact.result_sha256:
        raise RuntimeError("durable shadow terminal primary evidence mismatch")
    if result.shadow_exact_result_sha256 != result.shadow.result_sha256:
        raise RuntimeError("durable shadow terminal shadow evidence mismatch")
    if (
        result.shadow.input_sha256 != ticket.input_sha256
        or result.shadow.request_sha256 != ticket.shadow_request_sha256
    ):
        raise RuntimeError("durable shadow terminal frozen input evidence mismatch")
    return result


def _payload_string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"durable shadow terminal {field} must be a string")
    return value


def _payload_bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"durable shadow terminal {field} must be boolean")
    return value


def _payload_positive_int(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"durable shadow terminal {field} must be a positive integer")
    return value


def _payload_hex_float(value: object, field: str) -> float:
    return float.fromhex(_payload_string(value, field))
