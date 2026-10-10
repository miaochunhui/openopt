from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
from threading import Barrier, Event, Lock
import time

import pytest

from openpo.certification_policy import (
    CertificationDisposition,
    CertificationMode,
    CertificationPolicy,
    CertificationReason,
)
from openpo.durable_shadow import (
    DurableShadowCertificationService,
    ShadowBackpressureError,
    ShadowCircuitResetDisabledError,
    ShadowExecutionError,
    ShadowRouteCircuitOpenError,
    ShadowSubmissionConflictError,
    certification_backpressure_reasons,
    certification_estimated_drain_seconds,
)
from openpo.facade import solve_production
from openpo.shadow import ShadowQueueFullError
from openpo.shadow import ShadowTicket
from openpo.shadow_journal import ShadowCapacitySnapshot, SQLiteShadowJournal
from openpo.service_time_calibration import ServiceTimeCalibrationConfig
from tests.test_facade import _request


def _request_for_policy_selection(
    policy: CertificationPolicy,
    *,
    selected: bool,
):
    request = _request()
    for iterations in range(
        request.fast_config.iterations,
        request.fast_config.iterations + 100,
    ):
        candidate = replace(
            request,
            fast_config=replace(request.fast_config, iterations=iterations),
        )
        result = solve_production(candidate)
        is_selected = policy.sample_bucket(result.request_sha256) < policy.sample_numerator
        if is_selected is selected:
            return candidate
    raise AssertionError("failed to construct request for policy selection")


def test_durable_shadow_submit_collect_ack_and_retry_are_idempotent(tmp_path) -> None:
    path = tmp_path / "shadow.sqlite3"
    service = DurableShadowCertificationService(path, max_pending=1)
    first = service.submit(_request(), submission_id="order-1")
    duplicate = service.submit(_request(), submission_id="order-1")

    assert duplicate.ticket == first.ticket
    assert duplicate.online.result_sha256 == first.online.result_sha256
    result = service.collect(first.ticket, timeout=10.0)
    assert result.passed is True
    assert result.durable is True
    assert result.execution_attempt == 1
    assert result.fence_token == 1
    assert service.ack(first.ticket) is True
    assert service.ack(first.ticket) is False
    assert service.outstanding_count == 0
    service.shutdown()


def test_durable_shadow_rejects_submission_id_reuse_for_different_request(tmp_path) -> None:
    service = DurableShadowCertificationService(tmp_path / "shadow.sqlite3")
    request = _request()
    first = service.submit(request, submission_id="order-1")
    changed = replace(
        request,
        problem=replace(request.problem, alpha=request.problem.alpha + 0.01),
    )

    with pytest.raises(ShadowSubmissionConflictError, match="different request"):
        service.submit(changed, submission_id="order-1")

    service.collect(first.ticket, timeout=10.0)
    service.ack(first.ticket)
    service.shutdown()


def test_durable_shadow_terminal_result_survives_restart(tmp_path) -> None:
    path = tmp_path / "shadow.sqlite3"
    first_service = DurableShadowCertificationService(path)
    submission = first_service.submit(_request(), submission_id="order-1")
    first_result = first_service.collect(submission.ticket, timeout=10.0)
    first_service.shutdown(require_collected=False)

    second_service = DurableShadowCertificationService(path)
    restored = second_service.collect(submission.ticket, timeout=0.0)
    assert restored.result_sha256 == first_result.result_sha256
    assert restored.shadow.result_sha256 == first_result.shadow.result_sha256
    assert second_service.ack(submission.ticket) is True
    second_service.shutdown()


def test_durable_shadow_restart_rejects_terminal_scalar_tampering(tmp_path) -> None:
    path = tmp_path / "shadow.sqlite3"
    service = DurableShadowCertificationService(path)
    submission = service.submit(_request(), submission_id="order-1")
    service.collect(submission.ticket, timeout=10.0)
    service.shutdown(require_collected=False)

    with sqlite3.connect(path) as connection:
        encoded = connection.execute(
            "SELECT terminal_payload FROM jobs WHERE submission_id = ?",
            ("order-1",),
        ).fetchone()[0]
        payload = json.loads(encoded.decode("utf-8"))
        payload["objective_relative_delta"] = float(0.5).hex()
        connection.execute(
            "UPDATE jobs SET terminal_payload = ? WHERE submission_id = ?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"), "order-1"),
        )

    with pytest.raises(ValueError, match="terminal result hash mismatch"):
        DurableShadowCertificationService(path)


def test_durable_shadow_failure_survives_and_requires_ack(tmp_path, monkeypatch) -> None:
    from openpo import durable_shadow as durable_module

    def fail(request):
        raise RuntimeError("durable backend failed")

    monkeypatch.setattr(durable_module, "certify_exact", fail)
    service = DurableShadowCertificationService(tmp_path / "shadow.sqlite3")
    submission = service.submit(_request(), submission_id="order-1")

    with pytest.raises(ShadowExecutionError, match="durable backend failed") as captured:
        service.collect(submission.ticket, timeout=10.0)
    assert captured.value.error_type == "RuntimeError"
    assert service.outstanding_count == 1
    assert service.ack(submission.ticket) is True
    service.shutdown()


def test_durable_shadow_capacity_includes_terminal_unacked(tmp_path) -> None:
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        max_pending=1,
    )
    first = service.submit(_request(), submission_id="order-1")
    service.collect(first.ticket, timeout=10.0)

    with pytest.raises(ShadowQueueFullError, match="queue is full"):
        service.submit(_request(), submission_id="order-2")

    service.ack(first.ticket)
    second = service.submit(_request(), submission_id="order-2")
    service.collect(second.ticket, timeout=10.0)
    service.ack(second.ticket)
    service.shutdown()


def test_durable_shadow_queue_age_backpressure_is_explicit(tmp_path, monkeypatch) -> None:
    path = tmp_path / "shadow.sqlite3"
    service = DurableShadowCertificationService(
        path,
        max_pending=2,
        max_queue_age_seconds=1.0,
    )
    monkeypatch.setattr(service, "_schedule_next", lambda: None)
    first = service.submit(_request(), submission_id="order-1")
    assert first.ticket is not None
    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE jobs SET online_completed_at = ? WHERE submission_id = ?",
            (time.time() - 10.0, "order-1"),
        )

    with pytest.raises(ShadowBackpressureError) as captured:
        service.submit(_request(), submission_id="order-2")
    assert captured.value.reasons == ("queue_age_exceeded",)
    assert captured.value.capacity.outstanding == 1
    assert captured.value.capacity.available == 1
    assert captured.value.capacity.oldest_inflight_age_seconds >= 10.0
    service.shutdown(require_collected=False)


def test_certification_drain_estimate_and_backpressure_are_worker_aware() -> None:
    snapshot = ShadowCapacitySnapshot(
        max_outstanding=8,
        outstanding=5,
        online_ready=3,
        leased=2,
        terminal_unacked=0,
        oldest_outstanding_age_seconds=4.0,
        oldest_inflight_age_seconds=4.0,
        oldest_ready_age_seconds=3.0,
    )

    assert certification_estimated_drain_seconds(
        snapshot,
        worker_count=2,
        service_time_budget_seconds=10.0,
    ) == 30.0
    assert certification_backpressure_reasons(
        snapshot,
        max_queue_age_seconds=60.0,
        worker_count=2,
        service_time_budget_seconds=10.0,
        max_drain_seconds=20.0,
    ) == ("drain_time_exceeded",)
    assert certification_backpressure_reasons(
        snapshot,
        max_queue_age_seconds=60.0,
        worker_count=3,
        service_time_budget_seconds=10.0,
        max_drain_seconds=20.0,
    ) == ()
    assert certification_backpressure_reasons(
        replace(snapshot, online_ready=4, leased=0, outstanding=4),
        max_queue_age_seconds=60.0,
        worker_count=2,
        service_time_budget_seconds=10.0,
        max_drain_seconds=20.0,
        additional_jobs=1,
    ) == ("drain_time_exceeded",)
    assert certification_backpressure_reasons(
        replace(snapshot, max_outstanding=6),
        max_queue_age_seconds=60.0,
        worker_count=2,
        service_time_budget_seconds=10.0,
        max_drain_seconds=60.0,
        additional_jobs=2,
    ) == ("capacity_exhausted",)


def test_shutdown_serializes_with_projected_admission(tmp_path, monkeypatch) -> None:
    from openpo import durable_shadow as durable_module

    original = durable_module.solve_production
    entered = Event()
    release = Event()

    def blocked_solve(request):
        entered.set()
        assert release.wait(timeout=10.0)
        return original(request)

    monkeypatch.setattr(durable_module, "solve_production", blocked_solve)
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        max_pending=2,
    )
    monkeypatch.setattr(service, "_schedule_next", lambda: None)

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(service.submit, _request(), submission_id="closing")
        assert entered.wait(timeout=10.0)
        service.shutdown(wait=False, require_collected=False)
        release.set()
        with pytest.raises(RuntimeError, match="durable shadow service is closed"):
            future.result(timeout=10.0)

    assert service.capacity_snapshot().outstanding == 0


def test_durable_shadow_serializes_projected_drain_admission(
    tmp_path, monkeypatch
) -> None:
    from openpo import durable_shadow as durable_module

    original = durable_module.solve_production
    barrier = Barrier(2, timeout=10.0)

    def synchronized_solve(request):
        result = original(request)
        barrier.wait()
        return result

    monkeypatch.setattr(durable_module, "solve_production", synchronized_solve)
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        max_pending=4,
        worker_count=1,
        certification_service_time_budget_seconds=10.0,
        max_drain_seconds=10.0,
    )
    monkeypatch.setattr(service, "_schedule_next", lambda: None)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(service.submit, _request(), submission_id=f"race-{index}")
            for index in range(2)
        ]
    outcomes = []
    for future in futures:
        try:
            outcomes.append(future.result())
        except ShadowBackpressureError as error:
            outcomes.append(error)

    submissions = [outcome for outcome in outcomes if not isinstance(outcome, Exception)]
    errors = [outcome for outcome in outcomes if isinstance(outcome, Exception)]
    assert len(submissions) == 1
    assert len(errors) == 1
    assert errors[0].reasons == ("drain_time_exceeded",)
    assert service.capacity_snapshot().online_ready == 1
    service.shutdown(require_collected=False)


def test_durable_shadow_runs_configured_workers_concurrently(
    tmp_path, monkeypatch
) -> None:
    from openpo import durable_shadow as durable_module

    original = durable_module.certify_exact
    barrier = Barrier(2, timeout=10.0)
    counter_lock = Lock()
    active = 0
    maximum_active = 0

    def synchronized_certify(request):
        nonlocal active, maximum_active
        with counter_lock:
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            barrier.wait()
            return original(request)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(durable_module, "certify_exact", synchronized_certify)
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        max_pending=2,
        worker_count=2,
        certification_service_time_budget_seconds=10.0,
        max_drain_seconds=10.0,
    )
    first = service.submit(_request(), submission_id="worker-1")
    second = service.submit(_request(), submission_id="worker-2")

    assert service.collect(first.ticket, timeout=10.0).passed
    assert service.collect(second.ticket, timeout=10.0).passed
    assert maximum_active == 2
    assert service.ack(first.ticket)
    assert service.ack(second.ticket)
    service.shutdown()


def test_durable_shadow_service_time_calibration_uses_static_cold_start(
    tmp_path,
) -> None:
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        certification_service_time_budget_seconds=10.0,
        service_time_calibration=ServiceTimeCalibrationConfig(minimum_samples=2),
        max_drain_seconds=20.0,
    )

    admission = service.admission_snapshot()
    assert admission.service_time_budget_seconds == 10.0
    assert admission.service_time_budget_source == "static_cold_start"
    assert admission.service_time_calibration is not None
    assert admission.service_time_calibration.total_sample_count == 0
    assert admission.service_time_calibration.effective_budget_seconds == 10.0
    service.shutdown()


def test_durable_shadow_service_time_calibration_restores_accepted_samples(
    tmp_path,
) -> None:
    path = tmp_path / "shadow.sqlite3"
    journal = SQLiteShadowJournal(path, max_outstanding=1)
    job, inserted = journal.insert_online_ready(
        submission_id="seed",
        ticket_sha256="a" * 64,
        ticket_payload=b"ticket",
        request_payload=b"request",
        online_payload=b"online",
        submitted_at=10.0,
        online_completed_at=11.0,
    )
    assert inserted
    claimed = journal.claim_next(owner="seed-worker", lease_seconds=60.0, now=20.0)
    assert claimed is not None
    assert journal.complete_success(
        job_id=claimed.job_id,
        owner="seed-worker",
        fence_token=claimed.fence_token,
        terminal_payload=b"result",
        service_time_seconds=4.0,
    )
    journal.mark_observed(job.ticket_sha256)
    assert journal.ack(job.ticket_sha256)

    service = DurableShadowCertificationService(
        path,
        max_pending=1,
        certification_service_time_budget_seconds=10.0,
        service_time_calibration=ServiceTimeCalibrationConfig(
            window_size=4,
            minimum_samples=1,
        ),
        max_drain_seconds=20.0,
    )
    admission = service.admission_snapshot()
    assert admission.service_time_budget_source == "adaptive_quantile"
    assert admission.service_time_budget_seconds == 5.0
    assert admission.service_time_calibration is not None
    assert admission.service_time_calibration.total_sample_count == 1
    assert admission.service_time_calibration.retained_sample_count == 1
    assert admission.service_time_calibration.quantile_seconds == 4.0
    service.shutdown()


def test_durable_shadow_service_time_calibration_is_route_scoped(tmp_path) -> None:
    path = tmp_path / "shadow.sqlite3"
    journal = SQLiteShadowJournal(path, max_outstanding=1)
    job, inserted = journal.insert_online_ready(
        submission_id="seed",
        ticket_sha256="a" * 64,
        ticket_payload=b"ticket",
        request_payload=b"request",
        online_payload=b"online",
        submitted_at=10.0,
        online_completed_at=11.0,
    )
    assert inserted
    claimed = journal.claim_next(owner="seed-worker", lease_seconds=60.0, now=20.0)
    assert claimed is not None
    assert journal.complete_success(
        job_id=claimed.job_id,
        owner="seed-worker",
        fence_token=claimed.fence_token,
        terminal_payload=b"result",
        service_time_seconds=4.0,
        service_time_route_key="route-a",
    )
    journal.mark_observed(job.ticket_sha256)
    assert journal.ack(job.ticket_sha256)

    service = DurableShadowCertificationService(
        path,
        max_pending=1,
        certification_service_time_budget_seconds=10.0,
        service_time_calibration=ServiceTimeCalibrationConfig(
            window_size=4,
            minimum_samples=1,
        ),
        max_drain_seconds=20.0,
    )
    route_a = service.admission_snapshot(service_time_route_key="route-a")
    route_b = service.admission_snapshot(service_time_route_key="route-b")
    assert route_a.service_time_budget_source == "adaptive_quantile"
    assert route_a.service_time_budget_seconds == 5.0
    assert route_a.service_time_calibration is not None
    assert route_a.service_time_calibration.route_key == "route-a"
    assert route_b.service_time_budget_source == "static_cold_start"
    assert route_b.service_time_budget_seconds == 10.0
    assert route_b.service_time_calibration is not None
    assert route_b.service_time_calibration.route_key == "route-b"
    service.shutdown()


def test_durable_shadow_refills_worker_slots_without_starvation(
    tmp_path, monkeypatch
) -> None:
    from openpo import durable_shadow as durable_module

    original = durable_module.certify_exact
    release_first_wave = Event()
    counter_lock = Lock()
    started: list[int] = []
    active = 0
    maximum_active = 0

    def tracked_certify(request):
        nonlocal active, maximum_active
        with counter_lock:
            call_index = len(started)
            started.append(call_index)
            active += 1
            maximum_active = max(maximum_active, active)
        try:
            if call_index < 2 and not release_first_wave.wait(timeout=10.0):
                raise TimeoutError("first worker wave was not released")
            return original(request)
        finally:
            with counter_lock:
                active -= 1

    monkeypatch.setattr(durable_module, "certify_exact", tracked_certify)
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        max_pending=5,
        worker_count=2,
    )
    submissions = [
        service.submit(_request(), submission_id=f"refill-{index}")
        for index in range(5)
    ]
    release_first_wave.set()
    results = [service.collect(submission.ticket, timeout=10.0) for submission in submissions]

    assert maximum_active == 2
    assert started == list(range(5))
    assert all(result.execution_attempt == 1 for result in results)
    assert all(result.fence_token == 1 for result in results)
    assert all(service.ack(submission.ticket) for submission in submissions)
    service.shutdown()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_pending": 1, "worker_count": 2},
        {"certification_service_time_budget_seconds": 10.0},
        {"max_drain_seconds": 10.0},
    ],
)
def test_durable_shadow_rejects_invalid_worker_capacity_contract(
    tmp_path, kwargs
) -> None:
    with pytest.raises(ValueError):
        DurableShadowCertificationService(tmp_path / "shadow.sqlite3", **kwargs)


def test_durable_shadow_circuit_reset_is_disabled_by_default_and_audited(
    tmp_path,
) -> None:
    path = tmp_path / "shadow.sqlite3"
    policy = CertificationPolicy()
    disabled = DurableShadowCertificationService(
        path,
        certification_policy=policy,
    )
    assert disabled._policy_store is not None
    assert disabled._policy_store.open_circuit(
        route_key="cpu",
        policy_sha256=policy.policy_sha256,
        evidence_sha256="1" * 64,
        reason="test_divergence",
    )
    with pytest.raises(ShadowCircuitResetDisabledError, match="disabled"):
        disabled.reset_route_circuit(
            route_key="cpu",
            expected_evidence_sha256="1" * 64,
            approval_sha256="2" * 64,
            actor="risk-oncall",
            reason="incident reviewed",
        )
    disabled.shutdown()

    enabled = DurableShadowCertificationService(
        path,
        certification_policy=policy,
        circuit_reset_cooldown_seconds=0.0,
    )
    reset, changed = enabled.reset_route_circuit(
        route_key="cpu",
        expected_evidence_sha256="1" * 64,
        approval_sha256="2" * 64,
        actor="risk-oncall",
        reason="incident reviewed",
    )
    assert changed is True
    assert reset.actor == "risk-oncall"
    submission = enabled.submit(_request(), submission_id="after-reset")
    assert submission.ticket is not None
    enabled.collect(submission.ticket, timeout=10.0)
    enabled.ack(submission.ticket)
    enabled.shutdown()


def test_durable_shadow_rejects_forged_ticket(tmp_path) -> None:
    service = DurableShadowCertificationService(tmp_path / "shadow.sqlite3")
    submission = service.submit(_request(), submission_id="order-1")
    forged = replace(submission.ticket, attempt=2)

    with pytest.raises(KeyError, match="forged"):
        service.collect(forged, timeout=0.0)
    service.collect(submission.ticket, timeout=10.0)
    service.ack(submission.ticket)
    service.shutdown()


def test_durable_shadow_rejects_outstanding_job_with_different_policy(tmp_path) -> None:
    path = tmp_path / "shadow.sqlite3"
    service = DurableShadowCertificationService(path, weights_tolerance=1e-5)
    submission = service.submit(_request(), submission_id="order-1")
    service.collect(submission.ticket, timeout=10.0)
    service.shutdown(require_collected=False)

    with pytest.raises(RuntimeError, match="unsupported frozen policy"):
        DurableShadowCertificationService(path, weights_tolerance=1e-4)

    resumed = DurableShadowCertificationService(path, weights_tolerance=1e-5)
    resumed.collect(submission.ticket, timeout=0.0)
    resumed.ack(submission.ticket)
    resumed.shutdown()


def test_durable_shadow_replays_expired_attempt_after_process_kill(tmp_path) -> None:
    journal_path = tmp_path / "shadow.sqlite3"
    ticket_path = tmp_path / "ticket.json"
    entered_path = tmp_path / "entered"
    helper = Path(__file__).parent / "helpers" / "durable_shadow_crash_worker.py"
    process = subprocess.Popen(
        [
            sys.executable,
            str(helper),
            str(journal_path),
            str(ticket_path),
            str(entered_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not (
            ticket_path.exists() and entered_path.exists()
        ):
            if process.poll() is not None:
                raise RuntimeError(f"crash worker exited early with {process.returncode}")
            time.sleep(0.02)
        assert ticket_path.exists() and entered_path.exists()
        process.kill()
        assert process.wait(timeout=10.0) < 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10.0)

    ticket = ShadowTicket(**json.loads(ticket_path.read_text(encoding="utf-8")))
    time.sleep(0.6)
    recovered = DurableShadowCertificationService(
        journal_path,
        lease_seconds=0.5,
        heartbeat_seconds=0.1,
    )
    result = recovered.collect(ticket, timeout=10.0)

    assert result.passed is True
    assert result.execution_attempt == 2
    assert result.fence_token == 2
    assert recovered.ack(ticket) is True
    recovered.shutdown()


def test_durable_shadow_replays_two_expired_workers_after_process_kill(tmp_path) -> None:
    journal_path = tmp_path / "shadow.sqlite3"
    tickets_path = tmp_path / "tickets.json"
    entered_path = tmp_path / "entered"
    helper = Path(__file__).parent / "helpers" / "durable_shadow_multi_crash_worker.py"
    process = subprocess.Popen(
        [
            sys.executable,
            str(helper),
            str(journal_path),
            str(tickets_path),
            str(entered_path),
        ],
        cwd=Path(__file__).resolve().parents[1],
    )
    try:
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline and not (
            tickets_path.exists() and entered_path.exists()
        ):
            if process.poll() is not None:
                raise RuntimeError(
                    f"multi-worker crash helper exited early with {process.returncode}"
                )
            time.sleep(0.02)
        assert tickets_path.exists() and entered_path.exists()
        process.kill()
        assert process.wait(timeout=10.0) < 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10.0)

    tickets = [
        ShadowTicket(**payload)
        for payload in json.loads(tickets_path.read_text(encoding="utf-8"))
    ]
    time.sleep(0.6)
    recovered = DurableShadowCertificationService(
        journal_path,
        max_pending=2,
        worker_count=2,
        lease_seconds=0.5,
        heartbeat_seconds=0.1,
    )
    results = [recovered.collect(ticket, timeout=10.0) for ticket in tickets]

    assert all(result.passed for result in results)
    assert [result.execution_attempt for result in results] == [2, 2]
    assert [result.fence_token for result in results] == [2, 2]
    assert all(recovered.ack(ticket) for ticket in tickets)
    recovered.shutdown()


def test_sampled_policy_skip_is_explicit_idempotent_and_runs_no_shadow(
    tmp_path,
    monkeypatch,
) -> None:
    from openpo import durable_shadow as durable_module

    def forbidden_shadow(request):
        raise AssertionError("skipped request must not run Clarabel")

    monkeypatch.setattr(durable_module, "certify_exact", forbidden_shadow)
    policy = CertificationPolicy(
        mode=CertificationMode.SAMPLED,
        sample_numerator=1,
        sample_denominator=2,
    )
    request = _request_for_policy_selection(policy, selected=False)
    path = tmp_path / "shadow.sqlite3"
    service = DurableShadowCertificationService(
        path,
        certification_policy=policy,
    )

    first = service.submit(request, submission_id="skipped-order")
    replay = service.submit(request, submission_id="skipped-order")

    assert first.ticket is None
    assert first.policy_decision is not None
    assert first.policy_decision.disposition is CertificationDisposition.SKIPPED
    assert first.policy_decision.reasons == (
        CertificationReason.HASH_NOT_SELECTED,
    )
    assert replay.ticket is None
    assert replay.online.result_sha256 == first.online.result_sha256
    assert replay.online_elapsed_seconds == first.online_elapsed_seconds
    assert replay.policy_decision == first.policy_decision
    assert service.outstanding_count == 0
    service.shutdown()

    restored = DurableShadowCertificationService(
        path,
        certification_policy=policy,
    )
    restored_submission = restored.submit(request, submission_id="skipped-order")
    assert restored_submission.ticket is None
    assert restored_submission.online.result_sha256 == first.online.result_sha256
    assert restored_submission.online_elapsed_seconds == first.online_elapsed_seconds
    assert restored_submission.policy_decision == first.policy_decision
    restored.shutdown()


def test_skipped_policy_restart_rejects_scalar_evidence_tampering(tmp_path) -> None:
    policy = CertificationPolicy(
        mode=CertificationMode.SAMPLED,
        sample_numerator=1,
        sample_denominator=2,
    )
    request = _request_for_policy_selection(policy, selected=False)
    path = tmp_path / "shadow.sqlite3"
    service = DurableShadowCertificationService(path, certification_policy=policy)
    submission = service.submit(request, submission_id="tampered-skip")
    assert submission.ticket is None
    service.shutdown()

    with sqlite3.connect(path) as connection:
        connection.execute(
            "UPDATE policy_decisions SET reasons_json = ? WHERE submission_id = ?",
            ('["hash_sample"]', "tampered-skip"),
        )

    restored = DurableShadowCertificationService(path, certification_policy=policy)
    with pytest.raises(RuntimeError, match="evidence hash is inconsistent"):
        restored.submit(request, submission_id="tampered-skip")
    restored.shutdown()


def test_sampled_policy_selected_request_runs_durable_shadow(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        CertificationPolicy,
        "sample_bucket",
        lambda self, request_sha256: 0,
    )
    policy = CertificationPolicy(
        mode=CertificationMode.SAMPLED,
        sample_numerator=1,
        sample_denominator=2,
    )
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        certification_policy=policy,
    )

    submission = service.submit(_request(), submission_id="selected-order")

    assert submission.ticket is not None
    assert submission.policy_decision is not None
    assert submission.policy_decision.disposition is CertificationDisposition.QUEUED
    assert submission.policy_decision.reasons == (CertificationReason.HASH_SAMPLE,)
    result = service.collect(submission.ticket, timeout=10.0)
    assert result.passed
    assert service.ack(submission.ticket)
    service.shutdown()


def test_anomaly_policy_queues_high_online_gap_without_hash_sample(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        CertificationPolicy,
        "sample_bucket",
        lambda self, request_sha256: 1,
    )
    policy = CertificationPolicy(
        mode=CertificationMode.ANOMALY,
        sample_numerator=1,
        sample_denominator=2,
        anomaly_objective_gap=1e-9,
    )
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        certification_policy=policy,
    )

    submission = service.submit(_request(), submission_id="anomaly-order")

    assert submission.ticket is not None
    assert submission.policy_decision is not None
    assert CertificationReason.OBJECTIVE_GAP in submission.policy_decision.reasons
    assert CertificationReason.HASH_SAMPLE not in submission.policy_decision.reasons
    service.collect(submission.ticket, timeout=10.0)
    service.ack(submission.ticket)
    service.shutdown()


def test_divergence_opens_durable_route_circuit_and_policy_change_does_not_reset(
    tmp_path,
    monkeypatch,
) -> None:
    from openpo import durable_shadow as durable_module

    original_certify = durable_module.certify_exact

    def divergent_shadow(request):
        result = original_certify(request)
        return replace(result, objective=result.objective + 1.0)

    monkeypatch.setattr(durable_module, "certify_exact", divergent_shadow)
    path = tmp_path / "shadow.sqlite3"
    policy = CertificationPolicy()
    service = DurableShadowCertificationService(
        path,
        certification_policy=policy,
    )
    submission = service.submit(_request(), submission_id="divergent-order")
    assert submission.ticket is not None

    deadline = time.monotonic() + 10.0
    while not service.is_ready(submission.ticket):
        if time.monotonic() >= deadline:
            raise TimeoutError("divergent shadow did not complete")
        time.sleep(0.01)
    with pytest.raises(ShadowRouteCircuitOpenError, match="circuit-open"):
        service.submit(_request(), submission_id="blocked-before-collect")
    result = service.collect(submission.ticket, timeout=10.0)
    assert not result.passed
    assert service.ack(submission.ticket)
    with pytest.raises(ShadowRouteCircuitOpenError, match="circuit-open"):
        service.submit(_request(), submission_id="blocked-order")
    service.shutdown()

    changed_policy = CertificationPolicy(
        mode=CertificationMode.SAMPLED,
        sample_numerator=1,
        sample_denominator=2,
    )
    restored = DurableShadowCertificationService(
        path,
        certification_policy=changed_policy,
    )
    with pytest.raises(ShadowRouteCircuitOpenError, match="circuit-open"):
        restored.submit(_request(), submission_id="still-blocked")
    restored.shutdown()


def test_fenced_late_divergence_does_not_open_route_circuit(
    tmp_path,
    monkeypatch,
) -> None:
    from openpo import durable_shadow as durable_module

    original_certify = durable_module.certify_exact

    def divergent_shadow(request):
        result = original_certify(request)
        return replace(result, objective=result.objective + 1.0)

    monkeypatch.setattr(durable_module, "certify_exact", divergent_shadow)
    service = DurableShadowCertificationService(
        tmp_path / "shadow.sqlite3",
        certification_policy=CertificationPolicy(),
    )
    monkeypatch.setattr(service._journal, "complete_success", lambda **kwargs: False)
    submission = service.submit(_request(), submission_id="fenced-late-divergence")
    assert submission.ticket is not None

    deadline = time.monotonic() + 10.0
    while True:
        with service._lock:
            if not service._active:
                break
        if time.monotonic() >= deadline:
            raise TimeoutError("fenced-late divergent worker did not finish")
        time.sleep(0.01)

    assert service._policy_store is not None
    assert service._policy_store.circuit_state("cpu") is None
    service.shutdown(require_collected=False)
