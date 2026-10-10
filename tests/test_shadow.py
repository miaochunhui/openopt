from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import replace
from threading import Event

import pytest

from openpo.facade import FallbackPolicy, SolveStatus
from openpo.shadow import (
    ShadowCertificationService,
    ShadowDecision,
    ShadowQueueFullError,
)
from tests.test_facade import _replace_request, _request


def test_shadow_service_returns_online_osqp_then_collects_clarabel() -> None:
    request = _request(exact_backend="osqp")
    service = ShadowCertificationService(max_pending=1)

    submission = service.submit(request)
    result = service.collect(submission.ticket, timeout=10.0)
    assert service.collect(submission.ticket, timeout=0.0) is result
    assert service.ack(submission.ticket) is True
    assert service.ack(submission.ticket) is False
    service.shutdown()

    assert submission.online.status is SolveStatus.SOLVED_TO_TOLERANCE
    assert submission.online.exact.backend == "osqp"
    assert submission.online.input_sha256 == submission.ticket.input_sha256
    assert submission.ticket.service_epoch == service.service_epoch
    assert len(submission.ticket.ticket_nonce) == 32
    assert submission.ticket.policy_sha256 == service.policy_sha256
    assert submission.ticket.attempt == 1
    assert result.ticket == submission.ticket
    assert result.shadow.backend == "clarabel"
    assert result.shadow.status is SolveStatus.OPTIMAL
    assert result.decision is ShadowDecision.MATCH
    assert result.passed is True
    assert result.primary_exact_result_sha256 == submission.online.exact.result_sha256
    assert result.shadow_exact_result_sha256 == result.shadow.result_sha256
    assert result.objective_relative_delta <= 1e-6
    assert result.weights_max_abs_diff <= 1e-5
    assert result.shadow_queue_wait_seconds >= 0.0
    assert result.certification_age_seconds >= result.shadow_elapsed_seconds
    assert len(result.result_sha256) == 64


def test_shadow_service_rejects_wrong_primary_contract() -> None:
    service = ShadowCertificationService()
    with pytest.raises(ValueError, match="exact_backend='osqp'"):
        service.submit(_request(exact_backend="clarabel"))
    with pytest.raises(ValueError, match="fallback_policy=DENY"):
        service.submit(
            _replace_request(
                _request(),
                fallback_policy=FallbackPolicy.ALLOW_CPU_EXACT,
            )
        )
    service.shutdown()


def test_shadow_queue_is_bounded_and_never_silently_drops(monkeypatch) -> None:
    entered = Event()
    release = Event()
    from openpo import shadow as shadow_module

    original = shadow_module.certify_exact

    def blocked(request):
        entered.set()
        assert release.wait(timeout=10.0)
        return original(request)

    monkeypatch.setattr(shadow_module, "certify_exact", blocked)
    service = ShadowCertificationService(max_pending=1)
    first = service.submit(_request())
    assert entered.wait(timeout=10.0)
    assert service.inflight_count == 1

    with pytest.raises(ShadowQueueFullError, match="queue is full"):
        service.submit(_request())

    release.set()
    result = service.collect(first.ticket, timeout=10.0)
    assert result.passed is True
    with pytest.raises(ShadowQueueFullError, match="queue is full"):
        service.submit(_request())
    assert service.ack(first.ticket) is True
    service.shutdown()


def test_shadow_timeout_keeps_ticket_collectable(monkeypatch) -> None:
    entered = Event()
    release = Event()
    from openpo import shadow as shadow_module

    original = shadow_module.certify_exact

    def blocked(request):
        entered.set()
        assert release.wait(timeout=10.0)
        return original(request)

    monkeypatch.setattr(shadow_module, "certify_exact", blocked)
    service = ShadowCertificationService()
    submission = service.submit(_request())
    assert entered.wait(timeout=10.0)
    with pytest.raises(TimeoutError, match="timed out"):
        service.collect(submission.ticket, timeout=0.0)
    assert service.uncollected_count == 1

    release.set()
    assert service.collect(submission.ticket, timeout=10.0).passed is True
    assert service.ack(submission.ticket) is True
    service.shutdown()


def test_shadow_collect_delivers_terminal_result_after_timeout_completion_race(
    monkeypatch,
) -> None:
    service = ShadowCertificationService()
    submission = service.submit(_request())
    record = service._records[submission.ticket.ticket_sha256]
    expected = record.future.result(timeout=10.0)
    original_result = record.future.result
    call_count = 0

    def timeout_then_result(*, timeout=None):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise FutureTimeoutError
        return original_result(timeout=timeout)

    monkeypatch.setattr(record.future, "result", timeout_then_result)
    actual = service.collect(submission.ticket, timeout=0.0)

    assert actual is expected
    assert service.uncollected_count == 0
    assert service.ack(submission.ticket) is True
    service.shutdown()


def test_shadow_background_failure_is_collected_fail_loud(monkeypatch) -> None:
    from openpo import shadow as shadow_module

    def fail(request):
        raise RuntimeError("shadow backend failed")

    monkeypatch.setattr(shadow_module, "certify_exact", fail)
    service = ShadowCertificationService()
    submission = service.submit(_request())

    with pytest.raises(RuntimeError, match="shadow backend failed"):
        service.collect(submission.ticket, timeout=10.0)
    assert service.uncollected_count == 0
    assert service.outstanding_count == 1
    assert service.ack(submission.ticket) is True
    service.shutdown()


def test_shadow_worker_timeout_error_is_terminal_and_acknowledgeable(monkeypatch) -> None:
    from openpo import shadow as shadow_module

    def fail(request):
        raise TimeoutError("shadow backend timed out internally")

    monkeypatch.setattr(shadow_module, "certify_exact", fail)
    service = ShadowCertificationService()
    submission = service.submit(_request())

    with pytest.raises(TimeoutError, match="timed out internally"):
        service.collect(submission.ticket, timeout=10.0)
    assert service.uncollected_count == 0
    assert service.ack(submission.ticket) is True
    service.shutdown()


def test_shadow_shutdown_rejects_uncollected_and_future_submissions() -> None:
    service = ShadowCertificationService()
    submission = service.submit(_request())
    with pytest.raises(RuntimeError, match="unacknowledged ticket"):
        service.shutdown()
    assert service.collect(submission.ticket, timeout=10.0).passed is True
    assert service.ack(submission.ticket) is True
    with pytest.raises(RuntimeError, match="closed"):
        service.submit(_request())


def test_shadow_rejects_forged_ticket_and_cross_service_collisions() -> None:
    request = _request()
    first_service = ShadowCertificationService(service_epoch="1" * 32)
    second_service = ShadowCertificationService(service_epoch="2" * 32)
    first = first_service.submit(request)
    second = second_service.submit(request)

    assert first.ticket.ticket_sha256 != second.ticket.ticket_sha256
    forged = replace(first.ticket, attempt=2)
    with pytest.raises(KeyError, match="forged"):
        first_service.collect(forged, timeout=0.0)
    with pytest.raises(KeyError, match="forged"):
        first_service.ack(forged)

    assert first_service.collect(first.ticket, timeout=10.0).passed is True
    assert second_service.collect(second.ticket, timeout=10.0).passed is True
    assert first_service.ack(first.ticket) is True
    assert second_service.ack(second.ticket) is True
    first_service.shutdown()
    second_service.shutdown()


def test_shadow_nonce_prevents_same_epoch_cross_service_ticket_collision() -> None:
    request = _request()
    first_service = ShadowCertificationService(service_epoch="4" * 32)
    second_service = ShadowCertificationService(service_epoch="4" * 32)
    first = first_service.submit(request)
    second = second_service.submit(request)

    assert first.ticket.ticket_nonce != second.ticket.ticket_nonce
    assert first.ticket.ticket_sha256 != second.ticket.ticket_sha256
    with pytest.raises(KeyError, match="unknown or forged"):
        second_service.collect(first.ticket, timeout=0.0)

    first_service.collect(first.ticket, timeout=10.0)
    second_service.collect(second.ticket, timeout=10.0)
    first_service.ack(first.ticket)
    second_service.ack(second.ticket)
    first_service.shutdown()
    second_service.shutdown()


def test_shadow_policy_hash_binds_tolerances_and_attempt() -> None:
    request = _request()
    strict = ShadowCertificationService(
        weights_tolerance=1e-6,
        service_epoch="3" * 32,
    )
    loose = ShadowCertificationService(
        weights_tolerance=1e-5,
        service_epoch="3" * 32,
    )
    strict_submission = strict.submit(request, attempt=1)
    loose_submission = loose.submit(request, attempt=2)

    assert strict.policy_sha256 != loose.policy_sha256
    assert strict_submission.ticket.policy_sha256 == strict.policy_sha256
    assert loose_submission.ticket.policy_sha256 == loose.policy_sha256
    assert strict_submission.ticket.ticket_sha256 != loose_submission.ticket.ticket_sha256
    assert loose_submission.ticket.attempt == 2

    strict.collect(strict_submission.ticket, timeout=10.0)
    loose.collect(loose_submission.ticket, timeout=10.0)
    strict.ack(strict_submission.ticket)
    loose.ack(loose_submission.ticket)
    strict.shutdown()
    loose.shutdown()


def test_shadow_collect_and_ack_are_concurrently_idempotent() -> None:
    service = ShadowCertificationService(max_pending=1)
    submission = service.submit(_request())

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(
            executor.map(
                lambda _: service.collect(submission.ticket, timeout=10.0),
                range(4),
            )
        )
    assert all(result is results[0] for result in results)
    assert service.outstanding_count == 1

    with ThreadPoolExecutor(max_workers=4) as executor:
        acknowledgements = list(
            executor.map(lambda _: service.ack(submission.ticket), range(4))
        )
    assert acknowledgements.count(True) == 1
    assert acknowledgements.count(False) == 3
    assert service.outstanding_count == 0
    service.shutdown()


def test_shadow_ack_rejects_interrupted_collect_before_completion(monkeypatch) -> None:
    from openpo import shadow as shadow_module

    entered = Event()
    release = Event()
    original = shadow_module.certify_exact

    def blocked(request):
        entered.set()
        assert release.wait(timeout=10.0)
        return original(request)

    monkeypatch.setattr(shadow_module, "certify_exact", blocked)
    service = ShadowCertificationService()
    submission = service.submit(_request())
    assert entered.wait(timeout=10.0)
    record = service._records[submission.ticket.ticket_sha256]
    original_result = record.future.result

    def interrupted(*, timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(record.future, "result", interrupted)
    with pytest.raises(KeyboardInterrupt):
        service.collect(submission.ticket, timeout=0.0)
    with pytest.raises(RuntimeError, match="collected before ack"):
        service.ack(submission.ticket)

    monkeypatch.setattr(record.future, "result", original_result)
    release.set()
    service.collect(submission.ticket, timeout=10.0)
    assert service.ack(submission.ticket) is True
    service.shutdown()


def test_shadow_ack_tombstones_are_bounded() -> None:
    service = ShadowCertificationService(max_pending=2)
    tickets = []
    for _ in range(5):
        submission = service.submit(_request())
        service.collect(submission.ticket, timeout=10.0)
        assert service.ack(submission.ticket) is True
        tickets.append(submission.ticket)

    assert len(service._acked) == 2
    assert service.ack(tickets[-1]) is False
    with pytest.raises(KeyError, match="unknown or forged"):
        service.ack(tickets[0])
    service.shutdown()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_pending": 0}, "positive integer"),
        ({"objective_tolerance": -1.0}, "finite and nonnegative"),
        ({"weights_tolerance": float("nan")}, "finite and nonnegative"),
        ({"service_epoch": "NOT-HEX"}, "32 lowercase hexadecimal"),
    ],
)
def test_shadow_service_rejects_invalid_configuration(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShadowCertificationService(**kwargs)
