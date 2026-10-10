from __future__ import annotations

import sqlite3

import pytest

from openpo.shadow_journal import ShadowJournalFullError, SQLiteShadowJournal


def _insert(
    journal: SQLiteShadowJournal,
    *,
    submission_id: str = "submission-1",
    ticket_sha256: str = "a" * 64,
):
    return journal.insert_online_ready(
        submission_id=submission_id,
        ticket_sha256=ticket_sha256,
        ticket_payload=b"ticket",
        request_payload=b"request",
        online_payload=b"online",
        submitted_at=10.0,
        online_completed_at=11.0,
    )


def test_sqlite_shadow_journal_enables_wal_and_checks_integrity(tmp_path) -> None:
    path = tmp_path / "shadow.sqlite3"
    journal = SQLiteShadowJournal(path, max_outstanding=2)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    journal.quick_check()


def test_sqlite_shadow_journal_capacity_counts_terminal_unacked(tmp_path) -> None:
    journal = SQLiteShadowJournal(tmp_path / "shadow.sqlite3", max_outstanding=1)
    first, inserted = _insert(journal)
    assert inserted is True

    claimed = journal.claim_next(owner="worker-a", lease_seconds=60.0, now=20.0)
    assert claimed is not None
    assert journal.complete_success(
        job_id=claimed.job_id,
        owner="worker-a",
        fence_token=claimed.fence_token,
        terminal_payload=b"result",
    )
    with pytest.raises(ShadowJournalFullError, match="queue is full"):
        _insert(journal, submission_id="submission-2", ticket_sha256="b" * 64)

    journal.mark_observed(first.ticket_sha256)
    assert journal.ack(first.ticket_sha256) is True
    second, inserted = _insert(
        journal,
        submission_id="submission-2",
        ticket_sha256="b" * 64,
    )
    assert inserted is True
    assert second.state == "ONLINE_READY"


def test_service_time_samples_record_only_accepted_success(tmp_path) -> None:
    journal = SQLiteShadowJournal(tmp_path / "shadow.sqlite3", max_outstanding=2)
    _insert(journal)
    claimed = journal.claim_next(owner="worker-a", lease_seconds=60.0, now=20.0)
    assert claimed is not None

    assert journal.complete_success(
        job_id=claimed.job_id,
        owner="worker-a",
        fence_token=claimed.fence_token,
        terminal_payload=b"result",
        service_time_seconds=8.5,
    )
    assert not journal.complete_success(
        job_id=claimed.job_id,
        owner="worker-a",
        fence_token=claimed.fence_token,
        terminal_payload=b"late",
        service_time_seconds=99.0,
    )

    samples = journal.service_time_samples(window_size=4)
    assert samples.total_sample_count == 1
    assert samples.elapsed_seconds == (8.5,)


def test_service_time_samples_are_isolated_by_route(tmp_path) -> None:
    journal = SQLiteShadowJournal(tmp_path / "shadow.sqlite3", max_outstanding=2)
    _insert(journal)
    _insert(journal, submission_id="submission-2", ticket_sha256="b" * 64)

    first = journal.claim_next(owner="worker-a", lease_seconds=60.0, now=20.0)
    assert first is not None
    assert journal.complete_success(
        job_id=first.job_id,
        owner="worker-a",
        fence_token=first.fence_token,
        terminal_payload=b"result-a",
        service_time_seconds=8.5,
        service_time_route_key="route-a",
    )
    second = journal.claim_next(owner="worker-a", lease_seconds=60.0, now=21.0)
    assert second is not None
    assert journal.complete_success(
        job_id=second.job_id,
        owner="worker-a",
        fence_token=second.fence_token,
        terminal_payload=b"result-b",
        service_time_seconds=11.0,
        service_time_route_key="route-b",
    )

    route_a = journal.service_time_samples(window_size=4, route_key="route-a")
    route_b = journal.service_time_samples(window_size=4, route_key="route-b")
    legacy = journal.service_time_samples(window_size=4)
    assert route_a.elapsed_seconds == (8.5,)
    assert route_a.route_key == "route-a"
    assert route_b.elapsed_seconds == (11.0,)
    assert route_b.route_key == "route-b"
    assert legacy.total_sample_count == 0


def test_sqlite_shadow_journal_upgrades_v3_service_samples_to_v4(tmp_path) -> None:
    path = tmp_path / "shadow.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE service_time_samples(
                sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id INTEGER NOT NULL UNIQUE,
                elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds > 0.0),
                created_at REAL NOT NULL
            );
            PRAGMA user_version=3;
            """
        )

    SQLiteShadowJournal(path, max_outstanding=1)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
        columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(service_time_samples)"
            ).fetchall()
        }
    assert "route_key" in columns


@pytest.mark.parametrize("service_time_seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_service_time_samples_reject_invalid_success_measurements(
    tmp_path, service_time_seconds
) -> None:
    journal = SQLiteShadowJournal(tmp_path / "shadow.sqlite3", max_outstanding=1)
    _insert(journal)
    claimed = journal.claim_next(owner="worker-a", lease_seconds=60.0, now=20.0)
    assert claimed is not None

    with pytest.raises(ValueError, match="finite and positive"):
        journal.complete_success(
            job_id=claimed.job_id,
            owner="worker-a",
            fence_token=claimed.fence_token,
            terminal_payload=b"result",
            service_time_seconds=service_time_seconds,
        )

    assert journal.get_by_ticket("a" * 64).state == "LEASED"
    assert journal.service_time_samples(window_size=4).total_sample_count == 0


def test_service_time_samples_fail_loud_on_persisted_tampering(tmp_path) -> None:
    path = tmp_path / "shadow.sqlite3"
    journal = SQLiteShadowJournal(path, max_outstanding=1)
    _insert(journal)
    claimed = journal.claim_next(owner="worker-a", lease_seconds=60.0, now=20.0)
    assert claimed is not None
    assert journal.complete_success(
        job_id=claimed.job_id,
        owner="worker-a",
        fence_token=claimed.fence_token,
        terminal_payload=b"result",
        service_time_seconds=8.5,
    )
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA ignore_check_constraints=ON")
        connection.execute(
            "UPDATE service_time_samples SET elapsed_seconds = ?",
            (float("inf"),),
        )

    with pytest.raises(RuntimeError, match="persisted service-time sample is invalid"):
        journal.service_time_samples(window_size=4)


def test_sqlite_shadow_journal_capacity_snapshot_recomputes_ages(tmp_path) -> None:
    journal = SQLiteShadowJournal(tmp_path / "shadow.sqlite3", max_outstanding=3)
    first, _ = _insert(journal)

    ready = journal.capacity_snapshot(now=21.0)
    assert ready.max_outstanding == 3
    assert ready.outstanding == 1
    assert ready.available == 2
    assert ready.utilization == pytest.approx(1.0 / 3.0)
    assert ready.online_ready == 1
    assert ready.leased == 0
    assert ready.terminal_unacked == 0
    assert ready.oldest_outstanding_age_seconds == 10.0
    assert ready.oldest_inflight_age_seconds == 10.0
    assert ready.oldest_ready_age_seconds == 10.0

    claimed = journal.claim_next(owner="worker-a", lease_seconds=60.0, now=22.0)
    assert claimed is not None
    leased = journal.capacity_snapshot(now=23.0)
    assert leased.online_ready == 0
    assert leased.leased == 1
    assert leased.oldest_ready_age_seconds is None
    assert journal.complete_success(
        job_id=claimed.job_id,
        owner="worker-a",
        fence_token=claimed.fence_token,
        terminal_payload=b"result",
    )
    terminal = journal.capacity_snapshot(now=24.0)
    assert terminal.terminal_unacked == 1
    assert terminal.oldest_inflight_age_seconds is None
    journal.mark_observed(first.ticket_sha256)
    journal.ack(first.ticket_sha256)
    empty = journal.capacity_snapshot(now=25.0)
    assert empty.outstanding == 0
    assert empty.available == 3
    assert empty.oldest_outstanding_age_seconds is None


def test_sqlite_shadow_journal_submission_and_ack_are_idempotent(tmp_path) -> None:
    journal = SQLiteShadowJournal(tmp_path / "shadow.sqlite3", max_outstanding=1)
    first, inserted = _insert(journal)
    duplicate, inserted_again = _insert(journal)

    assert inserted is True
    assert inserted_again is False
    assert duplicate.job_id == first.job_id
    claimed = journal.claim_next(owner="worker-a", lease_seconds=60.0, now=20.0)
    assert claimed is not None
    assert journal.complete_failure(
        job_id=claimed.job_id,
        owner="worker-a",
        fence_token=claimed.fence_token,
        error_type="RuntimeError",
        error_message="failed",
        error_traceback="trace",
    )
    with pytest.raises(RuntimeError, match="terminal and observed"):
        journal.ack(first.ticket_sha256)
    journal.mark_observed(first.ticket_sha256)
    assert journal.ack(first.ticket_sha256) is True
    assert journal.ack(first.ticket_sha256) is False


def test_sqlite_shadow_journal_reclaims_expired_lease_and_fences_late_result(
    tmp_path,
) -> None:
    journal = SQLiteShadowJournal(tmp_path / "shadow.sqlite3", max_outstanding=1)
    first, _ = _insert(journal)
    old = journal.claim_next(owner="worker-old", lease_seconds=5.0, now=20.0)
    assert old is not None
    assert journal.heartbeat(
        job_id=old.job_id,
        owner="worker-old",
        fence_token=old.fence_token,
        lease_seconds=5.0,
    )

    replacement = journal.claim_next(owner="worker-new", lease_seconds=5.0, now=1e20)
    assert replacement is not None
    assert replacement.execution_attempt == old.execution_attempt + 1
    assert replacement.fence_token == old.fence_token + 1
    assert journal.complete_success(
        job_id=replacement.job_id,
        owner="worker-new",
        fence_token=replacement.fence_token,
        terminal_payload=b"new-result",
    )
    assert not journal.complete_success(
        job_id=old.job_id,
        owner="worker-old",
        fence_token=old.fence_token,
        terminal_payload=b"late-result",
    )

    stored = journal.get_by_ticket(first.ticket_sha256)
    assert stored is not None
    assert stored.terminal_payload == b"new-result"
    assert [event["event_type"] for event in journal.events(first.job_id)][-1] == "FENCED_LATE"


def test_sqlite_shadow_journal_rejects_missing_parent(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="parent directory"):
        SQLiteShadowJournal(tmp_path / "missing" / "shadow.sqlite3", max_outstanding=1)
