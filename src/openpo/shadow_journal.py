from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sqlite3
import time
from typing import Callable


class ShadowJournalFullError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShadowCapacitySnapshot:
    max_outstanding: int
    outstanding: int
    online_ready: int
    leased: int
    terminal_unacked: int
    oldest_outstanding_age_seconds: float | None
    oldest_inflight_age_seconds: float | None
    oldest_ready_age_seconds: float | None

    @property
    def available(self) -> int:
        return max(0, self.max_outstanding - self.outstanding)

    @property
    def utilization(self) -> float:
        return self.outstanding / self.max_outstanding


@dataclass(frozen=True)
class DurableShadowJob:
    job_id: int
    submission_id: str
    ticket_sha256: str
    ticket_payload: bytes
    request_payload: bytes
    online_payload: bytes
    state: str
    observed: bool
    execution_attempt: int
    fence_token: int
    lease_owner: str | None
    lease_deadline: float | None
    terminal_payload: bytes | None
    error_type: str | None
    error_message: str | None
    error_traceback: str | None
    submitted_at: float
    online_completed_at: float
    created_at: float
    updated_at: float


@dataclass(frozen=True)
class ShadowServiceTimeSamples:
    total_sample_count: int
    elapsed_seconds: tuple[float, ...]
    route_key: str | None = None


class SQLiteShadowJournal:
    def __init__(self, path: str | Path, *, max_outstanding: int) -> None:
        if not isinstance(max_outstanding, int) or isinstance(max_outstanding, bool) or max_outstanding <= 0:
            raise ValueError("max_outstanding must be a positive integer")
        self.path = Path(path).expanduser().resolve()
        if not self.path.parent.is_dir():
            raise FileNotFoundError(f"shadow journal parent directory does not exist: {self.path.parent}")
        self.max_outstanding = max_outstanding
        self._initialize()

    def allocate_sequence(self, service_epoch: str) -> int:
        key = f"sequence:{service_epoch}"
        with self._transaction() as connection:
            row = connection.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
            sequence = 0 if row is None else int(row["value"])
            connection.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, str(sequence + 1)),
            )
            return sequence

    def insert_online_ready(
        self,
        *,
        submission_id: str,
        ticket_sha256: str,
        ticket_payload: bytes,
        request_payload: bytes,
        online_payload: bytes,
        submitted_at: float,
        online_completed_at: float,
    ) -> tuple[DurableShadowJob, bool]:
        now = time.time()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM jobs WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            if existing is not None:
                return _job(existing), False
            outstanding = int(
                connection.execute(
                    "SELECT COUNT(*) AS count FROM jobs WHERE state != 'ACKED'"
                ).fetchone()["count"]
            )
            if outstanding >= self.max_outstanding:
                raise ShadowJournalFullError("durable Clarabel shadow queue is full")
            cursor = connection.execute(
                """
                INSERT INTO jobs(
                    submission_id, ticket_sha256, ticket_payload, request_payload,
                    online_payload, state, observed, execution_attempt, fence_token,
                    submitted_at, online_completed_at, created_at, updated_at
                ) VALUES(?, ?, ?, ?, ?, 'ONLINE_READY', 0, 0, 0, ?, ?, ?, ?)
                """,
                (
                    submission_id,
                    ticket_sha256,
                    ticket_payload,
                    request_payload,
                    online_payload,
                    submitted_at,
                    online_completed_at,
                    now,
                    now,
                ),
            )
            job_id = int(cursor.lastrowid)
            self._event(connection, job_id, "ONLINE_READY", "accepted durable shadow job")
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return _job(row), True

    def get_by_submission_id(self, submission_id: str) -> DurableShadowJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            return None if row is None else _job(row)

    def get_by_ticket(self, ticket_sha256: str) -> DurableShadowJob | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE ticket_sha256 = ?",
                (ticket_sha256,),
            ).fetchone()
            return None if row is None else _job(row)

    def outstanding_jobs(self) -> list[DurableShadowJob]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE state != 'ACKED' ORDER BY job_id"
            ).fetchall()
            return [_job(row) for row in rows]

    def claim_next(
        self,
        *,
        owner: str,
        lease_seconds: float,
        now: float | None = None,
    ) -> DurableShadowJob | None:
        claimed_at = time.time() if now is None else now
        lease_deadline = claimed_at + lease_seconds
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM jobs
                WHERE state = 'ONLINE_READY'
                   OR (state = 'LEASED' AND lease_deadline <= ?)
                ORDER BY job_id
                LIMIT 1
                """,
                (claimed_at,),
            ).fetchone()
            if row is None:
                return None
            job_id = int(row["job_id"])
            execution_attempt = int(row["execution_attempt"]) + 1
            fence_token = int(row["fence_token"]) + 1
            connection.execute(
                """
                UPDATE jobs
                SET state = 'LEASED', execution_attempt = ?, fence_token = ?,
                    lease_owner = ?, lease_deadline = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    execution_attempt,
                    fence_token,
                    owner,
                    lease_deadline,
                    claimed_at,
                    job_id,
                ),
            )
            self._event(
                connection,
                job_id,
                "LEASED",
                f"owner={owner} attempt={execution_attempt} fence={fence_token}",
            )
            claimed = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return _job(claimed)

    def heartbeat(
        self,
        *,
        job_id: int,
        owner: str,
        fence_token: int,
        lease_seconds: float,
    ) -> bool:
        now = time.time()
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET lease_deadline = ?, updated_at = ?
                WHERE job_id = ? AND state = 'LEASED'
                  AND lease_owner = ? AND fence_token = ?
                """,
                (now + lease_seconds, now, job_id, owner, fence_token),
            )
            return cursor.rowcount == 1

    def complete_success(
        self,
        *,
        job_id: int,
        owner: str,
        fence_token: int,
        terminal_payload: bytes,
        service_time_seconds: float | None = None,
        service_time_route_key: str | None = None,
    ) -> bool:
        if service_time_seconds is not None and (
            not math.isfinite(service_time_seconds) or service_time_seconds <= 0.0
        ):
            raise ValueError("service_time_seconds must be finite and positive or null")
        if service_time_route_key is not None and (
            not isinstance(service_time_route_key, str)
            or not service_time_route_key
            or len(service_time_route_key) > 256
        ):
            raise ValueError(
                "service_time_route_key must be a nonempty string of at most 256 characters or null"
            )
        return self._complete(
            job_id=job_id,
            owner=owner,
            fence_token=fence_token,
            state="SUCCEEDED",
            terminal_payload=terminal_payload,
            error_type=None,
            error_message=None,
            error_traceback=None,
            service_time_seconds=service_time_seconds,
            service_time_route_key=service_time_route_key,
        )

    def complete_failure(
        self,
        *,
        job_id: int,
        owner: str,
        fence_token: int,
        error_type: str,
        error_message: str,
        error_traceback: str,
    ) -> bool:
        return self._complete(
            job_id=job_id,
            owner=owner,
            fence_token=fence_token,
            state="FAILED",
            terminal_payload=None,
            error_type=error_type,
            error_message=error_message,
            error_traceback=error_traceback,
            service_time_seconds=None,
            service_time_route_key=None,
        )

    def service_time_samples(
        self,
        *,
        window_size: int,
        route_key: str | None = None,
    ) -> ShadowServiceTimeSamples:
        if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size <= 0:
            raise ValueError("window_size must be a positive integer")
        if route_key is not None and (
            not isinstance(route_key, str) or not route_key or len(route_key) > 256
        ):
            raise ValueError("route_key must be a nonempty string of at most 256 characters or null")
        predicate = "route_key IS NULL" if route_key is None else "route_key = ?"
        parameters: tuple[object, ...] = () if route_key is None else (route_key,)
        with self._connect() as connection:
            connection.execute("BEGIN")
            total = int(
                connection.execute(
                    f"SELECT COUNT(*) FROM service_time_samples WHERE {predicate}",
                    parameters,
                ).fetchone()[0]
            )
            rows = connection.execute(
                f"SELECT route_key, elapsed_seconds FROM service_time_samples "
                f"WHERE {predicate} ORDER BY sample_id DESC LIMIT ?",
                (*parameters, window_size),
            ).fetchall()
        values = tuple(float(row["elapsed_seconds"]) for row in reversed(rows))
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            raise RuntimeError("persisted service-time sample is invalid")
        if any(row["route_key"] != route_key for row in rows):
            raise RuntimeError("persisted service-time sample route is inconsistent")
        return ShadowServiceTimeSamples(
            total_sample_count=total,
            elapsed_seconds=values,
            route_key=route_key,
        )

    def mark_observed(self, ticket_sha256: str) -> DurableShadowJob:
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE ticket_sha256 = ?",
                (ticket_sha256,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown durable shadow ticket {ticket_sha256}")
            if row["state"] not in {"SUCCEEDED", "FAILED"}:
                raise RuntimeError("durable shadow ticket is not terminal")
            connection.execute(
                "UPDATE jobs SET observed = 1, updated_at = ? WHERE job_id = ?",
                (now, int(row["job_id"])),
            )
            self._event(connection, int(row["job_id"]), "OBSERVED", "terminal delivered")
            updated = connection.execute(
                "SELECT * FROM jobs WHERE job_id = ?",
                (int(row["job_id"]),),
            ).fetchone()
            return _job(updated)

    def ack(self, ticket_sha256: str) -> bool:
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE ticket_sha256 = ?",
                (ticket_sha256,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown durable shadow ticket {ticket_sha256}")
            if row["state"] == "ACKED":
                return False
            if row["state"] not in {"SUCCEEDED", "FAILED"} or not bool(row["observed"]):
                raise RuntimeError("durable shadow ticket must be terminal and observed before ack")
            connection.execute(
                """
                UPDATE jobs
                SET state = 'ACKED', acked_at = ?, lease_owner = NULL,
                    lease_deadline = NULL, updated_at = ?
                WHERE job_id = ?
                """,
                (now, now, int(row["job_id"])),
            )
            self._event(connection, int(row["job_id"]), "ACKED", "capacity released")
            return True

    @property
    def outstanding_count(self) -> int:
        return self._count("state != 'ACKED'")

    @property
    def unobserved_count(self) -> int:
        return self._count("state != 'ACKED' AND observed = 0")

    @property
    def inflight_count(self) -> int:
        return self._count("state IN ('ONLINE_READY', 'LEASED')")

    def capacity_snapshot(self, *, now: float | None = None) -> ShadowCapacitySnapshot:
        observed_at = time.time() if now is None else now
        if not isinstance(observed_at, (int, float)) or not math.isfinite(observed_at):
            raise ValueError("capacity snapshot time must be finite")
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT
                    SUM(CASE WHEN state != 'ACKED' THEN 1 ELSE 0 END) AS outstanding,
                    SUM(CASE WHEN state = 'ONLINE_READY' THEN 1 ELSE 0 END) AS online_ready,
                    SUM(CASE WHEN state = 'LEASED' THEN 1 ELSE 0 END) AS leased,
                    SUM(CASE WHEN state IN ('SUCCEEDED', 'FAILED') THEN 1 ELSE 0 END)
                        AS terminal_unacked,
                    MIN(CASE WHEN state != 'ACKED' THEN online_completed_at END)
                        AS oldest_outstanding_at,
                    MIN(CASE WHEN state IN ('ONLINE_READY', 'LEASED')
                        THEN online_completed_at END) AS oldest_inflight_at,
                    MIN(CASE WHEN state = 'ONLINE_READY' THEN online_completed_at END)
                        AS oldest_ready_at
                FROM jobs
                """
            ).fetchone()

        def age(field: str) -> float | None:
            value = row[field]
            return None if value is None else max(0.0, observed_at - float(value))

        return ShadowCapacitySnapshot(
            max_outstanding=self.max_outstanding,
            outstanding=int(row["outstanding"] or 0),
            online_ready=int(row["online_ready"] or 0),
            leased=int(row["leased"] or 0),
            terminal_unacked=int(row["terminal_unacked"] or 0),
            oldest_outstanding_age_seconds=age("oldest_outstanding_at"),
            oldest_inflight_age_seconds=age("oldest_inflight_at"),
            oldest_ready_age_seconds=age("oldest_ready_at"),
        )

    def events(self, job_id: int) -> list[sqlite3.Row]:
        with self._connect() as connection:
            return list(
                connection.execute(
                    "SELECT * FROM journal_events WHERE job_id = ? ORDER BY event_id",
                    (job_id,),
                ).fetchall()
            )

    def quick_check(self) -> None:
        with self._connect() as connection:
            result = connection.execute("PRAGMA quick_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError(f"shadow journal quick_check failed: {result}")

    def _complete(
        self,
        *,
        job_id: int,
        owner: str,
        fence_token: int,
        state: str,
        terminal_payload: bytes | None,
        error_type: str | None,
        error_message: str | None,
        error_traceback: str | None,
        service_time_seconds: float | None,
        service_time_route_key: str | None,
    ) -> bool:
        if service_time_seconds is not None and (
            not math.isfinite(service_time_seconds) or service_time_seconds <= 0.0
        ):
            raise ValueError("service_time_seconds must be finite and positive")
        now = time.time()
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE jobs
                SET state = ?, terminal_payload = ?, error_type = ?,
                    error_message = ?, error_traceback = ?, lease_owner = NULL,
                    lease_deadline = NULL, updated_at = ?
                WHERE job_id = ? AND state = 'LEASED'
                  AND lease_owner = ? AND fence_token = ?
                """,
                (
                    state,
                    terminal_payload,
                    error_type,
                    error_message,
                    error_traceback,
                    now,
                    job_id,
                    owner,
                    fence_token,
                ),
            )
            if cursor.rowcount == 1:
                if service_time_seconds is not None:
                    connection.execute(
                        "INSERT INTO service_time_samples("
                        "job_id, route_key, elapsed_seconds, created_at"
                        ") VALUES(?, ?, ?, ?)",
                        (job_id, service_time_route_key, service_time_seconds, now),
                    )
                self._event(connection, job_id, state, f"fence={fence_token}")
                return True
            self._event(
                connection,
                job_id,
                "FENCED_LATE",
                f"owner={owner} fence={fence_token} terminal={state}",
            )
            return False

    def _count(self, where: str) -> int:
        with self._connect() as connection:
            return int(connection.execute(f"SELECT COUNT(*) FROM jobs WHERE {where}").fetchone()[0])

    def _initialize(self) -> None:
        with self._connect() as connection:
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError(f"shadow journal failed to enable WAL mode: {mode}")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in {0, 1, 2, 3, 4}:
                raise RuntimeError(f"unsupported shadow journal schema version {version}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS jobs(
                    job_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT NOT NULL UNIQUE,
                    ticket_sha256 TEXT NOT NULL UNIQUE,
                    ticket_payload BLOB NOT NULL,
                    request_payload BLOB NOT NULL,
                    online_payload BLOB NOT NULL,
                    state TEXT NOT NULL CHECK(state IN (
                        'ONLINE_READY', 'LEASED', 'SUCCEEDED', 'FAILED', 'ACKED'
                    )),
                    observed INTEGER NOT NULL CHECK(observed IN (0, 1)),
                    execution_attempt INTEGER NOT NULL,
                    fence_token INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_deadline REAL,
                    terminal_payload BLOB,
                    error_type TEXT,
                    error_message TEXT,
                    error_traceback TEXT,
                    submitted_at REAL NOT NULL,
                    online_completed_at REAL NOT NULL,
                    acked_at REAL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS journal_events(
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL REFERENCES jobs(job_id),
                    event_type TEXT NOT NULL,
                    detail TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS jobs_state_lease_idx
                    ON jobs(state, lease_deadline, job_id);
                CREATE INDEX IF NOT EXISTS journal_events_job_idx
                    ON journal_events(job_id, event_id);
                CREATE TABLE IF NOT EXISTS service_time_samples(
                    sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id INTEGER NOT NULL UNIQUE REFERENCES jobs(job_id),
                    route_key TEXT,
                    elapsed_seconds REAL NOT NULL CHECK(elapsed_seconds > 0.0),
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS service_time_samples_created_idx
                    ON service_time_samples(created_at, sample_id);
                """
            )
            sample_columns = {
                str(row[1])
                for row in connection.execute(
                    "PRAGMA table_info(service_time_samples)"
                ).fetchall()
            }
            if "route_key" not in sample_columns:
                connection.execute(
                    "ALTER TABLE service_time_samples ADD COLUMN route_key TEXT"
                )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS service_time_samples_route_created_idx "
                "ON service_time_samples(route_key, created_at, sample_id)"
            )
            connection.execute("PRAGMA user_version=4")
        self.quick_check()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    def _transaction(self):
        return _Transaction(self._connect)

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        job_id: int,
        event_type: str,
        detail: str,
    ) -> None:
        connection.execute(
            "INSERT INTO journal_events(job_id, event_type, detail, created_at) VALUES(?, ?, ?, ?)",
            (job_id, event_type, detail, time.time()),
        )


class _Transaction:
    def __init__(self, connection_factory: Callable[[], sqlite3.Connection]) -> None:
        self._connection_factory = connection_factory
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self._connection = self._connection_factory()
        self._connection.execute("BEGIN IMMEDIATE")
        return self._connection

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._connection is not None
        try:
            if exc_type is None:
                self._connection.execute("COMMIT")
            else:
                self._connection.execute("ROLLBACK")
        finally:
            self._connection.close()


def _job(row: sqlite3.Row) -> DurableShadowJob:
    return DurableShadowJob(
        job_id=int(row["job_id"]),
        submission_id=str(row["submission_id"]),
        ticket_sha256=str(row["ticket_sha256"]),
        ticket_payload=bytes(row["ticket_payload"]),
        request_payload=bytes(row["request_payload"]),
        online_payload=bytes(row["online_payload"]),
        state=str(row["state"]),
        observed=bool(row["observed"]),
        execution_attempt=int(row["execution_attempt"]),
        fence_token=int(row["fence_token"]),
        lease_owner=None if row["lease_owner"] is None else str(row["lease_owner"]),
        lease_deadline=None if row["lease_deadline"] is None else float(row["lease_deadline"]),
        terminal_payload=None if row["terminal_payload"] is None else bytes(row["terminal_payload"]),
        error_type=None if row["error_type"] is None else str(row["error_type"]),
        error_message=None if row["error_message"] is None else str(row["error_message"]),
        error_traceback=None if row["error_traceback"] is None else str(row["error_traceback"]),
        submitted_at=float(row["submitted_at"]),
        online_completed_at=float(row["online_completed_at"]),
        created_at=float(row["created_at"]),
        updated_at=float(row["updated_at"]),
    )
