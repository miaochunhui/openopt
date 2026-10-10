from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3

from .facade import ProductionResult, SolveStatus


_SQLITE_USER_VERSION = 4
_SUPPORTED_SQLITE_USER_VERSIONS = frozenset(range(_SQLITE_USER_VERSION + 1))


class CertificationMode(str, Enum):
    FULL = "full"
    SAMPLED = "sampled"
    ANOMALY = "anomaly"


class CertificationDisposition(str, Enum):
    QUEUED = "queued"
    SKIPPED = "skipped"
    CIRCUIT_OPEN = "circuit_open"


class CertificationReason(str, Enum):
    FULL = "full"
    HASH_SAMPLE = "hash_sample"
    HASH_NOT_SELECTED = "hash_not_selected"
    OBJECTIVE_GAP = "objective_gap"
    PRIMAL_RESIDUAL = "primal_residual"
    FALLBACK = "fallback"
    STATUS = "status"
    MISSING_QUALITY = "missing_quality"
    CIRCUIT_OPEN = "circuit_open"
    NO_ANOMALY = "no_anomaly"


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA256 digest")
    return value


@dataclass(frozen=True)
class CertificationObservation:
    result_sha256: str
    status: str
    objective_gap: float | None
    primal_residual: float
    fallback_used: bool

    def __post_init__(self) -> None:
        _require_sha256(self.result_sha256, "result_sha256")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a nonempty string")
        if self.objective_gap is not None and (
            not math.isfinite(self.objective_gap) or self.objective_gap < 0.0
        ):
            raise ValueError("objective_gap must be finite and nonnegative or null")
        if not math.isfinite(self.primal_residual) or self.primal_residual < 0.0:
            raise ValueError("primal_residual must be finite and nonnegative")
        if not isinstance(self.fallback_used, bool):
            raise TypeError("fallback_used must be boolean")

    @classmethod
    def from_production_result(
        cls,
        result: ProductionResult,
    ) -> CertificationObservation:
        return cls(
            result_sha256=result.result_sha256,
            status=result.status.value,
            objective_gap=result.objective_gap,
            primal_residual=max(
                result.fast.residuals.primal_residual,
                result.exact.residuals.primal_residual,
            ),
            fallback_used=result.fallback_used,
        )


@dataclass(frozen=True)
class CertificationPolicy:
    mode: CertificationMode = CertificationMode.FULL
    sample_numerator: int = 1
    sample_denominator: int = 1
    anomaly_objective_gap: float = 0.01
    anomaly_primal_residual: float = 1e-5
    policy_version: str = "async-certification-v1"

    def __post_init__(self) -> None:
        if not isinstance(self.mode, CertificationMode):
            raise TypeError("mode must be a CertificationMode")
        if not isinstance(self.sample_numerator, int) or isinstance(
            self.sample_numerator, bool
        ):
            raise TypeError("sample_numerator must be an integer")
        if not isinstance(self.sample_denominator, int) or isinstance(
            self.sample_denominator, bool
        ):
            raise TypeError("sample_denominator must be an integer")
        if self.sample_denominator <= 0:
            raise ValueError("sample_denominator must be positive")
        if not 0 < self.sample_numerator <= self.sample_denominator:
            raise ValueError("sample_numerator must be in [1, sample_denominator]")
        if self.mode is CertificationMode.FULL and (
            self.sample_numerator != 1 or self.sample_denominator != 1
        ):
            raise ValueError("full mode must use a 1/1 sample ratio")
        for field, value in (
            ("anomaly_objective_gap", self.anomaly_objective_gap),
            ("anomaly_primal_residual", self.anomaly_primal_residual),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{field} must be finite and positive")
        if not isinstance(self.policy_version, str) or not self.policy_version:
            raise ValueError("policy_version must be a nonempty string")

    @property
    def policy_sha256(self) -> str:
        return _canonical_sha256(
            {
                "policy_version": self.policy_version,
                "mode": self.mode.value,
                "sample_numerator": self.sample_numerator,
                "sample_denominator": self.sample_denominator,
                "anomaly_objective_gap": float(self.anomaly_objective_gap).hex(),
                "anomaly_primal_residual": float(
                    self.anomaly_primal_residual
                ).hex(),
            }
        )

    def sample_bucket(self, request_sha256: str) -> int:
        request_digest = _require_sha256(request_sha256, "request_sha256")
        digest = hashlib.sha256(
            f"{self.policy_sha256}:{request_digest}".encode("ascii")
        ).digest()
        return int.from_bytes(digest[:8], "big") % self.sample_denominator

    def evaluate(
        self,
        request_sha256: str,
        observation: CertificationObservation,
        *,
        circuit_open: bool = False,
    ) -> CertificationPolicyDecision:
        request_digest = _require_sha256(request_sha256, "request_sha256")
        if not isinstance(circuit_open, bool):
            raise TypeError("circuit_open must be boolean")
        bucket = self.sample_bucket(request_digest)
        sampled = bucket < self.sample_numerator
        reasons: list[CertificationReason] = []
        if circuit_open:
            disposition = CertificationDisposition.CIRCUIT_OPEN
            reasons.append(CertificationReason.CIRCUIT_OPEN)
        elif self.mode is CertificationMode.FULL:
            disposition = CertificationDisposition.QUEUED
            reasons.append(CertificationReason.FULL)
        elif self.mode is CertificationMode.SAMPLED:
            disposition = (
                CertificationDisposition.QUEUED
                if sampled
                else CertificationDisposition.SKIPPED
            )
            if sampled:
                reasons.append(CertificationReason.HASH_SAMPLE)
            else:
                reasons.append(CertificationReason.HASH_NOT_SELECTED)
        else:
            if sampled:
                reasons.append(CertificationReason.HASH_SAMPLE)
            if observation.status != SolveStatus.SOLVED_TO_TOLERANCE.value:
                reasons.append(CertificationReason.STATUS)
            if observation.fallback_used:
                reasons.append(CertificationReason.FALLBACK)
            if observation.objective_gap is None:
                reasons.append(CertificationReason.MISSING_QUALITY)
            elif observation.objective_gap > self.anomaly_objective_gap:
                reasons.append(CertificationReason.OBJECTIVE_GAP)
            if observation.primal_residual > self.anomaly_primal_residual:
                reasons.append(CertificationReason.PRIMAL_RESIDUAL)
            disposition = (
                CertificationDisposition.QUEUED
                if reasons
                else CertificationDisposition.SKIPPED
            )
            if disposition is CertificationDisposition.SKIPPED:
                reasons.append(CertificationReason.NO_ANOMALY)
        evidence = {
            "policy_sha256": self.policy_sha256,
            "request_sha256": request_digest,
            "result_sha256": observation.result_sha256,
            "status": observation.status,
            "objective_gap": (
                None
                if observation.objective_gap is None
                else float(observation.objective_gap).hex()
            ),
            "primal_residual": float(observation.primal_residual).hex(),
            "fallback_used": observation.fallback_used,
            "sample_bucket": bucket,
            "disposition": disposition.value,
            "reasons": [reason.value for reason in reasons],
        }
        return CertificationPolicyDecision(
            policy_sha256=self.policy_sha256,
            request_sha256=request_digest,
            result_sha256=observation.result_sha256,
            disposition=disposition,
            reasons=tuple(reasons),
            sample_bucket=bucket,
            sample_numerator=self.sample_numerator,
            sample_denominator=self.sample_denominator,
            evidence_sha256=_canonical_sha256(evidence),
        )


@dataclass(frozen=True)
class CertificationPolicyDecision:
    policy_sha256: str
    request_sha256: str
    result_sha256: str
    disposition: CertificationDisposition
    reasons: tuple[CertificationReason, ...]
    sample_bucket: int
    sample_numerator: int
    sample_denominator: int
    evidence_sha256: str


def certification_decision_to_payload(
    decision: CertificationPolicyDecision,
) -> dict[str, object]:
    return {
        "schema": "certification-decision-v1",
        "policy_sha256": decision.policy_sha256,
        "request_sha256": decision.request_sha256,
        "result_sha256": decision.result_sha256,
        "disposition": decision.disposition.value,
        "reasons": [reason.value for reason in decision.reasons],
        "sample_bucket": decision.sample_bucket,
        "sample_numerator": decision.sample_numerator,
        "sample_denominator": decision.sample_denominator,
        "evidence_sha256": decision.evidence_sha256,
    }


def certification_decision_from_payload(
    payload: dict[str, object],
) -> CertificationPolicyDecision:
    expected = {
        "schema", "policy_sha256", "request_sha256", "result_sha256",
        "disposition", "reasons", "sample_bucket", "sample_numerator",
        "sample_denominator", "evidence_sha256",
    }
    if set(payload) != expected or payload.get("schema") != "certification-decision-v1":
        raise ValueError("unsupported certification decision payload")
    reasons = payload.get("reasons")
    if not isinstance(reasons, list) or not all(isinstance(item, str) for item in reasons):
        raise ValueError("certification decision reasons must be a string list")
    integer_fields = ("sample_bucket", "sample_numerator", "sample_denominator")
    integers: dict[str, int] = {}
    for field in integer_fields:
        value = payload.get(field)
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(f"certification decision {field} must be an integer")
        integers[field] = value
    decision = CertificationPolicyDecision(
        policy_sha256=_require_sha256(
            str(payload.get("policy_sha256")), "policy_sha256"
        ),
        request_sha256=_require_sha256(
            str(payload.get("request_sha256")), "request_sha256"
        ),
        result_sha256=_require_sha256(
            str(payload.get("result_sha256")), "result_sha256"
        ),
        disposition=CertificationDisposition(str(payload.get("disposition"))),
        reasons=tuple(CertificationReason(item) for item in reasons),
        sample_bucket=integers["sample_bucket"],
        sample_numerator=integers["sample_numerator"],
        sample_denominator=integers["sample_denominator"],
        evidence_sha256=_require_sha256(
            str(payload.get("evidence_sha256")), "evidence_sha256"
        ),
    )
    if (
        decision.sample_denominator <= 0
        or not 0 < decision.sample_numerator <= decision.sample_denominator
        or not 0 <= decision.sample_bucket < decision.sample_denominator
        or not decision.reasons
    ):
        raise ValueError("certification decision sampling evidence is invalid")
    return decision


def validate_certification_decision(
    decision: CertificationPolicyDecision,
    observation: CertificationObservation,
) -> None:
    expected_bucket = int.from_bytes(
        hashlib.sha256(
            f"{decision.policy_sha256}:{decision.request_sha256}".encode("ascii")
        ).digest()[:8],
        "big",
    ) % decision.sample_denominator
    if (
        decision.sample_denominator <= 0
        or not 0 < decision.sample_numerator <= decision.sample_denominator
        or decision.sample_bucket != expected_bucket
        or not decision.reasons
    ):
        raise RuntimeError("certification decision sampling evidence is inconsistent")
    if decision.result_sha256 != observation.result_sha256:
        raise RuntimeError("certification decision result evidence is inconsistent")
    evidence = {
        "policy_sha256": decision.policy_sha256,
        "request_sha256": decision.request_sha256,
        "result_sha256": observation.result_sha256,
        "status": observation.status,
        "objective_gap": (
            None
            if observation.objective_gap is None
            else float(observation.objective_gap).hex()
        ),
        "primal_residual": float(observation.primal_residual).hex(),
        "fallback_used": observation.fallback_used,
        "sample_bucket": decision.sample_bucket,
        "disposition": decision.disposition.value,
        "reasons": [reason.value for reason in decision.reasons],
    }
    if decision.evidence_sha256 != _canonical_sha256(evidence):
        raise RuntimeError("certification decision evidence hash is inconsistent")


@dataclass(frozen=True)
class CertificationCircuitState:
    route_key: str
    open: bool
    policy_sha256: str
    evidence_sha256: str
    reason: str
    opened_at: float


@dataclass(frozen=True)
class CertificationCircuitReset:
    reset_id: int
    route_key: str
    policy_sha256: str
    circuit_evidence_sha256: str
    circuit_reason: str
    circuit_opened_at: float
    approval_sha256: str
    actor: str
    reason: str
    reset_at: float
    reset_evidence_sha256: str


@dataclass(frozen=True)
class StoredCertificationPolicyDecision:
    submission_id: str
    route_key: str
    decision: CertificationPolicyDecision
    online_payload: bytes
    online_elapsed_seconds: float
    ticket_sha256: str | None


class CertificationPolicyConflictError(RuntimeError):
    pass


class CertificationCircuitResetCooldownError(RuntimeError):
    def __init__(self, remaining_seconds: float) -> None:
        self.remaining_seconds = remaining_seconds
        super().__init__(
            f"certification circuit reset cooldown has {remaining_seconds:.6f}s remaining"
        )


class SQLiteCertificationPolicyStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        if not self.path.parent.is_dir():
            raise FileNotFoundError(
                f"certification policy store parent does not exist: {self.path.parent}"
            )
        self._initialize()

    def record_decision(
        self,
        *,
        submission_id: str,
        route_key: str,
        decision: CertificationPolicyDecision,
        online_payload: bytes,
        online_elapsed_seconds: float,
        ticket_sha256: str | None = None,
    ) -> bool:
        if not isinstance(submission_id, str) or not submission_id:
            raise ValueError("submission_id must be a nonempty string")
        if not isinstance(route_key, str) or not route_key:
            raise ValueError("route_key must be a nonempty string")
        if not isinstance(online_payload, bytes) or not online_payload:
            raise ValueError("online_payload must be nonempty bytes")
        if (
            not math.isfinite(online_elapsed_seconds)
            or online_elapsed_seconds < 0.0
        ):
            raise ValueError("online_elapsed_seconds must be finite and nonnegative")
        if ticket_sha256 is not None:
            _require_sha256(ticket_sha256, "ticket_sha256")
        now = time.time()
        payload = (
            submission_id,
            route_key,
            decision.policy_sha256,
            decision.request_sha256,
            decision.result_sha256,
            decision.disposition.value,
            json.dumps([reason.value for reason in decision.reasons]),
            decision.sample_bucket,
            decision.sample_numerator,
            decision.sample_denominator,
            decision.evidence_sha256,
            online_payload,
            online_elapsed_seconds,
            ticket_sha256,
        )
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM policy_decisions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
            if existing is not None:
                existing_payload = tuple(existing[field] for field in (
                    "submission_id", "route_key", "policy_sha256", "request_sha256",
                    "result_sha256", "disposition", "reasons_json", "sample_bucket",
                    "sample_numerator", "sample_denominator", "evidence_sha256",
                    "online_payload", "online_elapsed_seconds",
                    "ticket_sha256",
                ))
                if existing_payload != payload:
                    raise CertificationPolicyConflictError(
                        "submission_id is already bound to different policy evidence"
                    )
                return False
            connection.execute(
                """
                INSERT INTO policy_decisions(
                    submission_id, route_key, policy_sha256, request_sha256,
                    result_sha256, disposition, reasons_json, sample_bucket,
                    sample_numerator, sample_denominator, evidence_sha256,
                    online_payload, online_elapsed_seconds, ticket_sha256, created_at
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (*payload, now),
            )
            return True

    def decision_record(
        self,
        submission_id: str,
    ) -> StoredCertificationPolicyDecision | None:
        if not isinstance(submission_id, str) or not submission_id:
            raise ValueError("submission_id must be a nonempty string")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM policy_decisions WHERE submission_id = ?",
                (submission_id,),
            ).fetchone()
        return None if row is None else self._decision_record(row)

    def decision_record_by_ticket(
        self,
        ticket_sha256: str,
    ) -> StoredCertificationPolicyDecision | None:
        ticket_digest = _require_sha256(ticket_sha256, "ticket_sha256")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM policy_decisions WHERE ticket_sha256 = ?",
                (ticket_digest,),
            ).fetchone()
        return None if row is None else self._decision_record(row)

    def _decision_record(
        self,
        row: sqlite3.Row,
    ) -> StoredCertificationPolicyDecision:
        reasons_payload = json.loads(str(row["reasons_json"]))
        if not isinstance(reasons_payload, list):
            raise RuntimeError("stored certification reasons must be a list")
        decision = CertificationPolicyDecision(
            policy_sha256=_require_sha256(
                str(row["policy_sha256"]), "stored policy_sha256"
            ),
            request_sha256=_require_sha256(
                str(row["request_sha256"]), "stored request_sha256"
            ),
            result_sha256=_require_sha256(
                str(row["result_sha256"]), "stored result_sha256"
            ),
            disposition=CertificationDisposition(str(row["disposition"])),
            reasons=tuple(CertificationReason(str(reason)) for reason in reasons_payload),
            sample_bucket=int(row["sample_bucket"]),
            sample_numerator=int(row["sample_numerator"]),
            sample_denominator=int(row["sample_denominator"]),
            evidence_sha256=_require_sha256(
                str(row["evidence_sha256"]), "stored evidence_sha256"
            ),
        )
        online_elapsed_seconds = float(row["online_elapsed_seconds"])
        if not math.isfinite(online_elapsed_seconds) or online_elapsed_seconds < 0.0:
            raise RuntimeError("stored online_elapsed_seconds is invalid")
        return StoredCertificationPolicyDecision(
            submission_id=str(row["submission_id"]),
            route_key=str(row["route_key"]),
            decision=decision,
            online_payload=bytes(row["online_payload"]),
            online_elapsed_seconds=online_elapsed_seconds,
            ticket_sha256=(
                None
                if row["ticket_sha256"] is None
                else _require_sha256(str(row["ticket_sha256"]), "stored ticket_sha256")
            ),
        )

    def open_circuit(
        self,
        *,
        route_key: str,
        policy_sha256: str,
        evidence_sha256: str,
        reason: str,
    ) -> bool:
        if not isinstance(route_key, str) or not route_key:
            raise ValueError("route_key must be a nonempty string")
        _require_sha256(policy_sha256, "policy_sha256")
        _require_sha256(evidence_sha256, "evidence_sha256")
        if not isinstance(reason, str) or not reason:
            raise ValueError("reason must be a nonempty string")
        now = time.time()
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT * FROM route_circuits WHERE route_key = ?", (route_key,)
            ).fetchone()
            if existing is not None:
                return False
            # A late publisher or terminal reader must not undo an audited
            # reset of this evidence. New divergence evidence still opens.
            reset = connection.execute(
                """
                SELECT * FROM circuit_resets
                WHERE route_key = ? AND policy_sha256 = ?
                    AND circuit_evidence_sha256 = ?
                ORDER BY reset_id DESC LIMIT 1
                """,
                (route_key, policy_sha256, evidence_sha256),
            ).fetchone()
            if reset is not None:
                self._circuit_reset(reset)
                return False
            connection.execute(
                """
                INSERT INTO route_circuits(
                    route_key, policy_sha256, evidence_sha256, reason, opened_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (route_key, policy_sha256, evidence_sha256, reason, now),
            )
            return True

    def circuit_state(self, route_key: str) -> CertificationCircuitState | None:
        if not isinstance(route_key, str) or not route_key:
            raise ValueError("route_key must be a nonempty string")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM route_circuits WHERE route_key = ?", (route_key,)
            ).fetchone()
        if row is None:
            return None
        return CertificationCircuitState(
            route_key=str(row["route_key"]),
            open=True,
            policy_sha256=str(row["policy_sha256"]),
            evidence_sha256=str(row["evidence_sha256"]),
            reason=str(row["reason"]),
            opened_at=float(row["opened_at"]),
        )

    def reset_circuit(
        self,
        *,
        route_key: str,
        expected_evidence_sha256: str,
        approval_sha256: str,
        actor: str,
        reason: str,
        cooldown_seconds: float,
        now: float | None = None,
    ) -> tuple[CertificationCircuitReset, bool]:
        if not isinstance(route_key, str) or not route_key or len(route_key) > 256:
            raise ValueError("route_key must be a nonempty string of at most 256 characters")
        expected_digest = _require_sha256(
            expected_evidence_sha256, "expected_evidence_sha256"
        )
        approval_digest = _require_sha256(approval_sha256, "approval_sha256")
        if not isinstance(actor, str) or not actor or len(actor) > 256:
            raise ValueError("actor must be a nonempty string of at most 256 characters")
        if not isinstance(reason, str) or not reason or len(reason) > 1024:
            raise ValueError("reason must be a nonempty string of at most 1024 characters")
        if not math.isfinite(cooldown_seconds) or cooldown_seconds < 0.0:
            raise ValueError("cooldown_seconds must be finite and nonnegative")
        reset_at = time.time() if now is None else now
        if not math.isfinite(reset_at):
            raise ValueError("reset time must be finite")
        with self._transaction() as connection:
            previous = connection.execute(
                "SELECT * FROM circuit_resets WHERE approval_sha256 = ?",
                (approval_digest,),
            ).fetchone()
            if previous is not None:
                record = self._circuit_reset(previous)
                if (
                    record.route_key != route_key
                    or record.circuit_evidence_sha256 != expected_digest
                    or record.actor != actor
                    or record.reason != reason
                ):
                    raise CertificationPolicyConflictError(
                        "approval_sha256 is already bound to different reset evidence"
                    )
                return record, False
            circuit = connection.execute(
                "SELECT * FROM route_circuits WHERE route_key = ?", (route_key,)
            ).fetchone()
            if circuit is None:
                raise CertificationPolicyConflictError(
                    f"certification route {route_key!r} is not circuit-open"
                )
            if str(circuit["evidence_sha256"]) != expected_digest:
                raise CertificationPolicyConflictError(
                    "circuit reset evidence does not match the open circuit"
                )
            circuit_opened_at = float(circuit["opened_at"])
            remaining = circuit_opened_at + cooldown_seconds - reset_at
            if remaining > 0.0:
                raise CertificationCircuitResetCooldownError(remaining)
            reset_evidence = {
                "schema": "certification-circuit-reset-v1",
                "route_key": route_key,
                "policy_sha256": str(circuit["policy_sha256"]),
                "circuit_evidence_sha256": expected_digest,
                "circuit_reason": str(circuit["reason"]),
                "circuit_opened_at": float(circuit_opened_at).hex(),
                "approval_sha256": approval_digest,
                "actor": actor,
                "reason": reason,
                "reset_at": float(reset_at).hex(),
            }
            reset_evidence_sha256 = _canonical_sha256(reset_evidence)
            cursor = connection.execute(
                """
                INSERT INTO circuit_resets(
                    route_key, policy_sha256, circuit_evidence_sha256,
                    circuit_reason, circuit_opened_at, approval_sha256,
                    actor, reason, reset_at, reset_evidence_sha256
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    route_key,
                    str(circuit["policy_sha256"]),
                    expected_digest,
                    str(circuit["reason"]),
                    circuit_opened_at,
                    approval_digest,
                    actor,
                    reason,
                    reset_at,
                    reset_evidence_sha256,
                ),
            )
            deleted = connection.execute(
                "DELETE FROM route_circuits WHERE route_key = ? AND evidence_sha256 = ?",
                (route_key, expected_digest),
            )
            if deleted.rowcount != 1:
                raise RuntimeError("circuit reset lost its evidence fence")
            row = connection.execute(
                "SELECT * FROM circuit_resets WHERE reset_id = ?",
                (int(cursor.lastrowid),),
            ).fetchone()
            return self._circuit_reset(row), True

    def circuit_reset_by_approval(
        self, approval_sha256: str
    ) -> CertificationCircuitReset | None:
        approval_digest = _require_sha256(approval_sha256, "approval_sha256")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM circuit_resets WHERE approval_sha256 = ?",
                (approval_digest,),
            ).fetchone()
        return None if row is None else self._circuit_reset(row)

    @staticmethod
    def _circuit_reset(row: sqlite3.Row) -> CertificationCircuitReset:
        record = CertificationCircuitReset(
            reset_id=int(row["reset_id"]),
            route_key=str(row["route_key"]),
            policy_sha256=_require_sha256(
                str(row["policy_sha256"]), "stored reset policy_sha256"
            ),
            circuit_evidence_sha256=_require_sha256(
                str(row["circuit_evidence_sha256"]),
                "stored reset circuit_evidence_sha256",
            ),
            circuit_reason=str(row["circuit_reason"]),
            circuit_opened_at=float(row["circuit_opened_at"]),
            approval_sha256=_require_sha256(
                str(row["approval_sha256"]), "stored reset approval_sha256"
            ),
            actor=str(row["actor"]),
            reason=str(row["reason"]),
            reset_at=float(row["reset_at"]),
            reset_evidence_sha256=_require_sha256(
                str(row["reset_evidence_sha256"]),
                "stored reset reset_evidence_sha256",
            ),
        )
        if (
            not math.isfinite(record.circuit_opened_at)
            or not math.isfinite(record.reset_at)
            or not record.circuit_reason
            or not record.actor
            or not record.reason
        ):
            raise RuntimeError("stored circuit reset evidence is invalid")
        expected = _canonical_sha256(
            {
                "schema": "certification-circuit-reset-v1",
                "route_key": record.route_key,
                "policy_sha256": record.policy_sha256,
                "circuit_evidence_sha256": record.circuit_evidence_sha256,
                "circuit_reason": record.circuit_reason,
                "circuit_opened_at": float(record.circuit_opened_at).hex(),
                "approval_sha256": record.approval_sha256,
                "actor": record.actor,
                "reason": record.reason,
                "reset_at": float(record.reset_at).hex(),
            }
        )
        if record.reset_evidence_sha256 != expected:
            raise RuntimeError("stored circuit reset evidence hash is inconsistent")
        return record

    def quick_check(self) -> None:
        with self._connect() as connection:
            result = connection.execute("PRAGMA quick_check").fetchone()[0]
            if result != "ok":
                raise RuntimeError(
                    f"certification policy store quick_check failed: {result}"
                )

    def _initialize(self) -> None:
        with self._connect() as connection:
            mode = connection.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            if str(mode).lower() != "wal":
                raise RuntimeError(
                    f"certification policy store failed to enable WAL mode: {mode}"
                )
            connection.execute("PRAGMA synchronous=FULL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in _SUPPORTED_SQLITE_USER_VERSIONS:
                raise RuntimeError(
                    f"unsupported certification policy schema version {version}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS policy_decisions(
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT NOT NULL UNIQUE,
                    route_key TEXT NOT NULL,
                    policy_sha256 TEXT NOT NULL,
                    request_sha256 TEXT NOT NULL,
                    result_sha256 TEXT NOT NULL,
                    disposition TEXT NOT NULL CHECK(disposition IN (
                        'queued', 'skipped', 'circuit_open'
                    )),
                    reasons_json TEXT NOT NULL,
                    sample_bucket INTEGER NOT NULL,
                    sample_numerator INTEGER NOT NULL,
                    sample_denominator INTEGER NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    online_payload BLOB NOT NULL,
                    online_elapsed_seconds REAL NOT NULL,
                    ticket_sha256 TEXT UNIQUE,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS route_circuits(
                    route_key TEXT PRIMARY KEY,
                    policy_sha256 TEXT NOT NULL,
                    evidence_sha256 TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    opened_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS circuit_resets(
                    reset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    route_key TEXT NOT NULL,
                    policy_sha256 TEXT NOT NULL,
                    circuit_evidence_sha256 TEXT NOT NULL,
                    circuit_reason TEXT NOT NULL,
                    circuit_opened_at REAL NOT NULL,
                    approval_sha256 TEXT NOT NULL UNIQUE,
                    actor TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    reset_at REAL NOT NULL,
                    reset_evidence_sha256 TEXT NOT NULL UNIQUE
                );
                CREATE INDEX IF NOT EXISTS circuit_resets_route_idx
                    ON circuit_resets(route_key, reset_id);
                """
            )
            if version < _SQLITE_USER_VERSION:
                connection.execute(f"PRAGMA user_version={_SQLITE_USER_VERSION}")
                upgraded = int(connection.execute("PRAGMA user_version").fetchone()[0])
                if upgraded != _SQLITE_USER_VERSION:
                    raise RuntimeError(
                        "certification policy store failed to upgrade SQLite user_version"
                    )

    def _connect(self) -> sqlite3.Connection:
        import sqlite3

        connection = sqlite3.connect(self.path, timeout=30.0, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _transaction(self) -> sqlite3.Connection:
        return _SQLiteTransaction(self._connect())


class _SQLiteTransaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        self.connection.execute("BEGIN IMMEDIATE")
        return self.connection

    def __exit__(self, exc_type, exc_value, traceback_value) -> None:
        try:
            if exc_type is None:
                self.connection.execute("COMMIT")
            else:
                self.connection.execute("ROLLBACK")
        finally:
            self.connection.close()
