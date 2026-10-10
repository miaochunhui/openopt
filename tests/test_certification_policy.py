from __future__ import annotations

from dataclasses import replace
import sqlite3
import subprocess
import sys

import pytest

from openpo.certification_policy import (
    CertificationCircuitResetCooldownError,
    CertificationDisposition,
    CertificationMode,
    CertificationObservation,
    CertificationPolicy,
    CertificationPolicyConflictError,
    CertificationReason,
    SQLiteCertificationPolicyStore,
    certification_decision_from_payload,
    certification_decision_to_payload,
    validate_certification_decision,
)


def _digest(value: int) -> str:
    return f"{value:064x}"


def _observation(**changes: object) -> CertificationObservation:
    values: dict[str, object] = {
        "result_sha256": _digest(9000),
        "status": "solved_to_tolerance",
        "objective_gap": 0.002,
        "primal_residual": 1e-7,
        "fallback_used": False,
    }
    values.update(changes)
    return CertificationObservation(**values)  # type: ignore[arg-type]


def _request_for_bucket(policy: CertificationPolicy, bucket: int) -> str:
    for value in range(1, 10000):
        request_sha256 = _digest(value)
        if policy.sample_bucket(request_sha256) == bucket:
            return request_sha256
    raise AssertionError(f"no request hash found for bucket {bucket}")


def test_openpo_import_after_highspy_does_not_load_sqlite_backend() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "import highspy; import openpo"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_full_policy_always_queues_and_is_hashed() -> None:
    policy = CertificationPolicy()
    decision = policy.evaluate(_digest(1), _observation())

    assert decision.disposition is CertificationDisposition.QUEUED
    assert decision.reasons == (CertificationReason.FULL,)
    assert decision.policy_sha256 == policy.policy_sha256
    assert len(decision.evidence_sha256) == 64


def test_certification_decision_payload_roundtrip_is_strict() -> None:
    decision = CertificationPolicy().evaluate(_digest(1), _observation())
    payload = certification_decision_to_payload(decision)

    assert certification_decision_from_payload(payload) == decision
    with pytest.raises(ValueError, match="unsupported certification decision payload"):
        certification_decision_from_payload({**payload, "unexpected": True})
    with pytest.raises(ValueError, match="sampling evidence is invalid"):
        certification_decision_from_payload({**payload, "sample_bucket": 1})


def test_certification_decision_validation_rejects_evidence_tampering() -> None:
    policy = CertificationPolicy(
        mode=CertificationMode.SAMPLED,
        sample_numerator=1,
        sample_denominator=4,
    )
    request_sha256 = _request_for_bucket(policy, 3)
    observation = _observation()
    decision = policy.evaluate(request_sha256, observation)

    validate_certification_decision(decision, observation)
    with pytest.raises(RuntimeError, match="sampling evidence is inconsistent"):
        validate_certification_decision(
            replace(decision, sample_bucket=0),
            observation,
        )
    with pytest.raises(RuntimeError, match="evidence hash is inconsistent"):
        validate_certification_decision(
            replace(decision, reasons=(CertificationReason.HASH_SAMPLE,)),
            observation,
        )


def test_full_policy_rejects_non_full_sampling_ratio() -> None:
    with pytest.raises(ValueError, match="full mode must use a 1/1 sample ratio"):
        CertificationPolicy(sample_numerator=1, sample_denominator=4)


def test_sampled_policy_is_request_hash_deterministic() -> None:
    policy = CertificationPolicy(
        mode=CertificationMode.SAMPLED,
        sample_numerator=1,
        sample_denominator=4,
    )
    selected = _request_for_bucket(policy, 0)
    skipped = _request_for_bucket(policy, 3)

    first = policy.evaluate(selected, _observation())
    replay = policy.evaluate(selected, _observation())
    rejected = policy.evaluate(skipped, _observation())

    assert first == replay
    assert first.disposition is CertificationDisposition.QUEUED
    assert first.reasons == (CertificationReason.HASH_SAMPLE,)
    assert rejected.disposition is CertificationDisposition.SKIPPED
    assert rejected.reasons == (CertificationReason.HASH_NOT_SELECTED,)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"objective_gap": 0.02}, CertificationReason.OBJECTIVE_GAP),
        ({"primal_residual": 2e-5}, CertificationReason.PRIMAL_RESIDUAL),
        ({"fallback_used": True}, CertificationReason.FALLBACK),
        ({"status": "numerical_failure"}, CertificationReason.STATUS),
        ({"objective_gap": None}, CertificationReason.MISSING_QUALITY),
    ],
)
def test_anomaly_policy_queues_explicit_quality_anomalies(
    changes: dict[str, object],
    reason: CertificationReason,
) -> None:
    policy = CertificationPolicy(
        mode=CertificationMode.ANOMALY,
        sample_numerator=1,
        sample_denominator=8,
    )
    request_sha256 = _request_for_bucket(policy, 7)
    decision = policy.evaluate(request_sha256, _observation(**changes))

    assert decision.disposition is CertificationDisposition.QUEUED
    assert reason in decision.reasons
    assert CertificationReason.HASH_SAMPLE not in decision.reasons


def test_anomaly_policy_keeps_a_deterministic_baseline_sample() -> None:
    policy = CertificationPolicy(
        mode=CertificationMode.ANOMALY,
        sample_numerator=1,
        sample_denominator=8,
    )
    selected = policy.evaluate(_request_for_bucket(policy, 0), _observation())
    skipped = policy.evaluate(_request_for_bucket(policy, 7), _observation())

    assert selected.disposition is CertificationDisposition.QUEUED
    assert selected.reasons == (CertificationReason.HASH_SAMPLE,)
    assert skipped.disposition is CertificationDisposition.SKIPPED
    assert skipped.reasons == (CertificationReason.NO_ANOMALY,)


def test_open_circuit_overrides_every_policy_mode() -> None:
    policy = CertificationPolicy()
    decision = policy.evaluate(_digest(2), _observation(), circuit_open=True)

    assert decision.disposition is CertificationDisposition.CIRCUIT_OPEN
    assert decision.reasons == (CertificationReason.CIRCUIT_OPEN,)


def test_policy_hash_binds_all_sampling_and_anomaly_parameters() -> None:
    baseline = CertificationPolicy(
        mode=CertificationMode.ANOMALY,
        sample_numerator=1,
        sample_denominator=8,
    )
    variants = (
        replace(baseline, sample_numerator=2),
        replace(baseline, sample_denominator=16),
        replace(baseline, anomaly_objective_gap=0.02),
        replace(baseline, anomaly_primal_residual=2e-5),
        replace(baseline, policy_version="async-certification-v2"),
    )

    assert len({baseline.policy_sha256, *(item.policy_sha256 for item in variants)}) == 6


def test_policy_store_initializes_shared_sqlite_user_version_v4(tmp_path) -> None:
    path = tmp_path / "policy.sqlite3"

    store = SQLiteCertificationPolicyStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    store.quick_check()


def test_policy_store_upgrades_legacy_user_version_to_v4(tmp_path) -> None:
    path = tmp_path / "policy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=2")

    SQLiteCertificationPolicyStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4


def test_policy_store_upgrades_shared_shadow_v3_database_to_v4(tmp_path) -> None:
    path = tmp_path / "shared.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("PRAGMA user_version=3")

    store = SQLiteCertificationPolicyStore(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 4
    decision = CertificationPolicy().evaluate(_digest(10), _observation())
    assert store.record_decision(
        submission_id="submission-1",
        route_key="cuda:0",
        decision=decision,
        online_payload=b"online-result",
        online_elapsed_seconds=0.5,
        ticket_sha256=_digest(77),
    )


def test_policy_store_records_idempotent_decisions_and_rejects_conflicts(
    tmp_path,
) -> None:
    store = SQLiteCertificationPolicyStore(tmp_path / "policy.sqlite3")
    policy = CertificationPolicy(
        mode=CertificationMode.SAMPLED,
        sample_numerator=1,
        sample_denominator=4,
    )
    decision = policy.evaluate(_digest(10), _observation())

    assert store.record_decision(
        submission_id="submission-1",
        route_key="cuda:0",
        decision=decision,
        online_payload=b"online-result",
        online_elapsed_seconds=0.5,
        ticket_sha256=_digest(77),
    )
    assert not store.record_decision(
        submission_id="submission-1",
        route_key="cuda:0",
        decision=decision,
        online_payload=b"online-result",
        online_elapsed_seconds=0.5,
        ticket_sha256=_digest(77),
    )
    with pytest.raises(CertificationPolicyConflictError, match="different policy evidence"):
        store.record_decision(
            submission_id="submission-1",
            route_key="cuda:1",
            decision=decision,
            online_payload=b"online-result",
            online_elapsed_seconds=0.5,
            ticket_sha256=_digest(77),
        )

    record = store.decision_record("submission-1")
    assert record is not None
    assert record.decision == decision
    assert record.online_payload == b"online-result"
    assert record.online_elapsed_seconds == 0.5
    assert record.ticket_sha256 == _digest(77)
    assert store.decision_record_by_ticket(_digest(77)) == record


def test_policy_store_circuit_is_durable_and_never_auto_resets(tmp_path) -> None:
    path = tmp_path / "policy.sqlite3"
    policy = CertificationPolicy()
    evidence_sha256 = _digest(33)
    store = SQLiteCertificationPolicyStore(path)

    assert store.open_circuit(
        route_key="cuda:0",
        policy_sha256=policy.policy_sha256,
        evidence_sha256=evidence_sha256,
        reason="shadow_diverged",
    )
    assert not store.open_circuit(
        route_key="cuda:0",
        policy_sha256=CertificationPolicy(
            mode=CertificationMode.SAMPLED,
            sample_numerator=1,
            sample_denominator=4,
        ).policy_sha256,
        evidence_sha256=_digest(34),
        reason="new_policy_must_not_reset",
    )

    reopened = SQLiteCertificationPolicyStore(path)
    state = reopened.circuit_state("cuda:0")
    assert state is not None
    assert state.open
    assert state.policy_sha256 == policy.policy_sha256
    assert state.evidence_sha256 == evidence_sha256
    assert state.reason == "shadow_diverged"
    reopened.quick_check()


def test_policy_store_circuit_reset_requires_cooldown_and_is_idempotent(
    tmp_path,
) -> None:
    store = SQLiteCertificationPolicyStore(tmp_path / "policy.sqlite3")
    policy = CertificationPolicy()
    evidence_sha256 = _digest(33)
    approval_sha256 = _digest(44)
    assert store.open_circuit(
        route_key="cuda:0",
        policy_sha256=policy.policy_sha256,
        evidence_sha256=evidence_sha256,
        reason="shadow_diverged",
    )
    state = store.circuit_state("cuda:0")
    assert state is not None

    with pytest.raises(CertificationCircuitResetCooldownError) as captured:
        store.reset_circuit(
            route_key="cuda:0",
            expected_evidence_sha256=evidence_sha256,
            approval_sha256=approval_sha256,
            actor="risk-oncall",
            reason="incident reviewed",
            cooldown_seconds=60.0,
            now=state.opened_at + 59.0,
        )
    assert captured.value.remaining_seconds == pytest.approx(1.0)

    reset, changed = store.reset_circuit(
        route_key="cuda:0",
        expected_evidence_sha256=evidence_sha256,
        approval_sha256=approval_sha256,
        actor="risk-oncall",
        reason="incident reviewed",
        cooldown_seconds=60.0,
        now=state.opened_at + 60.0,
    )
    assert changed is True
    assert reset.route_key == "cuda:0"
    assert reset.approval_sha256 == approval_sha256
    assert store.circuit_state("cuda:0") is None

    replay, changed = store.reset_circuit(
        route_key="cuda:0",
        expected_evidence_sha256=evidence_sha256,
        approval_sha256=approval_sha256,
        actor="risk-oncall",
        reason="incident reviewed",
        cooldown_seconds=60.0,
        now=state.opened_at + 120.0,
    )
    assert changed is False
    assert replay == reset
    with pytest.raises(CertificationPolicyConflictError, match="different reset evidence"):
        store.reset_circuit(
            route_key="cuda:1",
            expected_evidence_sha256=evidence_sha256,
            approval_sha256=approval_sha256,
            actor="risk-oncall",
            reason="incident reviewed",
            cooldown_seconds=0.0,
        )


def test_policy_store_circuit_reset_rejects_evidence_and_stored_tampering(
    tmp_path,
) -> None:
    store = SQLiteCertificationPolicyStore(tmp_path / "policy.sqlite3")
    policy = CertificationPolicy()
    assert store.open_circuit(
        route_key="cuda:0",
        policy_sha256=policy.policy_sha256,
        evidence_sha256=_digest(33),
        reason="shadow_diverged",
    )
    with pytest.raises(CertificationPolicyConflictError, match="does not match"):
        store.reset_circuit(
            route_key="cuda:0",
            expected_evidence_sha256=_digest(34),
            approval_sha256=_digest(44),
            actor="risk-oncall",
            reason="incident reviewed",
            cooldown_seconds=0.0,
        )
    reset, _ = store.reset_circuit(
        route_key="cuda:0",
        expected_evidence_sha256=_digest(33),
        approval_sha256=_digest(44),
        actor="risk-oncall",
        reason="incident reviewed",
        cooldown_seconds=0.0,
    )
    with store._connect() as connection:
        connection.execute(
            "UPDATE circuit_resets SET actor = ? WHERE reset_id = ?",
            ("tampered", reset.reset_id),
        )
    with pytest.raises(RuntimeError, match="evidence hash is inconsistent"):
        store.circuit_reset_by_approval(_digest(44))


def test_policy_store_rejects_unknown_schema_version(tmp_path) -> None:
    path = tmp_path / "policy.sqlite3"
    store = SQLiteCertificationPolicyStore(path)
    with store._connect() as connection:
        connection.execute("PRAGMA user_version=99")

    with pytest.raises(RuntimeError, match="unsupported certification policy schema"):
        SQLiteCertificationPolicyStore(path)
