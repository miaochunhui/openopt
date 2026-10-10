"""A visible divergent terminal result must already enforce the route circuit."""

from dataclasses import replace
from threading import Event

import pytest

from openpo import durable_shadow as durable_module
from openpo.certification_policy import CertificationPolicy
from openpo.durable_shadow import (
    DurableShadowCertificationService,
    ShadowRouteCircuitOpenError,
)
from tests.test_durable_shadow import _request


def test_ready_divergence_enforces_circuit_before_worker_publication_finishes(
    tmp_path, monkeypatch,
):
    original_certify = durable_module.certify_exact
    original_open = durable_module.SQLiteCertificationPolicyStore.open_circuit
    entered = Event()
    release = Event()
    calls = 0

    def divergent(request):
        result = original_certify(request)
        return replace(result, objective=result.objective + 1.0)

    def delayed_open(store, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            entered.set()
            if not release.wait(10):
                raise TimeoutError("circuit publication probe not released")
        return original_open(store, **kwargs)

    monkeypatch.setattr(durable_module, "certify_exact", divergent)
    monkeypatch.setattr(durable_module.SQLiteCertificationPolicyStore, "open_circuit", delayed_open)
    service = DurableShadowCertificationService(
        tmp_path / "order.sqlite3", certification_policy=CertificationPolicy(),
    )
    submission = None
    try:
        submission = service.submit(_request(), submission_id="divergent")
        assert submission.ticket is not None
        assert entered.wait(10), "worker did not reach the terminal/circuit window"
        assert service._policy_store.circuit_state("cpu") is None
        assert service.is_ready(submission.ticket)
        with pytest.raises(ShadowRouteCircuitOpenError, match="circuit-open"):
            service.submit(_request(), submission_id="must-be-blocked")
    finally:
        release.set()
        if submission is not None:
            service.collect(submission.ticket, timeout=10)
            service.ack(submission.ticket)
        service.shutdown()
