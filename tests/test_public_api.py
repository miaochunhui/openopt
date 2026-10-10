from __future__ import annotations

import openpo
from openpo import (
    PreparedExactWorkspace,
    PreparedProductionWorkspace,
    prepare_exact_workspace,
    prepare_production_workspace,
    solve_exact_prepared,
    solve_prepared_production,
    ShadowCertificationService,
    ShadowTicket,
)
from openpo import exact_solver, facade


def test_prepared_workspace_helpers_are_public_api() -> None:
    assert PreparedExactWorkspace is exact_solver.PreparedExactWorkspace
    assert prepare_exact_workspace is exact_solver.prepare_exact_workspace
    assert solve_exact_prepared is exact_solver.solve_exact_prepared
    assert PreparedProductionWorkspace is facade.PreparedProductionWorkspace
    assert prepare_production_workspace is facade.prepare_production_workspace
    assert solve_prepared_production is facade.solve_prepared_production
    assert ShadowCertificationService is openpo.shadow.ShadowCertificationService
    assert ShadowTicket is openpo.shadow.ShadowTicket


def test_prepared_workspace_helpers_are_in_all() -> None:
    expected = {
        "PreparedExactWorkspace",
        "prepare_exact_workspace",
        "solve_exact_prepared",
        "PreparedProductionWorkspace",
        "prepare_production_workspace",
        "solve_prepared_production",
        "ShadowCertificationService",
        "ShadowTicket",
    }
    assert expected <= set(openpo.__all__)
