"""New direction statuses need independent vectors, not status-only acceptance."""

from copy import deepcopy

import pytest

from openopt.corpus import generate_case
from openopt.qp import QPConfig
from scripts.qualify_qp_corpus import assess, run_solver


@pytest.mark.parametrize("family", ["infeasible", "unbounded"])
def test_generated_directions_qualify_and_mutations_fail(family):
    case = generate_case(family, 8)
    raw = run_solver(case, "openopt", QPConfig(direction_termination=True))
    assert assess(case, "openopt", raw, 5e-6, 1e-6)["verdict"] == "QUALIFIED"
    key = "primal_certificate" if family == "infeasible" else "dual_certificate"
    for value in (None, -raw[key]):
        mutated = deepcopy(raw)
        mutated[key] = value
        assert assess(case, "openopt", mutated, 5e-6, 1e-6)["verdict"] == "UNQUALIFIED"
    malformed = deepcopy(raw)
    malformed[key] = raw[key] * 0
    with pytest.raises(ValueError, match="zero"):
        assess(case, "openopt", malformed, 5e-6, 1e-6)


def test_unbounded_needs_actual_feasible_base_not_planted_witness():
    case = generate_case("unbounded", 8)
    raw = run_solver(case, "openopt", QPConfig(direction_termination=True))
    raw["feasible_base"] = None
    assert assess(case, "openopt", raw, 5e-6, 1e-6)["verdict"] == "UNQUALIFIED"
