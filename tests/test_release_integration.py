"""Integration contracts across independently developed experimental modules."""

from types import SimpleNamespace

import pytest
import torch

from openopt.batch import BatchQPProblem, PreparedBatchQP
from openopt.qp import QPConfig, QPProblem


@pytest.mark.parametrize("option", ["dual_gap_termination", "direction_termination"])
def test_batch_never_silently_ignores_unsupported_termination(option):
    p = QPProblem(
        torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64),
        torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )
    config = SimpleNamespace(
        validate=lambda: None, adaptive_rho=False,
        dual_gap_termination=False, direction_termination=False,
    )
    setattr(config, option, True)
    with pytest.raises(ValueError, match="does not support"):
        PreparedBatchQP(BatchQPProblem.stack([p, p]), config)


def test_reference_device_requests_never_fall_back():
    from openopt.conic import ConicConfig
    from openopt.integer import MILPConfig

    for config in (ConicConfig(device="cuda:0"), MILPConfig(backend="highspy", device="cuda:0")):
        with pytest.raises((RuntimeError, ValueError), match="CUDA|cpu"):
            config.validate()


def test_default_qp_config_keeps_new_modes_opt_in():
    config = QPConfig()
    assert not getattr(config, "dual_gap_termination", False)
    assert not getattr(config, "direction_termination", False)


def test_experimental_exports_preserve_module_identity():
    import openopt
    from openopt.scaling import solve_scaled_qp
    from openopt.conic import solve_conic
    from openopt.integer import solve_milp
    from openopt.operators import JacobiPreconditioner

    for name, value in (
        ("solve_scaled_qp", solve_scaled_qp), ("solve_conic", solve_conic),
        ("solve_milp", solve_milp), ("JacobiPreconditioner", JacobiPreconditioner),
    ):
        assert name in openopt.__all__ and getattr(openopt, name) is value
