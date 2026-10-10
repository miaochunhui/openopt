"""Formal explicit-CUDA tests; execute on authorized K8s hardware."""

import pytest
import torch

from openopt import QPConfig, QPProblem, solve_qp


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("linear_solver", ["cholesky", "triangular"])
def test_general_qp_runs_on_explicit_cuda(dtype, linear_solver):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable: run on K8s for GPU acceptance")
    p = QPProblem(
        torch.eye(8, dtype=dtype), -torch.arange(8, dtype=dtype) / 8,
        torch.eye(8, dtype=dtype), torch.zeros(8, dtype=dtype),
        torch.ones(8, dtype=dtype),
    )
    r = solve_qp(p, QPConfig(
        device="cuda:0", dtype=dtype, linear_solver=linear_solver,
        eps_abs=1e-5 if dtype == torch.float32 else 1e-7,
        eps_rel=1e-5 if dtype == torch.float32 else 1e-7,
    ))
    assert r.status == "solved"
    assert r.x.device == r.y.device == torch.device("cuda:0")
    assert r.x.dtype == r.y.dtype == dtype
    assert torch.allclose(r.x.cpu(), -p.q, atol=1e-5)
