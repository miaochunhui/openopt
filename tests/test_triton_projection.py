import pytest
import torch

from openpo.projection import _project_bounded_simplex_torch
from openpo.triton_projection import triton_project_bounded_simplex


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("assets", [32, 250, 1000])
def test_triton_projection_matches_torch(assets: int) -> None:
    generator = torch.Generator(device="cuda").manual_seed(17 + assets)
    values = torch.randn(8, assets, generator=generator, device="cuda")
    lower = torch.zeros(assets, device="cuda")
    upper = torch.full((assets,), min(0.1, 10.0 / assets), device="cuda")
    expected = _project_bounded_simplex_torch(values, lower, upper, bisection_steps=20)
    actual = triton_project_bounded_simplex(values, lower, upper, bisection_steps=20)
    assert torch.allclose(actual.sum(dim=1), torch.ones(8, device="cuda"), atol=3e-6)
    assert torch.all(actual >= lower) and torch.all(actual <= upper)
    assert torch.allclose(actual, expected, atol=3e-6, rtol=0.0)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_triton_projection_uses_input_device() -> None:
    torch.cuda.set_device(0)
    device = torch.device("cuda:1")
    values = torch.randn(3, 64, device=device)
    lower = torch.zeros(64, device=device)
    upper = torch.full((64,), 0.1, device=device)

    actual = triton_project_bounded_simplex(values, lower, upper, bisection_steps=20)

    assert actual.device == device
    assert torch.allclose(actual.sum(dim=1), torch.ones(3, device=device), atol=3e-6)
