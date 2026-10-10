import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpo.constraints import LinearConstraintResidual, LinearConstraints


def test_linear_constraints_validate_and_compute_residuals() -> None:
    A_eq = torch.as_tensor([[1.0, 1.0], [1.0, -1.0]])
    b_eq = torch.as_tensor([1.0, 0.0])
    A_ineq = torch.as_tensor([[1.0, 0.0], [0.0, 1.0]])
    lower = torch.as_tensor([0.2, 0.1])
    upper = torch.as_tensor([0.8, 0.9])

    constraints = LinearConstraints(A_eq, b_eq, A_ineq, lower, upper)

    assert constraints.num_variables == 2
    assert constraints.device == torch.device("cpu")
    assert constraints.dtype == torch.float32
    moved = constraints.to("cpu")
    assert moved == constraints

    weights = torch.tensor([0.7, 0.1])
    residual = constraints.residual(weights)

    expected_equality = weights @ A_eq.T - b_eq
    expected_values = weights @ A_ineq.T
    expected_lower = torch.clamp(lower - expected_values, min=0)
    expected_upper = torch.clamp(expected_values - upper, min=0)

    assert torch.allclose(residual.equality, expected_equality)
    assert torch.allclose(residual.lower_violation, expected_lower)
    assert torch.allclose(residual.upper_violation, expected_upper)
    assert torch.allclose(residual.max_abs_equality, expected_equality.abs().amax())
    assert torch.allclose(
        residual.max_inequality_violation,
        torch.stack([expected_lower.amax(), expected_upper.amax()]).amax(),
    )
    assert torch.allclose(
        residual.max_violation,
        torch.stack([expected_equality.abs().amax(), expected_lower.amax(), expected_upper.amax()]).amax(),
    )


@pytest.mark.parametrize(
    "kwargs, exc_type, message",
    [
        ({}, ValueError, "at least one constraint block must be provided"),
        (
            {"A_eq": torch.as_tensor([[1.0, 1.0]])},
            ValueError,
            "A_eq and b_eq must both be provided or both be None",
        ),
        (
            {
                "A_ineq": torch.as_tensor([[1.0, 1.0]]),
                "lower": torch.as_tensor([1.0]),
            },
            ValueError,
            "A_ineq, lower, and upper must all be provided or all be None",
        ),
        (
            {
                "A_eq": torch.as_tensor([[1.0, 1.0]]),
                "b_eq": torch.as_tensor([1.0, 2.0]),
            },
            ValueError,
            "A_eq and b_eq row counts differ",
        ),
        (
            {
                "A_ineq": torch.as_tensor([[1.0, 1.0]]),
                "lower": torch.as_tensor([1.0, 1.0]),
                "upper": torch.as_tensor([1.0, 1.0]),
            },
            ValueError,
            "A_ineq, lower, and upper row counts differ",
        ),
        (
            {
                "A_eq": torch.as_tensor([[1, 2]], dtype=torch.int64),
                "b_eq": torch.as_tensor([1], dtype=torch.int64),
            },
            TypeError,
            "constraint tensors must use floating-point dtypes",
        ),
        (
            {
                "A_ineq": torch.as_tensor([[1.0, 0.0], [0.0, 1.0]]),
                "lower": torch.as_tensor([0.0, float("nan")]),
                "upper": torch.as_tensor([1.0, 1.0]),
            },
            ValueError,
            "constraint tensors must be finite",
        ),
        (
            {
                "A_eq": torch.as_tensor([[1.0, 1.0]]),
                "b_eq": torch.as_tensor([1.0]),
                "A_ineq": torch.as_tensor([[1.0, 1.0, 1.0]]),
                "lower": torch.as_tensor([0.0]),
                "upper": torch.as_tensor([1.0]),
            },
            ValueError,
            "constraint matrices have inconsistent variable counts",
        ),
        (
            {
                "A_ineq": torch.as_tensor([[1.0, 1.0]]),
                "lower": torch.as_tensor([1.0]),
                "upper": torch.as_tensor([0.5]),
            },
            ValueError,
            "lower bounds exceed upper bounds",
        ),
    ],
)
def test_linear_constraints_validation_errors(kwargs, exc_type, message) -> None:
    with pytest.raises(exc_type, match=message):
        LinearConstraints(**kwargs)


def test_linear_constraint_residual_scalar_like_properties() -> None:
    residual = LinearConstraintResidual(None, None, None)

    assert residual.max_abs_equality.shape == ()
    assert residual.max_inequality_violation.shape == ()
    assert residual.max_violation.shape == ()
