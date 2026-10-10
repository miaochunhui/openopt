from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearConstraintResidual:
    equality: torch.Tensor | None
    lower_violation: torch.Tensor | None
    upper_violation: torch.Tensor | None

    @staticmethod
    def _scalar_like(
        tensors: tuple[torch.Tensor | None, ...],
    ) -> torch.Tensor:
        for tensor in tensors:
            if tensor is not None:
                return torch.zeros((), device=tensor.device, dtype=tensor.dtype)
        return torch.zeros(())

    @property
    def max_abs_equality(self) -> torch.Tensor:
        if self.equality is None or self.equality.numel() == 0:
            return self._scalar_like(
                (self.equality, self.lower_violation, self.upper_violation)
            )
        return self.equality.abs().amax()

    @property
    def max_inequality_violation(self) -> torch.Tensor:
        parts = [
            tensor
            for tensor in (self.lower_violation, self.upper_violation)
            if tensor is not None
        ]
        if parts and all(tensor.numel() == 0 for tensor in parts):
            return self._scalar_like(
                (self.equality, self.lower_violation, self.upper_violation)
            )
        if parts:
            return torch.stack([tensor.amax() for tensor in parts]).amax()
        return self._scalar_like((self.equality, self.lower_violation, self.upper_violation))

    @property
    def max_violation(self) -> torch.Tensor:
        return torch.stack([self.max_abs_equality, self.max_inequality_violation]).amax()


@dataclass(frozen=True)
class LinearConstraints:
    A_eq: torch.Tensor | None = None
    b_eq: torch.Tensor | None = None
    A_ineq: torch.Tensor | None = None
    lower: torch.Tensor | None = None
    upper: torch.Tensor | None = None

    def __post_init__(self) -> None:
        has_eq = self.A_eq is not None or self.b_eq is not None
        has_ineq = (
            self.A_ineq is not None or self.lower is not None or self.upper is not None
        )
        if not has_eq and not has_ineq:
            raise ValueError("at least one constraint block must be provided")
        if (self.A_eq is None) != (self.b_eq is None):
            raise ValueError("A_eq and b_eq must both be provided or both be None")
        if (
            (self.A_ineq is None) != (self.lower is None)
            or (self.A_ineq is None) != (self.upper is None)
        ):
            raise ValueError(
                "A_ineq, lower, and upper must all be provided or all be None"
            )

        tensors = [
            tensor
            for tensor in (self.A_eq, self.b_eq, self.A_ineq, self.lower, self.upper)
            if tensor is not None
        ]
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("constraint tensors must use floating-point dtypes")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all constraint tensors must be on the same device")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("all constraint tensors must use the same dtype")
        if any(not torch.isfinite(tensor).all().item() for tensor in tensors):
            raise ValueError("constraint tensors must be finite")

        num_variables = None
        if self.A_eq is not None:
            if self.A_eq.ndim != 2:
                raise ValueError("A_eq must have shape [rows, variables]")
            if self.b_eq is None or self.b_eq.ndim != 1:
                raise ValueError("b_eq must have shape [rows]")
            if self.A_eq.shape[0] != self.b_eq.shape[0]:
                raise ValueError("A_eq and b_eq row counts differ")
            num_variables = self.A_eq.shape[1]
        if self.A_ineq is not None:
            if self.A_ineq.ndim != 2:
                raise ValueError("A_ineq must have shape [rows, variables]")
            if self.lower is None or self.lower.ndim != 1:
                raise ValueError("lower must have shape [rows]")
            if self.upper is None or self.upper.ndim != 1:
                raise ValueError("upper must have shape [rows]")
            if self.A_ineq.shape[0] != self.lower.shape[0] or self.A_ineq.shape[0] != self.upper.shape[0]:
                raise ValueError("A_ineq, lower, and upper row counts differ")
            if num_variables is None:
                num_variables = self.A_ineq.shape[1]
            elif self.A_ineq.shape[1] != num_variables:
                raise ValueError("constraint matrices have inconsistent variable counts")
            if torch.any(self.lower > self.upper):
                raise ValueError("lower bounds exceed upper bounds")

    @property
    def num_variables(self) -> int:
        if self.A_eq is not None:
            return self.A_eq.shape[1]
        if self.A_ineq is not None:
            return self.A_ineq.shape[1]
        raise ValueError("at least one constraint block must be provided")

    @property
    def device(self) -> torch.device:
        for tensor in (self.A_eq, self.b_eq, self.A_ineq, self.lower, self.upper):
            if tensor is not None:
                return tensor.device
        raise ValueError("at least one constraint block must be provided")

    @property
    def dtype(self) -> torch.dtype:
        for tensor in (self.A_eq, self.b_eq, self.A_ineq, self.lower, self.upper):
            if tensor is not None:
                return tensor.dtype
        raise ValueError("at least one constraint block must be provided")

    def to(self, device: torch.device | str) -> LinearConstraints:
        return LinearConstraints(
            A_eq=None if self.A_eq is None else self.A_eq.to(device),
            b_eq=None if self.b_eq is None else self.b_eq.to(device),
            A_ineq=None if self.A_ineq is None else self.A_ineq.to(device),
            lower=None if self.lower is None else self.lower.to(device),
            upper=None if self.upper is None else self.upper.to(device),
        )

    def residual(self, weights: torch.Tensor) -> LinearConstraintResidual:
        if weights.ndim not in (1, 2):
            raise ValueError("weights must have shape [variables] or [batch, variables]")
        if weights.shape[-1] != self.num_variables:
            raise ValueError("weights and constraints have inconsistent variable counts")
        if weights.device != self.device:
            raise ValueError("weights and constraint tensors must be on the same device")
        if weights.dtype != self.dtype:
            raise ValueError("weights and constraint tensors must use the same dtype")
        if not weights.is_floating_point():
            raise TypeError("weights must use a floating-point dtype")
        if not torch.isfinite(weights).all():
            raise ValueError("weights must be finite")

        batch_weights = weights.unsqueeze(0) if weights.ndim == 1 else weights

        eq_residual = None
        if self.A_eq is not None:
            eq_residual = batch_weights @ self.A_eq.T - self.b_eq

        lower_violation = None
        upper_violation = None
        if self.A_ineq is not None:
            ineq_values = batch_weights @ self.A_ineq.T
            lower_violation = torch.clamp(self.lower - ineq_values, min=0)
            upper_violation = torch.clamp(ineq_values - self.upper, min=0)

        if weights.ndim == 1:
            if eq_residual is not None:
                eq_residual = eq_residual.squeeze(0)
            if lower_violation is not None:
                lower_violation = lower_violation.squeeze(0)
            if upper_violation is not None:
                upper_violation = upper_violation.squeeze(0)

        return LinearConstraintResidual(
            equality=eq_residual,
            lower_violation=lower_violation,
            upper_violation=upper_violation,
        )
