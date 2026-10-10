from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class FactorRiskModel:
    exposures: torch.Tensor
    factor_variance: torch.Tensor
    specific_variance: torch.Tensor

    def __post_init__(self) -> None:
        if self.exposures.ndim != 2:
            raise ValueError("exposures must have shape [assets, factors]")
        assets, factors = self.exposures.shape
        if self.factor_variance.shape != (factors,):
            raise ValueError("factor_variance must have shape [factors]")
        if self.specific_variance.shape != (assets,):
            raise ValueError("specific_variance must have shape [assets]")
        tensors = (self.exposures, self.factor_variance, self.specific_variance)
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("risk model tensors must use floating-point dtypes")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("risk model tensors must be on the same device")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("risk model tensors must use the same dtype")
        if torch.any(self.factor_variance <= 0) or torch.any(self.specific_variance <= 0):
            raise ValueError("all variances must be positive")

    @property
    def assets(self) -> int:
        return self.exposures.shape[0]

    def matvec(self, weights: torch.Tensor) -> torch.Tensor:
        factor_load = weights @ self.exposures
        return (
            (factor_load * self.factor_variance) @ self.exposures.T
            + weights * self.specific_variance
        )

    def to(self, device: torch.device | str) -> FactorRiskModel:
        return FactorRiskModel(
            exposures=self.exposures.to(device),
            factor_variance=self.factor_variance.to(device),
            specific_variance=self.specific_variance.to(device),
        )


@dataclass(frozen=True)
class PortfolioProblem:
    alpha: torch.Tensor
    risk: FactorRiskModel
    lower: torch.Tensor
    upper: torch.Tensor
    budget: float = 1.0
    risk_aversion: float = 1.0
    current: torch.Tensor | None = None
    trade_cost: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.budget):
            raise ValueError("budget must be finite")
        if not math.isfinite(self.risk_aversion) or not math.isfinite(self.trade_cost):
            raise ValueError("risk_aversion and trade_cost must be finite")
        if self.alpha.ndim not in (1, 2):
            raise ValueError("alpha must have shape [assets] or [batch, assets]")
        if self.alpha.ndim == 2 and self.alpha.shape[0] == 0:
            raise ValueError("alpha batch must not be empty")
        if self.alpha.shape[-1] != self.risk.assets:
            raise ValueError("alpha and risk model asset counts differ")
        if self.lower.shape != (self.risk.assets,) or self.upper.shape != (self.risk.assets,):
            raise ValueError("bounds must have shape [assets]")
        if torch.any(self.lower > self.upper):
            raise ValueError("lower bounds exceed upper bounds")
        if float(self.lower.sum()) > self.budget or float(self.upper.sum()) < self.budget:
            raise ValueError("budget is infeasible under supplied bounds")
        if self.current is not None and self.current.shape[-1] != self.risk.assets:
            raise ValueError("current holdings and risk model asset counts differ")
        tensors = [self.alpha, self.lower, self.upper]
        if self.current is not None:
            tensors.append(self.current)
        tensors.extend(
            [
                self.risk.exposures,
                self.risk.factor_variance,
                self.risk.specific_variance,
            ]
        )
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("all portfolio problem tensors must be on the same device")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("all portfolio problem tensors must use the same dtype")
        if self.risk_aversion <= 0 or self.trade_cost < 0:
            raise ValueError("risk_aversion must be positive and trade_cost nonnegative")

    @property
    def batched_alpha(self) -> torch.Tensor:
        return self.alpha.unsqueeze(0) if self.alpha.ndim == 1 else self.alpha

    def to(self, device: torch.device | str) -> PortfolioProblem:
        return PortfolioProblem(
            alpha=self.alpha.to(device),
            risk=self.risk.to(device),
            lower=self.lower.to(device),
            upper=self.upper.to(device),
            budget=self.budget,
            risk_aversion=self.risk_aversion,
            current=None if self.current is None else self.current.to(device),
            trade_cost=self.trade_cost,
        )
