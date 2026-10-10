from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .model import FactorRiskModel, PortfolioProblem


def _require_floating_tensor(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype")


def _require_finite_tensor(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all().item():
        raise ValueError(f"{name} must be finite")


def _require_vector_or_matrix(name: str, tensor: torch.Tensor, assets: int) -> None:
    if tensor.ndim == 1:
        if tensor.shape[0] != assets:
            raise ValueError(f"{name} must have shape [assets]")
        return
    if tensor.ndim == 2:
        if tensor.shape[-1] != assets:
            raise ValueError(f"{name} must have shape [batch, assets]")
        if tensor.shape[0] == 0:
            raise ValueError(f"{name} batch must not be empty")
        return
    raise ValueError(f"{name} must have shape [assets] or [batch, assets]")


def _batch_size(tensor: torch.Tensor) -> int:
    return 1 if tensor.ndim == 1 else tensor.shape[0]


def _expand_to_batch(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] != 1:
        raise ValueError("batch dimensions are incompatible")
    return tensor.expand(batch_size, -1).clone()


def _align_batch_tensors(
    tensors: dict[str, torch.Tensor | None],
    assets: int,
) -> tuple[dict[str, torch.Tensor | None], bool]:
    present = {name: tensor for name, tensor in tensors.items() if tensor is not None}
    if not present:
        return tensors, False
    batch_size = max(_batch_size(tensor) for tensor in present.values())
    for name, tensor in present.items():
        if tensor.shape[-1] != assets:
            raise ValueError(f"{name} must have {assets} assets")
        if _batch_size(tensor) not in (1, batch_size):
            raise ValueError("batch dimensions are incompatible")
    broadcasted = {
        name: None if tensor is None else _expand_to_batch(tensor, batch_size)
        for name, tensor in tensors.items()
    }
    return broadcasted, any(tensor is not None and tensor.ndim == 2 for tensor in present.values())


@dataclass(frozen=True)
class BarraRiskData:
    exposures: torch.Tensor
    factor_covariance: torch.Tensor
    specific_variance: torch.Tensor
    benchmark: torch.Tensor

    def __post_init__(self) -> None:
        tensors = (
            self.exposures,
            self.factor_covariance,
            self.specific_variance,
            self.benchmark,
        )
        if any(not tensor.is_floating_point() for tensor in tensors):
            raise TypeError("barra risk tensors must use floating-point dtypes")
        if len({tensor.device for tensor in tensors}) != 1:
            raise ValueError("barra risk tensors must be on the same device")
        if len({tensor.dtype for tensor in tensors}) != 1:
            raise ValueError("barra risk tensors must use the same dtype")
        if self.exposures.ndim != 2:
            raise ValueError("exposures must have shape [assets, factors]")
        assets, factors = self.exposures.shape
        if self.factor_covariance.ndim != 2:
            raise ValueError("factor_covariance must have shape [factors, factors]")
        if self.factor_covariance.shape != (factors, factors):
            raise ValueError("factor_covariance must have shape [factors, factors]")
        if self.specific_variance.shape != (assets,):
            raise ValueError("specific_variance must have shape [assets]")
        _require_vector_or_matrix("benchmark", self.benchmark, assets)
        for name, tensor in (
            ("exposures", self.exposures),
            ("factor_covariance", self.factor_covariance),
            ("specific_variance", self.specific_variance),
            ("benchmark", self.benchmark),
        ):
            _require_finite_tensor(name, tensor)
        if torch.any(self.specific_variance <= 0):
            raise ValueError("specific_variance must be positive")
        if not torch.equal(self.factor_covariance, self.factor_covariance.transpose(-1, -2)):
            raise ValueError("factor_covariance must be symmetric")
        try:
            _, info = torch.linalg.cholesky_ex(self.factor_covariance)
        except RuntimeError as error:
            raise ValueError("factor_covariance must be positive definite") from error
        if int(info.item()) != 0:
            raise ValueError("factor_covariance must be positive definite")

    @property
    def assets(self) -> int:
        return self.exposures.shape[0]

    @property
    def factors(self) -> int:
        return self.exposures.shape[1]

    def to(self, device: torch.device | str) -> BarraRiskData:
        return BarraRiskData(
            exposures=self.exposures.to(device),
            factor_covariance=self.factor_covariance.to(device),
            specific_variance=self.specific_variance.to(device),
            benchmark=self.benchmark.to(device),
        )

    def to_factor_risk_model(self) -> FactorRiskModel:
        chol = torch.linalg.cholesky(self.factor_covariance)
        return FactorRiskModel(
            exposures=self.exposures @ chol,
            factor_variance=torch.ones(
                self.factors,
                device=self.exposures.device,
                dtype=self.exposures.dtype,
            ),
            specific_variance=self.specific_variance,
        )

    def to_portfolio_problem(
        self,
        alpha: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        *,
        budget: float = 1.0,
        risk_aversion: float = 1.0,
        current: torch.Tensor | None = None,
        trade_cost: float = 0.0,
    ) -> PortfolioProblem:
        if alpha.device != self.exposures.device or alpha.dtype != self.exposures.dtype:
            raise ValueError("alpha must use the same device and dtype as the Barra data")
        if current is not None and (
            current.device != self.exposures.device or current.dtype != self.exposures.dtype
        ):
            raise ValueError("current must use the same device and dtype as the Barra data")
        if not math.isfinite(budget):
            raise ValueError("budget must be finite")
        if not math.isfinite(risk_aversion) or risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive and finite")
        if not math.isfinite(trade_cost) or trade_cost < 0:
            raise ValueError("trade_cost must be finite and nonnegative")
        if not torch.isfinite(alpha).all().item():
            raise ValueError("alpha must be finite")
        if current is not None and not torch.isfinite(current).all().item():
            raise ValueError("current must be finite")

        aligned, batched = _align_batch_tensors(
            {
                "alpha": alpha,
                "benchmark": self.benchmark,
                "current": current,
            },
            self.assets,
        )
        alpha_batched = aligned["alpha"]
        benchmark_batched = aligned["benchmark"]
        current_batched = aligned["current"]

        if alpha_batched is None or benchmark_batched is None:
            raise AssertionError("internal batch alignment failed")
        risk_model = self.to_factor_risk_model()
        alpha_adjusted = alpha_batched + risk_aversion * risk_model.matvec(benchmark_batched)
        problem_current = None if current_batched is None else current_batched
        if not batched:
            alpha_adjusted = alpha_adjusted.squeeze(0)
            if problem_current is not None:
                problem_current = problem_current.squeeze(0)
        return PortfolioProblem(
            alpha=alpha_adjusted,
            risk=risk_model,
            lower=lower,
            upper=upper,
            budget=budget,
            risk_aversion=risk_aversion,
            current=problem_current,
            trade_cost=trade_cost,
        )

    def active_objective(
        self,
        alpha: torch.Tensor,
        weights: torch.Tensor,
        *,
        risk_aversion: float = 1.0,
        current: torch.Tensor | None = None,
        trade_cost: float = 0.0,
    ) -> torch.Tensor:
        if alpha.device != self.exposures.device or alpha.dtype != self.exposures.dtype:
            raise ValueError("alpha must use the same device and dtype as the Barra data")
        if weights.device != self.exposures.device or weights.dtype != self.exposures.dtype:
            raise ValueError("weights must use the same device and dtype as the Barra data")
        if current is not None and (
            current.device != self.exposures.device or current.dtype != self.exposures.dtype
        ):
            raise ValueError("current must use the same device and dtype as the Barra data")
        if not math.isfinite(risk_aversion) or risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive and finite")
        if not math.isfinite(trade_cost) or trade_cost < 0:
            raise ValueError("trade_cost must be finite and nonnegative")
        for name, tensor in (("alpha", alpha), ("weights", weights), ("current", current)):
            if tensor is not None and not torch.isfinite(tensor).all().item():
                raise ValueError(f"{name} must be finite")

        aligned, _ = _align_batch_tensors(
            {
                "alpha": alpha,
                "weights": weights,
                "benchmark": self.benchmark,
                "current": current,
            },
            self.assets,
        )
        alpha_batched = aligned["alpha"]
        weights_batched = aligned["weights"]
        benchmark_batched = aligned["benchmark"]
        current_batched = aligned["current"]
        if alpha_batched is None or weights_batched is None or benchmark_batched is None:
            raise AssertionError("internal batch alignment failed")

        risk_model = self.to_factor_risk_model()
        active_weights = weights_batched - benchmark_batched
        risk = 0.5 * risk_aversion * (
            active_weights * risk_model.matvec(active_weights)
        ).sum(dim=1)
        utility = (alpha_batched * weights_batched).sum(dim=1)
        if current_batched is None or trade_cost == 0:
            trade = torch.zeros_like(risk)
        else:
            trade = 0.5 * trade_cost * ((weights_batched - current_batched) ** 2).sum(dim=1)
        return risk - utility + trade


def active_objective(
    data: BarraRiskData,
    alpha: torch.Tensor,
    weights: torch.Tensor,
    *,
    risk_aversion: float = 1.0,
    current: torch.Tensor | None = None,
    trade_cost: float = 0.0,
) -> torch.Tensor:
    return data.active_objective(
        alpha,
        weights,
        risk_aversion=risk_aversion,
        current=current,
        trade_cost=trade_cost,
    )
