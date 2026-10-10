from __future__ import annotations

import base64
from dataclasses import asdict
import hashlib
import json
import math
from typing import Any

import torch

from .constrained_solver import PrimalDualConfig
from .constraints import LinearConstraints
from .facade import (
    FallbackPolicy,
    LaneResult,
    ProductionRequest,
    ProductionResult,
    ResidualEvidence,
    SolveStatus,
)
from .model import FactorRiskModel, PortfolioProblem


_DTYPES = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
}


def encode_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def decode_payload(encoded: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("durable shadow payload is not valid UTF-8 JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("durable shadow payload must be a JSON object")
    return payload


def payload_sha256(encoded: bytes) -> str:
    return hashlib.sha256(encoded).hexdigest()


def tensor_to_payload(tensor: torch.Tensor | None) -> dict[str, Any] | None:
    if tensor is None:
        return None
    value = tensor.detach().contiguous().cpu()
    dtype_name = str(value.dtype)
    if dtype_name not in _DTYPES:
        raise TypeError(f"durable shadow tensor dtype is unsupported: {dtype_name}")
    if not torch.isfinite(value).all().item():
        raise ValueError("durable shadow tensors must be finite")
    raw = value.numpy().tobytes(order="C")
    return {
        "dtype": dtype_name,
        "shape": list(value.shape),
        "data_base64": base64.b64encode(raw).decode("ascii"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def tensor_from_payload(payload: object, *, nullable: bool = False) -> torch.Tensor | None:
    if payload is None:
        if nullable:
            return None
        raise ValueError("durable shadow tensor payload must not be null")
    if not isinstance(payload, dict):
        raise ValueError("durable shadow tensor payload must be an object")
    dtype_name = payload.get("dtype")
    dtype = _DTYPES.get(dtype_name)
    if dtype is None:
        raise ValueError(f"durable shadow tensor dtype is unsupported: {dtype_name!r}")
    shape = payload.get("shape")
    if (
        not isinstance(shape, list)
        or any(not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape)
    ):
        raise ValueError("durable shadow tensor shape is invalid")
    encoded = payload.get("data_base64")
    if not isinstance(encoded, str):
        raise ValueError("durable shadow tensor data_base64 is invalid")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("durable shadow tensor base64 is invalid") from error
    expected_sha = payload.get("sha256")
    if not isinstance(expected_sha, str) or hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("durable shadow tensor hash mismatch")
    item_size = torch.empty((), dtype=dtype).element_size()
    expected_count = math.prod(shape)
    if len(raw) != expected_count * item_size:
        raise ValueError("durable shadow tensor byte length does not match shape")
    return torch.frombuffer(bytearray(raw), dtype=dtype).reshape(tuple(shape)).clone()


def request_to_payload(request: ProductionRequest) -> dict[str, Any]:
    config = asdict(request.fast_config)
    for field_name in (
        "tolerance",
        "constraint_tolerance",
        "step_safety",
        "warm_start_max_alpha_delta",
    ):
        value = config[field_name]
        config[field_name] = None if value is None else float(value).hex()
    return {
        "schema": "production-request-v1",
        "problem": {
            "alpha": tensor_to_payload(request.problem.alpha),
            "risk_exposures": tensor_to_payload(request.problem.risk.exposures),
            "risk_factor_variance": tensor_to_payload(request.problem.risk.factor_variance),
            "risk_specific_variance": tensor_to_payload(request.problem.risk.specific_variance),
            "lower": tensor_to_payload(request.problem.lower),
            "upper": tensor_to_payload(request.problem.upper),
            "budget": float(request.problem.budget).hex(),
            "risk_aversion": float(request.problem.risk_aversion).hex(),
            "current": tensor_to_payload(request.problem.current),
            "trade_cost": float(request.problem.trade_cost).hex(),
        },
        "constraints": {
            "A_eq": tensor_to_payload(request.constraints.A_eq),
            "b_eq": tensor_to_payload(request.constraints.b_eq),
            "A_ineq": tensor_to_payload(request.constraints.A_ineq),
            "lower": tensor_to_payload(request.constraints.lower),
            "upper": tensor_to_payload(request.constraints.upper),
        },
        "fast_device": request.fast_device,
        "exact_backend": request.exact_backend,
        "fallback_policy": request.fallback_policy.value,
        "fast_config": config,
        "residual_tolerance": float(request.residual_tolerance).hex(),
        "objective_gap_tolerance": float(request.objective_gap_tolerance).hex(),
    }


def request_from_payload(payload: dict[str, Any]) -> ProductionRequest:
    if payload.get("schema") != "production-request-v1":
        raise ValueError("unsupported durable production request schema")
    problem = _mapping(payload.get("problem"), "problem")
    constraints = _mapping(payload.get("constraints"), "constraints")
    config = _mapping(payload.get("fast_config"), "fast_config").copy()
    for field_name in (
        "tolerance",
        "constraint_tolerance",
        "step_safety",
        "warm_start_max_alpha_delta",
    ):
        value = config.get(field_name)
        config[field_name] = None if value is None else float.fromhex(_string(value, field_name))
    risk = FactorRiskModel(
        exposures=_tensor(problem, "risk_exposures"),
        factor_variance=_tensor(problem, "risk_factor_variance"),
        specific_variance=_tensor(problem, "risk_specific_variance"),
    )
    portfolio = PortfolioProblem(
        alpha=_tensor(problem, "alpha"),
        risk=risk,
        lower=_tensor(problem, "lower"),
        upper=_tensor(problem, "upper"),
        budget=float.fromhex(_string(problem.get("budget"), "budget")),
        risk_aversion=float.fromhex(_string(problem.get("risk_aversion"), "risk_aversion")),
        current=tensor_from_payload(problem.get("current"), nullable=True),
        trade_cost=float.fromhex(_string(problem.get("trade_cost"), "trade_cost")),
    )
    linear = LinearConstraints(
        A_eq=tensor_from_payload(constraints.get("A_eq"), nullable=True),
        b_eq=tensor_from_payload(constraints.get("b_eq"), nullable=True),
        A_ineq=tensor_from_payload(constraints.get("A_ineq"), nullable=True),
        lower=tensor_from_payload(constraints.get("lower"), nullable=True),
        upper=tensor_from_payload(constraints.get("upper"), nullable=True),
    )
    return ProductionRequest(
        problem=portfolio,
        constraints=linear,
        fast_device=_string(payload.get("fast_device"), "fast_device"),
        exact_backend=_string(payload.get("exact_backend"), "exact_backend"),
        fallback_policy=FallbackPolicy(_string(payload.get("fallback_policy"), "fallback_policy")),
        fast_config=PrimalDualConfig(**config),
        residual_tolerance=float.fromhex(_string(payload.get("residual_tolerance"), "residual_tolerance")),
        objective_gap_tolerance=float.fromhex(_string(payload.get("objective_gap_tolerance"), "objective_gap_tolerance")),
    )


def residual_to_payload(residual: ResidualEvidence) -> dict[str, Any]:
    return {
        key: None if value is None else float(value).hex()
        for key, value in asdict(residual).items()
    }


def residual_from_payload(payload: dict[str, Any]) -> ResidualEvidence:
    values = {
        key: None if value is None else float.fromhex(_string(value, key))
        for key, value in payload.items()
    }
    return ResidualEvidence(**values)


def lane_to_payload(lane: LaneResult) -> dict[str, Any]:
    return {
        "lane": lane.lane,
        "status": lane.status.value,
        "certificate": lane.certificate,
        "requested_backend": lane.requested_backend,
        "backend": lane.backend,
        "backend_version": lane.backend_version,
        "device": lane.device,
        "input_sha256": lane.input_sha256,
        "request_sha256": lane.request_sha256,
        "weights_sha256": lane.weights_sha256,
        "result_sha256": lane.result_sha256,
        "weights": tensor_to_payload(lane.weights),
        "objective": tensor_to_payload(lane.objective),
        "objective_gap": _optional_hex(lane.objective_gap),
        "residuals": residual_to_payload(lane.residuals),
        "iterations": lane.iterations,
        "elapsed_seconds": float(lane.elapsed_seconds).hex(),
        "raw_statuses": list(lane.raw_statuses),
        "fallback_used": lane.fallback_used,
        "fallback_reason": lane.fallback_reason,
        "prepared": lane.prepared,
        "template_sha256": lane.template_sha256,
        "setup_seconds": _optional_hex(lane.setup_seconds),
        "cold_seconds": _optional_hex(lane.cold_seconds),
    }


def lane_from_payload(payload: dict[str, Any]) -> LaneResult:
    statuses = payload.get("raw_statuses")
    if not isinstance(statuses, list) or any(not isinstance(value, str) for value in statuses):
        raise ValueError("durable lane raw_statuses are invalid")
    return LaneResult(
        lane=_string(payload.get("lane"), "lane"),
        status=SolveStatus(_string(payload.get("status"), "status")),
        certificate=_string(payload.get("certificate"), "certificate"),
        requested_backend=_string(payload.get("requested_backend"), "requested_backend"),
        backend=_string(payload.get("backend"), "backend"),
        backend_version=_string(payload.get("backend_version"), "backend_version"),
        device=_string(payload.get("device"), "device"),
        input_sha256=_string(payload.get("input_sha256"), "input_sha256"),
        request_sha256=_string(payload.get("request_sha256"), "request_sha256"),
        weights_sha256=_string(payload.get("weights_sha256"), "weights_sha256"),
        result_sha256=_string(payload.get("result_sha256"), "result_sha256"),
        weights=_tensor(payload, "weights"),
        objective=_tensor(payload, "objective"),
        objective_gap=_optional_float(payload.get("objective_gap"), "objective_gap"),
        residuals=residual_from_payload(_mapping(payload.get("residuals"), "residuals")),
        iterations=payload.get("iterations"),
        elapsed_seconds=float.fromhex(_string(payload.get("elapsed_seconds"), "elapsed_seconds")),
        raw_statuses=tuple(statuses),
        fallback_used=_boolean(payload.get("fallback_used"), "fallback_used"),
        fallback_reason=_optional_string(payload.get("fallback_reason"), "fallback_reason"),
        prepared=_boolean(payload.get("prepared"), "prepared"),
        template_sha256=_optional_string(payload.get("template_sha256"), "template_sha256"),
        setup_seconds=_optional_float(payload.get("setup_seconds"), "setup_seconds"),
        cold_seconds=_optional_float(payload.get("cold_seconds"), "cold_seconds"),
    )


def production_to_payload(result: ProductionResult) -> dict[str, Any]:
    return {
        "schema": "production-result-v1",
        "status": result.status.value,
        "input_sha256": result.input_sha256,
        "request_sha256": result.request_sha256,
        "result_sha256": result.result_sha256,
        "objective_gap_sha256": result.objective_gap_sha256,
        "objective_gap": _optional_hex(result.objective_gap),
        "selected_lane": result.selected_lane,
        "fallback_used": result.fallback_used,
        "fallback_reason": result.fallback_reason,
        "fast": lane_to_payload(result.fast),
        "exact": lane_to_payload(result.exact),
        "prepared": result.prepared,
        "production_template_sha256": result.production_template_sha256,
        "exact_template_sha256": result.exact_template_sha256,
        "prepared_setup_seconds": _optional_hex(result.prepared_setup_seconds),
    }


def production_from_payload(payload: dict[str, Any]) -> ProductionResult:
    if payload.get("schema") != "production-result-v1":
        raise ValueError("unsupported durable production result schema")
    return ProductionResult(
        status=SolveStatus(_string(payload.get("status"), "status")),
        input_sha256=_string(payload.get("input_sha256"), "input_sha256"),
        request_sha256=_string(payload.get("request_sha256"), "request_sha256"),
        result_sha256=_string(payload.get("result_sha256"), "result_sha256"),
        objective_gap_sha256=_string(payload.get("objective_gap_sha256"), "objective_gap_sha256"),
        objective_gap=_optional_float(payload.get("objective_gap"), "objective_gap"),
        selected_lane=_string(payload.get("selected_lane"), "selected_lane"),
        fallback_used=_boolean(payload.get("fallback_used"), "fallback_used"),
        fallback_reason=_optional_string(payload.get("fallback_reason"), "fallback_reason"),
        fast=lane_from_payload(_mapping(payload.get("fast"), "fast")),
        exact=lane_from_payload(_mapping(payload.get("exact"), "exact")),
        prepared=_boolean(payload.get("prepared"), "prepared"),
        production_template_sha256=_optional_string(payload.get("production_template_sha256"), "production_template_sha256"),
        exact_template_sha256=_optional_string(payload.get("exact_template_sha256"), "exact_template_sha256"),
        prepared_setup_seconds=_optional_float(payload.get("prepared_setup_seconds"), "prepared_setup_seconds"),
    )


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"durable shadow {name} must be an object")
    return value


def _string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"durable shadow {name} must be a string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"durable shadow {name} must be boolean")
    return value


def _tensor(payload: dict[str, Any], name: str) -> torch.Tensor:
    value = tensor_from_payload(payload.get(name))
    assert value is not None
    return value


def _optional_hex(value: float | None) -> str | None:
    return None if value is None else float(value).hex()


def _optional_float(value: object, name: str) -> float | None:
    return None if value is None else float.fromhex(_string(value, name))
