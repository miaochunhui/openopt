from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import math
from threading import BoundedSemaphore, Lock
import time
import uuid

import torch

from .constraints import LinearConstraints
from .certification_policy import CertificationPolicyDecision
from .facade import (
    FallbackPolicy,
    LaneResult,
    ProductionRequest,
    ProductionResult,
    SolveStatus,
    certify_exact,
    input_sha256,
    solve_production,
)
from .model import FactorRiskModel, PortfolioProblem


class ShadowQueueFullError(RuntimeError):
    pass


class ShadowDecision(str, Enum):
    MATCH = "match"
    DIVERGED = "diverged"


@dataclass(frozen=True)
class ShadowTicket:
    ticket_sha256: str
    ticket_nonce: str
    service_epoch: str
    sequence: int
    attempt: int
    policy_sha256: str
    input_sha256: str
    primary_request_sha256: str
    shadow_request_sha256: str


@dataclass(frozen=True)
class OnlineShadowSubmission:
    ticket: ShadowTicket | None
    online: ProductionResult
    online_elapsed_seconds: float
    policy_decision: CertificationPolicyDecision | None = None


@dataclass(frozen=True)
class ShadowCertificationResult:
    ticket: ShadowTicket
    decision: ShadowDecision
    passed: bool
    primary_exact_result_sha256: str
    shadow_exact_result_sha256: str
    shadow: LaneResult
    objective_relative_delta: float
    weights_max_abs_diff: float
    shadow_queue_wait_seconds: float
    shadow_elapsed_seconds: float
    certification_age_seconds: float
    total_elapsed_seconds: float
    result_sha256: str
    execution_attempt: int = 1
    fence_token: int = 1
    durable: bool = False


@dataclass
class _ShadowRecord:
    ticket: ShadowTicket
    future: Future[ShadowCertificationResult]
    observed: bool = False


def _canonical_sha256(payload: dict[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clone_tensor(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is None:
        return None
    return tensor.detach().to("cpu").clone()


def _is_worker_exception(future: Future[object], error: BaseException) -> bool:
    return (
        future.done()
        and not future.cancelled()
        and future.exception(timeout=0.0) is error
    )


def _freeze_request(request: ProductionRequest, *, exact_backend: str) -> ProductionRequest:
    before = input_sha256(request.problem, request.constraints)
    problem = PortfolioProblem(
        alpha=_clone_tensor(request.problem.alpha),
        risk=FactorRiskModel(
            exposures=_clone_tensor(request.problem.risk.exposures),
            factor_variance=_clone_tensor(request.problem.risk.factor_variance),
            specific_variance=_clone_tensor(request.problem.risk.specific_variance),
        ),
        lower=_clone_tensor(request.problem.lower),
        upper=_clone_tensor(request.problem.upper),
        budget=request.problem.budget,
        risk_aversion=request.problem.risk_aversion,
        current=_clone_tensor(request.problem.current),
        trade_cost=request.problem.trade_cost,
    )
    constraints = LinearConstraints(
        A_eq=_clone_tensor(request.constraints.A_eq),
        b_eq=_clone_tensor(request.constraints.b_eq),
        A_ineq=_clone_tensor(request.constraints.A_ineq),
        lower=_clone_tensor(request.constraints.lower),
        upper=_clone_tensor(request.constraints.upper),
    )
    after = input_sha256(request.problem, request.constraints)
    frozen = ProductionRequest(
        problem=problem,
        constraints=constraints,
        fast_device=request.fast_device,
        exact_backend=exact_backend,
        fallback_policy=request.fallback_policy,
        fast_config=request.fast_config,
        residual_tolerance=request.residual_tolerance,
        objective_gap_tolerance=request.objective_gap_tolerance,
    )
    frozen_hash = input_sha256(frozen.problem, frozen.constraints)
    if before != after or before != frozen_hash:
        raise RuntimeError("production request changed while creating shadow snapshot")
    return frozen


class ShadowCertificationService:
    def __init__(
        self,
        *,
        max_pending: int = 1,
        objective_tolerance: float = 1e-6,
        weights_tolerance: float = 1e-5,
        service_epoch: str | None = None,
    ) -> None:
        if not isinstance(max_pending, int) or isinstance(max_pending, bool) or max_pending <= 0:
            raise ValueError("max_pending must be a positive integer")
        for name, value in (
            ("objective_tolerance", objective_tolerance),
            ("weights_tolerance", weights_tolerance),
        ):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        self._objective_tolerance = objective_tolerance
        self._weights_tolerance = weights_tolerance
        self._service_epoch = uuid.uuid4().hex if service_epoch is None else service_epoch
        if (
            len(self._service_epoch) != 32
            or any(character not in "0123456789abcdef" for character in self._service_epoch)
        ):
            raise ValueError("service_epoch must be 32 lowercase hexadecimal characters")
        self._policy_sha256 = _canonical_sha256(
            {
                "policy_version": "shadow-certification-v1",
                "comparison_rule": "exact-objective-relative-and-weights-max-abs-v1",
                "primary_backend": "osqp",
                "shadow_backend": "clarabel",
                "objective_tolerance": float(objective_tolerance).hex(),
                "weights_tolerance": float(weights_tolerance).hex(),
            }
        )
        self._capacity = BoundedSemaphore(max_pending)
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="openpo-clarabel-shadow",
        )
        self._lock = Lock()
        self._closed = False
        self._sequence = 0
        self._records: dict[str, _ShadowRecord] = {}
        self._ack_tombstone_limit = max_pending
        self._acked: OrderedDict[str, ShadowTicket] = OrderedDict()

    def __enter__(self) -> ShadowCertificationService:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.shutdown(wait=True, require_collected=exc_type is None)

    @property
    def uncollected_count(self) -> int:
        with self._lock:
            return sum(not record.observed for record in self._records.values())

    @property
    def outstanding_count(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def inflight_count(self) -> int:
        with self._lock:
            return sum(not record.future.done() for record in self._records.values())

    @property
    def service_epoch(self) -> str:
        return self._service_epoch

    @property
    def policy_sha256(self) -> str:
        return self._policy_sha256

    def submit(
        self,
        request: ProductionRequest,
        *,
        attempt: int = 1,
    ) -> OnlineShadowSubmission:
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt <= 0:
            raise ValueError("attempt must be a positive integer")
        if request.exact_backend != "osqp":
            raise ValueError("shadow service requires exact_backend='osqp' for the online primary")
        if request.fallback_policy is not FallbackPolicy.DENY:
            raise ValueError("shadow service requires fallback_policy=DENY")
        with self._lock:
            if self._closed:
                raise RuntimeError("shadow service is closed")
        if not self._capacity.acquire(blocking=False):
            raise ShadowQueueFullError("Clarabel shadow queue is full")
        try:
            primary_request = _freeze_request(request, exact_backend="osqp")
            shadow_request = _freeze_request(request, exact_backend="clarabel")
            started = time.perf_counter()
            online = solve_production(primary_request)
            online_elapsed = time.perf_counter() - started
            online_completed_at = time.perf_counter()
            if online.status is not SolveStatus.SOLVED_TO_TOLERANCE:
                raise RuntimeError(
                    f"online OSQP production certification failed with {online.status.value}"
                )
            if online.exact.status is not SolveStatus.OPTIMAL:
                raise RuntimeError(
                    f"online OSQP exact lane failed with {online.exact.status.value}"
                )
            with self._lock:
                if self._closed:
                    raise RuntimeError("shadow service closed during submission")
                sequence = self._sequence
                self._sequence += 1
                ticket_nonce = uuid.uuid4().hex
                ticket = ShadowTicket(
                    ticket_sha256=_canonical_sha256(
                        {
                            "ticket_nonce": ticket_nonce,
                            "service_epoch": self._service_epoch,
                            "sequence": sequence,
                            "attempt": attempt,
                            "policy_sha256": self._policy_sha256,
                            "input_sha256": online.input_sha256,
                            "primary_request_sha256": online.request_sha256,
                            "shadow_request_sha256": _request_hash(shadow_request),
                        }
                    ),
                    ticket_nonce=ticket_nonce,
                    service_epoch=self._service_epoch,
                    sequence=sequence,
                    attempt=attempt,
                    policy_sha256=self._policy_sha256,
                    input_sha256=online.input_sha256,
                    primary_request_sha256=online.request_sha256,
                    shadow_request_sha256=_request_hash(shadow_request),
                )
                shadow_enqueued_at = time.perf_counter()
                future = self._executor.submit(
                    self._run_shadow,
                    ticket,
                    shadow_request,
                    online.exact,
                    started,
                    online_completed_at,
                    shadow_enqueued_at,
                )
                self._records[ticket.ticket_sha256] = _ShadowRecord(
                    ticket=ticket,
                    future=future,
                )
            return OnlineShadowSubmission(
                ticket=ticket,
                online=online,
                online_elapsed_seconds=online_elapsed,
            )
        except BaseException:
            self._capacity.release()
            raise

    def collect(
        self,
        ticket: ShadowTicket,
        *,
        timeout: float | None = None,
    ) -> ShadowCertificationResult:
        if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
            raise ValueError("timeout must be finite and nonnegative")
        with self._lock:
            record = self._records.get(ticket.ticket_sha256)
            if record is None or record.ticket != ticket:
                raise KeyError(f"unknown or forged shadow ticket {ticket.ticket_sha256}")
        try:
            result = record.future.result(timeout=timeout)
        except FutureTimeoutError as error:
            if _is_worker_exception(record.future, error):
                with self._lock:
                    record.observed = True
                raise
            if not record.future.done():
                raise TimeoutError(
                    f"shadow certification timed out for ticket {ticket.ticket_sha256}"
                ) from error
            try:
                result = record.future.result(timeout=0.0)
            except BaseException as terminal_error:
                if _is_worker_exception(record.future, terminal_error):
                    with self._lock:
                        record.observed = True
                raise
        except BaseException as error:
            if _is_worker_exception(record.future, error):
                with self._lock:
                    record.observed = True
            raise
        if result.ticket != ticket:
            raise RuntimeError("shadow result ticket does not match registered ticket")
        with self._lock:
            record.observed = True
        return result

    def is_ready(self, ticket: ShadowTicket) -> bool:
        with self._lock:
            record = self._records.get(ticket.ticket_sha256)
            if record is None or record.ticket != ticket:
                raise KeyError(f"unknown or forged shadow ticket {ticket.ticket_sha256}")
            return record.future.done()

    def ack(self, ticket: ShadowTicket) -> bool:
        with self._lock:
            acknowledged_ticket = self._acked.get(ticket.ticket_sha256)
            if acknowledged_ticket is not None:
                if acknowledged_ticket != ticket:
                    raise KeyError(f"unknown or forged shadow ticket {ticket.ticket_sha256}")
                self._acked.move_to_end(ticket.ticket_sha256)
                return False
            record = self._records.get(ticket.ticket_sha256)
            if record is None or record.ticket != ticket:
                raise KeyError(f"unknown or forged shadow ticket {ticket.ticket_sha256}")
            if not record.observed:
                raise RuntimeError("shadow ticket must be collected before ack")
            if not record.future.done():
                raise RuntimeError("shadow ticket cannot be acked before completion")
            del self._records[ticket.ticket_sha256]
            self._acked[ticket.ticket_sha256] = ticket
            if len(self._acked) > self._ack_tombstone_limit:
                self._acked.popitem(last=False)
        self._capacity.release()
        return True

    def shutdown(self, *, wait: bool = True, require_collected: bool = True) -> None:
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=wait, cancel_futures=False)
        if require_collected:
            with self._lock:
                outstanding = len(self._records)
            if outstanding:
                raise RuntimeError(
                    f"shadow service closed with {outstanding} unacknowledged ticket(s)"
                )

    def _run_shadow(
        self,
        ticket: ShadowTicket,
        request: ProductionRequest,
        primary_exact: LaneResult,
        submitted_at: float,
        online_completed_at: float,
        shadow_enqueued_at: float,
    ) -> ShadowCertificationResult:
        started = time.perf_counter()
        queue_wait = started - shadow_enqueued_at
        shadow = certify_exact(request)
        completed_at = time.perf_counter()
        shadow_elapsed = completed_at - started
        if shadow.input_sha256 != ticket.input_sha256:
            raise RuntimeError("Clarabel shadow used a different frozen input")
        if shadow.request_sha256 != ticket.shadow_request_sha256:
            raise RuntimeError("Clarabel shadow request hash does not match its ticket")
        denominator = primary_exact.objective.abs().clamp_min(1e-12)
        objective_delta = float(
            ((shadow.objective - primary_exact.objective).abs() / denominator).amax().item()
        )
        weights_delta = float(
            (shadow.weights - primary_exact.weights).abs().amax().item()
        )
        passed = (
            shadow.status is SolveStatus.OPTIMAL
            and shadow.backend == "clarabel"
            and shadow.residuals.primal_residual <= request.residual_tolerance
            and objective_delta <= self._objective_tolerance
            and weights_delta <= self._weights_tolerance
        )
        decision = ShadowDecision.MATCH if passed else ShadowDecision.DIVERGED
        result_hash = _canonical_sha256(
            {
                "ticket_sha256": ticket.ticket_sha256,
                "policy_sha256": ticket.policy_sha256,
                "decision": decision.value,
                "primary_exact_result_sha256": primary_exact.result_sha256,
                "shadow_exact_result_sha256": shadow.result_sha256,
                "objective_relative_delta": float(objective_delta).hex(),
                "weights_max_abs_diff": float(weights_delta).hex(),
            }
        )
        return ShadowCertificationResult(
            ticket=ticket,
            decision=decision,
            passed=passed,
            primary_exact_result_sha256=primary_exact.result_sha256,
            shadow_exact_result_sha256=shadow.result_sha256,
            shadow=shadow,
            objective_relative_delta=objective_delta,
            weights_max_abs_diff=weights_delta,
            shadow_queue_wait_seconds=queue_wait,
            shadow_elapsed_seconds=shadow_elapsed,
            certification_age_seconds=completed_at - online_completed_at,
            total_elapsed_seconds=completed_at - submitted_at,
            result_sha256=result_hash,
        )

def _request_hash(request: ProductionRequest) -> str:
    from .facade import _request_sha256

    return _request_sha256(request)
