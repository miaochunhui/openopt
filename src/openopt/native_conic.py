"""Opt-in dense PyTorch proximal ADMM for zero/nonnegative/SOC/RSOC conic QP.

min .5*x.T*P*x + q.T*x, subject to A*x + s = b, s in K.
No third-party solving, automatic routing, infeasibility certificates or HSD.
Only the independent original-coordinate CPU checker can accept a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from .certificates import NumericalEvidence
from .conic import (
    ConicProblem, NonnegativeCone, RotatedSecondOrderCone, SecondOrderCone,
    ZeroCone, check_conic_candidate,
)

__all__ = [
    "NativeConicConfig", "NativeConicResult", "PreparedNativeConic",
    "project_cones", "prepare_native_conic", "solve_native_conic",
]

Cone = ZeroCone | NonnegativeCone | SecondOrderCone | RotatedSecondOrderCone


def _finite(value: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"native conic nonfinite {context}")


def _norm(value: torch.Tensor) -> float:
    return float(value.abs().max()) if value.numel() else 0.0


def _scalar(value: float, context: str) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"native conic nonfinite {context}")
    return value


def _cones(cones: object, rows: int) -> tuple[Cone, ...]:
    if not isinstance(cones, (tuple, list)):
        raise TypeError("cones must be an ordered tuple or list")
    result = tuple(cones)
    if any(type(cone) not in (
        ZeroCone, NonnegativeCone, SecondOrderCone, RotatedSecondOrderCone,
    ) for cone in result):
        raise TypeError("native conic supports Zero, Nonnegative, SOC and RSOC only")
    if sum(cone.dimension for cone in result) != rows:
        raise ValueError("sum of cone dimensions must equal vector/A/b rows")
    return result


def _rotated_projection(block: torch.Tensor) -> None:
    """Scale only the SOC calculation; recover its small boundary head stably."""
    tail = block[2:].clone()
    tail_scale = tail.abs().max()
    has_tail = bool(tail_scale > 0)
    radius_mantissa, radius_exponent = torch.frexp(tail_scale)
    if has_tail:
        normalized_tail_radius = torch.linalg.vector_norm(tail / tail_scale)
        radius_mantissa, norm_exponent = torch.frexp(
            radius_mantissa * normalized_tail_radius,
        )
        radius_exponent = radius_exponent + norm_exponent
    _finite(radius_mantissa, "RSOC tail radius mantissa")
    first, second = block[0].clone(), block[1].clone()

    def restore_exponent(mantissa, exponent):
        # Some torch runtimes materialize 2**exponent before multiplication.
        limit = 1023 if block.dtype == torch.float64 else 127
        floor = -1022 if block.dtype == torch.float64 else -126
        initial = exponent.clamp(min=floor, max=limit)
        return torch.ldexp(torch.ldexp(mantissa, initial), exponent - initial)

    def inside(a, b):
        if bool((a < 0) | (b < 0)):
            return False
        if not has_tail:
            return True
        if not bool((a > 0) & (b > 0)):
            return False
        a_mantissa, a_exponent = torch.frexp(a)
        b_mantissa, b_exponent = torch.frexp(b)
        sum_exponent = a_exponent + b_exponent
        half_exponent = torch.div(sum_exponent, 2, rounding_mode="floor")
        product_root = torch.sqrt(a_mantissa * b_mantissa)
        if bool(sum_exponent - 2 * half_exponent):
            product_root = product_root * math.sqrt(2.0)
        ratio = restore_exponent(
            radius_mantissa / product_root, radius_exponent - half_exponent,
        )
        return bool(ratio <= math.sqrt(2.0))

    if inside(first, second):
        return
    if inside(-first, -second):
        block.zero_()
        return
    scale = block.abs().max()
    normalized = block / scale
    inverse_root_two = 1.0 / math.sqrt(2.0)
    head = inverse_root_two * normalized[0] + inverse_root_two * normalized[1]
    difference = inverse_root_two * normalized[0] - inverse_root_two * normalized[1]
    transformed_tail = torch.cat((difference.reshape(1), normalized[2:]))
    transformed_radius = torch.linalg.vector_norm(transformed_tail)
    _finite(transformed_radius, "RSOC scaled radius")
    if not bool(transformed_radius > 0):
        raise RuntimeError("native conic unresolved RSOC projection radius")
    if bool(head < 0):
        # Rationalize R+head; the direct half-sum erases near-polar solutions.
        normalized_radius = torch.linalg.vector_norm(normalized[2:])
        numerator = normalized_radius ** 2 - 2 * normalized[0] * normalized[1]
        weight = numerator / (
            2 * transformed_radius * (transformed_radius - head)
        )
    else:
        weight = 0.5 + 0.5 * head / transformed_radius
    if not bool(weight > 0):
        raise RuntimeError("native conic unresolved RSOC projection weight")
    projected_head = weight * transformed_radius
    large = (inverse_root_two * projected_head
             + inverse_root_two * torch.abs(weight * difference)) * scale
    _finite(large, "RSOC projected large head")
    projected_tail = weight * tail
    small = torch.zeros_like(large)
    if has_tail and bool(weight > 0):
        if not bool(large > 0):
            raise RuntimeError("native conic unresolved RSOC projected head")
        weight_mantissa, weight_exponent = torch.frexp(weight)
        large_mantissa, large_exponent = torch.frexp(large)
        small = restore_exponent(
            (weight_mantissa * radius_mantissa) ** 2 / large_mantissa,
            2 * weight_exponent + 2 * radius_exponent - large_exponent - 1,
        )
    if bool(first >= second):
        block[0], block[1] = large, small
    else:
        block[0], block[1] = small, large
    block[2:] = projected_tail
    _finite(block, "RSOC projection output")


@torch.no_grad()
def project_cones(
    value: torch.Tensor, cones: tuple[Cone, ...], *, dual: bool = False,
) -> torch.Tensor:
    """Euclidean projection onto the ordered product K (or K*).

    Zero's dual is unrestricted. Nonnegative, SOC and RSOC are self-dual.
    RSOC uses an orthogonal SOC transform, preserving Euclidean projection.
    """
    if not isinstance(value, torch.Tensor):
        raise TypeError("cone projection requires a tensor")
    if (value.layout != torch.strided or value.dtype not in (torch.float32, torch.float64)
            or value.ndim != 1 or value.device.type not in ("cpu", "cuda")):
        raise ValueError("projection requires a dense CPU/CUDA float32/64 vector")
    if not isinstance(dual, bool):
        raise ValueError("dual must be a boolean")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("cone projection input must be finite")
    cones = _cones(cones, value.numel())
    result = value.detach().clone()
    offset = 0
    for cone in cones:
        block = result[offset:offset + cone.dimension]
        offset += cone.dimension
        if isinstance(cone, ZeroCone):
            if not dual:
                block.zero_()
        elif isinstance(cone, NonnegativeCone):
            block.clamp_(min=0)
        elif isinstance(cone, RotatedSecondOrderCone):
            _rotated_projection(block)
        else:
            t, tail = block[0].clone(), block[1:].clone()
            scale = tail.abs().max()
            radius = scale * torch.linalg.vector_norm(tail / scale) if bool(scale > 0) else scale
            _finite(radius, "SOC radius")
            if bool(radius <= t):
                pass
            elif bool(radius <= -t):
                block.zero_()
            else:
                block[0] = 0.5 * t + 0.5 * radius
                block[1:] = (0.5 + 0.5 * (t / radius)) * tail
    _finite(result, "cone projection output")
    return result


@dataclass(frozen=True)
class NativeConicConfig:
    enabled: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    rho: float = 1.0
    sigma: float = 1e-6
    max_iterations: int = 10000
    check_interval: int = 25
    eps_abs: float = 1e-7
    eps_rel: float = 1e-7

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean")
        if not self.enabled:
            raise ValueError("native conic is opt-in; enabled=True is required")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("native conic supports float32/float64 only")
        for name in ("rho", "sigma", "eps_abs", "eps_rel"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be a real number")
            if not math.isfinite(value) or (value < 0 if name == "eps_rel" else value <= 0):
                raise ValueError(f"{name} must be finite and {'nonnegative' if name == 'eps_rel' else 'positive'}")
        if self.eps_rel >= 1:
            raise ValueError("eps_rel must be less than one")
        for name in ("rho", "sigma"):
            represented = torch.tensor(getattr(self, name), dtype=self.dtype)
            if not bool(torch.isfinite(represented)) or not bool(represented > 0):
                raise ValueError(f"{name} must be positive and finite in the selected dtype")
        for name in ("max_iterations", "check_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda") or (device.type == "cpu" and device.index is not None):
            raise ValueError("native conic supports device='cpu' or an explicit CUDA device")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("explicit CUDA requested but unavailable; no CPU fallback")
            if device.index is not None and device.index >= torch.cuda.device_count():
                raise ValueError("CUDA device index out of range")


@dataclass(frozen=True)
class NativeConicResult:
    x: torch.Tensor
    s: torch.Tensor
    z: torch.Tensor
    status: str
    iterations: int
    objective: float
    evidence: NumericalEvidence
    device: str
    dtype: torch.dtype
    backend: str = "native_torch_proximal_admm"
    certified: bool = False
    qualification: str = "original-coordinate approximate numerical checks; not an exact proof"


def _host_tensor(value: object, name: str) -> torch.Tensor:
    try:
        tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
    except (TypeError, ValueError, RuntimeError) as error:
        raise ValueError(f"{name} must be real numeric dense data") from error
    if (tensor.layout != torch.strided or tensor.is_complex() or tensor.dtype == torch.bool
            or tensor.device.type not in ("cpu", "cuda")):
        raise ValueError(f"{name} must be real dense CPU/CUDA data, not boolean")
    # Python floats must not pass through PyTorch's default float32 dtype.
    if not isinstance(value, torch.Tensor):
        tensor = torch.as_tensor(value, dtype=torch.float64)
    result = tensor.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must be finite")
    return result


class PreparedNativeConic:
    """Frozen original model and a reusable device factor; each solve resets state.

    x+ solves (P + rho*A.T*A + sigma*I)x+ =
    sigma*x - q + A.T*(rho*(b-s) - z).
    s+ = projection_K(b - A*x+ - z/rho);
    z+ = z + rho*(A*x+ + s+ - b).
    sigma regularizes the iteration, never the model or independent dual gap.
    """

    @torch.no_grad()
    def __init__(
        self, problem: ConicProblem, config: NativeConicConfig = NativeConicConfig(),
    ):
        if not isinstance(config, NativeConicConfig):
            raise TypeError("config must be a NativeConicConfig")
        config.validate()
        if not isinstance(problem, ConicProblem):
            raise TypeError("problem must be a ConicProblem")
        P, q, A, b = [_host_tensor(getattr(problem, name), name) for name in ("P", "q", "A", "b")]
        if q.ndim != 1 or not q.numel():
            raise ValueError("q must be a nonempty vector")
        n = q.numel()
        if P.shape != (n, n) or A.ndim != 2 or A.shape[1] != n:
            raise ValueError("P/A dimensions must match q")
        if b.shape != (A.shape[0],):
            raise ValueError("b must match A rows")
        cones = _cones(problem.cones, b.numel())
        difference = P - P.T
        _finite(difference, "original symmetry difference")
        symmetry = _scalar(64 * torch.finfo(torch.float64).eps * n * _norm(P), "symmetry threshold")
        if _norm(difference) > symmetry:
            raise ValueError("P must be a full symmetric matrix, not upper-triangular storage")
        symmetric_P = 0.5 * P + 0.5 * P.T
        eigenvalues = torch.linalg.eigvalsh(symmetric_P)
        _finite(eigenvalues, "original PSD spectrum")
        threshold = _scalar(
            64 * torch.finfo(torch.float64).eps * n * _norm(eigenvalues), "PSD threshold",
        )
        if float(eigenvalues[0]) < -threshold:
            raise ValueError("P must be positive semidefinite")
        self.config = config
        self._original = ConicProblem(P, q, A, b, cones)
        self._P, self._q, self._A, self._b = [
            tensor.to(device=config.device, dtype=config.dtype).clone()
            for tensor in (symmetric_P, q, A, b)
        ]
        self.cones = cones
        for name, original in zip(("P", "q", "A", "b"), (symmetric_P, q, A, b)):
            cast = getattr(self, f"_{name}")
            _finite(cast, f"device-cast {name}")
            if bool(((original != 0).to(cast.device) & (cast == 0)).any()):
                raise ValueError(f"native conic device-cast {name} underflow lost nonzero data")
        system = self._P + config.rho * (self._A.T @ self._A)
        system.diagonal().add_(config.sigma)
        _finite(system, "linear system")
        try:
            factor, info = torch.linalg.cholesky_ex(system)
        except RuntimeError as error:
            raise RuntimeError("native conic linear system factorization failed") from error
        if int(info) != 0:
            raise RuntimeError(f"native conic linear system factorization failed info={int(info)}")
        _finite(factor, "linear system factor")
        self._factor = factor

    @property
    def original_problem(self) -> ConicProblem:
        """Return a copy, not an alias into the immutable replay model."""
        return ConicProblem(*[
            getattr(self._original, name).clone() for name in ("P", "q", "A", "b")
        ], self._original.cones)

    def _check(
        self, x: torch.Tensor, s: torch.Tensor, z: torch.Tensor, iteration: int,
    ) -> NumericalEvidence:
        host = [value.detach().to(device="cpu", dtype=torch.float64) for value in (x, s, z)]
        try:
            return check_conic_candidate(
                self._original, *host, atol=self.config.eps_abs, rtol=self.config.eps_rel,
            )
        except (ValueError, RuntimeError, ArithmeticError) as error:
            raise RuntimeError(
                f"native conic independent candidate check failed iteration={iteration}",
            ) from error

    @torch.no_grad()
    def solve(self) -> NativeConicResult:
        c = self.config
        P, q, A, b = self._P, self._q, self._A, self._b
        x = torch.zeros_like(q)
        s = project_cones(b, self.cones)
        z = torch.zeros_like(b)
        status = "max_iterations"
        evidence = None
        for iteration in range(1, c.max_iterations + 1):
            rhs = c.sigma * x - q + A.T @ (c.rho * (b - s) - z)
            _finite(rhs, f"rhs iteration={iteration}")
            x = torch.cholesky_solve(rhs[:, None], self._factor).squeeze(1)
            Ax = A @ x
            argument = b - Ax - z / c.rho
            for name, value in (("x", x), ("Ax", Ax), ("projection argument", argument)):
                _finite(value, f"{name} iteration={iteration}")
            s = project_cones(argument, self.cones)
            z = z + c.rho * (Ax + s - b)
            _finite(z, f"z iteration={iteration}")
            if iteration % c.check_interval == 0 or iteration == c.max_iterations:
                evidence = self._check(x, s, z, iteration)
                if evidence.passed:
                    status = "solved"
                    break
        assert evidence is not None  # The last iteration always invokes the check.
        objective = evidence.metrics["primal_objective"]
        if objective is None:
            raise RuntimeError("native conic independent objective is unavailable")
        _scalar(objective, f"original objective iteration={iteration}")
        return NativeConicResult(
            x=x, s=s, z=z, status=status, iterations=iteration,
            objective=objective, evidence=evidence, device=str(x.device), dtype=x.dtype,
        )


def prepare_native_conic(
    problem: ConicProblem, config: NativeConicConfig = NativeConicConfig(),
) -> PreparedNativeConic:
    return PreparedNativeConic(problem, config)


def solve_native_conic(
    problem: ConicProblem, config: NativeConicConfig = NativeConicConfig(),
) -> NativeConicResult:
    return prepare_native_conic(problem, config).solve()
