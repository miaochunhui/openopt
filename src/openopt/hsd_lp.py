"""Experimental, explicitly enabled dense homogeneous self-dual LP splitting.

The original model is min q.T*x, A*x+s=b, s in K, with P exactly zero.
Only zero, nonnegative and Lorentz cones, binary64, and small finite dense
models are supported. There is no solver fallback, model regularization,
scaling, warm start, sparse path, exact certificate or commercial guarantee.
Device LU/DR computation and explicit host original-model auditing are separate.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

from .conic import ConicProblem, NonnegativeCone, SecondOrderCone, ZeroCone
from .native_conic import project_cones

__all__ = [
    "HSDLPConfig", "HSDLPCheck", "HSDLPResult", "PreparedHSDLP",
    "check_hsd_candidate", "check_hsd_primal_ray", "check_hsd_recession_ray",
    "prepare_hsd_lp", "solve_hsd_lp",
]

Cone = ZeroCone | NonnegativeCone | SecondOrderCone
MAX_EMBEDDING_DIMENSION = 256
MAX_ITERATIONS = 10000


def _finite(value: torch.Tensor, context: str) -> None:
    if not bool(torch.isfinite(value).all()):
        raise RuntimeError(f"HSD nonfinite {context}")


def _number(value: object, name: str, *, positive: bool = True) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _integer(value: object, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True)
class HSDLPConfig:
    enabled: bool = False
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    max_iterations: int = 2000
    check_interval: int = 10
    max_embedding_dimension: int = 256
    eps_abs: float = 1e-7
    eps_rel: float = 1e-7
    eps_ray: float = 1e-7
    tau_min: float = 1e-10
    linear_residual_tolerance: float = 1e-12

    def validate(self) -> torch.device:
        if type(self.enabled) is not bool or not self.enabled:
            raise ValueError("HSD LP is opt-in; enabled=True is required")
        if self.dtype != torch.float64:
            raise ValueError("bounded HSD LP supports torch.float64 only")
        if type(self.device) is not str:
            raise ValueError("device must be an explicit CPU/CUDA string")
        device = torch.device(self.device)
        if device.type not in ("cpu", "cuda") or (
            device.type == "cpu" and device.index is not None
        ):
            raise ValueError("HSD supports cpu or an explicit CUDA device only")
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("explicit CUDA unavailable; no CPU fallback")
            index = torch.cuda.current_device() if device.index is None else device.index
            if not 0 <= index < torch.cuda.device_count():
                raise ValueError("CUDA device index out of range")
            device = torch.device("cuda", index)
        _integer(self.max_iterations, "max_iterations", MAX_ITERATIONS)
        _integer(self.check_interval, "check_interval", MAX_ITERATIONS)
        _integer(self.max_embedding_dimension, "max_embedding_dimension", MAX_EMBEDDING_DIMENSION)
        for name in ("eps_abs", "eps_rel", "eps_ray"):
            value = _number(getattr(self, name), name, positive=name != "eps_rel")
            if value > 1e-7:
                raise ValueError(f"{name} cannot loosen the original 1e-7 gate")
        _number(self.tau_min, "tau_min")
        tolerance = _number(self.linear_residual_tolerance, "linear_residual_tolerance")
        if tolerance > 1e-10:
            raise ValueError("linear_residual_tolerance must be <= 1e-10")
        return device


@dataclass(frozen=True)
class HSDLPCheck:
    kind: str
    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float]
    thresholds: dict[str, float]
    qualification: str = "bounded original-coordinate numerical evidence, not exact proof"


@dataclass(frozen=True)
class HSDLPResult:
    status: str
    iterations: int
    x: torch.Tensor | None
    s: torch.Tensor | None
    y: torch.Tensor | None
    primal_ray: torch.Tensor | None
    recession_ray: torch.Tensor | None
    feasible_base: torch.Tensor | None
    candidate_check: HSDLPCheck | None
    primal_ray_check: HSDLPCheck | None
    recession_ray_check: HSDLPCheck | None
    base_check: HSDLPCheck | None
    u: torch.Tensor
    v: torch.Tensor
    w: torch.Tensor
    trace: tuple[dict[str, object], ...]
    model_sha256: str
    source_sha256: dict[str, str]
    settings: dict[str, object]
    runtime: dict[str, object]
    device: str
    linear_solve_calls: int
    backend: str = "experimental_native_dense_hsd_lp_v1"
    certified: bool = False
    qualification: str = "numerical candidates/rays only; no GPU or commercial qualification"


def _array(value: object, name: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError(f"{name}: audit requires an explicit host copy, not implicit transfer")
        if value.layout != torch.strided or value.is_complex() or value.dtype == torch.bool:
            raise ValueError(f"{name} must be real dense numeric data")
        value = value.detach().to(torch.float64).numpy()
    array = np.asarray(value)
    if array.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be real dense numeric data")
    with np.errstate(over="raise", under="raise", invalid="raise"):
        try:
            result = np.array(array, dtype=np.float64, copy=True)
        except (FloatingPointError, OverflowError) as error:
            raise ValueError(f"{name}: binary64 conversion failed") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _model(problem: ConicProblem, cap: int = MAX_EMBEDDING_DIMENSION) -> ConicProblem:
    if not isinstance(problem, ConicProblem):
        raise TypeError("problem must be a ConicProblem")
    # Check shapes before copying dense data or constructing any factor.
    shapes = {name: np.shape(getattr(problem, name)) for name in ("P", "q", "A", "b")}
    if len(shapes["q"]) != 1 or not shapes["q"][0]:
        raise ValueError("q must be a nonempty vector")
    n = shapes["q"][0]
    if len(shapes["A"]) != 2 or shapes["A"][1] != n:
        raise ValueError("A columns must match q")
    m = shapes["A"][0]
    if n + m + 1 > cap:
        raise ValueError(f"dense embedding dimension exceeds bounded cap={cap}")
    if shapes["P"] != (n, n) or shapes["b"] != (m,):
        raise ValueError("P/b dimensions must match q/A")
    if not isinstance(problem.cones, (tuple, list)):
        raise TypeError("cones must be an ordered tuple/list")
    cones = tuple(problem.cones)
    if any(type(cone) not in (ZeroCone, NonnegativeCone, SecondOrderCone) for cone in cones):
        raise TypeError("HSD LP supports zero/nonnegative/SOC only; no RSOC/generic cones")
    if sum(cone.dimension for cone in cones) != m:
        raise ValueError("cone dimensions must match A rows")
    arrays = [_array(getattr(problem, name), name) for name in ("P", "q", "A", "b")]
    if np.any(arrays[0] != 0):
        raise ValueError("HSD LP requires P exactly zero; no QP/PSD approximation")
    for array in arrays:
        array.setflags(write=False)
    return ConicProblem(*arrays, cones)


def _hash_model(problem: ConicProblem) -> str:
    digest = hashlib.sha256()
    for name in ("P", "q", "A", "b"):
        array = getattr(problem, name)
        digest.update(json.dumps([name, list(array.shape), "<f8"], separators=(",", ":")).encode())
        digest.update(np.asarray(array, dtype="<f8").tobytes(order="C"))
    digest.update(json.dumps(
        [[type(cone).__name__, cone.dimension] for cone in problem.cones],
        separators=(",", ":"),
    ).encode())
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    paths = {
        "openopt.hsd_lp": Path(__file__).resolve(),
        "openopt.conic": Path(sys.modules[ConicProblem.__module__].__file__).resolve(),
        "openopt.native_conic": Path(sys.modules[project_cones.__module__].__file__).resolve(),
    }
    return {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in paths.items()}


def _f(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _represented(value: Fraction, name: str) -> float:
    try:
        result = float(value)
    except OverflowError as error:
        raise RuntimeError(f"HSD audit overflow: {name}") from error
    if not math.isfinite(result) or (value != 0 and result == 0):
        raise RuntimeError(f"HSD audit nonrepresentable/underflow: {name}")
    return result


def _dot(left: np.ndarray, right: np.ndarray) -> Fraction:
    # Exact binary64 input products and accumulation, independent of torch/BLAS.
    return sum((_f(a) * _f(b) for a, b in zip(left, right)), Fraction(0))


def _matvec(matrix: np.ndarray, value: np.ndarray, name: str) -> np.ndarray:
    return np.array([_represented(_dot(row, value), name) for row in matrix])


def _vector(value: object, size: int, name: str) -> np.ndarray:
    array = _array(value, name)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return array


def _cone(
    value: np.ndarray, cones: tuple[Cone, ...], *, dual: bool,
    strict: bool, eps_abs: float, eps_rel: float,
) -> tuple[bool, dict[str, float], dict[str, float]]:
    passed, metrics, limits = True, {}, {}
    offset = 0
    for index, cone in enumerate(cones):
        block = value[offset:offset + cone.dimension]
        offset += cone.dimension
        if isinstance(cone, ZeroCone):
            violations = np.zeros_like(block) if dual else np.abs(block)
            thresholds = np.zeros_like(block) if strict else np.full_like(block, eps_abs)
            good = bool(np.all(violations <= thresholds))
        elif isinstance(cone, NonnegativeCone):
            violations = np.maximum(-block, 0)
            thresholds = np.zeros_like(block) if strict else eps_abs + eps_rel * np.abs(block)
            good = bool(np.all(violations <= thresholds))
        else:
            radius = math.hypot(*map(float, block[1:]))
            if not math.isfinite(radius):
                raise RuntimeError("HSD audit SOC norm overflow")
            violations = np.array([max(radius - block[0], 0.0)])
            thresholds = np.array([0.0 if strict else eps_abs + eps_rel * max(abs(block[0]), radius)])
            good = (
                block[0] >= 0 and _dot(block[1:], block[1:]) <= _f(block[0]) ** 2
                if strict else bool(np.all(violations <= thresholds))
            )
        for row, (violation, threshold) in enumerate(zip(violations, thresholds)):
            metrics[f"cone_{index}_component_{row}"] = float(violation)
            limits[f"cone_{index}_component_{row}"] = float(threshold)
        passed = passed and good
    return passed, metrics, limits


def _check(kind: str, checks: dict[str, bool], metrics: dict[str, float],
           limits: dict[str, float]) -> HSDLPCheck:
    if not checks:
        raise RuntimeError("empty HSD audit is not acceptance")
    if not all(math.isfinite(value) for value in (*metrics.values(), *limits.values())):
        raise RuntimeError("HSD audit nonfinite metric/threshold")
    checks = {name: bool(value) for name, value in checks.items()}
    return HSDLPCheck(kind, all(checks.values()), checks, metrics, limits)


def _tolerances(eps_abs: float, eps_rel: float) -> None:
    if _number(eps_abs, "eps_abs") > 1e-7 or _number(eps_rel, "eps_rel", positive=False) > 1e-7:
        raise ValueError("audit cannot loosen original 1e-7 gates")


def _rows(residual: np.ndarray, scales: np.ndarray, name: str, eps_abs: float,
          eps_rel: float, metrics: dict, limits: dict) -> bool:
    good = True
    for index, (error, scale) in enumerate(zip(residual, scales)):
        key = f"{name}_{index}"
        threshold = _represented(_f(eps_abs) + _f(eps_rel) * _f(scale), key + " threshold")
        metrics[key], limits[key] = abs(float(error)), threshold
        good = good and abs(error) <= threshold
    return good


def _primal(p: ConicProblem, x: np.ndarray, s: np.ndarray, eps_abs: float,
            eps_rel: float) -> HSDLPCheck:
    Ax = _matvec(p.A, x, "Ax")
    actual = np.array([_represented(_f(b) - _f(a), "original slack") for a, b in zip(Ax, p.b)])
    residual = np.array([_represented(_f(a) + _f(v) - _f(b), "primal equation")
                         for a, v, b in zip(Ax, s, p.b)])
    metrics, limits = {}, {}
    equation = _rows(residual, np.maximum.reduce([np.abs(Ax), np.abs(s), np.abs(p.b)]),
                     "primal_equation", eps_abs, eps_rel, metrics, limits)
    cone_pass = True
    for label, value in (("slack", s), ("original_slack", actual)):
        good, cone_metrics, cone_limits = _cone(
            value, p.cones, dual=False, strict=False, eps_abs=eps_abs, eps_rel=eps_rel,
        )
        metrics.update({f"{label}_{key}": val for key, val in cone_metrics.items()})
        limits.update({f"{label}_{key}": val for key, val in cone_limits.items()})
        cone_pass = cone_pass and good
    return _check("hsd_original_primal", {"primal_equation": equation, "primal_cone": cone_pass},
                  metrics, limits)


def check_hsd_candidate(
    problem: ConicProblem, x: object, s: object, y: object, *,
    eps_abs: float = 1e-7, eps_rel: float = 1e-7,
) -> HSDLPCheck:
    """Six original-model numerical gates; dual cone signs are strict.

    No projection is used. LP stationarity/gap are numerical conditions, not
    a claim that an inexact multiplier defines an exact finite dual bound.
    """
    _tolerances(eps_abs, eps_rel)
    p = _model(problem)
    x, s, y = (_vector(value, size, name) for value, size, name in
               ((x, p.q.size, "x"), (s, p.b.size, "s"), (y, p.b.size, "y")))
    primal = _primal(p, x, s, eps_abs, eps_rel)
    metrics, limits = dict(primal.metrics), dict(primal.thresholds)
    dual, cone_metrics, cone_limits = _cone(
        y, p.cones, dual=True, strict=True, eps_abs=eps_abs, eps_rel=eps_rel,
    )
    metrics.update({f"dual_{key}": val for key, val in cone_metrics.items()})
    limits.update({f"dual_{key}": val for key, val in cone_limits.items()})
    ATy = _matvec(p.A.T, y, "A.T y")
    stationarity = np.array([_represented(_f(a) + _f(q), "stationarity")
                             for a, q in zip(ATy, p.q)])
    stationary = _rows(stationarity, np.maximum(np.abs(ATy), np.abs(p.q)),
                       "stationarity", eps_abs, eps_rel, metrics, limits)
    complementary, offset = True, 0
    for index, cone in enumerate(p.cones):
        block = slice(offset, offset + cone.dimension)
        offset += cone.dimension
        blocks = (
            [(s[j:j + 1], y[j:j + 1]) for j in range(block.start, block.stop)]
            if not isinstance(cone, SecondOrderCone) else [(s[block], y[block])]
        )
        for component, (left, right) in enumerate(blocks):
            value = abs(_dot(left, right))
            scale = sum((abs(_f(a) * _f(b)) for a, b in zip(left, right)), Fraction(0))
            threshold = _f(eps_abs) + _f(eps_rel) * scale
            key = f"complementarity_{index}_{component}"
            metrics[key], limits[key] = _represented(value, key), _represented(threshold, key)
            complementary = complementary and value <= threshold
    primal_value, minus_dual = _dot(p.q, x), _dot(p.b, y)
    gap, threshold = abs(primal_value + minus_dual), (
        _f(eps_abs) + _f(eps_rel) * max(abs(primal_value), abs(minus_dual))
    )
    metrics.update(primal_objective=_represented(primal_value, "objective"),
                   dual_candidate_objective=_represented(-minus_dual, "dual candidate objective"),
                   duality_gap=_represented(gap, "gap"))
    limits["duality_gap"] = _represented(threshold, "gap threshold")
    checks = dict(primal.checks)
    checks.update(dual_cone=dual, stationarity=stationary,
                  complementarity=complementary, duality_gap=gap <= threshold)
    return _check("hsd_original_candidate_six_gates", checks, metrics, limits)


def _ray_limit(p: ConicProblem, normal: np.ndarray, eps_ray: float) -> float:
    _number(eps_ray, "eps_ray")
    if eps_ray > 1e-7:
        raise ValueError("eps_ray cannot loosen 1e-7")
    maximum = float(np.max(np.abs(p.A))) if p.A.size else 0.0
    norm = float(np.max(np.abs(normal))) if normal.size else 0.0
    ratio = min(Fraction(1), _f(maximum) / _f(norm)) if norm > 0 else Fraction(1)
    # A exactly zero has an exact zero residual; no positive tolerance is needed.
    return _represented(_f(eps_ray) * ratio, "ray threshold")


def check_hsd_primal_ray(
    problem: ConicProblem, y: object, *, eps_ray: float = 1e-7,
) -> HSDLPCheck:
    """Normalized numerical Farkas test: A.T y ~ 0, y in K*, b.T y ~ -1.

    Numerical annihilation is not exact proof. Cone membership is exact for
    represented dyadics, so a tiny wrong-sign multiplier is never clipped.
    The data-relative residual limit only tightens the absolute 1e-7 gate.
    """
    p = _model(problem)
    y = _vector(y, p.b.size, "primal ray")
    residual = _matvec(p.A.T, y, "Farkas A.T y")
    limit = _ray_limit(p, p.b, eps_ray)
    dual, metrics, limits = _cone(
        y, p.cones, dual=True, strict=True, eps_abs=eps_ray, eps_rel=0,
    )
    annihilation = _rows(residual, np.zeros_like(residual), "annihilation",
                         limit if limit > 0 else eps_ray, 0, metrics, limits)
    if limit == 0:
        annihilation = bool(np.all(residual == 0))
        limits.update({f"annihilation_{j}": 0.0 for j in range(residual.size)})
    normalization = _dot(p.b, y)
    metrics["b_dot_y"] = _represented(normalization, "b.T y")
    limits["normalization"] = eps_ray
    return _check("hsd_original_numerical_farkas", {
        "dual_cone": dual, "annihilation": annihilation,
        "strict_negative": normalization < 0,
        "normalization": abs(normalization + 1) <= _f(eps_ray),
    }, metrics, limits)


def check_hsd_recession_ray(
    problem: ConicProblem, direction: object, *, eps_ray: float = 1e-7,
) -> HSDLPCheck:
    """Check -A*d in K and q.T*d ~ -1; no fabricated feasible base.

    Passing alone is reported as a dual-infeasibility/recession numerical ray,
    NOT primal unboundedness. A separate independently checked base is required.
    """
    p = _model(problem)
    direction = _vector(direction, p.q.size, "recession ray")
    slack = -_matvec(p.A, direction, "recession A d")
    limit = _ray_limit(p, p.q, eps_ray)
    cone, metrics, limits = _cone(
        slack, p.cones, dual=False, strict=True, eps_abs=eps_ray, eps_rel=0,
    )
    # Strict original cone membership is stronger than approximate A*d+s=0.
    descent = _dot(p.q, direction)
    metrics.update(q_dot_d=_represented(descent, "q.T d"), tightened_residual_limit=limit)
    limits["normalization"] = eps_ray
    return _check("hsd_original_numerical_recession", {
        "recession_cone": cone, "strict_descent": descent < 0,
        "normalization": abs(descent + 1) <= _f(eps_ray),
    }, metrics, limits)


class PreparedHSDLP:
    """Frozen original model, one native LU, and cold state on every solve.

    Q=[[0,A.T,q],[-A,0,b],[-q.T,-b.T,0]], C=R^n x K* x R+.
    t=(I+Q)^-1 w; u=projection_C(2*t-w); w=w+u-t.
    v=u+w_old-2*t belongs to C* at a fixed point, with v=Q*u.
    Explicit host copies for hashing/audits are not a CPU solver fallback.
    """

    @torch.no_grad()
    def __init__(self, problem: ConicProblem, config: HSDLPConfig = HSDLPConfig()):
        if not isinstance(config, HSDLPConfig):
            raise TypeError("config must be HSDLPConfig")
        device = config.validate()
        if not isinstance(problem, ConicProblem):
            raise TypeError("problem must be ConicProblem")
        inputs = []
        self._input_uploads = 0
        self._input_host_audit_copies = 0
        for name in ("P", "q", "A", "b"):
            value = getattr(problem, name)
            if isinstance(value, torch.Tensor) and value.device.type != "cpu":
                if value.device != device or value.layout != torch.strided or value.is_complex():
                    raise ValueError(f"{name}: input device/layout mismatch; no implicit CPU fallback")
                self._input_host_audit_copies += 1
                value = value.detach().to(device="cpu", dtype=torch.float64)
            inputs.append(value)
        self._original = _model(ConicProblem(*inputs, problem.cones), config.max_embedding_dimension)
        self._model_sha256 = _hash_model(self._original)
        self._source_sha256 = _source_hashes()
        self.config, self._device = config, device
        self._config_seal = config
        self._projector = project_cones
        self._runtime_seal = self._runtime()
        self._q, self._A, self._b = [
            torch.tensor(getattr(self._original, name), device=device, dtype=torch.float64)
            for name in ("q", "A", "b")
        ]
        for name, tensor in (("q", self._q), ("A", self._A), ("b", self._b)):
            self._guard(tensor, f"original device {name}")
        self._input_uploads = 3 if device.type == "cuda" else 0
        n, m = self._q.numel(), self._b.numel()
        self._n, self._m = n, m
        Q = torch.zeros((n + m + 1, n + m + 1), device=device, dtype=torch.float64)
        Q[:n, n:n + m], Q[n:n + m, :n] = self._A.T, -self._A
        Q[:n, -1], Q[-1, :n] = self._q, -self._q
        Q[n:n + m, -1], Q[-1, n:n + m] = self._b, -self._b
        self._Q = Q
        self._system = torch.eye(n + m + 1, device=device, dtype=torch.float64) + Q
        self._guard(self._system, "resolvent system")
        try:
            self._lu, self._pivots, info = torch.linalg.lu_factor_ex(self._system)
        except RuntimeError as error:
            raise RuntimeError("HSD device LU factorization failed; no fallback") from error
        self._guard(self._lu, "LU factor")
        self._guard(self._pivots, "LU pivots", floating=False)
        if info.device != device or int(info) != 0:
            raise RuntimeError(f"HSD LU factorization failed/device mismatch info={int(info)}")
        self._tensor_seals = {
            name: getattr(self, name).clone() for name in
            ("_q", "_A", "_b", "_Q", "_system", "_lu", "_pivots")
        }

    def _runtime(self) -> dict[str, object]:
        return {
            "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
            "python_version": sys.version, "torch_cuda_runtime": torch.version.cuda,
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "device": str(self._device) if hasattr(self, "_device") else self.config.device,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }

    def _guard(self, value: torch.Tensor, name: str, *, floating: bool = True) -> None:
        if not isinstance(value, torch.Tensor) or value.device != self._device:
            raise RuntimeError(f"HSD device residency mismatch: {name}; no CPU fallback")
        if value.layout != torch.strided or (floating and value.dtype != torch.float64):
            raise RuntimeError(f"HSD tensor layout/dtype mismatch: {name}")
        _finite(value, name)

    def _integrity(self) -> None:
        if self.config != self._config_seal or _hash_model(self._original) != self._model_sha256:
            raise RuntimeError("HSD frozen model/config changed")
        if _source_hashes() != self._source_sha256 or project_cones is not self._projector:
            raise RuntimeError("HSD source/projection binding changed")
        if self._runtime() != self._runtime_seal:
            raise RuntimeError("HSD runtime/thread configuration changed")
        for name, sealed in self._tensor_seals.items():
            current = getattr(self, name)
            self._guard(current, name, floating=name != "_pivots")
            if not torch.equal(current, sealed):
                raise RuntimeError(f"HSD frozen device tensor changed: {name}")

    @property
    def original_problem(self) -> ConicProblem:
        return ConicProblem(*[getattr(self._original, name).copy()
                              for name in ("P", "q", "A", "b")], self._original.cones)

    def _host(self, value: torch.Tensor, name: str) -> np.ndarray:
        self._guard(value, name)
        return value.detach().to(device="cpu", dtype=torch.float64).numpy().copy()

    def _normalize(self, value: torch.Tensor, normal: torch.Tensor, name: str) -> torch.Tensor | None:
        products = value * normal
        self._guard(products, name + " products")
        if bool(((value != 0) & (normal != 0) & (products == 0)).any()):
            raise RuntimeError(f"HSD {name} normalization product underflow")
        denominator = -products.sum()
        self._guard(denominator, name + " denominator")
        if not bool(denominator > 0):
            return None
        result = value / denominator
        self._guard(result, name)
        if bool(((value != 0) & (result == 0)).any()):
            raise RuntimeError(f"HSD {name} normalization lost nonzero coordinate")
        return result

    @torch.no_grad()
    def solve(self, *, feasible_base: object | None = None) -> HSDLPResult:
        self._integrity()
        c, p = self.config, self._original
        base_tensor, base_check = None, None
        if feasible_base is not None:
            if isinstance(feasible_base, torch.Tensor) and feasible_base.device.type != "cpu":
                raise ValueError("feasible_base requires explicit original host coordinates")
            base = _vector(feasible_base, self._n, "feasible_base")
            applied_base = _matvec(p.A, base, "base A x")
            slack = np.array([_represented(_f(b) - _f(a), "base slack")
                              for a, b in zip(applied_base, p.b)])
            base_check = _primal(p, base, slack, c.eps_abs, c.eps_rel)
            if not base_check.passed:
                raise ValueError("supplied feasible base failed original-model checks")
            base_tensor = torch.tensor(base, device=self._device, dtype=torch.float64)
        w = torch.zeros(self._n + self._m + 1, device=self._device, dtype=torch.float64)
        w[-1] = 1
        status, trace = "max_iterations", []
        x = s = y = primal_ray = recession_ray = None
        candidate_check = primal_check = recession_check = None
        previous_primal = previous_recession = False
        host_audit_vectors = 0
        max_linear_error = 0.0
        for iteration in range(1, c.max_iterations + 1):
            old_w = w
            try:
                tilde = torch.linalg.lu_solve(self._lu, self._pivots, old_w[:, None]).squeeze(1)
            except RuntimeError as error:
                raise RuntimeError(f"HSD native resolvent solve failed iteration={iteration}") from error
            self._guard(tilde, f"resolvent iteration={iteration}")
            applied = self._system @ tilde
            self._guard(applied, f"resolvent residual iteration={iteration}")
            scale = torch.maximum(applied.abs().max(), old_w.abs().max()).clamp_min(1)
            error = (applied / scale - old_w / scale).abs().max()
            self._guard(error, "resolvent backward residual")
            linear_error = float(error)
            max_linear_error = max(max_linear_error, linear_error)
            if linear_error > c.linear_residual_tolerance:
                raise RuntimeError(f"HSD resolvent residual failed iteration={iteration} error={linear_error}")
            argument = 2 * tilde - old_w
            self._guard(argument, "DR projection argument")
            u = argument.clone()
            projected = self._projector(argument[self._n:-1], p.cones, dual=True)
            self._guard(projected, "projected dual cone")
            u[self._n:-1], u[-1] = projected, argument[-1].clamp_min(0)
            v = u - argument
            w = old_w + u - tilde
            for name, value in (("u", u), ("v", v), ("w", w)):
                self._guard(value, f"{name} iteration={iteration}")
            if iteration % c.check_interval != 0 and iteration != c.max_iterations:
                continue
            candidate_check = primal_check = recession_check = None
            x = s = y = primal_ray = recession_ray = None
            tau, kappa = float(u[-1]), float(v[-1])
            if tau > c.tau_min:
                x, s, y = u[:self._n] / u[-1], v[self._n:-1] / u[-1], u[self._n:-1] / u[-1]
                for name, before, after in (
                    ("x", u[:self._n], x), ("s", v[self._n:-1], s),
                    ("y", u[self._n:-1], y),
                ):
                    self._guard(after, "decoded candidate " + name)
                    if bool(((before != 0) & (after == 0)).any()):
                        raise RuntimeError(f"HSD candidate {name} division lost nonzero coordinate")
                candidate_check = check_hsd_candidate(
                    p, self._host(x, "candidate x"), self._host(s, "candidate s"),
                    self._host(y, "candidate y"), eps_abs=c.eps_abs, eps_rel=c.eps_rel,
                )
                host_audit_vectors += 3
            primal_ray = self._normalize(u[self._n:-1], self._b, "Farkas ray")
            recession_ray = self._normalize(u[:self._n], self._q, "recession ray")
            if primal_ray is not None:
                primal_check = check_hsd_primal_ray(p, self._host(primal_ray, "Farkas ray"),
                                                   eps_ray=c.eps_ray)
                host_audit_vectors += 1
            if recession_ray is not None:
                recession_check = check_hsd_recession_ray(
                    p, self._host(recession_ray, "recession ray"), eps_ray=c.eps_ray,
                )
                host_audit_vectors += 1
            primal_good = primal_check is not None and primal_check.passed
            recession_good = recession_check is not None and recession_check.passed
            embedding_error = self._Q @ u - v
            self._guard(embedding_error, "embedding residual")
            trace.append({
                "iteration": iteration, "tau": tau, "kappa": kappa,
                "embedding_residual_inf": float(embedding_error.abs().max()),
                "resolvent_backward_error": linear_error,
                "candidate": None if candidate_check is None else asdict(candidate_check),
                "farkas": None if primal_check is None else asdict(primal_check),
                "recession": None if recession_check is None else asdict(recession_check),
            })
            if candidate_check is not None and candidate_check.passed:
                status = "solved_candidate"
                break
            if primal_good and previous_primal:
                status = "primal_infeasible_ray"
                x = s = y = None
                break
            if recession_good and previous_recession:
                status = "unbounded_candidate" if base_check is not None else "dual_infeasible_ray"
                x = s = y = None
                break
            previous_primal, previous_recession = primal_good, recession_good
        self._integrity()
        settings = asdict(c)
        settings["dtype"] = str(c.dtype)
        runtime = dict(self._runtime_seal)
        runtime.update(
            factor_device=str(self._lu.device), pivots_device=str(self._pivots.device),
            embedding_device=str(u.device), factorization_calls=1,
            linear_solve_calls=iteration, projection_calls=iteration,
            input_uploads=self._input_uploads,
            input_host_audit_copies=self._input_host_audit_copies,
            explicit_host_audit_vectors=host_audit_vectors,
            device_integrity_checks="native torch.equal; no host factor solve",
            max_resolvent_backward_error=max_linear_error,
            solve_state="cold_reset; prepared factor reused; no warm start",
        )
        return HSDLPResult(
            status, iteration, x, s, y, primal_ray, recession_ray, base_tensor,
            candidate_check, primal_check, recession_check, base_check, u, v, w,
            tuple(trace), self._model_sha256, dict(self._source_sha256), settings,
            runtime, str(self._device), iteration,
        )


def prepare_hsd_lp(
    problem: ConicProblem, config: HSDLPConfig = HSDLPConfig(),
) -> PreparedHSDLP:
    return PreparedHSDLP(problem, config)


def solve_hsd_lp(
    problem: ConicProblem, config: HSDLPConfig = HSDLPConfig(), *,
    feasible_base: object | None = None,
) -> HSDLPResult:
    return prepare_hsd_lp(problem, config).solve(feasible_base=feasible_base)
