"""Independent CUDA RSOC regression, not the fixed thirteen-case producer.

Run with scripts/verify.py and external artifacts. CPU mechanisms deliberately
reuse the numerical gates but never qualify CUDA. No solver is called.
"""

from collections import Counter
from dataclasses import dataclass
from decimal import localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils._pytree import tree_flatten
from torch.utils._python_dispatch import TorchDispatchMode

import openopt.conic as conic
import openopt.native_conic as native
from tests import test_rotated_conic_independent as oracle

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src/openopt/conic.py", "src/openopt/native_conic.py",
    "tests/test_rotated_conic_independent.py",
    "tests/test_rotated_conic_cuda_extremes.py",
)


@dataclass(frozen=True)
class Case:
    name: str
    dtype: torch.dtype
    value: tuple[float, ...]
    region: str


def _cases(dtype):
    f64 = dtype == torch.float64
    label, power = ("f64", 600) if f64 else ("f32", 100)
    head = 2. ** power
    middle, tiny = (2. ** 200, 2. ** -450) if f64 else (2. ** 30, 2. ** -60)
    tails = (
        ("unit-last", (head, head, 1.)),
        ("unit-first", (1., head, head)),
        ("unit-middle", (head, 1., head)),
        ("signed", (-head, head, -1.)),
        ("middle-last", (head, head, middle)),
        ("middle-first", (middle, head, head)),
        ("two-levels-last", (head, head, 1., tiny)),
        ("two-levels-first", (tiny, 1., head, head)),
    )
    result = [Case(f"{label}-remote-{name}", dtype, (-head, -head, *tail), "polar-exterior")
              for name, tail in tails]
    root = math.sqrt(2.) if f64 else float(np.nextafter(np.float32(math.sqrt(2.)), np.float32(np.inf)))
    polar_head = 2. ** (500 if f64 else 100)
    result += [
        Case(f"{label}-polar-sqrt2", dtype, (-1., -1., root), "polar-exterior"),
        Case(f"{label}-polar-dyadic", dtype,
             (-polar_head, -polar_head, polar_head, polar_head, 1.), "polar-exterior"),
        Case(f"{label}-negative-head", dtype, (-1e20, 0., 1e8), "mixed"),
        Case(f"{label}-small-head", dtype,
             (-1e300, 0., 1e210) if f64 else (-1e30, 0., 1e17), "mixed"),
        Case(f"{label}-overflowing-norm", dtype,
             (0., 0., 1.3e308, 1.3e308) if f64 else (0., 0., 3e38, 3e38), "exterior"),
    ]
    previous = math.nextafter(head, 0.) if f64 else float(np.nextafter(np.float32(head), np.float32(0.)))
    for name, tail, region in (
        ("unit-last", (head, head, 1.), "exterior"),
        ("unit-first", (1., head, head), "exterior"),
        ("unit-middle", (head, 1., head), "exterior"),
        ("boundary", (head, head, 0.), "boundary"),
        ("inside-last", (head, previous, 1.), "inside"),
        ("inside-first", (1., previous, head), "inside"),
    ):
        result.append(Case(f"{label}-positive-{name}", dtype, (head, head, *tail), region))
    return tuple(result)


CASES = _cases(torch.float64) + _cases(torch.float32)
PARAMETERS = [pytest.param(case, id=case.name) for case in CASES]
CUDA_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA unavailable: independent extreme CUDA nodes are UNVERIFIED, not CPU PASS",
)


def _source_hashes():
    for module, relative in ((conic, SOURCE_PATHS[0]), (native, SOURCE_PATHS[1]),
                             (oracle, SOURCE_PATHS[2])):
        assert Path(module.__file__).resolve() == ROOT / relative, "foreign loaded numerical source"
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_PATHS}


class DeviceTrace(TorchDispatchMode):
    """Reject any off-device tensor operand/result during real projection ops."""

    def __init__(self, device):
        super().__init__()
        self.device = torch.device(device)
        self.operations = Counter()
        self.dtypes = set()
        self.tensor_count = 0

    def _check(self, value):
        for item in tree_flatten(value)[0]:
            if isinstance(item, torch.Tensor):
                assert item.device == self.device, ("off-device projection tensor", item.device, self.device)
                self.tensor_count += 1
                self.dtypes.add(str(item.dtype))

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        self._check((args, kwargs))
        result = function(*args, **kwargs)
        self._check(result)
        self.operations[str(function)] += 1
        return result

    def report(self):
        assert self.tensor_count > 0 and self.operations, "empty placement trace is not evidence"
        return {"device": str(self.device), "tensor_count": self.tensor_count,
                "dtypes": sorted(self.dtypes), "operations": dict(self.operations)}


def _project(value, dual, monkeypatch):
    trace = DeviceTrace(value.device)

    def forbidden_host_transfer(*args, **kwargs):
        raise AssertionError("CPU projection transfer is forbidden")

    # Host reference copies happen outside this scope. Scalar Boolean control
    # in the existing implementation is allowed; CPU tensor projection is not.
    with monkeypatch.context() as patch:
        for method in ("cpu", "numpy", "tolist"):
            patch.setattr(torch.Tensor, method, forbidden_host_transfer)
        with trace:
            projected = native.project_cones(value, (conic.RSOC(len(value)),), dual=dual)
    assert projected.device == value.device and projected.dtype == value.dtype
    a, b, *tail = [Fraction.from_float(float(x)) for x in value.detach().to("cpu")]
    excess = sum(x * x for x in tail) - 2 * a * b
    early = (a >= 0 and b >= 0 or a <= 0 and b <= 0) and excess <= 0
    norms = sum(count for name, count in trace.operations.items() if name.startswith("aten.linalg_vector_norm."))
    assert norms == (1 if early else 2), ("incorrect inside/polar early return", excess, norms)
    assert "torch.float64" in trace.dtypes, "device-local float64 intermediates were not exercised"
    return projected, trace.report()


def _exercise(case, dual, device, monkeypatch, tmp_path, request):
    before_source = _source_hashes()
    value = torch.tensor(case.value, dtype=case.dtype, device=device)
    original = value.detach().to("cpu").clone()
    assert torch.isfinite(original).all()
    a, b, *tail = [Fraction.from_float(float(x)) for x in original]
    excess = sum(x * x for x in tail) - 2 * a * b
    assert (excess > 0 if case.region in ("exterior", "polar-exterior", "mixed")
            else excess == 0 if case.region == "boundary" else excess < 0)
    expected, negative_expected = oracle.original_projection(original), oracle.original_projection(-original)
    if case.region == "polar-exterior":
        assert a < 0 and b < 0 and float(expected[0]) > 0 and float(expected[1]) > 0
    positive, positive_trace = _project(value, dual, monkeypatch)
    negative, negative_trace = _project(-value, not dual, monkeypatch)
    p, n = positive.detach().to("cpu"), negative.detach().to("cpu")
    oracle.assert_coordinates(p, expected)
    oracle.assert_coordinates(n, negative_expected)
    oracle.assert_scaled_moreau(original, p, n)
    torch.testing.assert_close(value.detach().to("cpu"), original, atol=0, rtol=0)
    if case.region in ("inside", "boundary"):
        torch.testing.assert_close(p, original, atol=0, rtol=0)
    if a >= 0 and b >= 0:
        assert conic.check_cone_feasibility(
            original, (conic.RSOC(len(original)),), atol=math.ulp(0.), rtol=0.,
        ).passed is (excess <= 0)
    assert _source_hashes() == before_source, "source bytes changed during case"
    with localcontext() as context:
        context.prec = 1100
        saved = {
            "schema_version": 1, "case": case.name, "nodeid": request.node.nodeid,
            "verdict": "CUDA_NUMERICAL_REGRESSION_PASS" if device == "cuda:0" else "CPU_MECHANISM_PASS",
            "qualification": "projection regression only; no commercial, performance or remote attestation",
            "requested_device": device, "observed_device": str(positive.device), "dtype": str(case.dtype),
            "gpu_name": torch.cuda.get_device_name(0) if device == "cuda:0" else None,
            "input_hex": [float(x).hex() for x in original],
            "positive_hex": [float(x).hex() for x in p],
            "negative_hex": [float(x).hex() for x in n],
            "expected_decimal": list(map(str, expected)), "negative_expected_decimal": list(map(str, negative_expected)),
            "exact_excess": {"numerator": str(excess.numerator), "denominator": str(excess.denominator)},
            "dual": dual, "source_sha256": before_source,
            "trace": {"positive": positive_trace, "negative": negative_trace},
            "original_coordinate_check": True, "scaled_moreau_check": True,
        }
    path = tmp_path / "rotated-conic-extreme.json"
    path.write_text(json.dumps(saved, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    assert json.loads(path.read_bytes()) == saved
    request.node.user_properties.extend((
        ("rsoc_cuda_extremes_path", str(path)),
        ("rsoc_cuda_extremes_sha256", hashlib.sha256(path.read_bytes()).hexdigest()),
    ))


@CUDA_ONLY
@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
@pytest.mark.parametrize("case", PARAMETERS)
def test_cuda_rotated_extreme_coordinates(case, dual, monkeypatch, tmp_path, request):
    _exercise(case, dual, "cuda:0", monkeypatch, tmp_path, request)
    torch.cuda.synchronize(0)


@pytest.mark.parametrize("dual", [False, True], ids=["primal", "dual"])
@pytest.mark.parametrize("case", PARAMETERS)
def test_cpu_mechanism_rotated_extreme_coordinates(case, dual, monkeypatch, tmp_path, request):
    _exercise(case, dual, "cpu", monkeypatch, tmp_path, request)


def test_cpu_mechanism_guard_rejects_cpu_tensor_in_cuda_scope():
    with pytest.raises(AssertionError, match="off-device projection tensor"):
        with DeviceTrace("cuda:0"):
            torch.ones(2, device="cpu")


def test_cpu_mechanism_guard_rejects_off_device_output():
    with pytest.raises(AssertionError, match="off-device projection tensor"):
        with DeviceTrace("cpu"):
            torch.empty(2, device="meta")


def test_cpu_mechanism_guard_observes_real_operations():
    trace = DeviceTrace("cpu")
    with trace:
        torch.frexp(torch.ones(2, dtype=torch.float64))
    assert trace.report()["device"] == "cpu"
    assert any(name.startswith("aten.frexp.") for name in trace.operations)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["f32", "f64"])
def test_cpu_mechanism_oracle_rejects_erased_representable_polar_heads(dtype):
    case = next(case for case in CASES if case.dtype == dtype and case.name.endswith("remote-unit-last"))
    value = torch.tensor(case.value, dtype=dtype)
    with pytest.raises(AssertionError):
        oracle.assert_coordinates(torch.zeros_like(value), oracle.original_projection(value))


def test_cpu_mechanism_exact_case_membership():
    assert len(CASES) == len({case.name for case in CASES}) == 38
    assert Counter(case.dtype for case in CASES) == {torch.float64: 19, torch.float32: 19}
    assert Counter(case.region for case in CASES) == {
        "polar-exterior": 20, "mixed": 4, "exterior": 8, "boundary": 2, "inside": 4,
    }
    _source_hashes()
