"""Device-local COO structural validation; CPU mechanisms are not CUDA proof.

The old original CUDA Placement and producer are imported read-only, never
modified. Explicit selection of the new CUDA entry requests real CUDA and must
fail when unavailable. Source scans expose counts and hashes, never secrets.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from types import SimpleNamespace
from xml.etree import ElementTree

import pytest
import torch

import openopt.operators as operators
from tests.test_operator_original_cuda import Placement


ROOT = Path(__file__).resolve().parents[1]


def coo(indices, values, shape=(2, 3), *, device="cpu", dtype=torch.float64, coalesced=False):
    index = torch.tensor(indices, device=device, dtype=torch.int64).reshape(2, -1)
    value = torch.tensor(values, device=device, dtype=dtype)
    return torch.sparse_coo_tensor(index, value, shape, device=device, dtype=dtype,
                                  check_invariants=False, is_coalesced=coalesced)


def test_device_local_rebuild_does_not_enter_backend_invariant_checker(monkeypatch):
    matrix = coo([[1, 0, 1], [2, 0, 2]], [1, 3, 2])
    saved_indices, saved_values = matrix._indices().clone(), matrix._values().clone()
    original_constructor = torch.sparse_coo_tensor
    calls = []

    def observed(indices, values, size, **kwargs):
        calls.append(kwargs.copy())
        assert kwargs["check_invariants"] is False
        assert kwargs["is_coalesced"] is False
        assert indices.device == values.device == kwargs["device"] == matrix.device
        return original_constructor(indices, values, size, **kwargs)

    monkeypatch.setattr(torch, "sparse_coo_tensor", observed)
    with Placement(matrix.device) as trace:
        rebuilt = operators._checked_device_coo(matrix)
    assert len(calls) == 1
    assert rebuilt.layout == torch.sparse_coo and rebuilt.device == matrix.device
    assert torch.equal(rebuilt._indices(), saved_indices)
    assert torch.equal(rebuilt._values(), saved_values)
    assert torch.equal(matrix._indices(), saved_indices)
    assert torch.equal(matrix._values(), saved_values)
    assert any("lt" in op or "ge" in op for op in trace.operations)


INVALID_COO = [
    pytest.param([[-1], [0]], [1], (2, 3), id="negative-row"),
    pytest.param([[2], [0]], [1], (2, 3), id="row-at-size"),
    pytest.param([[0], [-1]], [1], (2, 3), id="negative-column"),
    pytest.param([[0], [3]], [1], (2, 3), id="column-at-size"),
    pytest.param([[2**63-1], [0]], [1], (2, 3), id="int64-extreme"),
    pytest.param([[0], [0]], [1], (0, 3), id="zero-row-nonempty"),
    pytest.param([[0], [0]], [1], (2, 0), id="zero-column-nonempty"),
    pytest.param([[0], [0]], [float("nan")], (2, 3), id="nan-value"),
    pytest.param([[0], [0]], [float("inf")], (2, 3), id="inf-value"),
]


@pytest.mark.parametrize("indices,values,shape", INVALID_COO)
def test_invalid_coo_rejected_before_rebuild_or_math(indices, values, shape, monkeypatch):
    matrix = coo(indices, values, shape)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid original COO reached reconstruction/coalesce/mm")

    monkeypatch.setattr(torch, "sparse_coo_tensor", forbidden)
    monkeypatch.setattr(torch.Tensor, "coalesce", forbidden)
    monkeypatch.setattr(torch.sparse, "mm", forbidden)
    with pytest.raises((RuntimeError, ValueError), match="COO"):
        operators._checked_device_coo(matrix)


@pytest.mark.parametrize("case", ["indices-rank", "indices-shape", "indices-dtype", "indices-device",
                                 "values-rank", "values-count", "values-dtype", "values-device"])
def test_raw_coo_component_metadata_rejected(case, monkeypatch):
    matrix = coo([[0], [1]], [2])
    indices, values = matrix._indices(), matrix._values()
    original_indices, original_values = torch.Tensor._indices, torch.Tensor._values
    if case == "indices-rank":
        indices = indices.reshape(-1)
    elif case == "indices-shape":
        indices = indices[:1]
    elif case == "indices-dtype":
        indices = indices.to(torch.int32)
    elif case == "indices-device":
        indices = torch.empty((2, 1), device="meta", dtype=torch.int64)
    elif case == "values-rank":
        values = values.reshape(1, 1)
    elif case == "values-count":
        values = values[:0]
    elif case == "values-dtype":
        values = values.to(torch.float32)
    else:
        values = torch.empty(1, device="meta", dtype=torch.float64)
    monkeypatch.setattr(torch.Tensor, "_indices", lambda self: indices if self is matrix else original_indices(self))
    monkeypatch.setattr(torch.Tensor, "_values", lambda self: values if self is matrix else original_values(self))
    with pytest.raises(RuntimeError, match="COO"):
        operators._checked_device_coo(matrix)


def test_coo_helper_requires_original_two_dimensional_sparse_float():
    for matrix in (torch.eye(2, dtype=torch.float64), torch.eye(2, dtype=torch.float64).to_sparse_csr()):
        with pytest.raises(RuntimeError, match="sparse COO"):
            operators._checked_device_coo(matrix)
    hybrid = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.ones((1, 2), dtype=torch.float64),
                                    (2, 2), check_invariants=False)
    with pytest.raises(RuntimeError, match="two sparse dimensions"):
        operators._checked_device_coo(hybrid)
    for dtype in (torch.float16, torch.complex128):
        with pytest.raises(ValueError, match="float32/float64"):
            operators._checked_device_coo(coo([[0], [1]], [1], dtype=dtype))


def test_finite_duplicate_sum_overflow_still_rejected():
    maximum = torch.finfo(torch.float64).max
    rebuilt = operators._checked_device_coo(coo([[0, 0], [0, 0]], [maximum, maximum]))
    with pytest.raises(ValueError, match="finite"):
        operators.TensorOperator(rebuilt)


LEGAL_COO = [
    pytest.param([[1, 0, 1], [2, 0, 2]], [1, 3, 2], (2, 3), False, id="unsorted-duplicates"),
    pytest.param([[1, 0, 1], [2, 0, 2]], [1, 3, 2], (2, 3), True, id="false-coalesced-claim"),
    pytest.param([[], []], [], (0, 3), False, id="empty-rows"),
    pytest.param([[], []], [], (2, 0), False, id="empty-columns"),
    pytest.param([[], []], [], (0, 0), False, id="empty-both"),
    pytest.param([[0, 1], [2, 1]], [0, -4], (2, 3), False, id="zero-and-negative-values"),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("indices,values,shape,claimed", LEGAL_COO)
def test_device_local_coo_cpu_legal_sparse_math(indices, values, shape, claimed, dtype, monkeypatch):
    matrix = coo(indices, values, shape, dtype=dtype, coalesced=claimed)
    saved_indices, saved_values = matrix._indices().clone(), matrix._values().clone()
    dense = torch.zeros(shape, dtype=dtype)
    for row, column, value in zip(indices[0], indices[1], values):
        dense[row, column] += value

    def forbidden(*args, **kwargs):
        raise AssertionError("COO operation attempted densification")

    monkeypatch.setattr(torch.Tensor, "to_dense", forbidden)
    with Placement(matrix.device):
        rebuilt = operators._checked_device_coo(matrix)
        operator = operators.TensorOperator(rebuilt)
        vector = torch.ones(shape[1], dtype=dtype)
        reverse = torch.ones(shape[0], dtype=dtype)
        actual = operator.matvec(vector)
        transposed = operator.rmatvec(reverse)
    assert torch.equal(actual, dense @ vector)
    assert torch.equal(transposed, dense.T @ reverse)
    assert operator.matrix.is_coalesced()
    assert torch.equal(matrix._indices(), saved_indices)
    assert torch.equal(matrix._values(), saved_values)


@pytest.mark.parametrize("layout", ["coo", "csr"])
def test_cpu_original_backend_invariant_validation_retained(layout, monkeypatch):
    dense = torch.eye(2, dtype=torch.float64)
    matrix = dense.to_sparse_coo() if layout == "coo" else dense.to_sparse_csr()
    original = torch.sparse_coo_tensor if layout == "coo" else torch.sparse_csr_tensor
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["check_invariants"] is True
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "sparse_coo_tensor" if layout == "coo" else "sparse_csr_tensor", observed)
    operator = operators.TensorOperator(matrix)
    assert len(calls) == 1
    assert torch.equal(operator.matvec(torch.ones(2, dtype=torch.float64)), torch.ones(2, dtype=torch.float64))


@pytest.mark.parametrize("crow,col", [([0, 1, 2], [0, 3]), ([0, 1, 2], [-1, 0]),
                                     ([1, 1, 2], [0, 1]), ([0, 2, 1], [0, 1]),
                                     ([0, 0, 1], [0, 1]), ([0, 2, 2], [0, 0])])
def test_original_invalid_csr_still_rejected(crow, col):
    matrix = torch.sparse_csr_tensor(torch.tensor(crow), torch.tensor(col),
                                    torch.ones(2, dtype=torch.float64), (2, 2), check_invariants=False)
    with pytest.raises(RuntimeError):
        operators.TensorOperator(matrix)


def cuda_entry_selected(config):
    names = ("test_cuda_coo_structure_entry", "test_cuda_coo_invalid_entry")
    return any(any("::" + name in str(arg) for arg in config.args) or name in config.option.keyword
               for name in names)


def required_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("explicit COO CUDA unavailable; no CPU fallback")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("explicit COO CUDA index unavailable; no CPU fallback")
    return torch.device("cuda:0")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("indices,values,shape,claimed", LEGAL_COO)
def test_cuda_coo_structure_entry(indices, values, shape, claimed, dtype, monkeypatch, request):
    if not cuda_entry_selected(request.config):
        pytest.skip("CUDA structure entry not explicitly selected; execution UNVERIFIED")
    device = required_cuda()
    matrix = coo(indices, values, shape, device=device, dtype=dtype, coalesced=claimed)
    reference = torch.zeros(shape, device=device, dtype=dtype)
    for row, column, value in zip(indices[0], indices[1], values):
        reference[row, column] += value
    original_constructor = torch.sparse_coo_tensor
    calls = []

    def observed(*args, **kwargs):
        calls.append(kwargs.copy())
        assert kwargs["check_invariants"] is False
        assert kwargs["device"] == device and kwargs["is_coalesced"] is False
        return original_constructor(*args, **kwargs)

    monkeypatch.setattr(torch, "sparse_coo_tensor", observed)
    with Placement(device) as trace:
        operator = operators.TensorOperator(matrix)
        forward = operator.matvec(torch.ones(shape[1], device=device, dtype=dtype))
        reverse = operator.rmatvec(torch.ones(shape[0], device=device, dtype=dtype))
    assert len(calls) == 1
    assert torch.equal(forward, reference.sum(dim=1))
    assert torch.equal(reverse, reference.sum(dim=0))
    assert operator.matrix.device == operator._transpose.device == device
    assert trace.tensors > 0


@pytest.mark.parametrize("indices,values,shape", INVALID_COO)
def test_cuda_coo_invalid_entry(indices, values, shape, monkeypatch, request):
    if not cuda_entry_selected(request.config):
        pytest.skip("CUDA structure entry not explicitly selected; execution UNVERIFIED")
    device = required_cuda()
    matrix = coo(indices, values, shape, device=device)

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid CUDA COO reached reconstruction/coalesce/mm")

    monkeypatch.setattr(torch, "sparse_coo_tensor", forbidden)
    monkeypatch.setattr(torch.Tensor, "coalesce", forbidden)
    monkeypatch.setattr(torch.sparse, "mm", forbidden)
    with Placement(device):
        with pytest.raises((ValueError, RuntimeError), match="COO"):
            operators.TensorOperator(matrix)


def test_explicit_cuda_entry_cpu_unavailable_fails_never_skips(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    request = SimpleNamespace(config=SimpleNamespace(
        args=["tests/test_operator_coo_device.py::test_cuda_coo_structure_entry"],
        option=SimpleNamespace(keyword="")))
    for entry in (test_cuda_coo_structure_entry, test_cuda_coo_invalid_entry):
        with pytest.raises(RuntimeError, match="explicit COO CUDA unavailable; no CPU fallback"):
            if entry is test_cuda_coo_structure_entry:
                entry([[0], [1]], [2], (2, 3), False, torch.float64, monkeypatch, request)
            else:
                entry([[-1], [0]], [2], (2, 3), monkeypatch, request)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(RuntimeError, match="index unavailable"):
        required_cuda()


def test_cuda_collection_selection_default_off():
    for targets, keyword, expected in (
        (["tests/test_operator_coo_device.py"], "", False),
        (["tests/test_operator_coo_device.py"], "test_cuda", False),
        (["tests/test_operator_coo_device.py::test_cuda_coo_structure_entry"], "", True),
        (["tests/test_operator_coo_device.py::test_cuda_coo_invalid_entry"], "", True),
        (["tests/test_operator_coo_device.py"], "test_cuda_coo_structure_entry or test_cuda_coo_invalid_entry", True),
    ):
        assert cuda_entry_selected(SimpleNamespace(args=targets, option=SimpleNamespace(keyword=keyword))) is expected


def diagnose(run_root, output):
    """Read archived bytes only, no artifact/source execution or trust upgrade."""
    run_root, output = run_root.resolve(), output.resolve()
    if output.is_relative_to(ROOT) or output.is_relative_to(run_root):
        raise ValueError("diagnosis must be external and not change historical evidence")
    cases = sorted((run_root / "case-0").glob("cuda-check-*"))
    if len(cases) != 1:
        raise ValueError("expected one archived CUDA case directory")
    case = cases[0]
    paths = {name: case / name for name in ("result.json", "pytest.xml", "pytest.log")}
    paths["source-hashes.json"] = run_root / "source/source-hashes.json"
    paths["operators.py"] = run_root / "source/src/openopt/operators.py"
    paths["producer.py"] = run_root / "source/tests/test_operator_original_cuda.py"
    payloads = {name: path.read_bytes() for name, path in paths.items()}
    result = json.loads(payloads["result.json"])
    if result["verdict"] != "FAIL" or result["returncode"] != 1:
        raise ValueError("historical diagnosis must retain original FAIL and exit code")
    for key, name in (("src/openopt/operators.py", "operators.py"),
                       ("tests/test_operator_original_cuda.py", "producer.py")):
        if result["source_sha256"][key] != hashlib.sha256(payloads[name]).hexdigest():
            raise ValueError("archive source hash mismatch")
    root = ElementTree.fromstring(payloads["pytest.xml"])
    nodes = root.findall(".//testcase")
    failed = [node for node in nodes if node.find("failure") is not None]
    if len(nodes) != 53 or len(failed) != 9 or any(node.find("skipped") is not None for node in nodes):
        raise ValueError("historical exact membership/count mismatch")
    failures = []
    for node in failed:
        text = node.find("failure").text or ""
        if not node.attrib["name"].endswith("-coo]") or not all(needle in text for needle in (
                "src/openopt/operators.py:81", "torch.sparse_coo_tensor", "tensor([0, 0])", "off-device operator tensor")):
            raise ValueError("historical failure differs from observed COO placement first cause")
        failures.append(node.attrib["name"])
    report = {"schema_version": 1, "verdict": "HISTORICAL_FAIL_DIAGNOSED",
              "collected_at": datetime.now(timezone.utc).isoformat(),
              "counts": {"tests": 53, "passed": 44, "failed": 9, "skipped": 0},
              "failure_nodes": failures, "hardware": result["hardware"],
              "qualification": "archived diagnosis only; repaired CUDA execution UNVERIFIED",
              "first_cause": "COO invariant constructor produced CPU index-bound tensor rejected by unchanged Placement",
              "pins": {name: {"path": str(paths[name]), "sha256": hashlib.sha256(payload).hexdigest(),
                              "bytes": len(payload)} for name, payload in payloads.items()}}
    output.mkdir(parents=True, exist_ok=True)
    with (output / "diagnosis.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print("COO_DIAGNOSIS verdict=HISTORICAL_FAIL_DIAGNOSED tests=53 passed=44 failed=9 skipped=0")
    return 0


def scan(argv=None):
    parser = argparse.ArgumentParser(description="Scoped device COO source credential scan")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--scan-only", action="store_true")
    actions.add_argument("--diagnose-run", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.diagnose_run is not None:
        return diagnose(args.diagnose_run, args.artifact_dir)
    output = args.artifact_dir.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("scan evidence must be external")
    output.mkdir(parents=True, exist_ok=True)
    spec = importlib.util.spec_from_file_location("safe_scanner", "/data/dev/infra/ops/dsh/scan-secrets-precommit.py")
    scanner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanner)
    needles = scanner.read_needles(argparse.Namespace(secret_hex=None, secrets_file=None, no_needle=False,
                         needle_from=str(Path.home() / ".config/dsh-n600-token")))
    if not needles:
        raise RuntimeError("credential scan needle unavailable")
    paths = [ROOT / "src/openopt/operators.py", Path(__file__).resolve(), ROOT / "docs/research/operator-coo-device.md"]
    literal, _ = scanner.scan_literal([str(path) for path in paths], needles)
    pattern = re.compile(r"BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY|IDENTIFIED" r" BY|"
                         r"(?i:password|token|secret)\s*[\"\x27]?\s*[:=]\s*[\"\x27][^\"\x27\s]{6,}[\"\x27]")
    shape = sum(bool(pattern.search(path.read_text())) for path in paths)
    verdict = "PASS" if literal == shape == 0 else "FAIL"
    report = {"verdict": verdict, "literal_hits": literal, "shape_hits": shape,
              "files": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}
    with (output / "source-security-scan.json").open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"COO_SOURCE_SCAN verdict={verdict} literal_hits={literal} shape_hits={shape} files={len(paths)}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(scan())
