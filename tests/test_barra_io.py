from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

from openpo.barra import BarraRiskData
from openpo.barra_io import (
    ARRAY_NAMES,
    NPZ_KEYS,
    BarraInputCase,
    BarraNpzError,
    load_barra_npz,
    save_barra_npz,
)


def _case(dtype: torch.dtype = torch.float64) -> BarraInputCase:
    data = BarraRiskData(
        exposures=torch.tensor(
            [[1.0, 0.2], [0.5, -0.3], [-0.1, 0.7]], dtype=dtype
        ),
        factor_covariance=torch.tensor([[0.3, 0.04], [0.04, 0.2]], dtype=dtype),
        specific_variance=torch.tensor([0.1, 0.2, 0.15], dtype=dtype),
        benchmark=torch.tensor([0.4, 0.35, 0.25], dtype=dtype),
    )
    return BarraInputCase(
        data=data,
        alpha=torch.tensor([[0.02, -0.01, 0.03], [0.01, 0.04, -0.02]], dtype=dtype),
        current=torch.tensor([0.4, 0.35, 0.25], dtype=dtype),
        box_lower=torch.tensor([0.0, 0.0, 0.0], dtype=dtype),
        box_upper=torch.tensor([0.8, 0.8, 0.8], dtype=dtype),
        factor_exposure_lower=torch.tensor([-0.5, -0.4], dtype=dtype),
        factor_exposure_upper=torch.tensor([0.8, 0.6], dtype=dtype),
        factor_labels=("industry_0", "style_0"),
    )


def _replace(case: BarraInputCase, **changes) -> BarraInputCase:
    values = {
        "data": case.data,
        "alpha": case.alpha,
        "current": case.current,
        "box_lower": case.box_lower,
        "box_upper": case.box_upper,
        "factor_exposure_lower": case.factor_exposure_lower,
        "factor_exposure_upper": case.factor_exposure_upper,
        "factor_labels": case.factor_labels,
    }
    values.update(changes)
    return BarraInputCase(**values)


def _rewrite_npz(path: Path, transform) -> None:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key], copy=True) for key in archive.files}
    transform(payload)
    np.savez(path, **payload)


def _load_benchmark_module():
    root = Path(__file__).resolve().parents[1]
    source = root / "src"
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    path = root / "benchmarks" / "benchmark_cntr_6000.py"
    spec = importlib.util.spec_from_file_location("barra_io_benchmark_cntr", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_complete_case_roundtrip_is_strict_and_auditable(tmp_path, dtype) -> None:
    path = tmp_path / "barra.npz"
    original = _case(dtype)
    user_metadata = {"as_of": "2026-08-08", "source": "authorized-cntr", "revision": 7}

    saved_audit = save_barra_npz(path, original, metadata=user_metadata)
    loaded, loaded_audit = load_barra_npz(path, return_audit=True)

    assert isinstance(loaded, BarraInputCase)
    for name in ARRAY_NAMES:
        if hasattr(loaded, name):
            loaded_tensor = getattr(loaded, name)
            original_tensor = getattr(original, name)
        else:
            loaded_tensor = getattr(loaded.data, name)
            original_tensor = getattr(original.data, name)
        assert torch.equal(loaded_tensor, original_tensor)
    assert loaded.factor_labels == original.factor_labels
    assert loaded.batch == 2
    assert loaded_audit == saved_audit
    assert loaded_audit.metadata == {
        "factor_labels": ["industry_0", "style_0"],
        "user_metadata": user_metadata,
    }
    assert len(loaded_audit.combined_sha256) == 64
    assert set(loaded_audit.array_sha256) == set(ARRAY_NAMES)
    assert all(len(digest) == 64 for digest in loaded_audit.array_sha256.values())
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == NPZ_KEYS
        assert all(archive[name].dtype.kind != "O" for name in archive.files)


def test_loaded_case_duck_types_into_real_build_barra_problem(tmp_path) -> None:
    path = tmp_path / "benchmark-input.npz"
    save_barra_npz(path, _case())
    loaded = load_barra_npz(path)
    benchmark = _load_benchmark_module()

    problem, constraints = benchmark.build_barra_problem(loaded, loaded.alpha, device="cpu")

    assert problem.alpha.shape == (2, 3)
    assert problem.lower.shape == (3,)
    assert problem.upper.shape == (3,)
    assert constraints.A_ineq is not None
    assert constraints.A_ineq.shape == (2, 3)
    assert torch.equal(constraints.lower, loaded.factor_exposure_lower)
    assert torch.equal(constraints.upper, loaded.factor_exposure_upper)


def test_case_accepts_singleton_broadcast_batches() -> None:
    case = _case()
    batched_data = BarraRiskData(
        exposures=case.data.exposures,
        factor_covariance=case.data.factor_covariance,
        specific_variance=case.data.specific_variance,
        benchmark=case.data.benchmark.unsqueeze(0),
    )
    broadcast = _replace(case, data=batched_data, current=case.current.unsqueeze(0))

    assert broadcast.batch == 2


def test_case_rejects_incompatible_batches() -> None:
    case = _case()
    with pytest.raises(ValueError, match="batch dimensions are incompatible"):
        _replace(case, current=case.current.repeat(3, 1))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"box_lower": torch.tensor([0.6, 0.6, 0.0], dtype=torch.float64)}, "budget 1.0"),
        ({"box_upper": torch.tensor([0.2, 0.2, 0.2], dtype=torch.float64)}, "budget 1.0"),
        ({"box_lower": torch.tensor([0.9, 0.0, 0.0], dtype=torch.float64)}, "must not exceed"),
        (
            {"factor_exposure_lower": torch.tensor([0.9, -0.4], dtype=torch.float64)},
            "factor_exposure_lower must not exceed",
        ),
        ({"factor_labels": ("only_one",)}, "count must equal factors"),
    ],
)
def test_case_rejects_infeasible_bounds_and_labels(changes, message) -> None:
    with pytest.raises(ValueError, match=message):
        _replace(_case(), **changes)


def test_case_rejects_mixed_float_dtypes() -> None:
    with pytest.raises(TypeError, match="same float dtype"):
        _replace(_case(), alpha=_case().alpha.float())

    with pytest.raises(TypeError, match="sequence of strings"):
        _replace(_case(), factor_labels="ab")


def test_save_is_deterministic_for_same_case_and_metadata(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    metadata = {"source": "cntr", "nested": {"b": 2, "a": 1}}

    first_audit = save_barra_npz(first, _case(), metadata=metadata)
    second_audit = save_barra_npz(second, _case(), metadata=metadata)

    assert first_audit == second_audit


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("exposures", np.zeros((3, 3), dtype=np.float64), "factor_covariance must have shape"),
        ("alpha", np.zeros((2, 2), dtype=np.float64), "alpha must have shape"),
        ("box_lower", np.zeros((3, 1), dtype=np.float64), "box_lower and box_upper"),
        ("factor_exposure_upper", np.zeros(3, dtype=np.float64), "factor exposure bounds"),
    ],
)
def test_load_rejects_wrong_shapes(tmp_path, key, value, message) -> None:
    path = tmp_path / "wrong-shape.npz"
    save_barra_npz(path, _case())
    _rewrite_npz(path, lambda payload: payload.__setitem__(key, value))

    with pytest.raises(BarraNpzError, match=message):
        load_barra_npz(path)


def test_load_rejects_missing_and_extra_keys(tmp_path) -> None:
    path = tmp_path / "keys.npz"
    save_barra_npz(path, _case())
    _rewrite_npz(path, lambda payload: payload.pop("alpha"))
    with pytest.raises(BarraNpzError, match=r"missing=\['alpha'\]"):
        load_barra_npz(path)

    save_barra_npz(path, _case())
    _rewrite_npz(path, lambda payload: payload.__setitem__("unexpected", np.asarray(1)))
    with pytest.raises(BarraNpzError, match=r"extra=\['unexpected'\]"):
        load_barra_npz(path)


@pytest.mark.parametrize("dtype", [np.float16, np.int64, object])
def test_load_rejects_forbidden_dtypes_without_pickle(tmp_path, dtype) -> None:
    path = tmp_path / "dtype.npz"
    save_barra_npz(path, _case())
    _rewrite_npz(path, lambda payload: payload.__setitem__("alpha", payload["alpha"].astype(dtype)))

    with pytest.raises(BarraNpzError, match="failed to read|dtype must"):
        load_barra_npz(path)


def test_load_rejects_nonfinite_and_invalid_covariance(tmp_path) -> None:
    path = tmp_path / "invalid.npz"
    save_barra_npz(path, _case())

    def nonfinite(payload) -> None:
        payload["current"][0] = np.nan

    _rewrite_npz(path, nonfinite)
    with pytest.raises(BarraNpzError, match="current must be finite"):
        load_barra_npz(path)

    save_barra_npz(path, _case())
    _rewrite_npz(
        path,
        lambda payload: payload.__setitem__(
            "factor_covariance", np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
        ),
    )
    with pytest.raises(BarraNpzError, match="positive definite"):
        load_barra_npz(path)


def test_load_rejects_corrupted_array_and_combined_hashes(tmp_path) -> None:
    path = tmp_path / "hash.npz"
    save_barra_npz(path, _case())

    def corrupt_array_hash(payload) -> None:
        manifest = json.loads(payload["manifest_json"].item())
        manifest["arrays"]["alpha"]["sha256"] = "0" * 64
        payload["manifest_json"] = np.asarray(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )

    _rewrite_npz(path, corrupt_array_hash)
    with pytest.raises(BarraNpzError, match="alpha SHA256 mismatch"):
        load_barra_npz(path)

    save_barra_npz(path, _case())

    def corrupt_combined_hash(payload) -> None:
        manifest = json.loads(payload["manifest_json"].item())
        manifest["combined_sha256"] = "f" * 64
        payload["manifest_json"] = np.asarray(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )

    _rewrite_npz(path, corrupt_combined_hash)
    with pytest.raises(BarraNpzError, match="manifest combined_sha256 mismatch"):
        load_barra_npz(path)


def test_load_rejects_metadata_tampering_missing_manifest_and_bad_labels(tmp_path) -> None:
    path = tmp_path / "audit.npz"
    save_barra_npz(path, _case(), metadata={"source": "cntr"})

    def tamper_metadata(payload) -> None:
        metadata = json.loads(payload["metadata_json"].item())
        metadata["user_metadata"]["source"] = "other"
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )

    _rewrite_npz(path, tamper_metadata)
    with pytest.raises(BarraNpzError, match="metadata_sha256 mismatch"):
        load_barra_npz(path)

    save_barra_npz(path, _case())

    def remove_manifest_field(payload) -> None:
        manifest = json.loads(payload["manifest_json"].item())
        manifest.pop("batch")
        payload["manifest_json"] = np.asarray(
            json.dumps(manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )

    _rewrite_npz(path, remove_manifest_field)
    with pytest.raises(BarraNpzError, match=r"missing=\['batch'\]"):
        load_barra_npz(path)

    save_barra_npz(path, _case())

    def remove_label(payload) -> None:
        metadata = json.loads(payload["metadata_json"].item())
        metadata["factor_labels"].pop()
        payload["metadata_json"] = np.asarray(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        )

    _rewrite_npz(path, remove_label)
    with pytest.raises(BarraNpzError, match="factor_labels count"):
        load_barra_npz(path)


def test_save_rejects_non_json_or_nonfinite_metadata(tmp_path) -> None:
    with pytest.raises(BarraNpzError, match="not strict JSON"):
        save_barra_npz(tmp_path / "set.npz", _case(), metadata={"bad": {1, 2}})
    with pytest.raises(BarraNpzError, match="not strict JSON"):
        save_barra_npz(tmp_path / "nan.npz", _case(), metadata={"bad": float("nan")})
