from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import torch

from .barra import BarraRiskData


SCHEMA = "openpo.barra.cntr.npz"
SCHEMA_VERSION = 1
RISK_ARRAY_NAMES = (
    "exposures",
    "factor_covariance",
    "specific_variance",
    "benchmark",
)
CASE_ARRAY_NAMES = (
    "alpha",
    "current",
    "box_lower",
    "box_upper",
    "factor_exposure_lower",
    "factor_exposure_upper",
)
ARRAY_NAMES = RISK_ARRAY_NAMES + CASE_ARRAY_NAMES
NPZ_KEYS = frozenset((*ARRAY_NAMES, "manifest_json", "metadata_json"))
MANIFEST_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "assets",
        "factors",
        "batch",
        "arrays",
        "metadata_sha256",
        "combined_sha256",
    }
)
ARRAY_MANIFEST_KEYS = frozenset({"dtype", "shape", "sha256"})
METADATA_KEYS = frozenset({"factor_labels", "user_metadata"})
ALLOWED_DTYPE_STRINGS = frozenset({"<f4", "<f8"})


class BarraNpzError(ValueError):
    pass


@dataclass(frozen=True)
class BarraInputCase:
    data: BarraRiskData
    alpha: torch.Tensor
    current: torch.Tensor
    box_lower: torch.Tensor
    box_upper: torch.Tensor
    factor_exposure_lower: torch.Tensor
    factor_exposure_upper: torch.Tensor
    factor_labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.data, BarraRiskData):
            raise TypeError("data must be BarraRiskData")
        if isinstance(self.factor_labels, (str, bytes)):
            raise TypeError("factor_labels must be a sequence of strings")
        try:
            labels = tuple(self.factor_labels)
        except TypeError as error:
            raise TypeError("factor_labels must be a sequence of strings") from error
        object.__setattr__(self, "factor_labels", labels)
        _validate_case(self)

    @property
    def assets(self) -> int:
        return self.data.assets

    @property
    def factors(self) -> int:
        return self.data.factors

    @property
    def batch(self) -> int:
        return max(
            _batch_size(self.data.benchmark),
            _batch_size(self.alpha),
            _batch_size(self.current),
        )


@dataclass(frozen=True)
class BarraNpzAudit:
    manifest: dict[str, Any]
    metadata: dict[str, Any]

    @property
    def combined_sha256(self) -> str:
        return str(self.manifest["combined_sha256"])

    @property
    def array_sha256(self) -> dict[str, str]:
        arrays = self.manifest["arrays"]
        return {name: str(arrays[name]["sha256"]) for name in ARRAY_NAMES}


def _batch_size(tensor: torch.Tensor) -> int:
    return 1 if tensor.ndim == 1 else int(tensor.shape[0])


def _case_tensors(case: BarraInputCase) -> dict[str, torch.Tensor]:
    return {
        "exposures": case.data.exposures,
        "factor_covariance": case.data.factor_covariance,
        "specific_variance": case.data.specific_variance,
        "benchmark": case.data.benchmark,
        "alpha": case.alpha,
        "current": case.current,
        "box_lower": case.box_lower,
        "box_upper": case.box_upper,
        "factor_exposure_lower": case.factor_exposure_lower,
        "factor_exposure_upper": case.factor_exposure_upper,
    }


def _require_asset_vector_or_batch(name: str, tensor: torch.Tensor, assets: int) -> None:
    if tensor.ndim == 1 and tensor.shape == (assets,):
        return
    if tensor.ndim == 2 and tensor.shape[0] > 0 and tensor.shape[1] == assets:
        return
    raise ValueError(f"{name} must have shape [assets] or [batch, assets]")


def _validate_case(case: BarraInputCase) -> None:
    tensors = _case_tensors(case)
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if tensor.dtype not in (torch.float32, torch.float64):
            raise TypeError(f"{name} dtype must be float32 or float64")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must be finite")
    if len({tensor.dtype for tensor in tensors.values()}) != 1:
        raise TypeError("all Barra input arrays must use the same float dtype")

    assets = case.data.assets
    factors = case.data.factors
    _require_asset_vector_or_batch("alpha", case.alpha, assets)
    _require_asset_vector_or_batch("current", case.current, assets)
    batch_sizes = {
        "benchmark": _batch_size(case.data.benchmark),
        "alpha": _batch_size(case.alpha),
        "current": _batch_size(case.current),
    }
    batch = max(batch_sizes.values())
    incompatible = {name: size for name, size in batch_sizes.items() if size not in (1, batch)}
    if incompatible:
        raise ValueError(f"benchmark, alpha, and current batch dimensions are incompatible: {batch_sizes}")

    if case.box_lower.shape != (assets,) or case.box_upper.shape != (assets,):
        raise ValueError("box_lower and box_upper must have shape [assets]")
    if torch.any(case.box_lower > case.box_upper).item():
        raise ValueError("box_lower must not exceed box_upper")
    lower_budget = float(case.box_lower.sum().item())
    upper_budget = float(case.box_upper.sum().item())
    if lower_budget > 1.0 or upper_budget < 1.0:
        raise ValueError(
            "box bounds are infeasible for budget 1.0: "
            f"sum(lower)={lower_budget}, sum(upper)={upper_budget}"
        )

    if case.factor_exposure_lower.shape != (factors,) or case.factor_exposure_upper.shape != (
        factors,
    ):
        raise ValueError("factor exposure bounds must have shape [factors]")
    if torch.any(case.factor_exposure_lower > case.factor_exposure_upper).item():
        raise ValueError("factor_exposure_lower must not exceed factor_exposure_upper")
    if len(case.factor_labels) != factors:
        raise ValueError("factor_labels count must equal factors")
    if any(not isinstance(label, str) or not label for label in case.factor_labels):
        raise ValueError("factor_labels must contain non-empty strings")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json_object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise BarraNpzError(f"value is not strict JSON: {error}") from error


def _parse_json_object(text: str, name: str) -> dict[str, Any]:
    try:
        value = json.loads(
            text,
            object_pairs_hook=_strict_json_object_pairs,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError) as error:
        raise BarraNpzError(f"{name} must contain valid strict JSON: {error}") from error
    if not isinstance(value, dict):
        raise BarraNpzError(f"{name} must contain a JSON object")
    if text != _canonical_json(value):
        raise BarraNpzError(f"{name} must use canonical JSON encoding")
    return value


def _normalize_metadata(
    factor_labels: Sequence[str], metadata: Mapping[str, Any] | None
) -> tuple[dict[str, Any], str]:
    if metadata is None:
        user_metadata: dict[str, Any] = {}
    elif not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    else:
        user_metadata = dict(metadata)
    payload = {
        "factor_labels": list(factor_labels),
        "user_metadata": user_metadata,
    }
    canonical = _canonical_json(payload)
    return _parse_json_object(canonical, "metadata_json"), canonical


def _validate_metadata(payload: dict[str, Any]) -> tuple[str, ...]:
    _require_exact_keys(set(payload), METADATA_KEYS, "metadata")
    labels = payload["factor_labels"]
    if not isinstance(labels, list) or any(not isinstance(label, str) or not label for label in labels):
        raise BarraNpzError("metadata factor_labels must be an array of non-empty strings")
    if not isinstance(payload["user_metadata"], dict):
        raise BarraNpzError("metadata user_metadata must be an object")
    return tuple(labels)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    descriptor = _canonical_json(
        {"dtype": array.dtype.str, "shape": [int(size) for size in array.shape]}
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(descriptor)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _array_manifest(array: np.ndarray) -> dict[str, Any]:
    return {
        "dtype": array.dtype.str,
        "shape": [int(size) for size in array.shape],
        "sha256": _array_sha256(array),
    }


def _combined_sha256(manifest_without_combined: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_json(dict(manifest_without_combined)).encode("utf-8"))


def _require_exact_keys(actual: set[str], expected: frozenset[str], name: str) -> None:
    if actual == expected:
        return
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    raise BarraNpzError(f"{name} key set mismatch: missing={missing}, extra={extra}")


def _case_arrays(case: BarraInputCase) -> dict[str, np.ndarray]:
    if not isinstance(case, BarraInputCase):
        raise TypeError("case must be BarraInputCase")
    _validate_case(case)
    return {
        name: np.array(tensor.detach().cpu().contiguous().numpy(), copy=True, order="C")
        for name, tensor in _case_tensors(case).items()
    }


def _validate_array_dtypes(arrays: Mapping[str, np.ndarray]) -> None:
    for name, array in arrays.items():
        if array.dtype.str not in ALLOWED_DTYPE_STRINGS:
            raise BarraNpzError(f"{name} dtype must be little-endian float32 or float64")
    if len({array.dtype.str for array in arrays.values()}) != 1:
        raise BarraNpzError("all Barra input arrays must use the same float dtype")


def _case_from_arrays(
    arrays: Mapping[str, np.ndarray], factor_labels: tuple[str, ...]
) -> BarraInputCase:
    _validate_array_dtypes(arrays)
    tensors = {name: torch.from_numpy(array) for name, array in arrays.items()}
    try:
        data = BarraRiskData(
            exposures=tensors["exposures"],
            factor_covariance=tensors["factor_covariance"],
            specific_variance=tensors["specific_variance"],
            benchmark=tensors["benchmark"],
        )
        return BarraInputCase(
            data=data,
            alpha=tensors["alpha"],
            current=tensors["current"],
            box_lower=tensors["box_lower"],
            box_upper=tensors["box_upper"],
            factor_exposure_lower=tensors["factor_exposure_lower"],
            factor_exposure_upper=tensors["factor_exposure_upper"],
            factor_labels=factor_labels,
        )
    except (TypeError, ValueError) as error:
        raise BarraNpzError(f"invalid Barra input case: {error}") from error


def _build_manifest(
    arrays: Mapping[str, np.ndarray], metadata_json: str, *, batch: int
) -> dict[str, Any]:
    exposures = arrays["exposures"]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "assets": int(exposures.shape[0]),
        "factors": int(exposures.shape[1]),
        "batch": batch,
        "arrays": {name: _array_manifest(arrays[name]) for name in ARRAY_NAMES},
        "metadata_sha256": _sha256_bytes(metadata_json.encode("utf-8")),
    }
    manifest["combined_sha256"] = _combined_sha256(manifest)
    return manifest


def _validate_manifest(
    manifest: dict[str, Any],
    arrays: Mapping[str, np.ndarray],
    metadata_json: str,
    *,
    batch: int,
) -> None:
    _require_exact_keys(set(manifest), MANIFEST_KEYS, "manifest")
    if manifest["schema"] != SCHEMA:
        raise BarraNpzError(f"manifest schema must be {SCHEMA!r}")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise BarraNpzError(f"manifest schema_version must be {SCHEMA_VERSION}")
    for name in ("assets", "factors", "batch"):
        if type(manifest[name]) is not int:
            raise BarraNpzError(f"manifest {name} must be an integer")
    array_entries = manifest["arrays"]
    if not isinstance(array_entries, dict):
        raise BarraNpzError("manifest arrays must be an object")
    _require_exact_keys(set(array_entries), frozenset(ARRAY_NAMES), "manifest arrays")
    for name in ARRAY_NAMES:
        entry = array_entries[name]
        if not isinstance(entry, dict):
            raise BarraNpzError(f"manifest arrays.{name} must be an object")
        _require_exact_keys(set(entry), ARRAY_MANIFEST_KEYS, f"manifest arrays.{name}")

    expected = _build_manifest(arrays, metadata_json, batch=batch)
    for name in ARRAY_NAMES:
        actual_entry = array_entries[name]
        expected_entry = expected["arrays"][name]
        if actual_entry["dtype"] != expected_entry["dtype"]:
            raise BarraNpzError(f"{name} manifest dtype mismatch")
        if actual_entry["shape"] != expected_entry["shape"]:
            raise BarraNpzError(f"{name} manifest shape mismatch")
        if actual_entry["sha256"] != expected_entry["sha256"]:
            raise BarraNpzError(f"{name} SHA256 mismatch")
    for name in ("assets", "factors", "batch", "metadata_sha256", "combined_sha256"):
        if manifest[name] != expected[name]:
            raise BarraNpzError(f"manifest {name} mismatch")


def _read_json_scalar(archive: Any, key: str) -> str:
    value = archive[key]
    if value.ndim != 0 or value.dtype.kind != "U":
        raise BarraNpzError(f"{key} must be a scalar Unicode array")
    return str(value.item())


def save_barra_npz(
    path: str | os.PathLike[str],
    case: BarraInputCase,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> BarraNpzAudit:
    destination = Path(path)
    arrays = _case_arrays(case)
    metadata_payload, metadata_json = _normalize_metadata(case.factor_labels, metadata)
    manifest = _build_manifest(arrays, metadata_json, batch=case.batch)
    manifest_json = _canonical_json(manifest)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez(
                temporary,
                **arrays,
                manifest_json=np.asarray(manifest_json),
                metadata_json=np.asarray(metadata_json),
            )
            temporary.flush()
            os.fsync(temporary.fileno())
        temporary_path.replace(destination)
    except OSError:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return BarraNpzAudit(manifest=manifest, metadata=metadata_payload)


def load_barra_npz(
    path: str | os.PathLike[str],
    *,
    return_audit: bool = False,
) -> BarraInputCase | tuple[BarraInputCase, BarraNpzAudit]:
    source = Path(path)
    try:
        with np.load(source, allow_pickle=False) as archive:
            keys = tuple(archive.files)
            if len(keys) != len(set(keys)):
                raise BarraNpzError("NPZ archive contains duplicate keys")
            _require_exact_keys(set(keys), NPZ_KEYS, "NPZ archive")
            arrays = {
                name: np.array(archive[name], copy=True, order="C") for name in ARRAY_NAMES
            }
            manifest_json = _read_json_scalar(archive, "manifest_json")
            metadata_json = _read_json_scalar(archive, "metadata_json")
    except BarraNpzError:
        raise
    except (OSError, ValueError) as error:
        raise BarraNpzError(f"failed to read Barra NPZ {source}: {error}") from error

    metadata_payload = _parse_json_object(metadata_json, "metadata_json")
    factor_labels = _validate_metadata(metadata_payload)
    case = _case_from_arrays(arrays, factor_labels)
    manifest = _parse_json_object(manifest_json, "manifest_json")
    _validate_manifest(manifest, arrays, metadata_json, batch=case.batch)
    audit = BarraNpzAudit(manifest=manifest, metadata=metadata_payload)
    if return_audit:
        return case, audit
    return case
