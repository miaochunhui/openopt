from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_SEED = 20260809


@dataclass(frozen=True)
class ConstrainedCase:
    name: str
    seed: int
    assets: int
    industries: int
    styles: int
    budget: float
    long_short: bool
    feasible_weights: np.ndarray
    box_lower: np.ndarray
    box_upper: np.ndarray
    industry_loadings: np.ndarray
    industry_lower: np.ndarray
    industry_upper: np.ndarray
    style_loadings: np.ndarray
    style_lower: np.ndarray
    style_upper: np.ndarray
    market_loading: np.ndarray
    market_target: float = 0.0

    @property
    def equality_matrix(self) -> np.ndarray:
        return np.vstack(
            [
                np.ones(self.assets, dtype=np.float64),
                self.market_loading,
            ]
        )

    @property
    def equality_target(self) -> np.ndarray:
        return np.array([self.budget, self.market_target], dtype=np.float64)

    @property
    def inequality_matrix(self) -> np.ndarray:
        return np.vstack([self.industry_loadings, self.style_loadings])

    @property
    def inequality_lower(self) -> np.ndarray:
        return np.concatenate([self.industry_lower, self.style_lower])

    @property
    def inequality_upper(self) -> np.ndarray:
        return np.concatenate([self.industry_upper, self.style_upper])

    def input_sha256(self) -> str:
        digest = hashlib.sha256()
        _update_digest(digest, "name", self.name)
        _update_digest(digest, "seed", self.seed)
        _update_digest(digest, "assets", self.assets)
        _update_digest(digest, "industries", self.industries)
        _update_digest(digest, "styles", self.styles)
        _update_digest(digest, "budget", self.budget)
        _update_digest(digest, "long_short", self.long_short)
        for field_name, values in (
            ("feasible_weights", self.feasible_weights),
            ("box_lower", self.box_lower),
            ("box_upper", self.box_upper),
            ("industry_loadings", self.industry_loadings),
            ("industry_lower", self.industry_lower),
            ("industry_upper", self.industry_upper),
            ("style_loadings", self.style_loadings),
            ("style_lower", self.style_lower),
            ("style_upper", self.style_upper),
            ("market_loading", self.market_loading),
            ("equality_target", self.equality_target),
        ):
            _update_digest(digest, field_name, values)
        return digest.hexdigest()

    def residual_report(self, weights: np.ndarray | None = None) -> dict[str, float]:
        vector = self.feasible_weights if weights is None else np.asarray(weights, dtype=np.float64)
        equality_residual = self.equality_matrix @ vector - self.equality_target
        inequality_values = self.inequality_matrix @ vector
        return {
            "budget_error": float(abs(vector.sum() - self.budget)),
            "market_error": float(abs(float(self.market_loading @ vector) - self.market_target)),
            "max_box_lower_violation": float(np.maximum(self.box_lower - vector, 0.0).max()),
            "max_box_upper_violation": float(np.maximum(vector - self.box_upper, 0.0).max()),
            "max_inequality_lower_violation": float(
                np.maximum(self.inequality_lower - inequality_values, 0.0).max()
            ),
            "max_inequality_upper_violation": float(
                np.maximum(inequality_values - self.inequality_upper, 0.0).max()
            ),
            "max_equality_error": float(np.abs(equality_residual).max()),
            "min_box_slack": float(np.minimum(vector - self.box_lower, self.box_upper - vector).min()),
            "min_inequality_slack": float(
                np.minimum(
                    inequality_values - self.inequality_lower,
                    self.inequality_upper - inequality_values,
                ).min()
            ),
        }

    def to_numpy_bundle(self) -> dict[str, np.ndarray]:
        return {
            "name": np.array(self.name),
            "seed": np.array(self.seed, dtype=np.int64),
            "assets": np.array(self.assets, dtype=np.int64),
            "industries": np.array(self.industries, dtype=np.int64),
            "styles": np.array(self.styles, dtype=np.int64),
            "budget": np.array(self.budget, dtype=np.float64),
            "long_short": np.array(self.long_short, dtype=np.bool_),
            "feasible_weights": self.feasible_weights,
            "box_lower": self.box_lower,
            "box_upper": self.box_upper,
            "industry_loadings": self.industry_loadings,
            "industry_lower": self.industry_lower,
            "industry_upper": self.industry_upper,
            "style_loadings": self.style_loadings,
            "style_lower": self.style_lower,
            "style_upper": self.style_upper,
            "market_loading": self.market_loading,
            "market_target": np.array(self.market_target, dtype=np.float64),
            "equality_matrix": self.equality_matrix,
            "equality_target": self.equality_target,
            "inequality_matrix": self.inequality_matrix,
            "inequality_lower": self.inequality_lower,
            "inequality_upper": self.inequality_upper,
            "input_sha256": np.array(self.input_sha256()),
        }


def _update_digest(digest: hashlib._Hash, field_name: str, value: np.ndarray | int | float | bool | str) -> None:
    digest.update(field_name.encode("utf-8"))
    digest.update(b"\0")
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode("utf-8"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
        return
    if isinstance(value, str):
        digest.update(value.encode("utf-8"))
        return
    if isinstance(value, bool):
        digest.update(np.asarray([value], dtype=np.bool_).tobytes())
        return
    if isinstance(value, int):
        digest.update(np.asarray([value], dtype=np.int64).tobytes())
        return
    if isinstance(value, float):
        digest.update(np.asarray([value], dtype=np.float64).tobytes())
        return
    raise TypeError(f"unsupported digest value type: {type(value)!r}")


def _make_feasible_weights(rng: np.random.Generator, *, assets: int, long_short: bool) -> np.ndarray:
    if long_short:
        long_count = (assets + 1) // 2
        short_count = assets - long_count
        positive = rng.uniform(0.03, 0.07, size=long_count)
        if short_count == 0:
            weights = positive - positive.sum() / positive.size
        else:
            negative = rng.uniform(0.03, 0.07, size=short_count)
            negative *= positive.sum() / negative.sum()
            weights = np.concatenate([positive, -negative])
            rng.shuffle(weights)
            weights[-1] -= weights.sum()
    else:
        base = rng.uniform(0.04, 0.09, size=assets)
        weights = base / base.sum()
    return np.asarray(weights, dtype=np.float64)


def _orthogonal_market_loading(rng: np.random.Generator, weights: np.ndarray) -> np.ndarray:
    loading = rng.normal(0.0, 1.0, size=weights.shape[0])
    loading -= loading.mean()
    norm = float(np.linalg.norm(loading))
    if norm == 0.0:
        loading = np.zeros_like(weights)
        loading[0] = 1.0
        loading[1] = -1.0
    loading = loading - (loading @ weights) / float(weights @ weights) * weights
    norm = float(np.linalg.norm(loading))
    if norm == 0.0:
        loading = np.zeros_like(weights)
        loading[0] = 1.0
        loading[1] = -1.0
        loading = loading - (loading @ weights) / float(weights @ weights) * weights
        norm = float(np.linalg.norm(loading))
    return loading / norm


def _make_industry_loadings(assets: int, industries: int) -> np.ndarray:
    membership = np.arange(assets, dtype=np.int64) % industries
    rows = np.arange(industries, dtype=np.int64)[:, None]
    return (membership[None, :] == rows).astype(np.float64)


def _make_style_loadings(rng: np.random.Generator, assets: int, styles: int) -> np.ndarray:
    loadings = rng.normal(0.0, 0.35, size=(styles, assets))
    loadings -= loadings.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(loadings, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return loadings / norms


def build_case(
    *,
    seed: int = DEFAULT_SEED,
    long_short: bool = False,
    assets: int = 16,
    industries: int = 4,
    styles: int = 3,
) -> ConstrainedCase:
    if assets < 4:
        raise ValueError("assets must be at least 4")
    if industries < 2:
        raise ValueError("industries must be at least 2")
    if styles < 1:
        raise ValueError("styles must be at least 1")

    derived_seed = seed + 1_000_003 * int(long_short) + 97 * assets + 17 * industries + 7 * styles
    rng = np.random.default_rng(derived_seed)
    budget = 0.0 if long_short else 1.0
    feasible_weights = _make_feasible_weights(rng, assets=assets, long_short=long_short)
    if not long_short:
        feasible_weights *= budget / float(feasible_weights.sum())
    box_slack = rng.uniform(0.015, 0.025, size=assets)
    box_lower = feasible_weights - box_slack
    box_upper = feasible_weights + box_slack
    if not long_short:
        box_lower = np.maximum(box_lower, 0.0)

    industry_loadings = _make_industry_loadings(assets, industries)
    style_loadings = _make_style_loadings(rng, assets, styles)

    industry_reference = industry_loadings @ feasible_weights
    style_reference = style_loadings @ feasible_weights
    industry_margin = rng.uniform(0.04, 0.07, size=industries)
    style_margin = rng.uniform(0.03, 0.06, size=styles)
    industry_lower = industry_reference - industry_margin
    industry_upper = industry_reference + industry_margin
    style_lower = style_reference - style_margin
    style_upper = style_reference + style_margin
    market_loading = _orthogonal_market_loading(rng, feasible_weights)

    name = "long_short_market_neutral" if long_short else "long_only_budgeted"
    case = ConstrainedCase(
        name=name,
        seed=seed,
        assets=assets,
        industries=industries,
        styles=styles,
        budget=budget,
        long_short=long_short,
        feasible_weights=feasible_weights,
        box_lower=np.asarray(box_lower, dtype=np.float64),
        box_upper=np.asarray(box_upper, dtype=np.float64),
        industry_loadings=np.asarray(industry_loadings, dtype=np.float64),
        industry_lower=np.asarray(industry_lower, dtype=np.float64),
        industry_upper=np.asarray(industry_upper, dtype=np.float64),
        style_loadings=np.asarray(style_loadings, dtype=np.float64),
        style_lower=np.asarray(style_lower, dtype=np.float64),
        style_upper=np.asarray(style_upper, dtype=np.float64),
        market_loading=np.asarray(market_loading, dtype=np.float64),
    )
    residuals = case.residual_report()
    if residuals["budget_error"] > 1e-12 or residuals["market_error"] > 1e-12:
        raise RuntimeError(
            "constructed case is not exactly feasible: "
            f"budget_error={residuals['budget_error']:.3e}, market_error={residuals['market_error']:.3e}"
        )
    if residuals["min_box_slack"] <= 0.0 or residuals["min_inequality_slack"] <= 0.0:
        raise RuntimeError("constructed case is not strictly feasible")
    return case


def build_suite(seed: int = DEFAULT_SEED) -> tuple[ConstrainedCase, ConstrainedCase]:
    return (
        build_case(seed=seed, long_short=False),
        build_case(seed=seed, long_short=True),
    )


def save_case(case: ConstrainedCase, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{case.name}.npz"
    np.savez(output_path, **case.to_numpy_bundle())
    return output_path


def save_suite(output_dir: Path, seed: int = DEFAULT_SEED) -> list[dict[str, object]]:
    manifests: list[dict[str, object]] = []
    for case in build_suite(seed):
        output_path = save_case(case, output_dir)
        manifests.append(
            {
                "name": case.name,
                "path": str(output_path),
                "input_sha256": case.input_sha256(),
                "seed": case.seed,
                "budget": case.budget,
                "long_short": case.long_short,
                "residuals": case.residual_report(),
            }
        )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifests, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark-results/constrained-cases"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = save_suite(args.output_dir, seed=args.seed)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
