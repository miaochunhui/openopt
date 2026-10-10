"""Tests for benchmarks/mpo_t1_benchmark.py (SPEC section 9).

No real backtests are run: ``run_seed`` is replaced through the module's
RUN_SEED_OVERRIDE hook, and sibling modules (stats/report) are injected or
blanked via sys.modules. All runs use --jobs 1 (inline execution path).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "benchmarks"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mpo_t1_benchmark as bench  # noqa: E402
from openpo.mpo.config import ExperimentConfig, config_hash  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRunSeed:
    """Callable stand-in for openpo.mpo.backtest.run_seed with call ledger."""

    def __init__(self):
        self.calls: list[tuple[int, str]] = []

    def __call__(self, seed, cfg):
        self.calls.append((seed, config_hash(cfg)))
        return {
            "mpo": {"net_sharpe": 1.0 + seed % 7 * 0.1, "turnover": 0.5},
            "fixed_weight": {"net_sharpe": 0.8, "turnover": 0.4},
            "buy_and_hold": {"net_sharpe": 0.2, "turnover": 0.0},
            "seed": seed,
        }


class FakeRunSeedLambdas(FakeRunSeed):
    """FakeRunSeed that also records the fixed_lambdas kwarg it was handed."""

    def __init__(self):
        super().__init__()
        self.lambda_calls: list = []

    def __call__(self, seed, cfg, fixed_lambdas=None):
        self.lambda_calls.append(None if fixed_lambdas is None else list(fixed_lambdas))
        return super().__call__(seed, cfg)


class FakeRunSeedRoster(FakeRunSeed):
    """FakeRunSeed that records the ``strategies`` roster and honours it.

    The real ``run_seed`` returns one entry per requested strategy, so the fake
    has to as well -- otherwise the artifact's roster block would be right for
    the wrong reason.
    """

    def __init__(self):
        super().__init__()
        self.roster_calls: list = []

    def __call__(self, seed, cfg, strategies=None, fixed_lambdas=None):
        self.roster_calls.append(None if strategies is None else list(strategies))
        base = super().__call__(seed, cfg)
        if strategies is None:
            return base
        out = {"seed": seed}
        for i, name in enumerate(strategies):
            out[name] = {"net_sharpe": 1.0 - 0.1 * i + 0.01 * (seed % 5), "turnover": 0.5}
        return out


class FakeTuneScorer:
    """Deterministic stand-in for the real (seed, cfg, lambdas) -> score worker.

    Score = dot(lambdas, coeffs) + a fixed per-seed offset, so the argmax over
    a simplex lattice is the corner with the largest coefficient and no QP is
    ever solved.
    """

    def __init__(self, coeffs=(0.1, 0.9, 0.3)):
        self.coeffs = coeffs
        self.calls: list[tuple[int, tuple]] = []

    def __call__(self, seed, cfg, lambdas):
        self.calls.append((seed, tuple(float(x) for x in lambdas)))
        base = sum(c * float(lam) for c, lam in zip(self.coeffs, lambdas))
        return base + 0.001 * (seed % 3)


def make_args(**overrides):
    argv = ["--mode", overrides.pop("mode", "smoke")]
    for key, val in overrides.items():
        flag = "--" + key.replace("_", "-")
        if val is True:
            argv.append(flag)
        elif isinstance(val, list):
            for v in val:
                argv.extend([flag, str(v)])
        else:
            argv.extend([flag, str(val)])
    return bench.build_parser().parse_args(argv)


@pytest.fixture()
def fake_run_seed(monkeypatch):
    fake = FakeRunSeed()
    monkeypatch.setattr(bench, "RUN_SEED_OVERRIDE", fake)
    return fake


@pytest.fixture()
def fake_run_seed_lambdas(monkeypatch):
    fake = FakeRunSeedLambdas()
    monkeypatch.setattr(bench, "RUN_SEED_OVERRIDE", fake)
    return fake


@pytest.fixture()
def fake_run_seed_roster(monkeypatch):
    fake = FakeRunSeedRoster()
    monkeypatch.setattr(bench, "RUN_SEED_OVERRIDE", fake)
    return fake


@pytest.fixture()
def fake_tune_scorer(monkeypatch):
    scorer = FakeTuneScorer()
    monkeypatch.setattr(bench, "TUNE_SCORE_OVERRIDE", scorer)
    return scorer


@pytest.fixture()
def no_sibling_modules(monkeypatch):
    """Force ImportError for stats/report regardless of what siblings shipped."""
    monkeypatch.setitem(sys.modules, "openpo.mpo.stats", None)
    monkeypatch.setitem(sys.modules, "openpo.mpo.report", None)


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------


def test_parser_defaults():
    args = bench.build_parser().parse_args(["--mode", "main"])
    assert args.mode == "main"
    assert args.seeds is None
    assert args.jobs == 1
    assert args.output is None
    assert args.html_output is None
    assert args.axis is None
    assert args.resume is False


def test_parser_mode_required_and_validated():
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args([])
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--mode", "bogus"])


def test_parser_axis_repeatable_and_validated():
    args = bench.build_parser().parse_args(
        ["--mode", "sensitivity", "--axis", "cost", "--axis", "t1"]
    )
    assert args.axis == ["cost", "t1"]
    args = bench.build_parser().parse_args(
        ["--mode", "sensitivity", "--axis", "gamma", "--axis", "exec"]
    )
    assert args.axis == ["gamma", "exec"]
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--mode", "sensitivity", "--axis", "nope"])


def test_parser_full_flag_set(tmp_path):
    args = make_args(
        mode="sensitivity",
        seeds=3,
        jobs=4,
        output=tmp_path / "a.json",
        html_output=tmp_path / "a.html",
        axis=["decay"],
        resume=True,
    )
    assert args.seeds == 3
    assert args.jobs == 4
    assert args.resume is True
    assert args.html_output == tmp_path / "a.html"


# ---------------------------------------------------------------------------
# Mode -> config construction
# ---------------------------------------------------------------------------


def test_smoke_config_matches_spec():
    cfg = bench.build_smoke_config()
    assert cfg.n_seeds == 2
    assert cfg.synth.n_assets == 100
    assert cfg.synth.n_days == 20
    assert cfg.mpo.calib_days == 10
    assert cfg.eval_start_day == 10


def test_main_config_is_frozen_defaults():
    cfg = bench.build_main_config()
    assert cfg == ExperimentConfig()
    assert cfg.n_seeds == 24


def test_sensitivity_cells_cover_spec_grids():
    cells = bench.build_sensitivity_cells(sorted(bench.SENSITIVITY_AXES))
    by_axis = {}
    for c in cells:
        by_axis.setdefault(c["axis"], []).append(c["value"])
    assert by_axis["cost"] == [0.5, 1.0, 2.0]
    assert by_axis["ic"] == [0.5, 1.0, 1.5]
    assert by_axis["decay"] == [0.5, 1.0, 2.0]
    assert by_axis["stages"] == [1, 2, 4]
    assert by_axis["t1"] == [True, False]
    assert by_axis["gamma"] == [2.0, 5.0, 10.0, 20.0]
    assert by_axis["exec"] == [30, 60]
    assert all(c["config"].n_seeds == 8 for c in cells)
    assert set(by_axis) == set(bench.SENSITIVITY_AXES)


def test_sensitivity_cells_replace_correct_field():
    cells = {c["cell_id"]: c["config"] for c in bench.build_sensitivity_cells(["cost", "ic", "t1"])}
    base = ExperimentConfig()
    assert cells["cost=2.0"].mpo.cost_mult == 2.0
    assert cells["cost=2.0"].synth == base.synth  # only the targeted field moves
    assert cells["ic=1.5"].synth.ic_mult == 1.5
    assert cells["ic=1.5"].mpo == base.mpo
    assert cells["t1=False"].mpo.t1_enabled is False
    # base config must remain untouched (dataclasses.replace, no mutation)
    assert ExperimentConfig() == base


def test_gamma_axis_replaces_only_mpo_gamma():
    base = ExperimentConfig()
    assert base.mpo.gamma == 10.0  # frozen default sits inside the grid
    cells = {c["cell_id"]: c for c in bench.build_sensitivity_cells(["gamma"])}
    assert list(cells) == ["gamma=2.0", "gamma=5.0", "gamma=10.0", "gamma=20.0"]
    for cell_id, gamma in (
        ("gamma=2.0", 2.0),
        ("gamma=5.0", 5.0),
        ("gamma=10.0", 10.0),
        ("gamma=20.0", 20.0),
    ):
        cfg = cells[cell_id]["config"]
        assert cfg.mpo.gamma == gamma
        assert cells[cell_id]["value"] == gamma
        assert cfg.synth == base.synth  # only the targeted field moves
        assert cfg.mpo == dataclasses.replace(base.mpo, gamma=gamma)
    # the gamma=10.0 cell is the base config -> shares the base config hash
    assert config_hash(cells["gamma=10.0"]["config"]) == config_hash(
        dataclasses.replace(base, n_seeds=bench.SENSITIVITY_SEEDS_PER_CELL)
    )
    assert ExperimentConfig() == base


def test_exec_axis_replaces_only_mpo_exec_minutes():
    base = ExperimentConfig()
    assert base.mpo.exec_minutes == 30  # frozen default sits inside the grid
    cells = {c["cell_id"]: c for c in bench.build_sensitivity_cells(["exec"])}
    assert list(cells) == ["exec=30", "exec=60"]
    for cell_id, minutes in (("exec=30", 30), ("exec=60", 60)):
        cfg = cells[cell_id]["config"]
        assert cfg.mpo.exec_minutes == minutes
        assert cells[cell_id]["value"] == minutes
        assert cfg.synth == base.synth
        assert cfg.mpo == dataclasses.replace(base.mpo, exec_minutes=minutes)
    # stage-grid A/B only touches S1 length: epochs/horizon/max_stages frozen
    wide = cells["exec=60"]["config"].mpo
    assert (wide.horizon_minutes, wide.max_stages) == (
        base.mpo.horizon_minutes,
        base.mpo.max_stages,
    )
    # 60 trading minutes still fits inside a 240-minute session
    assert wide.exec_minutes < base.synth.minutes_per_day
    # the exec=30 cell is the base config -> shares the base config hash
    assert config_hash(cells["exec=30"]["config"]) == config_hash(
        dataclasses.replace(base, n_seeds=bench.SENSITIVITY_SEEDS_PER_CELL)
    )
    assert ExperimentConfig() == base


def test_sensitivity_axis_filter():
    cells = bench.build_sensitivity_cells(["decay"])
    assert [c["axis"] for c in cells] == ["decay"] * 3
    assert [c["axis"] for c in bench.build_sensitivity_cells(["gamma"])] == ["gamma"] * 4
    assert [c["axis"] for c in bench.build_sensitivity_cells(["exec"])] == ["exec"] * 2


def test_seeds_are_seed_base_offsets():
    cfg = bench.build_smoke_config()
    assert bench.seeds_for(cfg, None) == [cfg.seed_base, cfg.seed_base + 1]
    assert bench.seeds_for(cfg, 3) == [cfg.seed_base + i for i in range(3)]


# ---------------------------------------------------------------------------
# Atomic write / sha256 helpers
# ---------------------------------------------------------------------------


def test_atomic_write_json_handles_numpy_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "sub" / "x.json"
    bench.atomic_write_json(path, {"a": np.float64(1.5), "b": np.array([1, 2])})
    assert json.loads(path.read_text()) == {"a": 1.5, "b": [1, 2]}
    assert not list(path.parent.glob("*.tmp"))


def test_sha256_sidecar_matches_content(tmp_path):
    path = tmp_path / "x.json"
    bench.atomic_write_json(path, {"k": 1})
    digest = bench.write_sha256_sidecar(path)
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    text = (Path(str(path) + ".sha256")).read_text()
    assert text == f"{digest}  x.json\n"


# ---------------------------------------------------------------------------
# End-to-end smoke run with fake run_seed
# ---------------------------------------------------------------------------


def test_smoke_run_writes_checkpoints_and_artifact(tmp_path, fake_run_seed, no_sibling_modules):
    out = tmp_path / "results" / "mpo-t1-smoke.json"
    artifact = bench.run_benchmark(make_args(mode="smoke", output=out))

    cfg = bench.build_smoke_config()
    h8 = config_hash(cfg)[:8]
    seeds = [cfg.seed_base, cfg.seed_base + 1]
    assert sorted(s for s, _ in fake_run_seed.calls) == seeds

    ckpt_dir = out.parent / "mpo-t1-smoke-checkpoints"
    for seed in seeds:
        ckpt = ckpt_dir / f"ckpt-{h8}-seed{seed}.json"
        assert ckpt.exists()
        payload = json.loads(ckpt.read_text())
        assert payload["seed"] == seed
        assert payload["config_hash"] == config_hash(cfg)
        assert payload["result"]["seed"] == seed

    assert out.exists()
    data = json.loads(out.read_text())
    assert data["schema"] == "mpo-t1.synthetic.v1"
    assert data["mode"] == "smoke"
    assert data["n_cells"] == 1
    cell = data["cells"][0]
    assert cell["config_hash"] == config_hash(cfg)
    assert cell["seeds"] == seeds
    assert set(cell["per_seed"]) == {str(s) for s in seeds}
    assert sorted(cell["strategies"]) == ["buy_and_hold", "fixed_weight", "mpo"]
    assert cell["config"]["synth"]["n_assets"] == 100
    # env + provenance
    assert data["env"]["python"] == sys.version.split()[0]
    assert data["env"]["numpy"] is not None
    assert "hostname" in data["env"]
    assert "git_rev" in data
    assert data["wall_clock_seconds"] >= 0
    # sha256 sidecar present and correct
    digest = Path(str(out) + ".sha256").read_text().split()[0]
    assert digest == hashlib.sha256(out.read_bytes()).hexdigest()
    # no temp leftovers from the atomic write
    assert not list(out.parent.glob("*.tmp"))
    assert artifact["_output_path"] == str(out)


def test_seeds_flag_overrides_seed_count(tmp_path, fake_run_seed, no_sibling_modules):
    out = tmp_path / "r.json"
    bench.run_benchmark(make_args(mode="smoke", output=out, seeds=1))
    assert len(fake_run_seed.calls) == 1


# ---------------------------------------------------------------------------
# Resume / checkpoint semantics
# ---------------------------------------------------------------------------


def test_resume_skips_completed_tasks(tmp_path, fake_run_seed, no_sibling_modules, monkeypatch):
    out = tmp_path / "r.json"
    bench.run_benchmark(make_args(mode="smoke", output=out))
    assert len(fake_run_seed.calls) == 2

    fresh = FakeRunSeed()
    monkeypatch.setattr(bench, "RUN_SEED_OVERRIDE", fresh)
    artifact = bench.run_benchmark(make_args(mode="smoke", output=out, resume=True))
    assert fresh.calls == []  # everything served from checkpoints
    assert artifact["cells"][0]["seeds"] == [
        bench.build_smoke_config().seed_base + i for i in range(2)
    ]


def test_without_resume_recomputes(tmp_path, fake_run_seed, no_sibling_modules):
    out = tmp_path / "r.json"
    bench.run_benchmark(make_args(mode="smoke", output=out))
    bench.run_benchmark(make_args(mode="smoke", output=out))
    assert len(fake_run_seed.calls) == 4


def test_resume_recomputes_corrupt_checkpoint(tmp_path, fake_run_seed, no_sibling_modules):
    out = tmp_path / "r.json"
    bench.run_benchmark(make_args(mode="smoke", output=out))
    cfg = bench.build_smoke_config()
    h8 = config_hash(cfg)[:8]
    bad = out.parent / "r-checkpoints" / f"ckpt-{h8}-seed{cfg.seed_base}.json"
    bad.write_text("{not json")
    n_before = len(fake_run_seed.calls)
    bench.run_benchmark(make_args(mode="smoke", output=out, resume=True))
    # exactly the corrupt seed reruns
    assert [s for s, _ in fake_run_seed.calls[n_before:]] == [cfg.seed_base]


def test_sensitivity_dedups_identical_default_cells(tmp_path, fake_run_seed, no_sibling_modules):
    # cost=1.0 and decay=1.0 are both the base config -> one checkpoint, one run per seed
    out = tmp_path / "s.json"
    artifact = bench.run_benchmark(
        make_args(mode="sensitivity", output=out, axis=["cost", "decay"], seeds=2)
    )
    cells = {c["cell_id"]: c for c in artifact["cells"]}
    assert cells["cost=1.0"]["config_hash"] == cells["decay=1.0"]["config_hash"]
    # 6 cells x 2 seeds = 12 nominal tasks, but the two default cells share work
    assert len(fake_run_seed.calls) == 10
    # both default cells still report full per-seed results
    assert len(cells["cost=1.0"]["per_seed"]) == 2
    assert len(cells["decay=1.0"]["per_seed"]) == 2


def test_gamma_and_exec_axes_run_end_to_end(tmp_path, fake_run_seed, no_sibling_modules):
    out = tmp_path / "ge.json"
    artifact = bench.run_benchmark(
        make_args(mode="sensitivity", output=out, axis=["gamma", "exec"], seeds=2)
    )
    assert artifact["n_cells"] == 6
    assert [c["cell_id"] for c in artifact["cells"]] == [
        "gamma=2.0",
        "gamma=5.0",
        "gamma=10.0",
        "gamma=20.0",
        "exec=30",
        "exec=60",
    ]
    cells = {c["cell_id"]: c for c in artifact["cells"]}
    # gamma=10.0 and exec=30 are both the frozen default -> shared config hash
    assert cells["gamma=10.0"]["config_hash"] == cells["exec=30"]["config_hash"]
    # 5 distinct configs x 2 seeds (the two default cells share their work)
    assert len(fake_run_seed.calls) == 10
    assert len({h for _, h in fake_run_seed.calls}) == 5
    assert all(len(c["per_seed"]) == 2 for c in artifact["cells"])
    # the perturbed values reach the encoded per-cell configs
    assert cells["gamma=2.0"]["config"]["mpo"]["gamma"] == 2.0
    assert cells["exec=60"]["config"]["mpo"]["exec_minutes"] == 60
    assert cells["exec=60"]["config"]["mpo"]["gamma"] == 10.0
    assert cells["gamma=20.0"]["config"]["mpo"]["exec_minutes"] == 30


# ---------------------------------------------------------------------------
# Aggregation / paired stats
# ---------------------------------------------------------------------------


def test_paired_stats_called_for_each_baseline(tmp_path, fake_run_seed, monkeypatch):
    recorded = []

    def fake_paired(per_seed_metrics, a, b):
        recorded.append((a, b, len(per_seed_metrics)))
        return {"t": 2.0, "ci": [0.1, 0.5], "winrate": 0.9}

    stats_mod = types.ModuleType("openpo.mpo.stats")
    stats_mod.paired_stats = fake_paired
    monkeypatch.setitem(sys.modules, "openpo.mpo.stats", stats_mod)
    monkeypatch.setitem(sys.modules, "openpo.mpo.report", None)

    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(make_args(mode="smoke", output=out))
    cell = artifact["cells"][0]
    assert set(cell["paired"]) == {"mpo_vs_fixed_weight", "mpo_vs_buy_and_hold"}
    assert all(a == "mpo" and n == 2 for a, _, n in recorded)
    assert cell["paired"]["mpo_vs_fixed_weight"]["winrate"] == 0.9


def test_paired_stats_error_is_contained(tmp_path, fake_run_seed, monkeypatch):
    stats_mod = types.ModuleType("openpo.mpo.stats")

    def boom(*a, **k):
        raise ValueError("degenerate cell")

    stats_mod.paired_stats = boom
    monkeypatch.setitem(sys.modules, "openpo.mpo.stats", stats_mod)
    monkeypatch.setitem(sys.modules, "openpo.mpo.report", None)

    artifact = bench.run_benchmark(make_args(mode="smoke", output=tmp_path / "r.json"))
    paired = artifact["cells"][0]["paired"]
    assert paired["mpo_vs_fixed_weight"] == {"error": "degenerate cell"}


def test_missing_stats_module_warns_but_completes(tmp_path, fake_run_seed, no_sibling_modules):
    with pytest.warns(UserWarning, match="stats unavailable"):
        artifact = bench.run_benchmark(make_args(mode="smoke", output=tmp_path / "r.json"))
    assert artifact["cells"][0]["paired"] == {}


def test_aggregate_cell_pairs_mpo_against_the_shrinkage_arm():
    """``mpo_vs_mpo_shrunk`` must fall out of aggregation with NO benchmark edit.

    The shrinkage arm is an opt-in strategy name (openpo.mpo.backtest.
    OPT_IN_STRATEGIES -- demoted out of the default roster by the neutral
    Iteration 40 verdict, still requestable via ``run_seed(strategies=[...])``).
    Whichever way it enters, it is just another dict-valued key in each seed's
    result, so ``aggregate_cell`` pairs mpo against it like any other baseline.
    Uses the REAL paired_stats.
    """
    from openpo.mpo.backtest import DEFAULT_STRATEGIES, KNOWN_STRATEGIES, OPT_IN_STRATEGIES

    assert "mpo_shrunk" not in DEFAULT_STRATEGIES
    assert "mpo_shrunk" in OPT_IN_STRATEGIES and "mpo_shrunk" in KNOWN_STRATEGIES

    payloads = [
        {
            "seed": 100 + i,
            "elapsed_seconds": 1.0,
            "result": {
                "mpo": {"net_sharpe_annual": 1.0 + 0.1 * i},
                "mpo_shrunk": {"net_sharpe_annual": 1.2 + 0.1 * i},
                "buy_hold": {"net_sharpe_annual": 0.2},
                "seed": 100 + i,
            },
        }
        for i in range(4)
    ]
    cell = {
        "cell_id": "default",
        "axis": "default",
        "value": None,
        "config": ExperimentConfig(),
    }
    out = bench.aggregate_cell(cell, payloads)

    assert "mpo_shrunk" in out["strategies"]
    assert set(out["paired"]) == {"mpo_vs_mpo_shrunk", "mpo_vs_buy_hold"}
    pair = out["paired"]["mpo_vs_mpo_shrunk"]
    assert "error" not in pair
    assert pair["n_seeds"] == 4
    assert pair["mean_diff"] == pytest.approx(-0.2)  # paired: mpo - mpo_shrunk


def test_strategy_names_intersection():
    res = [
        {"mpo": {}, "fixed_weight": {}, "seed": 1},
        {"mpo": {}, "buy_and_hold": {}, "fixed_weight": {}},
    ]
    assert bench._strategy_names(res) == ["mpo", "fixed_weight"]
    assert bench._strategy_names([]) == []


# ---------------------------------------------------------------------------
# HTML report hook
# ---------------------------------------------------------------------------


def test_html_output_calls_render_html(tmp_path, fake_run_seed, monkeypatch):
    calls = []
    report_mod = types.ModuleType("openpo.mpo.report")

    def fake_render(artifact, path):
        calls.append((artifact["schema"], path))
        Path(path).write_text("<html></html>")

    report_mod.render_html = fake_render
    monkeypatch.setitem(sys.modules, "openpo.mpo.report", report_mod)
    monkeypatch.setitem(sys.modules, "openpo.mpo.stats", None)

    html = tmp_path / "rep" / "r.html"
    bench.run_benchmark(make_args(mode="smoke", output=tmp_path / "r.json", html_output=html))
    assert calls == [("mpo-t1.synthetic.v1", str(html))]
    assert html.read_text() == "<html></html>"


def test_missing_report_module_warns_not_fails(tmp_path, fake_run_seed, no_sibling_modules):
    with pytest.warns(UserWarning, match="report unavailable"):
        artifact = bench.run_benchmark(
            make_args(mode="smoke", output=tmp_path / "r.json", html_output=tmp_path / "r.html")
        )
    assert artifact["schema"] == "mpo-t1.synthetic.v1"
    assert not (tmp_path / "r.html").exists()


def test_render_html_exception_is_contained(tmp_path, fake_run_seed, monkeypatch):
    report_mod = types.ModuleType("openpo.mpo.report")
    report_mod.render_html = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "openpo.mpo.report", report_mod)
    monkeypatch.setitem(sys.modules, "openpo.mpo.stats", None)
    with pytest.warns(UserWarning, match="render_html failed"):
        bench.run_benchmark(
            make_args(mode="smoke", output=tmp_path / "r.json", html_output=tmp_path / "r.html")
        )


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


def test_main_prints_summary_and_returns_zero(tmp_path, fake_run_seed, no_sibling_modules, capsys):
    rc = bench.main(["--mode", "smoke", "--output", str(tmp_path / "r.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mpo-t1 smoke" in out
    assert "2 seed-run(s)" in out


def test_config_encoding_roundtrips_frozen_defaults():
    enc = bench.encode_config(ExperimentConfig())
    assert enc["n_seeds"] == 24
    assert enc["mpo"]["gamma"] == 10.0
    assert enc["synth"]["target_ics"] == [0.05, 0.04, 0.03]
    json.dumps(enc)  # must be JSON-safe as-is


def test_run_seed_import_is_deferred(monkeypatch):
    # Without the override and without backtest.py the resolver must raise
    monkeypatch.setattr(bench, "RUN_SEED_OVERRIDE", None)
    monkeypatch.setitem(sys.modules, "openpo.mpo.backtest", None)
    with pytest.raises(ImportError):
        bench._get_run_seed()


# ---------------------------------------------------------------------------
# FixedWeight lambdas: --lambdas parsing / validation
# ---------------------------------------------------------------------------


def test_parse_lambdas_normalizes():
    assert bench.parse_lambdas("1,1,2") == pytest.approx((0.25, 0.25, 0.5))
    assert bench.parse_lambdas(" 0.2 , 0.3 , 0.5 ") == pytest.approx((0.2, 0.3, 0.5))
    assert bench.parse_lambdas("0,0,7") == pytest.approx((0.0, 0.0, 1.0))
    assert sum(bench.parse_lambdas("3,4,5")) == pytest.approx(1.0)


@pytest.mark.parametrize("bad", ["1,2", "1,2,3,4", "a,b,c", "1,-1,1", "0,0,0", ""])
def test_parse_lambdas_rejects_invalid(bad):
    with pytest.raises(ValueError):
        bench.parse_lambdas(bad)


@pytest.mark.parametrize("bad", ["1,2", "1,-1,1", "0,0,0", "x,1,1"])
def test_cli_rejects_invalid_lambdas(bad):
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--mode", "smoke", "--lambdas", bad])


def test_parser_tuning_flag_defaults_and_validation():
    args = bench.build_parser().parse_args(["--mode", "smoke"])
    assert args.lambdas is None
    assert args.tune_lambdas is None
    assert args.tune_grid_step == bench.DEFAULT_TUNE_GRID_STEP == 0.1

    args = bench.build_parser().parse_args(
        ["--mode", "main", "--tune-lambdas", "5", "--tune-grid-step", "0.25"]
    )
    assert args.tune_lambdas == 5
    assert args.tune_grid_step == 0.25

    for bad in ("0", "-2"):
        with pytest.raises(SystemExit):
            bench.build_parser().parse_args(["--mode", "smoke", "--tune-lambdas", bad])


# ---------------------------------------------------------------------------
# Explicit --lambdas path
# ---------------------------------------------------------------------------


def test_explicit_lambdas_flow_into_run_seed_and_artifact(
    tmp_path, fake_run_seed_lambdas, no_sibling_modules
):
    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(make_args(mode="smoke", output=out, lambdas="1,1,2"))

    assert fake_run_seed_lambdas.lambda_calls == [[0.25, 0.25, 0.5]] * 2
    assert artifact["tuning"] == {"source": "cli", "lambdas": [0.25, 0.25, 0.5]}
    # also persisted in the artifact on disk and in the per-seed checkpoints
    assert json.loads(out.read_text())["tuning"]["lambdas"] == [0.25, 0.25, 0.5]
    ckpts = sorted((out.parent / "r-checkpoints").glob("ckpt-*-lam*-seed*.json"))
    assert len(ckpts) == 2
    assert json.loads(ckpts[0].read_text())["fixed_lambdas"] == [0.25, 0.25, 0.5]


def test_explicit_lambdas_get_own_checkpoint_namespace(
    tmp_path, fake_run_seed_lambdas, no_sibling_modules
):
    out = tmp_path / "r.json"
    bench.run_benchmark(make_args(mode="smoke", output=out, lambdas="1,0,0"))
    bench.run_benchmark(make_args(mode="smoke", output=out, lambdas="0,1,0", resume=True))
    # different lambdas must not be served from the other lambda's checkpoints
    assert len(fake_run_seed_lambdas.lambda_calls) == 4
    names = sorted(p.name for p in (out.parent / "r-checkpoints").glob("ckpt-*.json"))
    assert len(names) == 4  # 2 seeds x 2 distinct lambda tags


def test_explicit_lambdas_beat_tune_flag(
    tmp_path, fake_run_seed_lambdas, fake_tune_scorer, no_sibling_modules
):
    out = tmp_path / "r.json"
    with pytest.warns(UserWarning, match="ignoring --tune-lambdas"):
        artifact = bench.run_benchmark(
            make_args(mode="smoke", output=out, lambdas="0,0,1", tune_lambdas=2)
        )
    assert artifact["tuning"]["source"] == "cli"
    assert fake_tune_scorer.calls == []  # tuning skipped entirely


# ---------------------------------------------------------------------------
# Default (untuned) path stays exactly as before
# ---------------------------------------------------------------------------


def test_default_path_does_not_pass_lambdas(tmp_path, fake_run_seed_lambdas, no_sibling_modules):
    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(make_args(mode="smoke", output=out))

    # run_seed is called with no fixed_lambdas -> its own IC-proportional default
    assert fake_run_seed_lambdas.lambda_calls == [None, None]
    ics = bench.build_smoke_config().synth.target_ics
    expected = [x / sum(ics) for x in ics]
    assert artifact["tuning"] == {"source": "default", "lambdas": pytest.approx(expected)}
    # legacy checkpoint names (no lambda tag) are preserved
    cfg = bench.build_smoke_config()
    h8 = config_hash(cfg)[:8]
    for seed in (cfg.seed_base, cfg.seed_base + 1):
        assert (out.parent / "r-checkpoints" / f"ckpt-{h8}-seed{seed}.json").exists()
    assert not list((out.parent / "r-checkpoints").glob("*lam*"))
    assert not list((out.parent / "r-checkpoints").glob("tune-*"))


# ---------------------------------------------------------------------------
# Parallel tuning: grid, selection, determinism
# ---------------------------------------------------------------------------


def test_tuning_grid_matches_backtest_simplex_grid():
    from openpo.mpo import backtest

    grid = bench.tuning_grid(0.5)
    assert grid == [tuple(float(x) for x in row) for row in backtest.simplex_grid(0.5)]
    assert len(grid) == 6
    assert len(bench.tuning_grid(0.1)) == 66
    assert all(isinstance(x, float) for pt in grid for x in pt)


def test_tuning_seeds_come_from_tuning_seed_base():
    cfg = bench.build_smoke_config()
    assert bench.tuning_seeds_for(cfg, 3) == [cfg.tuning_seed_base + i for i in range(3)]
    assert bench.tuning_config("smoke") == bench.build_smoke_config()
    assert bench.tuning_config("sensitivity") == ExperimentConfig()


def test_tuning_selects_argmax_with_faked_scorer(
    tmp_path, fake_run_seed_lambdas, fake_tune_scorer, no_sibling_modules
):
    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, tune_lambdas=2, tune_grid_step=0.5)
    )

    # coeffs (0.1, 0.9, 0.3) -> best lattice point is the pure middle-signal corner
    tuning = artifact["tuning"]
    assert tuning["source"] == "tuned"
    assert tuning["lambdas"] == pytest.approx([0.0, 1.0, 0.0])
    # every (seed, grid point) pair was scored exactly once
    assert len(fake_tune_scorer.calls) == 6 * 2
    assert len(set(fake_tune_scorer.calls)) == 12
    # the tuned lambdas reach every eval seed run
    assert fake_run_seed_lambdas.lambda_calls == [pytest.approx([0.0, 1.0, 0.0])] * 2


def test_tuning_artifact_block_shape(
    tmp_path, fake_run_seed_lambdas, fake_tune_scorer, no_sibling_modules
):
    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, tune_lambdas=2, tune_grid_step=0.5)
    )
    tuning = artifact["tuning"]
    assert set(tuning) >= {
        "lambdas",
        "grid_step",
        "tuning_seeds",
        "scores_top5",
        "wall_clock_seconds",
    }
    cfg = bench.build_smoke_config()
    assert tuning["tuning_seeds"] == [cfg.tuning_seed_base, cfg.tuning_seed_base + 1]
    assert tuning["grid_step"] == 0.5
    assert tuning["n_grid_points"] == 6
    assert tuning["wall_clock_seconds"] >= 0
    scores = [entry["score"] for entry in tuning["scores_top5"]]
    assert len(tuning["scores_top5"]) == 5
    assert scores == sorted(scores, reverse=True)
    assert tuning["scores_top5"][0]["lambdas"] == pytest.approx(tuning["lambdas"])
    # survives the JSON round trip (artifact on disk carries the same block)
    assert json.loads(out.read_text())["tuning"]["lambdas"] == pytest.approx(tuning["lambdas"])


def test_tuning_is_deterministic_across_runs(
    tmp_path, fake_run_seed_lambdas, fake_tune_scorer, no_sibling_modules
):
    a = bench.run_benchmark(
        make_args(mode="smoke", output=tmp_path / "a.json", tune_lambdas=3, tune_grid_step=0.5)
    )
    b = bench.run_benchmark(
        make_args(mode="smoke", output=tmp_path / "b.json", tune_lambdas=3, tune_grid_step=0.5)
    )
    assert a["tuning"]["lambdas"] == b["tuning"]["lambdas"]
    assert a["tuning"]["grid_index"] == b["tuning"]["grid_index"]
    assert [e["grid_index"] for e in a["tuning"]["scores_top5"]] == [
        e["grid_index"] for e in b["tuning"]["scores_top5"]
    ]


def test_select_lambdas_breaks_ties_on_lowest_grid_index():
    grid = bench.tuning_grid(0.5)
    payloads = [
        {"grid_index": gi, "seed": seed, "score": 1.0}
        for gi in range(len(grid))
        for seed in (7, 8)
    ]
    chosen = bench.select_lambdas(list(reversed(payloads)), grid, n_seeds=2)
    assert chosen["grid_index"] == 0
    assert chosen["lambdas"] == pytest.approx(list(grid[0]))
    assert chosen["score"] == pytest.approx(2.0)  # summed over the two seeds


def test_select_lambdas_sums_over_seeds_and_ignores_incomplete_points():
    grid = bench.tuning_grid(0.5)
    payloads = [
        {"grid_index": 0, "seed": 1, "score": 1.0},
        {"grid_index": 0, "seed": 2, "score": 1.0},
        {"grid_index": 3, "seed": 1, "score": 5.0},  # only one seed -> not eligible
    ]
    chosen = bench.select_lambdas(payloads, grid, n_seeds=2)
    assert chosen["grid_index"] == 0
    assert chosen["n_scored_points"] == 1
    with pytest.raises(RuntimeError):
        bench.select_lambdas([{"grid_index": 1, "seed": 1, "score": 1.0}], grid, n_seeds=2)


# ---------------------------------------------------------------------------
# Tuning-seed disjointness guard
# ---------------------------------------------------------------------------


def test_tuning_seeds_must_be_disjoint_from_eval_seeds(
    tmp_path, fake_run_seed_lambdas, fake_tune_scorer, no_sibling_modules, monkeypatch
):
    cfg = bench.build_smoke_config()
    colliding = dataclasses.replace(cfg, tuning_seed_base=cfg.seed_base)
    monkeypatch.setattr(bench, "tuning_config", lambda mode: colliding)

    with pytest.raises(ValueError, match="overlap eval seeds"):
        bench.run_benchmark(
            make_args(
                mode="smoke", output=tmp_path / "r.json", tune_lambdas=2, tune_grid_step=0.5
            )
        )
    assert fake_tune_scorer.calls == []  # guard fires before any tuning work


def test_disjointness_guard_helper_accepts_frozen_defaults():
    cells = bench.build_cells("sensitivity", sorted(bench.SENSITIVITY_AXES))
    seeds = bench.tuning_seeds_for(ExperimentConfig(), 5)
    bench.assert_tuning_seeds_disjoint(seeds, cells, None)  # frozen bases are disjoint
    with pytest.raises(ValueError, match="dedicated tuning seeds"):
        bench.assert_tuning_seeds_disjoint(
            [ExperimentConfig().seed_base + 1], cells, None
        )


# ---------------------------------------------------------------------------
# Tuning checkpoints / resume
# ---------------------------------------------------------------------------


def test_tuning_checkpoint_names_carry_hash_seed_and_grid_index(
    tmp_path, fake_run_seed_lambdas, fake_tune_scorer, no_sibling_modules
):
    out = tmp_path / "r.json"
    bench.run_benchmark(
        make_args(mode="smoke", output=out, tune_lambdas=2, tune_grid_step=0.5)
    )
    cfg = bench.build_smoke_config()
    cfg_hash = config_hash(cfg)
    ckpt_dir = out.parent / "r-checkpoints"
    files = sorted(ckpt_dir.glob("tune-*.json"))
    assert len(files) == 12
    for seed in (cfg.tuning_seed_base, cfg.tuning_seed_base + 1):
        for gi in range(6):
            path = bench.tuning_checkpoint_path(ckpt_dir, cfg_hash, seed, gi, 0.5)
            assert path.exists(), path
            assert cfg_hash[:8] in path.name and str(seed) in path.name
            payload = json.loads(path.read_text())
            assert payload["grid_index"] == gi
            assert payload["seed"] == seed
            assert payload["grid_step"] == 0.5
            assert isinstance(payload["score"], float)
    # a different grid step must not reuse those files
    other = bench.tuning_checkpoint_path(ckpt_dir, cfg_hash, cfg.tuning_seed_base, 0, 0.1)
    assert not other.exists()


def test_tuning_resume_skips_completed_tasks(
    tmp_path, fake_run_seed_lambdas, no_sibling_modules, monkeypatch
):
    out = tmp_path / "r.json"
    first = FakeTuneScorer()
    monkeypatch.setattr(bench, "TUNE_SCORE_OVERRIDE", first)
    a = bench.run_benchmark(
        make_args(mode="smoke", output=out, tune_lambdas=2, tune_grid_step=0.5)
    )
    assert len(first.calls) == 12

    second = FakeTuneScorer(coeffs=(9.0, 0.0, 0.0))  # would pick a different corner
    monkeypatch.setattr(bench, "TUNE_SCORE_OVERRIDE", second)
    b = bench.run_benchmark(
        make_args(mode="smoke", output=out, tune_lambdas=2, tune_grid_step=0.5, resume=True)
    )
    assert second.calls == []  # every tuning task served from its checkpoint
    assert b["tuning"]["lambdas"] == pytest.approx(a["tuning"]["lambdas"])


def test_tuning_resume_reruns_only_missing_and_corrupt(
    tmp_path, fake_run_seed_lambdas, no_sibling_modules, monkeypatch
):
    out = tmp_path / "r.json"
    first = FakeTuneScorer()
    monkeypatch.setattr(bench, "TUNE_SCORE_OVERRIDE", first)
    bench.run_benchmark(make_args(mode="smoke", output=out, tune_lambdas=2, tune_grid_step=0.5))

    cfg = bench.build_smoke_config()
    ckpt_dir = out.parent / "r-checkpoints"
    cfg_hash = config_hash(cfg)
    missing = bench.tuning_checkpoint_path(ckpt_dir, cfg_hash, cfg.tuning_seed_base, 2, 0.5)
    missing.unlink()
    corrupt = bench.tuning_checkpoint_path(
        ckpt_dir, cfg_hash, cfg.tuning_seed_base + 1, 4, 0.5
    )
    corrupt.write_text("{not json")

    second = FakeTuneScorer()
    monkeypatch.setattr(bench, "TUNE_SCORE_OVERRIDE", second)
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, tune_lambdas=2, tune_grid_step=0.5, resume=True)
    )
    assert len(second.calls) == 2
    assert {seed for seed, _ in second.calls} == {
        cfg.tuning_seed_base,
        cfg.tuning_seed_base + 1,
    }
    assert artifact["tuning"]["lambdas"] == pytest.approx([0.0, 1.0, 0.0])
    assert missing.exists() and json.loads(corrupt.read_text())["grid_index"] == 4


def test_tuning_parallel_jobs_path(
    tmp_path, fake_run_seed_lambdas, fake_tune_scorer, no_sibling_modules
):
    # jobs>1 uses multiprocessing.Pool; the fork-inherited override keeps it hermetic
    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, tune_lambdas=2, tune_grid_step=0.5, jobs=2)
    )
    assert artifact["tuning"]["lambdas"] == pytest.approx([0.0, 1.0, 0.0])
    assert len(list((out.parent / "r-checkpoints").glob("tune-*.json"))) == 12


def test_tune_scorer_resolution_prefers_override(monkeypatch):
    monkeypatch.setattr(bench, "TUNE_SCORE_OVERRIDE", None)
    assert bench._get_tune_scorer() is bench.score_fixed_weight_point
    sentinel = FakeTuneScorer()
    monkeypatch.setattr(bench, "TUNE_SCORE_OVERRIDE", sentinel)
    assert bench._get_tune_scorer() is sentinel


def test_main_prints_tuned_lambdas(
    tmp_path, fake_run_seed_lambdas, fake_tune_scorer, no_sibling_modules, capsys
):
    rc = bench.main(
        [
            "--mode",
            "smoke",
            "--output",
            str(tmp_path / "r.json"),
            "--tune-lambdas",
            "2",
            "--tune-grid-step",
            "0.5",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "mpo-t1 smoke" in out
    assert "lambdas* = [0.00, 1.00, 0.00]" in out


# ---------------------------------------------------------------------------
# --strategies roster (Iteration 42 leg 2: the mpo_delayaware A/B arm)
# ---------------------------------------------------------------------------


def test_parse_strategies_accepts_known_names():
    from openpo.mpo.backtest import DEFAULT_STRATEGIES, KNOWN_STRATEGIES

    assert bench.parse_strategies("mpo,mpo_delayaware") == ("mpo", "mpo_delayaware")
    assert bench.parse_strategies(" mpo , buy_hold ") == ("mpo", "buy_hold")
    assert bench.parse_strategies("mpo") == ("mpo",)
    # order is preserved (it is part of the checkpoint fingerprint)
    assert bench.parse_strategies("mpo_delayaware,mpo") == ("mpo_delayaware", "mpo")
    # every advertised name really parses
    assert bench.parse_strategies(",".join(KNOWN_STRATEGIES)) == tuple(KNOWN_STRATEGIES)
    assert bench.parse_strategies(",".join(DEFAULT_STRATEGIES)) == tuple(DEFAULT_STRATEGIES)


@pytest.mark.parametrize(
    "bad", ["", " ", ",", "mpo,", "mpo,mpo", "nope", "mpo,nope", "MPO", "mpo delayaware"]
)
def test_parse_strategies_rejects_invalid(bad):
    with pytest.raises(ValueError):
        bench.parse_strategies(bad)


def test_parse_strategies_error_messages_are_actionable():
    with pytest.raises(ValueError, match=r"unknown name\(s\) \['typo_arm'\]"):
        bench.parse_strategies("mpo,typo_arm")
    with pytest.raises(ValueError, match=r"duplicate name\(s\) \['mpo'\]"):
        bench.parse_strategies("mpo,mpo")
    with pytest.raises(ValueError, match="no empty entries"):
        bench.parse_strategies("mpo,,mpo_delayaware")
    # the roster is printed so the operator does not have to grep the source
    with pytest.raises(ValueError, match="mpo_delayaware"):
        bench.parse_strategies("typo_arm")


@pytest.mark.parametrize("bad", ["", "nope", "mpo,mpo", "mpo,nope"])
def test_cli_rejects_invalid_strategies(bad):
    with pytest.raises(SystemExit):
        bench.build_parser().parse_args(["--mode", "smoke", "--strategies", bad])


def test_parser_strategies_defaults_to_none_and_parses_roster():
    assert bench.build_parser().parse_args(["--mode", "smoke"]).strategies is None
    args = bench.build_parser().parse_args(
        ["--mode", "smoke", "--strategies", "mpo,mpo_delayaware"]
    )
    assert args.strategies == ("mpo", "mpo_delayaware")


def test_strategies_flow_into_run_seed_and_artifact(
    tmp_path, fake_run_seed_roster, no_sibling_modules
):
    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, strategies="mpo,mpo_delayaware")
    )

    assert fake_run_seed_roster.roster_calls == [["mpo", "mpo_delayaware"]] * 2
    assert artifact["strategies"] == ["mpo", "mpo_delayaware"]
    assert json.loads(out.read_text())["strategies"] == ["mpo", "mpo_delayaware"]
    assert artifact["cells"][0]["strategies"] == ["mpo", "mpo_delayaware"]

    ckpts = sorted((out.parent / "r-checkpoints").glob("ckpt-*-strat*-seed*.json"))
    assert len(ckpts) == 2
    assert json.loads(ckpts[0].read_text())["strategies"] == ["mpo", "mpo_delayaware"]


def test_default_path_keeps_legacy_checkpoint_names_and_no_roster_kwarg(
    tmp_path, fake_run_seed_roster, no_sibling_modules
):
    """Compatibility: the default run must be byte-identical to before leg 2."""
    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(make_args(mode="smoke", output=out))

    assert fake_run_seed_roster.roster_calls == [None, None]
    assert artifact["strategies"] is None
    cfg = bench.build_smoke_config()
    h8 = config_hash(cfg)[:8]
    for seed in (cfg.seed_base, cfg.seed_base + 1):
        assert (out.parent / "r-checkpoints" / f"ckpt-{h8}-seed{seed}.json").exists()
    assert not list((out.parent / "r-checkpoints").glob("*strat*"))


def test_explicit_default_roster_normalizes_to_the_legacy_namespace(
    tmp_path, fake_run_seed_roster, no_sibling_modules
):
    """``--strategies <the default list>`` is the default run, so reuse its cache.

    Spelling out the default roster asks for exactly what ``run_seed(None)``
    does, so inventing a second checkpoint namespace for it would only cause
    needless recomputation.
    """
    from openpo.mpo.backtest import DEFAULT_STRATEGIES

    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, strategies=",".join(DEFAULT_STRATEGIES))
    )
    assert fake_run_seed_roster.roster_calls == [None, None]
    assert artifact["strategies"] is None
    assert not list((out.parent / "r-checkpoints").glob("*strat*"))


def test_strategies_get_their_own_checkpoint_namespace(
    tmp_path, fake_run_seed_roster, no_sibling_modules
):
    """cache-key-must-fingerprint-all-inputs: a different roster is a different run."""
    out = tmp_path / "r.json"
    bench.run_benchmark(make_args(mode="smoke", output=out, strategies="mpo"))
    bench.run_benchmark(
        make_args(mode="smoke", output=out, strategies="mpo,mpo_delayaware", resume=True)
    )
    # the second roster must NOT be served from the first roster's checkpoints
    assert fake_run_seed_roster.roster_calls == [
        ["mpo"], ["mpo"], ["mpo", "mpo_delayaware"], ["mpo", "mpo_delayaware"]
    ]
    names = sorted(p.name for p in (out.parent / "r-checkpoints").glob("ckpt-*.json"))
    assert len(names) == 4  # 2 seeds x 2 distinct roster tags
    # ... and order matters too, since run_seed's output order follows the roster
    assert bench._strategies_tag(("mpo", "mpo_delayaware")) != bench._strategies_tag(
        ("mpo_delayaware", "mpo")
    )
    assert bench._strategies_tag(("mpo",)) != bench._strategies_tag(("mpo", "mpo"))


def test_strategies_resume_reuses_its_own_checkpoints(
    tmp_path, fake_run_seed_roster, no_sibling_modules
):
    out = tmp_path / "r.json"
    roster = "mpo,mpo_delayaware"
    bench.run_benchmark(make_args(mode="smoke", output=out, strategies=roster))
    assert len(fake_run_seed_roster.roster_calls) == 2
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, strategies=roster, resume=True)
    )
    assert len(fake_run_seed_roster.roster_calls) == 2  # nothing recomputed
    assert artifact["strategies"] == ["mpo", "mpo_delayaware"]


def test_checkpoint_path_tag_composition():
    """The two optional tags stack in a fixed order, and neither is lossy."""
    base = bench.checkpoint_path("/d", "0123456789abcdef", 7)
    assert base == Path("/d/ckpt-01234567-seed7.json")
    lam = bench.checkpoint_path("/d", "0123456789abcdef", 7, fixed_lambdas=(0.5, 0.5, 0.0))
    strat = bench.checkpoint_path("/d", "0123456789abcdef", 7, strategies=("mpo",))
    both = bench.checkpoint_path(
        "/d", "0123456789abcdef", 7, fixed_lambdas=(0.5, 0.5, 0.0), strategies=("mpo",)
    )
    assert len({base, lam, strat, both}) == 4
    assert "-lam" in both.name and "-strat" in both.name
    assert both.name.index("-lam") < both.name.index("-strat")


def test_roster_with_lambdas_keeps_both_tags(
    tmp_path, fake_run_seed_roster, no_sibling_modules
):
    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, strategies="mpo", lambdas="1,1,2")
    )
    assert fake_run_seed_roster.roster_calls == [["mpo"], ["mpo"]]
    assert artifact["strategies"] == ["mpo"]
    assert artifact["tuning"]["lambdas"] == [0.25, 0.25, 0.5]
    ckpts = sorted((out.parent / "r-checkpoints").glob("ckpt-*-lam*-strat*-seed*.json"))
    assert len(ckpts) == 2
    payload = json.loads(ckpts[0].read_text())
    assert payload["strategies"] == ["mpo"] and payload["fixed_lambdas"] == [0.25, 0.25, 0.5]


def test_paired_stats_pairs_mpo_against_the_delay_arm(tmp_path, monkeypatch):
    """The A/B verdict line: mpo vs mpo_delayaware must be a paired comparison."""
    fake = FakeRunSeedRoster()
    monkeypatch.setattr(bench, "RUN_SEED_OVERRIDE", fake)
    seen = []

    def fake_paired_stats(results, a, b):
        seen.append((a, b))
        return {"mean_diff": 0.1, "p_value": 0.04, "n": len(results)}

    stats_mod = types.ModuleType("openpo.mpo.stats")
    stats_mod.paired_stats = fake_paired_stats
    monkeypatch.setitem(sys.modules, "openpo.mpo.stats", stats_mod)
    monkeypatch.setitem(sys.modules, "openpo.mpo.report", None)

    out = tmp_path / "r.json"
    artifact = bench.run_benchmark(
        make_args(mode="smoke", output=out, strategies="mpo,mpo_delayaware")
    )
    assert seen == [("mpo", "mpo_delayaware")]
    assert artifact["cells"][0]["paired"]["mpo_vs_mpo_delayaware"]["p_value"] == 0.04


def test_main_prints_the_roster_when_overridden(
    tmp_path, fake_run_seed_roster, no_sibling_modules, capsys
):
    rc = bench.main(
        ["--mode", "smoke", "--output", str(tmp_path / "r.json"),
         "--strategies", "mpo,mpo_delayaware"]
    )
    assert rc == 0
    printed = capsys.readouterr().out
    assert "strategies" in printed and "mpo_delayaware" in printed


def test_main_does_not_print_a_roster_line_by_default(
    tmp_path, fake_run_seed_roster, no_sibling_modules, capsys
):
    rc = bench.main(["--mode", "smoke", "--output", str(tmp_path / "r.json")])
    assert rc == 0
    assert "strategies" not in capsys.readouterr().out

def test_strategies_roster_comes_from_backtest_not_a_second_hardcoded_list():
    """One roster, one owner: the flag must accept whatever backtest advertises.

    A local copy of the name list would silently reject a future arm (or accept
    a retired one), so the check is that every advertised name round-trips.
    """
    from openpo.mpo import backtest

    assert "KNOWN_STRATEGIES" in Path(bench.__file__).read_text()
    for name in backtest.KNOWN_STRATEGIES:
        assert bench.parse_strategies(name) == (name,)
