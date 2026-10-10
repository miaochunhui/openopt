"""Tests for openpo.mpo.report.render_html (self-contained HTML report).

Covers: full artifact rendering (sections + numbers present), CI lower-bound
highlighting, sensitivity tables, and fault tolerance on missing/empty/broken
artifacts ('n/a' shown, no crash).

Iteration 38 additions -- the legs that would have caught the hollow-report
incident (see openpo.mpo.report's module docstring):

- ``TestEndToEndBenchmarkArtifact``: real ``build_artifact()`` output (smoke
  cell, fake run_seed whose per-strategy dicts come from the *real*
  ``metrics.summarize``) rendered by the real ``render_html``, asserting the
  numbers land in the HTML and no required region degrades to 'n/a'.
- ``TestRequiredRegionsRaise``: every required region, removed one at a time,
  must raise ``ReportContractError`` instead of rendering a shell.
- ``TestSummarizeKeyContract``: the metric keys the adapter reads must exist in
  ``metrics.summarize``'s real output (drift guard).
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "benchmarks"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mpo_t1_benchmark as bench  # noqa: E402

from openpo.mpo.metrics import summarize  # noqa: E402
from openpo.mpo.report import (  # noqa: E402
    REQUIRED_SEED_METRICS,
    ReportContractError,
    adapt_benchmark_artifact,
    render_html,
)
from openpo.mpo.types import BacktestResult  # noqa: E402


@pytest.fixture()
def full_artifact():
    return {
        "mode": "main",
        "config_hash": "abc123def456",
        "git_rev": "352ae02",
        "results_sha256": "feedbeef" * 8,
        "seeds": [20260821 + i for i in range(24)],
        "env": {
            "python": "3.11.8",
            "numpy": "1.26.4",
            "scipy": "1.13.0",
            "osqp": "0.6.5",
            "host": "a100-node-3",
        },
        "strategies": {
            "MPO": {
                "sharpe_mean": 1.2345,
                "sharpe_std": 0.3210,
                "annual_net_return": 0.1876,
                "daily_turnover": 0.0812,
                "cost_bp": 9.8765,
                "max_drawdown": -0.0654,
                "t1_violations": 0,
            },
            "FixedWeight": {
                "sharpe_mean": 0.9111,
                "sharpe_std": 0.4001,
                "annual_net_return": 0.1402,
                "daily_turnover": 0.0933,
                "cost_bp": 11.5000,
                "max_drawdown": -0.0899,
                "t1_violations": 0,
            },
        },
        "pairwise": [
            {
                "baseline": "FixedWeight",
                "mean_diff": 0.3234,
                "ci95": [0.1001, 0.5467],
                "win_rate": 0.7917,
                "p_holm": 0.0042,
                "hedges_g": 0.6543,
            },
            {
                "baseline": "InformedTrading",
                "mean_diff": 0.0500,
                "ci_low": -0.1200,
                "ci_high": 0.2200,
                "win_rate": 0.5417,
                "p_holm": 0.6100,
                "hedges_g": 0.1000,
            },
        ],
        "sensitivity": {
            "cost_mult": [
                {"value": 0.5, "mpo_sharpe": 1.4000, "fixed_sharpe": 1.1000},
                {"value": 2.0, "mpo_sharpe": 0.9000, "fixed_sharpe": 0.6000},
            ],
            "t1": [
                {"value": "on", "mpo_sharpe": 1.2345},
                {"value": "off", "mpo_sharpe": 1.3456},
            ],
        },
        "oracle": {
            "oracle_sharpe": 2.1000,
            "mpo_sharpe": 1.2345,
            "structure_gain": 0.3234,
            "calibration_loss": 0.8655,
        },
        "qp_walltime": {
            "p50": 0.0123,
            "p95": 0.0456,
            "cold_p50": 0.0500,
            "hot_p95": 0.0300,
        },
        "bootstrap_iters": 10000,
    }


def _render(artifact, tmp_path, name="report.html"):
    out = tmp_path / name
    ret = render_html(artifact, str(out))
    assert ret == str(out)
    html = out.read_text(encoding="utf-8")
    return html


class TestFullArtifact:
    def test_all_sections_present(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        for title in [
            "项目目标与背景",
            "配置与环境",
            "主结果",
            "配对比较",
            "敏感性分析",
            "Oracle 增益分解",
            "QP 求解墙钟",
            "方法论与置信度说明",
        ]:
            assert title in html

    def test_background_mentions_methodology(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        assert "Axioma 051" in html
        assert "T+1" in html
        assert "return_1_30" in html
        assert "交易分钟" in html  # 隔夜=0分钟约定

    def test_config_env_values(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        assert "abc123def456" in html
        assert "352ae02" in html
        assert "3.11.8" in html
        assert "a100-node-3" in html
        assert "24 个" in html

    def test_main_table_numbers_4dp(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        assert "MPO" in html and "FixedWeight" in html
        assert "1.2345 ± 0.3210" in html
        assert "0.1876" in html  # 年化净收益
        assert "0.0812" in html  # 日换手
        assert "9.8765" in html  # 成本bp
        assert "-0.0654" in html  # MaxDD
        assert ">0<" in html  # T+1 违规整数 0

    def test_pairwise_table_and_significance(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        assert "0.3234" in html
        assert "[0.1001, 0.5467]" in html  # ci95 list form
        assert "[-0.1200, 0.2200]" in html  # ci_low/ci_high form
        assert "0.7917" in html
        assert "0.0042" in html
        # CI 下界 > 0 的行加粗绿 (class="sig"); 下界 < 0 的行不加
        assert html.count('<span class="sig">FixedWeight</span>') == 1
        assert '<span class="sig">InformedTrading</span>' not in html

    def test_sensitivity_tables_per_axis(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        assert "cost_mult" in html
        assert "1.4000" in html
        assert "0.9000" in html
        # t1 axis with string cell values
        assert "on" in html and "off" in html and "1.3456" in html

    def test_oracle_and_qp_sections(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        assert "2.1000" in html  # oracle sharpe
        assert "0.8655" in html  # calibration loss
        assert "0.0123" in html and "0.0456" in html  # qp p50/p95
        assert "0.0500" in html  # cold_p50

    def test_methodology_content(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        assert "bootstrap" in html
        assert "Holm" in html
        assert "预注册" in html
        assert "局限" in html

    def test_self_contained(self, full_artifact, tmp_path):
        html = _render(full_artifact, tmp_path)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "<style>" in html
        # 无外部资源引用
        assert "http://" not in html and "https://" not in html
        assert "<script" not in html and "<link" not in html
        assert 'charset="utf-8"' in html


class TestFaultTolerance:
    def test_empty_artifact(self, tmp_path):
        html = _render({}, tmp_path)
        assert "n/a" in html
        for title in ["项目目标与背景", "配对比较", "QP 求解墙钟"]:
            assert title in html

    def test_non_dict_artifact(self, tmp_path):
        html = _render(None, tmp_path)
        assert "n/a" in html

    def test_missing_metric_fields(self, tmp_path):
        artifact = {
            "strategies": {"MPO": {"sharpe_mean": 1.0}},  # 其余指标缺失
            "pairwise": [{"baseline": "X"}],  # 无 CI/p 值
        }
        html = _render(artifact, tmp_path)
        assert "1.0000 ± n/a" in html
        assert "[n/a, n/a]" in html

    def test_nan_and_bad_types(self, tmp_path):
        artifact = {
            "strategies": [
                {
                    "name": "MPO",
                    "sharpe_mean": float("nan"),
                    "sharpe_std": float("inf"),
                    "annual_net_return": "oops",
                    "t1_violations": None,
                }
            ],
            "qp_walltime": {"p50": "bad", "p95": None},
            "sensitivity": {"broken": ["not-a-dict", {"value": 1.0}]},
        }
        html = _render(artifact, tmp_path)
        assert "n/a ± n/a" in html
        assert "MPO" in html
        assert "1.0000" in html  # 敏感性中合法 cell 仍渲染

    def test_strategies_as_list(self, tmp_path):
        artifact = {"strategies": [{"strategy": "BuyAndHold", "sharpe_mean": 0.5}]}
        html = _render(artifact, tmp_path)
        assert "BuyAndHold" in html
        assert "0.5000" in html

    def test_html_escaping(self, tmp_path):
        artifact = {
            "git_rev": "<script>alert(1)</script>",
            "strategies": {"<b>evil</b>": {"sharpe_mean": 1.0}},
        }
        html = _render(artifact, tmp_path)
        assert "<script>alert(1)</script>" not in html
        assert "&lt;script&gt;" in html
        assert "&lt;b&gt;evil&lt;/b&gt;" in html


# ---------------------------------------------------------------------------
# Iteration 38: end-to-end build_artifact() -> render_html()
# ---------------------------------------------------------------------------

#: Baselines the production run_seed returns (SPEC section 9).
STRATEGIES = (
    "mpo",
    "single_30",
    "single_240",
    "single_2400",
    "fixed_weight",
    "informed",
    "oracle_mpo",
    "buy_hold",
)

#: (daily drift, daily vol, daily turnover) -- ordered so oracle > mpo >
#: fixed_weight > buy_hold on Sharpe, as in the real experiment.
PROFILES = {
    "mpo": (9.0e-4, 3.0e-3, 0.08),
    "single_30": (3.0e-4, 4.0e-3, 0.55),
    "single_240": (5.0e-4, 4.5e-3, 0.78),
    "single_2400": (6.0e-4, 4.2e-3, 0.21),
    "fixed_weight": (7.0e-4, 3.6e-3, 0.13),
    "informed": (8.0e-4, 4.0e-3, 0.77),
    "oracle_mpo": (1.6e-3, 3.2e-3, 1.62),
    "buy_hold": (2.0e-4, 5.0e-3, 0.012),
}


def _fake_result(seed: int, name: str, idx: int, n_days: int) -> dict:
    """One strategy's per-seed summary, produced by the REAL summarize()."""
    drift, vol, turn = PROFILES[name]
    rng = np.random.default_rng([int(seed), idx])
    rets = drift + vol * rng.standard_normal(n_days)
    nav = 1e8 * np.cumprod(np.concatenate([[1.0], 1.0 + rets]))
    turnover = np.full(n_days, turn)
    traded = float(np.sum(turnover * nav[:-1]))
    n_solves = 0 if name == "buy_hold" else n_days * 8
    counts = {} if n_solves == 0 else {"optimal": n_solves - 1, "max_iter_reached": 1}
    result = BacktestResult(
        strategy=name,
        seed=int(seed),
        daily_nav=nav,
        daily_turnover=turnover,
        costs={
            "commission": traded * 2.0e-4,
            "stamp_tax": traded * 2.5e-4,
            "slippage": traded * 5.0e-4,
            "impact_est": traded * 1.0e-5,
        },
        qp_solve_times=[0.01 + 0.0002 * i for i in range(n_solves)],
        qp_status_counts=counts,
        meta={"traded_value_cny": traded},
    )
    return summarize(result)


class RealisticFakeRunSeed:
    """``run_seed`` stand-in whose per-strategy dicts come from the real
    ``metrics.summarize``, so the test data cannot drift from production keys.

    (The pre-existing fake in test_benchmark_cli.py speaks a third, invented
    dialect -- which is precisely why it never exercised the renderer seam.)
    """

    def __init__(self):
        self.calls: list[int] = []

    def __call__(self, seed, cfg, fixed_lambdas=None):
        self.calls.append(int(seed))
        n_days = int(cfg.synth.n_days) - int(cfg.eval_start_day)
        out: dict = {
            name: _fake_result(seed, name, idx, n_days)
            for idx, name in enumerate(STRATEGIES)
        }
        out["seed"] = int(seed)
        return out


@pytest.fixture()
def realistic_run_seed(monkeypatch):
    fake = RealisticFakeRunSeed()
    monkeypatch.setattr(bench, "RUN_SEED_OVERRIDE", fake)
    return fake


def _run(tmp_path, mode="smoke", **extra):
    """Real run_benchmark -> (artifact, html) through the real renderer."""
    argv = ["--mode", mode, "--output", str(tmp_path / "art.json")]
    html_path = tmp_path / "rep.html"
    argv += ["--html-output", str(html_path)]
    for key, val in extra.items():
        flag = "--" + key.replace("_", "-")
        if isinstance(val, list):
            for v in val:
                argv += [flag, str(v)]
        else:
            argv += [flag, str(val)]
    args = bench.build_parser().parse_args(argv)
    artifact = bench.run_benchmark(args)
    return artifact, html_path.read_text(encoding="utf-8")


def _section_text(html: str, heading: str) -> str:
    """Tag-stripped text of one numbered section (up to the next <h2>)."""
    start = html.index(heading)
    nxt = html.find("<h2", start)
    chunk = html[start : nxt if nxt != -1 else len(html)]
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", chunk))


def _mean_metric(artifact, cell_idx, strategy, key):
    per_seed = artifact["cells"][cell_idx]["per_seed"]
    vals = [per_seed[s][strategy][key] for s in per_seed]
    return sum(vals) / len(vals)


class TestEndToEndBenchmarkArtifact:
    """The seam the incident escaped through: real artifact, real renderer."""

    def test_artifact_is_dialect_one(self, tmp_path, realistic_run_seed):
        artifact, _ = _run(tmp_path, seeds=4)
        assert artifact["schema"] == bench.SCHEMA
        cell = artifact["cells"][0]
        assert set(REQUIRED_SEED_METRICS) <= set(cell["per_seed"]["20260821"]["mpo"])
        assert "mpo_vs_fixed_weight" in cell["paired"]

    def test_main_results_numbers_reach_the_html(self, tmp_path, realistic_run_seed):
        artifact, html = _run(tmp_path, seeds=4)
        section = _section_text(html, "3. 主结果")
        for strategy in STRATEGIES:
            assert strategy in section
        for strategy in ("mpo", "fixed_weight"):
            for key in ("net_sharpe_annual", "net_return_annual", "avg_daily_turnover"):
                value = _mean_metric(artifact, 0, strategy, key)
                assert f"{value:.4f}" in section, f"{strategy}.{key} 未出现在主结果表"

    def test_no_required_region_degrades_to_na(self, tmp_path, realistic_run_seed):
        _, html = _run(tmp_path, seeds=4)
        for heading in ("3. 主结果", "4. 配对比较"):
            assert "n/a" not in _section_text(html, heading), f"{heading} 出现 n/a"
        config_section = _section_text(html, "2. 配置与环境")
        assert "config hash" in config_section
        assert "4 个: 20260821" in config_section

    def test_pairwise_numbers_reach_the_html(self, tmp_path, realistic_run_seed):
        artifact, html = _run(tmp_path, seeds=4)
        section = _section_text(html, "4. 配对比较")
        stats = artifact["cells"][0]["paired"]["mpo_vs_fixed_weight"]
        assert "fixed_weight" in section
        for key in ("mean_diff", "ci95_low", "ci95_high", "win_rate", "p_value", "hedges_g"):
            assert f"{stats[key]:.4f}" in section, f"paired.{key} 未出现在配对表"

    def test_config_hash_and_git_rev_are_real(self, tmp_path, realistic_run_seed):
        artifact, html = _run(tmp_path, seeds=4)
        digest = artifact["cells"][0]["config_hash"]
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert digest in html

    def test_qp_walltime_and_oracle_populated(self, tmp_path, realistic_run_seed):
        artifact, html = _run(tmp_path, seeds=4)
        qp = _section_text(html, "7. QP 求解墙钟")
        p50 = _mean_metric(artifact, 0, "mpo", "qp_time_p50")
        assert f"{p50:.4f}" in qp
        # 8 strategies x 4 seeds, buy_hold solves nothing; 1 non-optimal each.
        n_solves = 10 * 8 * 7 * 4
        assert str(n_solves) in qp
        assert str(7 * 4) in qp  # non-optimal count
        oracle = _section_text(html, "6. Oracle")
        assert "oracle Sharpe" in oracle and "n/a" not in oracle

    def test_sensitivity_axis_grid_renders_every_cell(self, tmp_path, realistic_run_seed):
        artifact, html = _run(tmp_path, mode="sensitivity", seeds=3, axis=["t1"])
        section = _section_text(html, "5. 敏感性分析")
        assert "轴: t1" in section
        assert len(artifact["cells"]) == 2
        for idx in range(2):
            sharpe = _mean_metric(artifact, idx, "mpo", "net_sharpe_annual")
            assert f"{sharpe:.4f}" in section
        assert "n/a" not in section

    def test_single_cell_mode_says_no_grid(self, tmp_path, realistic_run_seed):
        _, html = _run(tmp_path, seeds=4)
        # 单 cell 模式无敏感性网格: 这是真正可选区, 允许 n/a
        assert "本次运行无敏感性网格结果" in _section_text(html, "5. 敏感性分析")

    def test_self_contained_and_escaped(self, tmp_path, realistic_run_seed):
        _, html = _run(tmp_path, seeds=4)
        assert html.lstrip().startswith("<!DOCTYPE html>")
        assert "http://" not in html and "https://" not in html
        assert "<script" not in html and "<link" not in html


class TestRequiredRegionsRaise:
    """Missing required region => loud failure, never a hollow report."""

    @pytest.fixture()
    def artifact(self, tmp_path, realistic_run_seed):
        art, _ = _run(tmp_path, seeds=4)
        return art

    def _expect_raise(self, artifact, tmp_path, match):
        with pytest.raises(ReportContractError, match=match):
            render_html(artifact, str(tmp_path / "boom.html"))

    def test_no_cells(self, artifact, tmp_path):
        artifact["cells"] = []
        self._expect_raise(artifact, tmp_path, "cells")

    def test_missing_config_hash(self, artifact, tmp_path):
        artifact["cells"][0].pop("config_hash")
        self._expect_raise(artifact, tmp_path, "config_hash")

    def test_missing_seeds(self, artifact, tmp_path):
        artifact["cells"][0]["seeds"] = []
        self._expect_raise(artifact, tmp_path, "seeds")

    def test_missing_per_seed(self, artifact, tmp_path):
        artifact["cells"][0]["per_seed"] = {}
        self._expect_raise(artifact, tmp_path, "per_seed")

    def test_missing_paired(self, artifact, tmp_path):
        artifact["cells"][0]["paired"] = {}
        self._expect_raise(artifact, tmp_path, "paired")

    def test_missing_mpo_strategy(self, artifact, tmp_path):
        cell = artifact["cells"][0]
        cell["strategies"] = [s for s in cell["strategies"] if s != "mpo"]
        for rec in cell["per_seed"].values():
            rec.pop("mpo")
        self._expect_raise(artifact, tmp_path, "mpo")

    def test_foreign_metric_dialect_is_rejected(self, artifact, tmp_path):
        """The invented dialect of the old CLI fake must not render as 'n/a'."""
        cell = artifact["cells"][0]
        for rec in cell["per_seed"].values():
            for name in list(rec):
                if isinstance(rec[name], dict):
                    rec[name] = {"net_sharpe": 1.0, "turnover": 0.5}
        self._expect_raise(artifact, tmp_path, "缺少必需指标")

    def test_schema_rename_still_validated(self, artifact, tmp_path):
        """Shape-based detection: a renamed schema must not fall back to n/a."""
        artifact["schema"] = "mpo-t1.synthetic.v2"
        artifact["cells"][0]["paired"] = {}
        self._expect_raise(artifact, tmp_path, "paired")

    def test_recorded_pair_error_is_surfaced_not_raised(self, artifact, tmp_path):
        """A per-pair stats failure recorded by the benchmark stays visible."""
        artifact["cells"][0]["paired"]["mpo_vs_informed"] = {"error": "boom-1234"}
        html = render_html(artifact, str(tmp_path / "ok.html"))
        text = Path(html).read_text(encoding="utf-8")
        assert "boom-1234" in text


class TestSummarizeKeyContract:
    """Drift guard: the adapter reads keys the real summarize() must produce."""

    def test_summarize_provides_every_key_the_adapter_reads(self):
        keys = set(_fake_result(20260821, "mpo", 0, 10))
        assert set(REQUIRED_SEED_METRICS) <= keys
        for optional in (
            "cost_bp_breakdown",
            "max_drawdown",
            "t1_violations",
            "qp_time_p50",
            "qp_time_p95",
            "qp_status_counts",
        ):
            assert optional in keys, f"summarize 不再产出 {optional}, 适配层需同步"

    def test_adapter_aggregates_match_hand_computed_means(self, tmp_path, realistic_run_seed):
        artifact, _ = _run(tmp_path, seeds=4)
        adapted = adapt_benchmark_artifact(artifact)
        aggregated = adapted["strategy_groups"][0]["strategies"]["mpo"]
        assert aggregated["sharpe_mean"] == pytest.approx(
            _mean_metric(artifact, 0, "mpo", "net_sharpe_annual")
        )
        assert aggregated["annual_net_return"] == pytest.approx(
            _mean_metric(artifact, 0, "mpo", "net_return_annual")
        )
        assert aggregated["t1_violations"] == 0
        assert aggregated["n_seeds"] == 4
