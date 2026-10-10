"""MPO-T1 self-contained HTML report renderer (SPEC section 9/10).

Renders the aggregated benchmark artifact (a plain ``dict``, typically the
JSON produced by ``benchmarks/mpo_t1_benchmark.py``) into a single
self-contained Chinese HTML file: inline CSS, no external assets, pure
string templates. All numbers are formatted to 4 decimal places; every
missing / malformed field degrades to ``'n/a'`` instead of raising.

Time convention reminder (project-wide): all horizons are TRADING minutes,
tau = day*240 + minute, overnight counts as 0 minutes; the report only
displays results and never recomputes them (the sole exception is the Holm
correction of the stored raw p-values, shown alongside them).

Two input dialects
------------------
1. **Benchmark artifact** (``schema == 'mpo-t1.synthetic.v1'``): the real
   product of ``benchmarks/mpo_t1_benchmark.py:build_artifact`` --
   ``cells[] / per_seed / paired / tuning``. It is normalized by
   :func:`adapt_benchmark_artifact` into dialect 2 and, because it is the
   production path, its **required regions are validated and a missing one
   raises** :class:`ReportContractError` instead of degrading to ``'n/a'``.
2. **Internal dialect** (below): the pre-aggregated form the section
   renderers consume. Hand-built artifacts in this dialect stay fully
   fail-soft (every missing field -> ``'n/a'``, never raises).

Incident (Iteration 38, 2026-08-23): dialect 1 was never accepted -- the
renderer only ever spoke dialect 2, while the benchmark only ever emitted
dialect 1, so the unconditional ``'n/a'`` fail-soft silently turned all four
committed HTML reports (Iteration 37 main/sensitivity/smoke + Iteration 38
sensitivity2) into data-free shells that still looked well-formed. The escape
path was test-shaped too: ``tests/mpo/test_report.py`` only fed hand-built
dialect-2 dicts and ``tests/mpo/test_benchmark_cli.py`` monkeypatched the real
renderer out, so the seam was never crossed by real data. Hence the strict
required-region validation on dialect 1 plus the end-to-end
``build_artifact() -> render_html()`` tests: fail-soft is only allowed where a
field is genuinely optional.

Expected (all optional) internal-dialect fields
-----------------------------------------------
- ``mode``, ``config_hash``, ``git_rev``, ``results_sha256``, ``seeds``
  (list[int]), ``env`` (dict: python/numpy/scipy/osqp/host ...)
- ``strategies``: {name: metrics} or list of dicts with ``name``/``strategy``
  key. Metric keys used by the main table: ``sharpe_mean``, ``sharpe_std``,
  ``annual_net_return``, ``daily_turnover``, ``cost_bp``, ``max_drawdown``,
  ``t1_violations``.
- ``pairwise``: list of dicts ``{baseline, mean_diff, ci_low, ci_high (or
  ci95=[lo,hi]), win_rate, p_holm (or p_value), hedges_g}``; the row is
  bold-green when the CI95 lower bound > 0.
- ``sensitivity``: {axis_name: list[cell dict]} -- one generic table per axis.
- ``oracle``: dict of gain-decomposition numbers.
- ``qp_walltime``: dict with p50/p95 (optionally cold_*/hot_*) in seconds.

Multi-cell extensions of the internal dialect (used by the adapter; each
falls back to its singular legacy key when absent):

- ``strategy_groups``: [{``label``, ``strategies``}] -- one main table per cell.
- ``pairwise_groups``: [{``label``, ``pairwise``}] -- one paired table per cell.
- ``oracle_rows`` / ``qp_walltime_rows``: list of row dicts (one per cell).
- ``extra_meta``: [(label, value)] appended to the config/env table.
"""

from __future__ import annotations

import html as _html
import math
import numbers

BENCHMARK_SCHEMA = "mpo-t1.synthetic.v1"

MPO_STRATEGY = "mpo"

#: Per-seed metric keys the main table needs (``openpo.mpo.metrics.summarize``).
REQUIRED_SEED_METRICS = ("net_sharpe_annual", "net_return_annual", "avg_daily_turnover")


class ReportContractError(ValueError):
    """A benchmark artifact is missing a region the report must display.

    Raised only for ``schema == BENCHMARK_SCHEMA`` artifacts: on the production
    path a hollow report is worse than a loud failure (see module docstring's
    Iteration 38 incident note).
    """


# ---------------------------------------------------------------------------
# formatting helpers (all fail-soft: bad input -> 'n/a')
# ---------------------------------------------------------------------------

NA = "n/a"


def _esc(x) -> str:
    """HTML-escape any value's string form."""
    return _html.escape(str(x), quote=True)


def _fmt_num(x, digits: int = 4) -> str:
    """Format a scalar to fixed decimals; anything non-finite -> 'n/a'.

    Accepts python numbers and numpy scalars (via float()); bools are treated
    as non-numeric to avoid rendering True as 1.0000.
    """
    if x is None or isinstance(x, bool):
        return NA
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NA
    if not math.isfinite(v):
        return NA
    return f"{v:.{digits}f}"


def _fmt_int(x) -> str:
    if x is None or isinstance(x, bool):
        return NA
    try:
        v = float(x)
    except (TypeError, ValueError):
        return NA
    if not math.isfinite(v):
        return NA
    return str(int(round(v)))


def _fmt_any(x) -> str:
    """Generic cell formatter: numbers -> 4dp, ints kept整数, else escaped str."""
    if x is None:
        return NA
    if isinstance(x, bool):
        return _esc(x)
    if isinstance(x, numbers.Integral):
        return _fmt_int(x)
    if isinstance(x, numbers.Real):
        return _fmt_num(x)
    return _esc(x)


def _get(d, key, default=None):
    """dict.get that tolerates non-dict containers."""
    if isinstance(d, dict):
        return d.get(key, default)
    return default


# ---------------------------------------------------------------------------
# benchmark-artifact adapter: dialect 1 (build_artifact) -> dialect 2
# ---------------------------------------------------------------------------


def _is_finite(x) -> bool:
    if x is None or isinstance(x, bool):
        return False
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _finite(values) -> list[float]:
    return [float(v) for v in values if _is_finite(v)]


def _mean(values):
    vals = _finite(values)
    return sum(vals) / len(vals) if vals else None


def _std(values):
    """Sample std (ddof=1), matching the benchmark's own convention."""
    vals = _finite(values)
    if len(vals) < 2:
        return 0.0 if vals else None
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _sum_int(values):
    vals = _finite(values)
    return int(round(sum(vals))) if vals else None


def _seed_sort_key(seed):
    try:
        return (0, int(seed), "")
    except (TypeError, ValueError):
        return (1, 0, str(seed))


def _seed_records(cell, strategy) -> list[dict]:
    """Per-seed summaries of one strategy in one cell, ordered by seed."""
    per_seed = _get(cell, "per_seed")
    if not isinstance(per_seed, dict):
        return []
    out = []
    for seed in sorted(per_seed, key=_seed_sort_key):
        rec = per_seed[seed]
        metrics = _get(rec, strategy)
        if isinstance(metrics, dict):
            out.append(metrics)
    return out


def _cell_strategy_names(cell) -> list[str]:
    """Declared strategy names, else the first-seen union across per_seed."""
    names = _get(cell, "strategies")
    if isinstance(names, (list, tuple)) and names:
        return [str(n) for n in names]
    out: list[str] = []
    per_seed = _get(cell, "per_seed")
    if isinstance(per_seed, dict):
        for seed in sorted(per_seed, key=_seed_sort_key):
            rec = per_seed[seed]
            if isinstance(rec, dict):
                for name in rec:
                    if str(name) not in out:
                        out.append(str(name))
    return out


def _total_cost_bp(record):
    """Total cost in bp of traded value = sum of the per-component breakdown."""
    breakdown = _get(record, "cost_bp_breakdown")
    if not isinstance(breakdown, dict):
        return None
    vals = _finite(breakdown.values())
    return sum(vals) if vals else None


def _aggregate_strategy(records: list[dict]) -> dict:
    """Cross-seed aggregate of one strategy, in internal-dialect metric keys."""
    return {
        "sharpe_mean": _mean(r.get("net_sharpe_annual") for r in records),
        "sharpe_std": _std(r.get("net_sharpe_annual") for r in records),
        "annual_net_return": _mean(r.get("net_return_annual") for r in records),
        "daily_turnover": _mean(r.get("avg_daily_turnover") for r in records),
        "cost_bp": _mean(_total_cost_bp(r) for r in records),
        "max_drawdown": _mean(r.get("max_drawdown") for r in records),
        "t1_violations": _sum_int(r.get("t1_violations") for r in records),
        "n_seeds": len(records),
    }


def _attach_holm(rows: list[dict]) -> None:
    """Add ``p_holm`` to rows with a finite raw p (family = one cell's tests).

    The benchmark stores raw paired-t p-values only; the Holm step-down is the
    single display-time computation this module performs, and the raw value is
    always shown next to it so nothing is hidden.
    """
    idx = [i for i, r in enumerate(rows) if _is_finite(r.get("p_value"))]
    if not idx:
        return
    try:
        from openpo.mpo.stats import holm_correction
    except Exception:  # noqa: BLE001 -- stats missing must not sink the report
        return
    try:
        adjusted = holm_correction([float(rows[i]["p_value"]) for i in idx])
    except Exception:  # noqa: BLE001
        return
    for i, adj in zip(idx, list(adjusted)):
        rows[i]["p_holm"] = float(adj)


def _cell_pairwise(cell) -> list[dict]:
    """``paired`` dict -> internal-dialect pairwise rows (order preserved)."""
    paired = _get(cell, "paired")
    rows: list[dict] = []
    if not isinstance(paired, dict):
        return rows
    for key, stats in paired.items():
        if not isinstance(stats, dict):
            continue
        baseline = stats.get("strategy_b")
        if not baseline:
            prefix = f"{MPO_STRATEGY}_vs_"
            baseline = str(key)[len(prefix) :] if str(key).startswith(prefix) else str(key)
        row = {
            "baseline": str(baseline),
            "mean_diff": stats.get("mean_diff"),
            "ci_low": stats.get("ci95_low"),
            "ci_high": stats.get("ci95_high"),
            "win_rate": stats.get("win_rate"),
            "p_value": stats.get("p_value"),
            "hedges_g": stats.get("hedges_g"),
        }
        # A per-pair failure is recorded by the benchmark, not swallowed here.
        if stats.get("error"):
            row["error"] = str(stats["error"])
        rows.append(row)
    _attach_holm(rows)
    return rows


_SINGLE_PERIOD = ("single_30", "single_240", "single_2400")
_ORACLE_STRATEGY = "oracle_mpo"


def _cell_oracle_row(label: str, aggregates: dict):
    """Oracle gain decomposition for one cell, or None when no oracle ran."""
    oracle = aggregates.get(_ORACLE_STRATEGY)
    mpo = aggregates.get(MPO_STRATEGY)
    if not isinstance(oracle, dict) or not isinstance(mpo, dict):
        return None
    o_sharpe, m_sharpe = oracle.get("sharpe_mean"), mpo.get("sharpe_mean")
    best_name, best_sharpe = None, None
    for name in _SINGLE_PERIOD:
        s = _get(aggregates.get(name), "sharpe_mean")
        if _is_finite(s) and (best_sharpe is None or float(s) > best_sharpe):
            best_name, best_sharpe = name, float(s)
    both = _is_finite(o_sharpe) and _is_finite(m_sharpe)
    return {
        "cell": label,
        "oracle Sharpe": o_sharpe,
        "MPO Sharpe": m_sharpe,
        "标定损耗 (oracle−MPO)": (float(o_sharpe) - float(m_sharpe)) if both else None,
        "最佳单期基线": best_name or NA,
        "结构增益 (MPO−最佳单期)": (
            float(m_sharpe) - best_sharpe
            if (_is_finite(m_sharpe) and best_sharpe is not None)
            else None
        ),
    }


def _cell_qp_row(label: str, cell) -> dict:
    """MPO solve-time percentiles plus the whole cell's QP status health."""
    mpo_records = _seed_records(cell, MPO_STRATEGY)
    solves = 0
    non_optimal = 0
    for name in _cell_strategy_names(cell):
        for record in _seed_records(cell, name):
            counts = _get(record, "qp_status_counts")
            if isinstance(counts, dict):
                for status, n in counts.items():
                    if not _is_finite(n):
                        continue
                    solves += int(n)
                    if str(status) != "optimal":
                        non_optimal += int(n)
    return {
        "cell": label,
        "MPO p50": _mean(r.get("qp_time_p50") for r in mpo_records),
        "MPO p95": _mean(r.get("qp_time_p95") for r in mpo_records),
        "QP 求解数 (全策略)": solves,
        "非最优": non_optimal,
    }


def _sensitivity_row(cell, aggregates: dict, pairwise: list[dict]) -> dict:
    """One compact cross-cell comparison row for the per-axis table."""
    mpo = aggregates.get(MPO_STRATEGY) or {}
    fixed = aggregates.get("fixed_weight") or {}
    versus = next((p for p in pairwise if p.get("baseline") == "fixed_weight"), {})
    return {
        "值": _get(cell, "value"),
        "MPO Sharpe": mpo.get("sharpe_mean"),
        "FixedWeight Sharpe": fixed.get("sharpe_mean"),
        "差 (MPO−FW)": versus.get("mean_diff"),
        "CI95 下界": versus.get("ci_low"),
        "CI95 上界": versus.get("ci_high"),
        "p (raw)": versus.get("p_value"),
        "MPO 年化净收益": mpo.get("annual_net_return"),
        "MPO 日换手": mpo.get("daily_turnover"),
        "MPO MaxDD": mpo.get("max_drawdown"),
        "T+1违规": mpo.get("t1_violations"),
    }


def _cell_label(cell) -> str:
    """Section heading for one cell: id, axis=value and the config fingerprint."""
    cell_id = _get(cell, "cell_id") or NA
    bits = []
    axis = _get(cell, "axis")
    if axis:
        bits.append(f"轴 {axis} = {_get(cell, 'value')}")
    digest = str(_get(cell, "config_hash") or "")
    if digest:
        bits.append(f"config {digest[:12]}")
    return f"cell {cell_id}" + (f"（{', '.join(bits)}）" if bits else "")


def _config_hash_meta(cells: list[dict]) -> tuple[str, list[tuple[str, str]]]:
    """(summary, per-config rows). Multi-cell runs list every full hash so the
    provenance stays auditable instead of collapsing to a truncated digest."""
    unique: list[tuple[str, str]] = []
    for cell in cells:
        digest = str(_get(cell, "config_hash") or "")
        if digest and digest not in [d for _, d in unique]:
            unique.append((str(_get(cell, "cell_id") or NA), digest))
    if not unique:
        return NA, []
    if len(unique) == 1:
        return unique[0][1], []
    summary = f"{len(unique)} 个配置（逐 cell 全值见下）"
    return summary, [(f"config hash · {cid}", digest) for cid, digest in unique]


def _seed_union(cells: list[dict]) -> list:
    seen: list = []
    for cell in cells:
        seeds = _get(cell, "seeds")
        if isinstance(seeds, (list, tuple)):
            for s in seeds:
                if s not in seen:
                    seen.append(s)
    return sorted(seen, key=_seed_sort_key)


def looks_like_benchmark_artifact(artifact) -> bool:
    """True for dialect-1 artifacts, by schema **or** by shape.

    The shape check is deliberate defense in depth: a future schema rename must
    fail loudly on the strict path rather than silently fall back to the
    fail-soft path and re-create the Iteration 38 hollow reports.
    """
    if not isinstance(artifact, dict):
        return False
    if artifact.get("schema") == BENCHMARK_SCHEMA:
        return True
    cells = artifact.get("cells")
    return (
        isinstance(cells, list)
        and bool(cells)
        and any(isinstance(c, dict) and "per_seed" in c for c in cells)
    )


def validate_benchmark_artifact(artifact) -> None:
    """Raise :class:`ReportContractError` when a required region is missing.

    Required = everything the report must *show* to be worth publishing: the
    per-cell main-results inputs (``per_seed`` with the metric keys), the
    paired-comparison block, and the ``config_hash`` + ``seeds`` provenance.
    Genuinely optional regions (``results_sha256``, oracle strategy, the
    sensitivity grid in single-cell modes) are still allowed to render 'n/a'.
    """
    if not isinstance(artifact, dict):
        raise ReportContractError(f"artifact 必须是 dict, 收到 {type(artifact).__name__}")
    cells = artifact.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ReportContractError(
            f"artifact['cells'] 缺失或为空 (schema={artifact.get('schema')!r}); "
            "无 cell 即无主结果可渲染"
        )
    for i, cell in enumerate(cells):
        where = f"cells[{i}]"
        if not isinstance(cell, dict):
            raise ReportContractError(f"{where} 必须是 dict, 收到 {type(cell).__name__}")
        where = f"{where}(cell_id={cell.get('cell_id')!r})"
        if not str(cell.get("config_hash") or "").strip():
            raise ReportContractError(f"{where} 缺少 config_hash (元数据必需区)")
        seeds = cell.get("seeds")
        if not isinstance(seeds, (list, tuple)) or not seeds:
            raise ReportContractError(f"{where} 缺少 seeds (元数据必需区)")
        per_seed = cell.get("per_seed")
        if not isinstance(per_seed, dict) or not per_seed:
            raise ReportContractError(f"{where} 缺少 per_seed (主结果必需区)")
        names = _cell_strategy_names(cell)
        if not names:
            raise ReportContractError(f"{where} per_seed 中没有任何策略记录 (主结果必需区)")
        if MPO_STRATEGY not in names:
            raise ReportContractError(
                f"{where} 缺少 {MPO_STRATEGY!r} 策略 (主结果与配对比较的主体)"
            )
        for name in names:
            records = _seed_records(cell, name)
            if not records:
                raise ReportContractError(f"{where} 策略 {name!r} 在 per_seed 中无记录")
            missing = [
                key
                for key in REQUIRED_SEED_METRICS
                if not any(_is_finite(r.get(key)) for r in records)
            ]
            if missing:
                raise ReportContractError(
                    f"{where} 策略 {name!r} 的 per_seed 记录缺少必需指标 {missing}; "
                    f"实有键={sorted(records[0])}"
                )
        paired = cell.get("paired")
        if not isinstance(paired, dict) or not paired:
            raise ReportContractError(f"{where} 缺少 paired (配对比较必需区)")


def adapt_benchmark_artifact(artifact: dict) -> dict:
    """Normalize a ``mpo-t1.synthetic.v1`` artifact into the internal dialect.

    Validates first (raises on a missing required region), then aggregates each
    cell across seeds, converts ``paired`` into pairwise rows and groups the
    cells by sensitivity axis.
    """
    validate_benchmark_artifact(artifact)
    cells = artifact["cells"]
    multi_cell = len(cells) > 1

    strategy_groups, pairwise_groups, oracle_rows, qp_rows = [], [], [], []
    sensitivity: dict[str, list[dict]] = {}
    for cell in cells:
        label = _cell_label(cell) if multi_cell else None
        short = str(_get(cell, "cell_id") or NA)
        aggregates = {
            name: _aggregate_strategy(_seed_records(cell, name))
            for name in _cell_strategy_names(cell)
        }
        pairwise = _cell_pairwise(cell)
        strategy_groups.append({"label": label, "strategies": aggregates})
        pairwise_groups.append({"label": label, "pairwise": pairwise})
        oracle_row = _cell_oracle_row(short, aggregates)
        if oracle_row is not None:
            oracle_rows.append(oracle_row)
        qp_rows.append(_cell_qp_row(short, cell))
        axis = _get(cell, "axis")
        if axis:
            sensitivity.setdefault(str(axis), []).append(
                _sensitivity_row(cell, aggregates, pairwise)
            )

    config = _get(cells[0], "config")
    tuning = _get(artifact, "tuning") or {}
    hash_summary, hash_rows = _config_hash_meta(cells)
    extra_meta = list(hash_rows) + [
        ("schema", artifact.get("schema")),
        ("cells", len(cells)),
        (
            "FixedWeight lambdas",
            f"{tuning.get('lambdas')} (来源: {tuning.get('source', NA)})" if tuning else NA,
        ),
        ("created (UTC)", artifact.get("created_utc") or NA),
        ("wall clock (s)", _fmt_num(artifact.get("wall_clock_seconds"), 1)),
    ]
    return {
        "mode": _get(artifact, "mode"),
        "git_rev": _get(artifact, "git_rev"),
        "env": _get(artifact, "env"),
        "results_sha256": _get(artifact, "results_sha256"),
        "config_hash": hash_summary,
        "seeds": _seed_union(cells),
        "extra_meta": extra_meta,
        "strategy_groups": strategy_groups,
        "pairwise_groups": pairwise_groups,
        "oracle_rows": oracle_rows,
        "qp_walltime_rows": qp_rows,
        "sensitivity": sensitivity,
        "bootstrap_iters": _get(config, "bootstrap_iters", 10000),
    }


# ---------------------------------------------------------------------------
# small HTML builders
# ---------------------------------------------------------------------------


def _table(headers: list[str], rows: list[list[str]]) -> str:
    """Rows are already-formatted (escaped or trusted) cell strings."""
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows
    )
    if not rows:
        body = f'<tr><td colspan="{max(len(headers), 1)}">{NA}</td></tr>'
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _kv_table(pairs: list[tuple[str, str]]) -> str:
    return _table(["项", "值"], [[_esc(k), v] for k, v in pairs])


def _section(sid: str, title: str, inner: str) -> str:
    return f'<section id="{sid}"><h2>{_esc(title)}</h2>{inner}</section>'


# ---------------------------------------------------------------------------
# section renderers
# ---------------------------------------------------------------------------


def _sec_background(artifact: dict) -> str:
    mode = _get(artifact, "mode")
    mode_txt = f"（本次运行模式: {_esc(mode)}）" if mode else ""
    para = (
        "<p>本报告评估多周期信号（return_1_30 / return_1_240 / return_1_2400）"
        "联合优化下单策略 MPO-T1。方法论参考 Axioma 051 "
        "《Multi-Period Portfolio Optimization with Alpha Decay》：将未来 2400 交易分钟"
        "划分为至多 4 个阶段，按各信号的分段期限结构与 AR(1) 衰减把 alpha 摊到各阶段，"
        "在统一 QP 中同时优化各阶段目标权重，显式建模 A 股 T+1 可卖约束"
        "（今日买入次日方可卖出）与非对称交易成本（买 7bp / 卖 12bp 含印花税）。"
        "全部 horizon 与衰减按交易分钟计（隔夜记 0 分钟）。对照基线包括单信号单期、"
        "固定权重复合单期、Israelov-Katz 门控策略与买入持有锚；oracle-MPO 使用真实分段 "
        "alpha 给出增益上界。" + mode_txt + "</p>"
    )
    return _section("background", "1. 项目目标与背景", para)


def _sec_config_env(artifact: dict) -> str:
    seeds = _get(artifact, "seeds")
    if isinstance(seeds, (list, tuple)):
        seeds_txt = f"{len(seeds)} 个: " + ", ".join(str(s) for s in seeds[:32])
        if len(seeds) > 32:
            seeds_txt += ", ..."
    else:
        seeds_txt = NA
    pairs = [
        ("config hash", _esc(_get(artifact, "config_hash") or NA)),
        ("seeds", _esc(seeds_txt)),
        ("git rev", _esc(_get(artifact, "git_rev") or NA)),
        ("results sha256", _esc(_get(artifact, "results_sha256") or NA)),
    ]
    extra = _get(artifact, "extra_meta")
    if isinstance(extra, (list, tuple)):
        for item in extra:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), _esc(item[1] if item[1] is not None else NA)))
    env = _get(artifact, "env")
    if isinstance(env, dict) and env:
        for k in sorted(env):
            pairs.append((f"env.{k}", _esc(env[k])))
    else:
        pairs.append(("env", NA))
    return _section("config", "2. 配置与环境", _kv_table(pairs))


_MAIN_COLS = [
    ("净Sharpe (均值±std)", None),  # special composite column
    ("年化净收益", "annual_net_return"),
    ("日换手", "daily_turnover"),
    ("成本(bp)", "cost_bp"),
    ("MaxDD", "max_drawdown"),
    ("T+1违规", "t1_violations"),
]


def _iter_strategies(artifact: dict):
    """Yield (name, metrics_dict) tolerating dict / list layouts."""
    strategies = _get(artifact, "strategies")
    if isinstance(strategies, dict):
        for name, m in strategies.items():
            yield str(name), m if isinstance(m, dict) else {}
    elif isinstance(strategies, (list, tuple)):
        for m in strategies:
            if isinstance(m, dict):
                name = m.get("name") or m.get("strategy") or NA
                yield str(name), m


def _groups(artifact: dict, plural_key: str, singular_key: str):
    """Yield (label, payload) per cell; legacy single-payload artifacts -> one
    unlabeled group, so both dialects share one rendering path."""
    groups = _get(artifact, plural_key)
    if isinstance(groups, (list, tuple)) and groups:
        for g in groups:
            if isinstance(g, dict):
                yield _get(g, "label"), _get(g, singular_key)
    else:
        yield None, _get(artifact, singular_key)


def _sec_main_results(artifact: dict) -> str:
    headers = ["策略"] + [c[0] for c in _MAIN_COLS]
    parts = []
    for label, strategies in _groups(artifact, "strategy_groups", "strategies"):
        rows = []
        for name, m in _iter_strategies({"strategies": strategies}):
            sharpe = f"{_fmt_num(_get(m, 'sharpe_mean'))} ± {_fmt_num(_get(m, 'sharpe_std'))}"
            row = [_esc(name), sharpe]
            for _lbl, key in _MAIN_COLS[1:]:
                if key == "t1_violations":
                    row.append(_fmt_int(_get(m, key)))
                else:
                    row.append(_fmt_num(_get(m, key)))
            rows.append(row)
        if label:
            parts.append(f"<h3>{_esc(label)}</h3>")
        parts.append(_table(headers, rows))
    return _section("main-results", "3. 主结果（跨 seed 聚合）", "".join(parts))


def _pair_ci(p: dict) -> tuple[object, object]:
    """Extract (ci_low, ci_high) from either ci_low/ci_high or ci95=[lo,hi]."""
    lo, hi = _get(p, "ci_low"), _get(p, "ci_high")
    ci = _get(p, "ci95")
    if (lo is None or hi is None) and isinstance(ci, (list, tuple)) and len(ci) == 2:
        lo, hi = ci[0], ci[1]
    return lo, hi


def _pairwise_rows(pairwise) -> list[list[str]]:
    rows = []
    if not isinstance(pairwise, (list, tuple)):
        return rows
    for p in pairwise:
        if not isinstance(p, dict):
            continue
        lo, hi = _pair_ci(p)
        ci_txt = f"[{_fmt_num(lo)}, {_fmt_num(hi)}]"
        # CI95 下界 > 0 => 显著优于基线，加粗绿显示
        significant = False
        try:
            significant = lo is not None and math.isfinite(float(lo)) and float(lo) > 0
        except (TypeError, ValueError):
            significant = False
        baseline = _esc(_get(p, "baseline") or NA)
        # A recorded per-pair failure is surfaced, never rendered as a bare n/a.
        if _get(p, "error"):
            baseline += f' <span class="note">[统计失败: {_esc(p["error"])}]</span>'
        cells = [
            baseline,
            _fmt_num(_get(p, "mean_diff")),
            ci_txt,
            _fmt_num(_get(p, "win_rate")),
            _fmt_num(_get(p, "p_value")),
            _fmt_num(_get(p, "p_holm", _get(p, "p_value"))),
            _fmt_num(_get(p, "hedges_g")),
        ]
        if significant:
            cells = [f'<span class="sig">{c}</span>' for c in cells]
        rows.append(cells)
    return rows


def _sec_pairwise(artifact: dict) -> str:
    headers = ["基线", "Sharpe 均差", "CI95", "胜率", "p (raw)", "p (Holm)", "Hedges g"]
    note = (
        "<p class=\"note\">配对差 = MPO − 基线（同 seed 同市场路径）；"
        "CI95 下界 &gt; 0 的行以<span class=\"sig\">粗体绿色</span>标注。"
        "主检验（预注册）: MPO vs FixedWeight。p (Holm) 的校正族 = 同一 cell 内的"
        "全部基线比较；原始 p 值并列显示。</p>"
    )
    parts = [note]
    for label, pairwise in _groups(artifact, "pairwise_groups", "pairwise"):
        if label:
            parts.append(f"<h3>{_esc(label)}</h3>")
        parts.append(_table(headers, _pairwise_rows(pairwise)))
    return _section("pairwise", "4. 配对比较（MPO vs 各基线）", "".join(parts))


def _generic_cells_table(cells) -> str:
    """Render a list of dict cells as a table with the union of keys as columns."""
    if not isinstance(cells, (list, tuple)) or not cells:
        return _table(["值"], [])
    keys: list[str] = []
    for c in cells:
        if isinstance(c, dict):
            for k in c:
                if k not in keys:
                    keys.append(str(k))
    if not keys:
        return _table(["值"], [])
    rows = []
    for c in cells:
        if isinstance(c, dict):
            rows.append([_fmt_any(c.get(k)) for k in keys])
    return _table([_esc(k) for k in keys], rows)


def _sec_sensitivity(artifact: dict) -> str:
    sens = _get(artifact, "sensitivity")
    if not isinstance(sens, dict) or not sens:
        inner = f"<p>{NA}（本次运行无敏感性网格结果）</p>"
        return _section("sensitivity", "5. 敏感性分析", inner)
    parts = []
    for axis in sens:
        parts.append(f"<h3>轴: {_esc(axis)}</h3>")
        parts.append(_generic_cells_table(sens[axis]))
    return _section("sensitivity", "5. 敏感性分析", "".join(parts))


def _sec_oracle(artifact: dict) -> str:
    rows = _get(artifact, "oracle_rows")
    oracle = _get(artifact, "oracle")
    if isinstance(rows, (list, tuple)) and rows:
        inner = _generic_cells_table(rows)
    elif isinstance(oracle, dict) and oracle:
        pairs = [(str(k), _fmt_any(oracle[k])) for k in oracle]
        inner = _kv_table(pairs)
    else:
        inner = f"<p>{NA}</p>"
    note = (
        "<p class=\"note\">oracle-MPO 使用合成市场的真实分段 alpha；"
        "oracle − MPO 差 ≈ 标定损耗，MPO − 单期基线差 ≈ 优化结构增益。</p>"
    )
    return _section("oracle", "6. Oracle 增益分解", note + inner)


def _sec_qp_walltime(artifact: dict) -> str:
    rows = _get(artifact, "qp_walltime_rows")
    if isinstance(rows, (list, tuple)) and rows:
        note = (
            '<p class="note">p50 / p95 为 MPO 策略的单次 QP 求解墙钟（秒，跨 seed 均值）；'
            "求解数与非最优计数覆盖该 cell 的全部策略。</p>"
        )
        return _section("qp-walltime", "7. QP 求解墙钟（秒）", note + _generic_cells_table(rows))
    qp = _get(artifact, "qp_walltime")
    pairs: list[tuple[str, str]] = []
    if isinstance(qp, dict) and qp:
        # 常见键优先, 其余键按字典序补齐, 全部容错
        preferred = ["p50", "p95", "cold_p50", "cold_p95", "hot_p50", "hot_p95"]
        seen = set()
        for k in preferred:
            if k in qp:
                pairs.append((k, _fmt_num(qp[k])))
                seen.add(k)
        for k in sorted(qp):
            if k not in seen:
                pairs.append((str(k), _fmt_any(qp[k])))
    else:
        pairs = [("p50", NA), ("p95", NA)]
    inner = _kv_table([(k, v) for k, v in pairs])
    return _section("qp-walltime", "7. QP 求解墙钟（秒）", inner)


def _sec_methodology(artifact: dict) -> str:
    n_boot = _get(artifact, "bootstrap_iters", 10000)
    para = (
        "<p><b>配对设计</b>: 所有策略在同一 seed 生成的同一市场路径、同一执行层与同一"
        "T+1 台账下回测，指标差为逐 seed 配对差，消除市场路径方差。"
        f"<b>置信区间</b>: seed 层 bootstrap（{_fmt_int(n_boot)} 次重抽样）95% CI，"
        "并辅以配对 t 检验与 Hedges g 效应量；多基线比较采用 Holm 校正控制族错误率。"
        "<b>预注册主检验</b>: MPO vs FixedWeight 的净 Sharpe 配对差，其余比较为探索性。"
        "<b>局限</b>: ①市场为合成数据（合成日历无节假日，tau 即交易分钟计数），"
        "成本与冲击为近似模型；②期限结构在训练窗标定后冻结、不滚动重拟合；"
        "③成交模型为 30 分钟 VWAP 近似 + 参与率上限，未建模盘口深度；"
        "结论外推到实盘需谨慎。</p>"
    )
    return _section("methodology", "8. 方法论与置信度说明", para)


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

_CSS = """
body { font-family: 'Noto Sans CJK SC', 'PingFang SC', 'Microsoft YaHei',
       sans-serif; margin: 2em auto; max-width: 1100px; color: #222;
       background: #fafafa; line-height: 1.55; }
h1 { border-bottom: 3px solid #2c5f8a; padding-bottom: 0.3em; }
h2 { color: #2c5f8a; margin-top: 1.6em; }
h3 { color: #444; }
table { border-collapse: collapse; margin: 0.8em 0; width: 100%;
        background: #fff; font-variant-numeric: tabular-nums; }
th, td { border: 1px solid #ccc; padding: 5px 10px; text-align: right; }
th { background: #eef3f7; }
td:first-child, th:first-child { text-align: left; }
.sig { font-weight: bold; color: #1a7a1a; }
.note { color: #666; font-size: 0.92em; }
footer { margin-top: 2.5em; color: #999; font-size: 0.85em; }
"""


def render_html(artifact: dict, path: str) -> str:
    """Render the aggregated benchmark artifact to a self-contained HTML file.

    Parameters
    ----------
    artifact : dict
        Either a ``schema == BENCHMARK_SCHEMA`` benchmark artifact (normalized
        by :func:`adapt_benchmark_artifact`; a missing **required** region
        raises :class:`ReportContractError` rather than printing 'n/a') or a
        pre-aggregated internal-dialect dict (fully fail-soft: any missing
        field renders as ``'n/a'`` and this function never raises).
    path : str
        Output file path; written as UTF-8.

    Returns
    -------
    str
        The ``path`` argument (for chaining / logging).

    Raises
    ------
    ReportContractError
        Benchmark artifact missing a required region (main results, paired
        comparisons, or the config_hash/seeds provenance).
    """
    if not isinstance(artifact, dict):
        artifact = {}
    if looks_like_benchmark_artifact(artifact):
        artifact = adapt_benchmark_artifact(artifact)
    sections = [
        _sec_background(artifact),
        _sec_config_env(artifact),
        _sec_main_results(artifact),
        _sec_pairwise(artifact),
        _sec_sensitivity(artifact),
        _sec_oracle(artifact),
        _sec_qp_walltime(artifact),
        _sec_methodology(artifact),
    ]
    doc = (
        "<!DOCTYPE html>\n"
        '<html lang="zh-CN"><head><meta charset="utf-8">'
        "<title>MPO-T1 基准报告</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>MPO-T1 多周期组合优化基准报告</h1>"
        + "".join(sections)
        + "<footer>本报告为自包含 HTML（内联 CSS，无外部依赖），"
        "由 openpo.mpo.report.render_html 生成。</footer>"
        "</body></html>"
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path
