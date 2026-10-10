"""MPO-T1 benchmark runner (SPEC section 9).

Runs the synthetic multi-period-optimization experiment across seeds and
sensitivity cells, with per-task JSON checkpoints (resumable), atomic artifact
writes, and an optional HTML report.

Conventions (frozen, see docs/mpo-t1-SPEC.md):
- All horizons/decays are in TRADING minutes; overnight counts as 0 minutes.
- The synthetic calendar has no holidays, so ``tau`` is itself a trading-minute
  count and ``trading_minutes_between(a, b) == b - a``.

Modes
-----
smoke        2 seeds, N=100, D=20 (calib 10 + eval 10) -- fast全绿硬校验.
main         24 seeds, all defaults.
sensitivity  per-axis grids x 8 seeds
             (axes: cost, ic, decay, stages, t1, gamma, exec).

Checkpoints are written next to the artifact in ``<stem>-checkpoints/`` with
file names ``ckpt-<config_hash[:8]>[-lam<tag>][-strat<tag>]-seed<seed>.json`` so
identical cells across sensitivity axes share work while every input that
changes the payload (tuned lambdas, strategy roster) gets its own key.
``--resume`` skips tasks whose checkpoint loads.

Strategy roster
---------------
``--strategies a,b,c`` restricts / extends the roster, validated against
``backtest.KNOWN_STRATEGIES``. This is how the opt-in research arms are run,
e.g. the Iteration 42 delay-cost A/B::

    --strategies mpo,mpo_delayaware

Omitting the flag (or passing exactly ``DEFAULT_STRATEGIES``) is the unchanged
default path, checkpoint names included.

FixedWeight lambda tuning (SPEC section 7)
-----------------------------------------
The preregistered main test is MPO vs FixedWeight, and FixedWeight's lambdas
must be picked on *dedicated tuning seeds* (``tuning_seed_base+i``), disjoint
from the evaluation seeds (``seed_base+i``). Two flags drive this:

- ``--lambdas A,B,C``  explicit lambdas (normalized, non-negative), no tuning.
- ``--tune-lambdas K`` grid-search on K tuning seeds before running the cells,
  with ``--tune-grid-step`` (default 0.1 -> 66 lattice points).

Neither flag => unchanged legacy behaviour (``run_seed`` falls back to its
IC-proportional default). Tuning is parallelized here (one task per
(tuning seed, lattice point), each with its own JSON checkpoint) because
``backtest.tune_fixed_weights`` is serial and far too slow for the full grid;
the numeric semantics (per-point net annual Sharpe summed over tuning seeds,
argmax, ties -> lowest lattice index) are replicated exactly.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import multiprocessing
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from openpo.mpo.config import ExperimentConfig, config_hash  # noqa: E402

SCHEMA = "mpo-t1.synthetic.v1"
MPO_STRATEGY_NAME = "mpo"

# Sensitivity grids (SPEC section 9). Each entry: axis -> (target, field, values)
# where target is 'synth' or 'mpo' (which nested dataclass to replace).
SENSITIVITY_AXES: dict[str, tuple[str, str, tuple]] = {
    "cost": ("mpo", "cost_mult", (0.5, 1.0, 2.0)),
    "ic": ("synth", "ic_mult", (0.5, 1.0, 1.5)),
    "decay": ("mpo", "decay_mult", (0.5, 1.0, 2.0)),
    "stages": ("mpo", "max_stages", (1, 2, 4)),
    "t1": ("mpo", "t1_enabled", (True, False)),
    # Iteration 38: risk-aversion sweep. Iteration 37 found MPO wins on
    # risk-adjusted terms but deploys little capital at the frozen gamma=10
    # (1.9% annual return, MaxDD 0.7%); lowering gamma raises deployment, so
    # this axis answers whether the Sharpe edge survives at higher exposure.
    "gamma": ("mpo", "gamma", (2.0, 5.0, 10.0, 20.0)),
    # Iteration 38: stage-grid A/B on the S1 execution stage length (trading
    # minutes). Decision epochs stay 8/day (EPOCH_OFFSETS is untouched); only
    # the S1 stage widens. stages.py cuts S1 at today's close, so 60 is legal.
    "exec": ("mpo", "exec_minutes", (30, 60)),
}
SENSITIVITY_SEEDS_PER_CELL = 8

# Test hook: when set, used instead of importing openpo.mpo.backtest.run_seed.
# Inherited by fork-started Pool workers, so monkeypatching works under --jobs>1.
RUN_SEED_OVERRIDE = None

# Test hook for the tuning worker: callable(seed, cfg, lambdas) -> float score.
# When set, no real market is generated / no QP is solved.
TUNE_SCORE_OVERRIDE = None

DEFAULT_TUNE_GRID_STEP = 0.1


# ---------------------------------------------------------------------------
# Config construction per mode
# ---------------------------------------------------------------------------


def build_smoke_config() -> ExperimentConfig:
    """Smoke: 2 seeds, N=100, 20 days = 10 calib + 10 eval."""
    base = ExperimentConfig()
    return dataclasses.replace(
        base,
        synth=dataclasses.replace(base.synth, n_assets=100, n_days=20),
        mpo=dataclasses.replace(base.mpo, calib_days=10),
        eval_start_day=10,
        n_seeds=2,
    )


def build_main_config() -> ExperimentConfig:
    """Main: 24 seeds, all frozen defaults."""
    return ExperimentConfig()


def build_sensitivity_cells(axes: list[str]) -> list[dict]:
    """One cell per (axis, value); each cell carries its own ExperimentConfig."""
    base = ExperimentConfig()
    cells = []
    for axis in axes:
        target, field_name, values = SENSITIVITY_AXES[axis]
        for value in values:
            if target == "synth":
                cfg = dataclasses.replace(
                    base, synth=dataclasses.replace(base.synth, **{field_name: value})
                )
            else:
                cfg = dataclasses.replace(
                    base, mpo=dataclasses.replace(base.mpo, **{field_name: value})
                )
            cfg = dataclasses.replace(cfg, n_seeds=SENSITIVITY_SEEDS_PER_CELL)
            cells.append(
                {
                    "cell_id": f"{axis}={value}",
                    "axis": axis,
                    "value": value,
                    "config": cfg,
                }
            )
    return cells


def build_cells(mode: str, axes: list[str]) -> list[dict]:
    if mode == "smoke":
        return [{"cell_id": "smoke", "axis": None, "value": None, "config": build_smoke_config()}]
    if mode == "main":
        return [{"cell_id": "main", "axis": None, "value": None, "config": build_main_config()}]
    if mode == "sensitivity":
        return build_sensitivity_cells(axes)
    raise ValueError(f"unknown mode: {mode}")


def seeds_for(cfg: ExperimentConfig, n_seeds: int | None) -> list[int]:
    n = n_seeds if n_seeds is not None else cfg.n_seeds
    return [cfg.seed_base + i for i in range(n)]


# ---------------------------------------------------------------------------
# JSON / atomic-write helpers
# ---------------------------------------------------------------------------


def _json_default(obj):
    """Tolerant encoder for numpy scalars/arrays inside per-seed result dicts."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return encode_config(obj)
    return str(obj)


def atomic_write_json(path: Path, obj) -> None:
    """NamedTemporaryFile in the target dir + os.replace => atomic on POSIX."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = tempfile.NamedTemporaryFile(
        mode="w", dir=str(path.parent), suffix=".tmp", delete=False, encoding="utf-8"
    )
    try:
        with fd as f:
            json.dump(obj, f, indent=2, sort_keys=True, default=_json_default)
            f.flush()
            os.fsync(f.fileno())
        os.replace(fd.name, path)
    except BaseException:
        try:
            os.unlink(fd.name)
        except OSError:
            pass
        raise


def write_sha256_sidecar(path: Path) -> str:
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    sidecar = Path(str(path) + ".sha256")
    sidecar.write_text(f"{digest}  {Path(path).name}\n", encoding="utf-8")
    return digest


def encode_config(cfg) -> dict:
    """Nested dataclass -> JSON-safe dict (mirrors config.config_hash encoding)."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return {f.name: encode_config(getattr(cfg, f.name)) for f in dataclasses.fields(cfg)}
    if isinstance(cfg, (list, tuple)):
        return [encode_config(x) for x in cfg]
    return cfg


# ---------------------------------------------------------------------------
# Environment / provenance
# ---------------------------------------------------------------------------


def collect_env() -> dict:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    for mod in ("numpy", "scipy", "osqp"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001 -- version probing is best-effort
            env[mod] = None
    return env


def collect_git_rev() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 -- provenance must never kill the run
        return None


# ---------------------------------------------------------------------------
# Task execution (checkpointed)
# ---------------------------------------------------------------------------


def _get_run_seed():
    if RUN_SEED_OVERRIDE is not None:
        return RUN_SEED_OVERRIDE
    from openpo.mpo.backtest import run_seed  # deferred: sibling module

    return run_seed


def _lambda_tag(fixed_lambdas) -> str:
    """Short stable tag for a lambda triple (checkpoint key component)."""
    blob = ",".join(f"{float(x):.12g}" for x in fixed_lambdas)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def _strategies_tag(strategies) -> str:
    """Short stable tag for a strategy roster (checkpoint key component)."""
    blob = ",".join(str(s) for s in strategies)
    return hashlib.sha256(blob.encode()).hexdigest()[:8]


def checkpoint_path(
    ckpt_dir: Path, cfg_hash: str, seed: int, fixed_lambdas=None, strategies=None
) -> Path:
    """Eval-run checkpoint path.

    The checkpoint key must fingerprint EVERY input that changes the payload,
    else a resume silently serves a result computed under different settings:

    - ``fixed_lambdas is None`` -> legacy name kept verbatim (run_seed uses its
      IC-proportional default); tuned/explicit lambdas insert a ``lam`` tag.
    - ``strategies is None`` -> legacy name kept verbatim (run_seed uses
      ``DEFAULT_STRATEGIES``); an explicit roster inserts a ``strat`` tag.
      ``run_benchmark`` normalizes a roster equal to ``DEFAULT_STRATEGIES`` back
      to None, so the default run's checkpoint names never move.

    The roster tag is order sensitive on purpose: a reordered roster produces
    the same payload but a different file, which wastes work rather than risking
    a stale hit -- the conservative direction.
    """
    name = f"ckpt-{cfg_hash[:8]}"
    if fixed_lambdas is not None:
        name += f"-lam{_lambda_tag(fixed_lambdas)}"
    if strategies is not None:
        name += f"-strat{_strategies_tag(strategies)}"
    return Path(ckpt_dir) / f"{name}-seed{seed}.json"


def load_checkpoint(path: Path) -> dict | None:
    """Return checkpoint payload if the file exists and parses; else None."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "result" in data:
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def run_task(task: dict) -> dict:
    """Worker body: run one (seed, cell config) backtest and checkpoint it.

    ``task`` = {'seed', 'cfg', 'cfg_hash', 'ckpt_dir'} plus optional
    'fixed_lambdas' / 'strategies'. Returns the checkpoint payload. Safe under
    fork-based multiprocessing (pure-python arguments).

    Optional kwargs are passed to ``run_seed`` only when actually set, so the
    default path stays the exact two-argument call it has always been.
    """
    seed = task["seed"]
    cfg = task["cfg"]
    cfg_hash = task["cfg_hash"]
    fixed_lambdas = task.get("fixed_lambdas")
    strategies = task.get("strategies")
    run_seed = _get_run_seed()
    kwargs = {}
    if fixed_lambdas is not None:
        kwargs["fixed_lambdas"] = list(fixed_lambdas)
    if strategies is not None:
        kwargs["strategies"] = list(strategies)
    t0 = time.monotonic()
    result = run_seed(seed, cfg, **kwargs)
    payload = {
        "seed": seed,
        "config_hash": cfg_hash,
        "elapsed_seconds": time.monotonic() - t0,
        "result": result,
    }
    if fixed_lambdas is not None:
        payload["fixed_lambdas"] = [float(x) for x in fixed_lambdas]
    if strategies is not None:
        payload["strategies"] = [str(s) for s in strategies]
    atomic_write_json(
        checkpoint_path(task["ckpt_dir"], cfg_hash, seed, fixed_lambdas, strategies),
        payload,
    )
    return payload


def execute_tasks(tasks: list[dict], jobs: int, resume: bool) -> dict[tuple[str, int], dict]:
    """Run all (cfg_hash, seed) tasks, dedup'd, returning payload per key.

    With --resume, tasks whose checkpoint already loads are skipped. jobs<=1
    runs inline (no Pool), which keeps tests hermetic and errors legible.
    """
    unique: dict[tuple[str, int], dict] = {}
    for t in tasks:
        unique.setdefault((t["cfg_hash"], t["seed"]), t)

    done: dict[tuple[str, int], dict] = {}
    pending: list[dict] = []
    for key, t in unique.items():
        ckpt = None
        if resume:
            ckpt = load_checkpoint(
                checkpoint_path(
                    t["ckpt_dir"],
                    t["cfg_hash"],
                    t["seed"],
                    t.get("fixed_lambdas"),
                    t.get("strategies"),
                )
            )
        if ckpt is not None:
            done[key] = ckpt
        else:
            pending.append(t)

    if pending:
        if jobs <= 1:
            payloads = [run_task(t) for t in pending]
        else:
            with multiprocessing.Pool(processes=jobs) as pool:
                payloads = pool.map(run_task, pending)
        for t, payload in zip(pending, payloads):
            done[(t["cfg_hash"], t["seed"])] = payload
    return done


# ---------------------------------------------------------------------------
# FixedWeight lambda tuning (parallel; tuning seeds only)
# ---------------------------------------------------------------------------


def parse_lambdas(text: str) -> tuple[float, float, float]:
    """Parse/validate/normalize ``--lambdas A,B,C``.

    Raises ValueError (argparse turns this into a clean CLI error) on a wrong
    arity, non-numeric parts, negative entries, or a non-positive sum.
    """
    parts = [p.strip() for p in str(text).split(",")]
    if len(parts) != 3:
        raise ValueError(f"--lambdas needs exactly 3 comma-separated values, got {text!r}")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"--lambdas values must be numbers, got {text!r}") from None
    if any(v < 0 for v in vals):
        raise ValueError(f"--lambdas values must be non-negative, got {text!r}")
    total = sum(vals)
    if not total > 0:
        raise ValueError(f"--lambdas must have a positive sum, got {text!r}")
    return tuple(v / total for v in vals)


def parse_strategies(text: str) -> tuple[str, ...]:
    """Parse/validate ``--strategies a,b,c`` against ``backtest.KNOWN_STRATEGIES``.

    Raises ValueError (argparse turns this into a clean CLI error) on an empty
    list, an unknown name, or a duplicate. The roster is validated HERE rather
    than inside the worker so a typo costs a second, not a whole run; the import
    stays deferred so the module keeps importing without its siblings.
    """
    from openpo.mpo.backtest import KNOWN_STRATEGIES  # deferred: sibling module

    parts = [p.strip() for p in str(text).split(",")]
    if not parts or any(not p for p in parts):
        # Strict on purpose: a stray comma is a typo, and silently dropping the
        # empty slot would run a roster the operator did not ask for.
        raise ValueError(
            "--strategies needs a comma-separated list of names with no empty "
            f"entries, got {text!r}"
        )
    dupes = sorted({p for p in parts if parts.count(p) > 1})
    if dupes:
        raise ValueError(f"--strategies has duplicate name(s) {dupes} in {text!r}")
    unknown = [p for p in parts if p not in KNOWN_STRATEGIES]
    if unknown:
        raise ValueError(
            f"--strategies has unknown name(s) {unknown}; "
            f"roster: {', '.join(KNOWN_STRATEGIES)}"
        )
    return tuple(parts)


def positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise ValueError(f"expected a positive integer, got {text!r}")
    return n


def ic_proportional_lambdas(cfg: ExperimentConfig) -> list[float]:
    """run_seed's own default: lambda proportional to the target ICs."""
    ics = [float(x) for x in cfg.synth.target_ics]
    total = sum(ics)
    return [x / total for x in ics]


def tuning_config(mode: str) -> ExperimentConfig:
    """Config the lambda grid search runs under (one tuning per benchmark run).

    smoke tunes on the small smoke config; main/sensitivity tune on the frozen
    defaults (the sensitivity axes perturb the *evaluation* cells, not the
    preregistered lambda selection).
    """
    if mode == "smoke":
        return build_smoke_config()
    return build_main_config()


def tuning_seeds_for(cfg: ExperimentConfig, n: int) -> list[int]:
    return [cfg.tuning_seed_base + i for i in range(int(n))]


def tuning_grid(grid_step: float) -> list[tuple[float, float, float]]:
    """Simplex lattice from backtest.simplex_grid, as plain-python triples."""
    from openpo.mpo.backtest import simplex_grid  # deferred: sibling module

    return [tuple(float(x) for x in row) for row in simplex_grid(float(grid_step), dim=3)]


def tuning_checkpoint_path(
    ckpt_dir: Path, cfg_hash: str, seed: int, grid_index: int, grid_step: float
) -> Path:
    step_tag = f"{float(grid_step):g}".replace(".", "p").replace("-", "m")
    return Path(ckpt_dir) / (
        f"tune-{cfg_hash[:8]}-step{step_tag}-seed{seed}-g{int(grid_index):04d}.json"
    )


def load_tuning_checkpoint(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("score"), (int, float)):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def score_fixed_weight_point(seed: int, cfg: ExperimentConfig, lambdas) -> float:
    """Net annual Sharpe of ONE FixedWeight lattice point on ONE tuning seed.

    Mirrors ``backtest.tune_fixed_weights`` step for step (same market
    generation, same calibration inputs, same strategy construction, same
    score) but for a single lattice point, so the work can be sharded across
    processes. Regenerating the market per task costs ~a second and buys full
    task independence (checkpointable + resumable).
    """
    import importlib

    import numpy as np

    from openpo.mpo import backtest as bt

    synth = importlib.import_module("openpo.mpo.synth")
    term_structure = importlib.import_module("openpo.mpo.term_structure")
    strategies_mod = importlib.import_module("openpo.mpo.strategies")
    simulator_mod = importlib.import_module("openpo.mpo.simulator")

    scfg = cfg.synth
    mcfg = cfg.mpo
    eval_start = int(cfg.eval_start_day)
    if eval_start < mcfg.calib_days:
        raise ValueError(
            f"eval_start_day={eval_start} < calib_days={mcfg.calib_days}: lookahead"
        )

    market, signals = synth.generate_market(scfg, int(seed))
    ts_model = bt._instantiate(
        term_structure.fit_term_structure,
        {
            "market": market,
            "market_panel": market,
            "panel": market,
            "signals": signals,
            "signal_panel": signals,
            "cfg": mcfg,
            "config": mcfg,
            "mpo_cfg": mcfg,
            "mcfg": mcfg,
            "scfg": scfg,
            "calib_days": mcfg.calib_days,
        },
    )
    pool = bt._strategy_pool(market, signals, ts_model, cfg, scfg)
    strategy = bt._build_strategy(
        "fixed_weight", strategies_mod, pool, np.asarray(lambdas, dtype=np.float64)
    )
    sim = simulator_mod.Simulator(market, mcfg)
    result = sim.run(strategy, signals, range(eval_start, scfg.n_days))
    return float(result.net_sharpe_annual())


def _get_tune_scorer():
    if TUNE_SCORE_OVERRIDE is not None:
        return TUNE_SCORE_OVERRIDE
    return score_fixed_weight_point


def run_tuning_task(task: dict) -> dict:
    """Worker body: score one (tuning seed, lattice point) pair + checkpoint it.

    ``task`` = {'seed', 'cfg', 'cfg_hash', 'grid_index', 'lambdas',
    'grid_step', 'ckpt_dir'}.
    """
    seed = task["seed"]
    cfg = task["cfg"]
    cfg_hash = task["cfg_hash"]
    grid_index = int(task["grid_index"])
    lambdas = [float(x) for x in task["lambdas"]]
    t0 = time.monotonic()
    score = float(_get_tune_scorer()(seed, cfg, lambdas))
    payload = {
        "seed": seed,
        "config_hash": cfg_hash,
        "grid_index": grid_index,
        "lambdas": lambdas,
        "grid_step": float(task["grid_step"]),
        "score": score,
        "elapsed_seconds": time.monotonic() - t0,
    }
    atomic_write_json(
        tuning_checkpoint_path(
            task["ckpt_dir"], cfg_hash, seed, grid_index, task["grid_step"]
        ),
        payload,
    )
    return payload


def build_tuning_tasks(
    cfg: ExperimentConfig,
    cfg_hash: str,
    seeds: list[int],
    grid: list[tuple[float, float, float]],
    grid_step: float,
    ckpt_dir: Path,
) -> list[dict]:
    """Cartesian product (tuning seed x lattice point), in deterministic order."""
    tasks = []
    for seed in seeds:
        for gi, lam in enumerate(grid):
            tasks.append(
                {
                    "seed": int(seed),
                    "cfg": cfg,
                    "cfg_hash": cfg_hash,
                    "grid_index": gi,
                    "lambdas": [float(x) for x in lam],
                    "grid_step": float(grid_step),
                    "ckpt_dir": ckpt_dir,
                }
            )
    return tasks


def execute_tuning_tasks(tasks: list[dict], jobs: int, resume: bool) -> list[dict]:
    """Run all tuning tasks (parallel, checkpointed, resumable)."""
    unique: dict[tuple[str, int, int], dict] = {}
    for t in tasks:
        unique.setdefault((t["cfg_hash"], t["seed"], t["grid_index"]), t)

    done: dict[tuple[str, int, int], dict] = {}
    pending: list[dict] = []
    for key, t in unique.items():
        ckpt = None
        if resume:
            ckpt = load_tuning_checkpoint(
                tuning_checkpoint_path(
                    t["ckpt_dir"], t["cfg_hash"], t["seed"], t["grid_index"], t["grid_step"]
                )
            )
        if ckpt is not None:
            done[key] = ckpt
        else:
            pending.append(t)

    if pending:
        if jobs <= 1:
            payloads = [run_tuning_task(t) for t in pending]
        else:
            with multiprocessing.Pool(processes=jobs) as pool:
                payloads = pool.map(run_tuning_task, pending)
        for t, payload in zip(pending, payloads):
            done[(t["cfg_hash"], t["seed"], t["grid_index"])] = payload

    # Deterministic order regardless of completion / resume order.
    return sorted(done.values(), key=lambda p: (p["grid_index"], p["seed"]))


def select_lambdas(
    payloads: list[dict], grid: list[tuple[float, float, float]], n_seeds: int
) -> dict:
    """Argmax of per-lattice-point summed score; ties -> lowest grid index.

    Selection depends only on the payload scores (no wall-clock input), and
    the summation order is fixed by (grid_index, seed), so identical inputs
    always give an identical lambda*.
    """
    if n_seeds < 1:
        raise ValueError("need at least one tuning seed")
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    for p in sorted(payloads, key=lambda p: (p["grid_index"], p["seed"])):
        gi = int(p["grid_index"])
        sums[gi] = sums.get(gi, 0.0) + float(p["score"])
        counts[gi] = counts.get(gi, 0) + 1

    complete = [gi for gi in sorted(sums) if counts[gi] == n_seeds]
    if not complete:
        raise RuntimeError("no lattice point was scored on all tuning seeds")

    best = complete[0]
    for gi in complete[1:]:
        if sums[gi] > sums[best]:  # strict: ties keep the lower grid index
            best = gi

    ranked = sorted(complete, key=lambda gi: (-sums[gi], gi))
    top5 = [
        {
            "grid_index": gi,
            "lambdas": [float(x) for x in grid[gi]],
            "score": sums[gi],
            "mean_score": sums[gi] / n_seeds,
        }
        for gi in ranked[:5]
    ]
    return {
        "grid_index": best,
        "lambdas": [float(x) for x in grid[best]],
        "score": sums[best],
        "scores_top5": top5,
        "n_scored_points": len(complete),
    }


def tune_lambdas_parallel(
    mode: str,
    n_tuning_seeds: int,
    grid_step: float,
    ckpt_dir: Path,
    jobs: int,
    resume: bool,
) -> dict:
    """Full parallel tuning pass -> artifact ``tuning`` block."""
    cfg = tuning_config(mode)
    cfg_hash = config_hash(cfg)
    seeds = tuning_seeds_for(cfg, n_tuning_seeds)
    grid = tuning_grid(grid_step)
    tasks = build_tuning_tasks(cfg, cfg_hash, seeds, grid, grid_step, ckpt_dir)

    t0 = time.monotonic()
    payloads = execute_tuning_tasks(tasks, jobs=jobs, resume=resume)
    chosen = select_lambdas(payloads, grid, len(seeds))
    wall = time.monotonic() - t0

    return {
        "source": "tuned",
        "lambdas": chosen["lambdas"],
        "grid_step": float(grid_step),
        "grid_index": chosen["grid_index"],
        "n_grid_points": len(grid),
        "tuning_seeds": [int(s) for s in seeds],
        "tuning_config_hash": cfg_hash,
        "scores_top5": chosen["scores_top5"],
        "wall_clock_seconds": wall,
    }


def assert_tuning_seeds_disjoint(tuning_seeds: list[int], cells: list[dict], n_seeds) -> None:
    """Hard guard: a tuning seed may never appear in any cell's eval seeds."""
    tset = {int(s) for s in tuning_seeds}
    for cell in cells:
        eval_seeds = set(seeds_for(cell["config"], n_seeds))
        overlap = sorted(tset & eval_seeds)
        if overlap:
            raise ValueError(
                f"tuning seeds overlap eval seeds of cell {cell['cell_id']!r}: {overlap}; "
                "FixedWeight lambdas must be selected on dedicated tuning seeds "
                "(SPEC section 7)"
            )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _strategy_names(per_seed_results: list[dict]) -> list[str]:
    """Strategy keys present in every seed's result (dict-valued entries only)."""
    names: list[str] | None = None
    for res in per_seed_results:
        keys = [k for k, v in res.items() if isinstance(v, dict)]
        names = keys if names is None else [k for k in names if k in keys]
    return names or []


def aggregate_cell(cell: dict, payloads: list[dict]) -> dict:
    """Per-cell summary: per-seed results + paired stats (MPO vs each other)."""
    seeds = [p["seed"] for p in payloads]
    per_seed = {str(p["seed"]): p["result"] for p in payloads}
    results_ordered = [p["result"] for p in sorted(payloads, key=lambda p: p["seed"])]

    paired: dict[str, dict] = {}
    strategies = _strategy_names(results_ordered)
    if MPO_STRATEGY_NAME in strategies:
        try:
            from openpo.mpo.stats import paired_stats
        except ImportError as exc:
            warnings.warn(f"openpo.mpo.stats unavailable, skipping paired stats: {exc}")
            paired_stats = None
        if paired_stats is not None:
            for baseline in strategies:
                if baseline == MPO_STRATEGY_NAME:
                    continue
                try:
                    paired[f"{MPO_STRATEGY_NAME}_vs_{baseline}"] = paired_stats(
                        results_ordered, MPO_STRATEGY_NAME, baseline
                    )
                except Exception as exc:  # noqa: BLE001 -- one bad pair must not sink the artifact
                    paired[f"{MPO_STRATEGY_NAME}_vs_{baseline}"] = {"error": str(exc)}
    elif strategies:
        warnings.warn(
            f"no '{MPO_STRATEGY_NAME}' strategy key in results; paired stats skipped"
        )

    return {
        "cell_id": cell["cell_id"],
        "axis": cell["axis"],
        "value": cell["value"],
        "config": encode_config(cell["config"]),
        "config_hash": config_hash(cell["config"]),
        "seeds": seeds,
        "strategies": strategies,
        "per_seed": per_seed,
        "paired": paired,
        "wall_clock_seconds": sum(p.get("elapsed_seconds", 0.0) for p in payloads),
    }


def build_artifact(
    mode: str,
    cells: list[dict],
    cell_summaries: list[dict],
    wall_clock: float,
    tuning: dict | None = None,
    strategies=None,
) -> dict:
    return {
        "schema": SCHEMA,
        "mode": mode,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "env": collect_env(),
        "git_rev": collect_git_rev(),
        "n_cells": len(cells),
        "cells": cell_summaries,
        "tuning": tuning,
        # Provenance: None = the default roster was used (run_seed's own default).
        "strategies": None if strategies is None else [str(s) for s in strategies],
        "wall_clock_seconds": wall_clock,
    }


# ---------------------------------------------------------------------------
# HTML report (best-effort)
# ---------------------------------------------------------------------------


def maybe_render_html(artifact: dict, html_path: Path) -> bool:
    """Render via openpo.mpo.report if available; warn and skip otherwise."""
    try:
        from openpo.mpo.report import render_html
    except ImportError as exc:
        warnings.warn(f"openpo.mpo.report unavailable, skipping HTML report: {exc}")
        return False
    try:
        Path(html_path).parent.mkdir(parents=True, exist_ok=True)
        render_html(artifact, str(html_path))
        return True
    except Exception as exc:  # noqa: BLE001 -- report failure must not fail the run
        warnings.warn(f"render_html failed, skipping HTML report: {exc}")
        return False


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mpo_t1_benchmark",
        description="MPO-T1 synthetic benchmark runner (SPEC section 9).",
    )
    p.add_argument("--mode", choices=("smoke", "main", "sensitivity"), required=True)
    p.add_argument(
        "--seeds",
        type=int,
        default=None,
        help="number of seeds per cell (default: mode-specific; seeds are seed_base+i)",
    )
    p.add_argument("--jobs", type=int, default=1, help="worker processes (1 = inline)")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="artifact JSON path (default benchmark-results/mpo-t1-<mode>.json)",
    )
    p.add_argument(
        "--html-output",
        type=Path,
        default=None,
        help="optional HTML report path (needs openpo.mpo.report)",
    )
    p.add_argument(
        "--axis",
        action="append",
        choices=sorted(SENSITIVITY_AXES),
        default=None,
        help="sensitivity axes (repeatable; default: all)",
    )
    p.add_argument("--resume", action="store_true", help="skip tasks with valid checkpoints")
    p.add_argument(
        "--strategies",
        type=parse_strategies,
        default=None,
        metavar="A,B,C",
        help=(
            "comma-separated strategy roster (validated against "
            "backtest.KNOWN_STRATEGIES; opt-in arms such as mpo_delayaware must be "
            "named here); default = backtest.DEFAULT_STRATEGIES"
        ),
    )
    p.add_argument(
        "--lambdas",
        type=parse_lambdas,
        default=None,
        metavar="A,B,C",
        help=(
            "explicit FixedWeight lambdas (3 non-negative comma-separated values, "
            "normalized to sum 1); skips tuning"
        ),
    )
    p.add_argument(
        "--tune-lambdas",
        type=positive_int,
        default=None,
        metavar="K",
        help=(
            "tune FixedWeight lambdas on K dedicated tuning seeds "
            "(tuning_seed_base+i) before running the cells"
        ),
    )
    p.add_argument(
        "--tune-grid-step",
        type=float,
        default=DEFAULT_TUNE_GRID_STEP,
        help="simplex lattice spacing for --tune-lambdas (default 0.1 -> 66 points)",
    )
    return p


def run_benchmark(args: argparse.Namespace) -> dict:
    t_start = time.monotonic()
    axes = args.axis if args.axis else sorted(SENSITIVITY_AXES)
    cells = build_cells(args.mode, axes)

    output = args.output or (REPO_ROOT / "benchmark-results" / f"mpo-t1-{args.mode}.json")
    output = Path(output)
    ckpt_dir = output.parent / f"{output.stem}-checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- FixedWeight lambdas: explicit / tuned / legacy default -------------
    lambdas_cli = getattr(args, "lambdas", None)
    n_tune = getattr(args, "tune_lambdas", None)
    grid_step = float(getattr(args, "tune_grid_step", DEFAULT_TUNE_GRID_STEP))
    fixed_lambdas: tuple | None = None
    tuning: dict | None = None

    if lambdas_cli is not None:
        if n_tune:
            warnings.warn("--lambdas given, ignoring --tune-lambdas (explicit lambdas win)")
        fixed_lambdas = tuple(float(x) for x in lambdas_cli)
        tuning = {"source": "cli", "lambdas": [float(x) for x in fixed_lambdas]}
    elif n_tune:
        tcfg = tuning_config(args.mode)
        assert_tuning_seeds_disjoint(tuning_seeds_for(tcfg, n_tune), cells, args.seeds)
        tuning = tune_lambdas_parallel(
            args.mode,
            n_tuning_seeds=n_tune,
            grid_step=grid_step,
            ckpt_dir=ckpt_dir,
            jobs=args.jobs,
            resume=args.resume,
        )
        fixed_lambdas = tuple(tuning["lambdas"])
    else:
        # Unchanged legacy path: run_seed applies its IC-proportional default.
        tuning = {"source": "default", "lambdas": ic_proportional_lambdas(cells[0]["config"])}

    # --- strategy roster: explicit / default ---------------------------------
    # A roster equal to DEFAULT_STRATEGIES is normalized back to None so it takes
    # the byte-for-byte legacy path (same run_seed call, same checkpoint names):
    # run_seed(strategies=None) IS run_seed(strategies=DEFAULT_STRATEGIES).
    strategies = getattr(args, "strategies", None)
    if strategies is not None:
        strategies = tuple(str(s) for s in strategies)
        from openpo.mpo.backtest import DEFAULT_STRATEGIES  # deferred: sibling

        if strategies == tuple(DEFAULT_STRATEGIES):
            strategies = None

    # Materialize tasks; identical configs across cells share one checkpoint.
    tasks: list[dict] = []
    for cell in cells:
        cfg = cell["config"]
        cfg_hash = config_hash(cfg)
        cell["_hash"] = cfg_hash
        for seed in seeds_for(cfg, args.seeds):
            tasks.append(
                {
                    "seed": seed,
                    "cfg": cfg,
                    "cfg_hash": cfg_hash,
                    "ckpt_dir": ckpt_dir,
                    "fixed_lambdas": fixed_lambdas,
                    "strategies": strategies,
                }
            )

    payload_by_key = execute_tasks(tasks, jobs=args.jobs, resume=args.resume)

    cell_summaries = []
    for cell in cells:
        seeds = seeds_for(cell["config"], args.seeds)
        payloads = [payload_by_key[(cell["_hash"], s)] for s in seeds]
        cell_summaries.append(aggregate_cell(cell, payloads))

    artifact = build_artifact(
        args.mode,
        cells,
        cell_summaries,
        time.monotonic() - t_start,
        tuning=tuning,
        strategies=strategies,
    )
    atomic_write_json(output, artifact)
    write_sha256_sidecar(output)

    if args.html_output is not None:
        maybe_render_html(artifact, args.html_output)

    artifact["_output_path"] = str(output)
    return artifact


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    artifact = run_benchmark(args)
    n_seeds = sum(len(c["seeds"]) for c in artifact["cells"])
    print(
        f"mpo-t1 {artifact['mode']}: {artifact['n_cells']} cell(s), "
        f"{n_seeds} seed-run(s), wall {artifact['wall_clock_seconds']:.1f}s "
        f"-> {artifact['_output_path']}"
    )
    if artifact.get("strategies"):
        print(f"  strategies = [{', '.join(artifact['strategies'])}] (from --strategies)")
    tuning = artifact.get("tuning") or {}
    if tuning.get("source") == "tuned":
        lam = ", ".join(f"{x:.2f}" for x in tuning["lambdas"])
        print(
            f"  fixed_weight lambdas* = [{lam}] (tuned on "
            f"{len(tuning['tuning_seeds'])} tuning seed(s), "
            f"{tuning['n_grid_points']} grid points, step {tuning['grid_step']}, "
            f"tuning wall {tuning['wall_clock_seconds']:.1f}s)"
        )
    elif tuning.get("source") == "cli":
        lam = ", ".join(f"{x:.4f}" for x in tuning["lambdas"])
        print(f"  fixed_weight lambdas = [{lam}] (from --lambdas)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
