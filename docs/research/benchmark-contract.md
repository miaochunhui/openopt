# Frozen Solver Comparison Contract

`scripts/benchmark_solver_comparison.py` is an independent runner. It does not
change `benchmark_general_qp.py`, submit GPUs, install SDKs, probe licenses, or
read credentials. The coordinator owns GPU scheduling and final acceptance.
This seat's write scope is this contract, the new script and its test file;
existing generated reports/design documents remain untouched.

## Runner CLI / API

```bash
python3 -B scripts/benchmark_solver_comparison.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-comparison/new-run \
  --size 64 --seed 20261010 --repeats 5 --devices cpu cuda:0 \
  --variant baseline --eps-abs 1e-7 --eps-rel 1e-7 --cpu-threads 1
```

`main(argv: list[str] | None = None) -> int` accepts the same arguments.
`--check-result /absolute/path/result.json` performs read-only disk acceptance
and prints measured sample counts, per-arm latency and controls. It verifies
the frozen NPZ/hash/witness, original-coordinate quality recomputed from stored
host x/y for every sample, complete samples, statistics, alternating orders,
same-config controls and their computed noise floor, plus nonempty telemetry.
`run_comparison(args: argparse.Namespace, report: dict) -> None` fills a report;
use `main` to obtain finalization, failure artifacts and an exit code.
`--devices` defaults to `cpu`, accepts `cpu cuda:0` or either device alone.
`--size` and `--repeats` must be at least 3. `--max-iterations` defaults to 5000.
Output must be outside the worktree and new or empty. An existing nonempty
directory fails without altering old evidence; an empty stage directory is OK.

Exit 0 means **measured numerical arms passed**, not all references available.
Exit 1 means execution/configuration/quality failure. Invalid CLI is exit 2.
The cluster fixed qualification stage may invoke this script directly and read
`result.json.verdict`; separately inspect `reference_coverage` and each arm.
No OSQP/Clarabel SDK in a GPU image means two separate `UNVERIFIED` rows, not
failure of available openopt arms. Do not install packages in the Pod.
Explicit unavailable CUDA is `FAIL`, never CPU fallback.

## Same Problem / Quality

The existing synthetic known-KKT generator is reused with explicit size/seed.
`problem.npz` freezes float64 P/q/A/l/u, the primal witness and its dual.
The content hash covers names, shapes, dtypes and all original matrix/bound
bytes. Each successful call verifies the frozen source has not mutated.
No solver consumes another solver's solution or metrics as its oracle.

All solvers receive identical `eps_abs`/`eps_rel` and iteration budgets.
Openopt/OSQP use both as named. Clarabel uses `tol_gap_abs=eps_abs`,
`tol_gap_rel=eps_rel`, `tol_feas=eps_abs`; reduced acceptance tolerances are
also matched, and only `Solved` (not almost-solved) is accepted.
Different termination formulas are not assumed equivalent: **all** arms must
pass the same independent original-coordinate test. Polishing/warm-starting
are disabled for OSQP; other backend defaults are recorded.
OSQP adaptation remains enabled within a call, with an explicit 50-iteration
interval rather than a wall-time-derived schedule. Before every solve its
initial `rho=0.1` is restored inside timing: disabling iterate warm starts alone
does not reset adapted rho. Any reset/refactor overhead is included in prepared
latency, not hidden in untimed preparation. This prevents a later prepared
repeat from receiving a better penalty parameter learned by its predecessor.

The checker recomputes constraint projection, stationarity P*x+q+A.T*y,
normal-cone residual, objective and relative witness-objective gap on CPU.
Primal/normal tolerance is `eps_abs + eps_rel*max(||Ax||inf, ||proj(Ax)||inf)`;
dual tolerance uses max norms of Px/q/A.T*y. Objective gap tolerance is
`eps_abs+eps_rel`. Nonfinite/wrong-size solutions fail. Clarabel conic duals
are mapped back with equality/upper/lower row signs before this check.

Full eigenvalue PSD validation remains in every model-cold setup and the
one-time prepared setup: openopt uses its unmodified full eigenvalue validation;
reference adapters independently validate full P before sparse/conic conversion.
A positive diagonal is not substituted for PSD. The initial witness validation
does not exempt subsequent cold calls. No cheaper PSD/adaptive setting is
silently selected. `--variant adaptive` requires a declared
`QPConfig.adaptive_rho` field; missing field is FAIL. If that field exists,
baseline explicitly sets false and adaptive true. No future API is mandatory
for baseline.

## Timing / Controls

Every measured arm runs sequentially on **one host, one process, one clock**.
Repeat 0 runs forward, repeat 1 reverse, then alternates. The two modes are
separate; repetitions are latency pairs, not independent throughput tasks.
There are no discarded numerical warmups. Module imports/runtime/device
initialization are excluded for all arms; this is **model-cold**, not
process-cold. First-use overhead not removed by initialization remains visible.

Cold `wall_s` includes full PSD validation, model copies, sparse/conic conversion,
setup/factorization, solve and host x/y extraction. `setup_s` and
`solve_host_result_s` are non-overlapping subdivisions of that wall.
CUDA synchronization brackets timing and separates setup from solve.
Prepared `setup_once_s` is separate and never mixed with cold latency.
Prepared calls reuse model/workspace, start numerical iterates afresh (openopt
documented zero start, OSQP warm_starting false, Clarabel fresh solve), and time
solve plus host outputs. Independent quality and file I/O are outside latency.

Each available backend (openopt per device/variant, OSQP and Clarabel) has a separately constructed identical-config
control arm, equal config hashes, its own workspace, and matched repeat pairs.
The p95 absolute paired control delta is `empirical_noise_floor_s`, not a
hardcoded noise floor. Failed/incomplete controls have null floor and
`UNVERIFIED`, never zero. Raw p50/p95 retain every successful sample; a failed
arm cannot be promoted by its successful subset or used in a ranking.
No automatic winner is emitted; reported latencies are scoped working evidence.

CPU limits are explicit. CLI sets OMP/OpenBLAS/MKL limits before imports;
Torch limits and loaded OpenBLAS getter/setter measurements are also recorded
for API callers. OSQP uses its installed backend; Clarabel `max_threads` is
explicit. Proc thread counts, process CPU seconds, affinity, cgroup quota and
before/after CPU counters are retained per sample, with runtime telemetry and
peak RSS. Pool getter unavailability is recorded, not assumed single-threaded.
Cross-host CPU reference results must not be merged into a GPU timing ratio.
GPU allocated/reserved bytes are sampled; GPU utilization/profiler evidence
remains the coordinator's separate requirement, explicitly UNVERIFIED here.

## Failures / Artifacts

`result.json` has the terminal verdict/exit code, every arm/mode/sample,
configurations, orders, quality and controls. `partial.json` preserves progress.
`failure.json` retains first cause, exception type, traceback and arm/mode/repeat
coordinates; the first failed arm is not replaced by the outer summary error.
After a failure that arm is not retried and its other mode is stopped; other
arms can finish independently to preserve comparison evidence.
`execution.log` includes a dated terminal marker; `telemetry.json` and
`problem.npz` are real external outputs.

Gurobi/CPLEX/MOSEK discovery only checks whether modules exist. SDK missing,
SDK present but matched adapter not executed, and license not verified remain
`UNVERIFIED`. No license call is made, so credentials/endpoints cannot leak.
Commercial qualification and universal/GPU ranking remain UNVERIFIED even
when the numerical run exits 0.

Negative controls in `tests/test_solver_comparison_tools.py` cover corrupted
primal/dual/nonfinite/size outputs, positive-diagonal indefinite P, missing
adaptive field, missing reference SDKs, unavailable CUDA, first-cause failure
isolation, alternating controls and preservation of existing artifacts.

CPU acceptance uses the existing environment, without installation:

```bash
sudo -n -u infra /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/verify.py tests --target tests/test_solver_comparison_tools.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-benchmarks/tests
sudo -n -u infra /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/benchmark_solver_comparison.py --devices cpu --size 64 \
  --seed 20261010 --repeats 3 --variant baseline --cpu-threads 1 \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-benchmarks/cpu-baseline
```

Acceptance reads actual test counts plus every cold/prepared quality verdict,
OSQP/Clarabel coverage, frozen hash and control pairs from the artifacts.
Use a new directory for later revisions, then run `--check-result` on that
revision's `result.json`. Source hashes identify the exact runner/generator/core.
GPU and broader/ill-conditioned/commercial qualification are not covered by
this CPU smoke.
