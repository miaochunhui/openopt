# Independent QP Campaign Acceptance

This seat owns only `scripts/accept_qp_campaign.py`,
`tests/test_qp_campaign_acceptance.py` and this document. It does not launch
GPUs, install dependencies, invoke nested/Flash agents, modify infra or read
credentials. The main owner integrates code and generates final reports.

## Real Evidence Entry

The gate directly consumes `gpu_qp_campaign.py` result/run/source snapshots
and `benchmark_solver_comparison.py` result/problem/telemetry artifacts.
No producer module from an evidence directory is imported or executed.
Use the existing CPU environment; every CLI run creates a new external 3FS
directory and preserves failures with first cause, coordinate, traceback and
exit code.

```bash
sudo -n -u infra /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/accept_qp_campaign.py seal \
  --result /absolute/external/run/case-0/result.json \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-acceptance/gpu-seal
```

GPU source/run locations default to `case-0/../source` and `case-0/../run.json`.
An explicit `--run` / `--source-root` overrides these locations. A comparison
requires `--source-root` pointing to its exact producer working tree/snapshot.
Repeat `--result` and corresponding `--source-root` in the same order for a
multi-result bundle. Empty inputs, duplicate results/JSON keys, nonfinite JSON,
missing/empty artifacts and unsupported schema fail.

Sealing checks the producer source hashes and freezes the complete public
source set plus result and required sidecar bytes. **SEALED is not PASS.**
The owner pins the emitted bundle hash separately; acceptance requires it:

```bash
sudo -n -u infra /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/accept_qp_campaign.py --check \
  --bundle /absolute/external/seal-unique/bundle.json \
  --bundle-sha256 OWNER_PINNED_64_HEX_SHA256 \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-acceptance/recheck
```

Result/NPZ/source/manifest/profile/log changes or additions to the sealed
source set fail. A caller who can replace both evidence and the trusted pin
can forge provenance; this is integrity protection, not hostile-party
authentication or a production sandbox.

## Criteria And Ranking

Comparison: require strict backend solved status, nonempty complete repeats,
positive finite wall and recomputed p50/p95, separate cold/prepared accounting,
observed device placement, matched dtype/eps and requested/observed Torch/BLAS
threads. Recompute every host x/y's original-coordinate constraint,
stationarity, signed normal-cone and objective metrics against the frozen
PSD problem and planted KKT witness. Recompute alternating orders, control
configuration equality and measured paired p95 noise floors independently.
Never accept a failed solver from a successful subset of samples.

GPU: consume the exact `result.measurements.cold|prepared` three-arm schema,
including `batch_size`, `validation_parity`, `problem_hash`, latency and quality.
Require every quality record to retain `solver_status="solved"` (or legacy
`status="solved"`), finite residuals below the existing independent GPU ruler,
nonempty telemetry and profile evidence. Correctness mode requires dense
adaptive on/off, all three batch items, COO/CSR, their statuses and quality,
nonzero JUnit counts with zero skips/errors/failures and a completed pytest
summary. A producer's `PASS`, process existence or exit zero is insufficient.

The revised GPU performance schema retains `problem.npz`, `host_x/host_y`,
`result.dtype/eps_abs/eps_rel` and three-arm `arm_config`. The gate recomputes
every saved solution and checks same-problem/dtype plus allowed adaptive/CG
configuration differences. Correctness requires `problem-dense.npz` and
`problem-sparse.npz` (or explicit local basenames in `result.problem_files`
with keys `dense`/`sparse`) and every check's host x/y, including batch items.
Missing replay evidence is FAIL, never an admission PASS. Missing dtype/eps
is not defaulted. Sparse caller PSD assumptions do not establish dense
validation parity. Shared dtype/eps are the producer schema's common settings;
any explicit per-arm overrides must match.

Only within-result, same-host/problem/thread/dtype/eps/mode sequential
comparison pairs with both identical controls are eligible for a scoped
latency difference. Differences within the measured control noise remain
INCONCLUSIVE, not wins. Cold and prepared never share a ranking. Distinct
results/hosts never form speedup ratios. Commercial discovery or a declared
PASS is not matched solver evidence: Gurobi/CPLEX/MOSEK qualification and
commercial ranking remain UNVERIFIED, `commercial_win=false`.
GPU baseline/candidate uses its measured baseline/control paired noise floor;
comparison backends require both independent identical-control noise floors.
`--check` is the admission-friendly alias of action `check`; exit 1 leaves a
first-cause artifact and must block admission. Main owner owns wiring this
entry into cluster submission. No live cluster operation is performed here.

## Audit Findings

Initial read on 2026-10-10 found these owner-owned boundaries:

- GPU performance and batch correctness quality records discarded solver
  status. The gate rejects the resulting missing-status evidence. Persist
  each actual result status without inventing it from the quality verdict.
- GPU results omitted dtype/eps and host solutions/witness; the gate cannot
  independently replay them or establish per-arm ranking parity.
- Cluster campaign admission relied on a declared PASS and partial core
  hashes, without full source/result integrity or test artifact acceptance.
  This offline gate is not automatically installed in cluster admission.
- Comparison `check_result()` recomputed quality but did not check saved
  solver status, source integrity or actual thread limits. This independent
  gate checks them; mutations below demonstrate rejection.
- GPU cold documentation claimed input assembly, but `generate()` and NumPy
  extraction occur outside timed `prepare_arm()`. Dense batch stacking and
  sparse conversion are timed; original input generation is not. Interpret
  this as model-cold, not process/input-generation-inclusive timing.

These findings are not silently patched outside this seat's write scope.
The main owner must close them with changed source and new evidence before
claiming admission/report integration complete.

At 08:01 UTC the owner revised GPU performance to retain NPZ, host x/y and
common dtype/eps/arm configs, and clarified excluded input generation. The
checker now consumes those fields directly. Original missing-status GPU
`campaign-u4darfl7` is independently rejected; no old artifact is retroactively
patched or accepted. The owner then emitted both problem sidecars and all
batch host vectors in `campaign-_rge0l3h`; that new correctness result passed
independent replay at 08:06:37 UTC. The old incomplete runs remain rejected.

## CPU Verification

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/verify.py tests --target tests/test_qp_campaign_acceptance.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-acceptance/tests
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/iteration_gate.py smoke \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-acceptance/postflight
```

CPU tests exercise the producer schemas with planted KKT solutions, not
fabricated benchmark claims. Resealed mutants cover solver status, primal,
dual, saved quality, zero/empty timing, eps, dtype, threads, problem hash,
cold/prepared mixing, order, missing GPU status, residual/count/profile and
incomplete pytest logs. Pinned-byte mutants change source, result, problem,
bundle or source membership. First-cause CLI artifacts are checked on disk.
Separate real comparison/GPU acceptance runs are required and must be cited
from their actual completed evidence, never from this document's examples.

For independently reproducible mutants on an accepted **real comparison**
bundle, use the same caller-pinned hash with action `mutants`:

```bash
sudo -n -u infra /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/accept_qp_campaign.py mutants \
  --bundle /absolute/external/seal-unique/bundle.json \
  --bundle-sha256 OWNER_PINNED_64_HEX_SHA256 \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-20261010-acceptance/mutants
```

This first reaccepts the original, then saves solver-status, primal, dual,
zero-timing and forged-sourcehash variants plus their independent rejection
coordinates. It does not change the original source/result/sidecars, rerun a
solver or generate new performance evidence.

## Observed Evidence

2026-10-10 live disk verification (not producer self-report):

- CPU targeted `tests-v3/tests-hysc6hps/verification.json`: 45 tests, zero
  failures/errors/skips; postflight `smoke-29wwpsmo`: 83 executed passes,
  four explicit CUDA skips. Both under the acceptance artifact root below.
- Final parameter-parity revision
  `tests-config-final/tests-ljg38own/verification.json`: 47 tests, zero
  failures/errors/skips, completed at 08:06:15 UTC.
- Exact benchmarks-seat source root
  `/data/dev/openopt-wt-qp3-20261010-benchmarks` accepted the real
  `openopt-qp3-20261010-benchmarks/cpu-baseline-final/result.json`:
  36 host solutions independently recomputed. Do not substitute the changing
  main source tree for this result's producer root.
- Real comparison bundle `real-comparison/seal-ir4gfph0/bundle.json`, hash
  `412738e4fae65f021440cc4edd57d3f68e874f06229fb40540f02333a0bc8836`;
  `real-mutants/mutants-3t_b2cgy/acceptance.json` records all five semantic
  mutants rejected, with coordinates and original first causes.
- Actual GPU batch performance `campaign-8evqrvdb/case-{0,1,2}/result.json`
  accepted at 08:03:26 UTC: 720 per-problem solutions independently replayed
  across three results. Bundle `real-performance/seal-cu4xep7q/bundle.json`,
  hash `f51e739b7ff0b938684ad7a06f1ed84a0b0986f07eebe92e5e2d6088e5ff3d30`;
  accepted record `real-performance-check/check-hs96zbnr/acceptance.json`.
- GPU correctness `campaign-y4_wnv1x` remains strict-replay FAIL:
  `problem-dense.npz` missing at 08:03:07 UTC; batch host vectors also absent.
  First cause retained in `real-gpu-v3/seal-fmb7cvtb/acceptance.json`.
  Older `campaign-u4darfl7` was rejected at `correctness/batch_0` for missing
  solver status in `real-gpu-check/check-0rza31s1/acceptance.json`.
- New correctness `campaign-_rge0l3h`, operators `campaign-p5cr56h9` and
  adaptive `campaign-szfx788c` passed strict replay at 08:06:37 UTC: seven
  correctness solutions plus 165 JUnit tests with zero errors/failures/skips,
  and 30 solutions per performance path. Operators ranking stays UNVERIFIED
  because caller-PSD and dense spectral validation are not equivalent.
  Bundle `real-newpaths/seal-7h_owss3/bundle.json`, hash
  `d59822abf36c7ecc14607183e9ea7f2589e920713ce2bd35293a8db149764488`;
  accepted record `real-newpaths-check/check-c3oxakps/acceptance.json`.

All abbreviated acceptance paths above are under
`/hf3fs-jd/data/tmp/openopt-qp3-20261010-acceptance/`. GPU campaign inputs
are under `/3fs/research3fs/data/tmp/openopt-campaign-gpu/`. No old failed
evidence has been repaired in place. Correctness sidecar/replay gaps are
closed for the new run, not for historical incomplete results. Main owner
still owns admission wiring, final integration/report and qualified commercial
runs. CPU tests are not GPU execution evidence.
