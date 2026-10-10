# Synthetic QP/LP Stress Corpus

This is a reproducible **synthetic** dense stress corpus, not a representative
public or industrial problem collection. No public dataset was imported.
Sparse, conic, MIP, commercial parity and production coverage remain unverified.
It supplements, not replaces, the existing dense experiment/report contract;
no solver algorithm, defaults or generated report was changed.

## Cases And Independent Truth

All problems minimize `0.5*x'P*x + q'x` under `l <= A*x <= u`, with float64
CPU arrays frozen identically for openopt, OSQP and Clarabel.

| Family | Structure | Oracle scope |
| --- | --- | --- |
| `original_dense` | Original seeded dense SPD with active bounds/equalities | Byte-identical original generator; floating planted KKT, roundoff rather than exact certification |
| `dense_spd` | Dense dyadic `I+B'B`, mixed active bounds | Algebraically exact dyadic primal/dual convex KKT witness |
| `illconditioned` | One unit diagonal, other diagonals `2^-25` to `2^-30`, dense rank-one term | Exact dyadic KKT; no inaccurate-solution-distance gate |
| `psd_singular` | SPD block plus exact zero nullspace | Exact KKT; multiple minimizers intentionally permitted |
| `equality_redundant` | Duplicate/scaled equalities and zero row, active inequalities | Exact KKT with rank-deficient constraint rows |
| `lp` | Zero P, redundant box rows, positive/negative/zero costs | Exact primal/dual optimum; nonunique zero-cost coordinate |
| `infeasible` | Same row must be <=0 and >=1; valid one-sided bounds | Independent exact Farkas witness `A'y=0`, box support `-1` |
| `unbounded` | LP or singular QP, cost-decreasing free coordinate | Feasible base and exact ray `Pd=Ad=0`, `q'd=-1` |

Seed parity varies the infeasible/unbounded LP versus QP variant. Dimensions
and seeds are explicit; default qualification uses n=8/24/64, seed=20261010.
The oracle exists before any solver is called. Generation validates convexity
and the witness in independent NumPy; tests additionally evaluate the dyadic
KKT equations in exact rational arithmetic across n=3/7/16. This is not a
claim of formal exact certification for returned numerical solver solutions.

## Qualification Contract

`src/openopt/corpus.py` never calls a solver. It hashes shape-labelled,
ordered little-endian float64 P/q/A/l/u bytes. Metadata saves problem hash,
generator version, seed, class, expected property and exact oracle scope.
Each selected case saves `metadata.json` and `problem.npz` (including witness).
Every backend saves raw vectors and a JSON row; the aggregate is `result.json`.
The script rereads frozen bytes and checks matched input hashes after solves.
Source hashes bind the generator, baseline core and qualification adapter.
`missing_rows` lists every selected backend/case not completed after an error.

The original dense solver uses its unchanged Cholesky path. OSQP receives the
same bounds; Clarabel converts finite upper/lower bounds and signed duals,
including equalities without duplicate inequality cones. References are CPU;
an explicit `--device cuda:N` applies only to openopt and never falls back.
Missing libraries, invalid inputs, exceptions or empty selections are errors,
not skipped qualification. CUDA allocation/launch is the parent's task.

Optimal candidates require an exact accepted solver status plus independent
absolute primal, stationarity and normal-cone residuals <=5e-6 and relative
objective gap <=1e-6. The objective is evaluated against the preconstructed
witness, never against the current solver or a reference consensus. Distance
to the particular witness is informational, not a gate for nonunique or
ill-conditioned solutions. Inaccurate solver statuses are not silently accepted.
For reference infeasible/unbounded statuses, the returned certificate is also
checked independently against the input. No solver-reported residual is trusted.

One cold diagnostic solve includes setup, validation, copies and host results.
Elapsed time is evidence, not a statistically qualified latency/speed ranking.
No automatic retries, hidden parameter changes, polishing, GPU scheduling or
commercial qualification occurs.

Exit codes: **0** all selected rows qualified; **1** at least one unqualified
status/quality (full diagnostic rows retained); **2** execution/input error
(first cause and case/solver coordinates in `failure.log`, remaining rows not
claimed complete). The output directory must be new, exclusive and on 3FS tmp.
`execution.log` records progress and the terminal verdict; old evidence is never
overwritten. Unqualified rows remain in the aggregate failure list, not xfails.

## Reproduction And Sharding

Run from the corpus worktree with the already-provisioned CPU environment:

```bash
sudo -n -u infra /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/qualify_qp_corpus.py --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/qualification-v1
sudo -n -u infra /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_qp_corpus.py --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/tests-v1
```

The parent can shard using an exact canonical case ID or a family, choosing a
new artifact directory per shard. `--case` embeds n/seed and must not be mixed
with `--size/--seed`; families support repeatable `--size` and `--seed`.
`--solver openopt` runs only that explicitly selected backend; omitting it runs
all three. Compare hashes, not filenames, between CPU and parent CUDA shards.

```bash
python -B scripts/qualify_qp_corpus.py --case illconditioned-n64-seed20261010 --device cuda:0 --solver openopt --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/cuda-ill64-v1
python -B scripts/qualify_qp_corpus.py --family lp --size 24 --seed 83 --device cpu --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/lp24-seed83-v1
```

## Known Qualification Gaps

The baseline core has no infeasible/unbounded certificate. Its `max_iterations`
on these cases is an honest **UNQUALIFIED capability**, not proof of a solver
bug, nor a successful expected-property qualification. The independent oracle
proves the problem property; it does not turn the solver's limit into a
certificate. Numerical errors or exceptions remain execution errors.
Ill-conditioned feasible cases may also hit iteration limits; these must remain
visible with their actual independent residuals/objective, never silently caught
or labelled solved. CPU run outcomes and exact unresolved cases are recorded
below after qualification; no CUDA run is performed by this corpus seat.

Tests cover three dimensions, two seeds, eight families, rational witnesses,
original byte identity, nullspace minimizers, primal/dual/normal/objective decoys,
Farkas/ray corruption, invalid/empty input, false statuses, preserved exceptions,
exclusive evidence and strict unavailable-CUDA rejection.

Initial test evidence is retained at
`/hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/tests-v1/tests-j3054y6n/`.
It exposed and fixed n=3 LP seed collapse and an insufficiently ill-conditioned
n=3 spectrum. It also exposed a real reference qualification gap:
`infeasible-n8-seed83`, OSQP returned `primal infeasible`, but independent
normalized certificate annihilation residual was `2.4906024187254516e-5`,
above the `5e-6` gate. This is not a generator failure or evidence of exact
infeasibility certification by OSQP. The gate stays unchanged. The test now
checks that the actual independent certificate verdict is propagated, including
UNQUALIFIED, with a separate corrupted-certificate decoy proving rejection.

### CPU Qualification 2026-10-10

The first full run completed at `2026-10-10T07:52:05.804893+00:00`, with
NumPy 2.2.6, PyTorch 2.8.0+cu128 (CPU execution), OSQP 1.0.4, Clarabel 0.11.1
and SciPy 1.15.3. Evidence:
`/hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/qualification-v1/result.json`.
All 24 cases / 72 backend rows completed: **64 QUALIFIED, 8 UNQUALIFIED,
exit 1**. All original dense and feasible synthetic rows passed; this is
qualification only under the stated residual/objective/status gates.

| Unresolved case IDs (all seed20261010) | Backend | Observed failure |
| --- | --- | --- |
| `infeasible-n8`, `infeasible-n24`, `infeasible-n64` | openopt | `max_iterations` at 5000; missing certificate capability, not a solver bug |
| `unbounded-n8`, `unbounded-n24`, `unbounded-n64` | openopt | `max_iterations` at 5000; missing certificate capability, not a solver bug |
| `infeasible-n24` | OSQP | `primal infeasible`; normalized `A'y` residual `3.795037316112193e-5` >5e-6 |
| `infeasible-n64` | OSQP | `primal infeasible`; normalized `A'y` residual `5.277209940357519e-5` >5e-6 |

Owners: the parent/core track must implement and independently qualify the
openopt certificates; the reference qualification track must investigate OSQP
certificate accuracy in a separately authorized changed-config experiment.
This seat does not change either solver or retry those failures. Exact case IDs
append `-seed20261010`; use `--case` with a fresh directory for diagnosis.

Important numerical boundary: `illconditioned-n64-seed20261010` openopt had
primal/dual/normal residuals `6.69e-8 / 6.89e-8 / 3.58e-8` and relative objective
gap `1.25e-8`, while witness distance was `0.663227`. A small KKT/objective error
does **not** certify coordinate recovery on this ill-conditioned model. Do not
describe this row as an exact recovered solution or universal robustness proof.

Corpus tests v2 passed 113 tests; the original postflight smoke passed 83 tests.
Both retain XML, counts and logs under `tests-v2/tests-s5aoycdc` and
`postflight-v1/smoke-6myw3osf` in the same task root. Subsequent acceptance
adds exact-rational property witnesses, matched saved-vector checks and explicit
missing-row evidence without changing the generated problem bytes or gates.

### OSQP Certificate First Cause

Retained-failure diagnosis:
`/hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/osqp-certificate-diagnosis-v1/`.
It rereads each original problem and raw `primal_certificate`, verifies its hash,
and independently recomputes each column dot product with Python scalar sums.
These reproduce the original residuals; no certificate row/sign/scale remapping
occurs in the adapter. Support is near -1 with the correct direction. The
original failure is therefore a genuinely approximate returned direction,
**not a bound/sign conversion bug or a strict-1e-8 evaluation accident**.

There are three separate thresholds:

| Threshold | Original run value | Meaning |
| --- | --- | --- |
| OSQP `eps_abs`, `eps_rel` | 1e-7 | Optimal solution stopping criteria, not the infeasibility-certificate tolerance |
| OSQP `eps_prim_inf`, `eps_dual_inf` | Default 1e-4 | Separate infeasible/unbounded stopping thresholds, omitted by the baseline adapter |
| Independent qualification certificate gate | 5e-6 | Explicit caller argument to `check_infeasibility`; the helper's default 1e-8 was not used |

The installed OSQP 1.0.4 API/constants (`osqp_api_constants.h:161`,
`osqp_api.c:318`) set `eps_prim_inf=1e-4`. Its `auxil.c:460` separately checks
`||A'delta_y||inf < eps_prim_inf * ||delta_y||inf` and negative box support.
The baseline setup was reconstructed without repeating the failed solve;
effective settings confirm this default. Both original residuals are below
1e-4 but above 5e-6. This is a reference **stopping-tolerance/qualification
contract mismatch**; leaving certificate tolerances implicit was a metadata
and configuration clarity gap, not evidence that OSQP violated its own setting.
Future solver rows now persist effective OSQP settings.

One explicitly requested diagnostic changes **only** `eps_prim_inf` to 1e-7;
all optimal tolerances, iteration limit, scaling and matrix bytes remain fixed:

| Case | Original `A'y` residual | Changed `eps_prim_inf=1e-7` residual | Same 5e-6 gate | Additional strict 1e-8 diagnostic |
| --- | --- | --- | --- | --- |
| `infeasible-n24-seed20261010` | 3.795037316112193e-5 | 6.770505822650819e-9 | PASS | PASS |
| `infeasible-n64-seed20261010` | 5.277209940357519e-5 | 9.762857189343777e-12 | PASS | PASS |

The one-variable result supports the stopping-tolerance first cause. The new
certificates are not substituted into `qualification-v1`, whose OSQP result
remains 22/24. No independent gate or qualification default was weakened; no
baseline solver setting was changed. Diagnosis exit 0 means **DIAGNOSED**, not
corpus qualification PASS (its diagnostic cases are nonempty; solver rows are
intentionally not qualification rows). Empty diagnostic failure selection fails.

```bash
sudo -n -u infra /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/qualify_qp_corpus.py --diagnose-osqp /hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/qualification-v1 --diagnostic-eps-prim-inf 1e-7 --artifact-dir /hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/osqp-certificate-diagnosis-v2
```

This command creates new changed-parameter evidence, not a blind baseline retry.
Baseline code/core and all four-file scope remain unchanged outside this seat.
Latest corpus acceptance before this diagnosis-test addition: **120 passed**,
`tests-v3/tests-vyl7udfs`; original smoke **83 passed**.

Final four-file acceptance: **121 passed**, no failures/errors/skips in
`/hf3fs-jd/data/tmp/openopt-qp3-corpus-20261010/final-tests-v1/tests-vdn24fog/`.
The complete repository suite and actual CUDA execution were not run by this
seat. Baseline `src/openopt/qp.py` and the original benchmark are unchanged.
