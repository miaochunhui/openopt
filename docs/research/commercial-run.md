# Authorized Commercial LP/QP Runner

## Conclusion And Boundary

`scripts/benchmark_commercial.py` supplies CPU-only Gurobi, CPLEX and MOSEK
adapters for the same original dense continuous problem:
`min 0.5*x.T*P*x + q.T*x`, subject to `l <= A*x <= u`.
This is an executable qualification entry, not evidence of commercial parity,
GPU acceleration, conic/MIP support or a universal ranking.

Only these files belong to the commercial seat: the runner,
`tests/test_benchmark_commercial.py`, and this document. Existing algorithms,
design pages, generated reports, infra and other worktrees remain untouched;
the main agent owns report/design integration before any release claim.

Development authorization does **not** authorize buying, installing or using a
commercial license. Default discovery invokes only top-level
`importlib.util.find_spec` and distribution version metadata. It does not import
SDKs, inspect credentials/files/environment variables or contact license servers.
Installed SDK metadata never establishes license availability or permission.
Unexpected discovery exceptions produce rc=1 with redacted first-cause evidence;
missing SDKs produce `UNVERIFIED`, not `PASS`.

Actual SDK import and model/license use require **both** `--authorized-run` and
`--authorization-sha256`, the SHA256 of an externally approved, non-secret
authorization record. The record must cover the named solver, this host, usage,
license checkout/network access where applicable, and the authorized period.
The runner records only its digest; it does not read the record, inspect a
license or determine legal validity. The operator must establish permission
before supplying the gate. A fabricated digest or test fixture is not approval.
The report always keeps `legal_validity=UNVERIFIED` and
`record_is_legal_authorization=false`, including explicitly gated executions;
the digest is an index, never a legal authorization token.
Without a real approved record, do not invoke the execution mode.

SDKs and their installations retain their own terms. Nothing is installed,
downloaded from a license server, purchased or vendored by this runner. Native
stdout/stderr (including license diagnostics) are discarded during import,
model creation, solve and cleanup. Reports preserve stage, solver/mode/repeat,
exception chain classes and numeric errno/error_code, never raw vendor exception
text, license values/paths, credential environment values or server endpoints.
The runner is not a security sandbox against malicious Python import finders,
SDKs or operators; it requires a trusted process and sequential use.

## Frozen Inputs And Evidence

`--problem /absolute/public-problem.npz` loads only float64 P/q/A/l/u using
`allow_pickle=False`. Optional `expected`/`witness_y` must occur together and
pass independent original-coordinate checks. Source paths and arbitrary NPZ
members are not copied into reports. Supply only public or explicitly permitted
inputs; do not move proprietary data from another project into this worktree.
Without `--problem`, `--kind lp|qp --size N --seed S` generates a small bounded
synthetic known-KKT case. Its witness is never passed to an adapter.

Each new external run directory retains `problem.npz`, byte and canonical
array-content SHA256, dimensions/dtype/type, source Git revision and source
file hashes (all `src/openopt/**/*.py` modules plus this runner), NumPy/Python
versions, public requested/effective parameters and
their hashes, CPU model/architecture/count/affinity/quota, and per-attempt host
x/y. No hostname or credential environment is collected. Every adapter gets an
independent original-model copy; hashes reject input mutation. Full independent
PSD/shape/bound validation is included in every cold model setup and in the
one-time prepared setup. Empty, nonfinite, malformed and indefinite inputs fail.

Gurobi and CPLEX expand each finite lower/upper row, retain equality once, and
omit fully free rows. Their minimization row duals map to original coordinates
with `y[row] -= native_dual`; MOSEK keeps ranged rows and maps `y=-gety()`.
Gurobi receives the full quadratic expression with the explicit half factor;
CPLEX receives the full P matrix in its half-quadratic API; MOSEK receives the
lower triangle of P in its half-quadratic API. Variables are explicitly free.
No portfolio-specific proprietary adapter or data is copied.

Strict native optimal status is necessary but insufficient. For every sample,
`openopt.certificates.check_qp_candidate` independently recomputes original
per-bound feasibility, stationarity `P*x+q+A.T*y`, multiplier signs,
complementarity, dual objective/range and signed primal-dual gap on CPU.
Nonfinite/wrong-size x/y, false status and mutated models fail. Singular LP/QP
range restrictions are checked, not repaired with a regularized inverse.
These are approximate numerical certificates, not exact mathematical proofs.
Infeasible/unbounded native statuses currently retain failed attempts; this
runner does not extract rays or qualify those categories.

## Timing And Terminal States

Each solver is run sequentially for `--repeats >=3` in separate cold/prepared
modes, with no dropped warmups. All arms use one process and clock. This is
model-cold, not process-cold: module imports/runtime initialization are excluded
and not represented as a commercial speedup. No paired winner or cross-host
ratio is emitted; fixed order, one synthetic case and three repeats do not
establish representative performance, uncertainty or noise controls.

Cold `wall_s` includes full model validation/copying/conversion, model/license
setup, solve and host x/y serialization. `setup_s` and `solve_host_result_s`
are non-overlapping subdivisions. Prepared `setup_once_s` is separate; each
prepared `wall_s` includes reset/solve/host results. Gurobi uses `reset(1)`,
CPLEX uses `advance=0`, and MOSEK uses a constant interior-point start and no
basis construction. This is model reuse, not changing-data iterate warm start.
Gurobi/CPLEX use barrier and explicit CPU thread limits; CPLEX requests a
nonbasic solution. MOSEK freezes both LP and QP termination settings and
sets near-optimal relaxation factors to one. Different native formulas are
not presumed equivalent: the common independent certificate decides success.
Requested/effective public numerical settings must agree. CPU process seconds
and wall seconds are retained; certificate evaluation and disk I/O are outside
the measured interval. GPU operations are not part of this runner.

Every arm/mode/sample is retained in `result.json`; `partial.json` preserves
progress. The first failing attempt stops that arm without retry or switching
backends; other independent solvers may complete. `failure.json` preserves the
first cause and all cleanup/secondary failures. A native license/import/setup
failure keeps `licensed_run=UNVERIFIED`, while collection fails with rc=1;
an executed solution that fails the certificate has arm `FAIL` and qualification
`UNVERIFIED`. No incomplete mode publishes qualified latency statistics.

All selected arms need complete, independently accepted cold/prepared samples
for qualification `PASS`. Missing SDK/approval/execution remains `UNVERIFIED`.
rc=0 for discovery means **collection completed**, never that a commercial
benchmark passed. CLI/precondition/replay errors return 2; execution/discovery
errors return 1. `--check-result` performs read-only frozen-input, source-hash,
parameter, host-certificate, count, authorization and latency replay; its own
acceptance can be `PASS` while commercial qualification stays `UNVERIFIED`.
Replay detects mutation but does not authenticate a fabricated report or an
approval digest. It checks the recorded source bytes against the current tree;
replay on a later runner revision is intentionally rejected.

## Commands And Local Acceptance

Run from this commercial worktree in the existing environment, without install:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/benchmark_commercial.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/commercial/discovery --kind qp
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_benchmark_commercial.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/commercial/unit
```

Only after real approval, the execution command adds `--authorized-run
--authorization-sha256 <approved-record-sha256>` and optionally `--problem`.
The placeholder is not a valid digest and must not be treated as approval.
For replay use `--check-result /absolute/run-directory/result.json`, without
`--authorized-run`. New runs never overwrite prior run directories or unknown
external files. In-tree and `/data/dev`/`/data/wdev` artifact roots are refused.

Actual discovery at **2026-10-10T08:47:09 UTC**, using CPU-only validation env:
Gurobi/gurobipy, CPLEX/cplex and MOSEK/mosek were all absent, metadata versions
were null, planned=18, attempted=0, solve_attempted=0, qualification
**UNVERIFIED**. CPU: AMD EPYC 7702 64-Core Processor, x86_64, 256 logical CPUs.
Evidence under the task root:
`actual-v1/discovery-_injqbzn/result.json`, `problem.npz`, `execution.log`.
This preserves the source hash of that revision; final code requires fresh
discovery/replay, not rewriting that artifact.

CPU preflight: `preflight/smoke-b9qs8lif/pytest.xml`, 79 passed / 4 explicit CUDA
skips, plus provenance and CPU example. First unit attempts remain preserved:
`unit-v1/tests-bz4thr17` (45 passed / 1 failed) and `unit-v2/tests-nypcwskh`
(45 passed / 7 failed), caused by mutation/enum test-fixture mismatches.
`unit-v3/tests-2kcyt4u1/pytest.xml`: **52 passed**, no skips/failures/errors.
Final acceptance paths and counts are reported with the scoped commit handoff.
Pre-final combined unit check `final-unit/tests-sq6n_8av/pytest.xml`: 77 passed;
CPU smoke `final-smoke/smoke-aih6_di5/pytest.xml`: 79 passed / 4 CUDA skips.
`final-discovery/discovery-f_x57_x_/result.json` at 08:51:19 UTC was disk-replayed:
acceptance PASS, commercial qualification UNVERIFIED, planned=18/attempted=0.
Subsequent parser/missing-mode hardening gets fresh unit/discovery evidence;
these earlier source-pinned outputs remain preserved, not relabeled final.
Final hardened-source combined acceptance:
`final-unit-v2/tests-qjaduwqx/pytest.xml`, **79 passed** (54 runner / 25 existing
probe), zero failures/errors/skips, with provenance and CPU example accepted.
Fresh discovery `final-discovery-v2/discovery-mtmrzdmw/result.json` at
**2026-10-10T08:53:37 UTC**: all three SDKs absent, no SDK import/license use,
planned=18, attempted=0, qualification **UNVERIFIED**. A post-commit discovery
is additionally supplied in the handoff with its exact source Git SHA.
After main integrates new modules/API, run discovery again in that integrated
tree. Source membership is enumerated dynamically and hashed afresh; this
commercial worktree artifact cannot certify different main-source bytes.
The final `final-unit-v3` path and actual counts are supplied in the handoff.
Counts from these repeated runs overlap and must not be summed.

Offline controls include six solver-output mutants (x, y, NaN, shape, status,
model mutation), ten frozen-report mutants (x/y, parameters, content hash,
NPZ bytes, counts, approval, latency, source, missing mode), poisoned SDK discovery,
forbidden imports/sockets, missing SDK, malformed approval and private-sentinel
redaction including native file descriptors. Three recording SDK substitutes
exercise the actual adapter builders and verify quadratic factors, row/bound
mapping, explicit free variables, public parameters and returned dual signs.
All substitutes are offline tests, never proof of a real licensed solve.

External blockers remain: real approved license usage and authorized SDKs,
then licensed LP/QP execution and independent numerical replay on those SDKs.
No SDK installation, license-server connection, credential inspection, GPU
submission, infra modification or push occurred in this seat.

## Primary API Checks

Public official API pages were read on 2026-10-10 for adapter conventions;
no reference documents, SDKs, credentials or proprietary data were vendored.
They are implementation references, not licensed runtime evidence:

- Gurobi linear constraint attributes (Pi, BarPi):
  https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/constraintlinear.html
- MOSEK Task API (quadratic objective and host x/y access):
  https://docs.mosek.com/latest/pythonapi/optimizer-task.html
- MOSEK LP/QP tolerances and starting-point parameters:
  https://docs.mosek.com/latest/pythonapi/parameters.html
- IBM official DOcplex CPLEX parameter hierarchy (barrier, advance, solutiontype):
  https://raw.githubusercontent.com/IBMDecisionOptimization/docplex/master/docplex/mp/params/parameter_hierarchy_22200.py

One earlier IBM public-source request timed out after partial receipt; a later
complete fetch confirmed the parameter ranges. This is a research fetch failure,
not a solver/license failure or commercial qualification result.
