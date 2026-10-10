# Release CUDA Independent Acceptance

## Scope

`scripts/accept_release_cuda.py` accepts **correctness-suite execution**, not a
comprehensive GPU numerical certificate or performance/commercial qualification.
It runs on CPU, never submits Jobs, imports producer code, regenerates problems
from test factories, or compares its worktree HEAD to a run's informational HEAD.
Explicit `--current-root`, `--source-snapshot`, and `--run` select the real inputs.

The four 2026-10-10 runs have different frozen source bytes:

| Run | Selection | Completed | Limit |
| --- | --- | --- | --- |
| gpu-harness / w84kxkid | general CUDA + GPU execution | 7 | Includes CPU placement and unavailable-GPU negative |
| gpu-termination / vtd1pvg_ | cuda_opt_in | 14 | Gap/ray tests have no saved proof JSON |
| gpu-scaling-precondition / op3_v3cs | cuda | 10 | Three scaling JSONs lack frozen problems; PCG has no saved JSON |
| gpu-integrated / v2q7r14p | cuda or gpu | 33 | Includes the five target files and negative tests; same optional proof limitations |

All four producer JUnits report zero skips, failures and errors. These counts
do not prove independent replay, commercial parity, representative coverage,
speedup, or all execution on CUDA. A negative case absent from a selected suite
is explicitly `UNVERIFIED`, not silently borrowed from another suite.

## Binding Contract

Caller pins the immutable `case-0/cuda-check-*/result.json` SHA256 first; do not
use the mutable coordinator `case-0/result.json` mirror. `seal` records exact
path, byte count and SHA256 of result, JUnit, pytest/execution logs, telemetry,
run and source manifest. Caller separately pins the resulting bundle SHA256
for `accept`. Pins must come from a trusted independent caller channel; hashing
an attacker-replaced file and accepting its new pin does not authenticate it.

Acceptance rejects empty/unknown source sets, unsafe paths, symlinks, duplicate
JSON keys, nonfinite JSON, skipped/failed/incomplete/duplicate/out-of-target
testcases, absent completion summary, contradictory counts and artifact mutation.
Frozen AST must contain every JUnit testcase function. The pinned Job manifest
must match run identity, exact target/selection command and a non-retrying
one-card request; result claims cannot substitute a different selection.
Full frozen snapshot, manifest, run source set and bundle source set must match
in both directions and in bytes. Result source set must match precisely the
producer's top-level openopt core, explicit targets and runner, not an arbitrary
subset/superset. New nested core appears in current/frozen coverage differences.

Current core membership and hash differences are separately reported as
`missing`, `extra`, `changed`. Historical execution can pass while current core
coverage is `UNVERIFIED`; this is never current release qualification.

Machine-read proof paths are read only from JUnit testcase properties ending
in `_path`, restricted to JSON inside that exact run's shared 3FS artifact tree.
Their descriptor set must match the observed property set exactly; artifacts
cannot silently appear/disappear. Saved model/device evidence must agree with
run hardware. Run hardware/runtime, GPU UUID samples and frozen source placement
assertion test identities are retained, not converted into per-vector placement
proof when the producer did not save it.

## Numerical Replay

A saved sidecar can supply `problem` with original `P,q,A,l,u`, finite saved
`x,y`, and frozen `atol,rtol` (at most 1e-3). CPU NumPy independently recomputes
feasibility, stationarity, complementarity, PSD range, objective and dual gap.
PCG additionally requires `system,rhs,inner_x,inner_tolerance` and recomputes
the true residual. Ray replay requires saved `direction,status`; unbounded
replay also requires `feasible_base`. Independent support/transpose residual,
recession, curvature and descent are recomputed. JSON `null` denotes an open
lower/upper bound, not missing matrix/vector values. Positive directional
curvature never qualifies as unbounded. Invalid vectors or overbroad thresholds
are rejected. No exact or comprehensive qualification is issued, even for
successful scoped numerical replay.

Missing problem, tolerances or PCG vectors means `UNVERIFIED`; test source,
seed, oracle label and reported metrics are not substitutes for frozen input.
The observed three scaling sidecars save `x,y` but no `problem`, so their KKT
metrics cannot be independently certified from this evidence. Gap/ray and PCG
have no linked saved vectors. The tool does not invent any sidecar.

## CPU Reproduction

Use the existing validation interpreter with `CUDA_VISIBLE_DEVICES=-1`,
`PYTHONDONTWRITEBYTECODE=1`; all execution uses `sudo -n -u infra`.
Run the existing `scripts/verify.py --target tests/test_accept_release_cuda.py
--artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/gpu-audit/unit`.
The unit fixtures exercise real file/JUnit/CLI acceptance, including semantic
mutants resealed with new hashes and direct artifact mutation, without GPUs.

For each real run, invoke `seal` with the independently read result hash,
immutable result path, its `source` directory, `run.json`, and
`--current-root /data/dev/openopt`. Then invoke `accept --bundle ... --sha256 ...`
with its caller-pinned seal hash. Every CLI execution stores an external
`acceptance.json`, `execution.log` and, on failure, first-cause `failure.log`.
Use `/hf3fs-jd/data/tmp/openopt-release-20261010/gpu-audit` for this seat's logs.
Acceptance does not alter run/proof source trees or the main repository.

## Observed Audit

The 2026-10-10 09:03 UTC independent read accepted execution for 7/14/10/33
cases. Only integrated had empty current-core differences (9/9 core files).
Harness missed conic/integer/scaling and changed init/batch/operators/qp.
Termination missed scaling and changed init/operators. Scaling-precondition
changed init. These historical runs are not source-current qualification.
Integrated had six completed negative tests. Optional independent replay
remained `UNVERIFIED`: only three scaling JSONs were linked, all lacking frozen
problems; gap/ray/PCG saved sidecars were absent. The latest acceptance artifacts
under the seat's external directory are the machine-read evidence authority.

Final validator rerun at 2026-10-10 09:05 UTC preserved these conclusions.
External CPU unit evidence: `gpu-audit/unit-final/tests-nlf7tbfq/verification.json`,
42 passed, zero skips (including resealed Job/source/JUnit mutants).
Final integrated evidence:
`gpu-audit/integrated-final/accept-z8b6jx5a/acceptance.json`.
Its immutable result SHA256 is
`b1749e0da9af2136feb0d200e3d2b484fce68f5b2b4fda6b6cc766d802608ef0`;
caller-pinned bundle is `integrated-final/seal-3simas5m/bundle.json`,
SHA256 `16014a56cd1fc9e73f438599f83468fe45c4d0deb6304918be4caffda48c2fdc`.
All artifact paths in this section are relative to
`/hf3fs-jd/data/tmp/openopt-release-20261010/` (bundle relative to `gpu-audit/`).
The integrated frozen snapshot has 160 files, checked bidirectionally against
the run and manifest; **current-source comparison covers the 9 openopt core
files only**, not every current repository file. Nine matching core files
establish byte freshness, not that every module has a selected CUDA testcase.
