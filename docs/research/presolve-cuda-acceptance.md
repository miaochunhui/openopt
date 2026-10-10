# Presolve CUDA Acceptance

## Scope

`scripts/accept_presolve_cuda.py` provides CPU-only `seal` and `accept` for
the versioned `cuda_presolve` selection in `tests/test_qp_presolve.py`.
The denominator is exactly eight cases: float32/float64 crossed with coupled,
active, constant and infeasible. Six cases must pass every independent
original-coordinate QP gate; two must pass the independent original-model
Farkas-direction checker. A release execution PASS alone is not numerical
acceptance.

The tool never imports a frozen test, executes producer code, invokes a
solver, reruns presolve, repairs saved vectors or reconstructs absent original
arrays. Unsupported schemas, domains, source drift and incomplete evidence
fail closed. This is scoped synthetic numerical acceptance, not an exact
certificate, complete presolve, commercial qualification or a speedup.

## Frozen Inputs

Both commands require independently supplied paths and SHA256 pins for
`result.json`, `run.json`, JUnit and the collected terminal Job/Pod status.
`accept` additionally requires the sealed bundle path and SHA256.
Input files remain immutable; output must be outside both the run tree and
current source, under shared 3FS tmp.

The tool checks committed clean current source, bidirectional core/snapshot
membership, run source identity, source manifest, result source hashes and
sidecar producer/checker hashes. Loaded/current/frozen bytes of both
`accept_release_cuda.py` and `accept_presolve_cuda.py` must agree.
Explicit source files selected by the release auditor are included in run
identity membership, not replaced by a docs directory glob.

JUnit must contain all eight exact static parameter identities, no failures,
errors, skips or duplicates, and exact per-case JSON path/hash/status
properties. Each JSON binds its own `full-model.npz`; cases cannot share
JSON/NPZ files or escape the immutable run. Duplicate archive members,
pickled/object/complex data, missing arrays, nonfinite arithmetic and
oversized archives are rejected.

The submitted Job and collected actual Job/Pod must agree on execution
identity, command and single-card NVIDIA runtime. The Pod must have the
correct Job UID owner, matching reported node, zero retries and completed
exit zero. Hardware, nonempty GPU/cgroup telemetry and completion logs are
also bound. Actual vector/ray/mapping placement is supported by the pinned
successful CUDA test assertions; NPZ contains explicit host copies.
This is not hardware attestation, protection against a malicious producer,
or an independent measurement of every intermediate tensor's placement.

## Original Model And Mapping

The sidecar carries all original pre-cast float64 and working/reduced arrays
`P/q/A/l/u`, saved original `x/y`, objective offset, variable/row partitions,
fixed values/identity rows and row shifts. The infeasible cases also carry
the original-coordinate direction. Original data, not reduced data, is the
input to the independent checker.

Mapping acceptance requires disjoint complete partitions, exact original
finite +1 identity equalities, exact zero feasible removed rows, saved
submatrices, exact rational substitution for q, row shifts, finite bounds
and objective constants. Lossy casts cannot establish exact fixing.
Zero-variable reduced arrays have explicit empty dimensions and are never
fed into the nonempty QP interface. The original model remains nonempty.

Removed non-selected rows must have zero multipliers. Fixed-row signed
multipliers must satisfy the original-coordinate stationarity restoration
equation including retained-row contributions; saved y is not repaired.
Every solved case must independently pass primal feasibility, stationarity,
bound sign, complementarity, available dual objective and duality gap with
the producer's versioned original thresholds (float32: 2e-4; float64: 1e-7).
Saved metrics/objective must exactly match replay. Infeasible cases require
the independently accepted original ray, not the failed candidate's gates
or reduced-model status. Ray checks use the current checker's default
versioned thresholds, matching this producer.

The NPZ original arrays are the authoritative saved problem, linked to
the pinned producer/JUnit. This tool does not prove correspondence to an
external user input or industrial corpus without an additional independent
input commitment.

## API

Run with the existing CPU validation environment, without installing
dependencies or submitting GPU work:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/accept_presolve_cuda.py seal --suite presolve \
  --result <absolute-result> --result-sha256 <independent-pin> \
  --run <absolute-run-json> --run-sha256 <independent-pin> \
  --junit <absolute-pytest-xml> --junit-sha256 <independent-pin> \
  --status <absolute-job-status-json> --status-sha256 <independent-pin> \
  --current-root <clean-final-source> \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/presolve-audit/final
```

`seal` writes `bundle.json`, its hash and a `SEALED` report only after full
validation. Invoke `accept` with the same four input pins and options,
plus `--bundle <absolute-bundle> --sha256 <independent-bundle-pin>`.
The optional `--log-dir` is an alias for `--artifact-dir`.
Successful acceptance emits `OPENOPT_PRESOLVE_CUDA_ACCEPT verdict=PASS`,
eight per-case outcomes and `qualified=false`, `certified=false`,
`commercial=UNVERIFIED`, `performance=UNVERIFIED`.
Failures preserve the first coordinate/cause, traceback, timestamps and
nonzero return code in external `acceptance.json`, `failure.log` and
`execution.log`. Malformed CLI arguments/path setup fail before a run
directory exists and are reported by the CLI, not as a fabricated report.

Python interfaces are `seal(current_root=..., result_path=...,
result_sha256=..., run_path=..., run_sha256=..., junit_path=...,
junit_sha256=..., status_path=..., status_sha256=...)` and
`accept(bundle_path, bundle_sha256, current_root=..., **same_input_pins)`.
`validate(bundle, pins=..., current_root=...)` accepts exact pinned
descriptors, never derives caller trust from bundle contents.

## Validation And Historical Run

Unit fixtures create real committed source trees, eight NPZ files,
JUnit properties, logs and collected terminal records. They execute the real
acceptor and independent checker, not a solver or GPU producer. Resealed
mutants cover original x/y, fixed duals, mappings, shifts, bounds, objective
constants, source/device/status, rays, complete-case denominators and
terminal failures. Fixture acceptance is not evidence of CUDA execution.

The historical run
`/hf3fs-jd/data/tmp/openopt-release-20261010/gpu-presolve-v1/release-regression-t49p57y6`
has eight recorded GPU passes from source `6c3dbef`, but predates the final
acceptor and the release-auditor explicit-docs change. It is preserved as
historical execution/scope diagnostic evidence, not accepted against new
source. Do not update its source hashes or weaken source binding.
Formal GPU numerical acceptance requires the main coordinator to freeze
the final committed sources, rerun the eight cases and supply independent
new result/run/JUnit/status pins.
