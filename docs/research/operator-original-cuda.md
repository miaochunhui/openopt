# Original Operator Device Regression Producer

This two-file opt-in test producer exercises the committed original-input
engine, not a new sparse algorithm or independent general optimizer auditor.
Only `tests/test_operator_original_cuda.py` and this document are owned here.
Numerical source, author tests, independent oracle and shared publication
files remain untouched. Main/coordinator alone submits GPU Jobs; this scope
does not install dependencies, submit cluster work or alter exports/defaults.

## Scope And Limits

Dense, sparse COO/CSR and matrix-free requested-device problems exercise
float32/64 on four tiny analytic models: interior diagonal QP, equality QP,
bounded LP and a zero-constraint-row QP. COO/CSR retain sparse layout and the
test forbids `to_dense`/dense Cholesky while executing non-dense paths. These
small inputs are not representative sparse-performance/multi-GPU evidence.

Each representation separately exercises the contradictory zero row at a
remote 1e9 scale, reversed binary64 bounds hidden by float32 rounding, finite
lower/upper overflow and nonzero bound underflow. Conversion rejection cannot
be called a solve. The zero-row outcome must remain `max_iterations`, not an
infeasibility certificate. Matrix-free callbacks are constructed at the
requested dtype/device with original binary64 vector bounds; no callback
precision guarantee beyond these explicitly constructed actions is inferred.

Independent host `Fraction.from_float` calculations use the represented
original model, saved x/y, per-row feasibility, stationarity, signed dual
support/complementarity and analytic diagonal-PSD dual objective/range.
Negative signed gaps are never clipped/absolute-valued or certified. Analytic
primal/dual companions give zero exact gap; saved numerical candidates must
satisfy an explicit approximate KKT tolerance. Their raw signed gap is retained
as rejected if negative, and LP dual objectives remain unavailable if the exact
diagonal range condition fails. KKT-only acceptance never grants a certificate.
Fraction host work is disclosed validation
cost, not accelerated arbitrary precision, general PSD proof, original
engine gap certificate or commercial qualification. Strict tests may reveal
a CUDA arithmetic first cause; retain it rather than widen tolerances.

## CPU And CUDA Entry

Default CPU development:

```text
scripts/verify.py tests --target tests/test_operator_original_cuda.py --artifact-dir EXTERNAL_ROOT
```

Use the existing validation interpreter, infra identity, hidden CUDA and
single-thread OMP/OpenBLAS/MKL. CPU mechanisms run actual solves, but never
qualify CUDA. CUDA nodes default to explicit opt-in-disabled skips.

Coordinator formal one-card worker environment must include
`OPENOPT_RUN_OPERATOR_CUDA=1`. Then the unchanged existing entry is:

```text
scripts/gpu_release_check.py --target tests/test_operator_original_cuda.py --expression test_cuda --artifact-dir ASSIGNED_EXTERNAL_CASE
```

Opt-in CUDA unavailable, wrong device count or wrong hardware FAIL instead of
skipping/falling back. Actual placements must be `cuda:0`; a CUDA callback
returning CPU tensors must be rejected. The worker wrapper retains its normal
JUnit/source/resource telemetry and rejects failed/skipped/empty collection.
This producer does not launch it locally or from a cluster. Before dispatch,
main must independently review clean source and pin expected case membership.

## Evidence

Every model/conversion/zero-row node writes external `original.json`, then
`solution.json` and `proof.json`, byte descriptors and JUnit properties for
the final proof. Original model is frozen before solving; saved candidates
include JSON float values and binary64 hex identities. Source pins identify
operator engine, producer test and unchanged GPU wrapper; actual loaded
operator module path must be the assessed repository. Input tensors and source
bytes are checked unchanged. Proof records exact solver settings and pytest
node identity; neither is caller authority or an independently approved model.
Failed assertions keep original inputs and the
normal fail-loud pytest/JUnit logs; no retry or historical success backfill.

Dispatcher placement observes tensor operands/results inside prepare/solve,
including matrix-free actions, working/original vectors and sparse operations.
CUDA nodes also require positive actual CUDA profiler events and an external
Chrome trace with SHA/bytes; no-kernel trace fails. Scalar host control in the
existing solver is not hidden. Profiler is local evidence, not remote
attestation; authenticity, source/job/run/collection binding and independent
acceptance remain main/auditor duties. CPU sidecars explicitly label CUDA
UNVERIFIED. Execution attestation, performance and commercial verdicts remain
UNVERIFIED in all sidecars.

Portable self-tests need neither historical 3FS artifacts nor commercial SDKs.
The optional source-scan entry reads the established credential needle only
in memory and writes counts/pins, never values. Owner pre/post smoke, CPU
results, strict unavailable opt-in failure and clean freeze coordinates are
recorded externally after actual execution; this document is not acceptance.

## Retained First Causes

`operator-original-cuda/unit-v1/tests-0nx0h8q2/verification.json` under the
external release root records 20 failures, 34 passes and 53 opt-in-disabled
CUDA skips in 70.44s. Eight failures came from demanding exact candidate gap/
range for the approximate equality/LP API; twelve came from an unattainable
float32 inner tolerance or LP range. Equality binary64 x was
`[0.24999999999982012,-0.5000000146830866]`; its exact signed gap was negative.
LP binary64 y was `[0.9999999885422017,-0.9999999885422017,0.0]`, outside exact
zero-P dual range. Neither is reclassified as certified. Float32 CG true
residual 2.384185791015625e-7 exceeded 3.2360680103302006e-8; the new producer
uses the existing author-style explicit 1e-7 inner tolerance, retaining the
original failed run and disclosing the parameter change. Source was not edited.
New negatives require nearly feasible negative-gap and tiny-nonzero LP range
to remain uncertified. Saved numerical results are retained before independent
assertions so a later real device failure preserves the exact candidate.

## Owner CPU Receipts

External root: `/hf3fs-jd/data/tmp/openopt-release-20261010/`.
Base numerical source: `204934bc36284a40b8407b4e2fb68fa0d88184f0`.

- `operator-original-cuda/pre-smoke/smoke-2ltxv0cv/verification.json`:
  80 passed, four disabled local CUDA skips, 6.77s.
- `operator-original-cuda/unit-v2/tests-1nr6k_9w/verification.json`:
  development source only, 56 passed, 53 disabled CUDA skips, 26.67s;
  source consistency PASS. Receipt SHA256
  `bf5385a29a867034bacf9b5c69bfcea7614b009f2832b4b4858022e1b669b8bc`.
- `operator-original-cuda/strict-unavailable-v1/tests-250b0dm_/verification.json`:
  actual CLI with CUDA hidden and opt-in enabled, one failure, zero skips,
  3.16s, nonzero exit. Receipt SHA256
  `0bdd84d44082a1a5eb9b033adbea3c38017a9859cb0224550a7ea842b796a2ae`.
  First cause: explicit CUDA requested but unavailable; no CPU fallback.

The subsequent clean-source owner tests and smoke use separate roots
`operator-original-cuda/clean-unit-v1` and
`operator-original-cuda/clean-smoke-v1`; exact generated directories and hashes
must be read from their terminal receipts, not inferred from these names.
No CUDA solve, kernel trace, GPU acceptance, latency/speedup or commercial
comparison has been run by this owner. Independent review is pending before
coordinator dispatch. Owner CPU PASS never closes those gates.
