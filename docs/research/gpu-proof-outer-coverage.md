# Outer GPU Proof Contract Coverage

This is CPU unit coverage of the existing outer acceptor, not actual CUDA or
Kubernetes execution evidence. Only `tests/test_accept_gpu_proof_run.py` and this
document, plus the subsequently authorized review-script source selector, are
changed. Producer, acceptor, dispatcher, other tests and cluster state are
untouched. The branch starts from `d1fdcb7`.

## Real Numerics And Simulated Control

One module-scoped producer CPU run creates all nine original models, actual
solutions, sparse CG traces, scaling data and generated terminal directions.
The original P/q/A/l/u arrays and model digests are independently saved and
pinned before the first solve. The CPU result hash, source membership and
explicit CPU-only receipt are frozen; teardown verifies their whole inventory
is unchanged. In unbounded evidence the base equals actual terminal x and is
not replaced by an analytic witness.

Each outer run copies only frozen raw descriptors and arrays. Device and
coordinator metadata are explicitly fabricated unit fixtures. Requested 4500,
one GPU, exact command, actual-Job-shaped UID/owner, Pod-shaped node/image/
exit/restart records and runtime/completion logs exercise the contract but are
not fetched from Kubernetes. Every log and receipt labels this simulation.
An outer return of execution PASS is only expected verifier behavior on these
fake control records; actual CUDA execution remains UNVERIFIED.

The three positive groups cover all nine exactly once with three cases per
run, plus one-/two-case direction runs. Every result remains exactly one case
and numerical execution UNVERIFIED. Original model pins, coordinator/result
pins, numeric bundles, current source identity and full inventories are checked
by the real acceptor and producer seal/replay, not mocked success responses.

## Cost And Loaded Source

The inherited test file is already 582 lines, so the size exception was
reported before editing. No unapproved test files are introduced. Selectors
group cost by `outer_allnine`, existing inexpensive controls and negative
direction cases. Only one CPU production is executed per module invocation,
not per mutation. Sparse/scaling traces are reused, and each mutation copies
only referenced arrays for its one case rather than every case's evidence.

The fixture imports real `openopt` modules from its clean committed source
root after temporarily isolating the package cache/search path, and restores
the entire previous cache on teardown. It never relaxes or mocks Mendel's
loaded-source guard. Configuration and checker source filenames are asserted.
Foreign cached-module rejection is checked separately when the guard exists;
on the pre-fix baseline it is explicitly skipped, not falsely passed.

During outer audit solver preparation/solve/CG, factory, producer run/hardware,
CUDA operations and non-Git subprocesses raise. Git calls are restricted to
the real source-identity operations required by the acceptor. Certificate
functions and configuration classes remain unchanged so the loaded-source
guard observes genuine reviewed code.
The review script also selects the clean caller-pinned producer checkout before
ordinary `openopt` imports and restores previous modules/search paths after the
operation. This changes only loading strategy, not any numerical/source gate,
and never imports evidence snapshot code. Existing receipts remain bound to
their original reviewer bytes; a changed reviewer requires new receipts.
Bundle pin, exact schema and inventory rejection still precede source selection.
Seven bounded loader tests cover real trusted-root import, cache restoration
after success/exception, wrong commit and four pre-import pin/inventory negatives.

## Direction Negatives

Both infeasible and unbounded statuses cover terminal status substitution,
zero/boolean/fractional/over-limit/impossible-interval/disagreeing iteration,
ray sign/zero/missing ray, substituted or unexpected feasible base, missing
checkpoint, changed current iterate/difference/initial iterate, rho, selected
metrics and contradictory opposite summary. Mutations preserve original model
expectations and repin sidecar, result, collection and completion together.
The independent outer control audit must still pass before numerical rejection.
Thus negative tests cannot pass merely because a stale hash or simulated log
pin was overlooked. Rejection reason and caller pins are saved externally.

## Reproduction

Use the existing validation interpreter with `CUDA_VISIBLE_DEVICES=-1`,
`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, `PYTHONDONTWRITEBYTECODE=1`.
Full CPU verification:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/proof-outer-coverage/full --target tests/test_accept_gpu_proof_run.py
```

For bounded or fixed-source selectors, invoke pytest on the same file with
`-k outer_allnine`, an external `--basetemp`, cache directory and JUnit path.
`OPENOPT_PROOF_TEST_SOURCE_ROOT` explicitly selects a stable reviewed source
checkout; `OPENOPT_PROOF_TEST_REVISION` selects its committed fixture base.
An explicit external selector requires the complete 40-character commit and
a clean checkout at that HEAD, with actual bytes checked against Git.
Never execute a changing producer worktree. A changed selector is recorded
with the source hash; it does not exempt any loaded-source guard.

Test completion counts, negative receipts, CPU receipt and source consistency
are required before this document's coverage may be claimed verified. No
commercial qualification, GPU speedup or actual CUDA PASS is asserted.

## Preserved Evidence

Evidence root:
`/hf3fs-jd/data/tmp/openopt-release-20261010/proof-outer-coverage/`.

At 2026-10-10T10:29 UTC, `unit-v1/tests-5wh_2xos/pytest.xml` records 85 passed,
one explicit pre-Mendel-guard skip, zero failures/errors, in 496.35 seconds.
All three nine-case groups, both one-/two-ray denominators and all 36 coordinated
ray mutations ran. The enclosing `verification.json` is FAIL because the
authorized write set expanded during execution; it is not a release PASS.
`loader-v1/tests-vvbb4mpr/verification.json` independently records seven passed,
zero skipped and source_consistency PASS at 2026-10-10T10:29:52 UTC.

The initial real CPU receipt is
`unit-v1/tests-5wh_2xos/pytest-tmp/real-cpu-nine-not-cuda0/cpu-receipt.json`.
Its result pin is
`a34086bead36baaf077256a0d8bbaa8091e9ccc9af8ccbc16ed492d9133493b6`;
the independently pre-solve-frozen expectations pin is
`d5d64e99af8e8405f39be54542b6a2e6508771f89d63f77f438782b024200736`.
Every mutation directory under that run's `pytest-tmp/` contains an
`outer-ray-rejection-receipt.json` with case, mutation, first numerical cause,
caller collection pin, result pin and expectations pin.
For example, `test_outer_allnine_ray_semanti20/` records substituted base for
infeasible; the corresponding unbounded case is the next parameter instance.
Use the receipt's exact fields, not directory ordinal, as the authority.

Final unchanged-write-set verification uses `final-v2/tests-*/verification.json`;
its machine counts and source consistency are authoritative. Tests do not
claim compatibility was executed against a changing Mendel worktree. The
primary agent must select the formally frozen combined checkout for the
guard-present regression and independent review run/replay. No new public or
actual CUDA evidence is produced here.
