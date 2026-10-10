# Independent Tiny Bounded MIQP Review

## Scope

Reviewer tree starts at `acc199306936922b6d75330884310a58cde84a9b`.
Only `tests/test_branch_bound_independent.py` and this document are written.
Bacon owns producer, producer tests and producer documentation separately.
Dirty producer bytes are read for interface preparation only, never executed
or accepted. No main/infra/exports/QP edit, cluster, GPU, dependency install
or nested delegation is permitted.

Default tests check self-authored mathematics. Frozen producer tests skip
without an explicit clean root and full forty-character commit. Such skips
are not evidence of producer correctness. The review CLI requires both
subject and reviewer clean, pins all tracked file bytes and commit/tree,
and rechecks the subject after real bounded CPU tests. It uses the selected
current committed source, not test reconstruction as a replacement for
original arrays. A future API change is a new reviewed source, not an
automatic exemption.

The CLI collects this file through real pytest discovery and compares every
collected identity with JUnit in both directions, rejecting deletion,
duplication, unknown identities and skipped frozen cases. `--mechanism-only`
selects no producer and explicitly requires all frozen-subject cases to skip.
It never substitutes mechanism PASS for a frozen-subject numerical verdict.
Both modes bind the clean reviewer bytes as well as external collection,
verification and JUnit hashes. Execution is CPU-only with one math thread.

## Independent Mathematics

The oracle uses exact Fraction values of caller original float32/64 tensors.
No producer candidate/bound/PSD/summary/coverage helper is used as an oracle.
It separately derives strict represented-model row and integer feasibility,
the objective polynomial, exact symmetric diagonal-dominance PSD margins,
convex tangent lower bounds and signed infinite row support. Any nonzero
dual selecting an infinite endpoint is invalid, including a subnormal.
Independent polynomial/energy identities and tiny pure-integer enumeration
are test-only references, never production algorithms or commercial claims.

Coverage is reconstructed from root box, parent, split dimension/cut and
both children. Every original integer region must be present exactly once;
continuous boxes cannot silently shrink or be discretized. Terminal bounds,
pruning and global minimum bound are independently recomputed. Signed gap
is objective minus lower bound, never absolute value. A represented feasible
mixed point does not close a continuous region with a positive exact gap,
even under a loose declared threshold. `3*x_cont=1` has no binary64 feasible
point and must not acquire a fabricated incumbent.

Reference positives are analytic pure integer, mixed, singular PSD, LP,
nonrepresentable mixed optimum and mixed equality. Zero-node and zero-wall
budgets must call no QP. Explicit missing CUDA must fail, not invoke CPU.
Exact-DD rejection is tested with a negative margin hidden by binary64
summation and by float32 proposal casting. Host arithmetic overflow must
raise; no finite enclosure is invented. Proof metadata remains execution
and performance UNVERIFIED, with separately disclosed host/replay costs.

## Coordinated Negatives

Four fixed mutations preserve original model/config/source and first pass
both independent review and the producer replay on the baseline:

- Inflated lower bound, with refreshed outer report and envelope hashes.
- Fractional equality-violating incumbent plus candidate/objective/summary.
- Deleted coverage branch, retaining a freshly hashed incomplete tree.
- Wrong infinite-support dual sign, including the smallest positive subnormal.

Baseline and forged reports plus the rehashed envelope remain external,
with pins in JUnit properties. Both independent mathematics and real
producer replay must reject each mutant. A stale-hash rejection alone is
insufficient. The producer acceptance function is a subject under test,
not the independent mathematical oracle.

## Evidence State

Pre-edit smoke at
`branch-bound-independent/pre-smoke-v1/smoke-kw3cgypn/verification.json`
has 80 CPU passes and four explicit CUDA skips, 8.73 pytest seconds.
This predates reviewer/producer implementation and is not MIQP acceptance.

Producer was still dirty at initial interface read: base `acc1993`, three
untracked files. No dirty producer solve/replay was run by this reviewer.
Mechanism-only and clean frozen-subject results will have distinct receipts.
Final numerical signoff requires real zero-skip same-source review against
Bacon's explicit clean freeze; source presence and owner PASS are not signoff.

First mechanism run:
`branch-bound-independent/mechanism-v1/tests-l9yi1540/verification.json`,
six passed, 16 unselected-freeze skips, 2.90 pytest seconds. This is a dirty
reviewer development receipt, not a final source or producer signoff. Later
mechanism tests add four self-authored rehashed transcript controls and
four collection/JUnit controls without executing any dirty producer.

Frozen producer selected later:
`e2d15e11cfe490d4c8d44fd979f66ee3d7e146bf`, parent
`acc199306936922b6d75330884310a58cde84a9b`, in
`/data/dev/openopt-wt-release4-20261010-native-miqp`.
Independent development receipts, with the producer clean and unchanged:
`frozen-e2d15e-v1/tests-1audxuaw/verification.json` has 34/34 passes,
zero skip, 6.49 pytest seconds; `frozen-e2d15e-v2/tests-pfkk7_3g/verification.json`
has 57/57 passes, zero skip, 8.72 pytest seconds, consistency PASS.
These precede the final clean reviewer commit and are not its final receipt.

The producer's retained `frontier-first-cause/tests-1g1gt6vu/verification.json`
has one actual failure: mixed fixed-integer frontier leaves were incorrectly
upgraded under max_nodes=1 and abs_gap=1. Its SHA256 is
`ab4c25cc648cd67a2cfd24faacf4e371ad5b7894f177883208ac43e8ea92d284`.
The independent test constructs that same original mixed model without
importing producer tests, verifies unknown on the new freeze and rejects
a freshly hashed forged solved status. Other controls cover null incumbent,
null bound/model, typed bool/count/config/source binding, continuous-box
shrink, nonenclosing gap, empty integer and original row exclusion, actual
proposal dtype/nonfinite/status failures, and unsupported input with no QP.

The shared six-case fixture writes original-array descriptors, full reports,
subject pins, independent decisions and separate host proof/replay costs
outside the tree. Binary32/binary64 zero-budget controls prove preservation
before proposal dtype casting. Another test disables all producer mathematical
helpers, checker, factory and solver while replaying the independent oracle.
Numerical acceptance does not attest actual placement or GPU kernels, trusted
runtime mutation safety, general MIQP, arbitrary precision, or commercial
performance. CPU metadata cannot qualify CUDA.

## Clean Reproduction

Run the test file as a CLI with the existing CPU environment, setting
CUDA_VISIBLE_DEVICES=-1 and OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1:

```text
tests/test_branch_bound_independent.py
  --subject-repo /data/dev/openopt-wt-release4-20261010-native-miqp
  --subject-commit e2d15e11cfe490d4c8d44fd979f66ee3d7e146bf
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/branch-bound-independent/clean-final-v1
```

Use `--mechanism-only` without subject arguments for the distinct no-producer
mode. The CLI rejects dirty reviewer/subject roots and emits external
`review-*/receipt.json`, real collection and JUnit plus execution logs. Each
receipt binds both roots' commit/tree/all tracked hashes; failures retain
their first cause. Numerical testing has a 179-second process limit, not
permission to increase a cluster deadline. Final receipts are external:
recording them later must not retroactively change either frozen source pin.
