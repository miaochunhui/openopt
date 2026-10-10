# Independent Frozen Distribution Review

Producer freeze: `8ba0dafd74057527d5d3d63b4e40b641bf3cfac3`, including parent
`eb86b125641c51ed7d19748b12e12d9815782704`.
Only this document and `tests/test_distribution_rotated_independent.py` are
owned by this review. Producer scripts/tests/docs, main, infrastructure,
dependencies and GPU are read-only. No build, extraction or installation is
performed. All generated controls, subprocess logs and receipts are external.

## Historical Stop Gate

The existing frozen producer receipt is FAIL:
`/hf3fs-jd/data/tmp/openopt-release-20261010/wheel-rsoc-v2/clean-wheel/wheel-z8s6v_bx/distribution.json`.
Its installed receipt retains
`ValueError: installed unpinned bytecode/native executable forbidden`.
That installation must not be accepted or cleaned in place. The frozen
producer used `-I` without `-B`; `-I` ignores PYTHONDONTWRITEBYTECODE and
imports generated unpinned pyc, which the post-import check rejected.
The clean successor explicitly disables bytecode writes and also uses `-B`;
the original bytecode rejection and all numerical tolerances are unchanged.

## Independent Boundaries

The new tests exercise separate wrong wheel path, missing/extra module,
changed source SHA, wrong 0BSD expression/license payload, installed symlink
escape, cached executable and caller-manifest-SHA controls. Every synthetic
fixture first proves its unmodified positive baseline. Wrong-sign dual
controls directly recompute original-coordinate KKT and also challenge
the producer with an embedded false PASS.

The actual existing wheel's full Python membership and byte descriptors,
wheel SHA, complete frozen producer source, LICENSE, installed metadata and
historical installed receipt bytes are recomputed against the frozen evidence
and the reviewed checkout. Loaded installed project modules use an explicit
source-only, caller-pinned loader in a fresh `-I -B` subprocess. This never
executes pyc and does not delete or rewrite the historical installation.
It is an independent numerical diagnostic, NOT a replacement for standard
installed-wheel acceptance. A second fresh subprocess executes the unchanged
standard producer CLI on the clean successor installation, verifies all
loaded project paths/bytes and requires its genuine PASS, nine checks per
solve and six rejected controls. The clean successor contains no pyc/native
package executable. The standard probe is not bypassed.
All installed package and distribution files are hashed before and after.
Two native CPU float64 and two explicitly selected Clarabel CPU solves
use original RSOC LP/QP models, not transformed SOC inputs. Independent
equations, head signs, cone inequality, stationarity, complementarity,
analytic objective/solution and signed gap are recomputed without consulting
producer PASS. Installed checker results and wrong-sign rejection are
additional evidence, not substitutes for that algebra.

The review runner's rc0 means its assertions completed; its separate
`accept_distribution` is conditional on actual standard installed-probe PASS,
source consistency and all independent numerical/negative checks. CUDA, speed,
commercial solvers and dependency redistribution rights are unverified.
The owner successor build receipt is verified; this reviewer did not rebuild.

## Execution

Run from the independent worktree using the existing validation environment:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_distribution_rotated_independent.py run --evidence-dir /hf3fs-jd/data/tmp/openopt-release-20261010/wheel-rsoc-v2/clean-wheel-v2/wheel-9jkf1069 --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/distribution-rotated-independent
```

The runner calls existing verify with only this test target, enforces a
179-second process gate, validates separately collected exact nodeids against
complete zero-skip JUnit with two-way differences and duplicates all empty, and
pins both subprocess receipts. Plain pytest without explicit existing
evidence skips the two actual-installation tests and cannot qualify this
review. Terminal counts, timings, actual failures and commit follow only
after disk verification.

## Independent Oracle Correction

Historical first independent run
`distribution-rotated-independent/history-v1/review-lvk5b6ne` returned rc1:
17 passed, one failed, 13.87s pytest. Its additional arbitrary analytic dual
coordinate bound of 2e-6 rejected Clarabel QP at max error
3.66290645e-6, although direct original KKT passed. That bound was not part
of the producer acceptance contract and incorrectly reused the primal point
bound for a Hessian with diagonal entry 4. The independent check now records
the analytic dual error rather than treating it as an extra certification
criterion. Stationarity, primal/dual cone, complementarity, signed gap,
primal-point and objective bounds remain unchanged. The failed artifact is
retained; no producer failure or numerical accuracy improvement is inferred
from correcting this test oracle. To correct the freeze, only the producer
identity/standard-probe expected terminal were updated; all wheel, installed
byte, sign, direct original KKT and actual-solve assertions were preserved.

## Observed Development Evidence

Clean producer wheel receipt SHA256:
`6beb9b5b21bcaaddfcfeecff94405dd0cd49ea46a2a82319dbcfdc19f6f4e180`.
Its installed receipt SHA256:
`c391ae2e14567c4277d8d8a8b19d773b006db87a155ccd75edf1871a7ea784b4`.
Both were physically reread, not inferred from owner messages.
Owner clean CPU receipt has 93 passes/zero skips and source consistency PASS;
clean smoke has 80 passes/four explicit skips and source consistency PASS.
Those are owner evidence, not independent test totals.

Independent development terminal:
`distribution-rotated-independent/development-v2/review-7zur_ez1/review.json`,
22 passes, zero skip/failure, pytest 20.39s, process 29.5246s under 179s.
It separately performs four installed native/reference numerical solves and
the standard producer performs another four RSOC solves; neither is CUDA or
commercial execution. Development source is dirty only for these two new
review files and is byte-consistent across the run. Final clean-commit replay
and exact collection/JUnit receipts are separate evidence, not yet represented
by that development PASS.

Exact collection replay also passed at
`distribution-rotated-independent/development-v3/review-m5xcgo8m/review.json`:
22 passes, zero skip/failure, pytest 20.58s, 296 source files unchanged.
Collection/JUnit missing, extra and duplicate sets are all empty.

The `scan --artifact-dir EXTERNAL_3FS` entry uses the existing infra scanner
read-only on exactly these two review paths, with its 15 synthetic controls
and seven credential shapes. It does not invoke any cluster/credential reads
and does not claim a known-credential literal scan. Final clean replay is
recorded under `distribution-rotated-independent/clean-final`; that receipt
must show clean source and actual acceptance before integration.
Path scan `scans/path-scan-58upuenm/scan.json`: CLEAN, exactly two files,
zero hits, all 15 synthetic controls passed. A later scan is required after
any further review-source edits.

Known retained artifacts: historical producer eb86b12 bytecode FAIL, first
independent oracle FAIL, and two explicitly dirty development PASS runs.
No historical installation was repaired/cleaned, and no wheel was installed.
Only the two independent review paths are eligible for this review commit.
