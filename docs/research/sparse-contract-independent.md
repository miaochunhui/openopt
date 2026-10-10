# Independent Sparse Contract Review

## Scope And Oracle

This independent tree starts at `c9616c4199e3cd9859dd007cd752862baec7c127`.
Only this document and `tests/test_sparse_contract_independent.py` are writable.
The operator, dense QP, presolve, existing tests and main tree remain unchanged.
There is no `src/openopt/precondition.py` in this baseline: the existing
Jacobi/PCG implementation is in `operators.py`, with
`tests/test_qp_precondition.py` and `docs/research/precondition.md`.

The new tests use Fraction dot products on represented original bytes,
independent componentwise feasibility/stationarity/normal/sign/complementarity
and gap formulas, analytic small SPD systems and a zero-row Farkas
contradiction. No production checker, projection, reduction or solve is an
oracle. Sparse tests prohibit densification and dense Gram factorization.
Finite probes are not PSD proofs; all positive P matrices are analytically PD.
All actual solves are CPU-only synthetic cases, not public or industrial
coverage, GPU evidence or commercial qualification.

Each observed preparation/solve saves original dtype/shape/raw bytes/hash,
unchanged caller bytes, complete config, working vector bytes, actual returned
vectors/status/residuals/iterations and independent metrics before asserting.
The review keeps failed cases in the denominator. No xfail, clipping, threshold
relaxation, solver repair or historical requalification is permitted.
The independent componentwise gates use absolute/relative `1e-7`; the core's
global criterion is documented separately, not claimed to be the same gate.

## First Counterexample Targets

The unchanged operator solver computes a scalar primal tolerance from the
largest `Ax` or projected `z` coordinate. The original problem
`P=[1], q=[-1e9], A=[[1],[0]], l=[1e9,1], u=[1e9,2]` is infeasible:
its second row requires `1 <= 0 <= 2`. Independently, `y=[0,-1]` gives
`A.T y=0` and box support `-1`. A one-unit violation must not become an
original-model accepted candidate because another row has scale `1e9`.
The ordinary-scale companion and four legitimate representations distinguish
a convergence-gate defect from incorrect sparse multiplication/callbacks.

Preparation casts q/l/u before checking reversed bounds. Float64
`l=1+1e-12, u=1` becomes equal in float32. Finite bounds `+/-1e100` can
become infinities, and positive zero-row bounds `[1e-100,2e-100]` can become
`[0,0]`. Tests require fail-loud rejection for these explicit original-input
losses, rather than silently treating a different mathematical model as the
caller model. This is not a demand for bitwise-preserving all float64-to-float32
rounding, or a claim that arbitrary-precision solving is implemented.

## External Execution And Replay

Evidence root:
`/hf3fs-jd/data/tmp/openopt-release-20261010/sparse-contract-independent`.
Pre-smoke is `pre-smoke-v1/smoke-5ch0i1s8/verification.json`: 80 passed,
four explicitly local-CUDA-unavailable skips, source consistency PASS.
Actual development counts and preserved findings are below. A final clean run
will be identified by an external `clean-final-v1/review-*/receipt.json`,
not by modifying old development receipts.

Run from the independent tree with the existing validation environment:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_sparse_contract_independent.py --run --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/sparse-contract-independent/clean-final-v1
```

The new test-file CLI saves a live collection, full pytest log/JUnit, full
source snapshots before/after and a machine-readable receipt. On pytest FAIL,
the existing verifier omits its normal final counts/source check; the
independent receipt parses actual JUnit and performs those checks separately.
Evidence PASS does not change review FAIL or accept the sparse contract.
A clean final run requires an exact two-file diff from the base.
All focused nodes must appear once in both collection and JUnit; missing,
extra, duplicate and coordinated truncated sets have negative controls.

No-solve replay uses the same file with `--audit-artifacts RUN_DIR`.
It re-reads raw bytes/hashes, recomputes Fraction metrics, compares fresh live
collection against actual JUnit, verifies production pins and complete source
membership, and writes a new immutable `replay.json`. It is not a new solve.
Each verification invocation is bounded to 170 seconds; an incomplete timeout
is not a completed review. No CUDA submission, network, install, credential or
shared publication operation occurs.

## Actual Development Findings

Development-v1 is preserved, not overwritten:
`development-v1/review-3c6cbdn8/receipt.json`,
SHA256 `c2bd1c28b23f3df113ec0e4941d085f4542c36b39764e689ebf413594715f4a0`.
Its actual JUnit is `tests-5bbyuk8k/pytest.xml`,
SHA256 `0cb12fb3e1b14a06cfd988b8cdadf5d27a3c17b6962e343c2b26b99e95d1c701`.
It has 52 nodes, 36 pass / 16 fail / zero error / zero skip, 8.52 pytest
seconds and 13.37517778002075 verifier seconds. Independent evidence replay,
collection/JUnit membership and full source consistency PASS; review FAIL.
The source was explicitly dirty development, not frozen acceptance.

1. **Original componentwise acceptance is unsafe across row scales.**
   All dense/COO/CSR/legitimate matrix-free cases returned `solved` at outer
   iteration 25, total CG 13, `x=999999999.9148482`,
   `y=[-0.10218226909637451,-40.000000000000014]`. Original zero-row
   primal/normal residual is exactly 1 against `3e-7`, while the core's
   scalar primal tolerance is `100.0000001` because z's first equality
   coordinate is exactly 1e9. Stationarity residual is
   `0.1873340606689453` against the scalar dual tolerance `100.0000001`.
   Original componentwise feasibility/normal/complementarity FAIL.
   The ordinary-scale companions keep residual 1 and correctly return
   `max_iterations` at 100. This is not an incorrect sparse adjoint,
   false PSD declaration, overflow or inner-solver failure.
   The core's documented approximate scalar convergence is not a certified
   infeasibility claim, but cannot be used as an original componentwise
   numerical acceptance gate. No such release acceptance is granted here.

2. **Conversion precedes original-input validation.**
   Three representations each accept the original reversed interval
   `1.000000000001 <= x <= 1` as a prepared float32 equality, and accept
   each finite lower/upper `1e100` magnitude as an infinite working bound.
   These nine failures are preparation/input-contract failures, not nine
   additional solved models. Source locations are `operators.py` preparation
   near lines 551 and 565-571, plus `_coerce` near 477-483.
   The separate presolve code explicitly validates original order before cast
   and rejects finite-to-nonfinite/nonzero-to-zero conversion; it is read-only
   corroborating policy, not the mathematical oracle or a proposed fallback.

3. **Positive tiny bounds silently become zero.**
   Dense/COO/CSR original zero-row bounds `[1e-100,2e-100]` become `[0,0]`
   in float32 and return `solved` at 25, `x=y=0`, CG 0 and working residuals 0.
   These three FAILs concern nonzero input loss and the exact zero-row
   contradiction. The fixed `1e-7` approximate original six gates DO pass
   the tiny violation. This is deliberately not reported as a six-gate false
   positive or a reason to tighten/relax any numerical tolerance.

There are four actual positive QP solves on the same analytic model, returning
`solved` at iteration 50 with CG 99, primal/normal
`1.7715264810647113e-9`, stationarity `1.660920734281035e-9`, and objective
`-6.000000007493963` versus exact optimum -6. All independent original six
gates pass. Six standalone actual CG systems compare against the analytic
solution `[1/11,7/11]`, with/without Jacobi across tensor representations.
There are 15 actual outer solver calls/returns total: four positive, eight
zero-row controls and three cast-zero-row controls. All failed cases remain
in the denominator; standalone CG tests are not additional QP coverage.

Final evidence additionally stores independent original reversed rows, exact
infeasible zero rows, selected cast losses and the scalar tolerance interval.
These are derived from raw byte arrays during replay, not accepted labels.
The development FAIL above predates those extra diagnostics and remains
immutable. A clean frozen rerun is for provenance and diagnostic completeness,
not a retry intended to make the failed production model acceptable.

## Bounded Signoff

This review rejects original-model contract acceptance on the pinned operator
source. The two independent files are reproducible evidence for an owner
repair, not a production change or an all-sparse capability qualification.
Owner closure requires validating original input semantics before destructive
casts and retaining componentwise original-model acceptance independently of
global convergence. Unchanged negative tests must reject/keep unsolved the
counterexamples; numerical thresholds cannot be widened or tiny data clipped.
Sparse formats, callback ownership, broad PSD certification, GPU, large-scale
conditioning, genuine public models and general presolve remain outside this
bounded signoff. No permission to edit any existing author file is inferred.

## Caller-Selected Owner Repair

The reviewer stays frozen in this independent tree; do not copy or edit its
52 original pytest nodes, fixtures, numerical expectations or Fraction oracle
to accommodate a repair. The caller may select a different clean local owner
tree with all three explicit CLI arguments:
`--owner-root /absolute/canonical/owner-tree --owner-commit FULL40HEX
--operator-sha256 FULL64HEX`. Partial, shortened, dirty or mismatched choices
are rejected before collection or solving. Equivalent pytest environment
names are `OPENOPT_SPARSE_OWNER_ROOT`, `OPENOPT_SPARSE_OWNER_COMMIT` and
`OPENOPT_SPARSE_OPERATOR_SHA256`; these affect only source selection, not math.

The production module is actually imported from that owner tree. The reviewer
still verifies its own historical source pins, and separately records/checks
the owner's full file membership/byte snapshot before and after execution,
caller commit/module hash and actual loaded module/function path. Replay
requires the same explicit caller choice, not a root or source label trusted
from the receipt. A repaired owner is not required to preserve historical
operators bytes or the reviewer's two-file-only ancestry; the reviewer itself
still must have the exact authorized two-file diff from its base.

For a new owner invocation use a new external artifact directory. Run/replay
do not write the owner worktree, run its tests, cherry-pick anything, or update
old FAIL artifacts. Owner run acceptance remains bounded to these unchanged
52 controls, original componentwise gates, conversion guards and CPU scope.
The original historical 16 FAILs are not requalified by a later repaired run.

Diagnostics development-v2 has the same 52 nodes and same 36/16/0/0 verdict:
`development-v2/review-ob6y18ba/receipt.json`, SHA256
`6bd0b519b3ae65853bbc88d812025e09a455f2fc24d8b115440b775c4fca43d3`.
Verification wall is 12.514338144974317 seconds. Three actual no-solve entry
negatives (partial caller tuple, wrong operator SHA, wrong full commit) PASS
at `entry-controls-development-v1/entry-78t45cn0/receipt.json`, SHA256
`1fafb5a6e6a428f8e1583c98a3472af08304c306b2b9ee1390164689c6f84d28`.
Post-smoke `post-smoke-v1/smoke-ljrf8jav/verification.json` has 80 passed,
four explicit local-CUDA skips, 6.86 pytest seconds and source consistency PASS.
These development receipts remain immutable. Clean-final and clean external
caller runs use new directories and record full reviewer and owner snapshots.

The entry negatives have a separate bounded CLI:
`--entry-controls --artifact-dir EXTERNAL_DIR`. They do not add/remove pytest
nodes, mutate an owner, or claim a solve. Exact review checks comprise six
Fraction componentwise gates at fixed `1e-7`, original reversed bounds,
finite-to-nonfinite/nonzero-to-zero selected conversion losses, original byte
immutability, sparse matvec/adjoint/ownership, analytic CG/Jacobi, strict device,
literal source bytes/full membership, and unique live collection/JUnit equality.
Passing a repaired run is restricted to these controls, not universal acceptance.
