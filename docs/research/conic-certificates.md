# Bounded Original Conic Direction Checks

This independent opt-in module is a host checker, not a direction producer,
solver, HSD implementation, arbitrary-precision optimizer, GPU accelerator or
commercial qualification. Existing cone/candidate checks, NumPy defaults,
solver statuses and public exports are unchanged. Import directly from
`openopt.conic_certificates`; the solver does not call these functions.

## Original Mathematical Contract

For the supplied original dense problem
`min 0.5*x.T*P*x + q.T*x`, `A*x+s=b`, `s in K`:

- `check_conic_infeasibility_direction(problem, y)` requires `y in K*`,
  every original component of `A.T*y=0`, and `b.T*y` strictly negative beyond
  its independently recomputed margin. If a feasible pair existed, then
  `b.T*y = x.T*A.T*y + s.T*y >= 0`, a contradiction.
- `check_conic_unbounded_direction(problem, d, x0=..., s0=...)` requires BOTH
  supplied original feasible base parts: `A*x0+s0=b`, `s0 in K`. It also checks
  every component of `P*d=0`, `-A*d in K` and strict negative `q.T*d` beyond
  its recomputed margin. Then `x(t)=x0+t*d`, `s(t)=s0-t*A*d` are feasible for
  `t>=0` by cone addition; symmetry and `P*d=0` make the objective
  `f(x0)+t*q.T*d`. Missing base returns failed evidence with an unavailable
  reason. Dual infeasibility alone is NEVER called primal unboundedness.

Zero cone primal components must be zero; their dual components are
unrestricted. Nonnegative is componentwise self-dual. SOC is self-dual with
head nonnegative and original `head^2 >= sum(tail^2)`. RSOC is self-dual in
the original Euclidean coordinates with BOTH heads nonnegative and
`2*a*b >= sum(tail^2)`. No coordinate conversion or global cone-scale tolerance
can mask a tiny negative head, one-ULP boundary violation or another block.

## Precision And Unsupported Scope

Version `bounded_host_binary64_fraction_v1` accepts finite real dense double
arrays/CPU double tensors (copied, never mutated) or exactly representable
integers, with `1<=n<=16`, `0<=m<=128`. Float32, complex, boolean, sparse,
non-CPU inputs, inexact integer conversions and unsupported shapes/cones fail
loud. Explicit `device=cuda[:index]` raises even if CUDA is available; there
is no implicit transfer, accelerator claim or CPU fallback from CUDA.

The full original matrix must be exactly symmetric on its supplied binary64
entries, with no symmetrization, triangular expansion or regularization.
Bounded rational Schur complements check original PSD, including zero-pivot
rows and singular non-diagonally-dominant matrices. A zero pivot with nonzero
remaining row is indefinite, not a tolerated spectral kernel. Each rational
intermediate has a 32768-bit numerator/denominator budget; exceeding it raises
an unavailable-bound error, not PASS. This implementation is cubic host work
with bounded rational cost, NOT a scalable general conic algorithm.

Rational arithmetic independently evaluates the SUPPLIED binary64 data, not
an unknown intended real model. Necessary equalities and cone membership are
conservatively exact on those represented values, so ordinary approximate
solver rays/base points can be rejected even within their residual tolerance.
Residuals AND identity decisions are shown per component. No tiny positive
curvature is allowed to masquerade as `P*d=0`.

Every reported scalar has an outward binary64 lower/upper enclosure. A
rounded nearest scalar is never treated as exact. Dot product terms, signed
and absolute sums, cone square/product bounds, and thresholds must remain
representable. NaN, infinities, overflow, loss to zero by underflow or
unavailable outward bounds raise errors, including canceled overflow terms.
The supplied base objective and each original base-equation absolute sum
must also have finite bounds; a recession ray cannot hide an unavailable base.
These conservative limits can reject mathematically valid extreme directions.
The negative test uses `upper(dot) < -upper(atol+rtol*sum(abs(products)))`,
default `atol=rtol=1e-8`, with finite positive atol and `0<=rtol<1`. It is
strict, signed, uncapped and never clipped to zero. Large canceled terms
increase the margin rather than hide it. Tolerances do not enable relaxed
kernel, base feasibility or cone acceptance.

All returned `NumericalEvidence` explicitly states APPROXIMATE host numerical
evidence, not an exact certificate. Passed data checks do not attest execution,
source trust, original-model provenance or a producer. GPU/execution and
commercial qualification remain UNVERIFIED. No third-party solve is performed.

## Self-Contained Acceptance

Analytic original cases cover equality/nonnegative/SOC/RSOC Farkas directions,
unconstrained LP, each recession cone, equality QP and non-diagonal singular
PSD. Independent Fraction oracles recompute negative expressions. Mutants
cover signs, margin equality, absent/infeasible base, nonzero `A.T*y`, tiny
positive curvature, near-zero indefinite matrices, wrong cone axis, one-ULP
SOC/RSOC violation, cross-component masking, nonfinite values, overflow and
unsupported scope. Independence tests forbid solve/candidate/eigh calls.

Use the existing interpreter without dependency installation, hide CUDA and
keep artifacts outside the worktree:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --target tests/test_conic_certificates.py --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/conic-directions/clean-cpu
```

The receipt freezes actual original arrays, cone order, directions/base parts,
ten numerical examples, four retained negative controls, qualification and
SHA256 of the three scope files plus read-only conic/certificate dependencies.
Its solver-call count is zero and GPU execution is UNVERIFIED. On CLEAN source,
the CLI re-reads final JUnit/counts, verification/source consistency, hashes
and frozen original arrays, then rechecks directions without any solve/factory:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_conic_certificates.py --audit-artifacts EXTERNAL_TEST_RUN_DIRECTORY
```

Run `scripts/iteration_gate.py smoke` before/after changes. CPU checks and
default API smoke are not GPU, performance, general-conic or release acceptance.
No network, GPU job, credential change or public distribution is needed.

## Owner Development Evidence

External root: `/hf3fs-jd/data/tmp/openopt-release-20261010/conic-directions`.
Pre-smoke `pre-smoke/smoke-vcwh4692/verification.json` was 80 passed/four
explicit local CUDA skips, 9.41 pytest seconds. First CPU receipt
`development-cpu/tests-z4x_ukfu/verification.json` remains FAIL, 96 passed/three
failed, 8.82 seconds: the multi-dimensional empty-LP TEST fixture constructed
a 1x1 P regardless of q size, so dimension checks rejected before arithmetic
negative controls. Only the fixture was fixed; no production gate was relaxed.
The original three counterexamples then ran through the intended arithmetic
checks at `development-cpu-v2/tests-mg_4bp4p/verification.json`, 99 pass/zero
skip, 7.80 seconds. Extended original-coordinate tests subsequently passed
113/zero skip, 8.96 seconds in `development-cpu-v3/tests-kmynn0xf/verification.json`.
Development post-smoke `post-smoke-development/smoke-zhah8jcg/verification.json`
was 80 pass/four CUDA skips, 8.79 seconds. Both last runs had source consistency
PASS. These are development receipts, not clean freeze or independent signoff.
Final clean receipts live in `clean-cpu` and `clean-smoke`; use the real final
verification/JUnit and `conic-directions-audit.json` written by the re-read CLI,
not these development counts, for review. Original first-cause evidence stays.
