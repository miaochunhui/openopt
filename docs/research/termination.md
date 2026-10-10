# Opt-In Dense QP Numerical Termination

This extension applies only to `PreparedQP` / `solve_qp`, not the batch,
sparse or matrix-free solvers. Both options default to false:

```python
config = QPConfig(
    dual_gap_termination=True,
    direction_termination=True,
    direction_atol=1e-8,
    direction_rtol=1e-8,
    adaptive_rho=True,
)
result = solve_qp(problem, config)
```

`solved`, `infeasible` and `unbounded` with these options are approximate
numerical statuses, not exact proofs or rigorous certified bounds. The default
finite path preserves its arithmetic and residual-only stopping behavior.
The independent implementation in `certificates.py` is unchanged.

## Original-Coordinate Gap

The workspace owns a CPU float64 copy of the input before solver dtype
conversion and symmetrization. At each stopping checkpoint the independent
candidate report is retained. `dual_gap_termination` additionally requires
every independent candidate check
to pass using `eps_abs` / `eps_rel`: original feasibility, stationarity, strict
infinite-bound signs, complementarity, dual availability and signed dual gap.
The result includes `dual_objective`, `duality_gap` and `termination_evidence`.

The dual is `-0.5 b.T P^dagger b - support(y)` for `b = q + A.T y` in the
numerical range of the original PSD matrix. No ADMM factor, sigma, z, solver
residual or regularized inverse is reused. A singular unresolved spectrum,
range failure, or unavailable support prevents solved status; it is not
replaced by a surrogate gap. Signed gaps are not clipped. Tolerance-level range
membership is numerical evidence, never a mathematically certified lower bound.

## Generated Directions

At each stopping checkpoint, `direction_termination` takes differences from
the preceding checkpoint (zero-start for the first). These are sums of ADMM
iteration increments, not planted corpus witnesses. The multiplier is the
unscaled y; adaptive-rho changes do not rescale it.

Both direction candidates are independently checked against original inputs:

- `dy`: normalized Farkas separation requires `A.T dy` near zero, admissible
  box-support signs and support strictly below the stated separation margin.
- `dx`: normalized recession/descent requires `P dx = 0`, strict negative
  `q.T dx`, two-sided rows near zero, lower-only rows nonnegative and upper-only
  rows nonpositive. Free rows impose no recession condition.
- Unbounded status also requires independently feasible current x as a base.
  A descent direction alone, including on an infeasible primal, is insufficient.

The checker requires resolved PSD eigenvalues and a near-roundoff positive
eigenspace component, not just a loose `||P dx||` test. In particular, an LP
has `P dx = 0`; tiny positive QP curvature is not treated as an LP nullspace.
Infinite-support multiplier signs and one-sided recession signs are strict.
Zero, rejected, unavailable, or conflicting direction evidence does not
promote status: iteration continues and exhaustion remains `max_iterations`.

Accepted results expose CPU float64 `infeasibility_direction` or
`unbounded_direction`, the latter with `feasible_base`, plus the full independent
`termination_evidence`. x/y remain on the requested solver device. Optional
checking explicitly transfers copies to CPU; it is not a GPU-resident
termination/performance claim. Malformed/nonfinite evidence and arithmetic
errors raise with kind, iteration, rho and the chained first cause.

## Acceptance And Boundaries

Run CPU acceptance with the existing interpreter; artifacts and JUnit remain
outside the worktree:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py \
  --target tests/test_qp_termination.py --target tests/test_qp_adaptive.py \
  --target tests/test_qp_certificates.py --target tests/test_general_qp.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/termination/acceptance
```

The suite compares disabled behavior bitwise with frozen commit
`a698c6744a0c5a667217ea683ac5c898d65a0d67` across float32/float64, both dense
linear solvers, adaptive/fixed rho and finite/nonoptimal families. Known
infeasible/unbounded synthetic corpus sizes are 3, 8, 24, 64 and 128 with
both seed parities (LP and QP) and both rho modes. Oracle validation precedes
solving; generated vectors and feasible bases are replayed by two independent
NumPy checks, with sign-flipped negative controls. JSON reports, source hashes
and vector archives are round-tripped; JUnit records their external paths.

Explicit `cuda:0` tests cover gap on SPD/LP/singular QP with both dense linear
solvers, and infeasible/unbounded directions with both seed parities and rho
modes. They assert actual x/y placement and independently replay host evidence.
CPU-only runs explicitly skip these 14 CUDA cases; no GPU run is claimed here.
The coordinator can select them with `pytest tests/test_qp_termination.py -k
cuda_opt_in` on its authorized 4500 execution path. These CUDA tests do not
require `.git`. Only the frozen baseline fixture does: source-only snapshots
explicitly skip its 40 comparisons, rather than substitute the current source.
The existing adaptive suite has its own Git-dependent baseline fixture outside
this write set; the coordinator must account for it separately.

Unimplemented: complete ray discovery, presolve/scaling/refinement, spectral
nullspace repair, exact or interval certificates, homogeneous embeddings,
batch/operator termination, public/industrial or commercial qualification,
GPU execution/performance validation. Rejected ambiguous directions are not
repaired or claimed as proofs. The synthetic corpus is not representative of
general industrial instances.

The edit rule router in infra rejected this worktree path as outside its root
(exit 2); no infra edits or alternate identity were used. The explicit
three-file assignment governs scope. `docs/qp-design.md` and generated report
pages are outside this seat's write set and require coordinator integration.
Preflight smoke passed 79 tests with four explicit CUDA skips; changed-source
acceptance results must be read from the external verification artifacts.

The first targeted run `targeted-v1/tests-xkkkwus2` retained 109 passed and two
failed test constructions: a large objective admitted the intended negative
control under relative gap tolerance; a residual-ineligible range candidate
had no report. The control now has zero optimal objective and checks the gap
itself; every enabled gap checkpoint now retains its independent report.
These original failures are not overwritten or reclassified as passing.

## Measured CPU Evidence

Evidence root: `/hf3fs-jd/data/tmp/openopt-release-20261010/termination/`.
Its resolved shared-filesystem alias in reports is
`/3fs/research3fs/data/tmp/openopt-release-20261010/termination/`.
Each verification directory contains `verification.json`, `pytest.xml` and
fail-loud execution logs. These counts overlap and must not be added.

| Run | Outcome |
| --- | --- |
| `preflight/smoke-m3ghq1m2` | 79 passed, 4 CUDA skipped |
| `targeted-v1/tests-xkkkwus2` | 109 passed, 2 retained failures described above |
| `acceptance-v2/tests-50azd5r2` | 545 passed, 2 CUDA skipped; dense/adaptive/certificates/batch/operators/corpus |
| `final-smoke/smoke-o6cb7w60` | 79 passed, 4 CUDA skipped |
| `termination-final-v4/tests-yi6sh3zu` | 122 passed, 14 CUDA skipped; finished 2026-10-10T08:45:29 UTC |
| `full-v3/tests-w4ucl62n` | 1713 passed, 44 skipped; complete regression, 338.29 seconds |

The full run loaded the termination suite before the final added
real-rho-update test. That additional test and the complete final termination
file passed separately in `termination-final-v4`; the solver source was
unchanged between these runs. CUDA and empty CUDA parameter selections are
explicit skips, not acceptance. No performance ranking follows from run time.

The final termination run includes 40 bitwise frozen comparisons and 40 known
direction solves across all five sizes, both seed parities and both rho modes.
It also checks a Farkas direction after actual rho updates (initial rho `1e-4`,
interval 1), not merely an enabled adaptive flag. Corpus direction artifacts
reside in that run's `pytest-tmp/test_known_direction_corpus_*` directories;
JUnit identifies each exact path. `vectors.npz` stores generated directions and
unbounded feasible bases; `direction-acceptance.json` includes original problem
hashes, UTC time, source hashes and both independent replays.

For the n=128, seed=20261011 fixed-rho QP, infeasible termination occurs at 75
iterations with normalized annihilation residual `3.4861e-14`; unbounded
termination occurs at 50 iterations with base violation zero, curvature
residual `1.1603e-14` and slope `-1.0000000000000908`. These are tolerance-level
numerical measurements, not exact identities.

Solver source SHA256:
`955d741317e5de1c5743f6e6bc65406ba14560f9a8b4c989afe6d11a1fc58e40`.
Unmodified independent checker SHA256:
`e7ece6f726bb7613d504ff6d84eedc20fb1a5879213a653680a26f8a64eb4c9c`.
