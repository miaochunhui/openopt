# Public LP Candidate Diagnosis

## Boundary

`scripts/diagnose_public_qp.py` only replays saved candidates from an externally
pinned qualification result. No solves, download, installation, model change,
tiny-dual clipping, tolerance widening, status upgrade or new commercial/legal
qualification occurs. Its exact three-file scope excludes checker, solver,
runner, importer and generated reports. The branch starts at
`c451dcf2c0f2705c0566c0172b773eb85b38522d`.

The producer verifies result and sidecar byte hashes, exact on-disk inventory,
frozen input/model membership, original source eligibility, six ordered attempts,
and original assessment/aggregate replay using the saved importer, runner and
checker. It does not load or execute the solver implementation. The independent
LP calculation checks every constraint side, variable bound and stationarity
component at the **original** `eps_abs/eps_rel`, comparing all checks, metrics,
thresholds and unavailable reasons with the pinned checker.

The diagnostic is restricted to `P=0`. General singular QP spectral analysis is
not reimplemented. Component coordinates preserve original names, row order,
variable bounds and objective sign; no variable shift or offset normalization
is allowed to widen acceptance. Every multiplier is preserved, including
tiny nonzero wrong-sign values. Signed negative numerical gaps are not clipped.

## Rules And Interpretation

The governing rules are frozen `code/certificates.py`, hash
`e7ece6f726bb7613d504ff6d84eedc20fb1a5879213a653680a26f8a64eb4c9c`,
and frozen runner hash
`7c6f526575c21e92c8a6b685b7fd57eee61c65e2d9ee4b4649776f6cbe131cce`.
Local `docs/research/certificates.md` explains the rules; the tool records its
hash and exact pinned function locations, not a floating latest API claim.

Feasibility, stationarity and complementarity are componentwise tests. A
reported maximum residual below the maximum threshold can coexist with a
failing smaller-threshold component. The current report maxima are descriptive,
not the rule. Diagnosis fixes the visibility gap by exposing actual failing
coordinates, local residuals, limits and ratios; it does not loosen the gate.

A positive multiplier selects the upper side and a negative multiplier the
lower side. If that side is infinite, support is infinite for **any** nonzero
multiplier; the sign threshold is exactly zero. Tiny multipliers cannot be
deleted after solving to retroactively accept a different witness.

For an LP, the Lagrangian infimum over the original unrestricted x coordinates
is finite only when `b=q+A.T@y=0`; original variable restrictions are included
as identity rows. Otherwise a nonzero b admits an arbitrarily negative linear
infimum. The numerical range rule is `||b||inf <= atol+rtol*||b||inf`,
distinct from cost-scaled stationarity. The zero-range condition is necessary;
its numerical tolerance is a conservative policy, not an exact certificate.
The current checker can accept tiny nonzero numerical b, so a finite reported
dual is only approximate evidence. Substituting the larger stationarity limit
would not establish the missing Lagrangian range condition.

When bound support is unavailable, the tool shows an independently computed
unmodified LP b residual labelled **diagnostic only**; it does not manufacture
an available dual/gap or overwrite the original checker's null fields.

Iteration exhaustion proves the saved configuration did not meet acceptance.
It does not prove scaling is the unique cause. Coefficient magnitude spans,
settings and residual failures are evidence; no iteration trajectory or
controlled scaling ablation was captured, so scaling causality remains
UNVERIFIED. Future solver work belongs to separately authorized scopes.

## Original Evidence

Result:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-real-qualification-v1/result.json`;
caller SHA256
`b10d49a32a13a9f34b8bf86ea6bd8a126dc779bf72a47db3f069420f653456ef`.
Original terminal status remains `UNQUALIFIED`, two public LPs, six attempts.

Independent saved-candidate replay established:

- afiro/OSQP: strict solved, feasibility/stationarity pass; wrong infinite-side
  multiplier `2.7755575615628914e-17` at view row 42, variable bound `X22`,
  selects the absent upper side and blocks support/dual/gap. Independently,
  view row 58 (`X39`) complementarity is `1.0751789013827996e-7` against
  its local `1.0000001075178901e-7` limit, ratio `1.0751787857818451`.
  Removing the tiny multiplier would still not pass the original gate.
- afiro/Clarabel: strict solved, only componentwise complementarity fails.
  There are 21 failing rows; worst ratio is view row 58 (`X39`):
  `5.249453015076714e-7` against local `1.0000005249453014e-7`,
  ratio `5.249450259402465`. The unrelated maximum threshold
  `4.381714237173428e-5` cannot be used for that row.
- adlittle/Clarabel: strict solved, feasibility passes; stationarity and
  complementarity fail on 8 variables and 118 rows. Worst stationarity
  ratio is variable index 5 (`...105`): `1.4121967408945853e-5` against
  `1.000014121967409e-7`, ratio `141.21767981798658`. Worst complementarity
  is view row 78 (`...122`): `5.768178779444163e-5` against
  `1.0000576817877944e-7`, ratio `576.7846079770559`. LP range has
  52 failing components; its maximum is variable index 74 (`...174`):
  `2.2527900490842967e-5` against `1.0000225279004908e-7`, making dual/gap
  unavailable. This is not a range-only rejection.
- adlittle/Openopt and OSQP: iteration budget 5000 exhausted. Their original
  feasibility, stationarity and complementarity are independently outside
  tolerance: Openopt has 47/65/77 failing rows/variables/rows respectively,
  OSQP 48/53/75. Worst feasibility is constraint view row 4 (`....05`),
  ratios `322462.17123319925` and `354318.2948023402`; worst stationarity
  is variable index 42 (`...142`), ratios `127078.6674398441` and
  `468038.8485398736`. There are also 1 and 16 infinite-support sign failures.
- afiro/Openopt: original numerical candidate passes; it remains a positive
  replay control, not two-problem or commercial qualification.

Coordinates above are zero-based in the original augmented constraint view
(original constraints followed by variable bounds). Adlittle's nonzero A and
q magnitude spans are `53583.333333333336` and `1838.888888888889`; afiro's
are `22.700934579439252` and `31.25`. The saved reference settings disable
OSQP scaling and Clarabel equilibration. These facts motivate a separately
authorized robustness/scaling experiment, not a causal conclusion or a
tolerance relaxation. No solver repair is made by this diagnostic scope.

## Accepted Evidence

Real replay finished at `2026-10-10T09:39:19.982928+00:00`:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-diagnosis/real-v2/`.
Its six candidates independently agree with the original checker, including
the five failed attempts and afiro/Openopt positive control.
`diagnosis.json` SHA256:
`8d3af182738043b39e1b5a4d4deaddd4e84c0a03bde6a5708d9b91e509b03592`.
`summary.json` SHA256:
`f89652c1530b6cd7df90e52b58c4b088a99a928f4bd236c289624d61bb023e2c`.
Terminal verdict is `DIAGNOSED_NOT_REQUALIFIED`, original `UNQUALIFIED`,
`qualification_upgraded=false`. Source/legal rights are not upgraded either.

CPU regression finished at `2026-10-10T09:40:09.588792+00:00`:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-diagnosis/final-tests-v2/tests-6lzrz_9f/verification.json`.
All 131 tests passed, zero failures/errors/skips, with source consistency PASS
across 245 files. Targets are diagnostic, existing certificate and existing
public-qualification tests. The diagnostic suite contributes 22 tests,
including eight saved-input tamper controls (caller pin, NPZ, resealed NPZ,
rules, membership, row, status and aggregate), diagnosis upgrade and compact
summary tamper rejection, exact tiny-sign failure, component-threshold
counterexamples, malformed inputs and a JSON round-trip.
Synthetic existing-runner tests may solve their fixtures; the real diagnostic
replay itself performs zero solves. These counts are not public-model coverage.

CPU smoke finished at `2026-10-10T09:39:31.370251+00:00`:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-diagnosis/final-smoke-v2/smoke-hswokl6r/verification.json`.
79 tests passed with four explicit CUDA skips, source consistency PASS.
This is not a full-repository or GPU validation, and test counts overlap.
The producer/test SHA256 pins in these runs are
`312c0c4c29959e18ece4611ff0beff1d00ebf8589066251fcf963300dae3aef1` /
`52019ec111c72f87d4a6b71f140de6fba5aae018aa30ceaf30fed46fd4fe9f75`.
They ran against baseline `c451dcf2c0f2705c0566c0172b773eb85b38522d`
with exactly the three new scoped files untracked; this evidence predates
the final documentation-only handoff update.

The infra router rejected the explicit cross-repository edit target with
exit 2 (`unsafe/outside-root`). This is a routing scope mismatch, not permission
to edit infra. The applicable openopt AGENTS and explicit three-file assignment
govern this task; selected inspect/test/data/commit rule texts were read in full.
Checker, qp, runner and importer remain byte-unchanged relative to the baseline.

## Reproduce

Use the existing CPU environment with `CUDA_VISIBLE_DEVICES=-1` and
`PYTHONDONTWRITEBYTECODE=1`, infra identity. CLI:

```text
diagnose_public_qp.py diagnose --result EXTERNAL_RESULT_JSON --result-sha256 CALLER_RESULT_PIN --artifact-dir NEW_EXTERNAL_DIRECTORY
diagnose_public_qp.py verify --diagnosis DIAGNOSIS_JSON --diagnosis-sha256 CALLER_DIAGNOSIS_PIN
```

Evidence/log/JUnit root:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-diagnosis/`.
`diagnosis.json` holds all components and source pins. `summary.json` contains
machine-checked worst-ratio coordinates and failure counts and is compared
against the full diagnosis on replay, not accepted as a free-standing claim.
`status.json` plus
`diagnosis.log` records terminal timestamps, and failures retain traceback and
exit code 2. Existing output directories are refused. `DIAGNOSED_NOT_REQUALIFIED`
and `DIAGNOSIS_REPLAYED` mean diagnostic integrity, not public solver PASS.
Replay is read-only and needs the original pinned result/sidecars and matching
diagnostic source. Unit fixtures are synthetic, never industrial coverage.

From the scoped worktree, the exact read-only real replay command is:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/diagnose_public_qp.py verify --diagnosis /hf3fs-jd/data/tmp/openopt-release-20261010/public-diagnosis/real-v2/diagnosis.json --diagnosis-sha256 8d3af182738043b39e1b5a4d4deaddd4e84c0a03bde6a5708d9b91e509b03592
```
