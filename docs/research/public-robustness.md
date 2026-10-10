# Explicit Public CPU Robustness Experiment

This is a three-file, opt-in experiment, not a solver/checker/runner/importer
repair or a qualification report. Baseline is the caller-pinned negative
two-LP run; its six attempts are copied byte-for-byte and replayed, not rerun.
Each case receives exactly one new attempt per predeclared arm. Failed and
budget-exhausted rows remain in the denominator. No automatic retry, tiny-dual
clipping, tolerance widening, GPU, download or dependency installation occurs.
The original corpus arrays, variable bounds and objective offsets are unchanged.
Source review does not become legal permission: license_verified and public
redistribution remain false. No speed/commercial/industrial claim is made.

## Arms And Gates

All arms use CPU float64. Independent full candidate/status/reported-objective
checks remain eps_abs=eps_rel=1e-7, directions 1e-8. Solve targets are separate:

- Openopt: bounded Ruiz (10 iterations, factors [1e-6,1e6], objective scaling
  enabled), fresh adaptive-rho dense solve at 1e-10 with 10000 maximum
  iterations, then explicit original-coordinate active-set refinement
  (three corrections, active guess 1e-5/1e-7, independent acceptance 1e-7).
  Refinement cannot turn max_iterations into strict solved. Rejected/unknown
  proposals retain the restored candidate; last proposal/checks are saved too.
- OSQP: fresh workspace, library default scaling, strict 1e-10 solve target,
  10000 maximum iterations, no warm start. Effective settings are saved.
- Clarabel: library default equilibration, strict 1e-10 solve target,
  200 maximum iterations, one thread, presolve disabled to retain transparent
  finite-bound mapping. Original signed y and cone z/mapping are saved.

The Openopt arm changes both scaling and postprocessing and tightens the solve
target; reference arms change scaling and targets. This is a combined robustness
experiment, not an isolated causal scaling ablation or a fair timing race.
Original strict sign/range/gap rules are never replaced by backend status.

## Evidence And Replay

`run --baseline PATH --baseline-sha256 PIN --artifact-dir NEW_3FS_DIRECTORY`
copies pinned baseline/source files and saves one worker result/vector/
transformation sidecar per case/arm. Parent execution has an explicit at-most
180-second wall budget including input preflight and worker startup; exhaustion
is BUDGET_EXHAUSTED, never success. Logs retain worker errors and original
causes. Each worker runs one solver call; refinement is a candidate
transformation rather than a second solver attempt. Replay never solves.

`replay --result PATH --result-sha256 PIN` verifies raw byte hashes, exact
sidecar inventory, original baseline replay, case/arm/source membership,
unchanged model hashes, original complete independent checks and all failed
denominators. It checks scaling restoration and refinement nonworsening/
unchanged-rejection semantics and reconstructs Clarabel's signed dual mapping.
Frozen producer source must match the replaying producer. This is diagnostic
integrity, not an exact mathematical certificate or public qualification PASS.

Source baseline is `e566b0b8163f099a29a0d554fa3b87ad15e4fb67`.
The infra edit router rejects cross-repository targets as outside its root;
applicable openopt AGENTS and the explicit three-file user scope govern this
work. Shared design/report pages are outside scope, not edited here.
Only the main coordinator submits cluster work; this seat does not.

## Retained Failure And Real Results

The initial cheap unit run is retained at
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/unit-v1/tests-yfgpvmk0/`.
It recorded 7 passed, 3 failed and 9 errors. The first cause was producer misuse
of the importer's exclusive-create JSON writer for updating a worker row,
progress and result. No real model was run in this failed phase. Worker-original
and parent-final rows now have distinct files, progress files are append-only
numbered snapshots, and terminal result is written once. Unit-v2 passed 19;
unit-v3 passed 22 including status-upgrade and exact Ruiz-factor replay controls.
Old evidence is retained, not counted as acceptance or overwritten.

The one real batch finished `2026-10-10T09:56:26.886824+00:00` in
`25.376415964012267` seconds, below its explicit 180-second budget:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/real-v1/result.json`.
Result SHA256:
`aa584b62f97349f37afad438dfbd87320ace3408df4d7e574a0989370970e8e7`.
It replayed baseline 6/6 and new 6/6 at `2026-10-10T09:56:53.327460+00:00`.
New denominator is six, all attempted, zero FAIL/budget-exhausted, five
UNQUALIFIED and one numerical QUALIFIED. Terminal execution status is
`EXPERIMENT_COMPLETE_NOT_QUALIFICATION`; numerical and original verdicts
remain UNQUALIFIED. Execution exit zero means completed evidence, not all-pass.

| Arm | afiro | adlittle | New Qualified / Denominator |
| --- | --- | --- | --- |
| Openopt Ruiz + refinement | max_iterations 10000 | max_iterations 10000 | 0 / 2 |
| OSQP default scaling, strict | solved, strict sign fails | max_iterations 10000 | 0 / 2 |
| Clarabel default equilibration, strict | full numerical gates pass | component complementarity fails | 1 / 2 |

Both Openopt refinements are unknown with `singular_or_unresolved_kkt`; no
candidate proposal was accepted or baseline value replaced. Adlittle original
feasibility/stationarity/complementarity maxima are
`0.0006463185027314466`, `0.06076771075350962`, `0.6428541482602704`.
Afiro Openopt did not meet its tighter scaled solve target; the previously
passing original baseline candidate is not replaced by this failed arm.

Afiro OSQP has original feasibility `1.2660450465773465e-9`,
stationarity `2.977019359695254e-11` and complementarity maximum
`5.852098652344556e-9`, but wrong infinite-side multiplier
`2.8355450826891544e-17` still makes support/dual/gap unavailable.
Adlittle OSQP has feasibility `0.002561842997682362`, stationarity
`0.05176518931972396`, complementarity `4.516118764172443`, plus strict-sign
failure. These are failed numerical candidates, not hidden convergence wins.

Afiro Clarabel passes with original signed gap `2.090996531478595e-8`.
Adlittle Clarabel passes feasibility/stationarity/sign/range/gap, but its
componentwise complementarity fails (maximum `1.1127175773220218e-7`);
the unrelated maximum threshold `0.013005056723289846` cannot mask it.
Original dual range residual is `1.2423939210748358e-8` and signed gap
`4.293484380468726e-6`. No tolerance is widened to promote this candidate.
This combined-arm experiment does not isolate scaling causality and does
not establish general robustness, a speedup, or two-problem qualification.

Caller inputs remain:
baseline SHA256 `b10d49a32a13a9f34b8bf86ea6bd8a126dc779bf72a47db3f069420f653456ef`;
corpus SHA256 `3a434edfa8165167f56be25732840f03bbace1a5f5b7d1de7cdab989673d3f0f`;
source acceptance SHA256 `1f82c8df6e9258bd8598884dd2bfee77f9e97f2ce7ec9f46035c5604822519fa`.
All raw bytes, arrays/model hashes and offsets are verified in the copied
baseline. Model SHA256 is
`16cb1249ad14facc394b318d3da4b2270390ea3d1c5537388aa798ca779d7d96`
for afiro and
`fb4c43a1366a429976b185e4841f302da1357100a2c42f6ca0fd23e7614051f5`
for adlittle. Original checker SHA256 remains
`e7ece6f726bb7613d504ff6d84eedc20fb1a5879213a653680a26f8a64eb4c9c`.

The independent external replay receipt entry writes a new file outside the
sealed run inventory, and never solves:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B tests/test_experiment_public_robustness.py --check-result /hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/real-v1/result.json --sha256 aa584b62f97349f37afad438dfbd87320ace3408df4d7e574a0989370970e8e7 --output /hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/replay-receipt-new.json
```

## Final Acceptance

Existing receipt:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/real-replay-v1.json`,
SHA256 `c17f1f02d0350aab55c4fd1cca94b42e776bf4d1668eb1bea6f3ff5cb9c06e5a`.
Its independent read-only replay at `2026-10-10T09:58:19.122842+00:00`
checks old 6/6 and new 6/6, retains all five new failures, and round-trips JSON
with qualification_upgraded=false.

Final targeted CPU run:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/final-tests-v1/tests-grgz29il/verification.json`,
finished `2026-10-10T09:59:43.572573+00:00`: 213 passed, ten explicit CUDA
skips, zero failures/errors. Source consistency PASS, 254 files. Targets are
this experiment (23 tests), existing scaling/refinement and qualification.
Negative controls include caller pin, vector bytes, resealed transformation,
tolerance, denominator, baseline verdict, clipping claim, extra inventory,
source membership, CPU budget and failed-worker accounting. The synthetic
accepted-refinement/max_iterations control explicitly keeps UNQUALIFIED.
Unit fixtures are not real public or industrial coverage.

Final CPU smoke:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/final-smoke-v1/smoke-sg8yaurb/verification.json`,
finished `2026-10-10T09:58:29.732412+00:00`: 80 passed, four CUDA skips,
zero failures/errors, source consistency PASS. Counts overlap, not additive.
No full-repository, wheel, GPU, commercial, speed or legal acceptance is claimed.

The verified producer/test SHA256 pins are
`8ceb0f2b0dd732fccfae1f1c13afdf585ee9aba0a59cbb8038fd4c4874d42e18` /
`3d336ab8b71a63be04259e2b80bdd888d99dbcb52574a8fe4c2eb196625d409c`.
These tests ran on the stated base commit plus exactly the three scoped
untracked files; this final evidence section is a documentation-only update.
The original negative result SHA256 is still the caller pin. Main source,
solver/checker/runner/importer and reports have no changes in this scope.
