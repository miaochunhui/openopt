# Commercial Readiness Review

## Verdict And Scope

Review cut: 2026-10-10, live collection 10:03-10:09 UTC. Independent
worktree starts at `3792a86f9f5658642c5e250053a48d77c6425643`.
Only this document is changed. No SDK installation/import/license checkout,
credential inspection/change, new public/commercial qualification solve, cluster operation,
publication, shared-tool edit or push is performed by this review.
Existing offline discovery, public-document provenance and saved-candidate
replay entries were used; original sealed runs were not changed.
The final CPU smoke also executes existing synthetic tests/examples; those
calls are not new public/commercial coverage.

**Commercial readiness: UNVERIFIED. Public-data redistribution: NOT AUTHORIZED
BY THE REVIEWED EVIDENCE. Online publication: UNVERIFIED.**
Failure to find authorization is not authorization, a free-use entitlement,
or permission to publish. This is an evidence/engineering review, not a legal
opinion about all possible agreements held by the operator.

## Findings

### P1: No Actual Same-Problem Closed-Source Baseline

The fresh runner report has three absent SDKs, no imports/license-use
attempts, six planned cold/prepared samples per vendor, and no actual samples.
Gurobi, CPLEX and MOSEK each have **0 actual solve calls, 0 completed solves,
0 independently accepted solves and 0 matched public cases** in the reviewed
real evidence. Public afiro/adlittle experiments use Openopt/OSQP/Clarabel,
not commercial backends. Recording SDK substitutes under `pytest-tmp` are
test fixtures, not proprietary execution evidence.

Source coordinates at review base:
`scripts/benchmark_commercial.py:249`, `:302`, `:360` are the actual native
optimize/solve sites; `:489` sets the solve-attempt flag before the adapter call;
`:559` aggregates that flag. For a future failed adapter, this flag alone is
not a completed-native-call counter. Retain actual native-entry/completion,
exception stage, strict status, original x/y and independent check counts
separately. Current empty sample lists establish zero within this run,
not an assertion that no commercial solver has ever run anywhere.
`scripts/benchmark_commercial.py:663` returning replay acceptance PASS does
not change its qualification UNVERIFIED or authenticate execution/legal rights.

### P1: Source Acceptance Does Not Grant Dataset Or SDK Rights

Caller-pinned HiGHS distribution/model acceptance proves only the reviewed
byte/provenance/model relationships. The source evidence explicitly has
`license_status=benchmark-use-declaration`, `license_verified=false`,
`public_redistribution=false`; observed root MIT text explicitly does not
establish data scope. Both accepted cases prohibit upgrading raw or derived
array redistribution from this review. `PUBLIC_SOURCE_ACCEPTED` is not a
dataset-license verdict or permission to distribute a sealed result bundle.

Commercial-run authorization at `scripts/benchmark_commercial.py:378` checks
only syntax and an operator-supplied digest, not a contract.
`record_is_legal_authorization=false` and `legal_validity=UNVERIFIED` remain
necessary. No actual agreement, eligibility, public benchmark-disclosure
permission or SDK redistribution entitlement was established here.
Project `LICENSE`/0BSD does not relicense datasets, vendor SDKs, manuals,
GPU runtime libraries or their dependencies.

### P2: Reference-Arm Replay Does Not Bind All Experiment Settings

Reviewed external branch `codex/release4-20261010-public-robustness`,
commit `efc1567`, producer
`8ceb0f2b0dd732fccfae1f1c13afdf585ee9aba0a59cbb8038fd4c4874d42e18`.
Coordinates below refer to that branch's
`scripts/experiment_public_robustness.py`, not a file at this review base.

At `:255` and `:284` the reference arms request complete settings, but replay
at `:627` checks only OSQP scaling/eps_abs/eps_rel and Clarabel
equilibrate_enable/tol_feas. Max iterations, no warm starting, presolve,
threads and other requested tolerances are not compared in full.
Original-candidate checks still reproduce the saved numerical verdict, but
they cannot establish that every claimed reference configuration was used.

Minimal owner-side negative test: in an isolated copied fixture, change
OSQP max_iter/warm_starting or Clarabel max_iter/max_threads/presolve_enable
consistently in the report, worker and final-row JSON; refresh those sidecar
pins and supply a new caller result pin. Require experiment-contract rejection
without solving. This review did not create or run coordinated mutations,
and does not allege alteration of the pinned real result.
Fix should compare the complete requested/effective configuration and reject
missing/extra fields, independently of the candidate's mathematical verdict.

### P2: Runtime Identity And Budget Are Insufficient For General Claims

At `:252`/`:280` current OSQP/Clarabel are imported. The new report at `:392`
and source capture at `:412` do not bind Python/NumPy/SciPy/Torch/reference
versions or CPU model/affinity/quota/BLAS/thread settings for the new worker.
A baseline dependency list is not proof of the new worker's runtime.
`torch.set_num_threads(1)` at `:353` does not record all library thread limits.
`observed_device=cpu` plus CPU tensor checks is limited evidence, not remote
execution attestation. Source base commit at `:31` is a constant; frozen
source hashes, not that label alone, identify executed implementation bytes.

At `:429`, workers run in fixed case/arm order under a single remaining
180-second wall budget. A slow early arm can prevent later arms starting;
BUDGET_EXHAUSTED remains in the denominator, but comparisons become
order-dependent. The current real batch finished in 25.3764 seconds without
budget exhaustion, so this is a design risk, not its observed cause of failure.
Before comparative robustness/performance claims, predeclare per-arm resource
budgets, retain preparation/import/solve/check phases and actual call counts,
and use balanced order/repeats. Runtime/version drift and starvation need
negative controls; this review does not modify or launch that experiment.

### P2: Combined Arms And Two LPs Do Not Identify A Cause Or Ranking

At `:34` and `:160`, Openopt changes scaling, objective scaling, solve target,
adaptive rho and refinement; reference arms change scaling and solve targets.
These are combined-arm observations, not an isolated scaling ablation,
presolve benefit, fair latency contest or representative coverage.
Openopt's afiro baseline passed but its new stricter arm exhausted 10000
iterations; do not replace the passing baseline or describe this as universal
regression/improvement. Both new Openopt refinements report
`singular_or_unresolved_kkt`, without an accepted proposal.
Two LPs contain no real public QP/conic/MIP or licensed commercial coverage.
Strict infinite-bound sign, componentwise complementarity, LP dual range and
signed gap must remain unchanged; small wrong-sign duals are not clipped.

### P2: Shared Publication Faces Remain Coordinator-Owned

Live `git status --short` at 10:06-10:07 UTC found both `/data/dev/openopt`
and `/data/dev/openopt-wt-release4-20261010-publication` clean. No active
shared-file collision was observed; cleanliness is not exclusive ownership
or permission to publish. Generated `docs/design*.html`,
`docs/benchmark-report*.html`, report generators, publication plan/source
registry/index and infra `publish-all.sh` chain remain shared/coordinator
write faces. This seat neither stages nor edits them.

`docs/research/publication.md:3` distinguishes PREPARED from PUBLISHED;
the publication tool at `scripts/prepare_publication.py:411` retains online
and commercial UNVERIFIED. Disclosure minimization, clean source or a scalar
projection does not establish rights to redistribute its source data.
This internal review contains external artifact paths and official network
references; it is not a publishable static page under that tool's policy.
Main must separately review a sanitized, committed, rights-cleared plan,
avoid concurrent writes to the shared renderer/index/source registry, and
use the authorized chain and deployed readback. Do not expose raw/derived
public arrays, vendor material, private evidence or credentials by default.

## Actual Evidence And Counts

Evidence root for this review:
`/hf3fs-jd/data/tmp/openopt-release-20261010/commercial-readiness-review/`.

| Vendor | Current environment SDK/version | Actual native solve / completed / accepted | Matched afiro/adlittle | Qualification |
| --- | --- | --- | --- | --- |
| Gurobi | Not discovered / unknown | 0 / 0 / 0 | 0 | UNVERIFIED |
| CPLEX | Not discovered / unknown | 0 / 0 / 0 | 0 | UNVERIFIED |
| MOSEK | Not discovered / unknown | 0 / 0 / 0 | 0 | UNVERIFIED |

Current discovery at `2026-10-10T10:03:51.407775+00:00`:
`discovery/sdk-548d5ftv/result.json`, also cuOpt absent; no SDK imports,
network or license checks. Current-source commercial runner discovery at
`2026-10-10T10:05:04.536415+00:00`:
`runner-discovery/discovery-3bsk_wt3/result.json`, SHA256
`c14f3a58f79a8cf9a827eb0ad8ff7f2fdda0f18d0cf1f4c465d0060443be71fe`,
source commit `3792a86`, planned=18, attempted=solve_attempted=passed=failed=0.
Its read-only `--check-result` returned acceptance PASS,
qualification/verdict UNVERIFIED, with identical counts. No execution flag
or authorization digest was supplied. This is a synthetic QP discovery,
not a solved commercial baseline on either public LP.

Independent real robustness replay at
`2026-10-10T10:05:45.950549+00:00`:
`robustness-replay.json`, SHA256
`42d27fa102047103abc232c234f4cf439745934d3061be80d6fbb132bb791dd9`.
Original pinned result:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/real-v1/result.json`,
SHA256 `aa584b62f97349f37afad438dfbd87320ace3408df4d7e574a0989370970e8e7`.
Replay rechecked baseline 6/6 and new 6/6 without solving. New attempts=6,
QUALIFIED=1, UNQUALIFIED=5, infrastructure FAIL/budget exhaustion=0.
Openopt=0/2, OSQP=0/2, Clarabel=1/2. Overall/original UNQUALIFIED remains;
EXPERIMENT_REPLAYED_NOT_QUALIFICATION and exit zero mean replay completed.

Public baseline pin:
`b10d49a32a13a9f34b8bf86ea6bd8a126dc779bf72a47db3f069420f653456ef`;
corpus pin `3a434edfa8165167f56be25732840f03bbace1a5f5b7d1de7cdab989673d3f0f`.
Rights/source acceptance:
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-source-acceptance/real-v3/acceptance.json`,
pin `1f82c8df6e9258bd8598884dd2bfee77f9e97f2ce7ec9f46035c5604822519fa`;
adjacent `status.json` and
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-acquisition/source-v3/source.json`
explicitly preserve unverified license/nonredistribution. No rights upgrade.

## Current Primary SDK Sources

Official pages were newly read in memory, not vendored. URL/date/hash/bytes
receipts only are external under `sources/source-*/result.json`. These facts
identify publicly documented versions/eligibility boundaries, not an installed
build, contract or executed baseline:

| Source | Newly observed fact | Receipt directory |
| --- | --- | --- |
| https://docs.gurobi.com/projects/optimizer/en/current/reference/releasenotes/changes.html | Current 13.0.3 notes visible; GPU PDHG fully supported since 13.0.2; final uncrush can downgrade optimal status. Not universal GPU QP/MIP execution. | source-629jbe0_ |
| https://docs.gurobi.com/projects/optimizer/en/current/reference/releasenotes/platforms.html | GPU PDHG linux64/armlinux64, NVIDIA; CUDA 12 compute 7.0-12.0, CUDA 13 compute 7.5-12.0; driver compatibility required. Python 3.10 appears supported. No card/build execution verified here. | source-owp2me00 |
| https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html | PDHGGPU requires separate GPU build. Current adapter forces CPU barrier Method=2, not this GPU path. | source-uogvh6it |
| https://support.gurobi.com/hc/en-us/articles/360051597492-How-do-I-resolve-a-Model-too-large-for-size-limited-Gurobi-license-error | Restricted package license limits 2000 variables/linear constraints, 200 variables with quadratic terms; non-production restriction. Not evidence of this operator's full entitlement. | source-12buyfo6 |
| https://raw.githubusercontent.com/IBMDecisionOptimization/docplex/master/docplex/mp/params/cplex_params.py | IBM-maintained current parameter source recognizes/defaults to 22.2.0. This is not proof of a latest maintenance SDK build. | source-14ixcfgv |
| https://www.ibm.com/products/ilog-cplex-optimization-studio | Python API and Community Edition limits of 1000 variables/1000 constraints documented. No purchased/community entitlement checked. | source-d_by25tr |
| https://www.ibm.com/docs/en/icos/22.2.0 | HTTP 403 on fresh read; latest patch/support matrix and full terms not verified from this page. No automatic retry or capability inference. | source-l2biyjo_ |
| https://www.mosek.com/downloads/ | Stable 11.2.6, October 8, 2026; version 12 listed as beta, not stable/latest qualification. | source-46wjhwol |
| https://docs.mosek.com/latest/pythonapi/intro_info.html | Python optimizer API 11.2.6 observed; linear/conic capabilities are documented, not actual execution here. | source-29ndz5sk |
| https://www.mosek.com/products/academic-licenses/ | Academic use limited to research/education at degree-granting institutions, specified eligibility/period; not unrestricted commercial product rights. | source-ed907bcb |
| https://www.mosek.com/products/license-agreement/ | Issued licenses governed by MOSEK agreement unless otherwise agreed. Linked full agreement not fetched/reviewed; redistribution/disclosure entitlements remain UNVERIFIED. | source-n6w_4qhr |

Neither an academic/community/trial offering nor a public API/source page
establishes free production use or benchmark disclosure/rebundling permission.
No vendor code/manual payload is placed in Git. No unsupported conclusion
that CPLEX/MOSEK universally lack GPUs is drawn from these limited reads.

## Proof-Fix Handoff And Next Gate

Prior implementation commit `275cef7fa554ade582067dc2e9aef59261cf74de`
was integrated as `3792a86`; coordinator dispatch `242e0a1` is separate.
Final producer unit evidence:
`/hf3fs-jd/data/tmp/openopt-release-20261010/proof-fix/unit-v3/tests-omipz6_6/verification.json`,
68 passed, zero failures/errors/skips, source consistency PASS, finished
`2026-10-10T09:58:54.209944+00:00`, SHA256
`643b323e58fb267bf5089e4938a52a6dde284ea65a6dbe1544166751068729b3`.
Nine actual CPU CLI run/seal/replay cases:
`/hf3fs-jd/data/tmp/openopt-release-20261010/proof-fix/cli-replay/replay-rsq566wr/acceptance.json`,
SHA256 `dbe2f3724b14ec16e1acf1ff370c3db293bef1720c216f42f01004957ec804c6`,
numerical PASS, execution UNVERIFIED, CPU_SMOKE_REPLAY_PASS.
Raw hashes and terminal fields were re-read in this review. These tests
predate integration and do not certify current main CUDA execution.

Do not use this review to open the GPU release gate. Main owns current clean
source, expectations pin, coordinator/external run audit, CUDA original-model
candidate/certificate replay and formal execution verdict. A bare numerical
seal is not GPU-verified, a commercial comparison or publication qualification.

Read-only reproduction (existing interpreter, no install):

```text
scripts/benchmark_commercial.py --check-result <review-root>/runner-discovery/discovery-3bsk_wt3/result.json
```

Run that in this review worktree with infra identity and CPU-only environment.
For the independently pinned public replay use the existing receipt entry in
the public-robustness worktree:

```text
tests/test_experiment_public_robustness.py --check-result /hf3fs-jd/data/tmp/openopt-release-20261010/public-robustness/real-v1/result.json --sha256 aa584b62f97349f37afad438dfbd87320ace3408df4d7e574a0989370970e8e7 --output <new-external-receipt-path>
```

Closure requires actual externally approved same-model commercial native
calls/completions and original-coordinate checks; complete experiment
settings/runtime/budget evidence; representative held-out LP/QP/conic/MIP
strata with preserved failure denominators; explicit dataset/SDK/disclosure
rights review; and separately authorized, single-owner publication/readback.
Missing items stay UNVERIFIED. No new solver, full regression, CUDA, mutation
fixture, legal authorization or publication test was run by this document seat.

Final document-seat CPU smoke finished
`2026-10-10T10:09:17.726322+00:00`:
`/hf3fs-jd/data/tmp/openopt-release-20261010/commercial-readiness-review/smoke/smoke-yxkac8ng/verification.json`,
SHA256 `a0b9c8656bb48e273417395ab0b5ce3faf86bf8a65b0bffb973d2ce69e224803`.
JUnit total=84, passed=80, failed/errors=0, explicit CUDA skips=4;
snapshot/provenance, CPU example and source consistency PASS. Subsequent
edits only clarify this document and record that receipt. Reviewed code/tests
remain identical to base `3792a86`. Document credential-pattern scan returned
no matches; final path-scoped diff/commit must contain only this review.
