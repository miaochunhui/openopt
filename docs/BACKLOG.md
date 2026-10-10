# openopt Development Ledger

## Established Foundation

- Extract the complete OpenPO implementation without modifying source history.
- Preserve the `openpo` public API and provide an `openopt` compatibility entry.
- Provide independent correctness verification and explicit migration evidence.

These items are accepted only by current `scripts/verify.py` output, not by the
presence of files or old benchmark claims.

## Next Development Stages

The user selected 0BSD on 2026-10-10. Licensing is implemented in `LICENSE`,
package metadata and wheel acceptance; it does not relicense dependencies.
The product target and release gates are in `strategy.html`.

1. General model: an experimental real dense convex QP/LP contract is now
   implemented in `src/openopt/qp.py`, with explicit dtype/device, approximate
   result status and independent KKT validation. Extend the quadratic
   operator interface, constraints, dtype/device semantics and result status.
   Keep portfolio adapters outside the general core; design GPU-resident
   dense, sparse and matrix-free operators, batching and prepared workspaces.
2. Correctness and robustness: compare matched LP/QP with licensed Gurobi,
   CPLEX and MOSEK plus OSQP/Clarabel/HiGHS references. Cover infeasible,
   unbounded, degenerate, ill-conditioned, empty and non-finite inputs.
   Missing commercial runs remain UNVERIFIED, not passed.
3. GPU qualification: measure transfer-inclusive cold/hot latency, p50/p95,
   success rate, objective gap, primal/dual residuals, throughput, memory and
   hardware/software versions on identical paired cases. Keep explicit GPU
   requests strict and choose automatic CPU/GPU routing from measurements.
4. Convex conic scope: zero/nonnegative/SOC contracts, the explicit Clarabel
   CPU reference and an opt-in dense native PyTorch proximal ADMM prototype
   are implemented. RSOC original-coordinate checks and native projection
   extend that prototype, with extreme-coordinate regression and strict
   device semantics; fresh clean-source independent/CUDA checks are pending.
   Actual CUDA acceptance, HSD/directions, further cones and
   representative MOSEK/open-reference qualification remain required.
5. MILP/MIQP scope: MILP bounds/integrality and the explicit HiGHS CPU reference
   are implemented with independent incumbent checks. Native GPU search,
   presolve/cuts and MIQP remain required. Compare licensed Gurobi/CPLEX and
   SCIP/HiGHS. A fast heuristic is not an exact MIP solver.
6. Distribution: clean-wheel acceptance and CPU/CUDA CI, followed by separate
   reproducible commercial/GPU qualification. No license server or SDK is
   bundled; commercial benchmark execution requires authorized installations.

## Current Research Extensions

Optional residual-balanced rho, per-item dense batching, sparse/matrix-free CG,
independent candidate/direction evidence checks and a reproducible synthetic
stress corpus are implemented. Independent sealed evidence replay checks
statuses, all host solutions, parameters, source membership and negative
controls. These extensions do not close commercial qualification.

Bounded reversible Ruiz scaling, Jacobi PCG, opt-in independently checked gap
and generated checkpoint directions are implemented. The restricted audited
MPS/QPS importer and authorized commercial LP/QP replay adapters are implemented.
Real public-model coverage, full GPU proof sidecars and legal commercial runs
remain separate qualifications. The release contract is `release-contract.md`.
Opt-in original-coordinate active-set polishing and finite-precision KKT
residual correction are implemented; this is not arbitrary precision or a
complete active-set algorithm. Complete original checks and nonworsening
baseline accuracy are required. Frozen numerical proof producers and bounded
one-case-per-Job dispatch are implemented; current-source GPU execution and
independent cluster/source/sidecar binding remain separate acceptance gates.
Conservative exact zero-row/+1 identity-equality presolve is implemented and
publicly exported, disabled by default. Sealed maps, offset and signed dual
postsolve enforce complete original-model acceptance. Unsupported or
nonrepresentable fixed substitutions remain unchanged. Installed-wheel probes
exercise a genuine reduction and original-coordinate solve; full commercial
presolve, representative benefit and same-source CUDA qualification remain
unverified.

Historical evidence is linked by the generated v3 report. The earlier v2 experiment
is historical, byte-frozen evidence; its original source is not replaced.
The actual native subagent limit was six; all eight scoped seats were executed
with completion/refill, never claimed as eight simultaneous agents.

## Explicit Non-Claims

- No universal `minimize`, complete native GPU conic/MIP engine, MIQP or exact
  certificate is implemented. The limited native dense conic prototype,
  explicit CPU references and numerical checkpoint directions must not be
  described as those missing capabilities.
- No universal GPU speedup or production qualification is asserted.
- No commercial solver parity or world-leading ranking is asserted.
- The user authorized this experiment's K8s Jobs with at most three 4500 cards.
  No service, credential, RBAC, networking or shared publication changes were made.
- Historical research decisions are not current runtime validation.

## Current Experiment

The earlier 4500 experiment and failures are recorded externally under
`/hf3fs-jd/data/tmp/openopt-4500/`. `scripts/render_qp_campaign.py` generates the
current v3 design and benchmark pages; `scripts/render_qp_report.py` preserves
the historical reports from their frozen evidence.
Model-cold and prepared timings are separate. Comparisons to the project's CPU
prototype do not establish performance against OSQP/Clarabel/commercial solvers.
Three independent one-card Jobs do not establish a multi-GPU solver.

Remaining qualification: licensed commercial matched runs, representative
diverse/ill-conditioned/sparse corpora, certificates, actual multi-GPU scaling,
hardware lower-bound analysis and publication through an authorized project
entry. These are not promoted to DONE.

## Release4 Current Stop Gates

Current source/evidence and owners are in `handoff/release4-20261011.md`.
Loaded Product callable admission has two independently reproduced false accepts;
four exponential smooth positives remain independently rejected. Both are in
repair, not accepted. HSD tiny CPU epigraph owner37/independent53 and report
owner41/independent41 passed actual chair readback, with their bounded scopes
retained. CUDA harness candidate CPU29 passes after a retained missing-dependency
failure, but its caller-source review and GPU admission remain pending.
Outgoing main scan has two shape findings undergoing independent disposition;
no new push/publication. Actual commercial SDK/solve/license counts remain zero,
and the latest live snapshot retains three unattributed historical failures.
General capabilities, licensed comparisons, whole-source regression, multi-GPU
scaling and coordinated online publication remain incomplete. Goal stays active.

The following 2026-10-10 snapshots are historical:

The newest operational source/evidence is the 23:35 +08:00 section of
`handoff/release4-20261010.md`; older bullets below are historical snapshots.
Main0bd236e offline gate45 passed with zero skips; real known-evidence check
remains INCOMPLETE/29 missing/release_complete=false. PSD independent128
clean replay passes for bounded projection only, not CUDA/general SDP.
Git repaired-owner unchanged35 replay is 20 pass/15 fail due to two new
read-only graph queries refused by the observer. Narrow observation repair
and original-content parity are required; no Git signoff or new push.
Current live FAIL has the new COO failed Pod unattributed, despite four Ready
and zero active/cordoned; precise diagnosis/recovery binding is still required.
Real current operator CUDA53 failed: 44 pass/nine COO-device failures,
zero skip, exit one and no restart. Preserve original Placement/thresholds;
device-local structural repair, independent review and new one-card execution
are required before expansion. Original13 sidecar was corrupted by an audit
fixture; old47 acceptance is withdrawn and new57 admission is BLOCKED.
No restoration, repin or retroactive acceptance has occurred. The separate
permission incident also remains open.

Git owner repair is clean eb3b23a, following seven original independent failures.
Latest owner21 and smoke80/four local CUDA skips pass, but unchanged35 independent
clean review and full current outgoing-object scan are pending. No new push.
Offline release-insufficiency independent55 replay passed with
release_complete=false; it cannot qualify absent capability or commercial data.
Separate PSD/exp/power primitives and generated Openopt project catalog are
under isolated review, not whole solvers or online/public acceptance.

The following 22:55 evidence is historical, not a current all-source signoff:
Clean main73613fc passed same-source smoke80/four local CUDA skips,
report148/zero skips, renderer independent38/31 negatives/zero false accepts,
HSD author110/one local CUDA skip plus original96/zero skips, bounded reader
author126 plus independent144/zero skips, and installed wheel62 modules/0BSD.
Portable sparse original52/zero skips and no-solve replay passed; its harness
repair was subsequently integrated as `4ec39bf`, not retroactively substituted
into those source coordinates. A permission-helper argument-order incident
remains separately unresolved; no numerical PASS closes it.

Pending development is real fixed13 schema2 producer independent review,
strict sparse CUDA producer/independent review and cluster execution,
PSD/exponential cone primitives, followed by actual broader solver integration.
The new primitives alone will not close general SDP/exponential solver scope.
Commercial discovery remains SDK0/4 and zero licensed solves as of 22:44;
applicable runtime/licenses/disclosure rights remain external prerequisites.
Research Git synchronization and shared docs-site onboarding are separate:
Gitee is reachable at old master8def56e; no new push occurred. Shared registry
incremental merge/access scope and managed publication/readback remain pending.
Current-source whole regression, true multi-GPU scaling and general
commercial/universal qualification remain unfinished. Goal stays active.

Current detailed handoff: `handoff/release4-20261010.md`. The v4 renderer is
implemented and independently reviewed, but its current external report is
an intermediate historical-source projection, not final release acceptance.

- Accepted baseline `c0565b0`: separate 76 real RTX PRO 4500 regressions and
  independent original-coordinate replay; unchanged 28-node independent
  suite passes after retained nine false accepts were repaired. Main smoke
  is 80 pass/four local-CUDA skips. Fixed thirteen has actual execution but
  still needs Nash's dedicated independent numerical auditor.
- Accepted distribution: clean producer and independent 22-node review
  pass, with genuine standard installed CLI and original LP/QP checks.
  Main rebuilt clean wheel passes source membership/bytes/0BSD and numerical
  probes. This is CPU distribution acceptance, not all-model/GPU/commercial
  qualification. Original failed wheel and reviewer oracle failure remain.
- Regression coordinator: cluster fixture-v2 transport is implemented but
  archived execution had four independently reproduced false accepts on
  `ba675eb` and `9c46c8e`. Mendel repaired production at clean `8beb931`;
  Helmholtz is replaying Nash's unchanged negatives. Until independent green,
  do not admit related cluster full regression. Shared-component cost and
  current-source whole-regression remain separate unfinished gates.
- Public qualification owner Darwin: prior `a668d72` v3 preview has zero
  solves. Actual executable v3 settings/runtime/thread/source contract and
  bounded synthetic real solves are now in development, with Herschel's
  separate independent review. Real public robustness and redistribution
  restrictions are not closed by synthetic contract tests.
- Native bounded MIQP owner Bacon: new independent three-file implementation
  from `acc1993`, small finite boxes and exact PSD sufficient subset,
  disclosed host verification and strict device semantics. It remains
  unimplemented/unaccepted until code, original bounds, mixed-leaf coverage,
  negative controls and independent replay exist. No full general MIP claim.
- Commercial owner: zero real licensed commercial solves are recorded.
  Authorized actual SDK/license runtime and matched problem results are
  required; fixtures and open-source baselines cannot close this gate.
- Publication owner: shared infra index/generator/source files contain
  existing changes from other work. No collision merge, online publication
  or remote push was performed. Approved integration plus genuine readback
  and exact published bytes are required to close publication.
- Core owners: full sparse/presolve/HSD/SDP/exponential/power cones,
  native MIP/MIQP, nonlinear/global scope, representative robustness,
  transfer-inclusive commercial performance and real multi-GPU scaling
  remain incomplete. The 0BSD project license does not confer those abilities.
