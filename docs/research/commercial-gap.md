# Commercial Capability And Prototype Gap

Research cut: **2026-10-10**, public primary sources read between
07:55 and 08:01 UTC. Prototype reviewed at `fcefac1`:
`src/openopt/qp.py`, `docs/qp-design.md`, `README.md`, `docs/BACKLOG.md`,
and `pyproject.toml`. This is capability research and SDK discovery, **not a
licensed commercial benchmark**. Main owns integration and the unified report.

## Conclusion

GPU is not an exclusive differentiator. Gurobi GPU PDHG became **fully
supported in 13.0.2**, not merely beta; the current release-notes page already
contains **13.0.3** [G1]. NVIDIA cuOpt is an open-source GPU optimization
engine, with LP and QP paths and separately qualified beta scopes [N1-N4].
Neither fact proves that every algorithm or problem class runs on GPU.

No evidence supports promising that openopt will universally defeat Gurobi,
CPLEX or MOSEK. Problem classes, certificates, accuracy, robustness, throughput,
tail latency, warm-start behavior and licensing form distinct comparison axes,
not a single total ordering. Our current dense convex QP/LP prototype has large
functional and qualification gaps; even matched commercial continuous runs
remain **UNVERIFIED**.

## Current Capabilities

| Solver / version evidence | Continuous scope | Discrete scope | GPU boundary |
| --- | --- | --- | --- |
| Gurobi 13.0 family; 13.0.3 notes currently visible [G1] | LP, QP/QCP, convex and nonconvex quadratic constraints; SOCP/rotated SOCP expressible as quadratic constraints; nonlinear optimization is a separate scope [G2] | MILP, MIQP/MIQCP; globally valid bounds for nonconvex quadratic models, subject to numerical tolerances [G2-G3] | PDHG solves **LPs**; `Method=6` can also be selected in MIP, where this is a linear-relaxation algorithm. `PDHGGPU=1` requests GPU if available. A **separate GPU build** is required. GPU support fully supported since 13.0.2. This is not a claim of GPU QP, GPU branch-and-bound or all-GPU MIP [G1,G3]. |
| IBM CPLEX: official DOcplex registers **22.2.0** and defaults to its parameter hierarchy [I2-I3] | LP, QP and quadratic/SOCP modeling; simplex, barrier, presolve and related methods [I1,I3] | MIP, MIQP/MIQCP, branch-and-bound and cuts [I1,I3] | No supported GPU backend was confirmed by the inspected IBM product page and complete official 22.2.0 parameter hierarchy. This limited search is **not proof that every IBM release/product has no GPU support**. GPU capability remains unconfirmed here. |
| MOSEK **11.2.6**, official page updated 2026-10-08 [M1] | LP, convex QP/QCQP, SOCP, exponential/power cones, continuous SDP; simplex and interior-point methods [M1-M2] | MILP, mixed-integer quadratic and conic models; **mixed-integer SDP excluded**. Certain nonconvex MI(QC)QO instances can be reformulated into equivalent convex instances; others are rejected, not general nonconvex support [M3] | No supported GPU optimizer was confirmed by the inspected official optimizer/API pages. Multi-core parallelism and batch optimization are documented, not evidence of GPU execution [M2-M4]. |
| NVIDIA cuOpt **26.08** documented release; official latest-release API says `v26.08.00`, published **2026-08-06** [N1,N5] | GPU PDLP for LP; GPU barrier for LP/QP via cuDSS/cuSparse; concurrent mode races GPU PDLP, GPU barrier and CPU dual simplex. QCQP/SOCP are beta [N1-N2] | MIP is **beta**: GPU primal heuristics plus traditional CPU algorithms including branch-and-bound; optimality proof remains under development [N1,N3] | Actual open-source GPU competitor, not just routing. Documented two-GPU mode is restricted to LP concurrent mode, running algorithms on separate GPUs, not distributed strong scaling of one algorithm [N2]. |

The cuOpt `main` README displays 26.10 development metadata, while the
official latest release and current documentation identify 26.08. Do not
equate `main`, nightly packages and released qualification. Python package
metadata also cannot prove that a Gurobi GPU build, supported device or license
is available [G1,N5].

## Precision And Status Semantics

**Gurobi:** numerical optimality is not exact rational optimality.
`FeasibilityTol` and `OptimalityTol` default to `1e-6`, and `IntFeasTol`
to `1e-5`; these feasibility/integrality tests are absolute, making input
scaling relevant [G4]. PDHG has separate feasibility and convergence
criteria: primal/dual residuals satisfy the absolute criterion **or** the
relative criterion, in addition to convergence. Set `PDHGRelTol=0` when
requiring absolute-only feasibility. Record `PDHGAbsTol`, `PDHGRelTol` and
`PDHGConvTol`, not just generic simplex tolerances. MIP termination uses
incumbent-versus-bound gaps; an optimal status is tolerance-qualified [G3].
13.0.3 explicitly adds a warning when PDHG uncrush changes OPTIMAL to
SUBOPTIMAL: validate the final original-model status, not an intermediate one
[G1].

**CPLEX:** official 22.2.0 parameter definitions give simplex feasibility
and reduced-cost optimality defaults `1e-6`, integrality `1e-5`, relative
MIP gap `1e-4`, absolute MIP gap `1e-6`. The hierarchy separates barrier,
simplex, MIP and solution-target parameters [I3]. Equal-looking numbers do
not make their norm/scaling/gap rules interchangeable. IBM's dynamic docs
returned HTTP 403 for the 22.2.0 landing page in this research environment;
the authoritative IBM-maintained parameter source was read instead.
Do not infer exact arithmetic, GPU precision or a licensed run from it.

**MOSEK:** the homogeneous primal-dual formulation can return numerical
infeasibility evidence as well as solutions. Official termination rules
explicitly say exact optimal solutions/certificates are not computed:
feasibility is norm-scaled and primal-dual gap is part of acceptance [M2].
MIP relative gap is
`abs(incumbent - bound) / max(mio_rel_gap_const, abs(incumbent))`,
with defaults `mio_rel_gap_const=1e-10`, relative gap `1e-4`, absolute
gap `0`; either gap threshold can terminate. Integer feasibility has its own
tolerance [M3]. A numerical certificate is still independently checkable,
but must not be represented as an exact-arithmetic proof.

**cuOpt:** documented PDLP default target is `1e-4` relative accuracy.
Native precision is FP64 for double problems; `single` uses FP32 internally,
while `mixed` uses an FP32 constraint matrix for iteration SpMV and FP64
vectors/computation, checking convergence/restarts against the full FP64
matrix. Crossover can improve LP basic solutions, but is **not supported
with quadratic objectives or constraints** [N2]. Do not compare low-accuracy
PDLP iteration latency to tighter validated QP solves, or an early MIP
heuristic incumbent to an optimality-gap-qualified MIP result.

**openopt at `fcefac1`:** `QPConfig` defaults to float64 and
`eps_abs=eps_rel=1e-7`; float32 is also accepted. `solved` means primal,
stationarity and normal-cone residual gates passed under the core's
absolute-plus-relative rules. It does **not** independently bound objective
gap or certify infeasibility/unboundedness. `max_iterations` is not a proof
of either. Model tensor dtype is not by itself an audit of every kernel's
internal arithmetic. All comparisons need independent checks in the original
model units, not the solver's status string alone.

PDLP is not a novel exclusive GPU idea: the original Applegate et al. paper
(NeurIPS 2021; arXiv v1 2021-06-09, v2 2022-01-07) describes LP PDHG
with matrix-vector operations, preconditioning, presolve, adaptive steps and
restarts [P1]. Its instance-scoped results do not transfer to our ADMM
prototype or establish a universal ordering.

## Actual Prototype Gaps

| Area | What exists at the reviewed baseline | Missing evidence or functionality |
| --- | --- | --- |
| General continuous core | Real dense PSD `P`, linear two-sided/equality constraints, zero `P` for experimental LP; ADMM plus Cholesky/triangular factor application | Sparse/matrix-free operators, large sparse corpora, presolve, equilibration/scaling, robust degeneracy/ill-conditioning handling and matched licensed LP/QP qualification |
| Setup and reuse | Full eigenvalue convexity validation; dense `P + rho A.T A + sigma I` factor; owned prepared tensors | Cubic cold setup and quadratic storage remain; each prepared solve starts at zero. No changing-data warm-start contract; legacy portfolio batching is not a general batched-QP API |
| Certificates | Residual-qualified approximate solution; numerical-error/max-iterations status | Independently checked primal/dual infeasibility rays, unbounded directions, reliable ambiguous/unknown termination, objective-gap qualification |
| Cones and nonconvex models | No conic/nonconvex general backend | SOCP/exponential/power/SDP model contracts, cone feasibility/duality and appropriate references; PSD rejection is not global nonconvex optimization |
| MIP | No integrality model or solver | MILP/MIQP schema, integer feasibility, incumbent and valid bound, presolve/cuts/branch-and-bound, absolute/relative gaps and reproducible termination |
| GPU performance | Explicit CPU/CUDA and prepared workspace; prior synthetic dense GPU results scoped against our CPU prototype | Commercial/open-source paired end-to-end latency, transfers/setup/output included, diverse instances, failure denominators, tail bounds and actual multi-GPU algorithm scaling |
| Operations and distribution | 0BSD project and migrated portfolio compatibility | Licensed baseline authorization, repeatable commercial runs, supported-hardware/runtime qualification; dependency/license inventory remains separate |

The task does not modify algorithms, parameters, benchmark runners or
generated reports. No numerical/performance improvement is claimed by adding
this research or discovery tool.

## Fair Benchmark Gate

This is a proposed commercial qualification contract, not a change to the
existing benchmark implementation. Main must integrate it with the design
and generated report before claiming qualification.

1. Freeze original problem arrays/files, hashes, seeds and per-class strata:
   dense/sparse LP, convex QP, each cone class, and MILP/MIQP separately.
   Include infeasible, unbounded, degenerate, ill-conditioned, empty/invalid
   and nonfinite controls. Unsupported categories remain unsupported, not
   filtered into an apparently universal win.
2. Verify licensed authorization separately from installation. Freeze exact
   solver/build/version, hardware, dtype and effective parameters; include
   Gurobi CPU/GPU builds and cuOpt released CPU/GPU methods where authorized.
   Compare original-model accuracy targets, not identical parameter names.
3. Independent original-model feasibility, stationarity/duality and objective
   gap decide continuous success. MIP additionally requires integrality,
   incumbent, valid dual bound and absolute/relative optimality gap. Reject
   false certificates and feasible-only heuristics mislabeled optimal.
4. Report successful **and attempted** counts, failures, unsupported and
   UNVERIFIED cases. Publish p50/p95/p99 or tail distribution with sufficient
   repetitions and uncertainty, time-to-target, objective/gap distributions,
   memory and throughput. Do not select only jointly solved instances to
   conceal robustness losses; show coverage and censored time limits.
5. Pair identical instances on matched resources, alternate order, retain
   every failure and fix the corpus/tuning split in advance. Separate process
   startup, model-cold, prepared and changing-data warm starts. Count transfers,
   synchronization, validation, factorization and host results in end-to-end
   claims. Include controls where small GPU jobs or setup-heavy workloads lose.
6. Publish per-class performance/success profiles or Pareto comparisons, not
   a universal rank. Commercial parity requires the actual licensed cases;
   missing SDK/license execution is **UNVERIFIED**, never a default win.

## Safe SDK Probe And Local Evidence

`scripts/commercial_probe.py` uses top-level `importlib.util.find_spec` and
`importlib.metadata.version` only; it does not import SDKs, construct solver
environments/models, inspect credential variables/files or open a license
connection. Module locations, license values/paths and host identity are not
reported. `AVAILABLE` means module discovery plus version metadata only:
every `licensed_run` and overall qualification stays **UNVERIFIED**.
Metadata without a module, or a module without metadata, does not qualify.

Missing distributions are normal UNVERIFIED observations; unexpected
discovery errors produce `FAIL`, stage and exception class with rc=1.
Raw exception messages are redacted because they can contain private paths,
hosts or values; HTTP status/OS errno is retained when available. Empty,
duplicate and unknown selections are rejected. The process's normal import
finders must be trusted; this tool is not a sandbox against malicious finders.

Run in an existing CPU environment; no installation is necessary:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/commercial_probe.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-commercial-20261010/actual-cpu
sudo -n -u infra env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B \
  scripts/verify.py tests --target tests/test_commercial_probe.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-commercial-20261010/unit
```

Each run creates a distinct external directory, persists `result.json` plus
`execution.log` and rereads the result before completion. Exit zero for
UNVERIFIED means collection finished, **not commercial qualification**.

Actual CPU discovery at **2026-10-10T07:57:53 UTC**:
`gurobipy`, `cplex`, `mosek`, `cuopt` were all absent; versions were empty;
overall verdict **UNVERIFIED**. Artifact:
`/hf3fs-jd/data/tmp/openopt-commercial-20261010/actual-cpu/sdk-ep3xdul6/result.json`.
This initial artifact has its own source hash; subsequent edits require a
fresh run, not replacing that evidence.

Preflight: 79 passed / 4 CUDA skipped, CPU example and provenance passed.
Artifact: `preflight/smoke-150f5us6/verification.json` under the same task root.
Unit-v2: **25 passed**, zero skipped/failures/errors, with JUnit counts and
CPU example, at `unit-v2/tests-jbbqc0su/verification.json`.
Negative controls include missing SDK, installed-without-license, missing
metadata, discovery exceptions, invalid versions, empty/duplicate selection,
poisoned SDK `__init__` (must not execute), forbidden sockets/SDK imports,
private environment sentinels, in-tree outputs and disallowed redirects.
Final post-edit acceptance at **2026-10-10T08:02 UTC**, task-root-relative:

| Check | Actual outcome | External artifact |
| --- | --- | --- |
| CPU SDK probe | COMPLETE / UNVERIFIED; all four SDKs absent; no license/network attempt | `final-cpu/sdk-zjajtjvi/result.json` |
| Probe tests | 25 passed; no skips/failures/errors; provenance and CPU example passed | `final-unit/tests-1salftg3/verification.json` and `pytest.xml` |
| Project CPU smoke | 79 passed / 4 explicitly skipped CUDA tests; provenance and CPU example passed | `final-smoke/smoke-bny51wwc/verification.json` and `pytest.xml` |

Final probe source SHA-256:
`dc7bae1a5456dd12788063cc1d23268d0a668c91218270294dde977089260132`.
CUDA/GPU execution and licensed commercial solves were **not run** in this
seat. These CPU tests do not qualify performance or commercial parity.

The optional `--source-url` mode is **explicit public-network research**,
separate from offline SDK discovery; never call it a network-free probe.
It displays bounded inspection text and writes URL/date/hash/size provenance
only, **not downloaded commercial reference documents**. HTTPS hosts and
redirects are checked; no credentials are requested. Earlier HTTP failures
remain in `sources/source-*/result.json`. The 22.2.0 IBM landing-page failure
is `sources/source-una5xvql/result.json` (HTTP 403). Those are research-access
failures, not evidence against solver capabilities.

## Licensing Boundary

The project's `LICENSE`/package metadata are 0BSD; that covers this project's
code, not PyTorch/NumPy, CUDA components, proprietary SDKs, service terms or
cuOpt. cuOpt's primary license is Apache-2.0 [N4], with its own dependency
terms. No vendor code/docs are vendored or relicensed here. SDK installation
is not license authorization; an authorized installation is not an executed
licensed benchmark. No credentials, installations, GPU submissions, service
changes or infra edits are included in this task.

## Primary Sources

Every URL below was read on **2026-10-10**; dates are access dates unless
explicitly labeled as release/page-update/paper dates. Summaries are
paraphrases, not reproduced manual passages.

| ID | Primary URL / provenance |
| --- | --- |
| G1 | https://docs.gurobi.com/projects/optimizer/en/current/reference/releasenotes/changes.html ; page updated 2026-10-09; read hash `95ad27fa8fd88ec2f1dd077477c2c232a3feb5cf658d2196ccd5b8f9d0911816` |
| G2 | https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/constraints.html |
| G3 | https://docs.gurobi.com/projects/optimizer/en/current/reference/parameters.html ; read hash `09226fe902d92d79e4f1309cf01bbcbf54cb9f5fa4972a49ea03df2e95378749` |
| G4 | https://docs.gurobi.com/projects/optimizer/en/current/concepts/numericguide/tolerances_scaling.html |
| I1 | https://www.ibm.com/products/ilog-cplex-optimization-studio |
| I2 | https://raw.githubusercontent.com/IBMDecisionOptimization/docplex/master/docplex/mp/params/cplex_params.py ; maps 22.2.0 and defaults to that hierarchy |
| I3 | https://raw.githubusercontent.com/IBMDecisionOptimization/docplex/master/docplex/mp/params/parameter_hierarchy_22200.py ; read hash `1d63fae1683edb75e10b0b8278e51b39ab426a4df00b11a8b4b5e9c4404d80d8` |
| M1 | https://docs.mosek.com/latest/pythonapi/intro_info.html ; 11.2.6, page updated 2026-10-08 |
| M2 | https://docs.mosek.com/latest/pythonapi/solving-linear.html and https://docs.mosek.com/latest/pythonapi/cont-optimizers.html |
| M3 | https://docs.mosek.com/latest/pythonapi/mip-optimizer.html ; read hash `445f755f85ac193e5f584862d8ddc9bc6cc36a0693b201521bb2eecd20c372ac` |
| M4 | https://docs.mosek.com/latest/pythonapi/optimizer-env.html |
| N1 | https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html |
| N2 | https://docs.nvidia.com/cuopt/user-guide/latest/convex-features.html ; read hash `200e11608ab355bb5ea2856fac1822a11f9758f5e006fa2e8933eaf46f3c68b6` |
| N3 | https://docs.nvidia.com/cuopt/user-guide/latest/milp-features.html |
| N4 | https://raw.githubusercontent.com/NVIDIA/cuopt/main/LICENSE |
| N5 | https://api.github.com/repos/NVIDIA/cuopt/releases/latest ; `v26.08.00` published 2026-08-06; distinguish https://raw.githubusercontent.com/NVIDIA/cuopt/main/README.md |
| P1 | https://arxiv.org/abs/2106.04756 ; Applegate et al., Practical Large-Scale Linear Programming using Primal-Dual Hybrid Gradient, NeurIPS 2021, arXiv v2 revised 2022-01-07 |
