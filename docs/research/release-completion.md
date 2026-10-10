# Release Insufficiency Precheck

This opt-in offline tool implements only a necessary-evidence precheck for
`docs/release-contract.md` and `docs/strategy.html`, not an optimizer, complete
capability test, legal opinion, release authorization or remote attestation.
It does not change existing tools/exports, solve, import an SDK, execute input
code, submit GPU work, publish, download or inspect credentials.

## Terminal States

`INCOMPLETE` (exit 1) lists every insufficient section inspected, retaining
safe first-cause codes. Even fully consistent inputs yield `REVIEW_REQUIRED`
(exit 2), not PASS; `release_complete=false`, execution attestation, legal
authority and general capability stay UNVERIFIED. There is no exit-zero
release path and no synthetic selector in the production schema. Test-only
positive fixtures verify mechanics and cannot establish qualification.

Caller SHA256 pins freeze chosen bytes, not their truth. Different producer
and auditor source bytes are necessary but cannot authenticate independence.
Self-reported execution, legal approval or PASS receipts, even coordinated
and rehashed, never cause this precheck to sign off a release. An independent
coordinator must separately establish authority, actual execution, numerical
adequacy, coverage/representativeness, scientific fairness and publication.
The tool is not wired into a shared release pipeline in this scope.

## Input Contract

Invoke `scripts/check_release_completion.py --manifest ABSOLUTE_EXTERNAL_JSON
--manifest-sha256 CALLER_SHA256 --current-root CLEAN_REPOSITORY
--artifact-dir NEW_EXTERNAL_DIRECTORY` with the existing infra identity.
Both terminal states are nonzero; inspect the external `completion.json`
and `execution.log`, not merely the exit code. Invalid input uses INCOMPLETE.
Do not place credentials or authorization material in any JSON: approval
objects accept exactly `{status, pin}`, with a status and digest only.

Manifest exact keys: `schema_version=1`, `scope=insufficient-release-evidence-v1`,
`source_commit` (full Git SHA), `source_sha256`, `capability_scope`, `capabilities`,
`commercial`, `gpu_scaling`, `publication`, `licensing`, `historical_observations`.
Source membership is the clean, committed `src/`, `scripts/`, `LICENSE` and
`pyproject.toml` set, reread before/after inspection. No source from an input
artifact is imported or executed. Data/docs/test bytes need their own evidence
descriptors when used; this source set is not a whole-distribution SBOM.

`capability_scope` must equal the ordered `CAPABILITIES` tuple in the tool:
LP, convex QP, sparse/matrix-free, scaling, infeasibility/recession, signed
gap/status, SOC/RSOC, SDP, exponential/power, general MILP/convex MIQP,
nonlinear/global, presolve/postsolve, refinement, held-out public robustness,
perturbation/long-run, failure/tail controls, cold/prepared/changing data,
release/runtime compatibility. This is a minimum versioned review checklist,
not an assertion that the list exhausts all optimization capabilities. A new
scope requires a reviewed schema version; narrow prototypes cannot replace
general entries. Missing entries are each returned as INCOMPLETE.

Every descriptor is exactly `{path: absolute, sha256: 64 lowercase hex, bytes:
positive integer}` and points outside both the assessed repo and tool repo.
All bytes are actually read and pinned, with parent/file symlinks, special
files, empty/oversize inputs and mutation during reads rejected. The established
`/hf3fs-jd` mount alias is recognized. Duplicate JSON keys, booleans, nonfinite
numbers (including overflowed JSON exponent) and extra/unknown fields are
rejected. Only the explicit historical negative adapter can read legacy bools;
it cannot qualify any section or be used as a success adapter.

Each supplied section receipt is exactly `{schema_version:1, section,
source_sha256, producer, auditor, actual, acceptance}`. Source map must equal
the manifest. Producer bytes must also match a current committed source hash.
Producer/auditor are byte descriptors, never executable inputs.
Actual JSON is `{source_sha256, section, data}`. Independent acceptance JSON
is `{source_sha256, section, actual_sha256, auditor_sha256, verdict}`. Its
PASS must bind the actual/auditor bytes; changing actual data and rehashing the
manifest cannot preserve an old acceptance. Fresh self-approved acceptance
still only yields REVIEW_REQUIRED, never qualification.

## Necessary Section Checks

- Each capability needs a matching scope, independent PASS, nonempty original
  models/results/checks, positive negative controls and consistent counts.
- Gurobi/CPLEX/MOSEK each need positive actual native calls and completions,
  cold/prepared same-model Openopt pairs, resources and original-coordinate,
  signed-gap/status checks. Discovery/import/planned calls are not solves.
  Authorization `{status:AUTHORIZED,pin}` is an identity pointer for later
  legal review, never authenticated authorization here.
- GPU entries `1`, `2`, `3` each need exactly that many distinct RTX PRO 4500
  device records, positive per-device kernel events, source/result-bound
  coordinator receipt, transfer-inclusive end-to-end cold/prepared samples,
  same model/pair/resources across counts, and original-coordinate checks.
  Three independent one-card jobs cannot satisfy this. Timing is read, not
  speedup superiority inferred; workload, host-resource fairness and trace
  authenticity remain independent reviews.
- Publication needs PUBLISHED current committed source, exact published vs
  readback bytes, matching remote source map/commit and pinned managed
  readback transport bytes. Local preparation/HTML does not suffice. The tool
  makes no network request and cannot authenticate a remote response.
- Licensing needs exact source-bound 0BSD LICENSE bytes, complete selected
  source-origin redistribution review pointers and explicit runtime/build/
  optional-SDK inventory/SBOM membership. Dependencies retain their licenses;
  NOT_BUNDLED is distinct from a redistribution grant. Missing/UNKNOWN rights
  are insufficient, not free use. Data/manual/runtime/non-source assets need
  separate broader rights review; no legal completeness claim is made here.

Sample schema and per-section exact shapes are exercised by the self-contained
`Evidence` builder in `tests/test_release_completion.py`. That builder is
test-only, writes clearly synthetic fixtures, never calls a solver and is not
a public production producer. Counts retain attempted/accepted/failed/
unsupported/skipped; incomplete strata fail instead of removing denominators.

## Reproduction And Known Incompletions

Default CPU entry: `scripts/verify.py tests --target
tests/test_release_completion.py --artifact-dir EXTERNAL_DIRECTORY` using
the existing validation interpreter, hidden CUDA and single-thread limits.
Default tests need no historical artifact, commercial SDK, network or GPU.
They include real disposable Git source checks, actual CLI/receipt inspection,
full test-only REVIEW_REQUIRED, missing scope, numeric/bool/count mutations,
old acceptance vs rehashed actual, source drift, paired model/device/trace,
publication readback and origin/SBOM negative controls.

Optional evidence-only entry: `tests/test_release_completion.py --freeze-known
--current-root CLEAN_REPOSITORY --artifact-dir NEW_EXTERNAL_DIRECTORY`.
It fails loud if the historical receipt is missing/changed. It explicitly pins
the real commercial runner discovery from 2026-10-10T10:05:04.536415+00:00:

`/hf3fs-jd/data/tmp/openopt-release-20261010/commercial-readiness-review/runner-discovery/discovery-3bsk_wt3/result.json`

SHA256 `c14f3a58f79a8cf9a827eb0ad8ff7f2fdda0f18d0cf1f4c465d0060443be71fe`.
All three solvers have empty cold/prepared sample lists and actual attempted/
solve-attempted/accepted counts zero. This is historical zero evidence, not a
new SDK probe, authorized run, claim about every environment or current-source
qualification. No historical success is backfilled. The generated manifest
leaves every unproven current scope missing, including multi-card scaling and
publication, and preserves INCOMPLETE with zero commercial solves. Missing
scope evidence is not a filesystem-based assertion of nonimplementation.

Owner receipts and clean commit are reported externally after actual runs;
this document alone does not satisfy the precheck or user completion goal.

## Owner Development Evidence

External scope root:
`/hf3fs-jd/data/tmp/openopt-release-20261010/release-completion-gate/`.
Baseline `pre-smoke/smoke-v4viyurq/verification.json`: 80 pass, four explicit
local CUDA skips, 6.49 pytest seconds. Original first failure is retained at
`unit-v1/tests-jhg3zu3m/verification.json` and `pytest.log`: Python 3.10
f-string collection SyntaxError, zero numerical tests run. Two output lines
were repaired, not skipped. `unit-v2/tests-_zohgrel/verification.json` then
records 42 pass/zero skip, 139.63 seconds.

After adding three unchanged-contract negatives and reusing immutable
test-only fixture bytes, `unit-v3/tests-yfm9toef/verification.json` records
45 pass/zero skip, 41.18 seconds. `post-smoke-v1/smoke-evvle65r/verification.json`
records 80 pass/four explicit CUDA skips, 6.33 seconds. Both have source
consistency PASS, but precede the clean commit. Credential literal/shape
scan at `security-v1/source-security-scan.json` has zero/zero hits on the
three named files; clean-source scan must follow this documentation update.
Clean-source final counts, pins and historical INCOMPLETE receipt are handed
off separately without editing tested source during a freeze.
