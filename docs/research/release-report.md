# Public-Sanitized Release Report

## Boundary

`scripts/render_release_report.py` generates only v4 report/design bytes and
external support files. It never imports historical generators, invokes a
solver, accesses the network, installs dependencies or publishes. The parent
owns generation of `docs/benchmark-report-v4.html`, `docs/design-v4.html`,
stable entry points and history navigation updates. v1-v3 remain historical.

The Chinese conclusion precedes measurements. Correctness applies only to
the tested scope. Online publication and commercial qualification remain
`UNVERIFIED`; a missing SDK, successful importer or synthetic fixture cannot
qualify a commercial comparison.

## Caller-Pinned CLI

```text
sudo -n -u infra python3 -B scripts/render_release_report.py \
  --manifest ABSOLUTE_3FS_INPUT_MANIFEST \
  --manifest-sha256 CALLER_EXPECTED_SHA256 \
  --output-dir NEW_ABSOLUTE_3FS_DIRECTORY
```

Add `--check` to verify an existing exact output set without writing. CLI
output is `RELEASE_REPORT verdict=PASS|FAIL`; failures return nonzero with a
safe first-cause code and role coordinate, never rejected payload values.
Existing output directories are rejected, not replaced. Both 3FS mount
aliases are accepted; arbitrary symlinks, traversal and nonregular files are
rejected. Artifacts must remain outside the worktree. Do not publish the
private input manifest or `report-bundle.json`.

The Python integration API is
`generate(manifest: Path, manifest_sha256: str) -> dict[str, bytes]`.
It has no output side effects. The parent can select the two HTML byte
payloads for its explicitly authorized documentation paths.

Optional preparation accepts repeated
`--input ID ABSOLUTE_3FS_PATH CALLER_SHA256` for every required role except
`metadata`, together with `--prepare-inputs --source-sha NUMERICAL_COMMIT`,
`--modified-at ZONED_FIXED_EDIT_TIME --output-dir NEW_PRIVATE_INPUT_DIRECTORY`.
Every caller pin is verified. The tool writes `metadata.json` and
`input-manifest.json`, validates the complete input and emits
`RELEASE_REPORT_INPUT verdict=PREPARED manifest=... sha256=...`.
Optional flags are `--doc-source-sha`, `--regression-scope`,
`--cpu-binding-kind`, `--cpu-binding-value`; every nonscope binding also
requires `cpu_version`. No evidence is selected implicitly. Changed input
selections need new directories; rejected preparations are not overwritten.

## Input Manifest

The private manifest is a nonempty JSON object with exactly:

```json
{
  "schema_version": 1,
  "modified_at": "2026-10-10T17:00:00+08:00",
  "source_sha": "FULL_SOURCE_COMMIT_SHA",
  "inputs": [
    {"id": "cpu", "path": "ABSOLUTE_3FS_JSON_PATH", "sha256": "CALLER_PIN"}
  ]
}
```

The abbreviated example shows descriptor shape, not a complete runnable
manifest. IDs and input paths must be unique. All descriptors require their
actual file SHA256 and explicit absolute 3FS path. No glob, latest-file
selection, implicit JUnit neighbor, process status or cached measurement is
used. The manifest itself requires an independent caller pin.

Required IDs: `cpu`, `junit`, `wheel`, `gpu`, `gpu_run`, `commercial`, `corpus`,
`campaign`, `audit`, `live`, `metadata`. Optional IDs: `gpu_junit`, `cpu_version`,
`public_qualification`.
`junit` and `gpu_junit` are XML; all other inputs are JSON objects. Each input
is opened no-follow, read as a bounded regular file, hash-checked, then checked
again for content and file-identity changes before acceptance. A moving
snapshot is rejected, not automatically retried or silently frozen.

- `cpu`: verification JSON with `verdict`, `tests`, `coverage`, `finished_at`.
  When present, `source_consistency` must carry `verdict`, `same_commit`,
  `changed`, `missing`, `extra`, `files`. PASS is cross-checked against
  nonempty `source.commit/tree/clean/sha256`, matching commit, file count,
  same commit and empty change collections. A hidden consistency FAIL
  cannot accompany CPU PASS. Missing legacy fields project to `UNVERIFIED`
  with explicit missing-field provenance, never frozen-release acceptance.
- `junit`: actual leaf-suite counts must match actual testcase outcomes and
  CPU JSON counts exactly. Empty, forged-count and false-PASS reports fail.
- `gpu`: integrated numerical result with `verdict`, `tests`, `scope`,
  `finished_at`, nonempty `source_sha256`; `returncode` is checked if present.
- `gpu_run`: explicit run JSON with `source_sha`. Result/run descriptor hashes
  and the run source SHA appear in the public report. GPU source is not
  silently equated with the later release source.
- `wheel`: distribution JSON with `verdict`, positive `modules`,
  `license_expression` equal to `0BSD`. New source consistency fields use
  the CPU gate shape with matching numerical source commit, actual source
  hashes and `wheel_sha256`. A false PASS or hidden source failure is
  rejected. Missing legacy fields project to `UNVERIFIED`; frozen release
  requires both CPU and wheel gates to PASS with clean source.
- `commercial`: `verdict`, `qualification` equal to `UNVERIFIED`, `solvers`
  mapping with each solver's `status`, and `counts` containing `planned`,
  `attempted`, `solve_attempted`, `passed`, `failed`. Ordering constraints
  reject impossible counts; zero attempts remain actual zero, not success.
- `corpus`: `verdict`, `scope`, `rows` covering exactly 24 distinct synthetic
  case IDs; multiple solver rows are allowed without pretending they are
  distinct problems. Each row carries `case_id`, `solver`, `verdict`,
  `solver_status`, `status_matches_expected`, `expected_property`,
  `candidate_quality` and/or `certificate_quality`. A `QUALIFIED` optimal
  case requires the candidate check to pass; nonoptimal cases require the
  direction/certificate check to pass. Expected ordinary-candidate FAIL
  states are displayed, never discarded.
- `campaign`: nonempty `seats` mapping.
- `audit`: nonempty `seats` list with matching unique membership, `seat`,
  `verdict`, `outside_scope`; a PASS with nonempty outside scope is rejected.
  Legitimate FAIL statuses and violation counts are displayed.
- `live`: `verdict`, zoned `read_at`, nonempty `ready`, explicit `active`,
  `cordoned`, `retained_failures` collections. Only counts leave the private
  snapshot, not node identities, error text or connection material.
- Optional `public_qualification`: explicit `verdict`, public-safe `scope`,
  and `counts` with `planned`, `attempted`, `qualified`, `rejected`.
  Impossible counts and PASS with zero attempts are rejected. Omitted input
  produces no summary and never implies the public runner was executed.

`metadata` is explicit version and reporting policy input:

```json
{
  "version": 4,
  "source_sha": "FULL_SOURCE_COMMIT_SHA",
  "modified_at": "2026-10-10T17:00:00+08:00",
  "regression_scope": "intermediate_snapshot",
  "cpu_binding": {"kind": "scope_only"},
  "online_publication": "UNVERIFIED",
  "commercial_qualification": "UNVERIFIED"
}
```

Source SHA and `modified_at` must exactly match the manifest. Zoned edit time
is rendered to seconds in +08:00, independent of render time. The intermediate
full-merged run is not a frozen release regression: tools added during that
run are not certified by core source stability.

For explicit source binding use `cpu_binding.kind` of `code_hash` or
`version_metadata`, with `value`, and a pinned `cpu_version` JSON containing
the same `source_sha` plus `code_hash` or `version` respectively. `code_hash`
must be a full SHA256. `frozen_release` requires such binding, but is still
a caller declaration, not an independent proof that every source was frozen
throughout execution. The parent must supply its final frozen verification
and source-version evidence after all seats have been integrated.

With the new CPU source gate, `code_hash` is SHA256 of
`encoded(cpu["source"]["sha256"])`: sorted keys, two-space indentation,
UTF-8, non-ASCII unescaped, no nonfinite values, trailing newline.
Frozen scope additionally requires source consistency PASS, clean source
and full `coverage: tests`. This binds numerical source, not later docs.
`source_sha` is the numerical source commit. Optional `doc_source_sha`
is a distinct documentation coordinate and is never forced equal.
Normally omit it from the generated page to avoid a commit self-reference:
the publication plan later binds its own committed docs source and producer.
The report states this distinction instead of inventing one frozen whole
repository that already contains its own future generated documentation.

## Outputs And Claims

Public-safe output files are the two v4 HTML pages, `summary.json`,
`claims.json`, and `manifest.json`. The last binds their relative file names,
hashes, byte sizes, source SHA and fixed edit time. `summary.json` contains
only selected public scalar projections; no raw logs, internal paths,
hosts, addresses, credential fields or commercial license material.

`report-bundle.json` is **private external evidence**: explicit source
descriptors (ID/path/hash), source SHA, edit time, manifest pin and per-scalar
provenance. Each provenance record names the source JSON pointer and any
operation, such as verified test arithmetic or collection length. This
private file is not included in the public manifest. Preserve it for replay;
never include it in publication selection.

HTML text and SVG labels use the same summary scalars. `data-scalar` makes
the mapping inspectable. `claims.json` supplies exact text, scalar pointer,
document, kind and unit for publication integration. Repeated displayed
values share a scalar and a unique claim ID per document.
Numeric and enum-status claims are compatible with the publication plan;
legal text (`0BSD`), dates, hashes and scope strings are `text` and must not
be falsely relabeled as measured number/status claims. Public numbers/status
can use pinned `summary.json` as evidence with declared producer
`scripts/render_release_report.py`; private provenance retains the raw chain.

The pages use only static relative HTML links and fragment navigation,
inline CSS and inline SVG. No external JavaScript, handler attributes,
scripts, forms, active SVG, endpoint links or raw filesystem paths.
No external fonts or assets are loaded. Selected unsafe text is rejected;
unselected private metadata is not copied. Encoded unsafe content is also
rejected. This is public sanitization of an allowlisted projection, not a
promise that arbitrary secret-like human text can be detected perfectly.

## Acceptance

Use the existing CPU validation environment with external pytest artifact
directory and JUnit output, targeting `tests/test_render_release_report.py`.
Tests use real files on 3FS and prove exact render determinism, source
no-mutation, claims equality, static-only links, output byte readback,
no-overwrite, secret/path/host/event rejection, snapshot-race rejection,
changed hashes, empty data, duplicate JSON, false counts, failed JUnit,
false corpus qualification and explicit source-binding requirements.

All pinned JSON inputs are traversed for FAIL/REJECTED/NOT_QUALIFIED status
scalars. Their values remain visible even inside otherwise passing records;
private pointers retain exact coordinates without leaking internal keys or
error payloads. The count is status occurrences, not distinct failures.

Legitimate FAIL regression, nonqualified rows and scope-audit failures
remain visible. Passing renderer tests certify this tool's fixture scope,
not release solver correctness, browser rendering, screen-reader behavior,
online publication or commercial qualification. These latter checks belong
to their explicitly authorized final acceptance steps.

## Independent Stop-Gate Inputs

Optional caller-pinned roles `extension_native`, `extension_presolve` and
`extension_refinement` carry separate extension acceptance JSON, not a pytest
result substituted for an independent original-model replay. PASS requires
explicit final-source acceptance for native/refinement, complete counts, the
original source commit, and no commercial/performance/exact-certificate upgrade. FAIL
records may lack those success-only fields; their failure and timestamp remain
visible without publishing error payloads. Historical source coordinates never
become the report's newer source coordinate implicitly.
Presolve instead requires `current_core: PASS` and a separate caller-pinned
`extension_presolve_run` role matching the acceptance's original run descriptor.
Its source commit comes from that explicit run; missing final-source fields
are not invented or populated from a later checkout.

Optional `proof_contract_review` shows the CPU contract review status, finish
time and explicit producer commit. Even PASS does not qualify GPU coordinator
origin, remote attestation, exact certificates or commercial comparisons.

Optional `public_robustness` requires separate caller-pinned
`public_robustness_plan` and `public_robustness_corpus` roles. Exact plan/result
membership must equal corpus cases x predeclared arms, including original model
hashes, solvers and single-attempt identity. Reduced or changed result sets
cannot be accepted by merely resealing row counts.
It consumes complete attempted rows and per-arm counts.
Every case/arm identity is unique, every row belongs to an arm, and all
QUALIFIED/UNQUALIFIED/FAIL/BUDGET_EXHAUSTED counts are recomputed. Planned and
actual attempted denominators remain separate. The producer's overall
`PASS` is accepted only when every row is QUALIFIED; otherwise it must remain
UNQUALIFIED. Candidate qualification requires an actual attempt. Combined arms are not a
single-variable ablation or performance ranking. The selected contract does
not upgrade unverified dataset rights, clip tiny duals or claim speedups.
The original experiment version is displayed. Full settings/runtime binding
remains UNVERIFIED in this renderer; in particular v1 is not a v2 execution.
Successful independent extension acceptance requires the original complete
suite denominator (native 21, presolve 8, refinement 7) and zero skips, errors
or failures. This stricter scope does not ban ordinary CPU regression skips.
UNQUALIFIED status occurrences join the recursively retained failure
statuses; the count remains occurrences, not distinct failed problems.

## Optional RSOC Receipt Projection

The optional roles `rotated_cuda_acceptance` and `rsoc_extremes_acceptance`
are separate, independently caller-pinned auditor receipts. They are not
replaced by top-level GPU PASS, added to CPU counts, or required implicitly
by historical v4 inputs. Each success requires its own `PREFIX_run`,
`PREFIX_result` and `PREFIX_junit` descriptor, where PREFIX is `rotated_cuda`
or `rsoc_extremes`. JUnit roles are XML, other roles JSON. CLI preparation
uses the unchanged repeated `--input ID PATH SHA256` interface.

The 76 contract consumes `RSOC_EXTREMES_REPLAY_PASS`, exactly 76 unique
original-coordinate replay identities, zero failures/errors/skips, original
source commit, all original input descriptors and unique JUnit sidecar pins.
Its independently versioned domain is 38 represented vectors x primal/dual;
it is not inferred from supplied receipts or mutable test collection.
The source manifest, complete snapshot bytes, result source subset, isolated
run/output paths, result mirror, execution artifacts and all sidecars are
hash-checked and rechecked under the existing no-follow snapshot rules.

Fixed thirteen additionally requires `rotated_cuda_bundle` and
`rotated_cuda_pins`. The receipt must be mode `accept`, bind the explicit
pins manifest digest, and carry a valid auditor digest. Numeric-only replay
or seal receipts do not substitute for accept. The complete bundle must
match its explicit caller pins, versioned algorithm, execution source and
the exact ten positive / three expected-rejection denominator. Every
original LP/QP or projection check is structurally complete; positives must
pass, each expected rejected candidate must retain its passed baseline and
specified false checks. Projection component and Moreau outcomes must pass.
The complete thirteen JUnit identities and original sidecar set are checked
in both directions, without dropping expected negative cases.

An input FAIL remains visible with its timestamp and receipt digest even
when success-only fields are absent. Unknown success schemas, missing roles,
bad denominators, duplicates, resealed dropped cases, hidden check failures,
unmatched caller pins, changed snapshots and source substitutions reject
the render. No mathematical auditor or solver is executed by this renderer.
These checks bind and project the caller's receipt; they do not attest that
its author was independent, repeat its arithmetic, or cryptographically
attest its archived runtime. Owner PASS is not independent signoff.

Public scalars retain each execution source, scope hash and acceptance,
run/result/JUnit digests. `source_relation` is `historical` whenever the
execution commit differs from the report's declared numerical source.
`current` additionally requires the clean CPU source-consistency PASS and
matching executed numerical/runner/test hashes. Historical CUDA cannot be
retroactively promoted by a later CPU audit or documentation commit.
Two passing projections do not establish full release, commercial,
performance, multi-GPU or online qualification.

Self-contained synthetic archive tests are explicitly non-executed fixtures.
They prove renderer binding only. A separate real-file test uses two named,
fixed historical acceptance receipts and checks deterministic generation,
external byte readback and the actual renderer CLI without GPU/network.
Its optional absence is a skip, never actual historical acceptance.
The first development receipt `report-rsoc-extension/mechanism-v1/tests-li_jds8c`
retains 27 passes and two failed test assertions: a broad leakage assertion
matched the existing legitimate `synthetic-*` corpus IDs. The chain checks
passed; the assertion was narrowed to the actual private fixture payload/path.
No numerical threshold, deadline, denominator or receipt pin was relaxed.

Development validation at `report-rsoc-extension/full-v3/tests-sbhbel7h`
passed 112 nodes, zero failures/errors/skips, in 116.44 pytest seconds;
305 source files matched before/after, but that source was still dirty.
The separate `actual-v2/tests-u2b1rv7s` passed three nodes in 26.83 seconds.
Its `pytest-tmp/test_real_pinned_rsoc_archives0/evidence/real-rsoc-projection.json`
records both real historical receipt sets, their original run/result/JUnit
pins, `cuda_execution=false`, `qualified=false` and actual CLI byte readback.
This is projection evidence, not independent fixed13 signoff.

The additional CLI negatives retain their first rejection codes, logs,
rehashed input-manifest digests and absence of output bundles in each isolated
test artifact. They mutate denominators and drop real contract identities
while rehashing the changed JSON; an unchanged outer hash is not the reason
they reject. A same-source synthetic positive supplies the complete matching
clean CPU snapshot and still remains intermediate/unqualified.
These six controls passed with zero failures/errors/skips in 15.33 seconds
at `report-rsoc-extension/final-controls-v4/tests-458s5_wi`; before/after
source consistency was PASS. The unchanged whole-file clean-source rerun
is handed off under `report-rsoc-extension/clean-final-v5`; its verification
JSON and complete JUnit, not this planned coordinate, decide acceptance.
