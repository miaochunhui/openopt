# Project-owned site / scalar contract v1

This is a new project landing page and catalog, not a modification of any
existing numerical report. It never publishes, registers shared sources,
changes authentication, or establishes commercial/general/GPU qualification.
Content baseline is frozen separately from the generator's later commit.

## Files and publication boundary

The exact outputs are `docs/site/index.html` and `docs.publish.json`. The latter
uses the existing publisher's version1 project catalog: one `overview` route,
one `admin` audience with empty `path_prefix`, exact nine page fields, and no
aliases. The page explicitly states **existing super-only** onboarding scope.
Catalog audience metadata is not an authentication policy. The chair must
preserve every existing shared modification when registering the project,
review the effective Caddy access scope, use the established first-publication
chain, and independently read back exact deployed bytes.

The incremental registration coordinates, after accepted integration, are
homepage `/data/dev/openopt/docs/site/index.html`, catalog
`/data/dev/openopt/docs.publish.json`, project slug `openopt`, allowed root
`/data/dev/openopt/docs`. The existing publisher computes the approval hash;
this tool does not write `sources.tsv`, the shared homepage/generator, Caddy,
or any publisher state. A research Git push is a separate operation, also not
performed by this tool. No live/public PASS is inferred from local rendering.

## Input and provenance

`collect` creates a new external JSON input using explicit full content source,
second-resolution `+08:00` editing time, and each receipt's caller SHA-256 and
full original source commit. It accepts only the named `cpu-smoke` and
`psd-owner-cpu` owner CPU receipt shapes, plus an explicitly failed
`operator-cuda-failure` result/JUnit/log readback. It verifies regular Git blob membership
and every source byte hash against each original frozen commit, clean-source
flags, source-consistency fields, and the count denominator. PSD extraction
additionally requires exact collection/membership and explicit CPU-only runtime.
It extracts only counts, original commits, receipt hashes and finished timestamps.
It does not import a solver, rerun numerical checks, attest execution, accept
unlicensed SDKs or make synthetic data genuine evidence.

Input schema1 has exact keys: `schema_version`, `project`,
`content_source_commit`, `modified_at`, `license`, `boundary_sha256`, `access`,
`publication`, `qualification`, `evidence`. The code license and research
boundary hashes bind the baseline `LICENSE` and `AGENTS.md` bytes. There is no
free text field, input path, model vector, SDK field, or credential field.
Evidence carries `source_commit`, `receipt_sha256`, `finished_at`, `tests`,
`passed`, `failures`, `errors`, `skipped`, `relation`, `kind`. Historical source
and separate PSD freeze are disclosed; they cannot be relabeled same-source.
The page presents reported owner counts, **not independently accepted results**.
The optional `--cuda-failure RESULT RESULT_SHA JUNIT JUNIT_SHA LOG LOG_SHA
SOURCE_COMMIT` uses seven caller arguments, exact seventeen selected source
hashes, failed integer return status, actual JUnit counts/unique target identity,
and actual failure summary. Its public scalar record additionally includes
`junit_sha256` and `pytest_log_sha256`; kind is
`reported-failed-cuda-owner-checks` and failures must remain nonzero. It never
promotes that result to a seal, archive acceptance, performance or numerical
qualification. Fixed13 is not an accepted input kind and no fixed13 proof is
published. The currently reported operator failures remain explicitly red.

Commercial and complete-general scope remain PENDING; current-source CUDA and
fair comparative speed remain UNVERIFIED. Publication remains NOT_PUBLISHED.
Those are hard contract ceilings, not claims obtained from receipt PASS.
Project code is 0BSD; dependency licenses, proprietary execution authority and
benchmark disclosure permission remain separate prerequisites.

JSON duplicate keys, nonfinite values, booleans as integers, unknown fields,
empty evidence, count drift, malformed hashes, missing seconds, wrong offset,
future-to-edit evidence, raw arrays and claim/access promotions reject loudly.
Rejected payload contents are not echoed. Generated content is also checked
with the existing publication leakage/active-content scanner.
The scanner's IPv6 expression also matches clocks; only exact complete
timestamps already accepted by the strict second/offset/date grammar are
replaced for scanning. All other network, path and active-content tokens remain
subject to the unchanged scanner. Negative controls exercise that narrow scope.

## Commands

All receipts/input/logs are create-once and external, under the established 3FS
temporary mount. Both render destinations must be absent before rendering;
the tool never overwrites an old page. `check` is read-only for source/output
files and requires a full **current clean generator owner** commit. It compares
complete committed source membership/bytes, regenerates both files and compares
them exactly. It does not retrofit an old receipt to a new source.

```text
python scripts/render_project_site.py collect --input <3FS-input.json> \
  --content-source-commit <full-content-baseline> \
  --modified-at YYYY-MM-DDTHH:MM:SS+08:00 \
  --evidence cpu-smoke <receipt> <receipt-sha256> <original-full-commit> \
  --evidence psd-owner-cpu <receipt> <receipt-sha256> <original-full-commit> \
  --artifact-dir <3FS-artifacts>
python scripts/render_project_site.py render --input <3FS-input.json> \
  --input-sha256 <input-sha256> --artifact-dir <3FS-artifacts>
python scripts/render_project_site.py check --input <3FS-input.json> \
  --input-sha256 <input-sha256> --owner-commit <new-full-clean-owner> \
  --artifact-dir <3FS-artifacts>
```

Artifact `receipt.json` records input/output hashes, before/after actual source,
owned-file hashes, source consistency, safe first cause, exit and finish time.
`execution.log` contains the terminal verdict. Failure receipts are not hidden
or overwritten. No permission helper, root, install, cluster task or publication
action is part of this workflow.

Default tests use their own synthetic receipts and real isolated Git trees:
these are mechanism controls, not commercial/GPU evidence. No historical mount
or runtime is required by portable default tests. Real extraction is an explicit
CLI operation using caller paths/pins; absent evidence fails, never skips or
manufactures a qualified input. The committed page's real source pins and owner
clean validation are recorded externally, not by mutating this frozen document.

The portable owner CLI is `python tests/test_render_project_site.py --artifact-dir
<new-3FS-dir>`. It records complete actual collection/JUnit membership, zero-skip
counts, runtime/environment, source before/after and owned bytes. `--scan-only`
uses the existing literal scanner without disclosing any scan material and
records an independent synthetic sensitivity 1-to-0 control. Optional
`--publisher-check <existing-publish-docs.py>` calls only the existing read-only
catalog parser, hashes those actual tool bytes and never runs generate/refresh.

Initial owner development retained eight failures caused by the legacy
network scanner's clock/IPv6 collision (55 passes, eight failures).
After the exact timestamp-only scan exemption, all original controls plus
the explicit timestamp/network negative control passed. No numerical tolerance,
source pin, credential detection or publication boundary was weakened.
Two subsequent development owner CLI receipts retained FAIL after 70 and 82
zero-skip test passes: the receipt writer attempted to hash the not-yet-generated
two output files and raised KeyError. The writer now explicitly records absent
generated outputs during pre-render validation, rather than inventing hashes.
Final owner acceptance requires the actual five files and clean pinned check.
