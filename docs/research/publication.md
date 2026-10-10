# Local Publication Preparation

`scripts/prepare_publication.py` prepares a static, disclosure-minimized ZIP.
`PREPARED` means local source/evidence checks and byte readback passed.
It does **not** mean `PUBLISHED`, commercial qualification, solver correctness
qualification, permission to deploy, or permission to redistribute raw evidence.

## Input Contract

The caller supplies an explicit clean repository HEAD, documentation version
`vN`, and a SHA256-pinned private JSON plan under an external shared-storage
evidence root. Every selected document is a committed, nonempty UTF-8 file under
`docs/`; HTML filenames must end in the selected `-vN.html`.
The upstream renderer produces public-sanitized documents and commits them
before preparation. This tool does not render, substitute or modify docs.
Each claim names its `document` and exact displayed `text`.
Non-measurement prose, dates, version identifiers and mathematical constants
remain source content. This tool does not infer the meaning of static numbers:
the integrating owner must ensure all measured numbers/statuses have claims.

Plan schema (all keys required; unknown keys are rejected):

```text
schema_version: 1
source_commit: full Git commit SHA at current clean HEAD
docs_version: vN
documents: nonempty list of {path, sha256}
evidence: nonempty list of {id, path, sha256, producer}
claims: nonempty list of {id, evidence, pointer, kind, unit, document, text}
```

Evidence paths are explicit files relative to the external evidence root,
not directories, globs, nested bundles or automatic recursive discovery.
`producer` names a committed `scripts/*.py` origin. It is a **declared**
producer, not proof that the command ran or that a solver result is correct.
The public origin record contains its source hash, evidence content hash and
JSON pointer tokens (reconstructable as an RFC 6901 pointer), never its private
evidence path or execution environment.
Execution and independent numerical acceptance remain upstream responsibilities.

Claim IDs are lowercase identifiers. `pointer` is an explicit JSON pointer.
`kind=number` accepts finite integers/floats, not booleans; units are `s`, `ms`,
`count`, `ratio`, `percent`, `bytes`, `MiB`. `kind=status` requires empty unit
and one of the bounded research/status enumerations in the tool.
Numeric `text` must equal the source value exactly, allowing equivalent decimal
formatting such as `1.250`; rounded/derived values need their own upstream
machine-readable source. Status text must equal its source status.
The exact displayed text must occur in the selected document; every document
requires at least one claim. Text presence is not proof of semantic placement
or complete coverage: those are renderer/integrating-owner responsibilities.
Missing fields, arbitrary strings, containers, nonfinite JSON, duplicate JSON
keys, unused evidence/claims and sensitive field names are rejected.
The bundle-level commercial/online status is always `UNVERIFIED`, regardless
of individual numerical status claims.

Only these selected scalar projections enter `public-evidence.json`.
Unselected raw JSON fields, credentials, host identities, commercial license
materials, model data, logs and original evidence files are never copied.
There is no option to copy an entire raw evidence bundle.

## Safety And Output

The tool rejects uncommitted/staged/untracked source changes, an incorrect HEAD,
source/document hash drift, empty files, nonregular files, symlinks in any
input/output component, traversal and output overlapping source/evidence.
The established shared-storage mount alias is normalized explicitly; other
symlinks are not resolved or followed. Evidence is pinned before selection and
rechecked by bytes and file identity/metadata before and after output.

Public document names, source bytes, rendered bytes, selected pointers and
generated manifests are checked for internal absolute paths, host/network
identities, credential forms, commercial license material and active HTML.
Entity/percent/Unicode escape encodings are checked too. Rejected values are
not echoed in diagnostic output. This bounded detector is not a universal
secret classifier or a replacement for the existing publication security gate.
Opaque unknown credentials in ordinary prose still require integrating review.
The policy deliberately disallows external network links and active scripts;
prepare static relative-link documentation instead of widening the detector.
Project licensing prose such as `0BSD` is allowed, not commercial license files.

Selected documents are preserved byte-for-byte.
A leaking document is rejected, not silently altered or certified by redaction.
Existing research pages with internal evidence paths therefore cannot be
published unchanged; provide committed public-sanitized documents with claims.

The new output directory is external shared storage. Existing output is never
overwritten or cleaned. It contains:

- `bundle.zip`: sorted static docs, scalar evidence, manifest and version.
- `manifest.json` and `version.json`: byte-identical detached public metadata.
- `SHA256SUMS`: full SHA256 hashes of archive and detached metadata.
- `local-verification.json`: written last after deterministic regeneration and
  byte readback; mandatory for a locally accepted preparation.

ZIP timestamps, order, file modes and JSON formatting are deterministic.
Failed partial output is retained for diagnosis; no cleanup or production
rollback is performed. Marker presence alone is never an acceptance:
the complete `verify` operation and its successful audit record are required.
`verify` regenerates the expected bundle from the pinned inputs, compares all
bytes and exact member sets, and requires a caller-supplied archive hash.
Editing the archive and its own checksum cannot create a valid acceptance.
For verification after source integration, use a clean worktree at the original
explicit source commit; a different HEAD is not silently substituted.

## Execution And Integration

The CLI has `prepare` and `verify` actions. Both require `--repo`,
`--evidence-root`, `--plan`, `--plan-sha256`, and `--output`.
`verify` additionally requires `--bundle-sha256`, retained independently from
the successful preparation. `--log-dir` selects external shared-storage audit
records, defaulting to this campaign's `publication/logs`; success/failure
records retain exit code, time and safe first-cause coordinates without payloads.
Run with the established `infra` identity and
existing validation interpreter, with `CUDA_VISIBLE_DEVICES=-1`.
No installation, network operation, credential lookup, GPU work or publication
operation is performed.

CPU regression, logs, fixtures, JUnit and bundles are external:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/publication --target tests/test_prepare_publication.py
```

Tests exercise deterministic byte readback, real CLI preparation/verification,
nonempty claim provenance, leakage controls, resealed malicious documents/plans,
changed inputs, output tampering, symlinks and repeat-write refusal.
Fixtures are synthetic, never performance or commercial evidence.

Formal wiring and online qualification belong to the coordinating owner.
Do not write shared infra index/generator/source files from this tool.
The only publication entry remains `publish-all.sh`; its registered sources
are fixed, a new project first needs a full-site publication, and readback
belongs to the existing chain. This local ZIP is not permission to bypass any
of those gates, change an ingress, push, or claim online completion.
