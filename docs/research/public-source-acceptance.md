# Pinned Public Source Acceptance

## Boundary

`scripts/accept_public_source.py` is an offline producer for the existing
HiGHS afiro/adlittle acquisition. It never downloads, extracts, installs,
executes upstream code, solves, or changes the importer. Its independent
source checks are rooted in **three caller-trusted SHA256 pins**, not an
`origin=public` label, a plausible URL, or hand-written official attestation.

Source authenticity remains an explicit external trust boundary: the caller
must independently trust the acquisition/source pins. Hashes do not prove a
dishonestly repinned acquisition true. This tool rechecks captured official
commit API identity and complete commit URL spelling, but does not reauthenticate
TLS, verify a PGP signature or cryptographically reconstruct the upstream Git
tree. It does not promote repository MIT text into a third-party dataset license.

The exclusive write scope is these three new files: the producer, its tests and
this document. Existing importer/acquisition code, core APIs, infra and the
Noether runner are read-only dependencies.

## Verification

The producer checks acquisition/source/corpus pins and cross-pins, exact official
repository/ref/full commit/API/archive/per-case URLs, archive byte hash and size,
all archive headers, selected member completeness/duplicates/type/path/resource
limits, and all selected LICENSE/README bytes against the archive and pinned
review. It reads only selected member contents in memory, not broad extraction.

The existing mathematical replay checks raw records, all arrays, bounds,
objective offset, sense and names. The producer additionally links archive
members to acquired local bytes and frozen raw bytes, ordered case membership,
input declarations, metadata/NPZ/canonical array/model hashes, frozen reader logs,
recorded code hashes and original model summaries. Changed importer or acquisition
code is rejected, not silently treated as the producer's tested code.

Only the scoped two LPs are accepted. Integer/nonconvex/RANGES/unsupported model
policies remain the importer's unchanged gates. No synthetic fixture constitutes
public or industrial coverage.

## Output Contract

`acceptance.json` has the Noether `PUBLIC_SOURCE_ACCEPTED` schema:
exactly `schema_version`, `verdict`, `corpus_sha256`, ordered `cases`.
Rights/official repository/commit/review scope remain in `evidence.json` and
terminal rights also remain in `status.json`, not extra acceptance keys.
Each case has exactly
`case_id/source_url/input_sha256/model_sha256/source_review/evidence_sha256`.
Each `source_review` explicitly states no verified legal license and no public
redistribution of raw or derived arrays.

Every case's `evidence_sha256` pins the generated `evidence.json`, which links
all three caller anchors and every checked document/member/model digest. The
caller must supply the **acceptance file SHA256** to Noether; the evidence
sidecar hash is not a replacement for that caller trust anchor.

`verify` reopens the pinned acceptance/evidence, independently repeats all
source/model checks and compares the complete regenerated deterministic documents.
It needs the original local source/acquisition/frozen evidence and matching code;
it is read-only and neither repairs evidence nor reruns a solve.

`status.json` and `acceptance.log` retain timestamps and terminal verdicts;
failure preserves `failure.log`, first cause and exit code 2. Existing output
directories are refused. All runtime data/logs/JUnit are external. Successful
source acceptance and replay use exit 0, not numerical solver qualification.

## Reproduction

Use the existing CPU interpreter with `CUDA_VISIBLE_DEVICES=-1` and
`PYTHONDONTWRITEBYTECODE=1`. No installation or network is needed.

```text
accept_public_source.py accept --acquisition ACQUISITION_JSON --acquisition-sha256 CALLER_ACQUISITION_PIN --source SOURCE_JSON --source-sha256 CALLER_SOURCE_PIN --corpus FROZEN_DIRECTORY --corpus-sha256 CALLER_CORPUS_PIN --artifact-dir NEW_EXTERNAL_DIRECTORY
accept_public_source.py verify --acceptance ACCEPTANCE_JSON --acceptance-sha256 CALLER_ACCEPTANCE_PIN
accept_public_source.py check-consumer --acceptance ACCEPTANCE_JSON --acceptance-sha256 CALLER_ACCEPTANCE_PIN --runner LOCAL_MAIN_RUNNER --runner-sha256 CALLER_RUNNER_SOURCE_PIN --artifact-dir NEW_EXTERNAL_DIRECTORY
```

`check-consumer` loads the explicitly pinned local main runner and calls only
`source_eligibility` on the real frozen inputs. It records exact ordered
eligibility rows and runner source hash in `consumer.json`; it never solves,
changes that runner or treats consumer compatibility as numerical qualification.

The existing real bundle is rooted at
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-acquisition/`.
Use `real-v1/acquisition.json`, `source-v3/source.json`, `real-v1/frozen`:

- Acquisition: `9d8f6c56dd6da475b94cb0f7966bfd80d8355c47be193ce4198dfcb5fb82fbe3`.
- Source: `0f34c085305e86325e40194123dabf835a9caad55a28fead2b17cb983a431183`.
- Corpus: `3a434edfa8165167f56be25732840f03bbace1a5f5b7d1de7cdab989673d3f0f`.

The source is HiGHS `v1.15.1`, resolved commit
`04024d701f79feb8e2f18bc3df0dffc04ef05088`; archive SHA256 is
`2e121a70ae00f56db8e5ed993e4046c247248fa6280d839ff4808fa958971f0d`.
New producer evidence belongs under
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-source-acceptance/`.

Legal permissions remain `benchmark-use-declaration`, `license_verified=false`,
`public_redistribution=false`, including raw and derived arrays. Acquisition/
source acceptance is not solver, industrial, commercial or redistribution
qualification. Noether solving remains the main coordinator's responsibility.

## Accepted Evidence

Real source acceptance completed at 2026-10-10 09:22:06 UTC, two cases.
All paths below are relative to
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-source-acceptance/`:

- `real-v3/acceptance.json` SHA256:
  `1f82c8df6e9258bd8598884dd2bfee77f9e97f2ce7ec9f46035c5604822519fa`.
- `real-v3/evidence.json` SHA256:
  `990eff3c26e32cc9f991077ce325a0a716556e3017d73bb0eb7e1bf4428e1755`.
- `consumer-v3/consumer.json` SHA256:
  `a07829a4783b0743fc155b693a757d60cc4c9f35ad21b14d62bd24e74a0a852f`.

The main runner `/data/dev/openopt/scripts/qualify_public_qp.py`, SHA256
`7c6f526575c21e92c8a6b685b7fd57eee61c65e2d9ee4b4649776f6cbe131cce`,
accepted afiro/adlittle with `eligible=true`, `reason=null` at 09:22:25 UTC.
This was a direct real-input `source_eligibility` call, **zero solves**.
Read-only source replay returned `PUBLIC_SOURCE_REPLAYED`, case_count=2,
at 09:22:59 UTC. Strict four-key acceptance is intentional even for consumers
that optionally permit rights fields; no runner was changed to accommodate it.

Final CPU regression is `final-tests-v3/tests-7z_s4awi/verification.json`:
**126 passed, zero errors/failures/skips**, comprising 34 producer tests,
30 acquisition tests and 62 importer tests. Negative controls include all
three pins, raw/archive/doc/API/NPZ tamper, rehashed false URL/model/API claims,
member order, wrong source code, legal-permission elevation, duplicate/path/link
constructs, changed reader evidence, and a failed second pass before publication.
JSON was read back, not accepted only because a process exited zero.

Smoke is `final-smoke-v3/smoke-cyl71z3l/verification.json`: **79 passed,
four explicit CUDA-unavailable skips**. Earlier repeated counts are not summed.
No full-repository or GPU verification was performed in this seat.

Tested producer SHA256:
`597adf22c058368cc8f0901be927375a67b5a95b20aff4552c4071f605c3151e`.
Test SHA256:
`fa0c0bd80bbfb5523b1c52281fa821f62e583722d779ec69cc9d6f26763db71f`.
Original input/acquisition/corpus and code hashes remain unchanged.
