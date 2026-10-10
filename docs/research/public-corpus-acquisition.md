# Official Public LP Acquisition

## Boundary

`scripts/acquire_public_corpus.py` reuses the existing frozen MPS/QPS importer.
It does not modify parsing, the core API or solver qualification. The official
HiGHS repository tag `v1.15.1` is resolved through its commit API, then the full
resolved SHA is used in the official codeload archive URL. Only selected
LICENSE/README files are first extracted for human review; a separate
acknowledged operation extracts afiro/adlittle and freezes them unchanged.

Public availability does not establish data licensing. Root repository MIT
text is reviewed but is **not asserted to cover third-party Netlib instances**.
This acquisition records `benchmark-use-declaration`, `license_verified=false`,
`public_redistribution=false`. No raw or frozen arrays are committed, published
or bundled in a product. Internal benchmark use is the user's declaration,
not legal license verification. Main-coordinator solver qualification is separate.

## Execution

Before any network access read infra `docs/agents/rules/network.md` and the
complete `ops/astrill-jumphost/README.md`. The only fetch entry is the existing
`astrill-fetch.py url`. A read-only preflight uses that module's remote helper
to check infra identity and existing node/git/wget/rsync/setsid/timeout.
Missing prerequisites stop before the legacy auto-install branch can run.
No credentials, tools or service configuration are changed.

Transfers are serial (`network_parallelism=1`) and explicitly `--rate 512k`
(0.5 MiB/s), well below the authorized aggregate limit of 4 MiB/s.
`--tries 1` prevents automatic payload retries; the approved tool still has its
fixed HTTP header-probe implementation. Failures retain return codes/logs and
are not replayed automatically.

The approved URL entry samples remote download bytes, but its sampler stops
before pull-back. Its local sampler value is **not** a measured pull-rate
proof. Acquisition records `NA` distinctly, checks actual wget progress rates
when available and records full-command throughput with its limited scope.
Both remote wget and serial local rsync receive the configured rate limit.

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/acquire_public_corpus.py review --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-acquisition/source-v1
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/acquire_public_corpus.py freeze-reviewed --source /hf3fs-jd/data/tmp/openopt-release-20261010/public-acquisition/source-v1 --source-sha256 CALLER_PINNED_SOURCE_JSON_SHA256 --acknowledge-benchmark-only --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/public-acquisition/real-v1
```

Inspect the extracted LICENSE and README completely between these operations.
Existing output directories are refused. Archive extraction never uses
`extractall`, executes upstream code, or extracts broad directories. It checks
commit-prefixed paths, traversal, duplicates, links/devices, selected file sizes
and exact byte readback. Archive, commit API JSON, selected source bytes,
provenance and frozen arrays have explicit hashes.

Real-problem acceptance independently counts raw ROWS/COLUMNS, checks frozen
dimensions, coordinate membership, LP Hessian, offset and nonnegative variable
bounds, and replays the caller-pinned frozen manifest. Source provenance,
license limits and source/array hashes are external machine-readable evidence.
Unknown dataset license and no redistribution are never silently upgraded.

## Tests

Offline CPU tests create synthetic archive/format fixtures externally to check
safe extraction, path/link/duplicate/size/empty corruption, URL/hash/rate gates,
missing execution chain, no automatic retry, permission acknowledgement and
independent model semantic decoys. These fixtures do not establish public
corpus coverage. The live acquired archive and actual imported public LPs are
separate acceptance evidence, not mocked transport success.

The only importer compatibility change for acquired Netlib files is the
strictly recognized `NAME <name> SIZE: N=<integer>, M=<integer>, NZ=<integer>`
non-mathematical annotation. Original bytes remain unchanged; unknown NAME
arguments are still rejected and all mathematical reader comparisons remain
exact. Integer/nonconvex/RANGES/unsupported rejection policies are unchanged.

## Acquired Evidence

Official tag `v1.15.1` resolved to source commit
`04024d701f79feb8e2f18bc3df0dffc04ef05088`. The archive source URL is
`https://codeload.github.com/ERGO-Code/HiGHS/tar.gz/04024d701f79feb8e2f18bc3df0dffc04ef05088`;
its 3,061,980 bytes have SHA256
`2e121a70ae00f56db8e5ed993e4046c247248fa6280d839ff4808fa958971f0d`.
Commit API response SHA256:
`1e30d59a2f4dca1d71a04ce09a1a44e864780e0cf055da23f5f31757e939090c`.
Root LICENSE.txt and README.md were read completely; the selected locations
contain no instance-specific license/readme. No MIT coverage of data is inferred.
Repository provenance is official/public; historical Netlib authorship and
dataset license are not independently authenticated.

All paths below are relative to
`/hf3fs-jd/data/tmp/openopt-release-20261010/public-acquisition/`:
`source-v3/source.json` is the approved review manifest, SHA256
`0f34c085305e86325e40194123dabf835a9caad55a28fead2b17cb983a431183`.
`real-v1/acquisition.json` records independent raw dimension/coordinate,
LP Hessian, zero offset and nonnegative variable-bound checks.
`real-v1/frozen/corpus.json` SHA256:
`3a434edfa8165167f56be25732840f03bbace1a5f5b7d1de7cdab989673d3f0f`.
It replays `VERIFIED`, **two public-declared cases, zero qualified cases**.

| Case | Variables | Constraint Rows | A Nonzeros | Offset | Input SHA256 |
| --- | ---: | ---: | ---: | ---: | --- |
| afiro | 32 | 27 | 83 | 0 | `9cd304f02717cbd6f85068cb777b69d28539b22a4868ae0f0fb425f514f0eea5` |
| adlittle | 97 | 56 | 383 | 0 | `ed99da009e35279828219ff7f04a2cd4f170692bedf58f704b622338e4adc1f9` |

Constraint counts exclude the objective N row; the NAME SIZE metadata includes
it. Per-case source URLs include the full commit in `real-v1/input.json`.
Original bytes and NPZ arrays are under `real-v1/frozen/<case>/`.
No raw bytes or frozen data enter Git. Both instances are LP, not real QP
coverage and not evidence of representative industrial coverage.

The download chain was not missing any tool and installed nothing. Commit
metadata was downloaded once and the archive once. `source-v1` preserves a
post-transfer false positive caused by reading policy text "8MB/s" as a
measurement. The rate ruler now only accepts actual wget saved-summary lines
and has positive/negative controls. `source-v2` preserves the subsequent
`missing required archive members: LICENSE` first cause: actual path was
LICENSE.txt. Both corrections reused caller-pinned local bytes in `source-v3`,
with no network reattempt.

Archive runtime sampler reported remote maximum **0.525 MiB/s** and wget
saved-summary average **380 KB/s**. Explicit upstream/downstream rate was
512k, serial. Pull peak speed was not independently sampled by the legacy tool;
its local sample zero is not reported as proof of measured zero traffic.

Final CPU acceptance completed **2026-10-10 09:07:05 UTC**: **92 passed**,
zero errors/failures/skips (30 acquisition gates plus 62 importer regression
tests). Report/log/JUnit:
`final-tests/tests-8zp06be5/`. Post-edit smoke: **79 passed, four explicit
CUDA-unavailable skips**, `final-smoke/smoke-dwrddw4a/`. Both verification JSON
files were reread with PASS; full repository and GPU execution were not run.
Frozen real corpus was independently replayed again at 09:07:18 UTC,
`VERIFIED`, case_count=2.

Tested source SHA256:
`scripts/acquire_public_corpus.py` =
`4e0d2e2aa01843c78878fe91230a82f4d4d226974948a59fc76b53a68a95f478`;
`scripts/public_qp_corpus.py` =
`f34c1b071aa4bf018f325927f205b4572dc45577712db33bbec0faafd1ff01a7`;
`tests/test_acquire_public_corpus.py` =
`192b7b2751cc1001beeaee76b7a40e43c16e1cdc1a2321a97c90a8cf64d81cb3`.
The coordinator may use `real-v1/frozen` and its caller-pinned corpus hash for
independent solving immediately; this seat makes no solver/commercial/
industrial qualification claim.
