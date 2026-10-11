# Bounded Offline SPDX Delivery

## Source And Standard

Base: `aed5f343dd235ae01196fe83090c593c89131d63`. Only new
`scripts/build_sbom.py`, `tests/test_build_sbom.py` and this document are owned.
Existing package metadata, solvers, exports, mainline and infrastructure are
unchanged. Generation and reading use no network, SDK imports, licenses,
credential files, environment credential discovery or dependency installation.

The output is SPDX 2.3 JSON, validated with the official SPDX 2.3.1 schema
using the already installed `jsonschema` Draft201909Validator. The official
schema is caller supplied offline and must have SHA256
`4126dc29f15e92feec92e9622f4939131ad58b125d6f578aa93e980a6e6f3212`.
It has no external references. This is schema plus bounded package/relationship
validation, not a general SPDX tooling or legal compliance certification.

Primary sources:

- SPDX 2.3 specification:
  <https://spdx.github.io/spdx-spec/v2.3/>
- Official versioned schema:
  <https://raw.githubusercontent.com/spdx/spdx-spec/v2.3.1/schemas/spdx-schema.json>
- SPDX package license declaration rules:
  <https://spdx.github.io/spdx-spec/v2.3/package-information/#713-declared-license-field>
- Python package Core Metadata (Name, Version, Requires-Dist,
  License-Expression, Provides-Extra):
  <https://packaging.python.org/en/latest/specifications/core-metadata/>
- Wheel RECORD and installation:
  <https://packaging.python.org/en/latest/specifications/binary-distribution-format/>

Schema acquisition used the existing Astrill URL entry, rate `500k`, one
attempt; 47,602 bytes, both transport SHA256 values matched. Its official
upstream origin and version are recorded, not independently signed upstream.
The generation tool itself never fetches the schema.

## Exact Coverage

The complete caller-selected Git source membership is checked byte for byte
against committed blobs, including index flags hiding edits. Empty source,
untracked changes, nonordinary files and drift refuse. Source checksum is the
canonical membership manifest hash, explicitly not an archive hash.

The project declares 0BSD separately from dependencies. Each selected dependency
is read from an explicit directory's actual dist-info/METADATA, never imported.
PEP 508 markers, explicit target Python, selected extras, versions and transitive
Requires-Dist are evaluated within bounded 256-package/512-edge scope. Build
requirements and unselected extras remain declarations rather than an assertion
that they were used in a build. Optional absent distributions are distinct from
missing active dependencies. Missing or version-incompatible active dependencies
produce INCOMPLETE and nonzero exit; malformed/ambiguous input produces FAIL.
Direct-URL dependencies are explicitly unsupported, never fetched.

Metadata License-Expression is retained only for the small declared SPDX-ID
grammar supported by this tool. Unknown identifiers and absent or legacy License
text produce NOASSERTION; legacy text is hashed, not copied or guessed as 0BSD.
Unsupported expression syntax fails loudly. All dependency licenseConcluded
fields remain NOASSERTION. Schema dataLicense CC0-1.0 is for SPDX metadata and
does not relicense project or dependencies.

Dependency package checksums cover METADATA bytes only. Dependency binaries,
vendored libraries, CUDA drivers, SDKs, system packages and other unlisted
runtime distributions are NOT_ANALYZED/NOT_SAMPLED. Runtime execution and legal
license qualification remain NOT_TESTED/UNVERIFIED regardless of schema PASS.

An optional actual wheel is bound by caller SHA256, its exact RECORD membership
and hashes, Name/Version/license/dependency declarations and original package
source bytes. Installed package and wheel metadata files must match the wheel.
Installer-generated RECORD and entrypoints are not build attestation. A historical
wheel whose package bytes match the current source is only PACKAGE_BYTES_MATCH_ONLY;
its source commit provenance remains UNVERIFIED_WITHOUT_BUILD_ATTESTATION.
The existing813 wheel is never relabeled as an aed5 build.

## Execution

Use the existing base interpreter, which already has packaging, pytest and
jsonschema; do not install into the validation environment. Output directories
must be fresh and external shared3FS paths. Existing outputs refuse overwrite.
Generation failure retains failure.json with fixed first-cause codes, no raw
metadata/child stdout. Receipts pin the SPDX and evidence bytes; the reader
rechecks schema, source and metadata/wheel/installed hashes.

```bash
sudo -n -u infra /home/infra/anaconda3/bin/python -B \
  scripts/build_sbom.py --source-root /data/dev/openopt \
  --source-commit aed5f343dd235ae01196fe83090c593c89131d63 \
  --metadata-root /hf3fs-jd/data/tmp/openopt-validation-env/lib/python3.10/site-packages \
  --metadata-root /home/infra/anaconda3/envs/torch28cu128/lib/python3.10/site-packages \
  --target-python 3.10 \
  --schema /hf3fs-jd/data/tmp/openopt-release-20261011/sbom-delivery/standard/spdx-2.3.1-schema.json \
  --schema-sha256 4126dc29f15e92feec92e9622f4939131ad58b125d6f578aa93e980a6e6f3212 \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261011/sbom-delivery/FRESH_SOURCE
```

The same entry adds `--wheel`, `--wheel-sha256`, `--installed-root` for existing
external wheel/installation inputs. `--read-receipt receipt.json
--receipt-sha256 CALLER_SHA256 --schema ... --schema-sha256 ...` performs no solve
or network. Reading an INCOMPLETE receipt preserves nonzero qualification state.
Test CLI uses `tests/test_build_sbom.py --artifact-dir FRESH_TEST_DIRECTORY`,
real local Git and wheel fixtures, bounded collection and JUnit double-set checks.
Actual generation/reader output, not numbers added later to this frozen document,
is authoritative. Development receipts do not establish clean-source delivery.
