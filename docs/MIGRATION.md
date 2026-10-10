# OpenPO Extraction

Migration date: 2026-10-10 (Asia/Shanghai).

## Source And Boundary

- Source repository: `/data/dev/optimizer`, not the infra working tree.
- Source commit: `a0300f07efaf9669e18d236f71c739490d3afe2a`.
- Destination: `/data/dev/openopt`, remote `git@gitee.com:sdfwefqaa/openopt.git`.
- Method: a new, independently reviewed snapshot, not a mirror of source history.
- All `src/openpo` Python files are preserved byte-for-byte in the initial
  import commit. Independent fixes are separate follow-up commits.
- Selected source tests, synthetic benchmark tools and MPO specifications are
  preserved byte-for-byte. The complete machine-readable lists and SHA256
  digests are in `migration-manifest.json`.

Existing source changes are not imported. The source repository, its branches,
its index and its worktree are not modified.

## Exclusions

Source history, secrets, environment files, deployment manifests, private
production adapters in `deliverables`, binary delivery bundles, third-party
commercial PDFs, logs and historical experiment outputs are excluded.

Tests of excluded deployment/delivery code, the old iteration gate and the old
historical-report pipeline are excluded with a per-file reason in the manifest.
Core numerical, API, shadow-service and multi-period tests are retained. This
is not a claim that every source-repository test has been migrated.

## New Maintenance Surface

`openopt` is the distribution/project name; `openpo` remains the implementation
namespace. The `openopt` top-level API reexports the existing public objects.
This preserves identity and avoids a risky mechanical rename of solver code.

The new verification entry runs without the old source repository or its
benchmark artifacts. Imported specifications are historical research contracts,
not an endorsement of production deployment or a newly certified result.

## Licensing And Release

The selected source snapshot contains no `LICENSE` declaration. No permission
grant or attribution has been invented in this migration. Public open-source
release is blocked until the rights holder selects a license and confirms the
redistribution rights of the imported code. Dependencies remain under their
respective licenses and are installed separately, not vendored.

## Reproduction

```bash
python3 scripts/migrate_openpo.py --provenance
python3 scripts/iteration_gate.py smoke
python3 scripts/verify.py
```

`--check` verifies the exact imported working snapshot. After independent edits,
use `--provenance` to verify imported bytes in the repository's initial commit,
while the regression suite validates the current implementation. The normal
verification entry uses provenance, so legitimate algorithm edits are allowed.
Both modes use only this repository; provenance requires its initial Git commit
and therefore does not work in a history-free source archive.
Reimporting requires the explicit original repository and fixed commit:

```bash
python3 scripts/migrate_openpo.py --source /data/dev/optimizer --apply
```

Import is fail-loud and refuses divergent destination files. Do not use it as
a synchronization mechanism after independent development begins.

## Independent Follow-Up

The unchanged source baseline revealed a terminal-publication race in the
durable shadow service: a divergent job becomes `SUCCEEDED` before its route
circuit is persisted. A deterministic delayed-publication regression reproduces
the race. The initial commit preserves imported source bytes; the following
maintenance commit fixes this issue only in openopt, without changing optimizer.
Audited circuit resets also fence delayed publication or replay of the same
divergence evidence; a new divergence can still open the circuit.
