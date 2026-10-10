# Research Git Delivery

Research-code synchronization is separate from commercial qualification and
managed docs-site publication. A clean research branch may disclose incomplete
capabilities; a remote ref is not an online solver, published webpage or a
commercial benchmark result.

`scripts/audit_git_delivery.py` reads every reachable outgoing object relative
to a full caller-pinned ancestor, including obsolete blobs and commit messages.
It compares known local credential material in memory and bounded credential
shapes. It emits only object identities, hashes, sizes and hit counts, never
matched content or credential tails. The existing scanner selftest is required.
Empty histories, missing objects, oversized objects, missing scan material and
source drift fail loudly. Logs/receipts are external.

Raw reads explicitly disable Git replacement objects. Shallow repositories and
legacy graft files are rejected because their traversal is not a complete raw
commit/tree graph. Current ordinary source files are compared with actual
committed blob bytes before and after scanning; index flags cannot hide drift.
Missing repositories are caught inside the terminal receipt boundary. The
independent review reproduced four replacement false accepts, two hidden-source
false accepts and one missing terminal receipt on the original implementation;
those failed receipts remain historical FAIL. New owner controls and an unchanged
independent replay are required for the repaired source, rather than retroactively
relabeling its earlier scan.

The bounded shape detector is not a universal secret classifier. A PASS does
not authenticate all local credentials, source-origin rights, dependencies,
data redistribution or unknown proprietary content. The coordinating owner
must separately review the selected history and its original migration
boundary. This tool never fetches, pushes, changes refs or handles remote
credentials.

Run with the existing infra identity, a clean repository, full `--base` and
`--head`, and an external `--artifact-dir`. The original remote destination
remains the user's Gitee repository. Re-read the destination ref immediately
before a separately authorized non-forced push. Use an explicit source and
destination branch; never force, rewrite history or bundle the infra source
repository. Read the remote ref after the actual push and compare the exact
commit. Record unresolved commercial/GPU/general-solver/publication gates.

`--scan-only --artifact-dir EXTERNAL` is a separate precommit mode for exactly
the tool, its owner test and this document. It does not inspect outgoing history
or assert source consistency. Invented credential-shaped test payloads are
assembled at runtime, not stored verbatim in repository blobs.
