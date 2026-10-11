# Exact GitHub Target Permission

An account mismatch is not itself proof that the requested repository denies
collaborator access. This narrow inspection supplements the retained identity
receipt. It never treats token scope, account ownership or read access as push
permission. HTTP404 means missing OR inaccessible, not proof of absence.

`scripts/inspect_github_target.py` copies its public, source-addressed code to
the existing infra@astrill channel, verifies remote bytes before credential read,
then verifies GET `/user` is `miaochunhui` and queries only
GET `/repos/miaochunhui/openopt`. The user supplied the matching account file
`/home/mch2/.passwd_github_miaochunhui` on 2026-10-11; the old file is unchanged.
The credential stays in
stdin/memory, not URLs, arguments, configuration or receipts. No API writes,
Git operations, new identity, services or changes to trust/permissions occur.
Responses contain a fixed whitelist of target, HTTP status and boolean push
permission. Source changes or inconsistent responses fail.

Run with the existing validation interpreter and a fresh `--artifact-dir`
under `/hf3fs-jd/data/tmp/`. The create-once receipt and execution log are external.
`WRITE_PERMISSION_VERIFIED` is only access discovery, not a push/readback or
three-host completion. Missing access requires the owner to provide matching
credentials via the original file or grant the authenticated account access.
Do not create an `asdfadf-dsf/openopt` substitute.

`scripts/deliver_github.py` is a separate explicitly authorized writer.
It requires a pinned complete-history reader, exact clean master and current
source hashes. It validates the matching account, checks a complete fetched
bundle with real fsck, creates only the user's public empty `openopt` if absent,
and performs only a pinned ordinary master push. It isolates inherited Git
configuration, rejects unsafe local settings, disables redirects and compares
the destination master after pushing. No tags or development branches are pushed.
The worker returns only bound fields; fixed failures retain their first cause.
`--scan-owned` reads established local scan material and both user-specified
credential files in memory, records counts/hashes only and performs no network.
This content inspection is not proof of unknown-secret absence or legal rights.

Offline controls use synthetic input only, no actual token/SSH/API access.
The reference is GitHub's official REST "Get a repository" documentation:
`https://docs.github.com/en/rest/repos/repos#get-a-repository`.
