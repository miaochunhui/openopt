# Repository Maintenance

## Authority

The user selected GitLab on the R640 infrastructure as the primary home.
The existing OpenOpt history is preserved; infra history, development branches,
external datasets and benchmark artifacts are not part of this delivery.

| Role | Repository | Local remote |
| --- | --- | --- |
| Primary | `https://gitlab.ai4everyone.top/infra/openopt` | `origin` |
| Secondary | `https://gitee.com/sdfwefqaa/openopt` | `gitee` |
| Public secondary | `https://github.com/miaochunhui/openopt` | `github` |

The GitLab project was created private. No visibility change was requested or
performed. Repository synchronization does not qualify the solver, publish the
documentation site or close outstanding commercial/GPU acceptance gates.

## Synchronization

Maintain the primary branch on GitLab. Secondary copies receive the same
explicitly pinned, content-audited commit using ordinary fast-forward pushes.
Do not use force, mirror, all-branches or implicit tag pushes. Existing
development worktrees remain local and independently owned.

`master` tracks `origin/master`; the old Gitee tracking refs were retained
under `gitee`. The three remotes are separate: a normal push is not an atomic
three-host operation. No cron, hook or server-side background mirror was
installed. Read each destination's branch after pushing and compare the full
commit identity. A failed or blocked destination remains incomplete.

On the existing infra workstation, use the already verified internal GitLab
SSH channel. It selects the existing GitLab key and previously trusted host
identity without modifying global SSH configuration or known-host records:

```bash
sudo -n -u infra env \
  GIT_SSH_COMMAND='ssh -i /home/infra/.ssh/id_ed25519_gitlab -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o HostKeyAlias=192.168.1.181' \
  git -C /data/dev/openopt ls-remote origin refs/heads/master
```

After auditing the exact intended commit, push that full identity to
`refs/heads/master`, with `-c push.followTags=false`. Gitee uses the existing
infra SSH configuration. Do not put access tokens in URLs, arguments, Git
configuration, source or receipts.

## Pending Boundaries

The earlier credential authenticated a different account; its failed receipts
and file remain unchanged. The user supplied the matching account file on
2026-10-11 through the existing credential-file process. The exact account and
target were checked before creating the public empty `miaochunhui/openopt`
repository and performing the first ordinary master push at 20:36 +08:00.
No alternate account repository was created and no credential was changed.

The public GitLab SSH endpoint failed host-key verification. Its old trust
record was not removed or overwritten. The verified internal channel is used
instead; the public endpoint requires separate trusted host-key reconciliation.

Earlier WIP repository-sync tools failed independent safety controls and were
not adopted. Their failures and isolated development files are retained.
The separately reviewed `scripts/deliver_github.py` is chair-operated, not
an automatic mirror. It requires an exact clean master, a pinned complete-history
content-audit reader and matching account/target permissions. It transports
only the master bundle through the existing infra@astrill channel, refuses
divergent history and verifies the destination master after an ordinary push.
Credentials stay in memory/stdin, not arguments, URLs, persistent Git
configuration or receipts. Git authentication uses a process-only extra header
configuration delivered through the isolated child environment. Malformed
arguments produce a fixed content-free error.

Local remote configuration was backed up externally before migration.
Configuration content was byte-verified despite the shared filesystem refusing
permission-metadata preservation. Restoring local configuration cannot undo
remote pushes; no remote rollback or history rewrite is authorized.

## Read-Only Ref Receipt

`scripts/inspect_repository_refs.py` verifies the configured destinations,
the full caller-selected clean local HEAD and the two live master refs.
It writes a create-once external receipt. It never reads the GitHub credential,
contacts GitHub, changes configuration, creates a repository or pushes.
This no-GitHub-credential-read statement applies to the ordinary ref entry.
The separate `--scan-owned` development entry reuses existing credential needles
in memory; its receipt explicitly sets `existing_scan_credentials_read=true`.
Neither path records credential values. Ref inspection checks local master and
HEAD before and after the two remote queries.
`PRIMARY_AND_GITEE_PASS` is deliberately not three-host completion:
`github.state=NOT_VERIFIED` and `three_host_complete=false`.
Use a full 40-character commit, not a moving branch name:

```bash
sudo -n -u infra python3 -B /data/dev/openopt/scripts/inspect_repository_refs.py \
  --repo /data/dev/openopt --expected-head FULL_COMMIT \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-repository-refs/FRESH_RUN
```

On 2026-10-11 the existing trusted internal GitLab channel returned hostname
`r640-61`. This establishes the observed R640 host, not a current HA failover
test. At 19:23:34 +08:00 GitLab and Gitee master were both
`813e2e03a2e5e6f722928f1ad7baeacdd24421be`, after an explicit audited
fast-forward. Later receipts supersede this dated ref snapshot.
GitHub remains pending, and no automatic three-host mirror was installed.

That dated GitHub status is superseded by the 20:37 +08:00 readbacks on the
same day. Local, GitLab, Gitee and anonymous GitHub master were all
`aed5f343dd235ae01196fe83090c593c89131d63`. The first-delivery and separate
read-only receipts are recorded in `docs/handoff/release4-20261011-2037.md`.
Later commits require their own exact-head audit and destination readbacks;
the first-delivery receipts must not be relabeled as proof of a later version.
No automatic mirror or HA failover qualification was established.
