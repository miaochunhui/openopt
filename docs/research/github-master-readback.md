# GitHub Master Readback

This adds only a fixed anonymous public master query, not synchronization.
Base aed5f343dd235ae01196fe83090c593c89131d63. Existing local/GitLab/Gitee
inspection remains unchanged and separately evidenced. GitHub master alone
cannot claim three-host completion, automatic mirroring or failover.

Only https://github.com/miaochunhui/openopt.git and refs/heads/master are
supported. Expected and returned heads must be nonzero lowercase 40-digit
Git SHA1. The existing infra@astrill SSH channel is used without changing
login methods, SSH trust or credentials. No GitHub token/helper/login/API,
repository creation, push, fetch or clone is used. SSH uses the existing
authentication channel; the application does not read credential files.

Remote Git runs at / with env -i, fixed PATH, inaccessible HOME/XDG homes,
GIT_CONFIG_GLOBAL=/dev/null, GIT_CONFIG_NOSYSTEM=1, no inherited trace/proxy/
askpass/config injection, helper and extraHeader cleared, HTTPS-only, SSL
verification enabled and HTTP URL redirects disabled. Only an exact single
master line is accepted. Unexpected banners, extra lines/refs, invalid hashes,
empty output and nonzero return codes remain FAIL. Captured subprocess
diagnostics are never copied to receipts or stdout. A missing/not-yet-public
repository is FAIL when actually queried, not treated as successful delivery.

Use only after the chair confirms public delivery. This task performs no
real SSH/network query beforehand. Selftests are offline fixtures, not
remote verification. Create the external parent first; receipt.json is
create-once and reserved before the query, so an existing receipt prevents
any retry. It contains UTC collection times, exact expected/actual master,
return code, fixed failure cause and clean committed tool-source before/after.
Owned blob bytes must match the commit. No raw stdout/stderr is persisted.

```bash
sudo -n -u infra mkdir -p /hf3fs-jd/data/tmp/openopt-release-20261011/github-master-readback/ACTUAL_NEW_RUN
sudo -n -u infra python3 /data/dev/openopt-wt-github-master-readback-20261011/scripts/read_github_master.py \
  --expected-head FULL_DELIVERED_40_SHA \
  --receipt /hf3fs-jd/data/tmp/openopt-release-20261011/github-master-readback/ACTUAL_NEW_RUN/receipt.json
```

The new test CLI scans only the three additions with the existing shape
scanner and a synthetic noncredential sentinel. Known-credential literal
scan is NOT_RUN: no actual credential material is read. Offline tests cover
exact output, fake output, mismatches, missing refs, timeout, dirty/drifting
source and create-once behavior. Author tests need chair acceptance; neither
fixture green nor clean source is an actual GitHub readback.
