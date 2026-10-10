# Retained Cluster Failure Recovery

`scripts/verify_cluster_recovery.py` closes only the observed refinement
snapshot packaging failure. It never qualifies numerical CUDA execution,
performance, commercial capability or publication.

The original failed run is retained with exit 1, restart count 0 and seven
JUnit failures whose actual first cause is `FileNotFoundError` for
`docs/research/refinement.md`. A new changed-source run must contain the
document, pass the same seven tests without skips, exit 0 without restart,
use the same image and preserve all solver/checker/test bytes. Both runs
are bound to coordinator-collected Job/Pod UIDs, ownership, exact command,
resource selection, results, JUnit and logs. GPU quantity serialization
accepts only the exact integer/string-equivalent value, not bool or a
different quantity. Failed runs need not contain success-only counts or
telemetry; their actual JUnit supplies the full failure denominator.

Every input is hash-bound in `recovery.json`. Caller-pinned replay recomputes
the complete report. Live acceptance can take `--recovery REPORT SHA256`;
it verifies every original input before attributing only the corresponding
actual failed Pod UID, Job UID, owner and unchanged exit/restart condition.
A name-only historical label cannot close a new failure.

The accepted external report is:

```text
/hf3fs-jd/data/tmp/openopt-release-20261010/failure-recovery-real-v3/recovery.json
cd911cf2cbe4f1488511157b02cb8c81c33b6ad740ea5f6437915bf4dbbb6610
```

At `2026-10-10T09:52:36.203604+00:00`, replay returned
`CLUSTER_RECOVERY verdict=RECOVERY_VERIFIED`. The original solver/test bytes
were unchanged, and the changed-source recovery had seven passed CUDA
tests on the authorized hardware. This is a narrow recovery fact, not final
same-source extension qualification.

Initial verifier attempts failed because Kubernetes serialized the GPU
quantity as a string, then because tuple case identities did not survive
JSON roundtrip. These verifier failures were not solver regressions and
did not change original run evidence. The first attempt produced no report;
the second unaccepted report remains in `failure-recovery-real-v2`.
The corrected verifier has negative controls for wrong failure causes,
case membership, skips, duplicate JSON, nonfinite JSON, changed solver/test
bytes, quantity mismatch and a different failed Pod/Job/exit condition.

Record and replay use the existing validation interpreter, external paths
and infra identity. Replay does not write or submit anything:

```text
verify_cluster_recovery.py verify --report REPORT --sha256 CALLER_PIN
```

No existing Jobs are deleted and no permission, credential, service or
production resource is changed. Remove the optional recovery input to
restore the previous strict UNKNOWN result for an unrecognized failure.
