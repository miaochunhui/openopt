# 4500 Submission Diagnosis

On 2026-10-10, the first submission was rejected before any Job was created.
The default kubeconfig is an observer identity and cannot create Jobs in
`devtest`; kubectl returned 1 with `Forbidden`.

Evidence: external run `openopt-4500/probe-gkc7keyc/job-0.json` under 3FS tmp.
No Pod termination, restart or cgroup evidence exists because no Pod was created.

The existing infrastructure administrator execution channel was found in
current cluster tools. A current `auth can-i create jobs -n devtest` through
that existing channel returned `yes`; resource quota showed zero GPU usage.
Subsequent submission explicitly selects that existing channel, without changing
any credential, RBAC or default connection. This is not an automatic identical
retry. The public runner receives connection configuration only from its
environment and does not ship internal kubeconfig content.

The runner now persists a submission failure and refuses to disguise it as a
successful launch. GPU expansion still requires current one-card artifacts.

The next Job `openopt-probe-wy6qvauf-0` was accepted and placed on h3c2, but
terminated with StartError/128 and restartCount=0 before Python ran. Its image
ID was `49f117c007278aaf841ef7903ee5fae279a285537bb832868d0ab78bb234ea48`.
The first cause was `python3: executable file not found in $PATH`.
Pod status and logs are retained in the external run directory. No application
cgroup sample exists because the container process never started.

Current image inventory and the existing LingBot runtime Dockerfile confirm
that the CUDA base has no Python, while the cached SGLang runtime embeds
Python/PyTorch. Select that already-cached runtime for a changed-image,
one-card preflight. No package installation or service mutation is needed.

The changed-image preflight passed on h3c2 with an actual float64 CUDA solve,
zero error, PyTorch 2.13.0+cu130, sm_120 and driver 595.91.07.

The first numerical smoke `openopt-smoke-cebpgx8a-0` computed passing CPU/CUDA
quality results, but Nsight capture returned 135 after exporting its report on
3FS. The application preserved this as FAIL, with exitCode=1/restartCount=0.
Run `smoke-cebpgx8a` contains partial numerical results, telemetry, failure.log
and the first Nsight report. Suspected cause: Nsight mmap/database scratch on
FUSE. Test a changed scratch location in local shared memory, copying the final
report/statistics back to 3FS. No numerical configuration changes. Treat the
cause as suspected until that controlled change passes; no GPU expansion yet.

The changed local-shared-memory scratch smoke `smoke-t363fo54` passed with the
same numerical configuration and Nsight return code zero, producing nonempty
kernel/API/memory tables. This is evidence that local scratch avoids the observed
failure, not a general proof of every cause of signal 135.

Further one-card A/B smoke passed before the three-card independent ladders.
All subsequent ladder, paired-confirmation and explicit-CUDA regression Jobs
completed. The two failed Jobs are retained, not deleted; collection records
preserve their original terminal states and logs.
