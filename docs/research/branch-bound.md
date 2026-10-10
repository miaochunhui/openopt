# Tiny Native Bounded MIQP

Implementation scope on 2026-10-10, base acc1993: only branch_bound.py,
test_branch_bound.py and this document. Default exports/solver behavior are
unchanged. Import this separate module and explicitly set enabled=True.
This is not a general/commercial MIP optimizer or an enumeration speed claim.

## Model And Proof

Minimize f(x)=0.5*x.T*P*x+q.T*x subject to l<=A*x<=u and finite L<=x<=U;
designated variables are integers. Real dense original float32/64 arrays,
1<=n<=8, m<=64, at least one integer; exact integer ceil/floor endpoints
within [-2**24+1,2**24-1]. P must be exactly symmetric and have exact
nonnegative row margins Pii-sum(j!=i,abs(Pij)). The identity
x.T*P*x=sum(margin_i*xi**2)+sum(i<j,abs(Pij)*(xi+sign(Pij)*xj)**2)
proves this sufficient PSD subset. Other PSD matrices return unsupported via
NotImplementedError, not nonconvex/infeasible. Invalid data raises ValueError
or TypeError; execution/source/arithmetic failures raise, never retry.

For any finite proposed z,y, define h(y) as yi*ui for yi>0, yi*li for yi<0,
and zero for yi=0. Every nonzero dual must select a finite signed endpoint.
Convexity and y.T*A*x<=h(y) give the original-model bound
B=-0.5*z.T*P*z-h(y)+sum(min(ai*Li,ai*Ui)), a=P*z+q+A.T*y.
No feasibility, inverse, regularization, stationarity or QP status is required
of that witness. Identity-box duals are discarded; boxes are counted once.
The explicit zero-dual seed and candidate tangents are separate witnesses,
not a fallback after a broken signed QP witness. Both endpoint products are
evaluated. Verified parent bounds can strengthen child bounds.

All PSD, row support, scalar bounds, candidate feasibility, objectives,
integer endpoints and signed gaps use exact Fraction of original binary64
values, capped at 8192 numerator/denominator bits after each scalar operation.
Proof values must also have finite binary64 enclosures; overflow raises.
Directed nextafter encloses the exact rational only AFTER exact comparison;
it is not a claim about GPU arithmetic. This host scalar validator is costly
and limited, not an arbitrary-precision optimizer. Costs are recorded for QP,
host coordination/proof, and independent replay, including data transfers.

## Search And Acceptance

Native PyTorch QP proposals use the explicit requested device/dtype. Wrong
placement/dtype, unavailable CUDA, nonfinite output or failure status raises;
there is no third-party solver/CPU fallback. Nonconverged finite max_iterations
proposals remain heuristic witnesses and cannot declare infeasibility.
Finite original row bounds cannot overflow to infinity during proposal casts.
Endpoint candidates and integer-rounded proposals undergo strict original
primal checks. Continuous values are never rounded. Original source/model
bytes are frozen and checked after solve. Results retain binary64 incumbents.

Deterministic first non-singleton integer coordinate is split at proposal
floor if interior, otherwise exact domain midpoint. Children cover all integer
values disjointly; continuous boxes are never discretized. Pure singletons
can close using strict primal/objective proof. Mixed fixed-integer leaves
remain unresolved unless their exact lower bound closes against incumbent;
an approximate feasible point cannot prove continuous optimality. In
particular, 3*x_cont=1 may have no representable accepted incumbent.
The same rule covers unvisited fixed-integer mixed frontier leaves at limits;
large user gap tolerances cannot bypass an unclosed continuous leaf.

Every frontier, unresolved and terminal non-infeasible region retains a
verified bound. Prune only LB>=strict incumbent objective. Global LB is the
minimum bound in this complete partition. Signed gap=objective-LB must not
be negative. Relative gap uses max(1,abs(objective)); thresholds are exact
finite nonnegative abs_gap+rel_gap*scale, with no hidden tolerance.
Statuses: solved_bounded for covered declared gap without unclosed mixed
leaf; feasible for verified incumbent at unresolved budget; unknown for no
incumbent or unclosed mixed leaf; infeasible_bounded only for fully proved
empty/row-infeasible coverage. No unbounded/ray/optimal label.
QP iteration-limit counts, node-limit and wall-budget causes remain explicit.
The scalar incumbent objective is an exact rational upper bound, not a
nearest-rounded float. Reported binary64 intervals enclose that rational;
the exact rational signed gap/threshold controls acceptance.

check_bounded_miqp_result takes caller model/config/trusted current source
hashes and recomputes bounds/candidates/PSD/tree/gap without solve/factory.
Refreshed outer hashes cannot repair inflated bounds, fractional incumbents,
lost children, wrong infinite signs, source/model/device or gap mutations.
Numerical/coverage PASS remains execution UNVERIFIED: bare metadata cannot
attest actual CUDA. Actual GPU kernel trace tests are provided but skipped
when unavailable; only the main agent may submit GPU. Performance/commercial
qualification remain UNVERIFIED even after numerical acceptance.
Model/config binding compares canonical typed JSON, not True==1 dictionary
equality. Loaded QP dependency paths must match this frozen source root.

## Reproduction

Use the existing validation environment, CUDA hidden and one BLAS/OMP thread.
Run scripts/verify.py tests --target tests/test_branch_bound.py with external
--artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/native-miqp/cpu.
CUDA entry is tests/test_branch_bound.py::test_cuda_frozen_bound_tree, for main
dispatch only; it freezes original arrays, solutions/tree and a kernel trace.
Tiny enumeration appears only as a test oracle. Self-authored analytic pure,
mixed, singular, LP, equality, limit, overflow and coordinated negatives need
no public dataset, SDK, commercial license or historical 3FS fixtures.
One actual Clarabel CPU reference checks the same original mixed objective
on its accepted fixed-integer continuous face. It is test-only numerical
control, not native solve fallback or a commercial/global MIQP certificate.

Pre-smoke: pre-smoke/smoke-cuq64_1o/verification.json, 80 passed/four CUDA
skips, 8.15s, source consistency PASS. This precedes implementation and is
not MIQP qualification. Actual implementation receipts are external under
/hf3fs-jd/data/tmp/openopt-release-20261010/native-miqp; failures stay retained.
First implementation CPU run cpu-v1/tests-nbbpu8ue/verification.json:
48 passed/one CUDA skip, 6.65s, development source consistency PASS. No failure
occurred in this first bounded run; clean-source acceptance remains separate.
The second development CPU run cpu-v2/tests-qk2johgd/verification.json had
57 passed/one CUDA skip, 7.17s; post-smoke/smoke-qeb6fgkp/verification.json
had 80 passed/four CUDA skips, 8.25s. Later cast/receipt edits require new
receipts. A final patch attempt reported a missing document context after
applying the two code/test hunks; live reads confirmed those exact writes,
then a separate document-only patch completed this section. No files outside
the three owned paths changed; no test failure was suppressed.
Final development cpu-v3/tests-2cbp3avx/verification.json: 63 passed/one CUDA
skip, 7.73s. post-smoke-v3/smoke-f60zl1vf/verification.json: 80 passed/four
CUDA skips, 8.46s. Both source consistency PASS; neither is clean committed
source acceptance. Frozen clean verification and independent review remain
separate gates; no actual CUDA or commercial solve was performed here.

## Retained Numerical First Cause

frontier-first-cause/tests-1g1gt6vu/verification.json retained one failed
test, 3.45s: with max_nodes=1 and abs_gap=1, the original mixed analytic model
returned solved_bounded even though its fixed-integer continuous children
were unvisited frontier leaves. The gap was a valid global enclosure, but
the conservative mixed-leaf status contract was violated. The summary now
checks all covered mixed fixed-integer leaves, including frontier, not only
visited unresolved leaves. The unchanged counterexample and a forged-status
replay negative remain hard tests. No bound/primal tolerance was relaxed.
