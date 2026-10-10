# openopt Development Ledger

## Established Foundation

- Extract the complete OpenPO implementation without modifying source history.
- Preserve the `openpo` public API and provide an `openopt` compatibility entry.
- Provide independent correctness verification and explicit migration evidence.

These items are accepted only by current `scripts/verify.py` output, not by the
presence of files or old benchmark claims.

## Next Development Stages

1. Licensing: obtain the rights holder's license and redistribution decision
   before making an open-source release claim.
2. General model: specify a domain-neutral LP/QP problem contract, quadratic
   operator interface, constraints, dtype/device semantics and result status.
   Keep portfolio adapters outside the general core.
3. Correctness: compare the same problem formulation with OSQP and Clarabel;
   cover infeasible, unbounded, degenerate, empty and non-finite inputs.
4. GPU qualification: measure cold/hot, p50/p95, success rate, objective gap,
   residuals and hardware/software versions on shared paired cases.
   Decide CPU/GPU routing from measurements, not a GPU-first assumption.
5. Distribution: build/test a clean wheel and add CI for the supported CPU/CUDA
   matrix. Add discrete optimization only after its contract and SCIP/HiGHS
   reference tests exist.

## Explicit Non-Claims

- No general-purpose `minimize` interface is implemented in this extraction.
- No universal GPU speedup or production qualification is asserted.
- No cluster, service, credential or networking changes are authorized here.
- Historical research decisions are not current runtime validation.
