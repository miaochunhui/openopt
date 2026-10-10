# Operator Original-Model Acceptance Repair

The independent sparse review found a zero-row contradiction accepted at
iteration 25 when another row had magnitude 1e9. It also found reversed
binary64 bounds and finite/nonzero data lost during conversion to float32.
Those historical failures remain failures.

This repair validates original vector shapes, finite objective data, NaN and
infinity direction, and bound order before conversion. Dense/COO/CSR matrix
values and q/l/u reject finite-to-nonfinite or nonzero-to-zero conversion.
Ordinary representable float32 rounding remains supported.

A scalar working convergence nomination is now followed by original-model
componentwise lower/upper feasibility, stationarity, normal-cone,
multiplier support signs and complementarity. Lower and upper bounds have
separate row scales. Masks precede multiplication with infinite bounds.
Tensor operators retain original binary64 data on the requested device;
sparse products do not densify or form a Gram matrix. Matrix-free callbacks
remain borrowed actions at their declared dtype/device, not exact arithmetic
or an authenticated immutable original model.

The existing approximate result API, CG recurrence, defaults, caller PSD
declaration and explicit-device policy are unchanged. This is not a dual-gap
certificate, a convexity proof, a full sparse solver or commercial qualification.
Extra original products and storage have unmeasured performance cost.

The new author tests and the unchanged independent counterexamples must be
run against a clean freeze, with external logs/JUnit and source consistency.
Independent signoff, actual same-source CUDA and generated reports remain
separate requirements. No historical GPU result is upgraded by this repair.

The first author regression is retained as FAIL under the external campaign
`operator-original-contract/development-v1`: 174 passes, four failures and
six explicitly local CUDA skips. The failures are the old coarse-inner
control's asserted `solved` status, now correctly denied by a componentwise
stationarity limit. At 5000 iterations that control still has stationarity
3.325518260766103e-8; its global threshold was 4.001e-6, but smaller-coordinate
thresholds fail. The precondition comparison now explicitly records the
maximum componentwise excess and requires this coarse arm to remain unsolved.
Its original coarse global diagnostics and failed Jacobi arm are retained;
the unchanged tighter-inner positive comparisons must still solve.
