# Rotated Second-Order Cone

The experimental conic contracts now accept `RotatedSecondOrderCone(d)` and
the `openopt.conic.RSOC` alias, with d >= 3. An original block `(a,b,w)` obeys
a >= 0, b >= 0 and 2ab >= ||w||^2. Its dual is the same cone.
This extends the dense zero/nonnegative/SOC prototype, not all conic scope,
HSD, SDP, exponential/power cones, exact certificates or commercial qualification.

The orthogonal map is
`T(a,b,w)=((a+b)/sqrt(2),(a-b)/sqrt(2),w)` and T is its own inverse.
Native projection is `T projection_SOC(T value)`, on the explicitly requested
PyTorch device. It is Euclidean projection, not an unweighted projection
through a nonorthogonal rescaling. No third-party solver or device fallback
is introduced. Native calculations scale the Lorentz transformation, preserve
interior inputs bitwise, rationalize near-polar weights and reconstruct the
small boundary head without cancellation. Signed-product error compensation
retains the near-polar difference. The weight remains a mantissa/exponent pair
until each coordinate is restored, so weight underflow does not erase a
representable output. A zero tail projects both heads independently.
Float32 uses float64 intermediates on the requested device, not CPU projection.
Original feasibility independently checks each head sign, then compares
`S-2*M*m <= 2*M*atol + rtol*max(2*M*abs(m),S)` using bounded dyadic integers
from the binary64 inputs, where `S=sum(w_i^2)`, `M=max(a,b)` and `m=min(a,b)`.
The tolerance contract is unchanged. This exact input-inequality comparison
does not certify optimality or make the solver arbitrary precision.
Opposite-head scaling must not
hide a negative coordinate, and subtractive Lorentz radii must not hide a
small original deficit. No overflowing ab or w^2 is formed. Inputs, original
slack, dual membership, block complementarity,
stationarity, dual range and signed gap remain independently checked.

The explicit Clarabel CPU reference transforms only the corresponding A/b
rows, invokes its SOC reference, and maps both slack and dual back with T.
The numerical evidence is recomputed on original P/q/A/b and RSOC blocks;
reference status alone cannot qualify a candidate. Original arrays are owned
and are not changed. Unknown status and unavailable CUDA remain fail-loud.

`tests/test_rotated_conic.py` covers analytic original KKT, half-space signs,
zero/boundary/large coordinates, orthogonal SOC parity, Moreau decomposition,
mixed offsets, float32/64 projection, native/reference original solves,
planting false slack, unfinished iterations and explicit CUDA rejection.
These CPU tests are not actual CUDA execution or transfer-inclusive speedup.
Fresh same-source cluster/wheel and generated-document acceptance remain
required before a release claim.

The mathematical construction is independently implemented under 0BSD.
No vendor code is copied, and dependency licenses are unchanged.

## Development Receipts

External root: `/hf3fs-jd/data/tmp/openopt-release-20261010/`.
The original KKT fixture's equality multiplier had the wrong sign; the first
`rotated-conic-owner-unit-v1` report retains that test failure. Correcting the
analytic sign also added a wrong-sign negative control, without changing any
tolerance. The subsequent owner unit report records 150 passed in 11.05s.

Independent read-only review proposed six extreme-coordinate counterexamples.
All six failed in `rotated-conic-independent-counterexamples-v1/tests-x08h1hm8`
(31 passed, six failed, 4.33s). The first repair, independently retaining all
those tests, yielded 271 passed, 19 CUDA skips, 18.09s in
`rotated-conic-independent-repair-v2/tests-hvgbwj4d`.
That receipt is an earlier repair, not evidence for the later near-polar fix.
The separate near-negative-cone analytic input `(-1e20,0,1e8)` still lost its
nonzero projection in `rotated-conic-negative-boundary-v1/tests-csd7hf0f`;
the one failed test is preserved. Rationalized weighting addresses that
additional first cause. Final repair, clean source, independent review,
generated-document and actual CUDA checks must each be read independently.
No historical measurement is promoted to final release acceptance.

The second independent review supplied four further extreme inputs. They
failed in `rotated-conic-independent-extreme-v1/tests-679kg4h2` (four failed,
3.18s): a tail norm overflow although coordinates were representable, and
an intermediate weight square underflow although the small head was
representable. Tail norms and weights now retain mantissa/exponent pairs
until final coordinates. The initial repair retained one failure in
`rotated-conic-independent-extreme-repair-v2/tests-a8ck1xrv` (275 passed,
one failed, 19 CUDA skips, 19.00s); staged exponent restoration fixes that
last output overflow. The subsequent complete three-file conic regression
is `rotated-conic-independent-extreme-repair-v3/tests-mqzjw3en` (276 passed,
19 CUDA skips, 18.87s), with source-consistency PASS. These are development
bytes, not clean-commit or actual GPU execution receipts. The eleven
extreme-coordinate negative/analytic cases remain in the default unit suite.

## New Independent Counterexamples

Frozen independent test commit `74f5583` added 64 tests without changing the
production files or previous oracle. Its independent run and the owner's
`owner-rsoc-independent-red-v1/tests-xnyw1thl` both retain 54 passes/10 failures.
The failures cover tiny representable heads or tails lost by weight underflow,
near-polar signed-product cancellation, checker normalization underflow, and
an invented deficit at the exact huge boundary `[1.3e308]*4`.
No independent test or tolerance was relaxed.

The first owner repair
`owner-rsoc-exponent-green-v1/tests-1exryl_p/verification.json` records 340
passes/19 explicit CUDA skips, 20.70s, source-consistency PASS. This is a dirty
development receipt before the final exact tolerance comparison; it is not
clean-source acceptance, independent signoff or GPU qualification. The final
comparison, clean committed rerun and independent review remain separate gates.
