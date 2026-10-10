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
small boundary head without cancellation. Original feasibility independently
checks each head sign, then evaluates the required small head from the stable
tail norm using exponent-split `r^2/(2*large)`. Opposite-head scaling must not
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
