# Independent Rotated Conic CUDA Extremes

This separate numerical regression is based on clean `b6905ab`. Only
`tests/test_rotated_conic_cuda_extremes.py` and this document are added.
It does not change the main tree, production projection/checker, the fixed
thirteen-member Nash producer, its auditor, dispatch or cluster objects.
No dependency installation, GPU submission or nested delegation is performed.
The main owner alone may submit GPU work.

## Numerical Contract

The test imports the unchanged original-coordinate oracle helpers from the
same root's `tests/test_rotated_conic_independent.py`: Decimal precision 1100,
per-coordinate relative limits `2e-12` (float64) and `3e-5` (float32), exact
zero for unrepresentable coordinates, and the unchanged scaled Moreau gate.
The independent 103-case branch's new vectors inform this selection; the
baseline file still has 75 cases and is not rewritten or executed as CUDA.
Neither a transformed SOC oracle nor a solver is used.

Exact Fraction arithmetic classifies each represented input's original
`sum(w_i**2)-2*a*b`, before projection. Both severe historical polar examples
are included: binary64 `(-1,-1,sqrt(2))` and
`(-2**500,-2**500,2**500,2**500,1)`. Their binary32 counterparts are explicitly
different represented vectors: sqrt(2) is rounded upward to the next float32,
and the large head is `2**100`. Both have strictly positive exact excess and
representable positive projected heads; the erased-head oracle decoy fails.

Eight remote-tail permutations retain the original binary64 `2**600` head,
unit/middle/signed tails and two remote levels. Float32 cannot represent that
head, so its explicitly named counterparts use `2**100`, middle `2**30` and
tiny `2**-60`, rather than silently casting `2**600` to infinity. These
binary32 cases do not exercise the binary64 expansion's `>900` exponent gap.
Both dtypes also cover the rationalized negative-head weight, a small head,
overflowing unscaled tail norm and six positive inside/boundary/exterior
permutations. No tolerance or old oracle is relaxed.

Positive exterior outputs may round to the same coordinates as the input.
Therefore real unmodified norm operation counts additionally distinguish
the full projection path from a false early return. The original checker
uses its smallest legal positive binary64 absolute tolerance, `math.ulp(0)`,
and zero relative tolerance; exact Fraction signs remain the independent
oracle. Positive inside/boundary results must preserve every input bit.

## Device And Evidence Boundary

Every CUDA node explicitly constructs its input on `cuda:0`, verifies output
device and requested dtype, and synchronizes completion. A TorchDispatch
trace rejects all off-device tensor operands and results inside the actual
production projection; real norm/frexp operations and device-local float64
intermediates must be observed. Tensor `cpu`, `numpy` and `tolist` methods
are forbidden during that protected scope. Host copies for the independent
Decimal/Fraction checker occur only outside it. Existing scalar Boolean
control synchronization remains allowed. This is a local tensor-placement
check, not adversarial runtime or remote attestation.

Every numerical node writes a unique external `rotated-conic-extreme.json`.
It preserves exact hexadecimal input/positive/negative coordinates, full
Decimal references, exact Fraction excess, requested/observed device, dtype,
dual orientation, observed GPU name when applicable, loaded-source byte hashes,
operation trace and separate coordinate/Moreau checks. JUnit pins actual JSON
bytes using `rsoc_cuda_extremes_path` and `rsoc_cuda_extremes_sha256`.
CPU nodes are labeled `CPU_MECHANISM_PASS`; CUDA nodes have the distinct
`CUDA_NUMERICAL_REGRESSION_PASS` marker. No production acceptance allowlist
accepts this family automatically. Both markers exclude commercial, speed,
certificate and remote-origin qualification.

## Exact Selection

There are 38 vectors: 19 float64 and 19 float32. Each has explicit `primal`
and `dual` parameter IDs, producing 76 CUDA nodes and 76 CPU mechanism nodes.
Six additional CPU mechanism nodes exercise placement guard negatives,
nonempty trace, both erased-head decoys and exact case membership.
The whole file therefore collects exactly 158 pytest nodes. With CUDA hidden,
82 pass and all 76 CUDA nodes explicitly skip if the numerical gates pass.
Skipped GPU nodes are UNVERIFIED, never GPU PASS.

For actual GPU regression select only
`tests/test_rotated_conic_cuda_extremes.py::test_cuda_rotated_extreme_coordinates`;
the denominator is exactly 76 and every node must pass without error/skip.
Do not merge those identities into the fixed thirteen-case producer contract.

From this clean root, local CPU mechanism validation uses:

```bash
sudo -n -u infra env CUDA_VISIBLE_DEVICES=-1 PYTHONDONTWRITEBYTECODE=1 \
  /hf3fs-jd/data/tmp/openopt-validation-env/bin/python -B scripts/verify.py tests \
  --target tests/test_rotated_conic_cuda_extremes.py \
  --artifact-dir /hf3fs-jd/data/tmp/openopt-release-20261010/rotated-conic-cuda-extremes/CPU_NEW
```

The owner can invoke the same verifier with the exact CUDA function target
and `--require-cuda` from its approved GPU runner and frozen source. No
cluster command or submission is supplied here. Each CPU batch must finish
within 180 seconds. Actual receipt paths and source consistency are reported
after execution; no historical or synthetic receipt is promoted to GPU proof.

## Local Preparation Receipt

The actual development verification at
`rotated-conic-cuda-extremes/development-v1/tests-pqdppwvc/verification.json`
records 158 nodes, 82 CPU passes, 76 explicit CUDA skips, zero failures/errors,
12.67 seconds of pytest execution and source-consistency PASS. This source
snapshot was not yet committed. Every numerical CPU node retained its JSON
sidecar and JUnit byte pin. Clean committed-source evidence is generated
separately under `clean-wholefile` and `clean-cuda-selection` and reported
with actual receipt hashes after execution. Neither local receipt qualifies
CUDA execution.
