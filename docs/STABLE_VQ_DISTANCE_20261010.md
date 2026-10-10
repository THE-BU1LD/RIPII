# Opt-in stable VQ distances: development specification

Version: `ripii-vq-distance-v1`. Date: 2026-10-10. Dependency: external-content
verification PR #22 at `66ee627b32387e86bcc4bc0d57e725f63a21fe60`.

## Observation and mathematical contract

For float32 query `[10001]` and code vectors `[10000]`, `[10001]`, the retained
norm/matrix-product identity returns squared distances `[0, 0]` and selects the
first code. Independent float64 direct subtraction gives `[1, 0]` and selects
the second. A nearest-code operation must not confuse these represented inputs.

The explicit `StableHierarchicalVectorQuantizer` wrapper substitutes direct
Euclidean distance evaluation, squared after PyTorch's non-matrix-product
`cdist` kernel. Float64 is retained; float16/bfloat16 distance work uses float32.
Autocast is disabled for this kernel. Non-finite input or intermediate squared
distance is rejected. Every computed zero distance is checked against exact
equality of its represented operands; a nonzero separation that underflows to
zero is rejected instead of becoming a false exact match. Thus not every finite
input is representable at the chosen work precision. First-index exact ties are retained. This avoids the
cancellation-prone squared-norm identity without allocating an N-by-K-by-D
difference tensor; it still requires N-by-K distance output and O(N*K*D) work.

The inherited coarse/fine assignments, residual, straight-through estimator,
codebook/commitment/balance objectives and explicit revival method are otherwise
unchanged. Code revival remains an explicit training intervention with the
retained mutation semantics; no atomic revival or new revival schedule is
promised. The wrapper shares the exact original coarse/fine Parameters, preserves
state-dict keys and training mode, and consumes no initialization randomness.

## Selection and scientific boundary

```python
from ripii.models.stable_quantizer import StableHierarchicalVectorQuantizer

model.quantizer = StableHierarchicalVectorQuantizer(model.quantizer)
```

This explicit assignment changes prospective numerical behavior. Record the
wrapper version and source commit in any new model configuration/protocol. The
retained quantizer, model factory, training callers, checkpoints, protocols and
historical results remain unchanged. Compatible parameter shapes alone do not
identify which numerical implementation produced a checkpoint; callers must
retain this source/version selection alongside their run configuration.

The falsifier is a disagreement with direct float64 distances/assignments on
representable fixtures, a changed retained equation, lost parameter/RNG identity,
or a gradient mismatch on well-conditioned inputs beyond dtype tolerance.
Budget: bounded generated CPU tests only; no protected data, scientific study,
paid compute or efficacy claim. This repair does not explain the observed
codebook collapse or reopen any negative/no-advance study.

## Verification and independent review

The final generated-fixture check passed 47 cases: 37 new successor cases and
10 retained quantizer, mechanism and model/loss regressions. It covers direct
float64 distance and gradient oracles, finite differences, first-index exact
ties, half/bfloat16 promotion, CPU autocast, noncontiguous inputs, overflow and
underflow rejection, parameter/optimizer identity, strict checkpoint loading,
RNG preservation and a real model backward/optimizer step. Ruff passed using the
repository's existing rule selection. Runtime: Python 3.12 and isolated PyTorch
2.5.1+cpu, with one CPU thread per numerical library.

```sh
PYTHONPATH=<isolated-torch>:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 python -m pytest -q -o addopts='' \
  tests/test_stable_quantizer.py tests/test_quantizer_qualification.py \
  tests/test_mechanism_semantics.py tests/test_loss_step.py
python -m ruff check ripii/models/stable_quantizer.py tests/test_stable_quantizer.py
```

An earlier implementation snapshot completed a full-suite run with one failure:
`test_post_correction_preflight_reports_frozen_cell_count`. The retained runner
requires 10.00 GiB free, while the workspace had 9.16-9.17 GiB. The same test and
exact error were reproduced at unchanged dependency commit
`66ee627b32387e86bcc4bc0d57e725f63a21fe60`. Its gate remains intact. Final
underflow and zero-dimensional-construction cases were then included in the
47-case scoped verification; no exact-final-source full-suite pass is claimed.

Independent source review checked the mathematical contract and inherited
distance consumers. It identified finite overflow and false-zero underflow
hazards. Both now reject before assignment. The underflow witnesses use float32
query `[1e-30]` and float64 query `[1e-200]` against zero and an exact matching
code. Every computed zero pair is checked in bounded feature chunks, including
nonselected codes, because inherited soft-assignment losses also consume the
distance matrix. Actual equal-operand zeros retain their normal behavior.

Hosted CI is not triggered by this draft source commit (`[skip ci]`): inherited
workflows include portable study verification and optional self-hosted hardware
jobs. CUDA/MPS operation, numerical throughput and scientific efficacy remain
untested. This limited local verification does not change any retained result,
source freeze, pending external provenance requirement or advancement decision.
