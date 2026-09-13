# Platform and Determinism Support

## Tested contract

- Python: CI targets 3.10, 3.11, and 3.12 from the exact `uv.lock` resolution.
- Local audit: Python 3.14 and PyTorch 2.14 on macOS CPU.
- Exact resume: covered on CPU, including rollout curriculum and state-noise RNG state.
- CUDA: an opt-in self-hosted scheduled smoke is configured and requires an owner-set
  `RIPII_CUDA_RUNNER=enabled` repository variable plus a matching GPU runner; it has not
  yet produced certification evidence.
- MPS: a weekly `macos-14` forward/backward finite-value and same-process repeatability
  smoke is configured. Configuration is not evidence that the workflow has passed.

CPU is the reference reproducibility target. Cross-device bitwise identity is not
claimed. Numerical comparisons across devices must use metric tolerances declared before
results are inspected and must report Python, PyTorch, device, dtype, thread settings,
and deterministic-algorithm status.

## Performance boundary

Machine-local timings are descriptive. Use warmup, synchronized repeated measurements,
bootstrap intervals, peak memory, and recognized-operation counts from the profiling
scripts. Do not compare a single cold timing or interpret recognized FLOPs as complete
FLOPs. The overnight runner rejects non-increasing calibration timings rather than
clamping them into a nominal estimate.

## Verification needed before expanding support

Run and retain the scheduled CUDA and macOS/MPS smoke results, add exact-resume tests
where deterministic kernels permit, establish device-specific tolerance records, and
add long-rollout non-finite/stability checks. A passing one-step smoke is not platform
certification.
