# Alpha-aware prospective paired planning

Base: `317a6ec4163134a56cdc56ce3fd20f5a43decfb4`.

## Reproduced defect and correction

The retained v1 planning API accepts `alpha` but hard-codes a six-pair exact
sign-flip resolution floor. With generated differences `[-0.001, 0.001]` and
minimum detectable effect `1.0`, it recommends six pairs at alpha `0.01` and
`0.001`. Six nonzero pairs cannot attain either threshold: their minimum
two-sided p-value is `2 / 2**6 = 0.03125`.

The prospective v2 planner derives the smallest nonzero pair count with
`2 / 2**n <= alpha`, then takes the maximum of that count and the unchanged v1
normal-approximation estimate. Integer ratios avoid logarithm rounding at exact
powers of two and their neighboring floating-point values. The resulting floors
are six at `0.05`, eight at `0.01`, and eleven at `0.001`.

The variance estimator and normal power approximation retain their old
definitions. The resolution helper supports every positive finite float alpha;
the planner explicitly rejects thresholds too small for the retained normal-tail
calculation. The floor assumes nonzero pairs and is not a power guarantee.
The retained exact enumerator supports only 20 nonzero pairs. A larger proposed
sample count requires a separately predeclared feasible inference method.

## Executable prospective path

Run from a repository checkout with an authorized development summary:

```bash
python -m scripts.plan_power_v2 development_summary.json \
  --candidate multiscale --baseline graph --split more_objects \
  --bottleneck continuous --minimum-detectable-effect 0.05 \
  --alpha 0.01 --power 0.8 --output prospective_plan_v2.json

python -m scripts.plan_power_v2 --verify-output prospective_plan_v2.json
```

The CLI uses the existing summary/capsule admission, paired-seed extraction and
input provenance path. Its new payload identifies the v1 dependencies and v2
source. Verification recomputes planning arithmetic and checks pair/seed counts,
evidence status, source identities and the content digest. Re-signing an
internally inconsistent plan does not make it pass. The digest is not sender
authentication, and this verifier does not establish independent provenance or
scientific validity of the supplied development measurements.

Publication stages and fsyncs a complete JSON file, then uses a same-filesystem
hard link to publish a fresh destination. Existing files and symlinks are
refused; failed publication removes only its own staging file. This is an atomic
new-file publication guarantee on supported filesystems, not a directory-fsync
crash-durability claim. It never replaces a retained plan.

## Verification and research boundary

`python -m pytest tests/test_power_v2.py tests/test_power.py`:
**35 passed** (33 new cases and two retained tests). Ruff under repository policy,
compilation and whitespace checks pass. New tests use generated summaries only:
independent enumeration establishes minimal resolution at each threshold;
boundary tests cover neighboring powers of two and the smallest positive float;
actual CLI creation/verification/repeated-output checks, re-signed semantic
corruptions and publication failure fixtures exercise the working artifact path.

All existing repository files, including the source-pinned v1 planner, retained
planning artifact, frozen studies, negative/no-advance outcomes and protocols,
remain unchanged. This opt-in repair performs no model training, protected
evaluation, benchmark matrix or paid compute. It does not mark the research as
complete or reinterpret any past outcome. Commit uses `[skip ci]` because the
broader inherited suites execute training and evaluation; the reported checks
are local, focused engineering checks only.
