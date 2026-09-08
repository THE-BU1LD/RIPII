# Reproducibility

Install with `uv sync --locked --extra dev`. Run `./scripts/preflight.sh`,
`./scripts/test.sh`, `./scripts/run_smoke.sh`, `./scripts/analyze.sh`, and
`./scripts/verify_artifact.sh`. A fresh development benchmark uses
`./scripts/run_experiments.sh <output> [world CLI arguments]`.

Full ignored run directories contain checkpoints. Compact capsules retain protocols,
summaries, reports, manifests, and hashes but cannot re-evaluate omitted checkpoints.
Exact resume is verified on CPU. Hardware timing is descriptive and not portable.
External dataset reproduction remains `EXTERNAL_EXECUTION_REQUIRED`.

New world runs also retain validated dataset specifications, content SHA-256 values,
an immutable protocol format, and a manifested run-state history. Use
`python scripts/profile_world.py CHECKPOINT --output PROFILE.json` for repeated timing;
its profiler FLOP count is explicitly a recognized-operator lower bound, not a claim of
complete FLOP matching. Use `python scripts/clean_generated.py` to preview disposable
caches and add `--apply` only when desired. Historical `runs/` are never cleanup targets.

The provisional external seed calculation is reproducible with:

```bash
python scripts/plan_power.py \
  research/results/development/world_v3_convergence_capsule_v2.json \
  --minimum-detectable-effect 0.05 \
  --output runs/planning/external_power_plan.json
```

The retained self-checksummed plan recommends 18 paired seeds from synthetic development
variance. It is planning support, not confirmatory evidence; an external pilot may
require a larger frozen count.
