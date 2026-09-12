# Exact execution command

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark.py \
  --config configs/pilot_v2.yaml \
  --seeds 701 809 907 \
  --modes simple_objective simple_plus_equiv simple_plus_inv simple_plus_scale \
    simple_plus_proj simple_plus_spectral simple_plus_geom simple_plus_vq \
    simple_plus_node simple_plus_moment simple_plus_identity simple_plus_depth base \
  --steps 30 \
  --output research/results/development/objective_study_v1/summary.json \
  --retain-run-dir research/results/development/objective_study_v1/runs \
  --study-id legacy-objective-v1 \
  --protocol research/protocols/objective_study_v1.md \
  --protocol-sha256 8a9f765fc66cd9331de82c47f9e49840cde8eca30034f56cbdc7e309a3ae749e
```

This command was recorded before execution. The `runs/` subtree is locally retained
and ignored by Git; the raw seed rows in `summary.json`, summary table, report, and
manifest are retained as repository evidence.
