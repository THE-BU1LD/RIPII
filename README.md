# RIPII

RIPII is a structured latent learning system built around adaptive coarse-graining, sparse latent graph refinement, and hierarchical discrete motifs.

## What is included

- End-to-end training, evaluation, diagnostics, and benchmark scripts
- Synthetic structured dataset with paired transformed views
- Adaptive projective renormalization stack
- Sparse latent graph refinement
- Hierarchical vector quantization
- Ablation presets, benchmark sweeps, and benchmark reports
- Smoke tests, CLI tests, and model inspection

## Quick start

```bash
python scripts/train.py --config configs/smoke.yaml
python scripts/evaluate.py --config configs/smoke.yaml --checkpoint runs/ripii_smoke/final.pt
python scripts/diagnostics.py --config configs/smoke.yaml --checkpoint runs/ripii_smoke/final.pt --output runs/ripii_smoke/diagnostics.png
python scripts/benchmark.py --config configs/smoke.yaml --seeds 3 7 --steps 3 --modes base no_graph no_vq
python scripts/run_suite.py --config configs/smoke.yaml
python scripts/inspect_model.py --config configs/default.yaml
```

## Outputs

Each run writes:

- `config.yaml`
- `history.jsonl`
- `latest.pt`
- `final.pt`
- `best.pt`
- `eval.json`
- diagnostic plots

## Ablations

Ablations are handled by config flags and the `--mode` switch in the training and benchmark scripts.
