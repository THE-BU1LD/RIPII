# Frozen pilot command

Protocol SHA-256: `a081c76696ed09168d2a783def08ea3fdf487bf6ed913d8af3e2b1d61e5066e5`

Config SHA-256: `8b85a3db55b44caee7bf3cb3b726fcaac90ddf50f7201634e06cb3e035a69d1d`

```bash
python3 scripts/benchmark.py \
  --config configs/pilot_v1.yaml \
  --seeds 101 211 307 \
  --steps 30 \
  --modes base no_renorm no_graph no_vq no_action no_structured \
  --study-id ripii_component_pilot_v1 \
  --protocol research/protocols/pilot_v1.md \
  --protocol-sha256 a081c76696ed09168d2a783def08ea3fdf487bf6ed913d8af3e2b1d61e5066e5 \
  --output research/results/pilot_v1/summary.json \
  --retain-run-dir research/results/pilot_v1/runs
```
