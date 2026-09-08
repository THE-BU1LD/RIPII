# Frozen pilot command

Protocol SHA-256:
`c8c6c6d1b692513e15a81e2a853f04358cbf16e649e4448521ca4d7a4983e0e0`

Config SHA-256:
`3be16fd6a0ef7ab9c1f659482adf8054136ae2d6f93f947d3c6b922146740ae1`

```bash
python3 scripts/benchmark.py \
  --config configs/pilot_v2.yaml \
  --seeds 401 503 607 \
  --steps 30 \
  --modes base no_vq no_structured \
  --study-id ripii_corrected_component_pilot_v2 \
  --protocol research/protocols/pilot_v2.md \
  --protocol-sha256 c8c6c6d1b692513e15a81e2a853f04358cbf16e649e4448521ca4d7a4983e0e0 \
  --output research/results/pilot_v2/summary.json \
  --retain-run-dir research/results/pilot_v2/runs
```
