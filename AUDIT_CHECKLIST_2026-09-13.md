# Research audit and execution checklist — 2026-09-13

## Verdict and classification

Model/world pipeline, quantizer diagnostics, CLI, artifact verification, statistics, profiling, external-trajectory adapter, and extensive tests are **complete/real engineering**. `research/results/pilot_v2` is **complete real negative development evidence**: the full object-state model does not clear advancement gates and codebooks approach collapse. Development coupling/objective studies are **development-only**. External broad evidence, equal-compute replication, and independent reproduction are **missing**. Current local source must not be assumed identical to historical artifacts.

The discrete codebook claim requires assignment entropy/perplexity well above collapse and an intervention showing decoded predictions depend on object codes rather than a bypass. Reconstruction loss alone does not identify object-centric structure.

| Priority | WHAT / WHY | HOW / WHERE | VERIFY |
|---|---|---|---|
| P0 | Fresh results need source/config/data/checkpoint identity. | Historical artifacts cannot support changed code. | Use clean tagged clone and sealed manifests for successor runs. | Verifier rejects any dirty/mismatched input. |
| P1 | Full model loses to bypass/removal; codebook collapses. | Core mechanism is not useful/identified. | Instrument entropy, perplexity, occupancy, decoder bypass; freeze collapse thresholds. | Causal code replacement/ablation changes predicted objects as specified. |
| P1 | Equal-compute and external comparison incomplete. | Capacity/optimization can explain outcomes. | Three matched models within 5%, ≥18 paired seeds, external simulator/data, OOD. | Hierarchical multi-dataset analysis and retained nulls. |
| P2 | Add slot/object/graph baselines and object metrics. | Reconstruction is insufficient. | Slot consistency, segmentation, intervention response, relation accuracy. | Metric unit tests and untouched evaluation. |
| P3 | Durable external artifact store and independent reproduction. | Local evidence is fragile. | Archive weights/raw predictions/checksums. | Third party rebuilds report. |

Conference state: **engineering mature; scientific advancement negative/not ready**.
