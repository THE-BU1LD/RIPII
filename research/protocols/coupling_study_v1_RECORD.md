# Coupling study v1 execution record

Status: executed development study. This Markdown index was written after execution
and is not a preregistration; the machine-readable protocol was written before training.

- Protocol SHA-256:
  `0707865d479ca2868ffe9bd155c9de5290baa9099cfa3a2164361ef882015c2c`.
- Models: graph, global-pool, multiscale; continuous bottleneck.
- Regimes: global coupling 0 and 1.
- Seeds: 53, 59, 61. Data seed: 12031.
- Budget: 100 updates; 96 train scenes; 16 evaluation scenes.
- Primary outcome: more-objects position RMSE, multiscale versus global-pool.
- Decision: `no_advance`.
- Capsule: `research/results/development/world_v4_coupling_capsule.json`.
- Capsule signature:
  `b8428f377bbdd07e042d3efb71f843a129fbf6ffec2f93c29cc1f9eaf90c30af`.

The ignored full run contained 185 verified artifacts. The capsule retains the master
protocol, summary, report, manifest, and both raw child summaries. It omits checkpoints
and therefore cannot re-evaluate them.

Provenance limit: this first executed master protocol recorded the dirty commit but did
not hash `run_coupling_study.py` itself. Its child protocols/manifests bind the executed
model sources and outputs. The current runner now hashes its own source plus protocol,
physics, model, experiment, and statistics modules for future studies; that prospective
improvement does not retroactively upgrade this result beyond development evidence.
