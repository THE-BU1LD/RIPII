# RIPII

RIPII is an experimental structured-latent learning implementation built around
adaptive coarse-graining, sparse latent graph refinement, and hierarchical discrete
motifs. It has no validated performance, novelty, or publication claim; see
[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md).

Both frozen local pilots failed their advancement rules. In the corrected follow-up,
the full model reconstructed worse than both quantizer bypass and the four-mechanism
removal, while its codebooks remained near collapse. See
[`research/results/pilot_v2/analysis.md`](research/results/pilot_v2/analysis.md).

The complete audit is indexed by [`audit/REPOSITORY_MAP.md`](audit/REPOSITORY_MAP.md),
[`audit/CONFERENCE_READINESS_CHECKLIST.md`](audit/CONFERENCE_READINESS_CHECKLIST.md),
and [`FINAL_RESEARCH_REPORT.md`](FINAL_RESEARCH_REPORT.md). The current verdict is
**EVIDENCE_PARTIAL**: implementation quality is substantially verified, while all
scientific evidence is synthetic and the proposed mechanisms have negative results.
The actionable post-audit backlog—including explicit pseudocode, scaffold, bad-
implementation, improvement, addition, and abstraction classifications—is
[`audit/ULTIMATE_CHECKLIST.md`](audit/ULTIMATE_CHECKLIST.md).
Focused guides under `docs/` cover research status, architecture, experiments,
reproducibility, and preserved failures.

## Object-state world model

A complete action-conditioned dynamics workflow is now available: a 2D contact
simulator, MLP/graph/Transformer/global-pool/multiscale predictors,
continuous/FSQ/VQ bottlenecks, validation-selected checkpoints, three explicit
analytic references, multi-seed generalization reports, and an interactive
prediction workbench. See [WORLD_MODEL.md](WORLD_MODEL.md) for the model contracts,
controls, protocol, and limitations.

```bash
python -m ripii.world benchmark --output runs/world_experiment --steps 300 --seeds 3 7 11
python -m ripii.world verify runs/world_experiment
python -m ripii.world capture runs/world_experiment --output research/results/development/world_experiment_capsule.json
python -m ripii.world verify-capsule research/results/development/world_experiment_capsule.json
python -m ripii.world demo --checkpoint runs/world_experiment/multiscale_continuous/seed_3/best.pt
```

These are object-state inputs, not learned visual perception. The benchmark reports
whether the new hierarchy helps; its existence is not evidence of an advantage.
The fresh five-architecture `runs/world_v2_controls` experiment returned
`no_advance`; graph has the best mean test and OOD errors in that bounded run.
The stronger fresh-data, five-seed, 1,000-update `runs/world_v3_convergence`
study also returned `no_advance`: multiscale lost the graph OOD comparison on
all five seeds. The current evidence therefore argues against advancing the
hierarchy without a materially new hypothesis.

A targeted extension supplied that new hypothesis by adding a symmetric long-range
force. Its prospectively specified three-seed local-versus-coupled study again returned
`no_advance`: multiscale did not earn a consistent 5% advantage over global pooling.
The self-checksummed result is
`research/results/development/world_v4_coupling_capsule.json`.

## What is included

- End-to-end training, evaluation, diagnostics, and benchmark scripts
- Synthetic structured dataset with paired transformed views
- Adaptive projective renormalization stack
- Sparse latent graph refinement
- Hierarchical vector quantization
- Ablation presets, benchmark sweeps, and benchmark reports
- Smoke tests, CLI tests, and model inspection
- Validated dataset adapters with split metadata and deterministic content hashes
- Immutable protocol records and manifested `planned -> running -> complete/failed` states
- Self-checksummed failure-localization artifacts and a repeatable rollout profiling CLI

## Quick start

For the exact resolved development environment recorded in `uv.lock`:

```bash
uv sync --locked --extra dev
uv run pytest
uv run ruff check .
```

The standard-library environment path is also supported:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
pytest
ruff check .
```

Then run a mechanism-enabled development job:

```bash
python3 scripts/train.py --config configs/mechanism_smoke.yaml --output-dir runs/dev_mechanism_001
python3 scripts/evaluate.py --config configs/mechanism_smoke.yaml --checkpoint runs/dev_mechanism_001/final.pt
python3 scripts/diagnostics.py --config configs/mechanism_smoke.yaml --checkpoint runs/dev_mechanism_001/final.pt
python3 scripts/benchmark.py --config configs/mechanism_smoke.yaml --seeds 3 7 --steps 3 --modes base no_graph no_vq plain_ae
python3 scripts/qualify_quantizer.py --seed 17 --steps 300
python3 scripts/run_suite.py --config configs/mechanism_smoke.yaml
python3 scripts/inspect_model.py --config configs/default.yaml
```

Canonical repository-wide entry points are:

```bash
./scripts/preflight.sh
./scripts/test.sh
./scripts/run_smoke.sh
./scripts/analyze.sh
./scripts/verify_artifact.sh
```

Profile a retained world-model checkpoint with warmup, repeated synchronized timing,
a bootstrap interval, throughput, environment metadata, and a clearly labeled
PyTorch-recognized FLOP lower bound:

```bash
python3 scripts/profile_world.py runs/world_experiment/graph_continuous/seed_3/best.pt \
  --output runs/world_experiment/graph_profile.json
```

`python3 scripts/clean_generated.py` is dry-run only. Add `--apply` to remove explicit
build/test caches; the cleaner never considers `runs/` eligible.

`configs/smoke.yaml` is a plumbing-only profile: it intentionally disables the
projective, graph, quantizer, and action paths and cannot be used for mechanism
ablations. Outputs are mutable development diagnostics unless produced by a frozen
protocol and retained with a manifest. Verify the retained pilot without trusting its
committed summary files:

```bash
python3 scripts/verify_artifact.py \
  --manifest research/results/pilot_v2/manifest.json \
  --protocol research/protocols/pilot_v2.md \
  --portable-summary
```

This portable mode verifies the protocol hash and the committed JSON, CSV, and
Markdown summaries while reporting that the ignored `runs/` entries were skipped. A
full manifest verification intentionally remains stricter and requires the retained
local run directory, including its checkpoints.

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

Ablations are handled by config flags and the `--mode` switch in the training and
benchmark scripts. Learned multi-objective total losses are optimization diagnostics;
they must not be used to rank models with different active objectives.

`plain_ae` is a genuinely unstructured encoder/decoder baseline: it has no graph,
projective stack, action module, fusion gate, or quantizer. The benchmark selects a
real hidden width within 2% of the full model's trainable parameter count; it does not
pad the model with unused parameters. This controls parameter count only—not training
compute or convergence. `qualify_quantizer.py` is an isolated, deterministic
development gate and does not establish that quantization helps the full model.

## Release boundary

This repository is suitable for inspection as an experimental prototype, not as a
validated method. It has no owner-selected license, authorship metadata, external
dataset evaluation, adequately powered convergence/compute-matched comparison, or
independent reproduction. A one-seed, one-step parameter-matched diagnostic is
retained only as negative development evidence. Without a license, public readers
have no granted reuse rights.

## Legacy checkpoint and metric corrections

New legacy-training checkpoints count completed optimizer updates and retain RNG,
optimizer, scaler, and best-score state. Resume with the saved run configuration and
an increased `--steps` budget. Older unversioned checkpoints remain evaluable, but
exact continuation is unsupported; use `--initial-state` for a fresh development run.
Zero-weight/disabled mechanisms have no uncertainty offsets, evaluation weights
per-example metrics by batch size, constant features have zero effective rank, and
the zero synthetic transform is now the identity. These corrections change future
results; historical pilot artifacts were preserved unchanged.
