# RIPII

RIPII is an experimental object-state dynamics and structured-latent toolkit built
around learned soft grouping, latent graph refinement, and optional discrete motifs.
It has no validated performance, novelty, or publication claim; see
[`RESEARCH_STATUS.md`](RESEARCH_STATUS.md).

Both frozen local pilots failed their advancement rules. In the corrected follow-up,
the full model reconstructed worse than both quantizer bypass and the four-mechanism
removal, while its codebooks remained near collapse. See
[`research/results/pilot_v2/analysis.md`](research/results/pilot_v2/analysis.md).

The authoritative audit is
[`audit/END_TO_END_AUDIT_2026-09-13.md`](audit/END_TO_END_AUDIT_2026-09-13.md),
with status and historical reports indexed by [`audit/README.md`](audit/README.md).
The current verdict is
**EVIDENCE_PARTIAL**: implementation quality is substantially verified, while the
evidence is synthetic or external-simulator development work and the proposed
mechanisms have negative results.
The actionable post-audit backlog—including explicit pseudocode, scaffold, bad-
implementation, improvement, addition, and abstraction classifications—is
[`audit/ULTIMATE_CHECKLIST.md`](audit/ULTIMATE_CHECKLIST.md).
Focused guides under `docs/` cover research status, architecture, experiments,
reproducibility, and preserved failures.

## Object-state world model

A complete action-conditioned dynamics workflow is now available: a 2D contact
simulator, MLP/graph/Transformer/global-pool/multiscale/E(2)-equivariant predictors,
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

A fresh-seed 39-run legacy objective study also found that reconstruction+KL beat
every individual auxiliary addition and the complete objective on every paired seed
at the fixed 30-update budget. Use `--mode simple_objective` for that executable
reference. This is synthetic development evidence, not a convergence claim.

## External-simulator evidence

The NRI Springs and Charged study uses the independently implemented, commit-pinned
MIT-licensed NRI simulator. It trained five neural controls across three paired seeds
per domain and retained 185 verified artifacts. Its gate returned `no_advance`:
multiscale reached the required 5% improvement over global pooling in only one of six
domain-seed cells and was not the best model on either domain. On Charged, the analytic
constant-velocity baseline beat every neural model. This is external-simulator
development evidence, not real-world or confirmatory validation. See
[`research/protocols/nri_external_development_v1.md`](research/protocols/nri_external_development_v1.md)
and the self-checksummed capsule under `research/results/development/`.

## What is included

- End-to-end training, evaluation, diagnostics, and benchmark scripts
- Synthetic structured dataset with paired transformed views
- Multi-projector structured-refinement stack
- Sparse latent graph refinement
- Hierarchical vector quantization
- Ablation presets, benchmark sweeps, and benchmark reports
- Smoke tests, CLI tests, and model inspection
- Validated dataset adapters with split metadata and deterministic content hashes
- Immutable protocol records and manifested `planned -> running -> complete/failed` states
- Self-checksummed failure-localization artifacts and a repeatable rollout profiling CLI
- Explicit energy/contact/wall/momentum and symmetry diagnostics
- Deterministic complete-run archival and a rendered manuscript build

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

For the supported object-state inference API and its exact tensor/NPZ contracts, see
[`docs/USAGE.md`](docs/USAGE.md). Installed environments expose `ripii` and the legacy
`ripii-world` alias:

```bash
ripii inspect runs/world_experiment/graph_continuous/seed_3/best.pt
ripii rollout runs/world_experiment/graph_continuous/seed_3/best.pt \
  --input rollout.npz --output prediction.npz
```

Canonical repository-wide entry points are:

```bash
./scripts/preflight.sh
./scripts/test.sh
./scripts/run_smoke.sh
./scripts/analyze.sh
./scripts/verify_artifact.sh
./scripts/build_paper.sh
python3 scripts/index_runs.py
```

Profile a retained world-model checkpoint with warmup, repeated synchronized timing,
a bootstrap interval, throughput, environment metadata, and a clearly labeled
PyTorch-recognized FLOP lower bound:

```bash
python3 scripts/profile_world.py runs/world_experiment/graph_continuous/seed_3/best.pt \
  --output runs/world_experiment/graph_profile.json
python3 scripts/profile_world_efficiency.py runs/world_experiment \
  --output runs/world_experiment/suite_profile.json
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
Dead-code revival is an explicit, opt-in intervention through
`HierarchicalVectorQuantizer.revive_dead_codes` or
`qualify_quantizer.py --reset-dead-every`; it never mutates evaluation silently and its
schedule must be recorded in a new protocol.

Create a deterministic full archive of a retained run, including checkpoints and logs:

```bash
python3 scripts/archive_research_run.py runs/world_experiment \
  output/research/world_experiment.tar.gz
```

Dataset and model disclosure cards are under `docs/data/` and `docs/models/`.

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
an increased `--steps` budget. Older checkpoints must be evaluated with their recorded
historical source revision; 0.2 refuses to reinterpret them under corrected mechanism
semantics. Compatible tensors may still be imported with `--initial-state` for a fresh
development run, but that is a new model rather than historical reproduction.
Zero-weight/disabled mechanisms have no uncertainty offsets, evaluation weights
per-example metrics by batch size, constant features have zero effective rank, and
the zero synthetic transform is now the identity. These corrections change future
results; historical pilot artifacts were preserved unchanged.
