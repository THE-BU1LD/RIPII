# Dataset Card: RIPII Synthetic Soft Discs v1

## Identity and intended use

- Dataset ID: `ripii.synthetic.soft_discs`
- Version: `1`
- Source: generated locally by `ripii/world/physics.py`
- Intended use: controlled development tests for object-state dynamics prediction
- Not intended for: real-world performance, safety, robotics deployment, or claims of empirical physics fidelity
- License: `NOASSERTION`; generated tensors inherit the repository's unresolved license status

## Schema and units

Each state is `[x, y, vx, vy, radius, mass]`. Position and radius use arena units in
the square `[-1, 1]^2`; velocity uses arena-units/time; mass uses an arbitrary mass
unit. Actions are force vectors in mass*arena-unit/time^2. `mask=true` identifies live
objects; every padded state and action is exactly zero.

## Generation and splits

`DatasetSpec` records the generator version, seed, split, scene count, horizon,
capacity, and complete `Physics` parameters. Each named split has a disjoint RNG domain.
Training uses 2-4 objects and excludes the large/heavy combination. `more_objects` uses
5-8 objects, `composition` holds out large/heavy objects, and `fast` changes the initial
velocity scale. The generator records a SHA-256 digest over typed tensor contents.

## Known limitations and leakage boundary

All splits share one generator, renderer/integrator, state contract, and largely the
same parameter support. They are RNG-disjoint but not independently sourced domains.
The data are exact, noiseless simulator states and can expose generator-specific
shortcuts. Random piecewise-constant actions are not a realistic control policy. Model
seeds do not measure generator-seed uncertainty unless `data_seed` is also varied.

## Reproduction and verification

Construct a `DatasetSpec`, call `load_dataset`, and retain its returned record. Verify
the record's `content_sha256`, tensor shapes/dtypes, unique IDs, split-domain IDs, live
properties, and zero padding. Tests: `tests/test_dataset_spec.py` and
`tests/test_world.py`.

