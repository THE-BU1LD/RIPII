# Dataset Card: NRI External-Simulator Adapter

## Identity and boundary

This adapter consumes locally generated Neural Relational Inference Springs and Charged
arrays prepared from the commit-pinned upstream source recorded by
`scripts/prepare_nri_data.py`. NRI is an external simulator implementation, not an
empirical or real-world dataset. Raw arrays are intentionally not distributed in the
current repository and must be regenerated under the upstream license.

## Field mapping

- NRI location and velocity are divided by the declared box size of five.
- Radius `0.04` and mass `1.0` are constructed padding fields, not observations.
- Actions are zero because the source trajectories have no compatible action field.
- NRI relation/edge labels are not consumed by the current predictor.
- Split names and content hashes are recorded in the prepared-data metadata.

Metrics involving property prediction are non-informative because radius and mass are
copied constants. Results must say "external synthetic simulator" and must not imply
real-world validation or observed physical-property inference.

## Leakage and verification

Train, validation, and test arrays must be disjoint and checksum-pinned. Preparation
must fail if upstream revision, shapes, dtypes, finite values, or metadata disagree.
Tests: `tests/test_nri_data.py` and `tests/test_nri_study.py`.

