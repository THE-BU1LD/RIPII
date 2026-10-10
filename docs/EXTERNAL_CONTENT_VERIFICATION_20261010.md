# External trajectory content verification

## Observation and defect

At main `317a6ec4163134a56cdc56ce3fd20f5a43decfb4`, the generic external
dataset verifier checked artifact hashes and disjoint trajectory IDs. A generated
regression copied the entire training split into test, changed only the IDs, and
updated the manifest with the correct new archive size and hash. The actual
`verify_trajectory_dataset` returned `PASS`. A renamed copy is not a held-out
trajectory, and file integrity alone cannot establish split integrity.

## Implementation

`ripii/world/external_data.py` now computes a SHA-256 content fingerprint for
every validated trajectory. It hashes ordered live state and action values with
their field names and dimensions, using canonical little-endian float64 values.
Positive and negative zero are equivalent. All masked padding is removed before
hashing, so padding placement, contiguous storage, numeric dtype, and assigned
IDs cannot hide an exact copy of the same values. Live-object and time order are
preserved. Each split's verification record includes the fingerprint schema,
unique-content count, and a deterministic digest of its set of fingerprints.

The existing ID-overlap check remains active. Any pair of declared splits,
including additional OOD splits, is also rejected if their content sets overlap.
The new algorithm is versioned `ripii-trajectory-content-v1`; the NPZ interchange
format and model input values are unchanged. Working storage consists of one
canonical trajectory plus the fingerprint sets. This is a prospective check of
the generic external-data interface, not a reanalysis of retained studies.

## Required interpretation

An exact content match does not require identical file bytes or IDs. Conversely,
absence of a match does not establish independence. This check does not detect
overlapping windows from one source episode, near duplicates, different
precision approximations with different values, reordered live objects, or
common population provenance. Source/episode grouping and a protected split
protocol remain necessary. Identical trajectories repeated within one split are
reported through the unique-content count; this check does not redefine those
rows as independent experimental units.

The verifier already loads all declared splits. Do not invoke it on an
unapproved protected dataset: generated regression fixtures were the only data
used for this repair.

## Verification and history

On 2026-10-10, Python 3.12 with isolated CPU Torch 2.14.1 passed all **220 tests**
(211 existing and nine new). The targeted external-data set had 13 passing tests.
New fixtures cover full copied splits, one duplicate within unrelated rows,
additional OOD splits, canonical float32/float64 values, signed zero, shifted
padding, distinct action/state values, row-order-independent cohort hashes, and
the existing ID-overlap rejection. Existing actual train/evaluate and generic
benchmark smoke checks still pass. Ruff and whitespace checks pass on changed
source and tests.

No scientific training matrix, protected evaluation, external trajectory
download, or new statistical claim was run. All negative/no-advance pilot,
world, objective, repaired 0.2, and RIPII-MR evidence and frozen protocols remain
retained. `RESEARCH_TRUTH.md`, `EVIDENCE_LEDGER.md`, and
`research/protocols/README.md` remain authoritative for those histories. The
external confirmatory protocol remains a draft and unexecuted.
