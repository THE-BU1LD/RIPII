# External trajectory container identity

## Defects and implemented contract

Valid leaf hashes previously admitted NPZ files below symlinked directories,
including paths outside the declared dataset root. NPZ membership was checked
using a set, hiding duplicate names. Hashing a path and reopening it for parsing
also allowed a replaced archive to be loaded under an earlier digest; rereading
the manifest digest could bind old metadata to a new manifest.

Every lexical path component is now checked for symlinks, and resolved paths
must remain under the declared root. An archive must contain exactly four
members and the required states/actions/mask/ids names. Its compressed bytes are
read once; length and SHA-256 are checked on those bytes, which are then passed
to NumPy through BytesIO. Manifest JSON and digest are derived from one byte
snapshot. Complete-dataset verification shares that parsed snapshot across all
split loads and records, preserving the existing ID/content overlap checks.

## Discriminating evidence

The seven new fixtures include inside/outside parent symlinks, a valid real nested
directory, duplicate NPZ members, an archive replaced immediately before np.load,
a manifest replaced after parsing, and a manifest changed between split loads.
On the exact parent, six cases fail and the valid-directory control passes.
The complete affected external loader/content/container suite passes 20 tests,
including the retained one-step train/evaluate and benchmark plumbing fixtures.
The earlier 17-test check and four-case baseline attempt remain separately saved.

## Limits

The compressed archive now remains in memory during load; there is no memory
reduction or decompression-quota claim. These checks bind consumed bytes to their
recorded hashes and reject the tested path defects; they are not a guarantee
against privileged filesystem races or hostile resource exhaustion. Exact content
hashes do not establish independent source episodes or detect approximate copies
and overlapping windows. No protected data was accessed. All negative pilot,
objective and RIPII-MR conclusions and the draft external boundary are preserved.

## Verification identity and reproduction

Exact source parent: `66ee627b32387e86bcc4bc0d57e725f63a21fe60`. Revision: `ripii.external-container-identity.1`.
The source/test SHA-256 map and test environment are retained in
`research/verification/maintenance_20261010/receipt.json`. Failed constructed witnesses and any JUnit files
remain in the same directory. These are engineering artifacts, not scientific
results or a research-completion checkpoint.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest -o addopts='' -q tests/test_external_trajectory_data.py tests/test_external_content_overlap.py tests/test_external_container_identity.py
```
