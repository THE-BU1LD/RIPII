# Locked dependency security repair | 27 October 2026

## Observed failure

The scheduled security run [40305481234](https://github.com/THE-BU1LD/RIPII/actions/runs/40305481234), at source `a53db91a202cf8bed57a98e9c42f214eae49ec22`, successfully exported the locked production dependencies and then failed its unchanged strict dependency audit.

The auditor reported filelock 3.32.5 for CVE-2026-112940, CVE-2026-112913 and CVE-2026-112907, each with 4.2.1 as the fixed version; it reported setuptools 84.0.0 for GHSA-xmj3-9gf7-h52w, with 84.1.0 as the fixed version. CodeQL passed in that run.

## Bounded change

Only the filelock and setuptools records in `uv.lock` change:

| Dependency | Before | Candidate |
| --- | --- | --- |
| filelock | 3.32.5 | 4.2.4 |
| setuptools | 84.0.0 | 84.1.0 |

The candidate versions are published on [PyPI for filelock](https://pypi.org/project/filelock/4.2.4/) and [PyPI for setuptools](https://pypi.org/project/setuptools/84.1.0/). The exact source and wheel URLs and SHA-256 hashes were taken from those release pages. Optional artifact size and upload-time fields are omitted for these two records because exact byte sizes and timestamps were not retrieved; they are not guessed.

The filelock release requires Python >=3.11, matching this project's existing floor. Setuptools 84.1.0 requires Python >=3.10. Both packages have no mandatory runtime dependencies in their release source metadata. The [setuptools changelog](https://setuptools.pypa.io/en/latest/history.html#v84-1-0) documents the advisory repair. The [filelock changelog](https://py-filelock.readthedocs.io/en/latest/changelog.html) documents the 4.2.1 fixes and later patch releases.

All other package records, dependency edges, resolution markers, project metadata, workflows, scientific source and retained evidence remain byte-for-byte unchanged by this repair.

## Local verification

Performed on Python 3.12.14:

- Original retrieved source contents were verified against their Git blob SHA values.
- Both original and candidate pass `uv lock --offline --check`.
- Candidate passes the workflow's production requirements export with `--offline` added: `uv export --offline --locked --no-dev --no-emit-project --format requirements.txt`.
- Parsed original/candidate lock comparison confirms that the package count remains 58 and exactly the two named package records differ.
- Exported requirements contain the candidate versions and the four registry SHA-256 hashes.

These are lock consistency and source/provenance checks. Package installation, the strict live vulnerability audit, and the full compatibility suite must be evaluated by fresh hosted CI on the PR's current source. Local package download was unavailable because the workspace's configured network proxy could not be reached.

## CI and research boundaries

The existing PR workflows run CPU tests, quality/coverage/build checks, dependency audit, CodeQL and hosted macOS capability smoke. Existing optional self-hosted CUDA and real-MPS jobs remain governed by their existing variables; neither is enabled by this change. Both optional jobs were skipped in the latest observed device-smoke run, 40295978096.

No advisory ignore, exception, or relaxed gate is introduced. No scientific run is launched and no historical environment, result, claim or frozen source snapshot is relabeled. This is a maintained environment repair awaiting current-source hosted verification, not a new research result.

## Pre-merge refresh | 9 November 2026

The latest scheduled main audit, [42621155843](https://github.com/THE-BU1LD/RIPII/actions/runs/42621155843), still fails at base `a53db91a202cf8bed57a98e9c42f214eae49ec22`. Its dependency-audit job reports 13 advisories: 12 for filelock 3.32.5 and GHSA-xmj3-9gf7-h52w for setuptools 84.0.0. CodeQL passes. The filelock fixed versions reported by that audit are 4.2.1, except CVE-2026-112936 at 4.0.9; this candidate's 4.2.4 covers those reported minimums. The setuptools fixed version remains 84.1.0.

A fresh source review confirmed that PR #35 at `70708c7911b0c994976ea791c5483b5c110ff0b4` remains limited to these two lock records and this evidence document. No workflow, audit exception, scientific source, or retained result changes. The repository has no submitted or outstanding inline PR reviews, and GitHub reports a clean, mergeable PR. These observations do not replace the checks below.

Fresh local validation used the exact GitHub blobs on Python 3.12.14:

- Base lock blob: `6aa796058950cf4d12e46121f0f2cdc16ae0b3b5`.
- Candidate lock blob: `878fdf2f33d7f7aee69e72764335e5582493785c`; SHA-256 remains `65dba25cd57b41b3df303d5dae4eade3951b98934a0bc96a9cda0cbf3226d4cc`.
- Both base and candidate pass `uv lock --offline --check`.
- Candidate passes `uv export --offline --locked --no-dev --no-emit-project --format requirements.txt`.
- A parsed record-by-record comparison confirms 58 package records, exactly the filelock and setuptools records changed, all 56 other records unchanged, and all resolution metadata unchanged. NumPy has two version-specific records, so counting unique package names is not an equivalent check.

The four successful hosted workflows at the earlier PR head were run on 27 October. They are historical evidence, not a fresh live vulnerability audit. This evidence refresh leaves the existing CI and strict security workflows unchanged so the updated PR head receives current hosted validation before merge. After merge, verify the resulting main security run separately. Optional real-MPS and CUDA certification remain governed by their existing configuration and are not claimed by this repair.
