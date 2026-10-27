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
