# Review-branch environment repair

GitHub dependency audit reported eight Pillow 12.3.0 vulnerabilities, each listing 13.0.0 as the fix. The review branch now requires Python >=3.11 and Pillow >=13,<14, resolves the lock accordingly, and tests Python 3.11, 3.12 and 3.13. This deliberately ends Python 3.10 compatibility for the proposed maintained environment. The user's pre-existing dirty pyproject, lockfile and quality workflow in the working checkout were preserved; the change was constructed in an isolated temporary directory and is visible in PR32.

This changes the prospective maintained/reproduction environment, not the environment or bytes of any historical Kaggle result. Historical source and environment receipts remain intact. Re-running a frozen study under a different dependency set would require a separately identified attempt and budget.

The security advisory identifiers reported by CI were CVE-2026-112895, CVE-2026-112400, CVE-2026-112397, CVE-2026-112896, CVE-2026-112897, CVE-2026-112395, CVE-2026-112404, and CVE-2026-112899. No advisory was ignored or suppressed. Official package metadata confirms Pillow 13's Python >=3.11 floor: https://pypi.org/pypi/pillow/13.0.0/json.

Scoped Ruff configuration excludes preserved hosted snapshots, upstream/vendor copies, and old portable notebooks from active-source style edits. Maintained source keeps the same lint rules. The quality workflow additionally runs the 14 radial control tests, five archive-adversary tests, and exact portable-payload reproduction.

The Python 3.11 floor also requires the equivalent datetime.UTC spelling in five maintained utility files; the review branch migrates these timestamps without altering frozen source snapshots.
