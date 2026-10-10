# RIPII dependency reconciliation — 2026-10-10

## Purpose and scope

The failed September 19 checks on Dependabot PRs [#6](https://github.com/THE-BU1LD/RIPII/pull/6) and [#8](https://github.com/THE-BU1LD/RIPII/pull/8) used source older than the accepted audit and type repairs. This candidate combines their dependency changes on current main rather than duplicating those repairs or rewriting their failed-job history.

The source base is `317a6ec4163134a56cdc56ce3fd20f5a43decfb4`. Main already includes:

- Audit repair [#12](https://github.com/THE-BU1LD/RIPII/pull/12), merge `4b1821ecc3a9af779c3846c77e1156d6a5ae9fcf`. It exports the complete locked production dependency set without the unpublished local project and audits that set strictly.
- Type repair [#13](https://github.com/THE-BU1LD/RIPII/pull/13), merge `327d7abd1a409906d00b6261eed54baec43dbc9e`. All four repaired source files on this candidate are byte-identical to that accepted merge.

## Changes

| Original proposal | Recorded source head | Adopted change |
| --- | --- | --- |
| #6 | `b29a1860a587d39e4d59a3689e2ae53126df916e` | Permit `mypy>=1.17,<3` in project and lock metadata. The actual locked mypy version remains 1.20.2. |
| #8 | `1ef6e08a6a0059c8622ddda7923d2342e37cc760` | Pin the existing security job's `actions/setup-python` to `5fda3b95a4ea91299a34e894583c3862153e4b97` (v7.0.0). |

There are exactly three changed original lines in three original files. No package version, model implementation, objective, config, scientific protocol, retained result, test, coverage threshold, lint setting or type-check setting changes. Existing strict audit commands, workflow conditions, permissions and timeouts are preserved. No vulnerability IDs or type errors are ignored by this candidate.

PRs #7 and #19 remain separate action proposals. The later engineering drafts #20–#25 are not adopted by this dependency candidate. Their existing development records and closed budgets remain on their original branches. The latest inspected scope record is [PR #25 state at e7ceda3](https://github.com/THE-BU1LD/RIPII/blob/e7ceda31d94ce58595829d0b4ce607a212a9afc4/RESEARCH_STATE.json); its unmerged features are not represented as present on main.

## Audit interpretation

The accepted command remains:

```sh
uv export --locked --no-dev --no-emit-project --format requirements.txt --output-file /tmp/ripii-requirements.txt
python -m pip_audit --strict --no-deps -r /tmp/ripii-requirements.txt
```

The [official uv reference](https://docs.astral.sh/uv/reference/cli/#uv-export) specifies that `--no-emit-project` omits the local project while retaining its dependencies. The [official pip-audit documentation](https://github.com/pypa/pip-audit#usage) specifies that `--strict` rejects dependency-collection failures and `--no-deps` requires fully pinned input rather than resolving it again. This audits the exported third-party production dependencies; it is not a code audit of the unpublished local RIPII distribution or a claim about unselected optional extras.

The [setup-python v7 release](https://github.com/actions/setup-python/releases/tag/v7.0.0) removes the `pip-install` input. This repository uses only `python-version` and `cache` inputs, so that removed input is not part of its workflow contract. Hosted execution is still required to verify the action on the actual runner.

## Evidence and boundaries

- `historical_failures.json` retains the original strict audit and 22-error type-check failure excerpts, source heads, job IDs and URLs. These are historical failures, not current-main results.
- `main_source_manifest.json` lists the 262 original main blobs; only the three declared metadata/workflow paths differ. All other 259 original files remain byte-identical, including every runtime/test/scientific artifact path.
- `initial_checks.json` and `validation_checks.json` record commands, results and wall times. Absolute checkout prefixes are normalized to `<checkout>` for portable records; requirement pins and hashes are unmodified.
- Validation uses existing CPU resources and isolated tool dependencies. No scientific experiment, protected outcome access, paid compute, deployment, merge or evidence deletion is authorized by these checks.

The retained scientific disposition remains negative development evidence with external confirmation unresolved. A dependency integration or green CI check does not establish model usefulness, superiority, mechanism efficacy or research completion.

## Verification result

Local validation passed: lock freshness, production export, full repository Ruff, strict audit (34 selected third-party dependencies; zero skipped and zero known vulnerabilities), and package typing under both mypy 1.20.2 and 2.3.1 (42 source files each). The aligned local typing checks use the quality job's Matplotlib 3.11.1 and NumPy 2.4.6, while the local interpreter is Python 3.12.14 and the reused CPU Torch is 2.14.1+cpu. They are not represented as an exact installation of the complete hosted lock.

Earlier local environment checks failed with the workspace's older Matplotlib stubs and with Python 3.12-selected NumPy syntax against the package's Python 3.10 type target. Their raw diagnostics are retained. These were resolved by matching the actual quality-job dependency selection; no source or type setting was changed to pass.

`receipt.json` records local validation and its limits. Automatically triggered exact-head hosted results will be linked in the PR body after publication. The candidate remains an isolated unmerged review branch; original dependency branches and failed runs are retained.
