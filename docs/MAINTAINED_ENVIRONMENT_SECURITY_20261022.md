# Maintained environment security repair — 2026-10-22

## Observed failure

At PR #28 head `f27c947051d16741a4fe081d5f8f9c5748f45427`, the dependency audit exported the locked production environment successfully, then reported 24 known vulnerabilities in Pillow 12.3.0, each listing 13.0.0 as a fixed version. The failing run is [39770401159](https://github.com/THE-BU1LD/RIPII/actions/runs/39770401159), job [119796189777](https://github.com/THE-BU1LD/RIPII/actions/runs/39770401159/job/119796189777). This was an advisory finding, not a runner startup or lock-consistency failure. CodeQL succeeded.

## Repair and compatibility

The maintained dependency contract now requires Pillow >=13,<14 and Python >=3.11. Pillow's [13.0.0 release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html#python-3-10) explain that Python 3.10 support was removed. The CI matrix therefore covers Python 3.11, 3.12, and 3.13; the type-checker target agrees with the minimum runtime.

The compatible lock and equivalent UTC syntax come from the already reviewed environment subset of PR #32 at `62be826644ebf9b33d3589c9185464c3e27c1aad`. This extraction does not integrate that PR's scientific work. It retains PR #28's mypy >=1.17,<3 constraint in both project metadata and lock metadata.

The lock changes Pillow 12.3.0 to 13.0.0 and removes five alternative package resolutions needed only for the former Python 3.10 range: contourpy 1.3.2, exceptiongroup 1.3.1, matplotlib 3.10.9, networkx 3.4.2, and numpy 2.2.6. Every other shared package version remains unchanged. Four maintained utilities use the equivalent Python 3.11 `datetime.UTC` alias to satisfy the existing lint policy.

## Verification and evidence boundaries

The 58-package candidate lock passes `uv lock --offline --check` locally with Python 3.12.14. The changed Python files parse and compile without launching their entry points. Fresh GitHub dependency audit, tests, types, and coverage are required for this new head; previous green results do not certify it.

No advisory is ignored and no security or numerical gate is weakened. Historical Python 3.10 environments, retained receipts, frozen source snapshots, budgets, and scientific dispositions remain unchanged. This is a prospective maintained-environment repair, not a regeneration of research evidence.

