# Radial evidence guard repair — 2026-10-22

## Reproduced defects

Review of PR #32 at `62be826644ebf9b33d3589c9185464c3e27c1aad` found two prospective intake/transport defects. Local synthetic fixtures reproduced both without contacting Kaggle or running any scientific model.

The archive intake accepted four self-consistent charge records whose prior, current, elapsed, or reserve component was negative, while the cumulative total and allocation formula still agreed. All four returned PASS in the original implementation. This could understate a future resource charge.

The transport's free-GPU check used a Python assertion. Compiling the original source with optimization level 2 removed that assertion: a mocked quota response with paid scaling enabled reached the mocked push operation and wrote a local launch receipt. No real launch occurred.

## Narrow repair

Every charge component must now be a non-boolean integer or float, finite and within the nonnegative authorized 14,400 GPU-second envelope. The original cumulative equality, two-GPU allocation formula, full source binding, archive manifest checks, and total budget cap remain required. The maximum component check also rejects huge integers before float conversion.

Transport validation now uses explicit exceptions, including the notebook-format and HTTPS checks that previously also depended on assertions. A GPU push requires an explicit boolean false for paid scaling; missing or ambiguous quota information refuses launch. Existing private-notebook, project-prefix, duplicate-receipt, timeout, and internet defaults remain unchanged.

## Validation and preserved evidence

Eight archive-intake tests pass, including the four cancelling-negative cases, 30 malformed/nonfinite component cases, and the exact cumulative-budget boundary. Four local transport tests exercise optimization levels 0 and 2. Paid/unknown quota and invalid notebooks never reach the mocked push; non-HTTPS artifacts never open a URL. Confirmed free quota reaches only the mock and retains the private/no-internet request contract.

The quality workflow includes these transport regressions. Fresh exact-head GitHub checks are still required after publication.

These are prospective engineering validation changes. No frozen model, hosted receipt, retained negative attempt, budget value, result table, or scientific disposition is rewritten. No scientific training/evaluation or cloud operation was performed. The existing radial adaptation and optimized execution studies remain cost blocked and scientifically NOT_EVALUATED.

