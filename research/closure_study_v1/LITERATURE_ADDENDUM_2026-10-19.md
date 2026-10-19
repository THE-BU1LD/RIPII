# Post-specification literature addendum — 19 October 2026

## Retrieval timing and study boundary

The additional primary source below was retrieved on **2026-10-19**,
after implementation and development execution had begun.
The implementation was already published as
`6b99a71768e429ace84116116a7ca8ffca06f35f`.
This is a retrospective literature-coverage addition, not part of the original
prospective literature record. Its publication date precedes this study;
its discovery here does not establish that the original review was exhaustive.

This retrieval causes **no changes to hypotheses, architecture, tuning,
hyperparameters, endpoints, budgets, exclusion rules, analysis, or recorded
experimental evidence**. It does not motivate an extra run or a favorable-result
gate. Existing source, tests, specification, protocol, bibliography and
pre-development documents are preserved. No confirmation data or outcome was
accessed for this addendum.

## [A1] Additional close primary source

Vinay Sharma and Olga Fink. *Learning coarse-step dynamics and internal
mechanical response with graph networks*. arXiv:2609.30344v1, 24 September 2026.
[Primary full text](https://arxiv.org/html/2609.30344v1).
BibTeX key: `sharma2026coarsestep` in [addendum.bib](addendum.bib).

Relevant Methods and Supplementary sections were inspected. The following
compact account is the source-derived summary for this addendum.

| Aspect | Inspected overlap | Source section |
| --- | --- | --- |
| Substeps | Four recurrent substeps span each observation interval; drives and operators are reevaluated. | Methods 4.2 |
| External effects | Observed loads enter explicitly; otherwise external forcing is decoded from current state. | Methods 4.4.3; Supplement 10.6 |
| Exchanges/projection | Physical edges exchange antisymmetric fluxes; a massless hub uses collective zero-sum projections. | Methods 4.4.1 |
| Nonlocal coupling | An operator-weighted virtual hub supplies system-wide communication. | Methods 4.5 |
| Interpretation | Learned response operators are coarse-step indicators, not uniquely identified constitutive properties. | Introduction; Methods 4.6 |

**Conservation caveat:** Independent nodal Newmark solves need not preserve
force-level cancellation; the derived local linear-momentum residual is
`O(dt²)` under boundedness assumptions. Hub zero-sum fluxes do not alone
guarantee zero moment about a common origin. Complete angular and trained-rollout
conservation are not established (Supplement 11).

## Implications for this study

The overlaps above further narrow permissible novelty language. Combining the
listed ingredients cannot be presented here as an established original
architectural contribution. Differences in formulas, application domains or
implementation choices do not by themselves establish novelty, and this
addendum does not assert that RIPII's diagnostic is unique.

The unchanged contribution boundary remains a **controlled representability,
misspecification, and hierarchy diagnostic under a fixed synthetic development
and confirmation protocol**. Its hypotheses must be judged against that
protocol, including simpler controls and compute coverage. This paper is a
related method, not an executed baseline in this study. No relative accuracy,
efficiency or state-of-the-art claim against it is supported by this retrieval.

The conservation qualification reinforces the existing distinction between a
component-level identity and a guarantee for the complete returned state.
For RIPII, only its own derivations, numerical checks and retained measurements
can establish the scope of such a guarantee. No result from [A1] is substituted
for that evidence.

This source-specific pass establishes no additional correction to the metadata
of the original ten references. Their records remain as originally written;
any later verified correction must likewise be dated rather than silently
backdated into the prospective record.
