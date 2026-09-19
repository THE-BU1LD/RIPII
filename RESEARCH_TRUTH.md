# Research truth

Evidence status: **EVIDENCE_PARTIAL — negative development evidence**.

RIPII has real, tested implementations. It has no validated novelty, superiority,
real-world utility, physical renormalization, or publication claim. Frozen pilot v1 is
methodologically compromised by a now-fixed adaptive-loss bug and is retained only as
historical negative evidence. Corrected pilot v2 is credible local negative evidence:
the full model reconstructs worse than quantizer bypass and the structured-removal
control, with near-collapsed codebooks. The stronger world-model v3 development study
also rejects advancement: multiscale loses the graph OOD comparison on all five seeds.
The later NRI Springs/Charged external-simulator study likewise returns `no_advance`.

External-simulator development is not real-world or confirmatory validation. A smoke
test is not an experiment.
Assignment activity is not mechanism efficacy. Equal updates and similar parameter
counts are not equal compute. The compact v3 capsule verifies summaries/provenance but
does not contain checkpoints. No new confirmatory experiment was run in this audit.

The simplest evidence-supported method is the flat object graph model. The appropriate
next scientific action is an externally grounded, compute-matched comparison—not
post-hoc tuning of the failed hierarchy.

## Long-range extension

A prospectively specified development extension added a symmetric, momentum-preserving
long-range force. This makes global information relevant, but multiscale did not gain
the required consistent 5% advantage over a simple global-pool control. Its mean
coupling-specific interaction was essentially zero across three seeds. A separate
300-update follow-up likewise put global pool first in mean IID and every mean OOD
regime. Neither extension is confirmatory or external evidence. Under known nonlocal
dynamics, current evidence supports simple global aggregation, not learned grouping.

## Objective simplification and compute boundary

A frozen local 39-run study compared reconstruction+KL with eleven individual
auxiliary additions and the complete objective on three fresh paired seeds. Every
addition worsened fixed-budget reconstruction on every seed; the complete objective
was worse by 26.4%, 60.0%, and 89.5%. A post-training gradient diagnostic found large
scale differences and negative gradient cosines, including near-opposition between
projective and geometry terms. This supports reconstruction+KL as the legacy reference
at this budget, but it is not a convergence or external-data claim.

Machine-local profiling also shows that similar parameter counts do not imply equal
recognized operations and that latency ordering changes with host load. No efficiency
or compute-matched conclusion is authorized.

## Repaired 0.2 objective matrix

The prospectively frozen repaired-objective study completed all 180 cells and returned
the predefined `no_advance` decision. Across five paired seeds, repaired RIPII 0.2 had
mean reconstruction MSE 0.08410, versus 0.04687 for reconstruction+KL and 0.05573 for
the capacity-matched plain autoencoder; both controls beat the base model on every seed.
The lowest observed secondary mean was 0.04329 for reconstruction+KL plus equivariance,
but that 36-condition matrix result is exploratory and is not a post-hoc positive claim.

This is strong negative synthetic development evidence, not confirmatory evidence. Five
pairs give a minimum two-sided exact sign-flip p-value of 0.0625, and the multiplicity-
adjusted comparisons are not significant. The result therefore rejects advancement of
this repaired configuration without establishing a population-wide superiority claim
for any control.

## RIPII-MR development matrix

The separately frozen RIPII-MR matrix completed all 75 learned-model runs and returned
`no_advance`. Across the 15 paired data/model-seed cells per control, its mean OOD
relative improvements were -0.1596 against the equivariant control, -0.7824 against the
graph control, -1.1909 against global pool and -1.2134 against legacy multiscale. Every
one of the 60 candidate/control comparisons failed the rule requiring at least 5% OOD
improvement without more than 5% IID regression.

This is a negative synthetic development result. The three generator seeds are the
independent units; the five model seeds nested within each generator seed do not create
15 independent datasets, and the sign-flip calculations in the retained summary are
descriptive rather than powered confirmation. The result does not support accuracy,
novelty, external-validity, significance, efficiency or publication claims. Under the
predeclared failure criterion, this architecture version does not advance.

The current router suppresses the coarse contribution only after restriction, coarse
edge construction and coarse message passing have run. It is contribution gating, not
verified conditional computation, and the completed matrix does not establish an
accuracy-compute benefit.
