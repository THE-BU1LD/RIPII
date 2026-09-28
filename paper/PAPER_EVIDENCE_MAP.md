# RIPII paper evidence map

Status: manuscript provenance map for \`paper/ripii_negative_results.tex\`.

This file records the repository surfaces used to prepare the manuscript. Exact numerical claims remain subordinate to the retained JSON/CSV/capsule artifacts, not to rounded prose or tables in the paper.

## Primary scientific evidence

| Paper surface | Repository source | Git blob SHA | Role |
|---|---|---|---|
| Corrected structured pilot | \`research/results/pilot_v2/summary.json\` | \`b2e1a6657cfcbafb549cc2eaf4159a27d865a825\` | 3-seed base / no-VQ / no-structured reconstruction, probes, code usage |
| World-v3 result capsule | \`research/results/development/world_v3_convergence_capsule_v2.json\` | \`afa40b381f48c3b3e198561ef4f5d2514e418c91\` | 5-seed graph/global/multiscale IID + OOD means, advancement checks |
| World-v3 paired statistics | \`research/results/development/world_v3_convergence_statistics.json\` | \`e9ecd44dc02e461973623009b582bb55016b593f\` | paired differences, bootstrap intervals, exact sign-flip tests |
| World-v3 failure localization | \`research/results/development/world_v3_failure_analysis.json\` | \`bd8b3646bd45e27fbd6e09da2121f8f57b4d5ff5\` | post-result regime analysis; exploratory only |
| Conditional coupling | \`research/results/development/world_v4_coupling_capsule.json\` | \`a7cee4a9cd76077b2b3776a84ccb5c8fd010db8c\` | prospective local-vs-coupled hierarchy gate |
| Longer coupling follow-up | \`research/results/development/world_global_coupling_v1_capsule.json\` | \`432fb1f9ee7304e8c409908cbe6e7ad68cfa49ad\` | fresh-seed 300-update graph/global/multiscale comparison |
| NRI development extension | \`research/results/development/nri_external_development_v1.json\` | \`fb4fc530513572c9fb733ebb680466a21274e504\` | commit-pinned Springs/Charged simulator development study |
| Objective study | \`research/results/development/objective_study_v1/summary.json\` | \`0d7a57edfef41759a94914bc99d51a9383d66e35\` | 39-run, 13-mode paired objective study |
| Gradient diagnostic | \`research/results/development/objective_study_v1/gradient_diagnostics.json\` | \`b0457b64c27597c7594e0d214d68f9be827ee7e3\` | weighted gradient norms and pairwise cosine diagnostics |

## Protocol and interpretation sources

| Source | Git blob SHA | Use |
|---|---|---|
| \`research/MATHEMATICAL_SPEC.md\` | \`c3b2723bc5cbff4a5eb1bfb6d07407a295bf23f6\` | model equations, world-model transition, multiscale grouping, coupling definition |
| \`research/HYPOTHESES.md\` | \`40a4a91e0ea80885d8927c803d4c288c778079dc\` | predeclared gates and development/confirmatory boundaries |
| \`research/NOVELTY_AUDIT.md\` | \`70172f93777e03fa94ca7102c6f13fe1dda94813\` | prior-work boundary; no novelty/SOTA claim |
| \`audit/FINAL_AUDIT.md\` | \`1336114bae142e5f7eae4c0bb79def82f298a79f\` | integrated interpretation, limitations, verification state |
| \`EVIDENCE_LEDGER.md\` | \`b5d2a1a719780a8314db70eae2f646c5df9eb8b2\` | current cross-study claim ledger, including repaired 0.2 and RIPII-MR follow-ups |
| \`research/protocols/confirmatory_external_v1.md\` | \`975a244e3992d0b94b57f28a090864e6ecf2bf1c\` | external confirmation remains draft / not frozen / not run |

## Manuscript values and derivations

### Corrected pilot

Computed directly from the three retained runs in \`pilot_v2/summary.json\`:

- full base mean reconstruction MSE: **0.2979114006**
- no-VQ mean reconstruction MSE: **0.2363620003**
- no-structured mean reconstruction MSE: **0.2362208565**
- full-base minus no-VQ: **+0.0615494003**
- full-base mean held-out probe accuracy: **0.4912280702**
- no-VQ mean held-out probe accuracy: **0.5087719298**
- no-structured mean held-out probe accuracy: **0.5789473684**
- full-base mean coarse effective-code fraction: **0.1477816751**
- full-base mean fine effective-code fraction: **0.1815032065**

### World-v3

The capsule report is the source for manuscript mean ± sample-SD values:

- graph: IID **0.0904 ± 0.0075**, more objects **0.1164 ± 0.0057**, composition **0.1203 ± 0.0058**, fast **0.1755 ± 0.0254**
- global pool: **0.0920 ± 0.0070**, **0.1235 ± 0.0056**, **0.1325 ± 0.0054**, **0.2167 ± 0.0262**
- multiscale: **0.0983 ± 0.0097**, **0.1216 ± 0.0033**, **0.1301 ± 0.0074**, **0.2130 ± 0.0275**
- graph-control mean OOD relative improvement for multiscale: **-0.1304836773**
- paired advancement passes: **0/5**
- frozen parameter counts: graph **101,896**, global pool **101,160**, multiscale **97,672**; each control is within the protocol's 5% parameter-count tolerance.
- retained analytic-baseline mean position RMSE (IID / more objects / composition / fast):
  - persistence: **0.3366 / 0.3051 / 0.3467 / 0.4884**
  - constant velocity: **0.2276 / 0.2415 / 0.2128 / 0.5536**
  - force-kinematic: **0.2276 / 0.2419 / 0.2116 / 0.5518**

The paired-statistics artifact is the source for the manuscript differences and finite-seed intervals. The parameter counts and analytic baselines come from the retained world-v3 capsule protocol/report.

### Long-range coupling

Prospective conditional-hierarchy study:

- local multiscale-minus-global more-objects mean difference: **-0.0019911677**
- coupled difference: **-0.0020360351**
- coupled relative mean difference: **-0.0109081643**
- coupled exact two-sided sign-flip p: **0.75**
- decision: **no_advance**

Longer fresh-seed follow-up:

- global pool has the lowest reported mean position RMSE in IID, more-objects, composition, and fast-motion regimes.

### Objective study

From the retained 39-run summary:

- reconstruction+KL reference mean reconstruction MSE: **0.1706779003**
- +geometry: **0.1719963402**
- +identity: **0.1730075876**
- +VQ: **0.1776892046**
- +equivariance: **0.1850805084**
- +spectral: **0.2022667875**
- +invariance: **0.2766338785**
- full legacy objective: **0.2714189788**

The frozen audit records that all twelve additions move in the wrong direction on all three paired seeds.

Gradient diagnostic:

- reconstruction weighted gradient L2 mean: **1.319954366**
- invariance weighted gradient L2 mean: **6.042615794**
- 63/78 objective pairs have a negative cosine on at least one seed.
- reconstruction-vs-invariance cosine is negative on all three diagnostic seeds.

## Follow-up evidence cited from the current ledger

The main branch ledger additionally records:

- repaired RIPII 0.2: full model mean reconstruction MSE **0.08410**, reconstruction+KL **0.04687**, plain AE **0.05573**, with both simpler controls better on every paired seed;
- RIPII-MR: **no_advance**, 75 learned-model runs, 0/60 paired comparisons passing, negative mean OOD improvement against every learned control.

These are kept as follow-up evidence and are not used to rewrite historical studies under repaired code.

## Explicit non-claims

The paper must not state or imply:

- state-of-the-art performance;
- general failure of hierarchy, VQ, multiscale GNNs, or graph networks;
- powered population-level significance from the small seed samples;
- real-world or independent confirmatory validation;
- exact compute matching where only parameters/update counts were matched;
- algorithmic novelty for the component mechanisms;
- that the draft external confirmation protocol has been run.

## Human release gates

Before any submission or archival release, authorized humans must finalize:

- author list/order and contribution statement;
- affiliations;
- conflicts/funding acknowledgments;
- software/data license;
- venue-specific formatting and disclosure requirements;
- final rendered-PDF review.
