# Active Hierarchy Is Not Necessarily Useful Hierarchy
## A negative development study of multiscale object-state prediction

**Internal manuscript draft | 20 October 2026**  
**Status:** Complete text draft synthesizing retained development records. No new model run, raw-checkpoint re-evaluation, external validation, or novelty certification. Authorship, license, independent review, and release approval remain open.

### Abstract

A learned hierarchy can remain active while failing to improve prediction. We examine the retained RIPII world-model v3 development record, which compares graph, global-pool, and multiscale predictors on generated two-dimensional object dynamics. The study uses a fixed dataset seed, five optimization seeds, and a ceiling of 1,000 updates. Reported mean in-distribution position RMSE is 0.0904 for the graph model, 0.0920 for global pooling, and 0.0983 for the multiscale model. Graph prediction also has the lowest mean error in each of three shifted regimes. The retained report records a -13.05% mean relative out-of-distribution improvement for multiscale prediction versus graph, with losses on all five paired seeds. The multiscale assignment remains active, averaging 2.77 effective groups out of four, but this diagnostic does not establish utility. These are generated-data development findings, not a compute-matched confirmation, a real-world result, or evidence about recursive self-improvement. We preserve the failed advancement decision and distinguish tested behavior from missing evidence needed for publication. [1,2]

## 1. Research question

The tested question is whether learned multiscale grouping improves object-state rollout prediction beyond simpler graph and global-context controls under the recorded development protocol. The system receives known object states; it does not infer objects from pixels. A successful state-input result would therefore not validate visual perception, language reasoning, or autonomous self-improvement. [1,2]

Relational inductive biases motivate structured predictors, but general motivation does not establish the utility of a particular learned grouping mechanism. Graph networks and Neural Relational Inference provide related methodological context; the local graph control in this study is not a reproduction of either complete published system. We make no benchmark-equivalence claim. [3,4]

## 2. Environment and predictors

The archived environment uses two-dimensional damped soft-disc dynamics with contacts, walls, drag, and external forces. The stated observation interval is 0.05 seconds, with four simulation substeps. Each object has position, velocity, radius, and mass. Masks represent variable object counts. Contacts are a soft-disc approximation, not exact rigid-body physics. [2]

The graph predictor applies learned local object interactions using relative positions, relative velocities, and distance features. The global-pool control supplements local interactions with a global mean scene representation and object-level refinement. The multiscale predictor instead performs learned soft grouping, coarse-node interaction, unpooling, and object-level refinement. The global-pool model is an important control: improvement over a purely local graph alone would not isolate the value of learned grouping from the value of ordinary global information. [2]

All learned predictors add corrections to a constant-velocity integration step and preserve radius and mass. They do not call the simulator to produce learned predictions. The recorded objective combines one-step state error and half the mean multi-step state error, with position coordinates weighted four and velocity coordinates weighted one. These implementation descriptions are attributed to the repository documentation; a fresh source-to-execution audit remains outstanding. [2]

## 3. Protocol and interpretation rule

World-model v3 uses dataset seed 9091 and optimization/minibatch seeds 23, 29, 31, 37, and 43. Graph, global-pool, and multiscale models have a training ceiling of 1,000 updates. The documentation describes shared training and validation trajectories and validation selection using position RMSE plus 0.25 times velocity RMSE. Parameter counts are approximately matched through hidden-width choices; exact counts must be recovered from the executed records before a precise capacity-matching claim. [1,2]

The retained setup trains on scenes containing two to four objects. Shifted evaluation regimes contain more objects, larger/heavier compositions absent in combination from training, or higher initial velocities. The archived general workflow uses 16-step training trajectories, a four-step training-rollout objective, and 32-step evaluation rollouts. The complete v3 protocol must be checked before treating every workflow default as an authenticated v3 hyperparameter. [2]

The reported development rule requires at least a 5% improvement in average shifted-regime position RMSE over the graph model, with no more than 5% in-distribution regression, on every seed and at least three seeds. It is a development advancement rule, not a statistical significance test. A failed rule remains failed; a favorable secondary comparison cannot retrospectively replace it. [1,2]

The recorded protocol SHA-256 is `fb8c3a1ca56f68a848b32c3e012b170b3a171ef3e4b5aeed96129854f82c116f`. The repository identifies a compact capsule at `research/results/development/world_v3_convergence_capsule_v2.json`. This revision reads the status and workflow records, not all original checkpoints or the complete capsule. We therefore attribute rather than independently certify the report's historical statement that 124 artifacts verified. [1]

## 4. Retained results

### Table 1. Mean in-distribution position RMSE

| Predictor | Mean position RMSE |
|---|---:|
| Graph | 0.0904 |
| Global pool | 0.0920 |
| Multiscale | 0.0983 |

Values are rounded means from the retained development records. Seed-level uncertainty cannot be reconstructed from these means alone. [1]

### Table 2. Available shifted-regime means

| Regime | Graph | Multiscale |
|---|---:|---:|
| More objects | 0.1164 | 0.1216 |
| Composition | 0.1203 | 0.1301 |
| Fast dynamics | 0.1755 | 0.2130 |

Lower RMSE is better. Global-pool regime-specific values were not retrieved and are deliberately omitted rather than inferred from aggregate improvements. [2]

Graph has lower mean error than multiscale in all four reported regimes. The archived report states that multiscale loses the graph shifted-regime comparison on all five paired seeds, with mean relative improvement -13.05%. It also reports only a 1.58% average shifted-regime improvement over global pooling, a 7.49% in-distribution relative regression, and passage of the secondary descriptive 5%/5% rule on two of five seeds. These relative quantities are retained as report-defined statistics; they are not recalculated by dividing rounded entries in the two tables. A mean of paired relative errors need not equal a ratio of pooled means. [1]

The study's recorded decision is **no advance**. A mean effective group count of 2.77 out of four indicates that grouping was not simply collapsed to one group under the recorded diagnostic. It does not show that the groups contained predictive information unavailable to the graph or global-pool model. Activity and utility are separate empirical claims. [1]

## 5. What failed, and what remains unresolved

The primary advancement criterion failed. The stronger development budget did not reverse the mechanism's deficit relative to graph prediction. A small secondary average improvement over global pooling neither rescues the failed graph comparison nor establishes a useful practical tradeoff once in-distribution regression is included. [1]

These observations are consistent with several explanations, including inefficient optimization, information loss through grouping and unpooling, unnecessary nonlocal interactions, or a mismatch between the environment and the proposed hierarchy. The current retained tables do not identify which explanation is correct. The next useful experiment would need to discriminate among such explanations rather than simply enlarge the training sweep.

Related earlier vector-reconstruction pilots and later coupling studies remain separate records. This paper does not pool their datasets, seeds, losses, or conclusions with world-model v3. In particular, the October 18 Kaggle notebook named RIPII Radial Mixture v3 is not presumed to be this world-model v3 study. No numerical output from that Kaggle notebook was obtained in this revision. [1]

## 6. Limitations

The dataset is generated and fixed across the optimization seeds. Five training seeds are not five independent physical systems. The retained summaries do not establish calibrated uncertainty over real-world tasks. Equal update counts and approximate parameter matching do not establish equal FLOPs, elapsed compute, or inference cost. Previous CPU profiling cannot support a new GPU efficiency claim. [1,2]

The baseline suite includes documented analytic controls elsewhere in the workflow, but their complete v3 values were not retrieved. This draft therefore makes no assertion that the graph model beats every analytic predictor. The omitted seed-level outputs and checkpoints must be recovered before final tables, paired uncertainty estimates, or a reproducibility release can be completed.

The public research status also records unresolved licensing and release requirements. No external publication is performed here, and no author's institutional affiliation or contribution is inferred from repository access. [1]

## 7. Conclusion

The retained development evidence argues against advancing the current multiscale mechanism on the strength of the world-model v3 results. Learned grouping remained active but did not meet the prespecified development criterion and had worse reported mean error than graph prediction in each listed regime. The defensible contribution of this draft is a bounded negative result with explicit provenance and missing-evidence limits, not a general claim against hierarchy or a completed positive architecture paper.

## References and evidence ledger

[1] `THE-BU1LD/RIPII`, `RESEARCH_STATUS.md`, Git blob `f81fd21740bdd4ac247c9441f11b72be09bddd63`, read on 20 October 2026. Historical status and numerical summaries; not a new execution receipt.

[2] `THE-BU1LD/RIPII`, `WORLD_MODEL.md`, Git blob `e81bac289caca971ea6987c1077f455b4edb978a`, read on 20 October 2026. Workflow, implementation, and reported regime means.

[3] Battaglia and colleagues. Relational inductive biases, deep learning, and graph networks. 2018. arXiv:1806.01261. Background only, not a measured baseline in this draft.

[4] Kipf, Fetaya, Wang, Welling, and Zemel. Neural Relational Inference for Interacting Systems. ICML 2018, PMLR 80:2688-2697. Background only, not a reproduction claim.

## Release conditions still open

Recover and verify the exact v3 protocol, executed source, capsule, all seed-level summaries, and raw artifacts; regenerate tables from that frozen source; resolve independent review, authorship, and license; and obtain an explicit publication decision. Further GPU development, if pursued, needs a distinct admitted hypothesis and must not rewrite this failed study.
