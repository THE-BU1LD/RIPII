# Radial adaptation v1: convergence-qualified comparison

This new study preserves radial-mixture v3 and its terminal
`INCONCLUSIVE_UNDERTRAINED` result. It asks whether scene-adaptive mixing improves
64-step position prediction over the same two radial experts trained with a fixed
0.5 mixture, and whether either radial method beats graph, global-pool and
Transformer controls. It uses the exact source archived by the original v3 run,
copied under `vendor/`, with original source and evidence hashes retained. Historical
losses, seeds, budgets, protocols and decisions are not changed.

## Fair training and adequacy

Each model/domain receives two learning-rate trials on a separate calibration data
block. Both trials receive identical maximum updates, scheduler and stopping rules;
all trials, costs and failures are retained. Choose by minimum 64-step validation
position RMSE among adequate trials. No test data participates in this selection.
All models use their v3 widths, shared batch size and AdamW settings; log allocated,
gradient-active parameter counts, elapsed time, updates and selected checkpoints.
Equal tuning opportunities are not a claim of equal FLOPs or efficiency.

Train on horizons 1, 2, 4, 8, 16 and 32, then spend at least 3,000 updates on the full
64-step objective. Allow up to 16,000 updates. Validate every 500 updates. Learning
rate reductions use validation position RMSE only. Stop only when both best-so-far
training and validation errors improve by at most 1% over the last four checks and
the selected checkpoint's training position RMSE is at most 75% of the
constant-velocity reference on the same complete training set. This threshold is a
prospective operational adequacy requirement, not proof of global convergence.
A plateau at the initial predictor therefore cannot pass.

The same rule applies to every final candidate and baseline fit. An undertrained
fit is retained and prevents protected evaluation of the entire main matrix. No
post-result longer run, threshold relaxation, seed removal or replacement is allowed.

## Data and controls

Two domains, six fresh generator blocks and two nested initialization/minibatch seeds
give 120 main fits, plus 20 calibration fits. Data counts and simulator semantics
match v3: 512 train, 128 validation, 128 test scenes; five objects; eight observed
frames; 64 forecast frames; sample frequency 50; divide positions/velocities by five.
Fresh role-separated dataset seeds are checked against retained v3 identities.
Official simulator edges are retained for provenance and excluded from all inputs.
Primary standard data and three shifted regimes remain unopened until selection and
adequacy are complete for the entire matrix. No protected-test retries are allowed.

The fixed model retains the unused gate parameters of the original v3 control;
they are reported separately and are not advertised as equal effective capacity.
Both radial models have exactly the same two experts, boundary model and training
opportunities. The adaptive model additionally gets a frozen-checkpoint intervention:
replace its scene gate with its mean over training scenes and forecast steps, without
retraining. This tests input-dependent adaptation while preserving the fitted experts.
The mean is fixed before test generation. Report intervention results separately from
the independently trained fixed control; intervention distribution shift is a caveat.

## Inference and falsifier

Position RMSE is primary; velocity error, wall time, gate variation and per-horizon
errors are secondary. Average nested model seeds within each generator block; there
are six independent blocks per domain, not twelve independent datasets. Report all
paired block gains and exact one-sided sign-permutation p-values on log RMSE ratios,
with Holm correction over the two primary domain comparisons. The randomization
interpretation assumes paired exchangeability under the null.

An adaptive benefit requires >=2% mean paired improvement over the trained fixed
control in each domain, positive improvement in every independent block and both
Holm-adjusted p-values <=0.05. A claim specifically about scene-dependent adaptation
also requires >=2% mean gain over the train-mean gate intervention in each domain and
positive gains in every block. Otherwise report `ADAPTIVE_BENEFIT_NOT_ESTABLISHED`,
or `INCONCLUSIVE_UNDERTRAINED` when adequacy fails. Failure is not equivalence.
Strong-baseline >=5% gains and <=5% OOD regression gates are reported separately;
success against the fixed model alone does not imply baseline superiority.

## Resources, provenance and execution

Private free Kaggle only, no paid fallback. The prospective new budget is at most
4 allocated GPU-hours, conservatively charging two allocated GPUs for every host
second even though only GPU 0 is used. Maximum host wall time is two hours including
qualification, data, tuning, failed work and export. A full-shape timing stage must
project the complete maximum-update matrix plus data/evaluation and a 1.5 safety
factor within the remaining cap before any tuning. If infeasible, retain the profile
and stop; do not shrink the study silently. A new user budget decision can precede
any training, but cannot change the frozen inference rules after outcomes.

Local numerical work obeys the existing >=512 MiB and stable pageout/swap guard.
Hosted execution records two resource samples, installed versions, CUDA device,
source/protocol hashes and a full artifact manifest. Raw arrays, trajectories,
learning curves, selected checkpoints, checkpoint replays and independently
recomputed adequacy/metrics are mandatory. No positive interpretation is emitted
from a partial matrix. Each attempt has a distinct directory and contributes to
cumulative accounting; old attempts are never overwritten.
