# RIPII negative-result synthesis

Status: **evidence-bounded development/frozen-local negative result; no superiority or external-generalization claim**.

## Decision

Do not rescue-tune the current RIPII hierarchy. Two independently motivated hierarchy mechanisms have now failed their prospectively stated local advancement criteria:

1. the legacy projective/graph/vector-quantized structured stack fails the corrected frozen pilot v2; and
2. the later learned multiscale object-dynamics hierarchy fails the stronger five-seed, 1,000-update world-model development gate.

The scientifically justified default is the simpler flat graph reference model. A successor architecture must be treated as a new hypothesis with a separately frozen protocol before any outcome access.

## Evidence chain

### Legacy structured representation path

Pilot v1 is retained but is not used for causal component ranking because zero-weight objectives still received learned uncertainty-offset gradients and global clipping confounded component attribution. Its conservative no-advance decision remains valid as historical negative evidence.

Pilot v2 prospectively corrected that defect and used three fresh seeds plus retained exact initial states. The full RIPII model reconstructed worse than the no-quantizer control on all three seeds. Relative to `no_vq`, the base model's mean test reconstruction MSE was worse by 0.06155. Relative to removal of the four structured mechanisms, reconstruction was worse by 0.06169 and held-out probe accuracy was worse by 0.08772. Base effective-code fractions were 0.148 coarse and 0.182 fine, below the frozen 0.25 gate. These data do not support retaining the current hierarchical quantizer.

### Object-dynamics hierarchy path

The later world-model work isolates learned multiscale grouping from ordinary local graph processing and generic global context. World v1 and the five-architecture v2 controls both returned `no_advance`. The stronger v3 convergence study changed the dataset seed, used five fresh training/minibatch seeds (23, 29, 31, 37, 43), raised the training ceiling to 1,000 updates, and compared graph, global-pool, and multiscale models under the prospectively written protocol whose SHA-256 is `fb8c3a1ca56f68a848b32c3e012b170b3a171ef3e4b5aeed96129854f82c116f`.

The v3 result is again `no_advance`:

- mean IID test position RMSE: graph 0.0904, global-pool 0.0920, multiscale 0.0983;
- graph has the lowest mean error on every retained OOD split;
- multiscale's average OOD relative improvement versus graph is -13.05%;
- multiscale loses the graph OOD comparison on all 5/5 paired seeds;
- against global-pool, multiscale has only 1.58% mean OOD improvement while mean IID relative error is 7.49% worse;
- learned grouping remains active at 2.77 effective groups out of four, so the failure cannot be explained simply as numerical group collapse.

The signed compact v3 capsule remains the authoritative retained source for exact values and artifact identities.

## Mechanism failure analysis

The evidence supports several narrow inferences and rejects stronger ones.

**Supported:** activity is not utility. The multiscale assignment mechanism is numerically active, yet the simpler graph wins consistently in the strongest retained comparison. The graph already encodes the dominant local interaction prior of the soft-disc system, so unconstrained learned coarse groups appear to add approximation/optimization burden without enough compensating information.

**Supported:** the current quantizer is not justified. Near-collapse in the corrected pilot and worse reconstruction versus bypass are direct evidence against keeping the current discrete hierarchy.

**Not supported:** a claim that hierarchy is universally harmful, that graph networks are universally optimal, or that RIPII has been falsified on real physical systems. All current outcome-bearing evidence is synthetic/local and seed counts remain small.

## Successor-hypothesis audit

There are genuinely new falsifiable ideas, but the evidence does not justify immediately coding another rescue architecture. The highest-value next step is external negative confirmation: test whether a local graph equals or outperforms the current learned-grouping variant on at least two versioned public physical-dynamics datasets with validation-only tuning and matched wall-clock/FLOP budgets.

A later successor may test a **conservation-aware residual**: project predicted pairwise impulses onto a momentum-conserving subspace in regimes without walls/drag. This is materially different from the failed learned grouping mechanism. Its control must be the same graph backbone without the projection plus the analytic force reference; the hypothesis is falsified if conservation diagnostics improve but long-horizon predictive error does not. It must receive its own versioned preregistration before any test outcome is accessed.

A geometry-aware grouping successor is also falsifiable, but it is lower priority because the strongest evidence currently favors removing hierarchy rather than refining it.

## Reproducibility boundary

No historical result is rewritten by this synthesis. Frozen local pilot artifacts keep their original hashes. World v3 exact values are bound to the retained signed capsule. The canonical verification paths remain those documented in the repository README and `scripts/verify_artifact.sh`.

Current scientific state: **NEGATIVE_DEVELOPMENT_EVIDENCE_PACKAGE_COMPLETE / EXTERNAL_CONFIRMATION_REQUIRED**. This is not a submission-ready external result and does not authorize a post-hoc rescue of the failed mechanisms.
