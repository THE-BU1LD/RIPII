# Learned hierarchy does not improve the tested RIPII synthetic benchmarks

## Abstract

We test two forms of learned hierarchy in RIPII: a structured representation stack with hierarchical vector quantization, and learned soft multiscale grouping for object-state dynamics. Both fail their prospectively stated local advancement criteria. In the corrected frozen representation pilot, the full model reconstructs worse than quantizer bypass and structured-mechanism removal while its codebooks remain near collapse. In the strongest object-dynamics study, a five-seed, 1,000-update multiscale model loses the flat graph OOD comparison on every paired seed despite maintaining active group assignments. These are bounded synthetic/local negative results, not external validation, population inference, or a novelty claim. The present evidence favors the simpler graph reference and does not justify post-hoc rescue tuning of the failed mechanisms.

## Scope, method, and protocol

The legacy representation study evaluates the projective, graph, quantizer, and latent-action stack under frozen local pilot protocols. Pilot v1 is retained as historical negative evidence but is not used for causal component ranking because its loss-balancing defect and global gradient clipping confound that interpretation. Pilot v2 prospectively corrects the loss-balancer defect, uses fresh seeds and exact retained initial states, and evaluates disjoint train/test representations.

The later object-dynamics study uses exact object state and applied force; learned perception is excluded. Mathematical contracts are in `research/MATHEMATICAL_SPEC.md`. Training uses a four-step autoregressive objective and validation-only checkpoint selection. Test regimes change scene identity, object count, held-out radius/mass composition, and initial velocity. Persistence, constant-velocity, force-kinematic, MLP, Transformer, global-pool, and flat graph controls separate hierarchy from capacity and generic global context. The strongest v3 study uses a new data seed, five new training/minibatch seeds, and a 1,000-update ceiling under a prospectively written development protocol.

## Results

In corrected pilot v2, full RIPII mean reconstruction is worse than `no_vq` by 0.06155 and worse than removal of the four structured mechanisms by 0.06169; held-out probe accuracy is also 0.08772 lower than that structured-mechanism removal. Base effective-code fractions are 0.148 coarse and 0.182 fine, both below the frozen 0.25 gate. No retained result supports keeping the current quantizer as a beneficial mechanism.

For world v3, mean IID position RMSE is 0.0904 for graph, 0.0920 for global pool, and 0.0983 for multiscale. Graph has the lowest mean error in every retained OOD regime. Multiscale's mean OOD relative improvement versus graph is -13.05% and it loses all five paired seeds. Versus global pool, its mean OOD advantage is 1.58%, but paired direction is mixed and IID error is 7.49% worse. Learned assignments remain active at 2.77 effective groups of four, ruling out simple numerical inactivity while not establishing utility. Exact unrounded values and provenance are in the retained signed v3 capsule cited by `EVIDENCE_LEDGER.md`.

## Mechanism and failure analysis

The two failures are not identical. The legacy hierarchy exhibits a direct discrete-bottleneck problem: near-collapsed codebooks accompany worse reconstruction, and bypassing the quantizer improves the frozen local result. The world-model hierarchy does not simply collapse. Its learned grouping stays active, but activity fails to translate into predictive value. A plausible bounded explanation is that the flat graph already encodes the dominant local interaction structure of the soft-disc system; unconstrained coarse grouping therefore adds approximation and optimization burden without enough new information.

These observations support the narrower principle **activity is not utility**. They do not prove that all hierarchical representations are harmful, that a graph is universally optimal, or that a physical conservation prior would fail.

## Successor boundary

The current hierarchy should not be rescue-tuned against these outcomes. The next highest-value experiment is external negative confirmation: test whether a local graph equals or outperforms the current learned-grouping variant on at least two versioned public physical-dynamics datasets with validation-only tuning, prospective seeds, and matched wall-clock/FLOP budgets.

A materially new successor hypothesis does exist: a conservation-aware residual could project predicted pairwise impulses onto a momentum-conserving subspace in regimes without walls or drag. That mechanism is different from learned grouping. It must be separately preregistered before any test-outcome access and compared against the same graph backbone plus an analytic-force reference. Improvement in a conservation diagnostic without improvement in long-horizon prediction would falsify the useful-mechanism claim. No such successor result exists today.

## Limitations

All outcome-bearing studies remain synthetic/local. The world benchmark uses known states, one dynamics law, limited object counts, and five or fewer paired seeds. Parameters and update counts are approximately matched, but FLOPs are not. No real-world data, public external dynamics benchmark, learned perception, scaling study, or independent reproduction is available. Thus the evidence rejects only the current tested implementations at the tested scales.

## Related work and conclusion

Object-centric interaction networks, differentiable graph pooling, multiscale graph simulators, VQ, and FSQ all predate this repository; see `research/NOVELTY_AUDIT.md`. The current evidence does not support advancing RIPII's quantized or multiscale hierarchy. The simpler graph is the appropriate reference until a separately frozen external or genuinely new successor hypothesis provides contrary evidence. A submission-grade negative result still requires multiple public datasets, adequately powered paired seeds, compute matching, and independent review.

## Reproducibility statement

Frozen pilot v1/v2 artifacts preserve their original hashes and conclusions. The world path records source hashes, complete experiment configuration, seeds, validation selection, per-seed raw metrics, timing, checkpoints, success state, and artifact hashes. The strongest v3 result has a retained signed compact capsule, while full local run directories remain ignored mutable storage. Canonical commands and verification entry points are documented in the README and `scripts/verify_artifact.sh`. `research/NEGATIVE_RESULT_SYNTHESIS.md` adds only evidence synthesis and creates no new empirical result. No license has been selected, so reuse rights are not granted by this manuscript.
