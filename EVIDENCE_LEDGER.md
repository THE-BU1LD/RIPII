# Evidence ledger

| Claim | Classification | Evidence | Boundary |
|---|---|---|---|
| Core models execute and differentiate | engineering-verified | full pytest suite | correctness, not usefulness |
| Legacy pilot v1 did not advance | negative, frozen local | `research/results/pilot_v1` | loss-balancer defect invalidates causal ranking |
| Corrected full RIPII is worse than no-VQ on pilot v2 | negative, frozen local | v2 summary: +0.061549 MSE | 3 seeds, 30 steps, synthetic |
| Pilot v2 full structure is worse than structured-mechanism removal | negative, frozen local | v2 summary: +0.06169 reconstruction MSE and -0.08772 held-out probe accuracy | local synthetic pilot |
| Pilot v2 codebooks are near collapse | negative, frozen local | effective fractions 0.148/0.182 | batch/task-specific diagnostic |
| World v3 multiscale loses graph OOD on 5/5 seeds | negative development | signed v2 capsule and full local manifest | one simulator/data seed; not external confirmation |
| Graph has lower mean v3 IID RMSE | negative development | graph 0.0904 vs multiscale 0.0983 | descriptive rounding; exact values in capsule |
| Graph has the lowest mean error on every retained v3 OOD split | negative development | signed v2 capsule | same simulator law |
| Multiscale mean OOD relative improvement vs graph is -13.05% | negative development | signed v2 capsule/statistics | five paired seeds; no population claim |
| Multiscale assignments remain active | mechanistic development | effective groups 2.77/4 | activity does not imply utility |
| Current hierarchy should not be rescue-tuned post hoc | evidence-governance conclusion | `research/NEGATIVE_RESULT_SYNTHESIS.md`, frozen v2 + world-v3 decisions | successor must be a new preregistered hypothesis |
| Flat graph is the current reference model | bounded recommendation | world-v3 negative result + simpler control | not a novelty or universal-optimality claim |
| Conservation-aware residual is a genuinely new successor candidate | hypothesis only, not run | `research/NEGATIVE_RESULT_SYNTHESIS.md` / `research/HYPOTHESES.md` | requires separate preregistration before test access |
| Exact resume, manifests, fail-closed checks work | engineering-verified | tests and verification scripts | tested local CPU path |
| Novelty/superiority/external generalization | not established | `research/NOVELTY_AUDIT.md` | must not be claimed |
| External confirmatory protocol | not yet run | `research/protocols/confirmatory_external_v1.md` | EXTERNAL_EXECUTION_REQUIRED |

Raw values remain in retained JSON/CSV and signed capsules; prose rounding is never the authoritative source. Historical artifacts are not rewritten after code changes. `research/NEGATIVE_RESULT_SYNTHESIS.md` is a synthesis of already-retained evidence and creates no new empirical outcome.
