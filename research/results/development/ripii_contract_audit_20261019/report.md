# RIPII-MR deterministic mechanics audit

No training, checkpoint selection, scientific advancement decision, or historical matrix replay was performed. Values below are lower bounds on per-coordinate RMSE over eight steps, conditional on the conservative P/Q constraints verified on every retained initialized rollout. The single-object obstructions are independent of learned weights because there are no internal pair impulses. A zero bound does not establish accurate trajectories.

| Fixture | Position RMSE lower bound | Velocity RMSE lower bound | Closed-form reference max error |
|---|---:|---:|---:|
| free_motion | 4.82608549e-17 | 0 | 8.327e-17 |
| drag_light | 0.00103060219 | 0.00604192309 | 5.551e-16 |
| drag_heavy | 0.000517790509 | 0.0030406934 | 4.718e-16 |
| wall_contact | 0.0674185568 | 0.298216254 | 2.220e-16 |
| force_four_substeps | 0.00143466196 | 1.59163761e-17 | 5.551e-17 |
| force_one_substep | 6.9267136e-19 | 8.93519206e-18 | 1.388e-17 |
| internal_contact | 0 | 0 | not applicable |
| long_range_coupling | 0 | 0 | not applicable |
| five_objects_drag | 0.000348087243 | 0.0020420524 | 2.776e-16 |

The wall-contact analytic reference covers the first observation only. Free-motion, drag, and constant-force closed forms cover all eight observations. Contact and coupling controls are checked through their conserved sums. The multi-object bounds are conditional on actual conservation: mass floors and degenerate or tied coarse assignments prevent a universal all-scene guarantee.

Both variants impose P'=P+dt*sum(actions) and Q'=Q+dt*P', where Q is mass-weighted position. Simulator wall and drag forces are additional external impulses. Four simulator substeps also differ from MR's single semi-implicit position update under constant force. Internal contact and symmetric coupling are negative controls because their net forces cancel.

## Actual coarse execution and router task gradient

| Variant | Route on | Coarsener calls | Scenes processed | Router gradient L1 |
|---|---|---:|---:|---:|
| ripii_mr | False | 1 | 1 | 0.00019191493 |
| ripii_mr | True | 1 | 1 | 0.000172593803 |
| ripii_mr_v2 | False | 0 | 0 | 0 |
| ripii_mr_v2 | True | 1 | 1 | 0.000172593803 |

These counters measure execution, not an accuracy/latency benefit. The v2 skipped path has no task gradient to the router for that scene; the logits in this probe are ±1, not saturated. The dynamics equations and checkpoint schema were not changed. The diagnostic uses the disclosed untrained parameter interventions. The historical 75-run result remains source-reported because its full artifacts are absent from the checkout.

All fixture inputs, simulator targets, initialized model predictions, physics settings, weight files, source snapshots and integrity hashes are retained alongside this report. No claim apportions the historical performance gap to this obstruction.
