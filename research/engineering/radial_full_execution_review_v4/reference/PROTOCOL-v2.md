# Radial adaptation v2: zero-dollar budget successor

This successor preserves the terminal `QUALIFIED_COST_BLOCKED` v1 study and its
zero scientific fits. It changes only the authorized execution envelope and the
already-qualified execution implementation. The models, datasets, blocks, seeds,
learning-rate trials, objective, curriculum, minimum and maximum updates,
validation schedule, adequacy rule, protected-test gate, estimands, inference,
and decision thresholds are byte-identical to v1.

The execution envelope is at most nine host hours on a private Kaggle T4 x2
session, charged conservatively as at most 18 allocated GPU-hours. Paid scaling,
internet access, and paid fallback remain prohibited. The executor uses four
isolated training workers, two per allocated GPU, and CUDA graphs. It must qualify
the optimized kernels before generating development data and retain every worker
log, curve, checkpoint, adequacy receipt, source hash, and failure.

The v1 timing gate is not reused as a v2 admission veto because it assumed one
sequential worker on one of two allocated GPUs. V2 instead has a hard cumulative
wall deadline, live resource guards, complete-matrix checks, and fail-closed
export. A timeout or partial matrix is retained as terminal execution evidence,
not interpreted as a scientific result and not retried by silently changing the
matrix.

Protected test data remain nonexistent until all 120 main fits complete and every
fit independently replays the frozen adequacy rule and checkpoint hash. If any
calibration or main fit is inadequate, the study ends
`INCONCLUSIVE_UNDERTRAINED` with the test closed. Only a complete admitted matrix
can reach the byte-identical v1 summary and decision procedure.

