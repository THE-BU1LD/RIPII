# Changelog

## 0.2.0 — Unreleased

- Add a supported `WorldPredictor` inference API and `ripii` CLI alias.
- Make projector count control a learned projector bank.
- Train raw-basis conditioning rather than a tautological post-QR constraint.
- Correct the projection objective to minimize unexplained residual energy.
- Exclude graph self-edges and separate structural sparsity diagnostics from a
  differentiable attention-concentration objective.
- Replace hard VQ usage as an objective with differentiable soft-assignment balance.
- Default new world-model training to the empirically stronger flat graph control.
- Integrate the NRI Springs/Charged negative development result into public-facing
  research documentation.

Historical checkpoints and experiment artifacts remain versioned evidence of their
original implementations; they are not silently upgraded to 0.2 semantics.
