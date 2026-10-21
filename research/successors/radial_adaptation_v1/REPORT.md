# Radial adaptation v1: qualified, blocked by compute cost

The requested properly trained comparison is **not complete**. No new evidence establishes whether the adaptive gate helps. The new study ran its engineering qualification and cost profile on free Kaggle, then stopped before tuning because the frozen comparison cannot fit the authorized **4 allocated GPU-hours and $0**.

## What was completed

- Preserved original radial-mixture v3 results and its `INCONCLUSIVE_UNDERTRAINED` decision. Seven authoritative historical artifacts and all 49 vendored files still match their recorded hashes.
- Created a separate prospective study using the exact original v3 architecture: adaptive radial, separately trained fixed radial, graph, simple pooling, and Transformer. These are the matched predecessor baselines; this is not a new literature-wide SOTA comparison.
- Implemented 20 calibration fits and 120 main fits across springs and charged particles, six independent generator blocks, and two nested model seeds. Selection uses calibration validation only.
- Raised the training allowance to 6,000–16,000 updates with a rollout curriculum, learning-rate reduction, training/validation plateau checks, and a training-quality check. Test generation is prohibited until all selected fits pass adequacy. More updates alone do not certify sufficient training.
- Added both the separately trained fixed-radial comparison and a same-checkpoint training-mean gate intervention. This separates a radial architecture gain from evidence for input-dependent adaptation.
- Passed 14 local control tests, 10 original hosted CPU numerical tests, and full-horizon CUDA backward/checkpoint replay checks for all five models.
- Downloaded and verified the hosted archive, its exact local source/protocol identity, and all 19 manifested attempt artifacts. No comparison fits or evaluations exist. The hosted verifier's `numeric_replay: true` flag was requested with zero evaluation rows, so it supplies no test-metric validation.

## Measured execution cost

| Quantity | Result |
|---|---:|
| Authorized allocation | 4 GPU-hours, $0 |
| Kaggle accelerator | 2 T4 GPUs; worker uses GPU 0 and charges both |
| Run execution elapsed | 395.13 seconds |
| Conservative ledger charge, including 180-second export reserve | 0.3195 allocated GPU-hours |
| Remaining authorization | 3.6805 allocated GPU-hours |
| Minimum-update matrix estimate | 179.10 host-hours / 358.19 allocated GPU-hours |
| Maximum-update estimate, including safety factor | 721.99 host-hours / 1,443.99 allocated GPU-hours |
| Comparison fits completed | 0 / 140 |
| Test data opened | No |

The estimates extrapolate short full-horizon update timings across the frozen matrix. The minimum-update estimate scales training and validation costs to 6,000 updates; it does not credit the cheaper early curriculum and is not a mathematical lower bound. The maximum estimate includes data/evaluation costs and a 1.5 safety factor. Neither estimate guarantees convergence. Even these limitations cannot support admission under a two-hour host cap. No further run was launched.

Full-horizon update timings were 0.535 seconds for adaptive radial, 0.470 for fixed radial, 0.999 for graph, 0.959 for pooling, and 0.754 for Transformer. Timing used the maximum of five measured updates after warm-up. Data generation for the two calibration training sets took 278.41 seconds. The upstream charged simulator emitted warnings, retained in the log; every saved scene passed the worker's shape and finiteness checks.

## Scientific interpretation and next action

**Adaptive benefit remains unknown.** Qualification establishes executable models and checkpoint replay, not training adequacy or superiority. The old undertrained result remains the only existing comparison and is unchanged.

Within the current budget, the next useful work is execution optimization in a separately versioned implementation, followed by numerical/gradient equivalence checks and another cost profile. The measured gap is roughly 90 times the original host allowance for the minimum-update estimate, so a small speedup or merely using the second GPU would not make this full study feasible. A smaller-domain/model/seed study would answer a narrower question and require a new declared protocol; it must not be presented as this completed comparison. Do not rerun the unchanged payload or shorten training merely to produce scores. Carry the 0.3195 GPU-hour charge into any further authorized attempt.

## Evidence

- [Study state](STATE.json), [frozen protocol](PROTOCOL.md), [source hashes](FROZEN.json)
- [Intake and source-binding receipt](hosted/kaggle/v1/intake-receipt.json)
- [Cost profile](hosted/kaggle/v1/extracted/execution/attempt-1/profile.json), [charge](hosted/kaggle/v1/extracted/execution/attempt-1/charge.json), [failure](hosted/kaggle/v1/extracted/execution/attempt-1/failure.json)
- [CPU qualification](hosted/kaggle/v1/extracted/execution/attempt-1/qualification-cpu.json), [CUDA qualification](hosted/kaggle/v1/extracted/execution/attempt-1/qualification-cuda.json)
- [Original archive](hosted/kaggle/v1/results.zip), [retained API log](hosted/kaggle/v1/api-log.txt)
- [Private Kaggle version 1](https://www.kaggle.com/code/ryangomez010/ripii-radial-adaptation-v1-4138f62b0ba1?scriptVersionId=360654457)

Archive SHA-256: `9f7bdfcd5ae30fe7a9d4bbf8c5edb6a43b72d031f4545eba0b5d8bf4929cbc40`.
Source freeze SHA-256: `2904cd7cb3487219a6d0acb2586c4d3f0d81e5d6f7ccddebc3fab7fa576ee80e`.

## Later execution investigation (21 October 2026)

The separately retained [radial execution study](../../engineering/radial_execution_v1/REPORT.md) passed corrected five-model equivalence fixtures and reduced measured captured-update timings. The update-only minimum matrix projection remains 13.10 host-hours, before data and validation. The full scientific matrix is still not admitted. Across the original run and two engineering checks, the cumulative conservative charge is 0.5748 allocated GPU-hours, leaving 3.4252 of the original four. The original run-only budget table above is historical, not the current remaining authorization.

## Rebuilding from the GitHub review branch

Run `python scripts/rebuild_radial_capsule.py --output /tmp/radial-reproduced.ipynb`. This independently reconstructs the frozen 59-file payload and requires byte identity with the retained hosted notebook. It does not re-audit the 616 MB historical raw archive, which remains locally preserved and hash-bound in PREDECESSOR.json. The original `build.py` additionally requires that historical archive and its predecessor files; use the published reproducer for this successor-only checkout. Hosted scientific execution is still prohibited unless a fresh admissible cumulative budget is established; reproducing a package is not permission to reset the four-hour cap.
