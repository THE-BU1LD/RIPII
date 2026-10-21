# Radial adaptation evidence manuscript

Seven-page anonymous evidence report using the byte-for-byte retained user arxiv style. This is a complete report of the work performed, not a positive-efficacy or submission-ready paper.

- `main.tex`: model derivation, exact protocol, observed results, limits and references.
- `observed.tex` and `optimization.tex`: artifact-derived numerical surfaces.
- `timing.pdf`: vector engineering timing figure; comparisons are descriptive and unpaired.
- `arxiv-source.zip`: minimal compilable source, not an arXiv submission.
- `RELEASE_RECEIPT.json`: hashes, seven-page visual review, and remaining gates.
- Final PDF: `../../output/pdf/ripii-radial-adaptation-evidence.pdf`.
- `hosted/attempt-3`: successful free-Kaggle CPU build, compiler logs, rendered pages, and source-bound archive.

Two earlier malformed notebook launches failed before the compiler ran; both logs are preserved. The corrected transport now rejects notebooks missing Python kernel metadata. Rebuild using the pinned compiler (Tectonic 0.17.0) or `tectonic main.tex` from this folder. `scripts/build_radial_paper_notebook.py` prepares the free CPU build for machines with insufficient local resources. No paid service is used.

The older `paper/ripii_negative_results.tex` and prior negative/undertrained research remain intact. Author names, affiliations and whole-release licensing require owner-provided values.
