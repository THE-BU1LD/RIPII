# RIPII paper package

Current full manuscript:

- \`ripii_negative_results.tex\`
- \`references.bib\`
- \`PAPER_EVIDENCE_MAP.md\`

The manuscript is intentionally anonymous until the author list and contribution basis are finalized by the project owners.

## Build

From this directory:

\`\`\`bash
latexmk -pdf -interaction=nonstopmode -halt-on-error ripii_negative_results.tex
\`\`\`

Clean generated files with:

\`\`\`bash
latexmk -C ripii_negative_results.tex
\`\`\`

## Evidence rule

Do not hand-edit a result in the LaTeX source merely to make a table look better. Update a numerical claim only after checking the exact retained artifact listed in \`PAPER_EVIDENCE_MAP.md\`.

Rounded manuscript values are presentation surfaces. JSON/CSV/capsule artifacts remain the numerical source of record.

## Scientific boundary

This package presents RIPII as a bounded negative-result / falsification-first study.

It does not claim:

- state-of-the-art performance;
- general failure of hierarchy or vector quantization;
- real-world or confirmatory external validation;
- algorithmic novelty for the component mechanisms;
- population-level statistical significance from the small seed samples.

The draft external-confirmation protocol remains unexecuted.
