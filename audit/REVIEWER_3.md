# Reviewer 3 — reproducibility and systems

## Final hostile review

**Summary.** The repository has exact CPU resume, validation-only selection, immutable
protocol hashes, status transitions, manifests, signed compact artifacts, source/data
hashes, fail-closed CLIs, canonical scripts, package builds, and broad tests. New
objective and efficiency artifacts have independent semantic verifiers.

**Fatal concern.** No owner-approved license, authorship, affiliations, or conflicts
statement exists. **Major concerns.** Compact capsules omit checkpoints; durable
content-addressed archival is absent; external data and anonymous clean-machine
reproduction are untested. CPU timings are visibly load-sensitive and profiler FLOPs
are incomplete, so no efficiency claim is justified. **Minor concerns.** Shell entry
points default to `.venv`, and ignored local runs are needed for checkpoint-level
re-evaluation. **Missing work.** Archive full runs, add owner metadata, and have an
independent researcher reproduce installation, one external experiment, analysis, and
paper generation.

**Novelty assessment.** Outside the systems contribution claimed by the project.
**Reproducibility assessment.** High locally, incomplete for release evidence.
**Likely score:** 4/10 (reject). **Confidence:** 5/5.
