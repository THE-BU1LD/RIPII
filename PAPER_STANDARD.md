# RIPII paper-readiness standard

This file operationalizes the research-paper standard supplied on 2026-09-13. It is a
gate, not a formatting checklist. A polished manuscript cannot compensate for missing
scientific evidence.

## Research states

| State | Meaning | Minimum evidence |
|---|---|---|
| P0 | Research direction | Important problem, motivation, initial literature search |
| P1 | Operationalized problem | Formal statement, notation, assumptions, hypotheses, success/failure criteria |
| P2 | Evaluation locked | Frozen data, splits, metrics, baselines, ablations, seeds, tuning and statistics |
| P3 | Experiments complete | Every required cell executed; failures and negative outcomes retained |
| P4 | Evidence verified | Benchmarks, ablations, robustness, failure analysis, statistics and mathematics audited |
| P5 | Externally reproducible | A competent independent researcher reproduces the central result from a release |
| P6 | Submission ready | Scientific work, disclosures, citations, visuals and final manuscript pass every gate |

Only P6 may be described as submission-ready. States are cumulative: a project cannot
skip an unmet earlier gate.

## Mandatory scientific chain

```text
important problem -> precise formulation -> justified method -> correct mathematics
-> controlled experiments -> strong benchmarks -> ablations and robustness
-> interpretable evidence -> honest discussion -> reproducible conclusion
```

## Required pre-result record

Every primary experiment must record, before outcome inspection:

- experiment identifier and scientific question;
- falsifiable hypothesis and explicit failure condition;
- independent, dependent and controlled variables;
- dataset identity, provenance, version, license and immutable split policy;
- metrics and exact aggregation;
- baselines, tuning budget, compute budget and fairness policy;
- seeds, sample size, power rationale and statistical test;
- expected observation and advancement rule.

Test data may not select models, preprocessing, hyperparameters, hypotheses or stopping
rules. Failed or non-finite runs may not be silently excluded.

## Required manuscript content

The manuscript must make the actual unresolved problem, technical obstacle, mechanism,
evidence and boundary unmistakable. It must define variables, spaces, dimensions,
distributions, assumptions, constraints, objectives, optimization limitations and
algorithmic complexity where applicable. Each equation must add scientific content and
each theorem must include assumptions, proof, interpretation and empirical relevance.

The method must permit reconstruction of input, preprocessing, representation, model,
objective, optimization, inference and output. Diagrams are used only when they clarify
those relationships and must match code.

Results are organized by scientific question. They report uncertainty and practical
magnitude, include simple and strong baselines, isolate proposed mechanisms, test
sensitivity and distribution shift where claimed, and expose failure regimes. Discussion
must consider alternative explanations and use language proportional to evidence.

## Traceability requirement

Every central number must trace through:

```text
claim -> table/figure -> analysis code -> raw output -> run -> configuration
-> dataset version -> code revision
```

Generated paper tables must fail closed when their immutable evidence is absent or
changed. Hand-entered or selectively copied primary results are prohibited.

## Required audits

- mathematical and dimensional audit, including limiting/special cases;
- implementation-to-method agreement;
- data provenance, license, duplication and leakage audit;
- baseline fairness and compute-matching audit;
- ablation, sensitivity, robustness and failure analysis;
- statistical uncertainty and multiplicity audit;
- citation and closest-work audit;
- clean-machine end-to-end reproduction;
- figure, table and final-PDF visual audit;
- executable placeholder/stub/dead-code classification;
- license, authorship, contribution, funding and conflict approval.

## Readiness decision

Score bands are 95–100 exceptional, 90–94 strong conference-ready, 85–89
submission-ready with minor weaknesses, 80–84 borderline, 70–79 major scientific
revision, 60–69 incomplete research, and below 60 not paper-ready. A score never
overrides a blocking gate.

P6 additionally requires score >=85, zero critical failures, zero blocking scientific
issues, complete primary experiments/ablations/baselines, and passed mathematics,
reproducibility, citation, visual and PDF audits.

Automatic submission failure includes fabricated evidence or citations, train/test
leakage, irreproducible central results, unfair principal comparisons, unsupported
headline claims, central mathematical invalidity, unjustifiably omitted key baselines,
hidden manual result intervention, conclusion-relevant placeholders, or conclusions that
materially exceed the evidence.

The current project decision is maintained in `FINAL_RESEARCH_AUDIT.md`.
