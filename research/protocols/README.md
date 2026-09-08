# Protocol registry

`pilot_v1.md` and `pilot_v2.md` are frozen, executed local development protocols and
must not be edited to match later code. Their digest JSON files bind retained results.

`confirmatory_external_v1.md` is an unexecuted protocol template. It is deliberately
marked draft because dataset choice, licensing, compute budget, and an external
timestamp are unresolved. Editing it before execution is allowed only with a new digest
and changelog; once outcomes are inspected it must be frozen or superseded, never
silently rewritten.

`coupling_study_v1_RECORD.md` is a post-execution index for the completed v4
development intervention. The actual machine-readable protocol was written before
training and is retained verbatim in its self-checksummed capsule. The index must not be called
an external preregistration.

`../planning/external_power_plan_v1.json` is a self-checksummed provisional planning artifact,
not a frozen external protocol. It uses only retained synthetic development variance
and a fixed 5% effect threshold; external pilot variance may justify increasing its
18-pair recommendation before test outcomes are examined.
