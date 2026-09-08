# Contributing to RIPII

RIPII is a falsification-stage research prototype. Contributions must preserve failed
and negative results and must not promote development diagnostics into evidence.

## Development checks

Create the locked environment and run the local CI-equivalent checks:

```bash
uv sync --locked --extra dev
uv run ruff check ripii scripts tests
uv run python -m compileall -q ripii scripts tests
uv run pytest -q
uv build
```

Changes to retained experiments must also verify both immutable pilots:

```bash
uv run python scripts/verify_artifact.py --manifest research/results/pilot_v1/manifest.json --protocol research/protocols/pilot_v1.md
uv run python scripts/verify_artifact.py --manifest research/results/pilot_v2/manifest.json --protocol research/protocols/pilot_v2.md
```

Source drift reported by the verifier is expected after prospective code changes;
artifact hash or size failures are not. Never rewrite a frozen protocol, manifest, or
result to make a new implementation appear compatible with an old run. New evidence
needs a new prospectively fixed protocol, study identifier, output directory, and
manifest.

Do not include credentials, private data, machine-local paths, or untrusted checkpoint
files. An owner-selected license is still missing, so acceptance and reuse terms must
be resolved before external contributions are solicited.
