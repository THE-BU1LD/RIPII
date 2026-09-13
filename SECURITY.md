# Security policy

RIPII is research software and has not received an independent security audit.

## Supported versions

Only the latest tagged release is supported. The unreleased 0.2 development branch is
alpha software and may introduce incompatible checkpoint or dataset schemas.

## Reporting a vulnerability

Use GitHub's private security-advisory reporting flow for this repository. Do not open a
public issue containing exploit details, credentials, private data, or unsafe artifacts.
Include the affected version/commit, operating system, minimal reproduction, impact, and
whether the report may be credited publicly. No response-time SLA is currently offered.

Only load checkpoints produced by a trusted RIPII run. Checkpoints contain model,
optimizer, configuration, and metric state; they are not a safe interchange format
for untrusted uploads. The loader rejects symlinks and requests PyTorch's restricted
`weights_only` loader, but that is not a substitute for provenance verification.

Do not place credentials, private datasets, or personal data in configuration or run
directories. Public releases must be generated from a clean tree and scanned for
secrets and machine-local paths.

CI performs static CodeQL analysis and an installed-environment dependency audit. These
automated checks are defense in depth and do not replace manual review or provenance
verification. Repository administrators must separately enable GitHub secret scanning
and push protection where the hosting plan supports them.
