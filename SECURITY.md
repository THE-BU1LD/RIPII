# Security policy

RIPII is research software and has not received an independent security audit.

Only load checkpoints produced by a trusted RIPII run. Checkpoints contain model,
optimizer, configuration, and metric state; they are not a safe interchange format
for untrusted uploads. The loader rejects symlinks and requests PyTorch's restricted
`weights_only` loader, but that is not a substitute for provenance verification.

Do not place credentials, private datasets, or personal data in configuration or run
directories. Public releases must be generated from a clean tree and scanned for
secrets and machine-local paths.
