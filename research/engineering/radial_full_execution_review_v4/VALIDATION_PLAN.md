# Guarded validation and rebuild

Status: source corrections applied; all v3 tests and builds are NOT_RUN.
The predecessor's 24-test receipt does not qualify v3. Four new regression methods
cover the three-session allocation, old-duration rejection, cumulative failed
attempt charges, and the actual environment seen by a fresh Python child.

Before validation, use the existing admission implementation at
ripii_mr_next/control/resources.py. Require two samples at least three seconds
apart, >=512 MiB free-plus-speculative memory, stable pageout/swap counters,
>=10 GiB workspace and scratch disk, process visibility, no conflicting runner,
and no conflicting lock. A failed sample blocks testing and rebuilding. Preserve
the receipt and wait for the scheduled follow-up; do not busy-poll or relax gates.

Once admitted, allow at most 180 seconds for this Torch-free local validation batch:

1. Run `rtk proxy ruff format research/engineering/radial_full_execution_v3` and
   `rtk proxy ruff check research/engineering/radial_full_execution_v3`.
2. Run `rtk proxy python3 -m unittest discover -s research/engineering/radial_full_execution_v3 -p 'test_*.py' -v`.
   Do not run qualify.py, probe.py, cost_profile.py or other numerical entrypoints.
3. Run `rtk proxy python3 research/engineering/radial_full_execution_v3/build.py`.
4. Run `rtk proxy python3 research/engineering/radial_full_execution_v3/validate_package.py kaggle/intensive_20261025_r1 --output audit/radial-full-execution-v3/package-validation.json`.
5. Extract only the new source ZIP into a task-owned temporary directory and run
   the same Torch-free tests there. Build a second time to check deterministic
   notebook/source bytes. Keep logs and exact hashes; retain any failure.
6. Verify every predecessor hash in audit/radial-full-execution-v3/predecessor-snapshot.json,
   and verify the frozen protocol and matrix are unchanged. Reconcile new source
   hashes, tests and status without modifying old package bytes or old receipts.
7. Make a distinct r1 handoff archive under output/intensive_20261025_r1 with the
   corrected source, new notebook, protocol, manifests, useful test logs and these
   instructions. Include no internal conversations, memory/session exports or
   unrelated parent state. Verify member hashes and record size/SHA-256.

Stop on a resource failure, source mismatch or numerical requirement. Formatting
repairs are local and allowed; do not widen the scientific scope to force a pass.
No commit, push, upload, hosted launch, paid work, paper build or protected-data
access belongs to this validation batch. Preserve all old scientific evidence.

After a successful static rebuild, CUDA qualification and actual recovery remain
unverified. Publish the exact new archive path/hash and that boundary in this chat,
then stop the guarded follow-up. A static pass is not scientific completion.
