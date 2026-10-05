# M2b: Content-Bound, One-Time Locked-Test Approval

## Purpose

M2b removed the reusable static `APPROVED` string and split final locked-test access into two permission stages: a human issues approval, then the runner consumes it once. Brainstorm, Critic, Runner, and allowlisted executors cannot issue approvals. An ExperimentSpec cannot request locked access.

```text
completed + release + KEEP
  → human runs approve-locked-test
  → freeze spec / checkpoint / predictions / dataset fingerprint
  → issue one-time bearer token
  → locked-test atomically consumes token
  → RECORDED or FAILED; neither state can be replayed
```

## Issuance requirements

`approve-locked-test` issues an approval only when all conditions hold:

- The experiment is registered as `completed` in the `release` stage.
- Deterministic Evaluation decided `KEEP`.
- A non-empty `dataset_fingerprint` is registered.
- At least one `best_checkpoint` artifact is registered, and its disk SHA-256 matches the registry.
- The locked prediction file exists and has a stable SHA-256.
- A human explicitly supplies `reason` and `approved_by`.

Issuance atomically changes the experiment to `frozen` and writes one approval and review. The random token is displayed only once. SQLite stores only its SHA-256, not the bearer token. An experiment cannot receive a second approval.

Approval binds the SHA-256 of the canonical Registry ExperimentSpec; path, SHA-256, size, and bundle SHA-256 for all seed checkpoints under the selected artifact name; absolute path and SHA-256 of locked predictions; and the training/research `dataset_fingerprint`.

## Two-step commands

The human reviewer issues approval after freezing the model, predictions, and rationale:

```bash
ticknet-research \
  --registry results/registry.sqlite \
  approve-locked-test \
  --id EXP-RELEASE-001 \
  --predictions locked/predictions.parquet \
  --checkpoint-artifact-name best_checkpoint \
  --reason "Final one-time confirmation before formal release" \
  --approved-by "risk-reviewer"
```

Save the returned `token`. The execution process may consume it but cannot change any bound input:

```bash
ticknet-research \
  --registry results/registry.sqlite \
  locked-test \
  --id EXP-RELEASE-001 \
  --predictions locked/predictions.parquet \
  --token "<one-time-token>"
```

## Consumption and failure behavior

Before execution, the runner recalculates the spec, checkpoint bundle, prediction, and dataset fingerprints. Any mismatch is rejected before token consumption, allowing investigation. Once all bindings match, a conditional single-row update where `status='issued'` atomically consumes approval; only then are locked results read.

On successful audit, the experiment becomes `locked_tested`; the result review records approval ID, consumption time, all binding summaries, and Audit. If the audit itself fails, the token remains consumed and the experiment becomes `locked_test_failed`, with the failure reason recorded. This prevents repeated probing based on failure details. Reuse of a consumed token is rejected.

## Permission boundary

Content binding and one-time consumption protect against accidental reuse, content substitution, and replay; they do not authenticate a human. `approved_by` is an audit label, not a digital signature. In production, isolate `approve-locked-test` in an OS account unavailable to agents, a CI environment approval, or a separate service. Agents should receive only the one-time token after issuance. Apply least-privilege file permissions to both Registry and locked artifacts.

## Acceptance coverage

Synthetic tests verify that static `APPROVED` without an issued record is invalid; the raw token is absent from SQLite; changed predictions or checkpoints invalidate approval; non-release, non-KEEP, missing-fingerprint, or missing-checkpoint experiments cannot receive approval; successful consumption cannot be replayed; audit failure still consumes and records failure; and issuance and consumption work end-to-end through the CLI.
