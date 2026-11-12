# G3.1 corrected PILOT evaluation — 2026-11-12

## Status

**Exploratory only. Gate: INCONCLUSIVE. Threshold: TO_LOCK. Confirm was not accessed.**

This record summarizes the user-supplied Colab export `SOUL_AI_G31_Colab_20261112T123718Z.zip`. It records the output as received; it does not replace the committed legacy metrics or claim a successful model.

## Provenance

| Field | Value |
|---|---|
| Environment | Pendulum-v1 / Gymnasium 1.4.0 |
| Episodes | 5 |
| Recorded transitions | 100 |
| Scored transitions | 95 |
| Warm-up transitions skipped | 5 (one per episode) |
| Evaluator | Training-aligned warm start + prior mean + physical decode + real-observation teacher forcing |
| PILOT SHA-256 | `26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e` |
| Checkpoint A SHA-256 | `0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147` |
| Checkpoint B1 SHA-256 | `136017b650fc910043c559822d47fea900391e2823324413327de815a4d8c018` |
| Confirm accessed | `false` |
| Optimizer updates | `0` |
| Model immutable | `true` |

The archive contains separate `A/` and `B1/` metric/report outputs. The hashes match the intended checkpoint identities. Both use the corrected 95-transition evaluator and the same PILOT dataset.

## Aggregate results

Equal-episode mean MAE; lower is better.

| Metric | A | B1 | Persistence | TRAIN-only mean-delta |
|---|---:|---:|---:|---:|
| Theta MAE (rad) | 1.177582 | 1.200337 | 0.187212 | 0.190341 |
| Velocity MAE (rad/s) | 3.743483 | 3.768105 | 0.542969 | 0.550427 |

Paired model-minus-persistence MAE:

| Metric | A | B1 |
|---|---:|---:|
| Theta (rad) | +0.990370 | +1.013124 |
| Velocity (rad/s) | +3.200516 | +3.225137 |

Both checkpoints remain substantially worse than both simple baselines on this five-episode PILOT batch. B1 is slightly worse than A on these aggregate metrics. The paired differences are descriptive; five episodes do not establish broad generalization.

## Interpretation

1. The evaluator warm-start correction fixed a training/evaluation sequence-alignment mismatch, and the new archive confirms that the corrected path was used.
2. The corrected metrics remain poor. Therefore the earlier mismatch was real but was not the sole explanation for the model's large errors.
3. Do not promote either checkpoint, lock thresholds, or claim a successful world model from these results.
4. The next engineering investigation should trace the representation-to-physical-state decode, latent prior/posterior semantics, dynamics target alignment, action conditioning, and per-episode error trajectories. Add small deterministic unit tests for each mapping before any retraining.
5. Preserve the current checkpoints and this export as immutable evidence. Any future model change must produce a new checkpoint hash and a new, separately identified run.

## Research boundaries

- PILOT only; no Confirm data or seeds were accessed.
- No optimizer updates, checkpoint mutation, threshold selection or protocol lock.
- These results are not held-out generalization evidence and do not resolve the Phase 0 protocol review.
