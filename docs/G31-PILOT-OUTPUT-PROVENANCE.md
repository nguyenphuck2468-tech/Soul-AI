# G3.1 PILOT output provenance audit

Status: audit record; no protocol lock, no Confirm access.

## Checkpoint identity found in the committed outputs

The artifact filenames alone are not reliable provenance. The embedded checkpoint hashes were checked directly:

| Artifact set | Embedded checkpoint SHA-256 | Identity |
|---|---|---|
| `pilot_metrics.json` / `pilot_predictions.jsonl` / `pilot_report.json` | `0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147` | A |
| `pilot_b1_metrics.json` / `pilot_b1_predictions.jsonl` | `0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147` | A — filename is misleading |
| `staging_b1_metrics.json` / `staging_b1_predictions.jsonl` / `staging_b1_report.json` | `136017b650fc910043c559822d47fea900391e2823324413327de815a4d8c018` | B1 |

Therefore, the committed `pilot_b1_*` metric/prediction files must not be cited as B1 evidence. The staging B1 set is the only inspected pre-fix B1 output set with a matching B1 checkpoint hash.

## Historical pre-fix PILOT numbers

These numbers are preserved here only to explain the provenance audit. They came from the old evaluator that scored all 100 recorded transitions before the training-aligned recurrent warm-start correction.

| Old evaluator result | A | B1 | Persistence baseline |
|---|---:|---:|---:|
| theta RMSE (rad) | 1.370717 | 1.403557 | 0.201528 |
| velocity RMSE (rad/s) | 4.036020 | 4.052245 | 0.592749 |

They are exploratory, not confirmatory. Do not interpret small A/B1 differences as meaningful: the old evaluation initialization was misaligned with the training sequence and the sample has only five PILOT episodes.

## Corrected evaluation path

The Colab branch `colab/g31-pilot-evaluation` now:

1. Verifies both A and B1 checkpoint hashes and TRAIN-only provenance.
2. Runs the same updated scorer against the same PILOT data for each checkpoint.
3. Primes each episode using the training sequence's zero-state/zero-action warm-up.
4. Excludes the first unscored warm-up transition, leaving 95 aligned transitions (19 per episode).
5. Compares both models against the same persistence and TRAIN-only mean-delta baselines.

The new results are written to timestamped Colab output folders and are not allowed to overwrite tracked artifacts.

## Research boundary

- TRAIN is the only fitting split.
- PILOT is exploratory evaluation only.
- CONFIRM remains untouched.
- Threshold remains `TO_LOCK`; gate remains `INCONCLUSIVE`.
- No performance claim is made until the corrected outputs are reviewed and the owner-defined protocol decisions are resolved.


## TRAIN dataset identity and output commit safety

The scorer now reads the PILOT and TRAIN JSONL inputs as bounded regular-file snapshots. The PILOT digest is checked against its frozen SHA-256 before scoring. The TRAIN digest is computed from the exact bytes parsed by the loader and must match the `dataset_sha256`, TRAIN split, environment, seed list and transition count embedded in the pinned A/B1 checkpoint. The same TRAIN snapshot feeds the mean-delta baseline; it is not reopened after hashing.

The TRAIN-only training entry points also compute checkpoint provenance from the exact snapshot used to construct episodes. Prediction rows, metrics and the report carry the TRAIN SHA-256 so a later data-file change cannot silently alter the baseline's identity.

Output files are staged before replacement. Existing outputs are backed up and restored if an ordinary replacement operation fails partway through the set. This is rollback protection, not a crash-atomic multi-file transaction; backups are retained and reported if rollback itself fails.

No model was retrained, no checkpoint bytes were modified, and no new benchmark or Confirm result is claimed by this audit change.
