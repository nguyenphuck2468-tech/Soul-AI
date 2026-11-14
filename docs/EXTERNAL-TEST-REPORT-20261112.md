# External SOUL AI package test report — 2026-11-12

**Evidence class:** owner-supplied test summary, transcribed for project tracking.  
**Artifact verification:** pending receipt of `Soul_AI_Complete_Package.zip`. The ZIP's exact contents, file hashes, model metadata and metric JSON have not yet been independently checked from the archive. Values below must therefore be read as reported results, not as a reproducible CI result for the current branch.

## 1. Reported package inventory

The report describes 18 important files across four artifact groups:

- `artifacts/g31/g31_full_trained.pt` — reported 3.38 MB Pendulum-v1 model weights.
- `artifacts/phase0/` — MiniCraft RSSM weights, training report and dev score report.
- `artifacts/phase1_skills_v2/` — 12-level distilled task trees for `T_wood`, `T_tools`, and `T_iron`, with a batch manifest.
- `artifacts/phase1_compressed/` — merged CPU skills and safe fallback layer for all three tasks, with an aggregate manifest.

The report also describes compressed `.npz` training data and SHA-256 provenance. Exact file names and digest values remain unverified until the ZIP is available.

## 2. Reported pre-training test suite

- Pytest: **104/104 passed**, 17.49 seconds.
- Determinism: same seed and action sequence reportedly produce the same predicted state with error below `1e-6`.
- Seed separation: MiniCraft training seeds `0..999`, Pendulum training seeds `1001..1005`, separate from dev/confirm.
- Safety filter: report says hazardous actions (including lava traversal and invalid crafting) are blocked.

This is the report's pre-training suite. It is distinct from the current repository CI run, which has its own test count and commit SHA.

## 3. Reported G3.1 Pendulum-v1 training

- 2,000 optimizer updates; training set described as 5 episodes / 100 transitions.
- Reconstruction loss: `0.8702` at update 50 → `0.0050` at update 2,000 (174× lower).
- Reward loss: `3.5635` → `0.0105` (339× lower).
- Policy loss: `12.1158` → `0.5645` (21× lower).
- Continue loss: `0.0042` → `6.38e-07`.

Loss convergence alone does not establish closed-loop control quality. A reconstruction loss of `0.0050` is not automatically equivalent to `99.5%` accuracy; that conversion requires a defined, calibrated metric and normalization. The model still requires held-out trajectory scoring, action-channel ablation, and comparison with simple baselines before any control claim.

## 4. Reported Phase 0 MiniCraft RSSM

- Collection: 50 training maps, seeds `0..49`, up to 200 ticks per map; 9 discrete actions.
- Training: 2,000 updates.
- Final losses: reward `0.0884`, policy `0.0301`, continue `1.34e-06`.
- Dev maps: seeds `1000..1009`; unseen feature overflow reported as zero.
- Self-state MAE: `self.x=0.0041`, `self.y=0.0017`, `self.food=0.0069`.
- Facing error: `0.111` versus inertia baseline `0.302`.
- Mob relative movement error: `mobs.dx=0.192`, `mobs.dy=0.114`, versus reported baselines `0.389` and `0.242`.

The exact normalization, evaluation aggregation and baseline definitions need to be checked against the package's JSON reports. Ratios between MAE values may be reported as error reductions, but should not be described as accuracy multipliers without a defined accuracy metric.

## 5. Reported Phase 1 skill results

| Task | Raw tree, 6 levels / 50 maps | Distilled tree, 12 levels / 200 maps | Merged CPU system, 30 new maps | Learned agreement |
| --- | ---: | ---: | ---: | ---: |
| `T_wood` | 55.0% (27 nodes) | 66.0% (129 nodes) | 100.0% success | 99.2% |
| `T_tools` | 0.0% (25 nodes) | 32.0% (343 nodes) | 100.0% success | 95.1% |
| `T_iron` | 0.0% (25 nodes) | 6.0% (487 nodes) | 93.3% success | 87.3% |

The merged system is reported to run on CPU in under 1 ms per step. The report's `success` column and `learned_agreement` column are separate. The latter is agreement with the learned/planner behavior, not a second success rate.

## 6. Engineering interpretation

The strongest reported result is the gap between the standalone distilled trees and the merged system: the safe mapped fallback appears to contribute substantially to end-task success, especially for `T_tools` and `T_iron`. That is a useful engineering result, but it also means the merged system must be evaluated against the fallback-only baseline to measure the learned tree's incremental value.

Before promoting the package:

1. Verify the complete ZIP inventory and every SHA-256 digest.
2. Re-run manifest validation and all tests against the extracted package.
3. Reproduce all 30-map held-out outcomes from the archived manifests/traces.
4. Report fallback-only success, learned-only success, merged success, and agreement separately on identical maps.
5. Run ablations and safety fault injection; keep any confirm set sealed.
6. Keep MiniCraft and Minecraft/Horizon observation/action schemas separate until a dedicated 3D adapter is validated.

## 7. Artifact intake tooling\n\nA safe ZIP inspector is now available at `scripts/inspect_complete_package.py`. After receiving the archive, run:\n\n```bash\npython -m scripts.inspect_complete_package Soul_AI_Complete_Package.zip --output package-inventory.json\n```\n\nIt hashes every file, checks archive paths and resource limits, validates JSON syntax and duplicate keys, and never deserializes `.pt` weights or `.npz` arrays. It deliberately does not claim to verify package-specific manifest references or reproduce metrics; those checks require the actual package schema and evidence.\n\n## 8. Current repository boundary

This report does not change G3.1's existing evidence gate, checkpoint provenance, training state, or `TO_LOCK` threshold. It does not authorize CONFIRM access or automatic control of Horizon. The matching integration contract is [SOUL AI ↔ Horizon integration](SOUL-HORIZON-INTEGRATION.md).
