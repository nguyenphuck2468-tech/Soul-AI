# Phase 0 readiness — Soul-AI MiniCraft

Date: 2026-10-11  
State: protocol/implementation draft; no formal Phase 0 experiment has been run.

## Original objective

The original plan's core idea is that a capable agent starts with expensive
world-model planning, compresses repeated experience into fast skills, and
keeps the world model as a supervisor that intervenes when expected value
exceeds compute cost. The falsifiable claim is that mature skills reduce
world-model calls without losing normal performance, while OOD events increase
intervention and speed recovery at lower compute cost than always planning.

The user has confirmed that the v2.6 full report and the long-term project
direction document are the intended source of truth and were cross-checked
against one another. The earlier execution plan was created after AI errors;
its B6 endpoint is not a project stop condition. Continue toward the original
phased research objective. Keep open design values open where the source report
does not specify them; do not invent margins or claim evidence that has not
been collected.

## Existing implementation

| Requirement | Current evidence | Remaining work |
| --- | --- | --- |
| Structured, copyable simulator | `minicraft/env.py`: structured observation, full `get_state`/`set_state`/`clone`, deterministic RNG | Observation contract documented in `docs/PHASE0_CONTRACT.md`; no velocity exists because there is no momentum |
| Feasibility expert | `minicraft/experts.py`; B6 passed on 200 train seeds per task | None for the simulator feasibility gate |
| SafetyController | Specification draft in `docs/PHASE0_CONTRACT.md`; no runtime controller | Review hard constraint, expiry behavior, severity mapping, hysteresis, and logs before implementation |
| Soul adapter | `wm_api/soul_adapter.py` supports snapshots and prior rollout; `SoulAdapter.for_minicraft()` now creates a compatible 9-action model | `uncertainty` is zero-filled and `surprise` raises `NotImplementedError`; neither is suitable as a scientific metric yet |
| Phase 0 collection/training/scoring | `scripts/phase0_collect.py`, `scripts/phase0_train.py`, `scripts/phase0_score.py`; smoke pipeline ran on a temporary 60-transition set | Scorer now records dirty working-tree status hash and rejects overlapping derived episode seeds. Do not call smoke output evidence; score is poor vs persistence and the checkpoint had only 5 updates. Define/lock fit size, rollout horizons, feature weighting, action distribution, and uncertainty method |
| Phase 0 compute | `scripts/phase0_measure_cost.py` measures the actual 9-action Soul adapter at H={1,5,20,100} | `ρ(h)` awaits an implemented skill-step cost; measurement is local wall time with provenance |
| Adaptive-compute protocol | v2.6 report defines metrics, baselines and compute accounting; P0 draft is `configs/phase0_protocol_draft.yaml` | Review seed list, mappings, statistical procedure; numerical `δ`, `h_OOD`, `R_max`, ρ/σ sweep intentionally unset in source |
| Data separation | `configs/seeds.yaml` has train 0–999 and dev 1000–1999; no Confirm range | Keep Confirm absent and unused; document which train/dev seeds are used for collection, fitting, and model selection |

## Next sequence aligned to the original direction

1. Review the source-to-protocol decision register in `docs/PHASE0_CONTRACT.md`
   alongside `configs/phase0_protocol_draft.yaml`; resolve still-open task
   details without inventing values from the report. The register now separates
   the Phase 0 prerequisite measurement from Phase 5's OOD, headroom, Pareto,
   and intervention thresholds.
2. Keep code development/smoke data on train and dev only. Dev seeds 1000–1001
   have already been touched by the explicitly non-evidentiary pipeline smoke.
3. Review the schema-frozen multi-horizon scorer; specify and calibrate a real
   uncertainty method before exposing uncertainty to any detector.
4. Run formal train fitting and dev evaluation once task distribution,
   horizons, weighting, and summary procedure are frozen. Confirm remains
   absent and unused.
5. Lock Phase 5 margins and final analysis protocol before any Phase 5 test.

The Soul policy/model path now permits a fully discrete action space
(`discrete_dim == action_dim`) and imagined discrete actions use a
straight-through categorical estimator. A targeted test exercises a training
update with MiniCraft's nine-action encoding. This removes one adapter/model
compatibility blocker; it does not establish world-model quality or calibrated
uncertainty.

No test or Confirm seed has been used for adaptive-compute claims. The Phase 0
smoke diagnostic did touch dev seeds 1000–1001 and found the 5-update model
underperformed persistence; it is not formal evidence. B6's train
seed runs only establish that the hand-written expert can solve the simulator
tasks; they do not measure WM quality or the core hypothesis.
