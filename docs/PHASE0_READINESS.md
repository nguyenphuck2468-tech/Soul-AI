# Phase 0 readiness — Soul-AI MiniCraft

Date: 2026-10-11  
State: review draft; no Phase 0 model experiment has been run.

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
| Structured, copyable simulator | `minicraft/env.py`: structured observation, full `get_state`/`set_state`/`clone`, deterministic RNG | Map the observation fields explicitly to the v2.6 prototype contract; movement velocity is not represented because this world has no momentum |
| Feasibility expert | `minicraft/experts.py`; B6 passed on 200 train seeds per task | None for the simulator feasibility gate |
| SafetyController | Not implemented | Define permitted actions and hard constraints for MiniCraft; document expiry, mode changes, and logs before implementation |
| Soul adapter | `wm_api/soul_adapter.py` supports snapshots and prior rollout; `SoulAdapter.for_minicraft()` now creates a compatible 9-action model | `uncertainty` is zero-filled and `surprise` raises `NotImplementedError`; neither is suitable as a scientific metric yet |
| Adaptive-compute protocol | v2.6 report defines metrics, baselines and compute accounting | Record how each proposal maps to MiniCraft; numerical `δ`, `h_OOD`, `R_max`, ρ/σ sweep and statistical procedure are intentionally unset in the report |
| Data separation | `configs/seeds.yaml` has train 0–999 and dev 1000–1999; no Confirm range | Keep Confirm absent and unused; document which train/dev seeds are used for collection, fitting, and model selection |

## Next sequence aligned to the original direction

1. Write a P0 protocol draft mapping v2.6's structured-state track, baselines,
   outcome metrics, intervention metrics and `C_wall`/`C_norm` to MiniCraft.
2. Specify the SafetyController's hard constraints and allowed action sets. Keep
   it a filter with no path search or new multi-step planner.
3. Specify a MiniCraft WM task contract: 9 discrete actions, observation
   schema, rollout targets, persistence baseline, and separate prediction and
   uncertainty metrics. The existing zero uncertainty placeholder is not
   evidence of calibrated uncertainty.
4. Keep model development and exploratory measurements to train/dev data. Do
   not add or access a held-out Confirm split. Lock the Phase 5 numeric margins
   and final analysis protocol before any Phase 5 evaluation.
5. Implement the Phase 0 data collection, training and dev-only evaluation,
   then report predictive accuracy, action conditioning, uncertainty behavior,
   and compute cost before drawing claims about adaptive compute.

The Soul policy/model path now permits a fully discrete action space
(`discrete_dim == action_dim`) and imagined discrete actions use a
straight-through categorical estimator. A targeted test exercises a training
update with MiniCraft's nine-action encoding. This removes one adapter/model
compatibility blocker; it does not establish world-model quality or calibrated
uncertainty.

No test or Confirm seed has been used for adaptive-compute claims. B6's train
seed runs only establish that the hand-written expert can solve the simulator
tasks; they do not measure WM quality or the core hypothesis.
