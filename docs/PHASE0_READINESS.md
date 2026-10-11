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

The v1.0 execution plan ends at B6. The v2.6 report extends the concept into
Phases 0–8 and says three design areas must be settled before a prototype:
the SafetyController contract, structured observations, and the adaptive
compute benchmark. The report is a design proposal and does not provide the
scientific margin values or a task-by-task Phase 0 acceptance suite.

## Existing implementation

| Requirement | Current evidence | Remaining work |
| --- | --- | --- |
| Structured, copyable simulator | `minicraft/env.py`: structured observation, full `get_state`/`set_state`/`clone`, deterministic RNG | Map the observation fields explicitly to the v2.6 prototype contract; movement velocity is not represented because this world has no momentum |
| Feasibility expert | `minicraft/experts.py`; B6 passed on 200 train seeds per task | None for the simulator feasibility gate |
| SafetyController | Not implemented | Define permitted actions and hard constraints for MiniCraft; document expiry, mode changes, and logs before implementation |
| Soul adapter | `wm_api/soul_adapter.py` supports snapshots and prior rollout | `uncertainty` is zero-filled and `surprise` raises `NotImplementedError`; MiniCraft requires an explicit 9-action model config |
| Adaptive-compute protocol | v2.6 report defines metrics, baselines and compute accounting | Record how each proposal maps to MiniCraft; numerical `δ`, `h_OOD`, `R_max`, ρ/σ sweep and statistical procedure are intentionally unset in the report |
| Data separation | `configs/seeds.yaml` has train 0–999 and dev 1000–1999; no Confirm range | Keep Confirm absent and unused; document which train/dev seeds are used for collection, fitting, and model selection |

## Proposed next sequence

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
5. After the protocol draft is reviewed and the metrics are operational,
   implement the Phase 0 data collection, training and dev-only evaluation as
   separate, verifiable tasks.

No test or Confirm seed has been used for adaptive-compute claims. B6's train
seed runs only establish that the hand-written expert can solve the simulator
tasks; they do not measure WM quality or the core hypothesis.
