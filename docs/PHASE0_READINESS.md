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
| Structured, copyable simulator | `minicraft/env.py`: structured observation, full `get_state`/`set_state`/`clone`, deterministic RNG; `minicraft/counterfactual.py` branches candidate actions from one cloned state | Counterfactual helper is infrastructure only; no intervention/regret metric has been defined or evaluated. Observation contract documented in `docs/PHASE0_CONTRACT.md`; no velocity exists because there is no momentum |
| Feasibility expert | `minicraft/experts.py`; B6 passed on 200 train seeds per task | None for the simulator feasibility gate |
| Skill alpha | `minicraft/mapped_skill.py` provides an observation-only map-building Candidate; `scripts/phase1_mapped_skill.py` evaluates it against the privileged expert; `scripts/phase1_distill_skill.py` prototypes action distillation | Mapped skill reached 10/10 per task on one train smoke batch; held-out generalization remains untested. Decision-tree imitation failed on T_tools/T_iron. No skill is validated or compute-efficient by evidence; held-out distance and promotion margins remain open. See `docs/PHASE1_SKILL_ALPHA.md` |
| SafetyController | `minicraft/safety.py` implements the visible-adjacent-lava filter, noop fallback, and decision record; Phase 0 collector/scorer apply it and record candidate/applied actions | Mapping is only valid while action semantics are unchanged; `confirm_inverted_move` is not observable, so OOD lava protection is not guaranteed. Severity mapping, persistent modes, expiry and hysteresis remain open |
| Soul adapter | `wm_api/soul_adapter.py` supports snapshots, prior rollout, recurrent `observe`, and raw one-step surprise; `SoulAdapter.for_minicraft()` creates a compatible 9-action model | Uncertainty stays zero-filled for compatibility but is labeled `NOT_IMPLEMENTED_PLACEHOLDER`. Surprise is `UNCALIBRATED_RAW_MASKED_MSE`, diagnostic only; neither is yet suitable as a trigger or scientific claim. Rechecking the same observation for surprise no longer increments encoder overflow audit counts twice. |
| Phase 0 collection/training/scoring | `scripts/phase0_collect.py`, `scripts/phase0_train.py`, `scripts/phase0_score.py`; smoke pipeline ran on a temporary 60-transition set | Collector uses at most one terminal trajectory per train seed. Trainer audits schema-v2 safety decisions, one-episode-per-seed, tick continuity, terminal boundaries, and observation chains. Checkpoints embed train-only split, dataset, seed, protocol, and source provenance; scorer validates it before dev access. Train/score reports record protocol and working-tree provenance plus run settings. Scorer separates anchored/open-loop horizons, reports feature groups and unseen paths, excludes/counts empty-target origins, and rejects overlapping derived seeds. Smoke is not evidence; its older checkpoint predates the new embedded provenance and must be regenerated. Lock fit size, horizons, field weighting, action distribution, and uncertainty method |
| Phase 0 compute | `scripts/phase0_measure_cost.py` measures the 9-action Soul adapter at H={1,5,20,100} | CUDA work is synchronized around wall-clock timing. Cost records include dirty-tree and protocol hashes. This is local adapter wall time, not integrated pipeline cost; `ρ(h)` awaits an implemented skill-step denominator |
| Adaptive-compute protocol | v2.6 report defines metrics, baselines and compute accounting; P0 draft is `configs/phase0_protocol_draft.yaml` | Review seed list, mappings, statistical procedure; numerical `δ`, `h_OOD`, `R_max`, ρ/σ sweep intentionally unset in source |
| Data separation | `configs/seeds.yaml` has train 0–999 and dev 1000–1999; no Confirm range | Keep Confirm absent and unused; document which train/dev seeds are used for collection, fitting, and model selection |

## Next sequence aligned to the original direction

1. Review the source-to-protocol decision register in `docs/PHASE0_CONTRACT.md`
   alongside `configs/phase0_protocol_draft.yaml`; resolve still-open task
   details without inventing values from the report. The register now separates
   the Phase 0 prerequisite measurement from Phase 5's OOD, headroom, Pareto,
   and intervention thresholds.
   `python -m scripts.phase0_validate_protocol` checks this draft remains
   fail-closed while those decisions are open.
2. Use `python -m scripts.phase0_alpha_run` for a small end-to-end run-through;
   it writes to a unique directory under `work/` by default and marks its
   manifest non-evidentiary. Keep smoke data on train and dev only. Dev seeds
   1000–1003 have already been touched by explicitly non-evidentiary smoke runs.
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
underperformed persistence; it is not formal evidence. The alpha run on dev
seed 1002 exercised the provenance-gated scorer using a 2-update checkpoint and
a 20-tick episode. It remains a run-through: H=1 and H=5 model MSE exceeded
persistence, and H=100 had no complete window. See
`results/phase0_alpha_smoke_20261011_manifest.json`. B6's train
seed runs only establish that the hand-written expert can solve the simulator
tasks; they do not measure WM quality or the core hypothesis.
