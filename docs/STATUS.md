# Soul-AI project status

Updated: 2026-10-11

## Direction

The target remains the user's cross-checked long-term project direction and
v2.6 report: a world model that shifts from expensive planner to supervisor as
skills mature, while increasing intervention again under OOD/risk. The earlier
AI-generated execution plan and its B6 endpoint are not the project endpoint.

## Current phase

- Alpha completion is tracked in `docs/ALPHA_SCOPE.md`: Phase 0 smoke pipeline
  run-through is complete, while the Phase 1 engineering alpha is estimated at
  75% and remains in progress. Overall alpha progress against the long-term
  plan is roughly 40%; this is an engineering estimate, not a weighted
  acceptance score.

- Phase 0 protocol and implementation are still drafts; no formal Phase 0
  experiment or adaptive-compute claim has been completed.
- MiniCraft now provides structured observations, deterministic state
  snapshots/cloning, one-step state-matched counterfactual branches, and a
  minimal visible-lava safety filter.
- Collection, training, scoring, and local adapter cost-measurement scripts
  exist. Collection uses at most one episode per train seed; training audits
  safety logs and episode continuity. New checkpoints embed train-only split,
  dataset, seed, protocol, and source provenance, which the scorer checks
  before reading dev data. Older smoke checkpoints lack this metadata and must
  be regenerated before scoring. The scorer reports anchored and open-loop
  metrics, feature groups, and schema overflow paths.
- `python -m scripts.phase0_alpha_run` provides a small end-to-end run-through;
  it defaults to unique output under `work/` and labels the manifest smoke-only.
- `python -m scripts.phase0_validate_protocol` now fail-closes the Phase 0
  draft if train/dev ranges drift, Confirm becomes configured, or numeric
  thresholds are introduced before protocol review. It is configuration-only
  and does not access any seeds.
- Phase 1 now has a Candidate skill wrapper around the existing privileged
  MiniCraft expert. It reports its task goal, status, risks, and descriptive
  episode outcome, plus fixed-action replay comparison and a three-task batch
  sweep. Across train seeds 200–209, Candidate succeeded in 10/10 episodes
  per task; replay succeeded in 1/10 for T_wood and 0/10 for T_tools and
  T_iron. This remains exploratory train-seed evidence, not held-out
  generalization, skill validation, or a compute-savings result. See
  `docs/PHASE1_SKILL_ALPHA.md` and `results/phase1_skill_sweep_20261011/`.
- An observation-only mapped Candidate now builds its map from local patches
  and explores known frontiers without receiving the full environment state.
  It succeeded on train seeds 400–409 for all three tasks, matching the
  privileged expert on that smoke batch. A learned decision-tree distillation
  prototype was also run and failed to solve T_tools/T_iron on the sample; it
  is archived as a negative engineering result, not a released skill. See
  `results/phase1_mapped_skill_20261011/` and
  `results/phase1_distillation_20261011/`.
- A compressed alpha Candidate now combines that learned tree with the mapped
  observation-only planner through an agreement gate: learned actions control
  only on agreement, otherwise the mapped planner remains in control. This is
  runnable integration infrastructure, not a pure learned skill, promotion
  result, or compute-saving claim. Run
  `python -m scripts.phase1_compressed_skill`.
- The mapped Candidate now reports a one-step expected state signature;
  checked fields matched the next observation on all 539 recorded checks for
  seeds 400–409. Its cheap monitor reports deviations and invalidates cached
  plans; the expectation is hand-coded and uncalibrated.
- Candidate action-proposal wall time on those seeds was approximately
  0.124–0.136 ms/step versus 0.0007–0.0009 ms/step for fixed replay. Replay
  usually failed, and measurement excludes simulation, safety, and learned
  world-model costs; it is diagnostic only, not matched-performance compute
  evidence. Same-seed Candidate-only / replay-only / tie counts were 9/0/1
  for T_wood and 10/0/0 for T_tools and T_iron; descriptive, with no
  inferential test or promotion rule.
- `surprise()` is an uncalibrated raw masked prediction-error diagnostic.
  Predictive uncertainty remains a labeled zero placeholder. Neither is valid
  as a detector or intervention threshold yet.
- Cost measurements are synchronized local adapter wall time, including
  completed CUDA work; integrated pipeline cost and `rho(h)` await a
  skill-step implementation.

## Data and gates

- Train seeds: 0–999. Dev seeds: 1000–1999. Dev 1000–1003 were used for
  explicitly non-evidentiary smoke runs and are already seen. The latest
  alpha end-to-end run completed collection, fitting, provenance-gated scoring,
  and CPU cost measurement through `scripts/phase0_alpha_run.py`; the archived
  seed-1002 run details are in
  `results/phase0_alpha_smoke_20261011_manifest.json`.
- Confirm has no range and remains unused.
- Do not treat either smoke checkpoint as model-quality evidence. The older
  60-transition, 5-update checkpoint underperformed persistence; the latest
  236-transition, 2-update smoke also exceeded persistence error at H=1 and
  H=5 and had no complete H=100 window.
- Phase 5 values such as `delta`, `h_OOD`, headroom margins, `rho/sigma`
  sweeps, and `R_max` remain open because the source report supplies no numeric
  values.
- The remaining Phase 0 blocker is protocol review: seed subsets, feature
  weighting, statistical procedure, uncertainty method, and acceptance
  criteria are still unresolved. The validator intentionally preserves this
  fail-closed state rather than selecting values.

## Repository state

Recent implementation is committed and pushed to `origin/main`; check `git log`
for the current head. The pre-existing untracked `work/` directory is retained.
