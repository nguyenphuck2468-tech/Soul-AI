# Soul-AI project status

Updated: 2026-10-11

## Direction

The target remains the user's cross-checked long-term project direction and
v2.6 report: a world model that shifts from expensive planner to supervisor as
skills mature, while increasing intervention again under OOD/risk. The earlier
AI-generated execution plan and its B6 endpoint are not the project endpoint.

## Current phase

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
- Phase 1 now has a Candidate skill wrapper around the existing privileged
  MiniCraft expert. It reports its task goal, status, risks, and descriptive
  episode outcome, plus a fixed-action replay comparison. Three single-seed
  train-seed samples favored replanning over replay (Candidate 1, replay 0),
  but are not held-out generalization evidence and do not validate skill
  maturity. See `docs/PHASE1_SKILL_ALPHA.md`.
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

## Repository state

Recent implementation is committed and pushed to `origin/main`; check `git log`
for the current head. The pre-existing untracked `work/` directory is retained.
