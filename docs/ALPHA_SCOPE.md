# Alpha scope and completion

Updated: 2026-10-11

## What “alpha complete” means here

The source documents describe a long-term adaptive-compute research program,
not a finite alpha release checklist. This file defines the current software
alpha boundary so runnable engineering work can be tracked without treating
smoke outputs as scientific evidence.

### Phase 0 alpha run-through

**Status: 100% of the smoke run-through; not 100% of Phase 0.** The one-command
pipeline collects train-only transitions, fits a small RSSM, provenance-gates
dev scoring, and measures local adapter cost. At least one end-to-end smoke
run is archived. The Phase 0 protocol remains a draft; the run is
non-evidentiary and does not validate model quality or the project hypothesis.

### Phase 1 skill alpha checklist

| Item | Status | Evidence / remaining work |
| --- | --- | --- |
| Runnable Candidate skill contract | Done | `minicraft/skills.py`; full-state expert wrapper, goal expectation, status, risk and outcome card |
| Safety-filtered episode runner | Done | `scripts/phase1_skill_smoke.py`; per-step trace and manifest |
| Fixed action-replay baseline | Done | `scripts/phase1_replay_compare.py`; paired initial seeds |
| Multi-task train smoke | Done | seeds 200–209, three tasks; archived traces and per-task manifests |
| Paired outcomes and proposal-time diagnostics | Done | candidate-only/replay-only/ties and local proposal timings recorded; descriptive only |
| Reproducible batch command | Done | `python -m scripts.phase1_skill_sweep`; unique `work/` outputs by default |
| Observation-only task skill | Done (Candidate) | `minicraft/mapped_skill.py` accumulates a map from observed 7x7 patches and explores/plans without a full-state API; train smoke 400–409 succeeded 10/10 per task |
| Skill one-step expectation | Done (hand-coded, uncalibrated) | Predicts selected state fields after one action; matched 65/65, 151/151, and 323/323 checked fields on the same smoke episodes |
| Learned/compressed skill | Not done | Behavior-cloning decision tree was tried; it scored 4/10 T_wood and 0/10 T_tools/T_iron on train evaluation seeds, so it is not presented as a usable skill |
| Held-out variant family and distance `d` | Not defined | Must be specified from the source project direction before generalization evaluation |
| Locked promotion and performance margins | Not defined | Keep open; do not promote or use Confirm |
| Matched-performance integrated compute accounting | Not done | Current timing excludes simulation, safety, and learned world-model costs; replay mostly fails |

**Engineering alpha estimate: 75%** of the checklist above is implemented and
run. This is a progress estimate for Phase 1 alpha software only, not a
research-completion score. “Alpha complete” as a runnable software slice would
require a useful learned/compressed skill, a held-out variant protocol, and
integrated matched-performance compute accounting. Until then, status remains
**alpha in progress**.

## Run the available alpha paths

Phase 0 smoke:

```powershell
python -m scripts.phase0_alpha_run
```

Phase 1 skill smoke:

```powershell
python -m scripts.phase1_skill_sweep --seed-start 200 --seed-count 10
```

Observation-only mapped skill:

```powershell
python -m scripts.phase1_mapped_skill --eval-seeds 400,401,402,403,404,405,406,407,408,409
```

Both use explicitly non-evidentiary exploratory data. The Phase 1 sweep reads
the archived source traces under `results/phase1_skill_smoke_20261011/` and
writes a unique run directory under `work/` unless `--output-dir` is supplied.

## Next engineering steps

1. Develop a compressed learned skill that succeeds beyond the demonstration
   trajectories while keeping only observation inputs.
2. Define the held-out variant family and distance `d` from the source project
   direction; keep Confirm absent until a protocol is locked.
3. Measure skill, detector, safety and world-model costs in one runtime at
   matched task performance.
4. Keep per-task and aggregate manifests self-contained and hash-linked.
5. Provide a clean-checkout validation command that checks imports/CLI help,
   documented output layout, and archived manifest/hash consistency without
   accessing held-out or Confirm seeds.

The offline sanity command is `python -m scripts.alpha_sanity`. It checks four
alpha CLI help paths and archived Phase 1 trace/model/example/manifest hashes
without constructing the simulator or accessing any seeds.
