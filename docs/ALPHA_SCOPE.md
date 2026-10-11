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
| Non-privileged/compressed skill implementation | Not done | Current Candidate calls privileged full-state BFS on every action |
| Held-out variant family and distance `d` | Not defined | Must be specified from the source project direction before generalization evaluation |
| Locked promotion and performance margins | Not defined | Keep open; do not promote or use Confirm |
| Matched-performance integrated compute accounting | Not done | Current timing excludes simulation, safety, and learned world-model costs; replay mostly fails |

**Engineering alpha estimate: 70%** of the checklist above is implemented and
run. This is a progress estimate for Phase 1 alpha software only, not a
research-completion score. “Alpha complete” as a runnable software slice would
require either implementing the remaining rows or explicitly deciding that
the first alpha intentionally ships the privileged expert wrapper as a
placeholder. Until then, status remains **alpha in progress**.

## Run the available alpha paths

Phase 0 smoke:

```powershell
python -m scripts.phase0_alpha_run
```

Phase 1 skill smoke:

```powershell
python -m scripts.phase1_skill_sweep --seed-start 200 --seed-count 10
```

Both use explicitly non-evidentiary exploratory data. The Phase 1 sweep reads
the archived source traces under `results/phase1_skill_smoke_20261011/` and
writes a unique run directory under `work/` unless `--output-dir` is supplied.

## Next engineering steps

1. Make each per-task and aggregate manifest self-contained and hash-linked.
2. Provide a clean-checkout validation command that checks imports/CLI help,
   documented output layout, and archived manifest/hash consistency without
   accessing held-out or Confirm seeds.
3. Keep the skill wrapper explicitly labeled as a privileged placeholder until
   a compressed/non-privileged skill and source-defined variant protocol exist.

The offline sanity command is `python -m scripts.alpha_sanity`. It checks both
CLI help paths and Phase 1 archived trace/manifest hashes without constructing
the simulator or accessing any seeds.
