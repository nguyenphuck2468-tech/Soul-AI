# Phase 1 skill alpha

## Scope

This is the first runnable skill-shaped interface from the v2.6 direction. It
wraps the existing MiniCraft full-state expert as a **Candidate** skill so the
agent loop can exchange an action plus an explicit expected task outcome.
It is intentionally not presented as a learned, compressed, or mature skill.

## Current contract

- Context: one named MiniCraft task and privileged full-map environment state.
- Action: expert-proposed action; the caller passes it through
  `MiniCraftSafetyController` before execution.
- Expected outcome: task-level goal predicate, such as `inv.wood >= 5`.
- Confidence: `NOT_CALIBRATED`; this is not a one-step prediction.
- Status: `Candidate`. No promotion threshold or automatic validation is
  implemented.
- Evidence: completed episodes and successes are counted descriptively. One
  smoke episode is not a success-rate estimate.
- Risks/failure modes: privileged state, BFS replanning cost, unavailable route,
  safety-filtered action, hidden action shock, death, or timeout.

## Run one smoke episode

```powershell
python -m scripts.phase1_skill_smoke --task T_wood --seed 5 --max-ticks 300
```

The seed must remain in the train range (0–999). By default the trace, skill
card, and manifest go under a unique directory in `work/`. The smoke applies
the hard safety filter to every proposed action and records the expected goal
alongside observed transitions.

To compare a saved Candidate trace with replay on exploratory train seeds:

```powershell
python -m scripts.phase1_replay_compare --task T_wood --source-trace results/phase1_skill_smoke_20261011/T_wood/skill_trace.jsonl --eval-seeds 8,9,10
```

Both controllers start from the same initial state for each seed. The script
records raw outcomes and descriptive `A` without applying a threshold.

For a paired exploratory sweep over all three tasks:

```powershell
python -m scripts.phase1_skill_sweep --seed-start 200 --seed-count 10
```

This compares each Candidate expert against its task's saved fixed-action
trace over seeds 200–209 by default. The command archives per-step traces and
per-task manifests under a unique `work/phase1_skill_sweeps/` directory.

## What is still needed before calling it a generalized skill

The source report's held-out variant distance `d` and promotion margins remain
open. Evaluation needs `S_train`, `S_held`, `G = S_train - S_held`,
`A = S_held(skill) - S_held(replay)`, and compute per task against replay.
Until the variant family and thresholds are fixed, this implementation remains
Candidate and is excluded from claims about compute savings or skill maturity.

## Run-through record

One train-seed smoke was run per task: T_wood seed 5 succeeded in 104 steps,
T_tools seed 6 in 57 steps, and T_iron seed 7 in 227 steps. These are three
single-episode wiring checks, not success-rate estimates or evidence of
generalization. Traces, cards, and manifests are archived under
`results/phase1_skill_smoke_20261011/`.

A small action-replay comparison then used each trace on one different train
seed. The Candidate expert succeeded and replay failed once for each task:
T_wood source/eval 5/8 (92 steps), T_tools 6/9 (26), and T_iron 7/10 (53).
The resulting descriptive `A` was +1.0 per task on a one-episode sample. This
is an exploratory demonstration that fixed action traces can fail across new
world seeds while the privileged expert replans; it does not establish the
source report's held-out variant metric or skill generalization. Full traces
and manifests are in `results/phase1_skill_replay_20261011/`.

A follow-up paired sweep used train seeds 200–209 for each task. Candidate
succeeded in 10/10 episodes for T_wood, T_tools, and T_iron; fixed replay
succeeded in 1/10, 0/10, and 0/10 respectively. Descriptive `A` was +0.9,
+1.0, and +1.0. This is still exploratory train-seed evidence: it does not
measure held-out variants, lock `d` or a promotion margin, validate a mature
skill, or support a compute-savings claim. Full traces and manifests are in
`results/phase1_skill_sweep_20261011/`.

The sweep runner also reports local wall time for the action proposal alone.
Across these runs, Candidate used about 0.124–0.136 ms per proposed step and
replay about 0.0007–0.0009 ms. Replay usually failed, and these timings exclude
environment stepping, safety filtering, serialization, and any learned
world-model work. They diagnose this privileged BFS wrapper; they do not
compare matched task performance or provide integrated compute accounting.
The timing-enriched manifest is archived at
`results/phase1_skill_sweep_20261011/compute_manifest.json`.
