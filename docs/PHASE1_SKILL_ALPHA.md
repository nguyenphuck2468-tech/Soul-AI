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
