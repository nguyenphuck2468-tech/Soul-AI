# Phase 0 MiniCraft contract (draft)

Status: exploratory draft; not frozen. Source of truth: the user's cross-checked
v2.6 report and long-term project direction. No numeric acceptance margins are
inferred where those sources leave them open.

## Environment and observation

`MiniCraftEnv` is a deterministic seeded, structured-state simulator with
copy/restore support (`get_state`, `set_state`, `clone`). Its observation is:

| Field | Meaning |
| --- | --- |
| `self.{x,y,facing,health,food}` | Player position, orientation and survival state |
| `inv.*` | Six inventory counts |
| `patch[7,7]` | Local tile grid; tile IDs are defined in `minicraft/env.py` |
| `mobs[]` | Visible nearby zombies as relative `dx`, `dy`, and `hp` |
| `events.{last_result,damage,tick}` | Last action result, damage and episode tick |

There is no velocity field because movement has no momentum. The environment
does not expose global map geometry in the observation; the expert's privileged
full-map access is only a feasibility baseline and is not available to the WM.
The structured-state track tests adaptive control logic, not pixel perception,
Dreamer 4 scale, or the invariant that at least one cue bypasses learned
encoding.

## World-model task

The minimum task is action-conditioned prediction of the next structured
observation from the current observation and the action, anchored to the real
observation after every transition. Use fixed encoder slots fitted from train
only; dev scoring freezes that schema. Compare per-episode numeric-feature MSE
to persistence (repeat current observed values), giving each episode equal
weight. Report feature-wise errors and observed-feature counts as diagnostics.

Report prediction by horizon where the evaluation implementation supports it;
do not use recursively imagined states as real targets. Action permutation is a
diagnostic for action sensitivity, not a quality baseline. Reward prediction,
policy quality, planning, uncertainty calibration, adaptive compute, and
OOD/generalization claims are outside this minimum measurement until their
contracts are specified.

The adapter's uncertainty output remains a zero placeholder. It must not enter
metrics, triggers, or claims. A future uncertainty measure needs independent
calibration against held-out train/dev outcomes before use.

## Data and reproducibility

- Train seeds 0–999: collection and model fitting only.
- Dev seeds 1000–1999: exploratory diagnostics and implementation iteration.
- Confirm: no range exists; do not create, infer, or access Confirm seeds.
- Record seed, episode, tick, integer action, split, checkpoint hash, feature
  schema hash, model config, code revision, and raw predictions/targets.
- `scripts/phase0_collect.py` writes train-only transition JSONL;
  `scripts/phase0_train.py` fits and saves the schema/config with the checkpoint;
  `scripts/phase0_score.py` rejects non-dev seeds and reports exploratory
  model-vs-persistence at horizons 1, 5, 20, and 100 plus an action-shuffle
  diagnostic without applying thresholds. Report unseen-feature overflow.

Example pipeline (temporary paths shown):

```powershell
python -m scripts.phase0_collect --output results/phase0_train.jsonl --seed-count 20 --steps-per-seed 300
python -m scripts.phase0_train --data results/phase0_train.jsonl --checkpoint results/phase0_soul.pt
python -m scripts.phase0_score --checkpoint results/phase0_soul.pt --output results/phase0_dev_score.json --seed-start 1000 --seed-count 10 --max-ticks 300
```

The scorer's smoke run on dev seeds 1000–1001 was a pipeline diagnostic using a
5-update, 60-transition temporary checkpoint. It showed model MSE above
persistence and did not measure a trained model. Treat those two seeds as
already seen in dev iteration; do not use that output as formal evidence.

## SafetyController contract for MiniCraft (specification only)

Inputs: structured observation, hard constraints, alert severity, candidate
action, current mode/context. Outputs: safety mode, allowed action set, and an
expiry tick. Every source of action must pass through the filter before
execution; a world-model proposal never commits directly.

The only hard constraint proposed for the first simulator version is to forbid
a directional move when the corresponding visible adjacent `patch` cell is
lava (tile ID `LAVA`). Action `noop` (0) remains an always-available fallback.
The filter may reject or permit a candidate, but does not invent a route,
search the map, change the task, or create a multi-step plan. It applies equally
to expert/skill, policy, and future WM actions and logs the input, decision,
mode, allowed set, and expiry.

Severity-to-mode mapping, expiry duration, hysteresis, and any additional hard
constraints remain open for protocol review; no values are silently chosen in
this draft. Safety modes and fallback behavior must not wait for a WM call.

## Protocol items still to lock

Before the Phase 0 evaluation is treated as an experiment, review and lock its
seed list, task/action distribution, rollout horizons, feature weighting and
missing-feature handling, action-conditioning diagnostic, uncertainty method,
compute accounting, statistical procedure, and acceptance margins. Before
Phase 5, additionally lock δ, `h_OOD`, headroom margins, `ρ(h)`, `σ_det`, OOD
dev/test types, `R_max`, Pareto comparison, and the one-touch test procedure.
The source report intentionally supplies no numeric values for these items.
