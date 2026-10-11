# Phase 0 MiniCraft contract (draft)

Status: exploratory draft; not frozen. Source of truth: the user's cross-checked
v2.6 report and long-term project direction. This contract separates requirements
that the sources determine from implementation proposals and unresolved owner
choices. No numeric acceptance margins are inferred where those sources leave
them open.

## Source-to-protocol decision register

| Topic | Source-determined requirement | Current implementation proposal | Still open before formal use |
| --- | --- | --- | --- |
| Research claim | Mature skills should reduce Dreamer intervention and compute in normal conditions while preserving performance within a predeclared non-inferiority margin; intervention should rise again under OOD/risk (v2.6 §§1, 12) | Phase 0 measures only a prerequisite: action-conditioned WM prediction on structured MiniCraft state | Phase 0 is not evidence for the core claim; Phase 5 needs its own frozen protocol |
| Environment track | Prototype can use structured state; it does not establish pixel perception or Dreamer 4 scale (v2.6 §13) | MiniCraft structured observation and copy/restore | Confirm which long-term-design requirements are intentionally deferred to later phases |
| Baselines | Compare against simple baselines and retain added complexity only when it wins (v2.6 §§12, 12b) | Persistence baseline; action permutation is diagnostic only | Exact action diagnostic and its inferential role |
| Horizon | Report multi-step prediction and drift by horizon (v2.6 §12b and its referenced benchmark definitions) | Horizons 1, 5, 20, 100 | Confirm these concrete horizons are the intended Phase 0 set |
| Sampling | Predeclare seeds and task/action distributions; do not tune on test (v2.6 §12) | Seeded uniform random actions; train 0–999, exploratory dev 1000–1999 | Seed subset/count, episode length, task mix, action distribution, and untouched evaluation set; the v2.6 examples are not approved values |
| Metric | Define performance and report uncertainty/statistical procedure before confirmatory claims (v2.6 §12) | Equal-episode mean numeric-slot MSE vs persistence, per-feature diagnostics | Feature weighting, missing-feature handling, aggregation/inference procedure, acceptance criterion, uncertainty method |
| Compute | Report measured wall/GPU time and normalized accounting, including supervision and safety costs; measure rollout-dependent cost (v2.6 §12) | Local WM wall-time helper at listed horizons; no skill denominator exists yet | Integrated cost protocol, hardware/runtime, repetitions, uncertainty intervals, skill-step denominator, and later ρ(h)/σ sweep |
| Safety | Hard safety rules bypass learned intervention gates and apply to every action source (v2.6 §§3, 4.1c–d) | Draft rule forbids moving into visible adjacent lava; noop fallback | Validate the first constraint against simulator semantics; severity, expiry, hysteresis, response modes and logs |
| Data separation | Lock protocol before any test; test set is one-touch or replaced after exposure (v2.6 §12) | No Confirm split/range is defined or accessed; dev seeds 1000–1001 were used for smoke | Choose and document a still-untouched evaluation set before a formal Phase 0 run; never relabel already inspected seeds as untouched |

The v2.6 report's “must lock before Phase 0” list in §12 is broader than the
minimum prerequisite measurement currently implemented: it includes OOD
conditions, headroom, Pareto criteria, intervention metrics and Phase 5
thresholds. Those belong to the later adaptive-compute benchmark and must not
be mistaken for Phase 0 world-model prediction acceptance criteria. The long-
term direction remains the endpoint; this Phase 0 track is a prerequisite, not
a replacement or stopping condition.

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

The minimum task is action-conditioned prediction of future structured
observations from the current observation and action sequence. Report two
distinct modes: (a) **reality-anchored H-step prediction**, where each real
intermediate observation is encoded before the next transition and the final
observation remains held out; and (b) **open-loop H-step prediction**, where
the model receives no intermediate observations. Use fixed encoder slots fitted from train
only; dev scoring freezes that schema. Compare per-episode numeric-feature MSE
for both modes to persistence (repeat the origin observation), giving each
episode equal weight. Report feature-wise errors and observed-feature counts as
diagnostics. The `model_mse` output key is retained as a compatibility alias for
the reality-anchored score; new consumers should use the explicit
`anchored_model_mse` and `open_loop_model_mse` fields.

Do not use recursively imagined states as real targets. Action permutation is a
diagnostic for action sensitivity, not a quality baseline; it does not by itself
show causal action use or compare against a calibrated null. Reward prediction,
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
  schema hash, model config, code revision, dirty-tree indicator and status
  hash, action counts, and raw predictions, targets, masks, and persistence
  values. A commit hash alone does not identify modified working-tree code;
  source-file hashes cover the scorer and model/environment files listed in
  the report, while the dirty-tree status hash records paths, not their contents.
- `scripts/phase0_collect.py` writes train-only transition JSONL;
  `scripts/phase0_train.py` fits and saves the schema/config with the checkpoint;
  `scripts/phase0_score.py` rejects non-dev seeds and reports exploratory
  model-vs-persistence at horizons 1, 5, 20, and 100 plus an action-shuffle
  diagnostic without applying thresholds. It writes raw per-origin predictions
  to a JSONL sidecar and reports unseen-feature overflow. Derived episode
  seeds must be unique; overlapping ranges are rejected. `scripts/phase0_measure_cost.py`
  measures the same MiniCraft adapter call at those horizons with runtime and
  source provenance; it does not supply a skill-cost denominator before a skill
  implementation exists.

Example pipeline (temporary paths shown):

```powershell
python -m scripts.phase0_collect --output results/phase0_train.jsonl --seed-count 20 --steps-per-seed 300
python -m scripts.phase0_train --data results/phase0_train.jsonl --checkpoint results/phase0_soul.pt
python -m scripts.phase0_score --checkpoint results/phase0_soul.pt --output results/phase0_dev_score.json --seed-start 1000 --seed-count 10 --max-ticks 300
python -m scripts.phase0_measure_cost --output results/phase0_cost.jsonl --warmup 5 --reps 30 --threads 1
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
lava (tile ID `LAVA`): actions 1/2/3/4 map to north/south/west/east. Action
`noop` (0) remains an always-available fallback.
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
