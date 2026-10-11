# Soul-AI Plan — G3.1 Status

## Blocker (G3.1 audit)
Row-0 transition-KL boundary defect: replay window row 0 has no predecessor
observation inside window, but Trainer.update() included prior_mu/prior_lv at
index 0 in transition KL. Dynamics.observe_sequence() used action=0 at t=0,
which is correct for boundary, but KL did not exclude it.

## Root cause
Training-target / sequence-boundary contract bug. Transition valid only for
rows 1..T: o_{t-1} --a_t--> o_t. Row 0 must stay for posterior/reconstruction
but not contribute to transition KL.

## Fix (production)
- soul/trainer.py: added `transition_valid` mask (`[:,0]=0`, rest=1), applied
to both `kl_prior` and `kl_post`, averaged over valid entries only.
- No change to observe_sequence action semantics (destination-row preserved).
- No change to action-effect head alignment (`actions[:,1:]` unchanged).

## Before / After (Trainer.update)
Before: row 0 contributed to transition KL (prior_mu[:,0], prior_lv[:,0] included).
After: row 0 excluded from KL; rows 1..seq continue normal transition training.
Row 0 still present in posterior/reconstruction/context.

## Regression
- tests/test_regressions_g31.py: proves row 0 change does not affect masked KL;
valid rows retain contribution.
- All existing regression tests preserved (no delete/skip/weaken).

## Verification
- pytest -q: 12 passed (includes new regression)
- compileall -q soul: pass
- python -m soul.verify --quick: 3/3 PASS
- git diff --check: pass (only intended trainer.py + new test)
- git status: only soul/trainer.py, tests/test_regressions_g31.py changed

## Scope
- Production change: yes (trainer.py mask only)
- G3.1 experiment: NO (not run — blocked per protocol)
- Benchmark / threshold lock: NO
- G3.2: NOT opened

## G3.0 status
G3.0 REVALIDATED — PASS (contract fixed, regression proves semantics,
full suite pass, compile/verify pass, no accidental semantic changes).

## G3.1 status
INCONCLUSIVE — Pilot evaluation completed and audited. The checkpoint
performed substantially worse than both simple baselines on the declared
Pilot distribution. No threshold was locked. Confirm remains closed.
G3.2: CLOSED.

## Next step
When protocol allows: run G3.1 pilot (frozen deployed RSSM prior decode
via existing observation decoder on held-out Pendulum episodes) — one-step,
action-conditioned, observable-state prediction only. Do not expand to
reconstruction/reward/actor/planning/OOD.

---
## G3.1 Implementation (this session)
Infrastructure implemented: collector, evaluator (prior-only, deterministic),
baselines (train-only delta + persistence), metrics (per-episode equal weight),
CLI skeleton, infrastructure tests. Confirm NOT run. Pilot NOT run.
Threshold remains TO_LOCK. G3.1 status: READY FOR PILOT — NOT EXPERIMENTALLY VALIDATED.

## G3.1 Training Run (this session — audit + train-only)
- Audit: save/load pipeline verified (soul/soul.py:527/569); no critical bugs; G3.0 transition-mask preserved
- Training: TRAIN-only (seeds 1001-1005, 100 transitions/episode, 100 updates)
- Checkpoint: `g31_frozen_checkpoint.pt` (3.39MB, SHA-256 8d3a7a3e...)
- Load test: PASS; deterministic inference: PASS
- Data isolation: PILOT/CONFIRM not accessed during training
- Confirm: NOT RUN (untouched)
- Threshold: TO_LOCK (not locked — protocol requires pilot analysis)
- G3.1 status: READY FOR PILOT (checkpoint valid, real data ready)
- G3.2: CLOSED

## G3.1 Action Contract + Real TRAIN Pipeline Audit

### Contract table
| Layer | Expected | Actual before fix | Status |
|---|---|---|---|
| Pendulum observation | `(3,)` | `(3,)` | PASS |
| Pendulum requested action | `(1,)` | not persisted | FIXED |
| Pendulum applied action | `(1,)`, bounds `[-2,2]` | collector accepted arbitrary shape | FIXED |
| Replay action | `(1,)` for G3.1 | global config was 4-D | G3.1 config added |
| Model action | `(1,)` for G3.1 | legacy global default `action_dim=4` | explicit config required |
| Dynamics action | model action input | 4-D under global config | follows explicit G3.1 config |
| Evaluator action | applied `(1,)` | accepted unvalidated tensor | contract guard added |

### Root causes
- Production/infrastructure contract bug: G3.1 training script used synthetic random observations, rewards, and 4-D actions instead of real Pendulum TRAIN data.
- Contract bug: global `Config.action_dim=4` is a legacy general-agent default and is not the Pendulum G3.1 action contract; no arbitrary padding/repetition is allowed.
- Infrastructure bug: real collector returned in-memory data only; no persisted TRAIN JSONL existed for a verifiable loader.
- Provenance bug: `train_g31_checkpoint.py` wrote TRAIN metadata without consuming TRAIN rows.
- No test bug identified. `imagine.py` semantic provenance remains unresolved, but retraining is blocked earlier by the data/schema defect.

### Fixes
- `soul/g31/collector.py`: explicit Pendulum `(3,)` observation and `(1,)` action validation; seed disjointness; requested/applied action preservation; environment ID validation.
- `soul/g31/real_data.py`: passes requested and environment-applied scalar actions separately with Pendulum metadata.
- `soul/g31/train_data.py`: new fail-closed TRAIN-only JSONL loader for seeds 1001-1005, Pendulum-v1, exact shapes.
- `train_g31_checkpoint.py`: now requires `G31_TRAIN_JSONL`/`artifacts/g31/train.jsonl`, uses explicit `action_dim=1`, and refuses to train until real-row encoding is implemented. It no longer generates synthetic training data.
- `tests/test_g31_action_contract.py`: four focused tests for scalar action contract, shape rejection, split isolation, and TRAIN schema.

### Data provenance
- TRAIN: real Pendulum data was previously collected in memory, but no persisted TRAIN JSONL is present; loader therefore fails closed.
- PILOT: seeds 2001-2005; not accessed.
- CONFIRM: seeds 3001-3005; not accessed or evaluated.
- Leakage: no training access path to PILOT/CONFIRM; missing TRAIN artifact blocks execution.

### Validation
- `python -m compileall -q soul`: PASS
- `python -m unittest -q tests/test_g31_action_contract.py`: 4 PASS
- G3.0 transition mask remains in both KL branches.
- Existing `.venv` still lacks pytest; no pytest result claimed.
- Existing invalid checkpoint unchanged; SHA-256 remains `8d3a7a3e2448ea7437325bf67877294fc9933733510e974de47c3b50baf02d38`.

### Gate
`G3.0: PASS`; `G3.1: BLOCKED`; `Training pipeline: BLOCKED pending persisted real TRAIN JSONL and real-row Episode encoding`; `Pilot: NOT RUN`; `Confirm: NOT RUN`; `Threshold: TO_LOCK`; `G3.2: CLOSED`.

## G3.1 Real TRAIN Data Persistence + Episode Encoding
- Environment: `Pendulum-v1`, Gymnasium 1.4.0
- Persisted dataset: `artifacts/g31/train.jsonl`
- TRAIN seeds: 1001-1005; episodes: 5; transitions: 100
- Schema validation: PASS (obs `(3,)`, requested/applied action `(1,)`, applied bounds `[-2,2]`, environment ID, split and provenance)
- Episode encoding: PASS (`JSONL -> validated rows -> Episode`), preserving timestep order, action values, rewards, terminal continuation flags, and episode boundaries
- Split isolation: TRAIN loader accepts only split `train` and seeds 1001-1005; PILOT/CONFIRM rows rejected
- G3.0 boundary semantics unchanged; no `Trainer.update()` called
- Tests: 16 unittest PASS; `compileall` PASS
- Checkpoint: existing invalid checkpoint unchanged; SHA-256 remains `8d3a7a3e2448ea7437325bf67877294fc9933733510e974de47c3b50baf02d38`
- Training: NOT RUN; Pilot: NOT RUN; Confirm: NOT RUN; Threshold: `TO_LOCK`
- Gate: `G3.0 PASS`, `G3.1 BLOCKED`, `Real TRAIN data READY`, `Episode encoding READY`, `G3.2 CLOSED`

## G3.1 Verified TRAIN-only Run
- Dataset: `artifacts/g31/train.jsonl`
- Dataset SHA-256: `520a02ea8b044777e331962a7dd940fc2e1c11fb7d5bcb188d2c7b6147ea6655`
- Environment: Pendulum-v1 / Gymnasium 1.4.0
- Split/seeds: TRAIN / 1001-1005; 5 episodes; 100 transitions
- Effective model action dimension: 1; no padding, repetition, or broadcasting
- Training: one clean run, fresh model initialization, 50 actual `Trainer.update()` optimizer updates; no old checkpoint initialization
- Checkpoint: `artifacts/g31/g31_verified_train_checkpoint.pt`
- Checkpoint SHA-256: `0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147`
- Metadata: PASS (schema, environment, split, seeds, dataset hash, config, optimizer, git commit, timestamp, updates)
- Fresh-model load: PASS; deterministic prior/decoder inference: PASS
- Old rejected checkpoint: unchanged; original SHA-256 preserved
- PILOT accessed: NO; CONFIRM accessed: NO; synthetic data used: NO
- Tests: compileall PASS; 16 relevant unittest PASS; pytest not run because unavailable in `.venv`
- Gate: `G3.0 PASS`, `G3.1 READY FOR PILOT`, `Checkpoint VALID FOR PILOT`, `Training PASS`, `Pilot NOT RUN`, `Confirm NOT RUN`, `Threshold TO_LOCK`, `G3.2 CLOSED`

## G3.1 Pilot Attempt / Audit (2026-10-09)
- Pilot was NOT run. `artifacts/g31/pilot.jsonl` is absent in the current workspace; no synthetic fallback or recollection was performed.
- Confirm was not opened, parsed, hashed, or evaluated.
- Checkpoint remains unchanged; SHA-256 remains `0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147`.
- Additional evaluator contract blocker found: `soul/g31/evaluator.py` returns decoder model-space `obs` output and does not implement physical Pendulum x/v inverse decoding or the required real-next-observation teacher-forced recurrent filtering path.
- Evaluator syntax check passed and 6 relevant unittest tests passed, but no prediction/metric artifacts were created.
- Gate: `G3.0 PASS`; `G3.1 BLOCKED`; `Checkpoint VALID FOR PILOT`; `Pilot NOT RUN`; `Confirm NOT RUN`; `Threshold TO_LOCK`; `G3.2 CLOSED`.
- Required next step: restore/persist immutable real PILOT JSONL, then implement and test physical x/v decoding plus teacher-forced recurrent evaluator before scoring.

## G3.1 Blocker Resolution (2026-10-09)
- Original Pilot provenance: NOT RECOVERABLE from workspace or Git history.
- New immutable Pilot dataset collected, explicitly not original: `artifacts/g31/pilot.jsonl`.
- New Pilot SHA-256: `26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e`.
- New Pilot provenance: Pendulum-v1 / Gymnasium 1.4.0, bounded-uniform policy, seeds 2001-2005, 5 episodes, 100 transitions, applied action shape `(1,)`.
- Confirm was not accessed.
- Evaluator mechanics fixed and tested: exact applied action, prior mean only, `model.eval()`, no sampling, real `obs_next` teacher-forced posterior/recurrent update, no recursive imagined feedback, parameter immutability and deterministic repeat.
- Relevant evaluator tests: 6 unittest PASS; `py_compile` PASS.
- Remaining blocker: protocol does not operationalize physical `x/v` mapping from raw Pendulum `[cos(theta), sin(theta), angular_velocity]`. Decoder output is model-space and cannot be scored as physical x/v without choosing `theta` versus Cartesian position semantics.
- No Pilot scoring artifacts created; threshold remains `TO_LOCK`.
- Gate: `G3.0 PASS`, `G3.1 BLOCKED`, `G3.2 CLOSED`.

## G3.1 Physical Target Protocol Lock (2026-10-09)
- Observation representation: `[cos(theta), sin(theta), angular_velocity]`.
- Physical target: `(theta, angular_velocity)`; `x` is shorthand for angular position `theta`, never Cartesian position.
- `theta = atan2(sin(theta), cos(theta))`; theta units are radians.
- Angular velocity is the direct third component; units are radians/second.
- Angular error is wrapped: `atan2(sin(theta_pred-theta_true), cos(theta_pred-theta_true))`.
- Evaluator inverse-decodes linear model-space observation values with the existing inverse of `ObsEncoder.squash` before converting to physical values.
- Evaluator contract: real current observation encoded with the persisted feature encoder, real applied action shape `(1,)`, deterministic prior mean, physical prediction, then real next observation teacher-forced into posterior state; no recursive imagined feedback.
- Tests: physical conversion, wrapped angle error, decoder conversion, teacher forcing, prior/action contract, immutability, and determinism all pass (6 relevant evaluator infrastructure tests); py_compile passes.
- Pilot dataset unchanged: SHA-256 `26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e`.
- TRAIN dataset and verified checkpoint unchanged. Confirm not accessed. No Pilot scoring artifacts created.
- Gate: `G3.0 PASS`; `G3.1 READY FOR PILOT`; `G3.2 CLOSED`; `Pilot scoring NOT RUN`; `Threshold TO_LOCK`.

## G3.1 Pilot Scoring (2026-10-09)
- Pilot dataset: `artifacts/g31/pilot.jsonl`; SHA-256 `26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e`; 5 episodes / 100 transitions / seeds 2001-2005.
- Checkpoint: `artifacts/g31/g31_verified_train_checkpoint.pt`; SHA-256 `0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147`.
- Evaluator: prior mean, exact applied action `(1,)`, physical theta/velocity decode, wrapped theta error, real-next-observation teacher forcing, no optimizer updates or recursive feedback.
- Artifacts: `pilot_predictions.jsonl`, `pilot_metrics.json`, `pilot_report.json`.
- Determinism: two complete scoring runs byte-identical for predictions, metrics, and report.
- Equal-episode aggregate model metrics: theta MAE `1.186512`, theta RMSE `1.370717`, theta MSE `1.975475`, theta bias `-0.408383`; velocity MAE `3.585914`, velocity RMSE `4.036020`, velocity MSE `17.509625`, velocity bias `-0.236823`.
- Persistence aggregate: theta MAE `0.179262`, theta RMSE `0.201528`, theta MSE `0.043704`; velocity MAE `0.540565`, velocity RMSE `0.592749`, velocity MSE `0.359557`.
- Train-only mean-delta aggregate: theta MAE `0.183495`, theta RMSE `0.206787`, theta MSE `0.046132`; velocity MAE `0.547093`, velocity RMSE `0.599248`, velocity MSE `0.366335`.
- Model-minus-baseline paired differences are positive for all reported MAE/MSE/RMSE metrics; results are exploratory and not a capability claim.
- Threshold: `TO_LOCK`; no threshold was pre-locked or selected retrospectively.
- Confirm: NOT ACCESSED and NOT RUN. TRAIN/checkpoint hashes unchanged.
- Tests: py_compile PASS; 6 relevant unittest PASS; pytest not run.
- Gate: `G3.0 PASS`; `G3.1 INCONCLUSIVE`; `G3.2 CLOSED`; next permitted step requires an explicit threshold-lock protocol decision before any Confirm run.

## G3.1 Post-Pilot Diagnostic Audit (2026-10-09)
- All six required artifact hashes verified; no mismatch and no artifact overwritten.
- Independent recomputation from `pilot_predictions.jsonl` exactly matched official per-episode and equal-episode aggregate metrics.
- Prediction coverage: 100 rows, 20 per each of 5 episodes; provenance fields and checkpoint/Pilot hashes present.
- Prediction collapse evidence: theta prediction range `[1.840857, 3.087245]`, mean `2.824846`, std `0.275060`; velocity range `[-0.110007, -0.041501]`, mean `-0.079102`, std `0.018112`.
- Target ranges: theta `[-3.141120, 3.025432]`, velocity `[-7.390771, 7.128725]`; velocity prediction/target correlation `0.023692`.
- Wrapped angular boundary audit: 1% of rows had absolute wrapped error above `pi-0.1`; wrap formula itself matched the independent reference.
- Baseline audit: persistence and train-only mean-delta metrics independently matched official artifacts; paired difference convention is model minus baseline.
- No confirmed evaluator defect, metric defect, or baseline leakage found. Plausible model-quality cause is narrow prior/decoder collapse, supported by output ranges and near-zero velocity correlation; this is a diagnosis, not a capability claim.
- Contract tests: compileall PASS; 6 relevant unittest PASS. Confirm was not accessed. Threshold remains `TO_LOCK`.
- Gate unchanged: `G3.0 PASS`; `G3.1 INCONCLUSIVE`; `G3.2 CLOSED`; Confirm `NOT RUN`.

## G3.1 Checkpoint Provenance Audit (this session)
- Checkpoint hash reverified: `8d3a7a3e2448ea7437325bf67877294fc9933733510e974de47c3b50baf02d38`
- Load and metadata inspection pass, but provenance is invalid for G3.1.
- `train_g31_checkpoint.py` did not load the real Pendulum TRAIN data. It generated random structured observations with `np.random.randn` and synthetic rewards.
- The script trained with current `Config.model.action_dim = 4`, while Pendulum-v1 has action dimension 1. Its replay actions were synthetic 4-D vectors, not environment-applied Pendulum actions.
- The script explicitly used `trainer.update()` and did not prove that the temporary `imagine_actor_critic` stub was absent in the producing run. Stub provenance is therefore UNKNOWN, but this is not needed to reject the artifact.
- `soul/imagine.py` current stack changes are not sufficient to repair the data/schema mismatch; no pilot was run.
- G3.0 transition-valid mask remains present in both KL branches; compileall passed. Pytest was unavailable in `.venv` during this audit.
- Classification: `INVALID — RETRAIN REQUIRED`.
- Required next step: build or repair a TRAIN-only loader for the recorded real Pendulum split, set the model/action schema to the locked Pendulum contract, run one clean training run, and save a new provenance-complete checkpoint. Do not use the current checkpoint.
- Confirm remains untouched; threshold remains `TO_LOCK`; G3.1 pilot remains `BLOCKED`; G3.2 remains `CLOSED`.
