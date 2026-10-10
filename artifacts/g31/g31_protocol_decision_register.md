@nguyenphuck2468-tech ➜ /workspaces/Soul-AI (main) $  cat << 'EOF'
=== G3.1 PROTOCOL DECISION REGISTER — AUDIT-ONLY ===
MODE: read-only; zero edits; Confirm NOT ACCESSED; TEST NOT ACCESSED; threshold
TO_LOCK; gate INCONCLUSIVE

--- ITEM 1: PRIMARY METRIC (theta RMSE) ---
Technical fact: evaluator.py physical_metrics defines theta_rmse = sqrt(mean(wra
pped_angle_error(pred[:,0], tgt[:,0])**2)) — verified
Protocol statement: plan.md line 62 describes 'metrics (per-episode equal weight
)' — describes, does not approve metric choice
Owner approval: NONE (no explicit 'primary metric = theta RMSE' approval in plan
.md or manifest)
Evidence: soul/g31/evaluator.py:32-39; artifacts/g31/pilot_metrics.json; artifac
ts/g31/staging_b1_metrics.json
Status: OWNER DECISION REQUIRED — confirm theta RMSE as primary; if velocity als
o required, specify weight

--- ITEM 2: AGGREGATION (mean per-episode) ---
Technical fact: score_pilot.py aggregate_episode_metrics computes mean/median/di
spersion of per-episode values — verified
Protocol statement: plan.md line 62 'per-episode equal weight' — descriptive
Owner approval: NONE (no approval of mean vs median vs pooled)
Evidence: soul/g31/score_pilot.py:83-94; reproduced exact (A=1.3707175, B1=1.403
5570)
Status: OWNER DECISION REQUIRED — confirm mean of per-episode RMSE; specify if m
edian/dispersion required

--- ITEM 3: COMPARISON DIRECTION (B1 vs A) ---
Technical fact: score_pilot.py paired_differences computes model - baseline — ve
rified
EOFifacts: unchanged; no edits; no new commits; no experiments run 10, 9, 11, 4)
=== G3.1 PROTOCOL DECISION REGISTER — AUDIT-ONLY ===
MODE: read-only; zero edits; Confirm NOT ACCESSED; TEST NOT ACCESSED; threshold
TO_LOCK; gate INCONCLUSIVE

--- ITEM 1: PRIMARY METRIC (theta RMSE) ---
Technical fact: evaluator.py physical_metrics defines theta_rmse = sqrt(mean(wra
pped_angle_error(pred[:,0], tgt[:,0])**2)) — verified
Protocol statement: plan.md line 62 describes 'metrics (per-episode equal weight
)' — describes, does not approve metric choice
Owner approval: NONE (no explicit 'primary metric = theta RMSE' approval in plan
.md or manifest)
Evidence: soul/g31/evaluator.py:32-39; artifacts/g31/pilot_metrics.json; artifac
ts/g31/staging_b1_metrics.json
Status: OWNER DECISION REQUIRED — confirm theta RMSE as primary; if velocity als
o required, specify weight

--- ITEM 2: AGGREGATION (mean per-episode) ---
Technical fact: score_pilot.py aggregate_episode_metrics computes mean/median/di
spersion of per-episode values — verified
Protocol statement: plan.md line 62 'per-episode equal weight' — descriptive
Owner approval: NONE (no approval of mean vs median vs pooled)
Evidence: soul/g31/score_pilot.py:83-94; reproduced exact (A=1.3707175, B1=1.403
5570)
Status: OWNER DECISION REQUIRED — confirm mean of per-episode RMSE; specify if m
edian/dispersion required

--- ITEM 3: COMPARISON DIRECTION (B1 vs A) ---
Technical fact: score_pilot.py paired_differences computes model - baseline — ve
rified
Protocol statement: none explicit
Owner approval: NONE
Evidence: soul/g31/score_pilot.py:96-108; artifacts/g31/pilot_b1_predictions.jso
nl (invalid, copy of A); artifacts/g31/staging_b1_predictions.jsonl (valid B1)
Status: OWNER DECISION REQUIRED — confirm B1 vs A; specify if absolute or relati
ve difference

--- ITEM 4: PAIRING UNIT ---
Technical fact: (episode_id, timestep) pairing verified — 100 keys equal, 5×20,
targets/actions identical
Protocol statement: none explicit
Owner approval: NONE
Evidence: artifacts/g31/pilot_predictions.jsonl + staging_b1_predictions.jsonl;
pairing verified (0 mismatches)
Status: ALREADY DOCUMENTED (pairing method is implementation; no protocol approv
al needed if owner accepts)

--- ITEM 5: CHECKPOINT SELECTION & PROVENANCE ---
Technical fact: A=0fd3fbc... (g31_verified_train_checkpoint.pt); B1=136017b... (
g31_b1_checkpoint.pt); both verified by sha256; B1 has config_dict/enc_schema/mo
del_state_dict/optimizer_state_dict/training_updates; no Confirm fields
Protocol statement: plan.md 'Checkpoint VALID FOR PILOT'; manifest integrity PAS
S
Owner approval: NONE (protocol claims valid but does not approve specific hashes
)
Evidence: artifacts/g31/g31_verified_train_checkpoint.pt; artifacts/g31/g31_b1_c
heckpoint.pt; artifacts/g31/pilot_manifest.json
Status: EVIDENCE REQUIRED — identity verified; approval of specific checkpoints
for confirmatory use requires owner sign-off

--- ITEM 6: DATASET PROVENANCE ---
Technical fact: pilot.jsonl hash 26d63b1e...; split=pilot; seeds 2001-2005; 100
lines; observation_shape [3]; action_shape [1]; bounds [-2,2]; collection_policy
 bounded_uniform; dataset_status='NEW DATASET — NOT ORIGINAL'
Protocol statement: plan.md 'PILOT: seeds 2001-2005; not accessed'; manifest int
egrity PASS
Owner approval: NONE (dataset is new, not original — informational note)
Evidence: artifacts/g31/pilot.jsonl; artifacts/g31/pilot_manifest.json; pilot_ma
nifest.json (root)
Status: EVIDENCE REQUIRED — dataset identity verified; 'NOT ORIGINAL' note requi
res owner clarification (is this expected?)

--- ITEM 7: SAMPLE / SEED / SIZE ---
Technical fact: 5 episodes, 20 transitions/episode, 100 total; seeds 2001-2005;
train seeds 1001-1005 disjoint
Protocol statement: plan.md lines 107, 123; manifest seeds_train/seeds_pilot
Owner approval: NONE (no sample-size rationale documented)
Evidence: pilot_manifest.json; artifacts/g31/train.jsonl; artifacts/g31/pilot.js
onl
Status: OWNER DECISION REQUIRED — confirm 5/20 sufficient for confirmatory evalu
ation; if not, specify required size

--- ITEM 8: NON-FINITE / MISSING / EXCLUSIONS / ABORT ---
Technical fact: score_pilot raises ValueError if non-finite; no exclusion logic;
 no abort conditions; all 100 predictions finite
Protocol statement: NONE (plan.md does not specify exclusion/abort rules)
Owner approval: NONE
Evidence: soul/g31/score_pilot.py (ValueError); artifacts/g31/staging_b1_predict
ions.jsonl (all finite)
Status: OWNER DECISION REQUIRED — specify: exclude non-finite? abort on failure?
 rollback conditions?

--- ITEM 8b: EVALUATOR VERSION / REPRODUCIBILITY ---
Technical fact: evaluator.py has no __version__ or version string; report descri
bes semantics; source unchanged since audit
Protocol statement: NONE
Owner approval: NONE
Evidence: soul/g31/evaluator.py (no version); artifacts/g31/staging_b1_report.js
on (evaluator description)
Status: OWNER DECISION REQUIRED — add version tracking; confirm evaluator frozen
 for confirmatory run

--- ITEM 9: STATISTICAL ANALYSIS ---
Technical fact: none performed; no test defined; no significance claim made
Protocol statement: NONE (plan.md does not mention statistical analysis)
Owner approval: NONE
Evidence: no statistical test artifacts; no protocol reference
Status: OWNER DECISION REQUIRED — permitted? if yes, specify test (t-test? boots
trap?); if no, document

--- ITEM 10: ACCEPTANCE CRITERION ---
Technical fact: none defined; TO_LOCK requires criterion; no numeric threshold;
no scientific justification
Protocol statement: plan.md 'protocol requires pilot analysis' — describes need,
 not value
Owner approval: NONE
Evidence: all reports say 'threshold: TO_LOCK'; plan.md line 50 'No threshold wa
s locked'
Status: OWNER DECISION REQUIRED — define numeric delta + scientific justificatio
n (e.g., theta RMSE improvement ≥ X% with Y confidence)

--- ITEM 11: STOP / ABORT / ROLLBACK ---
Technical fact: none defined
Protocol statement: NONE
Owner approval: NONE
Evidence: none
Status: OWNER DECISION REQUIRED — specify conditions (e.g., >Z% predictions non-
finite, dataset hash mismatch, evaluator version change)

--- ITEM 12: CONFIRM ACCESS ---
Technical fact: manifest confirm_accessed=false; no Confirm artifacts in artifac
ts/g31/; no Confirm-derived metrics
Protocol statement: plan.md 'Confirm: NOT RUN'; 'Confirm NOT run'
Owner approval: LOCKED (not accessed; protocol requires it remain closed until l
ock)
Evidence: artifacts/g31/pilot_manifest.json; plan.md; all reports
Status: LOCKED (state correct; authorization for future access requires owner)

--- SUMMARY: SMALLEST SET OF OWNER DECISIONS BEFORE LOCK ---
1. Acceptance criterion (numeric delta + justification) — BLOCKING
2. Statistical analysis permitted? (yes/no + test) — BLOCKING
3. Stop/abort/rollback conditions — BLOCKING
4. Confirm access authorization (when to open) — BLOCKING (for confirmatory comp
arison)
5. Evaluator version tracking — NON-BLOCKING (minor)
6. Dataset 'NOT ORIGINAL' clarification — NON-BLOCKING (informational)

--- DECISION ---
NOT READY TO LOCK — blocked by MISSING protocol definitions (items 10, 9, 11, 4)
, not by data/evaluator errors.

--- CONFIRM / TEST / THRESHOLD / ARTIFACT STATUS ---
Confirm: NOT ACCESSED (verified)
TEST seeds: NOT ACCESSED (verified)
Threshold: TO_LOCK (unchanged)
Artifacts: unchanged; no edits; no new commits; no experiments run
@nguyenphuck2468-tech ➜ /workspaces/Soul-AI (main) $ 