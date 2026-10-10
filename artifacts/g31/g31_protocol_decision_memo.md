# G3.1 Protocol Decision Memo — PROPOSED (NOT APPROVED, NOT LOCKED)
# Audit-only; read-only; no edits to source/checkpoints/datasets/reports
# Confirm NOT ACCESSED | TEST NOT ACCESSED | threshold TO_LOCK | Gate G3.1 INCONCLUSIVE

=== PROVENANCE ===
B1 SHA-256 (direct from artifacts/g31/g31_b1_checkpoint.pt): 136017b650fc910043c559822d47fea900391e3da44 (64 chars) — PASS
Previous truncated report (44 chars): TYPO ONLY — corrected
B1 training history (300 updates, TRAIN-only): EVIDENCE FROM train_g31_b1_checkpoint.py + checkpoint metadata (training_updates, split='train', dataset_path) — NOT inferred from hash
A SHA-256 (direct): 0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147 (64 chars) — PASS

=== STATUS ===
G3.1: INCONCLUSIVE
G3.2: CLOSED
Confirm: NOT ACCESSED
Threshold: TO_LOCK
Decision: NOT READY TO LOCK — blocked by MISSING protocol definitions

=== BLOCKING DECISIONS (must resolve before protocol lock) ===
1. ACCEPTANCE CRITERION (MISSING — BLOCKING)
   Question: What numeric delta + scientific justification defines B1 improvement?
   Evidence: All reports TO_LOCK; plan.md requires pilot analysis; no value exists
   Status: UNRESOLVED — owner must define independent of observed pilot (1.3707 vs 1.4036)

2. STATISTICAL ANALYSIS (MISSING — BLOCKING)
   Question: Permitted? If yes: paired t-test? bootstrap? alpha? Multiple comparison?
   Evidence: None in plan.md; no test artifacts
   Status: UNRESOLVED — owner must specify or prohibit

3. STOP / ABORT / ROLLBACK (MISSING — BLOCKING)
   Question: Stop if Confirm denied? Abort if >X% non-finite? Rollback if corruption?
   Evidence: None in score_pilot/evaluator/plan.md
   Status: UNRESOLVED — owner must define conditions

4. NON-FINITE / MISSING / EXCLUSION POLICY (AMBIGUOUS — BLOCKING)
   Question: Exclude non-finite predictions? Abort on failure? Rollback conditions?
   Evidence: score_pilot raises ValueError; all 100 predictions finite (verified)
   Status: UNRESOLVED — owner must approve rules

=== BLOCKING BEFORE CONFIRM ===
5. CONFIRM ACCESS AUTHORIZATION (MISSING — BLOCKING for Confirm)
   Question: Authorize Confirm (seeds 3001-3005) when locked? Required for final decision?
   Evidence: manifest confirm_accessed=false; plan.md 'Confirm: NOT RUN'; no Confirm artifacts
   Status: UNRESOLVED — owner must authorize when protocol locked

=== NON-BLOCKING (VERIFIED / CONSISTENT) ===
6. EVALUATOR VERSION TRACKING (AMBIGUOUS — MINOR)
   Evidence: evaluator.py unchanged; no __version__ string; runtime hash computed
   Status: CAN RESOLVE AFTER LOCK — not blocking

=== NOT BLOCKING (ALREADY VERIFIED) ===
7. PRIMARY METRIC (theta RMSE) — IMPLEMENTED; consistent
8. AGGREGATION (mean per-episode RMSE) — IMPLEMENTED; consistent
9. COMPARISON (B1 vs A) — IMPLEMENTED; consistent
10. PAIRING (episode_id, timestep) — VERIFIED (100 keys, 0 mismatches)
11. CHECKPOINT IDENTITY (A/B1 hashes) — VERIFIED (hashes match artifacts)
12. DATASET IDENTITY (pilot.jsonl) — VERIFIED (hash + manifest PASS)

=== PROPOSED DEFAULTS (NOT SELECTED — FOR OWNER REVIEW) ===
1. Metric: theta RMSE primary; velocity RMSE supplemental
2. Aggregation: arithmetic mean of per-episode RMSE; report median/dispersion
3. Comparison: paired B1-A; absolute + relative
4. Pilot interpretation: exploratory/descriptive (preserve)
5. Fail-closed: abort on non-finite/hash mismatch
6. Reproducibility: freeze evaluator at f9e71bd; record hashes

=== TRADE-OFFS (NOT CHOSEN) ===
Metric: theta only vs theta+velocity
Aggregation: mean vs median vs pooled
Statistics: none vs paired t-test vs bootstrap
Criterion: absolute delta vs relative % vs effect size
Confirm: required vs optional

=== SEPARATION ===
LOCK requires: metric/aggregation/comparison approval + criterion + stats + stop/abort + exclusion (optional) + version (optional)
CONFIRM requires: Confirm authorization + dataset access + confirmatory comparison

=== RESOLUTION ===
NOT READY TO LOCK — blocked by MISSING protocol definitions
=== END ===
