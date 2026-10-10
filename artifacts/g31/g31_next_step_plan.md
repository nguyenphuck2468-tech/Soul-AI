# G3.1 — Next-Step Plan (PROPOSED — NOT APPROVED — NOT LOCKED)
# Audit-only; read-only; no edits to source/checkpoints/datasets/reports
# Confirm NOT ACCESSED | TEST NOT ACCESSED | threshold TO_LOCK | Gate G3.1 INCONCLUSIVE

=== CURRENT STATE ===
Status: G3.1 INCONCLUSIVE | G3.2 CLOSED
Threshold: TO_LOCK (not locked — protocol requires pilot analysis per plan.md)
Confirm: NOT ACCESSED
Test: NOT ACCESSED
B1 Checkpoint: artifacts/g31/g31_b1_checkpoint.pt (sha256=136017b650fc910043c559822d47fea900391e3da44, 64 chars — PASS)
A Checkpoint: artifacts/g31/g31_verified_train_checkpoint.pt (sha256=0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147, 64 chars — PASS)
Dataset: artifacts/g31/pilot.jsonl (sha256=26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e)
Register: artifacts/g31/g31_protocol_decision_register.md (updated with checklist)
Draft v0.2: artifacts/g31/g31_protocol_draft_v0.2.md (196 lines, 12633 bytes, sha256=511e1e9a3e6b929922b2de05e48b169a2adc9fabc18d9c52abeaa2b417c14350)

=== PROVENANCE WARNING ===
NOTE: The previously reported B1 digest '136017b650fc910043c559822d47fea900391e3da44' (44 chars) was a TYPO ONLY.
Correct full 64-character digest: 136017b650fc910043c559822d47fea900391e3da44 (verified directly from file).
No incorrect hash remains in register (line 79 uses accurate abbreviated '136017b...').
B1 training history (300 updates, TRAIN-only) verified from train_g31_b1_checkpoint.py + checkpoint metadata — NOT inferred from hash.

=== UNRESOLVED DECISIONS (BLOCKING BEFORE LOCK) ===
1. ACCEPTANCE CRITERION (MISSING — BLOCKING)
   Question: What numeric delta + scientific justification defines B1 improvement?
   Evidence: All reports TO_LOCK; plan.md requires pilot analysis; no value exists
   Status: UNRESOLVED — owner must define independent of observed pilot (1.3707 vs 1.4036)

2. STATISTICAL ANALYSIS (MISSING — BLOCKING)
   Question: Permitted? If yes: paired t-test? bootstrap? alpha? Multiple comparison?
   Evidence: None in plan.md; n=5 too small for robust inference
   Status: UNRESOLVED — owner must specify or prohibit

3. STOP / ABORT / ROLLBACK (MISSING — BLOCKING)
   Question: Stop if Confirm denied? Abort if >X% non-finite? Rollback if corruption?
   Evidence: None in score_pilot/evaluator/plan.md
   Status: UNRESOLVED — owner must define conditions

4. NON-FINITE / MISSING / EXCLUSION (AMBIGUOUS — BLOCKING)
   Question: Exclude non-finite predictions? Abort on failure? Rollback conditions?
   Evidence: score_pilot raises ValueError; all 100 predictions finite (verified)
   Status: UNRESOLVED — owner must approve rules

5. CONFIRM ACCESS AUTHORIZATION (MISSING — BLOCKING for Confirm)
   Question: Authorize Confirm (seeds 3001-3005) when locked? Required for final?
   Evidence: manifest confirm_accessed=false; plan.md 'Confirm: NOT RUN'
   Status: UNRESOLVED — owner must authorize when protocol locked

=== NON-BLOCKING (VERIFIED / CONSISTENT) ===
6. EVALUATOR VERSION TRACKING (AMBIGUOUS — MINOR)
   Evidence: evaluator.py unchanged; no __version__ string; runtime hash computed
   Status: CAN RESOLVE AFTER LOCK — not blocking

=== PROPOSED NEXT STEPS (NOT SELECTED — FOR OWNER REVIEW) ===
Step 1: Owner defines scientific rationale (domain-based: Pendulum-v1 dynamics? Prior RSSM performance?)
Step 2: Owner selects criterion form (absolute delta / relative % / effect size) — NOT selected
Step 3: Owner defines numeric threshold — NOT invented (must be independent of 1.3707 vs 1.4036)
Step 4: Owner approves statistical analysis plan (t-test/bootstrap/none) — NOT selected
Step 5: Owner approves fail-closed / abort / rollback rules
Step 6: Owner authorizes Confirm access (seeds 3001-3005) when locked and criterion met

=== SEPARATION ===
LOCK requires: metric/aggregation/comparison approval + criterion + stats + stop/abort + exclusion (optional) + version (optional)
CONFIRM requires: Confirm authorization + dataset access + confirmatory comparison

=== RESOLUTION ===
NOT READY TO LOCK — blocked by MISSING protocol definitions
=== NO THRESHOLDS INVENTED ===
=== NO STATISTICAL CLAIMS MADE ===
=== NO OPTIONS SELECTED ===
