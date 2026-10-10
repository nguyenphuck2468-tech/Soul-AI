# G3.1 Protocol Draft v0.1 — PROPOSED (NOT APPROVED, NOT LOCKED)

Status: PROPOSED DRAFT — for owner review only. Not approved. Not locked.
Audit-only; read-only; no edits to source/checkpoints/datasets/reports.
Confirm NOT ACCESSED; TEST NOT ACCESSED; threshold TO_LOCK; gate G3.1 INCONCLUSIVE; G3.2 CLOSED.

---

## 1. EXPERIMENTAL OBJECTIVE & HYPOTHESIS

Objective: Evaluate whether G3.1 B1 checkpoint (300 updates, TRAIN-only, fresh init) improves prediction against reality compared to Candidate A (verified train checkpoint), using physical theta/velocity evaluator on real Pilot dataset.

Hypothesis (descriptive, not confirmatory): B1 may show different per-episode theta RMSE vs A; direction and magnitude not established by pilot alone; requires confirmatory comparison.

Evidence: Pilot dataset (artifacts/g31/pilot.jsonl, sha256=26d63b1e...); A predictions (artifacts/g31/pilot_predictions.jsonl); B1 predictions (artifacts/g31/staging_b1_predictions.jsonl); evaluator.py physical_metrics.

Not claimed: Statistical significance, causation, generalization, or model-quality collapse. These require Confirm comparison and protocol-approved statistical analysis.

---

## 2. CHECKPOINT & DATASET PROVENANCE

### Checkpoints
- A: artifacts/g31/g31_verified_train_checkpoint.pt (sha256=0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147, 3386259 bytes)
- B1: artifacts/g31/g31_b1_checkpoint.pt (sha256=136017b650fc910043c559822d47fea900391e3da44, 3383579 bytes)
- Both verified by runtime sha256; both contain config_dict, enc_schema, model_state_dict, optimizer_state_dict, training_updates; no Confirm-derived fields
- Protocol claim (plan.md): "Checkpoint VALID FOR PILOT" — describes validity; does not approve specific hashes for confirmatory use
- Owner approval required: Confirm specific checkpoints for confirmatory evaluation

### Dataset
- Pilot: artifacts/g31/pilot.jsonl (sha256=26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e)
- Split: pilot; seeds: 2001-2005; episodes: 5; transitions: 100; observation_shape=[3]; action_shape=[1]; action_bounds=[-2.0, 2.0]; collection_policy=bounded_uniform
- Manifest (artifacts/g31/pilot_manifest.json): integrity PASS; dataset_status="NEW DATASET — NOT ORIGINAL" (informational note)
- Protocol (plan.md): "PILOT: seeds 2001-2005; not accessed"
- Owner approval required: Confirm dataset approved; clarify "NOT ORIGINAL" note

---

## 3. PRIMARY METRIC, AGGREGATION, COMPARISON (PROPOSED — NOT SELECTED)

### Primary metric (PROPOSED — requires owner approval)
- Theta RMSE (physical theta in radians, wrapped periodic error)
- Evidence: evaluator.py physical_metrics (line 32-39); score_pilot.py aggregates; pilot_metrics.json / staging_b1_metrics.json
- Implementation: verified; protocol approval: MISSING (plan.md describes evaluator, does not approve metric choice)
- Supplemental: Velocity RMSE (optional; not required unless owner approves)
- Trade-off: Theta only (simpler, focused on physical position) vs theta+velocity (richer, captures dynamics)

### Aggregation (PROPOSED — requires owner approval)
- Arithmetic mean of per-episode RMSE (current implementation: aggregate_episode_metrics)
- Evidence: score_pilot.py:83-94; reproduced exact (A=1.3707174686520727, B1=1.4035570432979587)
- Supplemental reporting: median, dispersion (std), episode_count
- Trade-off: Mean (sensitive to outliers) vs median (robust) vs pooled transition-level (different weighting)
- Note: Earlier direct-pool over all 100 transitions (1.4055/1.4287) was incorrect method; official is mean of per-episode RMSE

### Comparison direction (PROPOSED — requires owner approval)
- Paired episode-level B1 − A differences; report absolute and relative
- Evidence: score_pilot.py paired_differences; predictions paired by (episode_id, timestep)
- Trade-off: Absolute delta (interpretable in RMSE units) vs relative % (scale-invariant)

### Pairing (VERIFIED — requires owner approval of method)
- (episode_id, timestep) — 100 keys equal; 5 episodes × 20 transitions; 0 duplicates; 0 missing
- Targets and applied actions identical across A/B1 (max diff 0)
- Evidence: artifacts/g31/pilot_predictions.jsonl + artifacts/g31/staging_b1_predictions.jsonl

---

## 4. PILOT INTERPRETATION (ALREADY DOCUMENTED — PRESERVE)

- Status: Exploratory / descriptive only
- Evidence: All reports (pilot_report.json, staging_b1_report.json) say "pilot_scoring: EXPLORATORY; no confirm run"; plan.md "G3.1 READY FOR PILOT — NOT EXPERIMENTALLY VALIDATED"
- Not confirmatory: No Confirm dataset used; no Confirm-derived metrics; no statistical significance claimed
- Per-episode theta RMSE: mixed (ep2001/2003 B1 better; 2002/2004/2005 B1 worse); aggregate +2.40% (0.0328)
- Per-episode velocity RMSE: mixed (smaller magnitude +0.40%); not uniform
- Sample: 5 episodes, 20 transitions/episode — too small for significance
- No quality verdict: No threshold defined; no scientific justification drafted

---

## 5. FAIL-CLOSED / EXCLUSION / ABORT (PROPOSED — requires approval)

- Non-finite predictions: score_pilot raises ValueError; all 100 predictions finite (verified)
- Missing/non-finite: Abort evaluation; report failure; do not silently exclude
- Key mismatch: Abort (pairing requires exact match)
- Hash mismatch: Abort (checkpoint identity must match expected)
- Exclusions: None permitted without explicit approval
- Rollback: Define conditions (e.g., artifact corruption, dataset hash mismatch, evaluator version change)
- Evidence: score_pilot.py (ValueError); artifacts/g31/staging_b1_predictions.jsonl (all finite)
- Trade-off: Fail-closed (safe) vs fail-open (flexible) — recommend fail-closed for confirmatory

---

## 6. REPRODUCIBILITY (ALREADY IMPLEMENTED — PROPOSED FOR APPROVAL)

- Checkpoint hashes: A=0fd3fbc...; B1=136017b... (runtime computed, not hardcoded)
- Dataset hash: 26d63b1e... (verified)
- Evaluator source: soul/g31/evaluator.py (no version string currently)
- Code revision: f9e71bd (interface fix); evaluator unchanged
- Evidence: artifacts/g31/g31_protocol_decision_register.md; score_pilot.py runtime hash; git log
- Unresolved: Evaluator version tracking (no __version__); recommend adding version string or freezing source hash

---

## 7. STATISTICAL ANALYSIS (MISSING — BLOCKING)

- Status: No protocol reference; no test artifacts; no significance claim made
- Evidence: None (plan.md has no statistical-analysis section)
- Unresolved: Permitted? If yes: paired t-test? bootstrap? alpha? Multiple comparison correction?
- Trade-off: None (simple) vs paired t-test (requires normality assumption) vs bootstrap (non-parametric, more robust for n=5)
- Note: With n=5 episodes, statistical power is low regardless of test; any significance claim requires justification
- Recommendation (not selected): Document whether statistical analysis is permitted; if permitted, specify test and alpha; if prohibited, document prohibition

---

## 8. ACCEPTANCE CRITERION (MISSING — BLOCKING)

- Status: No numeric delta; no scientific justification; all reports say "threshold: TO_LOCK"
- Evidence: artifacts/g31/pilot_metrics.json; artifacts/g31/staging_b1_metrics.json; artifacts/g31/pilot_report.json; artifacts/g31/staging_b1_report.json; plan.md line 50
- Unresolved: Numeric delta (e.g., B1 theta RMSE ≤ A - X%); direction (lower RMSE = improvement?); scientific rationale (domain-based, not data-based)
- Critical: Must be independent of observed pilot outcome (1.3707 vs 1.4036). Cannot derive threshold from this pilot.
- Trade-off: Absolute delta (e.g., -0.05 RMSE) vs relative % (e.g., -5%) vs effect size
- Recommendation (not selected): Owner must define criterion with scientific justification before lock

---

## 9. STOP / ABORT / ROLLBACK (MISSING — BLOCKING)

- Status: No conditions in score_pilot, evaluator, or plan.md
- Evidence: None
- Unresolved: Stop if Confirm denied? Abort if >X% non-finite? Rollback if artifact corruption?
- Trade-off: Strict (abort on any deviation) vs lenient (allow minor deviations)
- Recommendation (not selected): Define fail-closed conditions; specify rollback trigger

---

## 10. CONFIRM ACCESS AUTHORIZATION (LOCKED — STATE CORRECT; FUTURE AUTHORIZATION MISSING)

- Status: Confirm NOT ACCESSED (verified); must remain closed until lock
- Evidence: artifacts/g31/pilot_manifest.json (confirm_accessed=false); plan.md "Confirm: NOT RUN"; all reports
- Unresolved: When to authorize Confirm access? Required for final decision?
- Trade-off: Confirm required for final decision vs optional (pilot-only decision)
- Recommendation (not selected): Authorize Confirm only after protocol locked and acceptance criterion met; specify Confirm dataset (seeds 3001-3005) and comparison protocol

---

## 11. SEPARATION: PILOT OBSERVATIONS VS CONFIRMATORY EVIDENCE

- Pilot observations (descriptive, not confirmatory):
  - Per-episode theta RMSE: mixed (ep2001/2003 B1 better; 2002/2004/2005 B1 worse)
  - Aggregate theta RMSE: A=1.3707, B1=1.4036, diff=+0.0328 (+2.40%)
  - Aggregate velocity RMSE: A=4.0360, B1=4.0522, diff=+0.0162 (+0.40%)
  - Sample: 5 episodes, 20 transitions/episode — too small for significance
  - No statistical significance claimed; no quality verdict; no causation
- Confirmatory evidence (NOT AVAILABLE — Confirm NOT ACCESSED):
  - Confirm dataset (seeds 3001-3005) — NOT ACCESSED
  - Confirm predictions/metrics — NOT GENERATED
  - Confirm comparison — NOT PERFORMED
  - Confirm-based quality decision — NOT POSSIBLE
- Note: Pilot observations must not be treated as confirmatory evidence.

---

## 12. PROPOSED DECISION (NOT SELECTED — FOR OWNER REVIEW)

- NOT READY TO LOCK — blocked by MISSING: acceptance criterion, statistical analysis, stop/abort, Confirm authorization
- NOT blocked by: evaluator, dataset, pairing, metrics, artifacts, checkpoint identity
- All proposed defaults (items 1-6 above) are presented for review; none selected
- All trade-offs (metric/aggregation/statistics/criterion/Confirm) presented; not chosen
- No thresholds invented; no statistical claims made; no scientific justifications drafted

---

## 13. REFERENCES

- Protocol register: artifacts/g31/g31_protocol_decision_register.md (9009B)
- Evaluator: soul/g31/evaluator.py (physical_metrics, wrapped_angle_error, initial_state, evaluate_one_step)
- Scoring: soul/g31/score_pilot.py (run, aggregate_episode_metrics, paired_differences, episode_metrics)
- Manifest: artifacts/g31/pilot_manifest.json; pilot_manifest.json (root)
- Plan: plan.md (G3.1 status, blocker, fix, verification)
- Checkpoints: artifacts/g31/g31_verified_train_checkpoint.pt; artifacts/g31/g31_b1_checkpoint.pt
- Dataset: artifacts/g31/pilot.jsonl
- Pilot artifacts: artifacts/g31/pilot_predictions.jsonl; artifacts/g31/pilot_metrics.json; artifacts/g31/pilot_report.json
- B1 staging artifacts: artifacts/g31/staging_b1_predictions.jsonl; artifacts/g31/staging_b1_metrics.json; artifacts/g31/staging_b1_report.json
- Evidence: artifacts/g31/g31_continuation_evidence.jsonl

---
## 14. STATUS PRESERVATION

- G3.1: INCONCLUSIVE
- G3.2: CLOSED
- Confirm: NOT ACCESSED
- Threshold: TO_LOCK
- Source edited: NO (only f9e71bd interface from prior task)
- Checkpoints changed: NO
- Datasets changed: NO
- Existing artifacts overwritten: NO
- New experiments: NO
- New commits: NO (audit-only)

---

## END — PROPOSED PROTOCOL DRAFT v0.1 — NOT APPROVED — NOT LOCKED

Note: This draft is a proposal for owner review. It does not approve any protocol choice, invent any threshold, or claim any scientific conclusion. All proposed defaults are presented with evidence and trade-offs; none are selected. The protocol remains NOT READY TO LOCK until all blocking decisions (acceptance criterion, statistical analysis, stop/abort, Confirm authorization, non-finite policy, evaluator version) are explicitly resolved by the owner with independent scientific justification.
EOF