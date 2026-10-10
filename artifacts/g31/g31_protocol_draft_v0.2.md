# G3.1 Protocol Draft v0.2 — PROPOSED (NOT APPROVED, NOT LOCKED)

Status: PROPOSED DRAFT v0.2 — for owner review only. Not approved. Not locked.
Audit-only; read-only; no edits to source/checkpoints/datasets/reports.
Confirm NOT ACCESSED; TEST NOT ACCESSED; threshold TO_LOCK; gate G3.1 INCONCLUSIVE; G3.2 CLOSED.

Note: This is a revised draft addressing methodological clarifications from v0.1. It does not approve any protocol choice, invent any threshold, or claim any scientific conclusion. All proposed defaults remain unselected.

---

## 1. EXPERIMENTAL OBJECTIVE & HYPOTHESIS (REVISED — NEUTRAL)

Objective: Evaluate whether the G3.1 B1 checkpoint produces predictions that differ from Candidate A predictions when evaluated against the real Pilot dataset using the physical theta/velocity evaluator.

Hypothesis (descriptive, not confirmatory, not directional): The B1 model may exhibit different per-episode theta RMSE compared to A; the direction and magnitude of any difference are not established by the pilot alone and require confirmatory comparison.

Evidence: Pilot dataset (artifacts/g31/pilot.jsonl, sha256=26d63b1e...); A predictions (artifacts/g31/pilot_predictions.jsonl); B1 predictions (artifacts/g31/staging_b1_predictions.jsonl); evaluator.py physical_metrics.

Not claimed: Statistical significance, causation, generalization, model-quality improvement, or model-quality collapse. These require Confirm comparison and protocol-approved statistical analysis.

---

## 2. CHECKPOINT & DATASET PROVENANCE (VERIFIED — NOT APPROVED)

### Checkpoints — Identity Verified; Approval Required
- A: artifacts/g31/g31_verified_train_checkpoint.pt (sha256=0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147, 3386259 bytes)
- B1: artifacts/g31/g31_b1_checkpoint.pt (sha256=136017b650fc910043c559822d47fea900391e3da44, 3383579 bytes)
- Both verified by runtime sha256; both contain config_dict, enc_schema, model_state_dict, optimizer_state_dict, training_updates; no Confirm-derived fields
- Protocol claim (plan.md): "Checkpoint VALID FOR PILOT" — describes validity; does not approve specific hashes for confirmatory use
- Implementation verified: score_pilot.py computes checkpoint_sha256 at runtime (not hardcoded)
- Owner approval required: Confirm specific checkpoints for confirmatory evaluation
- Status: IDENTITY LOCKED; APPROVAL MISSING

### Dataset — Identity Verified; Provenance Note Clarified
- Pilot: artifacts/g31/pilot.jsonl (sha256=26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e)
- Split: pilot; seeds: 2001-2005; episodes: 5; transitions: 100; observation_shape=[3]; action_shape=[1]; action_bounds=[-2.0, 2.0]; collection_policy=bounded_uniform
- Manifest (artifacts/g31/pilot_manifest.json): integrity PASS; dataset_status="NEW DATASET — NOT ORIGINAL" (informational note — does not imply non-representative; clarification needed)
- Protocol (plan.md): "PILOT: seeds 2001-2005; not accessed"
- Implementation verified: load_rows validates split and seeds; dataset hash verified
- Owner approval required: Confirm dataset approved; clarify whether "NOT ORIGINAL" affects confirmatory validity
- Status: IDENTITY LOCKED; PROVENANCE CLARIFICATION MISSING

---

## 3. PRIMARY METRIC, AGGREGATION, COMPARISON (PROPOSED — NOT SELECTED)

### Primary metric (PROPOSED — requires owner approval; NOT approved by protocol)
- Theta RMSE (physical theta in radians, wrapped periodic error)
- Evidence: evaluator.py physical_metrics (line 32-39); score_pilot.py aggregates; pilot_metrics.json / staging_b1_metrics.json
- Implementation: verified; protocol approval: MISSING (plan.md describes evaluator, does not approve metric choice)
- Supplemental: Velocity RMSE (optional; not required unless owner approves)
- Trade-off: Theta only (simpler, focused on physical position) vs theta+velocity (richer, captures dynamics)
- Note: This is an implementation fact, not a protocol approval. The metric is implemented; whether it is the approved primary metric requires owner decision.

### Aggregation (PROPOSED — requires owner approval; NOT approved by protocol)
- Arithmetic mean of per-episode RMSE (current implementation: aggregate_episode_metrics)
- Evidence: score_pilot.py:83-94; reproduced exact (A=1.3707174686520727, B1=1.4035570432979587)
- Note: Earlier direct-pool over all 100 transitions (1.4055/1.4287) was incorrect method; official is mean of per-episode RMSE
- Supplemental reporting: median, dispersion (std), episode_count
- Trade-off: Mean (sensitive to outliers) vs median (robust) vs pooled transition-level (different weighting)
- Note: Implementation is consistent; protocol approval of aggregation method is missing.

### Comparison direction (PROPOSED — requires owner approval; NOT approved by protocol)
- Paired episode-level B1 − A differences; report absolute and relative
- Evidence: score_pilot.py paired_differences; predictions paired by (episode_id, timestep)
- Trade-off: Absolute delta (interpretable in RMSE units) vs relative % (scale-invariant)
- Note: Implementation is consistent; protocol approval of comparison direction is missing.

### Pairing (VERIFIED — requires owner approval of method)
- (episode_id, timestep) — 100 keys equal; 5 episodes × 20 transitions; 0 duplicates; 0 missing
- Targets and applied actions identical across A/B1 (max diff 0)
- Evidence: artifacts/g31/pilot_predictions.jsonl + artifacts/g31/staging_b1_predictions.jsonl
- Note: Pairing verified by execution; protocol approval of pairing method is missing.

---

## 4. PILOT INTERPRETATION (ALREADY DOCUMENTED — PRESERVE)

- Status: Exploratory / descriptive only
- Evidence: All reports (pilot_report.json, staging_b1_report.json) say "pilot_scoring: EXPLORATORY; no confirm run"; plan.md "G3.1 READY FOR PILOT — NOT EXPERIMENTALLY VALIDATED"
- Not confirmatory: No Confirm dataset used; no Confirm-derived metrics; no statistical significance claimed
- Per-episode theta RMSE: mixed (ep2001/2003 B1 better; 2002/2004/2005 B1 worse); aggregate +2.40% (0.0328)
- Per-episode velocity RMSE: mixed (smaller magnitude +0.40%); not uniform
- Sample: 5 episodes, 20 transitions/episode — too small for significance
- No quality verdict: No threshold defined; no scientific justification drafted
- Note: Pilot observations must not be treated as confirmatory evidence.

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
- Note: Implementation is consistent; protocol approval of exclusion/abort/rollback rules is missing.

---

## 6. REPRODUCIBILITY (ALREADY IMPLEMENTED — PROPOSED FOR APPROVAL)

- Checkpoint hashes: A=0fd3fbc...; B1=136017b... (runtime computed, not hardcoded)
- Dataset hash: 26d63b1e... (verified)
- Evaluator source: soul/g31/evaluator.py (line 32-39 physical_metrics; line 83-94 aggregate_episode_metrics; line 96-108 paired_differences; line 118-135 evaluate_one_step)
- Code revision: f9e71bd (interface fix); evaluator unchanged since audit
- Evidence: artifacts/g31/g31_protocol_decision_register.md; score_pilot.py runtime hash; git log
- Unresolved: Evaluator version tracking (no __version__ string currently)
- Note: Reproducibility is supported by exact source identification and runtime hash computation; version tracking is a minor gap.

---

## 7. STATISTICAL ANALYSIS (MISSING — BLOCKING)

- Status: No protocol reference; no test artifacts; no significance claim made
- Evidence: None (plan.md has no statistical-analysis section)
- Unresolved: Permitted? If yes: paired t-test? bootstrap? alpha? Multiple comparison correction?
- Trade-off: None (simple) vs paired t-test (requires normality assumption) vs bootstrap (non-parametric, more robust for n=5)
- Note: With n=5 episodes, statistical power is low regardless of test; any significance claim requires justification independent of observed pilot outcome
- Recommendation (not selected): Document whether statistical analysis is permitted; if permitted, specify test and alpha; if prohibited, document prohibition
- Critical: Do not invent significance levels or claim statistical significance from this pilot.

---

## 8. ACCEPTANCE CRITERION (MISSING — BLOCKING)

- Status: No numeric delta; no scientific justification; all reports say "threshold: TO_LOCK"
- Evidence: artifacts/g31/pilot_metrics.json; artifacts/g31/staging_b1_metrics.json; artifacts/g31/pilot_report.json; artifacts/g31/staging_b1_report.json; plan.md line 50
- Unresolved: Numeric delta (e.g., B1 theta RMSE ≤ A - X%); direction (lower RMSE = improvement?); scientific rationale (domain-based, not data-based)
- Critical: Must be independent of observed pilot outcome (1.3707 vs 1.4036). Cannot derive threshold from this pilot.
- Trade-off: Absolute delta (e.g., -0.05 RMSE) vs relative % (e.g., -5%) vs effect size
- Recommendation (not selected): Owner must define criterion with scientific justification before lock
- Note: No threshold invented; no scientific justification drafted; no option selected.

---

## 9. STOP / ABORT / ROLLBACK (MISSING — BLOCKING)

- Status: No conditions in score_pilot, evaluator, or plan.md
- Evidence: None
- Unresolved: Stop if Confirm denied? Abort if >X% non-finite? Rollback if artifact corruption?
- Trade-off: Strict (abort on any deviation) vs lenient (allow minor deviations)
- Recommendation (not selected): Define fail-closed conditions; specify rollback trigger
- Note: No numeric limits invented; no conditions selected.

---

## 10. CONFIRM ACCESS AUTHORIZATION (LOCKED — STATE CORRECT; FUTURE AUTHORIZATION MISSING)

- Status: Confirm NOT ACCESSED (verified); must remain closed until lock
- Evidence: artifacts/g31/pilot_manifest.json (confirm_accessed=false); plan.md "Confirm: NOT RUN"; all reports
- Unresolved: When to authorize Confirm access? Required for final decision?
- Trade-off: Confirm required for final decision vs optional (pilot-only decision)
- Recommendation (not selected): Authorize Confirm only after protocol locked and acceptance criterion met; specify Confirm dataset (seeds 3001-3005) and comparison protocol
- Note: Current state is correct; future authorization requires owner decision.

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
- Note: Pilot observations must not be treated as confirmatory evidence. The observed +2.40% theta RMSE difference is descriptive only.

---

## 12. STATUS PRESERVATION

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

## END — PROPOSED PROTOCOL DRAFT v0.2 — NOT APPROVED — NOT LOCKED

Note: This draft is a proposal for owner review. It does not approve any protocol choice, invent any threshold, or claim any scientific conclusion. All proposed defaults are presented with evidence and trade-offs; none are selected. The protocol remains NOT READY TO LOCK until all blocking decisions (acceptance criterion, statistical analysis, stop/abort, Confirm authorization, non-finite policy, evaluator version) are explicitly resolved by the owner with independent scientific justification.
EOF