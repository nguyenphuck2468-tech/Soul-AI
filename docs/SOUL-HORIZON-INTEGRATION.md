# SOUL AI ↔ Horizon integration contract (proposal-only)

**Status:** design contract; no live Horizon actuation is enabled by this document.  
**Scope:** connect the current MiniCraft observation-only skills to Horizon through the existing MCP control surface without treating the MiniCraft action space as Minecraft controls.

## What the supplied test report establishes

The supplied `Soul_AI_Complete_Package.zip` report describes four artifact groups: G3.1 Pendulum weights, Phase 0 MiniCraft RSSM outputs, 12-level distilled task trees, and merged compressed CPU skills. It reports 104/104 pre-training tests and the following 30-map merged-skill outcomes:

| Task | Reported task success | Reported learned agreement |
| --- | ---: | ---: |
| `T_wood` | 100.0% | 99.2% |
| `T_tools` | 100.0% | 95.1% |
| `T_iron` | 93.3% | 87.3% |

These are user-supplied report values, not independently reproduced by this repository revision. The ZIP itself and its SHA-256 manifest must be inspected before the artifacts are treated as release inputs. In particular, task success and learned-action agreement are different metrics and must never be substituted for one another.

The checked-in `minicraft/compressed_skill.py` currently describes the learned tree as advisory: it proposes the learned action only when it agrees with the observation-only mapped planner, and otherwise uses the mapped fallback. Its confidence labels explicitly remain uncalibrated. Do not remove that boundary based only on the summary report.

## Compatibility boundary

The current MiniCraft skill contract is not a Minecraft control contract:

- The feature encoder consumes a 7×7 MiniCraft patch, a fixed inventory schema, up to three relative mob observations, and event fields.
- The tree emits one discrete action in the MiniCraft action vocabulary (0–8).
- The mapped fallback reasons over MiniCraft tile constants, a bounded 20×20 map, and MiniCraft task goals.
- Horizon exposes a separate game-control/MCP surface and a 3D world with different blocks, inventories, entities, coordinates, timing, and action semantics.

Therefore **never pass Horizon observations directly into `observation_features()`, and never forward a MiniCraft action integer to Horizon as a key or command**. A dedicated adapter must translate from Horizon observations to a versioned SOUL observation schema and from SOUL proposals to a constrained Horizon action proposal.

## Proposed bridge: read → propose → validate → execute

1. **Read-only discovery:** connect to the user's already configured `horizon-launcher mcp` server (or its `game_*` tools) and discover protocol/tool capabilities. Do not launch a game or change account/settings state as part of discovery.
2. **Observation adapter:** construct a new, explicit `horizon.minecraft.observation.v1` schema. Include only player-visible state, bounded nearby blocks/entities, inventory, health/food, and recent public events. Include world/session identity, tick or sequence, coordinate frame, and observation timestamp. Reject missing, stale, non-finite, oversized, or unknown-schema fields.
3. **Task adapter:** define Minecraft-specific task IDs and goals separately from `T_wood`, `T_tools`, and `T_iron`. MiniCraft task trees are not reusable until a mapping is trained and validated against this new feature schema.
4. **Proposal-only inference:** return a structured proposal with model/skill ID, schema versions, observation sequence, candidate action, expected outcome, leaf support, safety status, and provenance. The first integration must not execute proposals.
5. **Independent safety gate:** re-check the latest Horizon observation immediately before any future action. Reject if the session/world changed, observation is stale, the action is unsupported, a hazard is nearby, or the proposal violates the task allowlist. A SOUL proposal must never override Horizon's own hard safety checks or user input.
6. **Explicit opt-in actuation:** only after Minecraft-specific validation and a separate user opt-in may the bridge submit a single bounded action. Stop on user input, death, disconnect, world transition, timeout, repeated no-progress, or safety rejection.
7. **Audit trail:** record hashes of the exact skill artifact and adapter, schema versions, observation sequence, proposed/applied action, safety decision, and outcome. Never log credentials or tokens.

## Proposal envelope (v1 sketch)

```json
{
  "schema": "soul.horizon.proposal.v1",
  "mode": "proposal_only",
  "skill": {
    "id": "soul.minecraft.<task>",
    "version": "<artifact-version>",
    "sha256": "<verified-artifact-sha256>"
  },
  "observation": {
    "schema": "horizon.minecraft.observation.v1",
    "session_id": "<opaque-session-id>",
    "sequence": 0
  },
  "proposal": {
    "action": "<minecraft-specific-action>",
    "expected_outcome": {},
    "leaf_support": 0,
    "confidence": "uncalibrated"
  },
  "safety": {
    "decision": "proposal_only",
    "reasons": ["execution_not_enabled"]
  }
}
```

The envelope is illustrative, not a committed API guarantee. Replace placeholders with validated enums and numeric bounds before implementation. Never accept a client-supplied `safety.decision` as authority; the Horizon-side gate computes that decision itself.

## Promotion gates

A Minecraft skill is not eligible for execution until all of the following are true:

- The complete ZIP is available and every artifact is verified against a manifest of SHA-256 hashes.
- The skill's observation feature names/order, action vocabulary, task goal, training/evaluation seed split, and provenance are explicit and machine-validated.
- Independent held-out Minecraft maps demonstrate task success, not only agreement with another planner.
- A no-skill baseline, safe mapped baseline, and ablation are reported on the same held-out maps.
- Hazard tests cover lava, falls, hostile mobs, inventory/tool mismatches, stale observations, world changes, disconnects, and user interruption.
- The adapter has deterministic contract tests, replay tests, fuzz/property tests for malformed observations, and end-to-end tests against a disposable Horizon world.
- Proposal-only mode is the default. Execution is opt-in, bounded, interruptible, and auditable.

## Immediate next work

1. Obtain and verify `Soul_AI_Complete_Package.zip`; do not infer artifact contents from its summary.
2. Add an artifact inspection/verification command that checks all referenced files, duplicate JSON keys, schema versions, and SHA-256 hashes without loading untrusted model weights.
3. Freeze the MiniCraft observation/action schemas and add a separate Minecraft observation/action adapter.
4. Build a read-only MCP discovery prototype; do not enable action execution yet.
