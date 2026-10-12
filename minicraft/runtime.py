"""Reusable runtime loop for the observation-only compressed skill alpha."""

from __future__ import annotations

import time
from typing import Any

from .compressed_skill import CompressedObservationSkill
from .env import MiniCraftEnv
from .safety import MiniCraftSafetyController


class CompressedSkillRuntime:
    """Run one compressed-skill episode with explicit runtime accounting."""

    def __init__(self, task: str, artifact: dict[str, Any], *,
                 max_ticks: int = 300):
        if max_ticks < 1:
            raise ValueError("max_ticks must be positive")
        self.task = task
        self.artifact = artifact
        self.max_ticks = max_ticks

    def run_episode(self, seed: int) -> dict[str, Any]:
        env = MiniCraftEnv(seed=seed, max_ticks=self.max_ticks)
        skill = CompressedObservationSkill(self.task, self.artifact)
        safety = MiniCraftSafetyController()
        observation = env.observe()
        learned_agreements = 0
        fallbacks = 0
        safety_blocks = 0
        expectation_checks = 0
        expectation_mismatches = 0
        steps = 0
        proposal_seconds = 0.0
        safety_seconds = 0.0
        environment_seconds = 0.0
        monitor_seconds = 0.0
        while not env.done:
            started = time.perf_counter()
            proposal = skill.propose(observation)
            proposal_seconds += time.perf_counter() - started
            learned_agreements += int(proposal.source == "learned_agreement")
            fallbacks += int(proposal.source == "mapped_fallback")

            started = time.perf_counter()
            decision = safety.filter(observation, proposal.candidate_action)
            safety_seconds += time.perf_counter() - started
            safety_blocks += int(decision.blocked)

            started = time.perf_counter()
            observation, _, _, info = env.step(decision.applied_action)
            environment_seconds += time.perf_counter() - started

            started = time.perf_counter()
            monitor = skill.observe_outcome(
                observation, applied_action=decision.applied_action)
            monitor_seconds += time.perf_counter() - started
            expectation_checks += int(monitor["checked"])
            expectation_mismatches += len(monitor["mismatched_fields"])
            steps += 1

        runtime_seconds = {
            "skill_proposal": proposal_seconds,
            "safety_filter": safety_seconds,
            "environment_step": environment_seconds,
            "expectation_monitor": monitor_seconds,
        }
        runtime_seconds["integrated"] = sum(runtime_seconds.values())
        return {
            "task": self.task,
            "seed": seed,
            "success": bool(info["task_success"][self.task]),
            "steps": steps,
            "learned_agreement_steps": learned_agreements,
            "mapped_fallback_steps": fallbacks,
            "safety_blocks": safety_blocks,
            "expectation_checks": expectation_checks,
            "expectation_mismatches": expectation_mismatches,
            "runtime_seconds": runtime_seconds,
        }
