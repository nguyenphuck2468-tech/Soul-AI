"""Alpha compressed skill with an observation-only agreement gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .distilled_skill import DistilledObservationSkill
from .mapped_skill import MiniCraftMappedSkill


@dataclass(frozen=True)
class CompressedSkillProposal:
    skill_id: str
    version: str
    task: str
    candidate_action: int
    expected_outcome: dict[str, Any]
    confidence_status: str
    skill_status: str
    source: str
    learned_action: int
    fallback_action: int
    leaf_samples: int


class CompressedObservationSkill:
    """Combine a learned action prior with a safe observation-only fallback.

    The learned tree is advisory in this alpha slice. It can control only when
    it agrees with the mapped planner; disagreement stays on the existing
    observation-only route and is recorded in the proposal.
    """

    version = '0.1.0-alpha-compressed'

    def __init__(self, task: str, artifact: dict[str, Any]):
        self.task = task
        self.learned = DistilledObservationSkill(task, artifact)
        self.fallback = MiniCraftMappedSkill(task)
        self.status = 'Candidate'

    def propose(self, observation: dict[str, Any]) -> CompressedSkillProposal:
        learned = self.learned.propose(observation)
        fallback = self.fallback.propose(observation)
        agrees = learned.candidate_action == fallback.candidate_action
        return CompressedSkillProposal(
            skill_id=f'minicraft.compressed.{self.task}',
            version=self.version,
            task=self.task,
            candidate_action=(
                learned.candidate_action if agrees else fallback.candidate_action),
            expected_outcome=fallback.expected_outcome,
            confidence_status=(
                'UNCALIBRATED_LEARNED_AGREEMENT'
                if agrees else 'UNCALIBRATED_FALLBACK_DISAGREEMENT'),
            skill_status=self.status,
            source='learned_agreement' if agrees else 'mapped_fallback',
            learned_action=learned.candidate_action,
            fallback_action=fallback.candidate_action,
            leaf_samples=learned.leaf_samples,
        )

    def observe_outcome(self, observation: dict[str, Any], *,
                        applied_action: int | None = None) -> dict[str, Any]:
        return self.fallback.observe_outcome(
            observation, applied_action=applied_action)
