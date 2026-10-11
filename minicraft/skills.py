"""Alpha skill contract for task-level MiniCraft expert skills.

The current implementation wraps the privileged Phase B expert. It is a
Candidate implementation for wiring and run-throughs, not a validated or
compressed skill.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .experts import MiniCraftExpert


class SkillStatus(str, Enum):
    CANDIDATE = 'Candidate'
    VALIDATED = 'Validated'
    RETIRED = 'Retired'


_GOAL_CONDITIONS = {
    'T_wood': {'field': 'inv.wood', 'operator': '>=', 'value': 5},
    'T_tools': {'field': 'inv.stone_pickaxe', 'operator': '>=', 'value': 1},
    'T_iron': {'field': 'inv.iron', 'operator': '>=', 'value': 1},
}


@dataclass(frozen=True)
class SkillProposal:
    skill_id: str
    version: str
    task: str
    candidate_action: int
    expected_outcome: dict[str, Any]
    confidence_status: str
    skill_status: str


class MiniCraftExpertSkill:
    """Wrap one existing expert behind an explicit candidate-skill contract."""

    version = '0.1.0-alpha'

    def __init__(self, task: str):
        if task not in _GOAL_CONDITIONS:
            raise ValueError(f'task must be one of {sorted(_GOAL_CONDITIONS)}')
        self.task = task
        self.skill_id = f'minicraft.expert.{task}'
        self.status = SkillStatus.CANDIDATE
        self.expert = MiniCraftExpert(task)
        self.episodes_completed = 0
        self.episodes_succeeded = 0

    def propose(self, observation: dict[str, Any], env_info: Any) -> SkillProposal:
        """Propose one action and expose the episode goal as its expectation.

        The expectation is a goal condition, not a calibrated probability or
        one-step state prediction. The action must still pass through the
        caller's SafetyController before execution.
        """
        if self.status is SkillStatus.RETIRED:
            raise RuntimeError(f'skill {self.skill_id} is retired')
        action = self.expert.expert_action(observation, env_info)
        return SkillProposal(
            skill_id=self.skill_id,
            version=self.version,
            task=self.task,
            candidate_action=action,
            expected_outcome={
                'scope': 'episode_goal',
                'condition': dict(_GOAL_CONDITIONS[self.task]),
            },
            confidence_status='NOT_CALIBRATED',
            skill_status=self.status.value,
        )

    def record_episode_result(self, task_success: bool) -> None:
        """Record one completed episode outcome for a descriptive success rate."""
        if not isinstance(task_success, bool):
            raise TypeError('task_success must be boolean')
        self.episodes_completed += 1
        self.episodes_succeeded += int(task_success)

    @property
    def empirical_success_rate(self) -> float | None:
        if not self.episodes_completed:
            return None
        return self.episodes_succeeded / self.episodes_completed

    def retire(self) -> None:
        self.status = SkillStatus.RETIRED

    def card(self) -> dict[str, Any]:
        """Return the skill's context, expectation, evidence, and known limits."""
        return {
            'skill_id': self.skill_id,
            'version': self.version,
            'status': self.status.value,
            'implementation': 'privileged_full_state_expert_wrapper',
            'context': {
                'environment': 'MiniCraftEnv',
                'task': self.task,
                'requires_full_state': True,
            },
            'expected_outcome': {
                'scope': 'episode_goal',
                'condition': dict(_GOAL_CONDITIONS[self.task]),
                'confidence_status': 'NOT_CALIBRATED',
            },
            'episodes_completed': self.episodes_completed,
            'episodes_succeeded': self.episodes_succeeded,
            'empirical_success_rate': self.empirical_success_rate,
            'known_risks': [
                'Uses privileged full-map state unavailable to the learned agent.',
                'Expert replans with BFS each action; no compute advantage is established.',
                'OOD robustness and held-out generalization are not validated.',
            ],
            'failure_modes': [
                'No reachable resource currently yields a noop while the episode continues.',
                'Safety filtering may replace a proposed action with noop.',
                'Hidden action-mapping shocks can invalidate expert movement assumptions.',
                'Survival failure or episode timeout can prevent goal completion.',
            ],
            'validation_note': 'Candidate status only; no promotion threshold is locked.',
        }
