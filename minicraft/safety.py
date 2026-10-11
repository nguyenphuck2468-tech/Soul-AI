"""Fail-closed per-action safety filter for the MiniCraft structured state.

This implements only the currently specified hard rule: do not move into a
visible adjacent lava cell. It is not a route planner or a general OOD safety
system. In particular, it assumes the environment's published action mapping
has not changed.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .env import DIRECTIONS, LAVA, PATCH_RADIUS


ACTION_COUNT = 9
NOOP = 0


@dataclass(frozen=True)
class SafetyDecision:
    candidate_action: int
    applied_action: int
    allowed_actions: tuple[int, ...]
    blocked: bool
    reason: str | None
    mode: str
    expiry_tick: int | None = None


class MiniCraftSafetyController:
    """Apply the visible-adjacent-lava hard constraint to a candidate action."""

    def allowed_actions(self, observation: dict[str, Any]) -> tuple[int, ...]:
        patch = np.asarray(observation['patch'])
        expected = 2 * PATCH_RADIUS + 1
        if patch.shape != (expected, expected):
            raise ValueError(f'patch must have shape {(expected, expected)}')

        allowed = [NOOP]
        for action in range(1, ACTION_COUNT):
            delta = DIRECTIONS.get(action)
            if delta is None:
                # Interact/craft/eat do not move the player in this simulator.
                allowed.append(action)
                continue
            dx, dy = delta
            cell_x = PATCH_RADIUS + dx
            cell_y = PATCH_RADIUS + dy
            if int(patch[cell_y, cell_x]) != LAVA:
                allowed.append(action)
        return tuple(allowed)

    def filter(self, observation: dict[str, Any], candidate_action: int) -> SafetyDecision:
        if isinstance(candidate_action, bool) or not isinstance(
                candidate_action, (int, np.integer)):
            raise TypeError('candidate_action must be an integer from 0 to 8')
        action = int(candidate_action)
        if not 0 <= action < ACTION_COUNT:
            raise ValueError('candidate_action must be in the range 0..8')

        allowed = self.allowed_actions(observation)
        blocked = action not in allowed
        return SafetyDecision(
            candidate_action=action,
            applied_action=NOOP if blocked else action,
            allowed_actions=allowed,
            blocked=blocked,
            reason='visible_adjacent_lava' if blocked else None,
            mode='hard_rule' if blocked else 'normal',
        )
