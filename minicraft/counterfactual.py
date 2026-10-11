"""State-matched one-step action branches for MiniCraft."""

from dataclasses import dataclass
from typing import Any, Sequence

from .env import MiniCraftEnv
from .safety import MiniCraftSafetyController, SafetyDecision


@dataclass(frozen=True)
class CounterfactualOutcome:
    candidate_action: int
    applied_action: int
    next_observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]
    safety: SafetyDecision


def branch_actions(
        env: MiniCraftEnv,
        candidate_actions: Sequence[int],
        safety: MiniCraftSafetyController | None = None,
) -> tuple[CounterfactualOutcome, ...]:
    """Step independent clones from one exact state, filtering each action.

    The source environment is never stepped or otherwise mutated. Each clone
    begins with the same simulator and RNG snapshot; differences after the
    first action are consequences of the action branches.
    """
    if env.done:
        raise RuntimeError('cannot branch a completed episode')
    controller = safety or MiniCraftSafetyController()
    observation = env.observe()
    outcomes = []
    for candidate in candidate_actions:
        decision = controller.filter(observation, candidate)
        branch = env.clone()
        next_observation, reward, done, info = branch.step(decision.applied_action)
        outcomes.append(CounterfactualOutcome(
            candidate_action=decision.candidate_action,
            applied_action=decision.applied_action,
            next_observation=next_observation,
            reward=float(reward),
            done=bool(done),
            info=info,
            safety=decision,
        ))
    return tuple(outcomes)
