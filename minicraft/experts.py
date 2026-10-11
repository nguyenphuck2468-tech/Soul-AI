"""Deterministic hand-written experts for MiniCraft benchmark tasks.

The experts are intentionally privileged: they plan with the full environment
snapshot, as specified for the Phase B feasibility baseline.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .env import DIRECTIONS, GRASS, IRON_ORE, LAVA, STONE, TREE


_MOVE_FOR_DELTA = {delta: action for action, delta in DIRECTIONS.items()}
_TASKS = {"T_wood", "T_tools", "T_iron"}


class MiniCraftExpert:
    """A deterministic, full-map BFS expert for one MiniCraft task."""

    def __init__(self, task: str):
        if task not in _TASKS:
            raise ValueError(f"task must be one of {sorted(_TASKS)}")
        self.task = task

    def expert_action(self, state_obs: dict[str, Any], env_info: Any) -> int:
        """Return one action using an observation and full state/environment.

        ``env_info`` may be a ``MiniCraftEnv``-like object exposing
        ``get_state()``, or a previously captured state snapshot. The snapshot
        is used only for map and threat information; inventory and position
        come from ``state_obs`` to keep the interface explicit.
        """
        state = env_info.get_state() if hasattr(env_info, "get_state") else env_info
        if not isinstance(state, dict) or "grid" not in state:
            raise TypeError("env_info must be an environment or full state snapshot")

        actor = state_obs["self"]
        inventory = state_obs["inv"]
        x, y = int(actor["x"]), int(actor["y"])
        grid = np.asarray(state["grid"])

        if inventory.get("apple", 0) and actor["food"] <= 4:
            return 8

        target_tile: int | None = None
        if self.task == "T_wood":
            if inventory["wood"] >= 5:
                return 0
            target_tile = TREE
        else:
            if inventory["stone_pickaxe"]:
                if self.task == "T_tools":
                    return 0
                if inventory["iron"]:
                    return 0
                target_tile = IRON_ORE
            elif inventory["wood_pickaxe"]:
                if inventory["stone"] >= 2 and inventory["wood"] >= 1:
                    return 7
                if inventory["stone"] < 2:
                    target_tile = STONE
                else:
                    target_tile = TREE
            elif inventory["wood"] >= 2:
                return 6
            else:
                target_tile = TREE

        assert target_tile is not None
        mob_positions = {
            (int(mob["x"]), int(mob["y"])) for mob in state.get("zombies", ())
        }
        destination = self._nearest_interaction_position(
            grid=grid,
            start=(x, y),
            target_tile=target_tile,
            mob_positions=mob_positions,
        )

        # Combat is a last resort when a nearby zombie blocks every route.
        if destination is None:
            # Wait for a moving zombie to clear rather than step onto its tile.
            return 0

        target, path = destination
        if path:
            return _MOVE_FOR_DELTA[(path[0][0] - x, path[0][1] - y)]

        dx, dy = target[0] - x, target[1] - y
        if (dx, dy) not in _MOVE_FOR_DELTA:
            return 0
        desired_facing = {1: 0, 2: 1, 3: 2, 4: 3}[_MOVE_FOR_DELTA[(dx, dy)]]
        if int(actor["facing"]) != desired_facing:
            # Moving into the resource is blocked, but sets facing in MiniCraft.
            return _MOVE_FOR_DELTA[(dx, dy)]
        return 5

    @staticmethod
    def _distance(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _nearest_interaction_position(
        grid: np.ndarray,
        start: tuple[int, int],
        target_tile: int,
        mob_positions: set[tuple[int, int]],
    ) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
        """BFS over grass, selecting the nearest safely reachable resource."""
        size_y, size_x = grid.shape
        directions = tuple(DIRECTIONS.values())
        queue = deque([start])
        previous: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        distance = {start: 0}
        goals: list[tuple[tuple[int, int], tuple[int, int]]] = []
        best_distance: int | None = None

        while queue:
            current = queue.popleft()
            current_distance = distance[current]
            if best_distance is not None and current_distance > best_distance:
                break
            cx, cy = current
            for dx, dy in directions:
                tx, ty = cx + dx, cy + dy
                if not (0 <= tx < size_x and 0 <= ty < size_y):
                    continue
                if int(grid[ty, tx]) == target_tile:
                    goals.append((current, (tx, ty)))
                    best_distance = current_distance
            if best_distance is not None:
                continue
            for dx, dy in directions:
                nxt = (cx + dx, cy + dy)
                nx, ny = nxt
                if not (0 <= nx < size_x and 0 <= ny < size_y):
                    continue
                if nxt in previous or int(grid[ny, nx]) != GRASS:
                    continue
                if int(grid[ny, nx]) == LAVA:
                    continue
                if nxt in mob_positions or any(
                    MiniCraftExpert._distance(nxt, mob) == 1 for mob in mob_positions
                ):
                    continue
                previous[nxt] = current
                distance[nxt] = current_distance + 1
                queue.append(nxt)

        if not goals:
            return None
        stand, target = min(
            goals,
            key=lambda pair: (
                distance[pair[0]],
                pair[1][1], pair[1][0], pair[0][1], pair[0][0],
            ),
        )
        path: list[tuple[int, int]] = []
        cursor = stand
        while cursor != start:
            path.append(cursor)
            parent = previous[cursor]
            assert parent is not None
            cursor = parent
        path.reverse()
        return target, path

def expert_action(
    state_obs: dict[str, Any], env_info: Any, task: str = "T_iron",
) -> int:
    """Convenience function matching the Phase B expert interface."""
    return MiniCraftExpert(task).expert_action(state_obs, env_info)


def expert_wood_action(state_obs: dict[str, Any], env_info: Any) -> int:
    return MiniCraftExpert("T_wood").expert_action(state_obs, env_info)


def expert_tools_action(state_obs: dict[str, Any], env_info: Any) -> int:
    return MiniCraftExpert("T_tools").expert_action(state_obs, env_info)


def expert_iron_action(state_obs: dict[str, Any], env_info: Any) -> int:
    return MiniCraftExpert("T_iron").expert_action(state_obs, env_info)

