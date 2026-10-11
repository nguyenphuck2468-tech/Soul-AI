"""Observation-only online mapping skill for MiniCraft task goals."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from .env import DIRECTIONS, GRASS, IRON_ORE, LAVA, PATCH_RADIUS, STONE, TREE, WATER


_MOVE_FOR_DELTA = {delta: action for action, delta in DIRECTIONS.items()}
_GOALS = {
    'T_wood': {'field': 'inv.wood', 'operator': '>=', 'value': 5},
    'T_tools': {'field': 'inv.stone_pickaxe', 'operator': '>=', 'value': 1},
    'T_iron': {'field': 'inv.iron', 'operator': '>=', 'value': 1},
}
_CARDINALS = tuple(DIRECTIONS.values())


@dataclass(frozen=True)
class MappedSkillProposal:
    skill_id: str
    version: str
    task: str
    candidate_action: int
    expected_outcome: dict[str, Any]
    confidence_status: str
    skill_status: str
    map_coverage: float
    selected_subgoal: str


class MiniCraftMappedSkill:
    """Build a persistent map from local observations and plan on known cells.

    No environment object or hidden full-state data is accepted by this API.
    Unknown terrain is explored by routing to the nearest reachable frontier.
    """

    version = '0.1.0-alpha-mapped'

    def __init__(self, task: str, world_size: int = 20):
        if task not in _GOALS:
            raise ValueError(f'task must be one of {sorted(_GOALS)}')
        self.task = task
        self.world_size = int(world_size)
        self.map = np.full((self.world_size, self.world_size), -1, dtype=np.int8)
        self.status = 'Candidate'
        self._cached_actions: list[int] = []
        self._cached_signature = None
        self._cached_subgoal = 'initialize'
        self._pending_expected: dict[str, Any] | None = None
        self._pending_action: int | None = None
        self.expectation_checks = 0
        self.expectation_mismatches = 0

    def _update_map(self, observation: dict[str, Any]) -> None:
        x, y = int(observation['self']['x']), int(observation['self']['y'])
        patch = np.asarray(observation['patch'])
        expected = 2 * PATCH_RADIUS + 1
        if patch.shape != (expected, expected):
            raise ValueError(f'patch must have shape {(expected, expected)}')
        for py in range(expected):
            for px in range(expected):
                gx = x + px - PATCH_RADIUS
                gy = y + py - PATCH_RADIUS
                if 0 <= gx < self.world_size and 0 <= gy < self.world_size:
                    tile = int(patch[py, px])
                    if tile != 6:
                        self.map[gy, gx] = tile

    def _reachable(self, start: tuple[int, int], mobs: set[tuple[int, int]]):
        queue = deque([start])
        previous = {start: None}
        distance = {start: 0}
        while queue:
            current = queue.popleft()
            x, y = current
            for dx, dy in _CARDINALS:
                nxt = (x + dx, y + dy)
                nx, ny = nxt
                if not (0 <= nx < self.world_size and 0 <= ny < self.world_size):
                    continue
                if nxt in previous or int(self.map[ny, nx]) != GRASS:
                    continue
                if nxt in mobs or any(abs(nx - mx) + abs(ny - my) == 1
                                      for mx, my in mobs):
                    continue
                previous[nxt] = current
                distance[nxt] = distance[current] + 1
                queue.append(nxt)
        return previous, distance

    @staticmethod
    def _path_to(previous: dict, start: tuple[int, int], end: tuple[int, int]):
        path = []
        cursor = end
        while cursor != start:
            parent = previous.get(cursor)
            if parent is None:
                return None
            path.append(cursor)
            cursor = parent
        path.reverse()
        return path

    def _frontier(self, previous: dict, distance: dict):
        candidates = []
        for (x, y), dist in distance.items():
            if any(
                0 <= x + dx < self.world_size
                and 0 <= y + dy < self.world_size
                and self.map[y + dy, x + dx] == -1
                for dx, dy in _CARDINALS
            ):
                candidates.append((dist, y, x))
        if not candidates:
            return None
        _, y, x = min(candidates)
        return (x, y)

    def _choose_resource(self, target_tile: int, start: tuple[int, int],
                         previous: dict, distance: dict):
        candidates = []
        for (x, y), dist in distance.items():
            for dx, dy in _CARDINALS:
                tx, ty = x + dx, y + dy
                if (0 <= tx < self.world_size and 0 <= ty < self.world_size
                        and int(self.map[ty, tx]) == target_tile):
                    candidates.append((dist, ty, tx, y, x))
        if not candidates:
            return None
        _, ty, tx, y, x = min(candidates)
        return (x, y), (tx, ty), self._path_to(previous, start, (x, y))

    def _target(self, inventory: dict[str, int]):
        if self.task == 'T_wood':
            return (None, 'goal') if inventory['wood'] >= 5 else (TREE, 'gather_tree')
        if not inventory['wood_pickaxe']:
            if inventory['wood'] >= 2:
                return None, 'craft_wood_pickaxe'
            return TREE, 'gather_tree'
        if self.task == 'T_tools' and inventory['stone_pickaxe']:
            return None, 'goal'
        if self.task == 'T_iron' and inventory['stone_pickaxe']:
            if inventory['iron']:
                return None, 'goal'
            return IRON_ORE, 'gather_iron'
        if inventory['stone'] >= 2 and inventory['wood'] >= 1:
            return None, 'craft_stone_pickaxe'
        if inventory['stone'] < 2:
            return STONE, 'gather_stone'
        return TREE, 'gather_tree'

    def _action_plan(self, observation: dict[str, Any], tile: int,
                     start: tuple[int, int], mobs: set[tuple[int, int]]):
        previous, distance = self._reachable(start, mobs)
        target = self._choose_resource(tile, start, previous, distance)
        if target is not None:
            stand, resource, path = target
            actions = []
            for current, nxt in zip([start, *(path or [])], path or []):
                actions.append(_MOVE_FOR_DELTA[(nxt[0] - current[0], nxt[1] - current[1])])
            if path:
                prior = path[-2] if len(path) > 1 else start
                dx, dy = path[-1][0] - prior[0], path[-1][1] - prior[1]
                facing = {1: 0, 2: 1, 3: 2, 4: 3}[
                    _MOVE_FOR_DELTA[(dx, dy)]]
            else:
                facing = int(observation['self']['facing'])
            dx, dy = resource[0] - stand[0], resource[1] - stand[1]
            face_action = _MOVE_FOR_DELTA[(dx, dy)]
            desired_facing = {1: 0, 2: 1, 3: 2, 4: 3}[face_action]
            if facing != desired_facing:
                actions.append(face_action)
            actions.append(5)
            return actions

        frontier = self._frontier(previous, distance)
        if frontier is None or frontier == start:
            return []
        path = self._path_to(previous, start, frontier)
        actions = []
        for current, nxt in zip([start, *(path or [])], path or []):
            actions.append(_MOVE_FOR_DELTA[(nxt[0] - current[0], nxt[1] - current[1])])
        return actions

    def _cached_action_is_valid(self, observation: dict[str, Any],
                                mobs: set[tuple[int, int]]) -> bool:
        if not self._cached_actions:
            return False
        events = observation.get('events', {})
        if events.get('last_result') == 'blocked' or events.get('damage', 0) > 0:
            return False
        action = self._cached_actions[0]
        if action in DIRECTIONS:
            x, y = int(observation['self']['x']), int(observation['self']['y'])
            dx, dy = DIRECTIONS[action]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.world_size and 0 <= ny < self.world_size):
                return False
            if int(self.map[ny, nx]) != GRASS:
                return False
            if (nx, ny) in mobs or any(abs(nx - mx) + abs(ny - my) == 1
                                       for mx, my in mobs):
                return False
        return True

    def _expected_transition(self, observation: dict[str, Any], action: int,
                             subgoal: str) -> dict[str, Any]:
        actor = observation['self']
        inventory = observation['inv']
        expected: dict[str, Any] = {}
        if action in DIRECTIONS:
            dx, dy = DIRECTIONS[action]
            x, y = int(actor['x']), int(actor['y'])
            nx, ny = x + dx, y + dy
            tile = (int(self.map[ny, nx]) if 0 <= nx < self.world_size
                    and 0 <= ny < self.world_size else None)
            if tile == GRASS:
                expected['self.x'] = nx
                expected['self.y'] = ny
            else:
                expected['self.x'] = x
                expected['self.y'] = y
            expected['self.facing'] = {1: 0, 2: 1, 3: 2, 4: 3}[action]
        elif action == 5:
            key = {
                'gather_tree': 'wood', 'explore_for_gather_tree': 'wood',
                'gather_stone': 'stone', 'explore_for_gather_stone': 'stone',
                'gather_iron': 'iron', 'explore_for_gather_iron': 'iron',
            }.get(subgoal)
            if key is not None:
                expected[f'inv.{key}'] = int(inventory[key]) + 1
        elif action == 6:
            expected['inv.wood'] = int(inventory['wood']) - 2
            expected['inv.wood_pickaxe'] = int(inventory['wood_pickaxe']) + 1
        elif action == 7:
            expected['inv.wood'] = int(inventory['wood']) - 1
            expected['inv.stone'] = int(inventory['stone']) - 2
            expected['inv.stone_pickaxe'] = int(inventory['stone_pickaxe']) + 1
        elif action == 8:
            expected['self.food'] = min(10, int(actor['food']) + 4)
        return {
            'horizon': 1,
            'expected_fields': expected,
            'episode_goal': dict(_GOALS[self.task]),
        }

    def propose(self, observation: dict[str, Any]) -> MappedSkillProposal:
        self._update_map(observation)
        actor = observation['self']
        inventory = observation['inv']
        start = (int(actor['x']), int(actor['y']))
        visible_mobs = {
            (start[0] + int(mob['dx']), start[1] + int(mob['dy']))
            for mob in observation.get('mobs', ())
        }
        target_tile, subgoal = self._target(inventory)
        signature = (
            tuple(int(inventory[key]) for key in (
                'wood', 'stone', 'iron', 'apple', 'wood_pickaxe', 'stone_pickaxe')),
            tuple(sorted(visible_mobs)),
            subgoal,
        )
        if inventory.get('apple', 0) and actor['food'] <= 4:
            action, subgoal = 8, 'eat'
            self._cached_actions = []
        elif target_tile is None and subgoal != 'goal':
            action = {'craft_wood_pickaxe': 6, 'craft_stone_pickaxe': 7}[subgoal]
            self._cached_actions = []
        elif self._cached_signature == signature and self._cached_action_is_valid(
                observation, visible_mobs):
            action = self._cached_actions.pop(0)
            subgoal = self._cached_subgoal
        else:
            if target_tile is None:
                action = 0
                self._cached_actions = []
            else:
                self._cached_actions = self._action_plan(
                    observation, target_tile, start, visible_mobs)
                self._cached_signature = signature
                self._cached_subgoal = subgoal
                if not self._cached_actions:
                    action = 0
                else:
                    action = self._cached_actions.pop(0)
                if not any(int(self.map[y, x]) == target_tile
                           for y in range(self.world_size)
                           for x in range(self.world_size)):
                    subgoal = f'explore_for_{subgoal}'
                    self._cached_subgoal = subgoal

        expected_outcome = self._expected_transition(observation, int(action), subgoal)
        self._pending_expected = expected_outcome
        self._pending_action = int(action)
        return MappedSkillProposal(
            skill_id=f'minicraft.mapped.{self.task}',
            version=self.version,
            task=self.task,
            candidate_action=int(action),
            expected_outcome=expected_outcome,
            confidence_status='HAND_CODED_UNCALIBRATED',
            skill_status=self.status,
            map_coverage=float(np.mean(self.map >= 0)),
            selected_subgoal=subgoal,
        )

    def observe_outcome(self, observation: dict[str, Any], *,
                        applied_action: int | None = None) -> dict[str, Any]:
        """Compare the last one-step expectation with reality and invalidate plans.

        This is a cheap, hand-coded expectation check. It is not an OOD or
        uncertainty detector and does not invoke a world model.
        """
        expected = self._pending_expected
        proposed = self._pending_action
        self._pending_expected = None
        self._pending_action = None
        if expected is None or not expected['expected_fields']:
            return {'checked': False, 'matched': None, 'mismatched_fields': []}

        mismatches = []
        for dotted_path, expected_value in expected['expected_fields'].items():
            actual = observation
            for key in dotted_path.split('.'):
                actual = actual[key]
            if actual != expected_value:
                mismatches.append({
                    'field': dotted_path,
                    'expected': expected_value,
                    'actual': actual,
                })
        if applied_action is not None and proposed != applied_action:
            mismatches.append({
                'field': 'applied_action',
                'expected': proposed,
                'actual': applied_action,
            })
        matched = not mismatches
        self.expectation_checks += 1
        self.expectation_mismatches += int(not matched)
        if not matched:
            self._cached_actions.clear()
            self._cached_signature = None
        return {
            'checked': True,
            'matched': matched,
            'mismatched_fields': mismatches,
            'cached_plan_invalidated': not matched,
        }
