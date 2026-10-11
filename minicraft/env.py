"""Small, deterministic, structured survival world for the Soul project."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from typing import Any

import numpy as np


GRASS, TREE, STONE, WATER, LAVA, IRON_ORE = range(6)
WORLD_SIZE = 20
PATCH_RADIUS = 3
MAX_HEALTH = 10
MAX_FOOD = 10
MAX_ZOMBIES = 3

# Facing follows the project plan: north, south, west, east.
DIRECTIONS = {
    1: (0, -1),
    2: (0, 1),
    3: (-1, 0),
    4: (1, 0),
}
FACING_DIRECTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
ACTION_TO_FACING = {1: 0, 2: 1, 3: 2, 4: 3}

ACTION_NAMES = {
    0: "noop",
    1: "north",
    2: "south",
    3: "west",
    4: "east",
    5: "interact",
    6: "craft_wood_pickaxe",
    7: "craft_stone_pickaxe",
    8: "eat",
}

INVENTORY_KEYS = (
    "wood", "stone", "iron", "apple", "wood_pickaxe", "stone_pickaxe",
)


class MiniCraftEnv:
    """A single seeded episode with explicit state and reproducible randomness."""

    size = WORLD_SIZE

    def __init__(self, seed: int = 0, max_ticks: int = 300):
        if max_ticks < 1:
            raise ValueError("max_ticks must be positive")
        self.max_ticks = int(max_ticks)
        self.seed = int(seed)
        self.reset(seed=self.seed)

    def reset(self, seed: int | None = None):
        """Start a new episode and return its first observation."""
        if seed is not None:
            self.seed = int(seed)
        self.tick = 0
        self.health = MAX_HEALTH
        self.food = MAX_FOOD
        self.inventory = {key: 0 for key in INVENTORY_KEYS}
        self.facing = 0
        self.done = False
        self.death = False
        self.death_reason: str | None = None
        self._hunger_clock = 0
        self._starvation_clock = 0
        self._healing_clock = 0
        self._milestones_paid: set[str] = set()
        self._last_result = "none"
        self._last_damage = 0
        self._shocks: list[dict[str, Any]] = []
        self._generate_world()
        self._respawn_timers = {slot: 0 for slot in range(MAX_ZOMBIES)}
        self.zombies: list[dict[str, int]] = []
        for slot in range(MAX_ZOMBIES):
            self._spawn_zombie(slot)
        return self.observe()

    # -- map generation -------------------------------------------------
    def _generate_world(self) -> None:
        """Generate a map, advancing failed seeds by the specified offset."""
        for attempt in range(100):
            world_seed = self.seed + attempt * 1_000_000
            self.rng = np.random.default_rng(world_seed)
            grid = np.full((self.size, self.size), GRASS, dtype=np.int8)
            self._place_scattered(grid, TREE, round(self.size**2 * 0.12))
            self._place_scattered(grid, STONE, round(self.size**2 * 0.08))
            self._place_clusters(grid, WATER, round(self.size**2 * 0.04), 4)
            self._place_clusters(grid, LAVA, round(self.size**2 * 0.02), 2)
            if not self._place_iron_ore(grid, round(self.size**2 * 0.01)):
                continue

            lava = self._positions(grid, LAVA)
            spawn_options = [
                pos for pos in self._positions(grid, GRASS)
                if all(self._manhattan(pos, hazard) > 3 for hazard in lava)
            ]
            if not spawn_options:
                continue
            order = self.rng.permutation(len(spawn_options))
            chosen = None
            for index in order:
                candidate = spawn_options[int(index)]
                reachable = self._reachable_grass(grid, candidate)
                adjacent_resources = {
                    (nx, ny)
                    for x, y in reachable
                    for dx, dy in DIRECTIONS.values()
                    for nx, ny in ((x + dx, y + dy),)
                    if 0 <= nx < self.size and 0 <= ny < self.size
                    and grid[ny, nx] in (TREE, IRON_ORE)
                }
                accessible_trees = sum(
                    grid[y, x] == TREE for x, y in adjacent_resources
                )
                accessible_ore = any(
                    grid[y, x] == IRON_ORE for x, y in adjacent_resources
                )
                if accessible_trees >= 5 and accessible_ore:
                    chosen = candidate
                    break
            if chosen is None:
                continue

            self.grid = grid
            self.x, self.y = chosen
            self.world_seed = world_seed
            return
        raise RuntimeError("could not generate a reachable world after 100 seeds")

    def _place_scattered(self, grid: np.ndarray, tile: int, count: int) -> None:
        cells = self._positions(grid, GRASS)
        if len(cells) < count:
            raise RuntimeError("not enough free cells to place map resources")
        selected = self.rng.choice(len(cells), size=count, replace=False)
        for index in selected:
            x, y = cells[int(index)]
            grid[y, x] = tile

    def _place_clusters(
        self, grid: np.ndarray, tile: int, count: int, cluster_size: int,
    ) -> None:
        remaining = count
        while remaining:
            start_cells = self._positions(grid, GRASS)
            if not start_cells:
                raise RuntimeError("not enough free cells to place clusters")
            x, y = start_cells[int(self.rng.integers(len(start_cells)))]
            frontier = [(x, y)]
            placed = 0
            while frontier and remaining and placed < cluster_size:
                pick = int(self.rng.integers(len(frontier)))
                cx, cy = frontier.pop(pick)
                if grid[cy, cx] != GRASS:
                    continue
                grid[cy, cx] = tile
                placed += 1
                remaining -= 1
                neighbours = [
                    (cx + dx, cy + dy) for dx, dy in DIRECTIONS.values()
                    if 0 <= cx + dx < self.size and 0 <= cy + dy < self.size
                    and grid[cy + dy, cx + dx] == GRASS
                ]
                if neighbours:
                    order = self.rng.permutation(len(neighbours))
                    frontier.extend(neighbours[int(i)] for i in order)

    def _place_iron_ore(self, grid: np.ndarray, count: int) -> bool:
        candidates = []
        for x, y in self._positions(grid, GRASS):
            if any(
                0 <= x + dx < self.size and 0 <= y + dy < self.size
                and grid[y + dy, x + dx] == STONE
                for dx, dy in DIRECTIONS.values()
            ):
                candidates.append((x, y))
        if len(candidates) < count:
            return False
        order = self.rng.permutation(len(candidates))
        for index in order[:count]:
            x, y = candidates[int(index)]
            grid[y, x] = IRON_ORE
        return True

    @staticmethod
    def _positions(grid: np.ndarray, tile: int) -> list[tuple[int, int]]:
        ys, xs = np.where(grid == tile)
        return list(zip(xs.tolist(), ys.tolist()))

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _reachable_grass(
        self, grid: np.ndarray, start: tuple[int, int],
    ) -> set[tuple[int, int]]:
        pending = deque([start])
        reached = {start}
        while pending:
            x, y = pending.popleft()
            for dx, dy in DIRECTIONS.values():
                pos = (x + dx, y + dy)
                nx, ny = pos
                if (0 <= nx < self.size and 0 <= ny < self.size
                        and grid[ny, nx] == GRASS and pos not in reached):
                    reached.add(pos)
                    pending.append(pos)
        return reached

    # -- episode transitions -------------------------------------------
    def step(self, action: int):
        """Advance one tick; return ``(observation, reward, done, info)``."""
        if self.done:
            raise RuntimeError("episode is done; call reset() before stepping")
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer from 0 to 8")
        action = int(action)
        if action not in ACTION_NAMES:
            raise ValueError("action must be in the range 0..8")

        next_tick = self.tick + 1
        self._last_damage = 0
        self._last_result = "none"
        fatal_lava = False
        self._activate_scheduled_shocks(next_tick)

        if action in DIRECTIONS:
            direction = action
            if self._shock_active("confirm_inverted_move", next_tick):
                direction = {1: 2, 2: 1, 3: 4, 4: 3}[action]
            self.facing = ACTION_TO_FACING[direction]
            dx, dy = DIRECTIONS[direction]
            nx, ny = self.x + dx, self.y + dy
            if not self._inside(nx, ny) or self.grid[ny, nx] in (
                TREE, STONE, WATER, IRON_ORE,
            ):
                self._last_result = "blocked"
            else:
                self.x, self.y = nx, ny
                if self.grid[ny, nx] == LAVA:
                    fatal_lava = True
                    self.health = 0
                    self._last_damage = MAX_HEALTH
        elif action == 5:
            self._interact()
        elif action == 6:
            self._craft_wood_pickaxe()
        elif action == 7:
            self._craft_stone_pickaxe()
        elif action == 8:
            self._eat()

        self.tick = next_tick
        self._update_zombies()
        self._update_survival()
        self.death = fatal_lava or self.health <= 0
        if self.death:
            self.death_reason = "lava" if fatal_lava else "health"
        self.done = self.death or self.tick >= self.max_ticks

        reward = self._milestone_reward()
        reward -= 0.1 * self._last_damage
        if self.death:
            reward -= 1.0

        obs = self.observe()
        info = {
            "tick": self.tick,
            "death": self.death,
            "death_reason": self.death_reason,
            "damage": self._last_damage,
            "milestones": tuple(sorted(self._milestones_paid)),
            "task_success": {
                "T_wood": self.inventory["wood"] >= 5,
                "T_tools": self.inventory["stone_pickaxe"] >= 1,
                "T_iron": self.inventory["iron"] >= 1,
            },
        }
        return obs, float(reward), self.done, info

    def _interact(self) -> None:
        dx, dy = FACING_DIRECTIONS[self.facing]
        tx, ty = self.x + dx, self.y + dy
        if not self._inside(tx, ty):
            return
        zombie = next(
            (mob for mob in self.zombies if mob["x"] == tx and mob["y"] == ty),
            None,
        )
        if zombie is not None:
            zombie["hp"] -= 1
            self._last_result = "hit"
            if zombie["hp"] <= 0:
                self._respawn_timers[zombie["slot"]] = 30
                self.zombies.remove(zombie)
            return

        tile = int(self.grid[ty, tx])
        if tile == TREE:
            if self._shock_active("confirm_tree_dry", self.tick + 1):
                self._last_result = "none"
            else:
                self.inventory["wood"] += 1
                if self.rng.random() < 0.10:
                    self.inventory["apple"] += 1
                self._last_result = "ok"
        elif tile == 7 and self._shock_active("confirm_tree_dry", self.tick + 1):
            self.inventory["wood"] += 1
            self._last_result = "ok"
        elif tile == STONE:
            if self.inventory["wood_pickaxe"] < 1:
                self._last_result = "no_tool"
            else:
                self.inventory["stone"] += 1
                self.grid[ty, tx] = GRASS
                self._last_result = "ok"
                if self._shock_active("dev_tool_break", self.tick + 1):
                    shock = self._active_shock("dev_tool_break", self.tick + 1)
                    shock["hits"] += 1
                    if shock["hits"] >= 3:
                        self.inventory["wood_pickaxe"] = 0
        elif tile == IRON_ORE:
            if self.inventory["stone_pickaxe"] < 1:
                self._last_result = "no_tool"
            else:
                self.inventory["iron"] += 1
                self.grid[ty, tx] = GRASS
                self._last_result = "ok"
    def _craft_wood_pickaxe(self) -> None:
        if self.inventory["wood"] >= 2:
            self.inventory["wood"] -= 2
            self.inventory["wood_pickaxe"] += 1
            self._last_result = "crafted"

    def _craft_stone_pickaxe(self) -> None:
        if self.inventory["wood"] >= 1 and self.inventory["stone"] >= 2:
            self.inventory["wood"] -= 1
            self.inventory["stone"] -= 2
            self.inventory["stone_pickaxe"] += 1
            self._last_result = "crafted"

    def _eat(self) -> None:
        if self.inventory["apple"] < 1:
            return
        self.inventory["apple"] -= 1
        if self._shock_active("dev_poison_apple", self.tick + 1):
            self._damage(2)
            self._last_result = "ate"
            return
        self.food = min(MAX_FOOD, self.food + 4)
        self._starvation_clock = 0
        self._last_result = "ate"

    def _update_zombies(self) -> None:
        for slot, timer in tuple(self._respawn_timers.items()):
            if timer > 0:
                self._respawn_timers[slot] = timer - 1
            elif not any(mob["slot"] == slot for mob in self.zombies):
                self._spawn_zombie(slot)

        for zombie in sorted(self.zombies, key=lambda mob: mob["slot"]):
            dx, dy = self.x - zombie["x"], self.y - zombie["y"]
            distance = abs(dx) + abs(dy)
            if distance <= 1:
                if zombie["rest"] > 0:
                    zombie["rest"] -= 1
                else:
                    self._damage(2)
                    zombie["rest"] = 3
                continue

            if zombie["rest"] > 0:
                zombie["rest"] -= 1
            speed = 2 if self._shock_active("dev_fast_zombie", self.tick) else 1
            for _ in range(speed):
                dx, dy = self.x - zombie["x"], self.y - zombie["y"]
                distance = abs(dx) + abs(dy)
                if distance <= 1:
                    break
                candidates: list[tuple[int, int]] = []
                if distance <= 6 and self.rng.random() < 0.5:
                    if dx:
                        candidates.append((zombie["x"] + int(np.sign(dx)), zombie["y"]))
                    if dy:
                        candidates.append((zombie["x"], zombie["y"] + int(np.sign(dy))))
                    if candidates:
                        target = candidates[int(self.rng.integers(len(candidates)))]
                        self._move_zombie(zombie, target)
                        continue

                choices = [
                    (zombie["x"] + mx, zombie["y"] + my)
                    for mx, my in DIRECTIONS.values()
                ]
                self.rng.shuffle(choices)
                for target in choices:
                    if self._move_zombie(zombie, target):
                        break

    def _spawn_zombie(self, slot: int) -> bool:
        occupied = {(mob["x"], mob["y"]) for mob in self.zombies}
        candidates = [
            pos for pos in self._positions(self.grid, GRASS)
            if pos not in occupied and self._manhattan(pos, (self.x, self.y)) >= 6
        ]
        if not candidates:
            return False
        x, y = candidates[int(self.rng.integers(len(candidates)))]
        self.zombies.append({"slot": slot, "x": x, "y": y, "hp": 3, "rest": 0})
        self._respawn_timers[slot] = 0
        return True

    def _move_zombie(self, zombie: dict[str, int], pos: tuple[int, int]) -> bool:
        x, y = pos
        if not self._inside(x, y) or self.grid[y, x] != GRASS:
            return False
        if any(mob is not zombie and (mob["x"], mob["y"]) == pos for mob in self.zombies):
            return False
        zombie["x"], zombie["y"] = x, y
        return True

    def _update_survival(self) -> None:
        self._hunger_clock += 1
        if self._hunger_clock >= 25:
            self._hunger_clock = 0
            if self.food > 0:
                self.food -= 1

        if self.food == 0:
            self._starvation_clock += 1
            if self._starvation_clock >= 10:
                self._starvation_clock = 0
                self._damage(1)
        else:
            self._starvation_clock = 0

        if self.food >= 5:
            self._healing_clock += 1
            if self._healing_clock >= 20:
                self._healing_clock = 0
                self.health = min(MAX_HEALTH, self.health + 1)
        else:
            self._healing_clock = 0

    def _damage(self, amount: int) -> None:
        actual = min(self.health, int(amount))
        self.health -= actual
        self._last_damage += actual

    def _milestone_reward(self) -> float:
        checks = {
            "got_wood": self.inventory["wood"] >= 1,
            "wood_pickaxe": self.inventory["wood_pickaxe"] >= 1,
            "stone": self.inventory["stone"] >= 1,
            "stone_pickaxe": self.inventory["stone_pickaxe"] >= 1,
            "iron": self.inventory["iron"] >= 1,
        }
        reward = 0.0
        for name, reached in checks.items():
            if reached and name not in self._milestones_paid:
                self._milestones_paid.add(name)
                reward += 1.0
        return reward

    # -- observation helpers -------------------------------------------
    def observe(self) -> dict[str, Any]:
        patch = np.full((7, 7), 6, dtype=np.int64)
        for py, y in enumerate(range(self.y - PATCH_RADIUS, self.y + PATCH_RADIUS + 1)):
            for px, x in enumerate(range(self.x - PATCH_RADIUS, self.x + PATCH_RADIUS + 1)):
                if self._inside(x, y):
                    patch[py, px] = int(self.grid[y, x])
        visible = [
            {"dx": mob["x"] - self.x, "dy": mob["y"] - self.y, "hp": mob["hp"]}
            for mob in self.zombies
            if abs(mob["x"] - self.x) <= PATCH_RADIUS
            and abs(mob["y"] - self.y) <= PATCH_RADIUS
        ]
        visible.sort(key=lambda mob: (abs(mob["dx"]) + abs(mob["dy"]), mob["dy"], mob["dx"]))
        return {
            "self": {
                "x": int(self.x), "y": int(self.y), "facing": int(self.facing),
                "health": int(self.health), "food": int(self.food),
            },
            "inv": {key: int(value) for key, value in self.inventory.items()},
            "patch": patch,
            "mobs": visible,
            "events": {
                "last_result": self._last_result,
                "damage": int(self._last_damage),
                "tick": int(self.tick),
            },
        }

    # -- exact state snapshots -----------------------------------------
    def get_state(self) -> dict[str, Any]:
        """Return a deep snapshot of the whole episode, including RNG state."""
        return deepcopy({
            "schema_version": 1,
            "size": self.size,
            "seed": self.seed,
            "world_seed": self.world_seed,
            "max_ticks": self.max_ticks,
            "grid": self.grid,
            "x": self.x,
            "y": self.y,
            "facing": self.facing,
            "health": self.health,
            "food": self.food,
            "inventory": self.inventory,
            "tick": self.tick,
            "done": self.done,
            "death": self.death,
            "death_reason": self.death_reason,
            "hunger_clock": self._hunger_clock,
            "starvation_clock": self._starvation_clock,
            "healing_clock": self._healing_clock,
            "milestones_paid": self._milestones_paid,
            "last_result": self._last_result,
            "last_damage": self._last_damage,
            "zombies": self.zombies,
            "respawn_timers": self._respawn_timers,
            "shocks": self._shocks,
            "rng_state": self.rng.bit_generator.state,
        })

    def set_state(self, state: dict[str, Any]) -> None:
        """Restore an episode snapshot without sharing mutable objects."""
        if not isinstance(state, dict) or state.get("schema_version") != 1:
            raise ValueError("unsupported MiniCraft state snapshot")
        snapshot = deepcopy(state)
        if int(snapshot.get("size", -1)) != self.size:
            raise ValueError("snapshot world size does not match this environment")
        grid = np.asarray(snapshot.get("grid"))
        if grid.shape != (self.size, self.size) or grid.dtype.kind not in "iu":
            raise ValueError("snapshot grid must be a 20x20 integer array")
        if not np.isin(grid, (GRASS, TREE, STONE, WATER, LAVA, IRON_ORE, 7)).all():
            raise ValueError("snapshot grid contains an unknown tile")

        x, y = int(snapshot["x"]), int(snapshot["y"])
        if not self._inside(x, y):
            raise ValueError("snapshot agent position is outside the map")
        facing = int(snapshot["facing"])
        if facing not in FACING_DIRECTIONS:
            raise ValueError("snapshot facing is invalid")
        inventory = {key: int(snapshot["inventory"][key]) for key in INVENTORY_KEYS}
        if any(value < 0 for value in inventory.values()):
            raise ValueError("snapshot inventory values cannot be negative")

        self.seed = int(snapshot["seed"])
        self.world_seed = int(snapshot["world_seed"])
        self.max_ticks = int(snapshot["max_ticks"])
        if self.max_ticks < 1:
            raise ValueError("snapshot max_ticks must be positive")
        self.grid = grid.astype(np.int8, copy=True)
        self.x, self.y, self.facing = x, y, facing
        self.health = int(snapshot["health"])
        self.food = int(snapshot["food"])
        if not 0 <= self.health <= MAX_HEALTH or not 0 <= self.food <= MAX_FOOD:
            raise ValueError("snapshot health or food is outside its valid range")
        self.inventory = inventory
        self.tick = int(snapshot["tick"])
        self.done = bool(snapshot["done"])
        self.death = bool(snapshot["death"])
        self.death_reason = snapshot["death_reason"]
        self._hunger_clock = int(snapshot["hunger_clock"])
        self._starvation_clock = int(snapshot["starvation_clock"])
        self._healing_clock = int(snapshot["healing_clock"])
        self._milestones_paid = set(snapshot["milestones_paid"])
        self._last_result = str(snapshot["last_result"])
        self._last_damage = int(snapshot["last_damage"])
        self.zombies = [dict(mob) for mob in snapshot["zombies"]]
        self._respawn_timers = {
            int(slot): int(timer)
            for slot, timer in snapshot["respawn_timers"].items()
        }
        if set(self._respawn_timers) != set(range(MAX_ZOMBIES)):
            raise ValueError("snapshot must contain a respawn timer for each zombie slot")
        self._shocks = [dict(shock) for shock in snapshot.get("shocks", [])]
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = deepcopy(snapshot["rng_state"])

    def clone(self) -> "MiniCraftEnv":
        """Return an independent environment at exactly the current state."""
        clone = type(self).__new__(type(self))
        clone.set_state(self.get_state())
        return clone

    # -- scheduled dev/Confirm shocks ----------------------------------
    def inject_shock(self, kind: str, tick: int) -> None:
        """Schedule one isolated rule change to begin at ``tick``."""
        allowed = {
            "dev_poison_apple", "dev_fast_zombie", "dev_tool_break",
            "confirm_new_lava", "confirm_inverted_move", "confirm_tree_dry",
        }
        if kind not in allowed:
            raise ValueError(f"unknown shock kind: {kind}")
        if isinstance(tick, bool) or not isinstance(tick, (int, np.integer)):
            raise TypeError("shock tick must be an integer")
        if int(tick) < self.tick:
            raise ValueError("cannot inject a shock retroactively")
        self._shocks.append({
            "kind": kind, "tick": int(tick), "applied": False, "hits": 0,
        })

    def _activate_scheduled_shocks(self, tick: int) -> None:
        for shock in self._shocks:
            if shock["applied"] or shock["tick"] > tick:
                continue
            if shock["kind"] == "confirm_new_lava":
                protected = {
                    (self.x, self.y),
                    *((self.x + dx, self.y + dy) for dx, dy in DIRECTIONS.values()),
                }
                candidates = [
                    pos for pos in self._positions(self.grid, GRASS)
                    if pos not in protected
                ]
                count = round(len(candidates) * 0.05)
                if count:
                    selected = self.rng.choice(len(candidates), size=count, replace=False)
                    for index in selected:
                        x, y = candidates[int(index)]
                        self.grid[y, x] = LAVA
            elif shock["kind"] == "confirm_tree_dry":
                self.grid[self.grid == TREE] = 7
            shock["applied"] = True

    def _active_shock(self, kind: str, tick: int) -> dict[str, Any] | None:
        return next((
            shock for shock in reversed(self._shocks)
            if shock["kind"] == kind and shock["tick"] <= tick and shock["applied"]
        ), None)

    def _shock_active(self, kind: str, tick: int) -> bool:
        return self._active_shock(kind, tick) is not None

    # -- small utilities ------------------------------------------------
    def _inside(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size


__all__ = ["MiniCraftEnv", "GRASS", "TREE", "STONE", "WATER", "LAVA", "IRON_ORE"]
