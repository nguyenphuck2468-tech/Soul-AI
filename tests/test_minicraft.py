import unittest

import numpy as np

from minicraft import GRASS, LAVA, STONE, TREE, MiniCraftEnv
from soul.obs import ObsEncoder


class MiniCraftObservationTests(unittest.TestCase):
    def test_encoder_preserves_each_patch_cell_and_marks_zero_as_present(self):
        env = MiniCraftEnv(seed=0)
        observation = env.observe()
        encoder = ObsEncoder()

        encoded = encoder.encode(observation)
        self.assertEqual(encoded["values"].shape, (encoder.cfg.max_features,))
        self.assertEqual(encoded["mask"].shape, (encoder.cfg.max_features,))
        self.assertEqual(encoded["text"].shape, (encoder.cfg.text_buckets,))

        patch_features = [
            name for name in encoder.schema if name.startswith("patch[")
        ]
        self.assertEqual(len(patch_features), 49)
        for y in range(7):
            for x in range(7):
                feature = f"patch[{y}][{x}]"
                self.assertIn(feature, encoder.schema)
                index = encoder.schema[feature]
                self.assertEqual(encoded["mask"][index], 1.0)
                raw_value = int(observation["patch"][y, x])
                expected = raw_value / (1 + abs(raw_value))
                self.assertAlmostEqual(encoded["values"][index], expected)

    def test_encoder_keeps_legacy_one_dimensional_array_summary(self):
        encoder = ObsEncoder()
        encoded = encoder.encode({"legacy": np.array([2, 4], dtype=np.int64)})

        self.assertEqual(encoder.schema, {"legacy": 0})
        self.assertEqual(encoded["mask"][0], 1.0)
        self.assertAlmostEqual(encoded["values"][0], 2 / 3)


class MiniCraftStateTests(unittest.TestCase):
    def test_snapshot_restore_replays_identical_random_transitions(self):
        env = MiniCraftEnv(seed=23)
        for action in (0, 0, 3, 0):
            env.step(action)
        snapshot = env.get_state()
        restored = MiniCraftEnv(seed=999)
        restored.set_state(snapshot)

        for action in (0, 2, 0, 5, 1, 0, 8, 0):
            left = env.step(action)
            right = restored.step(action)
            np.testing.assert_array_equal(left[0]["patch"], right[0]["patch"])
            self.assertEqual(left[0]["self"], right[0]["self"])
            self.assertEqual(left[0]["inv"], right[0]["inv"])
            self.assertEqual(left[0]["mobs"], right[0]["mobs"])
            self.assertEqual(left[0]["events"], right[0]["events"])
            self.assertEqual(left[1:], right[1:])

    def test_snapshot_and_clone_do_not_share_mutable_state(self):
        env = MiniCraftEnv(seed=7)
        snapshot = env.get_state()
        original_tile = int(env.grid[0, 0])
        snapshot["grid"][0, 0] = (original_tile + 1) % 6
        self.assertEqual(int(env.grid[0, 0]), original_tile)

        clone = env.clone()
        clone.step(0)
        self.assertEqual(env.tick, 0)
        self.assertEqual(clone.tick, 1)
        self.assertFalse(np.shares_memory(env.grid, clone.grid))


class MiniCraftShockTests(unittest.TestCase):
    @staticmethod
    def quiet_env(seed=0):
        env = MiniCraftEnv(seed=seed)
        env.zombies = []
        env._respawn_timers = {slot: 1000 for slot in range(3)}
        return env

    def test_new_lava_activates_at_scheduled_tick_and_preserves_neighbours(self):
        left = self.quiet_env(seed=9)
        right = self.quiet_env(seed=9)
        original = left.grid.copy()
        protected = {
            (left.x, left.y),
            *((left.x + dx, left.y + dy) for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0))),
        }
        candidates = [
            (x, y) for y in range(20) for x in range(20)
            if original[y, x] == GRASS and (x, y) not in protected
        ]
        expected_count = round(len(candidates) * 0.05)
        for env in (left, right):
            env.inject_shock("confirm_new_lava", tick=2)
            env.step(0)
            np.testing.assert_array_equal(env.grid, original)
            env.step(0)

        new_lava = set(zip(*np.where(left.grid == LAVA))) - set(zip(*np.where(original == LAVA)))
        self.assertEqual(len(new_lava), expected_count)
        self.assertFalse(any((y, x) in protected for y, x in new_lava))
        np.testing.assert_array_equal(left.grid, right.grid)

    def test_inverted_move_swaps_north_and_south(self):
        env = self.quiet_env()
        env.grid[:, :] = GRASS
        env.x = env.y = 10
        env.inject_shock("confirm_inverted_move", tick=1)

        env.step(1)

        self.assertEqual((env.x, env.y), (10, 11))
        self.assertEqual(env.facing, 1)

    def test_tree_dry_replaces_trees_with_harvestable_dust(self):
        env = self.quiet_env()
        env.grid[:, :] = GRASS
        env.x = env.y = 10
        env.grid[9, 10] = TREE
        env.inject_shock("confirm_tree_dry", tick=1)

        env.step(5)

        self.assertEqual(env.inventory["wood"], 1)
        self.assertFalse(np.any(env.grid == TREE))
        self.assertEqual(int(env.grid[9, 10]), 7)

    def test_poison_apple_consumes_apple_without_restoring_food(self):
        env = self.quiet_env()
        env.health = 9
        env.food = 4
        env.inventory["apple"] = 1
        env.inject_shock("dev_poison_apple", tick=1)

        observation, _, done, info = env.step(8)

        self.assertFalse(done)
        self.assertEqual(env.health, 7)
        self.assertEqual(env.food, 4)
        self.assertEqual(env.inventory["apple"], 0)
        self.assertEqual(info["damage"], 2)
        self.assertEqual(observation["events"]["last_result"], "ate")

    def test_fast_zombie_moves_two_cells_in_one_tick(self):
        normal = self.quiet_env()
        normal.grid[:, :] = GRASS
        normal.x = normal.y = 2
        normal.zombies = [{"slot": 0, "x": 15, "y": 15, "hp": 3, "rest": 0}]
        fast = normal.clone()
        fast.inject_shock("dev_fast_zombie", tick=1)

        normal.step(0)
        fast.step(0)

        self.assertEqual(abs(normal.zombies[0]["x"] - 15) + abs(normal.zombies[0]["y"] - 15), 1)
        self.assertEqual(abs(fast.zombies[0]["x"] - 15) + abs(fast.zombies[0]["y"] - 15), 2)

    def test_wood_pickaxe_breaks_after_third_successful_stone_mine(self):
        env = self.quiet_env()
        env.x = env.y = 10
        env.inventory["wood_pickaxe"] = 1
        env.inject_shock("dev_tool_break", tick=1)

        for _ in range(3):
            env.grid[9, 10] = STONE
            env.step(5)

        self.assertEqual(env.inventory["stone"], 3)
        self.assertEqual(env.inventory["wood_pickaxe"], 0)


if __name__ == "__main__":
    unittest.main()
