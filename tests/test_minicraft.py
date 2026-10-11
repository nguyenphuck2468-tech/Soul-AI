import unittest

import numpy as np

from minicraft import MiniCraftEnv
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


if __name__ == "__main__":
    unittest.main()
