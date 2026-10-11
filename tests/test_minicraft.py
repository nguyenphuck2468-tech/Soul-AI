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


if __name__ == "__main__":
    unittest.main()
