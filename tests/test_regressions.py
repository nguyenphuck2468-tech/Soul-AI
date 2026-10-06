import unittest

import numpy as np

from soul.demo import PendulumEnv
from soul.obs import ObsEncoder


class SoulRegressionTests(unittest.TestCase):
    def test_time_feature_keeps_information_across_long_horizon(self):
        enc = ObsEncoder()
        t0 = enc.encode({'time': 0})['values'][enc._schema['time']]
        t50 = enc.encode({'time': 50})['values'][enc._schema['time']]
        t200 = enc.encode({'time': 200})['values'][enc._schema['time']]

        self.assertGreater(t50 - t0, 0.20)
        self.assertGreater(t200 - t50, 0.20)

    def test_pendulum_env_clips_action_to_safe_range(self):
        env_clipped = PendulumEnv(seed=7)
        env_raw = PendulumEnv(seed=7)
        env_clipped.reset()
        env_raw.reset()

        _, _, _, _ = env_clipped.step(np.array([3.0], dtype=np.float32))
        _, _, _, _ = env_raw.step(np.array([1.0], dtype=np.float32))

        self.assertAlmostEqual(env_clipped.x, env_raw.x, places=5)
        self.assertAlmostEqual(env_clipped.v, env_raw.v, places=5)


if __name__ == '__main__':
    unittest.main()
