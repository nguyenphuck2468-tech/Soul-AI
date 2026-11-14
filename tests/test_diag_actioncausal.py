import unittest

import torch

from soul.diag_actioncausal import _shuffle_action_timesteps


class ActionShuffleTests(unittest.TestCase):
    def test_shuffle_is_per_sequence_over_time_and_preserves_padding(self):
        actions = torch.tensor(
            [
                [[1.0], [2.0], [3.0]],
                [[10.0], [20.0], [30.0]],
            ]
        )
        valid = torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        torch.manual_seed(7)

        shuffled = _shuffle_action_timesteps(actions, valid)

        self.assertEqual(sorted(shuffled[0, :, 0].tolist()), [1.0, 2.0, 3.0])
        self.assertEqual(
            sorted(shuffled[1, valid[1] > 0, 0].tolist()),
            [10.0, 30.0],
        )
        self.assertEqual(shuffled[1, 1, 0].item(), 20.0)
        self.assertTrue(
            all(value in {1.0, 2.0, 3.0} for value in shuffled[0, :, 0].tolist())
        )
        self.assertTrue(
            all(value in {10.0, 20.0, 30.0} for value in shuffled[1, :, 0].tolist())
        )

    def test_shuffle_rejects_incompatible_shapes(self):
        with self.assertRaisesRegex(ValueError, r"actions must be"):
            _shuffle_action_timesteps(torch.zeros(2, 3), torch.ones(2, 3))


if __name__ == "__main__":
    unittest.main()
