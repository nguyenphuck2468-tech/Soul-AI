import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from soul.config import Config
from soul.g31.collector import record_transition
from soul.g31.train_data import load_train_jsonl, pendulum_model_config, rows_to_episodes
from soul.obs import ObsEncoder


class G31ActionContractTests(unittest.TestCase):
    def test_pendulum_model_action_contract_is_scalar(self):
        cfg = pendulum_model_config(Config())
        self.assertEqual(cfg.model.action_dim, 1)
        self.assertEqual(cfg.model.discrete_dim, 0)

    def test_record_transition_requires_pendulum_shapes(self):
        with self.assertRaises(ValueError):
            record_transition(
                'ep', 'train', 1001, 0,
                [1.0, 0.0], [0.0], [0.0], 0.0, False,
                [1.0, 0.0], {'env_id': 'Pendulum-v1'})

    def test_train_loader_rejects_non_train_rows(self):
        row = {
            'episode_id': 'pilot_ep0',
            'split': 'pilot',
            'episode_seed': 2001,
            'raw_observation_before': [1.0, 0.0, 0.0],
            'raw_observation_after': [1.0, 0.0, 0.0],
            'applied_action': [0.0],
            'environment_config': {'env_id': 'Pendulum-v1'},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'mixed.jsonl'
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_train_jsonl(str(path))

    def test_train_loader_accepts_only_real_train_schema(self):
        row = {
            'episode_id': 'train_ep0',
            'split': 'train',
            'episode_seed': 1001,
            'raw_observation_before': [1.0, 0.0, 0.0],
            'raw_observation_after': [0.99, 0.1, 0.2],
            'applied_action': [0.5],
            'environment_config': {'env_id': 'Pendulum-v1'},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'train.jsonl'
            path.write_text(json.dumps(row) + '\n', encoding='utf-8')
            rows = load_train_jsonl(str(path))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['episode_seed'], 1001)

    def test_rows_to_episodes_preserves_order_actions_and_terminal(self):
        rows = []
        for timestep in range(2):
            rows.append({
                'episode_id': 'train_ep0', 'split': 'train',
                'episode_seed': 1001, 'timestep': timestep,
                'raw_observation_before': [1.0, 0.0, float(timestep)],
                'raw_observation_after': [0.9, 0.1, float(timestep + 1)],
                'requested_action': [float(timestep)],
                'applied_action': [float(timestep)],
                'reward': float(timestep), 'terminal': timestep == 1,
                'environment_config': {'env_id': 'Pendulum-v1'},
            })
        episodes = rows_to_episodes(rows, ObsEncoder(), goal_dim=4)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0].length, 2)
        np.testing.assert_array_equal(episodes[0].actions[0], [0.0])
        np.testing.assert_array_equal(episodes[0].actions[1], [1.0])
        self.assertEqual(episodes[0].continues, [1.0, 0.0])

    def test_rows_to_episodes_rejects_timestep_gap(self):
        rows = [{
            'episode_id': 'train_ep0', 'timestep': 1,
            'raw_observation_after': [1.0, 0.0, 0.0],
            'applied_action': [0.0], 'reward': 0.0, 'terminal': False,
        }]
        with self.assertRaises(ValueError):
            rows_to_episodes(rows, ObsEncoder(), goal_dim=4)


if __name__ == '__main__':
    unittest.main()
