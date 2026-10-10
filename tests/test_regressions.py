import os
import tempfile
import unittest

import numpy as np

from soul.config import Config
from soul.demo import PendulumEnv
from soul.obs import ObsEncoder
from soul.soul import Soul


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

    def test_action_effect_predictor_is_available(self):
        cfg = Config()
        cfg.model.hidden_dim = 32
        cfg.model.latent_dim = 8
        cfg.train.batch_size = 2
        cfg.train.seq_len = 4
        soul = Soul(cfg)

        obs = {'time': 0.0, 'x': 0.0, 'y': 0.0}
        action = np.array([0.25], dtype=np.float32)
        pred = soul.predict_action_effect(obs, action)

        self.assertEqual(pred.shape, (cfg.model.latent_dim,))
        self.assertTrue(np.all(np.isfinite(pred)))

    def test_planned_action_uses_prediction(self):
        cfg = Config()
        cfg.model.hidden_dim = 32
        cfg.model.latent_dim = 8
        cfg.model.plan_samples = 8
        cfg.train.batch_size = 2
        cfg.train.seq_len = 4
        soul = Soul(cfg)

        obs = {'time': 0.0, 'x': 0.0, 'y': 0.0}
        action = soul.plan_action(obs, n_samples=8)

        self.assertEqual(action.shape, (cfg.model.action_dim,))
        self.assertTrue(np.all(np.isfinite(action)))
        self.assertTrue(np.all(np.abs(action) <= 4.0))

    def test_brain_has_cognition_modules(self):
        cfg = Config()
        cfg.model.hidden_dim = 32
        cfg.model.latent_dim = 8
        cfg.train.batch_size = 2
        cfg.train.seq_len = 4
        soul = Soul(cfg)

        obs = {'time': 0.0, 'x': 0.0, 'y': 0.0}
        brain = soul.brain

        state = brain.perceive(obs)
        pred = brain.predict(obs, np.array([0.25], dtype=np.float32))
        action = brain.plan(obs, n_samples=4)

        self.assertEqual(state.shape[0], soul.enc.input_dim)
        self.assertEqual(pred.shape, (cfg.model.latent_dim,))
        self.assertEqual(action.shape, (cfg.model.action_dim,))
        self.assertTrue(np.all(np.isfinite(state)))
        self.assertTrue(np.all(np.isfinite(pred)))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_skill_memory_extracts_and_recalls(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        event = {
            'time': 12,
            'reward': 0.8,
            'done': False,
            'action': np.array([0.5], dtype=np.float32),
            'context': {'goal': 'forward', 'risk': 'low'},
        }

        skill = mgr.extract_skill_from_event(event)
        self.assertIsNotNone(skill)
        self.assertTrue(skill['name'])
        self.assertGreater(len(mgr.skills.best()), 0)
        self.assertGreater(len(mgr.recall_skills({'goal': 'forward'})), 0)

    def test_archive_memory_can_persist_to_disk(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        event = {
            'time': 21,
            'reward': -0.3,
            'done': False,
            'action': np.array([0.2], dtype=np.float32),
            'context': {'goal': 'forward', 'risk': 'medium'},
        }
        mgr.record_event(event, important=True)
        mgr.extract_skill_from_event(event)

        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            mgr.archive.save(path)
            restored = MemoryManager()
            restored.archive.load(path)
            self.assertGreater(len(restored.archive.events), 0)
            self.assertGreater(len(restored.archive.skills), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_skill_confidence_decays_over_time(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        event = {
            'time': 5,
            'reward': 0.9,
            'done': False,
            'action': np.array([0.6], dtype=np.float32),
            'context': {'goal': 'forward', 'risk': 'low'},
        }
        skill = mgr.extract_skill_from_event(event)
        self.assertGreater(skill['confidence'], 0.5)

        mgr.skills.skills[0]['last_seen'] = 0
        mgr.refresh_skills(current_time=50, decay=0.08)
        self.assertLess(mgr.skills.skills[0]['confidence'], 0.9)

    def test_archive_summary_creates_reusable_skill(self):
        from soul.memory import MemoryManager

        mgr = MemoryManager()
        for i in range(3):
            event = {
                'time': i,
                'reward': 0.7 + 0.1 * i,
                'done': False,
                'action': np.array([0.5 + 0.2 * i], dtype=np.float32),
                'context': {'goal': 'forward', 'risk': 'low'},
            }
            mgr.record_event(event, important=True)
            mgr.extract_skill_from_event(event)

        skills = mgr.summarize_archive_to_skills(min_events=2, max_skills=4)
        self.assertTrue(skills)
        self.assertTrue(any(skill.get('context', {}).get('goal') == 'forward' for skill in skills))

    def test_soul_archive_can_persist_and_restore(self):
        cfg = Config()
        soul = Soul(cfg)
        event = {
            'time': 7,
            'reward': 0.8,
            'done': False,
            'action': np.array([0.25], dtype=np.float32),
            'context': {'goal': 'forward'},
        }
        soul.memory_manager.record_event(event, important=True)
        soul.memory_manager.extract_skill_from_event(event)

        fd, path = tempfile.mkstemp(suffix='.json')
        os.close(fd)
        try:
            soul.save_archive(path)
            restored = Soul(cfg)
            restored.load_archive(path)
            self.assertGreater(len(restored.memory_manager.archive.events), 0)
            self.assertGreater(len(restored.memory_manager.skills.skills), 0)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == '__main__':
    unittest.main()
