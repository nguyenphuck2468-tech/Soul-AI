import numpy as np
import torch

from minicraft import MiniCraftEnv
from soul.config import Config
from soul.soul import Soul
from wm_api.base import ImagineResult, LatentSnapshot, WorldModel
from wm_api.soul_adapter import SoulAdapter


def make_adapter():
    cfg = Config()
    cfg.model.hidden_dim = 8
    cfg.model.latent_dim = 4
    cfg.model.action_dim = 1
    cfg.model.discrete_dim = 0
    cfg.train.batch_size = 2
    cfg.train.seq_len = 3
    soul = Soul(cfg)
    soul._h = torch.randn(1, cfg.model.hidden_dim, device=soul.device)
    soul._z = torch.randn(1, cfg.model.latent_dim, device=soul.device)
    soul._prev_action = np.asarray([0.25], dtype=np.float32)
    soul.set_goal(np.zeros(cfg.model.goal_dim, dtype=np.float32))
    return soul, SoulAdapter(soul)


def test_snapshot_restore_copies_latent_exactly():
    soul, adapter = make_adapter()
    expected_h = soul._h.clone()
    expected_z = soul._z.clone()
    snap = adapter.snapshot()
    assert isinstance(snap, LatentSnapshot)

    soul._h.zero_()
    soul._z.zero_()
    adapter.restore(snap)

    assert torch.equal(soul._h, expected_h)
    assert torch.equal(soul._z, expected_z)
    assert soul._h.data_ptr() != snap.payload['h'].data_ptr()
    assert soul._z.data_ptr() != snap.payload['z'].data_ptr()


def test_imagine_is_repeatable_with_same_seed_and_actions():
    _, adapter = make_adapter()
    snap = adapter.snapshot()
    actions = [0, 1, 0]

    torch.manual_seed(1234)
    first = adapter.imagine(snap, actions)
    torch.manual_seed(1234)
    second = adapter.imagine(snap, actions)

    assert isinstance(first, ImagineResult)
    np.testing.assert_array_equal(first.pred_obs, second.pred_obs)
    np.testing.assert_array_equal(first.uncertainty, np.zeros(3, dtype=np.float32))
    assert first.horizon == 3
    assert first.pred_obs.shape == (3, adapter.soul.cfg.obs.max_features)


def test_imagine_does_not_change_live_soul_state():
    soul, adapter = make_adapter()
    before_h = soul._h.clone()
    before_z = soul._z.clone()
    before_action = soul._prev_action.copy()
    before_goal = soul._goal.copy()
    before_steps = soul.steps
    before_tick = adapter.snapshot().tick

    adapter.imagine(adapter.snapshot(), [1, 0, 1])

    assert torch.equal(soul._h, before_h)
    assert torch.equal(soul._z, before_z)
    np.testing.assert_array_equal(soul._prev_action, before_action)
    np.testing.assert_array_equal(soul._goal, before_goal)
    assert soul.steps == before_steps
    assert adapter.snapshot().tick == before_tick


def test_batched_action_embedding_preserves_imagine_sequence():
    soul, _ = make_adapter()
    dynamics = soul.model.dynamics
    h = torch.randn(1, dynamics.hidden)
    z = torch.randn(1, dynamics.latent)
    actions = torch.randn(1, 5, soul.cfg.model.action_dim)

    # Reference the pre-optimization per-timestep path with the same latent
    # noise, so this checks that batching action embeddings changes no output.
    torch.manual_seed(23)
    old_h, old_z = h, z
    old_hs, old_zs, old_mus, old_lvs = [], [], [], []
    for t in range(actions.shape[1]):
        old_h, mu, lv = dynamics.step(old_h, old_z, actions[:, t])
        old_z = mu + torch.randn_like(mu) * (0.5 * lv.clamp(-8.0, 4.0)).exp()
        old_hs.append(old_h)
        old_zs.append(old_z)
        old_mus.append(mu)
        old_lvs.append(lv)
    expected = {
        'h': torch.stack(old_hs, 1),
        'z': torch.stack(old_zs, 1),
        'mu': torch.stack(old_mus, 1),
        'logvar': torch.stack(old_lvs, 1),
    }

    torch.manual_seed(23)
    actual = dynamics.imagine_sequence(h, z, actions)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-6)


def test_integer_actions_are_encoded_as_one_hot_for_vector_action_models():
    cfg = Config()
    cfg.model.hidden_dim = 8
    cfg.model.latent_dim = 4
    cfg.model.action_dim = 3
    cfg.model.discrete_dim = 0
    soul = Soul(cfg)
    adapter = SoulAdapter(soul)

    np.testing.assert_array_equal(
        adapter._encode_action(2), np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


def test_nine_way_discrete_policy_has_no_synthetic_continuous_action():
    cfg = Config()
    cfg.model.hidden_dim = 8
    cfg.model.latent_dim = 4
    cfg.model.action_dim = 9
    cfg.model.discrete_dim = 9
    cfg.model.use_goal = False
    cfg.train.batch_size = 1
    cfg.train.seq_len = 2
    cfg.train.imagine_horizon = 2
    soul = Soul(cfg)
    adapter = SoulAdapter(soul)

    encoded = adapter._encode_action(8)
    np.testing.assert_array_equal(encoded, np.eye(9, dtype=np.float32)[8])
    h = torch.zeros(1, cfg.model.hidden_dim)
    z = torch.zeros(1, cfg.model.latent_dim)
    goal = torch.zeros(1, 0)
    mean, logstd, logits = soul.model.policy_head(h, z, goal)
    assert mean.shape == logstd.shape == (1, 0)
    assert logits.shape == (1, 9)
    action = soul.model.policy_head.sample(mean, logstd, logits, deterministic=True)
    assert action.shape == (1, 9)
    torch.testing.assert_close(action.sum(-1), torch.ones(1))
    assert torch.isfinite(soul.model.policy_head.log_prob(mean, logstd, logits, action)).all()
    assert torch.isfinite(soul.model.policy_head.entropy(logstd, logits)).all()

    env = MiniCraftEnv(seed=23, max_ticks=3)
    env.zombies = []
    env._respawn_timers = {slot: 1000 for slot in range(3)}
    while not env.done:
        obs, reward, done, _ = env.step(0)
        adapter.observe(obs, 0, reward, done)
    losses = soul.learn(force=True)
    assert losses is not None
    assert np.isfinite(losses["total"])


def test_minicraft_factory_configures_nine_discrete_actions():
    adapter = SoulAdapter.for_minicraft(hidden_dim=8, latent_dim=4, seed=7)
    assert adapter.soul.cfg.model.action_dim == 9
    assert adapter.soul.cfg.model.discrete_dim == 9
    np.testing.assert_array_equal(
        adapter._encode_action(5), np.asarray([0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=np.float32))
