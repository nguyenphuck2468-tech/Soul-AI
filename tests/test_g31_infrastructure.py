"""G3.1 infrastructure tests — semantics, not existence.
"""
import numpy as np
import torch
from soul.g31.collector import check_split_disjoint, record_transition
from soul.g31.baselines import fit_train_delta, predict_persistence
from soul.g31.metrics import compute_metrics, aggregate_episode_metrics
from soul.g31.evaluator import (
    decoded_observation_to_physical,
    evaluate_one_step,
    initial_state,
    physical_metrics,
    raw_observation_to_physical,
)


def test_split_disjoint():
    eps = [
        {'episode_id': 1, 'split': 'train'},
        {'episode_id': 2, 'split': 'pilot'},
    ]
    check_split_disjoint(eps)


def test_train_only_baseline():
    train = [
        {'raw_observation_before': [0, 0], 'raw_observation_after': [1, 2]},
    ]
    dx, dv = fit_train_delta(train)
    assert dx == 1.0 and dv == 2.0


def test_evaluator_uses_prior_not_posterior():
    class Dynamics:
        def __init__(self): self.action = None
        def initial_state(self, batch, device):
            return torch.zeros(batch, 2), torch.zeros(batch, 2)
        def step(self, h, z, action):
            self.action = action.clone()
            return h + 1, torch.full_like(z, 7), torch.zeros_like(z)
    class Encoder:
        def __call__(self, obs): return torch.full((1, 2), float(obs[0, 0])), torch.zeros(1, 2)
        def encode(self, obs):
            return {'values': np.asarray([obs[0], obs[1]], dtype=np.float32),
                    'mask': np.ones(2, dtype=np.float32), 'text': np.zeros(1, dtype=np.float32)}
    class Decoder:
        def __call__(self, h, z):
            assert torch.all(z == 7)
            # squash([cos(0), sin(0), velocity]) = [0.5, 0, 0.5]
            return {'obs': torch.tensor([[0.5, 0.0, 0.5]])}
    class Model:
        def __init__(self):
            self.dynamics, self.encoder, self.decoder = Dynamics(), Encoder(), Decoder()
            self.mode = None
        def eval(self): self.mode = 'eval'
    model = Model()
    state = initial_state(model.dynamics)
    pred, _ = evaluate_one_step(model, state, [1, 0, 0], [0.25], device='cpu', feature_encoder=model.encoder)
    np.testing.assert_array_equal(model.dynamics.action.numpy(), [[0.25]])
    np.testing.assert_allclose(pred, [0.0, 1.0])
    assert model.mode == 'eval'


def test_no_recursive_imagination():
    class Dynamics:
        def initial_state(self, batch, device): return torch.zeros(batch, 1), torch.zeros(batch, 1)
        def step(self, h, z, action): return h + 1, z + 2, z
    class Encoder:
        def __call__(self, obs): return torch.full((1, 1), float(obs[0, 0])), torch.zeros(1, 1)
        def encode(self, obs):
            return {'values': np.asarray([obs[0]], dtype=np.float32),
                    'mask': np.ones(1, dtype=np.float32), 'text': np.zeros(1, dtype=np.float32)}
    class Decoder:
        def __call__(self, h, z): return {'obs': torch.full((1, 2), 99.0)}
    class Model:
        def __init__(self):
            self.dynamics, self.encoder, self.decoder = Dynamics(), Encoder(), Decoder()
        def eval(self): pass
    model = Model()
    state = initial_state(model.dynamics)
    _, next_state = evaluate_one_step(
        model, state, [1, 0, 0], [0.0], obs_next=[0.2, 0.8, 0.4],
        feature_encoder=model.encoder)
    assert next_state.z.item() == 0.2
    assert next_state.z.item() != 99.0


def test_evaluator_keeps_model_parameters_unchanged_and_is_deterministic():
    class Dynamics:
        def initial_state(self, batch, device): return torch.zeros(batch, 1), torch.zeros(batch, 1)
        def step(self, h, z, action): return h + action, z + action, z
    class Encoder:
        def __call__(self, obs): return torch.zeros(1, 1), torch.zeros(1, 1)
        def encode(self, obs):
            return {'values': np.asarray([obs[0]], dtype=np.float32),
                    'mask': np.ones(1, dtype=np.float32), 'text': np.zeros(1, dtype=np.float32)}
    class Decoder:
        def __call__(self, h, z): return {'obs': z.clone()}
    class Model:
        def __init__(self):
            self.weight = torch.nn.Parameter(torch.tensor([3.0]))
            self.dynamics, self.encoder, self.decoder = Dynamics(), Encoder(), Decoder()
        def eval(self): pass
    model = Model()
    before = model.weight.detach().clone()
    first, _ = evaluate_one_step(model, initial_state(model.dynamics), [1, 0, 0], [0.2], feature_encoder=model.encoder)
    second, _ = evaluate_one_step(model, initial_state(model.dynamics), [1, 0, 0], [0.2], feature_encoder=model.encoder)
    np.testing.assert_array_equal(first, second)
    assert torch.equal(before, model.weight.detach())


def test_metrics_per_episode_equal_weight():
    m = [compute_metrics(np.array([1,2]), np.array([1,2]))]
    agg = aggregate_episode_metrics(m)
    assert agg['mse']['count'] == 1


def test_pendulum_observation_to_physical_state():
    theta = 2.0
    raw = np.array([np.cos(theta), np.sin(theta), 3.5], dtype=np.float32)
    np.testing.assert_allclose(raw_observation_to_physical(raw), [theta, 3.5], atol=1e-6)


def test_angle_error_wraps_at_pi_boundary():
    metrics = physical_metrics(
        np.array([np.pi - 0.01, 2.0]),
        np.array([-np.pi + 0.01, 2.0]),
    )
    assert metrics['theta_mae'] < 0.03
    assert metrics['velocity_mae'] == 0.0


def test_decoder_representation_decodes_to_physical_state():
    theta = -1.25
    velocity = 0.75
    # Encoder squash(v) = v/(1+abs(v)); decoder inversion restores v.
    decoded = [
        np.cos(theta) / (1 + abs(np.cos(theta))),
        np.sin(theta) / (1 + abs(np.sin(theta))),
        velocity / (1 + abs(velocity)),
    ]
    np.testing.assert_allclose(
        decoded_observation_to_physical(decoded), [theta, velocity], atol=1e-6)
