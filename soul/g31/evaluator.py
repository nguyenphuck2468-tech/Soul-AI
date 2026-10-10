"""One-step deterministic prior evaluation with real-observation filtering.

G3.1 locks ``x`` to angular position theta, not Cartesian position:
raw Pendulum observation = [cos(theta), sin(theta), angular_velocity].
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class EvaluatorState:
    h: torch.Tensor
    z: torch.Tensor


def raw_observation_to_physical(observation):
    """Convert raw Pendulum observation to (theta radians, velocity rad/s)."""
    raw = np.asarray(observation, dtype=np.float32)
    if raw.shape != (3,):
        raise ValueError('Pendulum observation must have shape (3,)')
    return np.asarray([np.arctan2(raw[1], raw[0]), raw[2]], dtype=np.float32)


def wrapped_angle_error(pred_theta, true_theta):
    """Return periodic angular error in radians."""
    delta = np.asarray(pred_theta) - np.asarray(true_theta)
    return np.arctan2(np.sin(delta), np.cos(delta))


def physical_metrics(predicted, target):
    """Calculate theta-wrapped and linear-velocity metrics in physical units."""
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    error[..., 0] = wrapped_angle_error(predicted[..., 0], target[..., 0])
    return {
        'theta_mse': float(np.mean(error[..., 0] ** 2)),
        'theta_mae': float(np.mean(np.abs(error[..., 0]))),
        'theta_rmse': float(np.sqrt(np.mean(error[..., 0] ** 2))),
        'theta_bias': float(np.mean(error[..., 0])),
        'velocity_mse': float(np.mean(error[..., 1] ** 2)),
        'velocity_mae': float(np.mean(np.abs(error[..., 1]))),
        'velocity_rmse': float(np.sqrt(np.mean(error[..., 1] ** 2))),
        'velocity_bias': float(np.mean(error[..., 1])),
    }


def inverse_squash(value):
    """Invert ObsEncoder.squash without silently clipping invalid output."""
    value = np.asarray(value, dtype=np.float32)
    if np.any(np.abs(value) >= 1.0):
        raise ValueError('decoder value outside invertible squash range')
    return value / (1.0 - np.abs(value))


def decoded_observation_to_physical(decoded_values):
    """Decode first three model-space values to theta radians and rad/s."""
    values = np.asarray(decoded_values, dtype=np.float32).reshape(-1)
    if values.size < 3:
        raise ValueError('decoder output must contain three observation values')
    raw = inverse_squash(values[:3])
    return raw_observation_to_physical(raw)


def observation_vector(observation, feature_encoder):
    """Build the exact values/mask/text vector used during training."""
    if feature_encoder is None:
        raise ValueError('feature_encoder is required for real checkpoint evaluation')
    encoded = feature_encoder.encode(observation)
    return np.concatenate(
        [encoded['values'], encoded['mask'], encoded['text']], axis=0
    ).astype(np.float32)


def initial_state(dynamics, device='cpu'):
    h, z = dynamics.initial_state(1, device)
    return EvaluatorState(h=h, z=z)


def evaluate_one_step(model, state, obs_current, action_applied,
                      obs_next=None, device='cpu', feature_encoder=None):
    """Predict with prior mean, then optionally teacher-force real obs_next.

    Returns physical (theta radians, angular_velocity rad/s) and state.
    ``feature_encoder`` is the persisted ObsEncoder used by the checkpoint.
    """
    action = np.asarray(action_applied, dtype=np.float32)
    current = np.asarray(obs_current, dtype=np.float32)
    if current.shape != (3,):
        raise ValueError('Pendulum observation must have shape (3,)')
    if action.shape != (1,):
        raise ValueError('Pendulum applied action must have shape (1,)')
    if np.any(action < -2.0) or np.any(action > 2.0):
        raise ValueError('Pendulum applied action outside [-2, 2]')
    if obs_next is not None and np.asarray(obs_next).shape != (3,):
        raise ValueError('Pendulum next observation must have shape (3,)')

    model.eval()
    with torch.no_grad():
        current_t = torch.as_tensor(
            observation_vector(current, feature_encoder), device=device
        ).reshape(1, -1)
        z_current, _ = model.encoder(current_t)
        action_t = torch.as_tensor(action, device=device).reshape(1, 1)
        h_prior, p_mu, _ = model.dynamics.step(state.h, z_current, action_t)
        decoded = model.decoder(h_prior, p_mu)['obs'].squeeze(0).cpu().numpy().copy()
        predicted_physical = decoded_observation_to_physical(decoded)

        # Teacher-force with the real next observation after prediction.
        if obs_next is None:
            return predicted_physical, EvaluatorState(h=h_prior.detach(), z=p_mu.detach())
        next_raw = np.asarray(obs_next, dtype=np.float32)
        next_t = torch.as_tensor(
            observation_vector(next_raw, feature_encoder), device=device
        ).reshape(1, -1)
        q_mu, _ = model.encoder(next_t)
        return predicted_physical, EvaluatorState(h=h_prior.detach(), z=q_mu.detach())
