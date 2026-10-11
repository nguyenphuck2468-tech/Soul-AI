"""WorldModel adapter over the existing Soul implementation.

`imagine` delegates the prior rollout to `Dynamics.imagine_sequence` and
decodes its latent states. `uncertainty` is currently a zero placeholder;
calibrated uncertainty is not implemented yet and is reserved for phase C.
`surprise` returns a raw masked one-step prediction MSE and is explicitly
uncalibrated; it is diagnostic only and must not be used as an intervention
threshold without separate calibration.

Scalar integer actions are mapped to one-hot vectors when Soul has more than
one action input dimension. A one-dimensional action input receives the
scalar as a float. The adapter does not change Soul's policy output format.
"""

from copy import deepcopy
import time
from typing import Any, Sequence

import numpy as np
import torch

from .base import ImagineResult, LatentSnapshot, WorldModel


class SoulAdapter(WorldModel):
    """Expose a Soul instance through the small `WorldModel` protocol."""

    @classmethod
    def for_minicraft(cls, *, device: str = 'cpu', seed: int = 0,
                      hidden_dim: int = 96, latent_dim: int = 24):
        """Build a Soul configured for MiniCraft's nine categorical actions."""
        try:
            from soul.config import Config
            from soul.soul import Soul
        except ImportError:  # pragma: no cover - package-relative execution
            from ..soul.config import Config
            from ..soul.soul import Soul

        cfg = Config(device=device)
        cfg.model.action_dim = 9
        cfg.model.discrete_dim = 9
        cfg.model.hidden_dim = hidden_dim
        cfg.model.latent_dim = latent_dim
        cfg.train.seed = seed
        return cls(Soul(cfg))

    def __init__(self, soul):
        self.soul = soul
        self._tick = 0
        self._wm_h = None
        self._wm_z = None
        self._wm_prev_action = None
        self._pending_surprise = None
        self.surprise_status = 'UNCALIBRATED_RAW_MASKED_MSE'

    def _encode_action(self, action: int) -> np.ndarray:
        if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
            raise TypeError('WorldModel action must be an integer.')
        action_dim = self.soul.cfg.model.action_dim
        if action_dim == 1:
            return np.asarray([action], dtype=np.float32)
        if not 0 <= int(action) < action_dim:
            raise ValueError(f'action must be in [0, {action_dim - 1}].')
        encoded = np.zeros(action_dim, dtype=np.float32)
        encoded[int(action)] = 1.0
        return encoded

    def reset(self) -> None:
        self.soul.reset_state()
        self._tick = 0
        self._wm_h = None
        self._wm_z = None
        self._wm_prev_action = None
        self._pending_surprise = None

    def observe(self, obs: Any, action: int, reward: float,
                done: bool) -> None:
        encoded_action = self._encode_action(action)
        encoded_obs = self.soul.enc.encode(obs)
        current_h = self._wm_h if self._wm_h is not None else self.soul._h
        current_z = self._wm_z if self._wm_z is not None else self.soul._z
        if current_h is None or current_z is None:
            h, z = self.soul.model.dynamics.initial_state(1, self.soul.device)
        else:
            h = torch.as_tensor(current_h, device=self.soul.device).detach().clone()
            z = torch.as_tensor(current_z, device=self.soul.device).detach().clone()
        action_tensor = torch.as_tensor(
            encoded_action, dtype=torch.float32, device=self.soul.device).unsqueeze(0)
        with torch.no_grad():
            h_next, prior_mu, _ = self.soul.model.dynamics.step(h, z, action_tensor)
            prediction = self.soul.model.decoder(h_next, prior_mu)['obs']
        self._pending_surprise = {
            'prediction': prediction.squeeze(0).cpu().numpy().astype(np.float32, copy=True),
            'values': encoded_obs['values'].copy(),
            'mask': encoded_obs['mask'].copy(),
            'text': encoded_obs['text'].copy(),
        }
        self.soul.observe(obs, encoded_action, reward, done)
        if done:
            self._wm_h = None
            self._wm_z = None
            self._wm_prev_action = None
        else:
            encoded_vector = np.concatenate([
                encoded_obs['values'], encoded_obs['mask'], encoded_obs['text']])
            with torch.no_grad():
                posterior_mu, _ = self.soul.model.encoder(torch.as_tensor(
                    encoded_vector, dtype=torch.float32,
                    device=self.soul.device).unsqueeze(0))
            self._wm_h = h_next.detach().clone()
            self._wm_z = posterior_mu.detach().clone()
            self._wm_prev_action = encoded_action.copy()
        self._tick += 1

    def snapshot(self) -> LatentSnapshot:
        """Deep-copy Soul's recurrent state and associated action/goal state."""
        def clone(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, np.ndarray):
                return value.copy()
            return deepcopy(value)

        payload = {
            'h': clone(self._wm_h if self._wm_h is not None else self.soul._h),
            'z': clone(self._wm_z if self._wm_z is not None else self.soul._z),
            'prev_action': clone(self._wm_prev_action
                                 if self._wm_prev_action is not None
                                 else self.soul._prev_action),
            'goal': clone(self.soul._goal),
        }
        return LatentSnapshot(payload=payload, tick=self._tick)

    def restore(self, snap: LatentSnapshot) -> None:
        """Restore copied recurrent state without aliasing the snapshot."""
        payload = snap.payload
        device = self.soul.device

        def clone_to_device(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.detach().clone().to(device)
            if isinstance(value, np.ndarray):
                return value.copy()
            return deepcopy(value)

        self.soul._h = clone_to_device(payload['h'])
        self.soul._z = clone_to_device(payload['z'])
        self.soul._prev_action = clone_to_device(payload['prev_action'])
        self.soul._goal = clone_to_device(payload['goal'])
        self._wm_h = clone_to_device(payload['h'])
        self._wm_z = clone_to_device(payload['z'])
        self._wm_prev_action = clone_to_device(payload['prev_action'])
        self._pending_surprise = None
        self._tick = int(snap.tick)

    def imagine(self, snap: LatentSnapshot,
                actions: Sequence[int]) -> ImagineResult:
        """Roll out prior actions, returning decoded normalized value slots.

        Callers that need repeatable stochastic latent samples should set the
        PyTorch RNG seed before each call. This method does not mutate Soul's
        live recurrent state. `uncertainty` is zero-filled pending phase C.
        """
        started = time.perf_counter()
        encoded = [self._encode_action(action) for action in actions]
        action_dim = self.soul.cfg.model.action_dim
        device = self.soul.device
        if encoded:
            action_tensor = torch.as_tensor(
                np.stack(encoded)[None, ...], dtype=torch.float32, device=device)
        else:
            action_tensor = torch.zeros((1, 0, action_dim),
                                        dtype=torch.float32, device=device)

        payload = snap.payload
        if payload['h'] is None or payload['z'] is None:
            h, z = self.soul.model.dynamics.initial_state(1, device)
        else:
            h = payload['h'].detach().clone().to(device)
            z = payload['z'].detach().clone().to(device)

        with torch.no_grad():
            rollout = self.soul.model.dynamics.imagine_sequence(
                h, z, action_tensor)
            pred = self.soul.model.decoder(
                rollout['h'], rollout['z'])['obs']
        pred_obs = pred.squeeze(0).cpu().numpy().astype(np.float32, copy=True)
        horizon = int(pred_obs.shape[0])
        return ImagineResult(
            pred_obs=pred_obs,
            uncertainty=np.zeros(horizon, dtype=np.float32),
            horizon=horizon,
            wall_seconds=time.perf_counter() - started,
            uncertainty_status='NOT_IMPLEMENTED_PLACEHOLDER',
        )

    def surprise(self, obs: Any) -> float:
        """Return uncalibrated masked MSE against the immediately prior prediction.

        This is a raw reality-violation diagnostic, not calibrated uncertainty
        and not suitable as an intervention threshold without a separate
        calibration protocol.
        """
        if self._pending_surprise is None:
            raise RuntimeError('surprise requires a preceding observe() call')
        observed = self.soul.enc.encode(obs)
        pending = self._pending_surprise
        matches = all(np.array_equal(observed[key], pending[key])
                      for key in ('values', 'mask', 'text'))
        if not matches:
            raise ValueError('surprise observation does not match the latest observe()')
        count = min(self.soul.cfg.obs.max_features,
                    len(pending['prediction']), len(pending['values']))
        mask = pending['mask'][:count] > 0
        if not np.any(mask):
            self._pending_surprise = None
            raise ValueError('surprise observation has no known numeric features')
        error = pending['prediction'][:count][mask] - pending['values'][:count][mask]
        self._pending_surprise = None
        return float(np.mean(np.square(error.astype(np.float64))))
