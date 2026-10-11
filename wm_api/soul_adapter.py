"""WorldModel adapter over the existing Soul implementation.

`imagine` delegates the prior rollout to `Dynamics.imagine_sequence` and
decodes its latent states. `uncertainty` is currently a zero placeholder;
calibrated uncertainty is not implemented yet and is reserved for phase C.
`surprise` is also not implemented because Soul does not retain the previous
prediction needed to measure the reality violation for the new observation.

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

    def observe(self, obs: Any, action: int, reward: float,
                done: bool) -> None:
        encoded_action = self._encode_action(action)
        self.soul.observe(obs, encoded_action, reward, done)
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
            'h': clone(self.soul._h),
            'z': clone(self.soul._z),
            'prev_action': clone(self.soul._prev_action),
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
        )

    def surprise(self, obs: Any) -> float:
        """NotImplemented: the previous-step prediction is not retained by Soul."""
        raise NotImplementedError(
            'SoulAdapter.surprise needs a retained prior prediction; '
            'reality-violation scoring is not implemented yet.')
