"""G3.1 data collector — episode-level disjoint splits.

No confirm evaluation; no pilot/confirm fitting; split integrity enforced.
"""
from typing import List, Dict, Any

import numpy as np

SPLITS = ('train', 'pilot', 'confirm')
PENDULUM_ENV_ID = 'Pendulum-v1'
PENDULUM_OBS_SHAPE = (3,)
PENDULUM_ACTION_SHAPE = (1,)
PENDULUM_ACTION_BOUNDS = (-2.0, 2.0)


def check_split_disjoint(episodes: List[Dict[str, Any]]) -> None:
    seen_ep = {}
    for ep in episodes:
        eid = ep.get('episode_id')
        split = ep.get('split')
        if eid in seen_ep and seen_ep[eid] != split:
            raise ValueError(f"episode {eid} in multiple splits")
        seen_ep[eid] = split
    seeds = {split: set() for split in SPLITS}
    for ep in episodes:
        split = ep.get('split')
        if split not in seeds:
            raise ValueError(f'unknown split: {split}')
        seeds[split].add(ep.get('episode_seed'))
    if seeds['train'] & seeds['pilot'] or seeds['train'] & seeds['confirm']:
        raise ValueError('train seed overlaps another split')
    if seeds['pilot'] & seeds['confirm']:
        raise ValueError('pilot seed overlaps confirm split')


def record_transition(ep_id, split, seed, timestep,
                      obs_before, requested_action, action_applied, reward, terminal,
                      obs_after, meta) -> Dict[str, Any]:
    before = np.asarray(obs_before, dtype=np.float32)
    requested = np.asarray(requested_action, dtype=np.float32)
    applied = np.asarray(action_applied, dtype=np.float32)
    after = np.asarray(obs_after, dtype=np.float32)
    if before.shape != PENDULUM_OBS_SHAPE or after.shape != PENDULUM_OBS_SHAPE:
        raise ValueError('Pendulum observations must have shape (3,)')
    if requested.shape != PENDULUM_ACTION_SHAPE or applied.shape != PENDULUM_ACTION_SHAPE:
        raise ValueError('Pendulum actions must have shape (1,)')
    if np.any(applied < PENDULUM_ACTION_BOUNDS[0]) or np.any(applied > PENDULUM_ACTION_BOUNDS[1]):
        raise ValueError('applied action is outside Pendulum-v1 bounds')
    if meta.get('env_id') != PENDULUM_ENV_ID:
        raise ValueError('G3.1 data must identify Pendulum-v1')
    return {
        'episode_id': ep_id,
        'split': split,
        'episode_seed': seed,
        'timestep': timestep,
        'raw_observation_before': before.tolist(),
        'requested_action': requested.tolist(),
        'applied_action': applied.tolist(),
        'reward': reward,
        'terminal': terminal,
        'raw_observation_after': after.tolist(),
        'environment_config': meta,
        'behavior_policy': meta.get('behavior_policy'),
        'rng_provenance': meta.get('rng'),
    }
