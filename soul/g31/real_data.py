"""Real G3.1 data collection from actual Pendulum-v1.

Creates disjoint TRAIN / PILOT / CONFIRM with deterministic seeds.
Records applied action, observation pairs, environment metadata.
Confirm untouched.
"""
import numpy as np
import gymnasium as gym
from pathlib import Path
import json
from soul.g31.collector import record_transition

ENV_ID = 'Pendulum-v1'
ENV_VERSION = 'gymnasium 1.4.0'
ACTION_BOUNDS = [-2.0, 2.0]


def collect_split(split, seed, episodes=5, steps_per_ep=20):
    seeds = list(range(seed, seed + episodes))
    rng = np.random.default_rng(seed)
    episodes = []
    for ep, episode_seed in enumerate(seeds):
        env = gym.make(ENV_ID)
        obs, info = env.reset(seed=episode_seed)
        transitions = []
        for t in range(steps_per_ep):
            # Behavior policy: bounded uniform over valid range
            requested_action = np.asarray([rng.uniform(-2.0, 2.0)], dtype=np.float32)
            applied_action = np.clip(requested_action, -2.0, 2.0)
            obs_before = np.array(obs, dtype=np.float32)
            obs_after, reward, terminated, truncated, info = env.step(applied_action)
            transitions.append(record_transition(
                f'{split}_ep{episode_seed}', split, episode_seed, t,
                obs_before, requested_action, applied_action,
                float(reward), bool(terminated or truncated),
                np.array(obs_after, dtype=np.float32).tolist(),
                {'env_id': ENV_ID, 'env_version': ENV_VERSION,
                 'action_bounds': ACTION_BOUNDS, 'seed': episode_seed,
                 'behavior_policy': 'bounded_uniform',
                 'requested_equals_applied': bool(np.array_equal(
                     requested_action, applied_action))}
            ))
            obs = obs_after
            if terminated or truncated:
                break
        env.close()
        episodes.append(transitions)
    return episodes


def run_collection():
    train = collect_split('train', 1001, episodes=5, steps_per_ep=20)
    pilot = collect_split('pilot', 2001, episodes=5, steps_per_ep=20)
    confirm = collect_split('confirm', 3001, episodes=5, steps_per_ep=20)
    return train, pilot, confirm


def persist_episodes(episodes, path):
    """Persist already collected transitions without mixing or overwriting."""
    target = Path(path)
    if target.exists():
        raise FileExistsError(f'refusing to overwrite existing dataset: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x', encoding='utf-8') as stream:
        for episode in episodes:
            for transition in episode:
                stream.write(json.dumps(transition, sort_keys=True) + '\n')
