"""Collect fixed-policy MiniCraft transitions for Phase 0 model fitting.

Only train seeds are accepted. Collection policy is uniform random and records
structured observations, integer actions, rewards, terminal flags, and seed
provenance. It does not train a model or touch dev/Confirm seeds.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from minicraft import MiniCraftEnv


TRAIN_MIN, TRAIN_MAX = 0, 999


def collect(path: str | Path, seeds: list[int], steps_per_seed: int,
            max_ticks: int = 300) -> int:
    if not seeds:
        raise ValueError('at least one train seed is required')
    if any(isinstance(seed, bool) or not TRAIN_MIN <= int(seed) <= TRAIN_MAX
           for seed in seeds):
        raise ValueError('Phase 0 collector accepts train seeds 0..999 only')
    if steps_per_seed < 1 or max_ticks < 1:
        raise ValueError('steps_per_seed and max_ticks must be positive')

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open('w', encoding='utf-8', newline='\n') as f:
        for seed in seeds:
            env = MiniCraftEnv(seed=int(seed), max_ticks=max_ticks)
            obs = env.observe()
            rng = np.random.default_rng(int(seed))
            episode = 0
            for index in range(steps_per_seed):
                if env.done:
                    env = MiniCraftEnv(seed=int(seed), max_ticks=max_ticks)
                    obs = env.observe()
                action = int(rng.integers(0, 9))
                next_obs, reward, done, info = env.step(action)
                record = {
                    'schema_version': 1,
                    'split': 'train',
                    'seed': int(seed),
                    'episode': episode,
                    'tick': int(info['tick']),
                    'observation': obs,
                    'action': action,
                    'reward': float(reward),
                    'done': bool(done),
                    'next_observation': next_obs,
                }
                record['observation'] = _jsonable(record['observation'])
                record['next_observation'] = _jsonable(record['next_observation'])
                f.write(json.dumps(record, allow_nan=False) + '\n')
                rows += 1
                obs = next_obs
                if done:
                    episode += 1
    return rows


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/phase0_train.jsonl')
    parser.add_argument('--seed-start', type=int, default=0)
    parser.add_argument('--seed-count', type=int, default=20)
    parser.add_argument('--steps-per-seed', type=int, default=300)
    parser.add_argument('--max-ticks', type=int, default=300)
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error('--seed-count must be positive')
    seeds = list(range(args.seed_start, args.seed_start + args.seed_count))
    try:
        rows = collect(args.output, seeds, args.steps_per_seed, args.max_ticks)
    except ValueError as exc:
        parser.error(str(exc))
    print(f'Wrote {rows} train transitions to {args.output}')


if __name__ == '__main__':
    main()
