"""G3.1 Pendulum contract and TRAIN-only JSONL loader."""
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from soul.memory import Episode
from soul.obs import ObsEncoder

PENDULUM_ENV_ID = 'Pendulum-v1'
PENDULUM_ACTION_DIM = 1
PENDULUM_OBS_SHAPE = (3,)
TRAIN_SEEDS = frozenset(range(1001, 1006))


def pendulum_model_config(cfg):
    """Return a copy-like config configured for the scalar Pendulum action."""
    cfg.model.action_dim = PENDULUM_ACTION_DIM
    cfg.model.discrete_dim = 0
    return cfg


def load_train_jsonl(path: str) -> List[Dict]:
    """Load only real TRAIN transitions; fail closed on split/schema mismatch."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f'real TRAIN JSONL not found: {source}')
    rows = []
    with source.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line)
            if row.get('split') != 'train':
                raise ValueError(f'non-TRAIN row at {source}:{line_no}')
            if row.get('episode_seed') not in TRAIN_SEEDS:
                raise ValueError(f'non-TRAIN seed at {source}:{line_no}')
            before = np.asarray(row['raw_observation_before'], dtype=np.float32)
            after = np.asarray(row['raw_observation_after'], dtype=np.float32)
            action = np.asarray(row['applied_action'], dtype=np.float32)
            if before.shape != PENDULUM_OBS_SHAPE or after.shape != PENDULUM_OBS_SHAPE:
                raise ValueError(f'bad observation shape at {source}:{line_no}')
            if action.shape != (PENDULUM_ACTION_DIM,):
                raise ValueError(f'bad applied action shape at {source}:{line_no}')
            if row.get('environment_config', {}).get('env_id') != PENDULUM_ENV_ID:
                raise ValueError(f'wrong environment at {source}:{line_no}')
            rows.append(row)
    if not rows:
        raise ValueError(f'empty TRAIN JSONL: {source}')
    return rows


def rows_to_episodes(rows: List[Dict], encoder: ObsEncoder,
                     goal_dim: int) -> List[Episode]:
    """Convert validated TRAIN rows without joining episode boundaries."""
    grouped = {}
    for row in rows:
        grouped.setdefault(row['episode_id'], []).append(row)
    episodes = []
    for episode_rows in grouped.values():
        ordered = sorted(episode_rows, key=lambda row: row['timestep'])
        if [row['timestep'] for row in ordered] != list(range(len(ordered))):
            raise ValueError('episode timesteps must be contiguous from zero')
        episode = Episode()
        for row in ordered:
            encoded = encoder.encode(row['raw_observation_after'])
            episode.append(
                encoded,
                np.asarray(row['applied_action'], dtype=np.float32),
                float(row['reward']),
                0.0 if row['terminal'] else 1.0,
                np.zeros(goal_dim, dtype=np.float32),
            )
        episodes.append(episode)
    return episodes
