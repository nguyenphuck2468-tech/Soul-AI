"""G3.1 Pendulum contract and TRAIN-only JSONL loader."""
import hashlib
import json
import os
import stat
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


MAX_TRAIN_JSONL_BYTES = 4 * 1024 * 1024
MAX_TRAIN_JSONL_ROWS = 10_000


def _file_identity(metadata):
    return (
        metadata.st_dev, metadata.st_ino, metadata.st_size,
        metadata.st_mtime_ns, metadata.st_ctime_ns,
    )


def read_file_snapshot(path, max_bytes):
    """Read one bounded regular-file snapshot and hash the exact bytes returned."""
    source = Path(path)
    path_before = source.lstat()
    if not stat.S_ISREG(path_before.st_mode):
        raise ValueError(f'dataset source must be a regular non-symlink file: {source}')
    with source.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or _file_identity(before) != _file_identity(path_before):
            raise RuntimeError(f'dataset changed before it could be read: {source}')
        if before.st_size > max_bytes:
            raise ValueError(f'dataset exceeds the maximum snapshot size of {max_bytes} bytes: {source}')
        raw = stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    if len(raw) > max_bytes:
        raise ValueError(f'dataset exceeds the maximum snapshot size of {max_bytes} bytes: {source}')
    try:
        path_after = source.lstat()
    except OSError as error:
        raise RuntimeError(f'dataset path changed while it was being read: {source}') from error
    if (
        len(raw) != before.st_size
        or _file_identity(before) != _file_identity(after)
        or _file_identity(after) != _file_identity(path_after)
    ):
        raise RuntimeError(f'dataset changed while it was being read: {source}')
    return raw, hashlib.sha256(raw).hexdigest()


def load_train_jsonl_snapshot(path: str):
    """Parse and hash one bounded TRAIN snapshot; the digest describes these exact rows."""
    source = Path(path)
    raw, digest = read_file_snapshot(source, MAX_TRAIN_JSONL_BYTES)
    rows = []
    for line_no, line in enumerate(raw.decode('utf-8').splitlines(), 1):
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
        if len(rows) > MAX_TRAIN_JSONL_ROWS:
            raise ValueError(f'TRAIN JSONL exceeds the row limit of {MAX_TRAIN_JSONL_ROWS}: {source}')
    if not rows:
        raise ValueError(f'empty TRAIN JSONL: {source}')
    return rows, digest


def load_train_jsonl(path: str) -> List[Dict]:
    """Compatibility wrapper returning validated TRAIN rows from one snapshot."""
    rows, _ = load_train_jsonl_snapshot(path)
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
