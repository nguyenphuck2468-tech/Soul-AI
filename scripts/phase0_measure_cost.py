"""Measure MiniCraft's actual SoulAdapter call cost by rollout horizon."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import numpy as np
import torch

from soul.config import Config
from soul.soul import Soul
from wm_api import CostMeter, SoulAdapter


HORIZONS = (1, 5, 20, 100)


def measure(output: str | Path, warmup: int = 5, reps: int = 30,
            threads: int = 1, device: str | None = None) -> list[dict]:
    if threads < 1:
        raise ValueError('threads must be positive')
    if warmup < 0 or reps < 1:
        raise ValueError('warmup must be >= 0 and reps >= 1')
    torch.set_num_threads(threads)
    cfg = Config(device=device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cfg.model.action_dim = cfg.model.discrete_dim = 9
    cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
    cfg.train.seed = 0
    soul = Soul(cfg)
    soul._h = torch.zeros(1, cfg.model.hidden_dim, device=soul.device)
    soul._z = torch.zeros(1, cfg.model.latent_dim, device=soul.device)
    adapter = SoulAdapter(soul)
    snapshot = adapter.snapshot()
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    status = subprocess.run(['git', 'status', '--porcelain'], capture_output=True,
                            text=True, check=False)
    source_files = ('scripts/phase0_measure_cost.py', 'wm_api/cost.py',
                    'wm_api/soul_adapter.py', 'soul/model.py')
    source_hashes = {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                     for name in source_files}
    try:
        affinity = sorted(__import__('os').sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None

    records = []
    for horizon in HORIZONS:
        actions = [0, 1, 2, 3, 4, 5, 6, 7, 8] * (horizon // 9) + list(range(horizon % 9))
        meter = CostMeter(warmup=warmup, reps=reps)
        median, p95, samples = meter.measure(
            lambda: adapter.imagine(snapshot, actions))
        records.append({
            'schema_version': 1,
            'kind': 'phase0_cost_measurement',
            'component': 'SoulAdapter.for_minicraft.imagine',
            'status': 'EXPLORATORY_MEASURED',
            'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'horizon': horizon,
            'action_pattern': 'repeating 0..8',
            'warmup_reps': warmup, 'measurement_reps': reps,
            'median_seconds': float(median), 'p95_seconds': float(p95),
            'samples_seconds': [float(value) for value in samples],
            'input_shape': {
                'h': [1, cfg.model.hidden_dim], 'z': [1, cfg.model.latent_dim],
                'actions': [1, horizon, cfg.model.action_dim],
                'decoded_values': [horizon, cfg.obs.max_features],
            },
            'model_config': cfg.to_dict()['model'],
            'runtime': {
                'python': platform.python_version(), 'torch': torch.__version__,
                'numpy': np.__version__, 'device': str(soul.device),
                'torch_threads': torch.get_num_threads(),
                'cpu': platform.processor() or platform.uname().processor,
                'machine': platform.machine(), 'cpu_affinity': affinity,
            },
            'git_revision': revision,
            'working_tree_dirty': bool(status.stdout.strip()),
            'source_sha256': source_hashes,
        })

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='\n') as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False) + '\n')
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='results/phase0_cost.jsonl')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    try:
        records = measure(args.output, args.warmup, args.reps,
                          args.threads, args.device)
    except ValueError as exc:
        parser.error(str(exc))
    for record in records:
        print(f"H={record['horizon']:>3}: median={record['median_seconds']:.6f}s "
              f"p95={record['p95_seconds']:.6f}s")


if __name__ == '__main__':
    main()
