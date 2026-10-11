"""Measure prior-imagination latency for the current Soul WM adapter.

The workload uses the current Pendulum configuration's observation/model
dimensions and a deterministic synthetic latent snapshot. Skill and safety
gate costs are emitted as explicitly deferred records until phases D and F
provide those components.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from soul.config import Config
from soul.soul import Soul
from wm_api import CostMeter, SoulAdapter


HORIZONS = (1, 2, 4, 8, 16, 32)


def _revision():
    result = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True,
        text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _source_metadata():
    root = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ['git', 'status', '--porcelain'], capture_output=True,
        text=True, check=False,
    )
    names = ('scripts/measure_cost.py', 'wm_api/cost.py',
             'wm_api/soul_adapter.py', 'soul/model.py')
    return {
        'git_revision': _revision(),
        'working_tree_dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
        'source_file_sha256': {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in names
        },
    }


def _hardware(device):
    gpu = None
    if device.type == 'cuda' and torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(device)
    try:
        affinity = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = None
    return {
        'cpu': platform.processor() or platform.uname().processor or 'unknown',
        'machine': platform.machine(),
        'gpu': gpu,
        'torch_device': str(device),
        'torch_num_threads': torch.get_num_threads(),
        'torch_num_interop_threads': torch.get_num_interop_threads(),
        'process_cpu_affinity': affinity,
        'thread_environment': {
            name: os.environ.get(name)
            for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
        },
        'torch_version': torch.__version__,
        'numpy_version': np.__version__,
        'python_version': platform.python_version(),
    }


def _measure_imagine(adapter, snapshot, horizon, meter, thread_mode):
    actions = [0] * horizon
    fn = lambda: adapter.imagine(snapshot, actions)
    median_s, p95_s, all_s = meter.measure(fn)
    rel_max = max(abs(sample - median_s) for sample in all_s) / max(median_s, 1e-12)
    return {
        'schema_version': 1,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'cost_measurement',
        'component': 'wm_api.soul_adapter.imagine',
        'status': 'measured',
        'horizon': horizon,
        'input_shape': {
            'h': [1, adapter.soul.cfg.model.hidden_dim],
            'z': [1, adapter.soul.cfg.model.latent_dim],
            'actions': [1, horizon, adapter.soul.cfg.model.action_dim],
            'decoded_values': [horizon, adapter.soul.cfg.obs.max_features],
        },
        'warmup_reps': meter.warmup,
        'reps': meter.reps,
        'median_seconds': median_s,
        'p95_seconds': p95_s,
        'max_abs_deviation_from_median_ratio': rel_max,
        'stability_threshold_ratio': 0.20,
        'stability_pass': rel_max < 0.20,
        'thread_mode': thread_mode,
        'all_seconds_sorted': all_s,
        'hardware': _hardware(adapter.soul.device),
        **_source_metadata(),
    }


def run(out_path, warmup=5, reps=30, threads=0, interop_threads=0):
    if threads < 0 or interop_threads < 0:
        raise ValueError('thread counts must be zero (leave unchanged) or positive.')
    if threads:
        torch.set_num_threads(threads)
    if interop_threads and torch.get_num_interop_threads() != interop_threads:
        torch.set_num_interop_threads(interop_threads)

    cfg = Config()
    cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
    cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
    cfg.train.batch_size, cfg.train.seq_len = 8, 16
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    soul = Soul(cfg)
    soul._h = torch.zeros(1, cfg.model.hidden_dim, device=soul.device)
    soul._z = torch.zeros(1, cfg.model.latent_dim, device=soul.device)
    adapter = SoulAdapter(soul)
    snapshot = adapter.snapshot()
    meter = CostMeter(warmup=warmup, reps=reps)

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for horizon in HORIZONS:
        record = _measure_imagine(adapter, snapshot, horizon, meter, 'initial')
        records.append(record)
        if not record['stability_pass'] and torch.get_num_threads() != 1:
            previous = torch.get_num_threads()
            torch.set_num_threads(1)
            retry_meter = CostMeter(warmup=warmup, reps=reps)
            retry = _measure_imagine(
                adapter, snapshot, horizon, retry_meter,
                f'retry_after_{previous}_threads_with_1_thread',
            )
            records.append(retry)

    deferred = {
        'schema_version': 1,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'kind': 'cost_measurement',
        'component': 'phase_d_skill_step',
        'status': 'deferred_prerequisite_missing',
        'reason': 'Phase D skill component is not implemented yet.',
        'hardware': _hardware(soul.device),
        **_source_metadata(),
    }
    deferred_gate = {
        **deferred,
        'component': 'phase_f_detector_and_gate',
        'reason': 'Phase F detector/gate components are not implemented yet.',
    }
    records.extend((deferred, deferred_gate))
    with path.open('a', encoding='utf-8', newline='\n') as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False) + '\n')

    for record in records:
        if record['status'] == 'measured':
            print(f"H={record['horizon']:>2}: median="
                  f"{record['median_seconds']:.6f}s p95="
                  f"{record['p95_seconds']:.6f}s stability="
                  f"{'PASS' if record['stability_pass'] else 'FAIL'} "
                  f"({record['thread_mode']})")
        else:
            print(f"{record['component']}: {record['status']} — "
                  f"{record['reason']}")
    print(f'JSONL: {os.fspath(path)}')
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='results/A5_cost.jsonl')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--reps', type=int, default=30)
    parser.add_argument(
        '--threads', type=int, default=0,
        help='set torch CPU thread count; 0 keeps current setting',
    )
    parser.add_argument(
        '--interop-threads', type=int, default=0,
        help='set torch inter-op thread count; 0 keeps current setting',
    )
    args = parser.parse_args(argv)
    if args.warmup < 0 or args.reps < 1:
        parser.error('--warmup must be >= 0 and --reps must be >= 1.')
    if args.threads < 0 or args.interop_threads < 0:
        parser.error('--threads and --interop-threads must be >= 0.')
    run(args.out, warmup=args.warmup, reps=args.reps,
        threads=args.threads, interop_threads=args.interop_threads)


if __name__ == '__main__':
    sys.exit(main())
