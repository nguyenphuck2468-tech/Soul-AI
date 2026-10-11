"""Run the lightweight Phase 0 alpha pipeline and save a smoke manifest.

This is a run-through helper, not a formal experiment. Defaults are deliberately
small and write under ``work/`` so exploratory artifacts do not mix with locked
results.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output_dir: str | Path | None = None, *,
        train_seed_start: int = 0, train_seed_count: int = 5,
        steps_per_seed: int = 50, max_ticks: int = 50,
        updates: int = 2, hidden_dim: int = 32, latent_dim: int = 8,
        seq_len: int = 8, batch_size: int = 4, train_seed: int = 0,
        dev_seed_start: int = 1003, dev_seed_count: int = 1,
        dev_max_ticks: int = 20, shuffle_repeats: int = 1,
        cost_warmup: int = 0, cost_reps: int = 1,
        threads: int = 1, device: str = 'cpu') -> dict:
    """Collect, fit, score, and measure cost with small alpha defaults."""
    if train_seed_count < 1 or dev_seed_count < 1:
        raise ValueError('train and dev seed counts must be positive')
    if train_seed_start < 0 or train_seed_start + train_seed_count - 1 > 999:
        raise ValueError('train seed range must stay within 0..999')
    if dev_seed_start < 1000 or dev_seed_start + dev_seed_count - 1 > 1999:
        raise ValueError('dev seed range must stay within 1000..1999')
    if train_seed_start + train_seed_count - 1 >= dev_seed_start:
        raise ValueError('train and dev seed ranges must not overlap')

    # Keep lightweight CLI operations such as --help responsive on machines
    # where importing PyTorch is slow.
    from scripts.phase0_collect import collect
    from scripts.phase0_train import train
    from scripts.phase0_score import score
    from scripts.phase0_measure_cost import measure

    if output_dir is None:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = Path('work') / 'phase0_alpha_runs' / f'alpha-{run_id}'
    else:
        output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    data_path = output / 'train.jsonl'
    checkpoint_path = output / 'soul.pt'
    train_report_path = output / 'train_report.json'
    score_path = output / 'dev_score.json'
    predictions_path = output / 'dev_predictions.jsonl'
    cost_path = output / 'cost.jsonl'
    manifest_path = output / 'manifest.json'

    train_seeds = list(range(train_seed_start,
                             train_seed_start + train_seed_count))
    dev_seeds = list(range(dev_seed_start, dev_seed_start + dev_seed_count))
    transition_count = collect(data_path, train_seeds, steps_per_seed, max_ticks)
    print(f'Collected {transition_count} transitions across {len(train_seeds)} train seeds.')

    train_report = train(
        data_path, checkpoint_path, updates=updates, hidden_dim=hidden_dim,
        latent_dim=latent_dim, seq_len=seq_len, batch_size=batch_size,
        seed=train_seed, device=device)
    train_report_path.write_text(
        json.dumps(train_report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    print(f'Fit {train_report["updates"]} updates; checkpoint {train_report["checkpoint_sha256"][:12]}.')

    score_report = score(
        checkpoint_path, score_path, dev_seeds, max_ticks=dev_max_ticks,
        shuffle_repeats=shuffle_repeats, raw_output=predictions_path)
    print(f'Scored {len(dev_seeds)} dev seeds (exploratory only).')

    cost_records = measure(
        cost_path, warmup=cost_warmup, reps=cost_reps,
        threads=threads, device=device)
    print(f'Measured adapter cost at {len(cost_records)} horizons.')

    revision = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
        check=False).stdout.strip() or None
    artifacts = {
        path.name: _sha256(path)
        for path in (data_path, checkpoint_path, train_report_path, score_path,
                     predictions_path, cost_path)
    }
    manifest = {
        'schema_version': 1,
        'kind': 'phase0_alpha_run',
        'status': 'NON_EVIDENTIARY_SMOKE_ONLY',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_revision': revision,
        'train_seeds': train_seeds,
        'dev_seeds': dev_seeds,
        'confirm_accessed': False,
        'transitions_collected': transition_count,
        'episodes_fit': train_report['episodes_loaded'],
        'transitions_used_for_fit': train_report['transitions_used_for_fit'],
        'updates': train_report['updates'],
        'cost_warmup_reps': cost_warmup,
        'cost_measurement_reps': cost_reps,
        'device': device,
        'score_status': score_report['status'],
        'artifacts_sha256': artifacts,
        'limits': [
            'Alpha run-through only; not a formal Phase 0 experiment.',
            'Small data and update counts are for execution checks, not model claims.',
            'Dev seeds are exploratory and are marked as seen after this run.',
            'Cost repetitions are not suitable for latency conclusions at the default.',
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    print(f'Alpha run complete. Manifest: {manifest_path}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--train-seed-start', type=int, default=0)
    parser.add_argument('--train-seed-count', type=int, default=5)
    parser.add_argument('--steps-per-seed', type=int, default=50)
    parser.add_argument('--max-ticks', type=int, default=50)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--hidden-dim', type=int, default=32)
    parser.add_argument('--latent-dim', type=int, default=8)
    parser.add_argument('--seq-len', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--train-seed', type=int, default=0)
    parser.add_argument('--dev-seed-start', type=int, default=1003)
    parser.add_argument('--dev-seed-count', type=int, default=1)
    parser.add_argument('--dev-max-ticks', type=int, default=20)
    parser.add_argument('--shuffle-repeats', type=int, default=1)
    parser.add_argument('--cost-warmup', type=int, default=0)
    parser.add_argument('--cost-reps', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    try:
        run(**vars(args))
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
