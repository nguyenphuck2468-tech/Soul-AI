"""Fit the Phase 0 MiniCraft RSSM on audited, train-only collector JSONL."""

import argparse
import hashlib
import json
from collections import OrderedDict
from pathlib import Path
import subprocess

import numpy as np
import torch

from soul.config import Config
from soul.memory import Episode
from soul.soul import Soul
from minicraft import MiniCraftSafetyController


TRAIN_MIN, TRAIN_MAX = 0, 999


def _canonical_observation(observation) -> str:
    return json.dumps(observation, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _validate_transition_rows(rows: list[dict]) -> dict:
    """Validate safety logs and episode continuity before building replay."""
    safety = MiniCraftSafetyController()
    grouped = OrderedDict()
    for row_index, row in enumerate(rows):
        if row.get('schema_version') != 2:
            raise ValueError(f'transition {row_index} must use collector schema_version 2')
        candidate = row.get('candidate_action')
        applied = row.get('action')
        if (isinstance(candidate, bool) or not isinstance(candidate, int)
                or not 0 <= candidate < 9):
            raise ValueError(f'transition {row_index} has invalid candidate_action')
        if (isinstance(applied, bool) or not isinstance(applied, int)
                or not 0 <= applied < 9):
            raise ValueError(f'transition {row_index} has invalid applied action')
        expected = safety.filter(row.get('observation', {}), candidate)
        if applied != expected.applied_action:
            raise ValueError(
                f'transition {row_index} applied action does not match safety filter')
        safety_record = row.get('safety')
        if not isinstance(safety_record, dict):
            raise ValueError(f'transition {row_index} is missing safety decision')
        if (safety_record.get('blocked') is not expected.blocked
                or safety_record.get('reason') != expected.reason
                or safety_record.get('mode') != expected.mode
                or safety_record.get('allowed_actions') != list(expected.allowed_actions)
                or safety_record.get('expiry_tick') != expected.expiry_tick):
            raise ValueError(f'transition {row_index} safety decision record is inconsistent')
        if not isinstance(row.get('done'), bool):
            raise ValueError(f'transition {row_index} done flag must be boolean')
        if not isinstance(row.get('tick'), int) or isinstance(row.get('tick'), bool):
            raise ValueError(f'transition {row_index} tick must be an integer')
        if row['tick'] < 1:
            raise ValueError(f'transition {row_index} tick must be positive')
        key = (int(row['seed']), int(row['episode']))
        grouped.setdefault(key, []).append((row_index, row))

    for key, entries in grouped.items():
        for position, (row_index, row) in enumerate(entries):
            if position and entries[position - 1][1]['done']:
                raise ValueError(f'episode {key} contains transitions after done')
            if position:
                prev_row_index, previous = entries[position - 1]
                if row['tick'] != previous['tick'] + 1:
                    raise ValueError(
                        f'episode {key} has non-contiguous ticks at rows '
                        f'{prev_row_index} and {row_index}')
                if _canonical_observation(previous.get('next_observation')) != \
                        _canonical_observation(row.get('observation')):
                    raise ValueError(
                        f'episode {key} observation chain breaks at rows '
                        f'{prev_row_index} and {row_index}')
    return grouped


def train(data_path: str | Path, checkpoint_path: str | Path,
          updates: int = 1000, hidden_dim: int = 64, latent_dim: int = 16,
          seq_len: int = 16, batch_size: int = 16, seed: int = 0,
          device: str | None = None) -> dict:
    rows = [json.loads(line) for line in Path(data_path).read_text(
        encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise ValueError('training JSONL is empty')
    if any(row.get('split') != 'train'
           or not TRAIN_MIN <= int(row.get('seed', -1)) <= TRAIN_MAX
           for row in rows):
        raise ValueError('training input must contain train split seeds 0..999 only')
    validated_groups = _validate_transition_rows(rows)
    if updates < 1 or seq_len < 1 or batch_size < 1:
        raise ValueError('updates, seq_len and batch_size must be positive')

    cfg = Config(device=device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    cfg.model.action_dim, cfg.model.discrete_dim = 9, 9
    cfg.model.hidden_dim, cfg.model.latent_dim = hidden_dim, latent_dim
    cfg.train.seed = seed
    cfg.train.seq_len = seq_len
    cfg.train.batch_size = batch_size
    cfg.train.replay_capacity = max(len(rows) + 1, 100)
    cfg.train.warmup_updates = 0
    cfg.train.imagine_horizon = 4
    soul = Soul(cfg)

    # Establish feature slots from train observations only, before encoding
    # any transition into replay. The exact schema is persisted in checkpoint.
    for row in rows:
        soul.enc.encode(row['observation'])
        soul.enc.encode(row['next_observation'])

    accepted, skipped = 0, 0
    training_rows = []
    for entries in validated_groups.values():
        episode_rows = [row for _, row in entries]
        if len(episode_rows) < seq_len + 1:
            skipped += 1
            continue
        ep = Episode()
        for row in episode_rows:
            enc = soul.enc.encode(row['next_observation'])
            action = np.eye(9, dtype=np.float32)[int(row['action'])]
            ep.append(enc, action, float(row['reward']),
                      0.0 if row['done'] else 1.0,
                      np.zeros(cfg.model.goal_dim, dtype=np.float32))
        soul.replay.add_episode(ep)
        training_rows.extend(episode_rows)
        accepted += 1
    if not accepted:
        raise ValueError('no train episodes are long enough for seq_len + 1')

    losses = []
    for _ in range(updates):
        loss = soul.learn(force=True)
        if loss is None or not np.isfinite(loss['total']):
            raise RuntimeError('training produced no finite update')
        losses.append(loss)

    checkpoint = Path(checkpoint_path)
    soul.save(str(checkpoint))
    git_revision = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
        check=False).stdout.strip() or None
    git_status = subprocess.run(
        ['git', 'status', '--porcelain'], capture_output=True, text=True,
        check=False).stdout.strip()
    return {
        'schema_version': 2,
        'kind': 'phase0_train',
        'status': 'TRAIN_ONLY',
        'data_path': str(data_path),
        'data_sha256': hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        'checkpoint_path': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        'seed_min': min(int(row['seed']) for row in rows),
        'seed_max': max(int(row['seed']) for row in rows),
        'code_revision': git_revision,
        'working_tree_dirty': bool(git_status),
        'working_tree_status_sha256': hashlib.sha256(
            git_status.encode('utf-8')).hexdigest() if git_status else None,
        'protocol_config_sha256': hashlib.sha256(
            Path('configs/phase0_protocol_draft.yaml').read_bytes()).hexdigest(),
        'source_sha256': {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ('scripts/phase0_train.py', 'soul/obs.py',
                         'soul/trainer.py', 'soul/model.py',
                         'scripts/phase0_collect.py', 'minicraft/safety.py',
                         'configs/phase0_protocol_draft.yaml')
        },
        'action_counts': np.bincount([int(row['action']) for row in rows], minlength=9).tolist(),
        'candidate_action_counts': np.bincount(
            [int(row['candidate_action']) for row in rows], minlength=9).tolist(),
        'applied_action_counts': np.bincount(
            [int(row['action']) for row in rows], minlength=9).tolist(),
        'safety_blocks': int(sum(bool(row['safety']['blocked']) for row in rows)),
        'transitions': len(rows),
        'transitions_used_for_fit': len(training_rows),
        'training_candidate_action_counts': np.bincount(
            [int(row['candidate_action']) for row in training_rows], minlength=9).tolist(),
        'training_applied_action_counts': np.bincount(
            [int(row['action']) for row in training_rows], minlength=9).tolist(),
        'training_safety_blocks': int(sum(
            bool(row['safety']['blocked']) for row in training_rows)),
        'episodes_loaded': accepted,
        'episodes_skipped_short': skipped,
        'updates': len(losses),
        'final_losses': losses[-1],
        'feature_schema_sha256': hashlib.sha256(json.dumps(
            sorted(soul.enc.schema.items(), key=lambda item: item[1]),
            ensure_ascii=False, separators=(',', ':')).encode('utf-8')).hexdigest(),
        'config': cfg.to_dict(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='results/phase0_train.jsonl')
    parser.add_argument('--checkpoint', default='results/phase0_soul.pt')
    parser.add_argument('--report', default='results/phase0_train_report.json')
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--latent-dim', type=int, default=16)
    parser.add_argument('--seq-len', type=int, default=16)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    result = train(args.data, args.checkpoint, args.updates, args.hidden_dim,
                   args.latent_dim, args.seq_len, args.batch_size, args.seed,
                   args.device)
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, ensure_ascii=False,
                                 allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in
                      ('transitions', 'episodes_loaded', 'updates', 'checkpoint_sha256', 'final_losses')},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
