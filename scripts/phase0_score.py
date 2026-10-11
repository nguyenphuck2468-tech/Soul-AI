"""Score MiniCraft world-model predictions on dev seeds.

Uses a train-only checkpoint whose encoder schema was frozen during training.
Reports model and persistence error per episode, plus an action-shuffle
diagnostic. No threshold is applied and uncertainty is explicitly unavailable.
"""

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess

import numpy as np
import torch

from minicraft import MiniCraftEnv
from soul.config import Config
from soul.soul import Soul
from wm_api.soul_adapter import SoulAdapter


DEV_MIN, DEV_MAX = 1000, 1999


def score(checkpoint: str | Path, output: str | Path, seeds: list[int],
          episodes_per_seed: int = 1, max_ticks: int = 100,
          shuffle_repeats: int = 5,
          raw_output: str | Path = 'results/phase0_dev_predictions.jsonl') -> dict:
    if not seeds or any(isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
                        or not DEV_MIN <= int(seed) <= DEV_MAX
                        for seed in seeds):
        raise ValueError('Phase 0 scorer accepts dev seeds 1000..1999 only')
    if len(set(map(int, seeds))) != len(seeds):
        raise ValueError('dev seed list must not contain duplicates')
    if episodes_per_seed < 1 or max_ticks < 2 or shuffle_repeats < 1:
        raise ValueError('episodes/repeats must be positive and max_ticks >= 2')
    if max(int(seed) + episodes_per_seed - 1 for seed in seeds) > DEV_MAX:
        raise ValueError('derived episode seeds must remain in the dev range 1000..1999')

    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = Config.from_dict(saved['config'])
    if cfg.model.action_dim != 9 or cfg.model.discrete_dim != 9:
        raise ValueError('checkpoint must use MiniCraft nine-way discrete actions')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    soul = Soul(cfg, checkpoint=str(checkpoint))
    adapter = SoulAdapter(soul)
    soul.model.eval()
    if not soul.enc.schema:
        raise ValueError('checkpoint encoder schema is empty; train checkpoint first')
    soul.enc.freeze_schema()

    feature_schema = sorted(soul.enc.schema.items(), key=lambda item: item[1])
    feature_names = {slot: name for name, slot in feature_schema
                     if slot < soul.cfg.obs.max_features}
    schema_hash = hashlib.sha256(
        json.dumps(feature_schema, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    ).hexdigest()
    numeric_indices: list[int] | None = None
    episodes = []
    raw_rows = []
    action_counts = np.zeros(9, dtype=np.int64)

    for seed in seeds:
        for episode_index in range(episodes_per_seed):
            env_seed = int(seed) + episode_index
            env = MiniCraftEnv(seed=env_seed,
                               max_ticks=max_ticks)
            obs = env.observe()
            predictions, targets, current_values, masks, actions = [], [], [], [], []
            contexts = []
            local_feature_indices: list[int] | None = None
            live_h, live_z = soul.model.dynamics.initial_state(1, soul.device)
            previous_action = None
            action_rng = np.random.default_rng(env_seed)
            for _ in range(max_ticks):
                current = soul.enc.encode(obs)
                current_vec = np.concatenate(
                    [current['values'], current['mask'], current['text']]).astype(np.float32)
                if numeric_indices is None:
                    numeric_indices = sorted(
                        slot for slot in soul.enc.schema.values()
                        if slot < soul.cfg.obs.max_features)
                    local_feature_indices = numeric_indices
                action = int(action_rng.integers(0, 9))
                action_counts[action] += 1
                next_obs, _, done, _ = env.step(action)
                next_encoded = soul.enc.encode(next_obs)
                target_vec = next_encoded['values'].astype(np.float32, copy=True)

                action_input = torch.as_tensor(
                    adapter._encode_action(action), device=soul.device
                ).unsqueeze(0)
                with torch.no_grad():
                    if previous_action is not None:
                        live_h, _, _ = soul.model.dynamics.step(
                            live_h, live_z, previous_action)
                    q_mu, _ = soul.model.encoder(torch.as_tensor(
                        current_vec, device=soul.device).unsqueeze(0))
                    live_z = q_mu
                    contexts.append((live_h.detach().clone(), live_z.detach().clone()))
                    h1, mu, _ = soul.model.dynamics.step(live_h, live_z, action_input)
                    pred = soul.model.decoder(h1, mu)['obs'].squeeze(0).cpu().numpy()

                predictions.append(pred)
                if local_feature_indices is None:
                    local_feature_indices = sorted(
                        slot for slot in soul.enc.schema.values()
                        if slot < soul.cfg.obs.max_features)
                targets.append(target_vec[:soul.cfg.obs.max_features])
                current_values.append(current['values'].copy())
                masks.append(next_encoded['mask'].copy())
                actions.append(action)
                previous_action = action_input
                obs = next_obs
                if done:
                    break

            pred = np.asarray(predictions, dtype=np.float32)
            target = np.asarray(targets, dtype=np.float32)
            current_values = np.asarray(current_values, dtype=np.float32)
            mask = np.asarray(masks, dtype=np.float32)
            if numeric_indices is None:
                raise RuntimeError('no numeric observation fields found')
            idx = np.asarray([i for i in (local_feature_indices or [])
                              if i < pred.shape[1]], dtype=np.int64)
            if idx.size == 0:
                raise ValueError('train-frozen encoder schema has no numeric fields')
            horizon_metrics = {}
            for horizon in (1, 5, 20, 100):
                model_errors, persistence_errors = [], []
                feature_model_sum = np.zeros(len(idx), dtype=np.float64)
                feature_persistence_sum = np.zeros(len(idx), dtype=np.float64)
                feature_counts = np.zeros(len(idx), dtype=np.int64)
                for start in range(max(0, len(actions) - horizon + 1)):
                    h_roll, z_roll = (contexts[start][0].clone(),
                                      contexts[start][1].clone())
                    prediction = None
                    with torch.no_grad():
                        for offset in range(horizon):
                            action_input = torch.as_tensor(
                                adapter._encode_action(actions[start + offset]),
                                device=soul.device).unsqueeze(0)
                            h_roll, z_roll, _ = soul.model.dynamics.step(
                                h_roll, z_roll, action_input)
                            prediction = soul.model.decoder(h_roll, z_roll)['obs']
                    prediction = prediction.squeeze(0).cpu().numpy()
                    target_index = start + horizon - 1
                    weights = mask[target_index, idx]
                    denom = max(float(weights.sum()), 1.0)
                    model_errors.append(float(
                        (((prediction[idx] - target[target_index, idx]) ** 2)
                         * weights).sum() / denom))
                    persistence_errors.append(float(
                        (((current_values[start, idx] - target[target_index, idx]) ** 2)
                         * weights).sum() / denom))
                    feature_model_sum += ((prediction[idx] - target[target_index, idx]) ** 2) * weights
                    feature_persistence_sum += np.square(
                        current_values[start, idx] - target[target_index, idx]) * weights
                    feature_counts += weights.astype(np.int64)
                    raw_rows.append({
                        'schema_version': 1, 'split': 'dev',
                        'seed': int(seed), 'environment_seed': env_seed,
                        'episode': episode_index, 'start_tick': start,
                        'horizon': horizon,
                        'actions': actions[start:start + horizon],
                        'feature_names': [feature_names.get(int(slot), f'feature_{slot}')
                                          for slot in idx],
                        'prediction': prediction[idx].astype(float).tolist(),
                        'target': target[target_index, idx].astype(float).tolist(),
                        'persistence': current_values[start, idx].astype(float).tolist(),
                        'observed_mask': weights.astype(int).tolist(),
                    })
                horizon_metrics[str(horizon)] = {
                    'origins': len(model_errors),
                    'model_mse': float(np.mean(model_errors)) if model_errors else None,
                    'persistence_mse': (float(np.mean(persistence_errors))
                                        if persistence_errors else None),
                    'per_feature': {
                        feature_names.get(int(slot), f'feature_{slot}'): {
                            'model_mse': (float(feature_model_sum[position] /
                                                feature_counts[position])
                                          if feature_counts[position] else None),
                            'persistence_mse': (float(feature_persistence_sum[position] /
                                                      feature_counts[position])
                                                if feature_counts[position] else None),
                            'observations': int(feature_counts[position]),
                        }
                        for position, slot in enumerate(idx)
                    },
                }
            weights = mask[:, idx]
            denom = np.maximum(weights.sum(axis=1), 1.0)
            rng = np.random.default_rng(int(seed) * 1009 + episode_index)
            shuffle_errors = []
            shuffled_actions = np.asarray(actions, dtype=np.int64)
            for _ in range(shuffle_repeats):
                permuted = shuffled_actions.copy()
                rng.shuffle(permuted)
                # Action-shuffle diagnostic is reported as a paired randomized
                # intervention in the same initial states, not as a baseline.
                alt_predictions = []
                for t, shuffled_action in enumerate(permuted):
                    act = torch.as_tensor(adapter._encode_action(int(shuffled_action)),
                                          device=soul.device).unsqueeze(0)
                    with torch.no_grad():
                        context_h, context_z = contexts[t]
                        h1, mu, _ = soul.model.dynamics.step(context_h, context_z, act)
                        alt_predictions.append(
                            soul.model.decoder(h1, mu)['obs'].squeeze(0).cpu().numpy())
                alt = np.asarray(alt_predictions, dtype=np.float32)
                per_step = (((alt[:, idx] - target[:, idx]) ** 2) * weights).sum(axis=1) / denom
                shuffle_errors.append(float(per_step.mean()))

            episodes.append({
                'seed': int(seed), 'episode': episode_index,
                'environment_seed': env_seed,
                'transitions': int(len(target)),
                'horizons': horizon_metrics,
                'action_shuffle_mse_mean': float(np.mean(shuffle_errors)),
                'action_shuffle_delta_vs_true': float(
                    np.mean(shuffle_errors) - horizon_metrics['1']['model_mse']),
                'action_shuffle_mse_samples': shuffle_errors,
            })

    report = {
        'schema_version': 1,
        'kind': 'phase0_dev_score',
        'status': 'EXPLORATORY_NO_THRESHOLDS_LOCKED',
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        'feature_schema_sha256': schema_hash,
        'unseen_feature_overflow': int(soul.enc.overflow),
        'split': 'dev', 'seeds': [int(seed) for seed in seeds],
        'code_revision': subprocess.run(
            ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
            check=False).stdout.strip() or None,
        'working_tree_dirty': bool(subprocess.run(
            ['git', 'status', '--porcelain'], capture_output=True, text=True,
            check=False).stdout.strip()),
        'source_sha256': {
            name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in ('scripts/phase0_score.py', 'soul/obs.py',
                         'soul/model.py', 'minicraft/env.py')
        },
        'runtime': {
            'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'device': str(soul.device),
            'torch_threads': torch.get_num_threads(),
        },
        'action_counts': action_counts.tolist(),
        'raw_predictions_path': str(raw_output),
        'episodes': episodes,
        'equal_episode_mean': {
            'horizons': {
                horizon: {
                    metric: (float(np.mean([ep['horizons'][horizon][metric]
                                            for ep in episodes
                                            if ep['horizons'][horizon][metric] is not None]))
                             if any(ep['horizons'][horizon][metric] is not None
                                    for ep in episodes) else None)
                    for metric in ('model_mse', 'persistence_mse')
                }
                for horizon in ('1', '5', '20', '100')
            },
            'action_shuffle_mse_mean': float(np.mean(
                [ep['action_shuffle_mse_mean'] for ep in episodes])),
            'action_shuffle_delta_vs_true': float(np.mean(
                [ep['action_shuffle_delta_vs_true'] for ep in episodes])),
        },
        'uncertainty': 'NOT_IMPLEMENTED',
        'limits': [
            'Structured-state prototype only; not pixel input or Dreamer 4.',
            'This is deterministic-mean open-loop prediction and does not establish adaptive-compute gains.',
            'Action-shuffle is diagnostic and has not been independently calibrated.',
        ],
    }
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = Path(raw_output)
    if raw_path.resolve() == path.resolve():
        raise ValueError('summary and raw prediction outputs must be different files')
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open('w', encoding='utf-8', newline='\n') as raw_file:
        for row in raw_rows:
            raw_file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    report['raw_predictions_sha256'] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', default='results/phase0_dev_score.json')
    parser.add_argument('--raw-output', default='results/phase0_dev_predictions.jsonl')
    parser.add_argument('--seed-start', type=int, default=1000)
    parser.add_argument('--seed-count', type=int, default=10)
    parser.add_argument('--episodes-per-seed', type=int, default=1)
    parser.add_argument('--max-ticks', type=int, default=100)
    args = parser.parse_args()
    if args.seed_count < 1:
        parser.error('--seed-count must be positive')
    try:
        result = score(args.checkpoint, args.output,
                       list(range(args.seed_start, args.seed_start + args.seed_count)),
                       args.episodes_per_seed, args.max_ticks,
                       raw_output=args.raw_output)
    except ValueError as exc:
        parser.error(str(exc))
    print(json.dumps(result['equal_episode_mean'], indent=2))


if __name__ == '__main__':
    main()
