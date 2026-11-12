"""G3.1 Pilot scorer. Reads Pilot + TRAIN only; never opens Confirm."""
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from soul.config import Config
from soul.g31.evaluator import (
    evaluate_one_step,
    prime_episode_state,
    physical_metrics,
    raw_observation_to_physical,
)
from soul.g31.train_data import pendulum_model_config
from soul.model import SoulModel
from soul.obs import ObsEncoder

PILOT_PATH = Path('artifacts/g31/pilot.jsonl')
TRAIN_PATH = Path('artifacts/g31/train.jsonl')
DEFAULT_CHECKPOINT_PATH = Path('artifacts/g31/g31_verified_train_checkpoint.pt')
DEFAULT_PREDICTIONS_PATH = Path('artifacts/g31/pilot_predictions.jsonl')
DEFAULT_METRICS_PATH = Path('artifacts/g31/pilot_metrics.json')
DEFAULT_REPORT_PATH = Path('artifacts/g31/pilot_report.json')

# Default A checkpoint (preserved for backward compatibility)
PILOT_HASH = '26d63b1e3a900d40a9532aa5b2e4c178b58129e88f70cb2cb9a6e07af596264e'
CHECKPOINT_HASH = '0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path, split, seeds):
    rows = [json.loads(line) for line in path.open(encoding='utf-8')]
    if any(row.get('split') != split for row in rows):
        raise ValueError(f'{path} contains a non-{split} row')
    if {row.get('episode_seed') for row in rows} != set(seeds):
        raise ValueError(f'{path} has unexpected seeds')
    return rows


def fit_train_mean_delta():
    rows = load_rows(TRAIN_PATH, 'train', range(1001, 1006))
    deltas = np.asarray([
        raw_observation_to_physical(row['raw_observation_after'])
        - raw_observation_to_physical(row['raw_observation_before'])
        for row in rows
    ], dtype=np.float64)
    deltas[:, 0] = np.arctan2(np.sin(deltas[:, 0]), np.cos(deltas[:, 0]))
    return np.mean(deltas, axis=0)


def episode_metrics(records, prediction_key):
    grouped = defaultdict(list)
    for record in records:
        grouped[record["episode_id"]].append(record)
    result = {}
    for episode_id, rows in grouped.items():
        pred = np.asarray([r[prediction_key] for r in rows])
        target = np.asarray([r['target_physical'] for r in rows])
        result[episode_id] = physical_metrics(pred, target)
    return result


def paired_differences(model_metrics, baseline_metrics):
    keys = sorted(model_metrics)
    metric_names = sorted(model_metrics[keys[0]])
    return {
        metric: {
            'mean': float(np.mean([model_metrics[k][metric] - baseline_metrics[k][metric] for k in keys])),
            'median': float(np.median([model_metrics[k][metric] - baseline_metrics[k][metric] for k in keys])),
            'dispersion': float(np.std([model_metrics[k][metric] - baseline_metrics[k][metric] for k in keys])),
        }
        for metric in metric_names
    }


def aggregate_episode_metrics(metrics_by_episode):
    episode_values = list(metrics_by_episode.values())
    names = sorted(episode_values[0])
    return {
        name: {
            'mean': float(np.mean([metrics[name] for metrics in episode_values])),
            'median': float(np.median([metrics[name] for metrics in episode_values])),
            'dispersion': float(np.std([metrics[name] for metrics in episode_values])),
            'episode_count': len(episode_values),
        }
        for name in names
    }


def _same_output_path(left, right):
    """Detect path aliases, including symlinks and existing hard links."""
    if left.resolve() == right.resolve():
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def validate_output_paths(outputs, protected_inputs):
    """Prevent outputs from aliasing each other or source evidence."""
    for index, output in enumerate(outputs):
        for other in outputs[index + 1:]:
            if _same_output_path(output, other):
                raise ValueError(f'Output paths must be distinct: {output} and {other}')
        for source in protected_inputs:
            if _same_output_path(output, source):
                raise ValueError(f'Output path must not overwrite input evidence: {output}')


def run(checkpoint_path=None, predictions_path=None, metrics_path=None,
        report_path=None, allow_overwrite=False):
    # Resolve paths
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else DEFAULT_CHECKPOINT_PATH
    predictions_path = Path(predictions_path) if predictions_path else DEFAULT_PREDICTIONS_PATH
    metrics_path = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH
    report_path = Path(report_path) if report_path else DEFAULT_REPORT_PATH

    # Never let outputs alias each other or overwrite source evidence, even with
    # allow_overwrite=True. This protects the frozen checkpoint and evaluation inputs.
    validate_output_paths(
        (predictions_path, metrics_path, report_path),
        (checkpoint_path, PILOT_PATH, TRAIN_PATH),
    )

    # Isolation guard: refuse to overwrite existing artifacts unless allowed
    for p in (predictions_path, metrics_path, report_path):
        if p.exists() and not allow_overwrite:
            raise FileExistsError(
                f'Output artifact already exists (isolation guard): {p}. '
                f'Use a different output prefix or set allow_overwrite=True.'
            )

    # Verify Pilot dataset identity (same for A or B1)
    if sha256(PILOT_PATH) != PILOT_HASH:
        raise RuntimeError('Pilot dataset hash mismatch')

    # Load checkpoint and compute runtime hash (never hardcoded)
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    checkpoint_sha256 = sha256(checkpoint_path)

    train_delta = fit_train_mean_delta()
    pilot_rows = load_rows(PILOT_PATH, 'pilot', range(2001, 2006))
    cfg = Config.from_dict(checkpoint['config_dict'])
    cfg = pendulum_model_config(cfg)
    encoder = ObsEncoder(cfg.obs)
    encoder.load_schema(checkpoint['enc_schema'])
    model = SoulModel(cfg, encoder.input_dim)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}

    grouped = defaultdict(list)
    for row in pilot_rows:
        grouped[row["episode_id"]].append(row)
    records = []
    for episode_id, rows in grouped.items():
        rows.sort(key=lambda row: row['timestep'])
        if len(rows) < 2:
            raise ValueError(f'{episode_id}: at least two rows are required for aligned scoring')
        if [row['timestep'] for row in rows] != list(range(len(rows))):
            raise ValueError(f'{episode_id}: timesteps must be contiguous from zero')
        # Dataset contract: each row's before-state must equal the previous
        # row's after-state. Fail closed instead of silently scoring a broken sequence.
        for previous, current_row in zip(rows, rows[1:]):
            if not np.allclose(
                previous['raw_observation_after'],
                current_row['raw_observation_before'],
                rtol=0.0, atol=1e-6,
            ):
                raise ValueError(f'{episode_id}: discontinuity before timestep {current_row["timestep"]}')
        # The first recorded row is the training-sequence warm-up: training
        # initializes h with (zero latent, zero action), then observes its
        # after-state. It has no trained prior target, so prime from that row
        # and score only rows 1..N-1.
        state = prime_episode_state(
            model, rows[0]['raw_observation_after'], encoder, device='cpu'
        )
        for row in rows[1:]:
            current = np.asarray(row['raw_observation_before'], dtype=np.float32)
            target = raw_observation_to_physical(row['raw_observation_after'])
            # Diagnostic only: hold the recurrent context fixed and compare the
            # learned prior with the action input zeroed. This is not a new
            # baseline or a causal claim; it measures checkpoint sensitivity
            # to its action channel on the same held-out PILOT transitions.
            zero_action = np.zeros_like(np.asarray(row['applied_action'], dtype=np.float32))
            zero_action_prediction, _ = evaluate_one_step(
                model, state, current, zero_action,
                obs_next=None,
                feature_encoder=encoder,
            )
            predicted, state = evaluate_one_step(
                model, state, current, row['applied_action'],
                obs_next=row['raw_observation_after'],
                feature_encoder=encoder,
            )
            current_physical = raw_observation_to_physical(current)
            persistence = current_physical
            mean_delta = current_physical + train_delta
            mean_delta[0] = current_physical[0] + train_delta[0]
            mean_delta[0] = np.arctan2(np.sin(mean_delta[0]), np.cos(mean_delta[0]))
            if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(zero_action_prediction)):
                raise ValueError('nonfinite physical prediction')
            records.append({
                'episode_id': episode_id,
                'timestep': row['timestep'],
                'applied_action': row['applied_action'],
                'target_physical': target.tolist(),
                'predicted_physical': predicted.tolist(),
                'zero_action_physical': zero_action_prediction.tolist(),
                'persistence_physical': persistence.tolist(),
                'mean_delta_physical': mean_delta.tolist(),
                'checkpoint_sha256': checkpoint_sha256,
                'pilot_sha256': PILOT_HASH,
            })
    after = model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)

    model_ep = episode_metrics(records, 'predicted_physical')
    persistence_ep = episode_metrics(records, 'persistence_physical')
    mean_ep = episode_metrics(records, 'mean_delta_physical')
    zero_action_ep = episode_metrics(records, 'zero_action_physical')
    metrics = {
        'episode_count': len(model_ep),
        'transition_count': len(records),
        'model': model_ep,
        'model_aggregate_equal_episode': aggregate_episode_metrics(model_ep),
        'persistence': persistence_ep,
        'persistence_aggregate_equal_episode': aggregate_episode_metrics(persistence_ep),
        'train_only_mean_delta': mean_ep,
        'zero_action_ablation': zero_action_ep,
        'zero_action_aggregate_equal_episode': aggregate_episode_metrics(zero_action_ep),
        'mean_delta_aggregate_equal_episode': aggregate_episode_metrics(mean_ep),
        'paired_model_minus_persistence': paired_differences(model_ep, persistence_ep),
        'paired_model_minus_mean_delta': paired_differences(model_ep, mean_ep),
        'paired_model_minus_zero_action': paired_differences(model_ep, zero_action_ep),
        'threshold': 'TO_LOCK',
        'pilot_scoring': 'EXPLORATORY; no confirm run',
        'checkpoint_path': str(checkpoint_path),
        'checkpoint_sha256': checkpoint_sha256,
    }
    report = {
        'dataset': str(PILOT_PATH), 'pilot_sha256': PILOT_HASH,
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_sha256,
        'environment': 'Pendulum-v1 / Gymnasium 1.4.0',
        'episodes': len(model_ep), 'transitions': len(records),
        'recorded_transitions': len(pilot_rows),
        'warmup_transitions_skipped': len(model_ep),
        'evaluator': 'training-aligned warm start + prior mean + physical decode + real obs teacher forcing',
        'action_ablation': {
            'kind': 'zero-action sensitivity diagnostic',
            'counterfactual': 'replace applied action with zero',
            'held_fixed': ['recurrent state', 'current observation', 'checkpoint', 'PILOT transitions'],
            'interpretation': 'sensitivity only; not a causal claim or gate baseline',
            'optimizer_updates': 0,
        },
        'confirm_accessed': False, 'optimizer_updates': 0,
        'deterministic': True, 'model_immutable': True,
        'threshold': 'TO_LOCK', 'gate': 'INCONCLUSIVE',
        'note': 'Pilot metrics are exploratory; no threshold was locked and Confirm was not run.',
    }
    # Stage all three files beside their destinations before replacing any output.
    # A failed serialization/write therefore cannot leave a partially written artifact.
    payloads = {
        predictions_path: ''.join(
            json.dumps(record, sort_keys=True) + '\\n' for record in records
        ),
        metrics_path: json.dumps(metrics, indent=2, sort_keys=True),
        report_path: json.dumps(report, indent=2, sort_keys=True),
    }
    staged = {}
    try:
        for destination, rendered in payloads.items():
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode='w',
                encoding='utf-8',
                dir=destination.parent,
                prefix=f'.{destination.name}.',
                suffix='.tmp',
                delete=False,
            ) as stream:
                staged[destination] = Path(stream.name)
                stream.write(rendered)
                stream.write('\\n')
                stream.flush()
                os.fsync(stream.fileno())
        for destination, temporary in staged.items():
            os.replace(temporary, destination)
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
    return metrics


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument('--predictions', type=str, default=str(DEFAULT_PREDICTIONS_PATH))
    parser.add_argument('--metrics', type=str, default=str(DEFAULT_METRICS_PATH))
    parser.add_argument('--report', type=str, default=str(DEFAULT_REPORT_PATH))
    parser.add_argument('--allow-overwrite', action='store_true')
    args = parser.parse_args()
    result = run(
        checkpoint_path=args.checkpoint,
        predictions_path=args.predictions,
        metrics_path=args.metrics,
        report_path=args.report,
        allow_overwrite=args.allow_overwrite,
    )
    print(json.dumps({
        'episodes': result['episode_count'],
        'transitions': result['transition_count'],
        'threshold': result['threshold'],
        'gate': 'INCONCLUSIVE',
        'checkpoint_sha256': result['checkpoint_sha256'],
        'checkpoint_path': result['checkpoint_path'],
    }, indent=2))
