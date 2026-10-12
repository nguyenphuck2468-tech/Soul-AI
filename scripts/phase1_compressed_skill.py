"""Run the alpha compressed observation-only skill."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from minicraft import MiniCraftEnv
from minicraft.compressed_skill import CompressedObservationSkill
from minicraft.distilled_skill import fit_decision_tree, observation_features
from minicraft.experts import MiniCraftExpert
from minicraft.runtime import CompressedSkillRuntime


TASKS = ('T_wood', 'T_tools', 'T_iron')
EXPECTED_STATUS = 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION'
RUNTIME_COMPONENTS = (
    'skill_proposal', 'safety_filter', 'environment_step',
    'expectation_monitor', 'integrated',
)


def _validate_seed_list(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f'{label} must be a non-empty list')
    if any(isinstance(seed, bool) or not isinstance(seed, int)
           or not 0 <= seed <= 999 for seed in value):
        raise ValueError(f'{label} must contain train seeds 0..999')
    if len(set(value)) != len(value):
        raise ValueError(f'{label} must not contain duplicates')
    return value


def _validate_fraction(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be numeric')
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f'{label} must be finite and in [0, 1]')
    return number


def _validate_nonnegative_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be numeric')
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f'{label} must be finite and non-negative')
    return number


def validate_manifest(output_dir: str | Path) -> dict:
    """Fail closed if a compressed-skill run is incomplete or tampered with."""
    output = Path(output_dir)
    aggregate_path = output / 'manifest.json'
    try:
        aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid compressed-skill aggregate manifest: {exc}') from exc
    if aggregate.get('kind') != 'phase1_compressed_skill_alpha':
        raise ValueError('compressed-skill aggregate kind mismatch')
    if aggregate.get('status') != EXPECTED_STATUS:
        raise ValueError('compressed-skill run must remain exploratory/no-promotion')
    if aggregate.get('schema_version') != 1:
        raise ValueError('compressed-skill aggregate schema mismatch')
    aggregate_train_seeds = _validate_seed_list(
        aggregate.get('train_seeds'), label='aggregate train_seeds')
    aggregate_evaluation_seeds = _validate_seed_list(
        aggregate.get('evaluation_seeds'), label='aggregate evaluation_seeds')
    if set(aggregate_train_seeds) & set(aggregate_evaluation_seeds):
        raise ValueError('aggregate train/evaluation seeds must be disjoint')
    if not isinstance(aggregate.get('tasks'), dict):
        raise ValueError('compressed-skill aggregate tasks are required')
    for task in TASKS:
        entry = aggregate.get('tasks', {}).get(task)
        if not isinstance(entry, dict):
            raise ValueError(f'missing compressed-skill task entry: {task}')
        for field in (
                'success_fraction', 'learned_agreement_fraction',
                'runtime_seconds_per_step', 'manifest_path',
                'manifest_sha256'):
            if field not in entry:
                raise ValueError(f'{task}: aggregate field is required: {field}')
        _validate_fraction(
            entry['success_fraction'], label=f'{task} aggregate success_fraction')
        _validate_fraction(
            entry['learned_agreement_fraction'],
            label=f'{task} aggregate learned_agreement_fraction')
        manifest_path = Path(entry['manifest_path'])
        if not manifest_path.is_absolute():
            manifest_path = output / task / manifest_path.name
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != entry.get('manifest_sha256'):
            raise ValueError(f'{task}: task manifest hash mismatch')
        manifest = json.loads(manifest_bytes.decode('utf-8'))
        if manifest.get('schema_version') != 1:
            raise ValueError(f'{task}: task manifest schema mismatch')
        if manifest.get('status') != EXPECTED_STATUS:
            raise ValueError(f'{task}: task manifest status mismatch')
        for field in (
                'task', 'train_seeds', 'evaluation_seeds', 'episodes',
                'success_fraction', 'learned_agreement_fraction',
                'runtime_seconds', 'runtime_seconds_per_step',
                'safety_blocks', 'expectation_checks',
                'expectation_mismatches', 'skill_path', 'skill_sha256'):
            if field not in manifest:
                raise ValueError(f'{task}: task manifest field is required: {field}')
        if manifest['task'] != task:
            raise ValueError(f'{task}: task manifest task mismatch')
        _validate_fraction(
            manifest['success_fraction'], label=f'{task} success_fraction')
        _validate_fraction(
            manifest['learned_agreement_fraction'],
            label=f'{task} learned_agreement_fraction')
        train_seeds = _validate_seed_list(
            manifest['train_seeds'], label=f'{task} train_seeds')
        evaluation_seeds = _validate_seed_list(
            manifest['evaluation_seeds'], label=f'{task} evaluation_seeds')
        if train_seeds != aggregate_train_seeds:
            raise ValueError(f'{task}: train seed provenance mismatch')
        if evaluation_seeds != aggregate_evaluation_seeds:
            raise ValueError(f'{task}: evaluation seed provenance mismatch')
        if set(train_seeds) & set(evaluation_seeds):
            raise ValueError(f'{task}: train/evaluation seeds must be disjoint')
        skill_path = Path(manifest['skill_path'])
        if not skill_path.is_absolute():
            skill_path = output / task / skill_path.name
        if hashlib.sha256(skill_path.read_bytes()).hexdigest() != manifest.get('skill_sha256'):
            raise ValueError(f'{task}: skill artifact hash mismatch')
        episodes = manifest.get('episodes', [])
        if len(episodes) != len(manifest.get('evaluation_seeds', [])):
            raise ValueError(f'{task}: episode/evaluation seed count mismatch')
        if not episodes or any(set(row.get('runtime_seconds', {})) != set(RUNTIME_COMPONENTS)
                               for row in episodes):
            raise ValueError(f'{task}: runtime component schema mismatch')
        for row in episodes:
            for field in (
                    'task', 'seed', 'success', 'steps',
                    'learned_agreement_steps', 'mapped_fallback_steps',
                    'safety_blocks', 'expectation_checks',
                    'expectation_mismatches', 'runtime_seconds'):
                if field not in row:
                    raise ValueError(f'{task}: episode field is required: {field}')
            if row['task'] != task:
                raise ValueError(f'{task}: episode task mismatch')
            if not isinstance(row['success'], bool):
                raise ValueError(f'{task}: episode success must be boolean')
        episode_seeds = [row['seed'] for row in episodes]
        _validate_seed_list(episode_seeds, label=f'{task} episode seeds')
        if episode_seeds != evaluation_seeds:
            raise ValueError(f'{task}: episode seed provenance mismatch')
        for field in (
                'safety_blocks', 'expectation_checks', 'expectation_mismatches'):
            if any(not isinstance(row.get(field), int) or row[field] < 0
                   for row in episodes):
                raise ValueError(f'{task}: invalid {field} diagnostic')
            if sum(row[field] for row in episodes) != manifest.get(field):
                raise ValueError(f'{task}: {field} total mismatch')
        for row in episodes:
            counters = (
                'learned_agreement_steps', 'mapped_fallback_steps',
                'safety_blocks', 'expectation_checks',
                'expectation_mismatches', 'steps',
            )
            if any(not isinstance(row.get(field), int) or row[field] < 0
                   for field in counters):
                raise ValueError(f'{task}: invalid episode counters')
            if (row['learned_agreement_steps']
                    + row['mapped_fallback_steps'] != row['steps']):
                raise ValueError(f'{task}: proposal source totals mismatch')
            if row['safety_blocks'] > row['steps']:
                raise ValueError(f'{task}: safety block count exceeds steps')
            if row['expectation_mismatches'] > row['expectation_checks']:
                raise ValueError(
                    f'{task}: expectation mismatch count exceeds checks')
        if not np.isclose(
                entry.get('success_fraction', float('nan')),
                manifest.get('success_fraction', float('nan')),
                rtol=1e-12, atol=1e-15):
            raise ValueError(f'{task}: aggregate success summary mismatch')
        if not np.isclose(
                entry.get('learned_agreement_fraction', float('nan')),
                manifest.get('learned_agreement_fraction', float('nan')),
                rtol=1e-12, atol=1e-15):
            raise ValueError(f'{task}: aggregate agreement summary mismatch')
        aggregate_runtime = entry.get('runtime_seconds_per_step', {})
        manifest_runtime = manifest.get('runtime_seconds_per_step', {})
        if set(aggregate_runtime) != set(RUNTIME_COMPONENTS):
            raise ValueError(f'{task}: aggregate runtime schema mismatch')
        for component in RUNTIME_COMPONENTS:
            if not np.isclose(
                    aggregate_runtime[component],
                    manifest_runtime.get(component, float('nan')),
                    rtol=1e-9, atol=1e-12):
                raise ValueError(
                    f'{task}: aggregate runtime summary mismatch for {component}')
        steps = sum(row['steps'] for row in episodes)
        for component in RUNTIME_COMPONENTS:
            _validate_nonnegative_number(
                manifest['runtime_seconds'].get(component),
                label=f'{task} runtime_seconds.{component}')
            _validate_nonnegative_number(
                manifest['runtime_seconds_per_step'].get(component),
                label=f'{task} runtime_seconds_per_step.{component}')
            measured = sum(row['runtime_seconds'][component] for row in episodes)
            recorded = manifest['runtime_seconds'][component]
            if not np.isclose(measured, recorded, rtol=1e-9, atol=1e-12):
                raise ValueError(f'{task}: runtime total mismatch for {component}')
            expected_per_step = measured / max(1, steps)
            if not np.isclose(
                    expected_per_step,
                    manifest['runtime_seconds_per_step'][component],
                    rtol=1e-9, atol=1e-12):
                raise ValueError(f'{task}: runtime per-step mismatch for {component}')
    return aggregate


def _fit_artifact(task: str, seeds: list[int], max_ticks: int) -> dict:
    features = []
    labels = []
    for seed in seeds:
        env = MiniCraftEnv(seed=seed, max_ticks=max_ticks)
        expert = MiniCraftExpert(task)
        obs = env.observe()
        while not env.done:
            action = expert.expert_action(obs, env)
            features.append(observation_features(obs))
            labels.append(action)
            obs, _, _, _ = env.step(action)
    tree, metrics = fit_decision_tree(
        np.stack(features), np.asarray(labels, dtype=np.int64))
    return {
        'schema_version': 1,
        'kind': 'minicraft_observation_only_compressed_skill',
        'task': task,
        'version': '0.1.0-alpha-compressed',
        'status': 'Candidate',
        'training_seeds': seeds,
        'fit_metrics': metrics,
        'tree': tree,
    }


def _episode(task: str, seed: int, artifact: dict, max_ticks: int) -> dict:
    return CompressedSkillRuntime(
        task, artifact, max_ticks=max_ticks).run_episode(seed)


def run(output_dir: str | Path, *, train_seeds: list[int],
        eval_seeds: list[int], max_ticks: int = 300) -> dict:
    if not train_seeds or not eval_seeds:
        raise ValueError('train and evaluation seeds must be non-empty')
    if any(isinstance(seed, bool) or not isinstance(seed, int)
           or not 0 <= seed <= 999 for seed in train_seeds + eval_seeds):
        raise ValueError('compressed skill uses train seeds 0..999 only')
    if set(train_seeds) & set(eval_seeds):
        raise ValueError('training and evaluation seeds must be disjoint')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        'schema_version': 1,
        'kind': 'phase1_compressed_skill_alpha',
        'status': EXPECTED_STATUS,
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'train_seeds': train_seeds,
        'evaluation_seeds': eval_seeds,
        'tasks': {},
        'limits': [
            'The learned tree is advisory and gated by mapped-planner agreement.',
            'Evaluation uses exploratory train seeds, not held-out variants.',
            'No promotion margin or compute-saving claim is applied.',
        ],
    }
    for task in TASKS:
        artifact = _fit_artifact(task, train_seeds, max_ticks)
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        skill_path = task_dir / 'skill.json'
        skill_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
        episodes = [_episode(task, seed, artifact, max_ticks) for seed in eval_seeds]
        manifest = {
            'schema_version': 1,
            'kind': 'phase1_compressed_skill_task_alpha',
            'status': aggregate['status'],
            'task': task,
            'train_seeds': train_seeds,
            'evaluation_seeds': eval_seeds,
            'episodes': episodes,
            'success_fraction': sum(row['success'] for row in episodes) / len(episodes),
            'learned_agreement_fraction': (
                sum(row['learned_agreement_steps'] for row in episodes)
                / max(1, sum(row['steps'] for row in episodes))),
            'safety_blocks': sum(row['safety_blocks'] for row in episodes),
            'expectation_checks': sum(row['expectation_checks'] for row in episodes),
            'expectation_mismatches': sum(
                row['expectation_mismatches'] for row in episodes),
            'runtime_seconds': {
                component: sum(
                    row['runtime_seconds'][component] for row in episodes)
                for component in RUNTIME_COMPONENTS
            },
            'runtime_seconds_per_step': {
                component: (
                    manifest_value / max(1, sum(row['steps'] for row in episodes)))
                for component, manifest_value in {
                    component: sum(
                        row['runtime_seconds'][component] for row in episodes)
                    for component in RUNTIME_COMPONENTS
                }.items()
            },
            'skill_path': str(skill_path),
            'skill_sha256': hashlib.sha256(skill_path.read_bytes()).hexdigest(),
            'fit_metrics': artifact['fit_metrics'],
            'limits': aggregate['limits'],
        }
        manifest_path = task_dir / 'manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
        aggregate['tasks'][task] = {
            'success_fraction': manifest['success_fraction'],
            'learned_agreement_fraction': manifest['learned_agreement_fraction'],
            'runtime_seconds_per_step': manifest['runtime_seconds_per_step'],
            'manifest_path': str(manifest_path),
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        print(
            f'{task}: success={manifest["success_fraction"]:.3f}, '
            f'learned_agreement={manifest["learned_agreement_fraction"]:.3f}')
    aggregate_path = output / 'manifest.json'
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    validate_manifest(output)
    print(f'Compressed-skill manifest: {aggregate_path}')
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-seeds', default=','.join(map(str, range(20, 40))))
    parser.add_argument('--eval-seeds', default=','.join(map(str, range(400, 410))))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-ticks', type=int, default=300)
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = Path('work') / 'phase1_compressed_skill' / f'compressed-{run_id}'
    try:
        run(
            output,
            train_seeds=[int(value) for value in args.train_seeds.split(',') if value],
            eval_seeds=[int(value) for value in args.eval_seeds.split(',') if value],
            max_ticks=args.max_ticks,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
