"""Run the alpha compressed observation-only skill."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
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
    for task in TASKS:
        entry = aggregate.get('tasks', {}).get(task)
        if not isinstance(entry, dict):
            raise ValueError(f'missing compressed-skill task entry: {task}')
        manifest_path = Path(entry['manifest_path'])
        if not manifest_path.is_absolute():
            manifest_path = output / task / manifest_path.name
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != entry.get('manifest_sha256'):
            raise ValueError(f'{task}: task manifest hash mismatch')
        manifest = json.loads(manifest_bytes.decode('utf-8'))
        if manifest.get('status') != EXPECTED_STATUS:
            raise ValueError(f'{task}: task manifest status mismatch')
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
        for field in (
                'safety_blocks', 'expectation_checks', 'expectation_mismatches'):
            if any(not isinstance(row.get(field), int) or row[field] < 0
                   for row in episodes):
                raise ValueError(f'{task}: invalid {field} diagnostic')
            if sum(row[field] for row in episodes) != manifest.get(field):
                raise ValueError(f'{task}: {field} total mismatch')
        steps = sum(row['steps'] for row in episodes)
        for component in RUNTIME_COMPONENTS:
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
