"""Run the alpha compressed observation-only skill."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from minicraft import MiniCraftEnv, MiniCraftSafetyController
from minicraft.compressed_skill import CompressedObservationSkill
from minicraft.distilled_skill import fit_decision_tree, observation_features
from minicraft.experts import MiniCraftExpert


TASKS = ('T_wood', 'T_tools', 'T_iron')


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
    env = MiniCraftEnv(seed=seed, max_ticks=max_ticks)
    skill = CompressedObservationSkill(task, artifact)
    safety = MiniCraftSafetyController()
    obs = env.observe()
    learned_agreements = fallbacks = steps = 0
    proposal_seconds = safety_seconds = environment_seconds = 0.0
    while not env.done:
        started = time.perf_counter()
        proposal = skill.propose(obs)
        proposal_seconds += time.perf_counter() - started
        learned_agreements += int(proposal.source == 'learned_agreement')
        fallbacks += int(proposal.source == 'mapped_fallback')
        started = time.perf_counter()
        decision = safety.filter(obs, proposal.candidate_action)
        safety_seconds += time.perf_counter() - started
        started = time.perf_counter()
        obs, _, _, info = env.step(decision.applied_action)
        environment_seconds += time.perf_counter() - started
        skill.observe_outcome(obs, applied_action=decision.applied_action)
        steps += 1
    return {
        'task': task,
        'seed': seed,
        'success': bool(info['task_success'][task]),
        'steps': steps,
        'learned_agreement_steps': learned_agreements,
        'mapped_fallback_steps': fallbacks,
        'runtime_seconds': {
            'skill_proposal': proposal_seconds,
            'safety_filter': safety_seconds,
            'environment_step': environment_seconds,
            'integrated': proposal_seconds + safety_seconds + environment_seconds,
        },
    }


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
        'status': 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION',
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
            'runtime_seconds': {
                component: sum(
                    row['runtime_seconds'][component] for row in episodes)
                for component in (
                    'skill_proposal', 'safety_filter', 'environment_step',
                    'integrated',
                )
            },
            'runtime_seconds_per_step': {
                component: (
                    manifest_value / max(1, sum(row['steps'] for row in episodes)))
                for component, manifest_value in {
                    component: sum(
                        row['runtime_seconds'][component] for row in episodes)
                    for component in (
                        'skill_proposal', 'safety_filter', 'environment_step',
                        'integrated',
                    )
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
