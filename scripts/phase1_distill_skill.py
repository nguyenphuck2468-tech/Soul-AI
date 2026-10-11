"""Distill privileged MiniCraft experts into observation-only tree skills."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from minicraft import MiniCraftEnv, MiniCraftSafetyController
from minicraft.distilled_skill import (
    DistilledObservationSkill,
    fit_decision_tree,
    observation_features,
)
from minicraft.experts import MiniCraftExpert


TASKS = ('T_wood', 'T_tools', 'T_iron')


def _run_episode(task: str, seed: int, max_ticks: int, *,
                 policy: DistilledObservationSkill | None = None) -> dict:
    env = MiniCraftEnv(seed=seed, max_ticks=max_ticks)
    expert = MiniCraftExpert(task) if policy is None else None
    safety = MiniCraftSafetyController()
    obs = env.observe()
    proposal_seconds = 0.0
    steps = 0
    blocks = 0
    final_info = None
    while not env.done:
        started = time.perf_counter()
        if policy is None:
            candidate = expert.expert_action(obs, env)
            leaf_samples = None
        else:
            proposal = policy.propose(obs)
            candidate = proposal.candidate_action
            leaf_samples = proposal.leaf_samples
        proposal_seconds += time.perf_counter() - started
        decision = safety.filter(obs, candidate)
        obs, _, _, info = env.step(decision.applied_action)
        final_info = info
        steps += 1
        blocks += int(decision.blocked)
    if final_info is None:
        raise RuntimeError('evaluation episode produced no transition')
    return {
        'task': task,
        'seed': seed,
        'controller': 'expert' if policy is None else 'distilled_tree',
        'success': bool(final_info['task_success'][task]),
        'steps': steps,
        'safety_blocks': blocks,
        'proposal_wall_seconds': proposal_seconds,
        'mean_leaf_training_samples': leaf_samples,
    }


def _collect(task: str, seeds: list[int], max_ticks: int):
    features = []
    labels = []
    episode_results = []
    safety = MiniCraftSafetyController()
    expert = MiniCraftExpert(task)
    for seed in seeds:
        env = MiniCraftEnv(seed=seed, max_ticks=max_ticks)
        obs = env.observe()
        while not env.done:
            candidate = expert.expert_action(obs, env)
            decision = safety.filter(obs, candidate)
            features.append(observation_features(obs))
            labels.append(decision.applied_action)
            obs, _, _, info = env.step(decision.applied_action)
        episode_results.append({
            'seed': seed,
            'success': bool(info['task_success'][task]),
            'steps': int(info['tick']),
        })
    return np.stack(features), np.asarray(labels, dtype=np.int64), episode_results


def run(output_dir: str | Path, *, train_seeds: list[int], eval_seeds: list[int],
        max_ticks: int = 300, max_depth: int = 8,
        min_samples_leaf: int = 8, max_thresholds: int = 16) -> dict:
    for name, seeds in (('train', train_seeds), ('evaluation', eval_seeds)):
        if not seeds or len(set(seeds)) != len(seeds) or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                or not 0 <= seed <= 999 for seed in seeds):
            raise ValueError(f'{name} seeds must be unique train seeds 0..999')
    if set(train_seeds) & set(eval_seeds):
        raise ValueError('distillation and evaluation seeds must be disjoint')
    if max_ticks < 1:
        raise ValueError('max_ticks must be positive')

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_results = {}
    for task in TASKS:
        task_dir = output / task
        task_dir.mkdir(parents=True, exist_ok=True)
        x, y, demo_episodes = _collect(task, train_seeds, max_ticks)
        tree, fit_metrics = fit_decision_tree(
            x, y, max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            max_thresholds=max_thresholds)
        artifact = {
            'schema_version': 1,
            'kind': 'minicraft_observation_only_distilled_skill',
            'task': task,
            'version': '0.1.0-alpha-distilled',
            'status': 'Candidate',
            'observation_feature_count': int(x.shape[1]),
            'training_seeds': train_seeds,
            'tree': tree,
        }
        skill_path = task_dir / 'skill.json'
        skill_path.write_text(
            json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
        examples_path = task_dir / 'training_examples.npz'
        np.savez_compressed(examples_path, features=x, labels=y)

        policy = DistilledObservationSkill(task, artifact)
        distilled = [_run_episode(task, seed, max_ticks, policy=policy)
                     for seed in eval_seeds]
        expert = [_run_episode(task, seed, max_ticks)
                  for seed in eval_seeds]
        distilled_rate = sum(row['success'] for row in distilled) / len(distilled)
        expert_rate = sum(row['success'] for row in expert) / len(expert)
        by_seed = {row['seed']: row for row in expert}
        pairs = [(row['success'], by_seed[row['seed']]['success']) for row in distilled]
        distilled_only = sum(a and not b for a, b in pairs)
        expert_only = sum(b and not a for a, b in pairs)
        timing = {
            'scope': 'action_proposal_only_local_wall_time',
            'distilled_mean_seconds_per_step': (
                sum(row['proposal_wall_seconds'] for row in distilled)
                / sum(row['steps'] for row in distilled)),
            'expert_mean_seconds_per_step': (
                sum(row['proposal_wall_seconds'] for row in expert)
                / sum(row['steps'] for row in expert)),
            'excludes': ['environment_step', 'safety_filter', 'serialization'],
        }
        task_manifest = {
            'schema_version': 1,
            'kind': 'phase1_distilled_skill_smoke',
            'status': 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION',
            'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
            'task': task,
            'train_seeds': train_seeds,
            'evaluation_seeds': eval_seeds,
            'max_ticks': max_ticks,
            'expert_demonstration_episodes': demo_episodes,
            'fit_metrics': fit_metrics,
            'evaluation': {
                'distilled_successes': sum(row['success'] for row in distilled),
                'expert_successes': sum(row['success'] for row in expert),
                'episodes_per_controller': len(eval_seeds),
                'distilled_success_fraction': distilled_rate,
                'expert_success_fraction': expert_rate,
                'paired_outcomes': {
                    'distilled_only_success': distilled_only,
                    'expert_only_success': expert_only,
                    'both_same_outcome': len(pairs) - distilled_only - expert_only,
                },
                'distilled_episodes': distilled,
                'expert_episodes': expert,
            },
            'timing': timing,
            'skill_path': str(skill_path),
            'skill_sha256': hashlib.sha256(skill_path.read_bytes()).hexdigest(),
            'training_examples_path': str(examples_path),
            'training_examples_sha256': hashlib.sha256(examples_path.read_bytes()).hexdigest(),
            'limits': [
                'Tree is behavior-cloned from a privileged expert using train-only demonstrations.',
                'Evaluation seeds are exploratory train seeds, not held-out variants.',
                'No promotion margin, variant distance, or scientific claim is applied.',
                'Timing excludes environment stepping and safety overhead.',
            ],
        }
        manifest_path = task_dir / 'manifest.json'
        manifest_path.write_text(
            json.dumps(task_manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
        task_results[task] = {
            'distilled_success_fraction': distilled_rate,
            'expert_success_fraction': expert_rate,
            'fit_metrics': fit_metrics,
            'timing': timing,
            'manifest_path': str(manifest_path),
            'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        }
        print(f'{task}: distilled={distilled_rate:.3f}, expert={expert_rate:.3f}, '
              f'tree_nodes={fit_metrics["node_count"]}')

    aggregate = {
        'schema_version': 1,
        'kind': 'phase1_distilled_skill_batch',
        'status': 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'train_seeds': train_seeds,
        'evaluation_seeds': eval_seeds,
        'tasks': task_results,
    }
    aggregate_path = output / 'batch_manifest.json'
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    print(f'Distillation batch manifest: {aggregate_path}')
    return aggregate


def _parse_seeds(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(',') if part.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-seeds', default=','.join(map(str, range(20, 40))))
    parser.add_argument('--eval-seeds', default=','.join(map(str, range(300, 310))))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-ticks', type=int, default=300)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--min-samples-leaf', type=int, default=8)
    parser.add_argument('--max-thresholds', type=int, default=16)
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = Path('work') / 'phase1_distillation' / f'distill-{run_id}'
    try:
        run(output, train_seeds=_parse_seeds(args.train_seeds),
            eval_seeds=_parse_seeds(args.eval_seeds), max_ticks=args.max_ticks,
            max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf,
            max_thresholds=args.max_thresholds)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
