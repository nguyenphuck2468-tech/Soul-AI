"""Run a descriptive multi-seed Candidate-skill versus replay sweep.

This is a train-seed smoke utility. It does not lock held-out variants or
promote any skill.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.phase1_replay_compare import run as compare_run


TASKS = ('T_wood', 'T_tools', 'T_iron')
SOURCE_SEEDS = {'T_wood': 5, 'T_tools': 6, 'T_iron': 7}


def run_sweep(seed_start: int, seed_count: int, source_root: str | Path,
              output_dir: str | Path, max_ticks: int = 300) -> dict:
    if seed_start < 0 or seed_count < 1 or seed_start + seed_count > 1000:
        raise ValueError('sweep seeds must be a non-empty range within 0..999')
    if max_ticks < 1:
        raise ValueError('max_ticks must be positive')
    seeds = list(range(seed_start, seed_start + seed_count))
    if any(seed in SOURCE_SEEDS.values() for seed in seeds):
        raise ValueError('sweep seeds must not include the source episode seeds 5, 6, or 7')

    source_root = Path(source_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    task_results = {}
    for task in TASKS:
        source = source_root / task / 'skill_trace.jsonl'
        task_manifest = compare_run(
            task, source, seeds, output / task, max_ticks=max_ticks)
        skill = [episode['success'] for episode in task_manifest['episodes']
                 if episode['controller'] == 'candidate_skill']
        replay = [episode['success'] for episode in task_manifest['episodes']
                  if episode['controller'] == 'replay']
        skill_episodes = [episode for episode in task_manifest['episodes']
                          if episode['controller'] == 'candidate_skill']
        replay_episodes = [episode for episode in task_manifest['episodes']
                           if episode['controller'] == 'replay']
        skill_decision_seconds = sum(
            episode['decision_wall_seconds'] for episode in skill_episodes)
        replay_decision_seconds = sum(
            episode['decision_wall_seconds'] for episode in replay_episodes)
        skill_steps = sum(episode['steps'] for episode in skill_episodes)
        replay_steps = sum(episode['steps'] for episode in replay_episodes)
        task_results[task] = {
            'source_seed': task_manifest['source_seed'],
            'source_trace_sha256': task_manifest['source_trace_sha256'],
            'candidate_skill_successes': sum(skill),
            'replay_successes': sum(replay),
            'episode_count_per_controller': len(seeds),
            'candidate_skill_success_fraction': sum(skill) / len(skill),
            'replay_success_fraction': sum(replay) / len(replay),
            'descriptive_A_skill_minus_replay':
                task_manifest['descriptive_A_skill_minus_replay'],
            'decision_compute': {
                'scope': 'controller_action_proposal_only',
                'candidate_total_wall_seconds': skill_decision_seconds,
                'replay_total_wall_seconds': replay_decision_seconds,
                'candidate_mean_wall_seconds_per_step':
                    skill_decision_seconds / skill_steps,
                'replay_mean_wall_seconds_per_step':
                    replay_decision_seconds / replay_steps,
                'candidate_mean_steps_per_episode': skill_steps / len(skill_episodes),
                'replay_mean_steps_per_episode': replay_steps / len(replay_episodes),
                'timing_note': (
                    'Local descriptive wall time; excludes environment simulation, '
                    'safety filtering, serialization, and any trained world-model '
                    'supervision or intervention.'
                ),
            },
            'manifest_path': str(output / task / 'manifest.json'),
        }

    manifest = {
        'schema_version': 1,
        'kind': 'phase1_skill_replay_multi_seed_sweep',
        'status': 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_seeds': seeds,
        'max_ticks': max_ticks,
        'tasks': task_results,
        'limits': [
            'Exploratory train-seed sweep only; not held-out generalization.',
            'Variant distance and promotion margins remain unlocked.',
            'Decision timing covers action proposal only and is local/descriptive; '
            'it is not integrated adaptive-compute accounting.',
            'No skill promotion or compute-savings claim is made.',
        ],
    }
    manifest_path = output / 'sweep_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    manifest['sweep_manifest_sha256'] = hashlib.sha256(
        manifest_path.read_bytes()).hexdigest()
    print(f'Sweep manifest: {manifest_path}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed-start', type=int, default=200)
    parser.add_argument('--seed-count', type=int, default=10)
    parser.add_argument('--source-root', default='results/phase1_skill_smoke_20261011')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-ticks', type=int, default=300)
    args = parser.parse_args()
    try:
        output = args.output_dir
        if output is None:
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            output = Path('work') / 'phase1_skill_sweeps' / f'skill-sweep-{run_id}'
        run_sweep(args.seed_start, args.seed_count, args.source_root,
                  output, args.max_ticks)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
