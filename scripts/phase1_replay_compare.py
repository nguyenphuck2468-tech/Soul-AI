"""Exploratory Candidate-skill versus action-replay comparison.

This uses fresh train seeds only and reports descriptive outcomes without
promoting the skill or applying an acceptance margin.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from minicraft import MiniCraftEnv, MiniCraftExpertSkill, MiniCraftSafetyController


TRAIN_MIN, TRAIN_MAX = 0, 999


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _load_replay(path: Path, task: str) -> tuple[list[int], int]:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()
            if line.strip()]
    if not rows:
        raise ValueError('source skill trace is empty')
    seeds = {row.get('seed') for row in rows}
    if (len(seeds) != 1 or any(row.get('split') != 'train'
                               or row.get('skill_id') != f'minicraft.expert.{task}'
                               for row in rows)):
        raise ValueError('source trace must be one train episode for the selected task')
    source_seed = next(iter(seeds))
    if (isinstance(source_seed, bool) or not isinstance(source_seed, int)
            or not TRAIN_MIN <= source_seed <= TRAIN_MAX):
        raise ValueError('source trace seed must be in the train range 0..999')
    actions = [row.get('applied_action') for row in rows]
    if any(isinstance(action, bool) or not isinstance(action, int)
           or not 0 <= action <= 8 for action in actions):
        raise ValueError('source trace contains an invalid applied action')
    return actions, source_seed


def _run_episode(task: str, seed: int, max_ticks: int, *,
                 replay_actions: list[int] | None) -> tuple[dict, list[dict]]:
    env = MiniCraftEnv(seed=seed, max_ticks=max_ticks)
    skill = MiniCraftExpertSkill(task) if replay_actions is None else None
    safety = MiniCraftSafetyController()
    obs = env.observe()
    action_index = 0
    decision_seconds = 0.0
    blocks = 0
    trace = []
    final_info = None

    while not env.done:
        started = time.perf_counter()
        if replay_actions is None:
            proposal = skill.propose(obs, env)
            candidate = proposal.candidate_action
            expected = proposal.expected_outcome
        else:
            candidate = (replay_actions[action_index]
                         if action_index < len(replay_actions) else 0)
            expected = {
                'scope': 'replayed_sequence',
                'source_action_index': action_index,
                'sequence_exhausted': action_index >= len(replay_actions),
            }
        decision_seconds += time.perf_counter() - started
        decision = safety.filter(obs, candidate)
        next_obs, reward, done, info = env.step(decision.applied_action)
        trace.append({
            'task': task,
            'seed': seed,
            'tick': int(info['tick']),
            'controller': 'candidate_skill' if skill is not None else 'replay',
            'candidate_action': candidate,
            'applied_action': decision.applied_action,
            'expected_outcome': expected,
            'safety_blocked': decision.blocked,
            'reward': float(reward),
            'done': bool(done),
            'task_success': bool(info['task_success'][task]),
            'observation': _jsonable(obs),
            'next_observation': _jsonable(next_obs),
        })
        blocks += int(decision.blocked)
        action_index += 1
        obs = next_obs
        final_info = info

    if final_info is None:
        raise RuntimeError('episode produced no transitions')
    success = bool(final_info['task_success'][task])
    if skill is not None:
        skill.record_episode_result(success)
    return {
        'task': task,
        'seed': seed,
        'controller': 'candidate_skill' if skill is not None else 'replay',
        'success': success,
        'steps': action_index,
        'safety_blocks': blocks,
        'decision_wall_seconds': decision_seconds,
        'skill_card': skill.card() if skill is not None else None,
    }, trace


def run(task: str, source_trace: str | Path, eval_seeds: list[int],
        output_dir: str | Path, max_ticks: int = 300) -> dict:
    if task not in ('T_wood', 'T_tools', 'T_iron'):
        raise ValueError('task must be T_wood, T_tools, or T_iron')
    if not eval_seeds or any(isinstance(seed, bool) or not isinstance(seed, int)
                             or not TRAIN_MIN <= seed <= TRAIN_MAX
                             for seed in eval_seeds):
        raise ValueError('comparison accepts train seeds 0..999 only')
    if len(set(eval_seeds)) != len(eval_seeds):
        raise ValueError('evaluation seed list must not contain duplicates')
    if max_ticks < 1:
        raise ValueError('max_ticks must be positive')

    source = Path(source_trace)
    replay_actions, source_seed = _load_replay(source, task)
    if source_seed in eval_seeds:
        raise ValueError('evaluation seeds must differ from the replay source seed')

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / 'comparison_trace.jsonl'
    summaries = []
    traces = []
    for seed in eval_seeds:
        for replay in (False, True):
            summary, rows = _run_episode(
                task, seed, max_ticks,
                replay_actions=replay_actions if replay else None)
            summaries.append(summary)
            traces.extend(rows)

    skill_successes = [row['success'] for row in summaries
                       if row['controller'] == 'candidate_skill']
    replay_successes = [row['success'] for row in summaries
                        if row['controller'] == 'replay']
    skill_rate = sum(skill_successes) / len(skill_successes)
    replay_rate = sum(replay_successes) / len(replay_successes)
    with trace_path.open('w', encoding='utf-8', newline='\n') as f:
        for row in traces:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')

    manifest = {
        'schema_version': 1,
        'kind': 'phase1_skill_replay_comparison',
        'status': 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'task': task,
        'source_seed': source_seed,
        'evaluation_seeds': eval_seeds,
        'source_action_count': len(replay_actions),
        'candidate_skill_success_fraction': skill_rate,
        'replay_success_fraction': replay_rate,
        'descriptive_A_skill_minus_replay': skill_rate - replay_rate,
        'episodes': summaries,
        'trace_path': str(trace_path),
        'source_trace_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'limits': [
            'Train-seed smoke comparison only; not a held-out generalization result.',
            'No promotion threshold or variant-distance protocol is locked.',
            'Candidate uses privileged full-state BFS; replay is a fixed action cache.',
            'Single small sample and wall-clock measurements are descriptive only.',
        ],
    }
    manifest_path = output / 'manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    print(f'{task}: candidate={skill_rate:.3f}, replay={replay_rate:.3f}, '
          f'descriptive_A={skill_rate - replay_rate:+.3f}')
    print(f'Comparison manifest: {manifest_path}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('T_wood', 'T_tools', 'T_iron'),
                        required=True)
    parser.add_argument('--source-trace', required=True)
    parser.add_argument('--eval-seeds', default='8,9,10')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-ticks', type=int, default=300)
    args = parser.parse_args()
    try:
        seeds = [int(part.strip()) for part in args.eval_seeds.split(',') if part.strip()]
        output = args.output_dir
        if output is None:
            run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            output = str(Path('work') / 'phase1_skill_comparisons'
                         / f'{args.task}-{run_id}')
        run(args.task, args.source_trace, seeds, output, args.max_ticks)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
