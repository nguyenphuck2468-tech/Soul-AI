"""Smoke-run an observation-only mapped skill against the privileged expert."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from minicraft import MiniCraftEnv, MiniCraftSafetyController
from minicraft.experts import MiniCraftExpert
from minicraft.mapped_skill import MiniCraftMappedSkill


TASKS = ('T_wood', 'T_tools', 'T_iron')


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


def _episode(task: str, seed: int, max_ticks: int, use_mapped: bool):
    env = MiniCraftEnv(seed=seed, max_ticks=max_ticks)
    skill = MiniCraftMappedSkill(task) if use_mapped else None
    expert = None if use_mapped else MiniCraftExpert(task)
    safety = MiniCraftSafetyController()
    obs = env.observe()
    steps = blocks = 0
    proposal_seconds = 0.0
    rows = []
    expectation_checks = expectation_matches = 0
    final_info = None
    while not env.done:
        started = time.perf_counter()
        if skill is not None:
            proposal = skill.propose(obs)
            candidate = proposal.candidate_action
            subgoal = proposal.selected_subgoal
            coverage = proposal.map_coverage
            expected_outcome = proposal.expected_outcome
        else:
            candidate = expert.expert_action(obs, env)
            subgoal = 'privileged_expert'
            coverage = None
            expected_outcome = None
        proposal_seconds += time.perf_counter() - started
        decision = safety.filter(obs, candidate)
        next_obs, reward, done, info = env.step(decision.applied_action)
        expectation_match = None
        if expected_outcome is not None and expected_outcome['expected_fields']:
            expectation_match = True
            for dotted_path, expected_value in expected_outcome['expected_fields'].items():
                actual = next_obs
                for key in dotted_path.split('.'):
                    actual = actual[key]
                expectation_match &= actual == expected_value
            expectation_checks += 1
            expectation_matches += int(expectation_match)
        rows.append({
            'task': task,
            'seed': seed,
            'controller': 'mapped_observation_skill' if skill else 'privileged_expert',
            'tick': int(info['tick']),
            'candidate_action': candidate,
            'applied_action': decision.applied_action,
            'safety_blocked': decision.blocked,
            'subgoal': subgoal,
            'map_coverage': coverage,
            'expected_outcome': expected_outcome,
            'expectation_match': expectation_match,
            'reward': float(reward),
            'done': bool(done),
            'task_success': bool(info['task_success'][task]),
            'observation': _jsonable(obs),
            'next_observation': _jsonable(next_obs),
        })
        blocks += int(decision.blocked)
        steps += 1
        obs = next_obs
        final_info = info
    if final_info is None:
        raise RuntimeError('mapped-skill episode produced no transitions')
    return {
        'task': task,
        'seed': seed,
        'controller': 'mapped_observation_skill' if skill else 'privileged_expert',
        'success': bool(final_info['task_success'][task]),
        'steps': steps,
        'safety_blocks': blocks,
        'proposal_wall_seconds': proposal_seconds,
        'mean_proposal_seconds_per_step': proposal_seconds / steps,
        'final_map_coverage': float(np.mean(skill.map >= 0)) if skill else None,
        'expectation_checks': expectation_checks,
        'expectation_matches': expectation_matches,
    }, rows


def run(eval_seeds: list[int], output_dir: str | Path,
        max_ticks: int = 300) -> dict:
    if not eval_seeds or len(set(eval_seeds)) != len(eval_seeds) or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            or not 0 <= seed <= 999 for seed in eval_seeds):
        raise ValueError('evaluation must use unique train seeds 0..999 only')
    if max_ticks < 1:
        raise ValueError('max_ticks must be positive')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    aggregate = {
        'schema_version': 1,
        'kind': 'phase1_observation_mapped_skill_smoke',
        'status': 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'evaluation_seeds': eval_seeds,
        'max_ticks': max_ticks,
        'tasks': {},
        'limits': [
            'Mapped skill consumes only structured observations; no full-state API.',
            'Train-seed smoke only; not held-out variant or promotion evidence.',
            'Privileged expert is a task-success reference, not matched-cost baseline.',
        ],
    }
    for task in TASKS:
        summaries = []
        traces = []
        for seed in eval_seeds:
            for mapped in (True, False):
                summary, rows = _episode(task, seed, max_ticks, mapped)
                summaries.append(summary)
                traces.extend(rows)
        mapped_rows = [s for s in summaries if s['controller'] == 'mapped_observation_skill']
        expert_rows = [s for s in summaries if s['controller'] == 'privileged_expert']
        mapped_by_seed = {s['seed']: s for s in mapped_rows}
        expert_by_seed = {s['seed']: s for s in expert_rows}
        pairs = [(mapped_by_seed[seed]['success'], expert_by_seed[seed]['success'])
                 for seed in eval_seeds]
        mapped_success = sum(a for a, _ in pairs)
        expert_success = sum(b for _, b in pairs)
        child = output / task
        child.mkdir(parents=True, exist_ok=True)
        trace_path = child / 'trace.jsonl'
        with trace_path.open('w', encoding='utf-8', newline='\n') as f:
            for row in traces:
                f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        manifest = {
            'schema_version': 1,
            'kind': 'phase1_observation_mapped_skill_task_smoke',
            'status': aggregate['status'],
            'task': task,
            'evaluation_seeds': eval_seeds,
            'mapped_successes': mapped_success,
            'expert_successes': expert_success,
            'episodes_per_controller': len(eval_seeds),
            'mapped_success_fraction': mapped_success / len(eval_seeds),
            'expert_success_fraction': expert_success / len(eval_seeds),
            'paired_outcomes': {
                'mapped_only_success': sum(a and not b for a, b in pairs),
                'expert_only_success': sum(b and not a for a, b in pairs),
                'both_same_outcome': sum(a == b for a, b in pairs),
            },
            'one_step_expectation_checks': sum(
                s['expectation_checks'] for s in mapped_rows),
            'one_step_expectation_matches': sum(
                s['expectation_matches'] for s in mapped_rows),
            'one_step_expectation_match_fraction': (
                sum(s['expectation_matches'] for s in mapped_rows)
                / max(1, sum(s['expectation_checks'] for s in mapped_rows))),
            'mapped_mean_proposal_seconds_per_step': float(np.mean([
                s['mean_proposal_seconds_per_step'] for s in mapped_rows])),
            'expert_mean_proposal_seconds_per_step': float(np.mean([
                s['mean_proposal_seconds_per_step'] for s in expert_rows])),
            'episodes': summaries,
            'trace_path': str(trace_path),
            'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
            'limits': aggregate['limits'],
        }
        manifest_path = child / 'manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
            encoding='utf-8')
        aggregate['tasks'][task] = {
            'mapped_success_fraction': manifest['mapped_success_fraction'],
            'expert_success_fraction': manifest['expert_success_fraction'],
            'paired_outcomes': manifest['paired_outcomes'],
            'manifest_path': str(manifest_path),
        }
        print(f'{task}: mapped={manifest["mapped_success_fraction"]:.3f}, '
              f'expert={manifest["expert_success_fraction"]:.3f}')

    aggregate_path = output / 'manifest.json'
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    print(f'Mapped-skill manifest: {aggregate_path}')
    return aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-seeds', default=','.join(map(str, range(400, 410))))
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--max-ticks', type=int, default=300)
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        output = Path('work') / 'phase1_mapped_skill' / f'mapped-{run_id}'
    try:
        seeds = [int(part.strip()) for part in args.eval_seeds.split(',') if part.strip()]
        run(seeds, output, args.max_ticks)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
