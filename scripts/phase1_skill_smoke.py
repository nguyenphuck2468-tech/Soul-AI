"""Run one Candidate MiniCraft expert-skill episode as an alpha smoke."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from minicraft import MiniCraftEnv, MiniCraftExpertSkill, MiniCraftSafetyController


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


def run(output_dir: str | Path, *, task: str = 'T_wood', seed: int = 5,
        max_ticks: int = 300) -> dict:
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= 999:
        raise ValueError('alpha skill smoke accepts train seeds 0..999 only')
    if max_ticks < 1:
        raise ValueError('max_ticks must be positive')

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trace_path = output / 'skill_trace.jsonl'
    card_path = output / 'skill_card.json'
    manifest_path = output / 'manifest.json'

    env = MiniCraftEnv(seed=seed, max_ticks=max_ticks)
    skill = MiniCraftExpertSkill(task)
    safety = MiniCraftSafetyController()
    observation = env.observe()
    steps = 0
    result = None
    with trace_path.open('w', encoding='utf-8', newline='\n') as trace:
        while not env.done:
            proposal = skill.propose(observation, env)
            decision = safety.filter(observation, proposal.candidate_action)
            next_observation, reward, done, info = env.step(decision.applied_action)
            result = info
            row = {
                'schema_version': 1,
                'kind': 'phase1_skill_smoke_transition',
                'status': 'NON_EVIDENTIARY_SMOKE_ONLY',
                'split': 'train',
                'seed': seed,
                'tick': int(info['tick']),
                'skill_id': proposal.skill_id,
                'skill_version': proposal.version,
                'skill_status': proposal.skill_status,
                'candidate_action': proposal.candidate_action,
                'applied_action': decision.applied_action,
                'expected_outcome': proposal.expected_outcome,
                'confidence_status': proposal.confidence_status,
                'safety': {
                    'blocked': decision.blocked,
                    'reason': decision.reason,
                    'allowed_actions': list(decision.allowed_actions),
                },
                'observation': _jsonable(observation),
                'next_observation': _jsonable(next_observation),
                'reward': float(reward),
                'done': bool(done),
                'task_success': bool(info['task_success'][task]),
            }
            trace.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            steps += 1
            observation = next_observation

    if result is None:
        raise RuntimeError('skill episode produced no transition')
    success = bool(result['task_success'][task])
    skill.record_episode_result(success)
    card = skill.card()
    card_path.write_text(
        json.dumps(card, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    revision = subprocess.run(
        ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
        check=False).stdout.strip() or None
    manifest = {
        'schema_version': 1,
        'kind': 'phase1_skill_smoke',
        'status': 'NON_EVIDENTIARY_SMOKE_ONLY',
        'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'git_revision': revision,
        'split': 'train',
        'seed': seed,
        'task': task,
        'steps': steps,
        'task_success': success,
        'skill_status': card['status'],
        'empirical_success_rate': card['empirical_success_rate'],
        'trace_sha256': hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        'skill_card_sha256': hashlib.sha256(card_path.read_bytes()).hexdigest(),
        'limits': [
            'One episode only; not a success-rate estimate.',
            'The wrapped expert uses privileged full-state BFS.',
            'Candidate skill is not calibrated, generalized, or compute-validated.',
        ],
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
        encoding='utf-8')
    print(f'{task}: success={success}, steps={steps}, skill={card["status"]}')
    print(f'Skill smoke manifest: {manifest_path}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--task', choices=('T_wood', 'T_tools', 'T_iron'),
                        default='T_wood')
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--max-ticks', type=int, default=300)
    args = parser.parse_args()
    output = args.output_dir
    if output is None:
        run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        output = str(Path('work') / 'phase1_skill_smokes' / f'skill-{run_id}')
    try:
        run(output, task=args.task, seed=args.seed, max_ticks=args.max_ticks)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
