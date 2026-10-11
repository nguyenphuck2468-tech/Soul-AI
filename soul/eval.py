"""Evaluate Soul against fixed controllers and write reconstructable JSONL records.

Example:
    python -m soul.eval --steps 3000 --episodes 60 --env-steps 200 --seeds 0 1 2

The built-in controller comparison is an evaluation, not a general benchmark
or proof that the architecture is ready for promotion.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import uuid

import numpy as np
import torch

try:
    from .demo import PendulumEnv
    from .soul import Soul
    from .train import health
    from .verify import collect_episodes, make_config
except ImportError:  # pragma: no cover - run as a script
    from demo import PendulumEnv
    from soul import Soul
    from train import health
    from verify import collect_episodes, make_config


def _banner(text):
    print(f'\n{"=" * 72}\n{text}\n{"=" * 72}')


def _summarize(episode_returns):
    values = np.asarray(episode_returns, dtype=np.float64)
    if values.size == 0:
        raise ValueError('Evaluation requires at least one episode.')
    return {
        'episodes': int(values.size),
        'mean_return': float(values.mean()),
        'return_std_population': float(values.std()),
        'episode_returns': values.tolist(),
    }


def _evaluate_baseline(env_fn, seed, episodes, steps, k, controller):
    episode_returns = []
    for episode in range(episodes):
        episode_seed = seed * 1000 + episode
        env = env_fn(seed=episode_seed)
        env.reset(np.random.default_rng(episode_seed))
        action_rng = np.random.default_rng(
            np.random.SeedSequence([seed, episode, 1]))
        total_reward = 0.0
        for _ in range(steps):
            if controller == 'pd':
                action = -k * env.x - 1.0 * env.v
            elif controller == 'zero':
                action = 0.0
            elif controller == 'random':
                action = float(action_rng.uniform(-1, 1))
            else:
                raise ValueError(f'Unknown baseline controller: {controller}')
            _, reward, done, _ = env.step(np.array([action], dtype=np.float32))
            total_reward += reward
            if done:
                break
        episode_returns.append(float(total_reward))
    return _summarize(episode_returns)


def eval_baseline(env_fn, seed, episodes, steps, k, dim):
    """Compatibility wrapper returning the historical (mean, std) tuple."""
    result = _evaluate_baseline(env_fn, seed, episodes, steps, k, dim)
    return result['mean_return'], result['return_std_population']


def _evaluate_soul(soul, seed, episodes, steps, env_fn=PendulumEnv):
    mode_returns = {}
    runtime_state = (
        None if soul._h is None else soul._h.clone(),
        None if soul._z is None else soul._z.clone(),
        None if soul._prev_action is None else soul._prev_action.copy(),
        None if soul._goal is None else soul._goal.copy(),
        soul._cur_ep,
        soul.ep_reward,
        soul.ep_len,
    )
    try:
        with torch.random.fork_rng():
            for mode_index, mode in enumerate(('deterministic', 'random')):
                torch.manual_seed(seed * 1000 + mode_index)
                episode_returns = []
                for episode in range(episodes):
                    episode_seed = seed * 1000 + episode
                    env = env_fn(seed=episode_seed)
                    env.reset(np.random.default_rng(episode_seed))
                    soul.reset_state()
                    soul.set_goal(env.goal_vec())
                    total_reward = 0.0
                    for _ in range(steps):
                        action = soul.act(
                            env._obs(), deterministic=(mode == 'deterministic'))
                        next_obs, reward, done, info = env.step(action)
                        total_reward += reward
                        soul.observe(
                            next_obs, action, reward, done,
                            applied_action=info['applied_action'],
                            record=False,
                        )
                        if done:
                            break
                    episode_returns.append(float(total_reward))
                mode_returns[mode] = _summarize(episode_returns)
    finally:
        (soul._h, soul._z, soul._prev_action,
         soul._goal, soul._cur_ep,
         soul.ep_reward, soul.ep_len) = runtime_state
    return mode_returns


def eval_soul(soul, seed, episodes, steps):
    """Compatibility wrapper returning the historical list of result tuples."""
    results = _evaluate_soul(soul, seed, episodes, steps)
    return [
        (mode, result['mean_return'], result['return_std_population'])
        for mode, result in results.items()
    ]


def train(steps, seed, updates_per_collect=25, warmup=30, plan_samples=0):
    if steps < 0:
        raise ValueError('steps must be non-negative.')
    if updates_per_collect < 1:
        raise ValueError('updates_per_collect must be positive.')
    if plan_samples < 0:
        raise ValueError('plan_samples must be non-negative.')

    cfg = make_config(seed=seed, plan_samples=plan_samples)
    if warmup < cfg.train.batch_size:
        raise ValueError(
            f'warmup must be at least {cfg.train.batch_size} episodes '
            'to fill the replay batch.')
    soul = Soul(cfg)
    rng = np.random.default_rng(seed)
    env = PendulumEnv(seed=seed)
    collect_episodes(soul, warmup, env, rng)
    while soul.trainer.updates < steps:
        collect_episodes(soul, 1, env, rng)
        remaining = steps - soul.trainer.updates
        for _ in range(min(updates_per_collect, remaining)):
            losses = soul.learn(force=True)
            if losses is None:
                raise RuntimeError(
                    'Training requested an update but replay produced no batch.')
    return soul


def _source_metadata():
    revision = subprocess.run(
        ['git', 'rev-parse', 'HEAD'],
        capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ['git', 'status', '--porcelain'],
        capture_output=True, text=True, check=False,
    )
    source_root = Path(__file__).resolve().parent
    source_files = (
        'config.py', 'demo.py', 'eval.py', 'imagine.py', 'memory.py',
        'model.py', 'obs.py', 'soul.py', 'train.py', 'trainer.py',
        'verify.py',
    )
    return {
        'git_revision': revision.stdout.strip() if revision.returncode == 0 else None,
        'working_tree_dirty': bool(status.stdout.strip()) if status.returncode == 0 else None,
        'source_file_sha256': {
            f'soul/{name}': hashlib.sha256(
                (source_root / name).read_bytes()).hexdigest()
            for name in source_files
        },
    }


def _validate_args(parser, args):
    if any(seed < 0 for seed in args.seeds):
        parser.error('seeds must be non-negative integers.')
    if args.steps < 0:
        parser.error('--steps must be non-negative.')
    if args.episodes < 1 or args.env_steps < 1:
        parser.error('--episodes and --env-steps must be positive.')
    if args.warmup_episodes < 1 or args.updates_per_collect < 1:
        parser.error('--warmup-episodes and --updates-per-collect must be positive.')
    if args.plan_samples < 0:
        parser.error('--plan-samples must be non-negative.')


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Soul against fixed Pendulum controllers.')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument(
        '--steps', type=int, default=3000,
        help='exact number of optimizer updates (default: 3000)',
    )
    parser.add_argument('--episodes', type=int, default=60)
    parser.add_argument('--env-steps', type=int, default=200)
    parser.add_argument('--warmup-episodes', type=int, default=30)
    parser.add_argument('--updates-per-collect', type=int, default=25)
    parser.add_argument(
        '--plan-samples', type=int, default=0,
        help='stochastic collection planner candidates; 0 uses the actor directly',
    )
    parser.add_argument('--out', type=str, default='runs/eval.jsonl')
    args = parser.parse_args()
    _validate_args(parser, args)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    env_template = PendulumEnv()
    k = env_template.k
    env_config = {
        'class': 'PendulumEnv',
        'dt': env_template.dt,
        'k': env_template.k,
        'noise_std': env_template.noise,
        'max_episode_steps': 200,
        'evaluation_step_limit': args.env_steps,
    }

    metric_definitions = {
        'episodes': {
            'definition': 'Number of evaluation episodes that produced a return.',
            'source': 'Evaluation loop',
            'aggregation': 'Count',
            'classification': 'direct measurement count',
            'unit': 'episodes',
        },
        'episode_returns': {
            'definition': 'Per-episode sum of raw environment rewards.',
            'source': 'PendulumEnv.step',
            'aggregation': 'None; one value per episode',
            'classification': 'direct measurement',
            'unit': 'environment reward',
        },
        'mean_return': {
            'definition': 'Arithmetic mean of episode_returns.',
            'source': 'PendulumEnv.step',
            'aggregation': 'Mean across configured evaluation episodes',
            'classification': 'direct measurement',
            'unit': 'environment reward',
        },
        'return_std_population': {
            'definition': 'Population standard deviation of episode_returns.',
            'source': 'PendulumEnv.step',
            'aggregation': 'NumPy std with ddof=0 across configured episodes',
            'classification': 'direct measurement',
            'unit': 'environment reward',
        },
        'reward_r2': {
            'definition': '100 * (1 - MSE / target variance) for reward predictions.',
            'source': 'Replay samples evaluated by train.health',
            'aggregation': 'All sampled replay targets in health batches',
            'classification': 'model diagnostic; not external task performance',
            'unit': 'percent',
        },
        'reward_pred_std': {
            'definition': 'Standard deviation of denormalized predicted rewards.',
            'source': 'Replay samples evaluated by train.health',
            'aggregation': 'All sampled replay predictions',
            'classification': 'model diagnostic',
            'unit': 'environment reward',
        },
        'reward_true_std': {
            'definition': 'Standard deviation of reward targets.',
            'source': 'Replay samples evaluated by train.health',
            'aggregation': 'All sampled replay targets',
            'classification': 'dataset diagnostic',
            'unit': 'environment reward',
        },
        'policy_action_std': {
            'definition': 'Standard deviation of policy mean outputs on replay sequences.',
            'source': 'Replay samples evaluated by train.health',
            'aggregation': 'All sampled policy mean components',
            'classification': 'proxy for action collapse/variation, not task quality',
            'unit': 'action units',
        },
        'recon_ratio': {
            'definition': 'Mean absolute reconstruction error divided by constant-mean baseline error for the most frequent observed feature.',
            'source': 'Replay samples evaluated by train.health',
            'aggregation': 'Mean over valid samples for the selected feature',
            'classification': 'model diagnostic proxy',
            'unit': 'dimensionless ratio',
        },
        'imagine_reward_std': {
            'definition': 'Standard deviation of imagined discounted returns.',
            'source': 'Stochastic latent rollouts in train.health',
            'aggregation': 'All imagined return values',
            'classification': 'model diagnostic proxy; not environment performance',
            'unit': 'model reward units',
        },
        'critic_std': {
            'definition': 'Standard deviation of critic values in replay-seeded imagined rollouts.',
            'source': 'Value head in train.health',
            'aggregation': 'All imagined value estimates',
            'classification': 'model diagnostic proxy',
            'unit': 'model value units',
        },
        'return_std': {
            'definition': 'imagine_reward_std multiplied by square root of imagination horizon.',
            'source': 'Derived in train.health',
            'aggregation': 'Single derived value',
            'classification': 'proxy; not an independently measured task return',
            'unit': 'model reward units',
        },
    }
    source = _source_metadata()
    print('Evaluation (fixed baselines; not a promotion benchmark)')
    print(f'config: episodes={args.episodes}, env_steps={args.env_steps}, '
          f'updates={args.steps}, warmup={args.warmup_episodes}, '
          f'plan_samples={args.plan_samples}')

    with output_path.open('a', encoding='utf-8') as log_file:
        for seed in args.seeds:
            _banner(f'Seed {seed} — train {args.steps} optimizer updates')
            baseline = {
                controller: _evaluate_baseline(
                    PendulumEnv, seed, args.episodes, args.env_steps, k, controller)
                for controller in ('pd', 'zero', 'random')
            }
            start_time = time.time()
            soul = train(
                args.steps, seed,
                updates_per_collect=args.updates_per_collect,
                warmup=args.warmup_episodes,
                plan_samples=args.plan_samples,
            )
            train_seconds = time.time() - start_time

            # health() contains stochastic imagined rollouts; make that
            # diagnostic repeatable independently of preceding training draws.
            torch.manual_seed(seed * 1000 + 2)
            health_metrics = health(soul)
            if health_metrics is None:
                raise RuntimeError('Health diagnostics require replay samples.')
            soul_results = _evaluate_soul(
                soul, seed, args.episodes, args.env_steps)
            det = soul_results['deterministic']

            print(f'  baseline zero: {baseline["zero"]["mean_return"]:.3f} '
                  f'+- {baseline["zero"]["return_std_population"]:.3f}')
            print(f'  Soul deterministic: {det["mean_return"]:.3f} '
                  f'+- {det["return_std_population"]:.3f}')
            print(f'  actual updates: {soul.trainer.updates}; train: {train_seconds:.1f}s')

            record = {
                'schema_version': 1,
                'experiment_id': str(uuid.uuid4()),
                'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
                'kind': 'evaluation',
                'source': source,
                'runtime': {
                    'python': platform.python_version(),
                    'numpy': np.__version__,
                    'torch': torch.__version__,
                    'torch_device': str(soul.device),
                    'torch_num_threads': torch.get_num_threads(),
                },
                'seed': seed,
                'training': {
                    'requested_optimizer_updates': args.steps,
                    'actual_optimizer_updates': soul.trainer.updates,
                    'warmup_episodes': args.warmup_episodes,
                    'updates_per_collect': args.updates_per_collect,
                    'collection_plan_samples': args.plan_samples,
                    'duration_seconds': train_seconds,
                },
                'environment': {
                    **env_config,
                    'episode_seeds': [
                        seed * 1000 + i for i in range(args.episodes)
                    ],
                },
                'evaluation': {
                    'episodes': args.episodes,
                    'step_limit': args.env_steps,
                    'baseline_controllers': list(baseline),
                    'treatment': 'Soul policy',
                    'baselines': baseline,
                    'soul': soul_results,
                },
                'model_config': soul.cfg.to_dict(),
                'health_diagnostics': health_metrics,
                'metric_definitions': metric_definitions,
                'comparison': {
                    'soul_deterministic_beats_zero_mean': (
                        det['mean_return'] > baseline['zero']['mean_return']),
                    'soul_minus_zero_mean_return': (
                        det['mean_return'] - baseline['zero']['mean_return']),
                    'soul_minus_pd_mean_return': (
                        det['mean_return'] - baseline['pd']['mean_return']),
                },
            }
            log_file.write(json.dumps(record, allow_nan=False) + '\n')
            log_file.flush()

    _banner('SUMMARY')
    print(f'JSONL records appended to: {os.fspath(output_path)}')


if __name__ == '__main__':
    main()
