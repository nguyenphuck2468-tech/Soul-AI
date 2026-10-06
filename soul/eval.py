"""Danh gia nghiem tuc: Soul so voi cac baseline that.

Ly do ton tai: trong suot qua trien khai, "policy hoc duoc" duoc bao
ve bang nhung con so KHONG co y nghia — loss giam, policy "chay". Bay
gio moi thu phai so voi mot thu gi cung noi giay, cung moi truong.

Chay 4 bo dieu khien, cung mot tap trong vai, cung moi truong:

    pd      bo dieu khien PD: a = -k*x - c*v   (muc tieu)
    zero    khong dieu khien: a = 0
    random  hanh dong ngau nhien trong [-1, 1]
    soul    policy hoc duoc, hai che do: tiet dinh va ngau nhien

Tieu chi ket luan:
    Soul > zero   → policy co gia tri so voi khong lam gi
    Soul ~ pd     → policy gan dung muc tieu
    Soul > pd     → dat muc tieu

Ket qua ghi ra JSONL, moi dong mot lan chay, de so sanh duoc giua
cac lan chay ma khong phai nhin vao man hinh.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from .demo import PendulumEnv
from .soul import Soul
from .train import health
from .verify import collect_episodes, make_config


def _banner(s):
    print(f'\n{"=" * 72}\n{s}\n{"=" * 72}')


def eval_baseline(env_fn, seed, episodes, steps, k, dim):
    """Chay mot bo dieu khien khong can mo hinh."""
    out = []
    for i in range(episodes):
        env = env_fn(seed=seed * 1000 + i)
        # PHẢI truyền rng khi reset. Khong truyen thi moi episode bat
        # dau o x = 0 va ket qua IDENTICAL — do lech chuan bang 0.00, va
        # moi so sanh tren do la ve khong.
        env.reset(np.random.default_rng(seed * 1000 + i))
        obs = env._obs()
        tot = 0.0
        for _ in range(steps):
            if dim == 'pd':
                a = -k * env.x - 1.0 * env.v
            elif dim == 'zero':
                a = 0.0
            else:
                a = float(env.rng.uniform(-1, 1))
            obs, r, done, _ = env.step(np.array([a], np.float32))
            tot += r
            if done:
                break
        out.append(tot)
    return float(np.mean(out)), float(np.std(out))


def eval_soul(soul, seed, episodes, steps):
    out = []
    for mode in ('deterministic', 'random'):
        vals = []
        for i in range(episodes):
            env = PendulumEnv(seed=seed * 1000 + i)
            env.reset(np.random.default_rng(seed * 1000 + i))
            soul.set_goal(env.goal_vec())
            soul.reset_state()
            tot = 0.0
            for _ in range(steps):
                a = soul.act(env._obs(), deterministic=(mode == 'deterministic'))
                obs, r, done, _ = env.step(a)
                soul.observe(obs, a, r, done)
                tot += r
                if done:
                    break
            vals.append(tot)
        out.append((mode, float(np.mean(vals)), float(np.std(vals))))
    return out


def train(steps, seed, updates_per_collect=25, warmup=30):
    cfg = make_config()
    torch.manual_seed(seed)
    s = Soul(cfg)
    rng = np.random.default_rng(seed)
    env = PendulumEnv(seed=seed)
    collect_episodes(s, warmup, env, rng)
    while s.trainer.updates < steps:
        collect_episodes(s, 1, env, rng)
        for _ in range(updates_per_collect):
            s.learn(force=True)
    return s


def main():
    ap = argparse.ArgumentParser(description='Danh gia Soul so voi baseline')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    ap.add_argument('--steps', type=int, default=3000,
                    help='So buoc hoc truoc khi danh gia')
    ap.add_argument('--episodes', type=int, default=60)
    ap.add_argument('--env-steps', type=int, default=200)
    ap.add_argument('--out', type=str, default='runs/eval.jsonl')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    k = PendulumEnv().k

    _banner('0. Baseline (khong can mo hinh)')
    base = {}
    for dim in ('pd', 'zero', 'random'):
        m, sd = eval_baseline(PendulumEnv, 0, args.episodes,
                              args.env_steps, k, dim)
        base[dim] = (m, sd)
        print(f'  {dim:<8} {m:10.3f} +- {sd:.3f}')
    print()
    print(f'  -> muc tieu: Soul phai VUOT qua zero ({base["zero"][0]:.1f}),')
    print(f'     va BAT cang gap voi pd ({base["pd"][0]:.1f}) de coi la hoc duoc.')
    print()

    logf = open(args.out, 'a', encoding='utf-8')
    try:
        for seed in args.seeds:
            _banner(f'Seed {seed} — huan {args.steps} buoc')
            t0 = time.time()
            soul = train(args.steps, seed)
            h = health(soul)
            res = eval_soul(soul, seed, args.episodes, args.env_steps)
            det = [r for r in res if r[0] == 'deterministic'][0]
            rnd = [r for r in res if r[0] == 'random'][0]

            print(f'  R2 phan thuong        = {h["reward_r2"]:.1f}%')
            print(f'  tai tao / nen         = {h["recon_ratio"]:.3f}')
            print(f'  do lech hanh dong     = {h["policy_action_std"]:.4f}')
            print(f'  do lech tuong tuong   = {h["imagine_reward_std"]:.3f} '
                  f'(thuc ~1.0)')
            print()
            print(f'  {"":<22}{"diem":>10}{"+-":>10}')
            print(f'  {"pd":<22}{base["pd"][0]:>10.2f}{base["pd"][1]:>10.2f}')
            print(f'  {"zero":<22}{base["zero"][0]:>10.2f}{base["zero"][1]:>10.2f}')
            print(f'  {"random":<22}{base["random"][0]:>10.2f}{base["random"][1]:>10.2f}')
            print(f'  {"soul (tiet dinh)":<22}{det[1]:>10.2f}{det[2]:>10.2f}')
            print(f'  {"soul (ngau nhien)":<22}{rnd[1]:>10.2f}{rnd[2]:>10.2f}')
            print()
            win = det[1] > base['zero'][0]
            gap = det[1] - base['pd'][0]
            print(f'  > zero? {"CO" if win else "KHONG"}   '
                  f'chenh lech voi pd = {gap:+.2f}   ({time.time() - t0:.0f}s)')
            logf.write(json.dumps({
                'seed': seed, 'steps': args.steps,
                'pd': base['pd'][0], 'zero': base['zero'][0],
                'random': base['random'][0],
                'soul_det': det[1], 'soul_det_std': det[2],
                'soul_rand': rnd[1], **h}) + '\n')
            logf.flush()
    finally:
        logf.close()

    _banner('TONG KET')
    print(f'  ghi tai: {args.out}')
    print(f'  Doc lai: Soul > zero va gap pd bi nho hon muc do lech giua cac '
          f'seed thi coi la HOC DUOC.')


if __name__ == '__main__':
    main()