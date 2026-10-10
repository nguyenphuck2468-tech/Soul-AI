"""BƯỚC 0 (v2): tach "mo hinh sai" khoi "dung mo hinh sai".

Bản v1 có ba lỗi trong chính bộ đo, đã sửa ở đây:

1. CEM đặt `h = 0` ở MỌI bước điều khiển. Lúc huấn và trong
   `soul.act()`, `h` mang lịch sử. CEM luôn bắt đầu ở trạng thái như
   đầu chuỗi → dùng một mốc trạng thái mà mô hình chưa được huấn ở
   đó. Bản này đi qua chính `soul.act()` để bộ lọc tiến đúng quy ước,
   rồi dùng `soul._h`, `soul._z` làm điểm xuất phát.
2. Mọi episode CEM dùng CÙNG một trạng thái đầu, còn baseline dùng
   trạng thái khác nhau → không so ghép cặp được. Bản này cho mọi bộ
   điều khiển cùng một tập trạng thái đầu, cùng thứ tự.
3. Test "phản thực" đo sai số TUYỆT ĐỐI của x, v chứ không phải Δ, và
   dùng `ac[:, k]` thay vì `ac[:, k+1]`. Bản này đo Δ thật, chạy một
   môi trường thật từ cùng trạng thái với hai hành động.

Ngoài ra thêm:
  - đo độ chính xác mở k = 1, 5, 10, 20 bước (prior + hành động thật),
    so với nền "giữ nguyên trạng thái", kèm đường sàn tái tạo.
  - lấy chỉ số ô từ `soul.enc.schema`, không giả định cột 0/1 là x/v.
"""

import argparse
import os
import sys
import time
from typing import List

import numpy as np
import torch

from .config import Config
from .demo import PendulumEnv
from .soul import Soul
from .verify import collect_episodes


def _unsquash(s):
    """Dao lai `squash(v) = v/(1+|v|)`: v = s/(1-|s|).

    Quan sat trong replay luon o thang da squash, con `PendulumEnv` lam viec
    tren gia tri tho. Do do moi so sanh giua "mo hinh" va "that" deu phai
    chuyen qua day — neu khong, bien do khac nhau mot he so, va do duong
    nguoi doc se hieu sai.

    Chấp nhận cả float và torch.Tensor.
    """
    if isinstance(s, torch.Tensor):
        return s / torch.clamp(1.0 - s.abs(), min=1e-6)
    return s / max(1.0 - abs(s), 1e-6)


def _banner(s):
    print(f'\n{"=" * 74}\n{s}\n{"=" * 74}')


def _slots(soul):
    """Chỉ số ô của x và v, lấy từ schema — không đoán."""
    sch = soul.enc.schema
    xi = sch.get('state.position')
    vi = sch.get('state.velocity')
    if xi is None or vi is None:
        raise RuntimeError(f'schema khong co state.position/velocity: {sch}')
    return xi, vi


# ----------------------------------------------------------------------
# A. Kiem chuc: loc co tien dung khong?
# ----------------------------------------------------------------------
def check_filter(soul, steps=40):
    """`h` cua soul.act() phai KHAC nhau giua cac buoc."""
    env = PendulumEnv(seed=3)
    env.reset(np.random.default_rng(3))
    soul.set_goal(env.goal_vec())
    soul.reset_state()
    hs = []
    for _ in range(steps):
        a = soul.act(env._obs(), deterministic=True)
        hs.append(soul._h.clone())
        obs, r, done, _ = env.step(a)
        soul.observe(obs, a, r, done)
    d = torch.stack([(hs[i] - hs[0]).abs().mean() for i in range(1, len(hs))])
    print(f'  ||h_t - h_0|| trung binh sau 1 buoc : {float(d.mean()):.6f}')
    ok = float(d.mean()) > 1e-6
    print(f'  -> loc co tien dung: {"CO" if ok else "KHONG"}')
    return ok


# ----------------------------------------------------------------------
# B. Do do chinh xac mo k buoc
# ----------------------------------------------------------------------
@torch.no_grad()
def open_loop(soul, ks=(1, 5, 10, 20), n_batches=6, rng=None):
    """Dung prior + hanh dong THAT de du doan o xa k buoc.

    Nền: "giữ nguyên trạng thái" — x_{t+k} = x_t. Một mô hình kể cả
    khi kém cũng phải thắng nền này, nếu không nó chẳng dự đoán gì.

    Sàn decoder: giai ma chính (h_t, z_t) — tức chỉ đọc lại trạng
    thái neo. Sai số dưới sàn này nghĩa là vấn đề nằm ở chỗ latent
    không mang thông tin, chứ không phải decoder kém.
    """
    rng = rng or np.random.default_rng(11)
    xi, vi = _slots(soul)
    acc = {}

    for _ in range(n_batches):
        b = soul.replay.sample(8, rng)
        if b is None:
            continue
        t = lambda a: torch.as_tensor(a, device=soul.device, dtype=torch.float32)
        v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
        ac, va = t(b['actions']), t(b['valid'])
        seq = v_.shape[1]
        oi = torch.cat([v_, m_, tx], -1)
        o = soul.model.dynamics.observe_sequence(oi, ac,
                                                 soul.model.encoder, va)
        h, z = o['h_seq'][:, 0], o['z_seq'][:, 0]
        base = soul.model.decoder(h, z)['obs']
        bx, bv = base[:, xi], base[:, vi]
        for k in range(1, min(max(ks), seq - 1) + 1):
            # `obs[k]` do `actions[k]` sinh ra. Để dự đoán `obs[k]` từ
            # `obs[k-1]` phải bước bằng `actions[k]` — KHÔNG phải
            # `actions[k-1]`. Bản trước dùng sai chỉ số này.
            h, mu, _ = soul.model.dynamics.step(h, z, ac[:, k])
            z = mu
            if k not in acc:
                acc[k] = {'x': [], 'v': [], 'hx': [], 'hv': [],
                          'fx': [], 'fv': []}
            d = soul.model.decoder(h, z)['obs']
            msk = (m_[:, k, xi] > 0.5) & (m_[:, k, vi] > 0.5)
            if msk.sum() == 0:
                continue
            tx_, tv_ = v_[:, k, xi], v_[:, k, vi]
            acc[k]['x'].append(float((d[:, xi] - tx_)[msk].abs().mean()))
            acc[k]['v'].append(float((d[:, vi] - tv_)[msk].abs().mean()))
            acc[k]['hx'].append(float((bx - tx_)[msk].abs().mean()))
            acc[k]['hv'].append(float((bv - tv_)[msk].abs().mean()))
            acc[k]['fx'].append(float((d[:, xi] - bx)[msk].abs().mean()))
            acc[k]['fv'].append(float((d[:, vi] - bv)[msk].abs().mean()))

    print(f'  {"k":>3}{"|Δx|":>10}{"nền giữ":>10}{"sàn":>9}{"tỉ lệ":>9}'
          f'{"   |Δv|":>10}{"nền giữ":>10}{"sàn":>9}{"tỉ lệ":>9}')
    ratios = []
    for k in ks:
        if k not in acc:
            continue
        a = acc[k]
        mx, hx, fx = np.mean(a['x']), np.mean(a['hx']), np.mean(a['fx'])
        mv, hv, fv = np.mean(a['v']), np.mean(a['hv']), np.mean(a['fv'])
        rx, rv = mx / max(hx, 1e-9), mv / max(hv, 1e-9)
        ratios.append(max(rx, rv))
        print(f'  {k:>3}{mx:>10.4f}{hx:>10.4f}{fx:>9.4f}{rx:>9.3f}'
              f'   {mv:>10.4f}{hv:>10.4f}{fv:>9.4f}{rv:>9.3f}')
    print()
    print('  "tỉ lệ" < 1 ⇒ mô hình tốt hơn nền giữ nguyên trạng thái.')
    return float(np.mean(ratios)) if ratios else float('nan')


# ----------------------------------------------------------------------
# C. Test phan thuc THAT
# ----------------------------------------------------------------------
@torch.no_grad()
def counterfactual(soul, n_batches=4, rng=None):
    """Cùng một trạng thái, a = -1 và a = +1: mô hình có đúng không?"""
    rng = rng or np.random.default_rng(11)
    xi, vi = _slots(soul)
    dev = soul.device
    rows = []
    print(f'  {"trạng thái":<18}{"a":>4}{"Δv thật":>10}{"Δv mô hình":>13}'
          f'{"Δx thật":>10}{"Δx mô hình":>13}')
    for _ in range(n_batches):
        b = soul.replay.sample(8, rng)
        if b is None:
            continue
        t = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float32)
        v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
        ac, va = t(b['actions']), t(b['valid'])
        seq = v_.shape[1]
        oi = torch.cat([v_, m_, tx], -1)
        o = soul.model.dynamics.observe_sequence(oi, ac,
                                                 soul.model.encoder, va)
        for k in range(min(3, seq - 1)):
            x0, v0 = v_[:, k, xi], v_[:, k, vi]
            if (m_[:, k, xi] > 0.5).sum() < 4:
                continue
            h, z = o['h_seq'][:, k], o['z_seq'][:, k]
            for a_val in (-1.0, 1.0):
                env = PendulumEnv(seed=0)
                dvs, dxs = [], []
                for bi in range(x0.shape[0]):
                    env.reset()
                    raw_x = _unsquash(float(x0[bi]))
                    raw_v = _unsquash(float(v0[bi]))
                    env.x, env.v, env.t = raw_x, raw_v, 0
                    env.noise = 0.0
                    env.step(np.array([a_val], np.float32))
                    dvs.append(env.v - raw_v)
                    dxs.append(env.x - raw_x)
                a_t = torch.full((x0.shape[0], 1), a_val, device=dev)
                hn, mu, _ = soul.model.dynamics.step(h, z, a_t)
                d = soul.model.decoder(hn, mu)['obs']
                dv_m = float((d[:, vi] - v0).mean())
                dx_m = float((d[:, xi] - x0).mean())
                dv_t, dx_t = float(np.mean(dvs)), float(np.mean(dxs))
                rows.append((abs(dv_t - dv_m), abs(dv_t),
                             abs(dx_t - dx_m), abs(dx_t)))
                print(f'  x={float(x0.mean()):+.2f}, v={float(v0.mean()):+.2f}'
                      f'{a_val:>4.1f}{dv_t:>10.4f}{dv_m:>13.4f}'
                      f'{dx_t:>10.4f}{dx_m:>13.4f}')
    if not rows:
        return float('nan')
    ev = np.mean([r[0] for r in rows]) / max(np.mean([r[1] for r in rows]), 1e-9)
    ex = np.mean([r[2] for r in rows]) / max(np.mean([r[3] for r in rows]), 1e-9)
    print()
    print(f'  sai số Δv tương đối : {ev:.3f}')
    print(f'  sai số Δx tương đối : {ex:.3f}')
    return max(ev, ex)


# ----------------------------------------------------------------------
# D. CEM dung loc cua Soul
# ----------------------------------------------------------------------
@torch.no_grad()
def cem_first_action(soul, h, z, horizon, iters, pop, elite, act_lo, act_hi):
    dev = soul.device
    gamma = soul.cfg.train.gamma
    mu = torch.zeros(1, horizon, 1, device=dev)
    sd = torch.full((1, horizon, 1), 1.5, device=dev)
    first = torch.zeros(1, 1, device=dev)
    B = pop
    for _ in range(iters):
        cand = (mu + sd * torch.randn(B, horizon, 1, device=dev)).clamp(
            act_lo, act_hi)
        hh = h.expand(B, -1).contiguous()
        zz = z.expand(B, -1).contiguous()
        tot = torch.zeros(B, device=dev)
        for t in range(horizon):
            candidate = cand[:, t].clone()
            continuous_dim = (
                candidate.shape[-1] - soul.cfg.model.discrete_dim)
            candidate[..., :continuous_dim].clamp_(
                soul.cfg.model.action_low,
                soul.cfg.model.action_high,
            )
            hh, m_, _ = soul.model.dynamics.step(hh, zz, candidate)
            zz = m_
            tot = tot + gamma ** t * soul.model.reward_head(hh, zz).squeeze(-1)
        idx = torch.topk(tot, elite).indices
        el = cand[idx]
        mu = mu + 0.3 * (el.mean(0, keepdim=True) - mu)
        sd = sd + 0.3 * (el.std(0, keepdim=True) - sd)
        first = el[0, :1]
    return first


def cem_control(soul, seeds: List[int], steps, horizon, iters, pop, elite,
                act_lo, act_hi):
    out = []
    for si in seeds:
        env = PendulumEnv(seed=si)
        env.reset(np.random.default_rng(si))
        soul.reset_state()
        soul.set_goal(env.goal_vec())
        tot, chosen = 0.0, []
        for _t in range(steps):
            # CEM uses the same live observation filter as Soul.act, but
            # keeps its sequence-search action separate from the live actor.
            soul._filter_observation(env._obs(), commit=True)
            a_t = cem_first_action(soul, soul._h, soul._z, horizon, iters,
                                   pop, elite, act_lo, act_hi)
            a = a_t.cpu().numpy().reshape(-1)
            chosen.append(float(a[0]))
            obs, r, done, info = env.step(a)
            soul.observe(
                obs, a, r, done,
                applied_action=info['applied_action'],
            )
            tot += r
            if done:
                break
        out.append((tot, chosen))
    return out


def baselines(seeds, steps, k):
    res = {}
    for name in ('pd', 'zero', 'random'):
        vals = []
        for si in seeds:
            env = PendulumEnv(seed=si)
            env.reset(np.random.default_rng(si))
            action_rng = np.random.default_rng(np.random.SeedSequence([si, 1]))
            tot = 0.0
            for _ in range(steps):
                if name == 'pd':
                    a = -k * env.x - 1.0 * env.v
                elif name == 'zero':
                    a = 0.0
                else:
                    a = float(action_rng.uniform(-1, 1))
                _, r, done, _ = env.step(np.array([a], np.float32))
                tot += r
                if done:
                    break
            vals.append(tot)
        res[name] = vals
    return res


def _sem(v):
    v = np.asarray(v, dtype=float)
    return float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=1500)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--episodes', type=int, default=20)
    ap.add_argument('--env-steps', type=int, default=120)
    ap.add_argument('--horizon', type=int, default=20)
    ap.add_argument('--pop', type=int, default=50)
    ap.add_argument('--iters', type=int, default=3)
    ap.add_argument('--elite', type=int, default=20)
    ap.add_argument('--ckpt', type=str, default='')
    ap.add_argument('--free-nats', type=float, default=-1.0)
    ap.add_argument('--w-kl', type=float, default=-1.0)
    args = ap.parse_args()

    k = PendulumEnv().k
    # Trạng thái đầu GHÉP CẶP: mọi bộ điều khiển cùng tập, cùng thứ tự.
    seeds = [args.seed * 1000 + i for i in range(args.episodes)]

    cfg = Config()
    cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
    cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
    cfg.train.batch_size, cfg.train.seq_len = 8, 16
    if args.free_nats >= 0:
        cfg.train.free_nats = args.free_nats
    if args.w_kl >= 0:
        cfg.train.w_kl = args.w_kl
    soul = Soul(cfg)

    _banner(f'Huấn {args.steps} bước (free_nats={cfg.train.free_nats}, '
            f'w_kl={cfg.train.w_kl})')
    t0 = time.time()
    if args.ckpt and os.path.exists(args.ckpt):
        soul.load(args.ckpt)
        print(f'  nạp {args.ckpt}')
    else:
        torch.manual_seed(args.seed)
        rng = np.random.default_rng(args.seed)
        env = PendulumEnv(seed=args.seed)
        collect_episodes(soul, 30, env, rng)
        while soul.trainer.updates < args.steps:
            collect_episodes(soul, 1, env, rng)
            for _ in range(25):
                soul.learn(force=True)
        soul.save(args.ckpt or '_cem.ckpt')
    print(f'  xong ({time.time() - t0:.0f}s)')

    _banner('A. Bộ lọc của Soul có tiến đúng không?')
    check_filter(soul)

    _banner('B. Độ chính xác mở k bước (prior + hành động thật)')
    ratio = open_loop(soul, rng=np.random.default_rng(11))

    _banner('C. Test phản thực THẬT: cùng trạng thái, a = −1 và a = +1')
    cf = counterfactual(soul, rng=np.random.default_rng(11))

    _banner(f'D. CEM ({args.episodes} episode ghép cặp)')
    base = baselines(seeds, args.env_steps, k)
    t0 = time.time()
    res = cem_control(soul, seeds, args.env_steps, args.horizon,
                      args.iters, args.pop, args.elite, -4.0, 4.0)
    cem_v = [r[0] for r in res]
    allacts = np.array([c for r in res for c in r[1]])

    print(f'  {"bộ điều khiển":<22}{"trung bình":>12}{"sai số chuẩn":>14}')
    for name, vals in (('pd', base['pd']), ('zero', base['zero']),
                       ('random', base['random'])):
        print(f'  {name:<22}{np.mean(vals):>12.2f}{_sem(vals):>14.2f}')
    print(f'  {"CEM (mô hình)":<22}{np.mean(cem_v):>12.2f}{_sem(cem_v):>14.2f}')
    dz = np.sqrt(_sem(cem_v) ** 2 + _sem(base['zero']) ** 2)
    print()
    print(f'  CEM - zero = {np.mean(cem_v) - np.mean(base["zero"]):+.2f} '
          f'(sai số chuẩn {dz:.2f})')
    print(f'  CEM - pd   = {np.mean(cem_v) - np.mean(base["pd"]):+.2f}')
    print(f'  hành động CEM chọn: |a| p50='
          f'{np.percentile(np.abs(allacts), 50):.2f}  p95='
          f'{np.percentile(np.abs(allacts), 95):.2f}')
    print(f'  ({time.time() - t0:.0f}s)')

    _banner('KẾT LUẬN')
    print(f'  độ chính xác mở k bước (nhỏ hơn nền = tốt): {ratio:.3f}')
    print(f'  sai số phản thực Δ                        : {cf:.3f}')
    print(f'  CEM − PD                                   : '
          f'{np.mean(cem_v) - np.mean(base["pd"]):+.2f}')


if __name__ == '__main__':
    sys.exit(main())