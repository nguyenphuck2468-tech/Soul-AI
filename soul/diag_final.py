"""Policy co thuc su HOC khong?

Tren bo cua no, phan thuong R2 = 92.4% (truoc khi sua `head_lr` thi
-14.4%). Tin hieu advantage cuoi cung da co that. Cau hoi bay gio la
cua policy: no co phan ung voi trang thai, va co lam diem cua episode
tot hon cua so khong dieu khien khong.

Ky vong: he la `v += (-k*x + a)*dt`, nen de dua x ve 0 can
`a ≈ -k*x` — tức `corr(x, a)` phai AM. Doan duoc gia tri am thi policy
dung huong; gia tri duong la policy chay nguoc.
"""

import sys
import time

import numpy as np
import torch

from .demo import PendulumEnv
from .soul import Soul
from .verify import collect_episodes, make_config


def _banner(s):
    print(f'\n{"=" * 70}\n{s}\n{"=" * 70}')


def train(updates=3000, seed=0):
    cfg = make_config()
    torch.manual_seed(seed)
    s = Soul(cfg)
    rng = np.random.default_rng(seed)
    env = PendulumEnv()
    collect_episodes(s, 30, env, rng)
    while s.trainer.updates < updates:
        collect_episodes(s, 1, env, rng)
        for _ in range(75):
            s.learn()
    return s, rng, env


def main(seed=0):
    t0 = time.time()
    _banner('1. Huan 3000 buoc')
    s, rng, env = train(3000, seed=seed)
    print(f'  xong ({time.time() - t0:.0f}s)')

    _banner('2. Phan thuong (tin hieu advantage co that chua)')
    b = s.replay.sample(16, rng)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
    ac, va, rw = t(b['actions']), t(b['valid']), t(b['rewards'])
    seq = b['values'].shape[1] - 1
    oi = torch.cat([v_, m_, tx], -1)
    with torch.no_grad():
        out = s.model.dynamics.observe_sequence(oi[:, :seq], ac, s.model.encoder, va)
        pr = s.model.reward_head(out['h_seq'], out['z_seq'])
    n = min(pr.numel(), rw.numel())
    # reward head học trên phần thưởng đã chuẩn hóa — đải ngược về thang
    # thật trước khi so.
    rs = s.trainer._reward_stats
    raw = pr.reshape(-1)[:n] * max(rs['std'], 1e-6) + rs['mean']
    mse = float(((raw - rw.reshape(-1)[:n]) ** 2).mean())
    base = float(((rw - rw.mean()) ** 2).mean())
    print(f'  std doan = {float(raw.std()):.4f}   std that = {float(rw.std()):.4f}')
    print(f'  R2 = {100 * (1 - mse / base):.1f}%')

    _banner('3. Policy co phan ung voi TRANG THAI khong?')
    rows = []
    for x in (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0):
        env.reset()
        env.x, env.v, env.t = x, 0.0, 0
        a = s.act(env._obs(), deterministic=True)
        rows.append((x, float(a[0])))
    print(f'  {"vi tri x":>10}{"hanh dong a":>13}')
    for x, a in rows:
        print(f'  {x:>10.2f}{a:>13.4f}')
    xs = np.array([r[0] for r in rows])
    as_ = np.array([r[1] for r in rows])
    c = float(np.corrcoef(xs, as_)[0, 1])
    print(f'  corr(x, a) = {c:+.4f}   '
          f'(can AM: a ≈ -{env.k:g}·x de dua x ve 0)')
    print(f'  do lech a giua cac trang thai = {as_.max() - as_.min():.4f}')

    _banner('4. Diem cua episode thuc te')
    def run_eval(det, n=12):
        tot = []
        r2g = np.random.default_rng(7)
        for i in range(n):
            env.reset(r2g)
            s.set_goal(env.goal_vec())
            s._h = None
            tot.append(run_one(s, env, det))
        return float(np.mean(tot)), float(np.std(tot))

    def run_one(soul, e, det):
        s_total = 0.0
        for _ in range(60):
            a = soul.act(e._obs(), deterministic=det)
            obs, r, done, _ = e.step(a)
            s_total += r
            soul.observe(obs, a, r, False)
        return s_total

    def run_zero(e, n=12):
        tot = []
        r2g = np.random.default_rng(7)
        for i in range(n):
            e.reset(r2g)
            t_ = 0.0
            for _ in range(60):
                obs, r, done, _ = e.step(np.zeros(4, np.float32))
                t_ += r
            tot.append(t_)
        return float(np.mean(tot)), float(np.std(tot))

    m_p, sd_p = run_eval(True)
    m_s, sd_s = run_eval(False)
    m_z, sd_z = run_zero(env)
    print(f'  policy (tiet dinh)  : {m_p:9.3f} +- {sd_p:.3f}')
    print(f'  policy (ngau nhien)  : {m_s:9.3f} +- {sd_s:.3f}')
    print(f'  khong dieu khien a=0 : {m_z:9.3f} +- {sd_z:.3f}')
    win = m_p > m_z
    print()
    print(f'  -> policy thang khong dieu khien? '
          f'{"CO" if win else "KHONG"}  ({m_p - m_z:+.3f})')

    _banner('5. TIN HIEU ADVANTAGE trong tuong tuong co that khong?')
    from .imagine import imagine_actor_critic
    b = s.replay.sample(16, rng)
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
    ac, va, rw = t(b['actions']), t(b['valid']), t(b['rewards'])
    gl = t(b['goals'])
    seq = b['values'].shape[1] - 1
    oi = torch.cat([v_, m_, tx], -1)
    with torch.no_grad():
        out = s.model.dynamics.observe_sequence(oi[:, :seq], ac, s.model.encoder, va)
    rts, vs, acts_s = [], [], []
    h, z = out['h_seq'][:, -1], out['z_seq'][:, -1]
    H_ = s.cfg.train.imagine_horizon
    for step in range(H_):
        # giu lai s_t TRUOC khi hanh dong — dung chinh no de do critic
        h_pre, z_pre = h, z
        mean, logstd, logits = s.model.policy_head(h, z, gl[:, -1])
        act = s.model.policy_head.sample(mean, logstd, logits)
        acts_s.append(act)
        h, mu, lv = s.model.dynamics.step(h, z, act)
        from .model import _logvar_to_std
        z = mu + torch.randn_like(mu) * _logvar_to_std(lv)
        with torch.no_grad():
            rts.append(s.model.reward_head(h, z).squeeze(-1))
            vs.append(s.model.value_head(h_pre, z_pre, gl[:, -1]).squeeze(-1))
    acts_s = torch.stack(acts_s, 1)
    rts = torch.stack(rts, 1)
    vs = torch.stack(vs, 1)
    tg = torch.zeros_like(rts)
    nx = torch.zeros(rts.shape[0])
    for step in reversed(range(rts.shape[1])):
        nx = rts[:, step] + s.cfg.train.gamma * nx
        tg[:, step] = nx
    adv = tg - vs
    print(f'  std reward tuong tuong = {float(rts.std()):.4f}   '
          f'(that {float(rw.std()):.4f})')
    print(f'  std muc tieu (return) = {float(tg.std()):.4f}')
    print(f'  std gia tri (critic)  = {float(vs.std()):.4f}')
    print(f'  std advantage         = {float(adv.std()):.4f}')
    if float(adv.std()) < 1e-4:
        print('  -> ADVANTAGE ~ 0. Critic qua manh: muc tieu va gia tri')
        print('     trung nhau, khong con gi de policy hoc.')
    c = float(np.corrcoef(adv.reshape(-1).numpy(),
                          acts_s[..., 0].detach().reshape(-1).numpy())[0, 1])
    print(f'  corr(advantage, hanh dong) = {c:+.4f}')
    print(f'  std hanh dong mau          = {float(acts_s.std()):.4f}')
    with torch.no_grad():
        mm, ls, _ = s.model.policy_head(out['h_seq'], out['z_seq'],
                                        gl[:, :seq])
    print(f'  std mean cua policy        = {float(mm.std()):.4f}')
    print(f'  std logstd cua policy      = {float(ls.std()):.4f}')
    print(f'  (thoi gian {time.time() - t0:.0f}s)')


if __name__ == '__main__':
    sys.exit(main())