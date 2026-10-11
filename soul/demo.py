"""Demo: tinh loss giam tren mot mo phong don gian.

Moi truong: x_{t+1} = 0.9 * x_t + 0.3 * a_t + noise
Muc tieu: dua x ve 0 (goal = -x).

Mo phong nay nho, chay nhanh tren CPU, nhung van du de kiem chung:
- Tai tao (recon) giam: mo hinh hoc duoc dong luc.
- KL giam: prior gan posterior.
- Thuong du doan dung: reward head hoc duoc he so.
- Policy: hanh dong dua x ve 0.

Chay:
    python -m soul.demo
"""

import argparse
import math
from pathlib import Path
from typing import Dict

import numpy as np

try:
    from .config import Config
    from .soul import Soul
except ImportError:  # pragma: no cover - chạy trực tiếp như script
    from config import Config
    from soul import Soul


class PendulumEnv:
    """Mot con lat giay don gian (khong co goc, chi co van toc).

    Trang thai: vi tri x, van toc v.
    Hanh dong: luc a.
    Dong luc: x' = x + v*dt; v' = v + (-k*x + a)*dt + noise.
    """

    def __init__(self, dt=0.05, k=2.0, noise=0.02, seed=0):
        # THAM SO QUAN TRONG. Do bang so do, khong phai chon theo cam tinh.
        #
        # voi dt=0.1, k=1.0 (gia tri cũ): luc hoi phuc la -1.0*x, hanh dong
        # toi da 1.0. Hai luc NHO NHAU, nen con lac hanh xu nhu quay tu
        # do — hanh dong khong lam thay doi gi. Do do moi thu "te" deu do
        # mo hinh, chu khong phai do mo hinh te: moi muc tieu deu nhu nhau.
        #
        # Do la loai moi truong gia rong nhe nhat ma van khong sot ai:
        # no chay, no tieu thu du lieu, no in loss binh thuong — nhung
        # khong co bai toan nao de giai.
        #
        # dt=0.05, k=2.0: luc hoi phuc manh gap hanh dong, nen can `a`
        # dau phai doi dau. Day moi la bai toan that.
        self.dt = dt
        self.k = k
        self.noise = noise
        # NHIEU phai di qua rng CUA MOI TRUONG.
        #
        # Truoc day dung `np.random.normal` — ham global cua numpy. Khong
        # seed duoc, nen hai lan chay cung ma khong cho cung ket qua, va
        # moi so do tren day deu lap lai. Do la thu lon nhat de ket qua
        # "policy khong on dinh theo hat giong" co that hay khong.
        self.rng = np.random.default_rng(seed)
        self.x = 0.0
        self.v = 0.0
        self.t = 0

    def reset(self, rng=None):
        if rng is not None:
            self.x = rng.uniform(-2.0, 2.0)
            self.v = rng.uniform(-1.0, 1.0)
        else:
            self.x = 0.0
            self.v = 0.0
        self.t = 0
        return self._obs()

    def _obs(self):
        return {
            'state': {
                'position': self.x,
                'velocity': self.v,
            },
            'time': self.t,
        }

    def step(self, action: np.ndarray):
        a = float(action[0]) if isinstance(action, (np.ndarray, list)) else float(action)
        # Giới hạn hành động theo môi trường. Nếu không chặn, policy có thể
        # so sánh với một không gian hành động khác và đánh giá bị lệch.
        a = float(np.clip(a, -1.0, 1.0))
        # Dong luc.
        self.v += (-self.k * self.x + a) * self.dt
        self.x += self.v * self.dt
        # Nhieu.
        if self.noise > 0:
            self.x += self.rng.normal(0, self.noise)
            self.v += self.rng.normal(0, self.noise)
        self.t += 1
        # Thuong: am binh phuong vi tri (muon ve 0).
        reward = -(self.x ** 2 + 0.1 * self.v ** 2)
        done = self.t >= 200 or abs(self.x) > 10.0
        return self._obs(), reward, done, {
            'applied_action': np.array([a], dtype=np.float32),
        }

    # Muc tieu: vi tri = 0. Dung vector one-hot 4 chieu: [1,0,0,0] = ve 0.
    def goal_vec(self):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


def run_episode(soul: Soul, env: PendulumEnv, rng, train: bool = True) -> Dict[str, float]:
    """Chay mot episode, tra ve tong thuong moi truong va loss trung binh."""
    env.reset(rng)
    soul.set_goal(env.goal_vec())
    total_reward = 0.0
    losses_accum = {}
    n_losses = 0
    steps_run = 0

    obs = env._obs()
    for _ in range(200):
        action = soul.act(obs, deterministic=not train)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        steps_run += 1
        soul.observe(obs, action, reward, done)
        if train:
            losses = soul.learn()
            if losses:
                # Nhan "loss_" vao truoc moi khoa: trainer cung cap MOT khoa
                # ten 'reward', neu ghep truc tiep se ghi de 'total_reward' va
                # dong bao cao se in loss chu khong phai phan thuong that —
                # nham lan trong khi mo hinh hoan toan khong hoc duoc gi.
                for k, v in losses.items():
                    key = f'loss_{k}'
                    losses_accum[key] = losses_accum.get(key, 0.0) + v
                n_losses += 1
        if done:
            break

    if n_losses:
        for k in losses_accum:
            losses_accum[k] /= n_losses
    return {'total_reward': total_reward, 'steps': steps_run, **losses_accum}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--episodes', type=int, default=200,
                    help='So episode chay (mac dinh 200)')
    ap.add_argument('--seed', type=int, default=42,
                    help='Random seed')
    ap.add_argument('--checkpoint', type=str, default='soul_ckpt.pt',
                    help='File luu checkpoint')
    ap.add_argument('--eval-only', action='store_true',
                    help='Chi chay danh gia, khong hoc')
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # Cau hinh nho cho demo.
    cfg = Config()
    cfg.model.hidden_dim = 64
    cfg.model.latent_dim = 16
    cfg.model.action_dim = 1
    cfg.model.discrete_dim = 0
    cfg.model.goal_dim = 4
    cfg.model.use_goal = True
    cfg.train.batch_size = 4
    cfg.train.seq_len = 16
    cfg.train.updates_per_batch = 1
    cfg.train.lr = 3e-4
    # KHONG ghim posterior_mix o day nua. Gia tri 0.05 nghia la latent
    # luc hoc 95% la prior, con `act()` luc suy luan dung 100% posterior
    # — hai che do lech nhau. Gia tri mac dinh trong config da la 1.0
    # kem loi leo dan, dung cho ca hai.
    cfg.train.w_recon = 1.0
    cfg.train.w_kl = 0.5
    # KHONG ghim `free_nats` o day. Mac dinh trong config da vao 0.05 vi
    # KL thuc chi ~0.4 nat; ghim 1.0 se cat het gradient cua prior.
    # Xem ghi chu `free_nats` trong config.py.
    cfg.train.w_reward = 1.0
    cfg.train.w_policy = 0.5
    cfg.train.w_value = 0.5
    cfg.device = 'cpu'

    soul = Soul(cfg)
    env = PendulumEnv()

    print(f"Soul demo — {soul.model.num_parameters():,} tham so")
    print(f"Config: hidden={cfg.model.hidden_dim}, latent={cfg.model.latent_dim}, "
          f"action={cfg.model.action_dim}, goal={cfg.model.goal_dim}")
    print(f"Device: {cfg.device}")
    print("-" * 60)

    # --- Huan luyen ---
    if not args.eval_only:
        print("[Huan luyen]")
        for ep in range(1, args.episodes + 1):
            res = run_episode(soul, env, rng, train=True)
            if ep % 20 == 0 or ep <= 5:
                msg = f"  Ep {ep:4d} | thuong={res['total_reward']:8.3f}"
                if 'loss_total' in res:
                    msg += (f" | loss={res['loss_total']:.4f} "
                            f"recon={res.get('loss_recon', 0):.4f} "
                            f"kl={res.get('loss_kl', 0):.4f} "
                            f"rew={res.get('loss_reward', 0):.4f} "
                            f"pol={res.get('loss_policy', 0):.4f} "
                            f"val={res.get('loss_value', 0):.4f}")
                print(msg)
        # Luu checkpoint.
        soul.save(args.checkpoint)
        print(f"\n[Da luu checkpoint] {args.checkpoint}")

    # --- Danh gia (deterministic) ---
    print("\n[Danh gia - deterministic]")
    eval_rewards = []
    for i in range(20):
        res = run_episode(soul, env, rng, train=False)
        eval_rewards.append(res['total_reward'])
        if i < 5:
            print(f"  thu {i+1}: thuong={res['total_reward']:.3f}")
    print(f"  Trung binh: {np.mean(eval_rewards):.3f} +- {np.std(eval_rewards):.3f}")

    # --- Kiem tra tai tao (sanity check) ---
    print("\n[Kiem tra tai tao quan sat]")
    soul.set_goal(env.goal_vec())
    env.reset(rng)
    obs = env._obs()
    for _ in range(5):
        action = soul.act(obs)
        obs, _, _, _ = env.step(action)
    recon = soul.explain_observation(env._obs(), top=8)
    print("  Truong do duoc (top 8):")
    for name, val in recon:
        print(f"    {name:25s} = {val:8.4f}")

    stats = soul.get_stats()
    print(f"\n[Thong ke] buoc={stats['steps']}, cap nhat={stats['updates']}, "
          f"replay={stats['replay_size']}, ep={stats['replay_episodes']}, "
          f"slot={stats['enc_slots_used']}, overflow={stats['enc_overflow']}")


if __name__ == '__main__':
    main()
