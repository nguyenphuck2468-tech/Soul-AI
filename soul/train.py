"""Chay huấn luyện, thiet bi phuc vu nham lon (co GPU).

Viet cho viec len hosting. Ba dieu quan trong nam o day:

1. BAO CAO SUC KHOE, KHONG PHAI CHI LOSS.
   Trong toan bo qua trinh nay, loss luon in binh thuong trong khi mo
   hinh da hong. Da gap: critic nhan gradient bang khong, policy ket o
   bien, reward head doan hang so — va tat ca deu "chay" binh thuong.
   nen `health()` do bang SO thay vi do bang LOSS:
       R2 phan thuong, tai tao so voi nen, do lech hanh dong.
   Lo so doan mau van dung, nguoi khac doc cung hieu.

2. DUNG DUNG MOT CHE DO CHAY LAI.
   `--resume` nap lai checkpoint. Tren may thuê, dung ro may thi
   dang chay lai tu dau, mat ca gio may.

3. BAO CAO GHI RA FILE.
   `--log` ghi tung dong JSON. Khong co no thi khong ve duoc gi khi
   dang chay o che do nen.
"""

import argparse
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch

try:
    from .demo import PendulumEnv
    from .soul import Soul
    from .verify import collect_episodes
except ImportError:  # pragma: no cover - chạy trực tiếp như script
    from demo import PendulumEnv
    from soul import Soul
    from verify import collect_episodes


def pick_device() -> str:
    """Tu chon thiet bi. Uu tien GPU, lui ve CPU neu khong co."""
    if torch.cuda.is_available():
        return 'cuda'
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        return 'xpu'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


# ----------------------------------------------------------------------
# Suc khoe
# ----------------------------------------------------------------------
def health(soul: Soul, n_batches: int = 4) -> Optional[Dict[str, float]]:
    """Do bang SO xem mo hinh co that su hoc khong.

    Chi lay nhung phu luc co du lieu. Day la thu tu bat buoc: neu khong
    tinh mau so tren nhung o co so 0, ket qua se bi phinh to (truoc day
    do sai 25 lan chinh vi lay mau so ca 256 o thay vi o co du lieu).
    """
    rng = np.random.default_rng(12345)
    cfg = soul.cfg
    tc = cfg.train
    dev = soul.device

    rew_pred, rew_true, top_slots = [], [], None
    errs, bases = {}, {}
    acts = []
    rets, vals = [], []

    for _ in range(n_batches):
        b = soul.replay.sample(tc.batch_size, rng)
        if b is None:
            continue
        t = lambda a: torch.as_tensor(a, device=dev, dtype=torch.float32)
        v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
        ac, va, rw = t(b['actions']), t(b['valid']), t(b['rewards'])
        gl = t(b['goals'])
        seq = v_.shape[1]
        obs_in = torch.cat([v_, m_, tx], -1)

        with torch.no_grad():
            out = soul.model.dynamics.observe_sequence(
                obs_in, ac, soul.model.encoder, va)
            h, z = out['h_seq'], out['z_seq']
            dec = soul.model.decoder(h, z)
            pr = soul.model.reward_head(h, z)

            n = min(pr.numel(), rw.numel())
            # reward head học trên phần thưởng ĐÃ CHUẨN HÓA. Muốn đo
            # trên thang thật thì phải đải ngược lại — bỏ bước này thì R²
            # luôn âm dù mô hình dự đoán đúng.
            rs = soul.trainer._reward_stats
            raw = pr.reshape(-1)[:n] * max(rs['std'], 1e-6) + rs['mean']
            rew_pred.append(raw.cpu())
            rew_true.append(rw.reshape(-1)[:n].cpu())

            # So o co du lieu nhieu nhat — chi do nhung o do.
            if top_slots is None:
                freq = m_[:, :seq].mean(dim=(0, 1))
                top_slots = torch.topk(freq, min(5, int((freq > 0).sum()))).indices.tolist()
            for i in top_slots:
                msk = m_[:, :seq, i] > 0.5
                if msk.sum() == 0:
                    continue
                g = v_[:, :seq, i][msk]
                p = dec['obs'][:, :, i][msk]
                errs.setdefault(i, []).append(float((p - g).abs().mean()))
                bases.setdefault(i, []).append(float((g.mean() - g).abs().mean()))

            with torch.no_grad():
                mean, _, _ = soul.model.policy_head(h, z, gl[:, :seq])
            acts.append(mean.reshape(-1).cpu())

            # Muoc tieu va gia tri trong tuong tuong — de xem critic co
            # theo kip khong. Truoc day critic phang (std 0.09) nen muc
            # tieu (std 16.7) hoan toan thuoc ve phan thuong.
            hh, zz = h[:, -1], z[:, -1]
            rt, vl = [], []
            for _step in range(tc.imagine_horizon):
                hp, zp = hh, zz
                am, als, _ = soul.model.policy_head(hh, zz, gl[:, -1])
                from .model import MIN_STD, _logstd_to_std
                lsc = torch.maximum(als, torch.full_like(als, float(np.log(MIN_STD))))
                aa = am + torch.randn_like(am) * _logstd_to_std(lsc)
                continuous_dim = aa.shape[-1] - soul.cfg.model.discrete_dim
                applied_aa = torch.cat((
                    aa[..., :continuous_dim].clamp(
                        soul.cfg.model.action_low,
                        soul.cfg.model.action_high,
                    ),
                    aa[..., continuous_dim:],
                ), dim=-1)
                hh, mu, lv = soul.model.dynamics.step(hh, zz, applied_aa)
                from .model import _logvar_to_std
                zz = mu + torch.randn_like(mu) * _logvar_to_std(lv)
                rt.append(soul.model.reward_head(hh, zz).squeeze(-1))
                vl.append(soul.model.value_head(hp, zp, gl[:, -1]).squeeze(-1))
            rt = torch.stack(rt, 1)
            vl = torch.stack(vl, 1)
            tg = torch.zeros_like(rt)
            nx = torch.zeros(rt.shape[0], device=dev)
            for k in reversed(range(rt.shape[1])):
                nx = rt[:, k] + tc.gamma * nx
                tg[:, k] = nx
            rets.append(tg.reshape(-1).cpu())
            vals.append(vl.reshape(-1).cpu())

    if not rew_true:
        return None

    p = torch.cat(rew_pred)
    y = torch.cat(rew_true)
    mse = float(((p - y) ** 2).mean())
    var = float(((y - y.mean()) ** 2).mean())
    r2 = 100 * (1 - mse / max(var, 1e-9))

    A = torch.cat(acts)
    return {
        'reward_r2': r2,
        'reward_pred_std': float(p.std()),
        'reward_true_std': float(y.std()),
        'policy_action_std': float(A.std()),
        'recon_ratio': (float(np.mean(errs[top_slots[0]]))
                        / max(float(np.mean(bases[top_slots[0]])), 1e-9)
                        if top_slots and top_slots[0] in errs else float('nan')),
        'imagine_reward_std': float(torch.cat(rets).std()),
        'critic_std': float(torch.cat(vals).std()),
        'return_std': float(torch.cat(rets).std() * tc.imagine_horizon ** 0.5),
    }


def fmt(h: Dict[str, float]) -> str:
    return (f"R2={h['reward_r2']:6.1f}%  "
            f"std_ph={h['reward_pred_std']:.3f}/{h['reward_true_std']:.3f}  "
            f"tai_tao/nen={h['recon_ratio']:.3f}  "
            f"do_lech_a={h['policy_action_std']:.4f}  "
            f"critic={h['critic_std']:.2f}")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description='Huan luyen Soul')
    ap.add_argument('--steps', type=int, default=20000,
                    help='Tong so buoc hoc')
    ap.add_argument('--collect-every', type=int, default=1,
                    help='So buoc hoc giua hai lan thu du lieu moi')
    ap.add_argument('--updates-per-collect', type=int, default=10,
                    help='So lan goi learn() sau moi lan thu du lieu')
    ap.add_argument('--warmup-episodes', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--device', type=str, default='')
    ap.add_argument('--out', type=str, default='runs/soul')
    ap.add_argument('--checkpoint-every', type=int, default=200)
    ap.add_argument('--health-every', type=int, default=200)
    ap.add_argument('--log', type=str, default='')
    ap.add_argument('--resume', type=str, default='')
    # Si gia de dung
    ap.add_argument('--batch-size', type=int, default=0)
    ap.add_argument('--seq-len', type=int, default=0)
    ap.add_argument('--lr', type=float, default=0.0)
    ap.add_argument('--head-lr', type=float, default=0.0)
    ap.add_argument('--policy-lr', type=float, default=0.0)
    ap.add_argument('--preset', type=str, default='gpu',
                    choices=('gpu', 'local'))
    args = ap.parse_args()

    from .config import Config
    cfg = Config()

    if args.preset == 'local':
        cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
        cfg.train.batch_size, cfg.train.seq_len = 8, 16
        cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
    else:
        # Co GPU. Batch lon hon nhieu — do la thu luong dinh nghia dang
        # chay rat cham tren may 2 nhan, khong phai gia tri dung.
        cfg.model.hidden_dim, cfg.model.latent_dim = 96, 24
        cfg.train.batch_size, cfg.train.seq_len = 64, 24
        cfg.model.action_dim, cfg.model.discrete_dim = 1, 0

    cfg.device = args.device or pick_device()
    for name, val in (('batch_size', args.batch_size),
                      ('seq_len', args.seq_len),
                      ('lr', args.lr), ('head_lr', args.head_lr),
                      ('policy_lr', args.policy_lr)):
        if val:
            setattr(cfg.train, name, val)
    cfg.train.seed = args.seed

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)
    ckpt_path = os.path.join(args.out, 'ckpt.pt')
    log_path = args.log or os.path.join(args.out, 'metrics.jsonl')

    soul = Soul(cfg)
    env = PendulumEnv()
    if args.resume:
        soul.load(args.resume)
        print(f'[train] Da nap: {args.resume}')
    soul.cfg.train.replay_capacity = cfg.train.replay_capacity

    print(f'[train] thiet bi      : {cfg.device}')
    print(f'[train] tham so       : {soul.model.num_parameters():,}')
    print(f'[train] batch/seq     : {cfg.train.batch_size}/{cfg.train.seq_len}')
    print(f'[train] lr            : rep={cfg.train.lr} '
          f'head={cfg.train.head_lr} policy={cfg.train.policy_lr}')
    print(f'[train] muc tieu      : {cfg.train.batch_size * args.updates_per_collect} '
          f'buoc, ghi moi {args.checkpoint_every}')
    print(f'[train] ghi           : {log_path}')
    print('-' * 72)

    print(f'[train] Thu {args.warmup_episodes} episode de lap day replay...')
    collect_episodes(soul, args.warmup_episodes, env, rng)
    print('[train] Xong. Bat dau hoc.')
    print('-' * 72)

    t0 = time.time()
    last_h = None
    logf = open(log_path, 'a', encoding='utf-8')
    try:
        while soul.trainer.updates < args.steps:
            collect_episodes(soul, 1, env, rng)
            for _ in range(args.updates_per_collect):
                soul.learn(force=True)
                if soul.trainer.updates % args.checkpoint_every == 0:
                    soul.save(ckpt_path)
                    print(f'[train] -> {ckpt_path}')
                if soul.trainer.updates % args.health_every == 0:
                    h = health(soul)
                    if h is None:
                        print('[train] replay chua du du lieu')
                        continue
                    el = time.time() - t0
                    print(f'[train] {soul.trainer.updates:>7} buoc | {el:>7.0f}s '
                          f'| {fmt(h)}')
                    logf.write(json.dumps({'step': soul.trainer.updates,
                                           'elapsed': el, **h}) + '\n')
                    logf.flush()
                    last_h = h
    except KeyboardInterrupt:
        print('\n[train] Dung. Luu checkpoint...')
    finally:
        soul.save(ckpt_path)
        logf.close()

    print('-' * 72)
    print(f'[train] Xong {soul.trainer.updates} buoc '
          f'trong {time.time() - t0:.0f}s. Checkpoint: {ckpt_path}')
    if last_h:
        print(f'[train] {fmt(last_h)}')
        if last_h['reward_r2'] < 0:
            print('[train] CANH BAO: reward head doan te hon doan hang so.')
        if last_h['recon_ratio'] > 1.0:
            print('[train] CANH BAO: tai tao te hon doan hang so.')


if __name__ == '__main__':
    main()