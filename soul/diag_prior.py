"""Do prior mot buoc theo cung phuoc do ma nguoi danh gia dung.

Bao cao truoc goi "sai so 1 buoc 10-14 lan so voi doan hang so". Nguoi
danh gia chi ra nen THAY BANG mot moc tham chieu manh hon: **hoi quy
tuyen tinh** — fit `x_{t+1} ~ x_t` va `v_{t+1} ~ v_t` bang mot duong
thang tren chinh tap do. Day la muc "gioi" ma mot mo hinh du don gian
phai vuot qua truoc khi ta noi no "hoc duoc dong luc".

Do them:
  - giữ nguyên trạng thái
  - hồi quy tuyến tính
  - prior cua mo hinh

Va theo doi `post_std` / `post_mean_std` de biet co che can bang nao
dang lam viec. Do mot lan chay bat buoc phai rat lau; cac diem do se
in ra ngay khi den.
"""

import argparse
import sys
import time

import numpy as np
import torch

from .config import Config
from .demo import PendulumEnv
from .model import _logvar_to_std
from .soul import Soul
from .verify import collect_episodes, make_config


def _banner(s):
    print(f'\n{"=" * 78}\n{s}\n{"=" * 78}')


@torch.no_grad()
def measure(soul, ks=(1, 5, 10), rng=None):
    """Sai số mở k bước so với hai mốc tham chiếu."""
    rng = rng or np.random.default_rng(11)
    xi = soul.enc.schema['state.position']
    vi = soul.enc.schema['state.velocity']
    K = max(ks)
    acc = {}

    for _ in range(6):
        b = soul.replay.sample(8, rng)
        if b is None:
            continue
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)
        v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
        ac, va = t(b['actions']), t(b['valid'])
        seq = v_.shape[1] - 1
        if seq - 1 < K:
            continue
        oi = torch.cat([v_, m_, tx], -1)
        o = soul.model.dynamics.observe_sequence(
            oi[:, :seq], ac, soul.model.encoder, va)
        h, z = o['h_seq'][:, 0], o['z_seq'][:, 0]
        base = soul.model.decoder(h, z)['obs']
        bx, bv = base[:, xi], base[:, vi]
        # Hồi quy tuyến tính: least squares trên (giá trị neo -> giá trị
        # ở bước k). Fit RIÊNG cho từng k, trên chính các mẫu đó.
        for k in range(1, K + 1):
            h, mu, _ = soul.model.dynamics.step(h, z, ac[:, k])
            z = mu
            d = soul.model.decoder(h, z)['obs']
            msk = (m_[:, k, xi] > 0.5) & (m_[:, k, vi] > 0.5)
            if msk.sum() < 8:
                continue
            acc.setdefault(k, {'px': [], 'pv': [], 'hx': [], 'hv': [],
                               'lx': [], 'lv': []})
            tx_, tv_ = v_[:, k, xi], v_[:, k, vi]
            acc[k]['px'].append(float((d[:, xi] - tx_)[msk].abs().mean()))
            acc[k]['pv'].append(float((d[:, vi] - tv_)[msk].abs().mean()))
            acc[k]['hx'].append(float((bx - tx_)[msk].abs().mean()))
            acc[k]['hv'].append(float((bv - tv_)[msk].abs().mean()))
            for key, anch, want in (('lx', bx, tx_), ('lv', bv, tv_)):
                a_, w_ = anch[msk], want[msk]
                Am = torch.stack([a_, torch.ones_like(a_)], 1)
                sol = torch.linalg.lstsq(Am, w_.unsqueeze(1)).solution
                pred = (Am @ sol).squeeze(1)
                acc[k][key].append(float((pred - w_).abs().mean()))
    return acc


def latent_stats(soul, rng):
    b = soul.replay.sample(8, rng)
    if b is None:
        return float('nan'), float('nan')
    t = lambda a: torch.as_tensor(a, dtype=torch.float32)
    v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
    ac, va = t(b['actions']), t(b['valid'])
    seq = v_.shape[1] - 1
    oi = torch.cat([v_, m_, tx], -1)
    with torch.no_grad():
        o = soul.model.dynamics.observe_sequence(
            oi[:, :seq], ac, soul.model.encoder, va)
        ps = float(_logvar_to_std(o['post_lv']).mean())
        pm = float(o['post_mu'].std())
    return ps, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=3000)
    ap.add_argument('--at', type=int, nargs='+', default=[500, 1500, 3000])
    args = ap.parse_args()

    cfg = make_config()
    torch.manual_seed(args.seed)
    soul = Soul(cfg)
    rng = np.random.default_rng(args.seed)
    env = PendulumEnv(seed=args.seed)
    collect_episodes(soul, 30, env, rng)

    _banner(f'Huấn {args.steps} bước — chuẩn hoá KL, trần std '
            f'{cfg.train.latent_std_max}')
    print(f'  kl_prior_weight={cfg.train.kl_prior_weight}  '
          f'kl_post_weight={cfg.train.kl_post_weight}  '
          f'free_nats={cfg.train.free_nats}')
    print()
    print(f'  {"bước":>6} | {"prior x":>9}{"giữ x":>8}{"tuyến x":>9}'
          f' | {"prior v":>9}{"giữ v":>8}{"tuyến v":>9}'
          f' | {"post_std":>10}{"post_mean_std":>14}')
    t0 = time.time()
    for mark in args.at:
        while soul.trainer.updates < mark:
            collect_episodes(soul, 1, env, rng)
            for _ in range(25):
                soul.learn(force=True)
        acc = measure(soul, rng=np.random.default_rng(11))
        ps, pm = latent_stats(soul, rng)
        if 1 not in acc:
            continue
        a = acc[1]
        print(f'  {soul.trainer.updates:>6} | {np.mean(a["px"]):>9.4f}'
              f'{np.mean(a["hx"]):>8.4f}{np.mean(a["lx"]):>9.4f}'
              f' | {np.mean(a["pv"]):>9.4f}{np.mean(a["hv"]):>8.4f}'
              f'{np.mean(a["lv"]):>9.4f}'
              f' | {ps:>10.4f}{pm:>14.4f}')
    print(f'  ({time.time() - t0:.0f}s)')

    _banner('Mo k buoc (bước cuối)')
    acc = measure(soul, ks=(1, 5, 10), rng=np.random.default_rng(11))
    print(f'  {"k":>3}{"prior x":>10}{"giữ x":>9}{"tuyến x":>9}'
          f'{"  |":>4}{"prior v":>10}{"giữ v":>9}{"tuyến v":>9}')
    for k in sorted(acc):
        a = acc[k]
        print(f'  {k:>3}{np.mean(a["px"]):>10.4f}{np.mean(a["hx"]):>9.4f}'
              f'{np.mean(a["lx"]):>9.4f}{"  |":>4}{np.mean(a["pv"]):>10.4f}'
              f'{np.mean(a["hv"]):>9.4f}{np.mean(a["lv"]):>9.4f}')
    print()
    print('  prior < tuyến tính  => mô hình học được động lực học.')
    print('  prior chỉ ~ giữ nguyên => mô hình chưa học được gì ngoài quán tính.')


if __name__ == '__main__':
    sys.exit(main())