"""TEST NHAN QUA: ban sua lech MOT BUOC co that khong?

`observe_sequence` truoc day dung `actions[:, t-1]` trong khi `obs[t]`
do `actions[t]` sinh ra. Ban sua doi kha nang dung hon, nhung:

  - doc code khong phai bang chung.
  - Do KL cua ca hai cach tren checkpoint da huanc o cau hinh CU cho ra
    0.4913 (dung) va 0.4879 (sai) — gan như bang nhau. Do la do tren
    mo hinh DA THICH NGHI voi quy uoc sai, nen no khong phan biet duoc.

Test nay do tren mo hinh HOC DUNG cach moi:
    Dung hanh dong thật   -> prior phai do khop tot
    Xao tron hanh dong     -> prior phai do te hon

Neu hai so nay cach nhau nhieu, prior THAT SU dung hanh dong, va ban
sua lech-mot-buoc da co hieu luc.

Khong dung lai the nao cung mot kiem chung: neu hai so bang nhau,
test bao "khong phan biet duoc" chu khong ket luan sai.
"""

import argparse
import os
import sys

import numpy as np
import torch

from .config import Config
from .soul import Soul
from .trainer import _kl_normal

CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    '_causal.ckpt')


def _banner(s):
    print(f'\n{"=" * 72}\n{s}\n{"=" * 72}')


def build(seed=0, steps=1200, warmup_episodes=30,
          learn_per_collect=20):
    cfg = Config()
    cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
    cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
    cfg.train.batch_size, cfg.train.seq_len = 8, 16
    torch.manual_seed(seed)
    s = Soul(cfg)
    rng = np.random.default_rng(seed)
    from .demo import PendulumEnv
    from .verify import collect_episodes
    env = PendulumEnv(seed=seed)
    collect_episodes(s, warmup_episodes, env, rng)
    while s.trainer.updates < steps:
        collect_episodes(s, 1, env, rng)
        for _ in range(learn_per_collect):
            s.learn(force=True)
    s.save(CKPT)
    return s


def _shuffle_action_timesteps(actions, valid):
    """Shuffle time positions independently per sequence, preserving padding."""
    if actions.ndim != 3 or valid.shape != actions.shape[:2]:
        raise ValueError('actions must be [batch, time, action] and valid [batch, time]')
    shuffled = actions.clone()
    for batch_index in range(actions.shape[0]):
        indices = torch.nonzero(valid[batch_index] > 0, as_tuple=False).flatten()
        if indices.numel() > 1:
            permutation = torch.randperm(indices.numel(), device=actions.device)
            shuffled[batch_index, indices] = actions[batch_index, indices[permutation]]
    return shuffled


def measure(s, rng, shuffle, n_batches=10):
    tot, n = 0.0, 0
    for _ in range(n_batches):
        b = s.replay.sample(8, rng)
        if b is None:
            continue
        t = lambda a: torch.as_tensor(a, dtype=torch.float32)
        v_, m_, tx = t(b['values']), t(b['masks']), t(b['texts'])
        ac, va = t(b['actions']), t(b['valid'])
        seq = ac.shape[1]
        # The replay batch adds one lookahead observation for target use,
        # while actions/valid cover only the input sequence.
        oi = torch.cat([v_, m_, tx], -1)[:, :seq]
        if shuffle:
            # Hoan vi cac moc thoi gian rieng trong tung chuoi; giu nguyen
            # tap hanh dong cua moi episode va cac vi tri padding.
            ac = _shuffle_action_timesteps(ac, va)
        with torch.no_grad():
            out = s.model.dynamics.observe_sequence(
                oi, ac, s.model.encoder, va)
            kl = _kl_normal(out['prior_mu'], out['prior_lv'],
                            out['post_mu'], out['post_lv']).sum(-1)
            kl = torch.maximum(kl, torch.tensor(0.0)).mean()
        tot += float(kl)
        n += 1
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--steps', type=int, default=1200)
    ap.add_argument('--warmup-episodes', type=int, default=30)
    ap.add_argument('--learn-per-collect', type=int, default=20)
    ap.add_argument('--batches', type=int, default=10)
    args = ap.parse_args()
    if os.path.exists(CKPT):
        print(f'  Load checkpoint: {CKPT}')
        cfg = Config()
        cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
        cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
        cfg.train.batch_size, cfg.train.seq_len = 8, 16
        s = Soul(cfg)
        s.load(CKPT)
    else:
        _banner('Huan 1200 buoc (lan dau)')
        s = build(args.seed, args.steps, args.warmup_episodes,
                  args.learn_per_collect)
        print('  xong.')
    rng = np.random.default_rng(args.seed + 7)

    _banner('KL prior||posterior cua hanh dung va hanh xao tron')
    real = measure(s, rng, False, n_batches=args.batches)
    torch.manual_seed(args.seed + 7)
    shuf = measure(s, np.random.default_rng(args.seed + 7), True,
                   n_batches=args.batches)
    print(f'  hanh dong THAT     : KL = {real:.4f} nats')
    print(f'  hanh dong XAO TRON : KL = {shuf:.4f} nats')
    print(f'  chenh lech         : {shuf - real:+.4f} nats')

    _banner('Ket luan')
    if shuf > real * 1.05:
        print('  Prior DO TOT HON khi dung hanh dong that. -> prior that su')
        print('  dung hanh dong, va ban sua lech-mot-buoc CO hieu luc.')
    elif shuf < real:
        print('  KHONG phan biet duoc (xao tron ra KL nho hon). Ban sua')
        print('  lech-mot-buoc co the KHONG co tac dung — can xem lai.')
    else:
        print('  KHONG phan biet duoc: hai so gan bang nhau. Chua co bang')
        print('  chung nao rang prior co that dung hanh dong hay khong.')


if __name__ == '__main__':
    sys.exit(main())
