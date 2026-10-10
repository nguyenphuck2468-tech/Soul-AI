"""Nap du lieu THAT vao replay.

Tren ca phien nay mo hinh chi tung thay con lac don gian. Module nay
la canh cua duong thu — mot tap du lieu co cau truc (so do, cam bien,
chi so) duoc dua vao `ReplayBuffer` y het du lieu sinh trong moi truong.

`ObsEncoder` da xu ly duoc kieu du lieu long nhau (dict, so, chuoi,
list), nen o day chung ta CHI can bien tep tep thanh cau truc long nhau
roi dua vao. Khong can sua gi o `obs.py`.

Dinh dang CSV
-------------
Tren dong = mot transition da thuc thi: observation la `o_{t+1}`, `a0...`
la action THUC SU AP DUNG `a_t`, va reward/done la ket qua cua transition do.
Khong ghi `o_t` canh `a_t`; neu nguon du lieu dang o dang do, phai can lech
observation sang `next_obs` truoc khi nap. Dong terminal la observation cuoi
cua transition cuoi, khong phai mot dong duplicate them.

Cot dac biet:

    ep        so thu tu episode (tu tang)
    done      1 neu day la buoc ket thuc episode
    reward    phan thuong nhan duoc o buoc nay
    a0, a1..  hanh dong lien tuc (cot bat buoc theo so luong chieu)
    <bat ky>  con lai = QUAN SÁT, ghep thanh dict long nhau theo dau cham

Vi du:

    ep,done,reward,a0,temperature,pressure,valve_open
    0,0,-0.12,0.4,21.3,101.2,0.0
    0,1,-0.08,0.2,21.4,101.1,0.0

Cot `temperature` → quan sat `{'temperature': 21.3}`.
Cot `a0` → hanh dong `[0.4]`.

Dinh dang .npz
--------------
Moi mang la mot bien, tat ca cung do dai:

    obs__temperature : (N,) float
    act              : (N, A) float
    reward           : (N,)
    done             : (N,)
    ep               : (N,) int

Moi phan giong `obs__` bien se ghep vao mot dict quan sat chung.
"""

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from .memory import Episode, ReplayBuffer
    from .obs import ObsEncoder
except ImportError:  # pragma: no cover - chạy trực tiếp như script
    from memory import Episode, ReplayBuffer
    from obs import ObsEncoder

# Cot dac biet, khong phai la quan sat.
_META = ('ep', 'done', 'reward', 'goal')
_ACTION_PREFIX = 'a'


def _action_cols(row: Dict[str, str], action_dim: int) -> List[str]:
    """Tên cột hành động theo thứ tự: `a0`, `a1`, ... theo `action_dim`."""
    return [f'{_ACTION_PREFIX}{i}' for i in range(action_dim)
            if f'{_ACTION_PREFIX}{i}' in row]


def _to_action(row: Dict[str, str], action_dim: int) -> np.ndarray:
    """Đọc hành động nhiều chiều từ các cột `a0`, `a1`, ...

    Ban v1 chi doc `a0` va bo qua phan con lai, trong khi tai lieu ghi ro
    `a0, a1..` — hanh dong nhieu chieu bi mat MOT PHAN im lang, va replay
    van " dung" nen khong ai thay.
    """
    a = np.zeros(action_dim, dtype=np.float32)
    for i, col in enumerate(_action_cols(row, action_dim)):
        raw = row.get(col)
        if raw not in (None, ''):
            a[i] = float(raw)
    return a


def _nest(row: Dict[str, str], skip: Sequence[str]) -> Any:
    """Ghep cac cot con lai thanh dict long nhau theo dau cham."""
    out: Dict[str, Any] = {}
    for k, v in row.items():
        if k in skip or v is None or v == '':
            continue
        try:
            val: Any = float(v)
        except ValueError:
            val = v          # giu chuoi nguyen van
        parts = k.split('.')
        cur = out
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
            if not isinstance(cur, dict):
                break
        else:
            cur[parts[-1]] = val
    return out


def _group(rows: List[Dict[str, str]], action_dim: int, goal_dim: int,
           keys: Sequence[str]) -> List[Dict[str, Any]]:
    """Tach cac dong thanh tung episode."""
    episodes: List[Dict[str, Any]] = []
    cur: Dict[str, Any] = {'obs': [], 'act': [], 'rew': [], 'done': []}
    prev_ep = None

    for row in rows:
        ep_id = row.get('ep')
        # Doi so thu tu episode, HOAC gap dong → bat dau episode moi.
        if prev_ep is not None and ep_id is not None and ep_id != prev_ep:
            episodes.append(cur)
            cur = {'obs': [], 'act': [], 'rew': [], 'done': []}
        prev_ep = ep_id

        skip = list(_META) + [k for k in row if k.startswith(_ACTION_PREFIX + '.')
                              or (len(k) > 1 and k[0] == _ACTION_PREFIX
                                  and k[1].isdigit())]
        cur['obs'].append(_nest(row, skip))
        cur['act'].append(_to_action(row, action_dim))
        cur['rew'].append(float(row.get('reward', 0.0) or 0.0))
        cur['done'].append(int(float(row.get('done', 0) or 0)))

    if cur['obs']:
        episodes.append(cur)
    return episodes


# ----------------------------------------------------------------------
# Doc tep
# ----------------------------------------------------------------------
def read_csv(path: str, action_dim: int, goal_dim: int) -> List[Dict[str, Any]]:
    with open(path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    return _group(rows, action_dim, goal_dim, list(rows[0].keys()))


def read_npz(path: str, action_dim: int, goal_dim: int) -> List[Dict[str, Any]]:
    z = np.load(path, allow_pickle=False)
    names = list(z.keys())
    if 'action' in names or 'act' in names:
        act = z['action'] if 'action' in names else z['act']
    else:
        act = np.zeros((len(z['reward']), action_dim), np.float32)
    n = len(act)

    # Gom cac bien `obs__*` vao mot dict chung.
    fields = {}
    for k in names:
        if k.startswith('obs__'):
            fields.setdefault(k[len('obs__'):], []).append(
                np.atleast_1d(z[k]))

    ep_ids = z['ep'] if 'ep' in names else np.zeros(n, int)
    dones = z['done'] if 'done' in names else np.zeros(n, int)
    rews = z['reward'] if 'reward' in names else np.zeros(n, np.float32)

    rows = []
    for i in range(n):
        row: Dict[str, str] = {
            'ep': str(int(ep_ids[i])),
            'done': str(int(dones[i])),
            'reward': str(float(rews[i])),
        }
        # GHI DAY DU tat ca chieu hanh dong. Ban v1 chi ghi `act[i][0]`
        # nen moi truong da doc lai tep npz nhieu chieu deu mat cac chieu
        # con lai.
        for j in range(act.shape[1]):
            row[f'{_ACTION_PREFIX}{j}'] = str(float(act[i][j]))
        for fld, arrs in fields.items():
            val = arrs[0][i]
            row[fld] = str(float(val))
        rows.append(row)
    return _group(rows, action_dim, goal_dim, list(rows[0].keys()))


# ----------------------------------------------------------------------
# Chuẩn hoá theo cột — BẮT BUỘC với dữ liệu thật
# ----------------------------------------------------------------------
def collect_fields(obs: Any, prefix: str = '',
                   acc: Dict[str, List[float]] = None) -> Dict[str, List[float]]:
    """Gom mọi số trong quan sát lồng nhau, theo đường dẫn đầy đủ."""
    if acc is None:
        acc = {}
    if isinstance(obs, dict):
        for k in sorted(obs.keys(), key=str):
            collect_fields(obs[k], f'{prefix}.{k}' if prefix else str(k), acc)
    elif isinstance(obs, (int, float)) and not isinstance(obs, bool):
        acc.setdefault(prefix, []).append(float(obs))
    elif isinstance(obs, (list, tuple)):
        for i, v in enumerate(obs):
            collect_fields(v, f'{prefix}[{i}]', acc)
    return acc


def norm_stats(eps: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Trung bình và độ lệch chuẩn của từng trường quan sát.

    KHÔNG THỂ BỎ QUA với dữ liệu thật. Đo được trên tập mẫu:
    `temperature` có giá trị 20.8..21.2. Nhưng `ObsEncoder` CẮT ở
    `clip = 20.0` TRƯỚC khi squash, nên toàn bộ cột đó sụp về đúng một
    số 0.95238 — std = 0, đúng MỘT giá trị, mất 100% biến thông tin.
    Mọi đại lượng vật lý (nhiệt độ, áp suất, vận tốc) đều vượt 20.

    Sau khi chuẩn hoá z-score, cùng cột đó giữ lại 43.3% biến thông tin
    (phần mất còn lại do squash — chấp nhận được).

    Vì vậy thống kê này BẮT BUỘC phải đi kèm checkpoint: suy luận
    dùng thang khác thang huấn thì mọi dự đoán sai.
    """
    acc: Dict[str, List[float]] = {}
    for e in eps:
        for o in e['obs']:
            collect_fields(o, '', acc)
    out = {}
    for k, vals in acc.items():
        a = np.asarray(vals, np.float64)
        out[k] = {'mean': float(a.mean()),
                  'std': float(a.std()) if a.std() > 1e-8 else 1.0}
    return out


def apply_norm(obs: Any, stats: Dict[str, Dict[str, float]],
               prefix: str = '') -> Any:
    """Chuẩn hoá z-score cho mọi trường có trong `stats`."""
    if isinstance(obs, dict):
        return {k: apply_norm(v, stats,
                             f'{prefix}.{k}' if prefix else str(k))
                for k, v in obs.items()}
    if isinstance(obs, (int, float)) and not isinstance(obs, bool):
        st = stats.get(prefix)
        if st is not None:
            return (float(obs) - st['mean']) / st['std']
        return float(obs)
    if isinstance(obs, (list, tuple)):
        return [apply_norm(v, stats, f'{prefix}[{i}]')
                for i, v in enumerate(obs)]
    return obs


# ----------------------------------------------------------------------
# Dua vao replay
# ----------------------------------------------------------------------
def to_episodes(soul, eps: List[Dict[str, Any]], goal_dim: int,
                stats: Dict[str, Dict[str, float]] = None) -> List[Episode]:
    """Bien cac episode thanh `Episode` da vector hoa san.

    `stats` = thong ke chuan hoa. Neu None, tu tinh tu chinh tap nay.
    """
    out = []
    g = np.zeros(goal_dim, np.float32)
    for e in eps:
        ep = Episode()
        for o, a, r, d in zip(e['obs'], e['act'], e['rew'], e['done']):
            applied = np.asarray(a, dtype=np.float32).reshape(-1)
            expected = soul._format_action(applied)
            if applied.size != soul.cfg.model.action_dim or not np.allclose(
                    applied, expected, rtol=0.0, atol=1e-6):
                raise ValueError(
                    'Imported replay actions must already be applied actions '
                    'within the configured action bounds.')
            if stats is not None:
                o = apply_norm(o, stats)
            enc = soul.enc.encode(o)
            ep.append(enc, applied, float(r), 0.0 if d else 1.0, g)
        if len(ep.actions):
            out.append(ep)
    return out


def ingest(soul, path: str) -> int:
    cfg = soul.cfg
    if path.endswith('.csv'):
        eps = read_csv(path, cfg.model.action_dim, cfg.model.goal_dim)
    elif path.endswith('.npz'):
        eps = read_npz(path, cfg.model.action_dim, cfg.model.goal_dim)
    else:
        raise ValueError(f'khong ho tro dinh dang: {path}')

    stats = norm_stats(eps)
    soul.obs_norm = stats
    built = to_episodes(soul, eps, cfg.model.goal_dim, stats)
    for ep in built:
        soul.replay.add_episode(ep)
    return len(built)


# ----------------------------------------------------------------------
# Sinh tep mau de kiem chung
# ----------------------------------------------------------------------
def make_sample_csv(path: str, n_ep: int = 8, length: int = 120,
                    seed: int = 0) -> str:
    """Sinh tep tep co cau truc: mot con lac nhung doc o dang CSV.

    Dung de kiem chung duong thu MA KHONG can du lieu ngoai tai. Khac
    voi `PendulumEnv` o cho day "mo truong" chi la con so tren giay,
    con thu ban ghi vao tep va doc lai qua duong nap that.
    """
    rng = np.random.default_rng(seed)
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['ep', 'done', 'reward', 'a0',
                    'position', 'velocity', 'temperature'])
        for e in range(n_ep):
            x = rng.uniform(-2, 2)
            v = rng.uniform(-1, 1)
            for t in range(length):
                a = float(np.clip(rng.normal(0, 0.8), -1.0, 1.0))
                v += (-2.0 * x + a) * 0.05
                x += v * 0.05
                x += rng.normal(0, 0.02)
                v += rng.normal(0, 0.02)
                rew = -(x ** 2 + 0.1 * v ** 2)
                done = int(t == length - 1)
                w.writerow([e, done, f'{rew:.6f}', f'{a:.6f}',
                            f'{x:.6f}', f'{v:.6f}', f'{21 + 0.1 * x:.4f}'])
    return path


def main():
    ap = argparse.ArgumentParser(description='Nap du lieu that vao replay')
    ap.add_argument('path', nargs='?', default='')
    ap.add_argument('--make-sample', type=str, default='',
                    help='Sinh tep CSV mau')
    ap.add_argument('--episodes', type=int, default=8)
    ap.add_argument('--length', type=int, default=120)
    ap.add_argument('--train', action='store_true',
                    help='Sau khi nap, chay huanc o')
    args = ap.parse_args()

    if args.make_sample:
        p = make_sample_csv(args.make_sample, args.episodes, args.length)
        print(f'Da sinh tep mau: {p}')
        return 0
    if not args.path:
        ap.error('can path hoac --make-sample')

    from .config import Config
    from .soul import Soul
    cfg = Config()
    cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
    cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
    cfg.train.batch_size, cfg.train.seq_len = 8, 16
    soul = Soul(cfg)

    n = ingest(soul, args.path)
    print(f'Da nap {n} episode vao replay.')
    print(f'  schema quan sat: {soul.enc.schema}')
    print('  chuan hoa theo cot:')
    for k, v in soul.obs_norm.items():
        print(f'    {k:<20} mean={v["mean"]:>10.4f}  std={v["std"]:.4f}')
    lens = [len(e.actions) for e in soul.replay.episodes]
    print(f'  do dai episode : {min(lens)}..{max(lens)}')
    print(f'  buoc tong      : {sum(lens)}')
    if soul.enc.overflow:
        print(f'  CANH BAO: {soul.enc.overflow} truong bi bo vi tran ngan sach.')

    if args.train:
        import time
        t0 = time.time()
        for i in range(300):
            soul.learn(force=True)
        print(f'  da hoc 300 buoc ({time.time() - t0:.0f}s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())