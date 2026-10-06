"""Kiem chung mo hinh that su hoc — do duoc, khong tin vao bao cao.

Tinh huong gay chanh: mot mo hinh co the chay, in ra loss binh thuong,
va khong thuc su hoc gi — vi mot loss duoc tinh sai, hoac gradient khong
chay toi mot phan nao do. Day la loai loi nguy hiem nhat: moi thu deu xanh
truoc mat nguoi doc, nhung phan hoc chinh thi khong ton tai.

Moi phep kiem o day deu la MOT SO DO doc lap, co bang chung in ra.

Chay:
    python -m soul.verify
"""

import argparse
import math
import time
from typing import Dict, List

import numpy as np
import torch

try:
    from .config import Config
    from .soul import Soul
    from .demo import PendulumEnv
    from .model import PolicyHead
except ImportError:  # pragma: no cover - chạy trực tiếp như script
    from config import Config
    from soul import Soul
    from demo import PendulumEnv
    from model import PolicyHead


def banner(title: str):
    print()
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)


def check(name: str, ok: bool, detail: str = ''):
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ''))
    return ok


def make_config(seq_len=16, batch=8) -> Config:
    cfg = Config()
    cfg.model.hidden_dim = 64
    cfg.model.latent_dim = 16
    cfg.model.action_dim = 1
    cfg.model.discrete_dim = 0
    cfg.model.goal_dim = 4
    cfg.train.batch_size = batch
    cfg.train.seq_len = seq_len
    cfg.train.updates_per_batch = 1
    cfg.device = 'cpu'
    return cfg


def collect_episodes(soul: Soul, n_ep: int, env: PendulumEnv, rng, action_fn=None):
    """Chay n episode, nap day replay, khong hoc."""
    for _ in range(n_ep):
        env.reset(rng)
        obs = env._obs()
        for _ in range(200):
            if action_fn is None:
                action = soul.act(obs, deterministic=False)
            else:
                action = action_fn(env, soul)
            obs, reward, done, _ = env.step(action)
            soul.observe(obs, action, reward, done)
            if done:
                break


# ----------------------------------------------------------------------
def test_weights_actually_move(seed=0):
    """Trong so co thay doi sau khi hoc khong?"""
    banner("1. Trong so co thay doi khong")
    torch.manual_seed(seed)
    cfg = make_config()
    soul = Soul(cfg)
    rng = np.random.default_rng(seed)
    env = PendulumEnv()

    before = {k: v.detach().clone() for k, v in soul.model.state_dict().items()
              if v.dtype.is_floating_point}

    collect_episodes(soul, 10, env, rng)
    n_updates = 0
    for _ in range(60):
        if soul.learn():
            n_updates += 1

    after = soul.model.state_dict()
    changed, total, biggest = 0, 0, 0.0
    for k, b in before.items():
        a = after[k]
        d = float((a - b).abs().max())
        total += 1
        if d > 1e-8:
            changed += 1
        biggest = max(biggest, d)

    ok = check("co trong so thay doi", changed > 0,
               f"{changed}/{total} tensor doi, bien do lon nhat {biggest:.2e}")
    check("so buoc hoc > 0", n_updates > 0, f"{n_updates} buoc")
    return ok and n_updates > 0


def test_policy_receives_gradient(seed=0):
    """Gradient co that su chay vao tham so cua policy khong?

    Day la phep kiem chong "policy la mo hinh khong hoc". Neu loss tinh sai
    (vi du nhan truoc khi tinh trung binh), loss se ra 0 va gradient se
    bang khong — moi thu van chay binh thuong.
    """
    banner("2. Gradient co chay vao policy khong")
    torch.manual_seed(seed)
    cfg = make_config()
    soul = Soul(cfg)
    rng = np.random.default_rng(seed)
    env = PendulumEnv()
    collect_episodes(soul, 10, env, rng)
    for _ in range(30):
        soul.learn()

    batch = soul.replay.sample(cfg.train.batch_size, rng)
    dev = soul.device

    def t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=dev)

    values, masks, texts = t(batch['values']), t(batch['masks']), t(batch['texts'])
    actions, goals, valid = t(batch['actions']), t(batch['goals']), t(batch['valid'])
    seq = batch['values'].shape[1] - 1
    obs_in = torch.cat([values, masks, texts], dim=-1)

    out = soul.model.dynamics.observe_sequence(
        obs_in[:, :seq], actions, soul.model.encoder, valid)
    h, z = out['h_seq'], out['z_seq']
    g = goals[:, :seq]

    mean, logstd, logits = soul.model.policy_head(h, z, g)
    logp = soul.model.policy_head.log_prob(mean, logstd, logits, actions)
    # Bo qua valid: tinh don gian tren ca chuoi.
    loss = -(logp * torch.randn_like(logp)).mean()
    soul.model.zero_grad(set_to_none=True)
    loss.backward()

    grads = {n: (p.grad.abs().max().item() if p.grad is not None else 0.0)
             for n, p in soul.model.policy_head.named_parameters()}
    alive = sum(1 for v in grads.values() if v > 0)
    ok = check("policy nhan gradient", alive > 0,
               f"{alive}/{len(grads)} tham so co grad > 0")
    top = sorted(grads.items(), key=lambda kv: -kv[1])[:3]
    print(f"       gradient lon nhat: {', '.join(f'{k}={v:.2e}' for k, v in top)}")
    return ok


def test_recon_improves(seed=0, steps=120):
    """Loss tai tao giam tren MOT lo co dinh khong?

    Day la phep kiem chung manh nhat cho "co that su hoc gi": neu mo hinh
    khong doi duoc gi, loss tren cung mot lo du lieu se khong doi.
    """
    banner("3. Loss tai tao giam tren mot lo co dinh")
    torch.manual_seed(seed)
    cfg = make_config()
    soul = Soul(cfg)
    rng = np.random.default_rng(seed)
    env = PendulumEnv()
    collect_episodes(soul, 10, env, rng)

    batch = soul.replay.sample(cfg.train.batch_size, rng)
    first = last = None
    history: List[float] = []
    for i in range(steps):
        losses = soul.trainer.update(batch)
        history.append(losses['recon'])
        if i == 0:
            first = losses['recon']
        last = losses['recon']

    drop = (first - last) / max(first, 1e-9) * 100
    print(f"       recon buoc 1 : {first:.5f}")
    print(f"       recon buoc {steps} : {last:.5f}")
    print(f"       giam {drop:+.1f}%")
    return check("recon giam it nhat 30%", drop > 30.0, f"{drop:+.1f}%")


def test_reward_predictor(seed=0, steps=120):
    """Dau du doan phan thuong co hoc khong?"""
    banner("4. Dau du doan phan thuong co hoc khong")
    torch.manual_seed(seed)
    cfg = make_config()
    soul = Soul(cfg)
    rng = np.random.default_rng(seed)
    env = PendulumEnv()
    collect_episodes(soul, 10, env, rng)

    batch = soul.replay.sample(cfg.train.batch_size, rng)
    first = last = None
    for i in range(steps):
        losses = soul.trainer.update(batch)
        if i == 0:
            first = losses['reward']
        last = losses['reward']

    drop = (first - last) / max(first, 1e-9) * 100
    print(f"       reward loss buoc 1 : {first:.4f}")
    print(f"       reward loss buoc {steps} : {last:.4f}")
    print(f"       giam {drop:+.1f}%")
    return check("reward loss giam it nhat 30%", drop > 30.0, f"{drop:+.1f}%")


def test_imagination_accuracy(seed=0):
    """Mo hinh doan duoc buoc tiep theo khong?

    Day la dac tinh cua mo hinh the gioi: sau khi huan luyen, KHONG can
    quan sat that, chi can hanh dong, no van chay duoc va do duoc.
    """
    banner("5. Do chinh xac khi tuong tuong (khong co quan sat that)")
    torch.manual_seed(seed)
    cfg = make_config()
    soul = Soul(cfg)
    rng = np.random.default_rng(seed)
    env = PendulumEnv()
    collect_episodes(soul, 12, env, rng)
    for _ in range(300):
        soul.learn()

    batch = soul.replay.sample(cfg.train.batch_size, rng)
    dev = soul.device

    def t(a):
        return torch.as_tensor(a, dtype=torch.float32, device=dev)

    values, masks, texts = t(batch['values']), t(batch['masks']), t(batch['texts'])
    actions = t(batch['actions'])
    seq = batch['values'].shape[1] - 1
    obs_in = torch.cat([values, masks, texts], dim=-1)

    # Do TAI SAU khi da co quan sat o dau. Khong do buoc 0: buoc do mo hinh
    # chua thay gi, phai doan trang thai khoi dau ngau nhien — dieu bat kha
    # thi, va truong hop do khong noi gi ve chat luong mo hinh.
    with torch.no_grad():
        # Nhan quan sat dau tien vao trang thai an.
        q_mu, _ = soul.model.encoder(obs_in[:, 0])
        h, z = soul.model.dynamics.initial_state(batch['values'].shape[0], dev)
        h, _, _ = soul.model.dynamics.step(h, z, torch.zeros_like(actions[:, 0]))
        z = q_mu

        errs: List[float] = []
        per_step: Dict[int, float] = {}
        for t_i in range(seq - 1):
            h, p_mu, _ = soul.model.dynamics.step(h, z, actions[:, t_i])
            z = p_mu
            dec = soul.model.decoder(h, z)['obs']
            m = masks[:, t_i + 1]
            e = float(((dec - values[:, t_i + 1]).pow(2) * m).sum()
                      / m.sum().clamp(min=1))
            errs.append(e)
            per_step[t_i + 1] = e

    mean_err = float(np.mean(errs))

    # Duong mau: gia tri nhe trung binh — doan ra tung bieu do trung binh 0.
    # Neu mo hinh chi "hoc duoc phan bien" thi sai so cua no se ngang
    # hoac lon hon muc nay.
    #
    # PHONG DO PHAI GIONG VOI CON SO O TREN. Sai so duoc tinh tren co o co
    # mask=1, nen phan bien mau tham chieu cung phai chi lay cac o do.
    # Tinh tren ca 256 o se lam mau so bay xuong: o chong trong luon bang 0,
    # chung keo phan bien ve gan 0, va tia le sai/do lech bi thoi phong len
    # hang chuc lan — bao cho mo hinh te khong dung, trong khi mo hinh
    # hoan toan khoe. (Do sai lot nay da lam mot lan.)
    v_ref = values[:, 1:seq].cpu().numpy()
    m_ref = masks[:, 1:seq].cpu().numpy() > 0.5
    n_active = int(m_ref.sum())
    base = float(v_ref[m_ref].var()) if n_active > 0 else 0.0
    ratio = mean_err / max(base, 1e-9)
    print(f"       so o co du lieu that      : {n_active}/{m_ref.size} "
          f"({100.0 * n_active / m_ref.size:.2f}%)")
    print(f"       sai so trung binh 1 buoc : {mean_err:.6f}")
    print(f"       phan bien mau tham chieu  : {base:.6f}")
    print(f"       ti le sai / do lech       : {ratio:.4f}")
    if ratio < 1.0:
        print(f"       -> mo hinh giai thich {100.0 * (1.0 - ratio):.1f}% "
              f"phan bien du lieu")
    for s in (1, 4, 8, 12):
        if s in per_step:
            print(f"       buoc {s:2d}: sai={per_step[s]:.6f} "
                  f"(x{per_step[s] / max(base, 1e-9):.2f} duong mau)")
    return check("du doan sau 1 buoc tot hon ngau nhienh nhieu", ratio < 1.0,
                 f"ti le {ratio:.4f}")


def test_policy_improves(seed=0, episodes=90):
    """Chinh sach co lam tot diem thuong khong?"""
    banner("6. Chinh sach co lam diem thuong tot hon khong")
    cfg = make_config()
    env = PendulumEnv()

    def evaluate(soul, n=12):
        vals = []
        r = np.random.default_rng(seed + 999)
        for _ in range(n):
            env.reset(r)
            obs = env._obs()
            soul.set_goal(env.goal_vec())
            tot = 0.0
            for _ in range(200):
                a = soul.act(obs, deterministic=True)
                obs, rew, done, _ = env.step(a)
                tot += rew
                if done:
                    break
            vals.append(tot)
        return float(np.mean(vals)), float(np.std(vals))

    # A. Chinh sach ngau nhienh — moc so doi.
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    s_rand = Soul(cfg)
    s_rand.replay = s_rand.replay  # giu nguyen
    r = np.random.default_rng(seed + 999)
    rand_vals = []
    for _ in range(12):
        env.reset(r)
        tot = 0.0
        for _ in range(200):
            a = np.random.uniform(-1, 1, size=cfg.model.action_dim).astype(np.float32)
            _, rew, done, _ = env.step(a)
            tot += rew
            if done:
                break
        rand_vals.append(tot)
    rand_mean = float(np.mean(rand_vals))

    # B. Truoc khi hoc.
    torch.manual_seed(seed)
    soul = Soul(cfg)
    collect_episodes(soul, 12, env, rng)
    before_mean, before_std = evaluate(soul)

    # C. Hoc.
    t0 = time.time()
    for _ in range(episodes):
        collect_episodes(soul, 1, env, rng)
        for _ in range(20):
            soul.learn()
    train_s = time.time() - t0

    after_mean, after_std = evaluate(soul)

    print(f"       ngau nhienh      : {rand_mean:10.2f}")
    print(f"       truoc khi hoc    : {before_mean:10.2f} +- {before_std:.2f}")
    print(f"       sau khi hoc     : {after_mean:10.2f} +- {after_std:.2f}")
    print(f"       thoi gian hoc    : {train_s:.1f}s")
    return check("sau khi hoc tot hon truoc khi hoc", after_mean > before_mean,
                 f"{before_mean:.1f} -> {after_mean:.1f}")


def test_save_load(seed=0):
    """Luu va nap co giu nguyen du lieu va hanh vi khong?"""
    banner("7. Luu / nap checkpoint")
    cfg = make_config()
    rng = np.random.default_rng(seed)
    env = PendulumEnv()
    soul = Soul(cfg)
    collect_episodes(soul, 10, env, rng)
    for _ in range(30):
        soul.learn()

    soul.save('verify_tmp_ckpt.pt')

    soul2 = Soul(cfg, checkpoint='verify_tmp_ckpt.pt')

    ok_schema = (soul.enc.schema == soul2.enc.schema)
    ok_w = all(torch.allclose(soul.model.state_dict()[k], soul2.model.state_dict()[k])
               for k in soul.model.state_dict())
    ok_steps = soul.steps == soul2.steps

    # Hanh vi phai giong: cung quan sat, cung trang thai, cung muc tieu.
    o = env._obs()
    a1 = soul.act(o, deterministic=True)
    a2 = soul2.act(o, deterministic=True)
    ok_act = bool(np.allclose(a1, a2))

    ok_replay = len(soul.replay.episodes) == len(soul2.replay.episodes)

    import os
    try:
        os.remove('verify_tmp_ckpt.pt')
    except OSError:
        pass

    check("schema giong nhau", ok_schema)
    check("trong so giong nhau", ok_w)
    check("so buoc giong nhau", ok_steps)
    check("replay giong nhau", ok_replay)
    check("hanh dong giong nhau", ok_act)
    return all([ok_schema, ok_w, ok_steps, ok_replay, ok_act])


def test_obs_encoder_independent(seed=0):
    """Encoder quan sat co on dinh khong (cung duong dan -> cung o)?"""
    banner("8. Bo chuan hoa du lieu on dinh va dung")
    try:
        from .obs import ObsEncoder
    except ImportError:  # pragma: no cover - chạy trực tiếp như script
        from obs import ObsEncoder
    cfg = Config()
    e1 = ObsEncoder(cfg.obs)
    e2 = ObsEncoder(cfg.obs)
    obs = {'a': {'b': 1.5, 'c': 'hello'}, 'n': 42, 'flag': True}
    v1, v2 = e1.encode(obs), e2.encode(obs)
    ok_same = np.array_equal(v1['values'], v2['values'])

    # True phai doc truoc int, neu khong True se thanh so 1.
    e3 = ObsEncoder(cfg.obs)
    v3 = e3.encode({'flag_true': True, 'flag_false': False})
    names = e3.feature_names()
    iv, im = names.index('flag_true'), names.index('flag_false')
    ok_bool = (v3['values'][iv] == 1.0 and v3['values'][im] == 0.0
               and v3['mask'][iv] == 1.0 and v3['mask'][im] == 1.0)

    # Gia tri lon phai bi bop ve, khong lam NaN.
    e4 = ObsEncoder(cfg.obs)
    v4 = e4.encode({'big': 1e9, 'neg': -1e9, 'nan': float('nan'), 'inf': float('inf')})
    ok_clamp = bool(np.all(np.isfinite(v4['values'])))
    # Truong NaN/inf co bi dua vao so khong? Khong — no phai bi bo qua
    # hoan toan, ngay ca ca o trong schema: dua vao thi mo hinh se hoc
    # "cho nay luon trong", roi mot ngay na do co so that no khong doan
    # duoc. Bo qua nghia la "khong co du lieu o cho", dung hon.
    names4 = e4.feature_names()
    ok_nan_absent = ('nan' not in names4) and ('inf' not in names4)
    ok_nan_absent = ok_nan_absent and ('big' in names4) and ('neg' in names4)

    check("hai encoder cho cung ket qua", ok_same)
    check("bool doc dung (khong phai 1/0)", ok_bool)
    check("gia tri lon khong tao NaN/Inf", ok_clamp)
    check("truong NaN/inf bi bo qua hoan toan", ok_nan_absent)
    return all([ok_same, ok_bool, ok_clamp, ok_nan_absent])


def test_no_nan(seed=0):
    """Co con so NaN nao lan ra khong?"""
    banner("9. Khong co NaN lan rao")
    cfg = make_config()
    rng = np.random.default_rng(seed)
    env = PendulumEnv()
    soul = Soul(cfg)
    collect_episodes(soul, 10, env, rng)
    bad = []
    for i in range(40):
        l = soul.learn()
        if not l:
            continue
        for k, v in l.items():
            if not np.isfinite(v):
                bad.append((i, k, v))
    for k, v in soul.model.state_dict().items():
        if v.dtype.is_floating_point and not torch.isfinite(v).all():
            bad.append(('param', k, float('nan')))
    if bad:
        for b in bad[:6]:
            print(f"       NaN tai {b}")
    return check("khong co Na/Inf trong loss hay trong so", len(bad) == 0,
                 f"{len(bad)} ket qua khong hop le")


def test_act_uses_observation(seed=0):
    """Hành động có PHỤ THUỘC quan sát không?

    Bắt đúng lỗi: `act` tính ra vector quan sát rồi bỏ đi, chỉ lan bằng
    prior — lúc đó policy mù hoàn toàn, hai thế giới khác nhau cho cùng
    một hành động. Mọi thứ vẫn chạy, loss vẫn giảm, nên lỗi này không bao
    giờ lộ ra nếu không hỏi đúng câu này.
    """
    banner("10. Hành động có phản ứng với quan sát không")
    torch.manual_seed(seed)
    cfg = make_config()
    soul = Soul(cfg)

    def act_from(obs, state):
        soul._h, soul._z, soul._prev_action = state
        a = soul.act(obs, deterministic=True)
        return a, (soul._h.clone(), soul._z.clone())

    # Cung mot trang thai an, doi quan sat, phai ra hanh dong khac nhau.
    base = {'state': {'position': 2.0, 'velocity': 0.5}, 'time': 0}
    far = {'state': {'position': -2.0, 'velocity': -0.5}, 'time': 0}

    st0 = (torch.zeros(1, soul.model.hidden), torch.zeros(1, soul.model.latent),
           np.zeros(cfg.model.action_dim, dtype=np.float32))
    a0, st1 = act_from(base, st0)
    a1, _ = act_from(far, (st1[0], st1[1], np.asarray(a0, dtype=np.float32)))

    check("cung trang thai, khac quan sat -> khac hanh dong",
          not np.allclose(a0, a1),
          f"{np.round(a0, 4).tolist()} vs {np.round(a1, 4).tolist()}")

    # Va phai phan biet duoc cac gia tri sat nhau.
    st2 = (torch.zeros(1, soul.model.hidden), torch.zeros(1, soul.model.latent),
           np.zeros(cfg.model.action_dim, dtype=np.float32))
    a2, _ = act_from({'state': {'position': 0.0, 'velocity': 0.0}, 'time': 0}, st2)
    a3, _ = act_from({'state': {'position': 0.5, 'velocity': 0.0}, 'time': 0}, st2)
    ok_dist = not np.allclose(a2, a3)
    check("vi tri sat nhau van phai ra hanh dong khac", ok_dist,
          f"{np.round(a2, 4).tolist()} vs {np.round(a3, 4).tolist()}")

    # Hanh dong phai nam trong khoang hop le.
    ok_range = bool(np.all(np.abs(a0) <= 1.0) and np.all(np.abs(a1) <= 1.0))
    check("hanh dong trong [-1, 1]", ok_range)
    return ok_dist and ok_range


def test_log_density_consistent(seed=0):
    """`log_prob` có khớp với phân bố mà `sample` thực sự lấy mẫu không?

    Bắt đúng lỗi: dùng chung một hàm `exp(0.5*x)` cho cả log-phương sai
    (encoder/dynamics) và log-độ-lệch-chuẩn (policy). Với policy, độ lệch
    chuẩn phải là `exp(logstd)`. Sai thì mẫu vẫn ra số hợp lệ, loss vẫn
    giảm — nhưng REINFORCE tối ưu sai mật độ, và policy học sai.

    Phép kiểm: mật độ Gaussian chuẩn hoá phải tích phân bằng 1.
    """
    banner("11. Mật độ xác suất của policy chuẩn hoá đúng")
    torch.manual_seed(seed)
    dev = torch.device('cpu')
    logstd = torch.tensor([[0.0, 0.5, -1.0]], device=dev)   # (1,3)

    # Tham chieu phai dung CHUNG std ma policy that su dung.
    #
    # `PolicyHead` dat san `MIN_STD`: `sample`, `log_prob` va `entropy`
    # deu chan std o san do. Do la MOT PHAN DINH NGhia cua policy —
    # std thuc su la `max(exp(logstd), MIN_STD)`.
    #
    # Ban v1 cua test bo qua san: lay `exp(logstd)` truc tiep. Phan tu
    # thu ba co `logstd = -1.0` tuc std = 0.368 < MIN_STD = 0.6, nen
    # tham chieu va model lech nhau `ln(0.6) - ln(0.368) = 0.489` — chinh
    # xac bang muc lech test bao loi.
    #
    # Khac phai sua model: sua tham chieu. `sample` va `log_prob` phai
    # cung std, neu khong REINFORCE uoc luong gradient cua mot phan bo
    # ma policy khong bao gio lay mau tu — ca loi mau hon.
    try:
        from .model import MIN_STD
    except ImportError:  # pragma: no cover - chạy trực tiếp như script
        from model import MIN_STD
    ls_eff = torch.maximum(logstd.clamp(-8.0, 2.0),
                           torch.full_like(logstd, math.log(MIN_STD)))
    std = ls_eff.exp()
    grid = torch.linspace(-12.0, 12.0, 24001, device=dev).view(1, -1, 1)
    dens = torch.exp(-0.5 * (grid / std.unsqueeze(1)).pow(2)) / (
        std.unsqueeze(1) * math.sqrt(2 * math.pi))
    step = float(grid[0, 1, 0] - grid[0, 0, 0])
    area = (dens * step).sum(1)  # (1,3)
    ok_norm = bool(torch.allclose(area, torch.ones_like(area), atol=2e-3))
    check("mật độ tích phân = 1", ok_norm,
          f"{[round(float(v), 5) for v in area.reshape(-1)]}")

    # `log_prob` phải bằng log của mật độ đó, đúng cả phần chuẩn hoá.
    try:
        from .model import MIN_STD
    except ImportError:  # pragma: no cover - chạy trực tiếp như script
        from model import MIN_STD
    head = PolicyHead(8, 4, 3, 0, 0)
    with torch.no_grad():
        mean = torch.zeros(1, 3)
        x = torch.tensor([[0.3, -0.7, 1.1]])
        got = head.log_prob(mean, logstd, None, x)
        want = (-0.5 * ((x - mean) / std).pow(2)
                - ls_eff - 0.5 * math.log(2 * math.pi)).sum(-1)
    ok_val = bool(torch.allclose(got, want, atol=1e-5))
    check("log_prob khop mat do", ok_val,
          f"got={float(got.reshape(-1)[0]):.5f} want={float(want.reshape(-1)[0]):.5f}")

    # Entropy phai khop cong thuc: 0.5*log(2*pi*e*s^2).
    with torch.no_grad():
        ent = head.entropy(logstd, None)
    # Entropy cua Gauss 1D = 0.5*log(2*pi*e*s^2) = 0.5*log(2*pi*e) + log(s)
    # va log(s) chinh la `logstd` (do day la log-LECH-CHUAN).
    # Entropy phai dung std HIEU LUC cua policy, da chan san.
    want_e = (0.5 * math.log(2 * math.pi * math.e) + ls_eff).sum(-1)
    ok_ent = bool(torch.allclose(ent, want_e, atol=1e-4))
    check("entropy khop cong thuc Gauss", ok_ent,
          f"got={float(ent.reshape(-1)[0]):.5f} want={float(want_e.reshape(-1)[0]):.5f}")
    return ok_norm and ok_val and ok_ent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true', help='bo qua bai test cham')
    args = ap.parse_args()

    print("Kiem chung mo hinh Soul — ket luan dua tren SO DO, khong tren bao cao")
    t0 = time.time()

    results = []
    results.append(('trong so doi', test_weights_actually_move()))
    results.append(('gradient vao policy', test_policy_receives_gradient()))
    results.append(('recon giam', test_recon_improves()))
    results.append(('reward loss giam', test_reward_predictor()))
    results.append(('chuan hoa du lieu', test_obs_encoder_independent()))
    results.append(('khong NaN', test_no_nan()))
    results.append(('tuong tuong chinh xac', test_imagination_accuracy()))
    results.append(('act dung quan sat', test_act_uses_observation()))
    results.append(('mat do policy chuan hoa', test_log_density_consistent()))
    results.append(('luu/nap', test_save_load()))
    if not args.quick:
        results.append(('policy tot diem thuong', test_policy_improves()))

    banner("TONG KET")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print()
    print(f"  {passed}/{len(results)} phep kiem dat")
    print(f"  thoi gian: {time.time() - t0:.1f}s")
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == '__main__':
    main()