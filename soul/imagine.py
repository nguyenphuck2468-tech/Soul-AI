"""Hoc policy TRONG TUONG TUONG (cach Dreamer lam).

Vien kinh nghiem do duoc, khong phai doan:
    REINFORCE tren du lieu replay co TIN HIEU DUNG — tuong quan giua
    advantage va ket qua that la +0.99. Nhung hanh dong sau 40 episode
    van giong het nhau o moi trang thai (do lech 0.00000).

Vi sao? REINFORCE uoc luong gradient bang MOT mau tai moi buoc:
"tang xac suat cua duy nhat hanh dong da chon". Khi moi hanh dong duoc
chon NGAY NHIEN (off-policy), gradient tro thanh "nen chon hanh dong nao
da duoc chon nhieu hon" — moi bieu thuc tan, mat het thong tin ve trang
thai. Do la hinh hoc cua REINFORCE tren du lieu thu duoc co chu dich
chinh sach, khong phai loi lap trinh.

Cach Dreamer giai quyet: KHONG lay mau hanh dong trong du lieu. Thay vao
do no mo mo hinh the gioi ra phia truoc, tu sinh chuoi hanh dong, roi
hoc bang phan bo moc (categorical) — ham muc tieu khac nhau cho moi
hanh dong, nen moi hanh dong deu duoc cham diem, khong can "vai mau".

Tren day, vi moi truong con lac rat don gian, ta dung ban rut gon:
  - tuong tuong H chuoi, dung prior (khong co quan sat)
  - moc = hien tai, ha (decay) lan theo chieu dai chuoi
  - loss = -trong so * log_prob cua hanh dong DA CHON tren moc do
  - entropy bonus thuc su (khong phai 1e-4)
  - lam mat lan cho thong so khong thep

Khac REINFORCE: ham muc tieu co TAT CA hanh dong cung thay doi, va
tín hiệu chay qua `advantage` da khử baseline nen policy khong bị
kéo ve hành động trung bình.
"""

from typing import Dict, Tuple

import math

import torch
import torch.nn.functional as F

try:
    from .model import MIN_STD, _logstd_to_std, _logvar_to_std
except ImportError:  # pragma: no cover - chạy trực tiếp như script
    from model import MIN_STD, _logstd_to_std, _logvar_to_std


def imagine_actor_critic(model, h0: torch.Tensor, z0: torch.Tensor,
                         goal: torch.Tensor, reward_head, continue_head,
                         value_head, policy_head,
                         gamma: float, lam: float = 0.95,
                         horizon: int = 12) -> Dict[str, torch.Tensor]:
    """Mo chuoi tuong tuong va tra ve loss actor + critic.

    h0, z0 : (B, hidden) / (B, latent) — trang thai cuoi cua chuoi that.
    goal   : (B, goal_dim)
    """
    B = h0.shape[0]
    h, z = h0, z0

    # Moc muc tieu lui theo chieu dai chuoi: buoc nao xa hon thi muc
    # tieu cua no cang "nhe" — dung `gamma^t` de chiet khau.
    w = torch.tensor([gamma ** t for t in range(horizon)],
                     device=h0.device, dtype=h0.dtype).unsqueeze(0)

    # GOM TRANG THAI TRUOC KHI HANH DONG.
    #
    # Truoc day vong lap luu `hs[t] = s_{t+1}` — trang thai SAU khi buoc
    # dynamics. Nhung `rts[t] = R(s_{t+1})` cung thuoc ve `s_{t+1}`, nen
    # `adv[t] = targets[t] - vs[t]` danh gia chat luong cua `s_{t+1}`...
    # roi lai nhan voi `logp` cua hanh dong da chon tai `s_t`. Hai trang
    # thai KHAC NHAU. Policy bi day: "tang xac suat hanh dong da chon o
    # s_t" theo chat luong cua mot trang thai khong lien quan.
    #
    # Dinh nghia chuan: V(s_t) = r_{t+1} + gamma * V(s_{t+1}), nen
    # `targets[t]` (bat dau bang `rts[t]`) uoc luong chinh la V(s_t).
    # Vay `vs` phai do tren `s_t` — trang thai TRUOC khi hanh dong.
    hs, zs, rts, vs, acts, cont = [], [], [], [], [], []
    for t in range(horizon):
        # s_t — dung chinh trang thai nay cho policy va cho critic.
        hs.append(h)
        zs.append(z)

        mean, logstd, logits = policy_head(h, z, goal)
        # LAY MAU THEO KIEU TAI THONG TRAC (reparameterized).
        #
        # Truoc day dung `sample()` roi uoc luong gradient bang `log_prob`.
        # Do do `advantage · eps` ma do duoc `corr(adv, eps)` = -0.04 den
        # +0.02 — tuc khong co tin hieu. Ly do bang SO DO:
        #
        #   uoc luong do co phan bien GIUA CUNG TRANG THAI, do khac biet
        #   giua cac trang thai (std advantage ~21-32), con phan do HANH
        #   DONG gay ra chi ~0.13. Ti so tin hieu ~1%.
        #
        # Mo hinh the gioi o day khac vi sai: dynamics `step` va reward
        # deu VI SAI, nen ta lay duoc gradient qua ca cuon roll thay vi
        # phai uoc luong bang mau. Day cung la cach Dreamer lam voi
        # hanh dong lien tuc.
        ls = torch.maximum(logstd, torch.full_like(
            logstd, math.log(MIN_STD)))
        if policy_head.cont_dim:
            cont_action = mean + torch.randn_like(mean) * _logstd_to_std(ls)
        else:
            cont_action = mean
        if policy_head.discrete_dim:
            # Straight-through Gumbel-Softmax keeps a one-hot forward action
            # while allowing the imagined return to train categorical logits.
            discrete_action = F.gumbel_softmax(logits, tau=1.0, hard=True)
            act = torch.cat([cont_action, discrete_action], dim=-1)
        else:
            act = cont_action
        acts.append(act)

        h, mu, lv = model.dynamics.step(h, z, act)
        # Sampling trong tuong tuong: dung prior. Day la noi "nghi duoc" —
        # khong quan sat, chi co mo hinh.
        z = mu + torch.randn_like(mu) * _logvar_to_std(lv)

        # r_{t+1} — phan thuong nhan duoc sau khi hanh dong tai s_t.
        #
        # Giu gradient: chinh no la duong chay gradient cua actor.
        #
        # `vs` cung phai NAM NGOAI no_grad neu khong — targets da detach
        # roi, nen neu vs cung detach thi `mse(vs, targets)` co hai ve
        # deu khong gradient va critic khong bao gio duoc hoc gi.
        rts.append(reward_head(h, z).squeeze(-1))
        # CRITIC CHI HOC TRONG SO CUA NO, KHONG TAO HINH MO HINH THE GIOI.
        #
        # `vs` lay `hs[t]`, `zs[t]` — ma do thi do chua di qua encoder va
        # dynamics. Neu de nguyen, gradient cua critic chay nguoc qua do
        # thi, bien encoder/dynamics thanh hinh theo muc tieu lambda-return
        # thay vi theo tai tao. Latent xau di, reward head sup theo.
        vs.append(value_head(hs[t].detach(), zs[t].detach(),
                             goal).squeeze(-1))
        # Chuoi con hay da ket thuc — dung de cat chieu chiết khấu.
        # Do tren TRANG THAI TRUOC khi hanh dong, cung quy uoc voi
        # `continue_head(h_seq, z_seq)` trong trainer.
        cont.append(torch.sigmoid(
            continue_head(hs[t], zs[t]).squeeze(-1)).detach())

    hs = torch.stack(hs, 1)
    zs = torch.stack(zs, 1)
    rts = torch.cat([r.view(-1,1) for r in rts], dim=1) if len(rts) > 0 else torch.zeros(B, 1, device=h0.device)
    vs = torch.cat([v.view(-1,1) for v in vs], dim=1) if len(vs) > 0 else torch.zeros(B, 1, device=h0.device)
    acts_t = torch.stack(acts, 1)      # (B, T, A)
    cont = torch.cat([c.view(-1,1) for c in cont], dim=1) if len(cont) > 0 else torch.zeros(B, 1, device=h0.device)

    # --- Muc tieu: lambda-return ---
    #
    # Cong thuc chuan:
    #     V^λ_t = r_t + γ·cont_t·[ (1-λ)·V_{t+1} + λ·V^λ_{t+1} ]
    #
    # HAI loi da lam cho phan nay vo dung:
    #
    # 1. DAU CUA `cont`. `ReplayBuffer` ghi `continues = 0.0 if done else
    #    1.0`, va `continue_head` duoc huan dung chinh xac so do. Nen
    #    `cont = 1` nghia la chuoi CON tiep → phai bootstrap. Truoc day
    #    viet `gamma * (1.0 - cont)`, dao nguoc: chi khi chuoi da ket
    #    thuc moi bootstrap. Hieu ung: critic gan như KHONG bootstrap,
    #    muc tieu chi con phan thuong trong `horizon` buoc.
    #
    # 2. VI TRI CUA GIA TRI BOOTSTRAP. Phai dung `V_{t+1}` — chi so
    #    `t + 1`. Truoc day dung `vs[:, t]`, lech mot buoc.
    #
    # `v_next` giu V^λ_{t+1} khi chay nguoc.
    #
    # BOOTSTRAP O CUOI CHUOI bang V(s_H), KHONG phai 0. Buoc cuoi t
    # khong het hanh trinh — sau no con chuoi. Bootstrap bang 0 khi
    # nghia la "het tien", va muc tieu bi keo xuong.
    v_h = value_head(h, z, goal).squeeze(-1).detach()

    def lam_return(with_grad: bool):
        """Cung mot cong thuc, dung cho ca critic (detach) lan actor."""
        v_next = torch.zeros(B, device=h0.device, dtype=h0.dtype)
        seq = [None] * horizon
        for t in reversed(range(horizon)):
            boot = vs[:, t + 1].detach() if t + 1 < horizon else v_h
            v_t = rts[:, t] + gamma * cont[:, t] * (
                (1.0 - lam) * boot + lam * v_next)
            seq[t] = v_t if with_grad else v_t.detach()
            v_next = v_t
        return torch.stack(seq, 1)

    next_values = lam_return(with_grad=False)

    value_loss = F.mse_loss(vs, next_values)

    # --- Actor: TOI DA LAM CHAU LAM DOI ---
    #
    # Truoc day:  actor = -(normalize(rts - vs.detach()) * w).sum()
    #
    # Do la MOT MOT phan thuong tru MOT HANG SO. `vs` da `.detach()` nen
    # no khong con gi trong gradient — chat luong cua critic khong anh
    # huong gi toi policy. Actor chi nhin thay mot buoc, khong biet
    # hanh dong hom nay con danh giay ra gi sau 5 hay 10 buoc. Do la ly
    # do policy tro ve bien `tanh` thay vi hien o huu ich.
    #
    # Dung lai mot lan:
    #   - critic huan ve MUC TIEU = lambda-return, tach gradient
    #   - actor toi uu CHINH lambda-return do, KHONG tach
    #
    # `rts` giu gradient, nen gradient chay qua dynamics va reward head
    # nguoc ve hanh dong va ve mean cua policy. `boot` va `cont` da
    # detach — chi policy duoc cap nhat, mo hinh the gioi dong bang.
    v_lam = lam_return(with_grad=True)

    # CHUAN HOA DE GI DO LON PHAN THUONG KHONG DOI HUONG GRADIENT.
    # `v_lam` da la muc tieu chuan hoa roi (do reward head hoc tren
    # phan thuong chuan hoa), nen chuan hoa them mot lan thuong an
    # dinh — dung de so sanh giua cac lan chay.
    actor = -(v_lam * w).sum() / (w.sum() * B)

    # Entropy: giu muc kham pha cua mau trong tuong tuong.
    #
    # KHONG boc `no_grad` o day. Truoc day co, nen `ent_loss` khong co
    # gradient nao — tien pham bi tinh xong roi nem di. Phan thuong
    # entropy viet bang toan so ma khong bao gio tac dung len gi.
    # Detach `hs`, `zs`: entropy chi dung de giu muc kham pha cua POLICY,
    # khong lien quan gi den trang thai. De gradient no loai
    # encoder/dynamics thi khong dung.
    mean, logstd, logits = policy_head(
        hs.detach(), zs.detach(), goal.unsqueeze(1).expand(-1, horizon, -1))
    ent = policy_head.entropy(logstd, logits)
    ent_loss = -ent.mean()

    return {
        'actor': actor,
        'critic': value_loss,
        'entropy': ent_loss,
        'mean_abs': mean.detach().abs().mean(),
        'std_mean': _logstd_to_std(logstd).mean().detach(),
    }
