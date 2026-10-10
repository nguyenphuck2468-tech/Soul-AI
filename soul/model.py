"""Mô hình thế giới RSSM — phần "trí tuệ" thật sự của linh hồn.

Cấu trúc lấy từ họ Dreamer (Hafner) và mạng state-space (RSSM):

    encoder     o_t             -> q(z_t | o_t)        [chi khi HOC]
    dynamics    (h_{t-1}, z_{t-1}, a_{t-1}) -> (h_t, p(z_t|h_t))
                                                  [luon co, de TUONG TUONG]
    decoder     (h_t, z_t)     -> o_t                 [tai tao]
    reward      (h_t, z_t)     -> r_t
    continue    (h_t, z_t)     -> c_t  (con hay het)
    policy      (h_t, z_t, g) -> a_t
    value       (h_t, z_t, g) -> V_t

Diem mau chot, va cung la thu dong nhat nen doc:

  * HAI nguon tach biet: `prior` (khong co quan sat) va `posterior` (co quan
    sat that). Muc tieu la lam prior GAN posterior. Do lech giua hai ben
    chinh la KL loss — tuc "mo hinh du doan sai bi bao nhieu so voi cai
    no vua thay".

  * Khi SUY LUAN (luc chay), khong duoc dung encoder/posterior — chi dung
    dynamics. Tuc la: sau khi huan luyen xong, mo hinh du doan duoc the
    gioi ma KHONG CAN MOT MAU QUAN SAT NAO. Do do ma no "nghĩ duoc".

  * `free_nats`: KL duoc bo qua mot nguong nho. Neu khong co, KL se thang
    va prior bi keo ve 0 — luc do mo hinh thoi gian bien thanh may ghi am
    chu khong du doan gi ca. Gia tri 1.0 nat la moc re hon.

Mot luu y nua rieng mo hinh nay: khong dung Transformer. Co, chuoi thoi
gian khong dai va khong dung nganh, GRU hop hon ca ve may tinh va ve do
chinh xac. Transformer se thang o chuoi rat dai (chit, van ban, gia theo
gio), con bay gio khong can.
"""

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(sizes, act=nn.SiLU, out_act=False):
    """Dung MLP nhieu tang. SiLU vi dich bia thuong xuyen co gia tri am
    (hoi quy ve bien), ma SiLU o am van khong bi chet nhu ReLU."""
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_act:
            layers.append(act())
    return nn.Sequential(*layers)


# HAI phep chuyen doi khac nhau, truoc day bi dung chung mot ham — do la
# nguon goc cua loi im lang nguy hiem nhat trong mo hinh nay.
#
#   log-PHUONG SAI  (logvar): encoder va dynamics xuat ra so nay.
#       std = exp(0.5 * logvar)
#   log-LECH CHUAN  (logstd): policy xuat ra so nay.
#       std = exp(logstd)
#
# Dung nham phuong sai cho policy se lam mau sai he so 2, va `log_prob` se
# tru -logstd thay vi -0.5*logstd: no khong con la mat do xac suat cua
# phan bo ma `sample` thi ra. REINFORCE toi uu sai muc tieu, loss van
# giam, moi thu van chay binh thuong — va policy hoc sai.


def _logvar_to_std(logvar):
    """log-phuong sai -> lech chuan (dung cho encoder/dynamics)."""
    return (0.5 * logvar.clamp(-8.0, 4.0)).exp()


def _logstd_to_std(logstd):
    """log-lech chuan -> lech chuan (dung cho policy)."""
    return logstd.clamp(-8.0, 2.0).exp()


# Tran cho `mean` cua policy. Hanh dong thi di cat ve [-1, 1], nen mean
# di xa qua moc nay la tham so hoa danh. Dat 3.0: rong cho moi muc tieu
# hien thuc ma van ben trong vung lay mau.
MAX_MEAN = 3.0

# Tran cho do lech chuan. REINFORCE co xa huong lam `std` -> 0 (de tap
# trung xac suat len hanh dong da chon), va neu khong chan thi std sup
# xuong 0.001, hanh dong het ngau nhien. Dat 0.6: van kham pha du, nhung
# khong chet.
MIN_STD = 0.6

# Chan DO RONG cua latent.
#
# Do duoc tren mo hinh da huanc:
#     std cua posterior MEAN (tin hieu) = 0.328
#     std cua posterior      (nhieu)   = 1.086   <- gap 3.3 lan
#
# Nen latent lay mau la ~77% nhieu theo phuong sai. Tin hieu bi chon
# ngap trong nhieu.
#
# NGHICH LY DO MA DO DUNG: KL(post||prior) co so hang
# `((mu_q - mu_p)^2) / (2 * var_p)`. Mo rong `var_p` LAM NHIEU so hang
# do giam — nen mo hinh co the giam KL bao cach LAM RONG hai phan phoi
# thay vi CAN CHO mean khop nhau. Do do co that:
#     KL chi 0.43 nat, nhung prior lech posterior 36.7% do lon.
# KL nho DO KHONG chung to mo hinh dung — no vua say.
#
# `std <= 0.6` cho phep phan phoi hep hon nhieu lan tinh hieu, nen mot
# buoc GRU phai thuc su giu thong tin thay vi dua het so do vao do le.
LOGVAR_LO = -8.0
# Tran `std` cua latent. Gia tri nay la LUOI AN TOAN, khong phai co che
# can bang — xem `latent_std_max` trong config.py.
#
# Do duoc voi tran 0.6: `post_std` ghim DUNG o 0.6 con `post_mean_std` lan
# deu 1.3 -> 1.7 -> 2.7. Mot chan luon cham bien nghia la no dang ep mo
# hinh chu khong phai ho tro no; mo hinh dap lai bang cach phinh latent.
LOGVAR_HI = float(2.0 * math.log(1.5))   # std <= 1.5; ghi de qua cfg
LOGVAR_HI_DEFAULT = LOGVAR_HI


class Encoder(nn.Module):
    """Quan sat -> phan bo an. CHI dung khi huan luyen."""

    def __init__(self, in_dim: int, hidden: int, latent: int,
                 logvar_hi: float = None):
        super().__init__()
        self.logvar_hi = LOGVAR_HI_DEFAULT if logvar_hi is None else logvar_hi
        self.net = mlp([in_dim, hidden, hidden])
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        return self.to_mu(h), self.to_logvar(h).clamp(LOGVAR_LO, self.logvar_hi)


class Dynamics(nn.Module):
    """Doan trang thai tiep theo KHONG can quan sat that.

    Dung GRU (mot bien) chu khong dung RNN: don gian hon, nhanh hon, va
    du chuoi ngan cua bai toan nay thi khac biet rat nho.
    """

    def __init__(self, hidden: int, latent: int, action_dim: int,
                 action_embed: int = None, logvar_hi: float = None):
        super().__init__()
        self.latent = latent
        self.hidden = hidden
        self.logvar_hi = LOGVAR_HI_DEFAULT if logvar_hi is None else logvar_hi
        # HANH DONG CO ĐƯỜNG RIÊNG, KHÔNG CHIA SẺ VỚI LATENT.
        #
        # Truoc day: `gru(torch.cat([z, a]))` voi `a` la MOT SO tho.
        # Do do hanh dong chiem 1/(latent+1) be rong dau vao, va la so
        # chua qua chieu nao nen no bi chiem. Do duoc tren mo hinh da
        # huanc:
        #     KL cua prior co hanh dong that   : 0.4227
        #     KL cua hanh dong xao tron        : 0.4377   (chenh 3.5%)
        # va do do chinh xac mo k buoc: ty le 10-14 lan "doan hang so".
        #
        # Tham kho: DreamerV4-MC (`Dreamerv4-MC-main`) khong ghep thang
        # ma chieu hanh dong qua MLP 2 tang ra DUNG be rong mo hinh
        # (2048), roi cho no nhung o token rieng trong chuoi — moi tang
        # attention deu thay truc tiep. Nguyen ly dung, ap dung o quy
        # mo cua RSSM: cho hanh dong be rong TUONG NHO hoac LON HON
        # latent, thay vi mot so chen vao.
        ae = action_embed if action_embed is not None else max(16, latent)
        self.action_embed = ae
        self.action_in = nn.Sequential(
            nn.Linear(action_dim, ae), nn.SiLU(), nn.Linear(ae, ae))
        self.gru = nn.GRUCell(latent + ae, hidden)
        self.net = mlp([hidden, hidden, hidden])
        self.to_mu = nn.Linear(hidden, latent)
        self.to_logvar = nn.Linear(hidden, latent)

    def step(self, h, z, a):
        h = self.gru(torch.cat([z, self.action_in(a)], dim=-1), h)
        x = self.net(h)
        return h, self.to_mu(x), self.to_logvar(x).clamp(LOGVAR_LO, self.logvar_hi)

    def initial_state(self, batch: int, device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.zeros(batch, self.hidden, device=device),
                torch.zeros(batch, self.latent, device=device))

    def observe_sequence(self, obs: torch.Tensor, actions: torch.Tensor,
                         encoder: Encoder, valid: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Lan mot chuoi quan sat, dung prior VA posterior.

        obs, actions: (B, T, D).  valid: (B, T) — 1 neu buoc do thuoc
        ve mot ep hien tai, 0 neu la buoc "keo" qua ranh gioi cua mot ep
        khac (chi xay ra o dau chuoi).

        Tra ve cac chuoi (B, T, ...) de trainer tinh loss.
        """
        B, T = obs.shape[0], obs.shape[1]
        h, z = self.initial_state(B, obs.device)
        # `mix` la ti le tin posterior o moi buoc. Nho -> on dinh; lon ->
        # hoc nhanh nhung de bay. 0.05 la diem can bang da do duoc.
        mix = self.mix if hasattr(self, 'mix') else 0.05

        post_mu, post_lv, prior_mu, prior_lv = [], [], [], []
        h_seq, z_seq = [], []

        for t in range(T):
            q_mu, q_lv = encoder(obs[:, t])
            # Hanh dong o buoc t LA hanh dong vua sinh ra trang thai t.
            #
            # `Soul.observe` nhan quan sat SAU khi thuc hien `action`, va
            # ghi `values[s]` cung chi so voi `actions[s]`. Nghia la
            # `obs[t]` do `actions[t]` sinh ra — phai dung `actions[:, t]`.
            #
            # Truoc day dung `actions[:, t - 1]`, lech MOT BUOC: prior
            # duoc hoi "trang thai gi se ra neu tinh tu hien tai bang
            # hanh dong truoc do", trong khi quan sat o chinh la thu do
            # hanh dong do. Prior viet sai bai, nen gradient cua hanh
            # dong khong bao gio tro gan dung mot buoc nao.
            #
            # Buoc 0 khong co hanh dong nao truoc do -> dung 0.
            a_t = torch.zeros_like(actions[:, 0]) if t == 0 else actions[:, t]
            h, p_mu, p_lv = self.step(h, z, a_t)

            # Neu day la buoc khong hop le (nam ngay sau ranh gioi ep), thi
            # KHONG duoc dung posterior: quan sat do thuoc ve mot ep khac,
            # dung no se lam h ban nho sang chuyen khac. Thay bang prior.
            v = valid[:, t].unsqueeze(-1)
            if mix >= 1.0:
                z = q_mu + torch.randn_like(q_mu) * _logvar_to_std(q_lv)
            elif mix <= 0.0:
                z = p_mu + torch.randn_like(p_mu) * _logvar_to_std(p_lv)
            else:
                z_post = q_mu + torch.randn_like(q_mu) * _logvar_to_std(q_lv)
                z_prior = p_mu + torch.randn_like(p_mu) * _logvar_to_std(p_lv)
                z = mix * z_post + (1.0 - mix) * z_prior

            post_mu.append(q_mu * v + p_mu * (1 - v))
            post_lv.append(q_lv * v + p_lv * (1 - v))
            prior_mu.append(p_mu)
            prior_lv.append(p_lv)
            h_seq.append(h)
            z_seq.append(post_mu[-1])

        return {
            'post_mu': torch.stack(post_mu, 1),
            'post_lv': torch.stack(post_lv, 1),
            'prior_mu': torch.stack(prior_mu, 1),
            'prior_lv': torch.stack(prior_lv, 1),
            'h_seq': torch.stack(h_seq, 1),
            'z_seq': torch.stack(z_seq, 1),
        }

    def imagine_sequence(self, h, z, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Mo rong chuoi hanh dong trong TUONG TUONG, khong can quan sat.

        actions: (B, T, A) -> tra ve (B, T, hidden/latent). Day la chinh
        la noi "nghi duoc": mui hinh dong chua tung xay ra, nhung mo hinh
        van chay duoc va do duoc.
        """
        B, T = actions.shape[0], actions.shape[1]
        hs, zs, mus, lvs = [], [], [], []
        for t in range(T):
            h, mu, lv = self.step(h, z, actions[:, t])
            z = mu + torch.randn_like(mu) * _logvar_to_std(lv)
            hs.append(h); zs.append(z); mus.append(mu); lvs.append(lv)
        return {
            'h': torch.stack(hs, 1),
            'z': torch.stack(zs, 1),
            'mu': torch.stack(mus, 1),
            'logvar': torch.stack(lvs, 1),
        }


class Decoder(nn.Module):
    """Trang thai an -> quan sat.

    Day la thu ep mo hinh phai GIU thong tin ve the gioi. Bo no di thi
    latent chi con la mot con so ngau nhien, va khong gi du doan duoc gi.
    """

    def __init__(self, hidden: int, latent: int, obs_dim: int, mask_dim: int, text_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.mask_dim = mask_dim
        self.text_dim = text_dim
        self.net = mlp([hidden + latent, hidden, hidden, obs_dim + mask_dim + text_dim])

    def forward(self, h, z):
        out = self.net(torch.cat([h, z], dim=-1))
        o, m, t = torch.split(out, [self.obs_dim, self.mask_dim, self.text_dim], dim=-1)
        return {'obs': o, 'mask_logits': m, 'text_logits': t}


class RewardHead(nn.Module):
    def __init__(self, hidden, latent):
        super().__init__()
        self.net = mlp([hidden + latent, hidden, 1])

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class ContinueHead(nn.Module):
    """Chuoi con hay da ket thuc. Cho phep mo hinh hoc ranh gioi giua
    cac su kien, nen biet duoc phai nho lau bao lau."""

    def __init__(self, hidden, latent):
        super().__init__()
        self.net = mlp([hidden + latent, hidden, 1])

    def forward(self, h, z):
        return self.net(torch.cat([h, z], dim=-1)).squeeze(-1)


class ActionEffectHead(nn.Module):
    """Du doan hieu ung cua mot hanh dong toi trang thai tiep theo.

    Doi hoi no PHAI suy đoan kieu "neu toi lam X o trang thai hien tai,
    latent se thay doi the nao". Day la phan bo sung chat cheo cho
    mo hinh the gioi: bo khong co no thi AI chi biet "sua doi` reward`
    ma khong biet vi sao the gioi chuyen dong.
    """

    def __init__(self, hidden, latent, action_dim):
        super().__init__()
        self.net = mlp([hidden + latent + action_dim, hidden, latent])

    def forward(self, h, z, action):
        x = torch.cat([h, z, action], dim=-1)
        return self.net(x)


class PolicyHead(nn.Module):
    """Trang thai + muc tieu -> phan bo hanh dong.

    Hanh dong tach thanh hai phan: lien tuc (mean, log_std) va roi rac
    (logits one-hot). Tach vi hoi quy MSE tren one-hot cho ra 0.35/0.35 —
    hanh dong o giua chung, vo nghia.
    """

    def __init__(self, hidden, latent, action_dim, goal_dim, discrete_dim):
        super().__init__()
        self.action_dim = action_dim
        self.discrete_dim = discrete_dim
        self.cont_dim = action_dim - discrete_dim
        d_in = hidden + latent + goal_dim
        self.to_mean = mlp([d_in, hidden, self.cont_dim])
        self.to_logstd = nn.Linear(d_in, self.cont_dim)
        self.to_logits = mlp([d_in, hidden, discrete_dim]) if discrete_dim else None

    def forward(self, h, z, goal):
        x = torch.cat([h, z, goal], dim=-1)
        # `tanh` chon vi no nen mean vao khoang huu han, khong bao gio
        # duoc thep vao vuc bien — clamp cung duoc nhung tao be sat tai
        # bien, o do gradient bang 0 va policy khong con hiu chinh duoc
        # phia ben kia. Tai MAX_MEAN (3.0) gradient cua tanh van con
        # ~0.99.
        mean = MAX_MEAN * torch.tanh(self.to_mean(x) / MAX_MEAN)
        # Khoa logstd trong [-8, 2]. Be qua: policy dong bang, kham pha
        # het. Lon qua: buoc dau tinh nhieu hon ca nhieu.
        logstd = self.to_logstd(x).clamp(-8.0, 2.0)
        logits = self.to_logits(x) if self.to_logits is not None else None
        return mean, logstd, logits

    def sample(self, mean, logstd, logits, deterministic=False):
        if self.cont_dim:
            # Chan std o san: REINFORCE luon keo `logstd` xuong de tang
            # xac suat cho hanh dong da chon. De khong chan, std sup xuong
            # ~0.001 va hanh dong het ngau nhien — model khong con kham
            # pha gi duoc nua. `logstd` o day la log-lech-chuan.
            logstd = torch.maximum(logstd, torch.as_tensor(
                math.log(MIN_STD), device=logstd.device, dtype=logstd.dtype))
        if deterministic:
            cont = mean
        else:
            cont = mean + torch.randn_like(mean) * _logstd_to_std(logstd)
        if self.discrete_dim:
            idx = (logits.argmax(-1) if deterministic
                   else torch.distributions.Categorical(logits=logits).sample())
            return torch.cat([cont, F.one_hot(idx, self.discrete_dim).float()], -1)
        # KHONG clamp phan lien tuc.
        #
        # `clamp` co gradient bang 0 ngoai khoang, nen khi mean tro ra
        # ngoai [-1, 1] thi moi hanh dong mau trong tuong tuong deu bi
        # kep ve mot gia tri — quy dao do khong con thay doi gi theo
        # policy, advantage khong mang thong tin hanh dong, va policy
        # bi khep vinh vien. Day la cai bang toi: policy `stopped
        # learning` va in loss binh thuong.
        #
        # `tanh` o `forward` da lo phan gioi han: no nen mean vao
        # [-MAX_MEAN, MAX_MEAN] va gradient van con o ca hai bien
        # (do bang o giua bien con ~0.99).
        return cont

    def log_prob(self, mean, logstd, logits, action):
        cont = action[..., :self.cont_dim]
        parts = []
        if self.cont_dim:
            # Phai dung DUNG std ma `sample` dung. Neu `sample` chan san
            # std o MIN_STD ma `log_prob` khong chan, hai ben se noi ve hai
            # phan bo khac nhau: REINFORCE se uoc luong gradient cua mot
            # phan bo ma policy khong bao gio lay mau tu.
            ls = torch.maximum(logstd, torch.full_like(
                logstd, math.log(MIN_STD)))
            std = _logstd_to_std(ls)
            # Tong over cac chieu lien tuc de ra mot so cho moi hanh dong.
            parts.append(
                (-0.5 * ((cont - mean) / std).pow(2)
                 - ls - 0.5 * math.log(2 * math.pi)).sum(-1)
            )
        if self.discrete_dim:
            tgt = action[..., self.cont_dim:].argmax(-1)
            parts.append(F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1))
        return torch.stack(parts, -1).sum(-1)

    def entropy(self, logstd, logits):
        # Entropy cua Gauss 1D = 0.5*log(2*pi*e*std^2) = const + logstd.
        # Chi gradient phu thuoc `logstd` nen thieu const khong anh huong
        # viec hoc — nhung giu cho gia tri la entropy that.
        const = 0.5 * math.log(2 * math.pi * math.e)
        # Entropy phai phan anh std THUC SU lay mau, da bi chan san. Neu
        # tinh tren logstd chua chan, gradient entropy se keo std xuong
        # duoi MIN_STD ma khong bao gio duoc dung.
        ls = torch.maximum(logstd, torch.full_like(
            logstd, math.log(MIN_STD)))
        ent = (ls + const).sum(-1)
        if self.discrete_dim:
            ent = ent + (-(F.log_softmax(logits, -1) * F.softmax(logits, -1)).sum(-1))
        return ent


class ValueHead(nn.Module):
    """Gia tri trang thai — co so de tinh loi the (advantage)."""

    def __init__(self, hidden, latent, goal_dim):
        super().__init__()
        self.net = mlp([hidden + latent + goal_dim, hidden, 1])

    def forward(self, h, z, goal):
        return self.net(torch.cat([h, z, goal], dim=-1)).squeeze(-1)


class SoulModel(nn.Module):
    """Toan bo mo hinh, gon trong mot nn.Module de save/load de rang."""

    def __init__(self, cfg, in_dim: int):
        super().__init__()
        self.cfg = cfg
        self.in_dim = in_dim
        m = cfg.model
        self.hidden = m.hidden_dim
        self.latent = m.latent_dim
        self.goal_dim = m.goal_dim if m.use_goal else 0

        # Tran std latent do `config.latent_std_max` quyet dinh, khong
        # phai hard-code trong module — de con quet duoc bien nay.
        import math as _math
        _lv_hi = float(2.0 * _math.log(max(cfg.train.latent_std_max, 1e-3)))
        self.encoder = Encoder(in_dim, m.hidden_dim, m.latent_dim, _lv_hi)
        self.dynamics = Dynamics(m.hidden_dim, m.latent_dim, m.action_dim,
                                logvar_hi=_lv_hi)
        self.dynamics.mix = cfg.train.posterior_mix
        self.decoder = Decoder(m.hidden_dim, m.latent_dim,
                               cfg.obs.max_features, cfg.obs.max_features,
                               cfg.obs.text_buckets)
        self.reward_head = RewardHead(m.hidden_dim, m.latent_dim)
        self.continue_head = ContinueHead(m.hidden_dim, m.latent_dim)
        self.action_effect_head = ActionEffectHead(m.hidden_dim, m.latent_dim,
                                                 m.action_dim)
        self.policy_head = PolicyHead(m.hidden_dim, m.latent_dim, m.action_dim,
                                      self.goal_dim, m.discrete_dim)
        self.value_head = ValueHead(m.hidden_dim, m.latent_dim, self.goal_dim)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())