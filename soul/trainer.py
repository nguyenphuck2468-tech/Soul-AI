"""Tinh loss va backward. Day la noi chung minh day la mot mo hinh THAT
chut khong phai mot lop random co ten la mo hinh.

Co nam thanh phan loss, tat ca deu co gradient that vao cac tham so khac
nhau. Bo mot thanh phan di thi phan do KHONG CON DUOC HOC GI — va cac
mo hinh hay bi phe trong "chay duoc nhung khong hoc gi" thuong do chính
vi thieu mot trong cac thanh phan nay.

  1. recon      tai tao quan sat tu trang thai an. Day la dong co GOC
                cua viec hoc: ep latent phai chua du thong tin ve the gioi.
  2. KL         keo prior (khong co quan sat) ve gan posterior (co quan
                sat). Day la cai kien tao "du doan duoc tuong lai".
                Co `free_nats` de khong bi ep ve 0.
  3. reward     do phan thuong. Day la cai gan hanh dong vao chuoi qua
                khong: khong co no, moi hanh dong deu nhu nhau.
  4. continue   chuoi con hay het.
  5. policy+value  dung loi the (advantage) de day mua hieu hanh dong nao
                gay ket qua tot.
"""

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

try:
    from .imagine import imagine_actor_critic
except ImportError:  # pragma: no cover - chạy trực tiếp như script
    from imagine import imagine_actor_critic


def _log_prob_normal(x, mu, logvar):
    return -0.5 * (((x - mu) ** 2) / logvar.exp() + logvar)


def _kl_normal(mu_p, lv_p, mu_q, lv_q):
    """KL(q || p) theo tung chieu, tong lai."""
    return 0.5 * (lv_p - lv_q + (lv_q.exp() - lv_p.exp()) / lv_p.exp()
                  + ((mu_q - mu_p) ** 2) / lv_p.exp())


class Trainer:
    """Giu optimizer va tinh loss. Mot lan goi `update` = mot buoc hoc."""

    def __init__(self, model, cfg, obs_dim: int):
        self.model = model
        self.cfg = cfg
        self.tc = cfg.train
        self.device = torch.device(cfg.device)
        # HAI NHOM tham so, hai toc do hoc khac nhau.
        #
        # Tang bieu dien (encoder + dynamics) xay ra latent, dung chung cho
        # moi dau, nen no phai buoc chay on dinh — `lr` nho.
        #
        # Cac dau du doan chi DOC latent. Do lai chung bi thieu buoc: cung
        # latent, cung kien truc, batch 16, 3000 buoc, `lr = 3e-4` cho
        # R2 = -13.2% (te hon ca doan hang so), con `lr = 3e-3` cho
        # R2 = 69.3%. Chi lai so mau nhin thay ma khong — it buoc thi
        # te hon — nen day la thieu buoc toi uu chu khong phai thieu du
        # lieu. Xem `diag_sweep.py` va ghi chu `head_lr` trong config.py.
        head_mods = [model.decoder, model.reward_head, model.continue_head,
                     model.value_head]
        head_ids = {id(p) for mod in head_mods for p in mod.parameters()}
        pol_ids = {id(p) for p in model.policy_head.parameters()}
        rep = [p for p in model.parameters()
               if id(p) not in head_ids and id(p) not in pol_ids]
        head = [p for p in model.parameters() if id(p) in head_ids]
        pol = [p for p in model.parameters() if id(p) in pol_ids]
        self.optimizer = torch.optim.Adam(
            [{'params': rep, 'lr': self.tc.lr},
             {'params': head, 'lr': self.tc.head_lr},
             {'params': pol, 'lr': self.tc.policy_lr}],
            weight_decay=self.tc.weight_decay)
        self.obs_dim = obs_dim
        self.updates = 0
        # Thong ke chuan hoa phan thuong, dung cho reward head va cho
        # phan thuong trong tuong tuong (ca hai cung mot don vi).
        self._reward_stats = {'mean': 0.0, 'std': 1.0}

    # ---------------------------------------------------------------
    def _to_tensor(self, arr):
        return torch.as_tensor(arr, dtype=torch.float32, device=self.device)

    def update(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        tc = self.tc
        m = self.model
        seq = batch['values'].shape[1] - 1  # buoc cuoi la o_{t+1}

        values = self._to_tensor(batch['values'])
        masks = self._to_tensor(batch['masks'])
        texts = self._to_tensor(batch['texts'])
        actions = self._to_tensor(batch['actions'])
        rewards = self._to_tensor(batch['rewards'])
        continues = self._to_tensor(batch['continues'])
        goals = self._to_tensor(batch['goals'])
        valid = self._to_tensor(batch['valid'])

        # Ghep 3 phan cua vector hoa thanh mot dau vao cua encoder/decoder.
        obs_in = torch.cat([values, masks, texts], dim=-1)
        mask_weight = masks.mean(dim=(0, 1)).clamp(min=1.0 / max(1, masks.shape[-1]))
        # Trong so mot theo nhung o thuong xuyen co du lieu. O nao
        # hiem khi xuat hien thi co trong so nho -> it dung sai
        # khi dau do dung mo hinh "o nay luon 0".
        w = mask_weight

        # --- Lan chuoi qua dynamics (co prior va posterior) ---
        # "Slow posterior": `mix` leo tu 0 len `posterior_mix` theo so buoc
        # da hoc. Giu no o mot gia tri co dinh (0.05) thi 95% latent
        # chay vao GRU la prior, khong phai quan sat — xem ghi chu
        # `posterior_mix` trong config.py.
        want = tc.posterior_mix
        if tc.slow_posterior_updates > 0:
            ramp = min(1.0, self.updates / float(tc.slow_posterior_updates))
            m.dynamics.mix = want * ramp
        else:
            m.dynamics.mix = want

        out = m.dynamics.observe_sequence(obs_in[:, :seq], actions, m.encoder, valid)
        post_mu, post_lv = out['post_mu'], out['post_lv']
        prior_mu, prior_lv = out['prior_mu'], out['prior_lv']
        h_seq, z_seq = out['h_seq'], out['z_seq']

        # --- 1. TAI TAO ---
        # Goi decoder MOT LAN. Ba lan bao gio cho ra ba tensor khac nhau
        # neu model co bat ky phep nao co trang thai (BatchNorm, dropout
        # dang bat) — loss se khong khop nhau va con so bao cao se sua
        # doi giua cac lan goi.
        dec = m.decoder(h_seq, z_seq)
        dec_loss_num = F.mse_loss(dec['obs'], values[:, :seq], reduction='none')
        recon_num = (dec_loss_num * w).sum(-1).mean()
        # O co du lieu: dung BCE de tao mask (mot so trong so 1, phan con
        # lai la o trong). O trong co gia tri 0 nen dung "du doan 0".
        recon_mask = F.binary_cross_entropy_with_logits(
            dec['mask_logits'], masks[:, :seq], reduction='none')
        recon_mask = (recon_mask * w).sum(-1).mean()
        # Phan text la nhieu nhip 0/1 -> BCE cho dung vi tri.
        recon_text = F.binary_cross_entropy_with_logits(
            dec['text_logits'], texts[:, :seq], reduction='none').mean()
        recon = recon_num + 0.5 * recon_mask + 0.5 * recon_text

        # --- 2. KL: prior gan posterior ---
        # Free nats: cat moi phan KL duoi 1.0 nat duoc bo qua. Cach nay
        # cho prior "mien phi" di mot quang nho, nen no co the bo qua
        # chi tiet khong quan trong va tap trung vao cau truc lon.
        # THU TU DOI SO: phai tinh KL(post || prior), nghĩa la `p` la
        # PRIOR, `q` la POSTERIOR.
        #
        # `_kl_normal(mu_p, lv_p, mu_q, lv_q)` tra KL(q || p). Truoc day
        # goi `(post, prior)` tuc la tinh KL(prior || post) — nguoc.
        #
        # Vi sao huong la quan trong, khong phai chuyen chi tiet: so hang
        # entropy cua KL(q||p) phat `log(sigma_p/sigma_q)`. Khi p = prior,
        # no phat prior co phuong sai TO; khi p = post (nguoc), no phat
        # POSTERIOR co phuong sai to. Cai sau dung huong nguoc: model
        # bi day de nhieu latent nhieu hon, dung thay vi it lai.
        # STOP-GRADIENT tren POSTERIOR.
        #
        # Dreamer dung KL(sg(post) || prior). Khong co `sg`, gradient
        # chay nguoc ca hai ve: encoder duoc keo de "khop" voi prior
        # thay vi prior duoc day de tien doan dung. Hieu ung do la bien
        # dai bieu dien chay theo prior thay vi ngan tien.
        #
        # Day la chi so DUY NHAT cua prior — loss tai tao khong the
        # giup: decoder nhan `z_seq = encoder(obs[t])`, tuc no co the
        # chep thang quan sat va bo qua hoan toan `h` cua dynamics (do
        # duoc: xao h lam dau ra decoder doi ~13% do lon tin hieu).
        # KL BALANCING (DreamerV3) — thay cho `sg` thuan.
        #
        # Do duoc tren ban hien tai: `post_std` bị GHIM ĐÚNG Ở trần 0.6,
        # còn `post_mean_std` lẫn đều 1.3 -> 1.7 -> 2.7. Tức chặn cứng
        # đang có hiệu lực từng bước, va mo hinh dap lai bang cach phinh
        # thang latent len. No van chua hoc duoc gi.
        #
        # Vì vậy KHONG cắt hẳn gradient khỏi posterior. Thay vao do
        # chia hai nhanh:
        #     kl = w_prior  * KL(sg(post) || prior)   <- day prior
        #        + w_post   * KL(post      || sg(prior)) <- day encoder
        #
        # `sg(prior)` o nhanh thu hai là điểm then chot: nhờ nó, số hạng
        # entropy `log(sigma_prior/sigma_post)` ở đó không kéo được phương
        # sai posterior về 0. Nhánh đó buộc prior bám theo phân phối
        # thật, còn nhánh thứ nhất buộc prior đoán được từ quá khứ. Trọng
        # số 0.1 cho posterior là để encoder không bị ngợp vì nhiệm vụ chính
        # là làm prior đoán tốt.
        kl_prior = _kl_normal(prior_mu, prior_lv,
                              post_mu.detach(), post_lv.detach()).sum(-1)
        kl_post = _kl_normal(post_mu, post_lv,
                             prior_mu.detach(), prior_lv.detach()).sum(-1)
        kl_prior = torch.maximum(kl_prior, kl_prior.new_tensor(tc.free_nats))
        kl_post = torch.maximum(kl_post, kl_post.new_tensor(tc.free_nats))
        kl = (tc.kl_prior_weight * kl_prior.mean()
              + tc.kl_post_weight * kl_post.mean())

# --- 3. THUONG ---
        # CHUAN HOA PHAN THUONG.
        #
        # Comment o duoi da mo ta tinh nay tu dau nhung code CHUA BAO GIO
        # lam — no van la MSE thuan. Khong chuan hoa thi do lon muc tieu
        # phu thuoc vao thang do cua bai, va actor se duoi theo cai
        # thang do do.
        #
        # Do dac bi ngay o cho actor: no toi uu phan thuong TANG TUONG.
        # Neu phan thuong do mo hinh doan ra o trang thai ngoai phan phoi
        # bi noi lon gap nhieu (do duoc: std phan thuong tuong tuong 21.7
        # so voi 2.37 that), actor se dung chinh cai sai do de tang diem.
        # Chuan hoa giu phan thuong tren toan cuc o don vi "do lech chuan"
        # nen do lon cua loi sai cung don vi, va uu tien khong con phu
        # thuoc vao thang do.
        rew_stats = self._reward_stats
        rewards_n = (rewards - rew_stats['mean']) / max(rew_stats['std'], 1e-6)
        pred_rew = m.reward_head(h_seq, z_seq)
        rew_pred_loss = F.mse_loss(pred_rew, rewards_n)
        # Cap nhat thong ke tren CHINH phan thuong thuc, khong phai ban
        # chuan hoa — nen khong feedback vo hinh.
        # `.item()` de giu lai float thuan: tensor dua vao checkpoint se
        # keo theo thiet bi, va lam checkpoint nang khong can.
        with torch.no_grad():
            rew_stats['mean'] += 0.01 * (rewards.mean().item()
                                         - rew_stats['mean'])
            rew_stats['std'] += 0.01 * (rewards.std().item()
                                        - rew_stats['std'])

        # --- 4. CONTINUE ---
        pred_cont = m.continue_head(h_seq, z_seq)
        cont_loss = F.binary_cross_entropy_with_logits(pred_cont, continues)

        # --- 5. POLICY + VALUE (trong tuong tuong) ---
        B = h_seq.shape[0]
        goal_t = goals[:, :seq]
        # --- POLICY: HOC TRONG TUONG TUONG ---
        #
        # KHONG dung REINFORCE tren du lieu replay nua. Ly do do bang SO DO,
        # khong phai doan:
        #
        #   REINFORCE uoc luong gradient bang MOT mau tai moi buoc: tang
        #   xac suat cua duy nhat hanh dong da chon. Khi hanh dong duoc
        #   chon NGAY NHIEN (off-policy), bien bieu do tro thanh "nen chon
        #   hanh dong nao da duoc chon nhieu hon" — tan hoan toan, mat het
        #   thong tin ve trang thai.
        #
        #   Do do: advantage co tin hieu DUNG (tuong quan +0.99 voi ket qua
        #   that), nhung sau 40 episode hanh dong van giong het nhau o moi
        #   trang thai — do lech 0.00000. Policy hoi tu uoc luong, khong
        #   phai loi lap trinh. Moi thu van in loss binh thuong.
        #
        # Giai phap: mo mo hinh the gioi ra phia truoc, tu sinh hanh dong
        # do CHINH POLICY phat ra, roi hoc bang muc tieu da khur baseline.
        # Moi hanh dong deu duoc cham diem, khong can "vai mau" nao.
        # Bat dau tuong tuong o buoc CUOI chuoi that.
        #
        # Da thu bat dau o buoc ngau nhien va do ra ket qua TE HON:
        # `std phan thuong tuong tuong` tang tu 3.28 len 8.77, va policy
        # lai sap ve hang so. Ly do: buoc giua chuoi thi trang thai co
        # nhieu phan thuong lon hon (chuoi that dang o giua con lac), nen
        # mo hinh bi day sai nhieu hon khi do lai phan thuong. Giu buoc
        # cuoi cho mo hinh do tren vung no da hoc.
        img = imagine_actor_critic(
            m, h_seq[:, -1], z_seq[:, -1], goal_t[:, -1],
            m.reward_head, m.continue_head, m.value_head, m.policy_head,
            tc.gamma, getattr(tc, 'lam', 0.95), tc.imagine_horizon)

        policy_loss = img['actor']
        value_loss = img['critic']
        ent_loss = img['entropy']
        ent_w = tc.entropy_weight

        # --- Gradient actor CHI cham vao policy head ---
        #
        # Gradient actor chay qua cuon roll thi di qua dynamics va reward
        # head. De no loang, hai mo hinh do "gia": phan thuong do loi se
        # thoi len khong kiem soa. Do duoc khi de no loang that:
        #     std phan thuong tuong tuong = 1551.7  (thuc chi 2.26)
        #     std muc tieu            = 18116.0
        #
        # Tuc lai: de gradient loang, PHAI tach no ra rieng, cong vao
        # `.grad` cua policy sau khi back-propagate phan con lai.
        # Gradient actor CHI cham vao policy head.
        #
        # `torch.autograd.grad` chi tinh gradient doi voi policy, nen
        # dynamics va reward head KHONG bi cap nhat boi loss nay.
        #
        # Da thu de let gradient chay qua (de mo hinh "tu sua loi o noi
        # policy dam me") — ket qua do duoc:
        #     tach rieng : std phan thuong tuong tuong   9.98 (thuc 2.32)
        #     cho qua   : std phan thuong tuong tuong 581.7  (thuc 2.45)
        # Nhanh khac no lam `corr(x, a)` DUNG dau (-0.77 truoc do +0.91),
        # nhung no lam mo hinh no 237 lan, va phan thuong do loi roi thi
        # chinh no la cai policy toi uu. KHONG phai huong nen thu.
        policy_params = list(m.policy_head.parameters())
        actor_grads = torch.autograd.grad(
            policy_loss, policy_params, retain_graph=True, allow_unused=True)

        total = (tc.w_recon * recon + tc.w_kl * kl + tc.w_reward * rew_pred_loss
                 + tc.w_continue * cont_loss
                 + tc.w_value * value_loss + ent_w * ent_loss)

        self.optimizer.zero_grad(set_to_none=True)
        total.backward()
        # Ghep gradient actor vao policy. `retain_graph=True` o tren giữ
        # do thi lai cho `total.backward()` lay.
        for p, g in zip(policy_params, actor_grads):
            if g is None:
                continue
            g = g * tc.w_policy
            p.grad = g if p.grad is None else p.grad + g
        # Cat gradient: mot mau rac thuong duoc du day len se lam trong so
        # bay nhay, mat het nhung gi da hoc.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.tc.grad_clip)
        self.optimizer.step()
        self.updates += 1

        return {
            # Khoá goi la 'total' chu khong phai 'loss': run_episode tiền tố
            # them 'loss_' vao moi khoá de phan biet loss voi phan thuong
            # that. Neu de ten 'loss' thi ra 'loss_loss' — doc nhu mot loi.
            'total': float(total.item()),
            'recon': float(recon.item()),
            'recon_num': float(recon_num.item()),
            'kl': float(kl.item()),
            'reward': float(rew_pred_loss.item()),
            'continue': float(cont_loss.item()),
            'policy': float(policy_loss.item()),
            'value': float(value_loss.item()),
            'grad_norm': float(grad_norm.item()),
        }