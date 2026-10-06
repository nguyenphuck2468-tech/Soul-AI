"""Lớp cao cấp: quan sát -> hành động -> học -> lưu/nạp.

Đây là interface duy nhất bạn cần dùng. Mọi chi tiết bên trong (mô hình,
optimizer, replay, encoder) đều được gói gọn. Khi cần gắn vào xác nào
(Minecraft, robot, giao diện web...), chỉ việc gọi:
    s = Soul(cfg)
    action = s.act(observation)
    s.observe(observation, action, reward, done, goal)

Checkpoint lưu đủ để phục hồi nguyên trạng: trọng số, schema encoder,
replay, stats. Nạp lại là tiếp tục ngay từ bước cũ.
"""

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from .config import Config
from .obs import ObsEncoder
from .model import SoulModel
from .memory import Episode, ReplayBuffer
from .trainer import Trainer


class Soul:
    """Một 'linh hồn' độc lập — chỉ biết quan sát, hành động, học."""

    def __init__(self, cfg: Optional[Config] = None, checkpoint: Optional[str] = None):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(self.cfg.train.seed)
        torch.manual_seed(self.cfg.train.seed)
        if self.cfg.device == 'cuda' and not torch.cuda.is_available():
            print('[Soul] CUDA khong co, chuyen sang CPU')
            self.cfg.device = 'cpu'
        self.device = torch.device(self.cfg.device)

        # Encoder quan sát.
        self.enc = ObsEncoder(self.cfg.obs)

        # Mô hình.
        self.model = SoulModel(self.cfg, self.enc.input_dim).to(self.device)
        self.trainer = Trainer(self.model, self.cfg, self.enc.input_dim)

        # Replay.
        self.replay = ReplayBuffer(self.cfg.train.replay_capacity, self.cfg.train.seq_len)
        self._cur_ep = Episode()

        # Thống kê.
        self.steps = 0
        self.updates = 0
        self.ep_reward = 0.0
        self.ep_len = 0

        # Thống kê chuẩn hoá quan sát (xem `dataset.py`).
        #
        # KHÔNG được khởi tạo ở chỗ khác rồi quên lưu: suy luận phải dùng
        # đúng thang của lúc huấn, lệch một hằng số là mọi dự đoán sai.
        self.obs_norm: Dict[str, Dict[str, float]] = {}

        # Trạng thái suy luận.
        self._h = None
        self._z = None
        self._prev_action = None
        self._goal = None

        if checkpoint:
            self.load(checkpoint)

    # -----------------------------------------------------------------
    def set_goal(self, goal: np.ndarray):
        """Đặt mục tiêu cho chuỗi hành động tới. Vector mục tiêu có thể
        là one-hot, hoặc bất kỳ embedding nào bạn cung cấp. Mô hình chỉ
        quan tâm rằng nó có cùng `goal_dim` đã khai báo trong config."""
        self._goal = np.asarray(goal, dtype=np.float32)
        if self._goal.ndim == 1:
            self._goal = self._goal.reshape(1, -1)

    def act(self, observation: Any, deterministic: bool = False) -> np.ndarray:
        """Dự đoán hành động từ quan sát hiện tại.

        Lần đầu gọi (khi `self._h` là None) sẽ khởi tạo trạng thái mới.
        `deterministic=True` dùng khi muốn hành động chắc chắn (không
        khám phá) — ví dụ lúc đánh giá.
        """
        # Vector hóa quan sát.
        enc = self.enc.encode(observation)
        obs_vec = np.concatenate([enc['values'], enc['mask'], enc['text']], axis=0)

        # Khởi tạo trạng thái nếu lần đầu.
        if self._h is None:
            self._h = torch.zeros(1, self.model.hidden, device=self.device)
            self._z = torch.zeros(1, self.model.latent, device=self.device)
            self._prev_action = np.zeros(self.cfg.model.action_dim, dtype=np.float32)
            if self._goal is None:
                self._goal = np.zeros((1, self.model.goal_dim), dtype=np.float32)

        # Tự chọn mục tiêu nếu chưa set.
        goal = self._goal if self._goal is not None else np.zeros((1, self.model.goal_dim), np.float32)

        with torch.no_grad():
            h = torch.as_tensor(self._h, device=self.device)
            z = torch.as_tensor(self._z, device=self.device)
            a = torch.as_tensor(self._prev_action, device=self.device).unsqueeze(0)
            g = torch.as_tensor(goal, device=self.device)
            o = torch.as_tensor(obs_vec, device=self.device).unsqueeze(0)

            # Lan dong luc: h_t = GRU(h_{t-1}, [z_{t-1}, a_{t-1}]).
            h, mu, _ = self.model.dynamics.step(h, z, a)

            # Dung QUAN SAT hien tai de gan latent. Day la mau co dac:
            # khi chay, ta co quan sat thi — nen phai dung no. Bo qua
            # encoder o day (chi chay prior) thi policy khong "thay" gi ca:
            # no chi doan mo, khong phan biet duoc hai the gioi khac nhau.
            #
            # Phan biet voi tuong tuong: tuong tuong la doan cac buoc
            # TUONG LAI khi khong con quan sat nao. Con buoc hien tai ta
            # co quan sat that, nen dung posterior. Day cung la cach Dreamer
            # lam — prior dung cho tuong tuong, posterior cho buoc dang co
            # mat.
            q_mu, _ = self.model.encoder(o)
            z = q_mu

            # Dự đoán hành động từ (h, z, goal).
            mean, logstd, logits = self.model.policy_head(h, z, g)
            action = self.model.policy_head.sample(
                mean, logstd, logits, deterministic=deterministic)
            action = action.squeeze(0).cpu().numpy()

        self._h = h
        self._z = z
        self._prev_action = action
        return action

    # -----------------------------------------------------------------
    def observe(self, observation: Any, action: np.ndarray,
                reward: float, done: bool, goal: Optional[np.ndarray] = None):
        """Ghi nhận kết quả của hành động vừa thực hiện.

        `observation` là quan sát SAU khi đã thực hiện `action`.
        `reward` là phần thưởng nhận được TẠI BƯỚC NÀY (sau hành động).
        `done` = True nếu chuỗi kết thúc (thắng, thua, reset...).
        `goal` (tùy chọn): ghi đè mục tiêu cho bước này.
        """
        # Vector hóa quan sát mới.
        enc = self.enc.encode(observation)
        g = np.asarray(goal if goal is not None else
                       (self._goal if self._goal is not None
                        else np.zeros(self.model.goal_dim, np.float32)),
                       dtype=np.float32).reshape(-1)
        self._cur_ep.append(enc, action, reward,
                           0.0 if done else 1.0,
                           g)
        self.ep_reward += reward
        self.ep_len += 1
        self.steps += 1

        if done:
            # Episode kết thúc -> đẩy vào replay, reset trạng thái.
            self.replay.add_episode(self._cur_ep)
            self._cur_ep = Episode()
            self._h = None
            self._z = None
            self._prev_action = None
            self.ep_reward = 0.0
            self.ep_len = 0

    # -----------------------------------------------------------------
    def learn(self, force: bool = False) -> Optional[Dict[str, float]]:
        """Thực hiện một (hoặc vài) bước cập nhật trọng số.

        `force=True`: bắt buộc học ngay cả khi replay chưa đủ số lượng.
        Trả về dict loss, hoặc None nếu chưa đủ dữ liệu.
        """
        tc = self.cfg.train
        if not force and not self.replay.enough(tc.batch_size):
            return None

        # Lấy lô từ replay.
        batch = self.replay.sample(tc.batch_size, self.rng)
        if batch is None:
            return None

        # Huấn luyện `updates_per_batch` lần.
        losses = None
        for _ in range(tc.updates_per_batch):
            losses = self.trainer.update(batch)
        self.updates += 1
        return losses

    # -----------------------------------------------------------------
    def step(self, observation: Any, reward: float, done: bool,
             deterministic: bool = False) -> Tuple[np.ndarray, Dict[str, float]]:
        """Gộp act -> observe -> learn thành một lệnh.

        Tiện khi chạy episode: `action, info = soul.step(obs, reward, done)`.
        """
        action = self.act(observation, deterministic)
        self.observe(observation, action, reward, done)
        losses = self.learn()
        return action, losses or {}

    # -----------------------------------------------------------------
    def reset_state(self):
        """Xoa trang thai recurrent giữa các episode.

        `act()` giữ `h`, `z`, `_prev_action` cho lần gọi sau. Khi đánh
        giá nhiều episode liên tiếp, trạng thái đó thuộc về episode trước
        và làm mỗi episode sau bắt đầu lệch — kết quả so sánh với
        baseline trở nên không công bằng.
        """
        self._h = None
        self._z = None
        self._prev_action = None
        self._cur_ep = Episode()

    def save(self, path: str):
        """Lưu checkpoint: trọng số, schema encoder, replay, stats."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.trainer.optimizer.state_dict(),
            'enc_schema': self.enc.schema,
            'enc_overflow': self.enc.overflow,
            # Thống kê chuẩn hoá đi kèm schema: suy luận phải dùng
            # đúng thang đã huấn.
            'obs_norm': self.obs_norm,
            'replay': {
                'episodes': self.replay.episodes,
                'capacity': self.replay.capacity,
                'seq_len': self.replay.seq_len,
            },
            'config': self.cfg.to_dict(),
            'stats': {
                'steps': self.steps,
                'updates': self.updates,
                # `trainer.updates` la bo dem RIENG. Ramp cua
                # `posterior_mix` doc dung so nay, nen neu khong luu
                # thi moi lan resume se reset ramp ve 0 va phai hoc lai
                # tu dau phan "slow posterior" — dang chay lai tren may
                # thue se ton mat phan do.
                'trainer_updates': self.trainer.updates,
                # Chuẩn hóa phần thưởng: mất đi thì mọi dự đoán cũ
                # lệch một hằng số, và R² đo ra âm dù mô hình đúng.
                'reward_stats': dict(self.trainer._reward_stats),
                'ep_reward': self.ep_reward,
                'ep_len': self.ep_len,
            },
            'infer_state': {
                'h': self._h.cpu().numpy() if self._h is not None else None,
                'z': self._z.cpu().numpy() if self._z is not None else None,
                'prev_action': self._prev_action,
                'goal': self._goal,
            },
        }, p)

    def load(self, path: str):
        """Nạp checkpoint, phục hồi nguyên trạng."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        # Trong so Adam khong tuong thich giua cac phien bien do cau truc
        # nhom tham so doi. Checkpoint cu co MOT nhom, optimizer hien tai
        # co HAI (tang bieu dien / dau du doan). `load_state_dict` se nem
        # loi neu khong kiem tra — va no se nem loi LUON, nen phai loi
        # ro rang thay vi de no no.
        #
        # Mat trong so Adam khong phai mat hoc: chi la phan bo bien phai
        # uoc luong lai trong vai buoc dau.
        try:
            self.trainer.optimizer.load_state_dict(ckpt['optimizer'])
        except (ValueError, KeyError) as e:
            print(f'[Soul] BO QUA trong so cua optimizer ({type(e).__name__}: '
                  f'{e}). Cau truc nhom tham so da doi; trong so se duoc '
                  f'khoi tao lai.')
        self.enc.load_schema(ckpt['enc_schema'])
        self.obs_norm = ckpt.get('obs_norm', {})
        self.enc.overflow = ckpt['enc_overflow']
        r = ckpt['replay']
        self.replay = ReplayBuffer(r['capacity'], r['seq_len'])
        self.replay.episodes = r['episodes']
        s = ckpt['stats']
        self.steps = s['steps']
        self.updates = s['updates']
        self.trainer.updates = s.get('trainer_updates', 0)
        # `reward_stats` nằm TRONG `ckpt['stats']`, khác với
        # `obs_norm` nằm ở cấp gốc. Đọc nhầm cấp thì `.get()` trả
        # None, im lặng bỏ qua, va moi lai nap checkpoint deu giu
        # {mean: 0, std: 1} — reward head doan tren thang sai, R2
        # do ra am dung mo hinh hoan toan dung.
        rs = s.get('reward_stats')
        if rs:
            self.trainer._reward_stats.update(rs)
        self.ep_reward = s['ep_reward']
        self.ep_len = s['ep_len']
        inf = ckpt.get('infer_state', {})
        if inf.get('h') is not None:
            self._h = torch.from_numpy(inf['h']).to(self.device)
            self._z = torch.from_numpy(inf['z']).to(self.device)
            self._prev_action = inf['prev_action']
            self._goal = inf['goal']
        print(f'[Soul] Loaded checkpoint: steps={self.steps}, updates={self.updates}')

    # -----------------------------------------------------------------
    def get_stats(self) -> Dict[str, Any]:
        return {
            'steps': self.steps,
            'updates': self.updates,
            'replay_size': self.replay.size,
            'replay_episodes': len(self.replay),
            'enc_slots_used': len(self.enc._schema),
            'enc_overflow': self.enc.overflow,
            'params': self.model.num_parameters(),
        }

    def explain_observation(self, observation: Any, top: int = 10):
        """In ra vài trường quan sát đã gán ô — dùng khi debug để xem
        'linh hồn' thực sự thấy được gì từ dữ liệu bạn đưa vào."""
        return self.enc.explain(observation, top)