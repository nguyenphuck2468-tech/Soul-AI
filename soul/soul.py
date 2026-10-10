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

try:
    from .config import Config
    from .obs import ObsEncoder
    from .model import SoulModel
    from .memory import Episode, ReplayBuffer, MemoryManager
    from .trainer import Trainer
except ImportError:  # pragma: no cover - chạy trực tiếp như script
    from config import Config
    from obs import ObsEncoder
    from model import SoulModel
    from memory import Episode, ReplayBuffer, MemoryManager
    from trainer import Trainer


class SoulBrain:
    """Bộ não của Soul: nhận biết -> dự đoán -> lập kế hoạch -> ghi nhớ.

    Đây là lớp trừu tượng giúp mô hình được mô tả như một agent có cấu trúc
    nhận thức rõ ràng, không đơn thuần là một khối phương thức `act()` và
    `learn()` rời rạc.
    """

    def __init__(self, soul: 'Soul'):
        self.soul = soul
        self.memory = []

    def perceive(self, observation: Any) -> np.ndarray:
        """Nhận biết quan sát hiện tại và biến thành vector đặc trưng."""
        enc = self.soul.enc.encode(observation)
        return np.concatenate([enc['values'], enc['mask'], enc['text']], axis=0).astype(np.float32)

    def predict(self, observation: Any, action: np.ndarray) -> np.ndarray:
        """Dự đoán trạng thái latent tiếp theo khi thực hiện hành động."""
        return self.soul.predict_action_effect(observation, action)

    def plan(self, observation: Any, n_samples: Optional[int] = None,
             deterministic: bool = False) -> np.ndarray:
        """Lập kế hoạch dựa trên mô hình thế giới."""
        return self.soul.plan_action(observation, n_samples=n_samples,
                                     deterministic=deterministic)

    def tag_event(self, observation: Any, action: np.ndarray,
                 reward: float, done: bool, goal: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Gắn nhãn trải nghiệm thành một event có bối cảnh và nhãn quan trọng."""
        event = {
            'observation': observation,
            'action': np.asarray(action, dtype=np.float32),
            'reward': float(reward),
            'done': bool(done),
            'goal': None if goal is None else np.asarray(goal, dtype=np.float32),
            'time': self.soul.steps,
            'context': {'done': bool(done), 'goal': None if goal is None else np.asarray(goal, dtype=np.float32)},
            'tags': ['runtime'],
        }
        if abs(float(reward)) > 0.05:
            event['tags'].append('salient')
        if bool(done):
            event['tags'].append('terminal')
        if float(reward) >= 0.0:
            event['tags'].append('success')
        else:
            event['tags'].append('failure')
        return event

    def store_event(self, event: Dict[str, Any], *, important: Optional[bool] = None):
        """Lưu sự kiện vào bộ nhớ làm việc và bộ nhớ lâu dài nội bộ."""
        if important is None:
            important = bool(event.get('done', False)) or abs(float(event.get('reward', 0.0))) > 0.05
        self.memory.append(event)
        if len(self.memory) > 200:
            self.memory = self.memory[-200:]
        self.soul.memory_manager.record_event(event, important=important)
        self.soul.memory_manager.refresh_skills(current_time=self.soul.steps, decay=0.02)
        if len(self.soul.memory_manager.archive.events) >= 2:
            self.soul.memory_manager.summarize_archive_to_skills(min_events=2, max_skills=4)
        if important:
            name = 'success_pattern' if float(event.get('reward', 0.0)) >= 0.0 else 'avoid_failure'
            self.soul.memory_manager.extract_skill(
                event,
                name=name,
                tags=['important', 'runtime'],
                confidence=min(0.99, 0.5 + abs(float(event.get('reward', 0.0)))),
            )

    def remember(self, observation: Any, action: np.ndarray,
                 reward: float, done: bool, goal: Optional[np.ndarray] = None):
        """Ghi nhớ trải nghiệm mới để học ngay khi nó xuất hiện."""
        event = self.tag_event(observation, action, reward, done, goal)
        self.store_event(event, important=(abs(float(reward)) > 0.05 or bool(done)))
        self.soul.observe(observation, action, reward, done, goal)

    def learn(self, force: bool = False, real_time: bool = False,
              online_steps: int = 1) -> Optional[Dict[str, float]]:
        """Học ngay từ dữ liệu mới khi `real_time=True`, không chờ replay đầy."""
        if real_time:
            # Học từng chuỗi mới vừa sinh ra; nếu chưa đủ dữ liệu thì bỏ qua.
            # Điều này cho phép agent cập nhật liên tục khi đang tương tác
            # với môi trường thực, thay vì chờ đủ `batch_size` episode.
            batch = self.soul._sample_online_batch(min(online_steps, max(1, self.soul.cfg.train.batch_size)))
            if batch is None:
                return None
            losses = None
            for _ in range(self.soul.cfg.train.updates_per_batch):
                losses = self.soul.trainer.update(batch)
            self.soul.updates += 1
            return losses
        return self.soul.learn(force=force)


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

        # Replay và bộ nhớ nội bộ.
        self.replay = ReplayBuffer(self.cfg.train.replay_capacity, self.cfg.train.seq_len)
        self._cur_ep = Episode()
        self.memory_manager = MemoryManager()
        self.brain = SoulBrain(self)

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

    def recall_skill_action(self, observation: Any) -> Optional[np.ndarray]:
        """Lấy skill phù hợp nhất với bối cảnh hiện tại để làm bias cho hành động."""
        enc = self.enc.encode(observation)
        context = {'goal': self._goal.copy().reshape(-1).tolist() if self._goal is not None else [],
                   'time': float(enc['values'][self.enc._schema.get('time', 0)] if 'time' in self.enc._schema else 0.0)}
        skills = self.memory_manager.recall_skills({'context': context})
        if not skills:
            return None
        action = skills[0].get('action')
        if action is None:
            return None
        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        if arr.size < self.cfg.model.action_dim:
            arr = np.concatenate([arr, np.zeros(self.cfg.model.action_dim - arr.size, dtype=np.float32)])
        elif arr.size > self.cfg.model.action_dim:
            arr = arr[:self.cfg.model.action_dim]
        return arr.astype(np.float32)

    def plan_action(self, observation: Any, n_samples: Optional[int] = None,
                   deterministic: bool = False) -> np.ndarray:
        """Lập kế hoạch với một vài hành động giả định, rồi chọn cái có
        giá trị đồng ý với mô hình thế giới nhất.

        Đây là bước trung gian giữa "đoán action" và "suy đoán hậu quả".
        Thay vì chọn ngay một mẫu từ chính policy, Soul thử nhiều hành động
        trong đầu và chấm điểm mỗi hành động bằng hiệu quả dự đoán được của
        hành động đó trên trạng thái tiếp theo.
        """
        if n_samples is None:
            n_samples = max(1, int(self.cfg.model.plan_samples))

        if self._h is None:
            self._h = torch.zeros(1, self.model.hidden, device=self.device)
            self._z = torch.zeros(1, self.model.latent, device=self.device)
            self._prev_action = np.zeros(self.cfg.model.action_dim, dtype=np.float32)
            if self._goal is None:
                self._goal = np.zeros((1, self.model.goal_dim), dtype=np.float32)

        goal = self._goal if self._goal is not None else np.zeros((1, self.model.goal_dim), np.float32)
        enc = self.enc.encode(observation)
        obs_vec = np.concatenate([enc['values'], enc['mask'], enc['text']], axis=0)

        with torch.no_grad():
            h = torch.as_tensor(self._h, device=self.device)
            z = torch.as_tensor(self._z, device=self.device)
            g = torch.as_tensor(goal, device=self.device)
            o = torch.as_tensor(obs_vec, device=self.device).unsqueeze(0)
            q_mu, _ = self.model.encoder(o)
            mean, logstd, logits = self.model.policy_head(h, q_mu, g)
            candidates = []
            if deterministic:
                candidates.append(self.model.policy_head.sample(
                    mean, logstd, logits, deterministic=True).squeeze(0))
            else:
                candidates.append(self.model.policy_head.sample(
                    mean, logstd, logits, deterministic=False).squeeze(0))
                for _ in range(max(1, n_samples - 1)):
                    candidates.append(self.model.policy_head.sample(
                        mean, logstd, logits, deterministic=False).squeeze(0))

            best_action = None
            best_score = -float('inf')
            for cand in candidates:
                action = cand.cpu().numpy().reshape(-1)
                if action.size < self.cfg.model.action_dim:
                    pad = np.zeros(self.cfg.model.action_dim - action.size, dtype=np.float32)
                    action = np.concatenate([action, pad])
                elif action.size > self.cfg.model.action_dim:
                    action = action[:self.cfg.model.action_dim]

                pred_state = self.predict_action_effect(observation, action)
                pred_state_t = torch.as_tensor(pred_state, device=self.device, dtype=torch.float32).unsqueeze(0)
                reward_hat = self.model.reward_head(h, pred_state_t).squeeze(0)
                action_penalty = 0.05 * float(np.mean(np.square(action)))
                score = float(reward_hat.item()) - action_penalty
                if score > best_score:
                    best_score = score
                    best_action = action.astype(np.float32)

            if best_action is None:
                best_action = np.zeros(self.cfg.model.action_dim, dtype=np.float32)
            return best_action

    def act(self, observation: Any, deterministic: bool = False) -> np.ndarray:
        """Dự đoán hành động từ quan sát hiện tại.

        Lần đầu gọi (khi `self._h` là None) sẽ khởi tạo trạng thái mới.
        `deterministic=True` dùng khi muốn hành động chắc chắn (không
        khám phá) — ví dụ lúc đánh giá.
        """
        # Bước lập kế hoạch trong đầu: thử một vài hành động, chọn cái có
        # điểm đánh giá cao nhất theo mô hình thế giới. Đây là cách gần nhất
        # với "con người nhìn hành động rồi đoán hậu quả".
        if not deterministic and self.cfg.model.plan_samples > 0:
            action = self.plan_action(observation, n_samples=self.cfg.model.plan_samples,
                                      deterministic=False)
            self._prev_action = action
            return action

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

        skill_action = self.recall_skill_action(observation)
        if skill_action is not None:
            action = 0.7 * np.asarray(action, dtype=np.float32) + 0.3 * skill_action

        self._h = h
        self._z = z
        self._prev_action = action
        return action

    # -----------------------------------------------------------------
    def predict_action_effect(self, observation: Any, action: np.ndarray) -> np.ndarray:
        """Dự đoán trạng thái latent sau khi thực hiện một hành động.

        Hàm này trả về vector latent "điều gì sẽ diễn ra tiếp theo nếu làm
        hành động này ở quan sát hiện tại" — đó là cách cơ bản để AI "tính
        trước hậu quả" như con người nhìn một động tác và suy đoán hậu quả.
        """
        enc = self.enc.encode(observation)
        obs_vec = np.concatenate([enc['values'], enc['mask'], enc['text']], axis=0)
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.size < self.cfg.model.action_dim:
            pad = np.zeros(self.cfg.model.action_dim - act.size, dtype=np.float32)
            act = np.concatenate([act, pad])
        elif act.size > self.cfg.model.action_dim:
            act = act[:self.cfg.model.action_dim]
        act = act.reshape(1, -1)

        if self._h is None:
            self._h = torch.zeros(1, self.model.hidden, device=self.device)
            self._z = torch.zeros(1, self.model.latent, device=self.device)
            if self._goal is None:
                self._goal = np.zeros((1, self.model.goal_dim), dtype=np.float32)

        with torch.no_grad():
            h = torch.as_tensor(self._h, device=self.device)
            z = torch.as_tensor(self._z, device=self.device)
            o = torch.as_tensor(obs_vec, device=self.device).unsqueeze(0)
            a = torch.as_tensor(act, device=self.device)

            h, _, _ = self.model.dynamics.step(h, z, a)
            q_mu, _ = self.model.encoder(o)
            pred_delta = self.model.action_effect_head(h, q_mu, a)
            pred_state = q_mu + pred_delta

        return pred_state.squeeze(0).cpu().numpy()

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
        event = {
            'time': self.steps,
            'action': np.asarray(action, dtype=np.float32),
            'reward': float(reward),
            'done': bool(done),
            'goal': g.copy(),
            'context': {'observation': enc, 'goal': g.copy()},
        }
        self.memory_manager.record_event(event, important=(abs(float(reward)) > 0.05 or bool(done)))
        self.memory_manager.refresh_skills(current_time=self.steps, decay=0.02)
        if abs(float(reward)) > 0.05 or bool(done):
            name = 'success_pattern' if float(reward) >= 0.0 else 'avoid_failure'
            self.memory_manager.extract_skill(
                event,
                name=name,
                tags=['important', 'runtime'],
                confidence=min(0.99, 0.5 + abs(float(reward))),
            )
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
    def _sample_online_batch(self, batch_size: Optional[int] = None):
        """Lấy batch học từ dữ liệu mới nhất khi replay chưa đủ.

        Mục tiêu là cho Agent học theo thời gian thực: ưu tiên `current_episode`
        trước, rồi mới đến replay dài hạn. Điều này tránh tình trạng model
        phải chờ nhiều episode mới bắt đầu cập nhật.
        """
        tc = self.cfg.train
        bs = tc.batch_size if batch_size is None else max(1, int(batch_size))

        # Ưu tiên dữ liệu mới nhất để học ngay.
        if self._cur_ep.length >= tc.seq_len + 1:
            tmp = ReplayBuffer(capacity=max(1, tc.replay_capacity), seq_len=tc.seq_len)
            tmp.episodes = [self._cur_ep]
            batch = tmp.sample(1, self.rng)
            if batch is not None:
                return batch

        if self.replay.enough(bs):
            return self.replay.sample(bs, self.rng)
        return None

    def learn(self, force: bool = False, real_time: bool = False,
              online_steps: int = 1) -> Optional[Dict[str, float]]:
        """Thực hiện một (hoặc vài) bước cập nhật trọng số.

        `force=True`: bắt buộc học ngay cả khi replay chưa đủ số lượng.
        `real_time=True`: ưu tiên batch từ trải nghiệm mới nhất để học ngay
        theo thời gian thực, không chờ đầy bộ nhớ replay.
        """
        tc = self.cfg.train
        if real_time:
            batch = self._sample_online_batch(batch_size=max(1, int(online_steps)))
            if batch is None:
                return None
            losses = None
            for _ in range(tc.updates_per_batch):
                losses = self.trainer.update(batch)
            self.updates += 1
            return losses

        if not force and not self.replay.enough(tc.batch_size):
            return None

        # Lấy lô từ replay.
        batch = self.replay.sample(tc.batch_size, self.rng)
        if batch is None:
            if force:
                batch = self._sample_online_batch(batch_size=tc.batch_size)
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
        Chế độ học ngay: sau mỗi trải nghiệm mới, agent ưu tiên học từ chuỗi
        gần nhất để phản ứng theo thời gian thực.
        """
        action = self.act(observation, deterministic)
        self.observe(observation, action, reward, done)
        losses = self.learn(real_time=True, online_steps=max(1, self.cfg.train.batch_size // 4))
        return action, losses or {}

    # -----------------------------------------------------------------
    def save_archive(self, path: str):
        """Lưu archive nhớ lâu dài ra file JSON."""
        self.memory_manager.archive.save(path)
        return path

    def load_archive(self, path: str):
        """Nạp archive nhớ lâu dài từ file JSON và đồng bộ skill trong bộ nhớ."""
        self.memory_manager.archive.load(path)
        if self.memory_manager.archive.skills:
            self.memory_manager.skills.skills = list(self.memory_manager.archive.skills)
        return len(self.memory_manager.archive.events)

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
            'memory': self.memory_manager.to_dict(),
        }, p)

    def load(self, path: str):
        """Nạp checkpoint, phục hồi nguyên trạng."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
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
        mem = ckpt.get('memory')
        if mem:
            self.memory_manager.from_dict(mem)
        archive_path = ckpt.get('archive_path')
        if archive_path and Path(archive_path).exists():
            self.load_archive(archive_path)
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