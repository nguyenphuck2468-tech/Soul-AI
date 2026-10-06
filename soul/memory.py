"""Luu tru experience de huan luyen.

Van de: mau don le khong hoc duoc gi. Muon hieu "hanh dong A gay ra
quan sat B", mo hinh phai thay A->B nhieu lan, theo THU TU. Nen phai luu
chuoi, khong phai tung mau roi.

Cau truc: replay giu danh sach episode. `sample` cat mot doan chuoi dai
`seq_len` tu mot ep, va ghi them 1 buoc cho moi doan de co san `o_{t+1}`.
"""

from typing import Any, Dict, List, Optional

import numpy as np


class Episode:
    """Mot lan chay lien tuc, khong bi cat ngan."""

    __slots__ = ('values', 'masks', 'texts', 'actions', 'rewards', 'continues',
                 'goals', 'length')

    def __init__(self):
        self.values: List[np.ndarray] = []
        self.masks: List[np.ndarray] = []
        self.texts: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.continues: List[float] = []
        self.goals: List[np.ndarray] = []
        self.length = 0

    def append(self, enc, action: np.ndarray, reward: float,
               cont: float, goal: np.ndarray):
        self.values.append(enc['values'])
        self.masks.append(enc['mask'])
        self.texts.append(enc['text'])
        self.actions.append(action)
        self.rewards.append(reward)
        self.continues.append(cont)
        self.goals.append(goal)
        self.length += 1

    def __len__(self):
        return self.length


class ReplayBuffer:
    """Kho experience, gioi han dung luong bang so ep."""

    def __init__(self, capacity: int = 50000, seq_len: int = 12):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes: List[Episode] = []
        self._total = 0

    def add_episode(self, ep: Episode):
        if ep.length == 0:
            return
        self.episodes.append(ep)
        self._total += ep.length
        # Xoa tu ep cuoi vao. Giu ep moi hon la giu ep dau tien: ep moi
        # phan anh hanh vi hien tai cua mo hinh, ep cu da loi thoi.
        while self._total > self.capacity and self.episodes:
            old = self.episodes.pop(0)
            self._total -= old.length

    @property
    def size(self) -> int:
        return self._total

    def __len__(self) -> int:
        return len(self.episodes)

    def enough(self, batch_size: int) -> bool:
        """Du du lieu de cat chuoi chua? Can it nhat `batch_size` ep, va
        moi ep can du dai gap `seq_len + 1` (vi can o_{t+1})."""
        need = self.seq_len + 1
        ok = [e for e in self.episodes if e.length >= need]
        return len(ok) >= batch_size

    def sample(self, batch_size: int, rng: np.random.Generator) -> Optional[Dict[str, np.ndarray]]:
        """Cat mot lo chuoi ngan. Tra None neu chua du du lieu."""
        need = self.seq_len + 1
        cands = [e for e in self.episodes if e.length >= need]
        if len(cands) < batch_size:
            return None

        idx = rng.integers(0, len(cands), size=batch_size)
        seq = self.seq_len
        out = {
            'values': np.zeros((batch_size, seq + 1, cands[0].values[0].shape[0]), np.float32),
            'masks': np.zeros((batch_size, seq + 1, cands[0].masks[0].shape[0]), np.float32),
            'texts': np.zeros((batch_size, seq + 1, cands[0].texts[0].shape[0]), np.float32),
            'actions': np.zeros((batch_size, seq, cands[0].actions[0].shape[0]), np.float32),
            'rewards': np.zeros((batch_size, seq), np.float32),
            'continues': np.zeros((batch_size, seq), np.float32),
            'goals': np.zeros((batch_size, seq, cands[0].goals[0].shape[0]), np.float32),
            'valid': np.ones((batch_size, seq), np.float32),
        }

        for b, ci in enumerate(idx):
            ep = cands[ci]
            # `high` = seq + 1 - 1 = seq, de duoc o_{t+1} o buoc cuoi.
            start = rng.integers(0, ep.length - need + 1)
            for t in range(seq):
                s = start + t
                out['values'][b, t] = ep.values[s]
                out['masks'][b, t] = ep.masks[s]
                out['texts'][b, t] = ep.texts[s]
                out['actions'][b, t] = ep.actions[s]
                out['rewards'][b, t] = ep.rewards[s]
                out['continues'][b, t] = ep.continues[s]
                out['goals'][b, t] = ep.goals[s]
                # Buoc ngay sau khi ep ket thuc thi khong hop le — dau
                # chuoi qua ranh gioi, khong giu bo nho sang ep sau.
                out['valid'][b, t] = ep.continues[s]
            s = start + seq
            out['values'][b, seq] = ep.values[s]
            out['masks'][b, seq] = ep.masks[s]
            out['texts'][b, seq] = ep.texts[s]
        return out