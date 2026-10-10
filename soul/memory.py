"""Luu tru experience de huan luyen.

Van de: mau don le khong hoc duoc gi. Muon hieu "hanh dong A gay ra
quan sat B", mo hinh phai thay A->B nhieu lan, theo THU TU. Nen phai luu
chuoi, khong phai tung mau roi.

Cau truc: replay giu danh sach episode. `sample` cat mot doan chuoi dai
`seq_len` tu mot ep, va ghi them 1 buoc cho moi doan de co san `o_{t+1}`.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np


class WorkingMemory:
    """Bộ nhớ tạm đang hoạt động: vài bước gần nhất, dùng cho phản ứng ngay."""

    def __init__(self, maxlen: int = 32):
        self.maxlen = maxlen
        self.events: List[Dict[str, Any]] = []

    def add(self, event: Dict[str, Any]):
        self.events.append(event)
        if len(self.events) > self.maxlen:
            self.events = self.events[-self.maxlen:]

    def recent(self, k: int = 8) -> List[Dict[str, Any]]:
        return self.events[-k:]


class ImportantMemory:
    """Bộ nhớ nội bộ cho các sự kiện quan trọng: nguy hiểm, thành công, thất bại."""

    def __init__(self, maxlen: int = 128):
        self.maxlen = maxlen
        self.items: List[Dict[str, Any]] = []

    def add(self, item: Dict[str, Any]):
        self.items.append(item)
        if len(self.items) > self.maxlen:
            self.items = self.items[-self.maxlen:]

    def summarize(self) -> List[Dict[str, Any]]:
        return list(self.items)


class SkillMemory:
    """Bộ nhớ kỹ năng đã tổng hợp từ trải nghiệm lặp lại. Đây là 'bộ não' lâu dài nội bộ."""

    def __init__(self, maxlen: int = 64):
        self.maxlen = maxlen
        self.skills: List[Dict[str, Any]] = []

    def add(self, skill: Dict[str, Any]):
        for existing in self.skills:
            if existing.get('name') == skill.get('name'):
                existing['confidence'] = float(existing.get('confidence', 0.0)) * 0.7 + float(skill.get('confidence', 0.0)) * 0.3
                existing['count'] = int(existing.get('count', 1)) + 1
                existing['last_seen'] = skill.get('last_seen')
                return
        self.skills.append({**skill, 'count': 1})
        if len(self.skills) > self.maxlen:
            self.skills = self.skills[-self.maxlen:]

    def decay(self, current_time: Optional[int] = None, decay: float = 0.02):
        """Làm suy yếu kỹ năng cũ theo thời gian để tránh giữ lại skill đã lỗi thời."""
        if not self.skills:
            return []
        if current_time is None:
            current_time = max(
                int(s.get('last_seen', 0) if s.get('last_seen') is not None else 0)
                for s in self.skills
            )
        updated = []
        for skill in self.skills:
            last_seen = skill.get('last_seen', current_time)
            if last_seen is None:
                last_seen = current_time
            try:
                age = max(0, int(current_time) - int(last_seen))
            except (TypeError, ValueError):
                age = 0
            confidence = float(skill.get('confidence', 0.0))
            decayed = confidence * math.exp(-max(0.0, decay) * age)
            skill['confidence'] = float(decayed)
            skill['last_seen'] = current_time if skill.get('last_seen') is None else skill.get('last_seen')
            if skill['confidence'] > 0.01:
                updated.append(skill)
        self.skills = updated
        return self.skills

    def best(self, tag: Optional[str] = None) -> List[Dict[str, Any]]:
        items = self.skills
        if tag is not None:
            items = [s for s in items if tag in s.get('tags', [])]
        return sorted(items, key=lambda s: s.get('confidence', 0.0), reverse=True)


class ArchiveMemory:
    """Bộ nhớ lưu trữ nhẹ: nhật ký trải nghiệm + skill đã học, không cần DB ngoài."""

    def __init__(self, max_events: int = 2000, max_skills: int = 256):
        self.max_events = max_events
        self.max_skills = max_skills
        self.events: List[Dict[str, Any]] = []
        self.skills: List[Dict[str, Any]] = []

    def _jsonable(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.generic,)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): self._jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._jsonable(v) for v in obj]
        return obj

    def add_event(self, event: Dict[str, Any]):
        self.events.append(self._jsonable(event))
        if len(self.events) > self.max_events:
            self.events = self.events[-self.max_events:]

    def add_skill(self, skill: Dict[str, Any]):
        self.skills.append(self._jsonable(skill))
        if len(self.skills) > self.max_skills:
            self.skills = self.skills[-self.max_skills:]

    def to_dict(self) -> Dict[str, Any]:
        return {'events': self.events, 'skills': self.skills}

    def from_dict(self, payload: Optional[Dict[str, Any]]):
        if not payload:
            return
        self.events = list(payload.get('events', []))
        self.skills = list(payload.get('skills', []))

    def save(self, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        self.from_dict(payload)


class MemoryManager:
    """Quản lý các lớp nhớ: tạm, quan trọng, kỹ năng, lưu trữ nhẹ."""

    def __init__(self):
        self.working = WorkingMemory()
        self.important = ImportantMemory()
        self.skills = SkillMemory()
        self.archive = ArchiveMemory()

    def record_event(self, event: Dict[str, Any], *, important: bool = False):
        self.working.add(event)
        if important:
            self.important.add(event)
        self.archive.add_event(event)

    def extract_skill(self, event: Dict[str, Any], *, name: str,
                     tags: Optional[List[str]] = None,
                     confidence: float = 0.5):
        skill = {
            'name': name,
            'tags': tags or [],
            'confidence': float(confidence),
            'last_seen': event.get('time') or event.get('step'),
            'context': event.get('context', {}),
            'action': event.get('action'),
        }
        self.skills.add(skill)
        self.archive.add_skill(skill)
        return skill

    def extract_skill_from_event(self, event: Dict[str, Any], *, name: Optional[str] = None,
                                threshold: float = 0.05):
        """Chuyển một trải nghiệm quan trọng thành một skill nếu nó mang lại
        tín hiệu rõ ràng (thành công / thất bại / done)."""
        reward = float(event.get('reward', 0.0))
        done = bool(event.get('done', False))
        if abs(reward) <= threshold and not done:
            return None
        skill_name = name or ('success_pattern' if reward >= 0 else 'avoid_failure')
        tags = ['important']
        if reward >= 0:
            tags.append('success')
        else:
            tags.append('failure')
        if done:
            tags.append('terminal')
        return self.extract_skill(
            event,
            name=skill_name,
            tags=tags,
            confidence=min(0.99, 0.5 + abs(reward)),
        )

    def refresh_skills(self, current_time: Optional[int] = None, decay: float = 0.02):
        """Cập nhật trạng thái của skill theo thời gian: cũ thì suy giảm độ tin cậy."""
        self.skills.decay(current_time=current_time, decay=decay)
        self.archive.skills = [s for s in self.archive.skills if float(s.get('confidence', 0.0)) > 0.01]
        return self.skills.skills

    def summarize_archive_to_skills(self, *, min_events: int = 2,
                                   max_skills: int = 4) -> List[Dict[str, Any]]:
        """Tổng hợp nhiều event tương tự từ archive thành một skill reusable.

        Ví dụ: nhiều lần thử hành động tương tự với cùng mục tiêu sẽ được gộp
        thành một skill dạng 'goal_forward_pattern', giúp agent học được một
        kiểu hành động ổn định thay vì chỉ nhớ từng bước riêng lẻ.
        """
        if not self.archive.events:
            return []

        grouped: Dict[str, Dict[str, Any]] = {}
        for event in self.archive.events:
            context = event.get('context', {})
            goal = 'generic'
            if isinstance(context, dict):
                goal = str(context.get('goal') or context.get('target') or 'generic')
            action = event.get('action')
            action_value = 0.0
            if action is not None:
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                if arr.size:
                    action_value = float(np.mean(arr))
            key = str(goal)
            bucket = grouped.setdefault(key, {
                'goal': goal,
                'events': [],
                'reward_total': 0.0,
                'action_values': [],
            })
            bucket['events'].append(event)
            bucket['reward_total'] += float(event.get('reward', 0.0))
            bucket['action_values'].append(action_value)

        created: List[Dict[str, Any]] = []
        for group in grouped.values():
            events = group['events']
            if len(events) < min_events:
                continue
            goal = group['goal']
            action_mean = float(np.mean(group['action_values'])) if group['action_values'] else 0.0
            reward_mean = group['reward_total'] / len(events)
            confidence = min(0.99, 0.35 + max(0.0, reward_mean) + 0.15 * min(len(events), 5))
            skill = {
                'name': f"goal_{str(goal).replace(' ', '_')}_pattern",
                'tags': ['archive', 'summarized'],
                'confidence': float(confidence),
                'last_seen': max(int(float(e.get('time', 0))) for e in events),
                'context': {'goal': goal},
                'action': np.asarray([action_mean], dtype=np.float32),
                'count': len(events),
            }
            self.skills.add(skill)
            self.archive.add_skill(skill)
            created.append(skill)

        return created[:max_skills]

    def recall_skills(self, context: Optional[Dict[str, Any]] = None,
                     *, limit: int = 5) -> List[Dict[str, Any]]:
        """Tìm các skill có cùng bối cảnh nhất."""
        def _matches(a, b):
            if isinstance(a, (list, tuple, np.ndarray)) or isinstance(b, (list, tuple, np.ndarray)):
                try:
                    return bool(np.array_equal(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)))
                except Exception:
                    return list(a) == list(b) if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)) else str(a) == str(b)
            return a == b

        if not self.skills.skills:
            return []
        if context is None:
            return self.skills.best()[:limit]

        ctx = context.get('context', context) if isinstance(context, dict) else context
        if not isinstance(ctx, dict):
            return self.skills.best()[:limit]

        matches = []
        for skill in self.skills.skills:
            score = 0
            s_ctx = skill.get('context', {})
            if isinstance(s_ctx, dict):
                for k, v in s_ctx.items():
                    if _matches(ctx.get(k), v):
                        score += 1
            if score > 0 or 'important' in skill.get('tags', []):
                confidence = float(skill.get('confidence', 0.0))
                matches.append((score, confidence, skill))
        if not matches:
            return self.skills.best()[:limit]
        matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
        return [skill for _, _, skill in matches[:limit]]

    def to_dict(self) -> Dict[str, Any]:
        return {'archive': self.archive.to_dict(), 'skills': [s for s in self.skills.skills]}

    def from_dict(self, payload: Optional[Dict[str, Any]]):
        if not payload:
            return
        self.archive.from_dict(payload.get('archive', {}))
        skills = payload.get('skills', [])
        self.skills.skills = list(skills)

    def save(self, path: str):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self, path: str):
        import json
        with open(path, 'r', encoding='utf-8') as f:
            payload = json.load(f)
        self.from_dict(payload)


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