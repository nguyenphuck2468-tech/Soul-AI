"""Chuẩn hoá dữ liệu thô thành vector — dùng chung cho mọi nguồn dữ liệu.

Lớp này nhận dict lồng nhau, list, số, chuỗi, bool, mảng NumPy — và trả về
ba mảng:

    values  (max_features,)    con số, da bop ve khoang [-1, 1]
    mask    (max_features,)    1 = o nay co du lieu that, 0 = trong
    text    (text_buckets,)    1 = ho nay xuat hien trong mau nay

`mask` la mau chot. Mot o trong PHAI mang gia tri 0 VA duoc danh dau 0.
Neu khong danh dau, mo hinh khong phan biet duoc "gia tri 0 that" voi
"khong co gi", va no se hoc rang du lieu la toan 0 — mot phan lon cac
tap du lieu duong deu khong phai the.

Vi sao khong dung mot con so chuan hoa dung chung cho moi thuoc tinh:
khong co cach nao dung chung. Mot truong do tu 0..1, truong khac do tu
0..1e6. Dinh nghia chung se lam truong lon nghiem khong the hoi duoc, hay
lam truong nho bi bo nhe. O day moi con so duoc TUT chuan hoa rieng bang
`value / (1 + |value|)` — ham nay giu dau, giu thu tu, va bo ve 1 moi
dau du khong bao gio, nen khong bao gio bien mat dau tin.
"""

import hashlib
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def stable_hash(text: str) -> int:
    """Bam chuoi thanh so nguyen on dinh.

    KHONG dung `hash()` cua Python: no bi xao tron theo tung lan chay
    (PYTHONHASHSEED). Dung no thi checkpoint nap lai se gan sai o, va mo
    hinh se xem nhu dang nhin mot the gioi khac — loss nhay vuc len ngay
    luc nap, ma khong co loi nao bao loi.
    """
    return int.from_bytes(
        hashlib.blake2b(str(text).encode('utf-8'), digest_size=8).digest(), 'big')


def squash(v: float) -> float:
    """Bop mot so bat ky ve [-1, 1], giu nguyen dau va thu tu."""
    if not math.isfinite(v):
        return 0.0
    return v / (1.0 + abs(v))


class ObsEncoder:
    """Bien du lieu tho thanh vector. Tu gan o theo duong dan truong."""

    def __init__(self, cfg=None):
        try:
            from .config import ObsConfig
        except ImportError:  # pragma: no cover - chạy trực tiếp như script
            from config import ObsConfig
        self.cfg = cfg or ObsConfig()
        self._schema: Dict[str, int] = {}
        # Dem so con so bi bo vi het ngan sach. Doc so nay de biet mo hinh
        # dang bo mat thong tin gi — im lang bo thi khong bao gio biet.
        self.overflow = 0

    # -- schema ---------------------------------------------------------
    @property
    def schema(self) -> Dict[str, int]:
        return dict(self._schema)

    def load_schema(self, schema: Dict[str, int]):
        """Nap lai tuong o tu checkpoint cu (xem soul.load)."""
        self._schema = {str(k): int(v) for k, v in schema.items()}

    def _slot(self, path: str) -> int:
        idx = self._schema.get(path)
        if idx is not None:
            return idx
        if len(self._schema) >= self.cfg.max_features:
            self.overflow += 1
            return -1
        idx = len(self._schema)
        self._schema[path] = idx
        return idx

    # -- vector hoa -----------------------------------------------------
    @property
    def input_dim(self) -> int:
        """So luong dau vao cua mang: values + mask + text."""
        return self.cfg.max_features * 2 + self.cfg.text_buckets

    def encode(self, obs: Any) -> Dict[str, np.ndarray]:
        values = np.zeros(self.cfg.max_features, dtype=np.float32)
        mask = np.zeros(self.cfg.max_features, dtype=np.float32)
        text = np.zeros(self.cfg.text_buckets, dtype=np.float32)
        self._walk('', obs, values, mask, text)
        return {'values': values, 'mask': mask, 'text': text}

    def encode_batch(self, batch: Sequence[Any]) -> Dict[str, np.ndarray]:
        enc = [self.encode(o) for o in batch]
        return {k: np.stack([e[k] for e in enc]).astype(np.float32)
                for k in ('values', 'mask', 'text')}

    def _walk(self, prefix: str, node: Any,
              values: np.ndarray, mask: np.ndarray, text: np.ndarray,
              depth: int = 0):
        # Gioi han do sau: du lieu that co the long vai chuc cap, con mau
        # hong co cap nao la huu ich.
        if depth > 8:
            return

        if isinstance(node, dict):
            for k in sorted(node.keys(), key=str):
                child = f'{prefix}.{k}' if prefix else str(k)
                self._walk(child, node[k], values, mask, text, depth + 1)
            return

        if isinstance(node, (bool, np.bool_)):
            # bool la con cua int trong Python — phai kiem bool TRUOC int,
            # neu khong True se thanh so 1 va mat y nghia khi co.
            #
            # Ghi THANG 1.0/0.0, KHONG qua squash: squash(1.0) = 0.5, trung
            # voi gia tri cua so thuc 0.5. Nhu vay bool True va so 0.5 tro
            # thanh mot — mat thong tin ngay tu khi vector hoa, va moi thu
            # van chay binh thuong.
            self._put(prefix, 1.0 if node else 0.0, values, mask, raw=True)
            return

        if isinstance(node, (int, float, np.integer, np.floating)):
            v = float(node)
            # NaN/inf tu du lieu ban: danh dau o trong thay vi day NaN vao
            # mo hinh, vi NaS lan ra moi trong so mot cach khong the can.
            if not math.isfinite(v):
                return
            # Trường `time` là đặc biệt: nếu ta clip thẳng theo 20.0, mọi thời
            # điểm >=20 đều gần như giống nhau. Để giữ thông tin trong cả chuỗi,
            # ta chuẩn hóa theo tỉ lệ thực của episode/horizon.
            if prefix == 'time' or prefix.endswith('.time'):
                scale = max(self.cfg.time_scale, 1e-6)
                self._put(prefix, np.clip(v / scale, -1.0, 1.0), values, mask, raw=True)
                return
            clip = self.cfg.clip
            self._put(prefix, max(-clip, min(clip, v)), values, mask)
            return

        if isinstance(node, str):
            if node:
                text[stable_hash(node) % self.cfg.text_buckets] = 1.0
            return

        if isinstance(node, (list, tuple)):
            for i, item in enumerate(node):
                self._walk(f'{prefix}[{i}]', item, values, mask, text, depth + 1)
            return

        if isinstance(node, np.ndarray):
            # Mang phang: moi phan tu lay chieu so thuc cua no.
            if node.dtype.kind in 'fiu':
                if node.size == 1:
                    self._walk(prefix, float(node.reshape(-1)[0]),
                               values, mask, text, depth + 1)
                else:
                    self._put(prefix, float(node.size), values, mask)
            return

        if node is None:
            return

        # Kieu la: van ghi nho rang "co thu gi do o day" thay vi bo trang
        # im — it nhat mo hinh biet co thu.
        text[stable_hash(f'{prefix}:{type(node).__name__}') % self.cfg.text_buckets] = 1.0

    def _put(self, path: str, value: float,
             values: np.ndarray, mask: np.ndarray, raw: bool = False):
        idx = self._slot(path)
        if idx < 0:
            return
        # `raw=True`: bo qua squash — dung cho bool, da san nam trong
        # khoang [0,1]. Qua squash thi True -> 0.5 trung voi so 0.5.
        v = max(-1.0, min(1.0, value)) if raw else squash(value)
        values[idx] = np.clip(v, -1.0, 1.0)
        mask[idx] = 1.0

    # -- tien ich -------------------------------------------------------
    def feature_names(self) -> List[str]:
        inv = {v: k for k, v in self._schema.items()}
        return [inv.get(i, f'<trong_{i}>') for i in range(self.cfg.max_features)]

    def explain(self, obs: Any, top: int = 10) -> List[Tuple[str, float]]:
        """Xem mo hinh 'thay' duoc gi tu mot mau — tien khi huan luyen,
        cach nhanh nhat de bat loi chuan hoa."""
        e = self.encode(obs)
        names = self.feature_names()
        out = [(names[i], float(e['values'][i]))
               for i in range(self.cfg.max_features) if e['mask'][i] > 0]
        return out[:top]