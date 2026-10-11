"""Thin interface shared by world-model implementations."""

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np


@dataclass
class LatentSnapshot:
    payload: Any
    tick: int


@dataclass
class ImagineResult:
    pred_obs: np.ndarray
    uncertainty: np.ndarray
    horizon: int
    wall_seconds: float
    uncertainty_status: str = 'NOT_IMPLEMENTED'


class WorldModel(Protocol):
    def reset(self) -> None: ...

    def observe(self, obs: Any, action: int, reward: float, done: bool) -> None: ...

    def snapshot(self) -> LatentSnapshot: ...

    def restore(self, snap: LatentSnapshot) -> None: ...

    def imagine(self, snap: LatentSnapshot,
                actions: Sequence[int]) -> ImagineResult: ...

    def surprise(self, obs: Any) -> float: ...
