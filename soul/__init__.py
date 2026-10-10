"""Soul — mô hình thế giới độc lập (RSSM/Dreamer-style).

Không phụ thuộc Minecraft, không phụ thuộc mineflayer. Chỉ PyTorch + NumPy.

Usage:
    from soul import Soul, Config
    s = Soul(Config())
    action = s.act(observation)
    s.observe(observation, action, reward, done, goal)
    s.learn()
    s.save('my_soul.pt')
"""

from .config import Config, ModelConfig, ObsConfig, TrainConfig
from .obs import ObsEncoder, stable_hash, squash
from .model import SoulModel
from .memory import Episode, ReplayBuffer, WorkingMemory, ImportantMemory, SkillMemory, MemoryManager
from .trainer import Trainer
from .soul import Soul, SoulBrain

__version__ = '0.1.0'
__all__ = [
    'Config', 'ModelConfig', 'ObsConfig', 'TrainConfig',
    'ObsEncoder', 'stable_hash', 'squash',
    'SoulModel',
    'Episode', 'ReplayBuffer', 'WorkingMemory', 'ImportantMemory', 'SkillMemory', 'MemoryManager',
    'Trainer',
    'Soul', 'SoulBrain',
]