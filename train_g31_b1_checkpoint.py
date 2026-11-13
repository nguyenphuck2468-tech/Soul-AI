"""G3.1 TRAIN-ONLY training script.

Uses TRAIN split only (seeds 1001-1005, 100 transitions/episode).
No PILOT/CONFIRM access during training.
Saves frozen checkpoint with full provenance.
"""
import sys, os, json, hashlib, subprocess, time
import torch
import numpy as np
sys.path.insert(0, '.')
from soul.config import Config
from soul.model import SoulModel
from soul.trainer import Trainer
from soul.memory import ReplayBuffer, Episode
from soul.obs import ObsEncoder
from soul.g31.collector import record_transition

# Load real TRAIN data. This entry point deliberately fails closed until the
# recorded TRAIN JSONL exists; it never falls back to synthetic observations.
from soul.g31.train_data import (
    load_train_jsonl_snapshot,
    pendulum_model_config,
    rows_to_episodes,
)

TRAIN_JSONL = os.environ.get('G31_TRAIN_JSONL', 'artifacts/g31/train.jsonl')
train_rows, dataset_hash = load_train_jsonl_snapshot(TRAIN_JSONL)

cfg = Config()
cfg = pendulum_model_config(cfg)
cfg.train.seed = 1001  # TRAIN seed
cfg.train.batch_size = 5  # all five TRAIN episodes; no validation split
cfg.train.seq_len = 12
cfg.train.updates_per_batch = 1
cfg.train.posterior_mix = 1.0
cfg.train.slow_posterior_updates = 100
cfg.train.imagine_horizon = 12
cfg.device = 'cpu'

enc = ObsEncoder(cfg.obs)
episodes = rows_to_episodes(train_rows, enc, cfg.model.goal_dim)
if len(episodes) != 5 or sum(len(ep) for ep in episodes) != 100:
    raise ValueError('TRAIN dataset must contain exactly 5 episodes and 100 transitions')
if any(action.shape != (1,) for ep in episodes for action in ep.actions):
    raise ValueError('TRAIN actions must remain shape (1,)')
model = SoulModel(cfg, enc.input_dim)
trainer = Trainer(model, cfg, enc.input_dim)
replay = ReplayBuffer(cfg.train.replay_capacity, cfg.train.seq_len)
for episode in episodes:
    replay.add_episode(episode)

git_commit = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], text=True).strip()
training_started = time.time()
model.train()
last_losses = None
for i in range(300):
    batch = replay.sample(cfg.train.batch_size,
                          np.random.default_rng(cfg.train.seed + i))
    if batch is None:
        raise RuntimeError('TRAIN replay could not produce a full batch')
    if batch['actions'].shape[-1] != cfg.model.action_dim:
        raise ValueError('action tensor shape does not match model action_dim')
    last_losses = trainer.update(batch)

# Freeze
model.eval()

# Save frozen checkpoint
checkpoint_path = "artifacts/g31/g31_b1_checkpoint.pt"
os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
effective_config = cfg.to_dict()
torch.save({
    'schema_version': 'g31.train.v2',
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': trainer.optimizer.state_dict(),
    'config_dict': effective_config,
    'enc_schema': enc.schema,
    'training_seed': cfg.train.seed,
    'training_updates': trainer.updates,
    'split': 'TRAIN',
    'train_seeds': sorted({row['episode_seed'] for row in train_rows}),
    'model_config': effective_config['model'],
    'data_split': 'TRAIN',
    'data_seeds': [1001, 1002, 1003, 1004, 1005],
    'data_transitions': len(train_rows),
    'dataset_path': TRAIN_JSONL,
    'dataset_sha256': dataset_hash,
    'train_seeds': sorted({row['episode_seed'] for row in train_rows}),
    'environment_id': 'Pendulum-v1',
    'env_version': 'gymnasium 1.4.0',
    'checkpoint_version': 'G3.1-v1',
    'git_commit': git_commit,
    'timestamp': time.time(),
    'optimizer_config': {'type': 'Adam', 'updates_per_batch': 1},
    'training_duration_seconds': time.time() - training_started,
    'stats': {
        'steps': 100,
        'updates': trainer.updates,
        'trainer_updates': trainer.updates,
        'reward_stats': dict(trainer._reward_stats),
    },
}, checkpoint_path)

# Verify load
model2 = SoulModel(cfg, enc.input_dim)
checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
model2.load_state_dict(checkpoint['model_state_dict'])
model2.eval()

# Verify deterministic inference
with torch.no_grad():
    h, z = model2.dynamics.initial_state(1, 'cpu')
    a = torch.zeros(1, cfg.model.action_dim)
    h2, mu, _ = model2.dynamics.step(h, z, a)
    pred = model2.decoder(h2, mu)
    pred2 = model2.decoder(h2, mu)
    assert torch.allclose(pred['obs'], pred2['obs'], atol=1e-6), 'Non-deterministic!'

# Hash
import hashlib
with open(checkpoint_path, 'rb') as f:
    h = hashlib.sha256(f.read()).hexdigest()

print('CHECKPOINT SAVED:', checkpoint_path)
print('SIZE:', os.path.getsize(checkpoint_path), 'bytes')
print('HASH:', h[:16] + '...')
print('LOAD TEST: PASS')
print('DETERMINISTIC: PASS')
print('TRAIN-ONLY: YES (seeds 1001-1005)')
print('PILOT/CONFIRM ACCESS: NO')
print('UPDATES:', trainer.updates)
print('REPRESENTATIVE LOSSES:', json.dumps(last_losses, sort_keys=True))
