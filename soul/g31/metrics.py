"""G3.1 metrics — per-episode then equal-weight aggregate.

No long-episode dominance; no invented threshold.
"""
import numpy as np


def compute_metrics(pred, target):
    diff = pred - target
    return {
        'mse': float(np.mean(diff**2)),
        'mae': float(np.mean(np.abs(diff))),
        'rmse': float(np.sqrt(np.mean(diff**2))),
        'bias': float(np.mean(diff)),
    }


def aggregate_episode_metrics(ep_metrics):
    # ep_metrics: list of dicts per episode
    keys = ['mse', 'mae', 'rmse', 'bias']
    out = {}
    for k in keys:
        vals = [m[k] for m in ep_metrics]
        out[k] = {
            'mean': float(np.mean(vals)),
            'median': float(np.median(vals)),
            'dispersion': float(np.std(vals)),
            'count': len(vals),
        }
    return out
