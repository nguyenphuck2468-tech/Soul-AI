"""G3.1 baselines — train-only mean delta + persistence.

No fitting on pilot/confirm.
"""
import numpy as np


def fit_train_delta(train_transitions):
    dx = [t['raw_observation_after'][0] - t['raw_observation_before'][0]
          for t in train_transitions]
    dv = [t['raw_observation_after'][1] - t['raw_observation_before'][1]
          for t in train_transitions]
    return float(np.mean(dx)), float(np.mean(dv))


def predict_persistence(obs_current):
    return obs_current.copy()


def predict_train_delta(obs_current, mean_dx, mean_dv):
    pred = obs_current.copy()
    pred[0] += mean_dx
    pred[1] += mean_dv
    return pred
