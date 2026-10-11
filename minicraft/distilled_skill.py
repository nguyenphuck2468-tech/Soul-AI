"""Small observation-only decision-tree skills distilled from expert episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


_EVENT_CODES = {
    'none': 0, 'ok': 1, 'blocked': 2, 'no_tool': 3,
    'hit': 4, 'crafted': 5, 'ate': 6,
}
_TASK_GOALS = {
    'T_wood': {'field': 'inv.wood', 'operator': '>=', 'value': 5},
    'T_tools': {'field': 'inv.stone_pickaxe', 'operator': '>=', 'value': 1},
    'T_iron': {'field': 'inv.iron', 'operator': '>=', 'value': 1},
}


def observation_features(observation: dict[str, Any]) -> np.ndarray:
    """Flatten only the published observation; never reads privileged state."""
    actor = observation['self']
    inventory = observation['inv']
    features = [
        actor['x'], actor['y'], actor['facing'], actor['health'], actor['food'],
    ]
    features.extend(inventory[key] for key in (
        'wood', 'stone', 'iron', 'apple', 'wood_pickaxe', 'stone_pickaxe'))
    patch = np.asarray(observation['patch'], dtype=np.float32)
    if patch.shape != (7, 7):
        raise ValueError('observation patch must have shape (7, 7)')
    features.extend(patch.reshape(-1).tolist())
    mobs = list(observation.get('mobs', ()))[:3]
    features.append(len(mobs))
    for index in range(3):
        if index < len(mobs):
            mob = mobs[index]
            features.extend((mob['dx'], mob['dy'], mob['hp']))
        else:
            features.extend((0, 0, 0))
    events = observation.get('events', {})
    features.extend((
        _EVENT_CODES.get(events.get('last_result', 'none'), 0),
        events.get('damage', 0),
    ))
    return np.asarray(features, dtype=np.float32)


def _gini(counts: np.ndarray) -> float:
    total = int(counts.sum())
    if total == 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(1.0 - np.square(probabilities).sum())


def fit_decision_tree(features: np.ndarray, labels: np.ndarray, *,
                      max_depth: int = 8, min_samples_leaf: int = 8,
                      max_thresholds: int = 16) -> tuple[dict, dict]:
    """Fit a compact deterministic CART classifier with bounded quantile splits."""
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    if x.ndim != 2 or y.ndim != 1 or len(x) != len(y) or len(y) == 0:
        raise ValueError('features and labels must be non-empty aligned arrays')
    if not np.isfinite(x).all() or np.any((y < 0) | (y > 8)):
        raise ValueError('training data must be finite with action labels 0..8')
    if max_depth < 1 or min_samples_leaf < 1 or max_thresholds < 2:
        raise ValueError('invalid tree size parameters')

    def grow(indices: np.ndarray, depth: int) -> dict:
        node_labels = y[indices]
        counts = np.bincount(node_labels, minlength=9)
        prediction = int(np.argmax(counts))
        parent_impurity = _gini(counts)
        node = {
            'n': int(len(indices)),
            'prediction': prediction,
            'class_counts': counts.tolist(),
        }
        if depth >= max_depth or len(indices) < 2 * min_samples_leaf or parent_impurity == 0:
            node['leaf'] = True
            return node

        best_gain = 0.0
        best_split = None
        current = x[indices]
        for feature_index in range(x.shape[1]):
            values = current[:, feature_index]
            unique = np.unique(values)
            if len(unique) < 2:
                continue
            if len(unique) > max_thresholds + 1:
                quantiles = np.linspace(0.0, 1.0, max_thresholds + 2)[1:-1]
                candidates = np.unique(np.quantile(unique, quantiles))
            else:
                candidates = (unique[:-1] + unique[1:]) / 2.0
            for threshold in candidates:
                left_mask = values <= threshold
                left_n = int(left_mask.sum())
                right_n = len(indices) - left_n
                if left_n < min_samples_leaf or right_n < min_samples_leaf:
                    continue
                left_counts = np.bincount(node_labels[left_mask], minlength=9)
                right_counts = counts - left_counts
                gain = parent_impurity - (
                    left_n * _gini(left_counts) + right_n * _gini(right_counts)
                ) / len(indices)
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_split = (feature_index, float(threshold), left_mask)

        if best_split is None:
            node['leaf'] = True
            return node
        feature_index, threshold, left_mask = best_split
        node.update({
            'leaf': False,
            'feature': feature_index,
            'threshold': threshold,
            'left': grow(indices[left_mask], depth + 1),
            'right': grow(indices[~left_mask], depth + 1),
        })
        return node

    tree = grow(np.arange(len(y)), 0)
    predictions = np.asarray([predict_tree(tree, row) for row in x], dtype=np.int64)
    leaf_count = 0
    max_observed_depth = 0

    def inspect(node: dict, depth: int = 0) -> None:
        nonlocal leaf_count, max_observed_depth
        max_observed_depth = max(max_observed_depth, depth)
        if node['leaf']:
            leaf_count += 1
        else:
            inspect(node['left'], depth + 1)
            inspect(node['right'], depth + 1)

    inspect(tree)
    metrics = {
        'sample_count': int(len(y)),
        'training_action_accuracy': float(np.mean(predictions == y)),
        'node_count': 2 * leaf_count - 1,
        'leaf_count': leaf_count,
        'depth': max_observed_depth,
        'max_depth_config': max_depth,
        'min_samples_leaf': min_samples_leaf,
    }
    return tree, metrics


def predict_tree(tree: dict, features: np.ndarray) -> int:
    row = np.asarray(features, dtype=np.float32)
    node = tree
    while not node['leaf']:
        node = node['left'] if row[node['feature']] <= node['threshold'] else node['right']
    return int(node['prediction'])


@dataclass(frozen=True)
class DistilledSkillProposal:
    skill_id: str
    version: str
    task: str
    candidate_action: int
    expected_outcome: dict[str, Any]
    confidence_status: str
    skill_status: str
    leaf_samples: int


class DistilledObservationSkill:
    """Run a trained policy tree using observations only."""

    def __init__(self, task: str, artifact: dict[str, Any]):
        if task not in _TASK_GOALS:
            raise ValueError(f'task must be one of {sorted(_TASK_GOALS)}')
        if artifact.get('task') != task or artifact.get('schema_version') != 1:
            raise ValueError('skill artifact task/schema mismatch')
        self.task = task
        self.tree = artifact['tree']
        self.status = 'Candidate'
        self.version = artifact.get('version', '0.1.0-alpha-distilled')

    def propose(self, observation: dict[str, Any]) -> DistilledSkillProposal:
        features = observation_features(observation)
        node = self.tree
        while not node['leaf']:
            node = (node['left'] if features[node['feature']] <= node['threshold']
                    else node['right'])
        return DistilledSkillProposal(
            skill_id=f'minicraft.distilled.{self.task}',
            version=self.version,
            task=self.task,
            candidate_action=int(node['prediction']),
            expected_outcome={
                'scope': 'episode_goal',
                'condition': dict(_TASK_GOALS[self.task]),
            },
            confidence_status='UNCALIBRATED_LEAF_SUPPORT',
            skill_status=self.status,
            leaf_samples=int(node['n']),
        )
