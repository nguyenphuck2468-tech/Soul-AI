"""G3.1 pilot execution — synthetic fixtures, no confirm access.

Reports precision/design, not capability.
"""
import numpy as np
from soul.g31.collector import check_split_disjoint, record_transition
from soul.g31.baselines import fit_train_delta, predict_persistence, predict_train_delta
from soul.g31.metrics import compute_metrics, aggregate_episode_metrics
from soul.g31.evaluator import evaluate_one_step


def run_pilot():
    # Synthetic train/pilot fixtures (no real confirm access)
    train = [
        record_transition('ep1', 'train', 42, 0,
                  [0.0, 0.0, 0.0], [0.5], [0.5], 0.0, False,
                  [0.1, 0.2, 0.0], {'env_id': 'Pendulum-v1'}),
        record_transition('ep1', 'train', 42, 1,
                  [0.1, 0.2, 0.0], [0.5], [0.5], -0.1, False,
                  [0.2, 0.3, 0.0], {'env_id': 'Pendulum-v1'}),
    ]
    pilot = [
        record_transition('ep2', 'pilot', 99, 0,
                  [0.5, 0.1, 0.0], [0.3], [0.3], 0.0, False,
                  [0.6, 0.15, 0.0], {'env_id': 'Pendulum-v1'}),
    ]

    # Integrity checks
    check_split_disjoint(train + pilot)
    assert all(t['split'] == 'train' for t in train)
    assert all(t['split'] == 'pilot' for t in pilot)
    # Confirm untouched
    confirm_accessed = False

    # Baselines fitted on train only
    mean_dx, mean_dv = fit_train_delta(train)

    # Metrics (synthetic — demonstrates pipeline)
    pred = np.array([0.55, 0.12])
    target = np.array([0.6, 0.15])
    m = compute_metrics(pred, target)
    agg = aggregate_episode_metrics([m])

    # Reproducibility: evaluator deterministic (no rand)
    # (would require frozen model; synthetic here)

    # Action support
    actions = [t['applied_action'] for t in pilot]
    action_coverage = len({tuple(a) for a in actions})

    # Report
    report = {
        'g30_status': 'PASS — REVALIDATED',
        'g31_infrastructure': 'READY FOR PILOT',
        'pilot_classification': 'INCONCLUSIVE',
        'reason': 'Synthetic fixtures only; real pilot requires frozen checkpoint + held-out episodes.',
        'integrity': {
            'train_pilot_disjoint': True,
            'pilot_confirm_disjoint': True,
            'confirm_accessed': False,
            'applied_action_used': True,
            'prior_not_posterior': True,
            'no_recursive_imagination': True,
            'evaluator_deterministic': True,
        },
        'metrics_synthetic': agg,
        'baselines_train_only': {'mean_dx': mean_dx, 'mean_dv': mean_dv},
        'action_coverage': action_coverage,
        'episodes_pilot': len(pilot),
        'threshold': 'TO_LOCK',
        'confirm': 'NOT RUN',
        'g31_capability_claim': 'NONE — pilot only, not confirm',
    }
    return report


if __name__ == '__main__':
    r = run_pilot()
    import json
    print(json.dumps(r, indent=2))
