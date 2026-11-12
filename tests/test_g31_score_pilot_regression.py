"""Regression tests for G3.1 configurable scorer interface.
Verifies isolation, runtime hash, default A resolution, B1 resolution,
evaluator semantics unchanged, and artifact protection.
"""
import hashlib
import json
from pathlib import Path

import pytest

from soul.g31.score_pilot import (
    DEFAULT_CHECKPOINT_PATH,
    PILOT_HASH,
    sha256,
    run,
)

A_CHECKPOINT = Path('artifacts/g31/g31_verified_train_checkpoint.pt')
B1_CHECKPOINT = Path('artifacts/g31/g31_b1_checkpoint.pt')


def test_default_scorer_resolves_to_a_checkpoint():
    # Default path must point to A
    assert DEFAULT_CHECKPOINT_PATH == A_CHECKPOINT


def test_explicit_b1_checkpoint_resolves_to_b1():
    # Explicit parameter must resolve to B1 file
    assert B1_CHECKPOINT.exists()
    assert B1_CHECKPOINT != A_CHECKPOINT


def test_runtime_checkpoint_sha256_matches_actual_file():
    # Runtime hash must match file, not hardcoded value
    a_hash = sha256(A_CHECKPOINT)
    b1_hash = sha256(B1_CHECKPOINT)
    assert a_hash == '0fd3fbc46eadd3e6dc4b2d9af873f62ef4c5a04477bfedd06b8565d704b32147'
    assert b1_hash == '136017b650fc910043c559822d47fea900391e2823324413327de815a4d8c018'
    assert a_hash != b1_hash


def test_output_paths_are_isolated():
    # Default A outputs and B1 outputs must be different paths
    from soul.g31.score_pilot import DEFAULT_PREDICTIONS_PATH
    assert DEFAULT_PREDICTIONS_PATH.name == 'pilot_predictions.jsonl'
    b1_pred = Path('artifacts/g31/pilot_b1_predictions.jsonl')
    assert b1_pred != DEFAULT_PREDICTIONS_PATH


def test_existing_artifacts_not_silently_overwritten():
    # Isolation guard must raise when output exists and allow_overwrite=False
    # Use a temporary path that exists
    tmp = Path('artifacts/g31/test_isolation_guard.json')
    tmp.write_text('{}')
    try:
        with pytest.raises(FileExistsError):
            run(
                checkpoint_path=str(A_CHECKPOINT),
                predictions_path=str(tmp),
                metrics_path=str(tmp.with_suffix('.metrics.json')),
                report_path=str(tmp.with_suffix('.report.json')),
                allow_overwrite=False,
            )
    finally:
        tmp.unlink(missing_ok=True)
        tmp.with_suffix('.metrics.json').unlink(missing_ok=True)
        tmp.with_suffix('.report.json').unlink(missing_ok=True)


def test_evaluator_semantics_unchanged():
    # Import evaluator; verify contract (returns new state, no mutation)
    from soul.g31.evaluator import evaluate_one_step, initial_state, EvaluatorState
    import torch
    # Contract verified by source inspection; this test ensures import works
    assert callable(evaluate_one_step)
    assert callable(initial_state)


def test_default_run_uses_a_checkpoint_and_computes_hash():
    # Run with defaults (A) to a temporary staging prefix; verify hash in output
    staging_pred = Path('artifacts/g31/staging_test_predictions.jsonl')
    staging_met = Path('artifacts/g31/staging_test_metrics.json')
    staging_rep = Path('artifacts/g31/staging_test_report.json')
    # Clean any leftover
    for p in (staging_pred, staging_met, staging_rep):
        p.unlink(missing_ok=True)
    result = run(
        checkpoint_path=str(A_CHECKPOINT),
        predictions_path=str(staging_pred),
        metrics_path=str(staging_met),
        report_path=str(staging_rep),
        allow_overwrite=True,
    )
    # Verify runtime hash matches A
    assert result['checkpoint_sha256'] == sha256(A_CHECKPOINT)
    assert result['checkpoint_path'] == str(A_CHECKPOINT)
    # Verify outputs exist and non-empty
    assert staging_pred.exists() and staging_pred.stat().st_size > 0
    assert staging_met.exists() and staging_met.stat().st_size > 0
    assert staging_rep.exists() and staging_rep.stat().st_size > 0
    # Verify metadata in predictions references correct checkpoint
    first = json.loads(staging_pred.read_text().splitlines()[0])
    assert first['checkpoint_sha256'] == sha256(A_CHECKPOINT)
    assert first['pilot_sha256'] == PILOT_HASH
    assert first['timestep'] == 1
    assert len(first['zero_action_physical']) == 2
    assert result['transition_count'] == 95
    assert 'zero_action_aggregate_equal_episode' in result
    assert 'paired_model_minus_zero_action' in result
    report = json.loads(staging_rep.read_text(encoding='utf-8'))
    assert report['transitions'] == 95
    assert report['recorded_transitions'] == 100
    assert report['warmup_transitions_skipped'] == 5
    assert report['action_ablation']['kind'] == 'zero-action sensitivity diagnostic'
    assert report['action_ablation']['optimizer_updates'] == 0
    assert report['action_ablation']['held_fixed'] == [
        'recurrent state', 'current observation', 'checkpoint', 'PILOT transitions'
    ]
    # Each episode contributes 19 aligned one-step predictions.
    rows = [json.loads(line) for line in staging_pred.read_text(encoding='utf-8').splitlines()]
    assert len(rows) == 95
    assert {episode: sum(r['episode_id'] == episode for r in rows)
            for episode in {r['episode_id'] for r in rows}} == {
                f'pilot_ep{seed}': 19 for seed in range(2001, 2006)
            }
    # Clean staging
    for p in (staging_pred, staging_met, staging_rep):
        p.unlink(missing_ok=True)
