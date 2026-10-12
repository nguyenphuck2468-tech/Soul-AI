import hashlib
import json

import numpy as np
import pytest

from minicraft import MiniCraftEnv
from minicraft.compressed_skill import CompressedObservationSkill
from minicraft.distilled_skill import fit_decision_tree, observation_features
from minicraft.runtime import CompressedSkillRuntime
from scripts.phase1_compressed_skill import _episode, validate_manifest


def test_compressed_skill_uses_mapped_fallback_on_disagreement():
    observation = MiniCraftEnv(seed=400).observe()
    features = observation_features(observation)
    tree, _ = fit_decision_tree(
        np.zeros((4, features.size), dtype=np.float32),
        np.full(4, 8, dtype=np.int64),
        min_samples_leaf=1,
    )
    skill = CompressedObservationSkill('T_wood', {
        'schema_version': 1,
        'task': 'T_wood',
        'tree': tree,
    })

    proposal = skill.propose(observation)

    assert proposal.source == 'mapped_fallback'
    assert proposal.candidate_action == proposal.fallback_action
    assert proposal.learned_action == 8


def test_compressed_episode_records_integrated_runtime_components():
    observation = MiniCraftEnv(seed=400).observe()
    features = observation_features(observation)
    tree, _ = fit_decision_tree(
        np.zeros((4, features.size), dtype=np.float32),
        np.full(4, 8, dtype=np.int64),
        min_samples_leaf=1,
    )
    result = _episode('T_wood', 400, {
        'schema_version': 1,
        'task': 'T_wood',
        'tree': tree,
    }, max_ticks=2)

    runtime = result['runtime_seconds']
    assert set(runtime) == {
        'skill_proposal', 'safety_filter', 'environment_step',
        'expectation_monitor', 'integrated'}
    assert runtime['integrated'] >= runtime['skill_proposal']
    assert runtime['integrated'] >= runtime['safety_filter']
    assert runtime['integrated'] >= runtime['environment_step']
    assert runtime['integrated'] >= runtime['expectation_monitor']


def test_runtime_records_safety_and_expectation_diagnostics():
    observation = MiniCraftEnv(seed=400).observe()
    features = observation_features(observation)
    tree, _ = fit_decision_tree(
        np.zeros((4, features.size), dtype=np.float32),
        np.full(4, 8, dtype=np.int64),
        min_samples_leaf=1,
    )
    result = CompressedSkillRuntime(
        'T_wood', {'schema_version': 1, 'task': 'T_wood', 'tree': tree},
        max_ticks=2).run_episode(400)

    assert result['steps'] == 2
    assert result['safety_blocks'] >= 0
    assert result['expectation_checks'] >= 0
    assert result['expectation_mismatches'] >= 0
    assert set(result['runtime_seconds']) == {
        'skill_proposal', 'safety_filter', 'environment_step',
        'expectation_monitor', 'integrated'}


def test_compressed_manifest_validator_rejects_tampered_skill(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    skill_path = tmp_path / 'T_wood' / 'skill.json'
    skill_path.write_text(skill_path.read_text(encoding='utf-8') + 'tampered\n',
                          encoding='utf-8')

    with pytest.raises(ValueError, match='skill artifact hash mismatch'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_inconsistent_episode_counters(
        tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['episodes'][0]['mapped_fallback_steps'] += 1
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='proposal source totals mismatch'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_aggregate_summary_drift(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['success_fraction'] = 1.0
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='aggregate success summary mismatch'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_missing_required_field(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    del task_manifest['expectation_checks']
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='task manifest field is required'):
        validate_manifest(tmp_path)
