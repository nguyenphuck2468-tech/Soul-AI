import hashlib
import json
import subprocess
import sys

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


def test_compressed_cli_validates_existing_archive(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'scripts.phase1_compressed_skill',
            '--validate-dir',
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    assert 'Validated compressed-skill manifest:' in result.stdout


def test_compressed_cli_rejects_missing_archive(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            '-m',
            'scripts.phase1_compressed_skill',
            '--validate-dir',
            str(tmp_path / 'missing'),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert 'invalid compressed-skill aggregate manifest' in result.stderr


def test_compressed_manifest_validator_rejects_extra_task(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['unexpected'] = {}
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='task set mismatch'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_fit_metric_drift(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['fit_metrics']['sample_count'] += 1
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='task/artifact fit metrics mismatch'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_bad_manifest_filename(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_path'] = 'wrong.json'
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='task manifest filename mismatch'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_manifest_path_escape(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_path'] = str(
        tmp_path / 'manifest.json')
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='task manifest escapes task archive'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_nested_manifest_path(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    nested = tmp_path / 'T_wood' / 'nested'
    nested.mkdir()
    original = tmp_path / 'T_wood' / 'manifest.json'
    nested_manifest = nested / 'manifest.json'
    nested_manifest.write_bytes(original.read_bytes())
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_path'] = str(nested_manifest)
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        nested_manifest.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='directly under task archive'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_duplicate_aggregate_keys(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    aggregate_path = tmp_path / 'manifest.json'
    raw = aggregate_path.read_text(encoding='utf-8')
    aggregate_path.write_text(
        raw.replace('"schema_version": 1,', '"schema_version": 1,\n'
                    '  "schema_version": 1,', 1),
        encoding='utf-8')

    with pytest.raises(ValueError, match='duplicate JSON key'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_duplicate_skill_keys(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    skill_path = tmp_path / 'T_wood' / 'skill.json'
    raw = skill_path.read_text(encoding='utf-8')
    skill_path.write_text(
        raw.replace('"status": "Candidate",', '"status": "Candidate",\n'
                    '  "status": "Candidate",', 1),
        encoding='utf-8')
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['skill_sha256'] = hashlib.sha256(
        skill_path.read_bytes()).hexdigest()
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='duplicate JSON key'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_limits_drift(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['limits'] = []
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='limits must be a non-empty list'):
        validate_manifest(tmp_path)


@pytest.mark.parametrize(
    'limits',
    [
        [],
        [''],
        ['duplicate', 'duplicate'],
        [1],
    ],
)
def test_compressed_manifest_validator_rejects_malformed_limits(
        tmp_path, limits):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['limits'] = limits
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='aggregate limits'):
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


def test_compressed_manifest_validator_rejects_seed_provenance_drift(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['episodes'][0]['seed'] = 401
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='episode seed provenance mismatch'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_nonfinite_runtime(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['runtime_seconds']['integrated'] = float('nan')
    task_path.write_text(
        json.dumps(task_manifest, indent=2, allow_nan=True) + '\n',
        encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='must be finite'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_skill_metadata_drift(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    skill_path = tmp_path / 'T_wood' / 'skill.json'
    skill = json.loads(skill_path.read_text(encoding='utf-8'))
    skill['task'] = 'T_tools'
    skill_path.write_text(
        json.dumps(skill, indent=2) + '\n', encoding='utf-8')
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['skill_sha256'] = hashlib.sha256(
        skill_path.read_bytes()).hexdigest()
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='skill artifact task mismatch'):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_invalid_tree_feature(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    skill_path = tmp_path / 'T_wood' / 'skill.json'
    skill = json.loads(skill_path.read_text(encoding='utf-8'))
    leaf = {
        'n': 1,
        'prediction': 0,
        'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'leaf': True,
    }
    skill['tree'] = {
        'n': 1,
        'prediction': 0,
        'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'leaf': False,
        'feature': skill['observation_feature_count'],
        'threshold': 0.0,
        'left': leaf,
        'right': leaf,
    }
    skill_path.write_text(
        json.dumps(skill, indent=2) + '\n', encoding='utf-8')
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['skill_sha256'] = hashlib.sha256(
        skill_path.read_bytes()).hexdigest()
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='outside observation schema'):
        validate_manifest(tmp_path)


@pytest.mark.parametrize(
    ('tree_mutation', 'message'),
    [
        (lambda tree: tree.update(prediction=9), 'prediction must be an action'),
        (lambda tree: tree.update(class_counts=[1] * 9), 'class_counts'),
        (lambda tree: tree.update(leaf='true'), 'leaf must be boolean'),
        (lambda tree: tree.update(threshold=float('nan')),
         'threshold must be finite'),
    ],
)
def test_tree_validator_rejects_malformed_nodes(tree_mutation, message):
    from scripts.phase1_compressed_skill import _validate_tree

    tree = {
        'n': 1,
        'prediction': 0,
        'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'leaf': False,
        'feature': 0,
        'threshold': 0.0,
        'left': {
            'n': 1,
            'prediction': 0,
            'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
            'leaf': True,
        },
        'right': {
            'n': 1,
            'prediction': 0,
            'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
            'leaf': True,
        },
    }
    tree_mutation(tree)

    with pytest.raises(ValueError, match=message):
        _validate_tree(tree, feature_count=1, label='tree')


def test_tree_validator_rejects_cycles_without_recursion_error():
    from scripts.phase1_compressed_skill import _validate_tree

    tree = {
        'n': 1,
        'prediction': 0,
        'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'leaf': False,
        'feature': 0,
        'threshold': 0.0,
    }
    tree['left'] = tree
    tree['right'] = {
        'n': 1,
        'prediction': 0,
        'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'leaf': True,
    }

    with pytest.raises(ValueError, match='contains a cycle'):
        _validate_tree(tree, feature_count=1, label='tree')


@pytest.mark.parametrize(
    ('mutation', 'message'),
    [
        (lambda tree: tree.update(
            n=3, class_counts=[3, 0, 0, 0, 0, 0, 0, 0, 0]),
         'n does not equal child totals'),
        (lambda tree: tree.update(class_counts=[1, 1, 0, 0, 0, 0, 0, 0, 0]),
         'class_counts do not equal child totals'),
        (lambda tree: tree.update(prediction=1),
         'prediction is not the majority action'),
    ],
)
def test_tree_validator_rejects_inconsistent_split_statistics(mutation, message):
    from scripts.phase1_compressed_skill import _validate_tree

    leaf = {
        'n': 1,
        'prediction': 0,
        'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
        'leaf': True,
    }
    tree = {
        'n': 2,
        'prediction': 0,
        'class_counts': [2, 0, 0, 0, 0, 0, 0, 0, 0],
        'leaf': False,
        'feature': 0,
        'threshold': 0.0,
        'left': leaf,
        'right': {
            'n': 1,
            'prediction': 0,
            'class_counts': [1, 0, 0, 0, 0, 0, 0, 0, 0],
            'leaf': True,
        },
    }
    mutation(tree)

    with pytest.raises(ValueError, match=message):
        _validate_tree(tree, feature_count=1, label='tree')


def test_compressed_manifest_validator_rejects_observation_schema_drift(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    skill_path = tmp_path / 'T_wood' / 'skill.json'
    skill = json.loads(skill_path.read_text(encoding='utf-8'))
    skill['observation_feature_count'] -= 1
    skill_path.write_text(
        json.dumps(skill, indent=2) + '\n', encoding='utf-8')
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['skill_sha256'] = hashlib.sha256(
        skill_path.read_bytes()).hexdigest()
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='invalid observation feature count'):
        validate_manifest(tmp_path)


@pytest.mark.parametrize(
    ('field', 'value', 'message'),
    [
        ('sample_count', 0, 'sample_count does not match tree'),
        ('leaf_count', 0, 'leaf_count does not match tree'),
        ('training_action_accuracy', float('inf'),
         'training_action_accuracy must be finite'),
    ],
)
def test_compressed_manifest_validator_rejects_fit_metric_drift(
        tmp_path, field, value, message):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    skill_path = tmp_path / 'T_wood' / 'skill.json'
    skill = json.loads(skill_path.read_text(encoding='utf-8'))
    skill['fit_metrics'][field] = value
    skill_path.write_text(
        json.dumps(skill, indent=2, allow_nan=True) + '\n',
        encoding='utf-8')
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['skill_sha256'] = hashlib.sha256(
        skill_path.read_bytes()).hexdigest()
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match=message):
        validate_manifest(tmp_path)


def test_compressed_manifest_validator_rejects_episode_integrated_drift(tmp_path):
    from scripts.phase1_compressed_skill import run

    run(tmp_path, train_seeds=[20], eval_seeds=[400], max_ticks=2)
    task_path = tmp_path / 'T_wood' / 'manifest.json'
    task_manifest = json.loads(task_path.read_text(encoding='utf-8'))
    task_manifest['episodes'][0]['runtime_seconds']['integrated'] = 0.0
    task_path.write_text(
        json.dumps(task_manifest, indent=2) + '\n', encoding='utf-8')
    aggregate_path = tmp_path / 'manifest.json'
    aggregate = json.loads(aggregate_path.read_text(encoding='utf-8'))
    aggregate['tasks']['T_wood']['manifest_sha256'] = hashlib.sha256(
        task_path.read_bytes()).hexdigest()
    aggregate_path.write_text(
        json.dumps(aggregate, indent=2) + '\n', encoding='utf-8')

    with pytest.raises(ValueError, match='episode integrated runtime mismatch'):
        validate_manifest(tmp_path)
