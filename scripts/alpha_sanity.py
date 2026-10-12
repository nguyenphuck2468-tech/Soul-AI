"""Offline sanity checks for the documented Phase 0/1 alpha entry points.

This validates CLI startup and archived metadata only. It does not instantiate
the simulator, access seeds, or make quality claims.
"""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _artifact_path(raw_path: str | Path) -> Path:
    """Resolve archived paths while preserving the recorded relative layout."""
    path = ROOT / Path(raw_path)
    if path.exists():
        return path
    normalized = Path(raw_path).as_posix()
    prefix = 'work/phase1_skill_sweeps/skill-sweep-alpha-audit-20261011/'
    if normalized.startswith(prefix):
        archived = ROOT / 'results' / 'phase1_skill_sweep_20261011' / \
            normalized[len(prefix):]
        if archived.exists():
            return archived
    raise FileNotFoundError(f'archived artifact is missing: {raw_path}')


def check_cli(module: str) -> None:
    result = subprocess.run(
        [sys.executable, '-m', module, '--help'],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f'{module} --help failed ({result.returncode}):\n'
            f'{result.stdout}\n{result.stderr}')


def check_phase1_archives() -> int:
    root = ROOT / 'results' / 'phase1_skill_sweep_20261011'
    sweep_path = root / 'compute_manifest.json'
    sweep = json.loads(sweep_path.read_text(encoding='utf-8-sig'))
    if sweep.get('status') != 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION':
        raise ValueError('phase1 sweep status is not exploratory/no-promotion')
    if sweep.get('evaluation_seeds') != list(range(200, 210)):
        raise ValueError('phase1 archived smoke seeds differ from documented 200..209')

    checked = 0
    for task, entry in sweep['tasks'].items():
        task_path = _artifact_path(entry['manifest_path'])
        manifest_bytes = task_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != entry['manifest_sha256']:
            raise ValueError(f'{task}: manifest hash mismatch')
        manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
        trace_path = _artifact_path(manifest['trace_path'])
        if hashlib.sha256(trace_path.read_bytes()).hexdigest() != manifest['trace_sha256']:
            raise ValueError(f'{task}: trace hash mismatch')
        if manifest.get('paired_success_outcomes') != entry.get('paired_success_outcomes'):
            raise ValueError(f'{task}: paired summary mismatch')
        if manifest.get('decision_compute') != entry.get('decision_compute'):
            raise ValueError(f'{task}: decision timing mismatch')
        checked += 1
    mapped_root = ROOT / 'results' / 'phase1_mapped_skill_20261011'
    mapped = json.loads((mapped_root / 'manifest.json').read_text(encoding='utf-8-sig'))
    for task, entry in mapped['tasks'].items():
        task_manifest_path = _artifact_path(entry['manifest_path'])
        task_manifest = json.loads(task_manifest_path.read_text(encoding='utf-8-sig'))
        trace_path = _artifact_path(task_manifest['trace_path'])
        if hashlib.sha256(trace_path.read_bytes()).hexdigest() != task_manifest['trace_sha256']:
            raise ValueError(f'{task}: mapped-skill trace hash mismatch')
        if task_manifest['mapped_success_fraction'] != 1.0:
            raise ValueError(f'{task}: archived mapped-skill smoke result mismatch')
        if task_manifest['one_step_expectation_match_fraction'] != 1.0:
            raise ValueError(f'{task}: archived expectation check mismatch')
        checked += 1

    distill_root = ROOT / 'results' / 'phase1_distillation_20261011'
    distilled = json.loads(
        (distill_root / 'batch_manifest.json').read_text(encoding='utf-8-sig'))
    for task, entry in distilled['tasks'].items():
        task_manifest_path = _artifact_path(entry['manifest_path'])
        manifest_bytes = task_manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != entry['manifest_sha256']:
            raise ValueError(f'{task}: distillation manifest hash mismatch')
        task_manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
        for path_key, hash_key in (
                ('skill_path', 'skill_sha256'),
                ('training_examples_path', 'training_examples_sha256')):
            artifact_path = _artifact_path(task_manifest[path_key])
            if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != task_manifest[hash_key]:
                raise ValueError(f'{task}: distillation {path_key} hash mismatch')
        checked += 1
    return checked


def main() -> None:
    check_cli('scripts.phase0_validate_protocol')
    check_cli('scripts.phase0_alpha_run')
    check_cli('scripts.phase1_skill_sweep')
    check_cli('scripts.phase1_mapped_skill')
    check_cli('scripts.phase1_distill_skill')
    check_cli('scripts.phase1_compressed_skill')
    checked = check_phase1_archives()
    print(f'Alpha sanity passed: 6 CLIs, {checked} Phase 1 task archives.')
    print('No simulator seeds were accessed; this is not a model-quality test.')


if __name__ == '__main__':
    main()
