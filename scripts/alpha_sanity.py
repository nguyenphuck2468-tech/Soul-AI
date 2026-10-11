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
        task_path = ROOT / entry['manifest_path']
        manifest_bytes = task_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != entry['manifest_sha256']:
            raise ValueError(f'{task}: manifest hash mismatch')
        manifest = json.loads(manifest_bytes.decode('utf-8-sig'))
        trace_path = ROOT / manifest['trace_path']
        if hashlib.sha256(trace_path.read_bytes()).hexdigest() != manifest['trace_sha256']:
            raise ValueError(f'{task}: trace hash mismatch')
        if manifest.get('paired_success_outcomes') != entry.get('paired_success_outcomes'):
            raise ValueError(f'{task}: paired summary mismatch')
        if manifest.get('decision_compute') != entry.get('decision_compute'):
            raise ValueError(f'{task}: decision timing mismatch')
        checked += 1
    return checked


def main() -> None:
    check_cli('scripts.phase0_alpha_run')
    check_cli('scripts.phase1_skill_sweep')
    checked = check_phase1_archives()
    print(f'Alpha sanity passed: 2 CLIs, {checked} Phase 1 task archives.')
    print('No simulator seeds were accessed; this is not a model-quality test.')


if __name__ == '__main__':
    main()
