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
CLI_TIMEOUT_SECONDS = 60
RUN_TIMEOUT_SECONDS = 180


def _reject_duplicate_json_keys(
        pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    value = json.loads(
        path.read_text(encoding='utf-8-sig'),
        object_pairs_hook=_reject_duplicate_json_keys)
    if not isinstance(value, dict):
        raise ValueError(f'JSON root must be an object: {path}')
    return value


def _sha256_file(path: Path) -> set[str]:
    """Return hashes for exact bytes and normalized text line endings."""
    raw = path.read_bytes()
    canonical = raw.replace(b'\r\n', b'\n')
    hashes = {
        hashlib.sha256(raw).hexdigest(),
        hashlib.sha256(canonical).hexdigest(),
        hashlib.sha256(canonical.replace(b'\n', b'\r\n')).hexdigest(),
    }
    return hashes


def _artifact_path(raw_path: str | Path) -> Path:
    """Resolve archived paths while preserving the recorded relative layout."""
    path = ROOT / Path(raw_path)
    if path.exists():
        return path
    normalized = str(raw_path).replace('\\', '/')
    prefix = 'work/phase1_skill_sweeps/skill-sweep-alpha-audit-20261011/'
    if normalized.startswith(prefix):
        archived = ROOT / 'results' / 'phase1_skill_sweep_20261011' / \
            normalized[len(prefix):]
        if archived.exists():
            return archived
    raise FileNotFoundError(f'archived artifact is missing: {raw_path}')


def check_cli(module: str, *arguments: str) -> None:
    timeout = CLI_TIMEOUT_SECONDS if '--help' in arguments else RUN_TIMEOUT_SECONDS
    try:
        result = subprocess.run(
            [sys.executable, '-m', module, *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        mode = 'help check' if '--help' in arguments else 'execution'
        raise RuntimeError(
            f'{module} {mode} timed out after {timeout} seconds') from exc
    if result.returncode != 0:
        mode = 'help check' if '--help' in arguments else 'execution'
        raise RuntimeError(
            f'{module} {mode} failed ({result.returncode}):\n'
            f'{result.stdout}\n{result.stderr}')


def check_phase1_archives() -> int:
    root = ROOT / 'results' / 'phase1_skill_sweep_20261011'
    sweep_path = root / 'compute_manifest.json'
    sweep = _read_json(sweep_path)
    if sweep.get('status') != 'EXPLORATORY_TRAIN_SEEDS_NO_PROMOTION':
        raise ValueError('phase1 sweep status is not exploratory/no-promotion')
    if sweep.get('evaluation_seeds') != list(range(200, 210)):
        raise ValueError('phase1 archived smoke seeds differ from documented 200..209')

    checked = 0
    for task, entry in sweep['tasks'].items():
        task_path = _artifact_path(entry['manifest_path'])
        manifest_bytes = task_path.read_bytes()
        if entry['manifest_sha256'] not in _sha256_file(task_path):
            raise ValueError(f'{task}: manifest hash mismatch')
        manifest = json.loads(
            manifest_bytes.decode('utf-8-sig'),
            object_pairs_hook=_reject_duplicate_json_keys)
        if not isinstance(manifest, dict):
            raise ValueError(f'{task}: manifest root is not an object')
        trace_path = _artifact_path(manifest['trace_path'])
        if manifest['trace_sha256'] not in _sha256_file(trace_path):
            raise ValueError(f'{task}: trace hash mismatch')
        if manifest.get('paired_success_outcomes') != entry.get('paired_success_outcomes'):
            raise ValueError(f'{task}: paired summary mismatch')
        if manifest.get('decision_compute') != entry.get('decision_compute'):
            raise ValueError(f'{task}: decision timing mismatch')
        checked += 1
    mapped_root = ROOT / 'results' / 'phase1_mapped_skill_20261011'
    mapped = _read_json(mapped_root / 'manifest.json')
    for task, entry in mapped['tasks'].items():
        task_manifest_path = _artifact_path(entry['manifest_path'])
        task_manifest = _read_json(task_manifest_path)
        trace_path = _artifact_path(task_manifest['trace_path'])
        if task_manifest['trace_sha256'] not in _sha256_file(trace_path):
            raise ValueError(f'{task}: mapped-skill trace hash mismatch')
        if task_manifest['mapped_success_fraction'] != 1.0:
            raise ValueError(f'{task}: archived mapped-skill smoke result mismatch')
        if task_manifest['one_step_expectation_match_fraction'] != 1.0:
            raise ValueError(f'{task}: archived expectation check mismatch')
        checked += 1

    distill_root = ROOT / 'results' / 'phase1_distillation_20261011'
    distilled = _read_json(distill_root / 'batch_manifest.json')
    for task, entry in distilled['tasks'].items():
        task_manifest_path = _artifact_path(entry['manifest_path'])
        manifest_bytes = task_manifest_path.read_bytes()
        if entry['manifest_sha256'] not in _sha256_file(task_manifest_path):
            raise ValueError(f'{task}: distillation manifest hash mismatch')
        task_manifest = json.loads(
            manifest_bytes.decode('utf-8-sig'),
            object_pairs_hook=_reject_duplicate_json_keys)
        if not isinstance(task_manifest, dict):
            raise ValueError(f'{task}: distillation manifest root is not an object')
        for path_key, hash_key in (
                ('skill_path', 'skill_sha256'),
                ('training_examples_path', 'training_examples_sha256')):
            artifact_path = _artifact_path(task_manifest[path_key])
            if task_manifest[hash_key] not in _sha256_file(artifact_path):
                raise ValueError(f'{task}: distillation {path_key} hash mismatch')
        checked += 1
    return checked


def main() -> None:
    check_cli('scripts.phase0_validate_protocol', '--help')
    check_cli('scripts.phase0_validate_protocol')
    check_cli('scripts.phase0_alpha_run')
    check_cli('scripts.phase1_skill_sweep')
    check_cli('scripts.phase1_mapped_skill')
    check_cli('scripts.phase1_distill_skill')
    check_cli('scripts.phase1_compressed_skill')
    checked = check_phase1_archives()
    print(f'Alpha sanity passed: 7 CLIs, {checked} Phase 1 task archives.')
    print('No simulator seeds were accessed; this is not a model-quality test.')


if __name__ == '__main__':
    main()
