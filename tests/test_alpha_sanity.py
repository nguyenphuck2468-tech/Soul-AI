import subprocess

import pytest

from scripts import alpha_sanity


def test_execution_paths_use_explicit_longer_timeout(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs['timeout'])
        return subprocess.CompletedProcess(args[0], 0, '', '')

    monkeypatch.setattr(alpha_sanity.subprocess, 'run', fake_run)

    alpha_sanity.check_cli('scripts.phase0_alpha_run')
    alpha_sanity.check_cli('scripts.phase1_distill_skill')

    assert calls == [alpha_sanity.RUN_TIMEOUT_SECONDS] * 2


def test_cli_checks_keep_short_timeout(monkeypatch):
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs['timeout'])
        return subprocess.CompletedProcess(args[0], 0, '', '')

    monkeypatch.setattr(alpha_sanity.subprocess, 'run', fake_run)

    alpha_sanity.check_cli('scripts.phase1_compressed_skill', '--help')

    assert calls == [alpha_sanity.CLI_TIMEOUT_SECONDS]


def test_cli_failure_remains_explicit(monkeypatch):
    def fake_run(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 2, 'stdout', 'stderr')

    monkeypatch.setattr(alpha_sanity.subprocess, 'run', fake_run)

    with pytest.raises(RuntimeError, match='scripts.phase1_compressed_skill'):
        alpha_sanity.check_cli('scripts.phase1_compressed_skill', '--help')
