import hashlib
from pathlib import Path

import torch

from scripts.measure_cost import _hardware, _source_metadata


def test_cost_metadata_hashes_the_measured_dynamics_source():
    root = Path(__file__).resolve().parents[1]
    metadata = _source_metadata()
    expected = hashlib.sha256((root / 'soul/model.py').read_bytes()).hexdigest()

    assert metadata['source_file_sha256']['soul/model.py'] == expected


def test_cost_metadata_records_thread_and_affinity_configuration():
    metadata = _hardware(torch.device('cpu'))

    assert metadata['torch_num_threads'] == torch.get_num_threads()
    assert metadata['torch_num_interop_threads'] == torch.get_num_interop_threads()
    assert metadata['process_cpu_affinity'] is None or isinstance(
        metadata['process_cpu_affinity'], list)
