"""Separate Codespace scheduling delay from CPU time for the A5 workload."""

from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from soul.config import Config
from soul.soul import Soul
from wm_api import SoulAdapter


HORIZONS = (1, 2, 4, 8, 16, 32)
SEED = 0
REPS = 30
WARMUP = 5


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="results/A5_diag_run79_threadwall.jsonl",
        help="new JSONL output path; existing files are never overwritten",
    )
    args = parser.parse_args(argv)

    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cfg = Config()
    cfg.model.hidden_dim, cfg.model.latent_dim = 64, 16
    cfg.model.action_dim, cfg.model.discrete_dim = 1, 0
    cfg.train.batch_size, cfg.train.seq_len = 8, 16
    cfg.device = "cpu"
    soul = Soul(cfg)
    soul._h = torch.zeros(1, cfg.model.hidden_dim, device=soul.device)
    soul._z = torch.zeros(1, cfg.model.latent_dim, device=soul.device)
    adapter = SoulAdapter(soul)
    snapshot = adapter.snapshot()

    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for horizon in HORIZONS:
            actions = [0] * horizon
            fn = lambda: adapter.imagine(snapshot, actions)
            for _ in range(WARMUP):
                fn()
            for rep in range(REPS):
                wall_start = time.perf_counter()
                cpu_start = time.thread_time()
                fn()
                cpu_seconds = time.thread_time() - cpu_start
                wall_seconds = time.perf_counter() - wall_start
                record = {
                    "task_id": "A5-DIAG-CPU-WALL",
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "git_commit": revision,
                    "seed": SEED,
                    "config_path": "A5 settings: H in {1,2,4,8,16,32}; batch=1; hidden=64; latent=16; action_dim=1",
                    "horizon": horizon,
                    "rep": rep,
                    "metric": "wall_seconds",
                    "value": wall_seconds,
                    "thread_cpu_seconds": cpu_seconds,
                    "offcpu_gap_seconds": wall_seconds - cpu_seconds,
                    "torch_threads": torch.get_num_threads(),
                    "torch_interop_threads": torch.get_num_interop_threads(),
                }
                stream.write(json.dumps(record, allow_nan=False) + "\n")

    rows = [json.loads(line) for line in output.read_text().splitlines()]
    print(f"file={output} records={len(rows)} revision={revision}")
    for horizon in HORIZONS:
        group = [row for row in rows if row["horizon"] == horizon]
        wall = sorted(row["value"] for row in group)
        cpu = sorted(row["thread_cpu_seconds"] for row in group)
        offcpu = [row["offcpu_gap_seconds"] for row in group]
        wall_median = (wall[14] + wall[15]) / 2
        cpu_median = (cpu[14] + cpu[15]) / 2
        print(
            f"H={horizon:>2} wall_median={wall_median * 1000:.6f}ms "
            f"wall_maxdev={max(abs(x - wall_median) for x in wall) / wall_median * 100:.2f}% "
            f"cpu_median={cpu_median * 1000:.6f}ms "
            f"cpu_maxdev={max(abs(x - cpu_median) for x in cpu) / cpu_median * 100:.2f}% "
            f"max_offcpu_gap={max(offcpu) * 1000:.6f}ms "
            f"n={len(group)}"
        )


if __name__ == "__main__":
    main()
