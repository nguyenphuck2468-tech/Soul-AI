"""Run the Phase B expert feasibility check and append raw JSONL outcomes."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from minicraft import MiniCraftEnv, MiniCraftExpert


TASKS = ("T_wood", "T_tools", "T_iron")


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True,
    ).strip()


def run(seed_count: int, output: Path) -> None:
    if seed_count < 1 or seed_count > 1000:
        raise ValueError("seed_count must be between 1 and 1000 (train split)")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite result file: {output}")

    output.parent.mkdir(parents=True, exist_ok=True)
    commit = _git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        for task in TASKS:
            successes = 0
            for seed in range(seed_count):
                env = MiniCraftEnv(seed=seed)
                expert = MiniCraftExpert(task)
                success = False
                while not env.done:
                    _, _, _, info = env.step(expert.expert_action(env.observe(), env))
                    if info["task_success"][task]:
                        success = True
                        successes += 1
                        break
                record = {
                    "task_id": "B6",
                    "git_commit": commit,
                    "seed": seed,
                    "config_path": "configs/seeds.yaml",
                    "metric": "expert_task_success",
                    "value": int(success),
                    "extra": {
                        "task": task,
                        "ticks": env.tick,
                        "death_before_success": bool(env.death and not success),
                        "successes_so_far": successes,
                        "timestamp_utc": timestamp,
                    },
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
                stream.flush()
            print(f"{task}: {successes}/{seed_count} ({successes / seed_count:.1%})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.seeds, args.out)


if __name__ == "__main__":
    main()
