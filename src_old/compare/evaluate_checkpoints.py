"""Evaluate every numbered checkpoint and export a frame-score CSV.

Example:
    python -m src.compare.evaluate_checkpoints --method img_cont \
        --checkpoint-dir src/compare/img_cont/checkpoints \
        --output-csv src/compare/img_cont/frame_score.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


EVALUATOR_MODULES = {
    "vec_discrete": "src.vec_discrete.vec_discrete_evaluate",
    "img_discrete": "src.img_discrete.img_discrete_evaluate",
    "vec_cont": "src.vec_cont.vec_cont_evaluate",
    "img_cont": "src.img_cont.img_cont_evaluate",
}


def checkpoint_step(path: Path) -> int:
    """Read the agent-decision count saved in a checkpoint."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return int(data.get("metadata", {}).get("global_step", 0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(EVALUATOR_MODULES))
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    checkpoints = list(args.checkpoint_dir.glob("step_*.pt"))
    threshold_checkpoint = args.checkpoint_dir / "threshold_reached.pt"
    if threshold_checkpoint.exists():
        checkpoints.append(threshold_checkpoint)
    checkpoints = sorted(checkpoints, key=checkpoint_step)
    if not checkpoints:
        raise FileNotFoundError(f"No step_*.pt checkpoints in {args.checkpoint_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method", "checkpoint", "agent_steps", "action_repeat",
        "environment_frames", "episodes", "seed_start", "mean_reward",
        "std_reward", "min_reward", "max_reward", "completed_laps",
        "completion_rate",
    ]
    rows: list[dict[str, object]] = []
    for checkpoint in checkpoints:
        with tempfile.TemporaryDirectory(prefix="checkpoint-evaluation-") as directory:
            result_path = Path(directory) / "result.json"
            command = [
                sys.executable, "-m", EVALUATOR_MODULES[args.method],
                "--checkpoint", str(checkpoint),
                "--episodes", str(args.episodes),
                "--seed-start", str(args.seed_start),
                "--output-json", str(result_path),
            ]
            print(f"Evaluating {checkpoint.name} ({args.episodes} episodes)...", flush=True)
            subprocess.run(command, check=True)
            result = json.loads(result_path.read_text(encoding="utf-8"))

        agent_steps = checkpoint_step(checkpoint)
        repeat = int(result.get("action_repeat", 1))
        rows.append({
            "method": args.method,
            "checkpoint": checkpoint.name,
            "agent_steps": agent_steps,
            "action_repeat": repeat,
            "environment_frames": agent_steps * repeat,
            "episodes": int(result["episodes"]),
            "seed_start": int(result["seed_start"]),
            "mean_reward": float(result["mean_reward"]),
            "std_reward": float(result["std_reward"]),
            "min_reward": float(result["min_reward"]),
            "max_reward": float(result["max_reward"]),
            "completed_laps": int(result["completed_laps"]),
            "completion_rate": float(result["completion_rate"]),
        })

    with args.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv.resolve()}")


if __name__ == "__main__":
    main()
