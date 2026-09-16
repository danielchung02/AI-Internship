"""Draw four reward-vs-environment-frame curves from comparison CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_TITLES = {
    "vec_discrete": "1. Vector + Discrete (Double DQN)",
    "img_discrete": "2. Image + Discrete (Double DQN)",
    "vec_cont": "3. Vector + Continuous (PPO)",
    "img_cont": "4. Image + Continuous (PPO)",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-dir", type=Path, default=Path("src/compare"))
    parser.add_argument("--output", type=Path, default=Path("src/compare/four_way_comparison.png"))
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    for axis, method in zip(axes.flat, METHOD_TITLES):
        csv_path = args.compare_dir / method / "frame_score.csv"
        if not csv_path.exists():
            axis.set_title(f"{METHOD_TITLES[method]}\n(no frame_score.csv yet)")
            axis.axhline(900, color="tab:red", linestyle="--", linewidth=1)
            continue
        with csv_path.open(encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        frames = [int(row["environment_frames"]) for row in rows]
        rewards = [float(row["mean_reward"]) for row in rows]
        axis.plot(frames, rewards, marker="o", linewidth=1.5, markersize=4)
        axis.axhline(900, color="tab:red", linestyle="--", linewidth=1, label="900")
        axis.set_title(METHOD_TITLES[method])
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")

    for axis in axes[-1]:
        axis.set_xlabel("Environment frames")
    for axis in axes[:, 0]:
        axis.set_ylabel("100-episode mean reward")
    figure.suptitle("CarRacing: fixed-seed deterministic evaluation", fontsize=14)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(f"Saved graph: {args.output.resolve()}")


if __name__ == "__main__":
    main()
