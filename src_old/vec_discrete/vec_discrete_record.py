"""Record greedy rollouts of an engineered-vector CarRacing DQN checkpoint.

Run from ``car_racing``::

    python -m src.vec_discrete.vec_discrete_record --checkpoint outputs/engineered_vector_dqn/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from src.agents.vec_discrete.dqn import DQNAgent
from src.vec_discrete.vec_discrete_evaluate import repeated_step
from src.envs.vec_discrete import make_vector_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/videos"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument(
        "--until-lap-finished",
        action="store_true",
        help="Record every episode until one finishes a lap; ignores --episodes.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_vector_env(render_mode="rgb_array")
    env = RecordVideo(
        env,
        video_folder=str(args.output_dir),
        episode_trigger=lambda episode_id: (
            args.until_lap_finished or episode_id < args.episodes
        ),
        name_prefix="engineered-vector-dqn",
    )
    try:
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.n)
        agent = DQNAgent(state_dim, action_dim, device)
        metadata = agent.load(args.checkpoint, load_optimizer=False)
        saved_config = metadata.get("config", {})
        action_repeat = int(args.action_repeat or saved_config.get("action_repeat", 2))
        rng = np.random.default_rng(args.seed_start)
        agent.online_network.eval()

        episode = 0
        while args.until_lap_finished or episode < args.episodes:
            state, _ = env.reset(seed=args.seed_start + episode)
            reward_sum = 0.0
            while True:
                action = agent.select_action(state, epsilon=0.0, rng=rng)
                state, reward, terminated, truncated, info = repeated_step(
                    env, action, action_repeat
                )
                reward_sum += reward
                if terminated or truncated:
                    lap_finished = bool(info.get("lap_finished", False))
                    print(
                        f"episode={episode} reward={reward_sum:.2f} "
                        f"lap_finished={lap_finished}"
                    )
                    break
            if args.until_lap_finished and lap_finished:
                print("A completed lap was recorded; stopping.")
                break
            episode += 1
    finally:
        env.close()

    print(f"Videos saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
