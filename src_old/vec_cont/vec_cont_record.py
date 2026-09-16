"""Record deterministic rollouts of an engineered-vector PPO checkpoint."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers import RecordVideo

from src.agents.vec_cont.ppo import PPOAgent
from src.envs.action_repeat import repeated_step
from src.envs.lap import lap_finished as is_lap_finished
from src.envs.vec_cont import make_vec_cont_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/vec_cont_ppo/videos"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--until-lap-finished", action="store_true")
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Keep videos only for episodes that finish a lap.",
    )
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_env = make_vec_cont_env()
    try:
        state_dim = int(np.prod(model_env.observation_space.shape))
        agent = PPOAgent(state_dim, model_env.action_space, device)
        metadata = agent.load(args.checkpoint, load_optimizer=False)
        saved_config = metadata.get("config", {})
        action_repeat = int(args.action_repeat or saved_config.get("action_repeat", 1))
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        agent.actor_critic.eval()

        if args.success_only:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            episode = 0
            while args.until_lap_finished or episode < args.episodes:
                # A failed attempt is recorded only in this temporary directory,
                # which is deleted automatically when the episode ends.
                with tempfile.TemporaryDirectory(prefix="vec-cont-ppo-") as temporary_dir:
                    env = RecordVideo(
                        make_vec_cont_env(render_mode="rgb_array"),
                        video_folder=temporary_dir,
                        episode_trigger=lambda _: True,
                        name_prefix="vec-cont-ppo",
                    )
                    try:
                        state, _ = env.reset(seed=args.seed_start + episode)
                        reward_sum = 0.0
                        while True:
                            action, _, _ = agent.select_action(state, deterministic=True)
                            state, reward, terminated, truncated, info = repeated_step(
                                env, action, action_repeat
                            )
                            reward_sum += reward
                            if terminated or truncated:
                                lap_finished = is_lap_finished(env, terminated)
                                print(f"episode={episode} reward={reward_sum:.2f} lap_finished={lap_finished}")
                                break
                    finally:
                        env.close()

                    if lap_finished:
                        video_paths = list(Path(temporary_dir).glob("*.mp4"))
                        if len(video_paths) != 1:
                            raise RuntimeError("Expected exactly one recorded video")
                        destination = args.output_dir / f"vec-cont-ppo-success-episode-{episode}.mp4"
                        shutil.move(str(video_paths[0]), destination)
                        print(f"Saved completed-lap video: {destination}")

                if args.until_lap_finished and lap_finished:
                    break
                episode += 1
            return

        env = RecordVideo(
            make_vec_cont_env(render_mode="rgb_array"),
            video_folder=str(args.output_dir),
            episode_trigger=lambda episode: args.until_lap_finished or episode < args.episodes,
            name_prefix="vec-cont-ppo",
        )
        episode = 0
        try:
            while args.until_lap_finished or episode < args.episodes:
                state, _ = env.reset(seed=args.seed_start + episode)
                reward_sum = 0.0
                while True:
                    action, _, _ = agent.select_action(state, deterministic=True)
                    state, reward, terminated, truncated, info = repeated_step(
                        env, action, action_repeat
                    )
                    reward_sum += reward
                    if terminated or truncated:
                        lap_finished = is_lap_finished(env, terminated)
                        print(f"episode={episode} reward={reward_sum:.2f} lap_finished={lap_finished}")
                        break
                if args.until_lap_finished and lap_finished:
                    break
                episode += 1
        finally:
            env.close()
    finally:
        model_env.close()
    print(f"Videos saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
