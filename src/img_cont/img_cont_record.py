"""Record deterministic rollouts of a raw-image PPO checkpoint."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import torch
from gymnasium.wrappers import RecordVideo

from src.agents.img_cont.ppo import PPOAgent
from src.envs.action_repeat import repeated_step
from src.envs.img_cont import make_img_cont_env
from src.envs.lap import lap_finished as is_lap_finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/img_cont_ppo/videos"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--until-lap-finished", action="store_true")
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--successful-videos", type=int, default=1)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")
    if args.successful_videos <= 0:
        raise ValueError("successful_videos must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.success_only:
        model_env = make_img_cont_env()
        try:
            agent = PPOAgent(model_env.action_space, model_env.observation_space.shape, device)
            metadata = agent.load(args.checkpoint, load_optimizer=False)
            saved_config = metadata.get("config", {})
            action_repeat = int(args.action_repeat or saved_config.get("action_repeat", 1))
            if action_repeat <= 0:
                raise ValueError("action_repeat must be positive")
            agent.actor_critic.eval()

            args.output_dir.mkdir(parents=True, exist_ok=True)
            episode = 0
            saved_videos = 0
            while saved_videos < args.successful_videos:
                # Failed attempts are created only in this temporary directory.
                with tempfile.TemporaryDirectory(prefix="img-cont-ppo-") as temporary_dir:
                    env = RecordVideo(
                        make_img_cont_env(render_mode="rgb_array"),
                        video_folder=temporary_dir,
                        episode_trigger=lambda _: True,
                        name_prefix="img-cont-ppo",
                    )
                    try:
                        image, _ = env.reset(seed=args.seed_start + episode)
                        reward_sum = 0.0
                        while True:
                            action, _, _ = agent.select_action(image, deterministic=True)
                            image, reward, terminated, truncated, info = repeated_step(
                                env, action, action_repeat
                            )
                            reward_sum += reward
                            if terminated or truncated:
                                finished_lap = is_lap_finished(env, terminated)
                                print(
                                    f"episode={episode} reward={reward_sum:.2f} "
                                    f"lap_finished={finished_lap}"
                                )
                                break
                    finally:
                        env.close()

                    if finished_lap:
                        video_paths = list(Path(temporary_dir).glob("*.mp4"))
                        if len(video_paths) != 1:
                            raise RuntimeError("Expected exactly one recorded video")
                        saved_videos += 1
                        destination = args.output_dir / (
                            f"img-cont-ppo-success-{saved_videos}-episode-{episode}.mp4"
                        )
                        shutil.move(str(video_paths[0]), destination)
                        print(f"Saved completed-lap video: {destination}")
                episode += 1
        finally:
            model_env.close()
        print(f"Videos saved in: {args.output_dir.resolve()}")
        return

    env = RecordVideo(make_img_cont_env(render_mode="rgb_array"), video_folder=str(args.output_dir), episode_trigger=lambda episode: args.until_lap_finished or episode < args.episodes, name_prefix="img-cont-ppo")
    try:
        agent = PPOAgent(env.action_space, env.observation_space.shape, device)
        metadata = agent.load(args.checkpoint, load_optimizer=False)
        saved_config = metadata.get("config", {})
        action_repeat = int(args.action_repeat or saved_config.get("action_repeat", 1))
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        agent.actor_critic.eval()
        episode = 0
        while args.until_lap_finished or episode < args.episodes:
            image, _ = env.reset(seed=args.seed_start + episode)
            reward_sum = 0.0
            while True:
                action, _, _ = agent.select_action(image, deterministic=True)
                image, reward, terminated, truncated, info = repeated_step(
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
    print(f"Videos saved in: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
