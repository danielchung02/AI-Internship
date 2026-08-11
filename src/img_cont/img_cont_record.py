"""Record deterministic rollouts of a raw-image PPO checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from gymnasium.wrappers import RecordVideo

from src.agents.img_cont.ppo import PPOAgent
from src.envs.img_cont import make_img_cont_env
from src.envs.lap import lap_finished as is_lap_finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/img_cont_ppo/videos"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--until-lap-finished", action="store_true")
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = RecordVideo(make_img_cont_env(render_mode="rgb_array"), video_folder=str(args.output_dir), episode_trigger=lambda episode: args.until_lap_finished or episode < args.episodes, name_prefix="img-cont-ppo")
    try:
        agent = PPOAgent(env.action_space, env.observation_space.shape, device)
        agent.load(args.checkpoint, load_optimizer=False)
        agent.actor_critic.eval()
        episode = 0
        while args.until_lap_finished or episode < args.episodes:
            image, _ = env.reset(seed=args.seed_start + episode)
            reward_sum = 0.0
            while True:
                action, _, _ = agent.select_action(image, deterministic=True)
                image, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
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
