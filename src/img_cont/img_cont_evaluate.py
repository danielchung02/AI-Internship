"""Evaluate a raw-image, continuous-action PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.agents.img_cont.ppo import PPOAgent
from src.envs.img_cont import make_img_cont_env
from src.envs.lap import lap_finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_img_cont_env()
    try:
        agent = PPOAgent(env.action_space, env.observation_space.shape, device)
        agent.load(args.checkpoint, load_optimizer=False)
        agent.actor_critic.eval()
        rewards: list[float] = []
        completed_laps = 0
        for episode in range(args.episodes):
            image, _ = env.reset(seed=args.seed_start + episode)
            reward_sum = 0.0
            while True:
                action, _, _ = agent.select_action(image, deterministic=True)
                image, reward, terminated, truncated, info = env.step(action)
                reward_sum += float(reward)
                if terminated or truncated:
                    completed_laps += int(lap_finished(env, terminated))
                    break
            rewards.append(reward_sum)
    finally:
        env.close()

    values = np.asarray(rewards, dtype=np.float64)
    result = {"checkpoint": str(args.checkpoint), "episodes": args.episodes, "seed_start": args.seed_start, "mean_reward": float(values.mean()), "std_reward": float(values.std()), "min_reward": float(values.min()), "max_reward": float(values.max()), "completed_laps": completed_laps, "completion_rate": float(completed_laps / args.episodes)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
