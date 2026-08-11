"""Evaluate an engineered-vector, continuous-action PPO checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from src.agents.vec_cont.ppo import PPOAgent
from src.envs.action_repeat import repeated_step
from src.envs.lap import lap_finished
from src.envs.vec_cont import make_vec_cont_env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()
    if args.episodes <= 0:
        raise ValueError("episodes must be positive")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_vec_cont_env()
    try:
        state_dim = int(np.prod(env.observation_space.shape))
        agent = PPOAgent(state_dim, env.action_space, device)
        metadata = agent.load(args.checkpoint, load_optimizer=False)
        saved_config = metadata.get("config", {})
        action_repeat = int(args.action_repeat or saved_config.get("action_repeat", 1))
        if action_repeat <= 0:
            raise ValueError("action_repeat must be positive")
        agent.actor_critic.eval()
        rewards: list[float] = []
        completed_laps = 0
        for episode in range(args.episodes):
            state, _ = env.reset(seed=args.seed_start + episode)
            reward_sum = 0.0
            while True:
                action, _, _ = agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = repeated_step(
                    env, action, action_repeat
                )
                reward_sum += reward
                if terminated or truncated:
                    completed_laps += int(lap_finished(env, terminated))
                    break
            rewards.append(reward_sum)
    finally:
        env.close()

    values = np.asarray(rewards, dtype=np.float64)
    result = {"checkpoint": str(args.checkpoint), "episodes": args.episodes, "seed_start": args.seed_start, "action_repeat": action_repeat, "mean_reward": float(values.mean()), "std_reward": float(values.std()), "min_reward": float(values.min()), "max_reward": float(values.max()), "completed_laps": completed_laps, "completion_rate": float(completed_laps / args.episodes)}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
