"""Evaluate an engineered-vector CarRacing DQN checkpoint.

Example (run from ``car_racing``)::

    python -m src.vec_discrete.vec_discrete_evaluate --checkpoint outputs/engineered_vector_dqn/checkpoints/best.pt

The default 100 deterministic episodes follows the metric used by the legacy
Gym leaderboard.  These scores are reproducible for a pinned Gymnasium
version and seed range, but are not leaderboard-comparable because this agent
uses privileged simulator state rather than only pixels.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.agents.vec_discrete.dqn import DQNAgent
from src.envs.vec_discrete import make_vector_env


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    checkpoint: str
    episodes: int
    seed_start: int
    action_repeat: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    completed_laps: int
    completion_rate: float


def repeated_step(env, action: int, action_repeat: int):
    """Repeat one discrete action and return its accumulated reward."""
    reward_sum = 0.0
    info: dict = {}
    for _ in range(action_repeat):
        state, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        if terminated or truncated:
            break
    return state, reward_sum, terminated, truncated, info


def evaluate_checkpoint(
    checkpoint: str | Path,
    *,
    episodes: int = 100,
    seed_start: int = 10_000,
    action_repeat: int | None = None,
    device: torch.device | None = None,
) -> EvaluationResult:
    """Run a greedy policy on deterministic, previously unseen episode seeds."""
    if episodes <= 0:
        raise ValueError("episodes must be positive")

    checkpoint = Path(checkpoint)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_vector_env()
    try:
        state_dim = int(np.prod(env.observation_space.shape))
        action_dim = int(env.action_space.n)
        agent = DQNAgent(state_dim, action_dim, device)
        metadata = agent.load(checkpoint, load_optimizer=False)
        saved_config = metadata.get("config", {})
        repeat = int(action_repeat or saved_config.get("action_repeat", 2))
        if repeat <= 0:
            raise ValueError("action_repeat must be positive")

        rng = np.random.default_rng(seed_start)
        rewards: list[float] = []
        completed_laps = 0
        agent.online_network.eval()

        for episode in range(episodes):
            state, _ = env.reset(seed=seed_start + episode)
            episode_reward = 0.0
            while True:
                action = agent.select_action(state, epsilon=0.0, rng=rng)
                state, reward, terminated, truncated, info = repeated_step(
                    env, action, repeat
                )
                episode_reward += reward
                if terminated or truncated:
                    completed_laps += int(bool(info.get("lap_finished", False)))
                    break
            rewards.append(episode_reward)
    finally:
        env.close()

    reward_array = np.asarray(rewards, dtype=np.float64)
    return EvaluationResult(
        checkpoint=str(checkpoint),
        episodes=episodes,
        seed_start=seed_start,
        action_repeat=repeat,
        mean_reward=float(reward_array.mean()),
        std_reward=float(reward_array.std()),
        min_reward=float(reward_array.min()),
        max_reward=float(reward_array.max()),
        completed_laps=completed_laps,
        completion_rate=float(completed_laps / episodes),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10_000)
    parser.add_argument("--action-repeat", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")

    result = evaluate_checkpoint(
        args.checkpoint,
        episodes=args.episodes,
        seed_start=args.seed_start,
        action_repeat=args.action_repeat,
        device=device,
    )
    result_dict = asdict(result)
    print(json.dumps(result_dict, indent=2, ensure_ascii=False))
    print("PASS (mean >= 800)" if result.mean_reward >= 800.0 else "NOT YET (mean < 800)")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result_dict, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
