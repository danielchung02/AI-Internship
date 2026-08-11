"""Train an MLP DQN on engineered 1D CarRacing-v3 observations.

Run from the project root:
    python -m src.vec_discrete.vec_discrete_train
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from src.agents.vec_discrete.dqn import DQNAgent
from src.agents.vec_discrete.replay_buffer import ReplayBuffer
from src.envs.vec_discrete import make_vector_env


class _NullSummaryWriter:
    """No-op writer used when TensorBoard is not installed."""

    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def make_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        print(
            "TensorBoard is not installed; scalar logging is disabled. "
            "Install it with: python -m pip install tensorboard"
        )
        return _NullSummaryWriter()
    return SummaryWriter(log_dir=str(log_dir))


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 42
    total_steps: int = 500_000
    replay_capacity: int = 200_000
    batch_size: int = 128
    warmup_steps: int = 5_000
    train_frequency: int = 4
    target_update_interval: int = 2_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 200_000
    action_repeat: int = 2
    evaluation_interval: int = 25_000
    evaluation_episodes: int = 3
    checkpoint_interval: int = 50_000
    log_interval: int = 1_000
    output_dir: str = "outputs/engineered_vector_dqn"
    resume_path: str | None = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--output-dir", type=str, default="outputs/engineered_vector_dqn")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--action-repeat", type=int, default=2)
    args = parser.parse_args()

    return TrainConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        resume_path=args.resume,
        action_repeat=args.action_repeat,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(seed: int) -> gym.Env:
    return make_vector_env(seed)


def epsilon_by_step(step: int, config: TrainConfig) -> float:
    fraction = min(max(step, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (
        config.epsilon_end - config.epsilon_start
    )


def repeated_step(
    env: gym.Env,
    action: int,
    action_repeat: int,
) -> tuple[np.ndarray, float, bool, bool, dict]:
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")

    total_reward = 0.0
    final_info: dict = {}

    for _ in range(action_repeat):
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        final_info = info
        if terminated or truncated:
            break

    return next_state, total_reward, terminated, truncated, final_info


@torch.no_grad()
def evaluate(
    agent: DQNAgent,
    config: TrainConfig,
    seed_offset: int,
) -> tuple[float, float]:
    env = make_env(config.seed + 10_000 + seed_offset)
    rng = np.random.default_rng(config.seed + 20_000 + seed_offset)
    rewards: list[float] = []

    agent.online_network.eval()
    try:
        for episode_index in range(config.evaluation_episodes):
            state, _ = env.reset(
                seed=config.seed + 30_000 + seed_offset + episode_index
            )
            episode_reward = 0.0

            while True:
                action = agent.select_action(state, epsilon=0.0, rng=rng)
                state, reward, terminated, truncated, _ = repeated_step(
                    env,
                    action,
                    config.action_repeat,
                )
                episode_reward += reward
                if terminated or truncated:
                    break

            rewards.append(episode_reward)
    finally:
        env.close()
        agent.online_network.train()

    return float(np.mean(rewards)), float(np.std(rewards))


def train(config: TrainConfig) -> None:
    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if config.warmup_steps < config.batch_size:
        raise ValueError("warmup_steps must be at least batch_size")

    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2, ensure_ascii=False)

    writer = make_summary_writer(output_dir / "tensorboard")
    env = make_env(config.seed)

    state_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(env.action_space.n)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        gamma=config.gamma,
        learning_rate=config.learning_rate,
    )
    replay_buffer = ReplayBuffer(
        capacity=config.replay_capacity,
        state_dim=state_dim,
        seed=config.seed,
    )

    start_step = 0
    episode_index = 0
    best_evaluation_reward = -math.inf

    if config.resume_path is not None:
        metadata = agent.load(config.resume_path)
        start_step = int(metadata.get("global_step", 0))
        episode_index = int(metadata.get("episode_index", 0))
        best_evaluation_reward = float(
            metadata.get("best_evaluation_reward", -math.inf)
        )
        print(
            f"Resumed from {config.resume_path} at global step {start_step}. "
            "Replay buffer starts empty."
        )

    state, _ = env.reset(seed=config.seed + episode_index)
    episode_reward = 0.0
    episode_length = 0
    last_metrics: dict[str, float] = {}
    current_step = start_step

    print(f"Device: {device}")
    print(f"State dimension: {state_dim}, action dimension: {action_dim}")

    try:
        for global_step in range(start_step + 1, config.total_steps + 1):
            current_step = global_step
            epsilon = epsilon_by_step(global_step, config)

            if global_step <= config.warmup_steps:
                action = int(env.action_space.sample())
            else:
                action = agent.select_action(state, epsilon, rng)

            next_state, reward, terminated, truncated, info = repeated_step(
                env,
                action,
                config.action_repeat,
            )

            replay_buffer.add(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
            )

            state = next_state
            episode_reward += reward
            episode_length += 1

            if (
                global_step >= config.warmup_steps
                and global_step % config.train_frequency == 0
                and len(replay_buffer) >= config.batch_size
            ):
                batch = replay_buffer.sample(config.batch_size, device)
                last_metrics = agent.update(batch)

                if agent.optimization_steps % config.target_update_interval == 0:
                    agent.hard_update_target()

            if terminated or truncated:
                writer.add_scalar("episode/reward", episode_reward, global_step)
                writer.add_scalar("episode/length", episode_length, global_step)
                writer.add_scalar(
                    "episode/lap_finished",
                    float(bool(info.get("lap_finished", False))),
                    global_step,
                )

                print(
                    f"step={global_step:>7} episode={episode_index:>5} "
                    f"reward={episode_reward:>8.2f} length={episode_length:>4} "
                    f"epsilon={epsilon:.3f}"
                )

                episode_index += 1
                state, _ = env.reset(seed=config.seed + episode_index)
                episode_reward = 0.0
                episode_length = 0

            if global_step % config.log_interval == 0:
                writer.add_scalar("train/epsilon", epsilon, global_step)
                writer.add_scalar("train/replay_size", len(replay_buffer), global_step)
                for name, value in last_metrics.items():
                    writer.add_scalar(f"train/{name}", value, global_step)

            if global_step % config.evaluation_interval == 0:
                mean_reward, std_reward = evaluate(agent, config, global_step)
                writer.add_scalar("evaluation/mean_reward", mean_reward, global_step)
                writer.add_scalar("evaluation/std_reward", std_reward, global_step)
                print(
                    f"evaluation step={global_step}: "
                    f"mean_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )

                if mean_reward > best_evaluation_reward:
                    best_evaluation_reward = mean_reward
                    agent.save(
                        checkpoint_dir / "best.pt",
                        global_step=global_step,
                        episode_index=episode_index,
                        best_evaluation_reward=best_evaluation_reward,
                        config=asdict(config),
                    )

            if global_step % config.checkpoint_interval == 0:
                agent.save(
                    checkpoint_dir / f"step_{global_step}.pt",
                    global_step=global_step,
                    episode_index=episode_index,
                    best_evaluation_reward=best_evaluation_reward,
                    config=asdict(config),
                )

        agent.save(
            checkpoint_dir / "final.pt",
            global_step=config.total_steps,
            episode_index=episode_index,
            best_evaluation_reward=best_evaluation_reward,
            config=asdict(config),
        )
    except KeyboardInterrupt:
        interrupted_path = checkpoint_dir / f"interrupted_step_{current_step}.pt"
        agent.save(
            interrupted_path,
            global_step=current_step,
            episode_index=episode_index,
            best_evaluation_reward=best_evaluation_reward,
            config=asdict(config),
        )
        print(
            f"Training interrupted. Saved checkpoint: {interrupted_path}. "
            "Resume with --resume and the same --output-dir."
        )
    finally:
        env.close()
        writer.close()


if __name__ == "__main__":
    train(parse_args())
