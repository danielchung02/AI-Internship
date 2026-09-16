"""Train a frame-stack CNN Double DQN on discrete CarRacing-v3.

Run from ``car_racing``::

    python -m src.img_discrete.img_discrete_train

The default 1,500,000 *agent decisions* use ``action_repeat=2`` and therefore
collect up to 3,000,000 CarRacing environment frames.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.agents.img_discrete.dqn import ImageDQNAgent
from src.agents.img_discrete.replay_buffer import ImageReplayBuffer
from src.envs.img_discrete import make_image_env
from src.img_discrete.img_discrete_utils import repeated_step


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 42
    total_steps: int = 1_500_000 #3000000
    replay_capacity: int = 25_000 #50000
    batch_size: int = 64
    warmup_steps: int = 10_000 #20000
    train_frequency: int = 4
    target_update_interval: int = 2_000
    gamma: float = 0.99
    learning_rate: float = 1e-4
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 1_000_000 #2000000
    action_repeat: int = 2
    evaluation_interval: int = 50_000 #100000
    evaluation_episodes: int = 3 #10
    checkpoint_interval: int = 50_000 #100000
    stop_mean_reward: float | None = None
    output_dir: str = "outputs/img_discrete_double_dqn"
    resume_path: str | None = None


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--replay-capacity", type=int, default=defaults.replay_capacity)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--warmup-steps", type=int, default=defaults.warmup_steps)
    parser.add_argument("--train-frequency", type=int, default=defaults.train_frequency)
    parser.add_argument("--target-update-interval", type=int, default=defaults.target_update_interval)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--epsilon-start", type=float, default=defaults.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=defaults.epsilon_end)
    parser.add_argument("--epsilon-decay-steps", type=int, default=defaults.epsilon_decay_steps)
    parser.add_argument("--evaluation-interval", type=int, default=defaults.evaluation_interval)
    parser.add_argument("--evaluation-episodes", type=int, default=defaults.evaluation_episodes)
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
    parser.add_argument(
        "--stop-mean-reward",
        type=float,
        default=defaults.stop_mean_reward,
        help="Stop after an evaluation whose mean reward reaches this value.",
    )
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--action-repeat", type=int, default=2)
    args = parser.parse_args()
    return TrainConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        train_frequency=args.train_frequency,
        target_update_interval=args.target_update_interval,
        gamma=args.gamma,
        learning_rate=args.learning_rate,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        checkpoint_interval=args.checkpoint_interval,
        stop_mean_reward=args.stop_mean_reward,
        output_dir=args.output_dir,
        resume_path=args.resume,
        action_repeat=args.action_repeat,
    )


def epsilon_by_step(step: int, config: TrainConfig) -> float:
    fraction = min(max(step, 0) / config.epsilon_decay_steps, 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    agent: ImageDQNAgent,
    config: TrainConfig,
    seed_offset: int,
) -> tuple[float, float]:
    """Evaluate the greedy policy on fresh, deterministic episode seeds."""
    env = make_image_env()
    rng = np.random.default_rng(config.seed + 20_000)
    rewards: list[float] = []
    agent.online_network.eval()
    try:
        for episode in range(config.evaluation_episodes):
            image, _ = env.reset(seed=config.seed + 30_000 + episode)
            episode_reward = 0.0
            while True:
                action = agent.select_action(image, epsilon=0.0, rng=rng)
                image, reward, terminated, truncated, _ = repeated_step(
                    env, action, config.action_repeat
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
    if config.total_steps < 0 or config.action_repeat <= 0:
        raise ValueError("total_steps must be non-negative and action_repeat positive")
    if config.total_steps == 0 and config.stop_mean_reward is None:
        raise ValueError("unbounded training requires stop_mean_reward")
    if config.train_frequency <= 0 or config.target_update_interval <= 0:
        raise ValueError("train_frequency and target_update_interval must be positive")
    if config.epsilon_decay_steps <= 0:
        raise ValueError("epsilon_decay_steps must be positive")
    if config.stop_mean_reward is not None and not math.isfinite(config.stop_mean_reward):
        raise ValueError("stop_mean_reward must be finite")
    if config.warmup_steps < config.batch_size:
        raise ValueError("warmup_steps must be at least batch_size")
    if config.replay_capacity < config.batch_size:
        raise ValueError("replay_capacity must be at least batch_size")
    if config.warmup_steps > config.replay_capacity:
        raise ValueError("warmup_steps must not exceed replay_capacity")

    set_global_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    env = make_image_env(config.seed)
    agent = ImageDQNAgent(
        env.action_space.n,
        env.observation_space.shape,
        device,
        gamma=config.gamma,
        learning_rate=config.learning_rate,
    )
    replay_buffer = ImageReplayBuffer(
        config.replay_capacity,
        env.observation_space.shape,
        config.seed,
    )

    start_step = 0
    episode_index = 0
    best_evaluation_reward = -math.inf
    if config.resume_path is not None:
        metadata = agent.load(config.resume_path)
        start_step = int(metadata.get("global_step", 0))
        episode_index = int(metadata.get("episode_index", 0))
        best_evaluation_reward = float(metadata.get("best_evaluation_reward", -math.inf))
        print(
            f"Resumed from {config.resume_path} at global step {start_step}. "
            "Replay buffer starts empty."
        )

    image, _ = env.reset(seed=config.seed + episode_index)
    episode_reward = 0.0
    episode_length = 0
    last_loss = float("nan")
    current_step = start_step
    print(
        f"Device: {device}; observation shape: {env.observation_space.shape}; "
        f"action dimension: {env.action_space.n}"
    )

    try:
        step_numbers = (
            itertools.count(start_step + 1)
            if config.total_steps == 0
            else range(start_step + 1, config.total_steps + 1)
        )
        for global_step in step_numbers:
            current_step = global_step
            current_epsilon = epsilon_by_step(global_step, config)
            # A resumed checkpoint has no serialized replay buffer. Refill it
            # before applying TD updates, just as at the beginning of training.
            replay_is_warm = len(replay_buffer) >= config.warmup_steps
            if not replay_is_warm:
                action = int(env.action_space.sample())
            else:
                action = agent.select_action(image, current_epsilon, rng)

            next_image, reward, terminated, truncated, info = repeated_step(
                env, action, config.action_repeat
            )
            replay_buffer.add(image, action, reward, next_image, terminated)
            image = next_image
            episode_reward += reward
            episode_length += 1

            if (
                replay_is_warm
                and global_step % config.train_frequency == 0
                and len(replay_buffer) >= config.batch_size
            ):
                last_loss = agent.update(replay_buffer.sample(config.batch_size, device))
                if agent.optimization_steps % config.target_update_interval == 0:
                    agent.hard_update_target()

            if terminated or truncated:
                print(
                    f"step={global_step:>8} episode={episode_index:>5} "
                    f"reward={episode_reward:>8.2f} length={episode_length:>4} "
                    f"epsilon={current_epsilon:.3f}"
                )
                episode_index += 1
                image, _ = env.reset(seed=config.seed + episode_index)
                episode_reward = 0.0
                episode_length = 0

            if global_step % config.evaluation_interval == 0:
                mean_reward, std_reward = evaluate(agent, config, global_step)
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

                if (
                    config.stop_mean_reward is not None
                    and mean_reward >= config.stop_mean_reward
                ):
                    agent.save(
                        checkpoint_dir / "threshold_reached.pt",
                        global_step=global_step,
                        episode_index=episode_index,
                        best_evaluation_reward=best_evaluation_reward,
                        config=asdict(config),
                    )
                    print(
                        f"Reached mean reward target {config.stop_mean_reward:.2f} "
                        f"at step {global_step}; stopping training."
                    )
                    break

            if global_step % config.checkpoint_interval == 0:
                agent.save(
                    checkpoint_dir / f"step_{global_step}.pt",
                    global_step=global_step,
                    episode_index=episode_index,
                    best_evaluation_reward=best_evaluation_reward,
                    config=asdict(config),
                )

            if global_step % 1_000 == 0 and global_step >= config.warmup_steps:
                print(
                    f"step={global_step:>8} epsilon={current_epsilon:.3f} "
                    f"loss={last_loss:.4f} replay_size={len(replay_buffer)}"
                )

        agent.save(
            checkpoint_dir / "final.pt",
            global_step=current_step,
            episode_index=episode_index,
            best_evaluation_reward=best_evaluation_reward,
            config=asdict(config),
        )
    except KeyboardInterrupt:
        path = checkpoint_dir / f"interrupted_step_{current_step}.pt"
        agent.save(
            path,
            global_step=current_step,
            episode_index=episode_index,
            best_evaluation_reward=best_evaluation_reward,
            config=asdict(config),
        )
        print(f"Training interrupted. Saved checkpoint: {path}")
    finally:
        env.close()


if __name__ == "__main__":
    train(parse_args())
