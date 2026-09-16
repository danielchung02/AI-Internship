"""Train PPO on engineered-vector, continuous-action CarRacing-v3.

Run from ``car_racing``:
    python -m src.vec_cont.vec_cont_train
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.agents.vec_cont.ppo import PPOAgent
from src.agents.vec_cont.rollout_buffer import RolloutBuffer
from src.envs.action_repeat import repeated_step
from src.envs.lap import lap_finished
from src.envs.vec_cont import make_vec_cont_env


class _NullSummaryWriter:
    def add_scalar(self, *args, **kwargs) -> None:
        return None

    def close(self) -> None:
        return None


def make_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ModuleNotFoundError:
        print("TensorBoard is not installed; scalar logging is disabled.")
        return _NullSummaryWriter()
    return SummaryWriter(log_dir=str(log_dir))


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 42
    total_steps: int = 1_000_000
    rollout_steps: int = 2_048
    update_epochs: int = 10
    minibatch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    end_learning_rate: float = 3e-4
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.0
    end_entropy_coefficient: float = 0.0
    max_grad_norm: float = 0.5
    initial_log_std: float = 0.0
    action_repeat: int = 2
    evaluation_interval: int = 50_000
    evaluation_episodes: int = 5
    checkpoint_interval: int = 100_000
    stop_mean_reward: float | None = None
    output_dir: str = "outputs/vec_cont_ppo"
    resume_path: str | None = None
    reset_best_evaluation: bool = False


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = TrainConfig()
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--total-steps", type=int, default=defaults.total_steps)
    parser.add_argument("--rollout-steps", type=int, default=defaults.rollout_steps)
    parser.add_argument("--update-epochs", type=int, default=defaults.update_epochs)
    parser.add_argument("--minibatch-size", type=int, default=defaults.minibatch_size)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--end-learning-rate", type=float, default=defaults.end_learning_rate)
    parser.add_argument("--clip-ratio", type=float, default=defaults.clip_ratio)
    parser.add_argument("--value-coefficient", type=float, default=defaults.value_coefficient)
    parser.add_argument("--entropy-coefficient", type=float, default=defaults.entropy_coefficient)
    parser.add_argument("--end-entropy-coefficient", type=float, default=defaults.end_entropy_coefficient)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--initial-log-std", type=float, default=defaults.initial_log_std)
    parser.add_argument("--action-repeat", type=int, default=defaults.action_repeat)
    parser.add_argument("--evaluation-interval", type=int, default=defaults.evaluation_interval)
    parser.add_argument("--evaluation-episodes", type=int, default=defaults.evaluation_episodes)
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
    parser.add_argument("--stop-mean-reward", type=float, default=defaults.stop_mean_reward)
    parser.add_argument("--output-dir", type=str, default=defaults.output_dir)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--reset-best-evaluation",
        action="store_true",
        help="Select a new best.pt using only evaluations from this run.",
    )
    args = parser.parse_args()
    return TrainConfig(
        seed=args.seed,
        total_steps=args.total_steps,
        rollout_steps=args.rollout_steps,
        update_epochs=args.update_epochs,
        minibatch_size=args.minibatch_size,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        learning_rate=args.learning_rate,
        end_learning_rate=args.end_learning_rate,
        clip_ratio=args.clip_ratio,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        end_entropy_coefficient=args.end_entropy_coefficient,
        max_grad_norm=args.max_grad_norm,
        initial_log_std=args.initial_log_std,
        action_repeat=args.action_repeat,
        evaluation_interval=args.evaluation_interval,
        evaluation_episodes=args.evaluation_episodes,
        checkpoint_interval=args.checkpoint_interval,
        stop_mean_reward=args.stop_mean_reward,
        output_dir=args.output_dir,
        resume_path=args.resume,
        reset_best_evaluation=args.reset_best_evaluation,
    )


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(agent: PPOAgent, config: TrainConfig, seed_offset: int) -> tuple[float, float]:
    env = make_vec_cont_env()
    rewards: list[float] = []
    agent.actor_critic.eval()
    try:
        for episode in range(config.evaluation_episodes):
            state, _ = env.reset(seed=config.seed + 30_000 + episode)
            reward_sum = 0.0
            while True:
                action, _, _ = agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, _ = repeated_step(
                    env, action, config.action_repeat
                )
                reward_sum += reward
                if terminated or truncated:
                    break
            rewards.append(reward_sum)
    finally:
        env.close()
        agent.actor_critic.train()
    return float(np.mean(rewards)), float(np.std(rewards))


def train(config: TrainConfig) -> None:
    if config.total_steps < 0 or config.rollout_steps <= 0:
        raise ValueError("total_steps must be non-negative and rollout_steps positive")
    if config.total_steps == 0 and config.stop_mean_reward is None:
        raise ValueError("unbounded training requires stop_mean_reward")
    if config.action_repeat <= 0:
        raise ValueError("action_repeat must be positive")
    if config.minibatch_size <= 0 or config.minibatch_size > config.rollout_steps:
        raise ValueError("minibatch_size must be in [1, rollout_steps]")
    if config.learning_rate <= 0 or config.end_learning_rate <= 0:
        raise ValueError("learning rates must be positive")
    if config.entropy_coefficient < 0 or config.end_entropy_coefficient < 0:
        raise ValueError("entropy coefficients must be non-negative")

    set_global_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(asdict(config), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    writer = make_summary_writer(output_dir / "tensorboard")
    env = make_vec_cont_env(config.seed)

    state_dim = int(np.prod(env.observation_space.shape))
    agent = PPOAgent(
        state_dim,
        env.action_space,
        device,
        learning_rate=config.learning_rate,
        clip_ratio=config.clip_ratio,
        value_coefficient=config.value_coefficient,
        entropy_coefficient=config.entropy_coefficient,
        max_grad_norm=config.max_grad_norm,
        initial_log_std=config.initial_log_std,
    )
    buffer = RolloutBuffer(config.rollout_steps, state_dim, int(np.prod(env.action_space.shape)))

    global_step = 0
    episode_index = 0
    best_evaluation_reward = -math.inf
    if config.resume_path is not None:
        metadata = agent.load(config.resume_path)
        global_step = int(metadata.get("global_step", 0))
        episode_index = int(metadata.get("episode_index", 0))
        best_evaluation_reward = float(metadata.get("best_evaluation_reward", -math.inf))
        if config.reset_best_evaluation:
            best_evaluation_reward = -math.inf
        print(f"Resumed from {config.resume_path} at global step {global_step}.")

    state, _ = env.reset(seed=config.seed + episode_index)
    episode_reward = 0.0
    episode_length = 0
    next_evaluation = ((global_step // config.evaluation_interval) + 1) * config.evaluation_interval
    next_checkpoint = ((global_step // config.checkpoint_interval) + 1) * config.checkpoint_interval
    print(
        f"Device: {device}; state dimension: {state_dim}; "
        f"action dimension: {agent.action_dim}; action repeat: {config.action_repeat}"
    )

    try:
        while config.total_steps == 0 or global_step < config.total_steps:
            while len(buffer) < config.rollout_steps and (
                config.total_steps == 0 or global_step < config.total_steps
            ):
                action, log_prob, value = agent.select_action(state)
                next_state, reward, terminated, truncated, info = repeated_step(
                    env, action, config.action_repeat
                )
                next_value = 0.0 if terminated else agent.value(next_state)
                buffer.add(
                    state, action, float(reward), terminated, terminated or truncated,
                    value, next_value, log_prob,
                )
                global_step += 1
                state = next_state
                episode_reward += reward
                episode_length += 1

                if terminated or truncated:
                    print(
                        f"step={global_step:>8} episode={episode_index:>5} "
                        f"reward={episode_reward:>8.2f} length={episode_length:>4}"
                    )
                    writer.add_scalar("episode/reward", episode_reward, global_step)
                    writer.add_scalar("episode/length", episode_length, global_step)
                    writer.add_scalar("episode/lap_finished", float(lap_finished(env, terminated)), global_step)
                    episode_index += 1
                    state, _ = env.reset(seed=config.seed + episode_index)
                    episode_reward = 0.0
                    episode_length = 0

            schedule_steps = config.total_steps if config.total_steps > 0 else 3_000_000
            progress_remaining = max(0.0, 1.0 - global_step / schedule_steps)
            current_learning_rate = config.end_learning_rate + (
                config.learning_rate - config.end_learning_rate
            ) * progress_remaining
            current_entropy_coefficient = config.end_entropy_coefficient + (
                config.entropy_coefficient - config.end_entropy_coefficient
            ) * progress_remaining
            agent.set_learning_rate(current_learning_rate)
            agent.set_entropy_coefficient(current_entropy_coefficient)
            writer.add_scalar("train/learning_rate", current_learning_rate, global_step)
            writer.add_scalar("train/entropy_coefficient", current_entropy_coefficient, global_step)

            metrics = agent.update(
                buffer.get(device, gamma=config.gamma, gae_lambda=config.gae_lambda),
                epochs=config.update_epochs,
                minibatch_size=config.minibatch_size,
            )
            for name, value in metrics.items():
                writer.add_scalar(f"train/{name}", value, global_step)

            if global_step >= next_evaluation:
                mean_reward, std_reward = evaluate(agent, config, global_step)
                print(f"evaluation step={global_step}: mean_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                writer.add_scalar("evaluation/mean_reward", mean_reward, global_step)
                writer.add_scalar("evaluation/std_reward", std_reward, global_step)
                if mean_reward > best_evaluation_reward:
                    best_evaluation_reward = mean_reward
                    agent.save(checkpoint_dir / "best.pt", global_step=global_step, episode_index=episode_index, best_evaluation_reward=best_evaluation_reward, config=asdict(config))
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
                    print(f"Reached mean reward target {config.stop_mean_reward:.2f}; stopping training.")
                    break
                next_evaluation += config.evaluation_interval

            if global_step >= next_checkpoint:
                agent.save(checkpoint_dir / f"step_{global_step}.pt", global_step=global_step, episode_index=episode_index, best_evaluation_reward=best_evaluation_reward, config=asdict(config))
                next_checkpoint += config.checkpoint_interval

        agent.save(checkpoint_dir / "final.pt", global_step=global_step, episode_index=episode_index, best_evaluation_reward=best_evaluation_reward, config=asdict(config))
    except KeyboardInterrupt:
        path = checkpoint_dir / f"interrupted_step_{global_step}.pt"
        agent.save(path, global_step=global_step, episode_index=episode_index, best_evaluation_reward=best_evaluation_reward, config=asdict(config))
        print(f"Training interrupted. Saved checkpoint: {path}")
    finally:
        env.close()
        writer.close()


if __name__ == "__main__":
    train(parse_args())
