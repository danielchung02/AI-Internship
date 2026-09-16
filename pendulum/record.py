from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

# These checkpoint files are written by pendulum_ppo.py, so recording must use
# that exact model architecture and the same input preprocessing.
from pendulum_ppo import ActorCritic, DEVICE, ENV_ID, HIDDEN_DIM

PROJECT_DIRECTORY = Path(__file__).resolve().parent
SOLVED_CHECKPOINT_PATH = PROJECT_DIRECTORY / "pposolved.pt"
LATEST_CHECKPOINT_PATH = PROJECT_DIRECTORY / "ppo_latest.pt"
VIDEO_ROOT_DIRECTORY = PROJECT_DIRECTORY / "ppo_videos"
NUM_RECORD_EPISODES = 10
# Match evaluate_policy() in pendulum_ppo.py so the recording score is directly
# comparable with the training log's "eval return".
SEED = 42


def load_trained_model(
    checkpoint_path: Path,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    action_low,
    action_high,
    device: torch.device,
) -> ActorCritic:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint file was not found: {checkpoint_path.resolve()}"
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = ActorCritic(
        observation_dim, action_dim, hidden_dim, action_low, action_high
    ).to(device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def record_policy(
    env_id: str,
    model: ActorCritic,
    video_directory: Path,
    num_episodes: int,
    device: torch.device,
) -> None:
    video_directory.mkdir(parents=True, exist_ok=True)
    episode_returns = []

    base_env = gym.make(env_id, render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(
        base_env,
        video_folder=str(video_directory),
        episode_trigger=lambda episode_index: True,
        name_prefix="ppo-policy",
    )

    try:
        for episode_index in range(num_episodes):
            observation, _ = env.reset(seed=SEED + episode_index)
            terminated = False
            truncated = False
            episode_return = 0.0

            while not (terminated or truncated):
                state = torch.as_tensor(
                    observation, dtype=torch.float32, device=device
                ).unsqueeze(0)

                with torch.no_grad():
                    action = model.deterministic_action(state).squeeze(0).cpu().numpy()
                observation, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)

            print(
                f"Episode {episode_index + 1}/{num_episodes} | "
                f"return: {episode_return:.1f}"
            )
            episode_returns.append(episode_return)
    finally:
        env.close()

    print(f"Mean return over {num_episodes} episodes: {np.mean(episode_returns):.1f}")


def main() -> None:
    probe_env = gym.make(ENV_ID)
    observation_dim = int(probe_env.observation_space.shape[0])
    action_dim = int(probe_env.action_space.shape[0])
    action_low = probe_env.action_space.low
    action_high = probe_env.action_space.high
    probe_env.close()

    # A solved checkpoint is preferred; during training only ppo_latest.pt
    # exists, so recording the latest policy still works.
    checkpoint_path = (
        SOLVED_CHECKPOINT_PATH
        if SOLVED_CHECKPOINT_PATH.is_file()
        else LATEST_CHECKPOINT_PATH
    )

    model = load_trained_model(
        checkpoint_path=checkpoint_path,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_dim=HIDDEN_DIM,
        action_low=action_low,
        action_high=action_high,
        device=DEVICE,
    )
    print(f"Recording checkpoint: {checkpoint_path.name}")
    run_directory = VIDEO_ROOT_DIRECTORY / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    print(f"Video directory: {run_directory}")
    record_policy(
        env_id=ENV_ID,
        model=model,
        video_directory=run_directory,
        num_episodes=NUM_RECORD_EPISODES,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()
