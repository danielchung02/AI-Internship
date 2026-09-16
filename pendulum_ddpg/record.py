import importlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


EXPERIMENT = "pendulum_ddpg"
NUM_EVAL_EPISODES = 10
EXPERIMENTS = {
    "pendulum_ddpg": {"algorithm": "ddpg", "module_name": "pendulum_ddpg", "env_id": "Pendulum-v1", "checkpoint_path": Path("pt/pendulum_ddpg/ddpg_latest.pt"), "videos_dir": Path("videos/pendulum_ddpg")},
    "halfcheetah_ddpg": {"algorithm": "ddpg", "module_name": "halfcheetah", "env_id": "HalfCheetah-v5", "checkpoint_path": Path("pt/halfcheetah_ddpg/ddpg_latest.pt"), "videos_dir": Path("videos/halfcheetah_ddpg")},
    "halfcheetah_td3": {"algorithm": "ddpg", "module_name": "halfcheetah_td3", "env_id": "HalfCheetah-v5", "checkpoint_path": Path("pt/halfcheetah_td3/td3_latest.pt"), "videos_dir": Path("videos/halfcheetah_td3")},
    "halfcheetah_ppo": {"algorithm": "ppo", "module_name": "halfcheetah_ppo", "env_id": "HalfCheetah-v5", "checkpoint_path": Path("pt/halfcheetah_ppo/ppo_latest.pt"), "videos_dir": Path("videos/halfcheetah_ppo")},
}


def select_action(model, state, algorithm, device):
    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if algorithm == "ddpg": return model(state_tensor).squeeze(0).cpu().numpy()
        return model.deterministic_action(state_tensor).squeeze(0).cpu().numpy()


def run_episode(env, model, config, module, device, seed):
    state, _ = env.reset(seed=seed)
    state = preprocess_state(module, state)
    episode_return, terminated, truncated = 0.0, False, False
    while not (terminated or truncated):
        action = select_action(model, state, config["algorithm"], device)
        next_state, reward, terminated, truncated, _ = env.step(action)
        state = preprocess_state(module, next_state)
        episode_return += float(reward)
    return episode_return


def preprocess_state(module, state):
    if hasattr(module, "normalize_state"): return module.normalize_state(state)
    return np.asarray(state, dtype=np.float32)


def load_model(config, module):
    device = module.DEVICE
    env = gym.make(config["env_id"])
    try:
        observation_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
        if config["algorithm"] == "ddpg": model = module.Actor(observation_dim, action_dim, module.HIDDEN_DIM, env.action_space.low, env.action_space.high).to(device)
        else: model = module.ActorCritic(observation_dim, action_dim, module.HIDDEN_DIM, env.action_space.low, env.action_space.high).to(device)
    finally:
        env.close()
    checkpoint = torch.load(config["checkpoint_path"], map_location=device)
    checkpoint_key = "actor_state_dict" if config["algorithm"] == "ddpg" else "model_state_dict"
    model.load_state_dict(checkpoint[checkpoint_key])
    model.eval()
    checkpoint_index = checkpoint.get("timestep", checkpoint.get("update_index"))
    return model, checkpoint_index, device


def evaluate_policy(model, config, module, device):
    env = gym.make(config["env_id"])
    try:
        episode_returns = [run_episode(env, model, config, module, device, module.SEED + episode_index) for episode_index in range(NUM_EVAL_EPISODES)]
    finally:
        env.close()
    return float(np.mean(episode_returns))


def record_policy(model, checkpoint_index, config, module, device):
    config["videos_dir"].mkdir(parents=True, exist_ok=True)
    env = gym.make(config["env_id"], render_mode="rgb_array")
    env = gym.wrappers.RecordVideo(env, video_folder=str(config["videos_dir"]), episode_trigger=lambda episode_index: True, name_prefix=f"{EXPERIMENT}_checkpoint_{checkpoint_index}")
    try:
        return run_episode(env, model, config, module, device, module.SEED)
    finally:
        env.close()


def main():
    if EXPERIMENT not in EXPERIMENTS: raise ValueError(f"unknown experiment: {EXPERIMENT}")
    config = EXPERIMENTS[EXPERIMENT]
    if not config["checkpoint_path"].exists(): raise FileNotFoundError(f"checkpoint not found: {config['checkpoint_path']}")
    module = importlib.import_module(config["module_name"])
    model, checkpoint_index, device = load_model(config, module)
    mean_eval_return = evaluate_policy(model, config, module, device)
    video_return = record_policy(model, checkpoint_index, config, module, device)
    print(f"experiment: {EXPERIMENT} | checkpoint: {config['checkpoint_path']} | checkpoint index: {checkpoint_index} | mean eval return (10 episodes): {mean_eval_return:.1f} | recorded return: {video_return:.1f}")


if __name__ == "__main__":
    main()
