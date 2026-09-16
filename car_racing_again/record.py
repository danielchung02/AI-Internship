import argparse
from pathlib import Path

import gymnasium as gym
import torch

from agents.dqn import DQNNetwork
from agents.ppo import ActorCritic
# from train import TrainConfig, build_encoder, build_env, select_action  # delete
from train import TrainConfig, build_encoder, build_env, build_ppo_components, select_action  # add


def clear_previous_video(video_dir, name_prefix):
    video_dir = Path(video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    # for video_path in video_dir.glob(f"{name_prefix}-episode-*"):  # delete
    for video_path in video_dir.glob(f"{name_prefix}*"):  # add
        if video_path.is_file():
            video_path.unlink()


def make_record_env(config, video_dir, name_prefix):
    # clear_previous_video(video_dir, name_prefix)  # delete
    env = build_env(config, render_mode="rgb_array")
    return gym.wrappers.RecordVideo(env, video_folder=str(video_dir), episode_trigger=lambda episode_id: True, name_prefix=name_prefix)


def record_best_of_episodes(run_episode, model, config, video_dir, name_prefix, num_episodes):  # add
    if num_episodes <= 0:  # add
        raise ValueError("num_episodes must be positive")  # add

    clear_previous_video(video_dir, name_prefix)  # add
    video_dir = Path(video_dir)  # add
    results = []  # add

    for episode_index in range(num_episodes):  # add
        attempt_prefix = f"{name_prefix}_{episode_index + 1:02d}"  # add
        env = make_record_env(config, video_dir, attempt_prefix)  # add
        episode_reward = run_episode(model, env, config, config.seed + episode_index)  # add
        env.close()  # add
        video_paths = list(video_dir.glob(f"{attempt_prefix}-episode-*.mp4"))  # add
        if not video_paths:  # add
            raise RuntimeError(f"No video was created for episode {episode_index + 1}")  # add
        results.append((episode_reward, video_paths))  # add
        print(f"episode: {episode_index + 1}, reward: {episode_reward:.1f}")  # add

    best_index = max(range(num_episodes), key=lambda index: results[index][0])  # add
    best_reward, best_video_paths = results[best_index]  # add
    for episode_index, (_, video_paths) in enumerate(results):  # add
        if episode_index != best_index:  # add
            for video_path in video_paths:  # add
                video_path.unlink()  # add

    print(f"best episode: {best_index + 1}, reward: {best_reward:.1f}")  # add
    print("kept video:", ", ".join(str(path) for path in best_video_paths))  # add
    return best_reward  # add


def run_dqn_episode(online_network, env, config, seed):
    state, info = env.reset(seed=seed)
    episode_reward = 0.0
    done = False

    while not done:
        action = select_action(online_network, state, 0.0, env.action_space.n, config.device)

        for _ in range(config.action_repeat):
            state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if done:
                break

    return episode_reward


def run_ppo_episode(actor_critic, env, config, seed):
    state, info = env.reset(seed=seed)
    episode_reward = 0.0
    done = False

    while not done:
        state_tensor = torch.as_tensor(state, device=config.device).unsqueeze(0)

        with torch.no_grad():
            action_tensor, _, _ = actor_critic.act(state_tensor, deterministic=True)

        env_action = action_tensor.squeeze(0).cpu().numpy()

        for _ in range(config.action_repeat):
            state, reward, terminated, truncated, info = env.step(env_action)
            episode_reward += reward
            done = terminated or truncated

            if done:
                break

    return episode_reward


# def record_dqn(config, checkpoint_path, video_dir, target_reward):  # delete
def record_dqn(config, checkpoint_path, video_dir, num_episodes):  # add
    probe_env = build_env(config)
    encoder = build_encoder(config, probe_env.observation_space)
    action_dim = probe_env.action_space.n
    probe_env.close()

    online_network = DQNNetwork(encoder, config.feature_dim, action_dim).to(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    online_network.load_state_dict(checkpoint["online_network"])
    online_network.eval()

    # attempt = 0  # delete
    # while True:  # delete
    #     env = make_record_env(config, video_dir, "dqn_lap")  # delete
    #     episode_reward = run_dqn_episode(online_network, env, config, config.seed + attempt)  # delete
    #     env.close()  # delete
    #     print(f"attempt: {attempt + 1}, reward: {episode_reward:.1f}")  # delete
    #     if episode_reward > target_reward:  # delete
    #         print(f"Reward above {target_reward:.1f}; video saved.")  # delete
    #         return  # delete
    #     attempt += 1  # delete
    return record_best_of_episodes(run_dqn_episode, online_network, config, video_dir, "dqn_lap", num_episodes)  # add


# def record_ppo(config, checkpoint_path, video_dir, target_reward):  # delete
def record_ppo(config, checkpoint_path, video_dir, num_episodes):  # add
    probe_env = build_env(config)
    # encoder = build_encoder(config, probe_env.observation_space)  # delete
    # action_dim = probe_env.action_space.shape[0]  # delete
    # action_low = probe_env.action_space.low  # delete
    # action_high = probe_env.action_space.high  # delete
    # actor_critic = ActorCritic(  # delete
    #     encoder,  # delete
    #     config.feature_dim,  # delete
    #     action_dim,  # delete
    #     action_low,  # delete
    #     action_high,  # delete
    # ).to(config.device)  # delete
    actor_critic, _ = build_ppo_components(config, probe_env)  # add
    probe_env.close()  # add
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()

    # attempt = 0  # delete
    # while True:  # delete
    #     env = make_record_env(config, video_dir, "ppo_lap")  # delete
    #     episode_reward = run_ppo_episode(actor_critic, env, config, config.seed + attempt)  # delete
    #     env.close()  # delete
    #     print(f"attempt: {attempt + 1}, reward: {episode_reward:.1f}")  # delete
    #     if episode_reward > target_reward:  # delete
    #         print(f"Reward above {target_reward:.1f}; video saved.")  # delete
    #         return  # delete
    #     attempt += 1  # delete
    return record_best_of_episodes(run_ppo_episode, actor_critic, config, video_dir, "ppo_lap", num_episodes)  # add


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--observation", choices=["vector", "image"], required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--video-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=2)
    # parser.add_argument("--target-reward", type=float, default=850.0)  # delete
    parser.add_argument("--episodes", type=int, default=10)  # add
    args = parser.parse_args()

    config = TrainConfig()
    config.algorithm = args.algorithm
    config.observation = args.observation
    config.device = args.device
    config.seed = args.seed
    config.feature_dim = args.feature_dim
    config.num_frames = args.num_frames
    config.action_repeat = args.action_repeat
    config.lr = 2.5e-4 if args.observation == "image" else 3e-4  # add

    if args.algorithm == "dqn":
        # record_dqn(config, args.checkpoint_path, args.video_dir, args.target_reward)  # delete
        record_dqn(config, args.checkpoint_path, args.video_dir, args.episodes)  # add
    else:
        # record_ppo(config, args.checkpoint_path, args.video_dir, args.target_reward)  # delete
        record_ppo(config, args.checkpoint_path, args.video_dir, args.episodes)  # add


if __name__ == "__main__":
    main()
