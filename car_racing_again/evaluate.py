import argparse  # add
import numpy as np
import torch

# from train import TrainConfig, build_env, build_encoder, select_action  # delete
from train import TrainConfig, build_env, build_encoder, build_ppo_components, select_action  # add
from agents.dqn import DQNNetwork
from agents.ppo import ActorCritic

def evaluate_dqn(config, checkpoint_path, num_episodes=100, render_mode=None):
    env = build_env(config, render_mode)
    encoder = build_encoder(config,env.observation_space)
    action_dim = env.action_space.n #env.action_space.n이 뭐야
    online_network = DQNNetwork(encoder, config.feature_dim, action_dim)
    online_network.to(config.device)
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    online_network.load_state_dict(checkpoint["online_network"])
    online_network.eval()
    episode_rewards = []

    for episode in range(num_episodes):
        state, info = env.reset(seed=config.seed + episode)
        episode_reward = 0.0
        done = False

        while not done:
            action = select_action(online_network, state, 0.0, action_dim, config.device)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            state = next_state
            episode_reward += reward

        episode_rewards.append(episode_reward)
        print(f"episode: {episode + 1}, reward: {episode_reward:.1f}")

    print(f"mean reward: {np.mean(episode_rewards):.1f}")
    print(f"std reward: {np.std(episode_rewards):.1f}")

    env.close()
    return episode_rewards

def evaluate_ppo(config, checkpoint_path, num_episodes=100, render_mode=None):
    env = build_env(config, render_mode)
    # encoder = build_encoder(config, env.observation_space)  # delete
    # action_dim = env.action_space.shape[0]  # delete
    # actor_critic = ActorCritic(encoder, config.feature_dim, action_dim,env.action_space.low, env.action_space.high).to(config.device)  # delete
    actor_critic, _ = build_ppo_components(config, env)  # add
    checkpoint = torch.load(checkpoint_path, map_location=config.device)
    actor_critic.load_state_dict(checkpoint["actor_critic"])
    actor_critic.eval()
    episode_rewards = []
    for episode in range(num_episodes):
        state, info = env.reset(seed=config.seed + episode)
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

        episode_rewards.append(episode_reward)
        print(f"episode: {episode + 1}, reward: {episode_reward:.1f}")

    print(f"mean reward: {np.mean(episode_rewards):.1f}")
    print(f"std reward: {np.std(episode_rewards):.1f}")

    env.close()
    return episode_rewards
    

def main():
    parser = argparse.ArgumentParser()  # add
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], default="ppo")  # add
    parser.add_argument("--observation", choices=["vector", "image"], default="image")  # add
    parser.add_argument("--device", type=str, default="cuda")  # add
    parser.add_argument("--checkpoint-path", type=str, required=True)  # add
    parser.add_argument("--episodes", type=int, default=10)  # add
    parser.add_argument("--seed", type=int, default=42)  # add
    parser.add_argument("--feature-dim", type=int, default=256)  # add
    parser.add_argument("--num-frames", type=int, default=4)  # add
    parser.add_argument("--action-repeat", type=int, default=2)  # add
    args = parser.parse_args()  # add

    # config = TrainConfig()  # delete
    # config.algorithm = "ppo"  # delete
    # config.observation = "vector"  # delete
    # config.feature_dim = 256  # delete
    # config.num_frames = 4  # delete
    # config.action_repeat = 2  # delete
    # config.seed = 42  # delete
    # config.device = "cpu"  # delete
    # checkpoint_path = "checkpoints/vec_ppo/best.pt"  # delete
    # evaluate_ppo(config, checkpoint_path, num_episodes=100)  # delete
    config = TrainConfig()
    config.algorithm = args.algorithm  # add
    config.observation = args.observation  # add
    config.device = args.device  # add
    config.seed = args.seed  # add
    config.feature_dim = args.feature_dim  # add
    config.num_frames = args.num_frames  # add
    config.action_repeat = args.action_repeat  # add
    config.lr = 2.5e-4 if args.observation == "image" else 3e-4  # add

    if args.algorithm == "dqn":  # add
        evaluate_dqn(config, args.checkpoint_path, num_episodes=args.episodes)  # add
    else:  # add
        evaluate_ppo(config, args.checkpoint_path, num_episodes=args.episodes)  # add

if __name__ == "__main__":
    main()
