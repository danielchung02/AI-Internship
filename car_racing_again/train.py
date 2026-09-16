import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import copy
from pathlib import Path

from encoders.mlp import MLPEncoder
from encoders.cnn import CNNEncoder
from image_observation import DashboardMaskedFrameStack  # add
from vector_observation import VectorObservation, make_vector_env
from agents.dqn import DQNNetwork, ReplayBuffer
# from agents.ppo import ActorCritic, collect_rollout, compute_R_A, update_ppo  # delete
from agents.ppo import ActorCritic, collect_rollout, compute_R_A, compute_n_step_R_A, update_ppo  # add

class TrainConfig():
    def __init__(self):
        self.algorithm = None
        self.observation = None
        self.env_id = "CarRacing-v3"
        self.feature_dim = 256  
        # self.total_steps = None  # delete
        self.total_steps = 0  # add
        self.resume_path = None  # add
        self.seed = 42 #?
        self.device = "cuda"
        self.checkpoint_dir= None
        self.save_interval= None
        self.batch_size= None
        self.buffer_capacity= None
        self.num_frames = 4

        self.gamma= None
        self.lr= None
        self.learning_starts= None
        self.target_update_interval= None
        self.epsilon_start= None
        self.epsilon_end= None
        self.epsilon_decay= None
        self.train_frequency = 4
        self.action_repeat = 2
        self.reward_window=10
        self.stop_reward = 820

        #ppo config
        self.num_steps = 2048
        self.update_epochs = 10
        self.num_minibatches = 32
        self.clip_coef = 0.2
        #self.gae_lambda = 0.95
        self.value_coef = 0.5
        self.entropy_coef = 0.01
        self.max_grad_norm = 0.5
        self.target_kl = None  # add
        self.n_step_return = None  # add
        self.schedule_steps = None  # add
        self.end_lr = None  # add
        self.initial_entropy_coef = None  # add
        self.end_entropy_coef = None  # add
        self.num_steps = 2048

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_encoder(config: TrainConfig, observation_space = None):
    if config.observation == "vector":
        state_dim = VectorObservation.state_dim
        hidden_dim = config.feature_dim
        encoder = MLPEncoder(state_dim, hidden_dim)
        return encoder
    elif config.observation == "image":
        observation_shape = observation_space.shape 
        feature_dim = config.feature_dim
        encoder =  CNNEncoder(observation_shape, feature_dim)
        return encoder
    else:
        raise ValueError("observation은 반드시 vector나 image여야 한다")

def build_env(config: TrainConfig, render_mode = None):
    continuous = config.algorithm == "ppo"
    if config.observation == "vector":
        env = make_vector_env(render_mode=render_mode, continuous=continuous)
        return env
    
    elif config.observation == "image":
        env = gym.make(config.env_id,continuous=continuous, domain_randomize=False,render_mode=render_mode,)
        # env = gym.wrappers.FrameStackObservation(env, stack_size=config.num_frames)  # delete
        env = DashboardMaskedFrameStack(env, num_frames=config.num_frames)  # add
        return env
    
def build_dqn_components(config: TrainConfig, env):
    encoder = build_encoder(config, env.observation_space) 
    action_dim = env.action_space.n
    online_network = DQNNetwork(encoder, config.feature_dim, action_dim)
    online_network.to(device = config.device)
    target_network = copy.deepcopy(online_network)
    optimizer = torch.optim.AdamW(online_network.parameters(), lr = config.lr, weight_decay = 0.0)
    replaybuffer = ReplayBuffer(capacity=config.buffer_capacity)
    return online_network, target_network, optimizer, replaybuffer

def build_ppo_components(config: TrainConfig, env):
    # encoder = build_encoder(config, env.observation_space)  # delete
    action_dim = env.action_space.shape[0]
    if config.observation == "image":  # add
        encoder = CNNEncoder(env.observation_space.shape, feature_dim=None)  # add
        ppo_feature_dim = encoder.output_dim  # add
        initial_log_std = -0.5  # add
        initial_action_mean_bias = [0.0, -1.0986123, -2.2975599]  # add
        head_dim = 512  # add
    else:  # add
        encoder = build_encoder(config, env.observation_space)  # add
        ppo_feature_dim = config.feature_dim  # add
        initial_log_std = 0.0  # add
        initial_action_mean_bias = None  # add
        head_dim = config.feature_dim  # add
    # actor_critic = ActorCritic(encoder, config.feature_dim, action_dim, env.action_space.low,env.action_space.high).to(config.device)  # delete
    actor_critic = ActorCritic(  # add
        encoder, ppo_feature_dim, action_dim, env.action_space.low, env.action_space.high,  # add
        # initial_log_std=initial_log_std, initial_action_mean_bias=initial_action_mean_bias  # delete
        initial_log_std=initial_log_std, initial_action_mean_bias=initial_action_mean_bias, head_dim=head_dim  # add
    ).to(config.device)  # add
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=config.lr, eps=1e-5)
    return actor_critic, optimizer

def select_action(online_network, state, epsilon, action_dim, device):
    if np.random.rand() < epsilon:
            return int(np.random.randint(action_dim))
    state_tensor = torch.as_tensor(state, device=device).unsqueeze(0)
    with torch.no_grad():
        qs = online_network(state_tensor)
        return int(qs.argmax(dim = 1).item())

def update_dqn(online_network, target_network, optimizer, replay_buffer, config):
    states, actions, rewards, next_states, dones = replay_buffer.sample(config.batch_size, config.device)

    qs = online_network(states)
    current_qs = qs.gather(1, actions.unsqueeze(1)).squeeze(1) 

    with torch.no_grad():
        next_qs = target_network(next_states)
        td_target = rewards + config.gamma * (1 - dones) * next_qs.max(dim=1).values 
    loss = ((current_qs - td_target)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_network.parameters(), max_norm=10.0)
    optimizer.step()
    return loss.item()

def train_dqn(config):
    set_seed(config.seed)

    env = build_env(config)
    online_network, target_network, optimizer, replay_buffer = build_dqn_components(config, env)

    state, info = env.reset(seed=config.seed)
    episode_reward = 0.0
    recent_episode_rewards = []
    step = 0

    while True:
        progress = min(step / config.epsilon_decay, 1.0)
        epsilon = config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
        action = select_action(online_network, state, epsilon, env.action_space.n, config.device)
        next_state = state
        total_reward = 0.0
        terminated = False
        truncated = False

        for _ in range(config.action_repeat):
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        done = terminated or truncated
        replay_buffer.add(state, action, total_reward, next_state, done)
        state = next_state
        episode_reward += total_reward

        if step >= config.learning_starts and len(replay_buffer.buffer) >= config.batch_size and step % config.train_frequency == 0:
            loss = update_dqn(online_network, target_network, optimizer, replay_buffer, config)

        if (step + 1) % config.target_update_interval == 0:
            with torch.no_grad():
                for target_param, online_param in zip(target_network.parameters(), online_network.parameters()):
                    target_param.copy_(online_param)

        if (step + 1) % config.save_interval == 0:
            save_dqn_checkpoint(online_network, target_network, optimizer, step + 1, config, "dqn_latest.pt")

        if done:
            recent_episode_rewards.append(episode_reward)

            if len(recent_episode_rewards) > config.reward_window:
                recent_episode_rewards.pop(0)

            if len(recent_episode_rewards) == config.reward_window:
                recent_mean = float(np.mean(recent_episode_rewards))
                print(f"step: {step + 1}, recent mean reward: {recent_mean:.1f}")

                if recent_mean >= config.stop_reward:
                    save_dqn_checkpoint(online_network, target_network, optimizer, step + 1, config, "last.pt")
                    break

            state, info = env.reset()
            episode_reward = 0.0

        step += 1

    env.close()
    return online_network

def train_ppo(config:TrainConfig):
    # next_save_step = config.save_interval  # delete
    solved = False
    set_seed(config.seed)
    env = build_env(config)
    actor_critic, optimizer = build_ppo_components(config, env)
    step = 0  # add
    best_evaluation_return = -float("inf")  # add
    if config.resume_path is not None:  # add
        checkpoint = torch.load(config.resume_path, map_location=config.device)  # add
        actor_critic.load_state_dict(checkpoint["actor_critic"])  # add
        optimizer.load_state_dict(checkpoint["optimizer"])  # add
        step = int(checkpoint.get("step", 0))  # add
        best_evaluation_return, _ = evaluate_ppo_policy(config, actor_critic, num_episodes=5)  # add
        print(f"resumed checkpoint: {config.resume_path}, step: {step}, baseline evaluation: {best_evaluation_return:.1f}")  # add
    state, info = env.reset(seed=config.seed)
    episode_return = 0.0
    recent_episode_rewards = []
    # step = 0  # delete
    evaluation_interval = 50000
    num_evaluation_episodes = 5
    # next_evaluation_step = evaluation_interval  # delete
    next_evaluation_step = ((step // evaluation_interval) + 1) * evaluation_interval  # add
    next_save_step = ((step // config.save_interval) + 1) * config.save_interval  # add
    try:
        while True:
            actor_critic.train()
            rollout_steps = config.num_steps if config.total_steps <= 0 else min(config.num_steps, config.total_steps - step)  # add
            if rollout_steps <= 0:  # add
                save_ppo_checkpoint(actor_critic, optimizer, step, config, "last.pt")  # add
                break  # add
            # rollout, state, episode_return, completed_episode_returns = collect_rollout(env,actor_critic,  # delete
            #                 state,config.num_steps, config.device, config.action_repeat, episode_return)  # delete
            rollout, state, episode_return, completed_episode_returns = collect_rollout(  # add
                env, actor_critic, state, rollout_steps, config.device, config.action_repeat, config.gamma, episode_return  # add
            )  # add
            # step += config.num_steps  # delete
            step += rollout_steps  # add
            
            for completed_return in completed_episode_returns:
                recent_episode_rewards.append(completed_return)
                if len(recent_episode_rewards) > config.reward_window:
                    recent_episode_rewards.pop(0)
                if len(recent_episode_rewards) == config.reward_window:
                    recent_mean = float(np.mean(recent_episode_rewards))
                    print(f"step: {step}, recent mean reward: {recent_mean:.1f}")
                    #aftertouch: solved is decided only by deterministic evaluation.
            
            if rollout["terminateds"][-1].item() == 1.0 or rollout["truncateds"][-1].item() == 1.0: #마지막
                next_value = torch.tensor(0.0, dtype=torch.float32, device=config.device)
            else:
                next_state_tensor = torch.as_tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
            
                with torch.no_grad():
                    next_value = actor_critic.get_value(next_state_tensor).squeeze(0)
            
            # returns, advantages = compute_R_A(rollout["rewards"], rollout["terminateds"],rollout["truncateds"], rollout["values"],next_value, rollout["truncated_bootstrap_values"], config.gamma)  # delete
            # returns, advantages = compute_R_A(  # delete
            #     rollout["rewards"], rollout["terminateds"], rollout["truncateds"], rollout["values"],  # delete
            #     next_value, rollout["truncated_bootstrap_values"], rollout["bootstrap_discounts"]  # delete
            # )  # delete
            if config.observation == "image":  # add
                returns, advantages = compute_n_step_R_A(  # add
                    rollout["rewards"], rollout["terminateds"], rollout["truncateds"], rollout["values"],  # add
                    next_value, rollout["truncated_bootstrap_values"], rollout["bootstrap_discounts"], config.n_step_return  # add
                )  # add
            else:  # add
                returns, advantages = compute_R_A(  # add
                    rollout["rewards"], rollout["terminateds"], rollout["truncateds"], rollout["values"],  # add
                    next_value, rollout["truncated_bootstrap_values"], rollout["bootstrap_discounts"]  # add
                )  # add
            if config.observation == "image":  # add
                progress_remaining = max(0.0, 1.0 - min(step / config.schedule_steps, 1.0))  # add
                current_lr = config.end_lr + (config.lr - config.end_lr) * progress_remaining  # add
                current_entropy_coef = config.end_entropy_coef + (config.initial_entropy_coef - config.end_entropy_coef) * progress_remaining  # add
                for parameter_group in optimizer.param_groups:  # add
                    parameter_group["lr"] = current_lr  # add
                config.entropy_coef = current_entropy_coef  # add
            # metrics = update_ppo(actor_critic, optimizer, rollout, returns, advantages)  # delete
            metrics = update_ppo(actor_critic, optimizer, rollout, returns, advantages, config)  # add
            
            print(
                f"step: {step}, "
                f"policy loss: {metrics['policy_loss']:.4f}, "
                f"value loss: {metrics['value_loss']:.4f}, "
                # f"entropy: {metrics['entropy']:.4f}"  # delete
                f"entropy: {metrics['entropy']:.4f}, "  # add
                f"approx kl: {metrics['approx_kl']:.6f}"  # add
            )
            
            if step >= next_evaluation_step:
                mean_eval_return, std_eval_return = evaluate_ppo_policy(
                    config,
                    actor_critic,
                    num_evaluation_episodes,
                )

                print(
                    f"evaluation step: {step}, "
                    f"mean reward: {mean_eval_return:.1f}, "
                    f"std: {std_eval_return:.1f}"
                )

                if mean_eval_return > best_evaluation_return:
                    best_evaluation_return = mean_eval_return
                    save_ppo_checkpoint(
                        actor_critic,
                        optimizer,
                        step,
                        config,
                        "best.pt",
                    )

                if mean_eval_return >= config.stop_reward:
                    solved = True

                while step >= next_evaluation_step:
                    next_evaluation_step += evaluation_interval

            #aftertouch: keep a recoverable latest PPO checkpoint between evaluations.
            if step >= next_save_step:
                save_ppo_checkpoint(actor_critic, optimizer, step, config, "ppo_latest.pt")

                while step >= next_save_step:
                    next_save_step += config.save_interval
            
            # if solved:  # delete
            if solved or (config.total_steps > 0 and step >= config.total_steps):  # add
                save_ppo_checkpoint(actor_critic, optimizer, step, config, "last.pt")
                break                
            
                       
    finally:
        env.close()
    return actor_critic

def evaluate_ppo_policy(config, actor_critic, num_episodes=5):
    env = build_env(config)
    episode_returns = []
    actor_critic.eval()

    try:
        with torch.no_grad():
            for episode_index in range(num_episodes):
                state, _ = env.reset(seed=config.seed + 30000 + episode_index)
                episode_return = 0.0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=config.device).unsqueeze(0)
                    action_tensor, _, _ = actor_critic.act(state_tensor, deterministic=True)
                    env_action = action_tensor.squeeze(0).cpu().numpy()

                    for _ in range(config.action_repeat):
                        state, reward, terminated, truncated, _ = env.step(env_action)
                        episode_return += reward

                        if terminated or truncated:
                            break

                episode_returns.append(episode_return)
    finally:
        env.close()
        actor_critic.train()

    return float(np.mean(episode_returns)), float(np.std(episode_returns))

def save_dqn_checkpoint(online_network, target_network, optimizer, step, config, filename):
    checkpoint_path = Path(config.checkpoint_dir) / filename
    checkpoint = {
        "step": step,
        "config": vars(config),
        "online_network": online_network.state_dict(),
        "target_network": target_network.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, checkpoint_path)

def save_ppo_checkpoint(actor_critic, optimizer, step, config, filename):
    checkpoint_path = Path(config.checkpoint_dir) / filename

    checkpoint = {
        "step": step,
        "config": vars(config),
        "actor_critic": actor_critic.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    torch.save(checkpoint, checkpoint_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=["dqn", "ppo"], required=True)
    parser.add_argument("--observation", choices=["vector", "image"], required=True)
    parser.add_argument("--device", type=str, required=True)
    parser.add_argument("--checkpoint-dir", type=str, required=True)
    parser.add_argument("--total-steps", type=int, default=0)  # add
    parser.add_argument("--resume-path", type=str, default=None)  # add
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-capacity", type=int, default=50000)
    # parser.add_argument("--gamma", type=float, default=0.99)  # delete
    parser.add_argument("--gamma", type=float, default=None)  # add
    parser.add_argument("--lr", type=float, default=None)  # 고친 줄
    parser.add_argument("--learning-starts", type=int, default=10000)
    parser.add_argument("--target-update-interval", type=int, default=8000)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=int, default=1500000)
    parser.add_argument("--reward-window", type=int, default=10)
    parser.add_argument("--stop-reward", type=float, default=820.0)
    parser.add_argument("--save-interval", type=int, default=10000)
    parser.add_argument("--train-frequency", type=int, default=4)
    parser.add_argument("--action-repeat", type=int, default=2)
    # parser.add_argument("--num-steps", type=int, default=2048)  # delete
    parser.add_argument("--num-steps", type=int, default=None)  # add
    # parser.add_argument("--update-epochs", type=int, default=10)  # delete
    parser.add_argument("--update-epochs", type=int, default=None)  # add
    # parser.add_argument("--num-minibatches", type=int, default=32)  # delete
    parser.add_argument("--num-minibatches", type=int, default=None)  # add
    # parser.add_argument("--clip-coef", type=float, default=0.2)  # delete
    parser.add_argument("--clip-coef", type=float, default=None)  # add
    #parser.add_argument("--gae-lambda", type=float, default=0.95)  # 고친 줄
    parser.add_argument("--value-coef", type=float, default=0.5)  # 고친 줄
    # parser.add_argument("--entropy-coef", type=float, default=0.0)  # delete
    # parser.add_argument("--entropy-coef", type=float, default=0.01)  # delete
    parser.add_argument("--entropy-coef", type=float, default=None)  # add
    parser.add_argument("--max-grad-norm", type=float, default=0.5)  # 고친 줄
    parser.add_argument("--target-kl", type=float, default=None)  # add

    args = parser.parse_args()
    '''
acton repeat는 agent가 홀수 프레임에는 눈을 감고 운전하고 짝수 프레임에만 결정을 내린다는것
그리고 train freq은  총 8frame지나서 4번의 행동을 했을떄 optimizer를 한번 업데이트 하는거고
타겟 업데이트 interval이 8000이라는거는 총 16000프레임 지나서 8000번의 행동을 했을때 타겟을 동기화한다는거
    '''

    config = TrainConfig()
    config.algorithm = args.algorithm  # 고친 줄
    config.observation = args.observation
    config.device = args.device
    # config.checkpoint_dir = args.checkpoint_dir  # delete
    config.checkpoint_dir = args.checkpoint_dir  # add
    config.total_steps = args.total_steps  # add
    config.resume_path = args.resume_path  # add
    config.seed = args.seed
    config.feature_dim = args.feature_dim
    config.num_frames = args.num_frames
    config.batch_size = args.batch_size
    config.buffer_capacity = args.buffer_capacity
    # config.gamma = args.gamma  # delete
    # config.lr = args.lr if args.lr is not None else (1e-4 if args.algorithm == "dqn" else 3e-4)  # delete
    if args.algorithm == "ppo" and args.observation == "image":  # add
        # ppo_defaults = {"gamma": 0.997497, "lr": 2.5e-4, "num_steps": 4096, "update_epochs": 8, "num_minibatches": 16, "clip_coef": 0.15, "entropy_coef": 0.005, "target_kl": 0.015}  # delete
        ppo_defaults = {"gamma": 0.997497, "lr": 2.5e-4, "end_lr": 2.5e-5, "schedule_steps": 5_000_000, "num_steps": 4096, "update_epochs": 8, "num_minibatches": 16, "clip_coef": 0.15, "entropy_coef": 0.005, "end_entropy_coef": 0.0005, "target_kl": 0.015, "n_step_return": 64}  # add
    elif args.algorithm == "ppo":  # add
        # ppo_defaults = {"gamma": 0.994987, "lr": 3e-4, "num_steps": 2048, "update_epochs": 10, "num_minibatches": 8, "clip_coef": 0.2, "entropy_coef": 0.0, "target_kl": None}  # delete
        ppo_defaults = {"gamma": 0.994987, "lr": 3e-4, "end_lr": None, "schedule_steps": None, "num_steps": 2048, "update_epochs": 10, "num_minibatches": 8, "clip_coef": 0.2, "entropy_coef": 0.0, "end_entropy_coef": None, "target_kl": None, "n_step_return": None}  # add
    else:  # add
        ppo_defaults = None  # add
    config.gamma = args.gamma if args.gamma is not None else (ppo_defaults["gamma"] if ppo_defaults is not None else 0.99)  # add
    config.n_step_return = ppo_defaults["n_step_return"] if ppo_defaults is not None else None  # add
    config.lr = args.lr if args.lr is not None else (ppo_defaults["lr"] if ppo_defaults is not None else 1e-4)  # add
    config.end_lr = ppo_defaults["end_lr"] if ppo_defaults is not None else None  # add
    config.schedule_steps = ppo_defaults["schedule_steps"] if ppo_defaults is not None else None  # add
    config.learning_starts = args.learning_starts
    config.target_update_interval = args.target_update_interval
    config.epsilon_start = args.epsilon_start
    config.epsilon_end = args.epsilon_end
    config.epsilon_decay = args.epsilon_decay
    config.reward_window = args.reward_window
    config.stop_reward = args.stop_reward
    config.save_interval = args.save_interval
    config.train_frequency = args.train_frequency
    config.action_repeat = args.action_repeat 
    # config.num_steps = args.num_steps  # delete
    # config.update_epochs = args.update_epochs  # delete
    # config.num_minibatches = args.num_minibatches  # delete
    # config.clip_coef = args.clip_coef  # delete
    config.num_steps = args.num_steps if args.num_steps is not None else (ppo_defaults["num_steps"] if ppo_defaults is not None else config.num_steps)  # add
    config.update_epochs = args.update_epochs if args.update_epochs is not None else (ppo_defaults["update_epochs"] if ppo_defaults is not None else config.update_epochs)  # add
    config.num_minibatches = args.num_minibatches if args.num_minibatches is not None else (ppo_defaults["num_minibatches"] if ppo_defaults is not None else config.num_minibatches)  # add
    config.clip_coef = args.clip_coef if args.clip_coef is not None else (ppo_defaults["clip_coef"] if ppo_defaults is not None else config.clip_coef)  # add
   # config.gae_lambda = args.gae_lambda  # 고친 줄
    config.value_coef = args.value_coef  # 고친 줄
    # config.entropy_coef = args.entropy_coef  # delete
    config.entropy_coef = args.entropy_coef if args.entropy_coef is not None else (ppo_defaults["entropy_coef"] if ppo_defaults is not None else config.entropy_coef)  # add
    config.initial_entropy_coef = config.entropy_coef  # add
    config.end_entropy_coef = ppo_defaults["end_entropy_coef"] if ppo_defaults is not None else None  # add
    config.max_grad_norm = args.max_grad_norm  # 고친 줄
    config.target_kl = args.target_kl if args.target_kl is not None else (ppo_defaults["target_kl"] if ppo_defaults is not None else None)  # add
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    if config.algorithm == "dqn":  # 고친 줄
        return train_dqn(config)  # 고친 줄

    return train_ppo(config)  # 고친 줄

if __name__ == "__main__":
    main()




