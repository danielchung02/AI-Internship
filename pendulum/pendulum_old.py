import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from pathlib import Path
from typing import Callable

ENV_ID = "Pendulum-v1"
SEED = 42
NUM_ENVS = 4

NUM_STEPS = 32
TOTAL_TIMESTEPS = 200_000
GAMMA = 0.99
GAE_LAMBDA = 0.95  
LEARNING_RATE = 3e-4
HIDDEN_DIM = 128
VALUE_COEF = 0.05
ENTROPY_COEF = 0.0001
MAX_GRAD_NORM = 0.5

SAVE_INTERVAL_UPDATES = 50
EVAL_INTERVAL_UPDATES = 100
NUM_EVAL_EPISODES = 10
SOLVED_MEAN_RETURN = -200.0 #reward = -(angle_error² + 0.1 × angular_velocity² + 0.001 × torque²)
LATEST_CHECKPOINT_PATH = Path("latest.pt")
SOLVED_CHECKPOINT_PATH = Path("solved.pt")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.feature_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)  
        action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)  
        action_scale = (action_high_tensor - action_low_tensor) / 2  
        action_bias = (action_high_tensor + action_low_tensor) / 2  
        self.register_buffer("action_scale", action_scale)  
        self.register_buffer("action_bias", action_bias) 

    def forward(self, states):
        features = self.feature_net(states)
        mean = self.actor_mean(features)
        std = torch.exp(self.log_std).expand_as(mean)
        values = self.critic(features).squeeze(-1)
        return mean, std, values

    def sample_action(self, states):
        mean, std, values = self.forward(states)
        distribution = Normal(mean, std)
        raw_actions = distribution.rsample()
        z = torch.tanh(raw_actions)
        env_actions = self.action_bias + self.action_scale * z  #(-2,2)안에 들어오게
        normal_log_probs = distribution.log_prob(raw_actions) #pdf에서 샘플이 나올 확률의 log
        tanh_correction = torch.log(self.action_scale * (1 - z.pow(2)) + 1e-6)
        log_probs = (normal_log_probs - tanh_correction).sum(dim=-1)
        entropy_raw_actions = distribution.rsample()  # 수정
        entropy_z = torch.tanh(entropy_raw_actions)  # 수정
        raw_entropies = distribution.entropy().sum(dim=-1)  # 수정
        log_jacobians = torch.log(self.action_scale * (1 - entropy_z.pow(2)) + 1e-6).sum(dim=-1)  # 수정
        entropies = raw_entropies + log_jacobians  # 수정
        # a = action_bias + action_scale * tanh(raw_actions)
        # da/d(raw_actions) = action_scale * (1 - tanh²(raw_actions)) = action_scale * (1 - z²)
        # p(a) = p(raw_actions) / |da/d(raw_actions)|
        # log π(a|s) = normal_log_probs - log(action_scale * (1 - z²))
        # action dimension만 합

        # log(da/du) = c(1-z^2)
        #entropies = raw_entropies + log_jacobians=>E(log(dA/dU))

        return env_actions, log_probs, entropies, values

    def deterministic_action(self, states):
        mean, _, _ = self.forward(states)
        z = torch.tanh(mean)
        env_actions = self.action_bias + self.action_scale * z
        return env_actions

def make_env(env_id, seed, worker_id):
    worker_seed = seed+ worker_id

    def thunk():
        env = gym.make(env_id)
        env.reset(seed=worker_seed)
        env.action_space.seed(worker_seed)
        return env
    return thunk

def make_vector_env(env_id, num_envs, seed):
    env_fns = []
    for worker in range(num_envs):
        thunk = make_env(env_id, seed, worker)
        env_fns.append(thunk)
    return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)  # 수정

def collect_rollout(envs, model, states, num_steps, device):
    rewards_list = []
    terminateds_list = []  # 수정
    truncateds_list = []  # 수정
    truncated_bootstrap_values_list = []  # 수정
    log_probs_list = []
    values_list=[]
    entropies_list=[]
    for _ in range(num_steps):
        state_tensor = torch.as_tensor(states, dtype = torch.float32, device = device)
        env_actions, log_probs, entropies, values = model.sample_action(state_tensor)
        env_actions = env_actions.detach().cpu().numpy()
        next_states, rewards, terminateds, truncateds, infos = envs.step(env_actions)
        truncated_bootstrap_values = torch.zeros(envs.num_envs, dtype=torch.float32, device=device)  # 수정
        if np.any(truncateds):  # 수정
            final_obs_mask = infos["_final_obs"]  # 수정
            final_states = np.stack(infos["final_obs"][final_obs_mask])  # 수정
            final_state_tensor = torch.as_tensor(final_states, dtype=torch.float32, device=device)  # 수정
            with torch.no_grad():  # 수정
                _, _, final_values = model(final_state_tensor)  # 수정
            final_obs_mask_tensor = torch.as_tensor(final_obs_mask, dtype=torch.bool, device=device)  # 수정
            truncated_bootstrap_values[final_obs_mask_tensor] = final_values  # 수정
        rewards_list.append(torch.as_tensor(rewards, dtype = torch.float32, device=device))
        terminateds_list.append(torch.as_tensor(terminateds, dtype=torch.bool, device=device))  # 수정
        truncateds_list.append(torch.as_tensor(truncateds, dtype=torch.bool, device=device))  # 수정
        truncated_bootstrap_values_list.append(truncated_bootstrap_values)  # 수정
        log_probs_list.append(log_probs)
        values_list.append(values)
        entropies_list.append(entropies)
        states = next_states

    rollout = {
        "rewards": torch.stack(rewards_list),
        "terminateds": torch.stack(terminateds_list), 
        "truncateds": torch.stack(truncateds_list), 
        "truncated_bootstrap_values": torch.stack(truncated_bootstrap_values_list),  # 수정
        "log_probs": torch.stack(log_probs_list),
        "values": torch.stack(values_list),
        "entropies": torch.stack(entropies_list)
    }
    return rollout, states

def compute_R_A(rewards, terminateds, truncateds, values, next_value, truncated_bootstrap_values, gamma, gae_lambda):  # 수정
    raw_advantages = torch.zeros_like(rewards)  
    gae = torch.zeros_like(next_value)  
    for step in reversed(range(rewards.shape[0])): 
        if step == rewards.shape[0] - 1:
            next_state_value = next_value 
        else: 
            next_state_value = values[step + 1].detach()  

        continuation_value = torch.where(truncateds[step], truncated_bootstrap_values[step], next_state_value)  # 수정
        not_terminated = 1.0 - terminateds[step].float()  # 수정
        delta = rewards[step] + gamma * continuation_value * not_terminated - values[step].detach()  # 수정
        not_episode_end = 1.0 - torch.logical_or(terminateds[step], truncateds[step]).float()  # 수정
        gae = delta + gamma * gae_lambda * not_episode_end * gae  # 수정
        raw_advantages[step] = gae  # 수정
    returns = raw_advantages + values.detach()  
    advantages = raw_advantages  
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)  
    #A정규화
    return returns, advantages

def compute_a2c_loss(rollout, returns, advantages, value_coef, entropy_coef):
    log_probs = rollout["log_probs"]
    values = rollout["values"]
    entropies = rollout["entropies"]
    policy_loss = -(log_probs * advantages.detach()).mean()
    value_loss = (returns - values).pow(2).mean()
    entropy = entropies.mean()
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    return total_loss, policy_loss, value_loss, entropy

def evaluate_policy(env_id, model, device, num_episodes):
    env = gym.make(env_id)
    episode_returns = []
    model.eval()
    with torch.no_grad():
        for _ in range(num_episodes):
            state, _ = env.reset()
            episode_return = 0.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                state_tensor = torch.as_tensor(state,dtype = torch.float32, device = device).unsqueeze(0)
                env_action = model.deterministic_action(state_tensor)
                action = env_action.squeeze(0).cpu().numpy()
                state, reward, terminated, truncated, _ = env.step(action)
                episode_return += float(reward)
            episode_returns.append(episode_return)
    env.close()
    model.train()
    return float(np.mean(episode_returns))


def save_checkpoint(model, optimizer, update_index, mean_eval_return, checkpoint_path):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "update_index": update_index,
        "mean_eval_return": mean_eval_return,
    }

    torch.save(checkpoint, checkpoint_path)

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    envs = make_vector_env(ENV_ID, NUM_ENVS, SEED)
    observation_dim = envs.single_observation_space.shape[0]  
    action_dim = envs.single_action_space.shape[0]            
    action_low = envs.single_action_space.low                
    action_high = envs.single_action_space.high             
    model = ActorCritic(observation_dim, action_dim, HIDDEN_DIM, action_low, action_high)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    states, _ = envs.reset(seed=SEED)
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    last_eval_return = float("nan")
    for update_index in range(1, num_updates + 1):
        model.train()
        rollout, states = collect_rollout(envs, model, states, NUM_STEPS, DEVICE)
        next_state_tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            _, _, next_value = model(next_state_tensor)
        returns, advantages = compute_R_A(rollout["rewards"], rollout["terminateds"], rollout["truncateds"], rollout["values"], next_value, rollout["truncated_bootstrap_values"], GAMMA, GAE_LAMBDA)  # 수정
        total_loss, policy_loss, value_loss, entropy = compute_a2c_loss(rollout, returns, advantages, VALUE_COEF, ENTROPY_COEF)
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()

        if update_index % EVAL_INTERVAL_UPDATES == 0:
            last_eval_return = evaluate_policy(ENV_ID, model, DEVICE, NUM_EVAL_EPISODES)
            print(f"update: {update_index} | policy loss: {policy_loss.item():.3f} | value loss: {value_loss.item():.3f} | entropy: {entropy.item():.3f} | eval return: {last_eval_return:.1f}")
            if last_eval_return >= SOLVED_MEAN_RETURN:
                save_checkpoint(model, optimizer, update_index, last_eval_return, LATEST_CHECKPOINT_PATH)
                save_checkpoint(model, optimizer, update_index, last_eval_return, SOLVED_CHECKPOINT_PATH)
                break
        if update_index % SAVE_INTERVAL_UPDATES == 0:
            save_checkpoint(model, optimizer, update_index, last_eval_return, LATEST_CHECKPOINT_PATH)
    envs.close()

if __name__ == "__main__":
    train()
