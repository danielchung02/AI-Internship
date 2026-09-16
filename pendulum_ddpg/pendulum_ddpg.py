import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ENV_ID = "Pendulum-v1"
SEED = 42
TOTAL_TIMESTEPS = 1000000
GAMMA = 0.99
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
HIDDEN_DIM = 128
BUFFER_CAPACITY = 100_000
BATCH_SIZE = 128
START_STEPS = 10_000 #초반 몇 step 동안 random action으로 탐험할지 이거 전까지는 actor(state)+explore가 아니라 random action
UPDATE_AFTER = 1_000 #초반 몇 step 동안 gradient update를 기다릴지 이거 전까지는 신경망 학습 안함
TAU = 0.005
EXPLORATION_NOISE_STD = 0.1
MAX_GRAD_NORM = 10.0
SAVE_INTERVAL_STEPS = 10_000
EVAL_INTERVAL_STEPS = 50_000 # add
NUM_EVAL_EPISODES = 10 # add
PT_DIR = Path("pt/pendulum_ddpg")
LATEST_CHECKPOINT_PATH = PT_DIR / "ddpg_latest.pt"
BEST_CHECKPOINT_PATH = PT_DIR / "ddpg_best.pt" # add
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim, action_low, action_high):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.register_buffer("action_scale", torch.as_tensor((action_high - action_low) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias", torch.as_tensor((action_high + action_low) / 2.0, dtype=torch.float32))

    def forward(self, states):
        return torch.tanh(self.net(states)) * self.action_scale + self.action_bias

class Critic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(observation_dim+ action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, states, actions):
        state_actions = torch.cat([states, actions], dim = -1)
        return self.net(state_actions).squeeze(dim = -1)

class ReplayBuffer:
    def __init__(self, observation_dim, action_dim, capacity):
        self.states = np.zeros((capacity, observation_dim), dtype = np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype = np.float32)
        self.rewards = np.zeros(capacity, dtype = np.float32)
        self.next_states = np.zeros((capacity, observation_dim), dtype = np.float32)
        self.truncateds = np.zeros(capacity, dtype=np.float32)
        self.terminateds = np.zeros(capacity, dtype=np.float32)
        self.capacity = capacity
        self.pointer = 0
        self.size = 0

    def add(self, state,action, reward, next_state, terminated, truncated):
        self.states[self.pointer] = state
        self.actions[self.pointer] = action
        self.rewards[self.pointer] = reward
        self.next_states[self.pointer] = next_state
        self.terminateds[self.pointer] = float(terminated) #terminated and truncated는 boolean이라서
        self.truncateds[self.pointer] = float(truncated)
        self.pointer = (self.pointer+1)%self.capacity
        self.size = min(self.size+1, self.capacity)

    def sample(self, batch_size, device):
        indices = np.random.randint(0, self.size, size = batch_size)
        return {
            "states": torch.as_tensor(self.states[indices], dtype=torch.float32, device=device),
            "actions": torch.as_tensor(self.actions[indices], dtype=torch.float32, device=device),
            "rewards": torch.as_tensor(self.rewards[indices], dtype=torch.float32, device=device),
            "next_states": torch.as_tensor(self.next_states[indices], dtype=torch.float32, device=device),
            "terminateds": torch.as_tensor(self.terminateds[indices], dtype=torch.float32, device=device),
            "truncateds": torch.as_tensor(self.truncateds[indices], dtype=torch.float32, device=device),
        }

    def __len__(self):
        return self.size

class DDPGAgent:
    def __init__(self, observation_dim, action_dim, hidden_dim,action_low, action_high, device):
        self.device = device
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.actor = Actor(observation_dim, action_dim, hidden_dim, action_low, action_high).to(device)
        self.critic = Critic(observation_dim, action_dim, hidden_dim).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device) #그냥 copy.copy를 하면 복사본을 수정했을떄 원본도 바뀌어 버린다
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = CRITIC_LEARNING_RATE)
        for parameter in self.target_actor.parameters():
            parameter.requires_grad_(False)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad_(False)

    def select_action(self, state, add_noise):
        state_tensor = torch.as_tensor(state, dtype = torch.float32, device = self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).cpu().numpy()
        if add_noise:
            action += np.random.normal(0.0, EXPLORATION_NOISE_STD, size=action.shape).astype(np.float32)
        return np.clip(action, self.action_low, self.action_high)

    @torch.no_grad()
    def soft_update(self, online_network, target_network):
        for online_parameter, target_parameter in zip(online_network.parameters(), target_network.parameters()): 
            target_parameter.mul_(1.0 - TAU).add_(TAU * online_parameter)
            #target += (1-TAU)* target_network + TAU * online_network 이건 왜 안되는거야?
    
    def update_ddpg(self, replay_buffer):
        batch = replay_buffer.sample(BATCH_SIZE, self.device)
        states, actions, rewards, next_states, terminateds, truncateds =batch["states"],batch["actions"],batch["rewards"],batch["next_states"],batch["terminateds"],batch["truncateds"]
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            target_qvalues = self.target_critic(next_states, target_actions)
            targets = rewards + GAMMA * (1-terminateds) * target_qvalues

        qvalues = self.critic(states, actions)
        critic_loss = (targets - qvalues).pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm =  float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM))
        self.critic_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM))
        self.actor_optimizer.step()

        for parameter in self.critic.parameters(): 
            parameter.requires_grad_(True)
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item(), "q_mean": qvalues.mean().item(), "target_q_mean": targets.mean().item(), "actor_grad_norm": actor_grad_norm, "critic_grad_norm": critic_grad_norm}

def normalize_state(state):
    normalized_state = np.array(state, dtype=np.float32, copy=True)
    normalized_state[..., 2] /= 8.0
    return normalized_state


def make_env(env_id, seed):
    env = gym.make(env_id)
    state, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, normalize_state(state)


def prepare_pt_dir():
    PT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(agent, timestep, checkpoint_path):
    checkpoint = {"actor_state_dict": agent.actor.state_dict(), "critic_state_dict": agent.critic.state_dict(), "target_actor_state_dict": agent.target_actor.state_dict(), "target_critic_state_dict": agent.target_critic.state_dict(), "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(), "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(), "timestep": timestep}
    torch.save(checkpoint, checkpoint_path)

def evaluate_policy(env_id, agent, num_episodes): # add
    eval_env = gym.make(env_id) # add
    episode_returns = [] # add
    agent.actor.eval() # add
    try: # add
        for episode_index in range(num_episodes): # add
            state, _ = eval_env.reset(seed=SEED + episode_index) # add
            state = normalize_state(state) # add
            episode_return, terminated, truncated = 0.0, False, False # add
            while not (terminated or truncated): # add
                action = agent.select_action(state, add_noise=False) # add
                next_state, reward, terminated, truncated, _ = eval_env.step(action) # add
                state = normalize_state(next_state) # add
                episode_return += float(reward) # add
            episode_returns.append(episode_return) # add
    finally: # add
        eval_env.close() # add
        agent.actor.train() # add
    return float(np.mean(episode_returns)) # add

def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    prepare_pt_dir()
    env, state = make_env(ENV_ID, SEED)
    observation_dim, action_dim = env.observation_space.shape[0], env.action_space.shape[0]
    agent = DDPGAgent(observation_dim, action_dim, HIDDEN_DIM, env.action_space.low, env.action_space.high, DEVICE)
    replay_buffer = ReplayBuffer(observation_dim, action_dim,BUFFER_CAPACITY)
    best_eval_return = float("-inf") # add
    metrics = {"actor_loss": float("nan"),
               "critic_loss": float("nan"),
               "q_mean": float("nan"),
               "target_q_mean": float("nan"),
               "actor_grad_norm": float("nan"),
               "critic_grad_norm": float("nan")}
    try:
        for timestep in range(1, TOTAL_TIMESTEPS+1):
            if timestep<=START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, add_noise = True)

            next_state, reward, terminated, truncated,_ = env.step(action)
            next_state = normalize_state(next_state)

            replay_buffer.add(state, action, reward, next_state, terminated, truncated)
            state = next_state
            if terminated or truncated:
                reset_state, _ = env.reset()
                state = normalize_state(reset_state)
            if timestep>= UPDATE_AFTER and len(replay_buffer)>= BATCH_SIZE:
                metrics = agent.update_ddpg(replay_buffer)
            if timestep % EVAL_INTERVAL_STEPS == 0: # add
                mean_eval_return = evaluate_policy(ENV_ID, agent, NUM_EVAL_EPISODES) # add
                if mean_eval_return > best_eval_return: # add
                    best_eval_return = mean_eval_return # add
                    save_checkpoint(agent, timestep, BEST_CHECKPOINT_PATH) # add
                print(f"timestep: {timestep} | mean eval return: {mean_eval_return:.1f} | best eval return: {best_eval_return:.1f}") # add
            if timestep % SAVE_INTERVAL_STEPS ==0:
                save_checkpoint(agent, timestep, LATEST_CHECKPOINT_PATH)
                print(f"timestep: {timestep} | actor loss: {metrics['actor_loss']:.3f} | critic loss: {metrics['critic_loss']:.3f} | Q/target Q: {metrics['q_mean']:.2f}/{metrics['target_q_mean']:.2f} | grad actor/critic: {metrics['actor_grad_norm']:.3g}/{metrics['critic_grad_norm']:.3g} | replay buffer: {len(replay_buffer)}")
    finally:
        env.close()

if __name__ == "__main__":
    train()
