import copy
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


ENV_ID = "HalfCheetah-v5" 
SEED = 42
TOTAL_TIMESTEPS = 1000000
GAMMA = 0.99
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
HIDDEN_DIM = 256 # penulum은 128
BUFFER_CAPACITY = 1_000_000 # 진자는 10만
BATCH_SIZE = 256 # 진자는 128
START_STEPS = 10_000
UPDATE_AFTER = 1_000 
TAU = 0.005
EXPLORATION_NOISE_STD = 0.1
POLICY_NOISE = 0.2 #td3 #target을 흔드는 정도
NOISE_CLIP = 0.5 #td3
POLICY_DELAY = 2 #td3
MAX_GRAD_NORM = 10.0
SAVE_INTERVAL_STEPS = 10_000  
EVAL_INTERVAL_STEPS = 50_000 # add
NUM_EVAL_EPISODES = 10 # add
# DDPG checkpoint를 저장하던 PT_DIR 경로를 지웠다. #td3
PT_DIR = Path("pt/halfcheetah_td3") #td3
# DDPG latest checkpoint 파일명을 정하던 줄을 지웠다. #td3
LATEST_CHECKPOINT_PATH = PT_DIR / "td3_latest.pt" #td3
# DDPG best checkpoint 파일명을 정하던 줄을 지웠다. #td3
BEST_CHECKPOINT_PATH = PT_DIR / "td3_best.pt" #td3
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

# 하나의 Q(s, a)만 계산하던 Critic 클래스를 지웠다. #td3
class TwinCritic(nn.Module): #td3
    def __init__(self, observation_dim, action_dim, hidden_dim):
        super().__init__()
        self.q1_net = nn.Sequential( #td3
            nn.Linear(observation_dim+ action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.q2_net = nn.Sequential( #td3
            nn.Linear(observation_dim+ action_dim, hidden_dim), #td3
            nn.ReLU(), #td3
            nn.Linear(hidden_dim, hidden_dim), #td3
            nn.ReLU(), #td3
            nn.Linear(hidden_dim, 1) #td3
        ) #td3

    def forward(self, states, actions):
        state_actions = torch.cat([states, actions], dim = -1)
        # 하나의 Q값만 반환하던 줄을 지웠다. #td3
        q1_values = self.q1_net(state_actions).squeeze(dim = -1) #td3
        q2_values = self.q2_net(state_actions).squeeze(dim = -1) #td3
        return q1_values, q2_values #td3

    def q1_forward(self, states, actions): #td3 #Q2는 actor loss에 직접 들어가지는 않고, Q1이 과대평가하지 않도록 견제만 하는 거임
        state_actions = torch.cat([states, actions], dim = -1) #td3
        return self.q1_net(state_actions).squeeze(dim = -1) #td3

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

# DDPGAgent 클래스를 지웠다. #td3
class TD3Agent: #td3
    def __init__(self, observation_dim, action_dim, hidden_dim,action_low, action_high, device):
        self.device = device
        self.action_low = np.asarray(action_low, dtype=np.float32)
        self.action_high = np.asarray(action_high, dtype=np.float32)
        self.action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32, device=device) #td3
        self.action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32, device=device) #td3
        self.actor = Actor(observation_dim, action_dim, hidden_dim, action_low, action_high).to(device)
        # Critic 하나를 생성하던 줄을 지웠다. #td3
        self.critic = TwinCritic(observation_dim, action_dim, hidden_dim).to(device) #td3
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr = ACTOR_LEARNING_RATE)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr = CRITIC_LEARNING_RATE)
        self.update_count = 0 #td3 이거 세개는 그냥 로그 출력을 위한 변수
        self.last_actor_loss = float("nan") #td3
        self.last_actor_grad_norm = float("nan") #td3
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
    
    # 매 update마다 actor와 target network를 갱신하던 update_ddpg 함수를 지웠다. #td3
    def update_td3(self, replay_buffer): #td3
        batch = replay_buffer.sample(BATCH_SIZE, self.device)
        states, actions, rewards, next_states, terminateds, truncateds =batch["states"],batch["actions"],batch["rewards"],batch["next_states"],batch["terminateds"],batch["truncateds"]
        with torch.no_grad():
            target_actions = self.target_actor(next_states)
            # target actor action을 그대로 target critic에 넣던 줄을 지웠다. #td3
            target_noise = (torch.randn_like(target_actions) * POLICY_NOISE).clamp(-NOISE_CLIP, NOISE_CLIP) #td3
            target_actions = target_actions + target_noise #td3
            target_actions = torch.max(torch.min(target_actions, self.action_high_tensor), self.action_low_tensor) #td3
            # target critic 하나의 Q값을 사용하던 줄을 지웠다. #td3
            target_q1_values, target_q2_values = self.target_critic(next_states, target_actions) #td3
            target_qvalues = torch.min(target_q1_values, target_q2_values) #td3
            targets = rewards + GAMMA * (1-terminateds) * target_qvalues

        # critic 하나의 Q값에 대한 loss를 계산하던 줄을 지웠다. #td3
        q1_values, q2_values = self.critic(states, actions) #td3
        critic_loss = (targets - q1_values).pow(2).mean() + (targets - q2_values).pow(2).mean() #td3
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm =  float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), MAX_GRAD_NORM))
        self.critic_optimizer.step()

        self.update_count += 1 #td3
        # 매 update마다 actor를 갱신하던 줄을 지웠다. #td3
        if self.update_count % POLICY_DELAY == 0: #td3
            for parameter in self.critic.parameters():
                parameter.requires_grad_(False)
            actor_loss = -self.critic.q1_forward(states, self.actor(states)).mean() #td3
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.last_actor_grad_norm = float(torch.nn.utils.clip_grad_norm_(self.actor.parameters(), MAX_GRAD_NORM)) #td3
            self.actor_optimizer.step()
            self.last_actor_loss = actor_loss.item() #td3

            for parameter in self.critic.parameters(): 
                parameter.requires_grad_(True)
            self.soft_update(self.actor, self.target_actor) #td3
            self.soft_update(self.critic, self.target_critic) #td3
        # Q 하나만 기록하던 metrics 반환 줄을 지웠다. #td3
        return {"actor_loss": self.last_actor_loss, "critic_loss": critic_loss.item(), "q1_mean": q1_values.mean().item(), "q2_mean": q2_values.mean().item(), "target_q_mean": targets.mean().item(), "actor_grad_norm": self.last_actor_grad_norm, "critic_grad_norm": critic_grad_norm} #td3


def make_env(env_id, seed):
    env = gym.make(env_id)
    state, _ = env.reset(seed=seed)
    env.action_space.seed(seed)
    return env, state 


def prepare_pt_dir():
    PT_DIR.mkdir(parents=True, exist_ok=True)

def save_checkpoint(agent, timestep, checkpoint_path):
    checkpoint = {"actor_state_dict": agent.actor.state_dict(), "critic_state_dict": agent.critic.state_dict(), 
                  "target_actor_state_dict": agent.target_actor.state_dict(), 
                  "target_critic_state_dict": agent.target_critic.state_dict(), 
                  "actor_optimizer_state_dict": agent.actor_optimizer.state_dict(), 
                  "critic_optimizer_state_dict": agent.critic_optimizer.state_dict(), "timestep": timestep}
    torch.save(checkpoint, checkpoint_path)

def evaluate_policy(env_id, agent, num_episodes): # add
    eval_env = gym.make(env_id) # add
    episode_returns = [] # add
    agent.actor.eval() # add
    try: # add
        for episode_index in range(num_episodes): # add
            state, _ = eval_env.reset(seed=SEED + episode_index) # add
            episode_return, terminated, truncated = 0.0, False, False # add
            while not (terminated or truncated): # add
                action = agent.select_action(state, add_noise=False) # add
                state, reward, terminated, truncated, _ = eval_env.step(action) # add
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
    # DDPGAgent를 생성하던 줄을 지웠다. #td3
    agent = TD3Agent(observation_dim, action_dim, HIDDEN_DIM, env.action_space.low, env.action_space.high, DEVICE) #td3
    replay_buffer = ReplayBuffer(observation_dim, action_dim,BUFFER_CAPACITY)
    best_eval_return = float("-inf") # add
    metrics = {"actor_loss": float("nan"),
               "critic_loss": float("nan"),
               # Q 하나의 평균을 초기화하던 줄을 지웠다. #td3
               "q1_mean": float("nan"), #td3
               "q2_mean": float("nan"), #td3
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

            replay_buffer.add(state, action, reward, next_state, terminated, truncated)
            state = next_state
            if terminated or truncated:
                reset_state, _ = env.reset()
                # state = normalize_state(reset_state)
                state = reset_state # add
            if timestep>= UPDATE_AFTER and len(replay_buffer)>= BATCH_SIZE:
                # DDPG의 update_ddpg를 호출하던 줄을 지웠다. #td3
                metrics = agent.update_td3(replay_buffer) #td3
            if timestep % EVAL_INTERVAL_STEPS == 0: # add
                mean_eval_return = evaluate_policy(ENV_ID, agent, NUM_EVAL_EPISODES) # add
                if mean_eval_return > best_eval_return: # add
                    best_eval_return = mean_eval_return # add
                    save_checkpoint(agent, timestep, BEST_CHECKPOINT_PATH) # add
                print(f"timestep: {timestep} | mean eval return: {mean_eval_return:.1f} | best eval return: {best_eval_return:.1f}") # add
            if timestep % SAVE_INTERVAL_STEPS ==0:
                save_checkpoint(agent, timestep, LATEST_CHECKPOINT_PATH)
                # Q 하나와 target Q만 출력하던 줄을 지웠다. #td3
                print(f"timestep: {timestep} | actor loss: {metrics['actor_loss']:.3f} | critic loss: {metrics['critic_loss']:.3f} | Q1/Q2/target Q: {metrics['q1_mean']:.2f}/{metrics['q2_mean']:.2f}/{metrics['target_q_mean']:.2f} | grad actor/critic: {metrics['actor_grad_norm']:.3g}/{metrics['critic_grad_norm']:.3g} | replay buffer: {len(replay_buffer)}") #td3
    finally:
        env.close()

if __name__ == "__main__":
    train()
