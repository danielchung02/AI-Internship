import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.distributions import Normal
import torch.nn.functional as F


ENV_ID = "Pendulum-v1"
SEED = 42
NUM_ENVS = 8
NUM_STEPS = 64
TOTAL_TIMESTEPS = 1000000
GAMMA = 0.99
LEARNING_RATE = 3e-4
HIDDEN_DIM = 128
VALUE_COEF = 0.1
ENTROPY_COEF = 0.0001
MAX_GRAD_NORM = 10
SAVE_INTERVAL_UPDATES = 50
EVAL_INTERVAL_UPDATES = 100
NUM_EVAL_EPISODES = 10
SOLVED_MEAN_RETURN = -200.0
LATEST_CHECKPOINT_PATH = Path("latest.pt")
SOLVED_CHECKPOINT_PATH = Path("solved.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim, action_low, action_high): #low high없앰
            super().__init__()
            '''
            self.feature_net = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.critic = nn.Linear(hidden_dim, 1)
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
            #action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
            #action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)
            '''
            #변경
            self.actor_mean = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
            self.critic = nn.Sequential(
                nn.Linear(observation_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self,states):
          #features = self.feature_net(states)
          raw_mean = self.actor_mean(states)
          mean = 2.0 * raw_mean / (1.0 + raw_mean.abs()) #softsign
          #mean = 2.0 * (2 * torch.sigmoid(raw_mean) - 1) #sigmoid
          std = torch.exp(self.log_std).expand_as(mean)
          values = self.critic(states).squeeze(dim = -1)
          return mean,std, values

    def sample_action(self, states):
          mean,std,values = self.forward(states)
          distribution = Normal(mean,std)
          raw_actions = distribution.sample()
          env_actions = torch.clamp(raw_actions, min= -2.0, max=2.0)
          #env_actions = 2.0 * (2 * torch.sigmoid(raw_actions) - 1)
          log_probs = distribution.log_prob(raw_actions).sum(dim = -1)
          entropies = distribution.entropy().sum(dim = -1)
          return env_actions, log_probs, entropies, values

    def deterministic_action(self, states):
          mean, std, values = self.forward(states)
          env_actions = torch.clamp(mean, min = -2.0, max = 2.0)
          #env_actions = 2*(2*torch.sigmoid(mean) - 1)
          return env_actions

def make_env(env_id, seed, worker_id):
          worker_seed = seed+worker_id
          def thunk():
                env = gym.make(env_id)
                env.reset(seed = worker_seed)
                env.action_space.seed(worker_seed)
                return env
          return thunk
    
def make_vector_env(env_id, num_envs, seed):
          env_fns = []
          for worker_id in range(num_envs):
                env_fns.append(make_env(env_id,seed, worker_id))
          return gym.vector.SyncVectorEnv(env_fns, autoreset_mode=gym.vector.AutoresetMode.SAME_STEP)
def collect_rollout(envs:gym.vector.SyncVectorEnv, model, states, num_steps, device):
          rewards_list = []
          terminateds_list = []
          truncateds_list = []
          truncated_bootstrapvalues_list = []
          log_probs_list = []
          value_list = []
          entropies_list = []
          for _ in range(num_steps):
                #변경
                states_tensor = torch.as_tensor(states, dtype=torch.float32,device=device).clone()
                states_tensor[..., 2] /= 8.0
                envs_actions, log_probs, entropies, values = model.sample_action(states_tensor)
                actions_for_env = envs_actions.detach().cpu().numpy()
                next_states, rewards, terminateds, truncateds, infos = envs.step(actions_for_env)
                truncated_bootstrap_values = torch.zeros(envs.num_envs, dtype = torch.float32, device = device)
                if np.any(truncateds):
                      final_observation_mask = infos["_final_obs"]
                      final_states = np.stack(infos["final_obs"][final_observation_mask])
                      #변경
                      final_state_tensor = torch.as_tensor(final_states, dtype=torch.float32, device=device).clone()
                      final_state_tensor[..., 2] /= 8.0
                      with torch.no_grad():
                            _,_, final_values = model.forward(final_state_tensor)
                      final_observation_mask_tensor = torch.as_tensor(final_observation_mask, dtype=torch.bool,device=device)
                      truncated_bootstrap_values[final_observation_mask_tensor] = final_values
                      
                rewards_list.append(torch.as_tensor(rewards*0.1, dtype=torch.float32, device=device))
                #critic gradient좀 낮추자
                terminateds_list.append(torch.as_tensor(terminateds, dtype=torch.bool, device=device))
                truncateds_list.append(torch.as_tensor(truncateds, dtype=torch.bool, device=device))
                truncated_bootstrapvalues_list.append(truncated_bootstrap_values)
                log_probs_list.append(log_probs)
                value_list.append(values)
                entropies_list.append(entropies)
                states = next_states
          rollout = {
                        "rewards": torch.stack(rewards_list),
                        "terminateds": torch.stack(terminateds_list),
                        "truncateds": torch.stack(truncateds_list),
                        "truncated_bootstrap_values": torch.stack(truncated_bootstrapvalues_list),
                        "log_probs": torch.stack(log_probs_list),
                        "values": torch.stack(value_list),
                        "entropies": torch.stack(entropies_list), #다 32,4
               }
          return rollout, states

def compute_R_A(rewards, terminateds, truncateds, values, next_value, truncated_bootstrap_values, gamma):
          returns = torch.zeros_like(rewards)
          R = next_value
          for step in reversed(range(rewards.shape[0])): #rewards는 (32,4)
                continuation_value = torch.where(truncateds[step],truncated_bootstrap_values[step], R)
                not_terminated = 1.0 - terminateds[step].float()
                R = rewards[step] + gamma * continuation_value * not_terminated
                returns[step] = R
          advantages = returns - values
          advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
          return returns, advantages
    
def compute_a2c_loss(rollout, returns, advantages, value_coef, entropy_coef):
        log_probs = rollout["log_probs"]
        values = rollout["values"]
        entropies = rollout["entropies"]
        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = (returns - values).pow(2).mean() #(32,4)에 있는 128개의 원소를 모두 평균 냄
        #value_loss = F.smooth_l1_loss(values, returns.detach()) #valueloss에 민감하기에
        entropy = entropies.mean()
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        return total_loss, policy_loss, value_loss, entropy


def evaluate_policy(env_id, model: ActorCritic , device, num_episodes ): #평가할 에피소드개수(10)
    env = gym.make(env_id)
    episode_returns = []
    model.eval()

    try:
        with torch.no_grad():
            for episode_index in range(num_episodes):
                state, _ = env.reset(seed=SEED + episode_index) 
                episode_return = 0.0
                terminated = False
                truncated = False

                while not (terminated or truncated):
                    #변경
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).clone() # (1,4,3)
                    state_tensor[..., 2] /= 8.0
                    state_tensor = state_tensor.unsqueeze(0)
                    action = model.deterministic_action(state_tensor).squeeze(0).cpu().numpy() #(4,1)
                    state, reward, terminated, truncated, _ = env.step(action)
                    episode_return += float(reward)

                episode_returns.append(episode_return)
    finally:
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
    observation_dim = envs.single_observation_space.shape[0] #3
    action_dim = envs.single_action_space.shape[0] #1
    action_low = envs.single_action_space.low #-2
    action_high = envs.single_action_space.high #2
    model = ActorCritic(observation_dim, action_dim, HIDDEN_DIM, action_low, action_high).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    states, _ = envs.reset(seed=SEED)
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    last_eval_return = float("nan")

    try:
        for update_index in range(1, num_updates + 1):
            model.train()
            rollout, states = collect_rollout(envs, model, states, NUM_STEPS, DEVICE)
            #변경
            next_state_tensor = torch.as_tensor(states, dtype=torch.float32, device=DEVICE).clone()
            next_state_tensor[..., 2] /= 8.0

            with torch.no_grad():
                _, _, next_value = model.forward(next_state_tensor)

            returns, advantages = compute_R_A(
                rollout["rewards"],
                rollout["terminateds"],
                rollout["truncateds"],
                rollout["values"],
                next_value,
                rollout["truncated_bootstrap_values"],
                GAMMA
            )
            total_loss, policy_loss, value_loss, entropy = compute_a2c_loss(rollout,returns,advantages,VALUE_COEF,ENTROPY_COEF)
            optimizer.zero_grad()
            total_loss.backward()
                #로그추가
            #actor_grad_norm = model.actor_mean.weight.grad.norm().item()
            #critic_grad_norm = model.critic.weight.grad.norm().item()
            #feature_grad_norm = model.feature_net[0].weight.grad.norm().item()
            actor_grad_norm = model.actor_mean[-1].weight.grad.norm().item()
            critic_grad_norm = model.critic[-1].weight.grad.norm().item()
            actor_feature_grad_norm = model.actor_mean[0].weight.grad.norm().item()

            critic_feature_grad_norm = model.critic[0].weight.grad.norm().item()
            log_std_grad_norm = model.log_std.grad.norm().item()

            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            )
                #로그추가
            optimizer.step()

            if update_index % EVAL_INTERVAL_UPDATES == 0:
                last_eval_return = evaluate_policy(ENV_ID, model, DEVICE, NUM_EVAL_EPISODES)
                #추가
                with torch.no_grad():
                    debug_mean, debug_std, debug_values = model(next_state_tensor)

                reward_mean = rollout["rewards"].mean().item()
                reward_std = rollout["rewards"].std(unbiased=False).item()

                return_mean = returns.mean().item()
                return_std = returns.std(unbiased=False).item()

                value_mean = rollout["values"].detach().mean().item()
                value_std = rollout["values"].detach().std(unbiased=False).item()

                mean_abs = debug_mean.abs().mean().item()
                mean_at_or_over_limit = (debug_mean.abs() >= 2.0).float().mean().item()
                std_mean = debug_std.mean().item()
                #추가

                print(
                    f"update: {update_index} | "
                    f"policy loss: {policy_loss.item():.3f} | "
                    f"value loss: {value_loss.item():.3f} | "
                    f"entropy: {entropy.item():.3f} | "
                    f"eval return: {last_eval_return:.1f}"
                    f"reward μ/σ: {reward_mean:.2f}/{reward_std:.2f} | "
                    #추가
                    f"return μ/σ: {return_mean:.1f}/{return_std:.1f} | "
                    f"value μ/σ: {value_mean:.1f}/{value_std:.1f} | "
                    f"policy |μ|/σ: {mean_abs:.3f}/{std_mean:.3f} | "
                    f"mean≥2: {mean_at_or_over_limit:.2%} | "
                    f"grad actor/critic/actor_net/critic_net/std/all: "
                    f"{actor_grad_norm:.3g}/{critic_grad_norm:.3g}/"
                    f"{actor_feature_grad_norm:.3g}/{critic_feature_grad_norm:.3g}/"
                    f"{log_std_grad_norm:.3g}/{total_grad_norm:.3g} | "
                )

                if last_eval_return >= SOLVED_MEAN_RETURN:
                    save_checkpoint(model, optimizer, update_index, last_eval_return, LATEST_CHECKPOINT_PATH)
                    save_checkpoint(model, optimizer, update_index, last_eval_return, SOLVED_CHECKPOINT_PATH)
                    break

            if update_index % SAVE_INTERVAL_UPDATES == 0:
                save_checkpoint(model, optimizer, update_index, last_eval_return, LATEST_CHECKPOINT_PATH)
    finally:
        envs.close()


if __name__ == "__main__":
    train()

            
        
