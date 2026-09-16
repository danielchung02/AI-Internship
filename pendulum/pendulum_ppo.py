import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.distributions import Normal


ENV_ID = "Pendulum-v1"
SEED = 42
NUM_STEPS = 2048
TOTAL_TIMESTEPS = 1000000
GAMMA = 0.99
LEARNING_RATE = 3e-4
HIDDEN_DIM = 128
VALUE_COEF = 0.05
ENTROPY_COEF = 0.0001
MAX_GRAD_NORM = 10
SAVE_INTERVAL_UPDATES = 50
EVAL_INTERVAL_UPDATES = 100
NUM_EVAL_EPISODES = 10
SOLVED_MEAN_RETURN = -200.0
LATEST_CHECKPOINT_PATH = Path("ppo_latest.pt")
SOLVED_CHECKPOINT_PATH = Path("pposolved.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#PPO params
PPO_CLIP_COEF = 0.2
PPO_EPOCHS = 10
MINIBATCH_SIZE = 64


class ActorCritic(nn.Module):
    def __init__(self, observation_dim, action_dim, hidden_dim, action_low, action_high): #low high없앰
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
            #action_low_tensor = torch.as_tensor(action_low, dtype=torch.float32)
            #action_high_tensor = torch.as_tensor(action_high, dtype=torch.float32)

    def forward(self,states):
          features = self.feature_net(states)
          raw_mean = self.actor_mean(features)
          mean = 2.0 * raw_mean / (1.0 + raw_mean.abs()) #softsign
          #mean = 2.0 * (2 * torch.sigmoid(raw_mean) - 1) #sigmoid
          std = torch.exp(self.log_std).expand_as(mean)
          values = self.critic(features).squeeze(dim = -1)
          return mean,std, values

    def sample_action(self, states):
          mean,std,values = self.forward(states)
          distribution = Normal(mean,std)
          raw_actions = distribution.sample()
          env_actions = torch.clamp(raw_actions, min= -2.0, max=2.0)
          #env_actions = 2.0 * (2 * torch.sigmoid(raw_actions) - 1)
          log_probs = distribution.log_prob(raw_actions).sum(dim = -1)
          entropies = distribution.entropy().sum(dim = -1)
          return env_actions, raw_actions, log_probs, entropies, values #raw_actions추가

    def evaluate_actions(self, states,raw_actions): # 함수 추가
          mean, std, values = self.forward(states)
          distribution = Normal(mean,std)
          log_probs = distribution.log_prob(raw_actions).sum(dim = -1) #new logprob
          entropies = distribution.entropy().sum(dim = -1)
          return log_probs, entropies, values
 
          
    def deterministic_action(self, states):
          mean, std, values = self.forward(states)
          env_actions = torch.clamp(mean, min = -2.0, max = 2.0)
          #env_actions = 2*(2*torch.sigmoid(mean) - 1)
          return env_actions

def make_env(env_id, seed):
     env = gym.make(env_id)
     state,_ = env.reset(seed = seed)
     env.action_space.seed(seed)
     return env, state
        
    
def collect_rollout(env, model, state, num_steps, device):
          state_list = []
          raw_action_list = [] #rollout data여러번 써야해서 state and action저장해야
          reward_list = []
          terminated_list = []
          truncated_list = []
          truncated_bootstrapvalue_list = []
          log_prob_list = []
          value_list = []
          
          for _ in range(num_steps):
                state_tensor = torch.as_tensor(state, dtype=torch.float32,device=device)
                with torch.no_grad():
                     env_action, raw_action, log_prob, entropy, value = model.sample_action(state_tensor)
                
                action_for_env = env_action.detach().cpu().numpy()
                next_state, reward, terminated, truncated, infos = env.step(action_for_env)
                truncated_bootstrap_value = torch.tensor(0.0, dtype = torch.float32, device = device)
                if truncated:
                    final_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=device)

                    with torch.no_grad():
                        _, _, truncated_bootstrap_value = model.forward(final_state_tensor)
                state_list.append(state_tensor)
                raw_action_list.append(raw_action)
                reward_list.append(torch.tensor(reward*0.1, dtype = torch.float32, device = device))
                terminated_list.append(torch.as_tensor(terminated, dtype=torch.bool, device=device))
                truncated_list.append(torch.as_tensor(truncated, dtype=torch.bool, device=device))
                truncated_bootstrapvalue_list.append(truncated_bootstrap_value)
                log_prob_list.append(log_prob)
                value_list.append(value)
                if terminated or truncated:
                     state, _ = env.reset()
                else: state = next_state

          rollout = {
                        "states": torch.stack(state_list),
                        "raw_actions": torch.stack(raw_action_list),
                        "rewards": torch.stack(reward_list),
                        "terminateds": torch.stack(terminated_list),
                        "truncateds": torch.stack(truncated_list),
                        "truncated_bootstrap_values": torch.stack(truncated_bootstrapvalue_list),
                        "log_probs": torch.stack(log_prob_list),
                        "values": torch.stack(value_list),
            
               }
          return rollout, state

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

#*
def compute_ppo_loss(model, state, raw_action, old_log_prob, returns, advantage,clip_coef, value_coef, entropy_coef):
        #state:(64,3) rawaction(64,1) oldlogprob(64,) advantage(64,) 
        new_log_prob, entropy, value= model.evaluate_actions(state, raw_action)
        log_ratio = new_log_prob - old_log_prob
        ratio = torch.exp(log_ratio)
        surrogate1 = ratio*advantage
        surrogate2 = torch.clamp(ratio, 1.0-clip_coef, 1.0+clip_coef) *advantage
        policy_loss = -torch.minimum(surrogate1, surrogate2).mean() #64개 평균
        value_loss = (returns - value).pow(2).mean()
        entropy = entropy.mean()
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        return total_loss, policy_loss, value_loss, entropy

#*
def update_ppo(model,optimizer, rollout, returns, advantage):
     state = rollout["states"]
     raw_action = rollout["raw_actions"]
     old_log_prob = rollout["log_probs"]
     batch_size = state.shape[0]
     policy_losses = []
     value_losses = []
     entropies = []
     for epoch in range(PPO_EPOCHS):
          indices = torch.randperm(batch_size, device = state.device)# shuffle
          for start in range(0, batch_size, MINIBATCH_SIZE):
               minibatch_indices = indices[start:start + MINIBATCH_SIZE]
               total_loss, policy_loss, value_loss, entropy = compute_ppo_loss(model,state[minibatch_indices],raw_action[minibatch_indices],
                    old_log_prob[minibatch_indices],returns[minibatch_indices], advantage[minibatch_indices],PPO_CLIP_COEF, VALUE_COEF, ENTROPY_COEF)
               optimizer.zero_grad()
               total_loss.backward()
               torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
               optimizer.step()

               policy_losses.append(policy_loss.item())
               value_losses.append(value_loss.item())
               entropies.append(entropy.item())

     metrics = {"policy_loss": np.mean(policy_losses),
                "value_loss": np.mean(value_losses),
                "entropy": np.mean(entropies)}
            

     return metrics
envs = make_env(ENV_ID, SEED)

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
                    state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)# (1,4,3)
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

    env, state = make_env(ENV_ID, SEED)
    observation_dim = env.observation_space.shape[0] 
    action_dim = env.action_space.shape[0] #1
    action_low = env.action_space.low #-2
    action_high = env.action_space.high #2
    model = ActorCritic(observation_dim, action_dim, HIDDEN_DIM, action_low, action_high).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    num_updates = TOTAL_TIMESTEPS //  NUM_STEPS
    last_eval_return = float("nan")

    try:
        for update_index in range(1, num_updates + 1):
            model.train()
            rollout, state = collect_rollout(env, model, state, NUM_STEPS, DEVICE)
            next_state_tensor = torch.as_tensor(state, dtype=torch.float32, device=DEVICE)

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

            metrics = update_ppo(model, optimizer, rollout, returns, advantages)

            if update_index % EVAL_INTERVAL_UPDATES == 0:
                last_eval_return = evaluate_policy(ENV_ID, model, DEVICE, NUM_EVAL_EPISODES)

                with torch.no_grad():
                    debug_mean, debug_std, debug_values = model(next_state_tensor)

                reward_mean = rollout["rewards"].mean().item()
                reward_std = rollout["rewards"].std(unbiased=False).item()

                return_mean = returns.mean().item()
                return_std = returns.std(unbiased=False).item()

                value_mean = rollout["values"].detach().mean().item()
                value_std = rollout["values"].detach().std(unbiased=False).item()

                mean_abs = debug_mean.abs().mean().item()
                std_mean = debug_std.mean().item()
                print(
                    f"update: {update_index} | "
                    f"timesteps: {update_index * NUM_STEPS} | "  # 추가
                    f"policy loss: {metrics['policy_loss']:.3f} | "  # 수정
                    f"value loss: {metrics['value_loss']:.3f} | "  # 수정
                    f"entropy: {metrics['entropy']:.3f} | "  # 수정
                    f"eval return: {last_eval_return:.1f} | "
                    f"reward μ/σ: {reward_mean:.2f}/{reward_std:.2f} | "
                    f"return μ/σ: {return_mean:.1f}/{return_std:.1f} | "
                    f"value μ/σ: {value_mean:.1f}/{value_std:.1f} | "
                    f"policy |μ|/σ: {mean_abs:.3f}/{std_mean:.3f} | "
                )

                if last_eval_return >= SOLVED_MEAN_RETURN:
                    save_checkpoint(model, optimizer, update_index, last_eval_return, LATEST_CHECKPOINT_PATH)
                    save_checkpoint(model, optimizer, update_index, last_eval_return, SOLVED_CHECKPOINT_PATH)

                    break


            if update_index % SAVE_INTERVAL_UPDATES == 0:
                save_checkpoint(model, optimizer, update_index, last_eval_return, LATEST_CHECKPOINT_PATH)


    finally:
        env.close() 

if __name__ == "__main__":
    train()

           