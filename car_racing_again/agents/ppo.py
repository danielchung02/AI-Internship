import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions.normal
import numpy as np
from torch.distributions import Normal


# PPO_CLIP_COEF = 0.2  # delete
# PPO_EPOCHS = 10  # delete
# MINIBATCH_SIZE = 256  # delete
# VALUE_COEF = 0.5  # delete
# ENTROPY_COEF = 0.0  # delete
# MAX_GRAD_NORM = 0.5  # delete

class ActorCritic(nn.Module):
    # def __init__(self, encoder, feature_dim, action_dim ,action_low, action_high):  # delete
    # def __init__(self, encoder, feature_dim, action_dim, action_low, action_high, initial_log_std=0.0, initial_action_mean_bias=None):  # delete
    def __init__(self, encoder, feature_dim, action_dim, action_low, action_high, initial_log_std=0.0, initial_action_mean_bias=None, head_dim=None):  # add
        super().__init__()
        self.encoder = encoder
        head_dim = feature_dim if head_dim is None else head_dim  # add
        # self.feature_layer = nn.Sequential(  # delete
        #     nn.Linear(feature_dim, feature_dim),  # delete
        #     nn.ReLU(),  # delete
        # )  # delete
        # self.actor_feature_layer = nn.Sequential(  # delete
        #     nn.Linear(feature_dim, feature_dim),  # delete
        self.actor_feature_layer = nn.Sequential(  # add
            nn.Linear(feature_dim, head_dim),  # add
            nn.ReLU(),  # add
        )  # add
        # self.critic_feature_layer = nn.Sequential(  # delete
        #     nn.Linear(feature_dim, feature_dim),  # delete
        self.critic_feature_layer = nn.Sequential(  # add
            nn.Linear(feature_dim, head_dim),  # add
            nn.ReLU(),  # add
        )  # add
        # self.actor_mean = nn.Linear(feature_dim, action_dim) #action dim=3  # delete
        # self.critic = nn.Linear(feature_dim, 1)  # delete
        self.actor_mean = nn.Linear(head_dim, action_dim)  # add
        self.critic = nn.Linear(head_dim, 1)  # add
        # self.log_std = nn.Parameter(torch.full((action_dim,), -1.5))  # delete
        self.log_std = nn.Parameter(torch.full((action_dim,), initial_log_std))  # add
        # self.register_buffer("action_low", torch.as_tensor(action_low, dtype=torch.float32))  # delete
        # self.register_buffer("action_high", torch.as_tensor(action_high, dtype=torch.float32))  # delete
        self.register_buffer("action_scale", (torch.as_tensor(action_high, dtype=torch.float32) - torch.as_tensor(action_low, dtype=torch.float32)) / 2.0)  # add
        self.register_buffer("action_bias", (torch.as_tensor(action_high, dtype=torch.float32) + torch.as_tensor(action_low, dtype=torch.float32)) / 2.0)  # add
        #추가
        self.apply(self._initialize)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.zeros_(self.actor_mean.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)
        if initial_action_mean_bias is not None:  # add
            if len(initial_action_mean_bias) != action_dim:  # add
                raise ValueError("initial_action_mean_bias has the wrong length")  # add
            with torch.no_grad():  # add
                self.actor_mean.bias.copy_(torch.as_tensor(initial_action_mean_bias, dtype=torch.float32))  # add

    @staticmethod
    def _initialize(module):
        # if isinstance(module, nn.Linear):  # delete
        if isinstance(module, (nn.Linear, nn.Conv2d)):  # add
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
            nn.init.zeros_(module.bias)

    def forward(self, states): #states(64,15)
        # encoded_states = self.encoder(states)  # delete
        # features = self.feature_layer(encoded_states)  #(64,256)  # delete
        # mean = self.actor_mean(features) #64,3  # delete
        # std = torch.exp(self.log_std).expand_as(mean) #64,3  # delete
        # values = self.critic(features).squeeze(dim=-1) #64,  # delete
        encoded_states = self.encoder(states)  # add
        actor_features = self.actor_feature_layer(encoded_states)  # add
        critic_features = self.critic_feature_layer(encoded_states)  # add
        mean = self.actor_mean(actor_features)  # add
        std = torch.exp(torch.clamp(self.log_std, -3.0, 1.0)).expand_as(mean)  # add
        values = self.critic(critic_features).squeeze(dim=-1)  # add
        return mean,std, values


    def sample_action(self, states):
        mean, std, values = self.forward(states)
        distribution = Normal(mean,std)
        raw_actions = distribution.sample() #64,3
        #env_actions = torch.maximum(torch.minimum(raw_actions, self.action_high), self.action_low)
        #log_probs = distribution.log_prob(raw_actions).sum(dim=-1)

        # unit_actions = torch.sigmoid(raw_actions)  # delete
        # action_scale = self.action_high - self.action_low  # delete
        # env_actions = self.action_low + action_scale * unit_actions  # delete
        # log_probs = distribution.log_prob(raw_actions)  # delete
        # log_probs -= torch.log(action_scale * unit_actions * (1.0 - unit_actions) + 1e-6)  # delete
        # log_probs = log_probs.sum(dim=-1)  # delete
        squashed_actions = torch.tanh(raw_actions)  # add
        env_actions = self.action_bias + self.action_scale * squashed_actions  # add
        log_probs = distribution.log_prob(raw_actions)  # add
        log_probs -= torch.log(self.action_scale * (1.0 - squashed_actions.square()) + 1e-6)  # add
        log_probs = log_probs.sum(dim=-1)  # add

        entropies = distribution.entropy().sum(dim=-1) #64,
        return env_actions, raw_actions, log_probs, entropies, values

    def evaluate_actions(self, states, raw_actions):
        mean,std,values = self.forward(states)
        distribution = Normal(mean, std)
        #log_probs = distribution.log_prob(raw_actions).sum(dim = -1)
        # unit_actions = torch.sigmoid(raw_actions)  # delete
        # action_scale = self.action_high - self.action_low  # delete
        # log_probs = distribution.log_prob(raw_actions)  # delete
        # log_probs -= torch.log(action_scale * unit_actions * (1.0 - unit_actions) + 1e-6)  # delete
        # log_probs = log_probs.sum(dim=-1)  # delete
        squashed_actions = torch.tanh(raw_actions)  # add
        log_probs = distribution.log_prob(raw_actions)  # add
        log_probs -= torch.log(self.action_scale * (1.0 - squashed_actions.square()) + 1e-6)  # add
        log_probs = log_probs.sum(dim=-1)  # add

        entropies = distribution.entropy().sum(dim = -1)
        return log_probs, entropies, values

    def deterministic_actions(self, states):
        mean, _,_ = self.forward(states)
        #env_actions = torch.maximum(torch.minimum(mean, self.action_high), self.action_low)
        # unit_actions = torch.sigmoid(mean)  # delete
        # action_scale = self.action_high - self.action_low  # delete
        # env_actions = self.action_low + action_scale * unit_actions  # delete
        env_actions = self.action_bias + self.action_scale * torch.tanh(mean)  # add
        return env_actions

        
    def act(self, states, deterministic = False):
        if deterministic == True:
            env_actions = self.deterministic_actions(states)
            _,_, values = self.forward(states)
            return env_actions, None, values
        env_actions, _, log_probs, _, values = self.sample_action(states)
        return env_actions, log_probs, values

    def get_value(self, states):
        _, _, values = self.forward(states)
        return values


# def collect_rollout(env, model,state, num_steps, device, action_repeat, episode_return):  # delete
def collect_rollout(env, model, state, num_steps, device, action_repeat, gamma, episode_return):  # add
    state_list = []
    raw_action_list = []
    reward_list = []
    terminated_list = []
    truncated_list = []
    truncated_bootstrap_value_list = []
    bootstrap_discount_list = []  # add
    log_prob_list = []
    value_list = []
    completed_episode_returns = []

    for _ in range(num_steps): #2048
        state_tensor = torch.as_tensor(state, dtype = torch.float32, device = device).unsqueeze(0)
        with torch.no_grad():
            env_action, raw_action, log_prob, _, value = model.sample_action(state_tensor)
        action_for_env = env_action.squeeze(0).cpu().numpy()
        total_reward = 0.0
        terminated = False
        truncated = False
        next_state = state
        bootstrap_discount = 1.0  # add
        for _ in range(action_repeat):
             next_state, reward, terminated, truncated, _  = env.step(action_for_env)
             # total_reward += reward  # delete
             total_reward += bootstrap_discount * reward  # add
             bootstrap_discount *= gamma  # add
             if terminated or truncated:
                  break
             
        done = terminated or truncated
        truncated_bootstrap_value = torch.tensor(0.0, dtype=torch.float32, device=device)
        if truncated:
            next_state_tensor = torch.as_tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                truncated_bootstrap_value = model.get_value(next_state_tensor).squeeze(0)

        state_list.append(state_tensor.squeeze(0))
        raw_action_list.append(raw_action.squeeze(0))
        reward_list.append(torch.tensor(total_reward,dtype = torch.float32, device = device))
        terminated_list.append(torch.tensor(float(terminated), dtype=torch.float32, device=device))
        truncated_list.append(torch.tensor(float(truncated), dtype=torch.float32, device=device))
        truncated_bootstrap_value_list.append(truncated_bootstrap_value)
        bootstrap_discount_list.append(torch.tensor(bootstrap_discount, dtype=torch.float32, device=device))  # add
        log_prob_list.append(log_prob.squeeze(0))
        value_list.append(value.squeeze(0))

        episode_return += total_reward

        if done:
            completed_episode_returns.append(episode_return)
            state, _ = env.reset()
            episode_return = 0.0
        else:
            state = next_state

    rollout = {
        "states": torch.stack(state_list),
        "raw_actions": torch.stack(raw_action_list),
        "rewards": torch.stack(reward_list),
        "terminateds": torch.stack(terminated_list),
        "truncateds": torch.stack(truncated_list),
        "truncated_bootstrap_values": torch.stack(truncated_bootstrap_value_list),
        "bootstrap_discounts": torch.stack(bootstrap_discount_list),  # add
        "log_probs": torch.stack(log_prob_list),
        "values": torch.stack(value_list)
    }
    return rollout, state, episode_return, completed_episode_returns


        
# def compute_R_A(rewards, terminateds, truncateds, values, next_value, truncated_bootstrap_values, gamma):  # delete
def compute_R_A(rewards, terminateds, truncateds, values, next_value, truncated_bootstrap_values, bootstrap_discounts):  # add
    returns = torch.zeros_like(rewards)
    R = next_value

    for step in reversed(range(rewards.shape[0])):
        continuation_value = torch.where(
            truncateds[step].bool(),
            truncated_bootstrap_values[step],
            R,
        )

        not_terminated = 1.0 - terminateds[step]
        # R = rewards[step] + gamma * continuation_value * not_terminated  # delete
        R = rewards[step] + bootstrap_discounts[step] * continuation_value * not_terminated  # add
        returns[step] = R

    advantages = returns - values

    advantages = (advantages - advantages.mean())/(advantages.std(unbiased=False) + 1e-8)

    return returns, advantages


def compute_n_step_R_A(rewards, terminateds, truncateds, values, next_value, truncated_bootstrap_values, bootstrap_discounts, n_step):  # add
    """Fixed n-step TD returns; this is not GAE."""  # add
    returns = torch.zeros_like(rewards)  # add
    rollout_length = rewards.shape[0]  # add

    for start_step in range(rollout_length):  # add
        R = torch.zeros((), dtype=rewards.dtype, device=rewards.device)  # add
        discount = torch.ones((), dtype=rewards.dtype, device=rewards.device)  # add
        for offset in range(n_step):  # add
            step = start_step + offset  # add
            if step >= rollout_length:  # add
                R = R + discount * next_value  # add
                break  # add

            R = R + discount * rewards[step]  # add
            next_discount = discount * bootstrap_discounts[step]  # add
            if terminateds[step].bool():  # add
                break  # add
            if truncateds[step].bool():  # add
                R = R + next_discount * truncated_bootstrap_values[step]  # add
                break  # add
            if offset + 1 == n_step:  # add
                bootstrap_value = values[step + 1] if step + 1 < rollout_length else next_value  # add
                R = R + next_discount * bootstrap_value  # add
                break  # add
            if step + 1 == rollout_length:  # add
                R = R + next_discount * next_value  # add
                break  # add
            discount = next_discount  # add
        returns[start_step] = R  # add

    advantages = returns - values  # add
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)  # add
    return returns, advantages  # add

def compute_ppo_loss(model, state, raw_action, old_log_prob, returns, advantage, clip_coef, value_coef, entropy_coef):
    new_log_prob, entropy, value =model.evaluate_actions(state, raw_action)
    log_ratio = new_log_prob - old_log_prob
    ratio = torch.exp(log_ratio)
    surrogate1 = ratio * advantage
    surrogate2 = torch.clamp(ratio, 1.0-clip_coef, 1.0+clip_coef)* advantage
    policy_loss = -torch.minimum(surrogate1, surrogate2).mean()
    value_loss = (returns - value).pow(2).mean()
    entropy = entropy.mean()
    approx_kl = (ratio - 1.0 - log_ratio).mean()  # add

    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    # return total_loss, policy_loss, value_loss, entropy  # delete
    return total_loss, policy_loss, value_loss, entropy, approx_kl  # add


# def update_ppo(model, optimizer, rollout, returns, advantage):  # delete
def update_ppo(model, optimizer, rollout, returns, advantage, config):  # add
    state = rollout["states"]
    raw_action = rollout["raw_actions"]
    old_log_prob = rollout["log_probs"]
    batch_size = state.shape[0]
    minibatch_size = max(1, batch_size // config.num_minibatches)  # add
    policy_losses = []
    value_losses = []
    entropies = []
    approx_kls = []  # add
    stop_early = False  # add

    # for _ in range(PPO_EPOCHS): #10  # delete
    for _ in range(config.update_epochs):  # add
        indices = torch.randperm(batch_size, device=state.device)

        # for start in range(0, batch_size, MINIBATCH_SIZE):  # delete
        for start in range(0, batch_size, minibatch_size):  # add
            # minibatch_indices = indices[start:start + MINIBATCH_SIZE]  # delete
            minibatch_indices = indices[start:start + minibatch_size]  # add

            # total_loss, policy_loss, value_loss, entropy = compute_ppo_loss(  # delete
            total_loss, policy_loss, value_loss, entropy, approx_kl = compute_ppo_loss(  # add
                model,
                state[minibatch_indices],
                raw_action[minibatch_indices],
                old_log_prob[minibatch_indices],
                returns[minibatch_indices],
                advantage[minibatch_indices],
                # PPO_CLIP_COEF,  # delete
                config.clip_coef,  # add
                # VALUE_COEF,  # delete
                config.value_coef,  # add
                # ENTROPY_COEF  # delete
                config.entropy_coef,  # add
            )

            optimizer.zero_grad()
            total_loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  # delete
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)  # add
            optimizer.step()

            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())
            entropies.append(entropy.item())
            approx_kls.append(approx_kl.item())  # add
            if config.target_kl is not None and approx_kl.item() > 1.5 * config.target_kl:  # add
                stop_early = True  # add
                break  # add

        if stop_early:  # add
            break  # add

    metrics = {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "approx_kl": float(np.mean(approx_kls)),  # add
    }

    return metrics


        
        

