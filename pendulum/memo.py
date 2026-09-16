"""A2C experiment for Gymnasium Pendulum-v1.

This version deliberately does *not* use GAE or tanh-squashed actions.
It keeps n-step returns and clipped Gaussian actions, while making the
critic/actor interaction and action clipping observable.
"""

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal


ENV_ID = "Pendulum-v1"
SEED = 42

# 8 x 64 keeps the rollout batch at 512 samples but halves the n-step horizon
# compared with the original 4 x 128 configuration.
NUM_ENVS = 8
NUM_STEPS = 64
TOTAL_TIMESTEPS = 1_000_000

GAMMA = 0.99
LEARNING_RATE = 3e-4
HIDDEN_DIM = 128
VALUE_COEF = 0.1
ENTROPY_COEF = 1e-4
MAX_GRAD_NORM = 10.0

# Keep this False for the requested MSE experiment.  Set it True only for a
# separate Huber ablation; do not compare the numerical loss values directly.
USE_HUBER_VALUE_LOSS = False

SAVE_INTERVAL_UPDATES = 50
EVAL_INTERVAL_UPDATES = 100
NUM_EVAL_EPISODES = 10
SOLVED_MEAN_RETURN = -200.0

# Separate networks prevent a large critic gradient from changing the actor's
# representation.  This is independent of GAE and action squashing.
LATEST_CHECKPOINT_PATH = Path("memo_latest.pt")
SOLVED_CHECKPOINT_PATH = Path("memo_solved.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def states_to_tensor(states: np.ndarray, device: torch.device) -> torch.Tensor:
    """Scale only Pendulum's angular velocity; cos(theta) and sin(theta) stay unchanged."""
    normalized_states = np.asarray(states, dtype=np.float32).copy()
    normalized_states[..., 2] /= 8.0  # Pendulum-v1 max angular speed
    return torch.as_tensor(normalized_states, dtype=torch.float32, device=device)


class ActorCritic(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.critic_net = nn.Sequential(
            nn.Linear(observation_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, states: torch.Tensor):
        actor_features = self.actor_net(states)
        critic_features = self.critic_net(states)

        raw_mean = self.actor_mean(actor_features)
        # No tanh: retain the existing softsign mean and clipped Gaussian action.
        mean = 2.0 * raw_mean / (1.0 + raw_mean.abs())
        std = torch.exp(self.log_std).expand_as(mean)
        values = self.critic(critic_features).squeeze(-1)
        return mean, std, values

    def sample_action(self, states: torch.Tensor):
        mean, std, values = self.forward(states)
        distribution = Normal(mean, std)
        raw_actions = distribution.sample()
        env_actions = torch.clamp(raw_actions, min=-2.0, max=2.0)
        log_probs = distribution.log_prob(raw_actions).sum(dim=-1)
        entropies = distribution.entropy().sum(dim=-1)
        return env_actions, raw_actions, log_probs, entropies, values

    def deterministic_action(self, states: torch.Tensor):
        mean, _, _ = self.forward(states)
        return torch.clamp(mean, min=-2.0, max=2.0)


def make_env(env_id: str, seed: int, worker_id: int):
    worker_seed = seed + worker_id

    def thunk():
        env = gym.make(env_id)
        env.reset(seed=worker_seed)
        env.action_space.seed(worker_seed)
        return env

    return thunk


def make_vector_env(env_id: str, num_envs: int, seed: int):
    env_fns = [make_env(env_id, seed, worker_id) for worker_id in range(num_envs)]
    return gym.vector.SyncVectorEnv(
        env_fns,
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )


def collect_rollout(envs, model, states, num_steps: int, device: torch.device):
    rewards_list = []
    terminateds_list = []
    truncateds_list = []
    truncated_bootstrap_values_list = []
    log_probs_list = []
    values_list = []
    entropies_list = []
    clip_fractions_list = []

    for _ in range(num_steps):
        states_tensor = states_to_tensor(states, device)
        env_actions, raw_actions, log_probs, entropies, values = model.sample_action(
            states_tensor
        )
        actions_for_env = env_actions.detach().cpu().numpy()
        next_states, rewards, terminateds, truncateds, infos = envs.step(actions_for_env)

        truncated_bootstrap_values = torch.zeros(
            envs.num_envs, dtype=torch.float32, device=device
        )
        if np.any(truncateds):
            final_observation_mask = infos["_final_obs"]
            final_states = np.stack(infos["final_obs"][final_observation_mask])
            with torch.no_grad():
                _, _, final_values = model.forward(states_to_tensor(final_states, device))
            final_mask_tensor = torch.as_tensor(
                final_observation_mask, dtype=torch.bool, device=device
            )
            truncated_bootstrap_values[final_mask_tensor] = final_values

        # Keep rewards scaled for a numerically manageable critic.  Evaluation
        # deliberately reports the original, unscaled environment reward.
        rewards_list.append(
            torch.as_tensor(rewards * 0.1, dtype=torch.float32, device=device)
        )
        terminateds_list.append(
            torch.as_tensor(terminateds, dtype=torch.bool, device=device)
        )
        truncateds_list.append(
            torch.as_tensor(truncateds, dtype=torch.bool, device=device)
        )
        truncated_bootstrap_values_list.append(truncated_bootstrap_values)
        log_probs_list.append(log_probs)
        values_list.append(values)
        entropies_list.append(entropies)
        clip_fractions_list.append((raw_actions.abs() >= 2.0).float().mean())
        states = next_states

    rollout = {
        "rewards": torch.stack(rewards_list),
        "terminateds": torch.stack(terminateds_list),
        "truncateds": torch.stack(truncateds_list),
        "truncated_bootstrap_values": torch.stack(truncated_bootstrap_values_list),
        "log_probs": torch.stack(log_probs_list),
        "values": torch.stack(values_list),
        "entropies": torch.stack(entropies_list),
        "clip_fractions": torch.stack(clip_fractions_list),
    }
    return rollout, states


def compute_returns_and_advantages(
    rewards,
    terminateds,
    truncateds,
    values,
    next_value,
    truncated_bootstrap_values,
    gamma: float,
):
    """Plain n-step return estimator: intentionally no GAE."""
    returns = torch.zeros_like(rewards)
    running_return = next_value

    for step in reversed(range(rewards.shape[0])):
        continuation_value = torch.where(
            truncateds[step], truncated_bootstrap_values[step], running_return
        )
        not_terminated = 1.0 - terminateds[step].float()
        running_return = rewards[step] + gamma * continuation_value * not_terminated
        returns[step] = running_return

    raw_advantages = returns - values.detach()
    advantages = (raw_advantages - raw_advantages.mean()) / (
        raw_advantages.std(unbiased=False) + 1e-8
    )
    return returns, advantages


def compute_a2c_loss(rollout, returns, advantages):
    policy_loss = -(rollout["log_probs"] * advantages).mean()
    if USE_HUBER_VALUE_LOSS:
        value_loss = F.smooth_l1_loss(rollout["values"], returns.detach())
    else:
        value_loss = F.mse_loss(rollout["values"], returns.detach())
    entropy = rollout["entropies"].mean()
    total_loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
    return total_loss, policy_loss, value_loss, entropy


def evaluate_policy(model: ActorCritic, device: torch.device, num_episodes: int):
    env = gym.make(ENV_ID)
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
                    state_tensor = states_to_tensor(state, device).unsqueeze(0)
                    action = model.deterministic_action(state_tensor).squeeze(0)
                    state, reward, terminated, truncated, _ = env.step(
                        action.cpu().numpy()
                    )
                    episode_return += float(reward)
                episode_returns.append(episode_return)
    finally:
        env.close()
        model.train()
    return float(np.mean(episode_returns))


def save_checkpoint(model, optimizer, update_index, mean_eval_return, checkpoint_path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update_index": update_index,
            "mean_eval_return": mean_eval_return,
            "config": {
                "num_envs": NUM_ENVS,
                "num_steps": NUM_STEPS,
                "value_coef": VALUE_COEF,
                "use_huber_value_loss": USE_HUBER_VALUE_LOSS,
            },
        },
        checkpoint_path,
    )


def train():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    envs = make_vector_env(ENV_ID, NUM_ENVS, SEED)
    observation_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    model = ActorCritic(observation_dim, action_dim, HIDDEN_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    states, _ = envs.reset(seed=SEED)
    num_updates = TOTAL_TIMESTEPS // (NUM_ENVS * NUM_STEPS)
    last_eval_return = float("nan")

    print(
        f"device={DEVICE} | updates={num_updates} | batch={NUM_ENVS * NUM_STEPS} | "
        f"value_loss={'huber' if USE_HUBER_VALUE_LOSS else 'mse'}"
    )

    try:
        for update_index in range(1, num_updates + 1):
            rollout, states = collect_rollout(envs, model, states, NUM_STEPS, DEVICE)
            with torch.no_grad():
                _, _, next_value = model.forward(states_to_tensor(states, DEVICE))

            returns, advantages = compute_returns_and_advantages(
                rollout["rewards"],
                rollout["terminateds"],
                rollout["truncateds"],
                rollout["values"],
                next_value,
                rollout["truncated_bootstrap_values"],
                GAMMA,
            )
            total_loss, policy_loss, value_loss, entropy = compute_a2c_loss(
                rollout, returns, advantages
            )

            optimizer.zero_grad()
            total_loss.backward()
            actor_grad_norm = model.actor_mean.weight.grad.norm().item()
            critic_grad_norm = model.critic.weight.grad.norm().item()
            actor_feature_grad_norm = model.actor_net[0].weight.grad.norm().item()
            critic_feature_grad_norm = model.critic_net[0].weight.grad.norm().item()
            log_std_grad_norm = model.log_std.grad.norm().item()
            total_grad_norm = float(
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            )
            optimizer.step()

            if update_index % EVAL_INTERVAL_UPDATES == 0:
                last_eval_return = evaluate_policy(model, DEVICE, NUM_EVAL_EPISODES)
                with torch.no_grad():
                    debug_mean, debug_std, _ = model.forward(states_to_tensor(states, DEVICE))

                values = rollout["values"].detach()
                target_variance = returns.var(unbiased=False)
                residual_variance = (returns - values).var(unbiased=False)
                explained_variance = 1.0 - residual_variance / (target_variance + 1e-8)

                print(
                    f"update: {update_index} | "
                    f"policy loss: {policy_loss.item():.3f} | "
                    f"value loss: {value_loss.item():.3f} | "
                    f"entropy: {entropy.item():.3f} | "
                    f"eval return: {last_eval_return:.1f} | "
                    f"reward mean/std: {rollout['rewards'].mean().item():.2f}/"
                    f"{rollout['rewards'].std(unbiased=False).item():.2f} | "
                    f"return mean/std: {returns.mean().item():.1f}/"
                    f"{returns.std(unbiased=False).item():.1f} | "
                    f"value mean/std: {values.mean().item():.1f}/"
                    f"{values.std(unbiased=False).item():.1f} | "
                    f"explained var: {explained_variance.item():.3f} | "
                    f"policy |mean|/std: {debug_mean.abs().mean().item():.3f}/"
                    f"{debug_std.mean().item():.3f} | "
                    f"action clip: {rollout['clip_fractions'].mean().item():.1%} | "
                    f"grad actor/critic/actor_net/critic_net/std/all: "
                    f"{actor_grad_norm:.3g}/{critic_grad_norm:.3g}/"
                    f"{actor_feature_grad_norm:.3g}/{critic_feature_grad_norm:.3g}/"
                    f"{log_std_grad_norm:.3g}/{total_grad_norm:.3g}"
                )

                if last_eval_return >= SOLVED_MEAN_RETURN:
                    save_checkpoint(
                        model, optimizer, update_index, last_eval_return, LATEST_CHECKPOINT_PATH
                    )
                    save_checkpoint(
                        model, optimizer, update_index, last_eval_return, SOLVED_CHECKPOINT_PATH
                    )
                    print(f"Solved: mean evaluation return {last_eval_return:.1f}")
                    break

            if update_index % SAVE_INTERVAL_UPDATES == 0:
                save_checkpoint(
                    model, optimizer, update_index, last_eval_return, LATEST_CHECKPOINT_PATH
                )
    finally:
        envs.close()


if __name__ == "__main__":
    train()
