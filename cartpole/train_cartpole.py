''' REINFORCE with baseline의 baseline으로 상수가 아니라 phi에 종속되는
V(s)로 두는 방식을 사용. phi도 학습하는 방식으로 하겠다'''

import numpy as np
import torch 
import torch.nn as nn
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from torch.distributions import Categorical
import os

ENV_ID              = "CartPole-v1"
GAMMA               = 0.99
POLICY_LR           = 1e-3
BASELINE_LR         = 1e-3
HIDDEN_DIM          = 128
NUM_EPISODES        = 1000
EVAL_INTERVAL       = 25
SEED                = 42
MODEL_PATH          = "reinforce_baseline_cartpole.pt"
VIDEO_DIR           = "videos"

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, states):
        action_scores = self.network(states)
        return action_scores


class BaselineNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,1)
        )
    def forward(self, states:torch.Tensor):
        baseline_value = self.network(states)
        return baseline_value.squeeze(-1)

def select_action(policy: PolicyNetwork, state):
    state = torch.as_tensor(state, dtype = torch.float32)
    state = state.unsqueeze(dim = 0)
    with torch.no_grad():
        action_scores = policy(state)
        action_distribution = Categorical(logits = action_scores)
        sampled_action = action_distribution.sample()
        action = sampled_action.item()
        return action
        #policy는 점수를 출력하고 그거를 categorical이 내부의 softmax에 넣어서 확률로 만들고
        #그 확률이 sample에 들어가서 0번 행동 또는 1번 행동으로 출력 (그확률대로)

def collect_episode(env, policy:PolicyNetwork):
    state, info = env.reset()
    states = []
    actions = []
    rewards = []
    episode_reward = 0.0
    done = False

    while done ==False:
        action = select_action(policy, state)
        next_state, reward, terminated, truncated, info = env.step(action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        episode_reward+= reward
        state = next_state
        done = terminated or truncated
    return states, actions, rewards, episode_reward

    
def compute_returns(rewards, gamma):
    returns = []
    running_return = 0.0
    for reward in reversed(rewards):
         running_return = reward + gamma * running_return
         returns.append(running_return)

    returns.reverse()
    returns_tensor = torch.tensor(returns, dtype = torch.float32)

    return returns_tensor

def reinforce_update(policy: PolicyNetwork, baseline: BaselineNetwork, policy_optimizer, baseline_optimizer, states, actions, rewards, gamma):
    states_tensor = torch.tensor(states, dtype = torch.float32)
    actions_tensor = torch.tensor(actions, dtype = torch.long)
    returns_tensor = compute_returns(rewards, gamma)
    action_scores = policy(states_tensor)
    action_distribution = Categorical(logits = action_scores)
    log_probs = action_distribution.log_prob(actions_tensor)
    baseline_values = baseline(states_tensor)
    advantages = returns_tensor-baseline_values.detach() #Gt-V(S)
    time_steps = torch.arange(len(states), dtype=torch.float32)
    discount_weights = gamma ** time_steps
    policy_loss = -(discount_weights * advantages * log_probs).mean()
    #policy loss = -1/T sigma(gamma^t At log(pi(at|st)))
    # log앞에 dell안붙는 이유는 이거를 policy_loss.backward()에서 미분하기 때문

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()

    baseline_loss = ((returns_tensor - baseline_values)**2).mean()
    # baseline loss = 1/T sigma(Gt- V(st))^2  nstep G가 아니라 그냥 MC방식의 G
    baseline_optimizer.zero_grad()
    baseline_loss.backward()
    baseline_optimizer.step()
    return policy_loss.item(), baseline_loss.item()


def evaluate_policy(env, policy: PolicyNetwork):
    state, info = env.reset()
    episode_reward = 0.0
    done = False
    while not done:
        state_tensor = torch.tensor(state, dtype = torch.float32).unsqueeze(dim = 0)
        with torch.no_grad():
            action_scores = policy(state_tensor)
            action = torch.argmax(action_scores).item()
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            state = next_state
            done =  terminated or truncated
    return episode_reward    

def save_checkpoint(
    policy,
    baseline,
    policy_optimizer,
    baseline_optimizer,
    episode,
    best_score,
    model_path,
):
    checkpoint = {
        "policy_state_dict": policy.state_dict(),
        "baseline_state_dict": baseline.state_dict(),
        "policy_optimizer_state_dict": policy_optimizer.state_dict(),
        "baseline_optimizer_state_dict": baseline_optimizer.state_dict(),
        "episode": episode,
        "best_score": best_score,
    }

    torch.save(checkpoint, model_path)

def load_checkpoint(
    policy,
    baseline,
    policy_optimizer,
    baseline_optimizer,
    model_path,
):
    checkpoint = torch.load(model_path, map_location="cpu")

    policy.load_state_dict(checkpoint["policy_state_dict"])
    baseline.load_state_dict(checkpoint["baseline_state_dict"])

    policy_optimizer.load_state_dict(
        checkpoint["policy_optimizer_state_dict"]
    )
    baseline_optimizer.load_state_dict(
        checkpoint["baseline_optimizer_state_dict"]
    )

    return checkpoint["episode"], checkpoint["best_score"]

def record_episode(policy, video_dir):
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(ENV_ID, render_mode="rgb_array")
    video_env = RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda episode_id: True,
        name_prefix="final",
    )

    state, info = video_env.reset()
    episode_reward = 0.0
    done = False

    while not done:
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
        ).unsqueeze(dim=0)

        with torch.no_grad():
            action_scores = policy(state_tensor)
            action = torch.argmax(
                action_scores,
                dim=-1,
            ).item()

        next_state, reward, terminated, truncated, info = video_env.step(
            action
        )

        episode_reward += reward
        state = next_state
        done = terminated or truncated

    video_env.close()

    return episode_reward

def main():
    train_env = gym.make(ENV_ID)
    eval_env = gym.make(ENV_ID)

    state_dim = train_env.observation_space.shape[0]
    action_dim = train_env.action_space.n

    policy = PolicyNetwork(state_dim=state_dim,action_dim=action_dim,hidden_dim=HIDDEN_DIM,)

    baseline = BaselineNetwork(state_dim=state_dim,hidden_dim=HIDDEN_DIM)

    policy_optimizer = torch.optim.Adam(policy.parameters(),lr=POLICY_LR)

    baseline_optimizer = torch.optim.Adam(baseline.parameters(),lr=BASELINE_LR)

    best_score = float("-inf")

    for episode in range(1, NUM_EPISODES + 1):
        states, actions, rewards, episode_reward = collect_episode(train_env,policy)

        policy_loss, baseline_loss = reinforce_update(
            policy,
            baseline,
            policy_optimizer,
            baseline_optimizer,
            states,
            actions,
            rewards,
            GAMMA,
        )

        if episode % EVAL_INTERVAL == 0:
            evaluation_score = evaluate_policy(eval_env, policy)

            print(
                f"Episode: {episode} | "
                f"Train reward: {episode_reward:.0f} | "
                f"Eval reward: {evaluation_score:.0f} | "
                f"Policy loss: {policy_loss:.4f} | "
                f"Baseline loss: {baseline_loss:.4f}"
            )

            if evaluation_score > best_score:
                best_score = evaluation_score

                save_checkpoint(
                    policy,
                    baseline,
                    policy_optimizer,
                    baseline_optimizer,
                    episode,
                    best_score,
                    MODEL_PATH,
                )

    train_env.close()
    eval_env.close()

    saved_episode, saved_score = load_checkpoint(
        policy,
        baseline,
        policy_optimizer,
        baseline_optimizer,
        MODEL_PATH,
    )

    recorded_reward = record_episode(policy, VIDEO_DIR)

    print(f"Best checkpoint episode: {saved_episode}")
    print(f"Best evaluation score: {saved_score:.0f}")
    print(f"Recorded episode reward: {recorded_reward:.0f}")
if __name__ == "__main__":
    main()
