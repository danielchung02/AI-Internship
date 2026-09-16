import os
# import gym
import gymnasium as gym
from stable_baselines3 import PPO  ## algorithm
from stable_baselines3.common.vec_env import DummyVecEnv  
from stable_baselines3.common.evaluation import evaluate_policy

environment_name = 'CartPole-v1'
env = gym.make(environment_name, render_mode="human")

episodes = 500
for episode in range(1,episodes+1) :
        
    state = env.reset()
    terminated = False
    truncated = False
    score = 0
    
    while not (terminated | truncated) :
        env.render()
        action = env.action_space.sample()
        n_state, reward, terminated, truncated, info = env.step(action)
        
        score += reward
    print("Episode:{} Score:{}".format(episode,score))