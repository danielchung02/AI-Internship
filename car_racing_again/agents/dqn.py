import random
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim

class ReplayBuffer():
    def __init__(self,capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.index = 0

    def add(self, state, action, reward, next_state, done):
        transition = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)

        else:
            self.buffer[self.index] = transition
        self.index = (self.index +1) % self.capacity

    def sample(self, batch_size: int, device):
        indices = np.random.choice(len(self.buffer), size = batch_size, replace= False)
        batch = [self.buffer[i] for i in indices]

        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        for transition in batch:
            state, action, reward, next_state, done = transition
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

        states = torch.as_tensor(np.array(states), device=device)
        actions = torch.as_tensor(actions, dtype=torch.long, device=device)
        rewards = torch.as_tensor(rewards,dtype=torch.float32,device=device)
        next_states = torch.as_tensor( np.array(next_states), device=device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=device)

        return states, actions, rewards, next_states, dones

class DQNNetwork(nn.Module):
    def __init__(self, encoder,feature_dim = 256, action_dim = 5):
        super().__init__()
        self.encoder = encoder
        self.qhead = nn.Sequential(
            nn.Linear(feature_dim, action_dim)
        )

    def forward(self, state):
        features = self.encoder(state)
        qvalues = self.qhead(features)
        return qvalues

        
        
                

            
