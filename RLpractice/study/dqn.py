import gymnasium as gym
import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from replay_buffer import ReplayBuffer

class QNet(nn.Module):
    def __init__(self, action_size):
        super().__init__()

        self.l1 = nn.Linear(4, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)

        return x
    
class DQNAgent:
    def __init__(self):
        self.gamma = 0.98
        self.lr = 0.0005
        self.epsilon = 0.1
        self.buffer_size = 10000
        self.batch_size = 32
        self.action_size = 2

        self.replay_buffer = ReplayBuffer(
            self.buffer_size,
            self.batch_size
        )

        self.qnet = QNet(self.action_size)
        self.qnet_target = QNet(self.action_size)

        self.optimizer = optim.Adam(
            self.qnet.parameters(),
            lr=self.lr
        )

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)

        else:
            state = state[np.newaxis, :]
            state = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                qs = self.qnet(state)

            return qs.argmax().item()

    def update(self, state, action, reward, next_state, done):
        self.replay_buffer.add(
            state,
            action,
            reward,
            next_state,
            done
        )

        if len(self.replay_buffer) < self.batch_size:
            return

        state, action, reward, next_state, done = self.replay_buffer.get_batch()

        state = torch.tensor(state, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)

        qs = self.qnet(state)

        q = qs[torch.arange(self.batch_size), action]

        with torch.no_grad():
            next_qs = self.qnet_target(next_state)
            next_q = next_qs.max(dim=1).values

            target = reward + (1 - done) * self.gamma * next_q

        loss = F.mse_loss(q, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_qnet(self):
        self.qnet_target.load_state_dict(self.qnet.state_dict())