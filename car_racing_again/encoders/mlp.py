import torch
import torch.nn as nn

class MLPEncoder(nn.Module):
    def __init__(self, state_dim = 15, hidden_dim = 256):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(self.state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self,x):
        features = self.network(x)
        return features
