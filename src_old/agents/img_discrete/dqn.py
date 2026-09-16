"""Convolutional Double DQN for stacked RGB observations and discrete actions."""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
from typing import Any
from torch import nn
from torch.nn import functional as F


class ImageQNetwork(nn.Module):
    """Maps a stack of dashboard-free RGB frames to one Q-value per action."""

    def __init__(self, action_dim: int, observation_shape: tuple[int, ...]) -> None:
        super().__init__()
        num_frames, height, width, channels = observation_shape
        if channels != 3:
            raise ValueError("Expected stacked RGB observations")
        self.observation_shape = observation_shape
        self.features = nn.Sequential(
            nn.Conv2d(num_frames * channels, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flattened_size = self.features(torch.zeros(1, num_frames * channels, height, width)).shape[1]
        self.head = nn.Sequential(nn.Linear(flattened_size, 512), nn.ReLU(), nn.Linear(512, action_dim))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (batch, time, height, width, RGB) -> (batch, time * RGB, height, width)
        batch_size, num_frames, height, width, channels = images.shape
        images = images.permute(0, 1, 4, 2, 3).reshape(
            batch_size, num_frames * channels, height, width
        ).float().div_(255.0)
        return self.head(self.features(images))


class ImageDQNAgent:
    """CNN online/target networks with Double-DQN bootstrapping."""

    def __init__(self, action_dim: int, observation_shape: tuple[int, ...], device: torch.device, *, gamma: float = 0.99, learning_rate: float = 1e-4) -> None:
        self.action_dim, self.device, self.gamma = action_dim, device, gamma
        self.observation_shape = observation_shape
        self.online_network = ImageQNetwork(action_dim, observation_shape).to(device)
        self.target_network = ImageQNetwork(action_dim, observation_shape).to(device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.AdamW(
            self.online_network.parameters(),
            lr=learning_rate,
            amsgrad=True,
        )
        self.optimization_steps = 0

    @torch.no_grad()
    def select_action(self, image: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
        if rng.random() < epsilon:
            return int(rng.integers(self.action_dim))
        image_tensor = torch.as_tensor(image, device=self.device).unsqueeze(0)
        return int(self.online_network(image_tensor).argmax(dim=1).item())

    def update(self, batch: tuple[torch.Tensor, ...]) -> float:
        images, actions, rewards, next_images, terminated = batch
        predicted = self.online_network(images).gather(1, actions[:, None]).squeeze(1)
        with torch.no_grad():
            # Double DQN: online selects the next action; target evaluates it.
            next_actions = self.online_network(next_images).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_images).gather(1, next_actions).squeeze(1)
            target = rewards + self.gamma * (1.0 - terminated) * next_q_values
        loss = F.smooth_l1_loss(predicted, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), 10.0)
        self.optimizer.step()
        self.optimization_steps += 1
        return float(loss.item())

    def hard_update_target(self) -> None:
        self.target_network.load_state_dict(self.online_network.state_dict())

    def save(self, path: str | Path, **metadata: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "action_dim": self.action_dim,
                "observation_shape": self.observation_shape,
                "online_network": self.online_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "optimization_steps": self.optimization_steps,
                "metadata": metadata,
            },
            path,
        )

    def load(self, path: str | Path, *, load_optimizer: bool = True) -> dict[str, Any]:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        if checkpoint["action_dim"] != self.action_dim:
            raise ValueError("Checkpoint action_dim does not match current agent")
        if tuple(checkpoint.get("observation_shape", ())) != self.observation_shape:
            raise ValueError("Checkpoint observation shape does not match current agent")
        self.online_network.load_state_dict(checkpoint["online_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.optimization_steps = int(checkpoint.get("optimization_steps", 0))
        return dict(checkpoint.get("metadata", {}))
