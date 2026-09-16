"""Replay buffer that retains native uint8 RGB Gymnasium frames."""

from __future__ import annotations

import numpy as np
import torch


class ImageReplayBuffer:
    def __init__(self, capacity: int, observation_shape: tuple[int, ...], seed: int = 0) -> None:
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._images = np.empty((capacity, *observation_shape), dtype=np.uint8)
        self._next_images = np.empty_like(self._images)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.float32)
        self._position = self._size = 0

    def add(self, image: np.ndarray, action: int, reward: float, next_image: np.ndarray, terminated: bool) -> None:
        index = self._position
        self._images[index], self._next_images[index] = image, next_image
        self._actions[index], self._rewards[index], self._terminated[index] = action, reward, terminated
        self._position = (index + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        indices = self._rng.choice(self._size, batch_size, replace=False)
        return (
            torch.as_tensor(self._images[indices], device=device),
            torch.as_tensor(self._actions[indices], device=device, dtype=torch.long),
            torch.as_tensor(self._rewards[indices], device=device),
            torch.as_tensor(self._next_images[indices], device=device),
            torch.as_tensor(self._terminated[indices], device=device),
        )

    def __len__(self) -> int:
        return self._size
