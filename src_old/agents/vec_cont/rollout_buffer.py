"""On-policy rollout storage for vector-observation PPO."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


class RolloutBuffer:
    """A fixed-length, on-policy buffer with GAE-Lambda advantages."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int) -> None:
        if capacity <= 0 or state_dim <= 0 or action_dim <= 0:
            raise ValueError("capacity, state_dim, and action_dim must be positive")
        self.capacity = capacity
        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._actions = np.empty((capacity, action_dim), dtype=np.float32)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.float32)
        self._episode_ends = np.empty(capacity, dtype=np.float32)
        self._values = np.empty(capacity, dtype=np.float32)
        self._next_values = np.empty(capacity, dtype=np.float32)
        self._log_probs = np.empty(capacity, dtype=np.float32)
        self._size = 0

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        terminated: bool,
        episode_end: bool,
        value: float,
        next_value: float,
        log_prob: float,
    ) -> None:
        if self._size >= self.capacity:
            raise RuntimeError("Rollout buffer is full")
        index = self._size
        self._states[index] = state
        self._actions[index] = action
        self._rewards[index] = reward
        self._terminated[index] = float(terminated)
        self._episode_ends[index] = float(episode_end)
        self._values[index] = value
        self._next_values[index] = next_value
        self._log_probs[index] = log_prob
        self._size += 1

    def get(
        self,
        device: torch.device,
        *,
        gamma: float,
        gae_lambda: float,
    ) -> RolloutBatch:
        if self._size == 0:
            raise RuntimeError("Cannot train from an empty rollout")

        advantages = np.zeros(self._size, dtype=np.float32)
        gae = 0.0
        for index in range(self._size - 1, -1, -1):
            bootstrap = 1.0 - self._terminated[index]
            continuation = 1.0 - self._episode_ends[index]
            delta = (
                self._rewards[index]
                + gamma * bootstrap * self._next_values[index]
                - self._values[index]
            )
            gae = delta + gamma * gae_lambda * continuation * gae
            advantages[index] = gae

        returns = advantages + self._values[: self._size]
        batch = RolloutBatch(
            states=torch.as_tensor(self._states[: self._size], device=device),
            actions=torch.as_tensor(self._actions[: self._size], device=device),
            old_log_probs=torch.as_tensor(
                self._log_probs[: self._size], device=device
            ),
            advantages=torch.as_tensor(advantages, device=device),
            returns=torch.as_tensor(returns, device=device),
        )
        self._size = 0
        return batch

    def __len__(self) -> int:
        return self._size
