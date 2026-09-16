"""NumPy-backed replay buffer used by the DQN agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class ReplayBatch:
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    terminated: torch.Tensor


class ReplayBuffer:
    """Fixed-size cyclic replay buffer.

    Only ``terminated`` is stored for Bellman masking. Gymnasium truncation ends
    the episode loop, but a time-limit truncation must still bootstrap from the
    final next state.
    """

    def __init__(self, capacity: int, state_dim: int, seed: int = 0) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")

        self.capacity = int(capacity)
        self.state_dim = int(state_dim)
        self._rng = np.random.default_rng(seed)

        self._states = np.empty((capacity, state_dim), dtype=np.float32)
        self._actions = np.empty(capacity, dtype=np.int64)
        self._rewards = np.empty(capacity, dtype=np.float32)
        self._next_states = np.empty((capacity, state_dim), dtype=np.float32)
        self._terminated = np.empty(capacity, dtype=np.float32)

        self._position = 0
        self._size = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        terminated: bool,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        next_state_array = np.asarray(next_state, dtype=np.float32)

        expected_shape = (self.state_dim,)
        if state_array.shape != expected_shape:
            raise ValueError(
                f"state must have shape {expected_shape}, got {state_array.shape}"
            )
        if next_state_array.shape != expected_shape:
            raise ValueError(
                "next_state must have shape "
                f"{expected_shape}, got {next_state_array.shape}"
            )

        index = self._position
        self._states[index] = state_array
        self._actions[index] = int(action)
        self._rewards[index] = float(reward)
        self._next_states[index] = next_state_array
        self._terminated[index] = float(terminated)

        self._position = (self._position + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> ReplayBatch:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._size < batch_size:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer size {self._size}"
            )

        indices = self._rng.choice(self._size, size=batch_size, replace=False)

        return ReplayBatch(
            states=torch.as_tensor(self._states[indices], device=device),
            actions=torch.as_tensor(
                self._actions[indices], device=device, dtype=torch.long
            ),
            rewards=torch.as_tensor(self._rewards[indices], device=device),
            next_states=torch.as_tensor(self._next_states[indices], device=device),
            terminated=torch.as_tensor(self._terminated[indices], device=device),
        )

    def __len__(self) -> int:
        return self._size
