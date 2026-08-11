"""Shared environment-step helpers for image-based discrete DQN."""

from __future__ import annotations

import gymnasium as gym
import numpy as np


def repeated_step(
    env: gym.Env,
    action: int,
    action_repeat: int,
) -> tuple[np.ndarray, float, bool, bool, dict]:
    """Hold one discrete action for up to ``action_repeat`` environment frames."""
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")

    total_reward = 0.0
    info: dict = {}
    for _ in range(action_repeat):
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    return observation, total_reward, terminated, truncated, info
