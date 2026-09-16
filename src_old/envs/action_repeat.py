"""Helpers for holding one action across several CarRacing frames."""

from __future__ import annotations

from typing import Any

import gymnasium as gym


def repeated_step(
    env: gym.Env,
    action: Any,
    action_repeat: int,
) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    """Execute ``action`` up to ``action_repeat`` frames and sum rewards."""
    if action_repeat <= 0:
        raise ValueError("action_repeat must be positive")

    reward_sum = 0.0
    info: dict[str, Any] = {}
    for _ in range(action_repeat):
        observation, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        if terminated or truncated:
            break
    return observation, reward_sum, terminated, truncated, info
