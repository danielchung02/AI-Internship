"""Frame-stack image CarRacing with the native continuous action space."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces

from src.envs.image_stack import DashboardMaskedFrameStack


def make_img_cont_env(
    seed: int | None = None,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create continuous CarRacing with a dashboard-free four-frame history.

    The returned observation has shape ``(4, 84, 96, 3)``.
    """
    env = gym.make(
        "CarRacing-v3",
        continuous=True,
        domain_randomize=False,
        render_mode=render_mode,
    )
    if not isinstance(env.observation_space, spaces.Box) or env.observation_space.shape != (96, 96, 3):
        env.close()
        raise TypeError("Expected CarRacing-v3's native 96x96 RGB observation")
    if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (3,):
        env.close()
        raise TypeError("Expected CarRacing continuous actions: steer, gas, brake")
    env = DashboardMaskedFrameStack(env)
    if seed is not None:
        env.action_space.seed(seed)
    return env
