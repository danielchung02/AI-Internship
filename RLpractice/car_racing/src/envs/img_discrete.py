"""Frame-stack image CarRacing environment for the CNN-DQN experiment."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces

from src.envs.image_stack import DashboardMaskedFrameStack


def make_image_env(seed: int | None = None, *, render_mode: str | None = None) -> gym.Env:
    """Create discrete CarRacing with a dashboard-free four-frame history.

    ``continuous=False`` selects CarRacing's built-in ``Discrete(5)`` action
    space.  The returned observation has shape ``(4, 84, 96, 3)``.
    """
    env = gym.make(
        "CarRacing-v3",
        continuous=False,
        domain_randomize=False,
        render_mode=render_mode,
    )
    if not isinstance(env.observation_space, spaces.Box) or env.observation_space.shape != (96, 96, 3):
        env.close()
        raise TypeError("Expected CarRacing-v3's native 96x96 RGB observation")
    if not isinstance(env.action_space, spaces.Discrete):
        env.close()
        raise TypeError("Expected continuous=False to create a discrete action space")

    env = DashboardMaskedFrameStack(env)

    if seed is not None:
        env.action_space.seed(seed)
    return env
