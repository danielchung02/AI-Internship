"""Engineered-vector CarRacing with the native continuous action space."""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces

from src.envs.vec_discrete import CarRacingVectorObservation


def make_vec_cont_env(
    seed: int | None = None,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create 15D vector-observation CarRacing with steer/gas/brake actions."""
    env = gym.make(
        "CarRacing-v3",
        continuous=True,
        domain_randomize=False,
        render_mode=render_mode,
    )
    env = CarRacingVectorObservation(env)
    if not isinstance(env.action_space, spaces.Box) or env.action_space.shape != (3,):
        env.close()
        raise TypeError("Expected CarRacing continuous actions: steer, gas, brake")
    if seed is not None:
        env.action_space.seed(seed)
    return env
