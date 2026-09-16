"""CarRacing-specific episode outcome helpers."""

from __future__ import annotations

import gymnasium as gym


def lap_finished(env: gym.Env, terminated: bool) -> bool:
    """Return whether CarRacing ended because its track was completed.

    CarRacing's ``info`` dictionary is empty. Its native completion condition
    is read from the unwrapped environment: all tiles were visited, or the
    start tile was revisited after the configured fraction (95% by default).
    Going out of bounds also terminates an episode, but is not a completion.
    """
    if not terminated:
        return False

    base_env = env.unwrapped
    track = getattr(base_env, "track", None)
    visited_tiles = getattr(base_env, "tile_visited_count", None)
    if track is None or visited_tiles is None:
        return False

    return bool(getattr(base_env, "new_lap", False) or visited_tiles >= len(track))
