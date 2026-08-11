"""Engineered 1D observations for discrete-action Gymnasium CarRacing-v3.

CarRacing's public observation is an RGB image.  This wrapper instead reads
the simulator's car and track state and exposes a compact 15-value state for
an MLP DQN.  It is deliberately separate from the raw-image CNN experiment.
"""

from __future__ import annotations

import math
from typing import Final

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CarRacingVectorObservation(gym.ObservationWrapper):
    """Replace RGB frames with speed, alignment, offset, and look-ahead data.

    State layout:
    ``[forward_speed, lateral_speed, angular_velocity, steering, offset,
    sin(heading_error), cos(heading_error), ...]``.  The remaining eight
    values are sin/cos heading errors at four future track points.
    """

    STATE_DIM: Final[int] = 15
    LOOKAHEADS: Final[tuple[int, ...]] = (5, 10, 20, 40)
    TRACK_HALF_WIDTH: Final[float] = 40.0 / 6.0
    MAX_STEERING_ANGLE: Final[float] = 0.4

    def __init__(self, env: gym.Env, *, speed_scale: float = 50.0, angular_velocity_scale: float = 5.0) -> None:
        super().__init__(env)
        if speed_scale <= 0 or angular_velocity_scale <= 0:
            raise ValueError("State normalization scales must be positive")
        self.speed_scale = speed_scale
        self.angular_velocity_scale = angular_velocity_scale
        self.observation_space = spaces.Box(-1.0, 1.0, (self.STATE_DIM,), np.float32)

    @staticmethod
    def _wrap_angle(angle: float) -> float:
        return (angle + math.pi) % (2.0 * math.pi) - math.pi

    def observation(self, image: np.ndarray) -> np.ndarray:
        """Derive state from the current simulator state; RGB pixels are unused."""
        del image
        base_env = self.env.unwrapped
        car, track = getattr(base_env, "car", None), getattr(base_env, "track", None)
        if car is None or not track:
            raise RuntimeError("Reset CarRacing before requesting a vector observation")

        car_position = np.asarray(car.hull.position, dtype=np.float32)
        car_angle = float(car.hull.angle)
        track_xy = np.asarray([(item[2], item[3]) for item in track], dtype=np.float32)
        nearest = int(np.argmin(np.sum((track_xy - car_position) ** 2, axis=1)))
        _, track_angle, track_x, track_y = track[nearest]
        track_angle = float(track_angle)

        velocity = np.asarray(car.hull.linearVelocity, dtype=np.float32)
        forward_axis = np.asarray([-math.sin(car_angle), math.cos(car_angle)], dtype=np.float32)
        lateral_axis = np.asarray([math.cos(car_angle), math.sin(car_angle)], dtype=np.float32)
        heading_error = self._wrap_angle(track_angle - car_angle)
        track_normal = np.asarray([math.cos(track_angle), math.sin(track_angle)], dtype=np.float32)
        lateral_offset = float(np.dot(car_position - (track_x, track_y), track_normal))
        steering = 0.5 * (float(car.wheels[0].joint.angle) + float(car.wheels[1].joint.angle))

        values = [
            np.clip(np.dot(velocity, forward_axis) / self.speed_scale, -1.0, 1.0),
            np.clip(np.dot(velocity, lateral_axis) / self.speed_scale, -1.0, 1.0),
            np.clip(float(car.hull.angularVelocity) / self.angular_velocity_scale, -1.0, 1.0),
            np.clip(steering / self.MAX_STEERING_ANGLE, -1.0, 1.0),
            np.clip(lateral_offset / self.TRACK_HALF_WIDTH, -1.0, 1.0),
            math.sin(heading_error),
            math.cos(heading_error),
        ]
        for offset in self.LOOKAHEADS:
            future_angle = float(track[(nearest + offset) % len(track)][1])
            error = self._wrap_angle(future_angle - car_angle)
            values.extend((math.sin(error), math.cos(error)))

        state = np.asarray(values, dtype=np.float32)
        if state.shape != (self.STATE_DIM,):
            raise RuntimeError(f"Expected {self.STATE_DIM} state values, got {state.shape}")
        return state


def make_vector_env(
    seed: int | None = None,
    *,
    render_mode: str | None = None,
) -> gym.Env:
    """Create CarRacing's engineered-vector / MLP-DQN variant."""
    env = gym.make(
        "CarRacing-v3",
        continuous=False,
        domain_randomize=False,
        render_mode=render_mode,
    )
    env = CarRacingVectorObservation(env)
    if seed is not None:
        env.action_space.seed(seed)
    return env
