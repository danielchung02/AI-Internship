"""Image-history preprocessing shared by the image-based CarRacing methods.

The bottom 12 pixel rows of CarRacing's RGB observation contain the dashboard.
They are deliberately removed here: the agent must estimate motion from visual
change across recent road images rather than read the speed indicator.
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


FRAME_STACK_SIZE = 4
DASHBOARD_HEIGHT = 12


class DashboardMaskedFrameStack(gym.Wrapper):
    """Return the latest RGB frames after removing CarRacing's dashboard.

    An observation has shape ``(4, 84, 96, 3)`` in oldest-to-newest order.
    On reset, the first frame is repeated four times because no past frames
    exist yet.  The CNN receives all four frames as 12 input channels.
    """

    def __init__(
        self,
        env: gym.Env,
        *,
        num_frames: int = FRAME_STACK_SIZE,
        dashboard_height: int = DASHBOARD_HEIGHT,
    ) -> None:
        super().__init__(env)
        if num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if not isinstance(env.observation_space, spaces.Box):
            raise TypeError("Expected a Box RGB observation space")
        height, width, channels = env.observation_space.shape
        if channels != 3 or dashboard_height <= 0 or dashboard_height >= height:
            raise ValueError("Expected RGB frames with a valid dashboard crop")

        self.num_frames = num_frames
        self.image_height = height - dashboard_height
        self._frames: deque[np.ndarray] = deque(maxlen=num_frames)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(num_frames, self.image_height, width, channels),
            dtype=np.uint8,
        )

    def _without_dashboard(self, image: np.ndarray) -> np.ndarray:
        # Copy because the environment is free to reuse its image array.
        return np.ascontiguousarray(image[: self.image_height]).copy()

    def _stacked_observation(self) -> np.ndarray:
        return np.stack(tuple(self._frames), axis=0)

    def reset(self, **kwargs):
        image, info = self.env.reset(**kwargs)
        frame = self._without_dashboard(image)
        self._frames.clear()
        for _ in range(self.num_frames):
            self._frames.append(frame.copy())
        return self._stacked_observation(), info

    def step(self, action):
        image, reward, terminated, truncated, info = self.env.step(action)
        self._frames.append(self._without_dashboard(image))
        return self._stacked_observation(), reward, terminated, truncated, info
