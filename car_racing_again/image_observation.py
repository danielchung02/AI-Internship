from collections import deque  # add

import gymnasium as gym  # add
import numpy as np  # add
from gymnasium import spaces  # add


class DashboardMaskedFrameStack(gym.Wrapper):  # add
    def __init__(self, env, num_frames=4, dashboard_height=12):  # add
        super().__init__(env)  # add
        height, width, channels = env.observation_space.shape  # add
        if dashboard_height <= 0 or dashboard_height >= height or channels != 3:  # add
            raise ValueError("Expected RGB observations with a valid dashboard crop")  # add
        self.num_frames = num_frames  # add
        self.image_height = height - dashboard_height  # add
        self.frames = deque(maxlen=num_frames)  # add
        self.observation_space = spaces.Box(  # add
            low=0, high=255, shape=(num_frames, self.image_height, width, channels), dtype=np.uint8  # add
        )  # add

    def _crop_dashboard(self, image):  # add
        return np.ascontiguousarray(image[: self.image_height]).copy()  # add

    def _stacked_observation(self):  # add
        return np.stack(tuple(self.frames), axis=0)  # add

    def reset(self, **kwargs):  # add
        image, info = self.env.reset(**kwargs)  # add
        frame = self._crop_dashboard(image)  # add
        self.frames.clear()  # add
        for _ in range(self.num_frames):  # add
            self.frames.append(frame.copy())  # add
        return self._stacked_observation(), info  # add

    def step(self, action):  # add
        image, reward, terminated, truncated, info = self.env.step(action)  # add
        self.frames.append(self._crop_dashboard(image))  # add
        return self._stacked_observation(), reward, terminated, truncated, info  # add
