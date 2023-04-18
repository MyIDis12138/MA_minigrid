
from __future__ import annotations

import math
import operator
from functools import reduce
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import ObservationWrapper, ObsType, Wrapper

class SignalObsWrapper(ObservationWrapper):
    """
    Use the first observation in the list as the only observation output.
    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import FirstObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = FirstObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image'])
    """

    def __init__(self, env):
        """A wrapper that makes the first observation the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces[0]

    def observation(self, obs):
        return obs[0]
    
class ImgObsWrapper(ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import ImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = ImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (7, 7, 9)
    """

    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces["image"]

    def observation(self, obs):
        return obs["image"]
    

class RGBImgObsWrapper(ObservationWrapper):
    """
    Wrapper to use fully observable RGB image as observation,
    This can be used to have the agent to solve the gridworld in pixel space.
    Example:
        >>> import gymnasium as gym
        >>> import matplotlib.pyplot as plt
        >>> from minigrid.wrappers import RGBImgObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![NoWrapper](../figures/lavacrossing_NoWrapper.png)
        >>> env = RGBImgObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> plt.imshow(obs['image'])  # doctest: +SKIP
        ![RGBImgObsWrapper](../figures/lavacrossing_RGBImgObsWrapper.png)
    """

    def __init__(self, env, tile_size=8):
        super().__init__(env)

        self.tile_size = tile_size

        new_image_space = spaces.Box(
            low=0,
            high=255,
            shape=(3, self.env.width * tile_size, self.env.height * tile_size),
            dtype="uint8",
        )

        self.observation_space = new_image_space

    def observation(self, obs):
        rgb_img = self.get_full_render(highlight=True, tile_size=self.tile_size)

        return rgb_img


class SingleAgentWrapper(Wrapper):
    """
    Use the image as the only observation output, no language/mission.
    Example:
        >>> import gymnasium as gym
        >>> from minigrid.wrappers import SimpleObsWrapper
        >>> env = gym.make("MiniGrid-Empty-5x5-v0")
        >>> obs, _ = env.reset()
        >>> obs.keys()
        dict_keys(['image', 'direction', 'mission'])
        >>> env = SimpleObsWrapper(env)
        >>> obs, _ = env.reset()
        >>> obs.shape
        (7, 7, 9)
    """

    def __init__(self, env):
        """A wrapper that makes image the only observation.
        Args:
            env: The environment to apply the wrapper
        """
        super().__init__(env)
        self.observation_space = env.observation_space.spaces[0]
        self.action_space = env.action_space.spaces[0]

    def observation(self, obs):
        obs = obs[0]
        return obs
    
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        obs,_ = super().reset(seed=seed, options=options)
        return self.observation(obs)
    
    def step(self, actions):
        obs, rewards, terminated, truncated, info = super().step([actions])
        done = terminated or truncated
        return self.observation(obs), rewards[0], done, info
    

