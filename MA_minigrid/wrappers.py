
from __future__ import annotations

import math
import random
import re
import operator
from functools import reduce
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import logger, spaces
from gymnasium.core import ObservationWrapper, ObsType, Wrapper

from MA_minigrid.envs.MAbabyai.utils.knowledge_graph import KG

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
    

class KGWrapper(Wrapper):
    """
    A wrapper that returns the connected component of the KG in observation.
    kg_repr = [one_hot, raw]
    one_hot: each sentence is encoded as an onehot channel of the image
    raw: return all raw sentences as a list in observation['kg_cc']
    """
    def __init__(self, 
                 env, 
                 penalize_query=False, 
                 cc_bonus=0.05, 
                 weighted_bonus=False, 
                 kg_repr='raw', 
                 mode='graph_overlap', 
                 n_gram=2,
                 args=None):
        super(KGWrapper, self).__init__(env)
        self.kg_repr = kg_repr
        n_channel = env.observation_space['image'].shape[-1]
        self.moving_actions = ['left', 'right', 'forward', 'pickup', 'drop', 'toggle', 'done']

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.agent_view_size, self.agent_view_size, n_channel),
            dtype='uint8'
        )
        self.observation_space = gym.spaces.Dict({
            'image': self.observation_space
        })
        mode = 'set' if mode == 'no_kg' else mode
        self.KG = KG(mode=mode, n_gram=n_gram)
        self.cc_bonus = cc_bonus if mode == 'graph_overlap' else 0
        self.penalize_query = penalize_query
        if self.penalize_query:
            self.query_penalty = -0.01
        self.weighted_bonus = weighted_bonus
        self.total_frames_per_proc = args.frames // args.procs if args else 1000
        self.cur_total_frames = 0
        self.decrease_bonus = args.decrease_bonus if args else False

    def bonus_coef(self):
        if not self.decrease_bonus:
            return 1
        anneal_th = 0.6 * self.total_frames_per_proc
        if self.cur_total_frames <= anneal_th:
            return 1
        else:
            return 1.05 - (self.cur_total_frames - anneal_th) / (self.total_frames_per_proc - anneal_th)

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if isinstance(action, list) and len(action) > 1 and action[0] not in self.moving_actions:
            for ans in obs['ans'].split(','):
                is_CC_increase, overlap = self.KG.update(self.pre_proc_asn(ans))
                if is_CC_increase:
                    if self.weighted_bonus:
                        reward += self.bonus_coef() * self.cc_bonus * overlap
                    else:
                        reward += self.bonus_coef() * self.cc_bonus
            if self.penalize_query:
                reward += self.query_penalty
        obs = self.observation(obs, self.KG.getCC())
        self.cur_total_frames += 1
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        self.KG.reset(self.pre_proc_asn(obs['mission']))

        obs = self.observation(obs, self.KG.getCC())
        return obs


    def observation(self, observation, CC):
        """
        :param observation: dictionary
        :param CC: list of tuples
        :return: modified observation
        """
        if self.kg_repr == 'one_hot':
            ans_channel = np.zeros((7, 7, len(self.tokens)))
            for ans in CC:
                for i, token in enumerate(self.tokens):
                    if token == ans:
                        ans_channel[:, :, i] = 1
                        break
            obs = np.concatenate((observation['image'], ans_channel), axis=2)
            observation['image'] = obs
        elif self.kg_repr == 'raw':
            raw_repr = []
            for node in CC:
                raw_repr.append(' '.join(node))
            observation['kg_cc'] = raw_repr
        else:
            raise NotImplementedError

        return observation

    def pre_proc_asn(self, ans):
        ans = re.findall("([a-z0-9]+)", ans.lower())
        if 'is' in ans:
            ans.remove('is')
        if 'in' in ans:
            ans.remove('in')
        return ans

