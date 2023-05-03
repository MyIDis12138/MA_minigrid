
from __future__ import annotations

import time

import gymnasium as gym
import pygame
from gymnasium import Env

from minigrid.core.actions import Actions
from MA_minigrid.envs.Danger_gound import DangerGroundEnv
from MA_minigrid.wrappers import ImgObsWrapper, RGBImgObsWrapper


class ManualControl:
    def __init__(
        self,
        env: Env,
        question_set: list[str] = None,
        seed: int=None,
        key_to_action=None,
    ) -> None:
        """
        Manual control of the environment with keyboard.
        :param env: environment to control, must be a single agent minigrid environment
        :param seed: seed for the environment
        :param key_to_action: dictionary mapping key names to actions
        """
        self.env = env
        self.seed = seed
        self.closed = False
        self.question_set = question_set
        if key_to_action is None:
            self.key_to_action = {
                "a": Actions.left,
                "d": Actions.right,
                "w": Actions.forward,
                " ": Actions.toggle,
                "pageup": Actions.pickup,
                "pagedown": Actions.drop,
                "tab": Actions.pickup,
                "left shift": Actions.drop,
                "enter": Actions.done,
            }
        else:
            self.key_to_action = key_to_action
            
        print("ManualControl: key_to_action= ", self.key_to_action)


    def start(self):
        """Start the window display with blocking event loop"""
        self.reset(self.seed)
        
        while not self.closed:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.env.close()
                    break
                if event.type == pygame.KEYDOWN:
                    event.key = pygame.key.name(int(event.key))
                    self.key_handler(event)

    def step(self, action: Actions):
        obs, reward, done, info = self.env.step(action)
        self.env.render()
        #print(self.env.__str__)
        print(f"step={self.env.step_count}, reward={reward:.2f}, info={info}")
        if done:
            print("done!")
            self.reset(self.seed)
        else:
            self.env.render()

    def reset(self, seed=None):
        self.env.reset()
        self.env.render()
        if self.question_set:
            for question in self.question_set:
                obs, reward, done, info = self.env.step(question)
                print(f"step={self.env.step_count}, reward={reward:.2f}, info={info}")
                time.sleep(1)
                self.env.render()

    def key_handler(self, event):
        key: str = event.key
        print("pressed", key)

        if key == "escape":
            self.env.close()
            return
        if key == "backspace":
            self.reset()
            return

        if key in self.key_to_action.keys():
            action = self.key_to_action[key]
            self.step(action)
        else:
            print(key)


if __name__ == "__main__":

    env = DangerGroundEnv()
    #env = RGBImgObsWrapper(env)
    #env = ImgObsWrapper(env)

    manual_control = ManualControl(env, seed=33)
    manual_control.start()