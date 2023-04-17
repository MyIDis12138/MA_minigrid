import MA_minigrid
import gymnasium as gym
from MA_minigrid.ManualControl import ManualControl
from MA_minigrid.wrappers import SingleAgentWrapper
from MA_minigrid.envs.MAbabyai.query_wrapper import MultiGrid_Safety_Query

from tqdm import tqdm
import time

def get_human_action(num):
    action_map = [
        "left",
        "right",
        "forward",
        "pickup",
        "drop",
        "toggle",
        "done",
    ]

    return [action_map[num]]


def main():
    env = gym.make('SQbabyai-DangerGround-v0')
    #env = gym.make('SQbabyai-DangerRoom-v0')
    #env = gym.make('SQbabyai-DangerAgent-v0')
    env = SingleAgentWrapper(env)
    #env = MultiGrid_Safety_Query(env)

    env.reset()
    
    #while running:
    for _ in tqdm(range(1000000)):
        env.render()
        time.sleep(0.1)

        a = env.action_space.sample()
        obs, reward, done, info = env.step(a)
        if done:
            env.reset()
    

if __name__ == "__main__":
    main()
