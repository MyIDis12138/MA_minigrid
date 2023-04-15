import gymnasium as gym
import MA_minigrid

if __name__ == '__main__':
    env = gym.make("")
    #env = SignalObsWrapper(env)
    obs = env.reset()
    env.render()

    while True:
        actions = [1]
        env.step(actions)
        env.render()







