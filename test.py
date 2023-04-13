from envs.MA_empty import EmptyEnv

if __name__ == '__main__':
    env = EmptyEnv([0])
    env.reset()
    env.render()

    while True:
        actions = [1]
        env.step(actions)
        env.render()
