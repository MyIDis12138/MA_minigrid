import math
from multiprocessing import Process, Pipe
import traceback

import gymnasium as gym

from MA_minigrid.wrappers import SingleAgentWrapper
from MA_minigrid.envs.MAbabyai.query_wrapper import MultiGrid_Safety_Query
from MA_minigrid.wrappers import KGWrapper
from MA_minigrid.envs.MAbabyai.utils.format import Vocabulary 
from MA_minigrid.envs.MAbabyai.query_GPT import OracleGPT

def make_envs(
        env_names: str = None, 
        oracle: OracleGPT = None, 
        num_envs: bool = None, 
        verbose: bool = False, 
        query: bool = False,
        n_q: int = 1,
        query_arch:str = 'flat',
        kg_wrapper: bool = False,
        vocab_path: str='/home/yang/MA_minigrid/MA_minigrid/envs/MAbabyai/vocab',
        **kg_kwargs
    ):
    mapping = {
        'SQbabyai-DangerGround-v0': MultiGrid_Safety_Query,
        'SQbabyai-DangerRoom-v0': MultiGrid_Safety_Query,
        'SQbabyai-DangerAgent-v0': MultiGrid_Safety_Query,
    }
    dir_wrapper_mapping = {
        'SQbabyai-DangerGround-v0': SingleAgentWrapper,
        'SQbabyai-DangerRoom-v0': SingleAgentWrapper,
        'SQbabyai-DangerAgent-v0': SingleAgentWrapper,
        #'BabyAI-SGoToFavoriteDangerroom-v0' : DRoom_directionwrapper
    }
    num_envs = num_envs
    #q_wrqpper = mapping[args.env] if args.env in mapping else ObjInBoxMulti_MultiOracle_Query
    envs = []
    env_names = env_names
    n_diff_envs = len(env_names)
    n_proc_per_env = math.ceil(num_envs / n_diff_envs)
    for env_name in env_names:
        q_wrqpper = mapping[env_name]
        for i in range(n_proc_per_env):
            env = gym.make(env_name)
            dir_wrapper = dir_wrapper_mapping[env_name]
            env = dir_wrapper(env)
            if query:
                env = q_wrqpper(env, 
                                oracle=oracle,
                                n_q=n_q,
                                restricted=False, 
                                verbose=verbose, 
                                flat= 'flat' in query_arch,
                                vocab_path=vocab_path,
                                query_limit=100, 
                                reliability=1, 
                            )
            if kg_wrapper:
                env = KGWrapper(env, **kg_kwargs)
            envs.append(env)
    return envs


def worker(conn, env):
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset()
                conn.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                conn.send(obs)
            else:
                raise NotImplementedError
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        raise e

class ParallelEnv(gym.Env):
    """A concurrent execution of environments in multiple processes."""

    def __init__(
        self, 
        env_names, 
        num_envs, 
        query,
        n_q,
        query_arch,
        query_mode, 
        vocab_path, 
        kg_wrapper,
        **kwargs
    ):
        if query_mode == 'GPT':
            self.oracle = OracleGPT(Vocabulary(vocab_path))
        else:
            self.oracle = None
            
        envs = make_envs(oracle=self.oracle, 
                         kg_wrapper=kg_wrapper,
                         query=query,
                         query_arch=query_arch,
                         env_names=env_names,
                         n_q=n_q,
                         num_envs=num_envs, 
                         verbose=False, 
                         vocab_path=vocab_path,
                         **kwargs)
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        self.locals = []
        self.processes = []
        for env in self.envs[1:]:
            local, remote = Pipe()
            self.locals.append(local)
            p = Process(target=worker, args=(remote, env))
            p.daemon = True
            p.start()
            remote.close()
            self.processes.append(p)

    def reset(self):
        for local in self.locals:
            local.send(("reset", None))
        results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
        return results

    def step(self, actions):
        for local, action in zip(self.locals, actions[1:]):
            local.send(("step", action))
        obs, reward, done, info = self.envs[0].step(actions[0])
        if done:
            obs = self.envs[0].reset()
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()