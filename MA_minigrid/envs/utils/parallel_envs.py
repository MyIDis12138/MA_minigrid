import math
from multiprocessing import Process, Pipe
import traceback

import gymnasium as gym

from MA_minigrid.wrappers import SingleAgentWrapper
from MA_minigrid.envs.MAbabyai.query_wrapper import MultiGrid_Safety_Query
from MA_minigrid.wrappers import KGWrapper
from MA_minigrid.envs.MAbabyai.utils.format import Vocabulary 
from MA_minigrid.envs.MAbabyai.query_GPT import OracleGPT

def make_envs(args, oracle: OracleGPT = None,num_envs: bool = None, verbose: bool = False, render: bool = False, vocab_path: str='/home/yang/MA_minigrid/MA_minigrid/envs/MAbabyai/vocab'):
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
    num_envs = args.procs if not num_envs else num_envs
    #q_wrqpper = mapping[args.env] if args.env in mapping else ObjInBoxMulti_MultiOracle_Query
    envs = []
    env_names = args.env.split(',')
    n_diff_envs = len(env_names)
    n_proc_per_env = math.ceil(num_envs / n_diff_envs)
    for env_name in env_names:
        q_wrqpper = mapping[env_name]
        for i in range(n_proc_per_env):
            env = gym.make(env_name)
            dir_wrapper = dir_wrapper_mapping[env_name]
            env = dir_wrapper(env)
            #env = SpecialWrapper(env)
            if args.query:
                env = q_wrqpper(env, 
                                mode=args.query_mode,
                                oracle=oracle,
                                restricted=False, 
                                flat='flat' in args.query_arch, 
                                n_q=args.n_query, 
                                verbose=verbose, 
                                query_limit=100, 
                                reliability=1, 
                                vocab_path=vocab_path,
                            )
            if args.kg_wrapper:
                assert not args.ans_image
                if args.kg_mode == 'no_kg':
                    if args.cc_bonus != 0:
                        print('Set CC Bonus to 0 in no_kg mode')
                        args.cc_bonus = 0
                env = KGWrapper(env, penalize_query=args.penalize_query, cc_bonus=args.cc_bonus,
                                weighted_bonus=args.weighted_bonus, kg_repr=args.kg_repr, mode=args.kg_mode, n_gram=args.n_gram,
                                distractor_file_path=args.distractors_path, n_distractors=args.n_distractors, args=args)
            env.seed(100 * args.seed + i)
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

    def __init__(self, args):
        self.oracle = OracleGPT(Vocabulary(args.vocab_path))
        envs = make_envs(args, oracle=self.oracle, num_envs=args.procs, verbose=args.verbose, render=args.render, vocab_path=args.vocab_path)
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
        # test = []
        # for local in self.locals:
        #     test.append(local.recv())
        results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])
        return results

    def render(self):
        raise NotImplementedError

    def __del__(self):
        for p in self.processes:
            p.terminate()