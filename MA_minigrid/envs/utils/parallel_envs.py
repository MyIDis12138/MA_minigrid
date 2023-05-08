import math
from multiprocessing import Process, Pipe
import traceback

import gymnasium as gym

from MA_minigrid.wrappers import SingleAgentWrapper
from MA_minigrid.envs.MAbabyai.query_wrapper import MultiGrid_Safety_Query
from MA_minigrid.wrappers import KGWrapper
from minigrid.core.actions import Actions
from MA_minigrid.envs.MAbabyai.utils.format import Vocabulary 
from MA_minigrid.envs.MAbabyai.query_GPT import OracleCentralizedGPT, OracleGPT

def make_envs(args, oracle: OracleGPT = None, num_envs: bool = None, verbose: bool = False, render: bool = False, vocab_path: str='/home/yang/MA_minigrid/MA_minigrid/envs/MAbabyai/vocab'):
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
                                args=args)
            env.seed(100 * args.seed + i)
            envs.append(env)
    return envs


def worker(conn, env, oracle: OracleGPT = None):
    try:
        while True:
            cmd, data, id = conn.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
                if done:
                    obs = env.reset(id)
                    oracle.reset_id(env, id)
                conn.send((obs, reward, done, info))
            elif cmd == "reset":
                obs = env.reset()
                oracle.reset_id(env, id)
                conn.send(obs)
            else:
                raise NotImplementedError
    except Exception as e:
        traceback.print_tb(e.__traceback__)
        raise e
    
def centralized_worker(conn, env):
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "step":
                obs, reward, done, info = env.step(data)
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
        self.oracle = OracleGPT(Vocabulary(args.vocab_path), env_num=args.procs)
        envs = make_envs(args, oracle=self.oracle, num_envs=args.procs, verbose=args.verbose, render=args.render, vocab_path=args.vocab_path)
        assert len(envs) >= 1, "No environment given."

        self.envs = envs
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space


    def reset(self):
        results = [env.reset() for env in self.envs]
        self.oracle.reset_all(self.envs)
        return results

    def step(self, actions):
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, done, info = env.step(action)
            if done:
                obs = env.reset()
                self.oracle.reset(env)
            results.append((obs, reward, done, info))

        batched_results = tuple(map(list, zip(*results)))
        return batched_results

    def render(self):
        raise NotImplementedError




# class ParallelCentralizedEnv(gym.Env):
#     """A concurrent execution of environments in multiple processes."""

#     def __init__(
#             self, 
#             args,
#             envs: list[gym.Env] | None = None,          
#             mode: str = 'rule',
#             vocab_path: str | None = None,
#             flat: bool = False, 
#             n_q: int = 1, 
#             restricted: bool = False, 
#             query_limit: int = -1, 
#             random_ans: bool = False,
#             reliability: float = 1.0, 
#         ):
#         assert len(envs) >= 1, "No environment given."
#         assert mode in ['rule', 'GPT'], "Mode should be either 'rule' or 'GPT'."

#         if flat:
#             for env in envs:
#                 env.action_space = gym.spaces.Discrete((self.env.action_space.n) + n_q)
#         else:
#             for env in envs:
#                 env.action_space = gym.spaces.MultiDiscrete(
#                     [env.action_space.n, n_q])
#         self.envs = envs
#         self.restricted = restricted
#         self.query_limit = query_limit
#         self.random_ans = random_ans
#         self.reliability = reliability
#         self.mode = mode

#         if vocab_path is None or vocab_path == '':
#             vocab_path = "../vocab/vocab.txt"
#         if mode == 'GPT':
#             self.oracle = OracleCentralizedGPT(Vocabulary(args.vocab_path))

#         self.mini_grid_actions_map = {
#             'left': Actions.left, 
#             'right': Actions.right, 
#             'forward': Actions.forward, 
#             'pickup': Actions.pickup, 
#             'drop': Actions.drop, 
#             'toggle': Actions.toggle,
#             'done': Actions.done,
#         }

    
#         self.envs = envs
#         self.observation_space = self.envs[0].observation_space
#         self.action_space = self.envs[0].action_space

#         self.locals = []
#         self.processes = []
#         for env in self.envs[1:]:
#             local, remote = Pipe()
#             self.locals.append(local)
#             p = Process(target=centralized_worker, args=(remote, env))
#             p.daemon = True
#             p.start()
#             remote.close()
#             self.processes.append(p)

#     def reset(self):
#         for local in self.locals:
#             local.send(("reset", None))
#         results = [self.envs[0].reset()] + [local.recv() for local in self.locals]
#         self.prev_ans = []
#         self.eps_steps = []
#         self.eps_n_q = []
#         if self.mode == 'GPT':
#             self.oracle.reset_all(self.envs)
#         for result in results:
#             result['ans'] = 'None'
#             self.prev_ans.append('None')
#             self.eps_steps.append(0)
#             self.eps_n_q.append(0)
#         return results
    
#     def reset_env(self, id):
#         self.prev_ans[id] = 'None'
#         self.eps_steps[id] = 0
#         self.eps_n_q[id] = 0
#         if id == 0:
#             obs = self.envs[id].reset()
#         else:
#             self.locals[id-1].send(("reset", None))
#             obs = self.locals[id-1].recv()
#         if self.mode == 'GPT':
#             self.oracle.reset(self.envs[id],id)
#         return obs

#     def step(self, actions):
#         self.action_preprocess(actions)
#         ans = list([] for _ in range(len(actions)))
#         query_ids = []

#         # step the first env
#         if actions[0][0] in self.mini_grid_actions_map:
#             obs, reward, done, info = self.envs[0].step(self.mini_grid_actions_map[actions[0][0]])
#             ans[0] = self.prev_ans[0]
#         else:
#             obs, reward, done, info = self.envs[0].step(Actions.done)
#             if self.query_limit > 0 and self.eps_n_q[0] >= self.query_limit:
#                 ans[id] = 'None'
#             else:
#                 query_ids.append(0)
#                 self.eps_n_q[0] += 1

#         # step the rest envs
#         for id, action in enumerate(actions[1:]):
#             if action[0] in self.mini_grid_actions_map:
#                 self.locals[id].send(("step", self.mini_grid_actions_map[action[0]]))
#                 ans[id+1] = self.prev_ans[id+1]
#             else:
#                 self.locals[id].send(("step", Actions.done))
#                 if self.query_limit > 0 and self.eps_n_q[id] >= self.query_limit:
#                     ans[id] = 'None'
#                 else:
#                     query_ids.append(id+1)
#                     self.eps_n_q[id+1] += 1

#         results = zip(*[(obs, reward, done, info)] + [local.recv() for local in self.locals])

#         # get the answers
#         if self.mode == 'rule':
#             for id in query_ids:
#                 ans[id] = self.envs[id].get_answer(actions[id][0])
#         elif self.mode == 'GPT':
#             queries = [actions[id][0] for id in query_ids]
           
#             if len(queries) > 0:
#                 answers = self.oracle.get_answer(queries, query_ids)
#                 for id, answer in zip(query_ids, answers):
#                     ans[id] = answer

#         # update the envs
#         for id, answer in enumerate(ans):
#             results[id][0]['ans'] = answer
#             if results[id][2]:
#                  results[id][0] = self.reset_env(id)
            
#         return results
    
#     def action_preprocess(self, actions):
#         for i, action in enumerate(actions):
#             assert isinstance(action, list)
#             if isinstance(action[0], list):
#                 action = action[0]

#     def render(self):
#         raise NotImplementedError

#     def __del__(self):
#         for p in self.processes:
#             p.terminate()