from __future__ import annotations

import copy
import random

import gymnasium
from gymnasium.core import Wrapper
from minigrid.core.actions import Actions

from MA_minigrid.envs.MAbabyai.query_GPT import OracleGPT
from MA_minigrid.envs.MAbabyai.utils.vocabulary import Vocabulary


class MultiGrid_Safety_Query(Wrapper):
    def __init__(
        self, 
        env, 
        mode: str = 'rule',
        vocab_path: str | None = None,
        verbose: bool = False, 
        flat: bool = False, 
        n_q: int = 1, 
        call: bool = True, 
        restricted: bool = False, 
        query_limit: int = -1, 
        random_ans: bool = False,
        reliability: float = 1.0, 
    ):
        """
        Wrapper for minigrid environment to add query action
        Alllow lanuage actions for the agent to ask questions to the environment
        params:
            env: single agent minigrid environment
            vocab_path: path to vocabulary file
            verbose: print query and answer
            restricted: restrict the question to be asked (not implemented yet)
            flat: flatten the action space
            call: whether to call or apporach to ask the query (not implemented yet)
            n_q: number of queries to be sampled in one step
            query_limit: maximum number of queries in one episode when restricted is set to True
            random_ans: whether to randomize the answer
            reliability: probability of answering correctly when random_ans is set to True
        """
        assert mode in ['rule', 'GPT']
        self.mode = mode
        if vocab_path is None:
            vocab_path = "./vocab/vocab1.txt"
        if self.mode == 'GPT':
            self.oracle = OracleGPT(vocab=Vocabulary(file_path=vocab_path))        

        super(MultiGrid_Safety_Query, self).__init__(env)
        self.flat = flat
        self.restricted = restricted
        self.query_limit = query_limit
        self.random_ans = random_ans
        self.reliability = reliability

        if self.flat:
            self.action_space = gymnasium.spaces.Discrete((self.env.action_space.n) + n_q)
        else:
            self.action_space = gymnasium.spaces.MultiDiscrete(
                [self.env.action_space.n, n_q])
            
        self.mini_grid_actions_map = {
            'left': Actions.left, 
            'right': Actions.right, 
            'forward': Actions.forward, 
            'pickup': Actions.pickup, 
            'drop': Actions.drop, 
            'toggle': Actions.toggle,
            'done': Actions.done,
        }

        self.verbose = verbose
        self.prev_query, self.prev_ans = 'None', 'None'
        self.eps_steps = 0
        self.eps_n_q = 0
        self.action = 'None'
        self.call = call

        if self.random_ans:
            self.vocab = list(Vocabulary(vocab_path=vocab_path).vocab.keys())
            shuffled_vocab = copy.deepcopy(self.vocab)
            random.shuffle(shuffled_vocab)
            self.word_mapping = {}
            for i in range(len(self.vocab)):
                self.word_mapping[self.vocab[i]] = shuffled_vocab[i]

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        obs['ans'] = 'None'
        self.prev_query, self.prev_ans = 'None', 'None'
        self.eps_steps = 0
        self.eps_n_q = 0
        self.action = 'None'
        if self.mode == 'GPT':
            self.oracle.reset(self.unwrapped)
        
        if self.random_ans:
            shuffled_vocab = copy.deepcopy(self.vocab)
            random.shuffle(shuffled_vocab)
            self.word_mapping = {}
            for i in range(len(self.vocab)):
                self.word_mapping[self.vocab[i]] = shuffled_vocab[i]

        return obs
    
    def step(self, action):
        """
        :param action:
        (1) minigrid action: 0 - 6
        (2) query action: generated by the self.env.get_answer function

        The answer is a string of words separated by space.
        """
        self.action = action
        self.eps_steps += 1
        assert isinstance(action, list)
        if isinstance(action[0], list):
            action = action[0]

        if action[0] in self.mini_grid_actions_map:
            if self.verbose:
                print('MiniGrid action:', action[0])
                self.unwrapped.verbose_text= 'MiniGrid action:' + action[0]
            obs, reward, done, info  = super().step(self.mini_grid_actions_map[action[0]])
            obs['ans'] = self.prev_ans
            return obs, reward, done, info

        obs, reward, done, info  = super().step(action=Actions.done)

        if self.query_limit > 0 and self.eps_n_q >= self.query_limit:
            ans = 'None'
        else:
            if self.mode == 'rule':
                ans = self.env.get_answer(action)
            elif self.mode == 'GPT':
                ans = self.oracle.get_answer(action)
            self.eps_n_q += 1
        
        obs['ans'] = ans

        if self.verbose:
            print('Q:', action)
            print('Ans:', ans)
            Q = ""
            for s in action:
                Q = Q+" "+s
        
            A = ""
            for s in ans:
                A = A+" "+s
                
            self.unwrapped.verbose_text= 'Q: ' + Q + '\n Ans: ' + A

        self.prev_query = ' '.join(action)
        self.prev_ans = ans

        if self.random_ans and ans != 'None' and random.random() < self.reliability:
            ans = ans.split()

            for i in range(len(ans)):
                w = ans[i]
                if w not in self.word_mapping: continue
                ans[i] = self.word_mapping[w]
            
            obs['ans'] = ' '.join(ans)
            
        return obs, reward, done, info
        