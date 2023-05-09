import os 
import re
from typing import List, Tuple

import openai

from MA_minigrid.MA_core.MAminigrid import MultiGridEnv
from MA_minigrid.envs.MAbabyai.utils.format import Vocabulary

openai.api_key  = os.getenv('OPENAI_API_KEY')

class OracleCentralizedGPT:
    def __init__(self, vocab:Vocabulary, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.vocab = vocab
        self.memory = self.build_memory()

    def build_memory(self):
        # Initialize memory as an empty dictionary
        history = {}
        return history

    def get_completion(self, prompt):
        messages = [{"role": "system", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0.1, 
        )   
        return response.choices[0].message["content"]
    
    def gen_prompts(self, queries:str, ids:List[int]):
        assert len(queries) == len(ids), "The number of queries, missions and knowledge facts should be the same."
        
        agent_quries = ""
        for query, id in zip(queries, ids):
            agent_quries += f"""\
            Agent question: ```{query}```
            The agents mission: ```{self.agent_mission[id]}```.
            The knowledge facts: ```{self.knowledge_facts[id]}```.\n
            """
        prompt = f"""\
        You are an oracle in a grid world. You are trying to answer the questions asked by the agents strictly based on the knowledge you have.
        If the agents ask you a question that you don't know the answer to, reply with "I dont know".
        The words available for your response: ```{self.vocab.vocab}```, your can not use any other words out of them.
        {agent_quries}
        Format your response as a list of answers separated by commas.
        """
        return prompt

    def get_answer(self, queries:Tuple[Tuple[str]], env_ids:Tuple[int]):
        assert len(queries) == len(env_ids), "The number of queries, missions and knowledge facts should be the same."
        answer = tuple([] for _ in range(len(queries)))
        ids_GPT = []
        for query in queries:
            query = " ".join(query)

        query_GPT = "" 
        for id, query in zip(env_ids, queries):
            if query in self.memory[self.env_encodes[id]]:
                answer[id] = self.memory[self.env_encodes[id]][query]
            else:
                ids_GPT.append(id)
                query_GPT += query + "\n"
                
        prompt = self.gen_prompts(query_GPT, ids_GPT)
        completion = self.get_completion(prompt)
        GPT_ans = completion.split(",")

        for id, ans in zip(ids_GPT, GPT_ans):
            answer[id] = ans
            self.memory[self.env_encodes[id]].update({queries[id]: ans})

        return answer
    

    def reset(self, env: MultiGridEnv, env_id: int):
        self.knowledge_facts[env_id] = env.knowledge_facts
        self.agent_mission[env_id] = env.missions
        self.env_encodes[env_id] = env.encode
        if env.encode not in self.memory:
            self.memory.update({env.encode: {}})

    def reset_all(self, envs: List[MultiGridEnv]):
        self.knowledge_facts = []
        self.agent_mission = []
        self.env_encodes = []
        for env in envs:
            self.knowledge_facts.append(env.knowledge_facts)
            self.agent_mission.append(env.missions)
            self.env_encodes.append(env.encode)
            if env.encode not in self.memory:
                self.memory.update({env.encode: {}})


class OracleGPT:
    def __init__(self, vocab:Vocabulary, model: str = "gpt-3.5-turbo", env_num: int = 1):
        self.model = model
        self.vocab = vocab
        self.memory = self.build_memory()
        self.knowledge_facts = {}
        self.agent_mission = {}

    def build_memory(self):
        # Initialize memory as an empty list
        history = {}
        return history

    def get_completion(self, prompt):
        messages = [{"role": "system", "content": prompt}]
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1, 
                )   
                if response:
                    break
            except:
                continue

        return response.choices[0].message["content"]
    
    def gen_prompts(self, query:str, env_encode:int):
        prompt = f"""\
        As an oracle in a grid world, your role is to answer questions posed by agents based solely on your existing knowledge. If you encounter a question for which you do not know the answer, simply respond with "I dont know."
        
        IMPORTANT: When formulating your responses, you must STRICTLY only use the words found in: ```{self.vocab.vocab}```. It is crucial to adhere to this rule and avoid using words outside of this list.
        
        Knowledge of the environment: ```{self.knowledge_facts[env_encode]}```
        Agent's mission: ```{self.agent_mission[env_encode]}```
        Agent's question: ```{query}```
        
        Once again, remember to use ONLY words from the allowed vocabulary: ```{self.vocab.vocab}```. Failure to follow this constraint is not acceptable.\
        """
        return prompt

    def get_answer(self, query:Tuple[str], env_encode:Tuple):
        query = " ".join(query)
        if query in self.memory[env_encode]:
            return self.memory[env_encode][query]
        prompt = self.gen_prompts(query, env_encode)
        completion = self.get_completion(prompt)
        while not self.is_response_valid(completion):
            prompt += f"\nInvalid response: {completion}"
            completion = self.get_completion(prompt)
            print("Invalid response regenerate: ", completion)
        self.memory[env_encode].update({query: completion})
        return completion
    
    def is_response_valid(self, response):
        response_words = re.findall("([a-z0-9]+)", response.lower())
        self.preprocess(response_words)
        for word in response_words:
            if word not in self.vocab.vocab:
                print(f"Invalid response: {response}, {word} is not in the vocab.")
                return False
        return True
    
    def preprocess(self, ans):
        if 'is' in ans:
            ans.remove('is')
        if 'in' in ans:
            ans.remove('in')
        if 's' in ans:
            ans.remove('s')
        if 'don' in ans:
            ans[ans.index('don')] = 'dont'
        if 't' in ans:
            ans.remove('t')


    def reset(self, env: MultiGridEnv):
        if env.encode not in self.knowledge_facts:
            self.knowledge_facts.update({env.encode: env.knowledge_facts})
            self.agent_mission.update({env.encode: env.missions})
        if env.encode not in self.memory:
            self.memory.update({env.encode: {}})

    def reset_all(self, envs: List[MultiGridEnv]):
        for env in envs:
            if env.encode not in self.knowledge_facts:
                self.knowledge_facts.update({env.encode: env.knowledge_facts})
                self.agent_mission.update({env.encode: env.missions})
            if env.encode not in self.memory:
                self.memory.update({env.encode: {}})