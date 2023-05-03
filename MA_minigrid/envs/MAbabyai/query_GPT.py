import os 
from typing import List, Tuple

import openai

from MA_minigrid.MA_core.MAminigrid import MultiGridEnv
from MA_minigrid.envs.MAbabyai.utils.format import Vocabulary

openai.api_key  = os.getenv('OPENAI_API_KEY')

class OracleGPT:
    def __init__(self, vocab:Vocabulary, model: str = "gpt-3.5-turbo"):
        self.model = model
        self.vocab = vocab
        self.memory = self.build_memory()

    def build_memory(self):
        # Initialize memory as an empty list
        history = {}
        return history

    def get_completion(self, prompt):
        messages = [{"role": "system", "content": prompt}]
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0, 
        )   
        return response.choices[0].message["content"]
    
    def gen_prompts(self, query:str):
        prompt = f"""\
        You are an oracle in a grid world. You are trying to answer the questions asked by the agents strictly based on the knowledge you have.\
        If the agents ask you a question that you don't know the answer to, reply with "I dont know".\
        The agents mission: ```{self.agent_mission}```.\
        The knowledge of the environment you have is as follows: ```{self.knowledge_facts}```. \
        The words available for your response: {self.vocab.vocab}, your response can not contain any other words not in them.\
        Agent question: ```{query}```\
        """
        return prompt

    def get_answer(self, query:Tuple[str], env_encode:Tuple):
        query = " ".join(query)
        if query in self.memory[env_encode]:
            return self.memory[env_encode][query]
        prompt = self.gen_prompts(query)
        completion = self.get_completion(prompt)
        self.memory[env_encode].update({query: completion})
        return completion
    

    def reset(self, env: MultiGridEnv):
        self.knowledge_facts = env.knowledge_facts
        self.agent_mission = env.missions
        if env.encode not in self.memory:
            self.memory.update({env.encode: {}})
