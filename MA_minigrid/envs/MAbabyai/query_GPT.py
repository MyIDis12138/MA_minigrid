import os 
import re
import time

import json
import ast
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
        self.invail_vocab = []
        self.knowledge_facts = {}
        self.agent_mission = {}

    def build_memory(self):
        # Initialize memory as an empty list
        history = {}
        return history

    def get_completion(self, prompt, temperature=0.1):
        messages = [{"role": "system", "content": prompt}]
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature, 
                )   
                if response:
                    break
            except Exception as e:
                if "rate limit reached" in str(e).lower():
                    print("Rate limit reached, waiting before retrying...")
                    time.sleep(60)
                continue
        return response.choices[0].message["content"]
    
    def gen_prompts(self, query:str, env_encode:int):
        prompt = f"""\
        As an oracle in a grid world, your role is to answer questions posed by agents based solely on your existing knowledge. If you encounter a question for which you do not know the answer, only respond with "I dont know."
        
        IMPORTANT: When formulating your responses, you MUST STRICTLY only use the words in vocabulary: \
        ```{self.vocab.vocab}```. It is crucial to obey to this rule and avoid using words outside of this list.
        
        Knowledge of the environment: ```{self.knowledge_facts[env_encode]}```
        Agent's mission: ```{self.agent_mission[env_encode]}```
        Agent's question: ```{query}```
        
        Once again, remember to ONLY use the words from the allowed vocabulary: ```{self.vocab.vocab}```. 
        Any question you can not answer based on your knowledge ONLY reply with `I dont know`.\
        """
        return prompt
    
    def regen_prompts(self, fail_prompt:str, env_encode:int):
        prompt = f"""\
        You are a respones filter of an oracle in a grid world. 
        You are trying to filter the responses generated by the oracle based on the knowledge you have.
        IMPORTANT: Your response also MUST STRICTLY only use the words in vocabulary in your response: ```{self.vocab.vocab}```.
        You have the knwoledge of the environment: ```{self.knowledge_facts[env_encode]}```.

        The oracle's response: ```{fail_prompt}``` has used the words outside of the allowed vocabulary.
        Find the information of this response that could be represented by the words in the allowed vocabulary and regenerate the response.
        If this response is shown to be unanswerable, reply with "I dont know".

        Example: If the response is "mike's room is unknown to me", reply with "I dont know".
                 If this response is "mary toy is loacted in room3" where "located" is out of vocabulary, reply with "mary toy is in room3".
 
        Keep in mind that you can only use the words in the allowed vocabulary: ```{self.vocab.vocab}```.
        Make a short response, without any explain or indication. Directly give your result.\
        """
        return prompt
    #        You should output a list of indexes of words in the vocabulary, split your answer with comma.\
    

    def postprocess(self, completion):
        answer = completion.split(",")
        answer = [int(ans) for ans in answer]
        answer = [self.vocab.vocab[ans] for ans in answer]
        response = " ".join(answer)
        return response



    def get_answer(self, query:Tuple[str], env_encode:Tuple):
        query = " ".join(query)
        temp = 0.1
        if query in self.memory[env_encode]:
            return self.memory[env_encode][query]
        
        prompt = self.gen_prompts(query, env_encode)
        completion = self.get_completion(prompt, temperature=temp)

        vaild, word = self.is_response_valid(completion)
        while not vaild:
            self.invail_vocab.append(word)
            prompt += f"\n\n Invaild response example: {completion}."
            prompt = self.regen_prompts(completion, env_encode)
            if self.count_tokens(prompt) > 4096:
                prompt = self.gen_prompts(query, env_encode)
            completion = self.get_completion(prompt, temperature=0.5)
            #completion = self.postprocess(completion)
            print(f"Regenerating response: {completion}")
            vaild, word = self.is_response_valid(completion)

        self.memory[env_encode].update({query: completion})
        return completion
    
    def is_response_valid(self, response):
        response_words = re.findall("([a-z0-9]+)", response.lower())
        self.preprocess(response_words)
        for word in response_words:
            if word not in self.vocab.vocab:
                print(f"Invalid response: {response}, {word} is not in the vocab.")
                return False, word
        return True, None
    
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

    def store_memory(self, path):
        json_data = {str(k): v for k, v in self.memory.items()}

        with open(path, "w") as file:
            json.dump(json_data, file)

    def load_memory(self, path):
        with open(path, "r") as file:
            json_data = json.load(file)

        # Convert string keys back to tuples
        self.memory = {ast.literal_eval(k): v for k, v in json_data.items()}

    def count_tokens(self, prompt):
        return len(prompt.split())