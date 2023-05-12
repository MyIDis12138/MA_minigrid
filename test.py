import MA_minigrid
import gymnasium as gym
from MA_minigrid.ManualControl import ManualControl
from MA_minigrid.wrappers import SingleAgentWrapper
from MA_minigrid.envs.MAbabyai.query_wrapper import MultiGrid_Safety_Query
from MA_minigrid.wrappers import KGWrapper

from MA_minigrid.envs.MAbabyai.utils.format import Vocabulary
from MA_minigrid.envs.MAbabyai.query_GPT import OracleGPT

key_to_action = {
    "a": [["left"]],
    "d": [["right"]],
    "w": [["forward"]],
    " ": [["toggle"]],
    "pageup": [["pickup"]],
    "pagedown": [["drop"]],
    "tab": [["pickup"]],
    "left shift": [["drop"]],
    "return": [["done"]],
}


questions_ground = [
            ["where", "is", "danger", "ground"], 
            ["where", "is", "danger", "zone"],
            ["where", "is", "danger", "floor"],
            ["where", "is", "danger", "robot"],
            ["what", "is", "danger", "ground"], 
            ["what", "is", "danger", "zone"],
            ["what", "is", "danger", "floor"],
]

questions_room = [
            ["what", "is", "jack", "room"],
            ["what", "is", "mary", "room"],
            ["what", "is", "tom", "room"],
            ["what", "is", "mike", "room"],
            ["where", "is", "jack", "room"],
            ["where", "is", "mary", "room"],
            ["where", "is", "tom", "room"],
            ["where", "is", "mike", "room"],
            ["what", "is", "danger", "room"],
            ["where", "is", "danger", "room"],
            ["what", "is", "jack", "toy"],
            ["what", "is", "mary", "toy"],
            ["what", "is", "tom", "toy"],
            ["what", "is", "mike", "toy"],
]

questions_agent =[
            ["what", "is", "danger", "robot"],
            ["where", "is", "danger", "robot"],
]


map = {
    'SQbabyai-DangerGround-v0': questions_ground,
    'SQbabyai-DangerGround_large-v0': questions_ground, # 'SQbabyai-DangerGround-v0
    'SQbabyai-DangerRoom-v0': questions_room,
    'SQbabyai-DangerAgent-v0': questions_agent,
}


if __name__ == "__main__":
    #env_name = 'SQbabyai-DangerGround-v0'
    #env_name = 'SQbabyai-DangerRoom-v0'
    #env_name = 'SQbabyai-DangerAgent-v0'
    env_name = 'SQbabyai-DangerAgent-v0'
    env = gym.make(env_name)
    Oracle = OracleGPT(Vocabulary(file_path='/home/yang/MA_minigrid/MA_minigrid/envs/MAbabyai/vocab/vocab1.txt'))
    query_mode = 'rule'
    env = SingleAgentWrapper(env)
    env = MultiGrid_Safety_Query(env, oracle=Oracle,verbose=True, mode=query_mode, vocab_path='/home/yang/MA_minigrid/MA_minigrid/envs/MAbabyai/vocab/vocab1.txt')
    env = KGWrapper(env, kg_repr='raw', mode='graph_overlap')
    manual_control = ManualControl(env, query_mode=query_mode, key_to_action=key_to_action, question_set=None)
    manual_control.start()

 