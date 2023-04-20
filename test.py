import MA_minigrid
import gymnasium as gym
from MA_minigrid.ManualControl import ManualControl
from MA_minigrid.wrappers import SingleAgentWrapper
from MA_minigrid.envs.MAbabyai.query_wrapper import MultiGrid_Safety_Query

key_to_action = {
    "a": [["left"]],
    "d": [["right"]],
    "w": [["forward"]],
    " ": [["toggle"]],
    "pageup": [["pickup"]],
    "pagedown": [["drop"]],
    "tab": [["pickup"]],
    "left shift": [["drop"]],
    "enter": [["done"]],
}


questions_ground = [
            ["where", "is", "danger", "ground"], 
            ["where", "is", "danger", "zone"],
            ["where", "is", "danger", "area"],
            ["where", "is", "danger", "robot"],
            ["what", "is", "danger", "ground"], 
            ["what", "is", "danger", "zone"],
            ["what", "is", "danger", "area"],
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
    'SQbabyai-DangerRoom-v0': questions_room,
    'SQbabyai-DangerAgent-v0': questions_agent,
}


if __name__ == "__main__":
    #env_name = 'SQbabyai-DangerGround-v0'
    #env_name = 'SQbabyai-DangerRoom-v0'
    env_name = 'SQbabyai-DangerAgent-v0'
    env = gym.make(env_name)
    env = SingleAgentWrapper(env)
    env = MultiGrid_Safety_Query(env, verbose=True)
    manual_control = ManualControl(env, key_to_action=key_to_action, question_set=map[env_name])
    manual_control.start()

