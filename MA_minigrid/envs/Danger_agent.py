from __future__ import annotations

import random
import numpy as np
from minigrid.core.actions import Actions
from MA_minigrid.MA_core.objects import Oracle, MAGoal
from MA_minigrid.envs.MAbabyai.core.MAroomgrid_level import MARoomGridLevel

class DangerAgentEnv(MARoomGridLevel):
    """
    Three agnets, two are danger agent, another is target agent
    A level with danger agents
    """
    def __init__(
            self, 
            radius=1,
            call = True,
            room_size=9,
            view_size=7
    ):
        self.robot_action = [Actions.left, Actions.right, Actions.forward, Actions.done]
        self.robot_colors = self._rand_subset(['yellow', 'green', 'blue'], 2)
        self.action_fwd = [0.1, 0.1, 0.3, 0.5]
        self.action_turn = [0.2, 0.2, 0.1, 0.5]
        self.agent_start_pos = (1, 1)
        super().__init__(
            agents_colors=["red"]+self.robot_colors,
            num_rows=1,
            num_cols=1,
            room_size=room_size, 
            agent_view_size=view_size,
            highlight=False,
        )
        self.call = call
        self.radius = radius
        self.goal_pos = (room_size - 2, room_size - 2)
        
    def gen_mission(self):
        # place the oracle
        if not self.call:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.oracle_pos = (3, 1)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))

        self.put_obj(MAGoal(color='green'), *self.goal_pos)
        self._gent_agents()

        self._gen_instr()

    def _gent_agents(self):
        self.robot_colors = self._rand_subset(['yellow', 'green', 'blue'], 2)

        for id, color in enumerate(self.robot_colors):
            self.agents[id+1].color = color

        self.put_obj(self.agents[0],*self.agent_start_pos)
        self.agents[0].dir = 0
        for id in range(2):
            self.place_agent(id+1)

    def _gen_instr(self, agent_id=0):
        danger_instr = self.instrs_controller._make_instr("danger_agent", agent_id=agent_id, danger_agent_id = 1, radius = self.radius)
        goal_instr = self.instrs_controller._make_instr("goal", agent_id=agent_id, goal_pos = self.goal_pos) 
        env_instr = self.instrs_controller._make_instr("constriant", agent_id=agent_id, cons_instr=danger_instr, goal_instr=goal_instr)
        self.instrs_controller.add_instr(agent_id=0, instr=env_instr)

        for agent in self.agents:
            agent.mission = self.instrs_controller.surface(self, agent.id)

        self.useful_answers = ['danger robot is {}'.format(self.robot_colors[0])]
        self.missions = {
            0: self.instrs_controller.surface(self, agent_id),
            1: "robot with no mission",
            2: "robot with no mission"
        }

    def get_answer(self, question, default_answer='I　dont　know'):
        if question[0] == 'what' and question[1] == 'is' and question[2] == 'danger' and question[3] == 'robot':
            return self.useful_answers[0]
        else:
            return default_answer

    def step(self, action):
        actions = []
        actions += action
        for i in range(2):
            fwd_cell = self.grid.get(*self.agents[i+1].front_pos)
            if fwd_cell is None:
                actions+=random.choices(self.robot_action, weights=self.action_fwd, k=1)
            else:
                actions+=random.choices(self.robot_action, weights=self.action_turn, k=1)    
        
        obs, reward, terminated, truncated, info = super().step(actions)
        if self.agents[0].terminated:
            terminated = True

        return obs, reward, terminated, truncated, info
