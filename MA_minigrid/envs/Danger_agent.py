from __future__ import annotations

import random
import numpy as np
from minigrid.core.actions import Actions
from MA_minigrid.MA_core.objects import Oracle, MAGoal, MAWall
from MA_minigrid.envs.MAbabyai.core.MAroomgrid_level import MARoomGridLevel

class DangerAgentEnv(MARoomGridLevel):
    """
    Three agnets, two are danger agent, another is target agent
    A level with danger agents
    """
    def __init__(
            self, 
            radius: int =1,
            call: bool = True,
            room_size: int = 7,
            view_size: int = 7,
            **kwargs
    ):
        self.robot_colors = self._rand_subset(['yellow', 'green', 'blue'], 2)
        super().__init__(
            agents_colors=["red"]+self.robot_colors,
            num_rows=1,
            num_cols=1,
            room_size=room_size, 
            agent_view_size=view_size,
            highlight=True,
            highlight_agents=[0],
            **kwargs
        )
        self.agent_start_pos = (1, 1)
        self.robot_poss = [
            (self.agent_start_pos[0], self.agent_start_pos[1]+2),
            (self.room_size-self.agent_start_pos[0]-1, self.agent_start_pos[1]+2),
            (self.agent_start_pos[0]+2, self.agent_start_pos[1]),
            (self.agent_start_pos[0]+2, self.room_size-self.agent_start_pos[1]-1),
        ]
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
            rand_pos = self._rand_int(0,2)
            self.put_obj(self.agents[id+1],*self.robot_poss[rand_pos+id*2])
            self.agents[id+1].dir = id

        self.robot_turns = [False, False]

    def _gen_instr(self, agent_id=0):
        danger_agent_id = self._rand_int(1, 3)
        danger_instr = self.instrs_controller._make_instr("danger_agent", agent_id=agent_id, danger_agent_id = danger_agent_id, radius = self.radius)
        goal_instr = self.instrs_controller._make_instr("goal", agent_id=agent_id, goal_pos = self.goal_pos) 
        env_instr = self.instrs_controller._make_instr("constriant", agent_id=agent_id, cons_instr=danger_instr, goal_instr=goal_instr)
        self.instrs_controller.add_instr(agent_id=0, instr=env_instr)

        for agent in self.agents:
            agent.mission = self.instrs_controller.surface(self, agent.id)

        self.knowledge_facts = ['danger robot is {}'.format(self.agents[danger_agent_id].color)]
        self.missions = {
            0: self.instrs_controller.surface(self, agent_id),
            1: "robot with no mission",
            2: "robot with no mission"
        }
        self.encode = (self.agents[danger_agent_id].color)

    def get_answer(self, question, default_answer='I　dont　know'):
        if question[0] == 'what' and question[1] == 'is' and question[2] == 'danger' and question[3] == 'robot':
            return self.knowledge_facts[0]
        else:
            return default_answer

    def step(self, action):
        actions = []
        actions += action
        # enable lazy action for danger agents

        for i in range(2):
            if (self.step_count+1)%2:
                fwd_cell = self.grid.get(*self.agents[i+1].front_pos)
                if self.robot_turns[i]:
                    actions.append(Actions.left)
                    self.robot_turns[i] = False
                    continue
                if isinstance(fwd_cell, MAWall):
                    actions.append(Actions.left)
                    self.robot_turns[i] = True
                else:
                    actions.append(Actions.forward) 
            else:
                actions.append(Actions.done)
        
        obs, reward, terminated, truncated, info = super().step(actions)
        if self.agents[0].terminated:
            terminated = True

        return obs, reward, terminated, truncated, info
    
    

