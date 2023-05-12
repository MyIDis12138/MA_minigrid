from __future__ import annotations

import gymnasium as gym
from MA_minigrid.envs.MAbabyai.core.MAroomgrid_level import MARoomGridLevel
from MA_minigrid.MA_core.objects import Oracle
from MA_minigrid.envs.MAbabyai.core.MA_verifier import MAObjDesc

class DangerRoomEnv(MARoomGridLevel):
    """
    Go to an object and avoid a danger room, the object may be in another room.
    Danger room located at one of the rest rooms. 
    Many distractors.
    """

    def __init__(
            self,             
            room_size: int = 5,
            num_rows: int = 3,
            num_cols: int = 3,
            num_dists: int = 3,
            doors_open: bool = True,
            all_doors: bool = True,
            call: bool = True,
            **kwargs
    ):
        self.names = ['jack', 'mary', 'tom', 'mike']
        self.doors_open = doors_open
        self.all_doors = all_doors
        self.num_dists = num_dists
        self.call = call
        super().__init__(
            agents_colors=['red'],
            room_size=room_size,
            num_rows=num_rows, 
            num_cols=num_cols,
            **kwargs
        )

    def gen_mission(self):
        self.agent_init_room = self._rand_int(0, self.num_rows * self.num_cols)
        self.place_agent_in_room_with_id(self.agent_init_room, 0)

        locked = False
        if self.all_doors:
            self.add_door(0, 0, door_idx=0, locked=locked)
            self.add_door(1, 0, door_idx=0, locked=locked)
            self.add_door(0, 1, door_idx=0, locked=locked)
            self.add_door(1, 1, door_idx=0, locked=locked)
            self.add_door(0, 2, door_idx=0, locked=locked)
            self.add_door(1, 2, door_idx=0, locked=locked)

            self.add_door(0, 0, door_idx=1, locked=locked)
            self.add_door(1, 0, door_idx=1, locked=locked)
            self.add_door(2, 0, door_idx=1, locked=locked)

            self.add_door(0, 1, door_idx=1, locked=locked)
            self.add_door(1, 1, door_idx=1, locked=locked)
            self.add_door(2, 1, door_idx=1, locked=locked)
        else:
            self.connect_all(self.agents)

        if self.doors_open:
            self.open_all_doors()

        while True:
            objs = self.add_distractors(num_distractors=self.num_dists, all_unique=True, type_set=['ball', 'box'])
            self.objs = objs
            self.obj = objs[0]
            #self.others_fav = self._rand_elem(objs[1:])
            if self.check_objs_reachable(self.agents): 
                break
        
        self._gen_instr()

        if not self.call:
            oracle = Oracle(color='red')
            self.place_in_room_with_id(self.agent_init_room, oracle)
            self.oracle = oracle

    def _gen_instr(self, agent_id = 0):
        fav_nameid, danger_nameid = self._rand_subset([0,1,2,3], 2)
        fav_room_id = self.room_from_pos(*self.obj.cur_pos).room_id
        danger_except = [self.agent_init_room, fav_room_id]
        danger_room_set = [i for i in range(self.num_rows * self.num_cols) if i not in danger_except]
        danger_room_id, other_room = self._rand_subset(danger_room_set,2)

        goal_instr = self.instrs_controller._make_instr("favoriate", agent_id=agent_id, obj_desc=MAObjDesc(self.obj.type, self.obj.color), name=f"{self.names[fav_nameid]}")
        danger_instr = self.instrs_controller._make_instr("danger_room", agent_id=agent_id, room_name=self.names[danger_nameid], room_id=danger_room_id)
        env_instr = self.instrs_controller._make_instr("constriant", agent_id=agent_id, cons_instr=danger_instr, goal_instr=goal_instr)
        self.instrs_controller.add_instr(instr=env_instr, agent_id=agent_id)

        self.knowledge_facts = [
                    '{} toy is {} {}'.format(self.names[fav_nameid], self.obj.color, self.obj.type),
                    '{} {} in room{}'.format(self.obj.color, self.obj.type, fav_room_id ),
                    '{}\'s room is room{}'.format(self.names[danger_nameid], danger_room_id), 
                ]
        # self.knowledge_facts = [
        #             '{} toy is {} {}'.format(self.names[fav_nameid], self.obj.color, self.obj.type),
        #             '{} {} in room{}'.format(self.obj.color, self.obj.type, fav_room_id ),
        #             '{}\'s room is room{}'.format(self.names[danger_nameid], danger_room_id), 
        #             '{} toy is {} {}'.format(self.names[danger_nameid], self.objs[1].color, self.objs[1].type),
        #             '{}\'s room is room{}'.format(self.names[fav_nameid], other_room), 
        #         ]
        
        for agent in self.agents:
            agent.mission = self.instrs_controller.surface(self, agent.id)
        self.missions = {agent_id: self.instrs_controller.surface(self, agent_id)}
        # self.encode = (fav_nameid, 
        #                danger_nameid,
        #                self.obj.color, 
        #                self.obj.type, 
        #                danger_room_id, 
        #                self.objs[1].color, 
        #                self.objs[1].type, 
        #                other_room
        #             )
        self.encode = (fav_nameid, 
                danger_nameid,
                self.obj.color, 
                self.obj.type, 
                danger_room_id, 
            )

    # map the question to the answer
    def get_answer(self, question, default_answer='I　dont　know'):
        #return default_answer
        if question[0] == 'what' and question[1] == 'is' and question[2] == f'{self.names[self.encode[0]]}' and question[3] == 'toy':
            return self.knowledge_facts[0]
        elif question[0] == 'where' and question[1] == 'is' and question[2] == f'{self.obj.color}' and question[3] == f'{self.obj.type}':
            return self.knowledge_facts[1]
        elif question[0] == 'where' and question[1] == 'is' and question[2] == f'{self.names[self.encode[1]]}' and question[3] == 'room':
            return self.knowledge_facts[2]
        else:
            return default_answer
