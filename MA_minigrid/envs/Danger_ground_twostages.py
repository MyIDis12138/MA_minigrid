from __future__ import annotations

from MA_minigrid.envs.MAbabyai.core.MAroomgrid_level import MARoomGridLevel
from MA_minigrid.MA_core.objects import MALava, MAGoal, Oracle
import gymnasium as gym

import itertools as itt

class DangerGroundEnv(MARoomGridLevel):
    def __init__(self, room_size=7, call=True, goal_pos=None, **kwargs):
        self.lava_colors = ['yellow', 'blue']
        self.danger_names = ['ground','zone','floor']
        self.n_target = 2
        self.call = call
        self.goal_pos = goal_pos if goal_pos else (room_size - 2, room_size - 2)
        self.agent_start_pos = (2, 1)
        self.oracle_pos = (3, 1)
        super().__init__(
            agents_colors=['red'],
            num_rows=1,
            num_cols=1,
            room_size=room_size,
            agent_view_size=7,
            max_steps=100,
            **kwargs
        )

    def gen_mission(self):
        self.danger_color_idx, self.danger_name_idx = self._gen_lava()

        self.put_obj(MAGoal('green'), *self.goal_pos)

        if not self.call:
            self.grid.set(1, 1, None)
            self.grid.set(2, 1, None)
            self.grid.set(3, 1, None)
            self.grid.set(*self.oracle_pos, Oracle(color='red'))
        
        self.grid.set(*self.agent_start_pos, None)
        self.put_obj(self.agents[0], *self.agent_start_pos)
        self.agents[0].cur_pos = self.agent_start_pos
        self.agents[0].dir = 1
        
        self._gen_instr()

    def _gen_instr(self, agent_id = 0):
        danger_instr = self.instrs_controller._make_instr("danger_ground", agent_id=agent_id, lava_color=self.lava_colors[self.danger_color_idx], lava_name=self.danger_names[self.danger_name_idx])
        goal_instr = self.instrs_controller._make_instr("goal", agent_id=agent_id, goal_pos=self.goal_pos)
        env_instr = self.instrs_controller._make_instr("constriant", agent_id=agent_id, cons_instr=danger_instr, goal_instr=goal_instr)
        self.instrs_controller.add_instr(instr=env_instr, agent_id=agent_id)

        for agent in self.agents:
            agent.mission = self.instrs_controller.surface(self, agent.id)

        self.knowledge_facts = ['danger {} is {}'.format(self.danger_names[self.danger_name_idx], self.lava_colors[self.danger_color_idx])]
        self.missions = {agent_id: self.instrs_controller.surface(self, agent_id)}

    def _gen_lava(self):
        danger_name_idx= self._rand_int(0, len(self.danger_names))
        danger_color_idx= self._rand_int(0, self.n_target)
        height, width = self.room_size, self.room_size
        # Place obstacles (lava or walls)
        self.num_crossings = self._rand_int(1, (self.room_size - 3))
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[:self.num_crossings]  # sample random rivers
        rivers_v = sorted([pos for direction, pos in rivers if direction is v])
        rivers_h = sorted([pos for direction, pos in rivers if direction is h])
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        colored_tiles = [set(), set()]
        for i, j in obstacle_pos:
            color_idx = self._rand_int(0,2)
            self.grid.set(i, j, MALava(self.lava_colors[color_idx]))
            colored_tiles[color_idx].add((i, j))

        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        openings = set()
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1]))
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1]))
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False

            self.grid.set(i, j, MALava(self.lava_colors[1- danger_color_idx]))
            openings.add((i, j))
            if (i, j) in colored_tiles[danger_color_idx]:
                a, b = None, None
                while colored_tiles[1-danger_color_idx]:
                    a, b = colored_tiles[1-danger_color_idx].pop()
                    if (a, b) not in openings:
                        break
                    if not colored_tiles[1-danger_color_idx]:
                        a, b = None, None
                        break
                if a is not None:
                    self.grid.set(a, b, MALava(self.lava_colors[danger_color_idx]))
            #self.grid.set(i, j, None)

        return danger_color_idx, danger_name_idx

    def get_answer(self, question, default_answer='I　dont　know'):
        if question[0] == 'what' and question[1] == 'is' and question[2] == 'danger' and question[3] == self.danger_names[self.danger_name_idx]:
            return self.knowledge_facts[0]
        else:
            return default_answer
        