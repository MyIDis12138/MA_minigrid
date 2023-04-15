from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from minigrid.core.constants import COLOR_NAMES, DIR_TO_VEC
from MA_minigrid.MA_core.MAminigrid import MultiGridEnv
from minigrid.envs.babyai.core.verifier import dot_product, pos_next_to, ObjDesc

# Object types we are allowed to describe in language
OBJ_TYPES = ["box", "ball", "key", "door"]

# Object types we are allowed to describe in language
OBJ_TYPES_NOT_DOOR = list(filter(lambda t: t != "door", OBJ_TYPES))

# Locations are all relative to the agent's starting position
LOC_NAMES = ["left", "right", "front", "behind"]

# Environment flag to indicate that done actions should be
# used by the verifier
use_done_actions = os.environ.get("BABYAI_DONE_ACTIONS", False)

class MAInstrsController:
    def __init__(self):
        self.instrs = {}
    
    def add_instr(self, instr_str, agent_id, **kwargs):
        self.instrs.update({agent_id:self._make_instr(instr_str, agent_id, **kwargs)})

    def _make_instr(self, instr_str, agent_id, **kwargs):
        instr_map = {
            "goal" : GoToGoalInstr,

        }
        instr = instr_map[instr_str](agent_id, **kwargs)
        return instr

    def reset_verifier(self, env):
        for instr in self.instrs.values():
            instr.reset_verifier(env)
    
    def verify(self, actions):
        res = []
        for agent_id, action in enumerate(actions):
            res.append(self.instrs[agent_id].verify(action))
        return res
    
    def surface(self, env):
        res = []
        for instr in self.instrs.values():
            res.append(instr.surface(env))
        return res
    
    def num_navs_needed(self):
        res = []
        for instr in self.instrs.values():
            res.append(self._num_navs_needed(instr))
        return res

    def _num_navs_needed(self, instr) -> int:
        """
        Compute the maximum number of navigations needed to perform
        a simple or complex instruction
        """
        if isinstance(instr, GoToGoalInstr):
            return 2
        else:
            raise NotImplementedError(
                "instr needs to be an instance of PutNextInstr, ActionInstr, or SeqInstr"
            )
    
    def update_objs_poss(self):
        for instr in self.instrs.values():
            instr.update_objs_poss()



class MAInstr(ABC):
    """
    Base class for all instructions in the baby language
    """
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.env: MultiGridEnv

    @abstractmethod
    def surface(self, env):
        """
        Produce a natural language representation of the instruction
        """

        raise NotImplementedError

    def reset_verifier(self, env):
        """
        Must be called at the beginning of the episode
        """

        self.env = env

    @abstractmethod
    def verify(self, action):
        """
        Verify if the task described by the instruction is incomplete,
        complete with success or failed. The return value is a string,
        one of: 'success', 'failure' or 'continue'.
        """

        raise NotImplementedError

    def update_objs_poss(self):
        """
        Update the position of objects present in the instruction if needed
        """
        potential_objects = ("desc", "desc_move", "desc_fixed")
        for attr in potential_objects:
            if hasattr(self, attr):
                getattr(self, attr).find_matching_objs(self.env, use_location=False)

class MAActionInstr(MAInstr):
    """
    Base class for all action instructions (clauses)
    """

    def __init__(self, agent_id):
        super().__init__(agent_id=agent_id)

        # Indicates that the action was completed on the last step
        self.lastStepMatch = False

    def verify(self, action):
        """
        Verifies actions, with and without the done action.
        """

        if not use_done_actions:
            return self.verify_action(action)

        if action == self.env.actions.done:
            if self.lastStepMatch:
                return "success"
            return "failure"

        res = self.verify_action(action)
        self.lastStepMatch = res == "success"

    @abstractmethod
    def verify_action(self):
        """
        Each action instruction class should implement this method
        to verify the action.
        """

        raise NotImplementedError

class GoToGoalInstr(MAActionInstr):
    """
    Go to a specific goal
    """

    def __init__(self, agent_id, goal_pos, surface=None):
        super().__init__(agent_id=agent_id)
        self.goal_pos = goal_pos
        self._surface = surface

    def surface(self, env):
        return "go to the " + self.goal.surface(env)

    def verify_action(self, action):
        pos = self.env.agents[self.agent_id].cur_pos
        cur_cell = self.env.grid.get(*pos)

        if cur_cell and cur_cell.type == 'lava' and cur_cell.color == self.env.lava_colors[self.env.danger_color_idx]:
            return 'failure'

        if np.array_equal(pos, self.goal_pos):
            return "success"

        return "continue"
    
    def surface(self, env):
        return self._surface