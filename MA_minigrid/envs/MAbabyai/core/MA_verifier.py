from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np

from minigrid.envs.babyai.core.roomgrid_level import RejectSampling
from MA_minigrid.MA_core.MAconstants import COLOR_NAMES, DIR_TO_VEC
from MA_minigrid.MA_core.MAminigrid import MultiGridEnv
from minigrid.envs.babyai.core.verifier import dot_product

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
        self.instr_map = {
            "goal" : GoToGoalInstr,
            "favoriate" : GoToFavoriteInstr,
            "danger_ground" : DangerGroundInstr,
            "danger_room" : DangerRoomInstr,
            "danger_agent" : DangerAgentInsr,
            "sequence" : MASeqInstr,
            "constriant" : GoalConstrainedInstr,
        }
    
    def add_instr(self, agent_id, instr: MAInstr):
        self.instrs.update({agent_id:instr})

    def _make_instr(self, instr_str: str, agent_id: int, **kwargs):
        assert instr_str in self.instr_map, "Unknown instruction type: {}".format(instr_str)
        instr = self.instr_map[instr_str](agent_id, **kwargs)
        return instr

    def reset_verifier(self, env):
        for instr in self.instrs.values():
            instr.reset_verifier(env)
    
    def verify(self, actions):
        res = []
        for agent_id, action in enumerate(actions):
            if agent_id in self.instrs.keys():
                res.append(self.instrs[agent_id].verify(action))
            else:
                res.append("continue")
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
        if isinstance(instr, MASeqInstr):
            return self._num_navs_needed(instr.instr_a) + self._num_navs_needed(instr.instr_b)
        elif isinstance(instr, GoToGoalInstr):
            return 0
        elif isinstance(instr, GoToFavoriteInstr):
            return 2
        elif isinstance(instr, DangerGroundInstr):
            return 1
        elif isinstance(instr, DangerRoomInstr):
            return 1
        elif isinstance(instr, DangerAgentInsr):
            return 1
        else:
            raise NotImplementedError(
                "instr needs to be an instance of PutNextInstr, ActionInstr, or SeqInstr"
            )
    
    def update_objs_poss(self):
        for instr in self.instrs.values():
            instr.update_objs_poss()

    def surface(self, env, agent_id):
        if agent_id in self.instrs.keys():
            return self.instrs[agent_id].surface(env)
        else:
            return None

    def validate_instrs(self, instrs, env):
        """
        Perform some validation on the generated instructions
        """
        # Gather the colors of locked doors
        for instr in instrs: 
            colors_of_locked_doors = []
            if hasattr(env, "unblocking") and env.unblocking:
                for i in range(env.num_cols):
                    for j in range(env.num_rows):
                        room = env.get_room(i, j)
                        for door in room.doors:
                            if door and door.is_locked:
                                colors_of_locked_doors.append(door.color)


            if isinstance(instr, MAActionInstr):
                if not hasattr(env, "unblocking") or not env.unblocking:
                    continue
                # TODO: either relax this a bit or make the bot handle this super corner-y scenarios
                # Check that the instruction doesn't involve a key that matches the color of a locked door
                potential_objects = ("desc", "desc_move", "desc_fixed")
                for attr in potential_objects:
                    if hasattr(instr, attr):
                        obj = getattr(instr, attr)
                        if obj.type == "key" and obj.color in colors_of_locked_doors:
                            raise RejectSampling(
                                "cannot do anything with/to a key that can be used to open a door"
                            )
                continue

            if isinstance(instr, MASeqInstr):
                self.validate_instrs([instr.instr_a], env)
                self.validate_instrs([instr.instr_b], env)
                continue


            assert False, "unhandled instruction type"
        
class MAObjDesc:
    """
    Description of a set of objects in an environment
    """

    def __init__(self, type, color=None, loc=None):
        assert type in [None, *OBJ_TYPES], type
        assert color in [None, *COLOR_NAMES], color
        assert loc in [None, *LOC_NAMES], loc

        self.color = color
        self.type = type
        self.loc = loc

        # Set of objects possibly matching the description
        self.obj_set = []

        # Set of initial object positions
        self.obj_poss = []

    def __repr__(self):
        return f"{self.color} {self.type} {self.loc}"

    def surface(self, agent_id, env):
        """
        Generate a natural language representation of the object description
        """

        self.find_matching_objs(env, agent_id)
        assert len(self.obj_set) > 0, "no object matching description"

        if self.type:
            s = str(self.type)
        else:
            s = "object"

        if self.color:
            s = self.color + " " + s

        if self.loc:
            if self.loc == "front":
                s = s + " in front of you"
            elif self.loc == "behind":
                s = s + " behind you"
            else:
                s = s + " on your " + self.loc

        # Singular vs plural
        if len(self.obj_set) > 1:
            s = "a " + s
        else:
            s = "the " + s

        return s

    def find_matching_objs(self, env, agent_id, use_location=True):
        """
        Find the set of objects matching the description and their positions.
        When use_location is False, we only update the positions of already tracked objects, without taking into account
        the location of the object. e.g. A ball that was on "your right" initially will still be tracked as being "on
        your right" when you move.
        """

        if use_location:
            self.obj_set = []
            # otherwise we keep the same obj_set

        self.obj_poss = []

        agent_room = env.room_from_pos(*env.agents[agent_id].cur_pos)

        for i in range(env.grid.width):
            for j in range(env.grid.height):
                cell = env.grid.get(i, j)
                if cell is None:
                    continue

                if not use_location:
                    # we should keep tracking the same objects initially tracked only
                    already_tracked = any([cell is obj for obj in self.obj_set])
                    if not already_tracked:
                        continue

                # Check if object's type matches description
                if self.type is not None and cell.type != self.type:
                    continue

                # Check if object's color matches description
                if self.color is not None and cell.color != self.color:
                    continue

                # Check if object's position matches description
                if use_location and self.loc in ["left", "right", "front", "behind"]:
                    # Locations apply only to objects in the same room
                    # the agent starts in
                    if not agent_room.pos_inside(i, j):
                        continue

                    # Direction from the agent to the object
                    v = (i - env.agents[agent_id].cur_pos[0], j - env.agents[agent_id].cur_pos[1])

                    # (d1, d2) is an oriented orthonormal basis
                    d1 = DIR_TO_VEC[env.agents[agent_id].dir]
                    d2 = (-d1[1], d1[0])

                    # Check if object's position matches with location
                    pos_matches = {
                        "left": dot_product(v, d2) < 0,
                        "right": dot_product(v, d2) > 0,
                        "front": dot_product(v, d1) > 0,
                        "behind": dot_product(v, d1) < 0,
                    }

                    if not (pos_matches[self.loc]):
                        continue

                if use_location:
                    self.obj_set.append(cell)
                self.obj_poss.append((i, j))

        return self.obj_set, self.obj_poss


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
                getattr(self, attr).find_matching_objs(self.env, self.agent_id, use_location=False)

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

class MASeqInstr(MAInstr, ABC):
    """
    Sequence of instructions
    """

    def __init__(self, agent_id, instr_a, instr_b, strict=False):
        super().__init__(agent_id=agent_id)
        assert isinstance(instr_a, MAActionInstr) or isinstance(instr_a, GoalConstrainedInstr)
        assert isinstance(instr_b, MAActionInstr) or isinstance(instr_b, GoalConstrainedInstr)
        self.instr_a = instr_a
        self.instr_b = instr_b
        self.strict = strict

class GoalConstrainedInstr(MASeqInstr):
    """
    Base class for all instructions that are constrained by instr_a to a goal instr_b
    """

    def __init__(self, agent_id, cons_instr, goal_instr):
        assert isinstance(cons_instr, MAActionInstr)
        assert isinstance(goal_instr, MAActionInstr)
        super().__init__(agent_id, cons_instr, goal_instr)
        self.instr_a.agent_id = agent_id
        self.instr_b.agent_id = agent_id

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.instr_a.reset_verifier(env)
        self.instr_b.reset_verifier(env)
        self.a_done = False
        self.b_done = False

    def verify(self, action):
        if self.a_done != "success":
            self.a_done = self.instr_a.verify(action)

        if self.b_done != "success":
            self.b_done = self.instr_b.verify(action)
        
        if self.a_done == "failure" or self.b_done == "failure":
            return "failure"

        if self.a_done == "continue" and self.b_done == "success":
            return "success"

        return "continue"
    
    def surface(self, env):
        return f"{self.instr_a.surface(env)}, {self.instr_b.surface(env)}"

class DangerGroundInstr(MAActionInstr):
    """
    Avoid the danger lava with specific color and name
    """

    def __init__(self, agent_id, lava_color, lava_name, surface=None):
        super().__init__(agent_id=agent_id)
        self.lava_color = lava_color
        self.lava_name = lava_name
        self._surface = surface

    def verify_action(self, action):
        pos = self.env.agents[self.agent_id].cur_pos
        cur_cell = self.env.grid.get(*pos)

        if cur_cell and cur_cell.type == 'lava' and cur_cell.color == self.lava_color:
            return 'failure'
        
        return "continue"
    
    def surface(self, env):
        if self._surface:
            return self._surface
        return f"avoid danger {self.lava_name}"

class DangerRoomInstr(MAActionInstr):
    """
    Avoid the danger room with specific room name and room id
    """

    def __init__(self, agent_id, room_name, room_id, surface=None):
        super().__init__(agent_id=agent_id)
        self.room_name = room_name
        self.room_id = room_id
        self._surface = surface

    def verify_action(self, action):
        pos = self.env.agents[self.agent_id].cur_pos
        cur_room = self.env.room_from_pos(*pos)

        if cur_room.room_id == self.room_id:
            return "failure"

        return "continue"
    
    def surface(self, env):
        if self._surface:
            return self._surface
        return f"avoid {self.room_name} room"

class DangerAgentInsr(MAActionInstr):
    """
    Avoid the danger agent with specific agent id
    """

    def __init__(self, agent_id, danger_agent_id, radius, surface=None):
        super().__init__(agent_id=agent_id)
        self.danger_agent_id = danger_agent_id
        self.radius = radius
        self._surface = surface

    def verify_action(self, action):
        pos = self.env.agents[self.agent_id].cur_pos
        danger_agent_pos = self.env.agents[self.danger_agent_id].cur_pos

        if np.linalg.norm(np.array(pos)-np.array(danger_agent_pos)) <= self.radius:
            return "failure"

        return "continue"
    
    def surface(self, env):
        if self._surface:
            return self._surface
        return "avoid danger robot"

class GoToGoalInstr(MAActionInstr):
    """
    Go to a specific goal
    """

    def __init__(self, agent_id, goal_pos, surface=None):
        super().__init__(agent_id=agent_id)
        self.goal_pos = goal_pos
        self._surface = surface

    def verify_action(self, action):
        pos = self.env.agents[self.agent_id].cur_pos

        if np.array_equal(pos, self.goal_pos):
            return "success"

        return "continue"
    
    def surface(self, env):
        if self._surface:
            return self._surface
        return "go to the goal"

class GoToFavoriteInstr(MAActionInstr):
    """
    Go next to (and look towards) an object matching a given description
    eg: go to the door
    """

    def __init__(
            self, 
            agent_id: int, 
            obj_desc: MAObjDesc, 
            name: str, 
            surface: str = None
        ):
        super().__init__(agent_id=agent_id)
        self.desc = obj_desc
        self.name = name
        self._surface = surface
        self.agent_id = agent_id

    def reset_verifier(self, env):
        super().reset_verifier(env)
        self.desc.find_matching_objs(self.env, self.agent_id)

    def surface(self, env):
        if self._surface:
            return self._surface
        return f"go to {self.name} favorite toy"

    def verify_action(self, action):

        for pos in self.desc.obj_poss:
            # If the agent is next to (and facing) the object
            if np.array_equal(pos, self.env.agents[self.agent_id].front_pos):
                return "success"

        return "continue"

