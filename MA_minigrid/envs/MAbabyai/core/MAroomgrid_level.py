from __future__ import annotations

from MA_minigrid.MA_core.MAroomgrid import MARoomgrid, Agent
from MA_minigrid.envs.MAbabyai.core.MA_verifier import GoToGoalInstr, MAInstrsController
from minigrid.minigrid_env import MissionSpace
from minigrid.envs.babyai.core.roomgrid_level import BabyAIMissionSpace, RejectSampling
from minigrid.envs.babyai.core.verifier import (
    ActionInstr,
    AfterInstr,
    AndInstr,
    BeforeInstr,
    PutNextInstr,
    SeqInstr,
)

class MARoomGridLevel(MARoomgrid):
    """
    Base for levels based on MARoomGrid.
    A level, generates missions generated from
    one or more patterns. Levels should produce a family of missions
    of approximately similar difficulty.
    """

    def __init__(
            self, 
            agents_index: list[int] = [0], 
            room_size=8, 
            max_steps: int | None = None, 
            **kwargs
        ):
        mission_space = BabyAIMissionSpace()
        self.instrs_controller = MAInstrsController()

        # If `max_steps` arg is passed it will be fixed for every episode,
        # if not it will vary after reset depending on the maze size.
        self.fixed_max_steps = False
        if max_steps is not None:
            self.fixed_max_steps = True
        else:
            max_steps = 0  # only for initialization
        super().__init__(
            agents_index=agents_index,
            room_size=room_size,
            mission_space=mission_space,
            **kwargs,
        )

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)

        # Recreate the verifier
        self.instrs_controller.reset_verifier(self)

        # Compute the time step limit based on the maze size and instructions
        nav_time_room = self.room_size**2
        nav_time_maze = nav_time_room * self.num_rows * self.num_cols
        num_navs = self.instrs_controller.num_navs_needed()

        if not self.fixed_max_steps:
            for num_nav in num_navs:
                max_steps = num_nav * nav_time_maze
                self.max_steps = max(max_steps, self.max_steps)

        return obs
    
    def step(self, actions):
        obs, reward, terminated, truncated, info = super().step(actions)

        for action in actions:
            # If we drop an object, we need to update its position in the environment
            if action == self.actions.drop:
                self.update_objs_poss()

        # If we've successfully completed the mission
        status = True
        status = self.instrs_controller.verify(actions)
        rewards = []

        info['success'] = False
        for id, s in enumerate(status):
            reward = 0 
            if s == "success":
                self.agents[id].terminated = True
                info['success'] = True
                reward = self._reward(id)
            elif s == "failure":
                self.agents[id].terminated = True
            elif status == 'continue':
                self.agents[id].terminated = False
            rewards.append(reward)

        terminated = all([agent.terminated for agent in self.agents])

        if self.unwrapped.steps_remaining <= 0:
            truncated = True
                
        return obs, rewards, terminated, truncated, info
    
    def update_objs_poss(self, instr=None):
        if instr is None:
            instr = self.instrs_controller
        if (
            isinstance(instr, BeforeInstr)
            or isinstance(instr, AndInstr)
            or isinstance(instr, AfterInstr)
        ):
            self.update_objs_poss(instr.instr_a)
            self.update_objs_poss(instr.instr_b)
        else:
            instr.update_objs_poss()

    def _gen_grid(self, width, height):
        # We catch RecursionError to deal with rare cases where
        # rejection sampling gets stuck in an infinite loop
        while True:
            try:
                super()._gen_grid(width, height)

                # Generate the mission
                self.gen_mission()

                # Validate the instructions
                self.validate_instrs(self.instrs_controller.instrs.values())

            except RecursionError as error:
                print("Timeout during mission generation:", error)
                continue

            except RejectSampling as error:
                print("Sampling rejected:", error)
                continue

            break

        
    def validate_instrs(self, instrs):
        """
        Perform some validation on the generated instructions
        """
        # Gather the colors of locked doors
        for instr in instrs: 
            colors_of_locked_doors = []
            if hasattr(self, "unblocking") and self.unblocking:
                for i in range(self.num_cols):
                    for j in range(self.num_rows):
                        room = self.get_room(i, j)
                        for door in room.doors:
                            if door and door.is_locked:
                                colors_of_locked_doors.append(door.color)

            if isinstance(instr, PutNextInstr):
                # Resolve the objects referenced by the instruction
                instr.reset_verifier(self)

                # Check that the objects are not already next to each other
                if set(instr.desc_move.obj_set).intersection(set(instr.desc_fixed.obj_set)):
                    raise RejectSampling(
                        "there are objects that match both lhs and rhs of PutNext"
                    )
                if instr.objs_next():
                    raise RejectSampling("objs already next to each other")

                # Check that we are not asking to move an object next to itself
                move = instr.desc_move
                fixed = instr.desc_fixed
                if len(move.obj_set) == 1 and len(fixed.obj_set) == 1:
                    if move.obj_set[0] is fixed.obj_set[0]:
                        raise RejectSampling("cannot move an object next to itself")

            if isinstance(instr, ActionInstr):
                if not hasattr(self, "unblocking") or not self.unblocking:
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

            if isinstance(instr, SeqInstr):
                self.validate_instrs(instr.instr_a)
                self.validate_instrs(instr.instr_b)
                continue

            if isinstance(instr, GoToGoalInstr):
                continue

            assert False, "unhandled instruction type"
        
    def gen_mission(self):
        """
        Generate a mission (instructions and matching environment)
        Derived level classes should implement this method
        """
        raise NotImplementedError
    
    @property
    def level_name(self):
        return self.__class__.level_name

    @property
    def gym_id(self):
        return self.__class__.gym_id
    
    def open_all_doors(self):
        """
        Open all the doors in the maze
        """

        for i in range(self.num_cols):
            for j in range(self.num_rows):
                room = self.get_room(i, j)
                for door in room.doors:
                    if door:
                        door.is_open = True
    
    def check_objs_reachable(self, agents: list[Agent], raise_exc=True):
        """
        Check that all objects are reachable from the agent's starting
        position without requiring any other object to be moved
        (without unblocking)
        """
        for agent in agents:
            if not self._check_objs_reachable(agent, raise_exc=raise_exc):
                return False
        
        return True

    def _check_objs_reachable(self, agent: Agent, raise_exc=True):
        """
        Check that all objects are reachable from the agent's starting
        position without requiring any other object to be moved
        (without unblocking)
        """

        # Reachable positions
        reachable = set()

        # Work list
        stack = [agent.cur_pos]

        while len(stack) > 0:
            i, j = stack.pop()

            if i < 0 or i >= self.grid.width or j < 0 or j >= self.grid.height:
                continue

            if (i, j) in reachable:
                continue

            # This position is reachable
            reachable.add((i, j))

            cell = self.grid.get(i, j)

            # If there is something other than a door in this cell, it
            # blocks reachability
            if cell and cell.type != "door":
                continue

            # Visit the horizontal and vertical neighbors
            stack.append((i + 1, j))
            stack.append((i - 1, j))
            stack.append((i, j + 1))
            stack.append((i, j - 1))

        # Check that all objects are reachable
        for i in range(self.grid.width):
            for j in range(self.grid.height):
                cell = self.grid.get(i, j)

                if not cell or cell.type == "wall":
                    continue

                if (i, j) not in reachable:
                    if not raise_exc:
                        return False
                    raise RejectSampling("unreachable object at " + str((i, j)))

        # All objects reachable
        return True