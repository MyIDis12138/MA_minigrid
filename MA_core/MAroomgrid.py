from __future__ import annotations

import numpy as np

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.mission import MissionSpace
from .MAgrid import MAGrid
from .objects import MABall, MABox, MADoor, MAKey, MAWorldObj, Agent
from .MAminigrid import MultiGridEnv

def reject_next_to(env: MultiGridEnv, pos: tuple[int, int]):
    """
    Function to filter out object positions that are right next to
    the agent's starting point
    """
    for agent in env.agents:
        if agent.cur_pos is None:
            continue
        sx, sy = agent.cur_pos 
        x, y = pos
        d = abs(sx - x) + abs(sy - y)
        if d < 2:
            return True
    return False

class MARoom:
    def __init__(self, top: tuple[int, int], size: tuple[int, int]):
        # Top-left corner and size (tuples)
        self.top = top
        self.size = size

        # List of door objects and door positions
        # Order of the doors is right, down, left, up
        self.doors: list[bool | MADoor | None] = [None] * 4
        self.door_pos: list[tuple[int, int] | None] = [None] * 4

        # List of rooms adjacent to this one
        # Order of the neighbors is right, down, left, up
        self.neighbors: list[MARoom | None] = [None] * 4

        # Indicates if this room is behind a locked door
        self.locked: bool = False

        # List of objects contained
        self.objs: list[MAWorldObj] = []
        self.agents: list[Agent] = []

    def rand_pos(self, env: MultiGridEnv) -> tuple[int, int]:
        topX, topY = self.top
        sizeX, sizeY = self.size
        return env._rand_pos(topX + 1, topX + sizeX - 1, topY + 1, topY + sizeY - 1)

    def pos_inside(self, x: int, y: int) -> bool:
        """
        Check if a position is within the bounds of this room
        """

        topX, topY = self.top
        sizeX, sizeY = self.size

        if x < topX or y < topY:
            return False

        if x >= topX + sizeX or y >= topY + sizeY:
            return False

        return True

# Multi-agent environment with multiple rooms
class MARoomgrid(MultiGridEnv):
    """
    Environment with multiple rooms and random objects.
    This is meant to serve as a base class for other MA environments.
    """
    def __init__(
        self,
        agents_index: list[int] = [],
        room_size: int = 7,
        num_rows: int = 3,
        num_cols: int = 3,
        max_steps: int = 100,
        agent_view_size: int = 7,
        **kwargs,
    ):
        assert room_size > 0
        assert room_size >= 3
        assert num_rows > 0
        assert num_cols > 0
        self.room_size = room_size
        self.num_rows = num_rows
        self.num_cols = num_cols

        height = (room_size - 1) * num_rows + 1
        width = (room_size - 1) * num_cols + 1

        # By default, this environment has no mission
        self.missions = []

        # create the agents
        agents = []
        for i in agents_index:
            agents.append(Agent(i, view_size=agent_view_size))

        self.missions = ""
        
        super().__init__(
            agents=agents,
            width=width,
            height=height,
            max_steps=max_steps,
            see_through_walls=False,
            **kwargs,
        )
    
    def room_from_pos(self, x, y) -> MARoom:
        """Get the room a given position maps to"""

        assert x >= 0
        assert y >= 0

        i = x // (self.room_size-1)
        j = y // (self.room_size-1)

        assert i < self.num_cols
        assert j < self.num_rows

        return self.room_grid[j][i]
    
    def get_room(self, i, j) -> MARoom:
        assert i < self.num_cols
        assert j < self.num_rows
        return self.room_grid[j][i]
    
    def _gen_grid(self, width, height):
        # Create the grid
        self.grid = MAGrid(width, height)

        self.room_grid = []

        # For each row of rooms
        for j in range(0, self.num_rows):
            row = []

            # For each column of rooms
            for i in range(0, self.num_cols):
                room = MARoom(
                    (i * (self.room_size-1), j * (self.room_size-1)),
                    (self.room_size, self.room_size)
                )
                row.append(room)

                # Generate the walls for this room
                self.grid.wall_rect(*room.top, *room.size)

            self.room_grid.append(row)

        # For each row of rooms
        for j in range(0, self.num_rows):
            # For each column of rooms
            for i in range(0, self.num_cols):
                room = self.room_grid[j][i]

                x_l, y_l = (room.top[0] + 1, room.top[1] + 1)
                x_m, y_m = (room.top[0] + room.size[0] - 1, room.top[1] + room.size[1] - 1)

                # Door positions, order is right, down, left, up
                if i < self.num_cols - 1:
                    room.neighbors[0] = self.room_grid[j][i+1]
                    room.door_pos[0] = (x_m, self._rand_int(y_l, y_m))
                if j < self.num_rows - 1:
                    room.neighbors[1] = self.room_grid[j+1][i]
                    room.door_pos[1] = (self._rand_int(x_l, x_m), y_m)
                if i > 0:
                    room.neighbors[2] = self.room_grid[j][i-1]
                    room.door_pos[2] = room.neighbors[2].door_pos[0]
                if j > 0:
                    room.neighbors[3] = self.room_grid[j-1][i]
                    room.door_pos[3] = room.neighbors[3].door_pos[1]

    def place_in_room(
            self, i: int, j: int, obj: MAWorldObj
    ) -> tuple[MAWorldObj, tuple[int, int]]:
        """
        Add an existing object to room (i, j)
        """

        room = self.get_room(i, j)

        pos = self.place_obj(
            obj,
            room.top,
            room.size,
            reject_fn=reject_next_to,
            max_tries=1000
        )

        room.objs.append(obj)

        return obj, pos
    
    def place_agent_in_room(
            self, i: int, j: int, agent: Agent
    ) -> tuple[MAWorldObj, tuple[int, int]]:
        room = self.get_room(i, j)

        pos = self.place_agent(
            agent,
            room.top,
            room.size,
            reject_fn=reject_next_to,
            max_tries=1000
        )

        room.agents.append(agent)

        return agent, pos

    def add_object(        
        self,
        i: int,
        j: int,
        kind: str | None = None,
        color: str | None = None,
    ) -> tuple[MAWorldObj, tuple[int, int]]:
        """
        Add a new object to room (i, j)
        """

        if kind == None:
            kind = self._rand_elem(['key', 'ball', 'box'])

        if color == None:
            color = self._rand_color()

        # TODO: we probably want to add an Object.make helper function
        assert kind in ['key', 'ball', 'box']
        if kind == 'key':
            obj = MAKey(color)
        elif kind == 'ball':
            obj = MABall(color)
        elif kind == 'box':
            obj = MABox(color)
        else:
            raise ValueError(
                f"{kind} object kind is not available in this environment."
            )

        return self.place_in_room(i, j, obj)
    
    def add_door(
        self,
        i: int,
        j: int,
        door_idx: int | None = None,
        color: str | None = None,
        locked: bool | None = None,
    ) -> tuple[MADoor, tuple[int, int]]:
        """
        Add a door to a room, connecting it to a neighbor
        """

        room = self.get_room(i, j)

        if door_idx == None:
            # Need to make sure that there is a neighbor along this wall
            # and that there is not already a door
            while True:
                door_idx = self._rand_int(0, 4)
                if room.neighbors[door_idx] and room.doors[door_idx] is None:
                    break

        if color == None:
            color = self._rand_color()

        if locked is None:
            locked = self._rand_bool()

        assert room.doors[door_idx] is None, "door already exists"

        room.locked = locked
        door = MADoor(color, is_locked=locked)

        pos = room.door_pos[door_idx]
        self.grid.set(*pos, door)
        door.cur_pos = pos

        assert door_idx is not None
        neighbor = room.neighbors[door_idx]
        assert neighbor is not None
        room.doors[door_idx] = door
        neighbor.doors[(door_idx + 2) % 4] = door

        return door, pos
    
    def remove_wall(self, i: int, j: int, wall_idx: int):
        """
        Remove a wall between two rooms
        """

        room = self.get_room(i, j)

        assert wall_idx >= 0 and wall_idx < 4
        assert room.doors[wall_idx] is None, "door exists on this wall"
        assert room.neighbors[wall_idx], "invalid wall"

        neighbor = room.neighbors[wall_idx]

        tx, ty = room.top
        w, h = room.size

        # Ordering of walls is right, down, left, up
        if wall_idx == 0:
            for i in range(1, h - 1):
                self.grid.set(tx + w - 1, ty + i, None)
        elif wall_idx == 1:
            for i in range(1, w - 1):
                self.grid.set(tx + i, ty + h - 1, None)
        elif wall_idx == 2:
            for i in range(1, h - 1):
                self.grid.set(tx, ty + i, None)
        elif wall_idx == 3:
            for i in range(1, w - 1):
                self.grid.set(tx + i, ty, None)
        else:
            assert False, "invalid wall index"

        # Mark the rooms as connected
        room.doors[wall_idx] = True
        assert neighbor is not None
        neighbor.doors[(wall_idx+2) % 4] = True

    def connect_all(
            self, door_colors=COLOR_NAMES, max_itrs=5000, agents: list[Agent] | None = None
    )-> list[MADoor]:
        """
        Make sure that all rooms are reachable by the agent from its
        starting position
        """

        added_doors = []
        
        for agent in agents:
            start_room = self.room_from_pos(*agent.cur_pos)

            def find_reach():
                reach = set()
                stack = [start_room]
                while len(stack) > 0:
                    room = stack.pop()
                    if room in reach:
                        continue
                    reach.add(room)
                    for i in range(0, 4):
                        if room.doors[i]:
                            stack.append(room.neighbors[i])
                return reach

            num_itrs = 0

            while True:
                # This is to handle rare situations where random sampling produces
                # a level that cannot be connected, producing in an infinite loop
                if num_itrs > max_itrs:
                    raise RecursionError('connect_all failed')
                num_itrs += 1

                # If all rooms are reachable, stop
                reach = find_reach()
                if len(reach) == self.num_rows * self.num_cols:
                    break

                # Pick a random room and door position
                i = self._rand_int(0, self.num_cols)
                j = self._rand_int(0, self.num_rows)
                k = self._rand_int(0, 4)
                room = self.get_room(i, j)

                # If there is already a door there, skip
                if not room.door_pos[k] or room.doors[k]:
                    continue

                if room.locked or room.neighbors[k].locked:
                    continue

                color = self._rand_elem(door_colors)
                door, _ = self.add_door(i, j, k, color, False)
                added_doors.append(door)

        return added_doors

    def add_distractors(
        self,
        i: int | None = None,
        j: int | None = None,
        num_distractors: int = 10,
        all_unique: bool = True,
    ) -> list[MAWorldObj]:
        """
        Add random objects that can potentially distract/confuse the agent.
        """

        # Collect a list of existing objects
        objs = []
        for row in self.room_grid:
            for room in row:
                for obj in room.objs:
                    objs.append((obj.type, obj.color))

        # List of distractors added
        dists = []

        while len(dists) < num_distractors:
            color = self._rand_elem(COLOR_NAMES)
            type = self._rand_elem(['key', 'ball', 'box'])
            obj = (type, color)

            if all_unique and obj in objs:
                continue

            # Add the object to a random room if no room specified
            room_i = i
            room_j = j
            if room_i == None:
                room_i = self._rand_int(0, self.num_cols)
            if room_j == None:
                room_j = self._rand_int(0, self.num_rows)

            dist, pos = self.add_object(room_i, room_j, *obj)

            objs.append(obj)
            dists.append(dist)

        return dists
