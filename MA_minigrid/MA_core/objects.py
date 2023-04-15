from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import math
import numpy as np
from minigrid.core.world_object import *

from minigrid.core.constants import (
    COLOR_TO_IDX,
    COLORS,
    IDX_TO_COLOR,
    IDX_TO_OBJECT,
    OBJECT_TO_IDX,
    DIR_TO_VEC
)
from minigrid.utils.rendering import (
    fill_coords,
    point_in_circle,
    point_in_line,
    point_in_rect,
    point_in_triangle,
    rotate_fn
)

if TYPE_CHECKING:
    from MA_minigrid.MA_core.MAminigrid import MultiGridEnv


class DoorWID(Door):
    def __init__(self, color, id, is_open=False, is_locked=False):
        super(DoorWID, self).__init__(color, is_open, is_locked)
        self.id = id

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, KeyWID) and env.carrying.id == self.id:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

class KeyWID(Key):
    def __init__(self, id, color='blue'):
        super(KeyWID, self).__init__(color)
        self.id = id

    def can_overlap(self):
        return True

class BoxWID(Box):
    def __init__(self, color, id, contains=None, locked=True):
        super(BoxWID, self).__init__(color)
        self.contains = contains
        self.is_locked = locked
        self.id = id

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        if self.is_locked:
            if isinstance(env.carrying, KeyWID) and env.carrying.id == self.id:
                self.is_locked = False
                self.is_open = True
                return True
            return False
        else:
            # Replace the box by its contents
            env.grid.set(*pos, self.contains)
        return True
    
class MAWorldObj(WorldObj):
    def __init__(self, type, color):
        super(MAWorldObj, self).__init__(type, color)
        self.id = None
        self.agent_on = None

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    def toggle(self, env: MultiGridEnv, pos: tuple[int, int]) -> bool:
        """Method to trigger/toggle an action this object performs"""
        return False

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 9-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = MAWall(color)
        elif obj_type == 'floor':
            v = MAFloor(color)
        elif obj_type == 'ball':
            v = MABall(color)
        elif obj_type == 'key':
            v = MAKey(color)
        elif obj_type == 'box':
            v = MABox(color)
        elif obj_type == 'door':
            v = MADoor(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = MAGoal()
        elif obj_type == 'lava':
            v = MALava(color)
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v


class Agent(MAWorldObj):
    def __init__(
            self, 
            id = None,
            index = 0,
            direction = 0, 
            view_size = 7
        ):
        assert index >= 0 and index < 6, "index must be between 0 and 5"
        super(Agent, self).__init__('agent', IDX_TO_COLOR[index])
        # Number of cells (width and height) in the agent view
        assert view_size % 2 == 1
        assert view_size >= 3
        self.view_size = view_size
        assert direction in [0,1,2,3], "direction must be between 0 and 3"
        self.dir = direction
        self.id = id or index
        self.index = index
        self.carrying: MAWorldObj | None = None
        self.terminated = False
        self.started = True
        self.paused = False
        assert id is not None, "id must be specified"

    def render(self, img):
        c = COLORS[self.color]
        tri_fn = point_in_triangle(
            (0.12, 0.19),
            (0.87, 0.50),
            (0.12, 0.81),
        )
        # Rotate the agent based on its direction
        tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
        fill_coords(img, tri_fn, c)

    def encode(self):
        # Encode the agent as a 5-tuple of integers
        if self.carrying:
            return (OBJECT_TO_IDX[self.type], self.index, self.dir, 1, OBJECT_TO_IDX[self.carrying.type], COLOR_TO_IDX[self.carrying.color])
        return ( OBJECT_TO_IDX[self.type], self.index, self.dir, 0, 0, 0)
    
    @staticmethod
    def decode(index, direction, carrying, carrying_type, carrying_color):
        v = Agent(index, direction)
        v.carrying = MAWorldObj.decode(carrying_type, carrying_color, 0) if carrying else None
        return v
    
    def reset(self):
        self.terminated = False
        self.started = True
        self.paused = False
        self.dir = None
        self.cur_pos = None
        self.init_pos = None
        self.carrying = None

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir >= 0 and self.dir < 4
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.cur_pos + self.dir_vec

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.cur_pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self, view_size=None):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """
        
        view_size = view_size or self.view_size

        # Facing right
        if self.dir == 0:
            topX = self.cur_pos[0]
            topY = self.cur_pos[1] - view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.cur_pos[0] - view_size // 2
            topY = self.cur_pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.cur_pos[0] - view_size + 1
            topY = self.cur_pos[1] - view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.cur_pos[0] - view_size // 2
            topY = self.cur_pos[1] - view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + view_size
        botY = topY + view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None
    
class MAGoal(MAWorldObj):
    def __init__(self, color='green'):
        super().__init__('goal', color=color)

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class MAFloor(MAWorldObj):
    """
    Colored floor tile the agent can walk over
    """

    def __init__(self, color='blue'):
        super().__init__('floor', color)

    def can_overlap(self):
        return True

    def render(self, r):
        # Give the floor a pale color
        c = COLORS[self.color]
        r.setLineColor(100, 100, 100, 0)
        r.setColor(*c/2)
        r.drawPolygon([
            (1          , TILE_PIXELS),
            (TILE_PIXELS, TILE_PIXELS),
            (TILE_PIXELS,           1),
            (1          ,           1)
        ])

class MALava(MAWorldObj):
    def __init__(self, color='red'):
        super().__init__('lava', color=color)

    def can_overlap(self):
        return True

    def render(self, img):
        if self.color == 'yellow':
            c = (255, 128, 0)
        elif self.color == 'blue':
            c = (0, 0, 255)
        else:
            raise NotImplementedError

        # Background color
        fill_coords(img, point_in_rect(0, 1, 0, 1), c)

        # Little waves
        for i in range(3):
            ylo = 0.3 + 0.2 * i
            yhi = 0.4 + 0.2 * i
            fill_coords(img, point_in_line(0.1, ylo, 0.3, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.3, yhi, 0.5, ylo, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.5, ylo, 0.7, yhi, r=0.03), (0,0,0))
            fill_coords(img, point_in_line(0.7, yhi, 0.9, ylo, r=0.03), (0,0,0))

class MAWall(MAWorldObj):
    def __init__(self, color='grey'):
        super().__init__('wall', color)

    def see_behind(self):
        return False

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

class MADoor(MAWorldObj):
    def __init__(self, color, is_open=False, is_locked=False):
        super().__init__('door', color)
        self.is_open = is_open
        self.is_locked = is_locked

    def can_overlap(self):
        """The agent can only walk over this cell when the door is open"""
        return self.is_open

    def see_behind(self):
        return self.is_open

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, Key) and env.carrying.color == self.color:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True

    def encode(self):
        """Encode the a description of this object as a 5-tuple of integers"""

        # State, 0: open, 1: closed, 2: locked
        if self.is_open:
            state = 0
        elif self.is_locked:
            state = 2
        elif not self.is_open:
            state = 1

        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], state, self.agent_id, self.agent_dir)

    def render(self, img):
        c = COLORS[self.color]

        if self.is_open:
            fill_coords(img, point_in_rect(0.88, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.92, 0.96, 0.04, 0.96), (0,0,0))
            return

        # Door frame and door
        if self.is_locked:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.06, 0.94, 0.06, 0.94), 0.45 * np.array(c))

            # Draw key slot
            fill_coords(img, point_in_rect(0.52, 0.75, 0.50, 0.56), c)
        else:
            fill_coords(img, point_in_rect(0.00, 1.00, 0.00, 1.00), c)
            fill_coords(img, point_in_rect(0.04, 0.96, 0.04, 0.96), (0,0,0))
            fill_coords(img, point_in_rect(0.08, 0.92, 0.08, 0.92), c)
            fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), (0,0,0))

            # Draw door handle
            fill_coords(img, point_in_circle(cx=0.75, cy=0.50, r=0.08), c)

class MAKey(MAWorldObj):
    def __init__(self, color='blue'):
        super(Key, self).__init__('key', color)

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Vertical quad
        fill_coords(img, point_in_rect(0.50, 0.63, 0.31, 0.88), c)

        # Teeth
        fill_coords(img, point_in_rect(0.38, 0.50, 0.59, 0.66), c)
        fill_coords(img, point_in_rect(0.38, 0.50, 0.81, 0.88), c)

        # Ring
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.190), c)
        fill_coords(img, point_in_circle(cx=0.56, cy=0.28, r=0.064), (0,0,0))

class MABall(MAWorldObj):
    def __init__(self, color='blue'):
        super(Ball, self).__init__('ball', color)

    def can_pickup(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_circle(0.5, 0.5, 0.31), COLORS[self.color])

class MABox(MAWorldObj):
    def __init__(self, color, contains=None):
        super(Box, self).__init__('box', color)
        self.contains = contains

    def can_pickup(self):
        return True

    def render(self, img):
        c = COLORS[self.color]

        # Outline
        fill_coords(img, point_in_rect(0.12, 0.88, 0.12, 0.88), c)
        fill_coords(img, point_in_rect(0.18, 0.82, 0.18, 0.82), (0,0,0))

        # Horizontal slit
        fill_coords(img, point_in_rect(0.16, 0.84, 0.47, 0.53), c)

    def toggle(self, env, pos):
        # Replace the box by its contents
        env.grid.set(*pos, self.contains)
        return True

class MADoorWID(MADoor):
    def __init__(self, color, is_open=False, is_locked=False):
        super(MADoorWID, self).__init__(color, is_open, is_locked)

    def toggle(self, env, pos):
        # If the player has the right key to open the door
        if self.is_locked:
            if isinstance(env.carrying, MAKeyWID) and env.carrying.id == self.id:
                self.is_locked = False
                self.is_open = True
                return True
            return False

        self.is_open = not self.is_open
        return True
    
class MAKeyWID(MAKey):
    def __init__(self, id, color='blue'):
        super(MAKeyWID, self).__init__(color)
        self.id = id

    def can_overlap(self):
        return True

class MABoxWID(MABox):
    def __init__(self, color, id, contains=None, locked=True):
        super(MABoxWID, self).__init__(color)
        self.contains = contains
        self.is_locked = locked
        self.id = id

    def can_pickup(self):
        return True

    def toggle(self, env, pos):
        if self.is_locked:
            if isinstance(env.carrying, MAKeyWID) and env.carrying.id == self.id:
                self.is_locked = False
                self.is_open = True
                return True
            return False
        else:
            # Replace the box by its contents
            env.grid.set(*pos, self.contains)
        return True
    

class Oracle(Ball):
    def __init__(self, color):
        super().__init__(color)

    def can_overlap(self):
        return True
    