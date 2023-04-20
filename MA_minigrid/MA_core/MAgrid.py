from __future__ import annotations

import numpy as np

from typing import Any, Callable

from minigrid.utils.rendering import downsample, highlight_img, fill_coords, point_in_rect

from MA_minigrid.MA_core.MAconstants import OBJECT_TO_IDX, TILE_PIXELS
from MA_minigrid.MA_core.objects import Agent, MAWorldObj, MAWall


# Multi-agent base grid
class MAGrid:
    """
    Represent a grid and operations on it
    """

    # Static cache of pre-renderer tiles
    tile_cache: dict[tuple[Any, ...], Any] = {}

    def __init__(self, width: int, height: int):
        assert width >= 3
        assert height >= 3

        self.width: int = width
        self.height: int = height
        
        self.grid: list[MAWorldObj | None] = [None] * (width * height)
        self.agent_grid: list[Agent | None] = [None] * (width * height)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, MAWorldObj):
            for e in self.grid:
                if e is key:
                    return True
        elif isinstance(key, tuple):
            for e in self.grid:
                if e is None:
                    continue
                if (e.color, e.type) == key:
                    return True
                if key[0] is None and key[1] == e.type:
                    return True
        return False
    
    def __eq__(self, other: MAGrid) -> bool:
        grid1 = self.encode()
        grid2 = other.encode()
        return np.array_equal(grid2, grid1)
    
    def __ne__(self, other: MAGrid) -> bool:
        return not self == other
    
    def copy(self) -> MAGrid:
        from copy import deepcopy

        return deepcopy(self)
    
    def set(self, i: int, j: int, v: MAWorldObj | None):
        assert (
            0 <= i < self.width
        ), f"column index {i} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"
        self.grid[j * self.width + i] = v

    def get(self, i: int, j: int) -> MAWorldObj | None:
        assert 0 <= i < self.width
        assert 0 <= j < self.height
        assert self.grid is not None
        return self.grid[j * self.width + i]
    
    def set_agent(self, i: int, j: int, v: Agent | None):
        assert (
            0 <= i < self.width
        ), f"column index {j} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"
        if v is None:
            self.agent_grid[j * self.width + i] = None
            return
        assert self.get_agent(i, j) is None, f"agent already at ({i}, {j})"
        assert(
            self.get(i,j) is None or self.get(i,j).can_overlap()
        ), f"cannot place agent on top of other object at ({i}, {j})"
        self.agent_grid[j * self.width + i] = v

    def get_agent(self, i: int, j: int) -> Agent | None:
        assert (
            0 <= i < self.width
        ), f"column index {j} outside of grid of width {self.width}"
        assert (
            0 <= j < self.height
        ), f"row index {j} outside of grid of height {self.height}"
        assert self.agent_grid is not None
        return self.agent_grid[j * self.width + i]

    def horz_wall(
        self, 
        x: int, 
        y: int, 
        length: int | None = None, 
        obj_type: Callable[[], MAWorldObj] = MAWall,
    ):
        if length is None:
            length = self.width - x
        for i in range(0, length):
            self.set(x + i, y, obj_type())

    def vert_wall(
        self,
        x: int,
        y: int,
        length: int | None = None,
        obj_type: Callable[[], MAWorldObj] = MAWall,
    ):
        if length is None:
            length = self.height - y
        for j in range(0, length):
            self.set(x, y + j, obj_type())

    def wall_rect(self, x: int, y: int, w: int, h: int):
        self.horz_wall(x, y, w)
        self.horz_wall(x, y + h - 1, w)
        self.vert_wall(x, y, h)
        self.vert_wall(x + w - 1, y, h)

    def rotate_left(self) -> MAGrid:
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = MAGrid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                w = self.get_agent(i, j)
                grid.set(j, grid.height - 1 - i, v)
                grid.set_agent(j, grid.height - 1 - i, w)

        return grid

    def slice(self, topX: int, topY: int, width: int, height: int) -> MAGrid:
        """
        Get a subset of the grid
        """

        grid = MAGrid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if 0 <= x < self.width and 0 <= y < self.height:
                    v = self.get(x, y)
                    w = self.get_agent(x, y)
                else:
                    v = MAWall()
                    w = None

                grid.set(i, j, v)
                grid.set_agent(i, j, w)

        return grid

    @classmethod
    def render_tile(
        cls, 
        obj: MAWorldObj | None = None,
        agent: Agent | None = None,
        highlights: bool = False, 
        tile_size: int = TILE_PIXELS, 
        subdivs: int = 3
    ) -> np.ndarray:
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key: tuple[Any, ...] = (highlights, tile_size)
        key = obj.encode() + key if obj else key
        key = agent.encode() + key if agent else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]
        
        img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3), dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj != None:
            obj.render(img)

        # Overlay the agent on top
        if agent != None:
            agent.render(img)

        # Highlight the cell  if needed
        if highlights:
            highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)
        
        # Cache the rendered tile
        cls.tile_cache[key] = img
        
        return img
    
    def render(
        self,
        tile_size: int,
        highlight_mask: np.ndarray | None = None,
    ):
        """
        Render this grid at a given scale
        :param r: target renderer object
        :param tile_size: tile size in pixels
        """
        
        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)
        
        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)
                agent = self.get_agent(i, j)

                # agent_here = np.array_equal(agent_pos, (i, j))
                tile_img = MAGrid.render_tile(
                    cell,
                    agent=agent,
                    highlights=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    def encode(self, vis_mask: np.ndarray| None = None) -> np.ndarray:
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)
        
        array = np.zeros((self.width, self.height, 9), dtype='uint8')

        for i in range(self.width):
            for j in range(self.height):
                assert vis_mask is not None
                if vis_mask[i, j]:
                    v = self.get(i, j)
                    w = self.get_agent(i, j)

                    if v is None:
                        array[i, j, 0] = OBJECT_TO_IDX['empty']
                        array[i, j, 1] = 0
                        array[i, j, 2] = 0
                    else:
                        array[i, j][0:3] = v.encode()

                    if w is None:
                        array[i, j, 3]= OBJECT_TO_IDX['empty']
                    else:
                        array[i, j][3:] = w.encode()

        return array

    @staticmethod
    def decode(array: np.ndarray) -> tuple[MAGrid, np.ndarray]:
        """
        Decode an array grid encoding back into a grid
        """

        width, height, channels = array.shape
        assert channels == 9

        vis_mask = np.ones(shape=(width, height), dtype=bool)

        grid = MAGrid(width, height)
        for i in range(width):
            for j in range(height):
                embedding = array[i, j]
                v, w = MAWorldObj.decode(embedding)
                grid.set(i, j, v)
                grid.set_agent(i, j, w)
                vis_mask[i, j] = (embedding[0] != OBJECT_TO_IDX['unseen']) 

        return grid, vis_mask
    
    def process_vis(self, agent_pos: tuple[int, int], agent_id: int) -> np.ndarray:
        mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        mask[agent_pos[0], agent_pos[1]] = True

        for j in reversed(range(0, self.height)):
            for i in range(0, self.width - 1):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                agent = self.get_agent(i, j)
                if agent and agent.id != agent_id:
                    continue

                mask[i + 1, j] = True
                if j > 0:
                    mask[i + 1, j - 1] = True
                    mask[i, j - 1] = True

            for i in reversed(range(1, self.width)):
                if not mask[i, j]:
                    continue

                cell = self.get(i, j)
                if cell and not cell.see_behind():
                    continue

                agent = self.get_agent(i, j)
                if agent and agent.id != agent_id:
                    continue

                mask[i - 1, j] = True
                if j > 0:
                    mask[i - 1, j - 1] = True
                    mask[i, j - 1] = True

        for j in range(0, self.height):
            for i in range(0, self.width):
                if not mask[i, j]:
                    self.set(i, j, None)
                    self.set_agent(i, j, None)

        return mask
