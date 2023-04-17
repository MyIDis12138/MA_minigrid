from __future__ import annotations

import numpy as np
from typing import Any, Tuple

from minigrid.core.grid import Grid, downsample, highlight_img, fill_coords, point_in_rect
from minigrid.core.constants import OBJECT_TO_IDX, TILE_PIXELS

from MA_minigrid.MA_core.objects import Agent, MAWorldObj, MAWall

Point = Tuple[int, int]

# Multi-agent base grid
class MAGrid(Grid):
    # Grid for multi-agent environments
    def __init__(self, width, height):
        assert width >= 3
        assert height >= 3

        self.width = width
        self.height = height
        
        self.agent_grid = [None] * width * height
        self.grid = [None] * width * height
    
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

    @classmethod
    def render_tile(
        cls, 
        obj: MAWorldObj | None = None,
        agent: Agent | None = None,
        highlights: bool = False, 
        tile_size: int=TILE_PIXELS, 
        subdivs: int=3
    ):
        """
        Render a tile and cache the result
        """
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

    def encode(self, vis_mask=None):
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
                    if v is None:
                        array[0] = OBJECT_TO_IDX['empty']
                    else:
                        array[..., 0:3] = v.encode()

                    w = self.get_agent(i, j)
                    if w is None:
                        array[3]= OBJECT_TO_IDX['empty']
                    else:
                        array[..., 3:] = w.encode()

        return array

    
    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = MAGrid(width, height)

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                        y >= 0 and y < self.height:
                    v = self.get(x, y)
                    w = self.get_agent(x, y)
                else:
                    v = MAWall()
                    w = None

                grid.set(i, j, v)
                grid.set_agent(i, j, w)

        return grid
    
    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = MAGrid(self.height, self.width)

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)
                w = self.get_agent(i, j)
                grid.set_agent(j, grid.height - 1 - i, w)
        return grid
    
    def horz_wall(self, x, y, length=None, obj_type=MAWall):
        return super().horz_wall(x, y, length, obj_type)

    def vert_wall(self, x, y, length=None, obj_type=MAWall):
        return super().vert_wall(x, y, length, obj_type)

    @staticmethod
    def decode(array: np.ndarray):
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
                v = MAWorldObj.decode(*embedding[:3])
                grid.set(i, j, v)
                grid.set_agent(i, j, None)
                if embedding[3] == OBJECT_TO_IDX['agent']:
                    w = Agent.decode(*embedding[4:])
                    grid.set_agent(i, j, w)
                vis_mask[i, j] = (embedding[0] != OBJECT_TO_IDX['unseen']) 

        return grid, vis_mask

    def process_vis(self, agent_pos: tuple[int, int]) -> np.ndarray:
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
                if agent and not agent.see_behind():
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
                if agent and not agent.see_behind():
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