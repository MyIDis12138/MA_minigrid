from __future__ import annotations

from MA_core.MAgrid import MAGrid
from MA_core.objects import MAGoal, Agent
from MA_core.MAminigrid import MultiGridEnv
from minigrid.core.mission import MissionSpace


class EmptyEnv(MultiGridEnv):
    """
    ## Description
    This environment is an empty room, and the goal of the agent is to reach the
    green goal square, which provides a sparse reward. A small penalty is
    subtracted for the number of steps to reach the goal. This environment is
    useful, with small rooms, to validate that your RL algorithm works
    correctly, and with large rooms to experiment with sparse rewards and
    exploration. The random variants of the environment have the agent starting
    at a random position for each episode, while the regular variants have the
    agent always starting in the corner opposite to the goal.
    ## Mission Space
    "get to the green goal square"
    ## Action Space
    | Num | Name         | Action       |
    |-----|--------------|--------------|
    | 0   | left         | Turn left    |
    | 1   | right        | Turn right   |
    | 2   | forward      | Move forward |
    | 3   | pickup       | Unused       |
    | 4   | drop         | Unused       |
    | 5   | toggle       | Unused       |
    | 6   | done         | Unused       |
    ## Observation Encoding
    - Each tile is encoded as a 3 dimensional tuple:
        `(OBJECT_IDX, COLOR_IDX, STATE)`
    - `OBJECT_TO_IDX` and `COLOR_TO_IDX` mapping can be found in
        [minigrid/minigrid.py](minigrid/minigrid.py)
    - `STATE` refers to the door state with 0=open, 1=closed and 2=locked
    ## Rewards
    A reward of '1 - 0.9 * (step_count / max_steps)' is given for success, and '0' for failure.
    ## Termination
    The episode ends if any one of the following conditions is met:
    1. The agent reaches the goal.
    2. Timeout (see `max_steps`).
    ## Registered Configurations
    - `MiniGrid-Empty-5x5-v0`
    - `MiniGrid-Empty-Random-5x5-v0`
    - `MiniGrid-Empty-6x6-v0`
    - `MiniGrid-Empty-Random-6x6-v0`
    - `MiniGrid-Empty-8x8-v0`
    - `MiniGrid-Empty-16x16-v0`
    """

    def __init__(
        self,
        agent_indexes: list,
        size=8,
        max_steps: int | None = None,
        **kwargs,
    ):
        agents = [Agent(i) for i in agent_indexes]

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            agents=agents,
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "get to the green goal square"

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = MAGrid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(MAGoal(), width - 2, height - 2)

        for agent in self.agents:
            self.place_agent(agent,top=(1,1))

        self.mission = "get to the green goal square"