from __future__ import annotations

from gymnasium.envs.registration import register
from MA_minigrid.MA_core.MAminigrid import MultiGridEnv
from MA_minigrid import wrappers

def register_SQbabyai_envs():
    register(
        id="SQbabyai-DangerGround-v0",
        entry_point="MA_minigrid.envs.Danger_gound:DangerGroundEnv",
    )

def register_SQbabyai_envs():
    register(
        id="SQbabyai-DangerRoom-v0",
        entry_point="MA_minigrid.envs.Danger_room:DangerRoomEnv",
    )


register_SQbabyai_envs()