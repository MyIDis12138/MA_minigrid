# Enumerates the different reward types
from __future__ import annotations

from enum import IntEnum

class RewardType(IntEnum):
    GLOBAL = 0
    INDIVIDUAL = 1
    TWO_STAGE = 2
