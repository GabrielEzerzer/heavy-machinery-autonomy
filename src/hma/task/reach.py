# The reach seam: can the bucket tip get to a point inside joint limits?
# MachineArm asks the machine model's inverse kinematics at one bucket pitch after another.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from hma.machine import inverse_kinematics

COARSE_PITCH_DEG = 15
PITCHES_DEG = tuple(sorted(range(-180, 180), key=lambda d: d % COARSE_PITCH_DEG != 0))


class Reach(Protocol):
    def reachable(self, x: float, y: float, z: float) -> bool: ...


@dataclass(frozen=True)
class MachineArm:
    def reachable(self, x: float, y: float, z: float) -> bool:
        point = np.array([x, y, z], float)
        return any(inverse_kinematics(point, math.radians(d)) is not None for d in PITCHES_DEG)
