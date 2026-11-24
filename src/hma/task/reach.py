# The reach seam: can the bucket tip get to a point inside joint limits?
# StandInArm is a planar three-link stand-in until the machine model supplies inverse kinematics.
# Joint signs follow the open 12 t excavator description: positive angles fold the arm down.
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class Reach(Protocol):
    def reachable(self, x: float, y: float, z: float) -> bool: ...


@dataclass(frozen=True)
class StandInArm:
    boom_m: float = 4.60
    arm_m: float = 2.52
    bucket_m: float = 1.23
    boom_pivot_estimate: tuple[float, float] = (0.14, 1.424)
    boom_deg: tuple[float, float] = (-70.0, 44.0)
    arm_deg: tuple[float, float] = (30.0, 152.0)
    bucket_deg: tuple[float, float] = (-33.0, 143.0)
    step_deg: float = 1.0

    def reachable(self, x: float, y: float, z: float) -> bool:
        px, pz = self.boom_pivot_estimate
        dx, dz = math.hypot(x, y) - px, z - pz
        dist = math.hypot(dx, dz)

        bucket = np.radians(np.arange(self.bucket_deg[0], self.bucket_deg[1] + 1e-9, self.step_deg))
        outer = self.arm_m + self.bucket_m * np.exp(-1j * bucket)
        outer_len, outer_dir = np.abs(outer), np.angle(outer)

        cos_bend = (dist**2 - self.boom_m**2 - outer_len**2) / (2 * self.boom_m * outer_len)
        ok = np.abs(cos_bend) <= 1.0
        if not ok.any():
            return False
        bend = np.arccos(np.clip(cos_bend[ok], -1.0, 1.0))
        outer_len, outer_dir = outer_len[ok], outer_dir[ok]

        for b in (bend, -bend):
            elevation = math.atan2(dz, dx) - np.arctan2(
                outer_len * np.sin(b), self.boom_m + outer_len * np.cos(b)
            )
            boom = np.degrees(-elevation)
            arm = np.degrees(outer_dir - b)
            arm = (arm + 180.0) % 360.0 - 180.0
            inside = (
                (boom >= self.boom_deg[0])
                & (boom <= self.boom_deg[1])
                & (arm >= self.arm_deg[0])
                & (arm <= self.arm_deg[1])
            )
            if inside.any():
                return True
        return False
