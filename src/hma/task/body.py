# The machine-body seam: lift ratings, plan size, track footprint, travel reach, boom-arm angles.
# Ratings are the manufacturer's chart (hma.machine.statics.CHART plus its at-max-reach column)
# and the model statics at the same cells, from hma.task.capacity_table. Frame as in kinematics.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from hma.machine import JOINT_NAMES, inverse_kinematics, load_scene, machine_prefix, statics
from hma.machine.tracks import OUTLINE
from hma.task.capacity_table import TABLE

PITCHES_DEG = tuple(range(-180, 180, 15))
REACH_SCAN_M = (12.0, 0.05)
BUCKET_KG = 400.0
MAX_REACH = {
    4.5: (6.32, "*2310", "1920"), 3.0: (6.86, "*2300", "1640"), 1.5: (7.04, "2250", "1530"),
    0.0: (6.89, "2290", "1550"), -1.5: (6.39, "2540", "1710"), -3.0: (5.43, "3240", "2170"),
}


@dataclass(frozen=True)
class Capacity:
    tipping_kg: float
    hydraulic_kg: float


@dataclass(frozen=True)
class Cell:
    height_m: float
    radius_m: float
    side_kg: float


class Body(Protocol):
    def lift_capacity(self, radius_m: float, height_m: float) -> Capacity | None: ...

    def plan_radius(self) -> float: ...

    def footprint(self) -> tuple[float, float, float, float]: ...

    def travel_reach(self, height_m: float) -> float: ...

    def boom_arm_angles(self, x: float, y: float, z: float) -> tuple[float, ...]: ...


def kg(rating: str) -> float:
    return float(rating.lstrip("*"))


def side_rows() -> dict[float, list[Cell]]:
    rows: dict[float, list[Cell]] = {}
    for (height, radius), (_, side) in statics.CHART.items():
        rows.setdefault(height, []).append(Cell(height, radius, kg(side)))
    for height, (radius, _, side) in MAX_REACH.items():
        rows.setdefault(height, []).append(Cell(height, radius, kg(side)))
    return {h: sorted(cells, key=lambda c: c.radius_m) for h, cells in rows.items()}


SIDE_ROWS = side_rows()


def rating_cells(radius_m: float, rows: dict[float, list[Cell]] = SIDE_ROWS) -> list[Cell] | None:
    found = []
    for cells in rows.values():
        if radius_m < cells[0].radius_m:
            return None
        if radius_m > cells[-1].radius_m:
            continue
        hi = next(i for i, c in enumerate(cells) if c.radius_m >= radius_m)
        found += [cells[hi]] if cells[hi].radius_m == radius_m else [cells[hi - 1], cells[hi]]
    return found or None


def model_capacity(radius_m: float, height_m: float) -> Capacity | None:
    found = TABLE[(height_m, radius_m)]
    return None if found is None else Capacity(*found)


def _poses(x: float, y: float, z: float) -> list[np.ndarray]:
    point = np.array([x, y, z], float)
    found = (inverse_kinematics(point, math.radians(d)) for d in PITCHES_DEG)
    return [q for q in found if q is not None]


@dataclass
class MachineBody:
    _model: mujoco.MjModel | None = field(default=None, repr=False)
    _data: mujoco.MjData | None = field(default=None, repr=False)
    _radius: float | None = field(default=None, repr=False)
    _reach: dict[float, float] = field(default_factory=dict, repr=False)

    def _load(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        if self._model is None:
            self._model = load_scene(1, weld=True)
            self._data = mujoco.MjData(self._model)
        return self._model, self._data

    def _home(self) -> tuple[mujoco.MjModel, mujoco.MjData, str]:
        model, data = self._load()
        prefix = machine_prefix(0)
        data.qpos[:] = model.qpos0
        for name in JOINT_NAMES:
            data.qpos[model.joint(prefix + name).qposadr[0]] = 0.0
        mujoco.mj_kinematics(model, data)
        return model, data, prefix

    def _corners(self, g: int) -> list[np.ndarray]:
        model, data = self._load()
        rot = data.geom_xmat[g].reshape(3, 3)
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh = model.geom_dataid[g]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            return list(data.geom_xpos[g] + model.mesh_vert[start:start + count] @ rot.T)
        centre, half = model.geom_aabb[g][:3], model.geom_aabb[g][3:]
        signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
        return [data.geom_xpos[g] + rot @ (centre + s * half) for s in signs]

    def lift_capacity(self, radius_m: float, height_m: float) -> Capacity | None:
        return model_capacity(radius_m, height_m)

    def footprint(self) -> tuple[float, float, float, float]:
        return OUTLINE

    def travel_reach(self, height_m: float) -> float:
        if height_m not in self._reach:
            far, step = REACH_SCAN_M
            x = far
            while x > 0 and not _poses(x, 0.0, height_m):
                x -= step
            self._reach[height_m] = max(x, 0.0)
        return self._reach[height_m]

    def plan_radius(self) -> float:
        if self._radius is None:
            model, data, prefix = self._home()
            bodies = {model.body(prefix + "base").id, model.body(prefix + "upper").id}
            radius = 0.0
            for g in range(model.ngeom):
                if model.geom_bodyid[g] in bodies:
                    radius = max(radius, *(math.hypot(c[0], c[1]) for c in self._corners(g)))
            self._radius = radius
        return self._radius

    def boom_arm_angles(self, x: float, y: float, z: float) -> tuple[float, ...]:
        return tuple(180.0 - math.degrees(q[2]) for q in _poses(x, y, z))
