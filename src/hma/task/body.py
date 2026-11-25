# The machine-body seam: rated lift capacity, plan size and boom-arm angles.
# The rated chart is the manufacturer's for this machine; lift_capacity is where the model's own
# statics plug in. Machine frame as in hma.machine.kinematics, heights from the ground under it.
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import Protocol

import mujoco
import numpy as np

from hma.machine import JOINT_NAMES, inverse_kinematics, load_scene, machine_prefix

PITCHES_DEG = tuple(range(-180, 180, 15))
BUCKET_KG = 400.0
SIDE_CHART = {
    4.5: ((3.0, 3350.0), (4.5, 2110.0), (6.32, 1920.0)),
    3.0: ((3.0, 5460.0), (4.5, 3200.0), (6.0, 2050.0), (6.86, 1640.0)),
    1.5: ((3.0, 5380.0), (4.5, 2980.0), (6.0, 1960.0), (7.04, 1530.0)),
    0.0: ((3.0, 5070.0), (4.5, 2810.0), (6.0, 1880.0), (6.89, 1550.0)),
    -1.5: ((1.5, 4630.0), (3.0, 5030.0), (4.5, 2740.0), (6.0, 1850.0), (6.39, 1710.0)),
    -3.0: ((1.5, 8540.0), (3.0, 5120.0), (4.5, 2780.0), (5.43, 2170.0)),
}


@dataclass(frozen=True)
class Capacity:
    tipping_kg: float
    hydraulic_kg: float


class Body(Protocol):
    def lift_capacity(self, radius_m: float, height_m: float) -> Capacity | None: ...

    def plan_radius(self) -> float: ...

    def boom_arm_angles(self, x: float, y: float, z: float) -> tuple[float, ...]: ...


def chart_rating(radius_m: float, chart: dict = SIDE_CHART) -> float | None:
    found = []
    for row in chart.values():
        radii = [r for r, _ in row]
        if radius_m < radii[0]:
            return None
        if radius_m > radii[-1]:
            continue
        hi = bisect.bisect_left(radii, radius_m)
        lo = hi if radii[hi] == radius_m else hi - 1
        found.append(min(row[lo][1], row[hi][1]))
    return min(found) if found else None


def _poses(x: float, y: float, z: float) -> list[np.ndarray]:
    point = np.array([x, y, z], float)
    found = (inverse_kinematics(point, math.radians(d)) for d in PITCHES_DEG)
    return [q for q in found if q is not None]


@dataclass
class MachineBody:
    _model: mujoco.MjModel | None = field(default=None, repr=False)
    _data: mujoco.MjData | None = field(default=None, repr=False)
    _radius: float | None = field(default=None, repr=False)

    def _load(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        if self._model is None:
            self._model = load_scene(1, weld=True)
            self._data = mujoco.MjData(self._model)
        return self._model, self._data

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
        return None

    def plan_radius(self) -> float:
        if self._radius is not None:
            return self._radius
        model, data = self._load()
        prefix = machine_prefix(0)
        data.qpos[:] = model.qpos0
        for name in JOINT_NAMES:
            data.qpos[model.joint(prefix + name).qposadr[0]] = 0.0
        mujoco.mj_kinematics(model, data)
        bodies = {model.body(prefix + "base").id, model.body(prefix + "upper").id}
        radius = 0.0
        for g in range(model.ngeom):
            if model.geom_bodyid[g] in bodies:
                radius = max(radius, *(math.hypot(c[0], c[1]) for c in self._corners(g)))
        self._radius = radius
        return radius

    def boom_arm_angles(self, x: float, y: float, z: float) -> tuple[float, ...]:
        return tuple(180.0 - math.degrees(q[2]) for q in _poses(x, y, z))
