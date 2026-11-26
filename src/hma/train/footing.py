"""The one seam for the ground and how a machine rests on it: the ground an episode draws, writing
it to the model, which shapes count as ground, which parts may touch it, where the parked load
rests, and the rear track pivot the baseline's rigid geometry turns about."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from hma.machine import BODY_NAMES, ground_axes, machine_prefix, write_ground
from hma.machine.contact import draw_friction, set_friction
from hma.machine.model import (
    LOAD,
    LOAD_TOP,
    RING,
    STEEL_FRICTION,
    TRACK_FEET,
    ground_geoms,
    rest_pose,
)

EDGE_X = 2.0
RANGES = {
    "step": (-0.3, 0.3),
    "grade": (-math.tan(math.radians(8.0)), math.tan(math.radians(8.0))),
    "cross": (-math.tan(math.radians(4.0)), math.tan(math.radians(4.0))),
}
HEIGHT_LIMITS = (-6.0, 2.0)
PARK = (-1.0, 11.0)
ALLOWED_LINK = "bucket"


@dataclass(frozen=True)
class Ground:
    step: float = 0.0
    grade: float = 0.0
    cross: float = 0.0
    friction: float = STEEL_FRICTION[0]
    seed: int | None = None


FLAT = Ground()


def draw_ground(seed: int) -> Ground:
    rng = np.random.default_rng((seed, 3, 0))
    return Ground(float(rng.uniform(*RANGES["step"])), float(rng.uniform(*RANGES["grade"])),
                  float(rng.uniform(*RANGES["cross"])), draw_friction(seed), seed)


def heights(ground: Ground) -> np.ndarray:
    xs, ys = ground_axes()
    ahead = xs[None, :] > EDGE_X
    plane = ground.step + ground.grade * (xs[None, :] - EDGE_X) + ground.cross * ys[:, None]
    return np.clip(np.where(ahead, plane, 0.0), *HEIGHT_LIMITS)


def apply_ground(model: mujoco.MjModel, index: int, ground: Ground) -> None:
    write_ground(model, index, heights(ground))
    set_friction(model, index, ground.friction)


def ground_shapes(model: mujoco.MjModel, index: int) -> np.ndarray:
    return ground_geoms(model, index)


def touches_ground(model: mujoco.MjModel, index: int, link: str) -> np.ndarray:
    body = model.body(machine_prefix(index) + link).id
    return np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype & (1 << index) != 0))


def forbidden_shapes(model: mujoco.MjModel, index: int) -> np.ndarray:
    feet = set(track_feet(model, index).tolist())
    return np.array([g for link in BODY_NAMES if link != ALLOWED_LINK
                     for g in touches_ground(model, index, link) if g not in feet], int)


def surface(ground: Ground, x: float, y: float) -> float:
    if x <= EDGE_X:
        return 0.0
    height = ground.step + ground.grade * (x - EDGE_X) + ground.cross * y
    return float(np.clip(height, *HEIGHT_LIMITS))


def track_feet(model: mujoco.MjModel, index: int) -> np.ndarray:
    prefix = machine_prefix(index)
    return np.array([g for g in range(model.ngeom)
                     if model.geom(g).name.startswith(prefix + TRACK_FEET)], int)


def rear_pivot(model: mujoco.MjModel) -> tuple[float, float, float]:
    feet = track_feet(model, 0)
    rear = feet[np.argmin(model.geom_pos[feet, 0])]
    return (float(model.geom_pos[rear, 0]), float(model.geom_pos[rear, 2]),
            float(model.geom_size[rear, 0]))


def front_feet(model: mujoco.MjModel, index: int) -> np.ndarray:
    feet = track_feet(model, index)
    x = model.geom_pos[feet, 0]
    return feet[np.isclose(x, x.max())]


def front_edge(model: mujoco.MjModel) -> np.ndarray:
    front = front_feet(model, 0)[0]
    return np.array([model.geom_pos[front, 0], 0.0,
                     model.geom_pos[front, 2] - model.geom_size[front, 0]])


class Parking:
    def __init__(self, model: mujoco.MjModel, n: int):
        scratch = mujoco.MjData(model)
        self.load, self.top = [], []
        for i in range(n):
            pose = rest_pose(model, scratch, i, *PARK, 0.0)
            scratch.qpos[model.joint(machine_prefix(i) + LOAD).qposadr[0]:][:7] = pose
            mujoco.mj_kinematics(model, scratch)
            self.load.append(pose.copy())
            self.top.append(scratch.site_xpos[model.site(machine_prefix(i) + LOAD_TOP).id].copy())

    def park(self, model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
        prefix = machine_prefix(index)
        load = model.joint(prefix + LOAD)
        ring = model.joint(prefix + RING + "_x")
        q0, v0 = int(load.qposadr[0]), int(load.dofadr[0])
        r0, w0 = int(ring.qposadr[0]), int(ring.dofadr[0])
        data.qpos[q0:q0 + 7] = self.load[index]
        data.qvel[v0:v0 + 6] = 0.0
        data.qacc_warmstart[v0:v0 + 6] = 0.0
        data.qpos[r0:r0 + 3] = self.top[index] - model.body_pos[model.body(prefix + RING).id]
        data.qvel[w0:w0 + 3] = 0.0
        data.qacc_warmstart[w0:w0 + 3] = 0.0
