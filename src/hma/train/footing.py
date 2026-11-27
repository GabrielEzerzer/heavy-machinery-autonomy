"""How the propping scene reads main's ground and undercarriage: which shapes count as ground, the
bucket, forbidden links and the fronts of the tracks, the rear wheel the baseline pivots about,
where drawn ground allows a prop, the ground's height under the machine, and where the load is
parked."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from hma.machine import BODY_NAMES, HOME_KEY, machine_prefix, tracks
from hma.machine.ground import Box, Ground
from hma.machine.model import (
    ANCHOR,
    LOAD,
    LOAD_TOP,
    PARKED,
    RING,
    ground_geoms,
    rest_pose,
)
from hma.task.checks import LIMITS
from hma.task.scene import DEMO_SCENE

ALLOWED_LINK = "bucket"
PARK = (-1.0, 11.0)
RAY_TOP = 20.0
BUCKET_RADIUS = DEMO_SCENE.bucket_width_m / 2
EVEN = 0.02


def _grid(low: float, high: float) -> np.ndarray:
    return np.linspace(low, high, math.ceil((high - low) / LIMITS.sample_step_m) + 1)


FOOTPRINT = np.array([(x, y) for x in _grid(*tracks.OUTLINE[:2])
                      for y in _grid(*tracks.OUTLINE[2:])])
PRESS = np.array([(x, y) for x in _grid(-BUCKET_RADIUS, BUCKET_RADIUS)
                  for y in _grid(-BUCKET_RADIUS, BUCKET_RADIUS)
                  if math.hypot(x, y) <= BUCKET_RADIUS + 1e-9])


def slope_deg(ground: Ground) -> float:
    return math.degrees(math.atan(math.hypot(*ground.grade)))


def _local(box: Box, points: np.ndarray) -> np.ndarray:
    c, s = math.cos(box.heading), math.sin(box.heading)
    return (np.atleast_2d(points) - [box.x, box.y]) @ np.array([[c, -s], [s, c]])


def _box_meets_outline(box: Box) -> bool:
    x0, x1, y0, y1 = tracks.OUTLINE
    c, s = math.cos(box.heading), math.sin(box.heading)
    half = np.array([[a * box.half_length, b * box.half_width] for a in (-1, 1) for b in (-1, 1)])
    corners = [box.x, box.y] + half @ np.array([[c, s], [-s, c]])
    if (corners[:, 0].max() < x0 or corners[:, 0].min() > x1 or corners[:, 1].max() < y0
            or corners[:, 1].min() > y1):
        return False
    rect = _local(box, [(x, y) for x in (x0, x1) for y in (y0, y1)])
    return not (rect[:, 0].min() > box.half_length or rect[:, 0].max() < -box.half_length
                or rect[:, 1].min() > box.half_width or rect[:, 1].max() < -box.half_width)


def even_stance(ground: Ground) -> bool:
    x0, x1, y0, y1 = tracks.OUTLINE
    for bump in ground.bumps:
        near = (min(max(bump.x, x0), x1), min(max(bump.y, y0), y1))
        if math.hypot(bump.x - near[0], bump.y - near[1]) < bump.radius:
            return False
    return not any(_box_meets_outline(box) for box in ground.boxes)


def flat_press(ground: Ground, x: float, y: float) -> bool:
    if any(math.hypot(b.x - x, b.y - y) < b.radius + BUCKET_RADIUS for b in ground.bumps):
        return False
    inside = 0
    for box in ground.boxes:
        u, v = np.abs(_local(box, [(x, y)])[0])
        if u <= box.half_length - BUCKET_RADIUS and v <= box.half_width - BUCKET_RADIUS:
            inside += 1
        elif math.hypot(max(u - box.half_length, 0.0),
                        max(v - box.half_width, 0.0)) <= BUCKET_RADIUS:
            return False
    return inside <= 1


def _link_shapes(model: mujoco.MjModel, index: int, link: str) -> np.ndarray:
    body = model.body(machine_prefix(index) + link).id
    return np.flatnonzero((model.geom_bodyid == body) & (model.geom_contype & (1 << index) != 0))


def bucket_shapes(model: mujoco.MjModel, index: int) -> np.ndarray:
    return _link_shapes(model, index, ALLOWED_LINK)


def forbidden_shapes(model: mujoco.MjModel, index: int) -> np.ndarray:
    return np.concatenate([_link_shapes(model, index, link) for link in BODY_NAMES
                           if link != ALLOWED_LINK])


def front_parts(model: mujoco.MjModel, index: int) -> list[np.ndarray]:
    prefix = machine_prefix(index)
    out = []
    for side in tracks.SIDES:
        wheels = [tracks.wheel_name(side, k) for k, x in enumerate(tracks.WHEEL_X) if x > 0.0]
        belts = [tracks.belt_name(side, k) for k, x in enumerate(tracks.BELT_X) if x > 0.0]
        names = [*wheels, *belts, tracks.shell_names(side)[2]]
        out.append(np.array([model.geom(prefix + name).id for name in names]))
    return out


def lowest_points(model: mujoco.MjModel, geoms: np.ndarray) -> np.ndarray:
    base = model.body(machine_prefix(0) + "base").id
    points = []
    for g in geoms:
        body = int(model.geom_bodyid[g])
        at = model.body_pos[body] if body != base else model.geom_pos[g]
        points.append(at - [0.0, 0.0, model.geom_size[g, 0]])
    return np.array(points)


def rear_pivot() -> tuple[float, float, float]:
    return tracks.WHEEL_X[0], tracks.WHEEL_RADIUS, tracks.WHEEL_RADIUS


def height_under(model: mujoco.MjModel, index: int, origin: np.ndarray, rot: np.ndarray,
                 points: np.ndarray) -> np.ndarray:
    parked = model.body_pos[model.body(machine_prefix(index) + ANCHOR).id][2] + PARKED / 2
    down = -rot[:, 2].copy()
    mat = np.zeros(9)
    out = []
    for x, y in np.atleast_2d(points):
        start = origin + rot @ [x, y, RAY_TOP]
        nearest = np.inf
        for g in ground_geoms(model, index):
            if model.geom_pos[g, 2] < parked:
                continue
            mujoco.mju_quat2Mat(mat, model.geom_quat[g])
            d = mujoco.mju_rayGeom(model.geom_pos[g], mat, model.geom_size[g], start, down,
                                   int(model.geom_type[g]))
            if d >= 0.0:
                nearest = min(nearest, d)
        out.append(RAY_TOP - nearest)
    return np.array(out)


def park(model: mujoco.MjModel, index: int) -> None:
    prefix = machine_prefix(index)
    key = model.key(HOME_KEY).id
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = model.key_qpos[key]
    load = int(model.joint(prefix + LOAD).qposadr[0])
    ring = int(model.joint(prefix + RING + "_x").qposadr[0])
    pose = rest_pose(model, scratch, index, *PARK, 0.0)
    mujoco.mj_kinematics(model, scratch)
    top = scratch.site_xpos[model.site(prefix + LOAD_TOP).id]
    model.key_qpos[key, load:load + 7] = pose
    model.key_qpos[key, ring:ring + 3] = top - model.body_pos[model.body(prefix + RING).id]
