"""Rigid ground for one machine from MuJoCo primitives: a plane, tilted for a slope, with a fixed
pool of boxes (steps, kerbs, rut banks) and buried spheres (bumps) placed at reset. settle() sets a
machine down and steps it to rest; stand() redraws a seed's ground until the machine can stand."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from hma.machine import tracks
from hma.machine.model import (
    ANCHOR,
    BASE_JOINT,
    GROUND_BOXES,
    GROUND_BUMPS,
    GROUND_X,
    GROUND_Y,
    PARKED,
    WELD,
    ground_geoms,
    machine_prefix,
    write_friction,
)
from hma.machine.scenery import REACH as SCENERY_REACH
from hma.machine.scenery import SHOULDER, SUN_DIR

RANGES = {
    "slope": (0.0, math.radians(20.0)),
    "steps": (0, 2),
    "step_height": (0.05, 0.40),
    "kerbs": (0, 2),
    "kerb_height": (0.10, 0.20),
    "kerb_width": (0.2, 0.4),
    "ruts": (0, 2),
    "rut_depth": (0.05, 0.15),
    "rut_width": (0.3, 0.6),
    "bumps": (0, 6),
    "bump_height": (0.03, 0.15),
    "bump_radius": (0.3, 1.5),
    "feature_reach": (-4.0, 4.0),
    "friction": tracks.FRICTION_RANGE,
}
REACH = 6.0
BANK = 2.0
FIT_TRIES = 40
FIT_SHRINK = 0.9
MIN_WIDTH = 1.5
SHADOW_REACH = SCENERY_REACH
BURY = 0.5
DROP = 0.01
SETTLE_MIN = 0.5
SETTLE_MAX = 6.0
REST_SPEED = 1e-3
REST_TIME = 0.25
DRAWS = 20


@dataclass(frozen=True)
class Box:
    x: float
    y: float
    heading: float
    half_length: float
    half_width: float
    height: float


@dataclass(frozen=True)
class Bump:
    x: float
    y: float
    radius: float
    height: float


@dataclass(frozen=True)
class Ground:
    grade: tuple[float, float] = (0.0, 0.0)
    friction: float = tracks.FRICTION
    boxes: tuple[Box, ...] = ()
    bumps: tuple[Bump, ...] = ()
    seed: int | None = None


def inside(points: np.ndarray) -> bool:
    points = np.atleast_2d(points)
    sun = np.array(SUN_DIR) / np.linalg.norm(SUN_DIR)
    rel = points - SHOULDER
    across = np.linalg.norm(rel - np.outer(rel @ sun, sun), axis=1)
    in_field = ((GROUND_X[0] <= points[:, 0]) & (points[:, 0] <= GROUND_X[1])
                & (GROUND_Y[0] <= points[:, 1]) & (points[:, 1] <= GROUND_Y[1]))
    return bool(np.all(in_field & (across <= SHADOW_REACH)))


def box_corners(box: Box) -> np.ndarray:
    n = np.array([math.cos(box.heading), math.sin(box.heading)])
    t = np.array([-n[1], n[0]])
    return np.array([[*(np.array([box.x, box.y]) + a * box.half_length * n
                       + b * box.half_width * t), box.height]
                     for a in (-1, 1) for b in (-1, 1)])


def _fit(rng: np.random.Generator, length: float, width: float, height: float,
         offsets: tuple[float, ...], grows: bool) -> list[Box]:
    heading = float(rng.uniform(0.0, 2 * math.pi))
    n = np.array([math.cos(heading), math.sin(heading)])
    at = float(rng.uniform(*RANGES["feature_reach"]))
    for _ in range(FIT_TRIES):
        boxes = []
        for offset in offsets:
            middle = at + offset + (length if grows else 0.0)
            x, y = middle * n
            boxes.append(Box(float(x), float(y), heading, length, width, height))
        if all(inside(box_corners(b)) for b in boxes):
            return boxes
        width *= FIT_SHRINK
        length *= FIT_SHRINK if grows else 1.0
        if width < MIN_WIDTH:
            break
    return []


def draw_ground(seed: int, attempt: int = 0) -> Ground:
    if seed < 0 or attempt < 0:
        raise ValueError("seed and attempt must be non-negative integers")
    rng = np.random.default_rng((seed, 3, attempt))
    slope = rng.uniform(*RANGES["slope"])
    direction = rng.uniform(0.0, 2 * math.pi)
    boxes: list[Box] = []
    for _ in range(rng.integers(RANGES["steps"][0], RANGES["steps"][1] + 1)):
        boxes += _fit(rng, REACH, REACH, rng.uniform(*RANGES["step_height"]), (0.0,), True)
    for _ in range(rng.integers(RANGES["kerbs"][0], RANGES["kerbs"][1] + 1)):
        boxes += _fit(rng, rng.uniform(*RANGES["kerb_width"]) / 2, REACH,
                      rng.uniform(*RANGES["kerb_height"]), (0.0,), False)
    for _ in range(rng.integers(RANGES["ruts"][0], RANGES["ruts"][1] + 1)):
        half = rng.uniform(*RANGES["rut_width"]) / 2 + BANK / 2
        boxes += _fit(rng, BANK / 2, REACH, rng.uniform(*RANGES["rut_depth"]), (-half, half),
                      False)
    bumps = []
    for _ in range(rng.integers(RANGES["bumps"][0], RANGES["bumps"][1] + 1)):
        radius = float(rng.uniform(*RANGES["bump_radius"]))
        x, y = (float(v) for v in rng.uniform(*RANGES["feature_reach"], size=2))
        height = float(rng.uniform(*RANGES["bump_height"]))
        rim = [[x + radius * c, y + radius * s, height] for c, s in ((1, 0), (-1, 0), (0, 1),
                                                                      (0, -1))]
        if inside(np.array(rim)):
            bumps.append(Bump(x, y, radius, height))
    grade = (math.tan(slope) * math.cos(direction), math.tan(slope) * math.sin(direction))
    return Ground(grade, float(rng.uniform(*RANGES["friction"])), tuple(boxes[:GROUND_BOXES]),
                  tuple(bumps[:GROUND_BUMPS]), seed)


def _offset(model: mujoco.MjModel, index: int) -> np.ndarray:
    return model.body_pos[model.body(machine_prefix(index) + ANCHOR).id].copy()


def _tilt(grade: tuple[float, float]) -> np.ndarray:
    normal = np.array([-grade[0], -grade[1], 1.0])
    normal /= np.linalg.norm(normal)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    axis = np.cross([0.0, 0.0, 1.0], normal)
    if np.linalg.norm(axis) > 0:
        mujoco.mju_axisAngle2Quat(quat, axis / np.linalg.norm(axis), math.acos(normal[2]))
    return quat


def _turned(tilt: np.ndarray, heading: float) -> np.ndarray:
    yaw = np.array([math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)])
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, tilt, yaw)
    return quat


def apply_ground(model: mujoco.MjModel, index: int, ground: Ground) -> None:
    if len(ground.boxes) > GROUND_BOXES or len(ground.bumps) > GROUND_BUMPS:
        raise ValueError(f"a ground holds at most {GROUND_BOXES} boxes and {GROUND_BUMPS} bumps")
    if not all(math.isfinite(g) for g in ground.grade):
        raise ValueError("the grade must be finite")
    offset = _offset(model, index)
    gx, gy = ground.grade
    normal = np.array([-gx, -gy, 1.0]) / math.sqrt(1.0 + gx * gx + gy * gy)
    tilt = _tilt(ground.grade)
    plane, boxes, bumps = ground_geoms(model, index)

    def on_plane(x: float, y: float, lift: float) -> np.ndarray:
        return offset + [x, y, gx * x + gy * y] + lift * normal

    centre = [(GROUND_X[0] + GROUND_X[1]) / 2, (GROUND_Y[0] + GROUND_Y[1]) / 2]
    model.geom_pos[plane] = on_plane(*centre, 0.0)
    model.geom_quat[plane] = tilt
    model.geom_size[plane, :2] = np.array([GROUND_X[1] - GROUND_X[0],
                                           GROUND_Y[1] - GROUND_Y[0]]) / 2 / normal[2]
    model.geom_pos[[*boxes, *bumps]] = offset + [0.0, 0.0, PARKED]
    model.geom_size[[*boxes, *bumps]] = [0.1, 0.1, 0.1]
    for geom, box in zip(boxes, ground.boxes, strict=False):
        half_height = (box.height + BURY) / 2
        model.geom_pos[geom] = on_plane(box.x, box.y, box.height - half_height)
        model.geom_quat[geom] = _turned(tilt, box.heading)
        model.geom_size[geom] = [box.half_length, box.half_width, half_height]
    for geom, bump in zip(bumps, ground.bumps, strict=False):
        radius = (bump.radius**2 + bump.height**2) / (2 * bump.height)
        model.geom_pos[geom] = on_plane(bump.x, bump.y, bump.height - radius)
        model.geom_size[geom] = [radius, 0.0, 0.0]
    for geom in boxes:
        model.geom_rbound[geom] = np.linalg.norm(model.geom_size[geom])
        model.geom_aabb[geom] = [0.0, 0.0, 0.0, *model.geom_size[geom]]
    for geom in bumps:
        r = model.geom_size[geom][0]
        model.geom_rbound[geom] = r
        model.geom_aabb[geom] = [0.0, 0.0, 0.0, r, r, r]
    write_friction(model, index, ground.friction)


def ground_height(model: mujoco.MjModel, index: int, points: np.ndarray) -> np.ndarray:
    plane, boxes, bumps = ground_geoms(model, index)
    offset = _offset(model, index)
    rel = np.atleast_2d(points)[:, :2] - offset[:2]
    normal = np.zeros(3)
    mujoco.mju_rotVecQuat(normal, [0.0, 0.0, 1.0], model.geom_quat[plane])
    height = offset[2] - (normal[0] * rel[:, 0] + normal[1] * rel[:, 1]) / normal[2]
    top = 1e3
    down = np.array([0.0, 0.0, -1.0])
    for geom in (*boxes, *bumps):
        pos = model.geom_pos[geom]
        if pos[2] < offset[2] + PARKED / 2:
            continue
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, model.geom_quat[geom])
        for k, (x, y) in enumerate(np.atleast_2d(points)[:, :2]):
            d = mujoco.mju_rayGeom(pos, mat, model.geom_size[geom], np.array([x, y, top]), down,
                                   int(model.geom_type[geom]))
            if d >= 0:
                height[k] = max(height[k], top - d)
    return height


def _own_joints(model: mujoco.MjModel, index: int) -> np.ndarray:
    prefix = machine_prefix(index)
    return np.array([j for j in range(model.njnt) if model.joint(j).name.startswith(prefix)])


def _wheel_points(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> np.ndarray:
    prefix = machine_prefix(index)
    angles = np.radians(np.arange(-90.0, 91.0, 15.0))
    local = np.stack([tracks.WHEEL_RADIUS * np.sin(angles), np.zeros_like(angles),
                      -tracks.WHEEL_RADIUS * np.cos(angles)], axis=1)
    points = []
    for side in tracks.SIDES:
        for k in range(tracks.WHEELS):
            body = model.body(prefix + tracks.wheel_name(side, k)).id
            rot = data.xmat[body].reshape(3, 3)
            for rail in (-tracks.RAIL, tracks.RAIL):
                points.append(data.xpos[body] + (local + [0.0, rail, 0.0]) @ rot.T)
    return np.concatenate(points)


def _put(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float, y: float,
         heading: float, joints: np.ndarray) -> None:
    prefix = machine_prefix(index)
    joint = model.joint(prefix + BASE_JOINT)
    q0 = int(joint.qposadr[0])
    plane, _, _ = ground_geoms(model, index)
    offset = _offset(model, index)
    data.qpos[q0:q0 + 3] = offset + [x, y, 0.0]
    data.qpos[q0 + 3:q0 + 7] = _turned(model.geom_quat[plane], heading)
    for j in joints:
        dof = int(model.jnt_dofadr[j])
        width = 6 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        data.qvel[dof:dof + width] = 0.0
        data.qacc_warmstart[dof:dof + width] = 0.0
    mujoco.mj_kinematics(model, data)
    points = _wheel_points(model, data, index)
    data.qpos[q0 + 2] += float(np.max(ground_height(model, index, points) - points[:, 2]))
    mujoco.mj_kinematics(model, data)


def _travel(model: mujoco.MjModel, index: int) -> list[int]:
    return [model.actuator(machine_prefix(index) + tracks.TRAVEL + s).id for s in tracks.SIDES]


def settle(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float = 0.0,
           y: float = 0.0, heading: float = 0.0) -> bool:
    if not all(math.isfinite(v) for v in (x, y, heading)):
        raise ValueError("the place and heading must be finite")
    prefix = machine_prefix(index)
    joints = _own_joints(model, index)
    data.ctrl[_travel(model, index)] = 0.0
    _put(model, data, index, x, y, heading, joints)
    base = model.joint(prefix + BASE_JOINT)
    q0 = int(base.qposadr[0])
    if data.eq_active[model.equality(prefix + WELD).id]:
        mocap = model.body_mocapid[model.body(prefix + ANCHOR).id]
        data.mocap_pos[mocap] = data.qpos[q0:q0 + 3]
        data.mocap_quat[mocap] = data.qpos[q0 + 3:q0 + 7]
    else:
        data.qpos[q0 + 2] += DROP
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    v0 = int(base.dofadr[0])
    dt = model.opt.timestep
    least, most, needed = (round(t / dt) for t in (SETTLE_MIN, SETTLE_MAX, REST_TIME))
    still = 0
    rested = False
    for n in range(1, most + 1):
        tracks.step(model, scratch)
        still = 0 if np.abs(scratch.qvel[v0:v0 + 6]).max() > REST_SPEED else still + 1
        if still >= needed and n >= least:
            rested = True
            break
    for j in joints:
        q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        nq, nv = (7, 6) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (1, 1)
        data.qpos[q:q + nq] = scratch.qpos[q:q + nq]
        data.qvel[v:v + nv] = scratch.qvel[v:v + nv]
        data.qacc_warmstart[v:v + nv] = scratch.qacc_warmstart[v:v + nv]
    tracks.write_caps(model, data)
    mujoco.mj_kinematics(model, data)
    return rested


def stand(model: mujoco.MjModel, data: mujoco.MjData, index: int, seed: int, x: float = 0.0,
          y: float = 0.0, heading: float = 0.0) -> Ground:
    for attempt in range(DRAWS):
        ground = draw_ground(seed, attempt)
        apply_ground(model, index, ground)
        if settle(model, data, index, x, y, heading):
            return ground
    raise RuntimeError(f"no ground from seed {seed} lets the machine stand at rest")


def off_features(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> bool:
    base = data.xpos[model.body(machine_prefix(index) + "base").id] - _offset(model, index)
    return not (GROUND_X[0] <= base[0] <= GROUND_X[1] and GROUND_Y[0] <= base[1] <= GROUND_Y[1])
