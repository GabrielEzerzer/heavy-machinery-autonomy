"""Rigid ground for one machine from MuJoCo primitives: a tilted, levelled plane with boxes and
buried spheres, the load resting on it. settle() steps a machine to rest, arm held, or refuses,
and puts a welded machine at home height; stand() redraws a seed until the machine stands."""

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
    HOME_KEY,
    JOINT_NAMES,
    LOAD,
    PARKED,
    RING,
    WELD,
    ground_height,
    ground_parts,
    machine_prefix,
    rest_home,
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
    "friction": tracks.FRICTION_RANGE,
}
FEATURE_X = (-3.0, 10.0)
FEATURE_Y = (-6.0, 6.0)
REACH = 6.0
BANK = 2.0
FIT_TRIES = 40
FIT_SHRINK = 0.9
MIN_WIDTH = 1.5
RIM_POINTS = 32
SHADOW_REACH = SCENERY_REACH
BURY = 0.5
DROP = 0.01
SETTLE_MIN = 0.5
SETTLE_MAX = 6.0
REST_SPEED = 1e-3
REST_TIME = 0.25
DRAWS = 20
CLEAR_TRIES = 8
MAX_TILT = 15.0
MAX_SHIFT = 0.3
HOLD = 1.0e9


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
    level: float = 0.0


def inside(points: np.ndarray) -> bool:
    points = np.atleast_2d(points)
    sun = np.array(SUN_DIR) / np.linalg.norm(SUN_DIR)
    rel = points - SHOULDER
    across = np.linalg.norm(rel - np.outer(rel @ sun, sun), axis=1)
    in_field = ((FEATURE_X[0] <= points[:, 0]) & (points[:, 0] <= FEATURE_X[1])
                & (FEATURE_Y[0] <= points[:, 1]) & (points[:, 1] <= FEATURE_Y[1]))
    return bool(np.all(in_field & (across <= SHADOW_REACH)))


def _normal(grade: tuple[float, float]) -> np.ndarray:
    return np.array([-grade[0], -grade[1], 1.0]) / math.hypot(1.0, *grade)


def _on_plane(grade: tuple[float, float], x: float, y: float, lift: float) -> np.ndarray:
    return np.array([x, y, grade[0] * x + grade[1] * y]) + lift * _normal(grade)


def place_box(grade: tuple[float, float], box: Box) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    half_height = (box.height + BURY) / 2
    return (_on_plane(grade, box.x, box.y, box.height - half_height),
            _turned(_tilt(grade), box.heading),
            np.array([box.half_length, box.half_width, half_height]))


def box_points(grade: tuple[float, float], box: Box) -> np.ndarray:
    pos, quat, size = place_box(grade, box)
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    top = np.array([[a * size[0], b * size[1], size[2]] for a in (-1, 1) for b in (-1, 1)])
    feet = top - [0.0, 0.0, box.height]
    return pos + np.concatenate([top, feet]) @ mat.reshape(3, 3).T


def place_bump(grade: tuple[float, float], bump: Bump) -> tuple[np.ndarray, float]:
    radius = (bump.radius**2 + bump.height**2) / (2 * bump.height)
    return _on_plane(grade, bump.x, bump.y, bump.height - radius), radius


def bump_points(grade: tuple[float, float], bump: Bump) -> np.ndarray:
    normal = _normal(grade)
    across = np.cross(normal, [1.0, 0.0, 0.0])
    across /= np.linalg.norm(across)
    along = np.cross(across, normal)
    angles = np.linspace(0.0, 2 * math.pi, RIM_POINTS, endpoint=False)
    rim = bump.radius * (np.outer(np.cos(angles), along) + np.outer(np.sin(angles), across))
    return np.vstack([_on_plane(grade, bump.x, bump.y, 0.0) + rim,
                      _on_plane(grade, bump.x, bump.y, bump.height)])


def _fit(rng: np.random.Generator, grade: tuple[float, float], length: float, width: float,
         height: float, offsets: tuple[float, ...], grows: bool) -> list[Box]:
    for _ in range(FIT_TRIES):
        heading = float(rng.uniform(0.0, 2 * math.pi))
        n = np.array([math.cos(heading), math.sin(heading)])
        at = np.array([rng.uniform(*FEATURE_X), rng.uniform(*FEATURE_Y)])
        boxes = [Box(*(float(v) for v in at + (offset + (length if grows else 0.0)) * n),
                     heading, length, width, height) for offset in offsets]
        if all(inside(box_points(grade, b)) for b in boxes):
            return boxes
        width = max(MIN_WIDTH, width * FIT_SHRINK)
        length = max(MIN_WIDTH, length * FIT_SHRINK) if grows else length
    raise RuntimeError("a drawn feature does not fit in the field")


def _bump(rng: np.random.Generator, grade: tuple[float, float]) -> Bump:
    radius = float(rng.uniform(*RANGES["bump_radius"]))
    height = float(rng.uniform(*RANGES["bump_height"]))
    for _ in range(FIT_TRIES):
        bump = Bump(float(rng.uniform(*FEATURE_X)), float(rng.uniform(*FEATURE_Y)), radius,
                    height)
        if inside(bump_points(grade, bump)):
            return bump
    raise RuntimeError("a drawn bump does not fit in the field")


def draw_ground(seed: int, attempt: int = 0) -> Ground:
    if seed < 0 or attempt < 0:
        raise ValueError("seed and attempt must be non-negative integers")
    rng = np.random.default_rng((seed, 3, attempt))
    slope = rng.uniform(*RANGES["slope"])
    direction = rng.uniform(0.0, 2 * math.pi)
    grade = (math.tan(slope) * math.cos(direction), math.tan(slope) * math.sin(direction))
    boxes: list[Box] = []
    for _ in range(rng.integers(RANGES["steps"][0], RANGES["steps"][1] + 1)):
        boxes += _fit(rng, grade, REACH, REACH, rng.uniform(*RANGES["step_height"]), (0.0,), True)
    for _ in range(rng.integers(RANGES["kerbs"][0], RANGES["kerbs"][1] + 1)):
        boxes += _fit(rng, grade, rng.uniform(*RANGES["kerb_width"]) / 2, REACH,
                      rng.uniform(*RANGES["kerb_height"]), (0.0,), False)
    for _ in range(rng.integers(RANGES["ruts"][0], RANGES["ruts"][1] + 1)):
        half = rng.uniform(*RANGES["rut_width"]) / 2 + BANK / 2
        boxes += _fit(rng, grade, BANK / 2, REACH, rng.uniform(*RANGES["rut_depth"]),
                      (-half, half), False)
    bumps = [_bump(rng, grade)
             for _ in range(rng.integers(RANGES["bumps"][0], RANGES["bumps"][1] + 1))]
    return Ground(grade, float(rng.uniform(*RANGES["friction"])), tuple(boxes), tuple(bumps),
                  seed)


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


GEOM_STATE = ("geom_pos", "geom_quat", "geom_size", "geom_rbound", "geom_aabb", "geom_friction",
              "pair_friction", "bvh_aabb", "geom_sameframe")


def apply_ground(model: mujoco.MjModel, index: int, ground: Ground) -> None:
    if len(ground.boxes) > GROUND_BOXES or len(ground.bumps) > GROUND_BUMPS:
        raise ValueError(f"a ground holds at most {GROUND_BOXES} boxes and {GROUND_BUMPS} bumps")
    if not all(math.isfinite(g) for g in (*ground.grade, ground.level)):
        raise ValueError("the grade and level must be finite")
    saved = {name: getattr(model, name).copy() for name in GEOM_STATE}
    try:
        _write_ground(model, index, ground)
        rest_home(model, index)
    except ValueError:
        for name, value in saved.items():
            getattr(model, name)[:] = value
        raise


def _write_ground(model: mujoco.MjModel, index: int, ground: Ground) -> None:
    offset = _offset(model, index) + [0.0, 0.0, ground.level]
    plane, boxes, bumps = ground_parts(model, index)
    model.geom_sameframe[[plane, *boxes, *bumps]] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
    centre = ((GROUND_X[0] + GROUND_X[1]) / 2, (GROUND_Y[0] + GROUND_Y[1]) / 2)
    model.geom_pos[plane] = offset + _on_plane(ground.grade, *centre, 0.0)
    model.geom_quat[plane] = _tilt(ground.grade)
    model.geom_size[plane, :2] = np.array([GROUND_X[1] - GROUND_X[0],
                                           GROUND_Y[1] - GROUND_Y[0]]) / 2 / _normal(
                                               ground.grade)[2]
    model.geom_pos[[*boxes, *bumps]] = _offset(model, index) + [0.0, 0.0, PARKED]
    model.geom_size[[*boxes, *bumps]] = [0.1, 0.1, 0.1]
    for geom, box in zip(boxes, ground.boxes, strict=False):
        pos, quat, size = place_box(ground.grade, box)
        model.geom_pos[geom] = offset + pos
        model.geom_quat[geom] = quat
        model.geom_size[geom] = size
    for geom, bump in zip(bumps, ground.bumps, strict=False):
        pos, radius = place_bump(ground.grade, bump)
        model.geom_pos[geom] = offset + pos
        model.geom_size[geom] = [radius, 0.0, 0.0]
    for geom in boxes:
        model.geom_rbound[geom] = np.linalg.norm(model.geom_size[geom])
        model.geom_aabb[geom] = [0.0, 0.0, 0.0, *model.geom_size[geom]]
    for geom in bumps:
        r = model.geom_size[geom][0]
        model.geom_rbound[geom] = r
        model.geom_aabb[geom] = [0.0, 0.0, 0.0, r, r, r]
    write_friction(model, index, ground.friction)
    _world_bounds(model)


def _world_bounds(model: mujoco.MjModel) -> None:
    start, count = int(model.body_bvhadr[0]), int(model.body_bvhnum[0])
    rot = np.zeros(9)
    for node in range(start + count - 1, start - 1, -1):
        geom = model.bvh_nodeid[node]
        if geom >= 0:
            mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
            mat = rot.reshape(3, 3)
            centre = model.geom_pos[geom] + mat @ model.geom_aabb[geom, :3]
            half = np.abs(mat) @ model.geom_aabb[geom, 3:]
        else:
            boxes = model.bvh_aabb[start + model.bvh_child[node]]
            low = np.min(boxes[:, :3] - boxes[:, 3:], axis=0)
            high = np.max(boxes[:, :3] + boxes[:, 3:], axis=0)
            centre, half = (low + high) / 2, (high - low) / 2
        model.bvh_aabb[node] = [*centre, *half]


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
    plane, _, _ = ground_parts(model, index)
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
    data.qpos[q0 + 2] += float(np.nanmax(ground_height(model, index, points) - points[:, 2],
                                         initial=-np.inf))
    base = model.body(prefix + "base").id
    for _ in range(CLEAR_TRIES):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        depth = _deepest(model, data, base)
        if depth <= 0.0:
            break
        data.qpos[q0 + 2] += depth
    mujoco.mj_kinematics(model, data)


def _hold_at(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float, y: float,
             heading: float, joints: np.ndarray) -> None:
    q0 = int(model.joint(machine_prefix(index) + BASE_JOINT).qposadr[0])
    key = model.key(HOME_KEY).id
    data.qpos[q0:q0 + 3] = _offset(model, index) + [x, y, model.key_qpos[key, q0 + 2]]
    data.qpos[q0 + 3:q0 + 7] = _turned(model.key_qpos[key, q0 + 3:q0 + 7], heading)
    for j in joints:
        dof = int(model.jnt_dofadr[j])
        width = 6 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        data.qvel[dof:dof + width] = 0.0
        data.qacc_warmstart[dof:dof + width] = 0.0
    mujoco.mj_kinematics(model, data)


def _deepest(model: mujoco.MjModel, data: mujoco.MjData, base: int) -> float:
    contacts = data.contact[:data.ncon]
    own = ((model.body_rootid[model.geom_bodyid[contacts.geom[:, 0]]] == base)
           | (model.body_rootid[model.geom_bodyid[contacts.geom[:, 1]]] == base))
    return float(np.max(-contacts.dist[own], initial=0.0))


def upright(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    plane, _, _ = ground_parts(model, index)
    normal = np.zeros(3)
    mujoco.mju_rotVecQuat(normal, [0.0, 0.0, 1.0], model.geom_quat[plane])
    up = data.xmat[model.body(machine_prefix(index) + "base").id].reshape(3, 3)[:, 2]
    return math.degrees(math.acos(float(np.clip(up @ normal, -1.0, 1.0))))


def _travel(model: mujoco.MjModel, index: int) -> list[int]:
    return [model.actuator(machine_prefix(index) + tracks.TRAVEL + s).id for s in tracks.SIDES]


def settle(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float = 0.0,
           y: float = 0.0, heading: float = 0.0) -> bool:
    if not all(math.isfinite(v) for v in (x, y, heading)):
        raise ValueError("the place and heading must be finite")
    prefix = machine_prefix(index)
    joints = _own_joints(model, index)
    data.ctrl[_travel(model, index)] = 0.0
    data.userdata[index] = 0.0
    tracks.used_power(model, data)[index] = 0.0
    base = model.joint(prefix + BASE_JOINT)
    q0 = int(base.qposadr[0])
    if data.eq_active[model.equality(prefix + WELD).id]:
        _hold_at(model, data, index, x, y, heading, joints)
        mocap = model.body_mocapid[model.body(prefix + ANCHOR).id]
        data.mocap_pos[mocap] = data.qpos[q0:q0 + 3]
        data.mocap_quat[mocap] = data.qpos[q0 + 3:q0 + 7]
        return True
    _put(model, data, index, x, y, heading, joints)
    data.qpos[q0 + 2] += DROP
    scratch = mujoco.MjData(model)
    mujoco.mj_copyData(scratch, model, data)
    v0 = int(base.dofadr[0])
    dt = model.opt.timestep
    least, most, needed = (round(t / dt) for t in (SETTLE_MIN, SETTLE_MAX, REST_TIME))
    still = 0
    rested = False
    arm = [int(model.joint(prefix + name).dofadr[0]) for name in JOINT_NAMES]
    kept = model.dof_frictionloss[arm].copy()
    model.dof_frictionloss[arm] = HOLD
    try:
        for n in range(1, most + 1):
            tracks.step(model, scratch)
            still = 0 if np.abs(scratch.qvel[v0:v0 + 6]).max() > REST_SPEED else still + 1
            if still >= needed and n >= least:
                rested = True
                break
    finally:
        model.dof_frictionloss[arm] = kept
    for j in joints:
        q, v = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        nq, nv = (7, 6) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else (1, 1)
        data.qpos[q:q + nq] = scratch.qpos[q:q + nq]
        data.qvel[v:v + nv] = scratch.qvel[v:v + nv]
        data.qacc_warmstart[v:v + nv] = scratch.qacc_warmstart[v:v + nv]
        data.qfrc_applied[v:v + nv] = scratch.qfrc_applied[v:v + nv]
    mujoco.mj_kinematics(model, data)
    shift = math.hypot(*(data.qpos[q0:q0 + 2] - _offset(model, index)[:2] - [x, y]))
    return (rested and upright(model, data, index) <= MAX_TILT and shift <= MAX_SHIFT
            and not off_features(model, data, index))


def stand(model: mujoco.MjModel, data: mujoco.MjData, index: int, seed: int, x: float = 0.0,
          y: float = 0.0, heading: float = 0.0) -> Ground:
    own = np.concatenate([np.arange(model.jnt_qposadr[j], model.jnt_qposadr[j]
                                    + (7 if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1))
                          for j in _own_joints(model, index)])
    start = data.qpos[own].copy()
    prefix = machine_prefix(index)
    load = int(model.joint(prefix + LOAD).qposadr[0])
    ring = int(model.joint(prefix + RING + "_x").qposadr[0])
    key = model.key(HOME_KEY).id
    for attempt in range(DRAWS):
        data.qpos[own] = start
        ground = draw_ground(seed, attempt)
        try:
            apply_ground(model, index, ground)
        except ValueError:
            continue
        for q, n in ((load, 7), (ring, 3)):
            data.qpos[q:q + n] = model.key_qpos[key, q:q + n]
        if settle(model, data, index, x, y, heading):
            return ground
    raise RuntimeError(f"no ground from seed {seed} lets the machine stand at rest")


def off_features(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> bool:
    base = data.xpos[model.body(machine_prefix(index) + "base").id] - _offset(model, index)
    return not (GROUND_X[0] <= base[0] <= GROUND_X[1] and GROUND_Y[0] <= base[1] <= GROUND_Y[1])
