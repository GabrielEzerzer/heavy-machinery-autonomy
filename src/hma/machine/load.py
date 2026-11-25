"""A precast concrete block slung from the bucket's lifting eye by a four-leg sling: drawn per
episode (size, mass, centre of mass, inertia, sling angle, friction), placed on the ground, hooked
and unhooked. The sling legs are tension-only tendon limits that act only while hooked."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from hma.machine.contact import ground_geoms
from hma.machine.model import (
    HOME_KEY,
    HOOK,
    LOAD,
    LOAD_CORNER,
    LOAD_DENSITY,
    LOAD_HOME,
    LOAD_POINTS,
    MACHINE_SPACING,
    SLING,
    box_inertia,
    leg_length,
    load_points,
    machine_prefix,
)

RANGES = {
    "mass": (200.0, 7500.0),
    "length_ratio": (1.0, 2.0),
    "height_ratio": (0.5, 1.0),
    "com_offset": (-0.05, 0.05),
    "leg_angle": (np.radians(15.0), np.radians(45.0)),
    "friction": (0.45, 0.70),
}
LEGS = 4
SLACK = 1e-3
RAY_TOP = 100.0
DOWN = np.array([0.0, 0.0, -1.0])
CORNERS = ((1, 1), (-1, 1), (-1, -1), (1, -1))


@dataclass(frozen=True)
class Load:
    mass: float
    half: tuple[float, float, float]
    com: tuple[float, float, float]
    leg_angle: float
    friction: float

    @property
    def leg(self) -> float:
        return leg_length(np.array(self.half), self.leg_angle)


def block(mass: float, length_ratio: float = 1.4, height_ratio: float = 0.8,
          com: tuple[float, float, float] = (0.0, 0.0, 0.0), leg_angle: float = np.radians(30.0),
          friction: float = 0.55) -> Load:
    width = (mass / LOAD_DENSITY / (length_ratio * height_ratio)) ** (1.0 / 3.0)
    half = (length_ratio * width / 2, width / 2, height_ratio * width / 2)
    return Load(float(mass), half, tuple(map(float, com)), float(leg_angle), float(friction))


def draw_load(seed: int) -> Load:
    rng = np.random.default_rng((seed, 3))
    shape = block(rng.uniform(*RANGES["mass"]), rng.uniform(*RANGES["length_ratio"]),
                  rng.uniform(*RANGES["height_ratio"]))
    com = tuple(float(f * 2 * h) for f, h in
                zip(rng.uniform(*RANGES["com_offset"], 3), shape.half, strict=True))
    return Load(shape.mass, shape.half, com, float(rng.uniform(*RANGES["leg_angle"])),
                float(rng.uniform(*RANGES["friction"])))


def _ids(model: mujoco.MjModel, index: int):
    prefix = machine_prefix(index)
    joint = model.joint(prefix + LOAD)
    return (prefix, model.body(prefix + LOAD).id, model.geom(prefix + LOAD).id,
            int(joint.qposadr[0]), int(joint.dofadr[0]),
            np.array([model.tendon(f"{prefix}{SLING}{k}").id for k in range(LEGS)]))


def ground_height(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float,
                  y: float) -> float | None:
    top = np.array([x, y + index * MACHINE_SPACING, RAY_TOP])
    best = np.inf
    for geom in ground_geoms(model, index):
        kind = model.geom_type[geom]
        if kind == mujoco.mjtGeom.mjGEOM_HFIELD:
            dist = mujoco.mj_rayHfield(model, data, geom, top, DOWN)
        elif kind == mujoco.mjtGeom.mjGEOM_MESH:
            dist = mujoco.mj_rayMesh(model, data, geom, top, DOWN)
        else:
            dist = mujoco.mju_rayGeom(data.geom_xpos[geom], data.geom_xmat[geom],
                                      model.geom_size[geom], top, DOWN, kind)
        if dist >= 0:
            best = min(best, dist)
    return None if best == np.inf else float(RAY_TOP - best)


def resting_height(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float, y: float,
                   yaw: float, half: np.ndarray) -> float:
    mujoco.mj_kinematics(model, data)
    c, s = np.cos(yaw), np.sin(yaw)
    u = np.linspace(-1.0, 1.0, 9)
    gx, gy = np.meshgrid(u * half[0], u * half[1])
    px, py = x + c * gx - s * gy, y + s * gx + c * gy
    heights = [ground_height(model, data, index, a, b)
               for a, b in zip(px.ravel(), py.ravel(), strict=True)]
    if None in heights:
        raise ValueError("the load must be placed on this machine's ground")
    return max(heights)


def apply_load(model: mujoco.MjModel, index: int, load: Load, scratch: mujoco.MjData) -> None:
    half, com = np.array(load.half), np.array(load.com)
    if not (load.mass > 0 and min(half) > 0 and load.friction >= 0):
        raise ValueError("load mass and size must be positive and friction not negative")
    if not 0 < load.leg_angle < np.pi / 2:
        raise ValueError("sling legs must lean between vertical and horizontal")
    if np.any(np.abs(com) >= half):
        raise ValueError("the centre of mass must lie inside the block")
    prefix, body, geom, _, _, tendons = _ids(model, index)
    inertia = np.diag(box_inertia(load.mass, half)) - load.mass * (com @ com * np.eye(3)
                                                                    - np.outer(com, com))
    values, axes = np.linalg.eigh(inertia)
    if np.linalg.det(axes) < 0:
        axes[:, 0] = -axes[:, 0]
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, axes.ravel())
    model.body_mass[body] = load.mass
    model.body_ipos[body] = com
    model.body_inertia[body] = values
    model.body_iquat[body] = quat
    model.geom_size[geom] = half
    model.geom_rbound[geom] = np.linalg.norm(half)
    model.geom_aabb[geom] = [0.0, 0.0, 0.0, *half]
    model.geom_friction[geom, 0] = load.friction
    for k, pos in enumerate(load_points(half)):
        point = model.geom(f"{prefix}{LOAD_POINTS}{k}").id
        model.geom_pos[point] = pos
        model.geom_friction[point, 0] = load.friction
    for k, (sx, sy) in enumerate(CORNERS):
        model.site_pos[model.site(f"{prefix}{LOAD_CORNER}{k}").id] = [sx * half[0], sy * half[1],
                                                                      half[2]]
    model.tendon_range[tendons] = [0.0, load.leg]
    model.tendon_limited[tendons] = 0
    rest_home(model, scratch, index)
    mujoco.mj_setConst(model, scratch)


def rest_home(model: mujoco.MjModel, scratch: mujoco.MjData, index: int) -> None:
    _, _, geom, q0, *_ = _ids(model, index)
    half = model.geom_size[geom]
    x, y = LOAD_HOME
    z = resting_height(model, scratch, index, x, y, 0.0, half) + half[2]
    model.key_qpos[model.key(HOME_KEY).id, q0:q0 + 7] = [
        x, y + index * MACHINE_SPACING, z,
        1.0, 0.0, 0.0, 0.0]


def place_load(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float, y: float,
               yaw: float = 0.0) -> None:
    _, _, geom, q0, v0, tendons = _ids(model, index)
    half = model.geom_size[geom]
    z = resting_height(model, data, index, x, y, yaw, half) + half[2]
    data.qpos[q0:q0 + 7] = [x, y + index * MACHINE_SPACING, z, np.cos(yaw / 2), 0.0, 0.0,
                            np.sin(yaw / 2)]
    data.qvel[v0:v0 + 6] = 0.0
    model.tendon_limited[tendons] = 0
    mujoco.mj_forward(model, data)


def hang_load(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
    prefix, _, geom, q0, v0, tendons = _ids(model, index)
    mujoco.mj_kinematics(model, data)
    hook = data.site_xpos[model.site(prefix + HOOK).id]
    half = model.geom_size[geom]
    drop = np.sqrt(model.tendon_range[tendons[0], 1] ** 2 - half[0] ** 2 - half[1] ** 2)
    data.qpos[q0:q0 + 7] = [*(hook - [0.0, 0.0, drop + half[2]]), 1.0, 0.0, 0.0, 0.0]
    data.qvel[v0:v0 + 6] = 0.0
    model.tendon_limited[tendons] = 1
    mujoco.mj_forward(model, data)
    if any(geom in (c.geom1, c.geom2) and c.dist < 0 for c in data.contact[:data.ncon]):
        raise ValueError("the load would hang inside the machine or the ground; curl the bucket "
                         "or raise the hook")


def _leg_lengths(model: mujoco.MjModel, data: mujoco.MjData, tendons: np.ndarray) -> np.ndarray:
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_tendon(model, data)
    return data.ten_length[tendons]


def hook_load(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
    *_, tendons = _ids(model, index)
    if np.any(_leg_lengths(model, data, tendons) > model.tendon_range[tendons, 1]):
        raise ValueError("the hook is too far from the load to attach the sling")
    model.tendon_limited[tendons] = 1


def unhook_load(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
    *_, tendons = _ids(model, index)
    if np.any(_leg_lengths(model, data, tendons) > model.tendon_range[tendons, 1] - SLACK):
        raise ValueError("the sling is still carrying the load; set it down first")
    model.tendon_limited[tendons] = 0


def hooked(model: mujoco.MjModel, index: int) -> bool:
    return bool(model.tendon_limited[_ids(model, index)[5]].all())


def sling_tension(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    tendons = _ids(model, index)[5]
    rows = ((data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)
            & np.isin(data.efc_id[:data.nefc], tendons))
    return float(np.abs(data.efc_force[:data.nefc][rows]).sum())
