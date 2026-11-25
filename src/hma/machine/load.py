"""A precast concrete block on a four-leg sling from a lifting point on the back of the bucket:
drawn per episode, placed, hooked and unhooked. Hooked or not lives in data.eq_active (two
connect equalities): it resets with MjData, and mj_getState saves it with mjSTATE_EQ_ACTIVE."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import mujoco
import numpy as np

from hma.machine.collision import LINK_MESH, link_triangles, segment_hits
from hma.machine.model import (
    CONTACT_MARGIN,
    CORNERS,
    HOOK,
    HOOK_POS,
    HOOKED,
    LINK_MASSES,
    LOAD,
    LOAD_CORNER,
    LOAD_DENSITY,
    LOAD_POINT_RADIUS,
    LOAD_POINTS,
    LOAD_TOP,
    RING,
    SLING,
    STOWED,
    box_inertia,
    ground_geoms,
    ground_overlap,
    leg_length,
    load_geoms,
    load_points,
    machine_prefix,
    placement_fault,
    rest_home,
    rest_pose,
    ring_at_load,
)
from hma.machine.statics import CHART

HOOK_REACH = float(np.linalg.norm(HOOK_POS))
LIFT_CLEARANCE = 0.3
SIDE_RATING = {height: max(float(side.strip("*")) for (h, _), (_, side) in CHART.items()
                          if h == height) for height, _ in CHART}
LIFT_HEIGHTS = {h: kg - LINK_MASSES["bucket"] for h, kg in SIDE_RATING.items()
                if h - HOOK_REACH - LIFT_CLEARANCE > 0}
RANGES = {
    "mass": (200.0, max(LIFT_HEIGHTS.values())),
    "length_ratio": (1.0, 2.0),
    "height_ratio": (0.5, 1.0),
    "com_offset": (-0.05, 0.05),
    "leg_angle": (np.radians(15.0), np.radians(45.0)),
    "friction": (0.45, 0.70),
}
LEGS = 4
TONNE_FORCE = 1000.0 * 9.80665
LEG_WLL = 3.15 * TONNE_FORCE
SLING_WLL = 6.70 * TONNE_FORCE
DESIGN_FACTOR = 4.0
RESTING_SHARE = 0.95
LOAD_FIELDS = ("body_mass", "body_ipos", "body_inertia", "body_iquat", "body_pos", "geom_size",
               "geom_rbound", "geom_aabb", "geom_friction", "geom_pos", "site_pos", "tendon_range",
               "key_qpos", "qpos0")
STILL_SPEED = 0.01


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


def liftable_mass(length_ratio: float, height_ratio: float, leg_angle: float) -> float:
    per_width = np.hypot(length_ratio, 1.0) / 2 / np.tan(leg_angle) + height_ratio
    return max(min(rating, LOAD_DENSITY * length_ratio * height_ratio
                   * ((height - HOOK_REACH - LIFT_CLEARANCE) / per_width) ** 3)
               for height, rating in LIFT_HEIGHTS.items())


def draw_load(seed: int) -> Load:
    rng = np.random.default_rng((seed, 3))
    while True:
        ratios = rng.uniform(*RANGES["length_ratio"]), rng.uniform(*RANGES["height_ratio"])
        angle = float(rng.uniform(*RANGES["leg_angle"]))
        top = liftable_mass(*ratios, angle)
        if top >= RANGES["mass"][0]:
            break
    shape = block(rng.uniform(RANGES["mass"][0], top), *ratios)
    com = tuple(float(f * 2 * h) for f, h in
                zip(rng.uniform(*RANGES["com_offset"], 3), shape.half, strict=True))
    return Load(shape.mass, shape.half, com, angle, float(rng.uniform(*RANGES["friction"])))


def _ids(model: mujoco.MjModel, index: int):
    prefix = machine_prefix(index)
    joint = model.joint(prefix + LOAD)
    ring = model.joint(prefix + RING + "_x")
    return (prefix, model.body(prefix + LOAD).id, model.geom(prefix + LOAD).id,
            int(joint.qposadr[0]), int(joint.dofadr[0]), int(ring.qposadr[0]),
            int(ring.dofadr[0]),
            np.array([model.tendon(f"{prefix}{SLING}{k}").id for k in range(LEGS)]))


def _equalities(model: mujoco.MjModel, index: int) -> tuple[int, int]:
    prefix = machine_prefix(index)
    return model.equality(prefix + HOOKED).id, model.equality(prefix + STOWED).id


def apply_load(model: mujoco.MjModel, index: int, load: Load, scratch: mujoco.MjData) -> None:
    half, com = np.array(load.half), np.array(load.com)
    if not (load.mass > 0 and min(half) > 0 and load.friction >= 0):
        raise ValueError("load mass and size must be positive and friction not negative")
    if min(half) < LOAD_POINT_RADIUS:
        raise ValueError(f"every side of the block must be at least {2 * LOAD_POINT_RADIUS} m, "
                         "the diameter of its contact spheres")
    if not 0 < load.leg_angle < np.pi / 2:
        raise ValueError("sling legs must lean between vertical and horizontal")
    if np.any(np.abs(com) >= half):
        raise ValueError("the centre of mass must lie inside the block")
    inertia = np.diag(box_inertia(load.mass, half)) - load.mass * (com @ com * np.eye(3)
                                                                    - np.outer(com, com))
    values, axes = np.linalg.eigh(inertia)
    if values[0] <= 0 or 2 * values.max() >= values.sum():
        raise ValueError("the centre of mass is too far off centre for a solid block: its "
                         "inertia about the centre of mass is not physical")
    if np.linalg.det(axes) < 0:
        axes[:, 0] = -axes[:, 0]
    saved = {name: getattr(model, name).copy() for name in LOAD_FIELDS}
    try:
        _write_load(model, index, load, values, axes)
        rest_home(model, index)
    except ValueError:
        for name, value in saved.items():
            getattr(model, name)[:] = value
        raise
    mujoco.mj_setConst(model, scratch)


def _write_load(model: mujoco.MjModel, index: int, load: Load, values: np.ndarray,
                axes: np.ndarray) -> None:
    half, com = np.array(load.half), np.array(load.com)
    prefix, body, geom, *_, tendons = _ids(model, index)
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
    model.site_pos[model.site(prefix + LOAD_TOP).id] = [0.0, 0.0, half[2]]
    model.tendon_range[tendons] = [0.0, load.leg]


def legs_cross_machine(model: mujoco.MjModel, data: mujoco.MjData, index: int,
                       start: np.ndarray) -> bool:
    prefix = machine_prefix(index)
    corners = [data.site_xpos[model.site(f"{prefix}{LOAD_CORNER}{k}").id] for k in range(LEGS)]
    for link in LINK_MESH:
        body = model.body(prefix + link).id
        rot, origin = data.xmat[body].reshape(3, 3), data.xpos[body]
        local = (start - origin) @ rot
        for corner in corners:
            if len(segment_hits(local, (corner - origin) @ rot, link_triangles(link))):
                return True
    return False


def _foreign_contacts(model: mujoco.MjModel, data: mujoco.MjData, index: int,
                      allow_ground: bool) -> bool:
    mine = set(load_geoms(model, index).tolist())
    ground = set(ground_geoms(model, index).tolist()) if allow_ground else set()
    for c in data.contact[:data.ncon]:
        pair = {c.geom1, c.geom2}
        if pair & mine and not pair & ground:
            return True
    return False


def _commit(model: mujoco.MjModel, data: mujoco.MjData, scratch: mujoco.MjData) -> None:
    mujoco.mj_forward(model, scratch)
    mujoco.mj_copyData(data, model, scratch)


def _stow(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
    *_, r0, w0, _ = _ids(model, index)
    hooked, stowed = _equalities(model, index)
    data.qpos[r0:r0 + 3] = ring_at_load(model, data, index)
    data.qvel[w0:w0 + 3] = 0.0
    data.eq_active[hooked], data.eq_active[stowed] = 0, 1


def _attach(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
    prefix, *_, r0, w0, _ = _ids(model, index)
    hooked, stowed = _equalities(model, index)
    site = model.site(prefix + HOOK).id
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site, velocity, 0)
    data.qpos[r0:r0 + 3] = data.site_xpos[site] - model.body_pos[model.body(prefix + RING).id]
    data.qvel[w0:w0 + 3] = velocity[3:]
    data.eq_active[hooked], data.eq_active[stowed] = 1, 0


def hooked(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> bool:
    return bool(data.eq_active[_equalities(model, index)[0]])


def place_load(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float, y: float,
               yaw: float = 0.0) -> None:
    if hooked(model, data, index):
        raise ValueError("unhook the load before placing it")
    _, _, _, q0, v0, *_ = _ids(model, index)
    scratch = copy.copy(data)
    scratch.qpos[q0:q0 + 7] = rest_pose(model, scratch, index, x, y, yaw)
    scratch.qvel[v0:v0 + 6] = 0.0
    _stow(model, scratch, index)
    fault = placement_fault(model, scratch, index)
    if fault is not None:
        raise ValueError(fault)
    _commit(model, data, scratch)


def hang_load(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
    prefix, _, geom, q0, v0, *_, tendons = _ids(model, index)
    scratch = copy.copy(data)
    mujoco.mj_kinematics(model, scratch)
    hook = scratch.site_xpos[model.site(prefix + HOOK).id].copy()
    half = model.geom_size[geom]
    drop = np.sqrt(model.tendon_range[tendons[0], 1] ** 2 - half[0] ** 2 - half[1] ** 2)
    scratch.qpos[q0:q0 + 7] = [*(hook - [0.0, 0.0, drop + half[2]]), 1.0, 0.0, 0.0, 0.0]
    scratch.qvel[v0:v0 + 6] = 0.0
    _attach(model, scratch, index)
    mujoco.mj_forward(model, scratch)
    if (_foreign_contacts(model, scratch, index, allow_ground=False)
            or ground_overlap(model, scratch, index) > -CONTACT_MARGIN):
        raise ValueError("the load would hang inside the machine or the ground; raise the hook")
    if legs_cross_machine(model, scratch, index, hook):
        raise ValueError("the sling legs would pass through the machine; curl the bucket")
    _commit(model, data, scratch)


def hook_load(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
    if hooked(model, data, index):
        raise ValueError("the load is already hooked")
    prefix, body, geom, *_, tendons = _ids(model, index)
    scratch = copy.copy(data)
    mujoco.mj_kinematics(model, scratch)
    hook = scratch.site_xpos[model.site(prefix + HOOK).id].copy()
    local = scratch.xmat[body].reshape(3, 3).T @ (hook - scratch.xpos[body])
    if local[2] <= model.geom_size[geom][2]:
        raise ValueError("the hook must be above the top of the load")
    corners = np.array([scratch.site_xpos[model.site(f"{prefix}{LOAD_CORNER}{k}").id]
                        for k in range(LEGS)])
    if np.any(np.linalg.norm(corners - hook, axis=1) > model.tendon_range[tendons, 1]):
        raise ValueError("the hook is too far from the load to attach the sling")
    if legs_cross_machine(model, scratch, index, hook):
        raise ValueError("the sling legs would pass through the machine; curl the bucket")
    _attach(model, scratch, index)
    _commit(model, data, scratch)


def ground_support(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    mine = set(load_geoms(model, index).tolist())
    ground = set(ground_geoms(model, index).tolist())
    force = np.zeros(6)
    total = 0.0
    for i, c in enumerate(data.contact[:data.ncon]):
        if {c.geom1, c.geom2} & mine and {c.geom1, c.geom2} & ground:
            mujoco.mj_contactForce(model, data, i, force)
            world = c.frame.reshape(3, 3).T @ force[:3]
            total += world[2] if c.geom1 in ground else -world[2]
    return float(total)


def unhook_load(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> None:
    if not hooked(model, data, index):
        raise ValueError("the load is not hooked")
    _, body, *_ = _ids(model, index)
    scratch = copy.copy(data)
    mujoco.mj_forward(model, scratch)
    weight = model.body_mass[body] * -model.opt.gravity[2]
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, scratch, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 0)
    if ground_support(model, scratch, index) < RESTING_SHARE * weight:
        raise ValueError("the sling is still carrying the load; set it down first")
    if max(np.linalg.norm(velocity[:3]), np.linalg.norm(velocity[3:])) > STILL_SPEED:
        raise ValueError("the load is still moving; wait for it to settle")
    _stow(model, scratch, index)
    _commit(model, data, scratch)


def leg_tensions(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> np.ndarray:
    tendons = _ids(model, index)[-1]
    limit = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON
    return np.array([np.abs(data.efc_force[:data.nefc][limit & (data.efc_id[:data.nefc] == t)])
                     .sum() for t in tendons])


def sling_tension(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    return float(leg_tensions(model, data, index).sum())


def hook_force(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    rows = ((data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            & (data.efc_id[:data.nefc] == _equalities(model, index)[0]))
    return float(np.linalg.norm(data.efc_force[:data.nefc][rows]))
