"""Scenes of 12 t class excavators, each on its own ground of (y, x) heights; names start with
machine_prefix(i). ctrl is a joint speed in rad/s, sent through a "<joint>_valve" joint: read
joints by name; actuator_force is not joint load. set_weld(off) leaves track contacts on."""

from __future__ import annotations

import itertools
from pathlib import Path

import mujoco
import numpy as np

from hma.machine.collision import add_collision_geoms
from hma.machine.scenery import dress_after_heightfields

ASSETS = Path(__file__).parent / "assets"
MACHINE_XML = ASSETS / "excavator.xml"

JOINT_NAMES = ("swing", "boom", "arm", "bucket")
BODY_NAMES = ("base", "upper", "boom", "arm", "bucket")
BASE_JOINT = "base_free"
TIP_SITE = "bucket_tip"
TRACK_FEET = "track_"
ANCHOR = "anchor"
WELD = "base_weld"
GROUND = "ground"
HOME_KEY = "home"
HOME_POSE = (0.0, -0.6, 1.9, 0.6)

LINK_MASSES = {"base": 4500.0, "upper": 5580.0, "boom": 1220.0, "arm": 600.0, "bucket": 400.0}
COUNTERWEIGHT_MASS = 2450.0
FORCE_CAPS = {"swing": 34.0e3, "boom": 330.0e3, "arm": 244.0e3, "bucket": 122.0e3}
SPEED_LIMITS = {"swing": 1.0, "boom": 0.6, "arm": 0.8, "bucket": 1.2}
VALVE = "_valve"
VALVE_INERTIA = 1.0e8
VALVE_GAIN = 5.0e9
VALVE_SOLIMP = (0.999, 0.9999, 0.001)

GROUND_CELL = 0.1
GROUND_HALF = 13.5
GROUND_X = (-GROUND_HALF - GROUND_CELL, GROUND_HALF + GROUND_CELL)
GROUND_Y = (-GROUND_HALF, GROUND_HALF)
GROUND_DEPTH = 6.0
GROUND_TOP = 2.0
GROUND_BASE = 0.5
MACHINE_SPACING = 2 * GROUND_HALF
MAX_MACHINES = 31

TIMESTEP = 0.005
SOLVER_TOLERANCE = 1e-14
CONTACT_SOLREF = (2 * TIMESTEP, 1.0)
CONTACT_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
IMPRATIO = 10.0
CONTACT_MARGIN = 0.005
REST_DEPTH = 1e-6
STEEL_FRICTION = (0.5, 0.005, 1e-4)
LOAD_FRICTION = (0.55, 0.005, 1e-4)

HOOK = "hook"
HOOK_POS = (-0.393, 0.0, -0.153)
LOAD = "load"
LOAD_CORNER = "load_corner_"
SLING = "sling_"
RING = "sling_ring"
RING_MASS = 2.2
HOOKED = "sling_hooked"
STOWED = "sling_stowed"
LOAD_TOP = "load_top"
CORNERS = ((1, 1), (-1, 1), (-1, -1), (1, -1))
RAY_TOP = 100.0
DOWN = np.array([0.0, 0.0, -1.0])
LOAD_DENSITY = 2400.0
NOMINAL_LOAD_SIZE = (0.535, 0.382, 0.306)
NOMINAL_LEG_ANGLE = 0.5236
LOAD_HOME = (7.0, 0.0)
LOAD_POINTS = "load_point_"
LOAD_POINT_RADIUS = 0.05
FRICTION_TENDON = int(mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)


def machine_prefix(index: int) -> str:
    return f"m{index}/"


def tendon_friction(data: mujoco.MjData, tendons: np.ndarray) -> np.ndarray:
    tendons = np.asarray(tendons, np.intp)
    rows = data.efc_type == FRICTION_TENDON
    forces = np.bincount(data.efc_id[rows], weights=data.efc_force[rows],
                         minlength=int(tendons.max(initial=-1)) + 1)
    return forces[tendons]


def ground_shape() -> tuple[int, int]:
    nrow = round((GROUND_Y[1] - GROUND_Y[0]) / GROUND_CELL) + 1
    ncol = round((GROUND_X[1] - GROUND_X[0]) / GROUND_CELL) + 1
    return nrow, ncol


def _turn_reference_to_home(joint: mujoco.MjsJoint, angle: float) -> None:
    body = joint.parent
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, np.asarray(joint.axis, float), angle)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.asarray(body.quat, float), turn)
    body.quat = quat
    joint.ref = angle


def _machine_spec(index: int) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MACHINE_XML))
    for body, mass in LINK_MASSES.items():
        mesh_geom = next(g for g in spec.body(body).geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH)
        mesh_geom.mass = mass - (COUNTERWEIGHT_MASS if body == "upper" else 0.0)
    spec.geom("counterweight").mass = COUNTERWEIGHT_MASS
    for geom in spec.body("base").geoms:
        if geom.name.startswith(TRACK_FEET):
            geom.contype = 1 << index
            geom.conaffinity = 0
    for name, home in zip(JOINT_NAMES, HOME_POSE, strict=True):
        joint = spec.joint(name)
        _turn_reference_to_home(joint, home)
        joint.solref_limit = [2 * TIMESTEP, 1.0]
        joint.solimp_limit = [*VALVE_SOLIMP, 0.5, 2.0]
        valve = spec.worldbody.add_body(name=name + VALVE, mass=1.0, inertia=[1.0, 1.0, 1.0],
                                        explicitinertial=True)
        valve.add_joint(name=name + VALVE, axis=[0.0, 0.0, 1.0], armature=VALVE_INERTIA)
        tendon = spec.add_tendon(name=name + VALVE, frictionloss=FORCE_CAPS[name],
                                 solref_friction=[2 * TIMESTEP, 1.0],
                                 solimp_friction=[*VALVE_SOLIMP, 0.5, 2.0])
        tendon.wrap_joint(name, 1.0)
        tendon.wrap_joint(name + VALVE, -1.0)
        act = spec.add_actuator(name=name, target=name + VALVE, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_velocity(kv=VALVE_GAIN)
        act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.ctrlrange = [-SPEED_LIMITS[name], SPEED_LIMITS[name]]
    add_collision_geoms(spec, 1 << index, CONTACT_SOLREF, CONTACT_SOLIMP, STEEL_FRICTION,
                        CONTACT_MARGIN)
    spec.body("bucket").add_site(name=HOOK, pos=list(HOOK_POS), size=[0.04],
                                 rgba=[0.1, 0.8, 0.1, 1], group=2)
    return spec


def box_inertia(mass: float, half: np.ndarray) -> np.ndarray:
    a, b, c = (2.0 * np.asarray(half, float)) ** 2
    return mass / 12.0 * np.array([b + c, a + c, a + b])


def leg_length(half: np.ndarray, angle: float) -> float:
    return float(np.hypot(half[0], half[1]) / np.sin(angle))


def load_points(half: np.ndarray) -> np.ndarray:
    inset = np.asarray(half, float) - LOAD_POINT_RADIUS
    return np.array([np.array(k) * inset for k in itertools.product((-1, 0, 1), repeat=3)
                     if any(k)])


def _add_load(scene: mujoco.MjSpec, index: int, prefix: str, offset: np.ndarray) -> None:
    half = np.array(NOMINAL_LOAD_SIZE)
    mass = LOAD_DENSITY * float(np.prod(2.0 * half))
    bit = 1 << index
    contact = {"priority": 1, "friction": list(LOAD_FRICTION), "solref": list(CONTACT_SOLREF),
               "solimp": list(CONTACT_SOLIMP), "margin": CONTACT_MARGIN, "mass": 0.0}
    body = scene.worldbody.add_body(name=prefix + LOAD, pos=offset + [*LOAD_HOME, half[2]],
                                    mass=mass, ipos=[0.0, 0.0, 0.0],
                                    inertia=box_inertia(mass, half).tolist(),
                                    explicitinertial=True)
    body.add_freejoint(name=prefix + LOAD)
    body.add_geom(name=prefix + LOAD, type=mujoco.mjtGeom.mjGEOM_BOX, size=half.tolist(),
                  contype=bit, conaffinity=0, rgba=[0.62, 0.62, 0.6, 1.0], **contact)
    body.add_site(name=prefix + LOAD_TOP, pos=[0.0, 0.0, half[2]], size=[0.03], group=2)
    points = body.add_body(name=prefix + LOAD_POINTS)
    for k, pos in enumerate(load_points(half)):
        points.add_geom(name=f"{prefix}{LOAD_POINTS}{k}", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[LOAD_POINT_RADIUS, 0.0, 0.0], pos=pos.tolist(), contype=bit,
                        conaffinity=0, group=3, **contact)
    scene.add_exclude(bodyname1="world", bodyname2=prefix + LOAD)
    for link in BODY_NAMES:
        scene.add_exclude(bodyname1=prefix + link, bodyname2=prefix + LOAD_POINTS)
    ring = scene.worldbody.add_body(name=prefix + RING, pos=offset + [*LOAD_HOME, 2 * half[2]],
                                    mass=RING_MASS, inertia=[1e-3, 1e-3, 1e-3],
                                    explicitinertial=True)
    for axis in np.eye(3):
        ring.add_joint(name=f"{prefix}{RING}_{'xyz'[int(np.argmax(axis))]}",
                       type=mujoco.mjtJoint.mjJNT_SLIDE, axis=axis.tolist())
    ring.add_site(name=prefix + RING, size=[0.04], rgba=[0.15, 0.15, 0.15, 1.0], group=2)
    for name, other, active in ((HOOKED, HOOK, 0), (STOWED, LOAD_TOP, 1)):
        scene.add_equality(name=prefix + name, type=mujoco.mjtEq.mjEQ_CONNECT,
                           objtype=mujoco.mjtObj.mjOBJ_SITE, name1=prefix + RING,
                           name2=prefix + other, active=active, solref=list(CONTACT_SOLREF),
                           solimp=list(CONTACT_SOLIMP))
    length = leg_length(half, NOMINAL_LEG_ANGLE)
    for k, (sx, sy) in enumerate(CORNERS):
        body.add_site(name=f"{prefix}{LOAD_CORNER}{k}", pos=[sx * half[0], sy * half[1], half[2]],
                      size=[0.03], group=2)
        tendon = scene.add_tendon(name=f"{prefix}{SLING}{k}",
                                  limited=mujoco.mjtLimited.mjLIMITED_TRUE,
                                  range=[0.0, length], solref_limit=list(CONTACT_SOLREF),
                                  solimp_limit=list(CONTACT_SOLIMP), width=0.012,
                                  rgba=[0.15, 0.15, 0.15, 1.0])
        tendon.wrap_site(prefix + RING)
        tendon.wrap_site(f"{prefix}{LOAD_CORNER}{k}")


def _add_ground(scene: mujoco.MjSpec, index: int, prefix: str, offset: np.ndarray) -> None:
    nrow, ncol = ground_shape()
    half_x = (GROUND_X[1] - GROUND_X[0]) / 2
    half_y = (GROUND_Y[1] - GROUND_Y[0]) / 2
    scene.add_hfield(name=prefix + GROUND, nrow=nrow, ncol=ncol,
                     size=[half_x, half_y, GROUND_DEPTH + GROUND_TOP, GROUND_BASE],
                     userdata=[0.0] * (nrow * ncol))
    centre = offset + [(GROUND_X[0] + GROUND_X[1]) / 2, (GROUND_Y[0] + GROUND_Y[1]) / 2,
                       -GROUND_DEPTH]
    scene.worldbody.add_geom(name=prefix + GROUND, type=mujoco.mjtGeom.mjGEOM_HFIELD,
                             hfieldname=prefix + GROUND, pos=centre, contype=0,
                             conaffinity=1 << index, material=GROUND, friction=[0.8, 0.005, 1e-4])


def _share_look(scene: mujoco.MjSpec) -> None:
    def shared(name: str) -> str:
        return name.split("/", 1)[-1]

    for element in [*scene.textures, *scene.materials]:
        if element.name.startswith(machine_prefix(0)):
            element.name = shared(element.name)
        elif "/" in element.name:
            scene.delete(element)
    rgb = mujoco.mjtTextureRole.mjTEXROLE_RGB
    for material in scene.materials:
        material.textures[rgb] = shared(material.textures[rgb])
    for geom in scene.geoms:
        geom.material = shared(geom.material)


def load_scene(n_machines: int = 1, weld: bool = False) -> mujoco.MjModel:
    if not 1 <= n_machines <= MAX_MACHINES:
        raise ValueError(f"n_machines must be between 1 and {MAX_MACHINES}")
    scene = mujoco.MjSpec()
    scene.modelname = "excavator_scene"
    scene.meshdir = str(ASSETS)
    scene.option.timestep = TIMESTEP
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.option.tolerance = SOLVER_TOLERANCE
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = IMPRATIO
    for i in range(n_machines):
        prefix = machine_prefix(i)
        offset = np.array([0.0, i * MACHINE_SPACING, 0.0])
        frame = scene.worldbody.add_frame(pos=offset)
        scene.attach(_machine_spec(i), prefix=prefix, frame=frame)
        _add_ground(scene, i, prefix, offset)
        _add_load(scene, i, prefix, offset)
        scene.worldbody.add_body(name=prefix + ANCHOR, pos=offset, mocap=True)
        scene.add_equality(name=prefix + WELD, type=mujoco.mjtEq.mjEQ_WELD,
                           objtype=mujoco.mjtObj.mjOBJ_BODY, name1=prefix + "base",
                           name2=prefix + ANCHOR, data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                           active=int(weld))
    _share_look(scene)
    offsets = [i * MACHINE_SPACING for i in range(n_machines)]
    footprints = [(*GROUND_X, y + GROUND_Y[0], y + GROUND_Y[1]) for y in offsets]
    dress_after_heightfields(scene, footprints, [(0.0, y) for y in offsets])
    scene.add_key(name=HOME_KEY)
    model = scene.compile()
    flat = np.zeros(ground_shape())
    for i in range(n_machines):
        write_ground(model, i, flat)
        if weld:
            tracks = [g for g in range(model.ngeom)
                      if model.geom(g).name.startswith(machine_prefix(i) + TRACK_FEET)]
            model.geom_contype[tracks] = 0
    return model


def write_ground(model: mujoco.MjModel, index: int, heights: np.ndarray) -> None:
    heights = np.asarray(heights, float)
    if heights.shape != ground_shape():
        raise ValueError(f"heights must have shape {ground_shape()} (y, x), got {heights.shape}")
    if not np.all(np.isfinite(heights) & (heights >= -GROUND_DEPTH) & (heights <= GROUND_TOP)):
        raise ValueError(f"heights must lie within [{-GROUND_DEPTH}, {GROUND_TOP}] m")
    hf = model.hfield(machine_prefix(index) + GROUND)
    cells = slice(int(hf.adr[0]), int(hf.adr[0]) + heights.size)
    before = model.hfield_data[cells].copy()
    model.hfield_data[cells] = ((heights + GROUND_DEPTH) / (GROUND_DEPTH + GROUND_TOP)).ravel()
    try:
        rest_home(model, index)
    except ValueError:
        model.hfield_data[cells] = before
        raise


def ground_geoms(model: mujoco.MjModel, index: int) -> np.ndarray:
    bit = 1 << index
    return np.flatnonzero((model.geom_bodyid == 0) & (model.geom_contype == 0)
                          & (model.geom_conaffinity & bit != 0))


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


def load_geoms(model: mujoco.MjModel, index: int) -> np.ndarray:
    prefix = machine_prefix(index)
    bodies = [model.body(prefix + LOAD).id, model.body(prefix + LOAD_POINTS).id]
    return np.flatnonzero(np.isin(model.geom_bodyid, bodies))


def footprint(half: np.ndarray, step: float = GROUND_CELL / 2) -> np.ndarray:
    u, v = (np.linspace(-h, h, max(2, int(np.ceil(2 * h / step)) + 1)) for h in half[:2])
    return np.array([(a, b, -half[2]) for a in u for b in v])


def ground_overlap(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    prefix = machine_prefix(index)
    geom = model.geom(prefix + LOAD).id
    rot, centre = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]
    worst = -np.inf
    for point in footprint(model.geom_size[geom]) @ rot.T + centre:
        height = ground_height(model, data, index, point[0], point[1] - index * MACHINE_SPACING)
        if height is not None:
            worst = max(worst, height - point[2])
    return float(worst)


def placement_fault(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> str | None:
    mujoco.mj_forward(model, data)
    mine = set(load_geoms(model, index).tolist())
    ground = set(ground_geoms(model, index).tolist())
    for c in data.contact[:data.ncon]:
        pair = {c.geom1, c.geom2}
        if pair & mine and not pair & ground:
            return "the load would sit against the machine there"
    if ground_overlap(model, data, index) > 0:
        return "the ground there would pass between the load's contact spheres"
    return None


def support_plane(xy: np.ndarray, heights: np.ndarray) -> np.ndarray:
    best, lowest = None, np.inf
    for trio in itertools.combinations(range(len(xy)), 3):
        rows = np.c_[np.ones(3), xy[list(trio)]]
        if abs(np.linalg.det(rows)) < 1e-9:
            continue
        plane = np.linalg.solve(rows, heights[list(trio)])
        if np.all(np.c_[np.ones(len(xy)), xy] @ plane >= heights - 1e-9) and plane[0] < lowest:
            best, lowest = plane, plane[0]
    return best


def rest_pose(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float, y: float,
              yaw: float) -> np.ndarray:
    prefix = machine_prefix(index)
    half = model.geom_size[model.geom(prefix + LOAD).id].copy()
    q0 = int(model.joint(prefix + LOAD).qposadr[0])
    mujoco.mj_kinematics(model, data)
    c, s = np.cos(yaw), np.sin(yaw)
    feet = load_points(half)
    feet = feet[np.isclose(feet[:, 2], feet[:, 2].min())]
    xy = np.array([x, y]) + feet[:, :2] @ np.array([[c, s], [-s, c]])
    heights = np.array([ground_height(model, data, index, a, b) for a, b in xy])
    if None in heights:
        raise ValueError("the load must be placed on this machine's ground")
    com = model.body_ipos[model.body(prefix + LOAD).id][:2] @ np.array([[c, s], [-s, c]])
    slope = support_plane(xy - [x, y] - com, heights.astype(float))
    normal = np.array([-slope[1], -slope[2], 1.0]) / np.hypot(np.hypot(*slope[1:]), 1.0)
    level, tilt, quat = np.zeros(4), np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(4)
    mujoco.mju_axisAngle2Quat(level, [0.0, 0.0, 1.0], yaw)
    axis = np.cross([0.0, 0.0, 1.0], normal)
    if np.linalg.norm(axis) > 1e-12:
        mujoco.mju_axisAngle2Quat(tilt, axis / np.linalg.norm(axis), np.arccos(normal[2]))
    mujoco.mju_mulQuat(quat, tilt, level)
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, quat)
    offsets = feet @ rot.reshape(3, 3).T
    under = [ground_height(model, data, index, x + o[0], y + o[1]) for o in offsets]
    if None in under:
        raise ValueError("the load must be placed on this machine's ground")
    gap = CONTACT_MARGIN - REST_DEPTH
    lift = (LOAD_POINT_RADIUS + gap) / normal[2]
    z = max(h + lift - o[2] for h, o in zip(under, offsets, strict=True))
    data.qpos[q0:q0 + 7] = [x, y + index * MACHINE_SPACING, z, *quat]
    mine = set(load_geoms(model, index).tolist())
    ground = set(ground_geoms(model, index).tolist())
    for _ in range(8):
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        dists = [c.dist for c in data.contact[:data.ncon]
                 if {c.geom1, c.geom2} & mine and {c.geom1, c.geom2} & ground]
        short = gap - min(dists, default=gap)
        if short < 1e-9:
            break
        data.qpos[q0 + 2] += short / normal[2]
    return data.qpos[q0:q0 + 7].copy()


def ring_at_load(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> np.ndarray:
    prefix = machine_prefix(index)
    mujoco.mj_kinematics(model, data)
    top = data.site_xpos[model.site(prefix + LOAD_TOP).id]
    return top - model.body_pos[model.body(prefix + RING).id]


def rest_home(model: mujoco.MjModel, index: int) -> None:
    prefix = machine_prefix(index)
    scratch = mujoco.MjData(model)
    key = model.key(HOME_KEY).id
    scratch.qpos[:] = model.key_qpos[key]
    q0 = int(model.joint(prefix + LOAD).qposadr[0])
    r0 = int(model.joint(prefix + RING + "_x").qposadr[0])
    load = rest_pose(model, scratch, index, *LOAD_HOME, 0.0)
    fault = placement_fault(model, scratch, index)
    if fault is not None:
        raise ValueError(f"no rest for the load at its home position: {fault}")
    scratch.qpos[r0:r0 + 3] = 0.0
    model.body_pos[model.body(prefix + RING).id] += ring_at_load(model, scratch, index)
    for qpos in (model.key_qpos[key], model.qpos0):
        qpos[q0:q0 + 7] = load
        qpos[r0:r0 + 3] = 0.0


def ground_axes() -> tuple[np.ndarray, np.ndarray]:
    nrow, ncol = ground_shape()
    return np.linspace(*GROUND_X, ncol), np.linspace(*GROUND_Y, nrow)


def set_weld(model: mujoco.MjModel, data: mujoco.MjData, index: int, on: bool) -> None:
    prefix = machine_prefix(index)
    base = model.body(prefix + "base").id
    if on:
        mocap = model.body_mocapid[model.body(prefix + ANCHOR).id]
        mujoco.mj_kinematics(model, data)
        data.mocap_pos[mocap] = data.xpos[base]
        data.mocap_quat[mocap] = data.xquat[base]
        data.qvel[_base_dofs(model, index)] = 0.0
    else:
        for geom in range(model.ngeom):
            if model.geom(geom).name.startswith(prefix + TRACK_FEET):
                model.geom_contype[geom] = 1 << index
        model.body_contype[base] |= 1 << index
    data.eq_active[model.equality(prefix + WELD).id] = int(on)


def _base_dofs(model: mujoco.MjModel, index: int) -> slice:
    adr = int(model.joint(machine_prefix(index) + BASE_JOINT).dofadr[0])
    return slice(adr, adr + 6)
