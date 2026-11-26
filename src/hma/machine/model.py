"""Scenes of 12 t class excavators, each on its own ground of primitive geoms; names start with
machine_prefix(i). ctrl is the torque on each driven joint in N m, which hma.hydraulics computes
from the oil pressures, or a track speed in m/s sent through a "_valve" joint. Step with
hma.machine.tracks.step, which limits travel power."""

from __future__ import annotations

import itertools
from pathlib import Path

import mujoco
import numpy as np

from hma.machine import mass, tracks
from hma.machine.collision import add_collision_geoms
from hma.machine.mass import body_solids, combine, full_inertia, point
from hma.machine.scenery import GROUND_SHAPE, dress_after_heightfields, plain_look

ASSETS = Path(__file__).parent / "assets"
MACHINE_XML = ASSETS / "excavator.xml"

JOINT_NAMES = ("swing", "boom", "arm", "bucket")
BODY_NAMES = ("base", "upper", "boom", "arm", "bucket")
BASE_JOINT = "base_free"
TIP_SITE = "bucket_tip"
TRAVEL_NAMES = tuple(tracks.TRAVEL + side for side in tracks.SIDES)
ANCHOR = "anchor"
WELD = "base_weld"
GROUND = "ground"
HOME_KEY = "home"
HOME_POSE = (0.0, -0.6, 1.9, 0.6)

COUNTERWEIGHT_MASS = mass.COUNTERWEIGHT_MASS
END_STOP_SOLIMP = (0.999, 0.9999, 0.001)
VALVE = "_valve"
VALVE_INERTIA = 1.0e8
VALVE_GAIN = 5.0e9

GROUND_HALF = 13.5
GROUND_X = (-GROUND_HALF, GROUND_HALF)
GROUND_Y = (-GROUND_HALF, GROUND_HALF)
GROUND_BOXES = 8
GROUND_BUMPS = 6
GROUND_BOX = "ground_box_"
GROUND_BUMP = "ground_bump_"
PARKED = -50.0
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
FOOTPRINT_STEP = 0.05


def machine_prefix(index: int) -> str:
    return f"m{index}/"


def wheel_points() -> list[tuple[float, tuple[float, float, float]]]:
    return [(m, (x, tracks.side_sign(side) * mass.GAUGE / 2, tracks.WHEEL_RADIUS))
            for side in tracks.SIDES
            for m, x in zip(tracks.WHEEL_MASSES, tracks.WHEEL_X, strict=True)]


def link_solids() -> dict[str, mass.Solid]:
    solids = dict(body_solids())
    solids["base"] = combine([solids["base"], *(point(-m, pos) for m, pos in wheel_points())])
    return solids


LINK_MASSES = {name: solid.mass for name, solid in link_solids().items()}


def _turn_reference_to_home(joint: mujoco.MjsJoint, angle: float) -> None:
    body = joint.parent
    turn = np.zeros(4)
    mujoco.mju_axisAngle2Quat(turn, np.asarray(joint.axis, float), angle)
    quat = np.zeros(4)
    mujoco.mju_mulQuat(quat, np.asarray(body.quat, float), turn)
    body.quat = quat
    joint.ref = angle


def _valve(spec: mujoco.MjSpec, name: str, joints: tuple[str, ...], cap: float, limit: float,
           solimp: tuple[float, float, float], coef: float) -> None:
    valve = spec.worldbody.add_body(name=name + VALVE, mass=1.0, inertia=[1.0, 1.0, 1.0],
                                    explicitinertial=True)
    valve.add_joint(name=name + VALVE, axis=[0.0, 0.0, 1.0], armature=VALVE_INERTIA)
    tendon = spec.add_tendon(name=name + VALVE, frictionloss=cap,
                             solref_friction=[2 * TIMESTEP, 1.0],
                             solimp_friction=[*solimp, 0.5, 2.0])
    for joint in joints:
        tendon.wrap_joint(joint, coef)
    tendon.wrap_joint(name + VALVE, -1.0)
    act = spec.add_actuator(name=name, target=name + VALVE, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_velocity(kv=VALVE_GAIN)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [-limit, limit]


def _add_track(spec: mujoco.MjSpec, side: str) -> None:
    y = tracks.side_sign(side) * mass.GAUGE / 2
    across = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
    drive = tracks.wheel_name(side, 0)
    for k, x in enumerate(tracks.WHEEL_X):
        name = tracks.wheel_name(side, k)
        wheel = spec.body("base").add_body(name=name, pos=[x, y, tracks.WHEEL_RADIUS],
                                           mass=tracks.WHEEL_MASSES[k], ipos=[0.0, 0.0, 0.0],
                                           inertia=[tracks.WHEEL_INERTIA] * 3,
                                           explicitinertial=True)
        wheel.add_joint(name=name, axis=[0.0, 1.0, 0.0],
                        armature=tracks.BELT_ARMATURE / tracks.WHEELS)
        wheel.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, quat=across,
                       size=[tracks.WHEEL_RADIUS, tracks.RAIL], contype=0, conaffinity=0,
                       group=3, rgba=[0.2, 0.6, 0.9, 0.4])
        if k:
            spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                              name1=name, name2=drive, data=[0.0, 1.0] + [0.0] * 10,
                              solref=list(tracks.COUPLING_SOLREF),
                              solimp=list(tracks.COUPLING_SOLIMP))
    base = spec.body("base")
    frame, rear, front, support = tracks.shell_names(side)
    lift, top, half_shoe = tracks.SHELL_LIFT, tracks.SHELL_TOP, mass.SHOE_WIDTH / 2
    shell = dict(contype=0, conaffinity=0, group=3, mass=0.0, rgba=[0.9, 0.6, 0.2, 0.3])
    base.add_geom(name=frame, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, y, (lift + top) / 2],
                  size=[mass.TUMBLER, half_shoe, (top - lift) / 2], **shell)
    for name, x in ((rear, -mass.TUMBLER), (front, mass.TUMBLER)):
        base.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CYLINDER, quat=across,
                      pos=[x, y, tracks.WHEEL_RADIUS + lift],
                      size=[tracks.WHEEL_RADIUS, half_shoe], **shell)
    base.add_geom(name=support, type=mujoco.mjtGeom.mjGEOM_BOX,
                  pos=[0.0, y, tracks.SUPPORT_GAP + tracks.SUPPORT_HALF_HEIGHT],
                  size=[mass.TUMBLER, tracks.RAIL, tracks.SUPPORT_HALF_HEIGHT], **shell)
    wheels = tuple(tracks.wheel_name(side, k) for k in range(tracks.WHEELS))
    _valve(spec, tracks.TRAVEL + side, wheels, tracks.TRACTION, tracks.TRAVEL_SPEED,
           tracks.BRAKE_SOLIMP, coef=tracks.WHEEL_RADIUS / tracks.WHEELS)


def _machine_spec(index: int) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MACHINE_XML))
    for name, solid in link_solids().items():
        body = spec.body(name)
        body.explicitinertial = True
        body.mass = solid.mass
        body.ipos = solid.com
        body.fullinertia = full_inertia(solid)
    for name, home in zip(JOINT_NAMES, HOME_POSE, strict=True):
        joint = spec.joint(name)
        _turn_reference_to_home(joint, home)
        joint.solref_limit = [2 * TIMESTEP, 1.0]
        joint.solimp_limit = [*END_STOP_SOLIMP, 0.5, 2.0]
        joint.solref_friction = [2 * TIMESTEP, 1.0]
        joint.solimp_friction = [*END_STOP_SOLIMP, 0.5, 2.0]
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_motor()
    for side in tracks.SIDES:
        _add_track(spec, side)
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
    half = [(GROUND_X[1] - GROUND_X[0]) / 2, (GROUND_Y[1] - GROUND_Y[0]) / 2]
    centre = offset + [(GROUND_X[0] + GROUND_X[1]) / 2, (GROUND_Y[0] + GROUND_Y[1]) / 2, 0.0]
    friction = [tracks.FRICTION, 0.005, 1e-4]
    scene.worldbody.add_geom(name=prefix + GROUND, type=mujoco.mjtGeom.mjGEOM_PLANE,
                             pos=centre, size=[*half, 1.0], contype=0, conaffinity=1 << index,
                             material=GROUND, friction=friction)
    parked = offset + [0.0, 0.0, PARKED]
    shapes = [(GROUND_BOX, GROUND_BOXES, mujoco.mjtGeom.mjGEOM_BOX, [0.1] * 3),
              (GROUND_BUMP, GROUND_BUMPS, mujoco.mjtGeom.mjGEOM_SPHERE, [0.1, 0.0, 0.0])]
    for name, count, kind, size in shapes:
        for k in range(count):
            scene.worldbody.add_geom(name=f"{prefix}{name}{k}", type=kind, pos=parked, size=size,
                                     contype=0, conaffinity=1 << index,
                                     material=GROUND_SHAPE, friction=friction)


def _add_track_pairs(scene: mujoco.MjSpec, prefix: str) -> None:
    grounds = [GROUND] + [f"{GROUND_BOX}{k}" for k in range(GROUND_BOXES)]
    grounds += [f"{GROUND_BUMP}{k}" for k in range(GROUND_BUMPS)]
    mu = tracks.FRICTION
    for side in tracks.SIDES:
        shells = tracks.shell_names(side)
        for part in tracks.contact_names(side):
            for ground in grounds:
                scene.add_pair(geomname1=prefix + part,
                               geomname2=prefix + ground, condim=1 if part in shells else 3,
                               friction=[mu, mu, 0.005, 1e-4, 1e-4],
                               solreffriction=[-0.0, -tracks.FRICTION_REFERENCE],
                               solimp=list(tracks.CONTACT_SOLIMP))


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
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_EULER
    scene.option.tolerance = SOLVER_TOLERANCE
    scene.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    scene.option.impratio = IMPRATIO
    scene.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    scene.nuserdata = 3 * n_machines
    for i in range(n_machines):
        prefix = machine_prefix(i)
        offset = np.array([0.0, i * MACHINE_SPACING, 0.0])
        frame = scene.worldbody.add_frame(pos=offset)
        scene.attach(_machine_spec(i), prefix=prefix, frame=frame)
        _add_ground(scene, i, prefix, offset)
        if not weld:
            _add_track_pairs(scene, prefix)
        _add_load(scene, i, prefix, offset)
        scene.worldbody.add_body(name=prefix + ANCHOR, pos=offset, mocap=True)
        scene.add_equality(name=prefix + WELD, type=mujoco.mjtEq.mjEQ_WELD,
                           objtype=mujoco.mjtObj.mjOBJ_BODY, name1=prefix + "base",
                           name2=prefix + ANCHOR, data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                           active=int(weld))
    plain_look(scene)
    _share_look(scene)
    offsets = [i * MACHINE_SPACING for i in range(n_machines)]
    footprints = [(*GROUND_X, y + GROUND_Y[0], y + GROUND_Y[1]) for y in offsets]
    dress_after_heightfields(scene, footprints, [(0.0, y) for y in offsets])
    scene.add_key(name=HOME_KEY)
    model = scene.compile()
    for i in range(n_machines):
        rest_home(model, i)
    return model


def write_friction(model: mujoco.MjModel, index: int, mu: float) -> None:
    if not (np.isfinite(mu) and mu > 0):
        raise ValueError("the friction coefficient must be finite and positive")
    prefix = machine_prefix(index)
    model.geom_friction[ground_geoms(model, index), 0] = mu
    names = [tracks.wheel_name(s, k) for s in tracks.SIDES for k in range(tracks.WHEELS)]
    names += [n for s in tracks.SIDES for n in tracks.shell_names(s)]
    parts = [model.geom(prefix + n).id for n in names]
    pairs = np.isin(model.pair_geom1, parts) | np.isin(model.pair_geom2, parts)
    model.pair_friction[pairs, :2] = mu


def ground_parts(model: mujoco.MjModel, index: int) -> tuple[int, np.ndarray, np.ndarray]:
    prefix = machine_prefix(index)
    plane = model.geom(prefix + GROUND).id
    boxes = np.array([model.geom(f"{prefix}{GROUND_BOX}{k}").id for k in range(GROUND_BOXES)])
    bumps = np.array([model.geom(f"{prefix}{GROUND_BUMP}{k}").id for k in range(GROUND_BUMPS)])
    return plane, boxes, bumps


def ground_geoms(model: mujoco.MjModel, index: int) -> np.ndarray:
    plane, boxes, bumps = ground_parts(model, index)
    return np.array([plane, *boxes, *bumps])


def ground_height(model: mujoco.MjModel, index: int, points: np.ndarray) -> np.ndarray:
    plane, boxes, bumps = ground_parts(model, index)
    offset = model.body_pos[model.body(machine_prefix(index) + ANCHOR).id]
    points = np.atleast_2d(np.asarray(points, float))
    rel = points[:, :2] - offset[:2]
    normal = np.zeros(3)
    mujoco.mju_rotVecQuat(normal, [0.0, 0.0, 1.0], model.geom_quat[plane])
    height = model.geom_pos[plane][2] - (
        normal[0] * (points[:, 0] - model.geom_pos[plane][0])
        + normal[1] * (points[:, 1] - model.geom_pos[plane][1])) / normal[2]
    for geom in (*boxes, *bumps):
        pos = model.geom_pos[geom]
        if pos[2] < offset[2] + PARKED / 2:
            continue
        mat = np.zeros(9)
        mujoco.mju_quat2Mat(mat, model.geom_quat[geom])
        for k, (x, y) in enumerate(points[:, :2]):
            d = mujoco.mju_rayGeom(pos, mat, model.geom_size[geom], np.array([x, y, RAY_TOP]),
                                   DOWN, int(model.geom_type[geom]))
            if d >= 0:
                height[k] = max(height[k], RAY_TOP - d)
    off = ((rel[:, 0] < GROUND_X[0]) | (rel[:, 0] > GROUND_X[1])
           | (rel[:, 1] < GROUND_Y[0]) | (rel[:, 1] > GROUND_Y[1]))
    height[off] = np.nan
    return height


def _height(model: mujoco.MjModel, index: int, x: float, y: float) -> float:
    return float(ground_height(model, index, [[x, y + index * MACHINE_SPACING]])[0])


def load_geoms(model: mujoco.MjModel, index: int) -> np.ndarray:
    prefix = machine_prefix(index)
    bodies = [model.body(prefix + LOAD).id, model.body(prefix + LOAD_POINTS).id]
    return np.flatnonzero(np.isin(model.geom_bodyid, bodies))


def footprint(half: np.ndarray, step: float = FOOTPRINT_STEP) -> np.ndarray:
    u, v = (np.linspace(-h, h, max(2, int(np.ceil(2 * h / step)) + 1)) for h in half[:2])
    return np.array([(a, b, -half[2]) for a in u for b in v])


def ground_overlap(model: mujoco.MjModel, data: mujoco.MjData, index: int) -> float:
    prefix = machine_prefix(index)
    geom = model.geom(prefix + LOAD).id
    rot, centre = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom]
    points = footprint(model.geom_size[geom]) @ rot.T + centre
    gaps = ground_height(model, index, points) - points[:, 2]
    return float(np.nanmax(gaps, initial=-np.inf))


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
    heights = np.array([_height(model, index, a, b) for a, b in xy])
    if np.isnan(heights).any():
        raise ValueError("the load must be placed on this machine's ground")
    com = model.body_ipos[model.body(prefix + LOAD).id][:2] @ np.array([[c, s], [-s, c]])
    slope = support_plane(xy - [x, y] - com, heights)
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
    under = [_height(model, index, x + o[0], y + o[1]) for o in offsets]
    if np.isnan(under).any():
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


def set_weld(model: mujoco.MjModel, data: mujoco.MjData, index: int, on: bool) -> None:
    prefix = machine_prefix(index)
    wheel = model.geom(prefix + tracks.wheel_name(tracks.SIDES[0], 0)).id
    if not on and not np.isin(wheel, [model.pair_geom1, model.pair_geom2]).any():
        raise ValueError("a scene loaded welded has no ground contact; load it released")
    base = model.body(prefix + "base").id
    if on:
        mocap = model.body_mocapid[model.body(prefix + ANCHOR).id]
        mujoco.mj_kinematics(model, data)
        data.mocap_pos[mocap] = data.xpos[base]
        data.mocap_quat[mocap] = data.xquat[base]
        data.qvel[_base_dofs(model, index)] = 0.0
    data.eq_active[model.equality(prefix + WELD).id] = int(on)


def _base_dofs(model: mujoco.MjModel, index: int) -> slice:
    adr = int(model.joint(machine_prefix(index) + BASE_JOINT).dofadr[0])
    return slice(adr, adr + 6)
