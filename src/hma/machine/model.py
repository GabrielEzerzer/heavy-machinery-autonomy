"""Scenes of 12 t class excavators, each on its own ground of primitive geoms; names start with
machine_prefix(i). ctrl is a joint speed in rad/s or a track speed in m/s, sent through a "_valve"
joint: read joints by name. Step with hma.machine.tracks.step, which limits travel power."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from hma.machine import mass, tracks
from hma.machine.mass import body_solids, combine, full_inertia, point
from hma.machine.scenery import dress_after_heightfields

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
FORCE_CAPS = {"swing": 34.0e3, "boom": 330.0e3, "arm": 244.0e3, "bucket": 122.0e3}
SPEED_LIMITS = {"swing": 1.0, "boom": 0.6, "arm": 0.8, "bucket": 1.2}
VALVE = "_valve"
VALVE_INERTIA = 1.0e8
VALVE_GAIN = 5.0e9
VALVE_SOLIMP = (0.999, 0.9999, 0.001)

GROUND_X = (-3.0, 10.0)
GROUND_Y = (-6.0, 6.0)
GROUND_BOXES = 8
GROUND_BUMPS = 6
GROUND_BOX = "ground_box_"
GROUND_BUMP = "ground_bump_"
PARKED = -50.0
MACHINE_SPACING = 14.0
MAX_MACHINES = 31

TIMESTEP = 0.005
SOLVER_TOLERANCE = 1e-14


def machine_prefix(index: int) -> str:
    return f"m{index}/"


def ground_geoms(model: mujoco.MjModel, index: int) -> tuple[int, np.ndarray, np.ndarray]:
    prefix = machine_prefix(index)
    plane = model.geom(prefix + GROUND).id
    boxes = np.array([model.geom(f"{prefix}{GROUND_BOX}{k}").id for k in range(GROUND_BOXES)])
    bumps = np.array([model.geom(f"{prefix}{GROUND_BUMP}{k}").id for k in range(GROUND_BUMPS)])
    return plane, boxes, bumps


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
           solimp: tuple[float, float, float] = VALVE_SOLIMP, coef: float = 1.0) -> None:
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


def _machine_spec(index: int, weld: bool) -> mujoco.MjSpec:
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
        joint.solimp_limit = [*VALVE_SOLIMP, 0.5, 2.0]
        _valve(spec, name, (name,), FORCE_CAPS[name], SPEED_LIMITS[name])
    for side in tracks.SIDES:
        _add_track(spec, side)
    return spec


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
                                     contype=0, conaffinity=1 << index, material=GROUND,
                                     friction=friction)


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
    scene.option.impratio = tracks.IMPRATIO
    scene.option.jacobian = mujoco.mjtJacobian.mjJAC_SPARSE
    scene.nuserdata = 3 * n_machines
    for i in range(n_machines):
        prefix = machine_prefix(i)
        offset = np.array([0.0, i * MACHINE_SPACING, 0.0])
        frame = scene.worldbody.add_frame(pos=offset)
        scene.attach(_machine_spec(i, weld), prefix=prefix, frame=frame)
        _add_ground(scene, i, prefix, offset)
        if not weld:
            _add_track_pairs(scene, prefix)
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
    for i in range(n_machines):
        plane, boxes, bumps = ground_geoms(model, i)
        model.geom_sameframe[[plane, *boxes, *bumps]] = mujoco.mjtSameFrame.mjSAMEFRAME_NONE
    return model


def write_friction(model: mujoco.MjModel, index: int, mu: float) -> None:
    if not (np.isfinite(mu) and mu > 0):
        raise ValueError("the friction coefficient must be finite and positive")
    prefix = machine_prefix(index)
    plane, boxes, bumps = ground_geoms(model, index)
    model.geom_friction[[plane, *boxes, *bumps], 0] = mu
    names = [tracks.wheel_name(s, k) for s in tracks.SIDES for k in range(tracks.WHEELS)]
    names += [n for s in tracks.SIDES for n in tracks.shell_names(s)]
    parts = [model.geom(prefix + n).id for n in names]
    pairs = np.isin(model.pair_geom1, parts) | np.isin(model.pair_geom2, parts)
    model.pair_friction[pairs, :2] = mu


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
