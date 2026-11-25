"""Scenes of 12 t class excavators, each on its own ground of (y, x) heights; names start with
machine_prefix(i). ctrl is a joint speed in rad/s or a track speed in m/s, sent through a "_valve"
joint: read joints by name. Step scenes with hma.machine.tracks.step, not mujoco.mj_step."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from hma.machine import mass, tracks
from hma.machine.mass import body_solids, full_inertia

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
LINK_MASSES = {name: solid.mass for name, solid in body_solids().items()}
FORCE_CAPS = {"swing": 34.0e3, "boom": 330.0e3, "arm": 244.0e3, "bucket": 122.0e3}
SPEED_LIMITS = {"swing": 1.0, "boom": 0.6, "arm": 0.8, "bucket": 1.2}
VALVE = "_valve"
VALVE_INERTIA = 1.0e8
VALVE_GAIN = 5.0e9
VALVE_SOLIMP = (0.999, 0.9999, 0.001)

GROUND_CELL = 0.1
GROUND_X = (-3.0, 10.0)
GROUND_Y = (-6.0, 6.0)
GROUND_DEPTH = 6.0
GROUND_TOP = 2.0
GROUND_BASE = 0.5
MACHINE_SPACING = 14.0
MAX_MACHINES = 31

TIMESTEP = 0.005
SOLVER_TOLERANCE = 1e-14
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


def _valve(spec: mujoco.MjSpec, name: str, joint: str, cap: float, limit: float,
           solimp: tuple[float, float, float] = VALVE_SOLIMP) -> None:
    valve = spec.worldbody.add_body(name=name + VALVE, mass=1.0, inertia=[1.0, 1.0, 1.0],
                                    explicitinertial=True)
    valve.add_joint(name=name + VALVE, axis=[0.0, 0.0, 1.0], armature=VALVE_INERTIA)
    tendon = spec.add_tendon(name=name + VALVE, frictionloss=cap,
                             solref_friction=[2 * TIMESTEP, 1.0],
                             solimp_friction=[*solimp, 0.5, 2.0])
    tendon.wrap_joint(joint, 1.0)
    tendon.wrap_joint(name + VALVE, -1.0)
    act = spec.add_actuator(name=name, target=name + VALVE, trntype=mujoco.mjtTrn.mjTRN_JOINT)
    act.set_to_velocity(kv=VALVE_GAIN)
    act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    act.ctrlrange = [-limit, limit]


def _add_belt(spec: mujoco.MjSpec, side: str) -> None:
    y = tracks.side_sign(side) * tracks.GAUGE / 2
    name = tracks.BELT + side
    belt = spec.body("base").add_body(name=name, mass=1e-3, inertia=[1e-6] * 3,
                                      explicitinertial=True)
    belt.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[-1.0, 0.0, 0.0],
                   armature=tracks.BELT_ARMATURE)
    half_height = tracks.IDLER_RADIUS
    belt.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=[0.0, y, half_height],
                  size=[tracks.TUMBLER + tracks.IDLER_RADIUS, tracks.SHOE_WIDTH / 2, half_height],
                  contype=0, conaffinity=0, group=3, rgba=[0.2, 0.6, 0.9, 0.3])
    _valve(spec, tracks.TRAVEL + side, name, tracks.TRACTION, tracks.TRAVEL_SPEED,
           tracks.BRAKE_SOLIMP)


def _machine_spec(index: int, weld: bool) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MACHINE_XML))
    for name, solid in body_solids().items():
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
        _valve(spec, name, name, FORCE_CAPS[name], SPEED_LIMITS[name])
    for side in tracks.SIDES:
        _add_belt(spec, side)
    return spec


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
                             conaffinity=1 << index, material=GROUND,
                             friction=[tracks.NOMINAL_FRICTION[0], 0.005, 1e-4])
    scene.add_numeric(name=prefix + tracks.FRICTION, size=2, data=list(tracks.NOMINAL_FRICTION))


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
    scene.option.impratio = tracks.IMPRATIO
    scene.visual.global_.offwidth = 1280
    scene.visual.global_.offheight = 720
    scene.visual.quality.shadowsize = 4096
    scene.add_texture(name=GROUND, type=mujoco.mjtTexture.mjTEXTURE_2D,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=64, height=64,
                      rgb1=[0.55, 0.45, 0.33], rgb2=[0.5, 0.41, 0.3])
    ground = scene.add_material(name=GROUND, texrepeat=[12, 12])
    ground.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = GROUND
    scene.worldbody.add_light(pos=[0, 0, 30], dir=[0.3, 0.2, -1],
                              type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=1)
    for i in range(n_machines):
        prefix = machine_prefix(i)
        offset = np.array([0.0, i * MACHINE_SPACING, 0.0])
        frame = scene.worldbody.add_frame(pos=offset)
        scene.attach(_machine_spec(i, weld), prefix=prefix, frame=frame)
        _add_ground(scene, i, prefix, offset)
        scene.worldbody.add_body(name=prefix + ANCHOR, pos=offset, mocap=True)
        scene.add_equality(name=prefix + WELD, type=mujoco.mjtEq.mjEQ_WELD,
                           objtype=mujoco.mjtObj.mjOBJ_BODY, name1=prefix + "base",
                           name2=prefix + ANCHOR, data=[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],
                           active=int(weld))
    scene.add_key(name=HOME_KEY)
    model = scene.compile()
    flat = np.zeros(ground_shape())
    for i in range(n_machines):
        write_ground(model, i, flat)
    return model


def write_ground(model: mujoco.MjModel, index: int, heights: np.ndarray) -> None:
    heights = np.asarray(heights, float)
    if heights.shape != ground_shape():
        raise ValueError(f"heights must have shape {ground_shape()} (y, x), got {heights.shape}")
    if not np.all(np.isfinite(heights) & (heights >= -GROUND_DEPTH) & (heights <= GROUND_TOP)):
        raise ValueError(f"heights must lie within [{-GROUND_DEPTH}, {GROUND_TOP}] m")
    hf = model.hfield(machine_prefix(index) + GROUND)
    start = int(hf.adr[0])
    model.hfield_data[start:start + heights.size] = (
        (heights + GROUND_DEPTH) / (GROUND_DEPTH + GROUND_TOP)
    ).ravel()


def write_friction(model: mujoco.MjModel, index: int, along: float, across: float) -> None:
    if not all(np.isfinite(v) and v > 0 for v in (along, across)):
        raise ValueError("friction coefficients must be finite and positive")
    adr = int(model.numeric(machine_prefix(index) + tracks.FRICTION).adr[0])
    model.numeric_data[adr:adr + 2] = (along, across)
    model.geom_friction[model.geom(machine_prefix(index) + GROUND).id, 0] = along


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
    data.eq_active[model.equality(prefix + WELD).id] = int(on)


def _base_dofs(model: mujoco.MjModel, index: int) -> slice:
    adr = int(model.joint(machine_prefix(index) + BASE_JOINT).dofadr[0])
    return slice(adr, adr + 6)
