"""Scenes of 12 t class excavators, each on its own ground of (y, x) heights; names start with
machine_prefix(i). ctrl is the torque on each driven joint in N m, which hma.hydraulics computes
from the oil pressures; set_weld(off) leaves track contacts on."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

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
END_STOP_SOLIMP = (0.999, 0.9999, 0.001)

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


def machine_prefix(index: int) -> str:
    return f"m{index}/"


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


def _machine_spec(index: int, weld: bool) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MACHINE_XML))
    for body, mass in LINK_MASSES.items():
        mesh_geom = next(g for g in spec.body(body).geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH)
        mesh_geom.mass = mass - (COUNTERWEIGHT_MASS if body == "upper" else 0.0)
    spec.geom("counterweight").mass = COUNTERWEIGHT_MASS
    for geom in spec.body("base").geoms:
        if geom.name.startswith(TRACK_FEET):
            geom.contype = 0 if weld else 1 << index
            geom.conaffinity = 0
    for name, home in zip(JOINT_NAMES, HOME_POSE, strict=True):
        joint = spec.joint(name)
        _turn_reference_to_home(joint, home)
        joint.solref_limit = [2 * TIMESTEP, 1.0]
        joint.solimp_limit = [*END_STOP_SOLIMP, 0.5, 2.0]
        joint.solref_friction = [2 * TIMESTEP, 1.0]
        joint.solimp_friction = [*END_STOP_SOLIMP, 0.5, 2.0]
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_motor()
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
    _share_look(scene)
    offsets = [i * MACHINE_SPACING for i in range(n_machines)]
    footprints = [(*GROUND_X, y + GROUND_Y[0], y + GROUND_Y[1]) for y in offsets]
    dress_after_heightfields(scene, footprints, [(0.0, y) for y in offsets])
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
