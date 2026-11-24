"""Scenes with one or more 12 t class excavators, each standing on its own heightfield ground.
Every body, joint, actuator, site and ground name is prefixed per machine by machine_prefix(i)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
MACHINE_XML = ASSETS / "excavator.xml"

JOINT_NAMES = ("swing", "boom", "arm", "bucket")
BODY_NAMES = ("base", "upper", "boom", "arm", "bucket")
BASE_JOINT = "base_free"
TIP_SITE = "bucket_tip"
TRACK_GEOMS = ("track_left", "track_right")
WELD = "base_weld"
GROUND = "ground"
HOME_KEY = "home"
HOME_POSE = (0.0, 0.0, 1.6, 0.4)

LINK_MASSES = {"base": 4500.0, "upper": 5580.0, "boom": 1220.0, "arm": 600.0, "bucket": 400.0}
COUNTERWEIGHT_MASS = 2450.0
FORCE_CAPS = {"swing": 34.0e3, "boom": 330.0e3, "arm": 244.0e3, "bucket": 122.0e3}
SERVO_STIFFNESS = {"swing": 5.0e7, "boom": 2.0e8, "arm": 5.0e7, "bucket": 1.0e7}

GROUND_CELL = 0.1
GROUND_X = (-3.0, 9.0)
GROUND_Y = (-6.0, 6.0)
GROUND_DEPTH = 3.0
GROUND_TOP = 2.0
GROUND_BASE = 0.5
MACHINE_SPACING = 14.0
MAX_MACHINES = 31

TIMESTEP = 0.002
_FREE_HOME = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)


def machine_prefix(index: int) -> str:
    return f"m{index}/"


def ground_shape() -> tuple[int, int]:
    nrow = round((GROUND_Y[1] - GROUND_Y[0]) / GROUND_CELL) + 1
    ncol = round((GROUND_X[1] - GROUND_X[0]) / GROUND_CELL) + 1
    return nrow, ncol


def _machine_spec(index: int) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(MACHINE_XML))
    for body, mass in LINK_MASSES.items():
        mesh_geom = next(g for g in spec.body(body).geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH)
        mesh_geom.mass = mass - (COUNTERWEIGHT_MASS if body == "upper" else 0.0)
    spec.geom("counterweight").mass = COUNTERWEIGHT_MASS
    for name in TRACK_GEOMS:
        spec.geom(name).contype = 1 << index
        spec.geom(name).conaffinity = 0
    for name in JOINT_NAMES:
        joint = spec.joint(name)
        joint.actfrclimited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.actfrcrange = [-FORCE_CAPS[name], FORCE_CAPS[name]]
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_intvelocity(kp=SERVO_STIFFNESS[name], dampratio=1.0,
                               inheritrange=name != "swing")
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
                             conaffinity=1 << index, material="soil", friction=[0.8, 0.005, 1e-4])


def load_scene(n_machines: int = 1, weld: bool = False) -> mujoco.MjModel:
    if not 1 <= n_machines <= MAX_MACHINES:
        raise ValueError(f"n_machines must be between 1 and {MAX_MACHINES}")
    scene = mujoco.MjSpec()
    scene.modelname = "excavator_scene"
    scene.meshdir = str(ASSETS)
    scene.option.timestep = TIMESTEP
    scene.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    scene.visual.global_.offwidth = 1280
    scene.visual.global_.offheight = 720
    scene.visual.quality.shadowsize = 4096
    scene.add_texture(name="soil", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=64, height=64,
                      rgb1=[0.55, 0.45, 0.33], rgb2=[0.5, 0.41, 0.3])
    soil = scene.add_material(name="soil", texrepeat=[12, 12])
    soil.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "soil"
    scene.worldbody.add_light(pos=[0, 0, 30], dir=[0.3, 0.2, -1],
                              type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, castshadow=1)
    for i in range(n_machines):
        prefix = machine_prefix(i)
        offset = np.array([0.0, i * MACHINE_SPACING, 0.0])
        frame = scene.worldbody.add_frame(pos=offset)
        scene.attach(_machine_spec(i), prefix=prefix, frame=frame)
        _add_ground(scene, i, prefix, offset)
        scene.add_equality(name=prefix + WELD, type=mujoco.mjtEq.mjEQ_WELD,
                           objtype=mujoco.mjtObj.mjOBJ_BODY, name1=prefix + "base",
                           data=[*offset, 0, 0, 0, 1, 0, 0, 0, 1, 0], active=int(weld))
    home_qpos = [[i * MACHINE_SPACING if k == 1 else v for k, v in enumerate(_FREE_HOME)]
                 + list(HOME_POSE) for i in range(n_machines)]
    scene.add_key(name=HOME_KEY, qpos=np.concatenate(home_qpos),
                  act=np.tile(HOME_POSE, n_machines))
    model = scene.compile()
    flat = np.zeros(ground_shape())
    for i in range(n_machines):
        write_ground(model, i, flat)
    return model


def write_ground(model: mujoco.MjModel, index: int, heights: np.ndarray) -> None:
    heights = np.asarray(heights, float)
    if heights.shape != ground_shape():
        raise ValueError(f"heights must have shape {ground_shape()}, got {heights.shape}")
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
    eq = model.equality(machine_prefix(index) + WELD).id
    if on:
        base = model.body(machine_prefix(index) + "base").id
        mujoco.mj_kinematics(model, data)
        model.eq_data[eq, 0:3] = data.xpos[base]
        model.eq_data[eq, 3:6] = 0.0
        mujoco.mju_negQuat(model.eq_data[eq, 6:10], data.xquat[base])
        data.qvel[_base_dofs(model, index)] = 0.0
    data.eq_active[eq] = int(on)


def _base_dofs(model: mujoco.MjModel, index: int) -> slice:
    adr = int(model.joint(machine_prefix(index) + BASE_JOINT).dofadr[0])
    return slice(adr, adr + 6)
