"""Offscreen pictures of excavator scenes. The hydraulic cylinders and bucket linkage exist only as
meshes: draw_hydraulics() poses them from the joint state into an mjvScene at draw time, so they
add nothing to the physics and cost nothing when no frame is drawn."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import mujoco
import numpy as np

from hma.machine.cylinders import PINS, coupler
from hma.machine.model import machine_prefix

WIDTH, HEIGHT = 1280, 720
CYLINDERS = {"boom": ("boom_foot", "boom_rod"), "arm": ("arm_head", "arm_rod"),
             "bucket": ("bucket_head", "coupler")}


def _parts() -> list[tuple[str, str, str, str, str]]:
    out = []
    for name, (base, rod) in CYLINDERS.items():
        out.append((f"{name}_barrel", base, base, rod, "paint"))
        out.append((f"{name}_rod", rod, base, rod, "steel"))
    out.append(("side_link", "side_link", "side_link", "coupler", "paint"))
    out.append(("bucket_link", "link_pin", "link_pin", "coupler", "paint"))
    return out


PARTS = _parts()
ALONG_Z = {f"{name}_{piece}" for name in CYLINDERS for piece in ("barrel", "rod")}


def pin_positions(model: mujoco.MjModel, data: mujoco.MjData, prefix: str) -> dict[str, np.ndarray]:
    """World positions of the cylinder and linkage pins; the coupler follows the bucket angle."""
    out = {}
    for name, (body, local) in PINS.items():
        b = model.body(prefix + body).id
        out[name] = data.xpos[b] + data.xmat[b].reshape(3, 3) @ local
    arm = model.body(prefix + "arm").id
    x, z = coupler(data.qpos[model.joint(prefix + "bucket").qposadr[0]])[0]
    out["coupler"] = data.xpos[arm] + data.xmat[arm].reshape(3, 3) @ np.array([x, 0.0, z])
    return out


def machine_count(model: mujoco.MjModel) -> int:
    count = 0
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, machine_prefix(count) + "arm") >= 0:
        count += 1
    return count


def _frame(axis: np.ndarray, pin_axis: np.ndarray, along_z: bool) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    y = pin_axis - (pin_axis @ axis) * axis
    y /= np.linalg.norm(y)
    if along_z:
        return np.column_stack([np.cross(y, axis), y, axis])
    return np.column_stack([axis, y, np.cross(axis, y)])


def draw_hydraulics(model: mujoco.MjModel, data: mujoco.MjData, scene: mujoco.MjvScene) -> None:
    for index in range(machine_count(model)):
        prefix = machine_prefix(index)
        pins = pin_positions(model, data, prefix)
        pin_axis = data.xmat[model.body(prefix + "upper").id].reshape(3, 3)[:, 1]
        for mesh_name, anchor, start, end, material in PARTS:
            if scene.ngeom >= scene.maxgeom:
                return
            mesh = model.mesh(prefix + mesh_name).id
            mat = model.material(prefix + material).id
            frame = _frame(pins[end] - pins[start], pin_axis, mesh_name in ALONG_Z)
            offset = np.zeros(9)
            mujoco.mju_quat2Mat(offset, model.mesh_quat[mesh])
            pose = frame @ offset.reshape(3, 3)
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_MESH, np.zeros(3),
                                pins[anchor] + frame @ model.mesh_pos[mesh], pose.ravel(),
                                model.mat_rgba[mat].astype(np.float32))
            geom.dataid = 2 * mesh
            geom.matid = mat
            geom.texid = model.mat_texid[mat, mujoco.mjtTextureRole.mjTEXROLE_RGB]
            geom.texcoord = int(model.mesh_texcoordnum[mesh] > 0)
            geom.texrepeat = model.mat_texrepeat[mat]
            geom.texuniform = model.mat_texuniform[mat]
            geom.emission = model.mat_emission[mat]
            geom.specular = model.mat_specular[mat]
            geom.shininess = model.mat_shininess[mat]
            geom.reflectance = model.mat_reflectance[mat]
            geom.category = mujoco.mjtCatBit.mjCAT_DYNAMIC
            geom.castshadow = 1
            scene.ngeom += 1


def view_option() -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.sitegroup[:] = 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    return opt


def apply_look(scene: mujoco.MjvScene) -> None:
    scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = True
    scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = True


def default_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    count = machine_count(model)
    first = data.xpos[model.body(machine_prefix(0) + "base").id]
    last = data.xpos[model.body(machine_prefix(count - 1) + "base").id]
    cam.lookat[:] = (first + last) / 2 + [2.5, 0.0, 1.6]
    cam.distance = 13.0 + float(np.linalg.norm(last - first))
    cam.azimuth, cam.elevation = -135.0, -18.0
    return cam


def render_frame(model: mujoco.MjModel, data: mujoco.MjData,
                 camera: mujoco.MjvCamera | None = None,
                 renderer: mujoco.Renderer | None = None) -> np.ndarray:
    """An RGB frame (height, width, 3) of the scene as last computed into data. Pass a renderer
    made once with mujoco.Renderer(model, height, width) when drawing many frames."""
    own = renderer is None
    if own:
        renderer = mujoco.Renderer(model, HEIGHT, WIDTH)
    try:
        renderer.update_scene(data, camera or default_camera(model, data), view_option())
        apply_look(renderer.scene)
        draw_hydraulics(model, data, renderer.scene)
        return renderer.render()
    finally:
        if own:
            renderer.close()


def write_png(path: Path, rgb: np.ndarray) -> None:
    rgb = np.ascontiguousarray(rgb, np.uint8)
    height, width, channels = rgb.shape
    rows = np.concatenate([np.zeros((height, 1), np.uint8), rgb.reshape(height, -1)], axis=1)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        body = kind + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", width, height, 8, {3: 2, 4: 6}[channels], 0, 0, 0)
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
                           + chunk(b"IDAT", zlib.compress(rows.tobytes(), 9)) + chunk(b"IEND", b""))
