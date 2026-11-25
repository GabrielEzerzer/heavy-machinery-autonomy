"""The look of an excavator scene: a plain grey machine, with its tracks and upper structure drawn
without fine detail, on a two-tone grey grid of 1 m squares under a pale sky and a sun with soft
shadows. Nothing here collides or has mass."""

from __future__ import annotations

from functools import cache

import mujoco
import numpy as np

from hma.machine import kinematics
from hma.machine.collision import LINK_MESH, link_triangles

SKY_TOP = (0.62, 0.66, 0.72)
SKY_HORIZON = (0.84, 0.85, 0.87)
GRID_LIGHT = (0.40, 0.41, 0.43)
GRID_DARK = (0.33, 0.34, 0.36)
GRID_SQUARE = 1.0
GRID_TILE = 2.0 * GRID_SQUARE
GRID_PIXELS = 256
MACHINE_GREY = (0.80, 0.81, 0.83, 1.0)
PAINT = "paint"
PLAIN_LINKS = ("base", "upper")
PLAIN_CELL = 0.05
PLAIN = "_plain"
DRAWN_GROUP, HIDDEN_GROUP = 1, 4
SITE_STEPS = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0)
SITE_DROP = 0.0
SUN_DIR = (-0.45, 0.3, -0.85)
SUN_BACKOFF = 20.0
SHADOW_MARGIN = 0.5
SHOULDER = np.array([kinematics.BOOM_PIVOT[0], 0.0,
                     kinematics.SWING_HEIGHT + kinematics.BOOM_PIVOT[2]])
REACH = kinematics.BOOM_LENGTH + kinematics.ARM_LENGTH + kinematics.BUCKET_LENGTH + SHADOW_MARGIN
NEAR, FAR = 0.1, 3000.0
FOG_START, FOG_END = 60.0, 900.0


@cache
def plain_mesh(link: str) -> tuple[tuple[float, ...], tuple[int, ...]]:
    points = link_triangles(link).reshape(-1, 3)
    _, cell, count = np.unique(np.floor(points / PLAIN_CELL).astype(np.int64), axis=0,
                               return_inverse=True, return_counts=True)
    cell = cell.ravel()
    verts = np.zeros((len(count), 3))
    np.add.at(verts, cell, points)
    faces = cell.reshape(-1, 3)
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2])
                  & (faces[:, 0] != faces[:, 2])]
    _, first = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return (tuple((verts / count[:, None]).ravel().tolist()),
            tuple(faces[np.sort(first)].ravel().tolist()))


def plain_look(scene: mujoco.MjSpec) -> None:
    rgb = mujoco.mjtTextureRole.mjTEXROLE_RGB
    used = set()
    for material in scene.materials:
        if material.name.split("/")[-1] == PAINT:
            used.add(material.textures[rgb])
            material.textures[rgb] = ""
            material.rgba = list(MACHINE_GREY)
    for texture in scene.textures:
        if texture.name in used:
            scene.delete(texture)
    detailed = {mesh: link for link, mesh in LINK_MESH.items() if link in PLAIN_LINKS}
    for geom in list(scene.geoms):
        link = detailed.get(geom.meshname.split("/")[-1])
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH or link is None:
            continue
        verts, faces = plain_mesh(link)
        scene.add_mesh(name=geom.meshname + PLAIN, uservert=list(verts), userface=list(faces),
                       inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
        geom.parent.add_geom(name=geom.name + PLAIN, type=mujoco.mjtGeom.mjGEOM_MESH,
                             meshname=geom.meshname + PLAIN, material=geom.material,
                             pos=geom.pos, quat=geom.quat, contype=0, conaffinity=0, mass=0.0,
                             group=DRAWN_GROUP)
        geom.group = HIDDEN_GROUP


def _add_grid(scene: mujoco.MjSpec, ground_repeat: tuple[float, float]) -> None:
    scene.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=GRID_PIXELS,
                      height=GRID_PIXELS, rgb1=list(GRID_LIGHT), rgb2=list(GRID_DARK))
    for name, repeat in (("ground", ground_repeat), ("site_ground", (1.0, 1.0))):
        mat = scene.add_material(name=name, texrepeat=list(repeat), specular=0.05, shininess=0.1)
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"


def _breaks(low: float, high: float, inner: list[float]) -> list[float]:
    steps = [low - d for d in SITE_STEPS] + sorted(set(inner)) + [high + d for d in SITE_STEPS]
    return sorted(set(steps))


def _surround(footprints: list[tuple[float, float, float, float]]) -> list[tuple[float, ...]]:
    xs = _breaks(min(f[0] for f in footprints), max(f[1] for f in footprints),
                 [v for f in footprints for v in f[:2]])
    ys = _breaks(min(f[2] for f in footprints), max(f[3] for f in footprints),
                 [v for f in footprints for v in f[2:]])
    cells = []
    for x0, x1 in zip(xs, xs[1:], strict=False):
        for y0, y1 in zip(ys, ys[1:], strict=False):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if not any(f[0] < mx < f[1] and f[2] < my < f[3] for f in footprints):
                cells.append((x0, x1, y0, y1))
    return cells


def _sun_box(swing_axes: list[tuple[float, float]]) -> tuple[np.ndarray, float, float]:
    shoulders = np.array([(x, y, 0.0) for x, y in swing_axes]) + SHOULDER
    centre = (shoulders.min(axis=0) + shoulders.max(axis=0)) / 2
    sun = np.array(SUN_DIR) / np.linalg.norm(SUN_DIR)
    along = (shoulders - centre) @ sun
    across = np.linalg.norm(shoulders - centre - np.outer(along, sun), axis=1)
    return centre, float(across.max()) + REACH, float(np.abs(along).max()) + REACH + SUN_BACKOFF


def dress_after_heightfields(scene: mujoco.MjSpec, footprints: list[tuple[float, ...]],
                             swing_axes: list[tuple[float, float]]) -> None:
    first = footprints[0]
    extent = float(np.hypot(first[1] - first[0], first[3] - first[2])) / 2
    centre, reach, distance = _sun_box(swing_axes)
    scene.stat.center = centre.tolist()
    scene.stat.extent = extent
    vis = scene.visual
    vis.global_.offwidth, vis.global_.offheight = 1280, 720
    vis.quality.shadowsize = 8192 if len(swing_axes) == 1 else 4096
    vis.quality.offsamples = 4
    vis.map.shadowclip = float(np.nextafter(np.float32(reach / extent), np.float32(np.inf)))
    vis.map.haze = 0.4
    vis.map.fogstart, vis.map.fogend = FOG_START / extent, FOG_END / extent
    vis.map.znear, vis.map.zfar = NEAR / extent, FAR / extent
    vis.rgba.haze = [*SKY_HORIZON, 1.0]
    vis.rgba.fog = [*SKY_HORIZON, 1.0]
    vis.headlight.ambient = [0.32, 0.32, 0.34]
    vis.headlight.diffuse = [0.25, 0.25, 0.25]
    vis.headlight.specular = [0.05, 0.05, 0.05]
    scene.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                      builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, width=128, height=768,
                      rgb1=list(SKY_TOP), rgb2=list(SKY_HORIZON))
    _add_grid(scene, ((first[1] - first[0]) / GRID_TILE, (first[3] - first[2]) / GRID_TILE))
    scene.worldbody.add_light(name="sun", type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL,
                              pos=(centre - distance * np.array(SUN_DIR)
                                   / np.linalg.norm(SUN_DIR)).tolist(),
                              dir=list(SUN_DIR), castshadow=True,
                              diffuse=[0.75, 0.72, 0.66], specular=[0.3, 0.3, 0.3],
                              ambient=[0.0, 0.0, 0.0])
    _add_site_ground(scene, footprints)
    for geom in scene.geoms:
        if geom.material in ("ground", "site_ground"):
            geom.castshadow = False


def _add_site_ground(scene: mujoco.MjSpec, footprints: list[tuple[float, ...]]) -> None:
    cx, cy = (footprints[0][0] + footprints[0][1]) / 2, (footprints[0][2] + footprints[0][3]) / 2
    verts, uvs, faces = [], [], []
    for x0, x1, y0, y1 in _surround(footprints):
        n = len(verts)
        for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
            verts.append((x, y, -SITE_DROP))
            uvs.append((0.5 + (x - cx) / GRID_TILE, 0.5 - (y - cy) / GRID_TILE))
        faces += [(n, n + 1, n + 2), (n, n + 2, n + 3)]
    verts.append((cx, cy, -SITE_DROP - 1.0))
    uvs.append((0.5, 0.5))
    scene.add_mesh(name="site_ground", uservert=np.ravel(verts).tolist(),
                   usertexcoord=np.ravel(uvs).tolist(), userface=np.ravel(faces).tolist(),
                   inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    scene.worldbody.add_geom(name="site_ground", type=mujoco.mjtGeom.mjGEOM_MESH,
                             meshname="site_ground", material="site_ground", contype=0,
                             conaffinity=0, group=0)
