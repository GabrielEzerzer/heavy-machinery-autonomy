"""Rebuild the machine assets from a checkout of github.com/pwri-opera/zx120_ros.
Reads the URDF and COLLADA meshes, writes binary STL meshes and excavator.xml into assets/.
Usage: python -I -m hma.machine.convert <path to zx120_ros checkout>"""

from __future__ import annotations

import argparse
import math
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent / "assets"
DAE_NS = {"c": "http://www.collada.org/2005/11/COLLADASchema"}

LINKS = {
    "track_link": ("base", "crawler"),
    "body_link": ("upper", "body"),
    "boom_link": ("boom", "boom"),
    "arm_link": ("arm", "arm"),
    "bucket_link": ("bucket", "bucket"),
}
JOINTS = {
    "swing_joint": "swing",
    "boom_joint": "boom",
    "arm_joint": "arm",
    "bucket_joint": "bucket",
}
TIP_JOINT = "bucket_end_joint"
EDGE_BAND = 0.05
COLORS = {
    "crawler": "0.18 0.18 0.19 1",
    "body": "0.93 0.62 0.12 1",
    "boom": "0.93 0.62 0.12 1",
    "arm": "0.93 0.62 0.12 1",
    "bucket": "0.30 0.30 0.32 1",
}
COUNTERWEIGHT_DEPTH = 0.6
COUNTERWEIGHT_HEIGHT = 1.0


def rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def matrix_quat(m: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(m, float).ravel())
    return quat


def read_dae(path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.parse(path).getroot()
    scale = float(root.find("c:asset/c:unit", DAE_NS).get("meter", "1"))
    up = root.findtext("c:asset/c:up_axis", "Z_UP", DAE_NS).strip()
    if up != "Z_UP":
        raise ValueError(f"{path.name}: unsupported up axis {up}")
    sources = {s.get("id"): s for s in root.iter(f"{{{DAE_NS['c']}}}source")}
    geometries = {g.get("id"): g for g in root.iter(f"{{{DAE_NS['c']}}}geometry")}
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    count = 0

    def positions(mesh: ET.Element) -> np.ndarray:
        vin = mesh.find("c:vertices/c:input[@semantic='POSITION']", DAE_NS)
        src = sources[vin.get("source").lstrip("#")]
        return np.array(src.find("c:float_array", DAE_NS).text.split(), float).reshape(-1, 3)

    def visit(node: ET.Element, parent: np.ndarray) -> None:
        nonlocal count
        local = np.eye(4)
        for m in node.findall("c:matrix", DAE_NS):
            local = local @ np.array(m.text.split(), float).reshape(4, 4)
        world = parent @ local
        for inst in node.findall("c:instance_geometry", DAE_NS):
            mesh = geometries[inst.get("url").lstrip("#")].find("c:mesh", DAE_NS)
            if any(mesh.find(f"c:{kind}", DAE_NS) is not None for kind in ("polylist", "polygons")):
                raise ValueError(f"{path.name}: only <triangles> primitives are supported")
            pos = positions(mesh)
            pos = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3] * scale
            for tri in mesh.findall("c:triangles", DAE_NS):
                inputs = tri.findall("c:input", DAE_NS)
                stride = max(int(i.get("offset")) for i in inputs) + 1
                voff = next(int(i.get("offset")) for i in inputs if i.get("semantic") == "VERTEX")
                idx = np.array(tri.find("c:p", DAE_NS).text.split(), int)
                idx = idx.reshape(-1, stride)[:, voff]
                faces.append(idx.reshape(-1, 3) + count)
            verts.append(pos)
            count += len(pos)
        for child in node.findall("c:node", DAE_NS):
            visit(child, world)

    for scene in root.iter(f"{{{DAE_NS['c']}}}visual_scene"):
        for node in scene.findall("c:node", DAE_NS):
            visit(node, np.eye(4))
    return np.vstack(verts), np.vstack(faces)


def write_stl(path: Path, verts: np.ndarray, faces: np.ndarray) -> None:
    tri = verts[faces]
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    record = np.zeros(len(faces), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    record["n"] = normals
    record["v"] = tri
    with path.open("wb") as f:
        f.write(b"binary STL converted from COLLADA".ljust(80, b" "))
        f.write(struct.pack("<I", len(faces)))
        f.write(record.tobytes())


def origin(el: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if el is None:
        return np.zeros(3), np.eye(3)
    xyz = np.array(el.get("xyz", "0 0 0").split(), float)
    rpy = tuple(float(v) for v in el.get("rpy", "0 0 0").split())
    return xyz, rpy_matrix(rpy)


def fmt(values: np.ndarray) -> str:
    return " ".join(f"{v:.6g}" for v in np.where(np.abs(values) < 1e-12, 0.0, values))


def link_meshes(urdf: ET.Element, meshes_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out = {}
    for link in urdf.findall("link"):
        if link.get("name") not in LINKS:
            continue
        visual = link.find("visual")
        filename = Path(visual.find("geometry/mesh").get("filename")).name
        xyz, rot = origin(visual.find("origin"))
        verts, faces = read_dae(meshes_dir / filename)
        out[link.get("name")] = (verts @ rot.T + xyz, faces)
    return out


def counterweight_box(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rear = body[:, 0].min()
    pts = body[body[:, 0] < rear + COUNTERWEIGHT_DEPTH]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    hi[2] = lo[2] + COUNTERWEIGHT_HEIGHT
    return (lo + hi) / 2, (hi - lo) / 2


def build_mjcf(urdf: ET.Element, meshes: dict[str, tuple[np.ndarray, np.ndarray]]) -> ET.Element:
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    root = ET.Element("mujoco", model="excavator_12t")
    ET.SubElement(root, "compiler", angle="radian", discardvisual="false")
    asset = ET.SubElement(root, "asset")
    for _, mesh in LINKS.values():
        ET.SubElement(asset, "mesh", name=mesh, file=f"{mesh}.stl", inertia="convex")
        ET.SubElement(asset, "material", name=f"{mesh}_paint", rgba=COLORS[mesh])
    world = ET.SubElement(root, "worldbody")

    base = ET.SubElement(world, "body", name="base")
    ET.SubElement(base, "freejoint", name="base_free")
    ET.SubElement(base, "geom", name="crawler", type="mesh", mesh="crawler",
                  material="crawler_paint", contype="0", conaffinity="0", group="1")

    parent = base
    for urdf_joint, name in JOINTS.items():
        j = joints[urdf_joint]
        xyz, rot = origin(j.find("origin"))
        body_name, mesh = LINKS[j.find("child").get("link")]
        body = ET.SubElement(parent, "body", name=body_name, pos=fmt(xyz),
                             quat=fmt(matrix_quat(rot)))
        axis = np.array(j.find("axis").get("xyz").split(), float)
        attrs = {"name": name, "axis": fmt(axis / np.linalg.norm(axis))}
        limit = j.find("limit")
        if j.get("type") == "revolute":
            attrs["range"] = f"{float(limit.get('lower')):.12g} {float(limit.get('upper')):.12g}"
        ET.SubElement(body, "joint", **attrs)
        ET.SubElement(body, "geom", name=mesh, type="mesh", mesh=mesh, material=f"{mesh}_paint",
                      contype="0", conaffinity="0", group="1")
        if body_name == "upper":
            centre, half = counterweight_box(meshes["body_link"][0])
            ET.SubElement(body, "geom", name="counterweight", type="box", pos=fmt(centre),
                          size=fmt(half), contype="0", conaffinity="0", group="3",
                          rgba="0.5 0.5 0.5 0.3")
        parent = body

    tip = joints[TIP_JOINT]
    xyz, rot = origin(tip.find("origin"))
    ET.SubElement(parent, "site", name="bucket_tip", pos=fmt(xyz), quat=fmt(matrix_quat(rot)),
                  size="0.04", rgba="1 0 0 1", group="2")
    for name, end in zip(("bucket_edge_right", "bucket_edge_left"),
                         cutting_edge_ends(meshes["bucket_link"][0], xyz), strict=True):
        ET.SubElement(parent, "site", name=name, pos=fmt(end), quat=fmt(matrix_quat(rot)),
                      size="0.03", rgba="1 0 0 1", group="2")
    return root


def cutting_edge_ends(bucket: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    along = tip / np.linalg.norm(tip)
    near_edge = bucket[bucket @ along >= np.linalg.norm(tip) - EDGE_BAND]
    return (tip + [0.0, near_edge[:, 1].min(), 0.0], tip + [0.0, near_edge[:, 1].max(), 0.0])


def convert(source: Path) -> None:
    desc = source / "zx120_description"
    urdf = ET.parse(desc / "urdf" / "zx120.urdf").getroot()
    meshes = link_meshes(urdf, desc / "meshes")
    ASSETS.mkdir(parents=True, exist_ok=True)
    for link, (verts, faces) in meshes.items():
        write_stl(ASSETS / f"{LINKS[link][1]}.stl", verts, faces)
    tree = ET.ElementTree(build_mjcf(urdf, meshes))
    ET.indent(tree)
    tree.write(ASSETS / "excavator.xml", encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="path to a zx120_ros checkout")
    convert(parser.parse_args().source)


if __name__ == "__main__":
    main()
