"""Rebuild the machine assets from a checkout of github.com/pwri-opera/zx120_ros.
Reads the URDF, COLLADA meshes and texture; writes OBJ meshes, a PNG texture and excavator.xml.
Usage: python -I -m hma.machine.convert <path to zx120_ros checkout>"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from hma.machine.render import write_png

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
CYLINDERS = {
    "boom": ("zx120-boom-cyl1.dae", "zx120-boom-cyl2.dae"),
    "arm": ("zx120-arm-cyl1.dae", "zx120-arm-cyl2.dae"),
    "bucket": ("zx120-bucket-cyl1.dae", "zx120-bucket-cyl2.dae"),
}
LINKAGE = {
    "side_link": ("zx120-p1.dae", (0.0, 0.0), (0.254, -0.422)),
    "bucket_link": ("zx120-p2.dae", (0.0042, 0.0054), (0.4204, 0.2191)),
}
TEXTURE = "zx120.png"
TEXTURE_SIZE = 1024
PAINT = {"specular": "0.3", "shininess": "0.35"}
STEEL = {"rgba": "0.78 0.8 0.82 1", "specular": "0.9", "shininess": "0.9", "reflectance": "0.15"}
TRACK_FOOTPRINT_DEPTH = 0.05
TRACK_FEET_ALONG = 6
TRACK_FOOT_RADIUS = 0.1
COUNTERWEIGHT_DEPTH = 0.6
COUNTERWEIGHT_HEIGHT = 1.0


@dataclass
class Mesh:
    verts: np.ndarray
    normals: np.ndarray
    uvs: np.ndarray
    corners: np.ndarray

    def transformed(self, rot: np.ndarray, shift: np.ndarray) -> Mesh:
        return Mesh(self.verts @ rot.T + shift, self.normals @ rot.T, self.uvs, self.corners)


def rpy_matrix(rpy: tuple[float, float, float]) -> np.ndarray:
    r, p, y = rpy
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    return rz @ ry @ rx


def y_rotation(angle: float) -> np.ndarray:
    return rpy_matrix((0.0, angle, 0.0))


def matrix_quat(m: np.ndarray) -> np.ndarray:
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(m, float).ravel())
    return quat


def read_dae(path: Path) -> Mesh:
    root = ET.parse(path).getroot()
    scale = float(root.find("c:asset/c:unit", DAE_NS).get("meter", "1"))
    up = root.findtext("c:asset/c:up_axis", "Z_UP", DAE_NS).strip()
    if up != "Z_UP":
        raise ValueError(f"{path.name}: unsupported up axis {up}")
    sources = {s.get("id"): s for s in root.iter(f"{{{DAE_NS['c']}}}source")}
    geometries = {g.get("id"): g for g in root.iter(f"{{{DAE_NS['c']}}}geometry")}
    parts: dict[str, list[np.ndarray]] = {"v": [], "n": [], "t": [], "c": []}
    count = np.zeros(3, int)

    def floats(source_id: str, width: int) -> np.ndarray:
        src = sources[source_id.lstrip("#")]
        return np.array(src.find("c:float_array", DAE_NS).text.split(), float).reshape(-1, width)

    def visit(node: ET.Element, parent: np.ndarray) -> None:
        nonlocal count
        local = np.eye(4)
        for m in node.findall("c:matrix", DAE_NS):
            local = local @ np.array(m.text.split(), float).reshape(4, 4)
        world = parent @ local
        normal_map = np.linalg.inv(world[:3, :3]).T
        for inst in node.findall("c:instance_geometry", DAE_NS):
            mesh = geometries[inst.get("url").lstrip("#")].find("c:mesh", DAE_NS)
            if any(mesh.find(f"c:{kind}", DAE_NS) is not None for kind in ("polylist", "polygons")):
                raise ValueError(f"{path.name}: only <triangles> primitives are supported")
            vin = mesh.find("c:vertices/c:input[@semantic='POSITION']", DAE_NS)
            pos = floats(vin.get("source"), 3)
            pos = (np.c_[pos, np.ones(len(pos))] @ world.T)[:, :3] * scale
            normals = uvs = None
            for tri in mesh.findall("c:triangles", DAE_NS):
                inputs = {i.get("semantic"): i for i in tri.findall("c:input", DAE_NS)}
                if not {"VERTEX", "NORMAL", "TEXCOORD"} <= inputs.keys():
                    raise ValueError(f"{path.name}: triangles need positions, normals and UVs")
                if normals is None:
                    normals = floats(inputs["NORMAL"].get("source"), 3) @ normal_map.T
                    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
                    uvs = floats(inputs["TEXCOORD"].get("source"), 2)
                stride = max(int(i.get("offset")) for i in inputs.values()) + 1
                idx = np.array(tri.find("c:p", DAE_NS).text.split(), int).reshape(-1, stride)
                offsets = [int(inputs[s].get("offset")) for s in ("VERTEX", "NORMAL", "TEXCOORD")]
                parts["c"].append(idx[:, offsets].reshape(-1, 3, 3) + count)
            parts["v"].append(pos)
            parts["n"].append(normals)
            parts["t"].append(uvs)
            count = count + [len(pos), len(normals), len(uvs)]
        for child in node.findall("c:node", DAE_NS):
            visit(child, world)

    for scene in root.iter(f"{{{DAE_NS['c']}}}visual_scene"):
        for node in scene.findall("c:node", DAE_NS):
            visit(node, np.eye(4))
    return Mesh(*(np.vstack(parts[k]) for k in ("v", "n", "t", "c")))


def write_obj(path: Path, mesh: Mesh) -> None:
    points = mesh.verts.astype(np.float32)
    index: dict[tuple, int] = {}
    number = {}
    for v in mesh.corners[:, :, 0].ravel():
        number[v] = index.setdefault(tuple(points[v]), len(index) + 1)
    lines = ["# converted from COLLADA"]
    lines += [f"v {x:.9g} {y:.9g} {z:.9g}" for x, y, z in index]
    lines += [f"vt {u:.6f} {v:.6f}" for u, v in mesh.uvs]
    lines += [f"vn {x:.5f} {y:.5f} {z:.5f}" for x, y, z in mesh.normals]
    for tri in mesh.corners:
        lines.append("f " + " ".join(f"{number[v]}/{t + 1}/{n + 1}" for v, n, t in tri))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def origin(el: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if el is None:
        return np.zeros(3), np.eye(3)
    xyz = np.array(el.get("xyz", "0 0 0").split(), float)
    rpy = tuple(float(v) for v in el.get("rpy", "0 0 0").split())
    return xyz, rpy_matrix(rpy)


def fmt(values: np.ndarray) -> str:
    return " ".join(f"{v:.6g}" for v in np.where(np.abs(values) < 1e-12, 0.0, values))


def link_meshes(urdf: ET.Element, meshes_dir: Path) -> dict[str, Mesh]:
    out = {}
    for link in urdf.findall("link"):
        if link.get("name") not in LINKS:
            continue
        visual = link.find("visual")
        filename = Path(visual.find("geometry/mesh").get("filename")).name
        xyz, rot = origin(visual.find("origin"))
        out[link.get("name")] = read_dae(meshes_dir / filename).transformed(rot, xyz)
    return out


def axis_frame(direction: np.ndarray, along: np.ndarray) -> np.ndarray:
    angle = math.atan2(direction[0], direction[1]) - math.atan2(along[0], along[2])
    return y_rotation(-angle)


def cylinder_parts(meshes_dir: Path) -> dict[str, Mesh]:
    out = {}
    for name, (barrel_file, rod_file) in CYLINDERS.items():
        barrel, rod = read_dae(meshes_dir / barrel_file), read_dae(meshes_dir / rod_file)
        xz = rod.verts[:, [0, 2]]
        axis = np.linalg.svd(xz - xz.mean(axis=0))[2][0]
        axis *= np.sign(axis @ xz.mean(axis=0))
        rot = axis_frame(axis, np.array([0.0, 0.0, 1.0]))
        end = (xz @ axis).max()
        out[f"{name}_barrel"] = barrel.transformed(rot, np.zeros(3))
        out[f"{name}_rod"] = rod.transformed(rot, np.array([0.0, 0.0, -end]))
    return out


def linkage_parts(meshes_dir: Path) -> dict[str, Mesh]:
    out = {}
    for name, (filename, a, b) in LINKAGE.items():
        a, b = np.array(a), np.array(b)
        rot = axis_frame(b - a, np.array([1.0, 0.0, 0.0]))
        out[name] = read_dae(meshes_dir / filename).transformed(rot, -rot @ [a[0], 0.0, a[1]])
    return out


def track_feet(crawler: np.ndarray) -> list[tuple[str, np.ndarray]]:
    bottom = crawler[:, 2] < crawler[:, 2].min() + TRACK_FOOTPRINT_DEPTH
    pts = crawler[bottom]
    xs = np.linspace(pts[:, 0].min(), pts[:, 0].max(), TRACK_FEET_ALONG)
    inner, outer = np.abs(pts[:, 1]).min(), np.abs(pts[:, 1]).max()
    feet = []
    for name, side in (("track_left", 1.0), ("track_right", -1.0)):
        for k, (x, y) in enumerate((x, y) for x in xs for y in (inner, outer)):
            feet.append((f"{name}_{k}", np.array([x, side * y, TRACK_FOOT_RADIUS])))
    return feet


def counterweight_box(body: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rear = body[:, 0].min()
    pts = body[body[:, 0] < rear + COUNTERWEIGHT_DEPTH]
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    hi[2] = lo[2] + COUNTERWEIGHT_HEIGHT
    return (lo + hi) / 2, (hi - lo) / 2


def build_mjcf(urdf: ET.Element, meshes: dict[str, Mesh], parts: list[str]) -> ET.Element:
    joints = {j.get("name"): j for j in urdf.findall("joint")}
    root = ET.Element("mujoco", model="excavator_12t")
    ET.SubElement(root, "compiler", angle="radian", discardvisual="false")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="zx120", type="2d", file=TEXTURE)
    ET.SubElement(asset, "material", name="paint", texture="zx120", **PAINT)
    ET.SubElement(asset, "material", name="steel", **STEEL)
    for _, mesh in LINKS.values():
        ET.SubElement(asset, "mesh", name=mesh, file=f"{mesh}.obj", inertia="convex")
    for part in parts:
        ET.SubElement(asset, "mesh", name=part, file=f"{part}.obj", inertia="shell")
    world = ET.SubElement(root, "worldbody")

    base = ET.SubElement(world, "body", name="base")
    ET.SubElement(base, "freejoint", name="base_free")
    ET.SubElement(base, "geom", name="crawler", type="mesh", mesh="crawler",
                  material="paint", contype="0", conaffinity="0", group="1")
    for name, centre in track_feet(meshes["track_link"].verts):
        ET.SubElement(base, "geom", name=name, type="sphere", pos=fmt(centre),
                      size=fmt(np.array([TRACK_FOOT_RADIUS])), mass="0",
                      rgba="0.2 0.6 0.9 0.4", group="3", friction="0.8 0.005 0.0001")

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
        ET.SubElement(body, "geom", name=mesh, type="mesh", mesh=mesh, material="paint",
                      contype="0", conaffinity="0", group="1")
        if body_name == "upper":
            centre, half = counterweight_box(meshes["body_link"].verts)
            ET.SubElement(body, "geom", name="counterweight", type="box", pos=fmt(centre),
                          size=fmt(half), contype="0", conaffinity="0", group="3",
                          rgba="0.5 0.5 0.5 0.3")
        parent = body

    tip = joints[TIP_JOINT]
    xyz, rot = origin(tip.find("origin"))
    ET.SubElement(parent, "site", name="bucket_tip", pos=fmt(xyz), quat=fmt(matrix_quat(rot)),
                  size="0.04", rgba="1 0 0 1", group="2")
    for name, end in zip(("bucket_edge_right", "bucket_edge_left"),
                         cutting_edge_ends(meshes["bucket_link"].verts, xyz), strict=True):
        ET.SubElement(parent, "site", name=name, pos=fmt(end), quat=fmt(matrix_quat(rot)),
                      size="0.03", rgba="1 0 0 1", group="2")
    return root


def cutting_edge_ends(bucket: np.ndarray, tip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    along = tip / np.linalg.norm(tip)
    near_edge = bucket[bucket @ along >= np.linalg.norm(tip) - EDGE_BAND]
    return (tip + [0.0, near_edge[:, 1].min(), 0.0], tip + [0.0, near_edge[:, 1].max(), 0.0])


def write_texture(source: Path, target: Path) -> None:
    spec = mujoco.MjSpec()
    spec.add_texture(name="t", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(source))
    model = spec.compile()
    h, w, c = int(model.tex_height[0]), int(model.tex_width[0]), int(model.tex_nchannel[0])
    k = h // TEXTURE_SIZE
    pixels = model.tex_data.reshape(h, w, c)[:, :, :3].astype(float)
    small = pixels.reshape(TEXTURE_SIZE, k, w // k, k, 3).mean(axis=(1, 3))
    write_png(target, np.round(small).astype(np.uint8))


def convert(source: Path) -> None:
    desc = source / "zx120_description"
    urdf = ET.parse(desc / "urdf" / "zx120.urdf").getroot()
    meshes = link_meshes(urdf, desc / "meshes")
    parts = {**cylinder_parts(desc / "meshes"), **linkage_parts(desc / "meshes")}
    ASSETS.mkdir(parents=True, exist_ok=True)
    for link, mesh in meshes.items():
        write_obj(ASSETS / f"{LINKS[link][1]}.obj", mesh)
    for name, mesh in parts.items():
        write_obj(ASSETS / f"{name}.obj", mesh)
    write_texture(desc / "meshes" / TEXTURE, ASSETS / TEXTURE)
    tree = ET.ElementTree(build_mjcf(urdf, meshes, list(parts)))
    ET.indent(tree)
    tree.write(ASSETS / "excavator.xml", encoding="unicode")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="path to a zx120_ros checkout")
    convert(parser.parse_args().source)


if __name__ == "__main__":
    main()
