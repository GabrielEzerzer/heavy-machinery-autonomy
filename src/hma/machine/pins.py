"""Derive the pin table in hma.machine.cylinders from a full-history zx120_ros checkout: pins from
the zx120.xacro before commit 31e8ae6, rod-end bosses and link lengths fitted on the meshes, all
turned into the model's link frames by the current URDF's visual origins."""

from __future__ import annotations

import argparse
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from hma.machine.convert import LINKAGE, fit_ring, link_holes, origin, read_dae

OLD_XACRO = "31e8ae6^:zx120_description/urdf/zx120.xacro"
LINKS = {"upper": "body_link", "boom": "boom_link", "arm": "arm_link", "bucket": "bucket_link"}
XACRO_PINS = {
    "boom_foot": ("upper", "boom_cylinder_head_joint"),
    "arm_head": ("boom", "arm_cylinder_head_joint"),
    "bucket_head": ("arm", "bucket_cylinder_head_joint"),
    "side_link": ("arm", "bucket_b_joint"),
    "link_pin": ("bucket", "bucket_a_joint"),
}
MESH_BOSSES = {
    "boom_rod": ("boom", "zx120-boom.dae", (1.2038, 1.5674), 0.10),
    "arm_rod": ("arm", "zx120-arm.dae", (0.20, 0.65), 0.05),
}


def derive(source: Path) -> tuple[dict[str, tuple[str, tuple[float, ...]]], dict[str, float]]:
    desc = source / "zx120_description"
    urdf = ET.parse(desc / "urdf" / "zx120.urdf").getroot()
    turn = {}
    for frame, link in LINKS.items():
        _, rot = origin(urdf.find(f"link[@name='{link}']/visual/origin"))
        turn[frame] = rot
    text = subprocess.run(["git", "-C", str(source), "show", OLD_XACRO], check=True,
                          capture_output=True, text=True).stdout
    xacro = ET.fromstring(text.replace("xacro:", "xacro_"))
    mesh_frame = {}
    for name, (frame, joint) in XACRO_PINS.items():
        xyz, _ = origin(xacro.find(f".//joint[@name='{joint}']/origin"))
        mesh_frame[name] = (frame, xyz)
    for name, (frame, mesh, seed, radius) in MESH_BOSSES.items():
        verts = read_dae(desc / "meshes" / mesh).verts
        x, z = fit_ring(verts, np.array(seed), radius)
        mesh_frame[name] = (frame, np.array([x, 0.0, z]))
    pins = {name: (frame, tuple(float(v) for v in np.round(turn[frame] @ xyz, 4)))
            for name, (frame, xyz) in mesh_frame.items()}
    lengths = {}
    for name, (mesh, seed, radius) in LINKAGE.items():
        a, b = link_holes(read_dae(desc / "meshes" / mesh).verts, seed, radius)
        lengths[name] = float(np.linalg.norm(b - a))
    return pins, lengths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", type=Path, help="path to a zx120_ros checkout with full history")
    pins, lengths = derive(parser.parse_args().source)
    for name, (frame, xyz) in pins.items():
        print(f"{name:12s} {frame:7s} {xyz}")
    for name, length in lengths.items():
        print(f"{name} length {length:.4f} m")


if __name__ == "__main__":
    main()
