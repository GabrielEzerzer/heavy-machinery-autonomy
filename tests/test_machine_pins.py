"""The pin-deriving command runs end to end on a tiny stand-in checkout: a URDF, an old xacro in git
history and ring-shaped meshes whose centres and spacing are known."""

import subprocess

import numpy as np

from hma.machine import pins
from hma.machine.convert import LINKAGE

TURN_RPY = (0.0, 0.0, np.pi / 2)
TURN = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
XACRO_XYZ = {"boom_cylinder_head_joint": (0.5, -0.2, 0.1),
             "arm_cylinder_head_joint": (-1.0, 0, 2.6),
             "bucket_cylinder_head_joint": (-0.5, 0, 0.4),
             "bucket_b_joint": (0.0, 0, 2.2),
             "bucket_a_joint": (-0.3, 0, -0.1)}
BOSS_CENTRE = {"zx120-boom.dae": (1.21, 1.56), "zx120-arm.dae": (0.22, 0.63)}
LINK_END = {"zx120-p1.dae": (0.26, -0.43), "zx120-p2.dae": (0.40, 0.24)}


def _ring(centre, radius, n=24):
    angle = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, z = centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle)
    return np.c_[x, np.zeros(n), z]


def _write_dae(path, rings):
    verts = np.vstack([np.vstack([ring, ring + [0, 0.05, 0]]) for ring in rings])
    tris, start = [], 0
    for ring in rings:
        n = len(ring)
        for k in range(n):
            a, b = start + k, start + (k + 1) % n
            tris += [(a, b, a + n), (b, b + n, a + n)]
        start += 2 * n
    p = " ".join(f"{v} 0 0" for tri in tris for v in tri)
    floats = " ".join(f"{x:.6f}" for x in verts.ravel())
    path.write_text(f"""<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit meter="1"/><up_axis>Z_UP</up_axis></asset>
<library_geometries><geometry id="g"><mesh>
<source id="pos"><float_array id="pa" count="{verts.size}">{floats}</float_array></source>
<source id="nrm"><float_array id="na" count="3">0 1 0</float_array></source>
<source id="uv"><float_array id="ua" count="2">0 0</float_array></source>
<vertices id="v"><input semantic="POSITION" source="#pos"/></vertices>
<triangles count="{len(tris)}"><input semantic="VERTEX" source="#v" offset="0"/>
<input semantic="NORMAL" source="#nrm" offset="1"/>
<input semantic="TEXCOORD" source="#uv" offset="2"/>
<p>{p}</p></triangles></mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="s"><node><instance_geometry url="#g"/></node>
</visual_scene></library_visual_scenes></COLLADA>""")


def _checkout(root):
    desc = root / "zx120_description"
    (desc / "urdf").mkdir(parents=True)
    (desc / "meshes").mkdir()
    rpy = " ".join(str(v) for v in TURN_RPY)
    links = "".join(f'<link name="{link}"><visual><origin xyz="0 0 0" rpy="{rpy}"/></visual></link>'
                    for link in pins.LINKS.values())
    (desc / "urdf" / "zx120.urdf").write_text(f'<robot name="zx120">{links}</robot>')
    joints = "".join(f'<joint name="{name}"><origin xyz="{" ".join(map(str, xyz))}"/></joint>'
                     for name, xyz in XACRO_XYZ.items())
    xacro = desc / "urdf" / "zx120.xacro"
    xacro.write_text(f'<robot xmlns:xacro="http://ros.org/wiki/xacro">{joints}</robot>')
    for _, mesh, _, radius in pins.MESH_BOSSES.values():
        _write_dae(desc / "meshes" / mesh, [_ring(BOSS_CENTRE[mesh], radius)])
    for mesh, _, radius in LINKAGE.values():
        rings = [_ring((0.0, 0.0), radius), _ring(LINK_END[mesh], radius)]
        _write_dae(desc / "meshes" / mesh, rings)
    git = ["git", "-C", str(root), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "stand-in"], check=True)


def test_pins_command_reads_a_checkout(tmp_path, monkeypatch, capsys):
    _checkout(tmp_path)
    monkeypatch.setattr(pins, "OLD_XACRO", "HEAD:zx120_description/urdf/zx120.xacro")
    monkeypatch.setattr("sys.argv", ["pins", str(tmp_path)])
    pins.main()
    out = capsys.readouterr().out
    found, lengths = pins.derive(tmp_path)
    for name, (frame, joint) in pins.XACRO_PINS.items():
        assert found[name][0] == frame
        np.testing.assert_allclose(found[name][1], TURN @ XACRO_XYZ[joint], atol=1e-4)
    for name, (_, mesh, _, _) in pins.MESH_BOSSES.items():
        x, z = BOSS_CENTRE[mesh]
        np.testing.assert_allclose(found[name][1], TURN @ (x, 0.0, z), atol=1e-4)
    for name, (mesh, _, _) in LINKAGE.items():
        np.testing.assert_allclose(lengths[name], np.hypot(*LINK_END[mesh]), atol=1e-6)
        assert f"{name} length {lengths[name]:.4f} m" in out
