"""The drawn cylinders and links sit on the pins the cylinder force model uses, at every pose.
The boom's two cylinders are one source mesh; each must sit 0.273 m off the boom's centre plane,
where the clevises on the upper structure are."""

import mujoco
import numpy as np

from hma.machine import load_scene
from hma.machine.cylinders import H_LINK, SIDE_LINK, cylinder_length
from hma.machine.model import JOINT_NAMES, machine_prefix
from hma.machine.render import CYLINDERS, PARTS, draw_hydraulics, pin_positions, view_option

BOOM_PAIR_OFFSET = 0.273


def _pose(model, data, rng):
    for i in range(2):
        for name in JOINT_NAMES:
            joint = model.joint(machine_prefix(i) + name)
            data.qpos[joint.qposadr[0]] = rng.uniform(*model.jnt_range[joint.id])
    mujoco.mj_forward(model, data)


def test_drawn_pins_match_the_cylinder_model():
    model = load_scene(n_machines=2, weld=True)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(0)
    for _ in range(20):
        _pose(model, data, rng)
        for i in range(2):
            prefix = machine_prefix(i)
            pins = pin_positions(model, data, prefix)
            axis = data.xmat[model.body(prefix + "arm").id].reshape(3, 3)[:, 1]

            def planar(a, b, pins=pins, axis=axis):
                gap = pins[b] - pins[a]
                return np.linalg.norm(gap - (gap @ axis) * axis)

            for joint, (base, rod) in CYLINDERS.items():
                q = data.qpos[model.joint(prefix + joint).qposadr[0]]
                np.testing.assert_allclose(planar(base, rod), cylinder_length(joint, q)[0],
                                           atol=1e-9)
            np.testing.assert_allclose(planar("side_link", "coupler"), SIDE_LINK, atol=1e-9)
            np.testing.assert_allclose(planar("link_pin", "coupler"), H_LINK, atol=1e-9)


def test_every_part_is_drawn_and_sites_are_hidden():
    model = load_scene(n_machines=2)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    scene = mujoco.MjvScene(model, maxgeom=50)
    draw_hydraulics(model, data, scene)
    assert scene.ngeom == 2 * len(PARTS)
    mujoco.mjv_updateScene(model, data, view_option(), None, mujoco.MjvCamera(),
                           mujoco.mjtCatBit.mjCAT_ALL, scene)
    assert all(scene.geoms[i].objtype != mujoco.mjtObj.mjOBJ_SITE for i in range(scene.ngeom))


def test_both_boom_cylinders_sit_on_their_clevises():
    model = load_scene(n_machines=2, weld=True)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(1)
    for _ in range(5):
        _pose(model, data, rng)
        scene = mujoco.MjvScene(model, maxgeom=50)
        draw_hydraulics(model, data, scene)
        for i in range(2):
            prefix = machine_prefix(i)
            pins = pin_positions(model, data, prefix)
            axis = data.xmat[model.body(prefix + "upper").id].reshape(3, 3)[:, 1]
            along = pins["boom_rod"] - pins["boom_foot"]
            along /= np.linalg.norm(along)
            for part, pin in (("boom_barrel", "boom_foot"), ("boom_rod", "boom_rod")):
                mesh = model.mesh(prefix + part)
                geom = next(scene.geoms[g] for g in range(scene.ngeom)
                            if scene.geoms[g].dataid == 2 * mesh.id)
                verts = model.mesh_vert[mesh.vertadr[0]:mesh.vertadr[0] + mesh.vertnum[0]]
                rel = verts @ geom.mat.reshape(3, 3).T + geom.pos - pins[pin]
                eye = rel[np.abs(rel @ along) < 0.1] @ axis
                for sign in (1.0, -1.0):
                    side = eye[sign * eye > 0]
                    centre = (side.min() + side.max()) / 2
                    np.testing.assert_allclose(centre, sign * BOOM_PAIR_OFFSET, atol=0.002)
