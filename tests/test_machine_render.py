"""The drawn cylinders and links sit on the pins the cylinder force model uses, at every pose."""

import mujoco
import numpy as np

from hma.machine import load_scene
from hma.machine.cylinders import CYLINDER_PINS, H_LINK, SIDE_LINK, cylinder_length
from hma.machine.model import JOINT_NAMES, machine_prefix
from hma.machine.render import PARTS, draw_hydraulics, pin_positions, view_option


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

            for joint, (base, rod) in CYLINDER_PINS.items():
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
