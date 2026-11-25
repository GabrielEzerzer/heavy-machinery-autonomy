"""The scene's clip, fog and haze distances are the same in metres at any machine count, the sun's
shadow box holds every machine at full reach, and the plain drawn meshes add no mass or contact."""

import mujoco
import numpy as np
import pytest

from hma.machine import load_scene
from hma.machine.model import MACHINE_SPACING, MAX_MACHINES, machine_prefix
from hma.machine.scenery import (
    DRAWN_GROUP,
    HIDDEN_GROUP,
    PLAIN,
    REACH,
    SHOULDER,
    SUN_DIR,
)

COUNTS = (1, 2, 8, MAX_MACHINES)


def _metres(model):
    extent = model.stat.extent
    vis = model.vis.map
    return [vis.znear * extent, vis.zfar * extent, vis.fogstart * extent, vis.fogend * extent]


def test_look_distances_do_not_depend_on_the_machine_count():
    first = _metres(load_scene(1))
    for count in COUNTS[1:]:
        np.testing.assert_allclose(_metres(load_scene(count)), first)


@pytest.mark.parametrize("count", COUNTS)
def test_sun_shadow_box_holds_every_machine(count):
    model = load_scene(count)
    sun = model.light("sun")
    direction = np.array(SUN_DIR) / np.linalg.norm(SUN_DIR)
    np.testing.assert_allclose(sun.dir, direction, atol=1e-6)
    half = model.vis.map.shadowclip * model.stat.extent
    near = model.vis.map.znear * model.stat.extent
    for i in range(count):
        shoulder = SHOULDER + [0.0, i * MACHINE_SPACING, 0.0] - sun.pos
        along = shoulder @ direction
        across = np.linalg.norm(shoulder - along * direction)
        assert across + REACH <= half + 1e-6
        assert along - REACH > near


def test_tracks_and_upper_structure_are_drawn_plain_without_mass_or_contact():
    model = load_scene(2)
    default = mujoco.MjvOption()
    assert default.geomgroup[DRAWN_GROUP] and not default.geomgroup[HIDDEN_GROUP]
    for i in range(2):
        for name in ("crawler", "body"):
            detailed = model.geom(machine_prefix(i) + name)
            plain = model.geom(machine_prefix(i) + name + PLAIN)
            assert detailed.group[0] == HIDDEN_GROUP and plain.group[0] == DRAWN_GROUP
            assert plain.contype[0] == 0 and plain.conaffinity[0] == 0
            assert plain.bodyid[0] == detailed.bodyid[0]
            fine, coarse = model.mesh(detailed.dataid[0]), model.mesh(plain.dataid[0])
            assert coarse.facenum[0] < 0.6 * fine.facenum[0]
    alone = load_scene(1)
    np.testing.assert_array_equal(alone.body_mass[:6], model.body_mass[:6])
