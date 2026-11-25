"""The scene's clip, fog and haze distances are the same in metres at any machine count, and the
sun's shadow box holds every machine at full reach, in front of the light."""

import numpy as np
import pytest

from hma.machine import load_scene
from hma.machine.model import MACHINE_SPACING, MAX_MACHINES
from hma.machine.scenery import REACH, SHOULDER, SUN_DIR

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
