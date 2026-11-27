"""The play loop runs both machines to the end of one seed's episode, the display scene shows each
where its own scene has it, and held-out seeds are refused."""

import mujoco
import numpy as np
import pytest

from hma.machine import machine_prefix
from hma.machine.model import MACHINE_SPACING
from hma.train.play import episodes, mirror
from hma.train.scene import PropScene
from hma.train.task import DEV_BASE, HELD_OUT_BASE

SHORTEST = DEV_BASE + 4


def test_the_baseline_twice_ends_alike_and_is_drawn_where_it_stands():
    *_, (scenes, outcomes) = episodes("nominal", SHORTEST)
    assert outcomes[0] == outcomes[1]
    assert outcomes[0]["seed"] == SHORTEST and outcomes[0]["decisions"] == 171
    display = PropScene(2, "nominal")
    for slot in range(2):
        display.reset(slot, SHORTEST)
    mirror(display, scenes)()
    for slot, scene in enumerate(scenes):
        mujoco.mj_kinematics(scene.model, scene.data)
        for body in ("base", "upper", "arm", "bucket"):
            shown = display.data.body(machine_prefix(slot) + body).xpos
            stands = scene.data.body(machine_prefix(0) + body).xpos
            np.testing.assert_allclose(shown - stands, [0.0, slot * MACHINE_SPACING, 0.0],
                                       atol=1e-9)


def test_a_held_out_seed_is_refused():
    with pytest.raises(ValueError, match="held-out"):
        next(episodes("nominal", HELD_OUT_BASE))
