import mujoco
import numpy as np

from hma.machine import JOINT_NAMES, load_scene, machine_prefix
from hma.machine.demo import TOLERANCE, WAYPOINTS, motion


def test_scripted_motion_reaches_every_waypoint():
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    qpos = [model.joint(machine_prefix(0) + j).qposadr[0] for j in JOINT_NAMES]
    last = {}
    for index in motion(model, data, 1):
        last[index] = data.qpos[qpos].copy()
    assert sorted(last) == list(range(1, len(WAYPOINTS)))
    for index, reached in last.items():
        assert np.abs(reached - WAYPOINTS[index]).max() <= TOLERANCE
