import mujoco
import numpy as np

from hma.machine import JOINT_NAMES, load_scene, machine_prefix
from hma.machine.demo import CONTROL_PERIOD, TOLERANCE, WAYPOINTS, motion


def test_scripted_motion_reaches_every_waypoint_at_ten_hertz_control():
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    qpos = [model.joint(machine_prefix(0) + j).qposadr[0] for j in JOINT_NAMES]
    assert CONTROL_PERIOD == 0.1
    last, times = {}, []
    for index in motion(model, data, 1):
        last[index] = data.qpos[qpos].copy()
        times.append(data.time)
    np.testing.assert_allclose(np.diff(times), CONTROL_PERIOD, atol=1e-9)
    assert sorted(last) == list(range(1, len(WAYPOINTS)))
    for index, reached in last.items():
        assert np.abs(reached - WAYPOINTS[index]).max() <= TOLERANCE
    assert np.isfinite(data.qpos).all()
