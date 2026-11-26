import mujoco
import numpy as np

from hma.machine import JOINT_NAMES, load_scene, machine_prefix
from hma.machine.demo import CONTROL_PERIOD, TOLERANCE, WAYPOINTS, motion


def test_scripted_motion_reaches_every_waypoint_with_the_swing_at_rest_at_ten_hertz_control():
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    joints = [model.joint(machine_prefix(0) + j) for j in JOINT_NAMES]
    qpos = [j.qposadr[0] for j in joints]
    swing = joints[JOINT_NAMES.index("swing")].dofadr[0]
    last, swing_speed, times = {}, {}, []
    for index in motion(model, data, 1):
        last[index] = data.qpos[qpos].copy()
        swing_speed[index] = abs(data.qvel[swing])
        times.append(data.time)
    np.testing.assert_allclose(np.diff(times), CONTROL_PERIOD, atol=1e-9)
    assert sorted(last) == list(range(1, len(WAYPOINTS)))
    for index, reached in last.items():
        assert np.abs(reached - WAYPOINTS[index]).max() <= TOLERANCE
        assert swing_speed[index] <= TOLERANCE / CONTROL_PERIOD
    assert np.isfinite(data.qpos).all()
