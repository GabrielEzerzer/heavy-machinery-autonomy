import math

import numpy as np

from hma.machine import JOINT_LIMITS, JOINT_NAMES, forward_kinematics


def test_every_pose_inside_joint_limits_is_reachable(arm):
    rng = np.random.default_rng(0)
    for _ in range(100):
        pose = [rng.uniform(-math.pi, math.pi)]
        pose += [rng.uniform(*JOINT_LIMITS[name]) for name in JOINT_NAMES[1:]]
        tip, _ = forward_kinematics(np.array(pose))
        assert arm.reachable(*tip), pose


def test_envelope_is_that_of_a_12_tonne_class_machine(arm):
    assert arm.reachable(5.5, 0, -1)
    assert not arm.reachable(5.5, 0, -8)
    assert not arm.reachable(9.0, 0, 0)
