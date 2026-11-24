import math
import random

from hma.task.checks import deepest_reachable


def tip(arm, boom, stick, bucket):
    px, pz = arm.boom_pivot_estimate
    a1 = -math.radians(boom)
    a2 = a1 - math.radians(stick)
    a3 = a2 - math.radians(bucket)
    r = px + arm.boom_m * math.cos(a1) + arm.arm_m * math.cos(a2) + arm.bucket_m * math.cos(a3)
    z = pz + arm.boom_m * math.sin(a1) + arm.arm_m * math.sin(a2) + arm.bucket_m * math.sin(a3)
    return r, z


def test_every_pose_inside_joint_limits_is_reachable(arm):
    rng = random.Random(0)
    for _ in range(300):
        pose = (rng.uniform(*arm.boom_deg), rng.uniform(*arm.arm_deg), rng.uniform(*arm.bucket_deg))
        r, z = tip(arm, *pose)
        if r > 0:
            angle = rng.uniform(-math.pi, math.pi)
            assert arm.reachable(r * math.cos(angle), r * math.sin(angle), z), pose


def test_envelope_is_that_of_a_12_tonne_class_machine(arm):
    assert arm.reachable(5.5, 0, -1)
    assert not arm.reachable(5.5, 0, -8)
    assert not arm.reachable(9.0, 0, 0)
    assert 4.0 < -deepest_reachable(arm, 5.5, 0) < 5.5
    assert deepest_reachable(arm, 20, 0) is None
