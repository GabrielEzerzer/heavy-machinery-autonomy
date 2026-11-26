import math

import numpy as np
import pytest

from hma.machine import JOINT_LIMITS
from hma.machine.cylinders import (
    BUCKET_PIVOT,
    CYLINDER_JOINTS,
    H_LINK,
    MAIN_RELIEF,
    PINS,
    SIDE_LINK,
    coupler,
    cylinder_length,
    joint_torque_limits,
    pin_on_child,
    seal_friction,
    xz,
)
from hma.machine.statics import CHART, Machine, digging, lift, reach_load_point


def sweep(joint, n=2001):
    return np.linspace(*JOINT_LIMITS[joint], n)


@pytest.mark.parametrize("joint", CYLINDER_JOINTS)
def test_lever_arm_is_the_derivative_of_cylinder_length(joint):
    q = sweep(joint)
    h = 1e-6
    numeric = (cylinder_length(joint, q + h)[0] - cylinder_length(joint, q - h)[0]) / (2 * h)
    np.testing.assert_allclose(cylinder_length(joint, q)[1], numeric, atol=1e-7)


@pytest.mark.parametrize("joint", CYLINDER_JOINTS)
def test_stroke_over_the_joint_range_fits_one_cylinder(joint):
    length, _ = cylinder_length(joint, sweep(joint))
    stroke = np.ptp(length)
    assert 0.5 < stroke < 1.3
    assert length.min() - stroke > 0.3


@pytest.mark.parametrize("joint", CYLINDER_JOINTS)
def test_lever_arm_keeps_one_sign_and_never_nears_zero(joint):
    _, lever = cylinder_length(joint, sweep(joint))
    assert np.all(np.sign(lever) == np.sign(lever[0]))
    assert np.abs(lever).min() > 0.08


def test_bucket_linkage_stays_away_from_dead_centre():
    q = sweep("bucket")
    c, _ = coupler(q)
    side = np.array(PINS["side_link"][1])[[0, 2]].reshape(2, 1)
    np.testing.assert_allclose(np.hypot(*(c - side)), SIDE_LINK, atol=1e-9)
    a, _ = pin_on_child(BUCKET_PIVOT, q, xz("link_pin"))
    np.testing.assert_allclose(np.hypot(*(c - a)), H_LINK, atol=1e-9)
    u, v = side - c, a - c
    angle = np.degrees(np.arccos((u * v).sum(0) / (np.hypot(*u) * np.hypot(*v))))
    assert 20 < angle.min() and angle.max() < 160


HAND = {
    "boom": (2, 0.105, 0.070, -1),
    "arm": (1, 0.115, 0.080, 1),
    "bucket": (1, 0.100, 0.070, 1),
}


@pytest.mark.parametrize("joint", CYLINDER_JOINTS)
def test_torque_limit_matches_a_hand_calculation_both_ways(joint):
    count, bore, rod, extend_sign = HAND[joint]
    head = count * math.pi / 4 * bore**2
    annulus = count * math.pi / 4 * (bore**2 - rod**2)
    for q in np.linspace(*JOINT_LIMITS[joint], 4):
        h = 1e-5
        lever = float(cylinder_length(joint, q + h)[0] - cylinder_length(joint, q - h)[0]) / (2 * h)
        assert np.sign(lever) == extend_sign
        plus, minus = joint_torque_limits(joint, q, 34.3e6, 30.0e6)
        push, pull = 34.3e6 * head * abs(lever), 30.0e6 * annulus * abs(lever)
        expected = (push, pull) if extend_sign > 0 else (pull, push)
        assert float(plus) == pytest.approx(expected[0], rel=1e-6)
        assert float(minus) == pytest.approx(expected[1], rel=1e-6)


def test_digging_forces_match_the_spec_sheet():
    dig = digging(Machine())
    assert dig["bucket_force"] == pytest.approx(86e3, rel=0.03)
    assert dig["arm_force"] == pytest.approx(65e3, rel=0.03)


def test_hydraulic_lift_capacity_matches_a_starred_chart_entry():
    rated = 0.87 * lift(Machine(), 1.5, 3.0, MAIN_RELIEF)["hydraulic"]
    assert rated == pytest.approx(7890, rel=0.03)


def test_total_mass_and_centre_follow_a_bucket_removed_at_run_time():
    machine = Machine()
    machine.pose(0.0, -0.3, 1.5, 0.5, with_bucket=False)
    model, data = machine.model, machine.data
    mass = sum(model.body_mass[b] for b in machine.subtree)
    moment = sum(model.body_mass[b] * data.xipos[b] for b in machine.subtree)
    total, com = machine.total()
    machine.done()
    assert machine.bodies["bucket"] in machine.subtree
    assert total == pytest.approx(mass)
    assert total == pytest.approx(float(model.body_subtreemass[machine.bodies["base"]])
                                  - machine.bucket_mass)
    np.testing.assert_allclose(com, moment / mass, atol=1e-12)


def test_every_chart_point_is_reached_at_the_pose_found():
    machine = Machine()
    for height, radius in CHART:
        pose = reach_load_point(machine, radius, height)
        assert pose is not None, (height, radius)
        machine.pose(0.0, *pose, 0.0, False)
        x, _, z = machine.anchor("bucket")
        assert abs(x - radius) < 1e-4 and abs(z - height) < 1e-4


@pytest.mark.parametrize(("head", "rod", "speed", "hand"), [
    ((11e6, 11e6, 11e6), (11e6, 11e6, 11e6), (0.15, -0.15, 0.15),
     (1.75 * 920.0 + 2.625 * (145.0 - 6.3 * 50.0 / 4.7), None, None)),
    ((20e6, 20e6, 20e6), (8.65e6, 8.65e6, 8.65e6), (0.30, 0.30, -0.30),
     (None, None, 0.875 * 777.5 + 1.25 * 249.55)),
    ((0.3e6, 0.3e6, 0.3e6), (34.3e6, 34.3e6, 34.3e6), (0.01, 0.01, 0.01),
     (None, 1.0 * (1195.0 + 22.0 * 18.3) + 1.4375 * (285.0 + 19.0 * 18.0), None)),
    ((22e6, 22e6, 22e6), (11e6, 11e6, 11e6), (0.10, 0.10, 0.10),
     (1.75 * (1085.0 + 920.0) / 2 + 2.625 * (190.0 + 195.0) / 2, None, None)),
])
def test_seal_friction_matches_the_published_table_worked_by_hand(head, rod, speed, hand):
    force = seal_friction(np.array(head), np.array(rod), np.array(speed))
    for value, expected in zip(force, hand, strict=True):
        if expected is not None:
            assert value == pytest.approx(expected, rel=1e-9)
