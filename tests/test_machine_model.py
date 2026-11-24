import math

import mujoco
import numpy as np
import pytest

from hma.machine import (
    BODY_NAMES,
    FORCE_CAPS,
    GROUND,
    HOME_KEY,
    HOME_POSE,
    JOINT_LIMITS,
    JOINT_NAMES,
    LINK_MASSES,
    TIP_SITE,
    ground_axes,
    ground_shape,
    inverse_kinematics,
    load_scene,
    machine_prefix,
    set_weld,
    write_ground,
)

SOURCE_LIMITS_DEG = {"boom": (-70.0, 44.0), "arm": (30.0, 152.0), "bucket": (-33.0, 143.0)}


def home_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    return data


def set_pose(model, data, index, pose):
    prefix = machine_prefix(index)
    for name, value in pose.items():
        data.qpos[model.joint(prefix + name).qposadr[0]] = value
        data.act[model.actuator(prefix + name).actadr[0]] = value


def base_position(model, data, index):
    return data.xpos[model.body(machine_prefix(index) + "base").id].copy()


def push_base(model, data, index, force, seconds):
    body = model.body(machine_prefix(index) + "base").id
    data.xfrc_applied[body, :3] = force
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
    data.xfrc_applied[body] = 0.0


def settle(model, data, seconds=1.0):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def test_total_mass_in_operating_weight_band():
    model = load_scene(2)
    for i in range(2):
        masses = {b: model.body(machine_prefix(i) + b).mass[0] for b in BODY_NAMES}
        assert masses == pytest.approx(LINK_MASSES)
        assert 12_000.0 <= sum(masses.values()) <= 12_600.0


def test_joint_limits_match_source():
    model = load_scene(1)
    for name, (lo, hi) in SOURCE_LIMITS_DEG.items():
        joint_range = model.joint(machine_prefix(0) + name).range
        np.testing.assert_allclose(np.degrees(joint_range), [lo, hi], atol=1e-6)
        np.testing.assert_allclose(joint_range, JOINT_LIMITS[name], atol=1e-12)
    assert not model.joint(machine_prefix(0) + "swing").limited


def test_force_caps_on_every_driven_joint():
    model = load_scene(1)
    for name in JOINT_NAMES:
        joint = model.joint(machine_prefix(0) + name)
        cap = FORCE_CAPS[name]
        np.testing.assert_allclose(model.jnt_actfrcrange[joint.id], [-cap, cap])
        assert model.actuator(machine_prefix(0) + name).trnid[0] == joint.id


def test_names_are_prefixed_per_machine():
    model = load_scene(2)
    for i in range(2):
        prefix = machine_prefix(i)
        for name in BODY_NAMES:
            model.body(prefix + name)
        for name in JOINT_NAMES:
            model.joint(prefix + name)
            model.actuator(prefix + name)
        model.hfield(prefix + GROUND)
        model.geom(prefix + GROUND)


def test_home_keyframe_puts_every_machine_in_its_home_pose():
    model = load_scene(2)
    data = home_data(model)
    for i in range(2):
        prefix = machine_prefix(i)
        for name, value in zip(JOINT_NAMES, HOME_POSE, strict=True):
            assert data.qpos[model.joint(prefix + name).qposadr[0]] == value
            assert data.act[model.actuator(prefix + name).actadr[0]] == value
        mujoco.mj_kinematics(model, data)
        np.testing.assert_allclose(base_position(model, data, i), [0.0, 14.0 * i, 0.0])


def test_each_machine_stands_on_its_own_ground():
    model = load_scene(2)
    data = home_data(model)
    settle(model, data)
    pairs = set()
    for contact in data.contact[: data.ncon]:
        pairs.add(tuple(sorted((model.geom(contact.geom1).name, model.geom(contact.geom2).name))))
    assert pairs == {
        (f"{p}{GROUND}", f"{p}{track}")
        for p in (machine_prefix(0), machine_prefix(1))
        for track in ("track_left", "track_right")
    }
    for i in range(2):
        assert abs(base_position(model, data, i)[2]) < 0.01


def test_weld_holds_base():
    model = load_scene(1, weld=True)
    data = home_data(model)
    settle(model, data, 0.2)
    start = base_position(model, data, 0)
    push_base(model, data, 0, [3.0e5, 2.0e5, 3.0e5], 1.0)
    assert np.linalg.norm(base_position(model, data, 0) - start) < 0.01


def test_base_moves_under_large_force_without_weld():
    model = load_scene(1, weld=False)
    data = home_data(model)
    settle(model, data, 0.2)
    start = base_position(model, data, 0)
    push_base(model, data, 0, [3.0e5, 2.0e5, 0.0], 1.0)
    assert np.linalg.norm(base_position(model, data, 0) - start) > 0.5


def test_weld_switches_at_run_time_and_holds_where_it_was_switched_on():
    model = load_scene(1, weld=True)
    data = home_data(model)
    set_weld(model, data, 0, False)
    push_base(model, data, 0, [3.0e5, 0.0, 0.0], 0.5)
    settle(model, data, 1.0)
    moved = base_position(model, data, 0)
    assert moved[0] > 0.3
    set_weld(model, data, 0, True)
    push_base(model, data, 0, [-3.0e5, 2.0e5, 0.0], 1.0)
    assert np.linalg.norm(base_position(model, data, 0) - moved) < 0.01


def test_arm_and_bucket_pass_through_ground_without_contact():
    model = load_scene(1, weld=True)
    data = home_data(model)
    xs, ys = ground_axes()
    heights = np.where(xs[None, :] > 3.0, 1.5, 0.0) * np.ones((len(ys), 1))
    write_ground(model, 0, heights)
    target = np.array([5.5, 0.0, 0.3])
    pose = dict(zip(JOINT_NAMES, inverse_kinematics(target, 1.0), strict=True))
    set_pose(model, data, 0, pose)
    settle(model, data, 1.0)
    np.testing.assert_allclose(data.site(machine_prefix(0) + TIP_SITE).xpos, target, atol=0.05)
    arm_bodies = {model.body(machine_prefix(0) + b).id for b in ("boom", "arm", "bucket")}
    for contact in data.contact[: data.ncon]:
        bodies = {model.geom_bodyid[contact.geom1], model.geom_bodyid[contact.geom2]}
        assert not bodies & arm_bodies
    for body in arm_bodies:
        assert np.allclose(data.cfrc_ext[body], 0.0)
    for name, value in pose.items():
        assert data.qpos[model.joint(machine_prefix(0) + name).qposadr[0]] == pytest.approx(
            value, abs=0.02
        )


def test_written_ground_heights_appear_in_the_heightfield():
    model = load_scene(1)
    data = home_data(model)
    xs, ys = ground_axes()
    heights = np.zeros(ground_shape())
    col, row = int(np.argmin(np.abs(xs - 6.0))), int(np.argmin(np.abs(ys - 1.0)))
    heights[row - 2 : row + 3, col - 2 : col + 3] = -1.2
    write_ground(model, 0, heights)
    mujoco.mj_forward(model, data)
    geomid = np.array([-1], dtype=np.int32)
    for x, y, expected in ((xs[col], ys[row], -1.2), (2.0, -3.0, 0.0)):
        distance = mujoco.mj_ray(model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]),
                                 None, 1, -1, geomid)
        assert model.geom(int(geomid[0])).name == machine_prefix(0) + GROUND
        assert 5.0 - distance == pytest.approx(expected, abs=1e-6)


def test_write_ground_rejects_bad_input():
    model = load_scene(1)
    with pytest.raises(ValueError):
        write_ground(model, 0, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        write_ground(model, 0, np.full(ground_shape(), -10.0))
    with pytest.raises(ValueError):
        write_ground(model, 0, np.full(ground_shape(), np.nan))


def test_ground_spacing_matches_cell_size():
    xs, ys = ground_axes()
    assert math.isclose(xs[1] - xs[0], 0.1) and math.isclose(ys[1] - ys[0], 0.1)
    assert xs[0] < -2.5 and xs[-1] > 8.5
