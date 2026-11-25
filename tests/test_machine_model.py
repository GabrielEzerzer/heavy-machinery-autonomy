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
    forward_kinematics,
    ground_axes,
    ground_shape,
    inverse_kinematics,
    load_scene,
    machine_prefix,
    set_weld,
    write_ground,
)
from hma.machine.load import apply_load, block, hang_load
from hma.machine.model import (
    GROUND_DEPTH,
    LOAD_POINTS,
    MACHINE_SPACING,
    MAX_MACHINES,
    SPEED_LIMITS,
    VALVE,
    ground_geoms,
)

LIFT_POSE = (0.0, -0.65, 1.0, 2.4)
SOURCE_LIMITS_DEG = {"boom": (-70.0, 44.0), "arm": (30.0, 152.0), "bucket": (-33.0, 143.0)}


def home_data(model):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    return data


def set_pose(model, data, index, pose):
    for name, value in pose.items():
        data.qpos[model.joint(machine_prefix(index) + name).qposadr[0]] = value
    mujoco.mj_forward(model, data)


def joint_angles(model, data, index=0):
    return np.array([data.qpos[model.joint(machine_prefix(index) + j).qposadr[0]]
                     for j in JOINT_NAMES])


def base_position(model, data, index):
    return data.xpos[model.body(machine_prefix(index) + "base").id].copy()


def base_up(model, data, index=0):
    return data.xmat[model.body(machine_prefix(index) + "base").id].reshape(3, 3)[:, 2]


def tilt_deg(model, data, index=0):
    return math.degrees(math.acos(min(1.0, base_up(model, data, index)[2])))


def push_base(model, data, index, force, seconds):
    body = model.body(machine_prefix(index) + "base").id
    data.xfrc_applied[body, :3] = force
    settle(model, data, seconds)
    data.xfrc_applied[body] = 0.0


def settle(model, data, seconds=1.0):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def hang(model, data, mass):
    apply_load(model, 0, block(mass), mujoco.MjData(model))
    hang_load(model, data, 0)


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


def test_every_driven_joint_has_a_speed_valve_capped_at_its_force_limit():
    model = load_scene(1)
    prefix = machine_prefix(0)
    for name in JOINT_NAMES:
        act = model.actuator(prefix + name)
        assert act.trnid[0] == model.joint(prefix + name + VALVE).id
        np.testing.assert_allclose(act.ctrlrange, [-SPEED_LIMITS[name], SPEED_LIMITS[name]])
        tendon = model.tendon(prefix + name + VALVE).id
        assert model.tendon_frictionloss[tendon] == FORCE_CAPS[name]
        start, count = model.tendon_adr[tendon], model.tendon_num[tendon]
        assert model.joint(prefix + name).id in model.wrap_objid[start:start + count]


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
        np.testing.assert_array_equal(joint_angles(model, data, i), HOME_POSE)
        np.testing.assert_allclose(base_position(model, data, i), [0.0, MACHINE_SPACING * i, 0.0])


@pytest.mark.parametrize("weld", [False, True])
def test_plain_reset_is_the_home_pose_at_rest(weld):
    model = load_scene(1, weld=weld)
    data = mujoco.MjData(model)
    np.testing.assert_array_equal(data.qpos, model.key(HOME_KEY).qpos)
    mujoco.mj_forward(model, data)
    start = base_position(model, data, 0)
    settle(model, data, 1.0)
    np.testing.assert_allclose(joint_angles(model, data), HOME_POSE, atol=1e-3)
    assert np.linalg.norm(base_position(model, data, 0) - start) < 1e-3


def test_each_machine_stands_on_its_own_ground_with_few_contacts():
    model = load_scene(2)
    data = home_data(model)
    settle(model, data)
    pairs = {}
    for contact in data.contact[: data.ncon]:
        pairs[(contact.geom1, contact.geom2)] = pairs.get((contact.geom1, contact.geom2), 0) + 1
    for (g1, g2), count in pairs.items():
        index, ground, other = on_ground(model, g1, g2)
        prefix = machine_prefix(index)
        name = model.geom(other).name
        assert name.startswith(prefix + "track_") and count <= 4 or name.startswith(
            prefix + LOAD_POINTS)
    assert 0 < data.ncon <= 2 * (24 * 2 + 26)
    for i in range(2):
        assert abs(base_position(model, data, i)[2]) < 1e-3


def test_welded_machines_have_no_contacts():
    model = load_scene(2, weld=True)
    data = home_data(model)
    settle(model, data, 0.5)
    for contact in data.contact[: data.ncon]:
        index, _, other = on_ground(model, contact.geom1, contact.geom2)
        assert model.geom(other).name.startswith(machine_prefix(index) + LOAD_POINTS)


def on_ground(model, g1, g2):
    for index in range(MAX_MACHINES):
        ground = ground_geoms(model, index)
        if g1 in ground or g2 in ground:
            return index, (g1 if g1 in ground else g2), (g2 if g1 in ground else g1)
    raise AssertionError("a contact that does not involve any machine's ground")


def test_free_base_rests_flush_and_tip_matches_kinematics():
    model = load_scene(1)
    data = home_data(model)
    pose = (0.4, -0.5, 1.2, 0.8)
    set_pose(model, data, 0, dict(zip(JOINT_NAMES, pose, strict=True)))
    settle(model, data, 2.0)
    assert abs(base_position(model, data, 0)[2]) < 1e-3
    position, _ = forward_kinematics(joint_angles(model, data))
    np.testing.assert_allclose(data.site(machine_prefix(0) + TIP_SITE).xpos, position, atol=1e-3)


def test_free_base_sits_still_for_ten_seconds():
    model = load_scene(1)
    data = home_data(model)
    settle(model, data, 0.5)
    start = base_position(model, data, 0)
    settle(model, data, 10.0)
    assert np.linalg.norm(base_position(model, data, 0) - start) < 1e-3
    assert tilt_deg(model, data) < 0.01


def test_yawed_base_sits_flat():
    model = load_scene(1)
    data = home_data(model)
    adr = model.joint(machine_prefix(0) + "base_free").qposadr[0]
    data.qpos[adr + 3:adr + 7] = [math.cos(math.pi / 8), 0.0, 0.0, math.sin(math.pi / 8)]
    settle(model, data, 2.0)
    assert tilt_deg(model, data) < 0.01
    assert abs(base_position(model, data, 0)[2]) < 1e-3


def roll_on_ridge(side):
    model = load_scene(1)
    xs, ys = ground_axes()
    heights = np.zeros(ground_shape())
    under_outer_half = (side * ys > 0.95) & (side * ys < 1.35)
    heights[np.ix_(under_outer_half, np.abs(xs) < 2.0)] = 0.05
    write_ground(model, 0, heights)
    data = home_data(model)
    data.qpos[2] = 0.06
    settle(model, data, 2.0)
    up = base_up(model, data)
    return math.degrees(math.atan2(-up[1], up[2]))


def test_ridge_under_one_track_rolls_the_machine_and_mirrors():
    left, right = roll_on_ridge(1.0), roll_on_ridge(-1.0)
    assert left > 0.5
    assert right == pytest.approx(-left, abs=0.05)


def test_ridge_across_the_middle_of_the_tracks_lifts_the_machine():
    model = load_scene(1)
    xs, _ = ground_axes()
    heights = np.zeros(ground_shape())
    heights[:, np.abs(xs) < 0.35] = 0.25
    write_ground(model, 0, heights)
    data = home_data(model)
    data.qpos[2] = 0.3
    settle(model, data, 3.0)
    assert base_position(model, data, 0)[2] > 0.15


@pytest.mark.parametrize("lip", [1.5, 1.4, 1.2, 1.0])
def test_loaded_machine_stays_level_with_track_ends_over_a_drop(lip):
    model = load_scene(1)
    xs, _ = ground_axes()
    heights = np.zeros(ground_shape())
    heights[:, xs > lip] = -1.0
    write_ground(model, 0, heights)
    data = home_data(model)
    set_pose(model, data, 0, dict(zip(JOINT_NAMES, LIFT_POSE, strict=True)))
    hang(model, data, 1200.0)
    settle(model, data, 3.0)
    assert tilt_deg(model, data) < 0.5


def worst_tilt(swing, payload):
    model = load_scene(1)
    data = home_data(model)
    set_pose(model, data, 0, dict(zip(JOINT_NAMES, (swing, *LIFT_POSE[1:]), strict=True)))
    hang(model, data, payload)
    worst, feet = 0.0, []
    for _ in range(round(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if tilt_deg(model, data) > worst:
            worst = tilt_deg(model, data)
            feet = [model.geom_pos[g][0] for c in data.contact[:data.ncon]
                    for g in (c.geom1, c.geom2) if model.geom(g).name.startswith("m0/track_")]
    return worst, feet


def test_loaded_bucket_tilts_alike_swung_left_and_right():
    left, right = (worst_tilt(s, 2200.0)[0] for s in (math.pi / 2, -math.pi / 2))
    assert left < 0.05 and right < 0.05
    left, right = (worst_tilt(s, 3000.0)[0] for s in (math.pi / 2, -math.pi / 2))
    assert left > 1.0 and right > 1.0
    assert left == pytest.approx(right, abs=1.0)


def test_load_beyond_capacity_tips_a_free_base_onto_its_front_track_ends():
    worst, feet = worst_tilt(0.0, 5000.0)
    assert worst > 1.0
    assert feet and all(x > 1.0 for x in feet)


@pytest.mark.parametrize("index", [0, 1])
def test_weld_holds_base(index):
    model = load_scene(2, weld=True)
    data = home_data(model)
    settle(model, data, 0.2)
    start = base_position(model, data, index)
    push_base(model, data, index, [3.0e5, 2.0e5, 3.0e5], 1.0)
    assert np.linalg.norm(base_position(model, data, index) - start) < 0.01


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
    settle(model, data, 0.5)
    assert abs(base_position(model, data, 0)[2]) < 1e-3
    push_base(model, data, 0, [3.0e5, 0.0, 0.0], 0.5)
    settle(model, data, 1.0)
    moved = base_position(model, data, 0)
    assert moved[0] > 0.3
    set_weld(model, data, 0, True)
    push_base(model, data, 0, [-3.0e5, 2.0e5, 0.0], 1.0)
    assert np.linalg.norm(base_position(model, data, 0) - moved) < 0.01


def test_reset_after_rewelding_elsewhere_returns_to_home():
    model = load_scene(1, weld=True)
    data = home_data(model)
    set_weld(model, data, 0, False)
    adr = model.joint(machine_prefix(0) + "base_free").qposadr[0]
    data.qpos[adr:adr + 3] = [23.0, 0.2, -3.5]
    set_weld(model, data, 0, True)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    settle(model, data, 1.0)
    np.testing.assert_allclose(base_position(model, data, 0), [0.0, 0.0, 0.0], atol=0.01)
    assert np.abs(data.qvel).max() < 0.01


def test_two_data_on_one_model_weld_independently():
    model = load_scene(1, weld=True)
    a, b = home_data(model), home_data(model)
    set_weld(model, a, 0, False)
    adr = model.joint(machine_prefix(0) + "base_free").qposadr[0]
    a.qpos[adr:adr + 3] = [3.0, 0.0, 0.5]
    set_weld(model, a, 0, True)
    settle(model, b, 1.0)
    np.testing.assert_allclose(base_position(model, b, 0), [0.0, 0.0, 0.0], atol=0.01)
    settle(model, a, 1.0)
    np.testing.assert_allclose(base_position(model, a, 0), [3.0, 0.0, 0.5], atol=0.01)


def test_bucket_lowered_onto_raised_ground_stops_on_its_surface():
    model = load_scene(1, weld=True)
    data = home_data(model)
    xs, ys = ground_axes()
    heights = np.where(xs[None, :] > 3.0, 1.5, 0.0) * np.ones((len(ys), 1))
    write_ground(model, 0, heights)
    pose = dict(zip(JOINT_NAMES, inverse_kinematics(np.array([5.5, 0.0, 2.3]), 1.0), strict=True))
    set_pose(model, data, 0, pose)
    data.ctrl[model.actuator(machine_prefix(0) + "boom").id] = 0.1
    settle(model, data, 6.0)
    mujoco.mj_rnePostConstraint(model, data)
    assert data.site(machine_prefix(0) + TIP_SITE).xpos[2] == pytest.approx(1.5, abs=0.05)
    bucket = model.body(machine_prefix(0) + "bucket").id
    touching = [c for c in data.contact[:data.ncon]
                if bucket in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])]
    assert touching and min(c.dist for c in touching) > -0.002
    assert data.cfrc_ext[bucket][5] > 0.5 * LINK_MASSES["bucket"] * 9.81


def test_written_ground_heights_appear_in_the_heightfield():
    model = load_scene(1)
    data = home_data(model)
    xs, ys = ground_axes()
    heights = np.zeros(ground_shape())
    col, row = int(np.argmin(np.abs(xs - 6.0))), int(np.argmin(np.abs(ys - 1.0)))
    heights[row - 2 : row + 3, col - 2 : col + 3] = -5.5
    write_ground(model, 0, heights)
    mujoco.mj_forward(model, data)
    geomid = np.array([-1], dtype=np.int32)
    for x, y, expected in ((xs[col], ys[row], -5.5), (2.0, -3.0, 0.0)):
        distance = mujoco.mj_ray(model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]),
                                 None, 1, -1, geomid)
        assert model.geom(int(geomid[0])).name == machine_prefix(0) + GROUND
        assert 5.0 - distance == pytest.approx(expected, abs=1e-5)


def test_write_ground_rejects_bad_input():
    model = load_scene(1)
    with pytest.raises(ValueError):
        write_ground(model, 0, np.zeros((3, 3)))
    with pytest.raises(ValueError):
        write_ground(model, 0, np.zeros(ground_shape()).T)
    with pytest.raises(ValueError):
        write_ground(model, 0, np.full(ground_shape(), -GROUND_DEPTH - 0.1))
    with pytest.raises(ValueError):
        write_ground(model, 0, np.full(ground_shape(), np.nan))


def test_ground_grid_covers_the_work_area_and_depth():
    xs, ys = ground_axes()
    assert ground_shape() == (len(ys), len(xs)) and len(xs) != len(ys)
    assert math.isclose(xs[1] - xs[0], 0.1) and math.isclose(ys[1] - ys[0], 0.1)
    assert xs[0] < -2.5 and xs[-1] > 9.0 and ys[0] < -5.0 and ys[-1] > 5.0
    assert GROUND_DEPTH >= 6.0
