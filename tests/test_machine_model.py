import math

import mujoco
import numpy as np
import pytest

from hma.machine import (
    BODY_NAMES,
    GROUND,
    HOME_KEY,
    HOME_POSE,
    JOINT_LIMITS,
    JOINT_NAMES,
    LINK_MASSES,
    TIP_SITE,
    Ground,
    forward_kinematics,
    inverse_kinematics,
    load_scene,
    machine_prefix,
    set_weld,
)
from hma.machine import settle as set_down
from hma.machine.ground import Box, apply_ground, draw_ground
from hma.machine.load import apply_load, block, hang_load
from hma.machine.model import (
    GROUND_BOXES,
    LOAD_POINTS,
    MACHINE_SPACING,
    MAX_MACHINES,
    ground_geoms,
    ground_height,
)
from hma.machine.tracks import BELT_MASS, BELT_ROLLERS, WHEEL_MASSES, WHEELS, step
from scenes import held, lock_joints

LIFT_POSE = (0.0, -0.65, 1.0, 2.4)
LOWER = 0.3
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
    layer = held(model, data)
    for _ in range(round(seconds / model.opt.timestep)):
        layer.step()


def hang(model, data, mass):
    apply_load(model, 0, block(mass), mujoco.MjData(model))
    hang_load(model, data, 0)


def test_total_mass_in_operating_weight_band():
    model = load_scene(2)
    for i in range(2):
        masses = {b: model.body(machine_prefix(i) + b).mass[0] for b in BODY_NAMES}
        assert masses == pytest.approx(LINK_MASSES)
        total = model.body_subtreemass[model.body(machine_prefix(i) + "base").id]
        lower = 2 * (sum(WHEEL_MASSES) + BELT_ROLLERS * BELT_MASS)
        assert total == pytest.approx(sum(masses.values()) + lower)
        assert 12_000.0 <= total <= 12_600.0


def test_joint_limits_match_source():
    model = load_scene(1)
    for name, (lo, hi) in SOURCE_LIMITS_DEG.items():
        joint_range = model.joint(machine_prefix(0) + name).range
        np.testing.assert_allclose(np.degrees(joint_range), [lo, hi], atol=1e-6)
        np.testing.assert_allclose(joint_range, JOINT_LIMITS[name], atol=1e-12)
    assert not model.joint(machine_prefix(0) + "swing").limited


def test_every_driven_joint_has_an_unlimited_torque_motor_on_the_joint():
    model = load_scene(1)
    prefix = machine_prefix(0)
    for name in JOINT_NAMES:
        act = model.actuator(prefix + name)
        assert act.trntype == mujoco.mjtTrn.mjTRN_JOINT
        assert act.trnid[0] == model.joint(prefix + name).id
        assert act.gainprm[0] == 1.0 and not act.biasprm.any()
        assert not act.ctrllimited and not act.forcelimited
    joint_wraps = model.wrap_type == mujoco.mjtWrap.mjWRAP_JOINT
    driven = [model.joint(prefix + name).id for name in JOINT_NAMES]
    assert not np.isin(model.wrap_objid[joint_wraps], driven).any()


def test_names_are_prefixed_per_machine():
    model = load_scene(2)
    for i in range(2):
        prefix = machine_prefix(i)
        for name in BODY_NAMES:
            model.body(prefix + name)
        for name in JOINT_NAMES:
            model.joint(prefix + name)
            model.actuator(prefix + name)
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
        index, _, other = ground_side(model, g1, g2)
        prefix = machine_prefix(index)
        name = model.geom(other).name
        assert name.startswith(prefix + "track_") and count <= 2 or name.startswith(
            prefix + LOAD_POINTS)
    assert 0 < data.ncon <= 2 * (2 * 2 * WHEELS + 26)
    for i in range(2):
        assert abs(base_position(model, data, i)[2]) < 1e-3


def test_welded_scenes_have_no_ground_contact_and_refuse_a_release():
    model = load_scene(2, weld=True)
    data = home_data(model)
    settle(model, data, 0.5)
    assert model.npair == 0
    for contact in data.contact[: data.ncon]:
        index, _, other = ground_side(model, contact.geom1, contact.geom2)
        assert model.geom(other).name.startswith(machine_prefix(index) + LOAD_POINTS)
    with pytest.raises(ValueError):
        set_weld(model, data, 0, False)


def ground_side(model, g1, g2):
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


def on_ground(*boxes, pose=None):
    model = load_scene(1)
    apply_ground(model, 0, Ground(boxes=boxes))
    data = home_data(model)
    if pose is not None:
        set_pose(model, data, 0, dict(zip(JOINT_NAMES, pose, strict=True)))
    assert set_down(model, data, 0)
    mujoco.mj_forward(model, data)
    return model, data


def roll_on_ridge(side):
    model, data = on_ground(Box(0.0, side * 1.15, 0.0, 2.0, 0.2, 0.05))
    settle(model, data, 2.0)
    up = base_up(model, data)
    return math.degrees(math.atan2(-up[1], up[2]))


def test_ridge_under_one_track_rolls_the_machine_and_mirrors():
    left, right = roll_on_ridge(1.0), roll_on_ridge(-1.0)
    assert left > 0.5
    assert right == pytest.approx(-left, abs=0.05)


def test_ridge_across_the_middle_of_the_tracks_lifts_the_machine():
    model, data = on_ground(Box(0.0, 0.0, 0.0, 0.35, 3.0, 0.25))
    settle(model, data, 3.0)
    assert base_position(model, data, 0)[2] > 0.15


@pytest.mark.parametrize("lip", [1.5, 1.4, 1.2, 1.0])
def test_loaded_machine_stays_level_with_track_ends_over_a_drop(lip):
    model = load_scene(1)
    apply_ground(model, 0, Ground(level=-1.0, boxes=(Box(lip - 6.0, 0.0, 0.0, 6.0, 6.0, 1.0),)))
    data = home_data(model)
    set_pose(model, data, 0, dict(zip(JOINT_NAMES, LIFT_POSE, strict=True)))
    assert set_down(model, data, 0)
    hang(model, data, 1200.0)
    settle(model, data, 3.0)
    assert tilt_deg(model, data) < 0.5


def worst_tilt(swing, payload):
    model = load_scene(1)
    data = home_data(model)
    set_pose(model, data, 0, dict(zip(JOINT_NAMES, (swing, *LIFT_POSE[1:]), strict=True)))
    hang(model, data, payload)
    lock_joints(model, data)
    base = model.body(machine_prefix(0) + "base").id
    worst, feet = 0.0, []
    for _ in range(round(3.0 / model.opt.timestep)):
        step(model, data)
        if tilt_deg(model, data) > worst:
            worst = tilt_deg(model, data)
            rot = data.xmat[base].reshape(3, 3)
            feet = [((data.geom_xpos[g] - data.xpos[base]) @ rot)[0]
                    for c in data.contact[:data.ncon] for g in (c.geom1, c.geom2)
                    if model.geom(g).name.startswith("m0/track_")]
    return worst, feet


def test_loaded_bucket_tilts_alike_swung_left_and_right():
    left, right = (worst_tilt(s, 1800.0)[0] for s in (math.pi / 2, -math.pi / 2))
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
    model = load_scene(1)
    data = home_data(model)
    set_weld(model, data, 0, True)
    set_weld(model, data, 0, False)
    settle(model, data, 0.5)
    assert abs(base_position(model, data, 0)[2]) < 1e-3
    push_base(model, data, 0, [3.0e5, 0.0, 0.0], 0.2)
    settle(model, data, 2.0)
    assert np.abs(data.qvel).max() < 0.01
    moved = base_position(model, data, 0)
    assert moved[0] > 0.3
    set_weld(model, data, 0, True)
    push_base(model, data, 0, [-3.0e5, 2.0e5, 0.0], 1.0)
    assert np.linalg.norm(base_position(model, data, 0) - moved) < 0.01


def test_switching_the_weld_at_run_time_writes_nothing_into_the_model():
    model = load_scene(1)
    data = home_data(model)
    fields = ("body_contype", "body_conaffinity", "geom_contype", "geom_conaffinity",
              "pair_geom1", "pair_geom2", "pair_friction", "eq_active0")
    before = {name: getattr(model, name).copy() for name in fields}
    set_weld(model, data, 0, True)
    settle(model, data, 0.5)
    set_weld(model, data, 0, False)
    settle(model, data, 0.5)
    for name in fields:
        assert np.array_equal(getattr(model, name), before[name]), name


def test_reset_after_rewelding_elsewhere_returns_to_home():
    model = load_scene(1)
    data = home_data(model)
    adr = model.joint(machine_prefix(0) + "base_free").qposadr[0]
    data.qpos[adr:adr + 3] = [23.0, 0.2, -3.5]
    set_weld(model, data, 0, True)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    settle(model, data, 1.0)
    np.testing.assert_allclose(base_position(model, data, 0), [0.0, 0.0, 0.0], atol=0.01)
    assert np.abs(data.qvel).max() < 0.01


def test_two_data_on_one_model_weld_independently():
    model = load_scene(1)
    a, b = home_data(model), home_data(model)
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
    held(model, data)
    apply_ground(model, 0, Ground(boxes=(Box(9.0, 0.0, 0.0, 6.0, 6.0, 1.5),)))
    pose = dict(zip(JOINT_NAMES, inverse_kinematics(np.array([5.5, 0.0, 2.3]), 1.0), strict=True))
    set_pose(model, data, 0, pose)
    held(model, data).act(np.array([[0.0, LOWER, 0.0, 0.0]]))
    settle(model, data, 6.0)
    mujoco.mj_rnePostConstraint(model, data)
    assert data.site(machine_prefix(0) + TIP_SITE).xpos[2] == pytest.approx(1.5, abs=0.05)
    bucket = model.body(machine_prefix(0) + "bucket").id
    touching = [c for c in data.contact[:data.ncon]
                if bucket in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2])]
    assert touching and min(c.dist for c in touching) > -0.002
    assert data.cfrc_ext[bucket][5] > 0.5 * LINK_MASSES["bucket"] * 9.81


@pytest.mark.parametrize("seed", [2, 6, 8, 13])
def test_drawn_ground_heights_appear_where_the_collision_finds_them(seed):
    model = load_scene(1)
    apply_ground(model, 0, draw_ground(seed))
    data = home_data(model)
    data.qpos[2] = -100.0
    mujoco.mj_forward(model, data)
    rng = np.random.default_rng(seed)
    points = np.column_stack([rng.uniform(-3.0, 4.0, 40), rng.uniform(-4.0, 4.0, 40)])
    expected = ground_height(model, 0, points)
    geomid = np.array([-1], dtype=np.int32)
    for (x, y), height in zip(points, expected, strict=True):
        distance = mujoco.mj_ray(model, data, np.array([x, y, 5.0]), np.array([0.0, 0.0, -1.0]),
                                 None, 1, -1, geomid)
        assert model.geom(int(geomid[0])).name.startswith(machine_prefix(0) + GROUND)
        assert 5.0 - distance == pytest.approx(height, abs=1e-6)


def test_ground_input_is_checked():
    model = load_scene(1)
    with pytest.raises(ValueError):
        box = Box(0.0, 0.0, 0.0, 1.0, 1.0, 0.1)
        apply_ground(model, 0, Ground(boxes=(box,) * (GROUND_BOXES + 1)))
    with pytest.raises(ValueError):
        apply_ground(model, 0, Ground(grade=(math.nan, 0.0)))
    with pytest.raises(ValueError):
        apply_ground(model, 0, Ground(friction=0.0))
    with pytest.raises(ValueError):
        draw_ground(-1)
