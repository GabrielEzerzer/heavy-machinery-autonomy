import math

import mujoco
import numpy as np
import pytest

from hma.hydraulics.params import NOMINAL
from hma.machine import HOME_KEY, JOINT_NAMES, ground_axes, ground_shape, load_scene, write_ground
from hma.machine.contact import steel_geoms
from hma.machine.kinematics import (
    ARM_LENGTH,
    BOOM_LENGTH,
    BOOM_PIVOT,
    BUCKET_LENGTH,
    SWING_HEIGHT,
)
from hma.machine.load import (
    HOOK_REACH,
    LIFT_CLEARANCE,
    LIFT_HEIGHTS,
    LOAD_FIELDS,
    RANGES,
    Load,
    apply_load,
    block,
    draw_load,
    hang_load,
    hook_load,
    hooked,
    place_load,
    sling_tension,
    unhook_load,
)
from hma.machine.model import (
    CONTACT_MARGIN,
    GROUND_HALF,
    LOAD,
    LOAD_HOME,
    LOAD_POINT_RADIUS,
    LOAD_POINTS,
    TRACK_FEET,
    ground_geoms,
    ground_overlap,
    machine_prefix,
)
from hma.machine.statics import CHART
from scenes import held, lock_joints

P = machine_prefix(0)
HANG_POSE = (0.0, -1.0, 1.6, 2.3)
LIFT_POSE = (0.0, -0.6, 1.6, 2.3)
CROSSING_POSE = (0.0, -1.0, 1.2, 2.0)
BURIED_POSE = (0.0, -0.2, 2.0, 2.3)
SEEDS = range(2000)
G = 9.81
SAG = 2e-4
STILL = 1e-3
RAISE = -1.0


def scene(mass=2000.0, n=1, weld=True):
    model = load_scene(n, weld=weld)
    for i in range(n):
        apply_load(model, i, block(mass), mujoco.MjData(model))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    return model, data


def pose(model, data, angles, index=0):
    for name, value in zip(JOINT_NAMES, angles, strict=True):
        data.qpos[model.joint(machine_prefix(index) + name).qposadr[0]] = value
    mujoco.mj_forward(model, data)


def hook_xy(model, data, index=0):
    hook = data.site_xpos[model.site(machine_prefix(index) + "hook").id]
    return hook[0], hook[1]


def placed_and_hooked(model, data):
    pose(model, data, LIFT_POSE)
    place_load(model, data, 0, *hook_xy(model, data))
    hook_load(model, data, 0)


def body_speed(model, data, name):
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, model.body(name).id,
                             velocity, 0)
    return float(np.linalg.norm(velocity[3:]))


def run(model, data, seconds):
    layer = held(model, data)
    for _ in range(round(seconds / model.opt.timestep)):
        layer.step()


def rigid(model, data, seconds):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)


def snapshot(model, data):
    state = np.zeros(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(model, data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    fields = {k: getattr(model, k).copy() for k in dir(model)
              if not k.startswith("_") and isinstance(getattr(model, k), np.ndarray)}
    return state, fields


def assert_refused(model, data, call, match):
    before = snapshot(model, data)
    with pytest.raises(ValueError, match=match):
        call(model, data, 0)
    after = snapshot(model, data)
    np.testing.assert_array_equal(before[0], after[0])
    for key, value in before[1].items():
        np.testing.assert_array_equal(value, after[1][key], err_msg=key)


def test_reset_returns_a_clean_unhooked_load_at_rest():
    model, data = scene()
    pose(model, data, HANG_POSE)
    hang_load(model, data, 0)
    load = model.body(P + LOAD).id
    data.xfrc_applied[load, :3] = [1e4, 0.0, 1e4]
    run(model, data, 1.0)
    held(model, data).reset(0, NOMINAL)
    assert not hooked(model, data, 0)
    assert not data.xfrc_applied[load].any()
    run(model, data, 0.5)
    assert sling_tension(model, data, 0) == 0.0
    assert body_speed(model, data, P + LOAD) < 1e-4
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    assert not hooked(model, data, 0)


def test_two_datas_on_one_model_hold_their_own_hooked_state():
    model, first = scene()
    second = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, second, model.key(HOME_KEY).id)
    pose(model, first, HANG_POSE)
    pose(model, second, HANG_POSE)
    hang_load(model, first, 0)
    run(model, first, 0.5)
    run(model, second, 0.5)
    assert hooked(model, first, 0) and not hooked(model, second, 0)
    assert sling_tension(model, first, 0) > 0.0 and sling_tension(model, second, 0) == 0.0
    assert body_speed(model, second, P + LOAD) < 1e-4


def test_hooked_state_saves_and_restores_with_mujoco_state():
    model, data = scene()
    pose(model, data, HANG_POSE)
    hang_load(model, data, 0)
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.zeros(mujoco.mj_stateSize(model, kind))
    lock_joints(model, data)
    mujoco.mj_getState(model, data, state, kind)
    rigid(model, data, 0.5)
    expected = data.qpos.copy()
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_setState(model, data, state, kind)
    assert hooked(model, data, 0)
    rigid(model, data, 0.5)
    np.testing.assert_array_equal(data.qpos, expected)


def test_refused_hangs_leave_model_and_data_unchanged():
    model, data = scene()
    pose(model, data, BURIED_POSE)
    assert_refused(model, data, hang_load, "inside the machine or the ground")
    pose(model, data, CROSSING_POSE)
    assert_refused(model, data, hang_load, "pass through the machine")
    xs, _ = ground_axes()
    heights = np.zeros(ground_shape())
    heights[:, xs > 3.0] = 2.0
    write_ground(model, 0, heights)
    pose(model, data, (0.0, -0.8, 1.6, 2.3))
    assert_refused(model, data, hang_load, "inside the machine or the ground")


def test_hang_refuses_a_block_inside_the_machine():
    model, data = scene(3000.0)
    pose(model, data, (math.pi, 0.0, 2.6, 2.3))
    assert_refused(model, data, hang_load, "inside the machine or the ground")


def test_refused_hooks_leave_model_and_data_unchanged():
    model, data = scene()
    pose(model, data, LIFT_POSE)
    assert_refused(model, data, hook_load, "too far")
    place_load(model, data, 0, *hook_xy(model, data))
    q0 = model.joint(P + LOAD).qposadr[0]
    upright = data.qpos[q0:q0 + 7].copy()
    data.qpos[q0 + 2] += 1.0
    mujoco.mj_forward(model, data)
    assert_refused(model, data, hook_load, "above the top")
    data.qpos[q0:q0 + 7] = upright
    data.qpos[q0 + 3:q0 + 7] = [0.0, 1.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)
    assert_refused(model, data, hook_load, "above the top")
    data.qpos[q0:q0 + 7] = upright
    pose(model, data, (0.0, -0.5, 1.4, 2.0))
    place_load(model, data, 0, *hook_xy(model, data))
    assert_refused(model, data, hook_load, "pass through the machine")
    placed_and_hooked(model, data)
    assert_refused(model, data, hook_load, "already hooked")


def test_refused_unhooks_leave_model_and_data_unchanged():
    model, data = scene()
    assert_refused(model, data, unhook_load, "not hooked")
    pose(model, data, HANG_POSE)
    hang_load(model, data, 0)
    run(model, data, 0.5)
    assert_refused(model, data, unhook_load, "still carrying")
    model, data = scene()
    placed_and_hooked(model, data)
    run(model, data, 0.5)
    data.qvel[model.joint(P + LOAD).dofadr[0]] = 0.05
    assert_refused(model, data, unhook_load, "still moving")


def test_refused_placements_leave_model_and_data_unchanged():
    model, data = scene()
    pose(model, data, (0.0, -0.45, 1.6, 2.3))
    assert_refused(model, data, lambda m, d, i: place_load(m, d, i, *hook_xy(m, d)),
                   "against the machine")
    assert_refused(model, data, lambda m, d, i: place_load(m, d, i, GROUND_HALF + 5.0, 0.0),
                   "this machine's ground")
    xs, _ = ground_axes()
    inset = model.geom_size[model.geom(P + LOAD).id][0] - LOAD_POINT_RADIUS
    heights = np.zeros(ground_shape())
    heights[:, np.abs(xs - (5.0 + inset / 2)) < 0.06] = 0.3
    write_ground(model, 0, heights)
    assert_refused(model, data, lambda m, d, i: place_load(m, d, i, 5.0, 4.0),
                   "between the load's contact spheres")
    model, data = scene()
    placed_and_hooked(model, data)
    assert_refused(model, data, lambda m, d, i: place_load(m, d, i, 7.0, 0.0), "unhook")


def test_apply_load_rejects_unphysical_inertia_and_thin_slabs():
    model, _ = scene()
    off_centre = Load(1000.0, (0.5, 0.5, 0.06), (0.45, 0.45, 0.0), 0.5, 0.5)
    with pytest.raises(ValueError, match="inertia"):
        apply_load(model, 0, off_centre, mujoco.MjData(model))
    slab = Load(200.0, (0.6, 0.6, 0.8 * LOAD_POINT_RADIUS), (0.0, 0.0, 0.0), 0.5, 0.5)
    with pytest.raises(ValueError, match="contact spheres"):
        apply_load(model, 0, slab, mujoco.MjData(model))


def resting_still(model, data, seconds=1.0):
    load = model.body(P + LOAD).id
    start = data.xpos[load].copy()
    speeds = []
    layer = held(model, data)
    for _ in range(round(seconds / model.opt.timestep)):
        layer.step()
        speeds.append(body_speed(model, data, P + LOAD))
    return float(np.linalg.norm(data.xpos[load] - start)), max(speeds)


def test_reset_and_placement_start_the_load_at_rest():
    model, data = scene(3000.0)
    moved, fastest = resting_still(model, data)
    assert moved < SAG and fastest < STILL
    place_load(model, data, 0, 5.0, 3.0, 0.7)
    moved, fastest = resting_still(model, data)
    assert moved < SAG and fastest < STILL


def test_load_placed_on_a_slope_rests_tilted_with_it():
    model, data = scene(3000.0)
    slope = math.radians(10.0)
    xs, _ = ground_axes()
    write_ground(model, 0, np.zeros(ground_shape()) + math.tan(slope) * (xs[None, :] - 5.0))
    place_load(model, data, 0, 7.0, 2.0)
    up = data.xmat[model.body(P + LOAD).id].reshape(3, 3)[:, 2]
    assert math.degrees(math.acos(up[2])) == pytest.approx(10.0, abs=0.01)
    moved, fastest = resting_still(model, data)
    assert moved < SAG and fastest < STILL


def test_reset_puts_the_load_on_newly_written_ground():
    model, data = scene(3000.0)
    write_ground(model, 0, np.full(ground_shape(), 1.0))
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    half = model.geom_size[model.geom(P + LOAD).id]
    assert data.xpos[model.body(P + LOAD).id][2] == pytest.approx(1.0 + half[2] + CONTACT_MARGIN,
                                                                  abs=1e-6)
    moved, fastest = resting_still(model, data)
    assert moved < SAG and fastest < STILL



def ridge(x):
    xs, _ = ground_axes()
    heights = np.zeros(ground_shape())
    heights[:, int(np.argmin(np.abs(xs - x)))] = 0.3
    return heights


def test_ground_or_load_with_no_rest_at_home_is_refused_and_changes_nothing():
    model, data = scene(2000.0)
    small = model.geom_size[model.geom(P + LOAD).id].copy()
    ground, key = model.hfield_data.copy(), model.key_qpos.copy()
    with pytest.raises(ValueError, match="home position"):
        write_ground(model, 0, ridge(LOAD_HOME[0] + small[0] / 2))
    np.testing.assert_array_equal(model.hfield_data, ground)
    np.testing.assert_array_equal(model.key_qpos, key)
    write_ground(model, 0, ridge(LOAD_HOME[0] + small[0] + 0.2))
    fields = {name: getattr(model, name).copy() for name in LOAD_FIELDS}
    with pytest.raises(ValueError, match="home position"):
        apply_load(model, 0, block(5000.0, length_ratio=2.0), mujoco.MjData(model))
    for name, value in fields.items():
        np.testing.assert_array_equal(getattr(model, name), value, err_msg=name)


def test_reset_rests_the_load_on_uneven_ground():
    model, data = scene(3000.0)
    xs, ys = ground_axes()
    write_ground(model, 0, 0.06 * np.abs(xs[None, :] - 7.2) + 0.03 * ys[:, None])
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    assert ground_overlap(model, data, 0) < 0
    moved, _ = resting_still(model, data)
    assert moved < 1e-3 and body_speed(model, data, P + LOAD) < STILL

def collides(model, a, b):
    ba, bb = model.geom_bodyid[a], model.geom_bodyid[b]
    if not (model.geom_contype[a] & model.geom_conaffinity[b]
            or model.geom_contype[b] & model.geom_conaffinity[a]):
        return False
    if min(ba, bb) > 0 and (model.body_parentid[ba] == bb or model.body_parentid[bb] == ba):
        return False
    signature = (min(ba, bb) << 16) + max(ba, bb)
    return signature not in set(model.exclude_signature.tolist())


def test_the_load_meets_each_other_shape_through_one_kind_of_its_own():
    model, _ = scene()
    box = model.geom(P + LOAD).id
    spheres = [g for g in range(model.ngeom) if model.geom(g).name.startswith(P + LOAD_POINTS)]
    feet = [g for g in range(model.ngeom) if model.geom(g).name.startswith(P + TRACK_FEET)]
    pieces = steel_geoms(model, 0)
    for piece in pieces:
        assert collides(model, piece, box) and not any(collides(model, piece, s) for s in spheres)
    for foot in feet:
        assert not collides(model, foot, box) and not any(collides(model, foot, s) for s in spheres)
    for ground in ground_geoms(model, 0):
        assert not collides(model, ground, box) and all(collides(model, ground, s) for s in spheres)


def test_every_draw_fits_below_the_hook_within_the_side_rating():
    for seed in SEEDS:
        load = draw_load(seed)
        half = np.array(load.half)
        assert RANGES["mass"][0] <= load.mass <= RANGES["mass"][1]
        drop = math.sqrt(load.leg ** 2 - half[0] ** 2 - half[1] ** 2)
        assert any(load.mass <= rating + 1e-9
                   and drop + 2 * half[2] <= height - HOOK_REACH - LIFT_CLEARANCE + 1e-9
                   for height, rating in LIFT_HEIGHTS.items())



def leg_drop(load):
    return math.sqrt(load.leg ** 2 - load.half[0] ** 2 - load.half[1] ** 2)


def pin_at(model, data, radius, height):
    px, pz = radius - BOOM_PIVOT[0], SWING_HEIGHT + BOOM_PIVOT[2] - height
    arm = math.acos((px * px + pz * pz - BOOM_LENGTH ** 2 - ARM_LENGTH ** 2)
                    / (2 * BOOM_LENGTH * ARM_LENGTH))
    boom = math.atan2(pz, px) - math.atan2(ARM_LENGTH * math.sin(arm),
                                           BOOM_LENGTH + ARM_LENGTH * math.cos(arm))
    pose(model, data, (math.pi / 2, boom, arm, 2.4))


def test_heaviest_and_tallest_draws_are_hooked_and_lifted_clear_over_the_side():
    radius = {h: max((float(side.strip("*")), r) for (hh, r), (_, side) in CHART.items()
                     if hh == h)[1] for h in LIFT_HEIGHTS}
    loads = sorted((draw_load(s) for s in SEEDS), key=lambda d: d.mass)[-2:]
    loads += sorted((draw_load(s) for s in SEEDS), key=lambda d: 2 * d.half[2] + leg_drop(d))[-2:]
    for load in loads:
        height = min(h for h, kg in LIFT_HEIGHTS.items() if load.mass <= kg
                     and leg_drop(load) + 2 * load.half[2] <= h - HOOK_REACH - LIFT_CLEARANCE)
        model = load_scene(1)
        apply_load(model, 0, load, mujoco.MjData(model))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
        target, pin = 2 * load.half[2] + 0.9 * leg_drop(load), height
        for _ in range(4):
            pin_at(model, data, radius[height], pin)
            pin += target - data.site_xpos[model.site(P + "hook").id][2]
        place_load(model, data, 0, *hook_xy(model, data))
        hook_load(model, data, 0)
        body, base = model.body(P + LOAD).id, model.body(P + "base").id
        layer = held(model, data)
        layer.act(np.array([[0.0, RAISE, 0.0, 0.0]]))
        for _ in range(round(3.0 / model.opt.timestep)):
            layer.step()
            assert data.xmat[base][8] > math.cos(math.radians(0.5))
            if data.xipos[body][2] - load.half[2] > LIFT_CLEARANCE + CONTACT_MARGIN:
                break
        assert data.xipos[body][2] - load.half[2] > LIFT_CLEARANCE + CONTACT_MARGIN
        anchor = data.xanchor[model.joint(P + "bucket").id]
        assert anchor[2] <= height and math.hypot(anchor[0], anchor[1]) <= radius[height]
        layer.act(np.zeros((1, 4)))
        run(model, data, 1.0)
        tension = []
        for _ in range(round(1.0 / model.opt.timestep)):
            layer.step()
            tension.append(sling_tension(model, data, 0))
        assert np.mean(tension) == pytest.approx(
            load.mass * G / math.cos(load.leg_angle), rel=0.02)

def test_ground_reaches_past_the_hook_plus_the_longest_sling_and_block():
    hook_reach = BOOM_PIVOT[0] + BOOM_LENGTH + ARM_LENGTH + HOOK_REACH
    tip_reach = BOOM_PIVOT[0] + BOOM_LENGTH + ARM_LENGTH + BUCKET_LENGTH
    farthest = max(draw_load(s).leg + 2 * np.linalg.norm(draw_load(s).half) for s in SEEDS)
    assert GROUND_HALF >= max(hook_reach + farthest, tip_reach)
    model, data = scene()
    for swing in np.radians(np.arange(0.0, 360.0, 30.0)):
        pose(model, data, (swing, -0.6, 1.6, 2.3))
        xy = np.array(hook_xy(model, data))
        assert np.all(np.abs(xy) + farthest <= GROUND_HALF)

