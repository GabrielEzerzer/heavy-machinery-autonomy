import math

import mujoco
import numpy as np
import pytest

from hma.machine import (
    HOME_KEY,
    ground_axes,
    ground_shape,
    load_scene,
    machine_prefix,
    place,
    set_weld,
    step,
)
from hma.machine.ground import RANGES, apply_ground, draw_ground
from hma.machine.measure import (
    CASES,
    base_frame,
    bucket_tip,
    chart_pose,
    hand_tipping_load,
    machine_com,
    released,
    run,
    slope_heights,
    tipping_case,
    tips_under,
    turned,
)
from hma.machine.model import GROUND_CELL
from hma.machine.tracks import REAR_LINE, SIDE_LINE, TRACTION, TRAVEL_SPEED

P = machine_prefix(0)


@pytest.mark.parametrize("name", list(CASES))
def test_bucket_pin_load_tips_the_machine_within_ten_percent_of_hand_statics(name):
    heights, pose, hand = tipping_case(name)
    for factor, tips in ((0.9, False), (1.1, True)):
        model, data = released(heights, friction=(1.0, 1.0), pose=pose, hold=True)
        angle = tips_under(model, data, factor * hand)
        assert (angle > 3.0) if tips else (angle < 0.5), (factor, angle)


@pytest.mark.parametrize("lip", [1.2, 1.0])
def test_track_end_over_a_drop_tips_about_the_edge(lip):
    xs, _ = ground_axes()
    heights = np.zeros(ground_shape())
    heights[:, xs > lip + 1e-9] = -1.0
    pose = (0.0, 0.0, 1.0, 0.5)
    model, data = released(heights, friction=(1.0, 1.0), pose=pose, hold=True)
    origin, rot = base_frame(model, data)
    hand = hand_tipping_load(model, data, origin + rot @ [lip, 0.0, 0.0], rot[:, 1])
    for factor, tips in ((0.95, False), (1.05, True)):
        model, data = released(heights, friction=(1.0, 1.0), pose=pose, hold=True)
        angle = tips_under(model, data, factor * hand)
        assert (angle > 3.0) if tips else (angle < 0.5), (factor, angle)


def test_boom_press_down_lifts_the_front_at_the_predicted_force():
    pose = chart_pose(0.0, 6.0)
    model, data = released(pose=pose, hold=True)
    mass, com = machine_com(model, data)
    tip = data.site(P + "bucket_tip").xpos.copy()
    needed = mass * 9.81 * (com[0] - REAR_LINE) / (tip[0] - REAR_LINE)
    for factor, lifts in ((0.9, False), (1.1, True)):
        model, data = released(pose=pose, hold=True)
        _, start = base_frame(model, data)
        run(model, data, 2.0, np.array([0.0, 0.0, factor * needed]), bucket_tip, body="bucket")
        assert (turned(model, data, start) > 3.0) if lifts else (turned(model, data, start) < 0.5)


@pytest.mark.parametrize("factor, slides", [(0.95, False), (1.05, True)])
def test_machine_slides_down_a_slope_steeper_than_its_friction_angle(factor, slides):
    along = 0.45
    angle = factor * math.atan(along)
    model, data = released(slope_heights(along=angle), friction=(along, along / 0.9))
    run(model, data, 0.5)
    start, _ = base_frame(model, data)
    run(model, data, 2.0)
    moved = np.linalg.norm(base_frame(model, data)[0] - start)
    assert (moved > 0.1) if slides else (moved < 1e-3), moved


@pytest.mark.parametrize("factor, tips", [(0.95, False), (1.05, True)])
def test_machine_tips_sideways_on_a_slope_steeper_than_its_tip_angle(factor, tips):
    model, data = released(friction=(3.0, 3.0), hold=True)
    mass, com = machine_com(model, data)
    origin, _ = base_frame(model, data)
    critical = math.atan((SIDE_LINE + com[1] - origin[1]) / (com[2] - origin[2]))
    model, data = released(slope_heights(across=factor * critical), friction=(3.0, 3.0),
                           hold=True)
    _, start = base_frame(model, data)
    run(model, data, 3.0)
    assert (turned(model, data, start) > 3.0) if tips else (turned(model, data, start) < 0.5)


def test_tracks_drive_forward_and_back_at_the_commanded_speed():
    for command in (TRAVEL_SPEED, -TRAVEL_SPEED, 0.5):
        model, data = released()
        data.ctrl[[model.actuator(P + t).id for t in ("travel_left", "travel_right")]] = command
        run(model, data, 2.0)
        start, _ = base_frame(model, data)
        run(model, data, 1.0)
        speed = (base_frame(model, data)[0] - start)[0]
        assert speed == pytest.approx(command, rel=0.01)


def turn_rate(left_command, right_command):
    model, data = released()
    left, right = (model.actuator(P + t).id for t in ("travel_left", "travel_right"))
    data.ctrl[left], data.ctrl[right] = left_command, right_command
    run(model, data, 2.0)
    start, before = base_frame(model, data)
    run(model, data, 1.0)
    end, after = base_frame(model, data)
    yaw = math.atan2(after[1, 0], after[0, 0]) - math.atan2(before[1, 0], before[0, 0])
    forward = (before.T @ (end - start))[0]
    return yaw, forward


def test_counter_rotation_spins_in_place_and_a_pivot_turns_beyond_the_braked_track():
    gauge = 2 * (SIDE_LINE - 0.07)
    yaw, forward = turn_rate(-1.0, 1.0)
    assert 0.2 < yaw < 2.0 / gauge
    assert abs(forward) < 0.2 * yaw * gauge / 2
    yaw, forward = turn_rate(0.0, 1.0)
    assert 0.1 < yaw < 1.0 / gauge
    assert gauge / 2 <= forward / yaw < 2 * gauge


@pytest.mark.parametrize("pull, keeps_going", [(0.9, True), (1.1, False)])
def test_drawbar_pull_is_capped_at_the_published_traction(pull, keeps_going):
    model, data = released(friction=(1.0, 1.0))
    data.ctrl[[model.actuator(P + t).id for t in ("travel_left", "travel_right")]] = 1.0
    run(model, data, 2.0)
    data.xfrc_applied[model.body(P + "base").id, 0] = -pull * 2 * TRACTION
    run(model, data, 1.0)
    start, _ = base_frame(model, data)
    run(model, data, 1.0)
    speed = (base_frame(model, data)[0] - start)[0]
    assert (speed == pytest.approx(1.0, rel=0.01)) if keeps_going else speed < 0.9


def test_parking_brake_holds_on_a_twenty_degree_slope():
    model, data = released(slope_heights(along=math.radians(20.0)), friction=(0.45, 0.5))
    run(model, data, 2.0)
    start, _ = base_frame(model, data)
    run(model, data, 5.0)
    assert np.linalg.norm(base_frame(model, data)[0] - start) < 1e-3


def test_machine_at_rest_gains_no_energy_and_does_not_sink():
    model, data = released()
    run(model, data, 1.0)
    start, _ = base_frame(model, data)
    peak = 0.0
    for _ in range(20):
        run(model, data, 0.5)
        mujoco.mj_energyVel(model, data)
        peak = max(peak, float(data.energy[1]))
    assert peak < 1e-2
    assert np.linalg.norm(base_frame(model, data)[0] - start) < 1e-4


@pytest.mark.parametrize("seed", [3, 11, 29])
def test_machine_rests_on_drawn_rough_ground(seed):
    model = load_scene(1)
    apply_ground(model, 0, draw_ground(seed))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    place(model, data, 0)
    run(model, data, 1.0)
    start, before = base_frame(model, data)
    run(model, data, 2.0)
    assert np.linalg.norm(base_frame(model, data)[0] - start) < 5e-3
    assert turned(model, data, before) < 0.2
    assert np.abs(data.qvel).max() < 0.01


def test_same_commands_on_the_same_ground_repeat_bit_for_bit():
    def drive():
        model = load_scene(2)
        for i in range(2):
            apply_ground(model, i, draw_ground(7 + i))
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
        for i in range(2):
            place(model, data, i)
        rng = np.random.default_rng(0)
        for _ in range(20):
            data.ctrl[:] = rng.uniform(-1.0, 1.0, model.nu) * model.actuator_ctrlrange[:, 1]
            step(model, data, 20)
        return data.qpos.copy(), data.qvel.copy()

    first, second = drive(), drive()
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_ground_draws_repeat_and_stay_in_range():
    a, b, c = draw_ground(5), draw_ground(5), draw_ground(6)
    assert np.array_equal(a.heights, b.heights) and a.along == b.along
    assert not np.array_equal(a.heights, c.heights)
    assert a.heights.shape == ground_shape()
    for seed in range(40):
        ground = draw_ground(seed)
        assert RANGES["friction"][0] <= ground.along <= RANGES["friction"][1]
        assert ground.across > ground.along
    with pytest.raises(ValueError):
        a.heights[0, 0] = 1.0


def test_step_edges_fall_on_grid_lines_so_the_height_field_holds_them():
    xs, _ = ground_axes()
    heights = np.zeros(ground_shape())
    heights[:, xs > 2.0 + GROUND_CELL / 2] = -0.3
    model, data = released(heights)
    assert data.qpos[2] == pytest.approx(0.0, abs=1e-9)


def test_welded_machine_gets_no_track_contacts_and_release_writes_no_model_fields():
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    place(model, data, 0)
    step(model, data, 10)
    assert data.ncon == 0
    before = {name: getattr(model, name).copy() for name in
              ("geom_contype", "geom_conaffinity", "body_contype", "body_conaffinity", "eq_data")}
    set_weld(model, data, 0, False)
    step(model, data, 10)
    assert data.ncon > 0
    for name, values in before.items():
        assert np.array_equal(getattr(model, name), values), name
