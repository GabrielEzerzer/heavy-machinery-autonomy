import itertools

import mujoco
import numpy as np
import pytest

from hma.machine import FORCE_CAPS, JOINT_NAMES, load_scene, machine_prefix
from hma.machine.kinematics import inverse_kinematics
from hma.soil import (
    BUCKET_CAPACITY,
    NOMINAL,
    Soil,
    SoilParams,
    joint_load,
    physics_steps,
    sample_soil,
    step,
    sync_viewer,
)
from hma.soil.demo import DRAG_FROM, DRAG_TO, PITCH_DIG, SWING_TO, cycle, start_pose

PREFIX = machine_prefix(0)
STIFF = SoilParams(105.0e3, 105.0e3, 0.8, 22.0e3, 0.4, 300.0)
LOOSE = SoilParams(0.0, 0.0, 0.3, 17.0e3, 0.2, 0.0)
BURIED = (np.array([6.0, 0.0, -0.25]), PITCH_DIG)


def machine(params=NOMINAL):
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    return model, data, Soil(model, 0, params)


def run_cycle(params=NOMINAL, control_hz=10.0, heights=None, limit=None):
    model, data, soil = machine(params)
    start_pose(model, data)
    soil.reset(model, data, heights)
    start = soil.total_volume()
    trace = []
    for phase in itertools.islice(cycle(model, data, soil, control_hz), limit):
        loads = {j: joint_load(model, data, 0, j) for j in JOINT_NAMES}
        trace.append({"phase": phase, "loads": loads, "bucket": soil.bucket_volume,
                      "weight": float(data.xfrc_applied[soil.bucket, 2]),
                      "tip": data.site_xpos[soil.tip].copy()})
    return soil, start, trace


def place(model, data, soil, point, pitch):
    mujoco.mj_resetData(model, data)
    data.qpos[[model.joint(PREFIX + j).qposadr[0] for j in JOINT_NAMES]] = (
        inverse_kinematics(point, pitch))
    soil.reset(model, data)


def actuators(model):
    return [model.actuator(PREFIX + j).id for j in JOINT_NAMES]


def advance(model, data, soil, steps):
    for _ in range(steps):
        soil.substep(model, data)
        mujoco.mj_step(model, data)


@pytest.fixture(scope="module")
def nominal():
    return run_cycle()


def cells_near(soil, x, y, radius):
    xs, ys = soil.grid.axes()
    return soil.heights[np.hypot(*np.meshgrid(xs - x, ys - y)) < radius]


def test_cycle_conserves_volume_exactly(nominal):
    soil, start, trace = nominal
    moved = -soil.heights[soil.heights < 0].sum() * soil.grid.cell_area
    assert soil.lost_volume == 0.0
    assert soil.bucket_volume == 0.0
    assert abs(soil.total_volume() - start) < 1e-9
    assert abs(soil.total_volume() - start) <= 0.02 * moved


def test_cycle_leaves_a_trench_and_a_pile(nominal):
    soil, _, trace = nominal
    assert max(step["bucket"] for step in trace) > 0.8 * BUCKET_CAPACITY
    assert [p for p, _ in itertools.groupby(step["phase"] for step in trace)] == [
        "lower", "drag", "curl", "lift", "swing", "dump"]
    assert cells_near(soil, (DRAG_FROM + DRAG_TO) / 2, 0.0, 0.5).min() < -0.25
    pile = cells_near(soil, DRAG_TO * np.cos(SWING_TO), DRAG_TO * np.sin(SWING_TO), 3.0)
    assert pile.max() > 0.2
    assert pile.sum() * soil.grid.cell_area > 0.8 * BUCKET_CAPACITY


def test_soil_in_the_bucket_weighs_on_it(nominal):
    soil, _, trace = nominal
    carried = [row for row in trace if row["phase"] in ("lift", "swing")]
    assert carried
    for row in carried:
        assert row["weight"] == pytest.approx(-NOMINAL.unit_weight * row["bucket"])
        assert row["bucket"] > 0.0


def test_joint_loads_rise_while_cutting(nominal):
    _, _, trace = nominal
    air_soil, _, air = run_cycle(heights=np.full(nominal[0].grid.shape, -3.0), limit=60)

    def peak(steps, joint):
        return max(abs(s["loads"][joint]) for s in steps if s["phase"] == "drag")

    assert air_soil.bucket_volume == 0.0
    for joint in ("arm", "bucket"):
        assert peak(trace, joint) > 3.0 * peak(air, joint)


def test_same_seed_and_commands_give_the_same_heights_bit_for_bit():
    first, _, _ = run_cycle(sample_soil(11), limit=40)
    second, _, _ = run_cycle(sample_soil(11), limit=40)
    other, _, _ = run_cycle(sample_soil(12), limit=40)
    assert np.array_equal(first.heights, second.heights)
    assert first.bucket_volume == second.bucket_volume
    assert not np.array_equal(first.heights, other.heights)


def test_control_rate_is_a_setting():
    model = load_scene(1, weld=True)
    assert physics_steps(model, 10.0) == 20
    assert physics_steps(model, 20.0) == 10
    with pytest.raises(ValueError):
        physics_steps(model, 30.0)
    soil, start, trace = run_cycle(control_hz=20.0)
    assert trace[-1]["phase"] == "dump"
    assert max(step["bucket"] for step in trace) > 0.8 * BUCKET_CAPACITY
    assert soil.bucket_volume == 0.0 and soil.lost_volume == 0.0
    assert abs(soil.total_volume() - start) < 1e-9
    assert soil.heights.min() < -0.25


def test_stiff_soil_stalls_the_drag_against_the_arm_cap():
    _, _, trace = run_cycle(STIFF, limit=30)
    last = trace[-10:]
    assert {s["phase"] for s in last} == {"drag"}
    assert all(abs(s["loads"]["arm"]) >= 0.99 * FORCE_CAPS["arm"] for s in last)
    assert np.linalg.norm(last[-1]["tip"] - last[0]["tip"]) < 0.01


def test_holding_the_bucket_at_its_limit_stalls_like_a_free_bucket():
    def push(pinned):
        model, data, soil = machine(STIFF)
        place(model, data, soil, np.array([5.8, 0.0, -0.3]), 0.8)
        bucket = model.joint(PREFIX + "bucket")
        if pinned:
            data.qpos[bucket.qposadr[0]] = bucket.range[0]
            soil.reset(model, data)
        start = data.site_xpos[soil.tip].copy()
        data.ctrl[model.actuator(PREFIX + "arm").id] = 0.2
        data.ctrl[model.actuator(PREFIX + "bucket").id] = -0.25 if pinned else 0.0
        resistance = []
        for k in range(400):
            advance(model, data, soil, 1)
            resistance.append(soil.resistance)
            if k % 20 == 19:
                soil.control_step(model, data)
        return np.linalg.norm(data.site_xpos[soil.tip] - start), np.mean(resistance)

    free_travel, free_resistance = push(False)
    pinned_travel, pinned_resistance = push(True)
    assert pinned_travel <= 2.0 * free_travel + 0.01
    assert pinned_resistance >= 0.9 * free_resistance > 100e3


def test_a_jittery_vertical_descent_changes_the_force_smoothly():
    model, data, soil = machine(LOOSE)
    point = np.array([6.0, 0.0, 0.3])
    place(model, data, soil, point, 1.0)
    joints = [model.joint(PREFIX + j).qposadr[0] for j in JOINT_NAMES]
    acts, rng = actuators(model), np.random.default_rng(0)
    forces, limits = [], []
    for _ in range(160):
        point = point - np.array([0.0, 0.0, 0.01])
        target = inverse_kinematics(point + np.array([0.0, rng.normal(0.0, 0.003), 0.0]), 1.0)
        data.ctrl[acts] = np.clip(3.0 * (target - data.qpos[joints]), -0.5, 0.5)
        for _ in range(20):
            advance(model, data, soil, 1)
            forces.append(soil.edge_force(data).copy())
            limits.append(soil.resistance)
        soil.control_step(model, data)
    forces, limits = np.array(forces), np.array(limits)
    both = (limits[1:] > 1e3) & (limits[:-1] > 1e3)
    jumps = np.linalg.norm(np.diff(forces, axis=0), axis=1)[both]
    assert both.sum() > 1500
    assert jumps.max() < 2.0e3

def test_a_sliding_edge_meets_exactly_the_resistance_against_its_motion():
    model, data, soil = machine()
    acts = actuators(model)
    jac = np.zeros((3, model.nv))
    moving, sliding, misaligned = 0, 0, 0
    for command in itertools.product((-0.3, 0.0, 0.3), repeat=4):
        if not any(command):
            continue
        place(model, data, soil, *BURIED)
        data.ctrl[acts] = command
        for _ in range(30):
            soil.substep(model, data)
            limit = soil.resistance
            mujoco.mj_step(model, data)
            mujoco.mj_jacSite(model, data, jac, None, soil.tip)
            force, velocity = soil.edge_force(data), jac @ data.qvel
            magnitude, speed = np.linalg.norm(force), np.linalg.norm(velocity)
            assert magnitude <= 1.001 * limit
            if limit > 1e3 and speed > 1e-2:
                moving += 1
                assert -(force @ velocity) > 0.0
                if magnitude >= 0.99 * limit:
                    sliding += 1
                    misaligned += -(force @ velocity) < 0.99 * magnitude * speed
    assert sliding > 1000
    assert sliding > 0.99 * moving
    assert misaligned < 0.01 * sliding


def test_a_stalled_swing_leaves_the_curl_no_cheaper():
    def curl_loads(swing_first):
        model, data, soil = machine()
        place(model, data, soil, *BURIED)
        if swing_first:
            data.ctrl[model.actuator(PREFIX + "swing").id] = 0.3
            advance(model, data, soil, 100)
            assert abs(joint_load(model, data, 0, "swing")) == pytest.approx(
                FORCE_CAPS["swing"], rel=0.01)
        data.ctrl[model.actuator(PREFIX + "bucket").id] = 0.3
        loads = []
        for _ in range(100):
            advance(model, data, soil, 1)
            loads.append(abs(joint_load(model, data, 0, "bucket")))
        return np.median(loads)

    assert curl_loads(True) >= 0.9 * curl_loads(False) > 0.2 * FORCE_CAPS["bucket"]


def test_sweeping_sideways_is_resisted_until_the_swing_stalls():
    model, data, soil = machine()
    place(model, data, soil, *BURIED)
    data.ctrl[model.actuator(PREFIX + "swing").id] = 0.3
    advance(model, data, soil, 100)
    assert soil.resistance > 0.0
    assert abs(joint_load(model, data, 0, "swing")) == pytest.approx(FORCE_CAPS["swing"], rel=0.01)
    assert abs(data.site_xpos[soil.tip][1]) < 0.02


def test_an_edge_coasting_into_soil_is_stopped_at_the_wall():
    model, data, soil = machine(STIFF)
    xs, ys = soil.grid.axes()
    gx, gy = np.meshgrid(xs, ys)
    angle, radius = np.arctan2(gy, gx), np.hypot(gx, gy)
    pit = np.where((abs(angle) < 0.25) & (radius > 4.5) & (radius < 7.5), -1.0, 0.0)
    place(model, data, soil, *BURIED)
    soil.reset(model, data, pit)
    swing = model.joint(PREFIX + "swing").qposadr[0]
    data.ctrl[model.actuator(PREFIX + "swing").id] = 1.0
    while data.qpos[swing] < 0.15:
        advance(model, data, soil, 1)
    data.ctrl[:] = 0.0
    peak = 0.0
    for _ in range(600):
        advance(model, data, soil, 1)
        peak = max(peak, soil.resistance)
    assert data.qpos[swing] < 0.25
    assert peak > 20e3


def test_moving_the_free_base_drags_the_edge_against_the_soil():
    def slide(point):
        model = load_scene(1, weld=False)
        data = mujoco.MjData(model)
        soil = Soil(model, 0, NOMINAL)
        place(model, data, soil, point, PITCH_DIG)
        advance(model, data, soil, 200)
        base, jac = soil.base, np.zeros((3, model.nv))
        start = data.xpos[base][0]
        for _ in range(200):
            data.xfrc_applied[base, 0] = -1.5e5
            advance(model, data, soil, 1)
            mujoco.mj_jacSite(model, data, jac, None, soil.tip)
            assert soil.edge_force(data) @ (jac @ data.qvel) <= 0.0
        return start - data.xpos[base][0], soil.resistance

    buried, resisted = slide(BURIED[0])
    free, _ = slide(np.array([6.0, 0.0, 1.5]))
    assert resisted > 10e3
    assert buried < 0.5 * free


def test_a_resting_edge_stays_quiet_and_an_edge_above_ground_feels_nothing():
    def rest(heights):
        model, data, soil = machine()
        place(model, data, soil, *BURIED)
        soil.reset(model, data, heights)
        data.ctrl[model.actuator(PREFIX + "arm").id] = 0.2
        advance(model, data, soil, 100)
        data.ctrl[:] = 0.0
        advance(model, data, soil, 200)
        jac, speeds = np.zeros((3, model.nv)), []
        for _ in range(400):
            advance(model, data, soil, 1)
            mujoco.mj_jacSite(model, data, jac, None, soil.tip)
            speeds.append(np.linalg.norm(jac @ data.qvel))
        return np.array(speeds), soil.resistance

    shape = machine()[2].grid.shape
    in_soil, resistance = rest(None)
    in_air, _ = rest(np.full(shape, -3.0))
    assert resistance > 10e3
    assert in_soil.max() < 1.5e-3 and in_air.max() < 1.5e-3
    assert np.abs(np.diff(in_soil)).max() < 1e-5
    assert in_soil[-1] < in_soil[0]
    model, data, soil = machine()
    place(model, data, soil, *BURIED)
    data.qvel[:] = 0.0
    soil.substep(model, data)
    assert soil.resistance == 0.0
    mujoco.mj_step(model, data)
    assert not soil.edge_force(data).any()
    place(model, data, soil, np.array([6.0, 0.0, 1.5]), PITCH_DIG)
    data.ctrl[actuators(model)] = 0.3
    for _ in range(100):
        advance(model, data, soil, 1)
        assert soil.resistance == 0.0
    assert data.site_xpos[soil.edges, 2].min() > 0.3


def test_the_back_of_the_bucket_spills_instead_of_filling():
    model, data, soil = machine(LOOSE)
    place(model, data, soil, *BURIED)
    start = soil.total_volume()
    data.ctrl[model.actuator(PREFIX + "arm").id] = -0.2
    advance(model, data, soil, 200)
    assert data.site_xpos[soil.tip][0] > BURIED[0][0] + 0.2
    assert soil.bucket_volume == 0.0
    assert soil.loose.sum() > 0.0
    assert soil.total_volume() == pytest.approx(start, abs=1e-12)


def test_an_edge_at_the_grid_border_is_resisted_as_it_is_inland():
    model, data, soil = machine()
    place(model, data, soil, np.array([0.0, 5.93, -0.25]), PITCH_DIG)
    start = soil.ground_volume()
    data.ctrl[model.actuator(PREFIX + "arm").id] = -0.2
    resistance = []
    for _ in range(40):
        advance(model, data, soil, 1)
        resistance.append(soil.resistance)
    assert data.site_xpos[soil.edges, 1].max() > soil.grid.y0 + soil.grid.cell * (
        soil.grid.shape[0] - 1) - soil.grid.cell
    assert min(resistance[5:]) > 50e3
    assert start - soil.ground_volume() < 0.005

def test_ground_under_the_tracks_is_protected():
    model, data, soil = machine()
    place(model, data, soil, *BURIED)
    xs, ys = soil.grid.axes()

    def protected(x, y):
        return soil.protected[np.argmin(abs(ys - y)), np.argmin(abs(xs - x))]

    assert protected(0.0, 0.0) and protected(1.6, 1.3) and protected(-1.6, -1.3)
    assert not protected(1.8, 0.0) and not protected(0.0, 1.5) and not protected(6.0, 0.0)


def test_joint_load_equals_the_load_on_a_held_joint():
    model, data, soil = machine()
    place(model, data, soil, np.array([4.0, 0.0, 1.5]), 3.0)
    swing = model.joint(PREFIX + "swing").dofadr[0]
    for _ in range(200):
        data.qfrc_applied[swing] = 2.0e4
        soil.substep(model, data)
        mujoco.mj_step(model, data)
    assert joint_load(model, data, 0, "swing") == pytest.approx(2.0e4, rel=1e-3)


def test_sync_viewer_uploads_this_machines_heightfield():
    class Viewer:
        uploaded = []

        def update_hfield(self, hfield):
            self.uploaded.append(hfield)

    model, _, soil = machine()
    viewer = Viewer()
    sync_viewer(viewer, soil)
    assert viewer.uploaded == [model.hfield(PREFIX + "ground").id]


def test_the_bucket_fills_to_capacity_and_spills_the_rest_ahead():
    model, data, soil = machine(LOOSE)
    place(model, data, soil, *BURIED)
    start = soil.total_volume()
    data.ctrl[model.actuator(PREFIX + "arm").id] = 0.2
    for _ in range(30):
        step(model, data, [soil])
    tip = data.site_xpos[soil.tip]
    xs, _ = soil.grid.axes()
    assert soil.bucket_volume == BUCKET_CAPACITY
    assert soil.loose[:, xs < tip[0] - 0.05].sum() * soil.grid.cell_area > 0.02
    assert soil.lost_volume == 0.0
    assert soil.total_volume() == pytest.approx(start, abs=1e-12)


def test_moving_the_pose_after_reset_cuts_nothing():
    model, data, soil = machine()
    mujoco.mj_resetData(model, data)
    soil.reset(model, data)
    before = soil.heights.copy()
    data.qpos[[model.joint(PREFIX + j).qposadr[0] for j in JOINT_NAMES]] = (
        inverse_kinematics(*BURIED))
    mujoco.mj_forward(model, data)
    soil.substep(model, data)
    assert np.array_equal(soil.heights, before)
    mujoco.mj_step(model, data)
    data.ctrl[model.actuator(PREFIX + "arm").id] = 0.2
    advance(model, data, soil, 100)
    assert soil.bucket_volume > 0.02


def test_joint_load_reads_one_machine_while_another_cuts():
    model = load_scene(2, weld=True)
    data = mujoco.MjData(model)
    soils = [Soil(model, 0), Soil(model, 1)]
    other = machine_prefix(1)
    data.qpos[[model.joint(PREFIX + j).qposadr[0] for j in JOINT_NAMES]] = (
        inverse_kinematics(*BURIED))
    data.qpos[[model.joint(other + j).qposadr[0] for j in JOINT_NAMES]] = (
        inverse_kinematics(np.array([4.0, 0.0, 1.5]), 3.0))
    for soil in soils:
        soil.reset(model, data)
    data.ctrl[model.actuator(PREFIX + "arm").id] = 0.2
    swing = model.joint(other + "swing").dofadr[0]
    for _ in range(200):
        data.qfrc_applied[swing] = 2.0e4
        for soil in soils:
            soil.substep(model, data)
        mujoco.mj_step(model, data)
    assert soils[0].resistance > 10e3
    assert joint_load(model, data, 1, "swing") == pytest.approx(2.0e4, rel=1e-3)
