import itertools

import mujoco
import numpy as np
import pytest

from hma.machine import FORCE_CAPS, JOINT_NAMES, load_scene, machine_prefix
from hma.machine.kinematics import inverse_kinematics
from hma.soil import (
    BUCKET_CAPACITY,
    FORCE_CAP,
    NOMINAL,
    Soil,
    SoilParams,
    joint_load,
    physics_steps,
    sample_soil,
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
    carried = [step for step in trace if step["phase"] in ("lift", "swing")]
    assert carried
    for step in carried:
        assert step["weight"] == pytest.approx(-NOMINAL.unit_weight * step["bucket"])
        assert step["bucket"] > 0.0


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


def test_stiff_soil_stalls_the_bucket_against_the_joint_caps():
    _, _, trace = run_cycle(STIFF, limit=30)
    assert {step["phase"] for step in trace} == {"lower"}
    last = trace[-10:]
    assert all(abs(s["loads"]["boom"]) >= 0.99 * FORCE_CAPS["boom"] for s in last)
    assert abs(last[-1]["tip"][2] - last[0]["tip"][2]) < 0.01
    assert last[-1]["tip"][2] > -0.2


def test_force_opposes_the_edge_motion_in_every_direction():
    model, data, soil = machine()
    acts = actuators(model)
    jac = np.zeros((3, model.nv))
    resisted = 0
    for command in itertools.product((-0.3, 0.0, 0.3), repeat=4):
        if not any(command):
            continue
        place(model, data, soil, *BURIED)
        data.ctrl[acts] = command
        for _ in range(30):
            soil.substep(model, data)
            mujoco.mj_jacSite(model, data, jac, None, soil.tip)
            mujoco.mj_step(model, data)
            force, velocity = soil.edge_force(data), jac @ data.qvel
            assert np.linalg.norm(force) <= FORCE_CAP
            if np.linalg.norm(force) > 1.0 and np.linalg.norm(velocity) > 1e-3:
                assert force @ velocity < 0.0
                resisted += 1
    assert resisted > 1500


def test_sweeping_sideways_is_resisted_until_the_swing_stalls():
    model, data, soil = machine()
    place(model, data, soil, *BURIED)
    data.ctrl[model.actuator(PREFIX + "swing").id] = 0.3
    advance(model, data, soil, 100)
    assert soil.resistance > 0.0
    assert abs(joint_load(model, data, 0, "swing")) == pytest.approx(FORCE_CAPS["swing"], rel=0.01)
    assert abs(data.site_xpos[soil.tip][1]) < 0.02


def test_no_force_at_rest_or_with_the_edge_above_ground():
    model, data, soil = machine()
    acts = actuators(model)
    place(model, data, soil, *BURIED)
    data.ctrl[model.actuator(PREFIX + "arm").id] = 0.2
    advance(model, data, soil, 100)
    moving = soil.resistance
    data.ctrl[:] = 0.0
    advance(model, data, soil, 400)
    assert soil.resistance < 0.05 * moving
    data.qvel[:] = 0.0
    soil.substep(model, data)
    assert soil.resistance == 0.0
    mujoco.mj_step(model, data)
    assert not soil.edge_force(data).any()
    place(model, data, soil, np.array([6.0, 0.0, 1.5]), PITCH_DIG)
    data.ctrl[acts] = 0.3
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
