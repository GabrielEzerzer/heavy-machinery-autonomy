from dataclasses import replace

import mujoco
import numpy as np
import pytest

import hma.machine.model
from hma.hydraulics import NOMINAL, RANGES, Hydraulics, draw_machine
from hma.machine import JOINT_LIMITS, JOINT_NAMES, load_scene, machine_prefix
from hma.machine.kinematics import inverse_kinematics
from hma.machine.model import GROUND, GROUND_DEPTH, GROUND_TOP, SOIL_AXES, SOIL_INVWEIGHT
from hma.sim import reset, step
from hma.sim.dig import (
    DUMP_AT,
    PITCH_DIG,
    TRENCH,
    admissible,
    breaks_rule,
    count_stalls,
    cut_volumes,
    depth_excess,
    dig,
    holding,
    home_path,
    machine_origin,
    off_grid,
    tip_error,
    tip_point,
)
from hma.soil import NOMINAL as NOMINAL_SOIL
from hma.soil import Soil, SoilParams, sample_soil

ARM = JOINT_NAMES.index("arm")
STIFF = SoilParams(105.0e3, 105.0e3, 0.8, 22.0e3, 0.4, 300.0)
DRAG = np.array([0.0, 0.0, 1.0, 0.0])
LIFT = np.array([0.0, -1.0, -1.0, 0.0])


def buried(depth, radial=6.0):
    return inverse_kinematics(np.array([radial, 0.0, -depth]), PITCH_DIG)


def scene(machines, soil_params, poses):
    model = load_scene(len(machines), weld=True)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, machines)
    soils = [Soil(model, i, p) for i, p in enumerate(soil_params) if p is not None]
    for i, (machine, pose) in enumerate(zip(machines, poses, strict=True)):
        reset(layer, soils, i, machine, pose)
    return layer, soils


def dofs(model, index):
    return np.array([model.joint(machine_prefix(index) + j).dofadr[0] for j in JOINT_NAMES])


def held_readings(layer, soils, index, hand, commands):
    model, data = layer.model, layer.data
    data.qfrc_applied[dofs(model, index)] = hand * layer.machines[index].force_limit
    seen = []
    for u in commands:
        step(layer, soils, u)
        seen.append(layer.true_state().loads[index].copy())
    return np.array(seen)


def check_cutter(layer, soil):
    model, data = layer.model, layer.data
    own = dofs(model, soil.index)
    rows = (data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON) & np.isin(
        data.efc_id, soil.tendons)
    generalised = np.zeros(model.nv)
    mujoco.mj_mulJacTVec(model, data, generalised, np.where(rows, data.efc_force, 0.0))
    from_soil = generalised[own]
    loads = layer.true_state().loads[soil.index] * layer.machines[soil.index].force_limit
    np.testing.assert_allclose(loads, data.qfrc_constraint[own] - from_soil, rtol=0, atol=1e-6)
    return np.abs(from_soil).max()


HAND = np.array([0.5, 0.0, 0.0, -0.3])


@pytest.mark.parametrize("cutter", [0, 1])
def test_a_machine_reads_its_own_load_while_its_neighbour_cuts(cutter):
    machines = [draw_machine(3), draw_machine(8)]
    soils = [sample_soil(4), sample_soil(9)]
    held = 1 - cutter
    poses = [None, None]
    poses[cutter] = buried(0.25)
    layer, scene_soils = scene(machines, soils, poses)
    commands = np.zeros((20, 2, 4))
    commands[:, cutter] = DRAG
    together = held_readings(layer, scene_soils, held, HAND, commands)
    cutting = next(s for s in scene_soils if s.index == cutter)
    assert cutting.bucket_volume > 0.0
    assert check_cutter(layer, cutting) > 1e3

    alone_layer, _ = scene([machines[held]], [None], [None])
    alone = held_readings(alone_layer, [], 0, HAND, np.zeros((20, 1, 4)))
    np.testing.assert_allclose(together, alone, rtol=0, atol=1e-6)
    unloaded_layer, _ = scene([machines[held]], [None], [None])
    unloaded = held_readings(unloaded_layer, [], 0, 0.0 * HAND, np.zeros((20, 1, 4)))
    np.testing.assert_allclose(together[-1] - unloaded[-1], -HAND, atol=1e-3)


def test_a_lone_cutter_reads_the_valve_force_and_not_the_soil():
    machine = draw_machine(3)
    layer, soils = scene([machine], [sample_soil(4)], [buried(0.25)])
    rows = set()
    for k in range(25):
        step(layer, soils, (DRAG if k < 10 else LIFT)[None])
        rows.add(int(np.sum(layer.data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)))
        check_cutter(layer, soils[0])
    assert {4, 7} <= rows
    assert soils[0].bucket_volume > 0.0


def episode(layer, soils, commands):
    seen = []
    for u in commands:
        step(layer, soils, u)
        seen.append(np.concatenate([np.stack(layer.true_state()).ravel(),
                                    np.stack(layer.observe()).ravel(),
                                    *(s.heights.ravel() for s in soils),
                                    [s.bucket_volume for s in soils]]))
    return np.array(seen)


def test_same_seeds_and_commands_repeat_bit_for_bit_with_both_layers():
    machines = [draw_machine(5), draw_machine(6)]
    soils = [sample_soil(7), sample_soil(8)]
    poses = [buried(0.2), buried(0.3, 6.4)]
    rng = np.random.default_rng(2)
    commands = DRAG + rng.uniform(-0.3, 0.3, size=(15, 2, 4))
    first = episode(*scene(machines, soils, poses), commands)
    assert np.array_equal(first, episode(*scene(machines, soils, poses), commands))

    layer, scene_soils = scene(machines, soils, poses)
    for stray in (commands[::-1], commands[:1]):
        episode(layer, scene_soils, stray)
        for i, machine in enumerate(machines):
            reset(layer, scene_soils, i, machine, poses[i])
        assert np.array_equal(first, episode(layer, scene_soils, commands))
        for i, machine in enumerate(machines):
            reset(layer, scene_soils, i, machine, poses[i])


def test_resetting_one_machine_leaves_its_cutting_neighbour_alone():
    machines = [draw_machine(5), draw_machine(6)]
    soils = [sample_soil(7), sample_soil(8)]
    poses = [buried(0.2), buried(0.3, 6.4)]
    commands = DRAG + np.random.default_rng(2).uniform(-0.3, 0.3, size=(15, 2, 4))

    def neighbour(reset_at):
        layer, scene_soils = scene(machines, soils, poses)
        seen = []
        for k, u in enumerate(commands):
            if k == reset_at:
                reset(layer, scene_soils, 0, machines[0], poses[0])
            step(layer, scene_soils, u)
            neighbour = scene_soils[1]
            seen.append(np.concatenate([layer.true_state().positions[1], neighbour.heights.ravel(),
                                        [neighbour.bucket_volume]]))
        return np.array(seen)

    kept = neighbour(None)
    assert kept[-1, -1] > 0.0
    np.testing.assert_allclose(neighbour(10), kept, rtol=0, atol=1e-12)


def test_the_soil_prices_the_edge_where_the_last_step_left_it():
    layer, soils = scene([NOMINAL], [NOMINAL_SOIL],
                         [inverse_kinematics(np.array([6.0, 0.0, 0.01]), PITCH_DIG)])
    model, data, soil = layer.model, layer.data, soils[0]
    scratch = mujoco.MjData(model)
    physics, seen = layer.step, []

    def current():
        scratch.qpos[:] = data.qpos
        mujoco.mj_kinematics(model, scratch)
        np.testing.assert_array_equal(data.site_xpos[soil.edges], scratch.site_xpos[soil.edges])

    def recorded():
        current()
        seen.append((data.site_xpos[soil.edges, 2].min(), max(soil.limits)))
        physics()

    layer.step = recorded
    for _ in range(3):
        step(layer, soils, np.array([[0.0, 1.0, 0.0, 0.0]]))
        current()
    edge, limit = np.array(seen).T
    below = np.flatnonzero(edge < soil.grid.z0)
    assert below[0] > 0
    assert np.all(limit[:below[0]] == 0.0)
    assert np.all(limit[below] > 0.0)


@pytest.fixture(scope="module")
def nominal_cycle():
    return dig(NOMINAL, NOMINAL_SOIL)


def test_a_dig_and_dump_through_the_hydraulics_conserves_volume(nominal_cycle):
    r = nominal_cycle
    assert r["lost"] == 0.0
    assert r["left_in_bucket"] == 0.0
    assert abs(r["volume_error"]) < 1e-12
    assert r["trench_out"] > 0.3
    assert r["pile"] > 0.3
    assert r["deepest"] > 0.25
    assert r["highest"] > 0.2


def test_the_nominal_cycle_tracks_its_path_and_keeps_the_scoring_rule(nominal_cycle):
    r = nominal_cycle
    assert r["dumped"]
    assert not r["broke_rule"]
    assert not r["stalls"]
    assert r["error_max"] < 0.06
    model = load_scene(1, weld=True)
    assert admissible(home_path(), model, Soil(model, 0))


def test_a_full_command_in_stiff_soil_stalls_at_full_scale_under_the_power_limit():
    layer, soils = scene([NOMINAL], [STIFF], [buried(0.3)])
    model, data = layer.model, layer.data
    arm = model.actuator(machine_prefix(0) + "arm").id
    trace = []
    for _ in range(40):
        layer.act(DRAG[None])
        for _ in range(layer.steps_per_decision):
            soils[0].substep(model, data)
            layer.step()
            mujoco.mj_kinematics(model, data)
            trace.append((layer.true_state().loads[0, ARM], data.ctrl[arm],
                          layer.true_state().velocities[0, ARM], layer._ratio[0]))
        soils[0].control_step(model, data)
    loads, commands, speeds, ratios = np.array(trace).T
    assert np.all(loads[-200:] >= 0.99)
    assert np.abs(speeds[-200:]).max() < 0.01
    assert ratios[-1] < 1.0
    assert commands[-1] == pytest.approx(NOMINAL.power / NOMINAL.force_limit[ARM], rel=1e-3)
    assert np.ptp(speeds[-200:]) < 1e-3

    def reversals(x):
        steps = np.diff(x)
        signs = np.sign(steps[np.abs(steps) > 1e-9])
        return int(np.sum(signs[1:] != signs[:-1]))

    engaged = int(np.argmax(ratios < 1.0))
    settled = int(np.flatnonzero(np.abs(np.diff(ratios)) > 1e-9)[-1]) + 1
    assert 0 < engaged < settled < len(ratios) - 200
    assert reversals(ratios[engaged:settled + 1]) == 0
    assert reversals(commands[engaged:settled + 1]) == 1


def test_a_delayed_lagged_drag_pays_force_times_distance():
    slow = replace(NOMINAL, delay=np.full(4, RANGES["delay"][1]), lag=np.full(4, RANGES["lag"][1]))
    layer, soils = scene([slow], [NOMINAL_SOIL], [buried(0.1, 6.6)])
    model, data, soil = layer.model, layer.data, soils[0]
    paid, priced, travelled = [], [], []
    physics = layer.step

    def recorded():
        limits, axes = np.array(soil.limits), soil.axes.copy()
        physics()
        jac = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data, jac, None, soil.tip)
        move = jac @ data.qvel * model.opt.timestep
        paid.append(-soil.edge_force(data) @ move)
        priced.append(limits @ np.abs(axes @ move))
        travelled.append(np.linalg.norm(move))

    layer.step = recorded
    edges = []
    for a in [0.0] * 2 + [1.0] * 10 + [0.0] * 5 + [-0.5] * 3 + [1.0] * 10:
        step(layer, soils, a * DRAG[None])
        edges.append(data.site_xpos[soil.edges].copy())
    idle, idle_soils = scene([slow], [NOMINAL_SOIL], [buried(0.1, 6.6)])
    for k in range(9):
        step(idle, idle_soils, np.zeros((1, 4)))
        resting = idle.data.site_xpos[idle_soils[0].edges]
        if k < 8:
            assert np.array_equal(edges[k], resting)
    assert np.abs(edges[8] - resting).max() > 0.005
    paid, priced = np.array(paid), np.array(priced)
    assert sum(travelled) > 0.5
    assert priced.sum() > 2e4
    assert np.all(paid <= priced + 1e-6)
    assert paid.sum() == pytest.approx(priced.sum(), rel=0.01)


def placed(model, data, soil, point, pitch=PITCH_DIG, shift=(0.0, 0.0)):
    data.qpos[:] = model.qpos0
    data.qpos[[model.joint(machine_prefix(soil.index) + j).qposadr[0] for j in JOINT_NAMES]] = (
        inverse_kinematics(np.asarray(point, float), pitch))
    base = model.joint(machine_prefix(soil.index) + "base_free").qposadr[0]
    data.qpos[base:base + 2] += shift
    mujoco.mj_forward(model, data)


@pytest.mark.parametrize("index", [0, 1])
def test_each_soil_reports_its_edge_depth_below_uncut_ground_and_whether_it_is_on_the_grid(index):
    model = load_scene(2, weld=False)
    data = mujoco.MjData(model)
    soil = Soil(model, index)
    soil.reset(model, data)
    placed(model, data, soil, (6.0, 0.0, -0.25))
    assert soil.uncut_depth(data) == pytest.approx(0.25, abs=0.02)
    assert soil.edge_inside(data)

    lowered = np.full(soil.grid.shape, -0.1)
    soil.reset(model, data, lowered)
    assert soil.uncut_depth(data) == pytest.approx(0.15, abs=0.02)

    soil.reset(model, data)
    near = np.hypot(soil.cells[..., 0] - data.site_xpos[soil.tip, 0],
                    soil.cells[..., 1] - data.site_xpos[soil.tip, 1]) < 1.0
    soil.heights[near] += 0.3
    soil.loose[near] += 0.3
    assert soil.uncut_depth(data) == pytest.approx(0.25, abs=0.02)

    placed(model, data, soil, (6.0, 0.0, 0.3))
    assert soil.uncut_depth(data) == 0.0
    placed(model, data, soil, (6.0, 0.0, 0.3), shift=(6.0, 0.0))
    assert not soil.edge_inside(data)

    placed(model, data, soil, (6.0, 0.0, -0.25))
    far = soil.edges[np.argmax(data.site_xpos[soil.edges, 1])]
    share = (data.site_xpos[far, 1] - soil.grid.y0) / soil.grid.cell % 1.0
    tip = data.site_xpos[soil.tip].copy()
    placed(model, data, soil, (6.0, 0.0, -0.25), shift=(0.0, (0.5 - share) * soil.grid.cell))
    x, y, z = data.site_xpos[far]
    soil.reset(model, data, np.where(soil.cells[..., 1] > y, 0.5, 0.0))
    share = (y - soil.grid.y0) / soil.grid.cell % 1.0
    assert share == pytest.approx(0.5)
    assert soil.uncut_depth(data) == pytest.approx(soil.grid.z0 - z + 0.5 * share, abs=1e-9)

    xs, ys = soil.grid.axes()
    placed(model, data, soil, (6.0, 0.0, -0.25), shift=(0.0, ys[-1] - 0.05 - tip[1]))
    tip = data.site_xpos[soil.tip]
    assert xs[0] < tip[0] < xs[-1] and ys[0] < tip[1] < ys[-1]
    assert not soil.edge_inside(data)


def test_a_step_settles_the_spoil_and_writes_the_height_field():
    layer, soils = scene([NOMINAL], [NOMINAL_SOIL], [None])
    model, soil = layer.model, soils[0]
    far = np.unravel_index(np.argmin(np.hypot(soil.cells[..., 0] - 8.0, soil.cells[..., 1] - 3.0)),
                           soil.grid.shape)
    soil.heights[far] = soil.loose[far] = 0.5
    step(layer, soils, np.zeros((1, 4)))
    assert 0.0 < soil.heights[far] < 0.5
    assert soil.total_volume() == pytest.approx(0.5 * soil.grid.cell_area, abs=1e-12)
    field = model.hfield(machine_prefix(0) + GROUND)
    written = model.hfield_data[field.adr[0]:field.adr[0] + soil.heights.size]
    np.testing.assert_allclose(written * (GROUND_DEPTH + GROUND_TOP) - GROUND_DEPTH,
                               soil.heights.ravel(), rtol=0, atol=1e-5)


def soil_softness(model, index=0):
    return model.tendon_invweight0[[model.tendon(machine_prefix(index) + a).id for a in SOIL_AXES]]


def test_the_soil_tendons_keep_the_nominal_machines_softness_at_the_dig_pose(monkeypatch):
    layer, soils = scene([NOMINAL], [NOMINAL_SOIL], [buried(0.25)])
    model, data, soil = layer.model, layer.data, soils[0]
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac, None, soil.tip)
    along = soil.axes @ jac
    inverse = np.zeros_like(along)
    mujoco.mj_solveM(model, data, inverse, along)
    np.testing.assert_allclose(np.sum(along * inverse, axis=1), SOIL_INVWEIGHT, rtol=1e-3)
    np.testing.assert_array_equal(soil_softness(model), SOIL_INVWEIGHT)

    drawn, _ = scene([draw_machine(3), draw_machine(4)], [None, None], [None, buried(0.25)])
    for index in (0, 1):
        np.testing.assert_array_equal(soil_softness(drawn.model, index), SOIL_INVWEIGHT)

    valves = [model.tendon(machine_prefix(0) + j + "_valve").id for j in JOINT_NAMES]
    monkeypatch.setattr(hma.machine.model, "HOME_POSE", (0.0, -0.3, 1.5, 0.3))
    other = load_scene(1, weld=True)
    assert not np.allclose(other.tendon_invweight0[valves], model.tendon_invweight0[valves])
    np.testing.assert_array_equal(soil_softness(other), SOIL_INVWEIGHT)


class Still:
    duration = 1.0

    def __init__(self, point, pitch=PITCH_DIG):
        self.point, self.pitch = np.asarray(point, float), pitch

    def __call__(self, t):
        return self.point, self.pitch


class Lowering:
    duration = 2.0

    def __call__(self, t):
        return np.array([6.0, 0.0, 0.5 if t < 1.0 else -0.3]), PITCH_DIG


def test_depth_is_scored_against_what_the_path_asks_at_that_moment():
    path = Lowering()
    assert depth_excess(0.25, path, 0.0, 0.5) == pytest.approx(0.25)
    assert depth_excess(0.45, path, 0.0, 1.5) == pytest.approx(0.15)
    assert depth_excess(0.45, path, -0.1, 1.5) == pytest.approx(0.25)
    assert breaks_rule(depth_excess(0.25, path, 0.0, 0.5), False)
    assert not breaks_rule(depth_excess(0.45, path, 0.0, 1.5), False)
    assert breaks_rule(0.0, True)
    assert off_grid(-0.01, 0.0, inside=False)
    assert not off_grid(-0.01, 0.0, inside=True)
    assert not off_grid(0.01, 0.0, inside=False)


def test_a_stall_is_ten_whole_control_steps_held_at_full_load():
    limits = np.array([[-np.inf, np.inf], *JOINT_LIMITS.values()])
    q = np.array([0.0, *[(lo + hi) / 2 for lo, hi in JOINT_LIMITS.values()]])
    load, v = np.array([0.0, 0.0, 0.995, 0.0]), np.zeros(4)
    held = holding(q, v, load, limits)
    assert held.tolist() == [False, False, True, False]
    assert not holding(q, v + 0.02, load, limits).any()
    at_stop = q.copy()
    at_stop[ARM] = limits[ARM, 1] - 0.01
    assert not holding(at_stop, v, load, limits).any()
    stalls, counts = set(), np.zeros(4, int)
    for _ in range(9):
        counts = count_stalls(stalls, counts, held, 10.0, "drag")
    counts = count_stalls(stalls, counts, np.zeros(4, bool), 10.0, "drag")
    for _ in range(9):
        counts = count_stalls(stalls, counts, held, 10.0, "drag")
    assert not stalls
    count_stalls(stalls, counts, held, 10.0, "drag")
    assert stalls == {"arm in drag"}


def test_a_path_with_an_edge_end_off_the_grid_or_below_its_pass_is_not_admissible():
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    soil = Soil(model, 0)
    side = Still(tip_point(6.45, 1.2, -0.25))
    data.qpos[[model.joint(machine_prefix(0) + j).qposadr[0] for j in JOINT_NAMES]] = (
        inverse_kinematics(*side(0.0)))
    mujoco.mj_kinematics(model, data)
    _, ys = soil.grid.axes()
    assert data.site_xpos[soil.tip, 1] < ys[-1] < data.site_xpos[soil.edges, 1].max()
    assert data.site_xpos[soil.edges, 2].max() < soil.grid.z0
    assert not admissible(side, model, soil, samples=3)
    assert admissible(Still(tip_point(6.0, 0.0, -0.25)), model, soil, samples=3)
    assert not admissible(Still(tip_point(6.0, 0.0, -0.45)), model, soil, samples=3)


def test_cut_volumes_count_the_trench_the_ground_cut_outside_it_and_the_pile():
    model = load_scene(1, weld=True)
    soil = Soil(model, 0)
    start = soil.heights.copy()
    (x0, _), (y0, _) = TRENCH
    x, y = soil.cells[..., 0], soil.cells[..., 1]
    trench = (x > x0 + 0.2) & (x < x0 + 0.7) & (y > y0 + 0.2) & (y < y0 + 0.7)
    outside = (x > 8.0) & (x < 8.5) & (y > -5.0) & (y < -4.5)
    pile = np.hypot(x - DUMP_AT[0], y - DUMP_AT[1]) < 0.5
    soil.heights[trench] -= 0.2
    soil.heights[outside] -= 0.1
    soil.heights[pile] += 0.3
    area = soil.grid.cell_area
    volumes = cut_volumes(soil, start)
    assert volumes["trench_out"] == pytest.approx(0.2 * trench.sum() * area)
    assert volumes["cut_outside"] == pytest.approx(0.1 * outside.sum() * area)
    assert volumes["pile"] == pytest.approx(0.3 * pile.sum() * area)


def test_the_tip_error_is_taken_in_the_world_against_where_the_machine_stands():
    model = load_scene(2, weld=False)
    data = mujoco.MjData(model)
    point = tip_point(6.0, 0.3, -0.2)
    errors = []
    for index in (0, 1):
        soil = Soil(model, index)
        placed(model, data, soil, point)
        errors.append(tip_error(data, soil, machine_origin(model, index), point))
    assert errors[0] == pytest.approx(errors[1], abs=1e-9)
    assert errors[0] < 1e-6
    placed(model, data, soil, point, shift=(0.1, 0.0))
    assert tip_error(data, soil, machine_origin(model, 1), point) == pytest.approx(0.1, abs=1e-6)
