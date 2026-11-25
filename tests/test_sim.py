from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, RANGES, Hydraulics, draw_machine
from hma.machine import JOINT_NAMES, load_scene, machine_prefix
from hma.machine.kinematics import inverse_kinematics
from hma.sim import reset, step
from hma.sim.dig import PITCH_DIG, admissible, dig, home_path
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
    jac = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac, None, soil.tip)
    from_soil = jac[:, own].T @ soil.edge_force(data)
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
    assert admissible(home_path(), Soil(model, 0))


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
            trace.append((layer.true_state().loads[0, ARM], data.ctrl[arm],
                          layer.true_state().velocities[0, ARM]))
        soils[0].control_step(model, data)
    loads, commands, speeds = np.array(trace[-200:]).T
    assert np.all(loads >= 0.99)
    assert np.abs(speeds).max() < 0.01
    assert layer._ratio[0] < 1.0
    assert commands[-1] == pytest.approx(NOMINAL.power / NOMINAL.force_limit[ARM], rel=1e-3)
    assert np.ptp(speeds) < 1e-3
    changes = np.diff(commands)
    assert not np.any(np.diff(np.sign(changes[np.abs(changes) > 1e-6])))


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
    for a in [0.0] * 2 + [1.0] * 10 + [0.0] * 5 + [-0.5] * 3 + [1.0] * 10:
        step(layer, soils, a * DRAG[None])
    paid, priced = np.array(paid), np.array(priced)
    assert sum(travelled) > 0.5
    assert priced.sum() > 2e4
    assert np.all(paid <= priced + 1e-6)
    assert paid.sum() == pytest.approx(priced.sum(), rel=0.01)


def placed(model, data, soil, point, pitch=PITCH_DIG, shift=0.0):
    data.qpos[:] = model.qpos0
    data.qpos[[model.joint(machine_prefix(soil.index) + j).qposadr[0] for j in JOINT_NAMES]] = (
        inverse_kinematics(np.asarray(point, float), pitch))
    data.qpos[model.joint(machine_prefix(soil.index) + "base_free").qposadr[0]] += shift
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
    placed(model, data, soil, (6.0, 0.0, 0.3), shift=6.0)
    assert not soil.edge_inside(data)
