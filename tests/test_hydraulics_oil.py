import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, RANGES, Hydraulics, draw_machine
from hma.hydraulics import circuit as c
from hma.hydraulics.layer import DISPLACEMENT, INFLOW, NODE_DROPS, VOLUME_SLOPE
from hma.machine import (
    BODY_NAMES,
    JOINT_LIMITS,
    JOINT_NAMES,
    TIP_SITE,
    load_scene,
    machine_prefix,
    statics,
)
from hma.machine.cylinders import MAIN_RELIEF, areas, cylinder_length

DT = 0.005
G = 9.81
SWING, BOOM, ARM, BUCKET = range(4)


def layer_for(machine, pose=None, n=1):
    model = load_scene(n, weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), [machine] * n)
    for i in range(n):
        layer.reset(i, machine, pose)
    return layer


def steps(layer, seconds, before=None):
    for _ in range(round(seconds / DT)):
        if before:
            before()
        layer.step()


def command(joint, value, n=1):
    u = np.zeros((n, 4))
    u[:, joint] = value
    return u


def test_pump_energy_is_the_work_done_plus_the_losses_plus_the_oil_stored():
    machine = draw_machine(3)
    layer = layer_for(machine)
    stored = layer.oil_energy()[0]
    for u, seconds in (([0, -1, 0, 0], 2.0), ([1, 0, 0, 0], 1.5), ([0, 0, 0, 0], 0.5),
                       ([0, 0.5, -1, 1], 2.5), ([-1, 1, 1, -1], 2.0)):
        layer.act(np.array([u], float))
        steps(layer, seconds)
    pump = layer.pump_energy[0]
    balance = layer.work_energy[0] + layer.lost_energy[0] + layer.oil_energy()[0] - stored
    assert pump > 1e5
    assert abs(pump - balance) < 0.01 * pump


def extreme_machine(link_mass):
    volume = np.full((4, 2), RANGES["dead_volume"][0])
    volume[0] = RANGES["swing_volume"][0]
    return replace(NOMINAL, bulk_modulus=RANGES["bulk_modulus"][1], dead_volume=volume,
                   pump_volume=RANGES["pump_volume"][0],
                   link_mass=np.full(len(BODY_NAMES), link_mass),
                   spool_flow=np.full(4, RANGES["spool_flow"][1]),
                   backlash=np.full(4, RANGES["backlash"][1]))


def chatter(pressures):
    change = np.diff(pressures, axis=0)
    flips = (change[1:] * change[:-1] < 0) & (np.abs(change[1:]) > 0.5e6) & (
        np.abs(change[:-1]) > 0.5e6)
    return flips.mean()


@pytest.mark.parametrize("joint", range(4))
@pytest.mark.parametrize("value", [1.0, -1.0])
def test_stiffest_oil_and_full_play_stay_above_the_cavitation_floor_without_chatter(joint, value):
    machines = [extreme_machine(m) for m in RANGES["link_mass"]]
    model = load_scene(2, weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), machines)
    layer.act(command(joint, value, 2))
    pressures, speeds = [], []
    for _ in range(round(4.0 / DT)):
        layer.step()
        pressures.append(layer._pressure.copy())
        speeds.append(layer.true_state().velocities.copy())
    pressures, speeds = np.array(pressures), np.array(speeds)
    assert np.isfinite(pressures).all()
    assert pressures.min() >= c.CAVITATION_PRESSURE - 1e-6
    assert max(chatter(pressures[:, i]) for i in range(2)) <= 0.005
    assert speeds[-100:, :, joint].std(axis=0).max() < 0.01


def test_held_boom_arm_and_bucket_drift_well_inside_the_service_limit():
    leaky = np.where(np.array([[0, 0], [1, 0], [0, 1], [0, 0]], bool),
                     RANGES["holding_leakage"][1], RANGES["spool_leakage"][1])
    layer = layer_for(replace(NOMINAL, leakage=leaky))
    steps(layer, 1.0)

    def lengths():
        q = layer.true_state().positions[0]
        return np.array([cylinder_length(j, q[k])[0] for k, j in ((1, "boom"), (2, "arm"),
                                                                    (3, "bucket"))]).ravel()

    start = lengths()
    steps(layer, 60.0)
    assert np.abs(lengths() - start).max() * 5.0 < 0.020 / 4


def prony_frequency(x):
    x = x - x.mean()
    a1, a2 = np.linalg.lstsq(np.c_[x[1:-1], x[:-2]], x[2:], rcond=None)[0]
    return abs(np.angle(np.roots([1.0, -a1, -a2])[0])) / DT / (2 * math.pi)


def test_boom_bounce_matches_the_oil_springs_on_the_linkage_inertia():
    model = load_scene(1, weld=True)
    model.eq_solref[:, 0] = 2 * DT
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [NOMINAL])
    steps(layer, 0.5)
    q = layer.true_state().positions[0]
    length, lever = c.geometry(q)
    volume = layer._volume0[0] + VOLUME_SLOPE * length[:, None]
    stiffness = NOMINAL.bulk_modulus * (c.AREA ** 2 / volume).sum(1) * lever ** 2
    model.dof_armature[layer._qvel[0]] = 0.0
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    front = layer._qvel[0, 1:]
    modes = np.linalg.eigvals(np.linalg.solve(mass[np.ix_(front, front)],
                                              np.diag(stiffness[1:])))
    hand = math.sqrt(np.sort(modes.real)[0]) / (2 * math.pi)
    steps(layer, 0.1)
    data.qvel[layer._qvel[0, BOOM]] += 2e-4
    trace = []
    for _ in range(400):
        layer.step()
        trace.append(layer.true_state().positions[0, BOOM])
    assert 3.0 < hand < 3.7
    assert prony_frequency(np.array(trace)) == pytest.approx(hand, rel=0.04)


NO_BUCKET = replace(NOMINAL, link_mass=np.array([1.0, 1.0, 1.0, 1.0, 1e-6]))


def boom_capacity(machine, boom, arm):
    machine.pose(0.0, boom, arm, 0.0, False)
    jacobian = machine.point_jacobian(machine.anchor("bucket"), "arm")
    per_kg = machine.g * jacobian[2, machine.dofs["boom"]]
    return statics.force_room(machine.bias("boom"), per_kg,
                              *statics.caps("boom", boom, (MAIN_RELIEF, MAIN_RELIEF)))


def lift_trial(machine, pose, kg):
    layer = layer_for(NO_BUCKET, pose)
    model, data = layer.model, layer.data
    arm = model.body(machine_prefix(0) + "arm").id
    pin = model.joint(machine_prefix(0) + "bucket").id
    load = {"kg": 0.0}

    def hang():
        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(model, data, np.array([0.0, 0.0, -load["kg"] * G]), np.zeros(3),
                          data.xanchor[pin].copy(), arm, data.qfrc_applied)

    ramp = round(2.0 / DT)
    for k in range(ramp):
        load["kg"] = kg * (k + 1) / ramp
        hang()
        layer.step()
    steps(layer, 0.5, hang)
    q = layer.true_state().positions[0]
    capacity = boom_capacity(machine, q[BOOM], q[ARM])
    layer.act(command(BOOM, -1.0))
    steps(layer, 0.5, hang)
    start = layer.true_state().positions[0, BOOM]
    steps(layer, 1.0, hang)
    return start - layer.true_state().positions[0, BOOM], capacity


def test_boom_lifts_just_below_its_hydraulic_capacity_and_stalls_just_above():
    machine = statics.Machine()
    point = statics.lift(machine, 3.0, 3.0, MAIN_RELIEF)
    assert point["limited_by"] == "boom"
    pose = (0.0, point["boom"], point["arm"], 0.0)
    _, capacity = lift_trial(machine, pose, point["hydraulic"])
    lifted, below = lift_trial(machine, pose, 0.97 * capacity)
    stalled, above = lift_trial(machine, pose, 1.04 * capacity)
    assert 0.97 * capacity < 0.985 * below and lifted > 2e-3
    assert 1.04 * capacity > 1.025 * above and stalled < 2e-3


def stall_moment(joint, pose):
    layer = layer_for(NOMINAL, pose)
    model, data = layer.model, layer.data
    dof, qpos = layer._qvel[0, joint], layer._qpos[0, joint]
    start = data.qpos[qpos]
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    inertia = mass[dof, dof]
    spring = min(2.0e6, 0.2 * inertia / DT ** 2)
    damping = min(2.0 * math.sqrt(spring * inertia), 0.5 * inertia / DT)

    def push():
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof] = -spring * (data.qpos[qpos] - start) - damping * data.qvel[dof]

    layer.act(command(joint, 1.0))
    steps(layer, 3.0, push)
    push()
    mujoco.mj_forward(model, data)
    return -data.qfrc_applied[dof] + data.qfrc_bias[dof], layer.chamber_pressures()[0, joint]


def test_digging_force_at_stall_matches_the_static_cylinder_limits():
    dig = statics.digging(statics.Machine())
    for joint, pose, moment in (
            (BUCKET, (0.0, 0.0, 1.5, dig["bucket_angle"]), dig["bucket_moment"]),
            (ARM, (0.0, 0.0, dig["arm_angle"], 0.6), dig["arm_moment"])):
        simulated, chambers = stall_moment(joint, pose)
        assert simulated == pytest.approx(moment, rel=0.04)
        assert MAIN_RELIEF < chambers[0] < MAIN_RELIEF + c.RELIEF_OVERRIDE + c.VALVE_DROP


def swing_inertia(layer):
    model, data = layer.model, layer.data
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    return mass[layer._qvel[0, SWING], layer._qvel[0, SWING]]


def test_swing_settles_at_the_motor_speed_turns_three_times_in_the_published_time_and_stops():
    layer = layer_for(NOMINAL)
    start = layer.true_state().positions[0, SWING]
    layer.act(command(SWING, 1.0))
    speeds, turned = [], None
    for k in range(round(15.0 / DT)):
        layer.step()
        speeds.append(layer.true_state().velocities[0, SWING])
        if turned is None and layer.true_state().positions[0, SWING] - start >= 6 * math.pi:
            turned = (k + 1) * DT
    speeds = np.array(speeds)
    steady = speeds[-round(2.0 / DT):]
    free_running = c.PUMP_FLOW / c.SWING_DISPLACEMENT
    assert steady.std() < 1e-3 * free_running
    assert steady.mean() == pytest.approx(free_running, rel=0.01)
    assert free_running < 1.06 * c.SWING_SPEED
    assert speeds.max() < 1.02 * steady.mean()
    assert 12.1 < turned < 14.1
    top = speeds[-1]
    inertia = swing_inertia(layer)
    start = layer.true_state().positions[0, SWING]
    layer.act(np.zeros((1, 4)))
    ports = []
    for _ in range(round((c.BRAKE_DELAY + 1.0) / DT)):
        layer.step()
        ports.append(np.sort(layer.chamber_pressures()[0, SWING]))
    ports = np.array(ports)
    relieving = ports[ports[:, 1] >= NOMINAL.port_relief[SWING, 0]]
    assert relieving[len(relieving) // 2:, 0].min() > c.TANK_PRESSURE - 0.01e6
    stop = layer.true_state().positions[0, SWING] - start
    braking = c.SWING_DISPLACEMENT * (NOMINAL.port_relief[SWING, 0] - c.TANK_PRESSURE)
    braking /= c.SWING_MECHANICAL
    hand = top ** 2 * inertia / (2 * braking) + top * (NOMINAL.delay[SWING] + NOMINAL.lag[SWING])
    assert stop == pytest.approx(hand, rel=0.05)
    assert abs(layer.true_state().velocities[0, SWING]) < 1e-3


def test_swing_accelerates_on_its_cross_port_relief():
    layer = layer_for(NOMINAL)
    inertia = swing_inertia(layer)
    layer.act(command(SWING, 1.0))
    trace = []
    for _ in range(round(3.0 / DT)):
        layer.step()
        trace.append(layer.true_state().velocities[0, SWING])
    reached = (np.argmax(np.array(trace) >= 0.9 * c.SWING_SPEED) + 1) * DT
    torque = c.SWING_DISPLACEMENT * (NOMINAL.port_relief[SWING, 0] - c.TANK_PRESSURE)
    torque *= c.SWING_MECHANICAL
    hand = 0.9 * c.SWING_SPEED * inertia / torque + NOMINAL.delay[SWING] + NOMINAL.lag[SWING]
    assert reached == pytest.approx(hand, rel=0.1)


def swing_with(other, pose=None, seconds=4.0):
    layer = layer_for(NOMINAL, pose)
    start = layer.true_state().positions[0, SWING]
    u = command(SWING, 1.0)
    if other is not None:
        u[0, other[0]] = other[1]
    layer.act(u)
    speeds, quarter = [], None
    for k in range(round(seconds / DT)):
        layer.step()
        speeds.append(layer.true_state().velocities[0, SWING])
        if quarter is None and layer.true_state().positions[0, SWING] - start >= math.pi / 2:
            quarter = (k + 1) * DT
    return np.array(speeds), quarter


def test_swing_keeps_its_priority_when_the_arm_shares_its_pump():
    alone, _ = swing_with(None)
    shared, _ = swing_with((ARM, 1.0))
    at = round(1.0 / DT) - 1
    assert shared[at] > 0.9 * alone[at]
    assert shared.max() < 1.1 * c.PUMP_FLOW / c.SWING_DISPLACEMENT


def test_swing_with_boom_raise_turns_ninety_degrees_in_the_published_time():
    speeds, quarter = swing_with((BOOM, -1.0), (0.0, 0.0, 1.0, 1.0))
    assert 3.0 < quarter < 3.6
    assert speeds.max() <= c.SWING_SPEED


def test_full_boom_raise_takes_both_pumps_into_the_head_chambers():
    layer = layer_for(NOMINAL, (0.0, 0.3, 1.9, 0.6))
    layer.act(command(BOOM, -1.0))
    steps(layer, 1.5)
    q = layer.true_state().positions[0]
    _, lever = c.geometry(q)
    rod_speed = abs(lever[BOOM] * layer.true_state().velocities[0, BOOM])
    head, _ = areas("boom")
    assert rod_speed == pytest.approx(c.PUMPS * c.PUMP_FLOW / head, rel=0.02)


RELIEF_SPREAD = (-0.5e6, 2.0e6)


@pytest.mark.parametrize(("joint", "value", "pumps"), [
    (ARM, 1.0, [0, 1]), (BUCKET, 1.0, [0]), (BOOM, 1.0, [0])],
    ids=["arm roll-in on both pumps", "bucket roll-in on one", "boom lower on one"])
def test_each_machine_stalls_at_its_own_main_relief(joint, value, pumps):
    machines = [replace(NOMINAL, main_relief=MAIN_RELIEF + d) for d in RELIEF_SPREAD]
    model = load_scene(2, weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), machines)
    layer.act(command(joint, value, 2))
    trace, inlets = [], []
    for _ in range(round(4.0 / DT)):
        layer.step()
        trace.append(layer.pump_pressures())
        inlets.append(layer.chamber_pressures()[:, joint].max(axis=1))
    settled = np.array(trace)[-round(0.5 / DT):]
    assert settled.std(axis=0).max() < 0.01e6
    loaded = np.array(trace)[-1][:, pumps]
    idle = np.delete(np.array(trace)[-1], pumps, axis=1)
    for i, machine in enumerate(machines):
        assert np.all(machine.main_relief < loaded[i])
        assert np.all(loaded[i] < machine.main_relief + c.RELIEF_OVERRIDE)
        assert np.all(idle[i] < 0.5 * machine.main_relief)
        assert inlets[-1][i] < NOMINAL.port_relief[joint].min()
    gap = loaded[1] - loaded[0]
    np.testing.assert_allclose(gap, RELIEF_SPREAD[1] - RELIEF_SPREAD[0], rtol=0.1)


@pytest.mark.parametrize("fraction, moves", [(0.9, False), (1.1, True)])
def test_a_held_bucket_gives_way_only_above_its_port_relief(fraction, moves):
    layer = layer_for(NOMINAL)
    dof = layer._qvel[0, BUCKET]
    steps(layer, 0.2)
    q = layer.true_state().positions[0, BUCKET]
    head, rod = areas("bucket")
    _, lever = cylinder_length("bucket", q)
    holding = (NOMINAL.port_relief[BUCKET, 0] * head - c.TANK_PRESSURE * rod) * abs(float(lever))
    bias = layer.data.qfrc_bias[dof]
    sign = np.sign(float(lever)) * c.JOINT_SIGN[BUCKET]

    def push():
        layer.data.qfrc_applied[dof] = bias - sign * fraction * holding

    steps(layer, 0.25, push)
    start = layer.true_state().positions[0, BUCKET]
    steps(layer, 0.25, push)
    moved = abs(layer.true_state().positions[0, BUCKET] - start)
    assert (moved > 0.01) if moves else (moved < 1e-3)


def swing_turn_under(layer, torque, seconds):
    dof = layer._qvel[0, SWING]
    start = layer.true_state().positions[0, SWING]
    steps(layer, seconds, lambda: layer.data.qfrc_applied.__setitem__(dof, torque))
    return abs(layer.true_state().positions[0, SWING] - start)


def test_the_parking_brake_holds_until_a_spool_opens_past_its_deadband_and_after_the_delay():
    machine = replace(NOMINAL, deadband=np.full(4, 0.2))
    layer = layer_for(machine)
    torque = 0.9 * machine.brake
    assert swing_turn_under(layer, torque, 1.0) < 1e-4
    layer.act(command(BOOM, 0.19))
    assert swing_turn_under(layer, torque, 2.0) < 1e-4
    layer.act(command(BOOM, 0.4))
    assert swing_turn_under(layer, torque, 1.0) > 0.05
    layer.act(np.zeros((1, 4)))
    assert swing_turn_under(layer, torque, c.BRAKE_DELAY - 1.0) > 0.05
    steps(layer, 1.5)
    assert swing_turn_under(layer, torque, 2.0) < 1e-4


def test_a_parked_machine_holds_2500_kg_on_a_ten_degree_side_slope():
    layer = layer_for(NOMINAL, (0.0, 0.0, 1.0, 0.5))
    model, data = layer.model, layer.data
    slope = math.radians(10.0)
    model.opt.gravity[:] = (0.0, -G * math.sin(slope), -G * math.cos(slope))
    arm = model.body(machine_prefix(0) + "arm").id
    pin = model.joint(machine_prefix(0) + "bucket").id
    dof = layer._qvel[0, SWING]

    def hang():
        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(model, data, 2500.0 * model.opt.gravity, np.zeros(3),
                          data.xanchor[pin].copy(), arm, data.qfrc_applied)

    steps(layer, 1.0, hang)
    start = layer.true_state().positions[0, SWING]
    steps(layer, 10.0, hang)
    assert abs(layer.true_state().positions[0, SWING] - start) < 1e-3
    assert abs(data.qfrc_bias[dof] - data.qfrc_applied[dof]) > 0.3 * NOMINAL.brake


RELIEF_OVERSHOOT = 3.0e6


def test_random_combined_commands_at_the_corners_keep_floor_relief_and_torque_consistent():
    machines = [extreme_machine(m) for m in RANGES["link_mass"]] + [NOMINAL]
    model = load_scene(len(machines), weld=True)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, machines)
    setting = np.zeros((len(machines), c.NODES))
    for i, m in enumerate(machines):
        setting[i, :c.PUMPS] = m.main_relief
        setting[i, c.PUMPS:] = m.port_relief.ravel()
    setting[:, c.PUMPS:c.PUMPS + 2] = np.inf
    rng = np.random.default_rng(11)
    pressures, mismatch = [], []
    for _ in range(100):
        layer.act(rng.choice([-1.0, 0.0, 1.0], size=(len(machines), 4)))
        for _ in range(layer.steps_per_decision):
            mismatch.append(stepped_force_mismatch(layer))
            pressures.append(layer._pressure.copy())
    pressures, mismatch = np.array(pressures), np.array(mismatch)
    assert np.isfinite(pressures).all()
    assert pressures.min() >= c.CAVITATION_PRESSURE - 1e-6
    assert (pressures - setting).max() < RELIEF_OVERSHOOT
    corners, nominal = mismatch[:, :2, 1:], mismatch[:, 2, 1:]
    assert np.quantile(corners, 0.99) < 2.0e3 and corners.max() < 20.0e3
    assert np.quantile(nominal, 0.999) < 1.0e3 and nominal.max() < 3.0e3
    assert mismatch[..., 0].max() < 1.0e3
    flips = np.array([chatter_per_node(pressures[:, i]) for i in range(len(machines))])
    assert flips[:2].max() <= 0.025
    assert flips[2].max() <= 0.003


def stepped_force_mismatch(layer):
    data = layer.data
    q = data.qpos[layer._qpos].copy()
    before = data.qvel[layer._qvel].copy()
    layer.step()
    ctrl = data.ctrl[layer._actuators]
    torque = ctrl - layer._stiff * (data.qvel[layer._qvel] - before)
    _, lever = c.geometry(q)
    force = -(layer._pressure @ DISPLACEMENT) * layer._torque_scale
    loaded = (np.abs(layer._gap) >= 1.0 - 1e-9) | (layer._half_play == 0.0)
    return np.where(loaded & (ctrl != 0.0), np.abs(torque / lever - force), 0.0)


def test_a_boom_driven_into_its_top_stop_jolts_the_held_arm_only_with_forces_the_oil_has():
    layer = layer_for(NOMINAL, (0.0, 0.5, 0.8, -0.3))
    layer.act(command(BOOM, 1.0))
    mismatch = np.array([stepped_force_mismatch(layer) for _ in range(round(1.5 / DT))])
    assert layer.data.qpos[layer._qpos][0, BOOM] > JOINT_LIMITS["boom"][1] - 0.01
    assert mismatch.max() < 3.0e3
    assert layer.redos > 0


def chatter_per_node(pressures):
    change = np.diff(pressures, axis=0)
    flips = (change[1:] * change[:-1] < 0) & (np.abs(change[1:]) > 0.5e6) & (
        np.abs(change[:-1]) > 0.5e6)
    return flips.mean(axis=0)


def oil_volumes(layer):
    length, _ = c.geometry(layer.data.qpos[layer._qpos])
    chambers = (layer._volume0 + VOLUME_SLOPE * length[..., None]).reshape(layer.n, -1)
    return np.concatenate([np.repeat(layer._pump_volume[:, None], c.PUMPS, 1), chambers], 1)


def test_oil_volume_balances_in_every_chamber_while_a_cavity_opens_and_refills():
    layer = layer_for(NOMINAL, (0.0, 0.0, 1.5, 0.5))
    data = layer.data
    dof = layer._qvel[0, BOOM]
    rows = np.arange(c.PUMPS + 2, c.NODES)
    worst, opened, closed = 0.0, False, False
    for k in range(400):
        data.qfrc_applied[dof] = -5.0e5 if k < 20 else (3.0e5 if k < 120 else 0.0)
        p0, v0, cavity0 = layer._pressure.copy(), oil_volumes(layer), layer._cavity.copy()
        q0 = data.qpos[layer._qpos].copy()
        layer.step()
        _, lever = c.geometry(q0)
        rod = lever * data.qvel[layer._qvel]
        change = layer._pressure - p0
        used = layer._edge_flow + layer._edge_slope * (change @ NODE_DROPS)
        compressed = v0 * layer._inv_bulk[:, None] * change
        missing = compressed - ((used @ INFLOW) + rod @ DISPLACEMENT.T) * DT
        grown = layer._cavity - cavity0
        worst = max(worst, np.abs(missing[0, rows] - grown[0, rows]).max())
        held = layer._cavity[0] > 0.0
        assert np.all(np.abs(layer._pressure[0, held] - c.CAVITATION_PRESSURE) < 1.0)
        opened |= bool(layer._cavity[0].max() > 1e-6)
        closed |= opened and layer._cavity[0].max() == 0.0
    assert opened and closed
    assert worst < 1e-9


def wobble_and_kick(layer, k):
    force = 60e3 * math.sin(2 * math.pi * 1.5 * k * DT)
    layer.data.qfrc_applied[layer._qvel[:, BOOM]] = force + (-5.0e5 if k % 400 < 10 else 0.0)
    layer.data.qfrc_applied[layer._qvel[:, ARM]] = -0.5 * force


def test_energy_closes_with_cavitation_and_outside_forces_once_the_integration_terms_count():
    machines = [draw_machine(s) for s in (100, 101)]
    model = load_scene(len(machines), weld=True)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, machines)
    rng = np.random.default_rng(5)
    stored = layer.oil_energy().copy()
    damping, growth, actuator, play = np.zeros(2), np.zeros(2), np.zeros(2), np.zeros(2)
    opened, k = False, 0
    for _ in range(100):
        layer.act(rng.uniform(-1.0, 1.0, (2, 4)))
        for _ in range(layer.steps_per_decision):
            wobble_and_kick(layer, k)
            k += 1
            p0, v0 = layer._pressure.copy(), oil_volumes(layer)
            before = data.qvel[layer._qvel].copy()
            layer.step()
            after = data.qvel[layer._qvel]
            torque = data.ctrl[layer._actuators] - layer._stiff * (after - before)
            actuator += (torque * after).sum(1) * DT
            rod_force = (layer._pressure @ DISPLACEMENT) * layer._torque_scale
            play -= np.where(layer._stiff > 0.0, 0.0, layer._rod_speed * rod_force).sum(1) * DT
            change = layer._pressure - p0
            damping += (0.5 * v0 * change ** 2).sum(1) * layer._inv_bulk
            growth += (0.5 * layer._pressure ** 2 * (oil_volumes(layer) - v0)).sum(1) * (
                layer._inv_bulk)
            opened |= bool(layer._cavity.max() > 1e-6)
    pump = layer.pump_energy
    residual = pump - layer.work_energy - layer.lost_energy - (layer.oil_energy() - stored)
    assert opened
    assert np.all(residual > 0.0)
    assert np.all(np.abs(residual - damping + growth) < 1e-9 * pump)
    assert np.all(np.abs(layer.work_energy - actuator - play) < 1e-4 * pump)


def test_held_joints_drift_slowly_but_measurably_through_their_leaks():
    leaky = np.where(np.array([[0, 0], [1, 0], [0, 1], [0, 0]], bool),
                     RANGES["holding_leakage"][1], RANGES["spool_leakage"][1])
    layer = layer_for(replace(NOMINAL, leakage=leaky))
    steps(layer, 1.0)
    start = layer.true_state().positions[0].copy()
    steps(layer, 20.0)
    assert np.abs(layer.true_state().positions[0] - start)[1:].max() > 1e-5


def stroke_time(pose, joint, value, stop=None, kg=0.0, seconds=8.0):
    layer = layer_for(NOMINAL, pose)
    model, data = layer.model, layer.data
    arm = model.body(machine_prefix(0) + "arm").id
    pin = model.joint(machine_prefix(0) + "bucket").id

    def hang():
        data.qfrc_applied[:] = 0.0
        if kg:
            mujoco.mj_applyFT(model, data, np.array([0.0, 0.0, -kg * G]), np.zeros(3),
                              data.xanchor[pin].copy(), arm, data.qfrc_applied)

    steps(layer, 1.0, hang)
    layer.act(command(joint, value))
    start = layer.true_state().positions[0, joint]
    trace, cavity = [], 0.0
    for _ in range(round(seconds / DT)):
        hang()
        layer.step()
        trace.append(layer.true_state().positions[0, joint])
        cavity = max(cavity, layer._cavity[0].sum())
        if stop is not None and (trace[-1] - stop) * np.sign(stop - start) >= 0:
            return len(trace) * DT, cavity
    trace = np.array(trace)
    moving = np.flatnonzero(np.abs(trace - trace[-1]) > 0.01 * abs(trace[-1] - trace[0]))
    return (moving[-1] + 1) * DT, cavity


ARM_OUT, ARM_IN = JOINT_LIMITS["arm"]
BUCKET_OUT, BUCKET_IN = JOINT_LIMITS["bucket"]


@pytest.mark.parametrize(("pose", "joint", "value", "published"), [
    ((0.0, 0.0, ARM_OUT + 0.02, BUCKET_IN - 0.02), BOOM, -1.0, 3.1),
    ((0.0, -0.7, ARM_OUT + 0.01, 1.2), ARM, 1.0, 3.4),
    ((0.0, -0.7, ARM_IN - 0.01, 1.2), ARM, -1.0, 2.6),
    ((0.0, -0.7, 1.6, BUCKET_OUT + 0.01), BUCKET, 1.0, 3.5),
    ((0.0, -0.7, 1.6, BUCKET_IN - 0.01), BUCKET, -1.0, 2.4),
], ids=["boom raise", "arm roll-in", "arm roll-out", "bucket roll-in", "bucket roll-out"])
def test_unloaded_strokes_take_the_published_cycle_times(pose, joint, value, published):
    taken, _ = stroke_time(pose, joint, value)
    assert abs(taken - published) <= 0.3


def test_boom_lowers_from_the_top_in_the_published_time():
    top = (0.0, JOINT_LIMITS["boom"][0] + 0.01, ARM_OUT + 0.02, BUCKET_IN - 0.02)
    taken, _ = stroke_time(top, BOOM, 1.0, stop=0.0)
    assert abs(taken - 2.2) <= 0.3


@pytest.mark.parametrize(("pose", "joint", "stop"), [
    ((0.0, -0.7, ARM_OUT + 0.02, 1.2), ARM, ARM_IN - 0.05),
    ((0.0, -1.15, 1.5, 1.2), BOOM, 0.0),
], ids=["arm roll-in", "boom lower"])
def test_an_overrunning_load_runs_at_most_twice_as_fast_as_unloaded_with_little_cavity(
        pose, joint, stop):
    empty, _ = stroke_time(pose, joint, 1.0, stop=stop)
    loaded, cavity = stroke_time(pose, joint, 1.0, stop=stop, kg=3000.0)
    assert empty / 2.0 < loaded < empty
    assert cavity < 0.5e-3


PAYLOAD = 1200.0


@pytest.mark.parametrize("joint", [ARM, BUCKET])
def test_a_1200_kg_payload_rides_through_the_arm_and_bucket_ranges_inside_the_end_stops(joint):
    lo, hi = JOINT_LIMITS[JOINT_NAMES[joint]]
    pose = [0.0, 0.0, 0.8, 0.5]
    pose[joint] = lo + 0.15
    layer = layer_for(NOMINAL, pose)
    model, data = layer.model, layer.data
    bucket = model.body(machine_prefix(0) + "bucket").id
    tip = model.site(machine_prefix(0) + TIP_SITE).id

    def hang():
        data.qfrc_applied[:] = 0.0
        mujoco.mj_applyFT(model, data, np.array([0.0, 0.0, -PAYLOAD * G]), np.zeros(3),
                          data.site_xpos[tip].copy(), bucket, data.qfrc_applied)

    seen = []
    for value in (1.0, -1.0):
        layer.act(command(joint, value))
        for _ in range(round(5.0 / DT)):
            hang()
            layer.step()
            seen.append(layer.true_state().positions[0, joint])
        assert abs(seen[-1] - (hi if value > 0 else lo)) < math.radians(1.0)
    assert lo - math.radians(0.6) < min(seen) and max(seen) < hi + math.radians(0.6)
