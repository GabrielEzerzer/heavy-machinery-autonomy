import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, RANGES, Hydraulics, draw_machine
from hma.hydraulics import circuit as c
from hma.hydraulics.layer import VOLUME_SLOPE
from hma.machine import BODY_NAMES, load_scene, machine_prefix, statics
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


def test_swing_runs_at_the_rated_speed_and_stops_within_the_relief_braking_angle():
    layer = layer_for(NOMINAL)
    model, data = layer.model, layer.data
    layer.act(command(SWING, 1.0))
    steps(layer, 5.0)
    top = layer.true_state().velocities[0, SWING]
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    inertia = mass[layer._qvel[0, SWING], layer._qvel[0, SWING]]
    start = layer.true_state().positions[0, SWING]
    layer.act(np.zeros((1, 4)))
    steps(layer, 4.0)
    stop = layer.true_state().positions[0, SWING] - start
    torque = c.SWING_DISPLACEMENT * (NOMINAL.port_relief[SWING, 0] - c.TANK_PRESSURE)
    hand = top ** 2 * inertia / (2 * torque) + top * (NOMINAL.delay[SWING] + NOMINAL.lag[SWING])
    assert top == pytest.approx(c.SWING_SPEED, rel=0.02)
    assert stop == pytest.approx(hand, rel=0.05)
    assert abs(layer.true_state().velocities[0, SWING]) < 1e-3


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


def test_each_machine_stalls_at_its_own_main_relief():
    machines = [replace(NOMINAL, main_relief=MAIN_RELIEF + d) for d in RELIEF_SPREAD]
    model = load_scene(2, weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), machines)
    layer.act(command(ARM, 1.0, 2))
    pumps, heads = [], []
    for _ in range(round(4.0 / DT)):
        layer.step()
        pumps.append(layer.pump_pressures())
        heads.append(layer.chamber_pressures()[:, ARM, 0])
    settled = np.array(pumps)[-round(0.5 / DT):]
    assert settled.std(axis=0).max() < 0.01e6
    for i, machine in enumerate(machines):
        assert np.all(machine.main_relief < settled[-1, i])
        assert np.all(settled[-1, i] < machine.main_relief + c.RELIEF_OVERRIDE)
        assert heads[-1][i] < NOMINAL.port_relief[ARM, 0]
    gap = settled[-1, 1] - settled[-1, 0]
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


def test_the_parking_brake_holds_the_swing_until_a_command_releases_it():
    layer = layer_for(NOMINAL)
    dof = layer._qvel[0, SWING]
    start = layer.true_state().positions[0, SWING]
    steps(layer, 1.0, lambda: layer.data.qfrc_applied.__setitem__(dof, 0.9 * NOMINAL.brake))
    assert abs(layer.true_state().positions[0, SWING] - start) < 1e-4
    assert layer.model.dof_frictionloss[dof] == NOMINAL.brake
    layer.act(command(BOOM, 0.3))
    layer.step()
    assert layer.model.dof_frictionloss[dof] == 0.0
    layer.act(np.zeros((1, 4)))
    steps(layer, c.BRAKE_DELAY - DT)
    assert layer.model.dof_frictionloss[dof] == 0.0
    steps(layer, 2 * DT)
    assert layer.model.dof_frictionloss[dof] == NOMINAL.brake
