from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import EFFECTS, NOMINAL, NOMINAL_SUPPLY, RANGES, Hydraulics, draw_machine
from hma.machine import (
    BODY_NAMES,
    FORCE_CAPS,
    JOINT_NAMES,
    LINK_MASSES,
    SPEED_LIMITS,
    load_scene,
    machine_prefix,
)
from hma.machine.model import COUNTERWEIGHT_MASS, VALVE_GAIN, VALVE_INERTIA, _machine_spec

DT = 0.005
SOLVER_TOLERANCE_WITHOUT_NEIGHBOUR_COUPLING = 1e-12
BUCKET = JOINT_NAMES.index("bucket")
SWING = JOINT_NAMES.index("swing")


def layer_for(machine, n=1, weld=True):
    model = load_scene(n, weld=weld)
    data = mujoco.MjData(model)
    return Hydraulics(model, data, [machine] * n)


def step_response(machine, joint, command, seconds):
    layer = layer_for(machine)
    u = np.zeros((1, 4))
    u[0, joint] = command
    layer.act(u)
    trace = []
    for _ in range(round(seconds / DT)):
        layer.step()
        trace.append(layer.true_state().velocities[0])
    return np.array(trace)


def steady_speeds(machine, commands, seconds=2.0):
    layer = layer_for(machine)
    layer.act(np.asarray(commands, float)[None])
    for _ in range(round(seconds / DT)):
        layer.step()
    return layer.true_state().velocities[0]


def only(effect, seed=7):
    return draw_machine(seed, (effect,))


def test_delay_is_the_drawn_dead_time_before_the_joint_moves():
    machine = only("delay")
    final = 0.5 * NOMINAL.speed_limit[BUCKET]

    def onset(m):
        speeds = step_response(m, BUCKET, 0.5, 1.5)[:, BUCKET]
        return np.argmax(speeds > 0.05 * final) * DT

    measured = onset(machine) - onset(replace(machine, delay=np.zeros(4)))
    assert measured == pytest.approx(machine.delay[BUCKET], abs=DT / 2 + 1e-9)


def test_lag_is_the_drawn_time_constant_after_the_delay():
    machine = only("delay")
    final = 0.5 * NOMINAL.speed_limit[BUCKET]
    speeds = step_response(machine, BUCKET, 0.5, 4.0)[:, BUCKET]
    assert speeds[-1] == pytest.approx(final, rel=1e-3)
    mean_response = np.sum(1.0 - speeds / final) * DT
    valve_lag = VALVE_INERTIA / VALVE_GAIN
    measured = mean_response - round(machine.delay[BUCKET] / DT) * DT - valve_lag
    assert measured == pytest.approx(machine.lag[BUCKET], abs=DT)


def test_deadband_swallows_commands_below_the_drawn_threshold():
    machine = only("valve")
    deadband = machine.deadband[BUCKET]
    idle = steady_speeds(machine, [0, 0, 0, 0], 1.0)
    assert np.array_equal(steady_speeds(machine, [0, 0, 0, 0.98 * deadband], 1.0), idle)
    assert steady_speeds(machine, [0, 0, 0, deadband + 0.02], 1.0)[BUCKET] > 1e-3


@pytest.mark.parametrize("command", [0.3, 0.55, 0.8])
def test_steady_speed_follows_the_drawn_curve_and_gain(command):
    machine = only("valve")
    j = BUCKET
    opening = (command - machine.deadband[j]) / (1.0 - machine.deadband[j])
    xs = np.linspace(0.0, 1.0, machine.curve.shape[1])
    expected = min(machine.gain[j] * np.interp(opening, xs, machine.curve[j]), 1.0)
    measured = steady_speeds(machine, [0, 0, 0, command])[j] / machine.speed_limit[j]
    assert measured == pytest.approx(expected, rel=2e-3)


def test_curve_gain_scales_a_straight_curve():
    machine = replace(NOMINAL, gain=np.array([1.0, 1.0, 1.0, 0.92]))
    assert steady_speeds(machine, [0, 0, 0, 0.6])[BUCKET] == pytest.approx(
        0.92 * 0.6 * NOMINAL.speed_limit[BUCKET], rel=2e-3)


@pytest.mark.parametrize("seed", [3, 11])
def test_flow_above_the_drawn_limit_is_shared_in_proportion(seed):
    machine = replace(only("flow", seed), flow=0.6)
    for commands in ([0, 1, -1, 0], [0, 1, -1, 1]):
        speeds = steady_speeds(machine, commands, 1.2)
        share = min(1.0, 0.6 * NOMINAL_SUPPLY / np.abs(commands).sum())
        np.testing.assert_allclose(speeds, share * NOMINAL.speed_limit * commands,
                                   rtol=2e-3, atol=1e-3)
    drawn = only("flow", seed)
    speeds = steady_speeds(drawn, [0, 1, -1, 1], 1.2)
    share = min(1.0, drawn.flow * NOMINAL_SUPPLY / 3)
    np.testing.assert_allclose(speeds, share * NOMINAL.speed_limit * [0, 1, -1, 1],
                               rtol=2e-3, atol=1e-3)


@pytest.mark.parametrize("joint", ["boom", "arm", "bucket"])
@pytest.mark.parametrize("direction", [-1.0, 1.0])
def test_full_command_runs_at_the_drawn_speed_limit(joint, direction):
    machine = only("limits")
    j = JOINT_NAMES.index(joint)
    u = np.zeros(4)
    u[j] = direction
    assert steady_speeds(machine, u, 1.0)[j] == pytest.approx(
        direction * machine.speed_limit[j], rel=2e-3)


def test_swing_reaches_its_drawn_speed_limit():
    machine = only("limits")
    assert steady_speeds(machine, [1, 0, 0, 0], 3.0)[SWING] == pytest.approx(
        machine.speed_limit[SWING], rel=2e-3)


def test_drawn_force_limit_is_what_a_held_joint_withstands_per_machine():
    machines = [draw_machine(seed, ("limits",)) for seed in (1, 2)]
    model = load_scene(2, weld=True)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, machines)
    caps = [m.force_limit[SWING] for m in machines]
    assert caps[0] != caps[1]
    for i, cap in enumerate(caps):
        tendon = model.tendon(machine_prefix(i) + "swing_valve").id
        assert model.tendon_frictionloss[tendon] == cap
    torque = float(np.mean(caps))
    dofs = [model.joint(machine_prefix(i) + "swing").dofadr[0] for i in range(2)]
    start = layer.true_state().positions[:, SWING].copy()
    for _ in range(200):
        data.qfrc_applied[dofs] = torque
        layer.step()
    moved = np.abs(layer.true_state().positions[:, SWING] - start)
    weak, strong = np.argmin(caps), np.argmax(caps)
    assert moved[weak] > 1e-3
    assert moved[strong] < 1e-4


@pytest.mark.parametrize("fraction", [0.5, 0.97])
def test_held_swing_withstands_below_its_cap_and_reads_the_load(fraction):
    machine = only("limits")
    layer = layer_for(machine)
    dof = layer.model.joint(machine_prefix(0) + "swing").dofadr[0]
    start = layer.true_state().positions[0, SWING]
    for _ in range(200):
        layer.data.qfrc_applied[dof] = fraction * machine.force_limit[SWING]
        layer.step()
    assert abs(layer.true_state().positions[0, SWING] - start) < 1e-4
    assert layer.true_state().loads[0, SWING] == pytest.approx(-fraction, rel=1e-3)


def test_load_signal_is_the_tendon_friction_holding_the_arm_against_gravity():
    layer = layer_for(NOMINAL)
    for _ in range(100):
        layer.step()
    data = layer.data
    dofs = [layer.model.joint(machine_prefix(0) + j).dofadr[0] for j in JOINT_NAMES]
    held = data.qfrc_bias[dofs] / NOMINAL.force_limit
    np.testing.assert_allclose(layer.true_state().loads[0], held, atol=1e-4)
    assert np.all(np.abs(held[1:]) > 1e-3)


def test_backlash_loses_the_drawn_play_on_each_reversal():
    machine = only("backlash")
    j = BUCKET

    def travels(m):
        layer = layer_for(m)
        positions = [layer.true_state().positions[0, j]]
        for command in (0.5, -0.5):
            u = np.zeros((1, 4))
            u[0, j] = command
            layer.act(u)
            for _ in range(200):
                layer.step()
            positions.append(layer.true_state().positions[0, j])
        return np.diff(positions)

    with_play, without = travels(machine), travels(NOMINAL)
    play = machine.backlash[j]
    assert play > 1e-3
    assert without[0] - with_play[0] == pytest.approx(play / 2, abs=2e-5)
    assert with_play[1] - without[1] == pytest.approx(play, abs=2e-5)


def test_sensors_report_the_state_from_the_drawn_latency_ago_with_bounded_noise():
    machine = only("sensors")
    quiet = replace(machine, position_noise=0.0, velocity_noise=0.0, load_noise=0.0)
    lag_steps = round(machine.sensor_latency / DT)
    for m, bound in ((quiet, 0.0), (machine, 1.0)):
        layer = layer_for(m)
        layer.act(np.array([[0.0, 0.5, -0.5, 0.5]]))
        history, errors = [], []
        for _ in range(60):
            layer.step()
            history.append(np.stack(layer.true_state()))
            if len(history) > lag_steps:
                errors.append(np.stack(layer.observe()) - history[-1 - lag_steps])
        errors = np.array(errors)
        amplitude = np.array([m.position_noise, m.velocity_noise, m.load_noise])[:, None, None]
        assert np.all(np.abs(errors) <= amplitude * bound + 1e-12)
        if bound:
            assert np.all(np.abs(errors).max(axis=(0, 2, 3)) > 0.9 * amplitude[:, 0, 0])


def test_changed_masses_shift_the_links_and_the_home_pose_still_rests():
    machine = only("mass")
    for weld in (True, False):
        layer = layer_for(machine, weld=weld)
        model = layer.model
        bodies = [model.body(machine_prefix(0) + b).id for b in BODY_NAMES]
        nominal = load_scene(1, weld=weld)
        upper = BODY_NAMES.index("upper")
        for k, body in enumerate(bodies):
            expected = nominal.body_mass[body] * machine.link_mass[k]
            if k == upper:
                cw = COUNTERWEIGHT_MASS
                expected = (nominal.body_mass[body] - cw) * machine.link_mass[k]
                expected += cw * machine.counterweight
            else:
                np.testing.assert_allclose(model.body_ipos[body],
                                           nominal.body_ipos[body] + machine.com_shift[k],
                                           atol=1e-12)
            assert model.body_mass[body] == pytest.approx(expected, rel=1e-12)
        assert model.body_subtreemass[bodies[0]] == pytest.approx(
            sum(model.body_mass[b] for b in bodies))
        start = layer.true_state().positions[0].copy()
        for _ in range(400):
            layer.step()
        assert np.abs(layer.true_state().positions[0] - start).max() < 2e-3
        assert np.abs(layer.true_state().velocities[0]).max() < 1e-3
        assert np.abs(layer.data.qvel[:6]).max() < 1e-3


def test_nominal_machine_leaves_the_model_as_loaded():
    layer = layer_for(only("mass"))
    layer.reset(0, NOMINAL)
    nominal = load_scene(1, weld=True)
    for field in ("body_mass", "body_ipos", "body_inertia", "body_iquat", "body_subtreemass",
                  "tendon_frictionloss", "dof_invweight0", "tendon_invweight0"):
        np.testing.assert_array_equal(getattr(layer.model, field), getattr(nominal, field))


def trajectory(machine):
    layer = layer_for(machine)
    commands = np.random.default_rng(99).uniform(-1.0, 1.0, size=(20, 1, 4))
    seen, states = [], []
    for u in commands:
        np.random.seed(int(np.random.randint(1 << 30)))
        layer.act(u)
        for _ in range(layer.steps_per_decision):
            layer.step()
            states.append(layer.data.qpos.copy())
        seen.append(np.stack(layer.observe()))
    return np.array(states), np.array(seen)


def test_same_seed_gives_the_same_machine_and_trajectory_bit_for_bit():
    a, b = draw_machine(42), draw_machine(42)
    for name, value in vars(a).items():
        assert np.array_equal(value, getattr(b, name))
    np.random.seed(1)
    first = trajectory(a)
    np.random.seed(2)
    second = trajectory(b)
    for x, y in zip(first, second, strict=True):
        assert np.array_equal(x, y)
    other = trajectory(draw_machine(43))
    assert not np.array_equal(first[0], other[0])
    assert not np.array_equal(first[1], other[1])


def test_switching_effects_keeps_each_draw_and_nominal_elsewhere():
    full = draw_machine(5)
    delay_only = draw_machine(5, ("delay",))
    assert np.array_equal(delay_only.delay, full.delay)
    assert np.array_equal(delay_only.lag, full.lag)
    for name in vars(NOMINAL):
        if name not in ("seed", "delay", "lag"):
            assert np.array_equal(getattr(delay_only, name), getattr(NOMINAL, name))
    assert not np.array_equal(draw_machine(5, ()).delay, full.delay)
    with pytest.raises(ValueError):
        draw_machine(5, ("wear",))


def test_every_draw_lies_in_its_range():
    lo_hi = {name: np.array(r) for name, r in RANGES.items()}
    for seed in range(300):
        m = draw_machine(seed, EFFECTS)
        for name in ("delay", "lag", "deadband", "gain", "flow", "backlash", "link_mass",
                     "counterweight", "sensor_latency"):
            assert np.all((lo_hi[name][0] <= getattr(m, name))
                          & (getattr(m, name) <= lo_hi[name][1]))
        for name in ("speed_limit", "force_limit", "power"):
            ratio = getattr(m, name) / getattr(NOMINAL, name)
            assert np.all((lo_hi[name][0] <= ratio) & (ratio <= lo_hi[name][1]))
        assert np.all(np.linalg.norm(m.com_shift, axis=1) <= RANGES["com_shift"][1])
        assert m.curve[:, 0].max() == 0.0
        np.testing.assert_allclose(m.curve[:, -1], 1.0)
        assert np.all(np.diff(m.curve, axis=1) > 0.0)


def test_nominal_speeds_at_the_top_of_their_range_fit_the_actuators():
    top = NOMINAL.speed_limit * RANGES["speed_limit"][1]
    assert np.all(top <= np.array([SPEED_LIMITS[j] for j in JOINT_NAMES]))
    assert np.array_equal(NOMINAL.force_limit, [FORCE_CAPS[j] for j in JOINT_NAMES])


def test_twenty_hertz_holds_each_command_for_ten_physics_steps():
    model = load_scene(1, weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), [NOMINAL], control_hz=20.0)
    assert layer.steps_per_decision == 10
    with pytest.raises(ValueError):
        Hydraulics(model, mujoco.MjData(model), [NOMINAL], control_hz=30.0)


def test_a_speed_limit_beyond_the_actuator_is_refused():
    with pytest.raises(ValueError):
        layer_for(replace(NOMINAL, speed_limit=NOMINAL.speed_limit * 1.5))


def test_mass_changes_match_the_compiler_for_the_same_masses():
    machine = replace(only("mass"), com_shift=np.zeros((len(BODY_NAMES), 3)))
    layer = layer_for(machine)
    spec = _machine_spec(0, True)
    for k, name in enumerate(BODY_NAMES):
        mesh = next(g for g in spec.body(name).geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH)
        own = LINK_MASSES[name] - (COUNTERWEIGHT_MASS if name == "upper" else 0.0)
        mesh.mass = own * machine.link_mass[k]
    spec.geom("counterweight").mass = COUNTERWEIGHT_MASS * machine.counterweight
    compiled = spec.compile()

    def tensor(model, body):
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.body_iquat[body])
        rot = rot.reshape(3, 3)
        return rot @ np.diag(model.body_inertia[body]) @ rot.T

    for name in BODY_NAMES:
        ours = layer.model.body(machine_prefix(0) + name).id
        theirs = compiled.body(name).id
        assert layer.model.body_mass[ours] == pytest.approx(compiled.body_mass[theirs], rel=1e-9)
        np.testing.assert_allclose(layer.model.body_ipos[ours], compiled.body_ipos[theirs],
                                   atol=1e-9)
        np.testing.assert_allclose(tensor(layer.model, ours), tensor(compiled, theirs),
                                   rtol=1e-7, atol=1e-6)


def quiet(machine):
    return replace(machine, position_noise=0.0, velocity_noise=0.0, load_noise=0.0)


def readings(machines, commands):
    model = load_scene(len(machines), weld=True)
    model.opt.tolerance = SOLVER_TOLERANCE_WITHOUT_NEIGHBOUR_COUPLING
    layer = Hydraulics(model, mujoco.MjData(model), machines)
    seen = []
    for u in commands:
        layer.act(u)
        for _ in range(layer.steps_per_decision):
            layer.step()
        seen.append([np.stack(layer.true_state()), np.stack(layer.observe())])
    return np.array(seen)


def test_each_machine_in_a_scene_gets_its_own_commands_and_readings():
    machines = [quiet(draw_machine(21)), quiet(draw_machine(22))]
    commands = np.random.default_rng(5).uniform(-1.0, 1.0, size=(15, 2, 4))
    together = readings(machines, commands)
    for i, machine in enumerate(machines):
        alone = readings([machine], commands[:, i:i + 1])
        np.testing.assert_allclose(together[..., i, :], alone[..., 0, :], rtol=0, atol=1e-9)
    assert np.abs(together[..., 0, :] - together[..., 1, :]).max() > 1e-2


def test_a_second_layer_on_the_same_model_starts_from_the_nominal_masses():
    machine = draw_machine(4, ("mass",))
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    Hydraulics(model, data, [machine])
    once = model.body_mass.copy(), model.body_ipos.copy(), model.body_inertia.copy()
    Hydraulics(model, data, [machine])
    for before, after in zip(once, (model.body_mass, model.body_ipos, model.body_inertia),
                             strict=True):
        np.testing.assert_array_equal(after, before)


@pytest.mark.parametrize("control_hz", [0.0, -10.0, np.inf, np.nan, 400.0])
def test_control_rate_must_be_finite_positive_and_divide_the_physics_rate(control_hz):
    model = load_scene(1, weld=True)
    with pytest.raises(ValueError):
        Hydraulics(model, mujoco.MjData(model), [NOMINAL], control_hz=control_hz)


@pytest.mark.parametrize("change", [
    {"delay": np.array([-0.05, 0.05, 0.05, 0.05])},
    {"sensor_latency": -0.01},
    {"lag": np.array([np.nan, 0.1, 0.1, 0.1])},
    {"gain": np.array([1.0, np.inf, 1.0, 1.0])},
])
def test_reset_refuses_a_machine_with_impossible_values(change):
    layer = layer_for(NOMINAL)
    with pytest.raises(ValueError):
        layer.reset(0, replace(NOMINAL, **change))
    assert layer.machines[0] is NOMINAL


def test_a_bad_pose_is_refused_before_anything_is_reset():
    layer = layer_for(NOMINAL)
    layer.act(np.array([[0.0, 0.0, 0.0, 0.5]]))
    for _ in range(40):
        layer.step()
    before = np.stack(layer.true_state()), layer.model.body_mass.copy(), layer._spool.copy()
    with pytest.raises(ValueError):
        layer.reset(0, only("mass"), pose=[0.0, 0.0, np.nan, 0.4])
    assert layer.machines[0] is NOMINAL
    after = np.stack(layer.true_state()), layer.model.body_mass, layer._spool
    for x, y in zip(before, after, strict=True):
        np.testing.assert_array_equal(x, y)


def test_load_readings_follow_friction_rows_that_appear_and_vanish():
    machines = [draw_machine(31, ("limits",)), draw_machine(32, ("limits",))]
    model = load_scene(2, weld=True)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, machines)
    dofs = np.array([[model.joint(machine_prefix(i) + j).dofadr[0] for j in JOINT_NAMES]
                     for i in range(2)])
    caps = np.array([m.force_limit for m in machines])
    toggled = model.tendon(machine_prefix(0) + "swing_valve").id
    rows_seen = set()
    for k in range(120):
        model.tendon_frictionloss[toggled] = 0.0 if 30 <= k < 90 else caps[0, SWING]
        data.qfrc_applied[dofs[:, SWING]] = 0.5 * caps[:, SWING]
        layer.step()
        rows_seen.add(int(np.sum(data.efc_type == mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON)))
        loads = layer.true_state().loads
        np.testing.assert_allclose(loads * caps, data.qfrc_constraint[dofs], rtol=1e-9, atol=1e-6)
        if 30 <= k < 90:
            assert loads[0, SWING] == 0.0
        if k >= 25:
            assert loads[1, SWING] == pytest.approx(-0.5, rel=1e-3)
    assert rows_seen == {7, 8}


def test_each_reading_carries_one_noise_draw_however_often_it_is_observed():
    machine = draw_machine(7)
    commands = np.random.default_rng(8).uniform(-1.0, 1.0, size=(12, 1, 4))

    def run(extra_calls):
        layer = layer_for(machine)
        seen = []
        for u in commands:
            layer.act(u)
            for _ in range(layer.steps_per_decision):
                layer.step()
                for _ in range(extra_calls):
                    first, again = np.stack(layer.observe()), np.stack(layer.observe())
                    assert np.array_equal(first, again)
            seen.append(np.stack(layer.observe()))
        return np.array(seen), layer.data.qpos.copy()

    rarely, often = run(0), run(3)
    assert np.array_equal(rarely[0], often[0])
    assert np.array_equal(rarely[1], often[1])


ARM = JOINT_NAMES.index("arm")


def pushed_arm(machine, load, command, seconds):
    layer = layer_for(machine)
    data = layer.data
    dof = layer.model.joint(machine_prefix(0) + "arm").dofadr[0]
    u = np.zeros((1, 4))
    u[0, ARM] = command
    layer.act(u)
    trace = []
    for _ in range(round(seconds / DT)):
        data.qfrc_applied[dof] = load(data.qvel[dof])
        layer.step()
        trace.append((data.ctrl[layer._actuators[0, ARM]], data.qvel[dof]))
    return layer, np.array(trace).T


@pytest.mark.parametrize("kind", ["steady", "viscous"])
def test_a_joint_pushing_near_its_cap_moves_at_the_power_limited_speed(kind):
    cap = NOMINAL.force_limit[ARM]
    drag = 2.0 * NOMINAL.power / NOMINAL.speed_limit[ARM] ** 2
    load = (lambda v: -0.9 * cap) if kind == "steady" else (lambda v: -drag * v)
    layer, (commands, _) = pushed_arm(NOMINAL, load, 1.0, 1.5)
    state = layer.true_state()
    power = state.loads[0, ARM] * cap * state.velocities[0, ARM]
    assert power == pytest.approx(NOMINAL.power, rel=1e-2)
    assert state.velocities[0, ARM] < 0.75 * NOMINAL.speed_limit[ARM]
    assert np.abs(np.diff(commands[-100:], 2)).max() < 1e-4


def test_power_limit_holds_still_at_a_stall_and_at_rest():
    cap = NOMINAL.force_limit[ARM]
    for load, command in ((lambda v: -1.2 * cap, 1.0), (lambda v: 0.0, -1.0),
                          (lambda v: 0.0, 0.0)):
        layer, (commands, speeds) = pushed_arm(NOMINAL, load, command, 4.0)
        assert np.ptp(commands[-200:]) < 1e-9
        assert np.ptp(speeds[-200:]) < 1e-4
        if command:
            assert abs(layer.true_state().loads[0, ARM]) == pytest.approx(1.0)
            assert abs(commands[-1]) == pytest.approx(NOMINAL.power / NOMINAL.force_limit[ARM])
    assert commands[-1] == 0.0


def test_free_air_motion_never_reaches_the_power_limit():
    unlimited = replace(NOMINAL, power=1e12)
    for u in ([0, 1, -1, 1], [1, -1, 1, 1], [-1, -1, 1, -1]):
        for machine in (NOMINAL, draw_machine(9)):
            runs = [steady_speeds(m, u, 1.5) for m in (machine, replace(machine, power=1e12))]
            assert np.array_equal(*runs)
    assert np.array_equal(steady_speeds(NOMINAL, [0, -1, 1, 1]),
                          steady_speeds(unlimited, [0, -1, 1, 1]))


def test_curve_gain_applies_before_the_drawn_speed_limit_clips():
    limits = only("limits")
    top = limits.speed_limit[BUCKET]
    weak = replace(limits, gain=np.array([1.0, 1.0, 1.0, 0.92]))
    strong = replace(limits, gain=np.array([1.0, 1.0, 1.0, 1.08]))
    assert steady_speeds(weak, [0, 0, 0, 1.0])[BUCKET] == pytest.approx(0.92 * top, rel=2e-3)
    for command in (0.95, 1.0):
        assert steady_speeds(strong, [0, 0, 0, command])[BUCKET] == pytest.approx(top, rel=2e-3)
    assert steady_speeds(strong, [0, 0, 0, 0.9])[BUCKET] == pytest.approx(0.972 * top, rel=2e-3)


@pytest.mark.parametrize("machine", [NOMINAL, draw_machine(2)], ids=["nominal", "drawn"])
def test_reversing_against_a_resisting_load_stays_within_the_power_limit(machine):
    layer = layer_for(machine)
    data = layer.data
    dofs, actuators = layer._qvel[0], layer._actuators[0]
    cap = machine.force_limit
    commands = np.random.default_rng(3).choice([-1.0, 1.0], size=(60, 1, 4))
    power = []
    for u in commands:
        layer.act(u)
        for _ in range(layer.steps_per_decision):
            data.qfrc_applied[dofs] = -0.95 * cap * np.sign(data.ctrl[actuators])
            layer.step()
            force = layer.true_state().loads[0] * cap
            power.append(np.maximum(force * data.ctrl[actuators], 0.0).sum())
    assert np.mean(power[200:]) <= 1.02 * machine.power
    assert np.mean(power[200:]) > 0.7 * machine.power
