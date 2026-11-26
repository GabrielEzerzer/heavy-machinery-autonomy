from dataclasses import replace

import mujoco
import numpy as np
import pytest

import hma.hydraulics.layer as layer_module
from hma.hydraulics import EFFECTS, NOMINAL, RANGES, Hydraulics, draw_machine
from hma.hydraulics import circuit as c
from hma.hydraulics.params import PORT_RELIEF_TOLERANCE, RELIEF_TOLERANCE
from hma.machine import BODY_NAMES, JOINT_NAMES, load_scene, machine_prefix
from hma.machine.cylinders import MAIN_RELIEF, seal_friction
from hma.machine.ground import Ground, apply_ground
from hma.machine.load import LIFT_CLEARANCE, apply_load, block, hook_load, place_load
from hma.machine.mass import Solid, body_parts, combine, full_inertia, point
from hma.machine.measure import LOAD_AWAY, TRAVEL_POSE, slope
from hma.machine.model import (
    CONTACT_MARGIN,
    COUNTERWEIGHT_MASS,
    LOAD,
    TRAVEL_NAMES,
    VALVE_GAIN,
    VALVE_INERTIA,
    _machine_spec,
    wheel_points,
)
from hma.machine.tracks import (
    BELT_ARMATURE,
    FRICTION,
    SIDES,
    TRAVEL_POWER,
    TRAVEL_SPEED,
    WHEEL_RADIUS,
    WHEELS,
    drive_forces,
    drives,
    wheel_name,
)
from scenes import dig_pit

DT = 0.005
SWING = JOINT_NAMES.index("swing")
ARM = JOINT_NAMES.index("arm")
BUCKET = JOINT_NAMES.index("bucket")


def layer_for(machine, n=1, weld=True, control_hz=10.0):
    model = dig_pit(load_scene(n, weld=weld), n)
    data = mujoco.MjData(model)
    return Hydraulics(model, data, [machine] * n, control_hz=control_hz)


def run(layer, commands, seconds):
    layer.act(np.asarray(commands, float).reshape(layer.n, -1))
    trace = []
    for _ in range(round(seconds / DT)):
        layer.step()
        trace.append(layer.true_state().velocities[0].copy())
    return np.array(trace)


def only(effect, seed=7):
    return draw_machine(seed, (effect,))


def quiet(machine):
    return replace(machine, position_noise=0.0, velocity_noise=0.0, pressure_noise=0.0)


def test_delay_is_the_drawn_dead_time_before_the_joint_moves():
    machine = only("delay")

    def onset(m):
        speeds = run(layer_for(m), [0, 0, 0, 0.5], 1.5)[:, BUCKET]
        return np.argmax(speeds > 0.02) * DT

    measured = onset(machine) - onset(replace(machine, delay=np.zeros(4)))
    assert measured == pytest.approx(machine.delay[BUCKET], abs=DT / 2 + 1e-9)


def test_spool_follows_its_delayed_command_with_the_drawn_lag():
    machine = only("delay")
    layer = layer_for(machine)
    layer.act(np.array([[0.0, 0.0, 0.0, 0.5]]))
    spool = []
    for _ in range(round(2.0 / DT)):
        layer.step()
        spool.append(layer._spool[0, BUCKET])
    spool = np.array(spool)
    delay = round(machine.delay[BUCKET] / DT)
    assert np.all(spool[:delay] == 0.0)
    k = np.arange(len(spool) - delay) + 1
    expected = 0.5 * (1.0 - (1.0 - layer._alpha[0, BUCKET]) ** k)
    np.testing.assert_allclose(spool[delay:], expected, rtol=1e-12)
    assert 1.0 - layer._alpha[0, BUCKET] == pytest.approx(np.exp(-DT / machine.lag[BUCKET]))


def test_deadband_swallows_commands_below_the_drawn_threshold():
    machine = only("valve")
    deadband = machine.deadband[BUCKET]
    idle = run(layer_for(machine), [0, 0, 0, 0], 1.0)[-1]
    layer = layer_for(machine)
    below = run(layer, [0, 0, 0, 0.98 * deadband], 1.0)[-1]
    assert layer._openings()[0, BUCKET] == 0.0
    np.testing.assert_allclose(below[1:], idle[1:], rtol=0, atol=1e-5)
    assert run(layer_for(machine), [0, 0, 0, deadband + 0.05], 1.0)[-1, BUCKET] > 1e-3


@pytest.mark.parametrize("command", [0.3, 0.55, 0.8, 1.0])
def test_valve_opening_follows_the_drawn_curve_and_gain(command):
    machine = only("valve")
    layer = layer_for(machine)
    run(layer, [0, 0, 0, command], 2.0)
    j = BUCKET
    opening = (command - machine.deadband[j]) / (1.0 - machine.deadband[j])
    xs = np.linspace(0.0, 1.0, machine.curve.shape[1])
    expected = min(machine.gain[j] * np.interp(opening, xs, machine.curve[j]), 1.0)
    assert layer._openings()[0, j] == pytest.approx(expected, rel=1e-7)


def test_pumps_stay_within_their_drawn_flow_and_the_engine_power():
    for machine in (NOMINAL, only("flow", 3), only("flow", 11)):
        layer = layer_for(machine)
        layer.act(np.array([[1.0, -1.0, 1.0, 1.0]]))
        power, flows = [], []
        for _ in range(round(6.0 / DT)):
            layer.step()
            power.append(layer.pump_power[0])
            flows.append(layer._pump[0].copy())
        limit = machine.power * c.PUMP_EFFICIENCY
        window = round(0.5 / DT)
        averaged = np.convolve(power, np.ones(window) / window, mode="valid")
        assert np.max(flows) <= c.PUMP_FLOW * machine.pump_flow * (1 + 1e-12)
        assert np.max(power) <= 1.3 * limit
        assert np.max(averaged) <= 1.1 * limit
        assert np.mean(power[-round(1.0 / DT):]) == pytest.approx(limit, rel=0.01)


def test_a_cylinder_reaching_its_end_stop_overshoots_the_power_only_while_the_pump_turns_back():
    layer = layer_for(NOMINAL)
    layer.act(np.array([[0.0, 0.0, 1.0, 0.0]]))
    power = []
    for _ in range(round(4.0 / DT)):
        layer.step()
        power.append(layer.pump_power[0])
    share = np.array(power) / (NOMINAL.power * c.PUMP_EFFICIENCY)
    window = round(0.5 / DT)
    assert 1.2 < share.max() < 1.8
    assert np.convolve(share, np.ones(window) / window, mode="valid").max() < 1.25
    assert share[-window:].mean() == pytest.approx(1.0, abs=0.01)


def test_pressure_readings_are_the_chambers_plus_the_drawn_offset_never_negative():
    machine = quiet(replace(only("sensors"), sensor_latency=0.0))
    layer = layer_for(machine)
    for _ in range(50):
        layer.step()
        seen = layer.observe().pressures[0]
        expected = np.maximum(layer.chamber_pressures()[0] + machine.pressure_offset, 0.0)
        np.testing.assert_allclose(seen, expected, rtol=0, atol=1e-6)
        assert np.all(seen >= 0.0)
    assert np.abs(machine.pressure_offset).max() > 0.05e6


def test_sensors_report_the_state_from_the_drawn_latency_ago_with_bounded_noise():
    machine = only("sensors")
    lag_steps = round(machine.sensor_latency / DT)
    for m, bound in ((quiet(machine), 0.0), (machine, 1.0)):
        layer = layer_for(m)
        layer.act(np.array([[0.0, 0.5, -0.5, 0.5]]))
        history, errors = [], []
        for _ in range(60):
            layer.step()
            truth = layer.true_state()
            history.append(np.concatenate([truth.positions[0], truth.velocities[0],
                                           np.maximum(truth.pressures[0] + m.pressure_offset,
                                                      0.0).T.ravel()]))
            if len(history) > lag_steps:
                seen = layer.observe()
                reading = np.concatenate([seen.positions[0], seen.velocities[0],
                                          seen.pressures[0].T.ravel()])
                errors.append(reading - history[-1 - lag_steps])
        errors = np.abs(np.array(errors))
        amplitude = np.repeat([m.position_noise, m.velocity_noise, m.pressure_noise,
                               m.pressure_noise], 4)
        assert np.all(errors <= amplitude * bound + 1e-6)
        if bound:
            assert np.all(errors.max(axis=0) > 0.8 * amplitude)


def test_backlash_crosses_the_drawn_play_when_the_load_reverses():
    machine = only("backlash")
    pose = [0.0, -0.6, 1.9, 0.6]

    def crowd(m):
        layer = layer_for(m)
        layer.reset(0, m, pose)
        layer.act(np.array([[0.0, 0.0, 0.3, 0.0]]))
        trace = []
        for _ in range(round(6.5 / DT)):
            layer.step()
            trace.append((layer.true_state().positions[0, ARM], layer._gap[0, ARM],
                          layer.data.qfrc_bias[layer._qvel[0, ARM]]))
        return np.array(trace).T

    q, gap, bias = crowd(machine)
    q_nominal, _, _ = crowd(NOMINAL)
    reversal = np.argmax(bias > 0.0)
    assert 0 < reversal < len(bias) - round(0.5 / DT)
    assert np.all(gap[:reversal] < 0.0)
    assert np.argmax(gap > 0.99) > reversal
    assert gap[-1] == 1.0
    assert q_nominal[-1] - q[-1] == pytest.approx(machine.backlash[ARM], rel=0.15)


def test_each_held_cylinder_drags_by_its_seals_at_its_own_head_and_rod_pressures():
    layer = layer_for(NOMINAL)
    run(layer, [0, 0, 0, 0], 0.2)
    data = layer.data
    chambers = layer.chamber_pressures()[0, 1:]
    speed = layer._piston_speed[0, 1:].copy()
    lever = np.abs(c.geometry(data.qpos[layer._qpos])[1][0, 1:])
    expected = seal_friction(chambers[:, 0], chambers[:, 1], speed) * lever
    swapped = seal_friction(chambers[:, 1], chambers[:, 0], speed) * lever
    assert (np.abs(swapped / expected - 1.0) > 0.05).all()
    layer.step()
    rows = data.efc_type[:data.nefc] == mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF
    loss = dict(zip(data.efc_id[:data.nefc][rows], data.efc_frictionloss[:data.nefc][rows],
                    strict=True))
    for dof, base, drag in zip(layer._qvel[0, 1:], layer._base_frictionloss[0, 1:], expected,
                               strict=True):
        assert loss[dof] == pytest.approx(base + drag, rel=1e-9)


def undercarriage(model, bodies):
    masses = model.body_mass[bodies]
    places = np.array([model.body_ipos[b] + (model.body_pos[b] if b != bodies[0] else 0.0)
                       for b in bodies])
    return masses.sum(), masses @ places / masses.sum()


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
            elif k > 0:
                np.testing.assert_allclose(model.body_ipos[body],
                                           nominal.body_ipos[body] + machine.com_shift[k],
                                           atol=1e-12)
            assert model.body_mass[body] == pytest.approx(expected, rel=1e-12)
        lower = [bodies[0], *(model.body(machine_prefix(0) + wheel_name(s, k)).id
                              for s in SIDES for k in range(WHEELS))]
        mass, com = undercarriage(model, lower)
        nominal_mass, nominal_com = undercarriage(nominal, lower)
        assert mass == pytest.approx(nominal_mass * machine.link_mass[0], rel=1e-12)
        np.testing.assert_allclose(com, nominal_com + machine.com_shift[0], atol=1e-12)
        dofs = model.body_dofadr[lower[1:]]
        assert model.dof_armature[dofs].sum() == pytest.approx(
            2 * BELT_ARMATURE * machine.link_mass[0], rel=1e-12)
        assert model.body_subtreemass[bodies[0]] == pytest.approx(
            sum(model.body_mass[b] for b in bodies)
            + machine.link_mass[0] * sum(m for m, _ in wheel_points()))
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
                  "dof_invweight0", "dof_armature"):
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
        seen.append(np.concatenate([x.ravel() for x in layer.observe()]))
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
    ranges = {name: np.array(r) for name, r in RANGES.items()}

    def within(value, name, scale=1.0):
        lo, hi = ranges[name] * scale
        return np.all((lo <= value) & (value <= hi))

    holding = np.array([[False, False], [True, False], [False, True], [False, False]])
    for seed in range(300):
        m = draw_machine(seed, EFFECTS)
        for name in ("delay", "lag", "deadband", "gain", "spool_flow", "pump_flow",
                     "bulk_modulus", "pump_volume", "brake", "backlash", "link_mass",
                     "counterweight", "sensor_latency", "pressure_offset", "seal_friction"):
            assert within(getattr(m, name), name)
        assert within(m.power, "power", c.ENGINE_POWER)
        assert within(m.dead_volume[1:], "dead_volume")
        assert within(m.dead_volume[0], "swing_volume")
        assert within(m.leakage[1:][holding[1:]], "holding_leakage")
        assert within(m.leakage[1:][~holding[1:]], "spool_leakage")
        assert within(m.leakage[0], "motor_leakage")
        lo, hi = RELIEF_TOLERANCE
        assert MAIN_RELIEF + lo <= m.main_relief <= MAIN_RELIEF + hi
        assert np.all(m.port_relief[0] == m.port_relief[0, 0])
        assert c.SWING_RELIEF + lo <= m.port_relief[0, 0] <= c.SWING_RELIEF + hi
        lo, hi = PORT_RELIEF_TOLERANCE
        extra = m.port_relief[1:] - NOMINAL.port_relief[1:]
        assert np.all((lo <= extra) & (extra <= hi))
        assert np.all(np.linalg.norm(m.com_shift, axis=1) <= RANGES["com_shift"][1])
        assert m.curve[:, 0].max() == 0.0
        np.testing.assert_allclose(m.curve[:, -1], 1.0)
        assert np.all(np.diff(m.curve, axis=1) > 0.0)


def test_twenty_hertz_holds_each_command_for_ten_physics_steps():
    model = load_scene(1, weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), [NOMINAL], control_hz=20.0)
    assert layer.steps_per_decision == 10
    with pytest.raises(ValueError):
        Hydraulics(model, mujoco.MjData(model), [NOMINAL], control_hz=30.0)


def test_mass_changes_match_the_compiler_for_the_same_masses():
    machine = replace(only("mass"), com_shift=np.zeros((len(BODY_NAMES), 3)))
    layer = layer_for(machine)
    spec = _machine_spec(0)
    parts = body_parts()
    parts["base"] = [*parts["base"], *(point(-m, pos) for m, pos in wheel_points())]
    for k, name in enumerate(BODY_NAMES):
        scaled = [Solid(p.mass * machine.link_mass[k], p.com, p.inertia * machine.link_mass[k])
                  for p in parts[name]]
        if name == "upper":
            cw = parts[name][-1]
            assert cw.mass == COUNTERWEIGHT_MASS
            scaled[-1] = Solid(cw.mass * machine.counterweight, cw.com,
                               cw.inertia * machine.counterweight)
        solid = combine(scaled)
        body = spec.body(name)
        body.mass, body.ipos, body.fullinertia = solid.mass, solid.com, full_inertia(solid)
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


def readings(machines, commands):
    model = load_scene(len(machines), weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), machines)
    seen = []
    for u in commands:
        layer.act(u)
        for _ in range(layer.steps_per_decision):
            layer.step()
        truth, sensed = layer.true_state(), layer.observe()
        seen.append([np.concatenate([truth.positions, truth.velocities], axis=1),
                     np.concatenate([sensed.positions, sensed.velocities], axis=1),
                     truth.pressures.reshape(len(machines), -1) / 1e6,
                     sensed.pressures.reshape(len(machines), -1) / 1e6])
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
    {"gain": np.array([1.0, 0.0, 1.0, 1.0])},
    {"spool_flow": np.array([1.0, 1.0, 0.0, 1.0])},
    {"deadband": np.array([0.0, -0.1, 0.0, 0.0])},
    {"backlash": np.array([0.0, 0.0, -0.01, 0.0])},
    {"link_mass": np.array([1.0, 1.0, -1.0, 1.0, 1.0])},
    {"counterweight": 0.0},
    {"pressure_noise": -0.01},
    {"curve": NOMINAL.curve[:, ::-1].copy()},
    {"main_relief": 0.5e6},
    {"port_relief": np.full((3, 2), 30e6)},
    {"bulk_modulus": 0.0},
    {"dead_volume": np.zeros((4, 2))},
    {"leakage": -NOMINAL.leakage},
    {"brake": -1.0},
    {"pump_flow": 0.0},
    {"delay": np.full(3, 0.05)},
    {"curve": NOMINAL.curve[:, :-1].copy()},
    {"leakage": NOMINAL.leakage.ravel()},
    {"port_relief": np.full(NOMINAL.port_relief.shape, c.TANK_PRESSURE)},
    {"curve": np.minimum(NOMINAL.curve + 0.1, 1.0)},
    {"curve": 0.9 * NOMINAL.curve},
])
def test_reset_refuses_a_machine_with_impossible_values_and_leaves_the_layer_alone(change):
    layer = layer_for(only("mass"))
    layer.act(np.array([[0.5, -0.5, 0.5, -0.5]]))
    for _ in range(40):
        layer.step()

    def snapshot():
        return [np.concatenate([x.ravel() for x in layer.true_state()]),
                layer.model.body_mass.copy(), layer.model.dof_armature.copy(),
                layer._spool.copy(), layer._relief.copy(), layer._pressure.copy(),
                layer._leakage.copy(), layer._slope.copy(), layer._delay_index.copy()]

    before, machine = snapshot(), layer.machines[0]
    with pytest.raises(ValueError):
        layer.reset(0, replace(NOMINAL, **change))
    assert layer.machines[0] is machine
    for x, y in zip(before, snapshot(), strict=True):
        np.testing.assert_array_equal(x, y)


def test_stepping_leaves_the_model_as_found_and_a_later_layer_matches_a_fresh_one():
    machine = draw_machine(5)
    commands = np.random.default_rng(3).uniform(-1.0, 1.0, size=(12, 1, 4))

    def drive(layer):
        seen = []
        for u in commands:
            layer.act(u)
            for _ in range(layer.steps_per_decision):
                layer.step()
            seen.append(np.concatenate([x.ravel() for x in layer.true_state()]))
        return np.array(seen)

    fresh = drive(layer_for(machine))
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [draw_machine(6)])
    loaded = model.dof_armature.copy(), model.dof_frictionloss.copy()
    drive(layer)
    np.testing.assert_array_equal(model.dof_armature, loaded[0])
    np.testing.assert_array_equal(model.dof_frictionloss, loaded[1])
    np.testing.assert_allclose(drive(Hydraulics(model, data, [machine])), fresh,
                               rtol=0, atol=1e-9)


def test_a_layer_needs_at_least_one_machine():
    model = load_scene(1, weld=True)
    with pytest.raises(ValueError):
        Hydraulics(model, mujoco.MjData(model), [])


def test_a_bad_pose_is_refused_before_anything_is_reset():
    layer = layer_for(NOMINAL)
    layer.act(np.array([[0.0, 0.0, 0.0, 0.5]]))
    for _ in range(40):
        layer.step()
    before = (np.concatenate([x.ravel() for x in layer.true_state()]),
              layer.model.body_mass.copy(), layer._spool.copy())
    with pytest.raises(ValueError):
        layer.reset(0, only("mass"), pose=[0.0, 0.0, np.nan, 0.4])
    assert layer.machines[0] is NOMINAL
    after = (np.concatenate([x.ravel() for x in layer.true_state()]), layer.model.body_mass,
             layer._spool)
    for x, y in zip(before, after, strict=True):
        np.testing.assert_array_equal(x, y)


def test_each_reading_carries_one_noise_draw_however_often_it_is_observed():
    machine = draw_machine(7)
    commands = np.random.default_rng(8).uniform(-1.0, 1.0, size=(12, 1, 4))

    def run_with(extra_calls):
        layer = layer_for(machine)
        seen = []
        for u in commands:
            layer.act(u)
            for _ in range(layer.steps_per_decision):
                layer.step()
                for _ in range(extra_calls):
                    first, again = layer.observe(), layer.observe()
                    for x, y in zip(first, again, strict=True):
                        assert np.array_equal(x, y)
            seen.append(np.concatenate([x.ravel() for x in layer.observe()]))
        return np.array(seen), layer.data.qpos.copy()

    rarely, often = run_with(0), run_with(3)
    assert np.array_equal(rarely[0], often[0])
    assert np.array_equal(rarely[1], often[1])


def episode(layer, index, machine, commands, others=None):
    layer.reset(index, machine)
    seen = []
    for u in commands:
        full = np.zeros((layer.n, 4)) if others is None else others.copy()
        full[index] = u
        layer.act(full)
        for _ in range(layer.steps_per_decision):
            layer.step()
        sensed = layer.observe()
        seen.append(np.concatenate([sensed.positions[index], sensed.velocities[index],
                                    sensed.pressures[index].ravel() / 1e6,
                                    layer.true_state().positions[index]]))
    return np.array(seen)


@pytest.mark.parametrize("machine", [draw_machine(5), NOMINAL], ids=["drawn", "nominal"])
@pytest.mark.parametrize("stray_steps", [0, 3])
def test_an_episode_after_a_reset_repeats_a_fresh_layer_bit_for_bit(stray_steps, machine):
    commands = np.random.default_rng(4).uniform(-1.0, 1.0, size=(12, 4))
    fresh = episode(layer_for(machine), 0, machine, commands)
    layer = layer_for(machine)
    episode(layer, 0, machine, commands[::-1])
    layer.act(np.ones((1, 4)))
    for _ in range(round(2.0 / DT)):
        layer.data.qfrc_applied[layer._qvel[0]] = -2.0e5
        layer.step()
    assert layer.chamber_pressures().max() > 30e6
    for _ in range(stray_steps):
        layer.step()
    assert np.array_equal(episode(layer, 0, machine, commands), fresh)


SCRATCH = {"_valve", "_demand_in", "_nodes", "_edge_flow", "_weights", "_edge_slope", "_capacity",
           "_noise_draws", "_rhs", "_state"}


def test_a_reset_leaves_every_state_array_as_a_fresh_layer_has_it():
    machine = draw_machine(5)
    commands = np.random.default_rng(4).uniform(-1.0, 1.0, size=(12, 4))
    fresh, used = layer_for(machine), layer_for(machine)
    episode(used, 0, machine, commands)
    used.act(np.ones((1, 4)))
    for _ in range(round(2.0 / DT)):
        used.data.qfrc_applied[used._qvel[0]] = -2.0e5
        used.step()
    assert np.abs(used._rod_speed).max() > 0.0
    for layer in (fresh, used):
        layer.reset(0, machine)
    for name, value in vars(fresh).items():
        if isinstance(value, np.ndarray) and name not in SCRATCH:
            assert np.array_equal(value, getattr(used, name)), name


def test_resetting_one_machine_leaves_the_other_machines_noise_alone():
    machines = [draw_machine(5), draw_machine(6)]
    commands = np.random.default_rng(6).uniform(-1.0, 1.0, size=(12, 4))
    running = np.full((2, 4), 0.4)

    def scene(reset_machine_1):
        model = load_scene(2, weld=True)
        layer = Hydraulics(model, mujoco.MjData(model), machines)
        first = episode(layer, 1, machines[1], commands, running)
        for _ in range(3):
            layer.step()
        if reset_machine_1:
            second = episode(layer, 1, machines[1], commands, running)
        else:
            layer.act(running)
            for _ in range(len(commands) * layer.steps_per_decision):
                layer.step()
            second = None
        sensed = layer.observe()
        return first, second, np.concatenate([sensed.positions[0], sensed.velocities[0],
                                              sensed.pressures[0].ravel() / 1e6])

    first, second, other = scene(True)
    _, _, undisturbed = scene(False)
    np.testing.assert_allclose(second, first, rtol=0, atol=1e-12)
    np.testing.assert_allclose(other, undisturbed, rtol=0, atol=1e-9)


def uphill_speed():
    model = load_scene(1)
    data = mujoco.MjData(model)
    apply_ground(model, 0, slope(along=np.radians(20.0)))
    layer = Hydraulics(model, data, [NOMINAL])
    layer.reset(0, NOMINAL, TRAVEL_POSE)
    place_load(model, data, 0, *LOAD_AWAY)
    data.ctrl[[model.actuator(machine_prefix(0) + t).id for t in TRAVEL_NAMES]] = TRAVEL_SPEED
    for _ in range(round(3.0 / DT)):
        layer.step()
    start = data.qpos[:3].copy()
    for _ in range(round(1.0 / DT)):
        layer.step()
    weight = model.body_subtreemass[model.body(machine_prefix(0) + "base").id] * 9.81
    return float(np.linalg.norm(data.qpos[:3] - start)), TRAVEL_POWER / (weight * np.sin(
        np.radians(20.0)))


def test_the_layer_steps_through_the_track_drive(monkeypatch):
    speed, powered = uphill_speed()
    assert speed == pytest.approx(powered, rel=0.03)
    monkeypatch.setattr(layer_module, "track_step", mujoco.mj_step)
    assert uphill_speed()[0] > 1.1 * powered


BETWEEN_STEPS = int(mujoco.mjtState.mjSTATE_INTEGRATION) & ~int(mujoco.mjtState.mjSTATE_CTRL)
INTEGRATION = mujoco.mjtState.mjSTATE_INTEGRATION


def state(model, data, kind=INTEGRATION):
    out = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, out, kind)
    return out


def working_and_travelling(seed, seconds, each=None):
    model = load_scene(1)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [draw_machine(seed)])
    rng = np.random.default_rng(seed)
    travel = [model.actuator(machine_prefix(0) + t).id for t in TRAVEL_NAMES]
    for _ in range(round(seconds * layer.control_hz)):
        layer.act(rng.uniform(-1.0, 1.0, (1, 4)))
        data.ctrl[travel] = rng.uniform(0.3, 1.0, 2) * TRAVEL_SPEED
        for _ in range(layer.steps_per_decision):
            layer.step()
            if each is not None and each(model, data, layer):
                return model, data, layer
    return model, data, layer


def test_a_retaken_step_with_the_tracks_driving_matches_one_step_from_the_same_state(monkeypatch):
    calls = []
    track_step = layer_module.track_step

    def recording(model, data):
        calls.append((state(model, data), model.dof_armature.copy(),
                      model.dof_frictionloss.copy()))
        track_step(model, data)

    monkeypatch.setattr(layer_module, "track_step", recording)
    belt = []

    def retaken(model, data, layer):
        attempts = calls[:]
        calls.clear()
        lay = drives(model)
        moving = np.abs(data.qvel[lay.dofs] * WHEEL_RADIUS).min() > 0.2
        if len(attempts) > 1 and moving:
            belt.append(attempts)
            return True
        return False

    model, data, layer = working_and_travelling(11, 30.0, retaken)
    assert belt, "no step was retaken with both tracks moving"
    attempts = belt[0]
    starts = [state_without_ctrl(model, a[0]) for a in attempts]
    for later in starts[1:]:
        assert np.array_equal(later, starts[0])
    final, armature, friction = attempts[-1]
    once = mujoco.MjData(model)
    mujoco.mj_setState(model, once, final, INTEGRATION)
    kept = model.dof_armature.copy(), model.dof_frictionloss.copy()
    model.dof_armature[:], model.dof_frictionloss[:] = armature, friction
    track_step(model, once)
    model.dof_armature[:], model.dof_frictionloss[:] = kept
    assert np.array_equal(state(model, once), state(model, data))


def state_without_ctrl(model, full):
    scratch = mujoco.MjData(model)
    mujoco.mj_setState(model, scratch, full, INTEGRATION)
    return state(model, scratch, BETWEEN_STEPS)


def test_runs_with_the_tracks_driving_and_the_arm_working_repeat_bit_for_bit():
    runs = [working_and_travelling(12, 10.0) for _ in range(2)]
    (model, first, first_layer), (_, second, second_layer) = runs
    assert np.array_equal(state(model, first), state(model, second))
    assert np.array_equal(first_layer.chamber_pressures(), second_layer.chamber_pressures())
    assert first_layer.redos > 0 and first_layer.redos == second_layer.redos


def test_scenes_sharing_one_model_stepped_in_turn_match_scenes_on_their_own():
    def drive(scenes, model, seeds):
        travel = [model.actuator(machine_prefix(0) + t).id for t in TRAVEL_NAMES]
        rngs = [np.random.default_rng(k) for k in seeds]
        for _ in range(40):
            for (data, layer), rng in zip(scenes, rngs, strict=True):
                layer.act(rng.uniform(-1.0, 1.0, (1, 4)))
                data.ctrl[travel] = rng.uniform(-1.0, 1.0, 2) * model.actuator_ctrlrange[travel, 1]
            for _ in range(20):
                for _, layer in scenes:
                    layer.step()
        return [data.qpos.copy() for data, _ in scenes]

    def scene(model):
        data = mujoco.MjData(model)
        return data, Hydraulics(model, data, [draw_machine(1)])

    shared = load_scene(1)
    together = drive([scene(shared) for _ in range(2)], shared, (0, 1))
    apart = []
    for k in range(2):
        model = load_scene(1)
        apart += drive([scene(model)], model, (k,))
    for a, b in zip(together, apart, strict=True):
        assert np.array_equal(a, b)


def free_machine(ground, pose, load=None):
    model = load_scene(1)
    apply_ground(model, 0, ground)
    if load is not None:
        apply_load(model, 0, load, mujoco.MjData(model))
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [NOMINAL])
    layer.reset(0, NOMINAL, pose)
    if load is None:
        place_load(model, data, 0, *LOAD_AWAY)
    else:
        hook = data.site_xpos[model.site(machine_prefix(0) + "hook").id]
        place_load(model, data, 0, hook[0], hook[1])
        hook_load(model, data, 0)
    return model, data, layer


def drive(model, data, speed):
    data.ctrl[[model.actuator(machine_prefix(0) + t).id for t in TRAVEL_NAMES]] = speed


def test_a_free_machine_carries_a_slung_block_straight_at_full_travel_speed():
    load = block(1000.0)
    model, data, layer = free_machine(Ground(), (0.0, -0.6, 1.6, 2.3), load)
    body = model.body(machine_prefix(0) + LOAD).id
    layer.act(np.array([[0.0, -1.0, 0.0, 0.0]]))
    for _ in range(round(4.0 / DT)):
        layer.step()
        if data.xipos[body][2] - load.half[2] > LIFT_CLEARANCE + CONTACT_MARGIN:
            break
    layer.act(np.zeros((1, 4)))
    run(layer, np.zeros(4), 1.0)
    drive(model, data, TRAVEL_SPEED)
    lowest = np.inf
    for k in range(round(12.0 / DT)):
        layer.step()
        lowest = min(lowest, data.xipos[body][2] - load.half[2])
        if k == round(6.0 / DT):
            start = data.qpos[0]
    assert (data.qpos[0] - start) / (6.0 - DT) == pytest.approx(TRAVEL_SPEED, rel=1e-3)
    assert lowest > CONTACT_MARGIN
    assert layer.capped == 0


def test_a_free_machine_climbs_ten_degrees_at_full_speed_with_the_grade_on_its_drives():
    grade = np.radians(10.0)
    model, data, layer = free_machine(slope(along=grade), TRAVEL_POSE)
    drive(model, data, TRAVEL_SPEED)
    weight = model.body_subtreemass[model.body(machine_prefix(0) + "base").id] * 9.81
    assert TRAVEL_POWER / (weight * np.sin(grade)) > TRAVEL_SPEED
    pulls, start = [], None
    for k in range(round(12.0 / DT)):
        if k % round(2.0 / DT) == 0:
            layer.act(np.array([[0.0, -0.6 if k // round(2.0 / DT) % 2 == 0 else 0.6, 0.0, 0.0]]))
        layer.step()
        if k == round(4.0 / DT):
            start = data.qpos[:3].copy()
        if start is not None:
            pulls.append(drive_forces(model, data)[0].sum())
    assert np.linalg.norm(data.qpos[:3] - start) / (8.0 - DT) == pytest.approx(TRAVEL_SPEED,
                                                                               rel=1e-3)
    assert np.mean(pulls) == pytest.approx(weight * np.sin(grade), rel=0.01)
    assert layer.capped == 0


def test_a_free_machine_stopped_from_full_speed_with_the_arm_out_slides_as_far_as_friction_allows():
    model, data, layer = free_machine(Ground(), (0.0, -0.3, 0.6, 1.0))
    drive(model, data, TRAVEL_SPEED)
    run(layer, np.zeros(4), 6.0)
    speed, start = data.qvel[0], data.qpos[0]
    drive(model, data, 0.0)
    run(layer, np.zeros(4), 4.0)
    coulomb = speed ** 2 / (2 * FRICTION * 9.81)
    assert coulomb <= data.qpos[0] - start <= coulomb + speed * VALVE_INERTIA / VALVE_GAIN
    assert abs(data.qvel[0]) < 1e-3
    assert layer.capped == 0
