from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import (
    NOISE,
    NOMINAL,
    RANGES,
    BaseController,
    Circle,
    Hydraulics,
    StraightLine,
    draw_machine,
)
from hma.machine import forward_kinematics, load_scene
from scenes import dig_pit

LINE = StraightLine((5.0, -1.5, 0.5), (6.5, 1.0, -0.5), 1.4, 4.0)
CIRCLE = Circle((5.5, 0.0, 0.0), 0.6, 1.4, 7.0)


def steepest_curve():
    low, high = RANGES["curve_weight"]
    weights = np.array([high, low, low, low])
    return np.tile(np.concatenate([[0.0], np.cumsum(weights) / weights.sum()]), (4, 1))


def worst_corner(deadband):
    top = {name: RANGES[name][1] for name in RANGES}
    return replace(NOMINAL, delay=np.full(4, top["delay"]), lag=np.full(4, top["lag"]),
                   sensor_latency=top["sensor_latency"], gain=np.full(4, top["gain"]),
                   speed_limit=NOMINAL.speed_limit * top["speed_limit"], curve=steepest_curve(),
                   deadband=np.full(4, deadband), backlash=np.full(4, top["backlash"]),
                   position_noise=NOISE["position"], velocity_noise=NOISE["velocity"],
                   load_noise=NOISE["load"], seed=4242)


def late_swing(machine, hold):
    model = dig_pit(load_scene(1, weld=True))
    data = mujoco.MjData(model)
    controller = BaseController(LINE, 10.0)
    layer = Hydraulics(model, data, [machine])
    layer.reset(0, machine, pose=controller.reference(0.0))
    tips = []
    for _ in range(round((LINE.duration + hold) * 10)):
        layer.act(controller(data.time, layer.observe().positions[0])[None])
        for _ in range(layer.steps_per_decision):
            layer.step()
        tips.append(forward_kinematics(layer.true_state().positions[0])[0])
    return np.ptp(np.array(tips[-20:]), axis=0).max()


def tip_errors(path, machine, control_hz, hold=0.5):
    model = dig_pit(load_scene(1, weld=True))
    data = mujoco.MjData(model)
    controller = BaseController(path, control_hz)
    layer = Hydraulics(model, data, [machine], control_hz=control_hz)
    layer.reset(0, machine, pose=controller.reference(0.0))
    errors = []
    for _ in range(round((path.duration + hold) * control_hz)):
        layer.act(controller(data.time, layer.observe().positions[0])[None])
        for _ in range(layer.steps_per_decision):
            layer.step()
            tip, _ = forward_kinematics(layer.true_state().positions[0])
            errors.append(np.linalg.norm(tip - path(data.time)[0]))
    return np.array(errors)


@pytest.mark.parametrize("control_hz", [10.0, 20.0])
def test_nominal_machine_tracks_a_straight_line_within_centimetres(control_hz):
    nominal = tip_errors(LINE, NOMINAL, control_hz)
    assert nominal.mean() < 0.01
    assert nominal.max() < 0.03
    delayed = tip_errors(LINE, replace(NOMINAL, delay=np.full(4, 0.5)), control_hz)
    assert delayed.mean() > 5 * nominal.mean()
    assert delayed.max() > 0.1


def test_base_controller_settles_on_the_slowest_machine_in_range():
    slowest = replace(NOMINAL, delay=np.full(4, RANGES["delay"][1]),
                      lag=np.full(4, RANGES["lag"][1]))
    errors = tip_errors(LINE, slowest, 10.0, hold=10.0)
    assert errors[-200:].max() < 0.1 * errors.max()


@pytest.mark.parametrize(("deadband", "hold"), [(0.0, 30.0), (RANGES["deadband"][1], 10.0)])
def test_base_controller_stops_swinging_on_the_worst_corner_of_every_range(deadband, hold):
    assert late_swing(worst_corner(deadband), hold) < 0.03


def test_base_controller_settles_on_drawn_machines_that_swing_at_a_higher_gain():
    for seed in (158, 174, 143, 247, 124):
        assert late_swing(draw_machine(seed), hold=10.0) < 0.02


def test_nominal_machine_tracks_a_circle_within_centimetres():
    errors = tip_errors(CIRCLE, NOMINAL, 10.0)
    assert errors.mean() < 0.01
    assert errors.max() < 0.03


def test_paths_start_and_end_at_rest_on_their_endpoints():
    np.testing.assert_allclose(LINE(0.0)[0], LINE.start)
    np.testing.assert_allclose(LINE(LINE.duration + 1.0)[0], LINE.end)
    np.testing.assert_allclose(CIRCLE(0.0)[0], CIRCLE(CIRCLE.duration)[0], atol=1e-12)
    radius = np.linalg.norm(CIRCLE(CIRCLE.duration / 3)[0] - CIRCLE.centre)
    assert radius == pytest.approx(CIRCLE.radius)
    for path in (LINE, CIRCLE):
        assert np.linalg.norm(path(1e-3)[0] - path(0.0)[0]) < 1e-6


def test_reference_follows_inverse_kinematics_and_refuses_unreachable_paths():
    controller = BaseController(LINE, 10.0)
    tip, pitch = forward_kinematics(controller.reference(2.0))
    np.testing.assert_allclose(tip, LINE(2.0)[0], atol=1e-9)
    assert pitch == pytest.approx(LINE.pitch)
    with pytest.raises(ValueError):
        BaseController(StraightLine((5, 0, 0), (20, 0, 0), 1.4, 2.0), 10.0).reference(2.0)


def test_command_is_feedforward_plus_proportional_error_in_valve_units():
    controller = BaseController(LINE, 10.0)
    end = LINE.duration + 1.0
    q = controller.reference(end)
    at_rest = controller(end, q)
    assert np.all(np.abs(at_rest) < 1e-12)
    pushed = controller(end, q - 0.01)
    np.testing.assert_allclose(pushed - at_rest, controller.gain * 0.01 / NOMINAL.speed_limit)
