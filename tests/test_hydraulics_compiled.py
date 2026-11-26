from dataclasses import replace

import mujoco
import numpy as np
import pytest

import numpy_step
from hma.hydraulics import NOMINAL, RANGES, Hydraulics, draw_machine
from hma.machine import BODY_NAMES, TRAVEL_NAMES, load_scene, machine_prefix, tracks
from hma.machine.cylinders import SEAL_PRESSURE, SEAL_SPEED, seal_friction
from scenes import dig_pit

DECISIONS = 60
WELDED_POSITION = 1e-12
WELDED_PRESSURE = 1e-4
WELDED_ENERGY = 1e-12
DRIVEN_POSITION = 1e-7
DRIVEN_SETTLED_PRESSURE = 5.0
DRIVEN_LATE_PRESSURE = 0.1
DRIVEN_END_ENERGY = 1e-7


def extreme_machine(link_mass):
    volume = np.full((4, 2), RANGES["dead_volume"][0])
    volume[0] = RANGES["swing_volume"][0]
    return replace(NOMINAL, bulk_modulus=RANGES["bulk_modulus"][1], dead_volume=volume,
                   pump_volume=RANGES["pump_volume"][0],
                   link_mass=np.full(len(BODY_NAMES), link_mass),
                   spool_flow=np.full(4, RANGES["spool_flow"][1]),
                   backlash=np.full(4, RANGES["backlash"][1]))


def pair(welded):
    if welded:
        machines = [extreme_machine(m) for m in RANGES["link_mass"]] + [NOMINAL]
        model = dig_pit(load_scene(len(machines), weld=True), len(machines))
    else:
        machines = [draw_machine(seed) for seed in (3, 7, 12)]
        model = load_scene(len(machines))
    return (Hydraulics(model, mujoco.MjData(model), machines),
            numpy_step.NumpyHydraulics(model, mujoco.MjData(model), machines))


def energies(layer):
    return np.concatenate([layer.pump_energy, layer.lost_energy, layer.work_energy])


def run_side_by_side(welded):
    new, old = pair(welded)
    travel = np.array([[new.model.actuator(machine_prefix(i) + name).id for name in TRAVEL_NAMES]
                       for i in range(new.n)])
    rng = np.random.default_rng(5)
    gaps = []
    for _ in range(DECISIONS):
        if welded:
            command = rng.choice([-1.0, 0.0, 1.0], size=(new.n, 4))
        else:
            command = rng.uniform(-1.0, 1.0, (new.n, 4))
        drive = rng.uniform(-1.0, 1.0, travel.shape) * new.model.actuator_ctrlrange[travel, 1]
        for layer in (new, old):
            layer.act(command)
            if not welded:
                layer.data.ctrl[travel] = drive
        if not welded:
            limited, reference = mujoco.MjData(new.model), mujoco.MjData(new.model)
            for copy in (limited, reference):
                mujoco.mj_copyData(copy, new.model, new.data)
            tracks.limit_power(new.model, limited)
            numpy_step._limit_power(new.model, reference)
            assert np.array_equal(limited.qfrc_applied, reference.qfrc_applied)
            assert np.array_equal(limited.userdata, reference.userdata)
        for _ in range(new.steps_per_decision):
            new.step()
            old.step()
            assert (new.redos, new.capped, new.check_bound) == (old.redos, old.capped,
                                                                 old.check_bound)
            gaps.append((np.abs(new.data.qpos - old.data.qpos).max(),
                         np.abs(new._pressure - old._pressure).max(),
                         np.abs(energies(new) - energies(old)).max() / np.abs(energies(old)).max()))
    return new, np.array(gaps)


def test_the_compiled_step_tracks_the_numpy_step_on_welded_extreme_machines():
    layer, gaps = run_side_by_side(welded=True)
    assert layer.redos > 0
    assert gaps[:, 0].max() <= WELDED_POSITION
    assert gaps[:, 1].max() <= WELDED_PRESSURE
    assert gaps[:, 2].max() <= WELDED_ENERGY


def test_the_compiled_step_tracks_the_numpy_step_on_drawn_machines_driving_over_ground():
    layer, gaps = run_side_by_side(welded=False)
    assert layer.redos > 0 and layer.capped > 0 and layer.check_bound > 0
    assert gaps[:, 0].max() <= DRIVEN_POSITION
    assert gaps[20:, 1].max() <= DRIVEN_SETTLED_PRESSURE
    assert gaps[len(gaps) // 2:, 1].max() <= DRIVEN_LATE_PRESSURE
    assert gaps[-1, 2] <= DRIVEN_END_ENERGY


@pytest.mark.parametrize("shape", [(200000, 3), (3,), (5, 3), (2, 4, 3)])
def test_compiled_seal_friction_matches_the_numpy_tables_bit_for_bit(shape):
    rng = np.random.default_rng(0)
    pressures = np.r_[SEAL_PRESSURE, 0.0, -1.0, 40e6, np.nextafter(SEAL_PRESSURE, 0)]
    speeds = np.r_[SEAL_SPEED, -SEAL_SPEED, 0.0, -0.0, 1.0, np.nextafter(SEAL_SPEED, 0)]

    def draw(edges, low, high):
        return np.where(rng.random(shape) < 0.1, rng.choice(edges, shape),
                        rng.uniform(low, high, shape))

    head, rod = draw(pressures, -1e6, 45e6), draw(pressures, -1e6, 45e6)
    speed = draw(speeds, -0.6, 0.6)
    assert np.array_equal(seal_friction(head, rod, speed),
                          numpy_step.seal_friction(head, rod, speed))


def test_compiled_valve_openings_match_the_numpy_openings_bit_for_bit():
    machines = [draw_machine(seed) for seed in (1, 2, 3)]
    model = load_scene(len(machines))
    new = Hydraulics(model, mujoco.MjData(model), machines)
    old = numpy_step.NumpyHydraulics(model, mujoco.MjData(model), machines)
    rng = np.random.default_rng(0)
    for step in range(3000):
        if step % 7 == 0:
            command = rng.uniform(-1.3, 1.3, (3, 4))
            new.act(command)
            old.act(command)
        assert np.array_equal(new._openings(), old._openings())
        assert np.array_equal(new._spool, old._spool)
        assert np.array_equal(new._piloted, old._piloted)


def test_the_step_reads_nothing_an_earlier_step_left_in_its_workspace():
    machines = [draw_machine(seed) for seed in (3, 7, 12)]
    model = load_scene(len(machines))
    clean = Hydraulics(model, mujoco.MjData(model), machines)
    scrambled = Hydraulics(model, mujoco.MjData(model), machines)
    travel = np.array([[model.actuator(machine_prefix(i) + name).id for name in TRAVEL_NAMES]
                       for i in range(clean.n)])
    rng = np.random.default_rng(5)
    for _ in range(20):
        command = rng.uniform(-1.0, 1.0, (clean.n, 4))
        drive = rng.uniform(-1.0, 1.0, travel.shape) * model.actuator_ctrlrange[travel, 1]
        for layer in (clean, scrambled):
            layer.act(command)
            layer.data.ctrl[travel] = drive
        for _ in range(clean.steps_per_decision):
            for array in scrambled._work:
                if array.dtype == bool:
                    np.logical_not(array, out=array)
                elif array.dtype.kind == "f":
                    array.fill(np.nan)
                else:
                    array.fill(7)
            clean.step()
            scrambled.step()
            assert np.array_equal(clean.data.qpos, scrambled.data.qpos)
            assert np.array_equal(clean._pressure, scrambled._pressure)
            assert np.array_equal(clean.data.userdata, scrambled.data.userdata)
    assert clean.redos > 0 and clean.redos == scrambled.redos
