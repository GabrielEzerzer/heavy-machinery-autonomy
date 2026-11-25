import math

import numpy as np

from hma.soil import FORCE_CAP, NOMINAL, SoilParams, cutting_force, sample_soil
from hma.soil.force import BACK_GAIN, fee_factors, rake_limit

WIDTH = 0.61
STIFF = SoilParams(105.0e3, 105.0e3, 0.8, 22.0e3, 0.4, 300.0)
LOOSE = SoilParams(0.0, 0.0, 0.3, 17.0e3, 0.2, 0.0)
SOILS = [NOMINAL, STIFF, LOOSE, *(sample_soil(seed) for seed in range(10))]
LOAD = 22.0e3 * 0.45


def force(soil, depth, rake, attack=1.0, load=LOAD):
    return cutting_force(soil, [depth] * 7, WIDTH, rake, attack, load)


def test_no_depth_gives_no_force():
    for soil in SOILS:
        for rake in np.linspace(-4.0, 4.0, 41):
            assert force(soil, 0.0, rake) == 0.0


def test_the_force_stays_within_the_force_cap():
    for soil in SOILS:
        for depth in (0.1, 0.5, 1.5, 3.0):
            values = [force(soil, depth, rake) for rake in np.linspace(-4.0, 4.0, 401)]
            assert max(values) <= FORCE_CAP


def test_no_spike_sweeping_the_rake_through_the_singular_region():
    rakes = np.linspace(0.05, math.pi, 20001)
    for soil in (NOMINAL, STIFF, LOOSE):
        values = np.array([force(soil, 0.3, rake) for rake in rakes])
        assert np.abs(np.diff(values)).max() < 600.0
        at_limit = force(soil, 0.3, rake_limit(soil))
        beyond = values[rakes > rake_limit(soil)]
        assert beyond.max() <= BACK_GAIN * at_limit * (1 + 1e-9)
        assert beyond.min() >= at_limit * (1 - 1e-9)


def test_force_is_continuous_in_depth_and_vanishes_with_it():
    depths = np.linspace(0.0, 1.5, 3001)
    for soil in (NOMINAL, STIFF, LOOSE):
        values = np.array([force(soil, depth, 0.8) for depth in depths])
        assert values[0] == 0.0
        assert np.all(np.diff(values) >= 0.0)
        assert np.abs(np.diff(values)).max() < 600.0
        assert force(soil, 1e-5, 0.8) < 100.0


def test_penetration_follows_the_attack_cosine_in_either_sense():
    edge_on, broadside = force(STIFF, 0.2, 0.8, 1.0), force(STIFF, 0.2, 0.8, 0.0)
    assert edge_on > broadside
    assert force(STIFF, 0.2, 0.8, -1.0) == edge_on


def test_fee_factors_stay_positive_for_every_rake():
    for soil in SOILS:
        for rake in np.linspace(-4.0, 4.0, 801):
            assert min(fee_factors(soil, rake)) >= 0.0


def test_spoil_costs_its_weight_without_cohesion_and_blends_continuously():
    depth, rake = 0.3, 1.0
    firm = force(STIFF, depth, rake, load=0.0)
    spoil = np.linspace(0.0, 0.5, 5001)
    values = np.array([cutting_force(STIFF, [depth] * 7, WIDTH, rake, 1.0, 0.0, s)
                       for s in spoil])
    assert values[0] == firm
    assert values[-1] < 0.1 * firm
    assert np.all(np.diff(values) <= 0.0)
    assert np.abs(np.diff(values)).max() < 1e-3 * firm
    assert values[-1] == cutting_force(STIFF, [depth] * 7, WIDTH, rake, 1.0, 0.0, depth)
    weight_only = SoilParams(0.0, 0.0, STIFF.friction_angle, STIFF.unit_weight,
                             STIFF.steel_friction_angle, 0.0)
    assert values[-1] == force(weight_only, depth, rake, load=0.0)
