import math

import pytest

from hma.soil import NOMINAL, SoilParams, sample_soil
from hma.soil.params import (
    COHESION,
    FRICTION_ANGLE,
    PENETRATION_FACTOR,
    STEEL_FRICTION_ANGLE,
    UNIT_WEIGHT,
)

RANGES = {
    "cohesion": COHESION,
    "friction_angle": FRICTION_ANGLE,
    "unit_weight": UNIT_WEIGHT,
    "steel_friction_angle": STEEL_FRICTION_ANGLE,
    "penetration_factor": PENETRATION_FACTOR,
}


def test_ranges_match_the_published_table_in_si_units():
    assert COHESION == (0.0, 105.0e3)
    assert FRICTION_ANGLE == (0.3, 0.8)
    assert UNIT_WEIGHT == (17.0e3, 22.0e3)
    assert STEEL_FRICTION_ANGLE == (0.2, 0.4)
    assert PENETRATION_FACTOR == (0.0, 300.0)


def test_nominal_soil_is_the_middle_of_every_range():
    for name, (lo, hi) in RANGES.items():
        assert getattr(NOMINAL, name) == pytest.approx((lo + hi) / 2)
    assert NOMINAL.adhesion == pytest.approx(NOMINAL.cohesion / 2)


def test_same_seed_gives_the_same_soil_and_another_seed_does_not():
    assert sample_soil(5) == sample_soil(5)
    assert sample_soil(5) != sample_soil(6)


def test_samples_stay_inside_the_ranges_with_adhesion_below_cohesion():
    for seed in range(200):
        soil = sample_soil(seed)
        for name, (lo, hi) in RANGES.items():
            assert lo <= getattr(soil, name) <= hi
        assert 0.0 <= soil.adhesion <= soil.cohesion


@pytest.mark.parametrize("change", [
    {"cohesion": 106.0e3},
    {"adhesion": 60.0e3, "cohesion": 50.0e3},
    {"friction_angle": 0.9},
    {"unit_weight": 16.0e3},
    {"steel_friction_angle": 0.1},
    {"penetration_factor": -1.0},
    {"friction_angle": math.nan},
])
def test_values_outside_the_ranges_are_refused(change):
    values = {**NOMINAL.__dict__, **change}
    with pytest.raises(ValueError):
        SoilParams(**values)
