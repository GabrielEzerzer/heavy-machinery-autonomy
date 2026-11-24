"""Soil parameters: the six that Egli et al. 2022 (Table I) randomise per episode, in SI units.
sample_soil(seed) draws them uniformly from the published ranges; NOMINAL is each range's middle."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

COHESION = (0.0, 105.0e3)
FRICTION_ANGLE = (0.3, 0.8)
UNIT_WEIGHT = (17.0e3, 22.0e3)
STEEL_FRICTION_ANGLE = (0.2, 0.4)
PENETRATION_FACTOR = (0.0, 300.0)


@dataclass(frozen=True)
class SoilParams:
    cohesion: float
    adhesion: float
    friction_angle: float
    unit_weight: float
    steel_friction_angle: float
    penetration_factor: float

    def __post_init__(self) -> None:
        for field in fields(self):
            if not np.isfinite(getattr(self, field.name)):
                raise ValueError(f"{field.name} must be finite")
        checks = (
            (COHESION, self.cohesion, "cohesion"),
            ((0.0, self.cohesion), self.adhesion, "adhesion"),
            (FRICTION_ANGLE, self.friction_angle, "friction_angle"),
            (UNIT_WEIGHT, self.unit_weight, "unit_weight"),
            (STEEL_FRICTION_ANGLE, self.steel_friction_angle, "steel_friction_angle"),
            (PENETRATION_FACTOR, self.penetration_factor, "penetration_factor"),
        )
        for (lo, hi), value, name in checks:
            if not lo <= value <= hi:
                raise ValueError(f"{name} must lie within [{lo}, {hi}], got {value}")


def _middle(bounds: tuple[float, float]) -> float:
    return (bounds[0] + bounds[1]) / 2


NOMINAL = SoilParams(
    cohesion=_middle(COHESION),
    adhesion=_middle(COHESION) / 2,
    friction_angle=_middle(FRICTION_ANGLE),
    unit_weight=_middle(UNIT_WEIGHT),
    steel_friction_angle=_middle(STEEL_FRICTION_ANGLE),
    penetration_factor=_middle(PENETRATION_FACTOR),
)


def sample_soil(seed: int | np.random.Generator) -> SoilParams:
    rng = np.random.default_rng(seed)
    cohesion = float(rng.uniform(*COHESION))
    return SoilParams(
        cohesion=cohesion,
        adhesion=float(rng.uniform(0.0, 1.0)) * cohesion,
        friction_angle=float(rng.uniform(*FRICTION_ANGLE)),
        unit_weight=float(rng.uniform(*UNIT_WEIGHT)),
        steel_friction_angle=float(rng.uniform(*STEEL_FRICTION_ANGLE)),
        penetration_factor=float(rng.uniform(*PENETRATION_FACTOR)),
    )
