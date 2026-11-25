"""Resistance of soil to the bucket's cutting edge: the fundamental earthmoving equation
(separation) plus a tooth penetration term, smoothly capped at FORCE_CAP.
Within SINGULAR_MARGIN of the singular rake, separation ramps up to BACK_GAIN times."""

from __future__ import annotations

import math

import numpy as np

from hma.soil.params import SoilParams

FORCE_CAP = 175.0e3
CAP_SHARPNESS = 4
EDGE_THICKNESS = 0.02
RAKE_MARGIN = 0.1
SINGULAR_MARGIN = 0.1
BACK_RAMP = 0.2
BACK_GAIN = 2.0
SURCHARGE_DEPTH = 0.05


def failure_angle(soil: SoilParams) -> float:
    return math.pi / 4 - soil.friction_angle / 2


def effective_rake(rake: float) -> float:
    return max(abs(math.atan2(math.sin(rake), math.cos(rake))), RAKE_MARGIN)


def rake_limit(soil: SoilParams) -> float:
    return (math.pi - soil.friction_angle - soil.steel_friction_angle - failure_angle(soil)
            - SINGULAR_MARGIN)


def fee_factors(soil: SoilParams, rake: float) -> tuple[float, float, float, float]:
    rho = min(effective_rake(rake), rake_limit(soil))
    phi, delta, beta = soil.friction_angle, soil.steel_friction_angle, failure_angle(soil)
    denominator = math.sin(delta + rho + phi + beta)
    n_gamma = (1 / math.tan(rho) + 1 / math.tan(beta)) * math.sin(phi + beta) / (2 * denominator)
    n_cohesion = math.cos(phi) / (math.sin(beta) * denominator)
    n_surcharge = math.sin(phi + beta) / denominator
    n_adhesion = max(0.0, -math.cos(rho + phi + beta) / (math.sin(rho) * denominator))
    return n_gamma, n_cohesion, n_surcharge, n_adhesion


def passive_coefficient(soil: SoilParams) -> float:
    return math.tan(math.pi / 4 + soil.friction_angle / 2) ** 2


def smooth_cap(force: float) -> float:
    force = max(force, 0.0)
    return force / (1.0 + (force / FORCE_CAP) ** CAP_SHARPNESS) ** (1.0 / CAP_SHARPNESS)


def depth_sums(depths: list[float], width: float) -> tuple[float, float, float]:
    d = [max(x, 0.0) for x in depths]
    slice_width = width / len(d)
    return (sum(d) * slice_width, sum(x * x for x in d) * slice_width,
            sum(min(x / SURCHARGE_DEPTH, 1.0) for x in d) / len(d))


def raw_force(soil: SoilParams, sums: tuple[float, float, float], rake: float,
              attack_cos: float, load: float) -> float:
    area, moment, wetted = sums
    n_gamma, n_cohesion, n_surcharge, n_adhesion = fee_factors(soil, rake)
    separation = (soil.unit_weight * moment * n_gamma
                  + (soil.cohesion * n_cohesion + soil.adhesion * n_adhesion) * area
                  + load * n_surcharge * wetted)
    past = min(max((effective_rake(rake) - rake_limit(soil)) / BACK_RAMP, 0.0), 1.0)
    separation *= 1.0 + (BACK_GAIN - 1.0) * past * past * (3.0 - 2.0 * past)
    penetration = (soil.penetration_factor * passive_coefficient(soil) * soil.unit_weight
                   * area * EDGE_THICKNESS * abs(attack_cos))
    return separation + penetration


def cutting_force(soil: SoilParams, depths: np.ndarray, width: float, rake: float,
                  attack_cos: float, load: float) -> float:
    sums = depth_sums([float(x) for x in np.ravel(depths)], width)
    return smooth_cap(raw_force(soil, sums, rake, attack_cos, load))
