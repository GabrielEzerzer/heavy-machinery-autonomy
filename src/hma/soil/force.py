"""Resistance of soil to the bucket's cutting edge: the fundamental earthmoving equation
(separation) plus a tooth penetration term, smoothly capped at FORCE_CAP. Cohesion, adhesion
and penetration act only on the undisturbed part of the cut; loose spoil adds weight alone."""

from __future__ import annotations

import functools
import math

import numpy as np

from hma.soil.params import SoilParams

FORCE_CAP = 175.0e3
CAP_SHARPNESS = 4
EDGE_THICKNESS = 0.02
RAKE_MARGIN = 0.1
SINGULAR_MARGIN = 0.1
SURCHARGE_DEPTH = 0.05
TABLE_RAKES = 256
TABLE_BETAS = 2048


@functools.lru_cache(maxsize=64)
def _failure_angles(soil: SoilParams) -> tuple[float, float, list[float], float]:
    phi, delta = soil.friction_angle, soil.steel_friction_angle
    top = math.pi - phi - delta
    rakes = np.linspace(RAKE_MARGIN, top - SINGULAR_MARGIN, TABLE_RAKES)[:, None]
    betas = np.linspace(0.0, 1.0, TABLE_BETAS)[1:-1][None, :] * (top - rakes)
    n_gamma = ((1 / np.tan(rakes) + 1 / np.tan(betas)) * np.sin(phi + betas)
               / np.sin(delta + rakes + phi + betas))
    floor = math.pi / 4 - phi / 2
    best = np.maximum(betas[np.arange(TABLE_RAKES), np.argmin(n_gamma, axis=1)], floor).tolist()
    table = (RAKE_MARGIN, float(rakes[1, 0] - rakes[0, 0]), best)
    limit = top - SINGULAR_MARGIN
    for _ in range(100):
        limit = top - _lookup(table, limit) - SINGULAR_MARGIN
    return (*table, limit)


def _lookup(table: tuple[float, float, list[float]], rake: float) -> float:
    start, step, values = table
    at = min(max((rake - start) / step, 0.0), len(values) - 1.0)
    i = min(int(at), len(values) - 2)
    return values[i] + (at - i) * (values[i + 1] - values[i])


def failure_angle(soil: SoilParams, rake: float) -> float:
    return _lookup(_failure_angles(soil)[:3], rake)


def effective_rake(rake: float) -> float:
    return max(abs(math.atan2(math.sin(rake), math.cos(rake))), RAKE_MARGIN)


def rake_limit(soil: SoilParams) -> float:
    return _failure_angles(soil)[3]


def fee_factors(soil: SoilParams, rake: float) -> tuple[float, float, float, float]:
    rho = min(effective_rake(rake), rake_limit(soil))
    phi, delta, beta = soil.friction_angle, soil.steel_friction_angle, failure_angle(soil, rho)
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


def depth_sums(depths: list[float], spoil: list[float],
               width: float) -> tuple[float, float, float, float]:
    d = [max(x, 0.0) for x in depths]
    firm = [max(x - max(s, 0.0), 0.0) for x, s in zip(d, spoil, strict=True)]
    slice_width = width / len(d)
    return (sum(d) * slice_width, sum(x * x for x in d) * slice_width,
            sum(min(x / SURCHARGE_DEPTH, 1.0) for x in d) / len(d), sum(firm) * slice_width)


def raw_force(soil: SoilParams, sums: tuple[float, float, float, float], rake: float,
              attack_cos: float, load: float) -> float:
    area, moment, wetted, firm = sums
    n_gamma, n_cohesion, n_surcharge, n_adhesion = fee_factors(soil, rake)
    separation = (soil.unit_weight * moment * n_gamma
                  + (soil.cohesion * n_cohesion + soil.adhesion * n_adhesion) * firm
                  + load * n_surcharge * wetted)
    penetration = (soil.penetration_factor * passive_coefficient(soil) * soil.unit_weight
                   * firm * EDGE_THICKNESS * abs(attack_cos))
    return separation + penetration


def cutting_force(soil: SoilParams, depths: np.ndarray, width: float, rake: float,
                  attack_cos: float, load: float, spoil: float | np.ndarray = 0.0) -> float:
    depths = np.ravel(depths).astype(float)
    spoil = np.broadcast_to(np.asarray(spoil, float), depths.shape)
    sums = depth_sums(depths.tolist(), spoil.tolist(), width)
    return smooth_cap(raw_force(soil, sums, rake, attack_cos, load))
