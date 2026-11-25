"""Rough rigid ground drawn from a seed: a tilted plane with steps, kerbs, ruts and bumps on it,
and a track friction pair. Straight features run along the grid axes with their top edge on a grid
line, so the height field holds the edge exactly and its one-cell bevel falls on the low side."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from hma.machine.model import (
    GROUND_CELL,
    GROUND_DEPTH,
    GROUND_TOP,
    ground_axes,
    write_friction,
    write_ground,
)

RANGES = {
    "slope": (0.0, math.radians(20.0)),
    "steps": (0, 2),
    "step_height": (0.05, 0.40),
    "kerbs": (0, 2),
    "kerb_height": (0.10, 0.20),
    "kerb_width": (0.2, 0.4),
    "ruts": (0, 2),
    "rut_depth": (0.05, 0.15),
    "rut_width": (0.3, 0.6),
    "bumps": (0, 6),
    "bump_height": (-0.15, 0.15),
    "bump_radius": (0.3, 1.5),
    "feature_reach": (-4.0, 4.0),
    "friction": (0.30, 0.55),
}
ACROSS_RATIO = 0.50 / 0.45
CLEARANCE = 0.01


@dataclass(frozen=True, eq=False)
class Ground:
    seed: int
    heights: np.ndarray
    along: float
    across: float


def _edge(axis: np.ndarray, at: float) -> float:
    return float(axis[np.argmin(np.abs(axis - at))])


def _band(rng: np.random.Generator, axis: np.ndarray, width: tuple[float, float]):
    start = _edge(axis, rng.uniform(*RANGES["feature_reach"]))
    cells = max(1, round(rng.uniform(*width) / GROUND_CELL))
    return start, start + cells * GROUND_CELL


def draw_ground(seed: int) -> Ground:
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng((seed, 2))
    xs, ys = ground_axes()
    x, y = np.meshgrid(xs, ys)
    slope = rng.uniform(*RANGES["slope"])
    direction = rng.uniform(0.0, 2 * math.pi)
    heights = math.tan(slope) * (math.cos(direction) * x + math.sin(direction) * y)
    eps = 1e-9

    for _ in range(rng.integers(RANGES["steps"][0], RANGES["steps"][1] + 1)):
        along_x = rng.random() < 0.5
        coord, axis = (x, xs) if along_x else (y, ys)
        edge = _edge(axis, rng.uniform(*RANGES["feature_reach"]))
        rise = rng.uniform(*RANGES["step_height"]) * rng.choice([-1.0, 1.0])
        high_beyond = rng.random() < 0.5
        high = coord >= edge - eps if high_beyond else coord <= edge + eps
        heights = heights + np.where(high, max(rise, 0.0), min(rise, 0.0))

    for kind, size in (("kerbs", "kerb"), ("ruts", "rut")):
        for _ in range(rng.integers(RANGES[kind][0], RANGES[kind][1] + 1)):
            along_x = rng.random() < 0.5
            coord, axis = (x, xs) if along_x else (y, ys)
            start, end = _band(rng, axis, RANGES[f"{size}_width"])
            if kind == "kerbs":
                inside = (coord >= start - eps) & (coord <= end + eps)
                heights = heights + np.where(inside, rng.uniform(*RANGES["kerb_height"]), 0.0)
            else:
                inside = (coord > start + eps) & (coord < end - eps)
                heights = heights - np.where(inside, rng.uniform(*RANGES["rut_depth"]), 0.0)

    for _ in range(rng.integers(RANGES["bumps"][0], RANGES["bumps"][1] + 1)):
        cx, cy = rng.uniform(*RANGES["feature_reach"], size=2)
        radius = rng.uniform(*RANGES["bump_radius"])
        height = rng.uniform(*RANGES["bump_height"])
        heights = heights + height * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * radius**2))

    heights = heights - max(0.0, heights.max() - (GROUND_TOP - CLEARANCE))
    if heights.min() < -GROUND_DEPTH:
        raise ValueError("drawn ground is deeper than the height field holds")
    along = float(rng.uniform(*RANGES["friction"]))
    heights.flags.writeable = False
    return Ground(seed, heights, along, along * ACROSS_RATIO)


def apply_ground(model: mujoco.MjModel, index: int, ground: Ground) -> None:
    write_ground(model, index, ground.heights)
    write_friction(model, index, ground.along, ground.across)
