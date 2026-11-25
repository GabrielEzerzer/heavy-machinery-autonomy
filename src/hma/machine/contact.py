"""Per-episode contact friction for the machine's steel against the ground, the load and itself."""

from __future__ import annotations

import mujoco
import numpy as np

from hma.machine.collision import PIECES
from hma.machine.model import machine_prefix

STEEL_FRICTION_RANGE = (0.30, 0.70)


def draw_friction(seed: int) -> float:
    return float(np.random.default_rng((seed, 2)).uniform(*STEEL_FRICTION_RANGE))


def steel_geoms(model: mujoco.MjModel, index: int) -> np.ndarray:
    prefix = machine_prefix(index)
    return np.array([model.geom(f"{prefix}{link}_{name}").id
                     for link, pieces in PIECES.items() for name in pieces])


def set_friction(model: mujoco.MjModel, index: int, friction: float) -> None:
    if not np.isfinite(friction) or friction < 0:
        raise ValueError("friction must be finite and not negative")
    model.geom_friction[steel_geoms(model, index), 0] = friction
