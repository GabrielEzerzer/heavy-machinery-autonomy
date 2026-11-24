"""Per-machine hydraulic parameters: the ranges they are drawn from, the nominal machine, and one
seeded draw per machine. Arrays follow JOINT_NAMES (four joints) or BODY_NAMES (five links)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from hma.machine import BODY_NAMES, FORCE_CAPS, JOINT_NAMES

NOMINAL_SPEEDS = {"swing": 0.8, "boom": 0.3, "arm": 0.6, "bucket": 0.8}
CURVE_SEGMENTS = 4
NOMINAL_SUPPLY = 2.5

RANGES = {
    "delay": (0.05, 0.6),
    "lag": (0.1, 0.4),
    "deadband": (0.0, 0.25),
    "curve_weight": (0.5, 1.5),
    "gain": (0.9, 1.1),
    "flow": (0.6, 1.0),
    "speed_limit": (0.8, 1.2),
    "force_limit": (0.85, 1.15),
    "backlash": (0.0, math.radians(0.5)),
    "link_mass": (0.9, 1.1),
    "com_shift": (0.0, 0.05),
    "counterweight": (0.9, 1.1),
    "sensor_latency": (0.0, 0.2),
}
NOISE = {"position": 0.02, "velocity": 0.05, "load": 0.05}

EFFECTS = ("delay", "valve", "flow", "limits", "backlash", "mass", "sensors")
GROUPS = {
    "delay": ("delay", "lag"),
    "valve": ("deadband", "curve", "gain"),
    "flow": ("flow",),
    "limits": ("speed_limit", "force_limit"),
    "backlash": ("backlash",),
    "mass": ("link_mass", "com_shift", "counterweight"),
    "sensors": ("sensor_latency", "position_noise", "velocity_noise", "load_noise"),
}


@dataclass(frozen=True, eq=False)
class Machine:
    seed: int
    delay: np.ndarray
    lag: np.ndarray
    deadband: np.ndarray
    curve: np.ndarray
    gain: np.ndarray
    flow: float
    speed_limit: np.ndarray
    force_limit: np.ndarray
    backlash: np.ndarray
    link_mass: np.ndarray
    com_shift: np.ndarray
    counterweight: float
    sensor_latency: float
    position_noise: float
    velocity_noise: float
    load_noise: float

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                frozen = np.array(value, float)
                frozen.flags.writeable = False
                object.__setattr__(self, name, frozen)


def _per_joint(values: dict[str, float]) -> np.ndarray:
    return np.array([values[j] for j in JOINT_NAMES])


NOMINAL = Machine(
    seed=0,
    delay=np.full(len(JOINT_NAMES), RANGES["delay"][0]),
    lag=np.full(len(JOINT_NAMES), RANGES["lag"][0]),
    deadband=np.zeros(len(JOINT_NAMES)),
    curve=np.tile(np.linspace(0.0, 1.0, CURVE_SEGMENTS + 1), (len(JOINT_NAMES), 1)),
    gain=np.ones(len(JOINT_NAMES)),
    flow=1.0,
    speed_limit=_per_joint(NOMINAL_SPEEDS),
    force_limit=_per_joint(FORCE_CAPS),
    backlash=np.zeros(len(JOINT_NAMES)),
    link_mass=np.ones(len(BODY_NAMES)),
    com_shift=np.zeros((len(BODY_NAMES), 3)),
    counterweight=1.0,
    sensor_latency=0.0,
    position_noise=0.0,
    velocity_noise=0.0,
    load_noise=0.0,
)


def _curve(rng: np.random.Generator) -> np.ndarray:
    weights = rng.uniform(*RANGES["curve_weight"], size=(len(JOINT_NAMES), CURVE_SEGMENTS))
    rise = np.cumsum(weights, axis=1) / weights.sum(axis=1, keepdims=True)
    return np.concatenate([np.zeros((len(JOINT_NAMES), 1)), rise], axis=1)


def _com_shift(rng: np.random.Generator) -> np.ndarray:
    direction = rng.standard_normal((len(BODY_NAMES), 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    return direction * rng.uniform(*RANGES["com_shift"], size=(len(BODY_NAMES), 1))


def draw_machine(seed: int, effects: tuple[str, ...] = EFFECTS) -> Machine:
    unknown = set(effects) - set(EFFECTS)
    if unknown:
        raise ValueError(f"unknown effects {sorted(unknown)}; choose from {EFFECTS}")
    if seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.default_rng((seed, 0))
    joints = len(JOINT_NAMES)
    drawn = {
        "delay": rng.uniform(*RANGES["delay"], size=joints),
        "lag": rng.uniform(*RANGES["lag"], size=joints),
        "deadband": rng.uniform(*RANGES["deadband"], size=joints),
        "curve": _curve(rng),
        "gain": rng.uniform(*RANGES["gain"], size=joints),
        "flow": float(rng.uniform(*RANGES["flow"])),
        "speed_limit": NOMINAL.speed_limit * rng.uniform(*RANGES["speed_limit"], size=joints),
        "force_limit": NOMINAL.force_limit * rng.uniform(*RANGES["force_limit"], size=joints),
        "backlash": rng.uniform(*RANGES["backlash"], size=joints),
        "link_mass": rng.uniform(*RANGES["link_mass"], size=len(BODY_NAMES)),
        "com_shift": _com_shift(rng),
        "counterweight": float(rng.uniform(*RANGES["counterweight"])),
        "sensor_latency": float(rng.uniform(*RANGES["sensor_latency"])),
        "position_noise": NOISE["position"],
        "velocity_noise": NOISE["velocity"],
        "load_noise": NOISE["load"],
    }
    kept = {name: drawn[name] for group in effects for name in GROUPS[group]}
    return replace(NOMINAL, seed=seed, **kept)
