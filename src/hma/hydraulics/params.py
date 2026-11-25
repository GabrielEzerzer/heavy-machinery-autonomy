"""Per-machine hydraulic parameters: the ranges they are drawn from, the nominal machine, and one
seeded draw per machine. Arrays follow JOINT_NAMES (four functions) or BODY_NAMES (five links);
chamber arrays add a last axis of (head or port A, rod or port B)."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from hma.hydraulics.circuit import ENGINE_POWER, SWING_RELIEF, SWING_TORQUE
from hma.machine import BODY_NAMES, JOINT_NAMES
from hma.machine.cylinders import CYLINDER_JOINTS, MAIN_RELIEF, PORT_RELIEF

CURVE_SEGMENTS = 4
LITRE = 1e-3
CM3_PER_MIN_AT_10MPA = 1e-6 / 60.0 / 10.0e6
RELIEF_TOLERANCE = (-0.5e6, 2.0e6)
PORT_RELIEF_TOLERANCE = (0.0, 1.0e6)
NOMINAL_PORT_RELIEF = np.array([(SWING_RELIEF, SWING_RELIEF),
                                *(PORT_RELIEF[j] for j in CYLINDER_JOINTS)])
HOLDING_PORTS = np.array([[False, False], [True, False], [False, True], [False, False]])

RANGES = {
    "delay": (0.05, 0.6),
    "lag": (0.1, 0.4),
    "deadband": (0.0, 0.25),
    "curve_weight": (0.5, 1.5),
    "gain": (0.9, 1.1),
    "spool_flow": (0.8, 1.2),
    "pump_flow": (0.9, 1.0),
    "power": (0.85, 1.0),
    "bulk_modulus": (0.7e9, 1.5e9),
    "dead_volume": (0.5 * LITRE, 1.5 * LITRE),
    "swing_volume": (0.3 * LITRE, 0.8 * LITRE),
    "pump_volume": (0.5 * LITRE, 1.0 * LITRE),
    "spool_leakage": (3.0 * CM3_PER_MIN_AT_10MPA, 13.0 * CM3_PER_MIN_AT_10MPA),
    "holding_leakage": (0.02 * CM3_PER_MIN_AT_10MPA, 0.36 * CM3_PER_MIN_AT_10MPA),
    "motor_leakage": (0.5 * LITRE / 60.0 / SWING_RELIEF, 1.6 * LITRE / 60.0 / SWING_RELIEF),
    "brake": (1.0 * SWING_TORQUE, 1.5 * SWING_TORQUE),
    "backlash": (0.0, math.radians(0.5)),
    "link_mass": (0.9, 1.1),
    "com_shift": (0.0, 0.05),
    "counterweight": (0.9, 1.1),
    "sensor_latency": (0.0, 0.2),
    "pressure_offset": (-0.25e6, 0.25e6),
}
NOISE = {"position": 0.02, "velocity": 0.05, "pressure": 0.1e6}

EFFECTS = ("delay", "valve", "flow", "reliefs", "oil", "holding", "backlash", "mass", "sensors")
GROUPS = {
    "delay": ("delay", "lag"),
    "valve": ("deadband", "curve", "gain", "spool_flow"),
    "flow": ("pump_flow", "power"),
    "reliefs": ("main_relief", "port_relief"),
    "oil": ("bulk_modulus", "dead_volume", "pump_volume"),
    "holding": ("leakage", "brake"),
    "backlash": ("backlash",),
    "mass": ("link_mass", "com_shift", "counterweight"),
    "sensors": ("sensor_latency", "position_noise", "velocity_noise", "pressure_noise",
                "pressure_offset"),
}


@dataclass(frozen=True, eq=False)
class Machine:
    seed: int
    delay: np.ndarray
    lag: np.ndarray
    deadband: np.ndarray
    curve: np.ndarray
    gain: np.ndarray
    spool_flow: np.ndarray
    pump_flow: float
    power: float
    main_relief: float
    port_relief: np.ndarray
    bulk_modulus: float
    dead_volume: np.ndarray
    pump_volume: float
    leakage: np.ndarray
    brake: float
    backlash: np.ndarray
    link_mass: np.ndarray
    com_shift: np.ndarray
    counterweight: float
    sensor_latency: float
    position_noise: float
    velocity_noise: float
    pressure_noise: float
    pressure_offset: np.ndarray

    def __post_init__(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, np.ndarray):
                frozen = np.array(value, float)
                frozen.flags.writeable = False
                object.__setattr__(self, name, frozen)


def _middle(name: str) -> float:
    lo, hi = RANGES[name]
    return (lo + hi) / 2.0


def _leakage(spool, holding, motor) -> np.ndarray:
    leak = np.where(HOLDING_PORTS, holding, spool)
    leak[0] = motor
    return leak


def _volumes(dead, swing) -> np.ndarray:
    volume = np.array(dead, float) * np.ones((len(JOINT_NAMES), 2))
    volume[0] = swing
    return volume


NOMINAL = Machine(
    seed=0,
    delay=np.full(len(JOINT_NAMES), RANGES["delay"][0]),
    lag=np.full(len(JOINT_NAMES), RANGES["lag"][0]),
    deadband=np.zeros(len(JOINT_NAMES)),
    curve=np.tile(np.linspace(0.0, 1.0, CURVE_SEGMENTS + 1), (len(JOINT_NAMES), 1)),
    gain=np.ones(len(JOINT_NAMES)),
    spool_flow=np.ones(len(JOINT_NAMES)),
    pump_flow=1.0,
    power=ENGINE_POWER,
    main_relief=MAIN_RELIEF,
    port_relief=NOMINAL_PORT_RELIEF,
    bulk_modulus=_middle("bulk_modulus"),
    dead_volume=_volumes(_middle("dead_volume"), _middle("swing_volume")),
    pump_volume=_middle("pump_volume"),
    leakage=_leakage(_middle("spool_leakage"), _middle("holding_leakage"),
                     _middle("motor_leakage")),
    brake=_middle("brake"),
    backlash=np.zeros(len(JOINT_NAMES)),
    link_mass=np.ones(len(BODY_NAMES)),
    com_shift=np.zeros((len(BODY_NAMES), 3)),
    counterweight=1.0,
    sensor_latency=0.0,
    position_noise=0.0,
    velocity_noise=0.0,
    pressure_noise=0.0,
    pressure_offset=np.zeros((len(JOINT_NAMES), 2)),
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
    joints, ports = len(JOINT_NAMES), (len(JOINT_NAMES), 2)
    drawn = {
        "delay": rng.uniform(*RANGES["delay"], size=joints),
        "lag": rng.uniform(*RANGES["lag"], size=joints),
        "deadband": rng.uniform(*RANGES["deadband"], size=joints),
        "curve": _curve(rng),
        "gain": rng.uniform(*RANGES["gain"], size=joints),
        "spool_flow": rng.uniform(*RANGES["spool_flow"], size=joints),
        "pump_flow": float(rng.uniform(*RANGES["pump_flow"])),
        "power": ENGINE_POWER * float(rng.uniform(*RANGES["power"])),
        "main_relief": MAIN_RELIEF + float(rng.uniform(*RELIEF_TOLERANCE)),
        "port_relief": NOMINAL_PORT_RELIEF + np.r_[
            np.full((1, 2), rng.uniform(*RELIEF_TOLERANCE)),
            rng.uniform(*PORT_RELIEF_TOLERANCE, size=(joints - 1, 2))],
        "bulk_modulus": float(rng.uniform(*RANGES["bulk_modulus"])),
        "dead_volume": _volumes(rng.uniform(*RANGES["dead_volume"], size=ports),
                                rng.uniform(*RANGES["swing_volume"], size=2)),
        "pump_volume": float(rng.uniform(*RANGES["pump_volume"])),
        "leakage": _leakage(rng.uniform(*RANGES["spool_leakage"], size=ports),
                            rng.uniform(*RANGES["holding_leakage"], size=ports),
                            rng.uniform(*RANGES["motor_leakage"], size=2)),
        "brake": float(rng.uniform(*RANGES["brake"])),
        "backlash": rng.uniform(*RANGES["backlash"], size=joints),
        "link_mass": rng.uniform(*RANGES["link_mass"], size=len(BODY_NAMES)),
        "com_shift": _com_shift(rng),
        "counterweight": float(rng.uniform(*RANGES["counterweight"])),
        "sensor_latency": float(rng.uniform(*RANGES["sensor_latency"])),
        "position_noise": NOISE["position"],
        "velocity_noise": NOISE["velocity"],
        "pressure_noise": NOISE["pressure"],
        "pressure_offset": rng.uniform(*RANGES["pressure_offset"], size=ports),
    }
    kept = {name: drawn[name] for group in effects for name in GROUPS[group]}
    return replace(NOMINAL, seed=seed, **kept)
