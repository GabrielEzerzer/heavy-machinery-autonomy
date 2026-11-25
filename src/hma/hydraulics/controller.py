"""The base controller: proportional on joint angle error, with reference speed feedforward and no
integral term, in valve command units. It follows a bucket-tip path of position and pitch over
time; StraightLine and Circle are free-air paths that start and end at rest."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from hma.hydraulics.params import NOMINAL, NOMINAL_SPEEDS
from hma.machine import JOINT_NAMES, inverse_kinematics
from hma.machine.model import VALVE_GAIN, VALVE_INERTIA

CONTROLLER_GAIN = 0.25
CONTROLLER_LEAD = float(NOMINAL.delay[0] + NOMINAL.lag[0]) + VALVE_INERTIA / VALVE_GAIN


def _ease(t: float, duration: float) -> float:
    s = min(max(t / duration, 0.0), 1.0)
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


@dataclass(frozen=True)
class StraightLine:
    start: tuple[float, float, float]
    end: tuple[float, float, float]
    pitch: float
    duration: float

    def __call__(self, t: float) -> tuple[np.ndarray, float]:
        s = _ease(t, self.duration)
        start = np.asarray(self.start, float)
        return start + s * (np.asarray(self.end, float) - start), self.pitch


@dataclass(frozen=True)
class Circle:
    centre: tuple[float, float, float]
    radius: float
    pitch: float
    duration: float

    def __call__(self, t: float) -> tuple[np.ndarray, float]:
        angle = 2.0 * math.pi * _ease(t, self.duration)
        centre = np.asarray(self.centre, float)
        outward = np.array([centre[0], centre[1], 0.0]) / math.hypot(centre[0], centre[1])
        offset = self.radius * (math.cos(angle) * outward + math.sin(angle) * np.array([0, 0, 1]))
        return centre + offset, self.pitch


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


class BaseController:
    def __init__(self, path, control_hz: float, gain: float = CONTROLLER_GAIN,
                 lead: float = CONTROLLER_LEAD):
        self.path = path
        self.period = 1.0 / control_hz
        self.gain = gain
        self.lead = lead
        self._speed = np.array([NOMINAL_SPEEDS[j] for j in JOINT_NAMES])

    def reference(self, t: float) -> np.ndarray:
        position, pitch = self.path(t)
        q = inverse_kinematics(position, pitch)
        if q is None:
            raise ValueError(f"the path at t = {t:.3f} s is out of reach")
        return q

    def __call__(self, t: float, positions: np.ndarray) -> np.ndarray:
        ahead = self.reference(t + self.lead)
        feedforward = _wrap(self.reference(t + self.lead + self.period) - ahead) / self.period
        error = _wrap(self.reference(t) - np.asarray(positions, float))
        return np.clip((feedforward + self.gain * error) / self._speed, -1.0, 1.0)
