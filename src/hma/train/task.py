"""The propping task: settings, rungs, seeded draws, the tilt reference against time, and the
baseline's tip path: put down, held where it touched while the tracks roll about their rear wheel
on level rigid ground to give the reference tilt, then lifted back along the same vertical path."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from hma.hydraulics.controller import CONTROLLER_LEAD, _ease
from hma.machine import inverse_kinematics
from hma.machine.kinematics import BOOM_PIVOT

DEV_BASE = 10_000_000
HELD_OUT_BASE = 20_000_000
DEV_SIZE = 64
CHECK_STEP = 0.05
MAX_ATTEMPTS = 200


@dataclass(frozen=True)
class Rung:
    machines: bool
    ground: bool


RUNGS = {
    "nominal": Rung(machines=False, ground=False),
    "machines": Rung(machines=True, ground=False),
    "ground": Rung(machines=True, ground=True),
}


@dataclass(frozen=True)
class Settings:
    target: tuple[float, float] = (math.radians(5.0), math.radians(20.0))
    rest: float = 4.0
    place: float = 2.5
    ramp: tuple[float, float] = (3.0, 6.0)
    ramp_down: tuple[float, float] = (6.0, 10.0)
    settle: float = 1.5
    hold: tuple[float, float] = (3.0, 10.0)
    final: float = 3.0
    prop_x: tuple[float, float] = (3.5, 6.5)
    lift: tuple[float, float] = (0.5, 1.5)
    pitch: tuple[float, float] = (math.radians(60.0), math.radians(90.0))
    tilt_limit: float = math.radians(35.0)
    roll_limit: float = math.radians(10.0)
    sigma: float = math.radians(3.0)
    floor: float = 0.2
    weights: dict = field(default_factory=lambda: {"change": 0.03, "roll": 0.03, "slide": 0.03,
                                                   "landing": 0.05, "set_down": 0.05,
                                                   "finish": 60.0})
    roll_scale: float = math.radians(5.0)
    slide_scale: float = 0.05
    hop: float = 0.025
    band: float = math.radians(3.0)
    hold_share: float = 0.8
    landing: float = 0.3
    gentle: float = 0.15
    carry: float = 0.5
    window: float = 0.5
    still: float = 0.02
    scale: float = 0.3


@dataclass(frozen=True)
class Task:
    seed: int
    target: float
    ramp_up: float
    hold: float
    ramp_down: float
    prop_x: float
    start_z: float
    pitch: float
    attempts: int
    rest: float
    place: float
    final: float
    settle: float = 0.0

    @property
    def length(self) -> float:
        return self.rest + self.ramp_up + self.hold + self.ramp_down + self.settle + self.final

    @property
    def hold_span(self) -> tuple[float, float]:
        start = self.rest + self.ramp_up
        return start, start + self.hold

    def tilt(self, t: float) -> float:
        up = t - self.rest
        down = up - self.ramp_up - self.hold
        if up <= 0.0:
            return 0.0
        if down <= 0.0:
            return self.target * _ease(up, self.ramp_up)
        return self.target * (1.0 - _ease(down, self.ramp_down))


def dev_seeds(size: int = DEV_SIZE) -> list[int]:
    return [DEV_BASE + k for k in range(size)]


def held_out_seeds(size: int) -> list[int]:
    return [HELD_OUT_BASE + k for k in range(size)]


def training_seed(rng: np.random.Generator) -> int:
    return int(rng.integers(0, DEV_BASE))


class PropPath:
    def __init__(self, task: Task, pivot: tuple[float, float, float], height: float = 0.0):
        self.task = task
        self.pivot = pivot
        self.height = height

    def __call__(self, t: float) -> tuple[np.ndarray, float]:
        task, side = self.task, float(BOOM_PIVOT[1])
        if t < task.place:
            z = self.height + (task.start_z - self.height) * (1.0 - _ease(t, task.place))
            return np.array([task.prop_x, side, z]), task.pitch
        down = task.length - task.final
        if t > down:
            z = self.height + (task.start_z - self.height) * _ease(t - down, task.place)
            return np.array([task.prop_x, side, z]), task.pitch
        theta = task.tilt(t)
        cx, cz, radius = self.pivot
        c, s = math.cos(theta), math.sin(theta)
        tx = cx * (1.0 - c) + cz * s - radius * theta
        tz = cz * (1.0 - c) - cx * s
        dx, dz = task.prop_x - tx, self.height - tz
        return np.array([c * dx + s * dz, side, -s * dx + c * dz]), task.pitch + theta


def reachable(path: PropPath, period: float) -> bool:
    end = path.task.length + CONTROLLER_LEAD + period
    for t in np.arange(0.0, end + CHECK_STEP, CHECK_STEP):
        position, pitch = path(float(t))
        if inverse_kinematics(position, pitch) is None:
            return False
    return True


def draw_task(seed: int, settings: Settings, pivot: tuple[float, float, float],
              ground_height, period: float) -> Task:
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng((seed, 10, attempt))
        prop_x = float(rng.uniform(*settings.prop_x))
        ground = ground_height(prop_x, float(BOOM_PIVOT[1]))
        task = Task(
            seed=seed,
            target=float(rng.uniform(*settings.target)),
            ramp_up=float(rng.uniform(*settings.ramp)),
            hold=float(rng.uniform(*settings.hold)),
            ramp_down=float(rng.uniform(*settings.ramp_down)),
            prop_x=prop_x,
            start_z=ground + float(rng.uniform(*settings.lift)),
            pitch=float(rng.uniform(*settings.pitch)),
            attempts=attempt + 1,
            rest=settings.rest,
            place=settings.place,
            final=settings.final,
            settle=settings.settle,
        )
        if reachable(PropPath(task, pivot), period) and reachable(
                PropPath(task, pivot, ground), period):
            return task
    raise ValueError(f"no reachable propping task found for seed {seed}")
