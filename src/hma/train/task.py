"""The propping and turning tasks: settings, rungs, seeded draws, the tilt and heading references
against time, and the baseline's tip path: put down, held where it touched while the tracks roll
about their rear wheel to give the reference tilt, held down through the wait, then lifted back."""

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
    hold: tuple[float, float] = (3.0, 10.0)
    ramp_down: tuple[float, float] = (6.0, 10.0)
    wait: float = 1.5
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
                                                   "rush": 1.0})
    roll_scale: float = math.radians(5.0)
    slide_scale: float = 0.05
    hop: float = 0.025
    band: float = math.radians(3.0)
    hold_share: float = 0.8
    landing: float = 0.3
    carry: float = 0.5
    window: float = 0.5
    still: float = 0.02
    scale: float = 0.3
    smoothing: float = 0.28
    descent: tuple[float, float] = (0.15, 3.5)
    rush_scale: float = 0.3


UNFILTERED = 1.0


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
    wait: float
    final: float

    @property
    def length(self) -> float:
        return self.rest + self.ramp_up + self.hold + self.ramp_down + self.wait + self.final

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
            wait=settings.wait,
            final=settings.final,
        )
        if reachable(PropPath(task, pivot), period) and reachable(
                PropPath(task, pivot, ground), period):
            return task
    raise ValueError(f"no reachable propping task found for seed {seed}")


TURN_SHAPE = 1.875


@dataclass(frozen=True)
class TurnSettings(Settings):
    target: tuple[float, float] = (math.radians(5.0), math.radians(10.0))
    rest: float = 3.0
    ramp: tuple[float, float] = (3.0, 5.0)
    ramp_down: tuple[float, float] = (4.0, 6.0)
    wait: float = 0.0
    final: float = 1.0
    prop_x: tuple[float, float] = (4.0, 6.5)
    heading: tuple[float, float] = (math.radians(5.0), math.radians(22.0))
    rate: tuple[float, float] = (math.radians(4.5), math.radians(7.0))
    settle: tuple[float, float] = (0.5, 1.5)
    recentre: tuple[float, float] = (6.0, 10.0)
    heading_band: float = math.radians(1.5)
    belts: float = math.radians(1.5)
    turn_slide: float = 0.15
    slide_limit: float = 0.3
    peak_roll: float = math.radians(2.0)
    centred: float = math.radians(2.0)
    overshoot: float = math.radians(5.0)
    propped: float = math.radians(1.0)
    stray_pass: float = 0.05
    stray_end: float = 0.25
    distance_pass: float = 0.25


@dataclass(frozen=True)
class TurnTask(Task):
    heading: float
    settle: float
    turn: float
    recentre: float

    @property
    def turn_span(self) -> tuple[float, float]:
        start = self.rest + self.ramp_up + self.settle
        return start, start + self.turn

    @property
    def hold_span(self) -> tuple[float, float]:
        return self.turn_span

    @property
    def recentre_span(self) -> tuple[float, float]:
        start = self.rest + self.ramp_up + self.hold + self.ramp_down + self.wait + self.place
        return start, start + self.recentre

    def heading_at(self, t: float) -> float:
        return self.heading * _ease(t - self.turn_span[0], self.turn)

    def heading_rate(self, t: float) -> float:
        s = min(max((t - self.turn_span[0]) / self.turn, 0.0), 1.0)
        return self.heading * 30.0 * s * s * (1.0 - s) ** 2 / self.turn


def draw_turn_task(seed: int, settings: TurnSettings, pivot: tuple[float, float, float],
                   period: float) -> TurnTask:
    for attempt in range(MAX_ATTEMPTS):
        rng = np.random.default_rng((seed, 13, attempt))
        settle = float(rng.uniform(*settings.settle))
        size = float(rng.uniform(*settings.heading))
        turn = TURN_SHAPE * size / float(rng.uniform(*settings.rate))
        recentre = float(rng.uniform(*settings.recentre))
        task = TurnTask(
            seed=seed,
            target=float(rng.uniform(*settings.target)),
            ramp_up=float(rng.uniform(*settings.ramp)),
            hold=settle + turn,
            ramp_down=float(rng.uniform(*settings.ramp_down)),
            prop_x=float(rng.uniform(*settings.prop_x)),
            start_z=float(rng.uniform(*settings.lift)),
            pitch=float(rng.uniform(*settings.pitch)),
            attempts=attempt + 1,
            rest=settings.rest,
            place=settings.place,
            wait=settings.wait,
            final=settings.place + recentre + settings.final,
            heading=size * float(rng.choice((-1.0, 1.0))),
            settle=settle,
            turn=turn,
            recentre=recentre,
        )
        if reachable(PropPath(task, pivot), period):
            return task
    raise ValueError(f"no reachable turning task found for seed {seed}")
