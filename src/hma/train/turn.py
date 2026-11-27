"""The turning scene: a PropScene whose machines prop on the bucket, counter-rotate their tracks to
a new heading while propped, set down and re-centre the upper structure; its six commands, sensed
heading, pass rule and early ends, and the scripted baseline it is measured against."""

from __future__ import annotations

import math

import numpy as np

from hma.hydraulics import BaseController
from hma.hydraulics.layer import JOINTS, SWING
from hma.machine.mass import GAUGE
from hma.train.scene import (
    CAUSES,
    CONTROL_HZ,
    EPISODE_ARRAYS,
    FRAME,
    FRAME_HIGH,
    FRAME_LOW,
    HISTORY,
    PREVIEW,
    PropScene,
    _matrices,
    _wrap,
)
from hma.train.task import RUNGS, PropPath, TurnSettings, TurnTask, draw_turn_task

ACTIONS = JOINTS + 2
TRACK_SPEED = 0.3
HALF_GAUGE = GAUGE / 2
SWING_OPEN = 0.3
RECENTRE_GAIN = 2.0
RECENTRE_CAP = 0.6
RECENTRE_FLOOR = 0.3
RECENTRE_PUSH = math.radians(1.0)
RECENTRE_DONE = math.radians(0.3)
TURN_FIELDS = (
    ("track", 2, 1.0, -1.0, 1.0),
    ("yaw_rate", 1, 2.0, -5.0, 5.0),
    ("heading_to_go", 1, math.pi / 2, -2.0, 2.0),
    ("heading_reference", len(PREVIEW), math.pi / 2, -1.0, 1.0),
)
TURN_SCALE = np.concatenate([np.full(f[1], f[2]) for f in TURN_FIELDS])
TURN_FRAME = FRAME + len(TURN_SCALE)
TURN_LOW = np.tile(np.concatenate([FRAME_LOW, *[np.full(f[1], f[3]) for f in TURN_FIELDS]]),
                   HISTORY)
TURN_HIGH = np.tile(np.concatenate([FRAME_HIGH, *[np.full(f[1], f[4]) for f in TURN_FIELDS]]),
                    HISTORY)


def _heading(rot: np.ndarray) -> np.ndarray:
    return np.arctan2(rot[..., 1, 0], rot[..., 0, 0])


class TurnController:
    def __init__(self, task: TurnTask, pivot: tuple[float, float, float],
                 control_hz: float = CONTROL_HZ):
        self.task = task
        self.arm = BaseController(PropPath(task, pivot), control_hz)
        rear = abs(pivot[0])
        self.reach = rear * (1.0 + rear / task.prop_x)

    def __call__(self, t: float, positions: np.ndarray, pressures: np.ndarray) -> np.ndarray:
        task = self.task
        out = np.zeros(ACTIONS)
        out[:JOINTS] = self.arm(t, positions, pressures)
        start, end = task.turn_span
        if start <= t < end:
            rate = task.heading_rate(t)
            forward = self.reach * abs(task.heading_at(t)) * abs(rate)
            out[JOINTS:] = np.array([forward - rate * HALF_GAUGE,
                                     forward + rate * HALF_GAUGE]) / TRACK_SPEED
            out[SWING] = -math.copysign(SWING_OPEN, task.heading)
        elif end <= t < task.recentre_span[0]:
            out[SWING] = 0.0
        elif t >= task.recentre_span[0]:
            error = -float(_wrap(np.asarray(positions[SWING], float)))
            push = float(np.clip(RECENTRE_GAIN * error, -RECENTRE_CAP, RECENTRE_CAP))
            if abs(error) > RECENTRE_PUSH:
                push = math.copysign(max(abs(push), RECENTRE_FLOOR), push)
            out[SWING] = 0.0 if abs(error) < RECENTRE_DONE else push
        return np.clip(out, -1.0, 1.0)


class TurnScene(PropScene):
    actions = ACTIONS
    frame = TURN_FRAME
    obs_low, obs_high = TURN_LOW, TURN_HIGH
    causes = (*CAUSES, "slide", "overshoot", "lost")
    episode_arrays = (*EPISODE_ARRAYS, "heading0", "heading_est", "head_total", "turn_tip",
                      "turn_started", "turn_slide", "peak_roll", "belt_heading")
    settings_type = TurnSettings

    def __init__(self, n: int, rung: str = "nominal", mode: str = "residual",
                 settings: TurnSettings | None = None):
        if rung in RUNGS and RUNGS[rung].ground:
            raise ValueError("the turning task runs on level ground only, the nominal and "
                             "machines rungs; rough ground is not built for it yet")
        super().__init__(n, rung, mode, settings)
        self._travel = self.layer._travel
        self.heading0 = np.zeros(n)
        self.heading_est = np.zeros(n)
        self.head_total = np.zeros(n)
        self.turn_tip = np.zeros((n, 2))
        self.turn_started = np.zeros(n, bool)
        self.turn_slide = np.zeros(n)
        self.peak_roll = np.zeros(n)
        self.belt_heading = np.zeros(n)

    def _draw(self, seed: int, surface) -> TurnTask:
        return draw_turn_task(seed, self.settings, self.pivot, self.period)

    def _controller(self, task: TurnTask) -> TurnController:
        return TurnController(task, self.pivot)

    def _base_rot(self) -> np.ndarray:
        return _matrices(self.data.qpos[self._base_q[:, None] + np.arange(3, 7)])

    def _begin(self, index: int) -> None:
        self.heading0[index] = _heading(self._base_rot()[index])
        self.heading_est[index] = 0.0
        self.head_total[index] = 0.0
        self.turn_tip[index] = 0.0
        self.turn_started[index] = False
        self.turn_slide[index] = 0.0
        self.peak_roll[index] = 0.0
        self.belt_heading[index] = 0.0

    def _sense(self, index: int, sensed, gyro: np.ndarray) -> None:
        if self.k[index] > 0:
            self.heading_est[index] += self.period * (gyro[2] - sensed.velocities[index, SWING])

    def _frame(self, index: int, sensed, gyro, accel, tilt) -> np.ndarray:
        task = self.tasks[index]
        t = self.k[index] * self.period
        extra = np.concatenate([self.sent[index, JOINTS:],
                                [gyro[2] - sensed.velocities[index, SWING],
                                 task.heading - self.heading_est[index]],
                                [task.heading_at(t + ahead) for ahead in PREVIEW]])
        return np.concatenate([super()._frame(index, sensed, gyro, accel, tilt),
                               extra / TURN_SCALE])

    def _send(self, commands: np.ndarray) -> None:
        self.layer.act(commands[:, :JOINTS])
        self.data.ctrl[self._travel] = commands[:, JOINTS:] * TRACK_SPEED

    def _prepare(self, times: np.ndarray, commands: np.ndarray) -> None:
        tasks = self.tasks
        self._head_ref = np.array([[task.heading_at(t) for t in row]
                                   for task, row in zip(tasks, times, strict=True)])
        spans = np.array([task.turn_span for task in tasks])
        self._in_turn = (times >= spans[:, :1]) & (times < spans[:, 1:])
        self._goal = np.array([task.heading for task in tasks])
        spin = (commands[:, JOINTS + 1] - commands[:, JOINTS]) / 2.0 * TRACK_SPEED
        self._turning = np.abs(spin) > self.settings.turning
        self._head_sum = np.zeros(self.n)
        self._lost = np.zeros(self.n, bool)
        self._over = np.zeros(self.n, bool)

    def _watch(self, step: int, rot: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> None:
        s = self.settings
        heading = _wrap(_heading(rot) - self.heading0)
        error = np.abs(heading - self._head_ref[:, step])
        self._head_sum += error
        self.head_total += error
        self.peak_roll = np.maximum(self.peak_roll, np.abs(roll))
        belts = self.data.actuator_velocity[self._travel]
        self.belt_heading += self.dt * (belts[:, 1] - belts[:, 0]) / GAUGE
        inside = self._in_turn[:, step]
        tip = self.data.site_xpos[self._tips, :2]
        first = inside & ~self.turn_started
        self.turn_tip[first] = tip[first]
        self.turn_started |= inside
        moved = np.linalg.norm(tip - self.turn_tip, axis=1)
        self.turn_slide = np.where(inside, np.maximum(self.turn_slide, moved), self.turn_slide)
        self._lost |= self._turning & (np.abs(pitch) < s.lost_tilt)
        self._over |= (heading - self._goal) * np.sign(self._goal) > s.overshoot

    def _error(self, tilt_error: np.ndarray, done_steps: int) -> np.ndarray:
        return tilt_error + self._head_sum / max(done_steps, 1)

    def _ends(self) -> dict[str, np.ndarray]:
        return {"slide": self.turn_slide > self.settings.slide_limit, "overshoot": self._over,
                "lost": self._lost}

    def _turn_reasons(self, info: dict) -> list[str]:
        s = self.settings
        checks = ((abs(info["heading_error"]) > s.heading_band,
                   f"final heading more than {math.degrees(s.heading_band):.1f} deg off the step"),
                  (info["turn_slide"] > s.turn_slide,
                   f"bucket slid more than {s.turn_slide} m during the turn"),
                  (info["peak_roll"] > s.peak_roll,
                   f"roll above {math.degrees(s.peak_roll):.0f} deg"),
                  (abs(info["heading"] - info["belt_heading"]) > s.belts,
                   f"heading more than {math.degrees(s.belts):.1f} deg from what the belts turned"),
                  (abs(info["swing_left"]) > s.centred,
                   f"upper structure more than {math.degrees(s.centred):.0f} deg off the tracks"))
        return [reason for failed, reason in checks if failed]

    def reasons(self, info: dict) -> list[str]:
        return super().reasons(info) + self._turn_reasons(info)

    def outcome(self, index: int, cause: str | None, final_tilt: float, carried: float,
                speed: float, bucket_down: bool, front_down: bool) -> dict:
        info = super().outcome(index, cause, final_tilt, carried, speed, bucket_down, front_down)
        heading = float(_wrap(_heading(self._base_rot()[index]) - self.heading0[index]))
        info.update({
            "heading": heading, "heading_error": heading - self.tasks[index].heading,
            "heading_tracking": float(self.head_total[index]
                                      / max(int(self.steps_total[index]), 1)),
            "belt_heading": float(self.belt_heading[index]),
            "turn_slide": float(self.turn_slide[index]), "peak_roll": float(self.peak_roll[index]),
            "swing_left": float(_wrap(self.data.qpos[self._swing_q[index]])),
        })
        info["passed"] = bool(info["passed"] and not self._turn_reasons(info))
        return info


TASKS = {"prop": PropScene, "turn": TurnScene}
