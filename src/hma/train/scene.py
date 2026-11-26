"""The propping scene: n machines, each an independent episode, stepped by one Hydraulics layer at
10 decisions per second. It draws and writes each episode, builds the sensed observations, scores
the reward, ends episodes early, and saves and restores everything a replay needs."""

from __future__ import annotations

import math

import mujoco
import numpy as np

from hma.hydraulics import NOMINAL, BaseController, Hydraulics, draw_machine
from hma.hydraulics.layer import JOINTS, STATE, SWING
from hma.machine import TIP_SITE, inverse_kinematics, load_scene, machine_prefix
from hma.machine.model import BASE_JOINT
from hma.train import footing
from hma.train.task import RUNGS, PropPath, Settings, Task, draw_task

CONTROL_HZ = 10.0
PREVIEW = (0.0, 0.5, 1.0)
HISTORY = 15
GYRO_BIAS = math.radians(0.2)
GYRO_NOISE = math.radians(0.1)
ACCEL_BIAS = 0.05
ACCEL_NOISE = 0.05
TILT_NOISE = math.radians(0.1)
FILTER_TIME = 2.0
CAUSES = ("unstable", "contact", "tilt", "roll")
MODES = ("residual", "direct")
GROUND, FORBIDDEN, BUCKET, FRONT = 1, 2, 3, 4
FIELDS = (
    ("position", JOINTS, math.pi, -1.0, 1.0),
    ("velocity", JOINTS, 2.0, -5.0, 5.0),
    ("pressure", 2 * JOINTS, 40e6, 0.0, 2.0),
    ("gyro", 3, 2.0, -5.0, 5.0),
    ("accel", 3, 9.81, -10.0, 10.0),
    ("tilt", 2, math.pi / 2, -2.0, 2.0),
    ("command", JOINTS, 1.0, -1.0, 1.0),
    ("reference", len(PREVIEW), math.pi / 2, -1.0, 1.0),
)
FRAME = sum(f[1] for f in FIELDS)
FRAME_LOW = np.concatenate([np.full(f[1], f[3]) for f in FIELDS])
FRAME_HIGH = np.concatenate([np.full(f[1], f[4]) for f in FIELDS])
OBS_LOW = np.tile(FRAME_LOW, HISTORY)
OBS_HIGH = np.tile(FRAME_HIGH, HISTORY)
SCALE = np.concatenate([np.full(f[1], f[2]) for f in FIELDS])
EPISODE_ARRAYS = ("k", "steps_total", "pitch0", "roll0", "est0", "sent", "history", "up", "v_prev",
                  "gyro_bias", "accel_bias", "imu_on", "tip_prev", "bucket_prev", "front_prev",
                  "landing", "in_band", "hold_steps", "err_total", "done", "active")


def _matrices(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat.T
    out = np.empty((len(quat), 3, 3))
    out[:, 0, 0] = 1 - 2 * (y * y + z * z)
    out[:, 0, 1] = 2 * (x * y - w * z)
    out[:, 0, 2] = 2 * (x * z + w * y)
    out[:, 1, 0] = 2 * (x * y + w * z)
    out[:, 1, 1] = 1 - 2 * (x * x + z * z)
    out[:, 1, 2] = 2 * (y * z - w * x)
    out[:, 2, 0] = 2 * (x * z - w * y)
    out[:, 2, 1] = 2 * (y * z + w * x)
    out[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    out[:, 0] = a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1]
    out[:, 1] = a[:, 2] * b[:, 0] - a[:, 0] * b[:, 2]
    out[:, 2] = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    return out


def _turn(vectors: np.ndarray, angle: np.ndarray) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    x, y = vectors[:, 0], vectors[:, 1]
    return np.stack([c * x - s * y, s * x + c * y, vectors[:, 2]], -1)


def _tilts(up: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.arctan2(up[:, 0], up[:, 2]), np.arctan2(up[:, 1], up[:, 2])


def _wrap(angle: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(angle), np.cos(angle))


class PropScene:
    def __init__(self, n: int, rung: str = "nominal", mode: str = "residual",
                 settings: Settings | None = None):
        if rung not in RUNGS:
            raise ValueError(f"rung must be one of {sorted(RUNGS)}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.n, self.rung, self.mode = n, rung, mode
        self.settings = settings or Settings()
        model = load_scene(n)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        self.model, self.data = model, mujoco.MjData(model)
        self.layer = Hydraulics(model, self.data, [NOMINAL] * n, CONTROL_HZ)
        self.steps = self.layer.steps_per_decision
        self.dt = self.layer.dt
        self.period = 1.0 / CONTROL_HZ
        self.parking = footing.Parking(model, n)
        self.pivot = footing.rear_pivot(model)
        self.gravity = model.opt.gravity.copy()
        self.filter_gain = self.dt / FILTER_TIME

        base = [model.joint(machine_prefix(i) + BASE_JOINT) for i in range(n)]
        self._base_q = np.array([int(j.qposadr[0]) for j in base])
        self._base_v = np.array([int(j.dofadr[0]) for j in base])
        self._swing_q = self.layer._qpos[:, SWING]
        self._swing_v = self.layer._qvel[:, SWING]
        self._upper_at = model.body(machine_prefix(0) + "upper").pos.copy()
        self._front_at = footing.front_edge(model)
        self._tips = np.array([model.site(machine_prefix(i) + TIP_SITE).id for i in range(n)])
        self._own_q = [np.concatenate([np.arange(model.jnt_qposadr[j],
                                                 model.jnt_qposadr[j] + size)
                                       for j, size in self._joints(i)]) for i in range(n)]
        self._role = np.zeros(model.ngeom, np.int8)
        self._owner = np.full(model.ngeom, -1)
        for i in range(n):
            for role, geoms in ((GROUND, footing.ground_shapes(model, i)),
                                (FORBIDDEN, footing.forbidden_shapes(model, i)),
                                (BUCKET, footing.touches_ground(model, i, footing.ALLOWED_LINK)),
                                (FRONT, footing.front_feet(model, i))):
                self._role[geoms] = role
                self._owner[geoms] = i

        self.seeds: list[int | None] = [None] * n
        self.tasks: list[Task | None] = [None] * n
        self.grounds: list[footing.Ground] = [footing.FLAT] * n
        self.machines = [NOMINAL] * n
        self.controllers: list[BaseController | None] = [None] * n
        self._imu_rngs: list[np.random.Generator | None] = [None] * n
        self.k = np.zeros(n, np.intp)
        self.steps_total = np.zeros(n, np.intp)
        self.pitch0 = np.zeros(n)
        self.roll0 = np.zeros(n)
        self.est0 = np.zeros((n, 2))
        self.sent = np.zeros((n, JOINTS))
        self.history = np.zeros((n, HISTORY, FRAME))
        self.up = np.zeros((n, 3))
        self.v_prev = np.zeros((n, 3))
        self.gyro_bias = np.zeros((n, 3))
        self.accel_bias = np.zeros((n, 3))
        self.imu_on = np.zeros(n)
        self.tip_prev = np.zeros((n, 3))
        self.bucket_prev = np.zeros(n, bool)
        self.front_prev = np.zeros(n, bool)
        self.landing = np.zeros(n)
        self.in_band = np.zeros(n, np.intp)
        self.hold_steps = np.zeros(n, np.intp)
        self.err_total = np.zeros(n)
        self.done = np.zeros(n, bool)
        self.active = np.zeros(n, bool)
        self._warnings = self._unstable_count()

    def _joints(self, index: int):
        prefix = machine_prefix(index)
        for j in range(self.model.njnt):
            if self.model.joint(j).name.startswith(prefix):
                kind = int(self.model.jnt_type[j])
                yield j, {0: 7, 1: 4}.get(kind, 1)

    def _unstable_count(self) -> int:
        return int(self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)

    def draw(self, index: int, seed: int) -> tuple:
        rung = RUNGS[self.rung]
        machine = draw_machine(seed) if rung.machines else NOMINAL
        ground = footing.draw_ground(seed) if rung.ground else footing.FLAT
        task = draw_task(seed, self.settings, self.pivot,
                         lambda x, y: footing.surface(ground, x, y), self.period)
        return machine, ground, task

    def _write(self, index: int, seed: int) -> None:
        machine, ground, task = self.draw(index, seed)
        path = PropPath(task, self.pivot)
        position, pitch = path(0.0)
        pose = inverse_kinematics(position, pitch)
        footing.apply_ground(self.model, index, ground)
        self.layer.reset(index, machine, pose)
        self.parking.park(self.model, self.data, index)
        self.seeds[index], self.tasks[index] = seed, task
        self.machines[index], self.grounds[index] = machine, ground
        self.controllers[index] = BaseController(path, CONTROL_HZ)

    def reset(self, index: int, seed: int) -> np.ndarray:
        self._write(index, seed)
        mujoco.mj_forward(self.model, self.data)
        rng = self._imu_rngs[index] = np.random.default_rng((seed, 11))
        on = self.imu_on[index] = float(RUNGS[self.rung].machines)
        self.gyro_bias[index] = rng.uniform(-1.0, 1.0, 3) * GYRO_BIAS * on
        self.accel_bias[index] = rng.uniform(-1.0, 1.0, 3) * ACCEL_BIAS * on
        quat = self.data.qpos[self._base_q[index] + 3:self._base_q[index] + 7]
        rot = _matrices(quat[None])[0]
        up = rot[2]
        self.pitch0[index], self.roll0[index] = math.atan2(up[0], up[2]), math.atan2(up[1], up[2])
        swing = self.data.qpos[self._swing_q[index]]
        self.up[index] = _turn(up[None], np.array([-swing]))[0]
        self.v_prev[index] = self._upper_velocity()[index]
        self.k[index] = 0
        self.steps_total[index] = 0
        self.sent[index] = 0.0
        self.tip_prev[index] = self.data.site_xpos[self._tips[index]]
        self.bucket_prev[index] = False
        self.front_prev[index] = True
        self.landing[index] = 0.0
        self.in_band[index] = 0
        self.hold_steps[index] = 0
        self.err_total[index] = 0.0
        self.done[index] = False
        self.active[index] = True
        sensed = self.layer.observe_mean()
        gyro = self.gyro_bias[index] + rng.uniform(-1.0, 1.0, 3) * GYRO_NOISE * on
        accel = (_matrices(quat[None])[0].T @ -self.gravity)
        accel = _turn(accel[None], np.array([-swing]))[0] + self.accel_bias[index]
        accel += rng.uniform(-1.0, 1.0, 3) * ACCEL_NOISE * on
        tilt = self._sensed_tilt(index, sensed.positions[index, SWING], rng)
        self.est0[index] = tilt
        frame = self._frame(index, sensed, gyro, accel, tilt)
        self.history[index] = frame
        return self.history[index].ravel().copy()

    def _sensed_tilt(self, index: int, swing: float, rng: np.random.Generator) -> np.ndarray:
        base_up = _turn(self.up[index][None], np.array([swing]))
        pitch, roll = _tilts(base_up)
        noise = rng.uniform(-1.0, 1.0, 2) * TILT_NOISE * self.imu_on[index]
        return np.array([pitch[0], roll[0]]) + noise

    def _frame(self, index: int, sensed, gyro, accel, tilt) -> np.ndarray:
        task = self.tasks[index]
        t = self.k[index] * self.period
        reference = [task.tilt(t + ahead) for ahead in PREVIEW]
        raw = np.concatenate([_wrap(sensed.positions[index]), sensed.velocities[index],
                              sensed.pressures[index].ravel(), gyro, accel,
                              tilt - self.est0[index], self.sent[index], reference])
        return raw / SCALE

    def _upper_velocity(self) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        quat = q[self._base_q[:, None] + np.arange(3, 7)]
        rot = _matrices(quat)
        omega = rot @ v[self._base_v[:, None] + np.arange(3, 6)][..., None]
        arm = rot @ self._upper_at
        return v[self._base_v[:, None] + np.arange(3)] + _cross(omega[..., 0], arm)

    def baseline(self) -> np.ndarray:
        sensed = self.layer.observe()
        out = np.zeros((self.n, JOINTS))
        for i, controller in enumerate(self.controllers):
            if controller is not None and self.active[i]:
                out[i] = controller(self.k[i] * self.period, sensed.positions[i],
                                    sensed.pressures[i])
        return out

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, float)
        if actions.shape != (self.n, JOINTS) or not np.all(np.isfinite(actions)):
            raise ValueError(f"actions must be finite with shape {(self.n, JOINTS)}")
        if not self.active.all():
            raise ValueError("every machine must be reset before the scene steps")
        commands = np.clip(actions, -1.0, 1.0)
        if self.mode == "residual":
            commands = np.clip(self.baseline() + commands, -1.0, 1.0)
        self.layer.act(commands)
        n, dt, data, s = self.n, self.dt, self.data, self.settings
        times = (self.k[:, None] * self.steps + np.arange(1, self.steps + 1)) * dt
        reference = np.array([[task.tilt(t) for t in row]
                              for task, row in zip(self.tasks, times, strict=True)])
        holds = np.array([task.hold_span for task in self.tasks])
        in_hold = (times >= holds[:, :1]) & (times < holds[:, 1:])
        err_sum = np.zeros(n)
        roll_sum = np.zeros(n)
        worst_tilt = np.zeros(n)
        worst_roll = np.zeros(n)
        slide = np.zeros(n)
        forbidden = np.zeros(n, bool)
        gyro_sum = np.zeros((n, 3))
        accel_sum = np.zeros((n, 3))
        gain = self.filter_gain
        base_q, base_v = self._base_q, self._base_v
        for step in range(self.steps):
            self.layer.step()
            q, v = data.qpos, data.qvel
            rot = _matrices(q[base_q[:, None] + np.arange(3, 7)])
            up = rot[:, 2, :]
            pitch, roll = _tilts(up)
            err = np.abs(pitch - self.pitch0 - reference[:, step])
            err_sum += err
            roll_sum += np.abs(roll - self.roll0)
            worst_tilt = np.maximum(worst_tilt, np.abs(pitch))
            worst_roll = np.maximum(worst_roll, np.abs(roll))
            self.in_band += in_hold[:, step] & (err <= s.band)
            self.hold_steps += in_hold[:, step]
            swing = q[self._swing_q]
            local = v[base_v[:, None] + np.arange(3, 6)]
            omega = rot @ local[..., None]
            arm = rot @ self._upper_at
            velocity = v[base_v[:, None] + np.arange(3)] + _cross(omega[..., 0], arm)
            force = (velocity - self.v_prev) / dt - self.gravity
            self.v_prev = velocity
            spin = local.copy()
            spin[:, 2] += v[self._swing_v]
            gyro = _turn(spin, -swing) + self.gyro_bias
            accel = _turn((np.swapaxes(rot, 1, 2) @ force[..., None])[..., 0], -swing)
            accel += self.accel_bias
            gyro_sum += gyro
            accel_sum += accel
            est = self.up - dt * _cross(gyro, self.up)
            norm = np.linalg.norm(accel, axis=1, keepdims=True)
            est += gain * (accel / np.maximum(norm, 1e-9) - est)
            self.up = est / np.linalg.norm(est, axis=1, keepdims=True)
            bucket, front, bad = self._contacts()
            forbidden |= bad
            tip = data.site_xpos[self._tips]
            both = bucket & self.bucket_prev
            slide += np.where(both, np.linalg.norm((tip - self.tip_prev)[:, :2], axis=1), 0.0)
            self.tip_prev, self.bucket_prev = tip.copy(), bucket
            landed = front & ~self.front_prev
            if landed.any():
                point = velocity - _cross(omega[..., 0], arm) + _cross(
                    omega[..., 0], rot @ self._front_at)
                self.landing = np.where(landed, np.maximum(self.landing, -point[:, 2]),
                                        self.landing)
            self.front_prev = front
        self.k += 1
        self.steps_total += self.steps
        self.err_total += err_sum
        sensed = self.layer.observe_mean()
        mean_err = err_sum / self.steps
        change = np.abs(commands - self.sent).mean(1) / 2.0
        self.sent = commands
        w = s.weights
        reward = (s.floor + (1.0 - s.floor) * np.exp(-mean_err / s.sigma)
                  - w["change"] * change
                  - w["roll"] * np.minimum(roll_sum / self.steps / s.roll_scale, 1.0)
                  - w["slide"] * np.minimum(slide / s.slide_scale, 1.0))
        unstable = self._unstable_count() != self._warnings
        self._warnings = self._unstable_count()
        causes: list[str | None] = []
        for i in range(n):
            finite = (np.all(np.isfinite(data.qpos[self._own_q[i]]))
                      and np.all(np.isfinite(self.layer._pressure[i])))
            flags = {"unstable": unstable or not finite, "contact": bool(forbidden[i]),
                     "tilt": worst_tilt[i] > s.tilt_limit, "roll": worst_roll[i] > s.roll_limit}
            causes.append(next((c for c in CAUSES if flags[c]), None))
        obs = np.zeros((n, HISTORY * FRAME))
        terminated = np.zeros(n, bool)
        infos: list[dict] = [{} for _ in range(n)]
        for i in range(n):
            rng = self._imu_rngs[i]
            on = self.imu_on[i]
            gyro = gyro_sum[i] / self.steps + rng.uniform(-1.0, 1.0, 3) * GYRO_NOISE * on
            accel = accel_sum[i] / self.steps + rng.uniform(-1.0, 1.0, 3) * ACCEL_NOISE * on
            tilt = self._sensed_tilt(i, sensed.positions[i, SWING], rng)
            self.history[i, :-1] = self.history[i, 1:]
            self.history[i, -1] = self._frame(i, sensed, gyro, accel, tilt)
            obs[i] = self.history[i].ravel()
            end = self.k[i] * self.period >= self.tasks[i].length - 1e-9
            if causes[i] is not None:
                reward[i] = 0.0
            if causes[i] is not None or end:
                terminated[i] = True
                self.done[i] = True
                self.active[i] = False
                infos[i] = self.outcome(i, causes[i], pitch[i] - self.pitch0[i])
        return obs, reward, terminated, infos

    def outcome(self, index: int, cause: str | None, final_tilt: float) -> dict:
        s = self.settings
        share = self.in_band[index] / max(int(self.hold_steps[index]), 1)
        passed = (cause is None and share >= s.hold_share and self.landing[index] < s.landing
                  and abs(final_tilt) < s.settled)
        return {"seed": self.seeds[index], "cause": cause, "passed": bool(passed),
                "hold_share": float(share), "landing": float(self.landing[index]),
                "final_tilt": float(final_tilt),
                "tilt_error": float(self.err_total[index] / max(int(self.steps_total[index]), 1)),
                "decisions": int(self.k[index])}

    def _contacts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        flags = np.zeros((5, self.n), bool)
        count = self.data.ncon
        if count:
            pairs = self.data.contact.geom[:count]
            a, b = self._role[pairs[:, 0]], self._role[pairs[:, 1]]
            grounded = (a == GROUND) | (b == GROUND)
            other = np.where(a == GROUND, pairs[:, 1], pairs[:, 0])[grounded]
            kind = self._role[other]
            flags[kind, self._owner[other]] = True
        return flags[BUCKET], flags[FRONT], flags[FORBIDDEN]

    def snapshot(self) -> dict:
        if any(seed is None for seed in self.seeds):
            raise ValueError("every machine must be reset before a snapshot")
        state = np.zeros(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, state, STATE)
        return {
            "n": self.n, "rung": self.rung, "mode": self.mode, "settings": self.settings,
            "seeds": list(self.seeds), "physics": state, "layer": self.layer.snapshot(),
            "episode": {name: np.copy(getattr(self, name)) for name in EPISODE_ARRAYS},
            "imu": [rng.bit_generator.state for rng in self._imu_rngs],
            "warnings": self._warnings,
        }

    def restore(self, snapshot: dict) -> None:
        if (snapshot["n"], snapshot["rung"], snapshot["mode"]) != (self.n, self.rung, self.mode):
            raise ValueError("the snapshot is of a different scene")
        if snapshot["settings"] != self.settings:
            raise ValueError("the snapshot was taken with different settings")
        for index, seed in enumerate(snapshot["seeds"]):
            self._write(index, seed)
        mujoco.mj_setState(self.model, self.data, snapshot["physics"], STATE)
        mujoco.mj_forward(self.model, self.data)
        self.layer.restore(snapshot["layer"])
        for name, value in snapshot["episode"].items():
            setattr(self, name, np.copy(value))
        self._imu_rngs = []
        for state in snapshot["imu"]:
            rng = np.random.default_rng()
            rng.bit_generator.state = state
            self._imu_rngs.append(rng)
        self._warnings = snapshot["warnings"]

