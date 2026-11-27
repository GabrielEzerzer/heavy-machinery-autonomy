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
from hma.machine.contact import draw_friction, set_friction
from hma.machine.ground import GEOM_STATE, Ground, apply_ground, draw_ground
from hma.machine.kinematics import BOOM_PIVOT
from hma.machine.model import (
    ANCHOR,
    BASE_JOINT,
    RING,
    STEEL_FRICTION,
    ground_geoms,
    set_const,
)
from hma.task.checks import LIMITS
from hma.train import footing
from hma.train.task import RUNGS, PropPath, Settings, Task, draw_task, reachable

CONTROL_HZ = 10.0
PREVIEW = (0.0, 0.5, 1.0)
HISTORY = 15
GYRO_BIAS = math.radians(0.2)
GYRO_NOISE = math.radians(0.1)
ACCEL_BIAS = 0.05
ACCEL_NOISE = 0.05
TILT_NOISE = math.radians(0.5) * math.sqrt(3.0)
FILTER_TIME = 2.0
CAUSES = ("unstable", "contact", "strike", "tilt", "roll")
STRIKERS = ("arm", "bucket")
STRUCK = ("base", "upper", "boom")
BAD = ("mjWARN_BADQPOS", "mjWARN_BADQVEL", "mjWARN_BADQACC")
MODES = ("residual", "direct")
GROUND, FORBIDDEN, BUCKET, FRONT = 1, 2, 3, 4
LEVEL = Ground()
HEIGHT_TOLERANCE = 0.05
PROP_DRAWS = 1000
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
EPISODE_ARRAYS = ("k", "steps_total", "stance", "est0", "sent", "history", "up", "v_prev",
                  "gyro_bias", "accel_bias", "imu_on", "tip_prev", "apart", "front_prev",
                  "corner_prev", "touched", "rest_load", "rest_count", "landing", "recent",
                  "end_load", "end_steps", "in_band", "hold_steps", "err_total", "done",
                  "active")


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


def reward(s: Settings, error, change, roll, slide, landing, shortfall) -> np.ndarray:
    w = s.weights
    return (s.floor + (1.0 - s.floor) * np.exp(-np.asarray(error) / s.sigma)
            - w["change"] * np.minimum(change, 1.0)
            - w["roll"] * np.minimum(np.asarray(roll) / s.roll_scale, 1.0)
            - w["slide"] * np.minimum(np.asarray(slide) / s.slide_scale, 1.0)
            - w["landing"] * np.minimum(np.asarray(landing) / s.landing, 1.0)
            - w["set_down"] * np.clip(shortfall, 0.0, 1.0))


def warm_compiled_step() -> None:
    model = load_scene(1)
    layer = Hydraulics(model, mujoco.MjData(model), [NOMINAL], CONTROL_HZ)
    layer.reset(0, NOMINAL)
    layer.step()


class PropScene:
    actions = JOINTS
    frame = FRAME
    obs_low, obs_high = OBS_LOW, OBS_HIGH
    causes = CAUSES
    episode_arrays = EPISODE_ARRAYS
    settings_type = Settings

    def __init__(self, n: int, rung: str = "nominal", mode: str = "residual",
                 settings: Settings | None = None):
        if rung not in RUNGS:
            raise ValueError(f"rung must be one of {sorted(RUNGS)}")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.n, self.rung, self.mode = n, rung, mode
        self.settings = settings or self.settings_type()
        model = load_scene(n)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
        self.model, self.data = model, mujoco.MjData(model)
        self.layer = Hydraulics(model, self.data, [NOMINAL] * n, CONTROL_HZ)
        self.steps = self.layer.steps_per_decision
        self.dt = self.layer.dt
        self.period = 1.0 / CONTROL_HZ
        self.pivot = footing.rear_pivot()
        self.gravity = model.opt.gravity.copy()
        self.filter_gain = self.dt / FILTER_TIME

        base = [model.joint(machine_prefix(i) + BASE_JOINT) for i in range(n)]
        self._base_q = np.array([int(j.qposadr[0]) for j in base])
        self._base_v = np.array([int(j.dofadr[0]) for j in base])
        self._swing_q = self.layer._qpos[:, SWING]
        self._swing_v = self.layer._qvel[:, SWING]
        self._upper_at = model.body(machine_prefix(0) + "upper").pos.copy()
        fronts = footing.front_parts(model, 0)
        self._per_track = len(fronts[0])
        self._points = footing.lowest_points(model, np.concatenate(fronts))
        self._base_body = np.array([model.body(machine_prefix(i) + "base").id for i in range(n)])
        self._ring_body = np.array([model.body(machine_prefix(i) + RING).id for i in range(n)])
        self._ring_pos = model.body_pos[self._ring_body].copy()
        self._ground_rows = [ground_geoms(model, i) for i in range(n)]
        self._ground_start = [{name: getattr(model, name)[rows].copy() for name in GEOM_STATE
                               if name.startswith("geom_")} for rows in self._ground_rows]
        self._tips = np.array([model.site(machine_prefix(i) + TIP_SITE).id for i in range(n)])
        self._own_q = [np.concatenate([np.arange(model.jnt_qposadr[j],
                                                 model.jnt_qposadr[j] + size)
                                       for j, size in self._joints(i)]) for i in range(n)]
        self._role = np.zeros(model.ngeom, np.int8)
        self._owner = np.full(model.ngeom, -1)
        self._corner = np.full(model.ngeom, -1)
        self._part_of = np.full(model.ngeom, -1)
        self._striker = np.zeros(model.ngeom, bool)
        self._struck = np.zeros(model.ngeom, bool)
        qpos_owner = np.full(model.nq, -1)
        dof_owner = np.full(model.nv, -1)
        for i in range(n):
            front = np.concatenate(footing.front_parts(model, i))
            self._corner[front] = np.arange(len(self._points))
            for links, kind in ((STRIKERS, self._striker), (STRUCK, self._struck)):
                for link in links:
                    body = model.body(machine_prefix(i) + link).id
                    geoms = np.flatnonzero((model.geom_bodyid == body)
                                           & ((model.geom_contype | model.geom_conaffinity) > 0))
                    kind[geoms] = True
                    self._part_of[geoms] = i
            for j, size in self._joints(i):
                qpos_owner[model.jnt_qposadr[j]:model.jnt_qposadr[j] + size] = i
                dofs = {0: 6, 1: 3}.get(int(model.jnt_type[j]), 1)
                dof_owner[model.jnt_dofadr[j]:model.jnt_dofadr[j] + dofs] = i
            for role, geoms in ((GROUND, ground_geoms(model, i)),
                                (FORBIDDEN, footing.forbidden_shapes(model, i)),
                                (BUCKET, footing.bucket_shapes(model, i)),
                                (FRONT, front)):
                self._role[geoms] = role
                self._owner[geoms] = i

        self.seeds: list[int | None] = [None] * n
        self.tasks: list[Task | None] = [None] * n
        self.grounds: list[Ground] = [LEVEL] * n
        self.frictions = [STEEL_FRICTION[0]] * n
        self.heights = [0.0] * n
        self.machines = [NOMINAL] * n
        self.normal = np.tile([0.0, 0.0, 1.0], (n, 1))
        self.controllers: list[BaseController | None] = [None] * n
        self._imu_rngs: list[np.random.Generator | None] = [None] * n
        self.k = np.zeros(n, np.intp)
        self.steps_total = np.zeros(n, np.intp)
        self.stance = np.tile([0.0, 0.0, 1.0], (n, 1))
        self.est0 = np.zeros((n, 2))
        self.sent = np.zeros((n, self.actions))
        self.history = np.zeros((n, HISTORY, self.frame))
        self.up = np.zeros((n, 3))
        self.v_prev = np.zeros((n, 3))
        self.gyro_bias = np.zeros((n, 3))
        self.accel_bias = np.zeros((n, 3))
        self.imu_on = np.zeros(n)
        self.tip_prev = np.zeros((n, 3))
        self._hop = round(self.settings.hop / self.dt)
        self.apart = np.full(n, self._hop + 1, np.intp)
        self.front_prev = np.zeros((n, len(self._points)), bool)
        self.corner_prev = np.zeros((n, len(self._points)))
        self.touched = np.zeros(n, bool)
        self.rest_load = np.zeros((n, 2))
        self.rest_count = np.zeros(n, np.intp)
        self.landing = np.zeros(n)
        self.recent = np.zeros((n, HISTORY))
        window = max(1, round(self.settings.window / self.period))
        self.end_load = np.zeros((n, window, 2))
        self.end_steps = np.zeros((n, window), np.intp)
        self.in_band = np.zeros(n, np.intp)
        self.hold_steps = np.zeros(n, np.intp)
        self.err_total = np.zeros(n)
        self.done = np.zeros(n, bool)
        self.active = np.zeros(n, bool)
        self._bad = [(self.data.warning[int(getattr(mujoco.mjtWarning, name))], owners)
                     for name, owners in zip(BAD, (qpos_owner, dof_owner, dof_owner), strict=True)]

    def _joints(self, index: int):
        prefix = machine_prefix(index)
        for j in range(self.model.njnt):
            if self.model.joint(j).name.startswith(prefix):
                kind = int(self.model.jnt_type[j])
                yield j, {0: 7, 1: 4}.get(kind, 1)

    def machine_for(self, index: int, seed: int):
        return draw_machine(seed) if RUNGS[self.rung].machines else NOMINAL

    def _apply(self, index: int, ground: Ground, friction: float) -> None:
        self.model.body_pos[self._ring_body[index]] = self._ring_pos[index]
        for name, value in self._ground_start[index].items():
            getattr(self.model, name)[self._ground_rows[index]] = value
        apply_ground(self.model, index, ground)
        footing.park(self.model, index)
        set_friction(self.model, index, friction)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.model.geom_quat[ground_geoms(self.model, index)[0]])
        self.normal[index] = rot.reshape(3, 3)[:, 2]

    def _plane_frame(self, index: int, ground: Ground) -> tuple[np.ndarray, np.ndarray]:
        plane = ground_geoms(self.model, index)[0]
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, self.model.geom_quat[plane])
        origin = self.model.body_pos[self.model.body(machine_prefix(index) + ANCHOR).id]
        return origin + [0.0, 0.0, ground.level], rot.reshape(3, 3)

    def _surface(self, index: int, origin: np.ndarray, rot: np.ndarray):
        origin, rot = origin.copy(), rot.copy()
        return lambda x, y: float(footing.height_under(self.model, index, origin, rot,
                                                       [(x, y)])[0])

    def height_under(self, index: int, x: float, y: float) -> float:
        body = self._base_body[index]
        return self._surface(index, self.data.xpos[body],
                             self.data.xmat[body].reshape(3, 3))(x, y)

    def _start(self, task: Task) -> np.ndarray:
        return inverse_kinematics(*PropPath(task, self.pivot)(0.0))

    def _settled_even(self, index: int, x: float, y: float) -> bool:
        body = self._base_body[index]
        origin, rot = self.data.xpos[body], self.data.xmat[body].reshape(3, 3)
        stance = footing.height_under(self.model, index, origin, rot, footing.FOOTPRINT)
        press = footing.height_under(self.model, index, origin, rot, footing.PRESS + [x, y])
        return bool(np.abs(stance).max() <= footing.EVEN and np.ptp(press) <= footing.EVEN)

    def _stand(self, index: int, seed: int, machine, friction: float) -> tuple:
        side = float(BOOM_PIVOT[1])
        for attempt in range(PROP_DRAWS):
            try:
                ground = draw_ground(seed, attempt)
            except RuntimeError:
                continue
            if (footing.slope_deg(ground) > LIMITS.bucket_ground_slope_deg
                    or not footing.even_stance(ground)):
                continue
            try:
                self._apply(index, ground, friction)
                surface = self._surface(index, *self._plane_frame(index, ground))
                task = self._draw(seed, surface)
                if not footing.flat_press(ground, task.prop_x, side):
                    continue
                self.layer.reset(index, machine, self._start(task))
            except (ValueError, RuntimeError):
                continue
            height = self.height_under(index, task.prop_x, side)
            if (self._settled_even(index, task.prop_x, side)
                    and abs(height - surface(task.prop_x, side)) <= HEIGHT_TOLERANCE
                    and reachable(PropPath(task, self.pivot, height), self.period)):
                return ground, task, height
        raise RuntimeError(f"no ground drawn from seed {seed} gives a propping task")

    def _write(self, index: int, seed: int, task: Task | None = None) -> None:
        machine = self.machine_for(index, seed)
        if RUNGS[self.rung].ground:
            if task is not None:
                raise ValueError("a given task runs only on level ground: the nominal and "
                                 "machines rungs")
            friction = draw_friction(seed)
            ground, task, height = self._stand(index, seed, machine, friction)
        else:
            friction, ground, height = STEEL_FRICTION[0], LEVEL, 0.0
            self._apply(index, ground, friction)
            if task is None:
                task = self._draw(seed, lambda x, y: 0.0)
            self.layer.reset(index, machine, self._start(task))
        self.seeds[index], self.tasks[index], self.machines[index] = seed, task, machine
        self.grounds[index], self.frictions[index], self.heights[index] = ground, friction, height
        self.controllers[index] = self._controller(task)

    def _draw(self, seed: int, surface) -> Task:
        return draw_task(seed, self.settings, self.pivot, surface, self.period)

    def _controller(self, task: Task):
        return BaseController(PropPath(task, self.pivot), CONTROL_HZ)

    def reset(self, index: int, seed: int, task: Task | None = None) -> np.ndarray:
        if task is None:
            self._write(index, seed)
        else:
            self._write(index, seed, task)
        mujoco.mj_forward(self.model, self.data)
        rng = self._imu_rngs[index] = np.random.default_rng((seed, 11))
        on = self.imu_on[index] = float(RUNGS[self.rung].machines)
        self.gyro_bias[index] = rng.uniform(-1.0, 1.0, 3) * GYRO_BIAS * on
        self.accel_bias[index] = rng.uniform(-1.0, 1.0, 3) * ACCEL_BIAS * on
        quat = self.data.qpos[self._base_q[index] + 3:self._base_q[index] + 7]
        rot = _matrices(quat[None])[0]
        up = rot[2]
        self.stance[index] = rot[:, 2]
        swing = self.data.qpos[self._swing_q[index]]
        self.up[index] = _turn(up[None], np.array([-swing]))[0]
        self.v_prev[index] = self._upper_velocity()[index]
        self.k[index] = 0
        self.steps_total[index] = 0
        self.sent[index] = 0.0
        self.tip_prev[index] = self.data.site_xpos[self._tips[index]]
        self.apart[index] = self._hop + 1
        self.front_prev[index] = self._contacts()[1][index]
        self.corner_prev[index] = self._approach(self._part_velocity())[index]
        self.touched[index] = False
        self.rest_load[index] = 0.0
        self.rest_count[index] = 0
        self.landing[index] = 0.0
        self.recent[index] = 0.0
        self.end_load[index] = 0.0
        self.end_steps[index] = 0
        self.in_band[index] = 0
        self.hold_steps[index] = 0
        self.err_total[index] = 0.0
        self.done[index] = False
        self.active[index] = True
        self._begin(index)
        sensed = self.layer.observe_mean()
        gyro = self.gyro_bias[index] + rng.uniform(-1.0, 1.0, 3) * GYRO_NOISE * on
        accel = (_matrices(quat[None])[0].T @ -self.gravity)
        accel = _turn(accel[None], np.array([-swing]))[0] + self.accel_bias[index]
        accel += rng.uniform(-1.0, 1.0, 3) * ACCEL_NOISE * on
        tilt = self._sensed_tilt(index, sensed.positions[index, SWING], rng)
        self.est0[index] = tilt
        self._sense(index, sensed, gyro)
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
                              tilt - self.est0[index], self.sent[index, :JOINTS], reference])
        return raw / SCALE

    def _upper_velocity(self) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        quat = q[self._base_q[:, None] + np.arange(3, 7)]
        rot = _matrices(quat)
        omega = rot @ v[self._base_v[:, None] + np.arange(3, 6)][..., None]
        arm = rot @ self._upper_at
        return v[self._base_v[:, None] + np.arange(3)] + _cross(omega[..., 0], arm)

    def _part_velocity(self, rot: np.ndarray | None = None) -> np.ndarray:
        q, v = self.data.qpos, self.data.qvel
        if rot is None:
            rot = _matrices(q[self._base_q[:, None] + np.arange(3, 7)])
        offsets = rot @ self._points.T
        omega = rot @ v[self._base_v[:, None] + np.arange(3, 6)][..., None]
        return v[self._base_v[:, None] + np.arange(3)][..., None] + np.cross(omega, offsets, axis=1)

    def _approach(self, velocity: np.ndarray) -> np.ndarray:
        return -np.einsum("nkp,nk->np", velocity, self.normal)

    def _failure(self, failed: np.ndarray, seen: bool) -> bool:
        for warning, owners in self._bad:
            if warning.number:
                if not seen:
                    where = int(warning.lastinfo)
                    owner = owners[where] if 0 <= where < len(owners) else -1
                    failed[owner if owner >= 0 else slice(None)] = True
                seen = True
                warning.number = 0
        return seen

    def tilt(self, index: int) -> tuple[float, float]:
        quat = self.data.qpos[self._base_q[index] + 3:self._base_q[index] + 7]
        up = self.stance[index] @ _matrices(quat[None])[0]
        return self.tasks[index].tilt(self.k[index] * self.period), float(_tilts(up[None])[0][0])

    def baseline(self) -> np.ndarray:
        sensed = self.layer.observe()
        out = np.zeros((self.n, self.actions))
        for i, controller in enumerate(self.controllers):
            if controller is not None and self.active[i]:
                out[i] = controller(self.k[i] * self.period, sensed.positions[i],
                                    sensed.pressures[i])
        return out

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, float)
        if actions.shape != (self.n, self.actions) or not np.all(np.isfinite(actions)):
            raise ValueError(f"actions must be finite with shape {(self.n, self.actions)}")
        if not self.active.all():
            raise ValueError("every machine must be reset before the scene steps")
        commands = np.clip(actions, -1.0, 1.0)
        if self.mode == "residual":
            commands = np.clip(self.baseline() + self.settings.scale * commands, -1.0, 1.0)
        self._send(commands)
        n, dt, data, s = self.n, self.dt, self.data, self.settings
        times = (self.k[:, None] * self.steps + np.arange(1, self.steps + 1)) * dt
        reference = np.array([[task.tilt(t) for t in row]
                              for task, row in zip(self.tasks, times, strict=True)])
        holds = np.array([task.hold_span for task in self.tasks])
        in_hold = (times >= holds[:, :1]) & (times < holds[:, 1:])
        resting = times > np.array([[task.length - task.final] for task in self.tasks])
        load_sum = np.zeros((n, 2))
        fresh = np.zeros(n)
        err_sum = np.zeros(n)
        roll_sum = np.zeros(n)
        worst_tilt = np.zeros(n)
        worst_roll = np.zeros(n)
        slide = np.zeros(n)
        forbidden = np.zeros(n, bool)
        struck = np.zeros(n, bool)
        failed = np.zeros(n, bool)
        reset_seen = False
        for warning, _ in self._bad:
            warning.number = 0
        gyro_sum = np.zeros((n, 3))
        accel_sum = np.zeros((n, 3))
        gain = self.filter_gain
        base_q, base_v = self._base_q, self._base_v
        touched = self.touched.copy()
        rot = _matrices(data.qpos[base_q[:, None] + np.arange(3, 7)])
        pitch = _tilts(np.einsum("nk,nkj->nj", self.stance, rot))[0]
        bucket, front = self.apart == 0, self.front_prev.copy()
        motion = self._part_velocity(rot)
        self._prepare(times, commands)
        done_steps = self.steps
        for step in range(self.steps):
            self.layer.step()
            reset_seen = self._failure(failed, reset_seen)
            if reset_seen:
                done_steps = step
                break
            q, v = data.qpos, data.qvel
            rot = _matrices(q[base_q[:, None] + np.arange(3, 7)])
            pitch, roll = _tilts(np.einsum("nk,nkj->nj", self.stance, rot))
            err = np.abs(pitch - reference[:, step])
            err_sum += err
            roll_sum += np.abs(roll)
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
            bucket, front, bad, strike = self._contacts()
            forbidden |= bad
            struck |= strike
            tip = data.site_xpos[self._tips]
            self.apart = np.where(bucket, 0, self.apart + 1)
            near = self.apart <= self._hop
            slide += np.where(near, np.linalg.norm((tip - self.tip_prev)[:, :2], axis=1), 0.0)
            self.tip_prev = tip.copy()
            touched |= bucket
            landed = front & ~self.front_prev
            if landed.any():
                fresh = np.maximum(fresh, np.where(landed, self.corner_prev, 0.0).max(1))
            self.front_prev = front
            if resting[:, step].any():
                load_sum += np.where(resting[:, step, None], self._front_load(), 0.0)
            motion = self._part_velocity(rot)
            self.corner_prev = self._approach(motion)
            self._watch(step, rot, pitch, roll)
        self.landing = np.maximum(self.landing, fresh)
        self.recent[:, :-1] = self.recent[:, 1:]
        self.recent[:, -1] = fresh
        aloft = ~touched & (not reset_seen)
        if aloft.any():
            load = self._front_load()
            self.rest_load[aloft] += load[aloft]
        self.rest_count += aloft
        self.touched = touched
        counted = resting[:, :done_steps].sum(1)
        settling = counted > 0
        self.end_load[settling, :-1] = self.end_load[settling, 1:]
        self.end_steps[settling, :-1] = self.end_steps[settling, 1:]
        self.end_load[settling, -1] = load_sum[settling]
        self.end_steps[settling, -1] = counted[settling]
        rest = self.rest_load / np.maximum(self.rest_count, 1)[:, None]
        held = self.end_load.sum(1) / np.maximum(self.end_steps.sum(1), 1)[:, None]
        carried = np.min(np.divide(held, rest, out=np.zeros_like(rest), where=rest > 0.0), axis=1)
        down = front.reshape(n, 2, self._per_track).any(2).all(1)
        clear = ~bucket & down
        shortfall = np.where(settling, np.where(clear, 1.0 - np.minimum(carried / s.carry, 1.0),
                                                1.0), 0.0)
        self.k += 1
        self.steps_total += done_steps
        self.err_total += err_sum
        sensed = self.layer.observe_mean()
        mean_err = err_sum / max(done_steps, 1)
        change = np.abs(commands - self.sent).mean(1) / 2.0
        self.sent = commands
        earned = reward(s, self._error(mean_err, done_steps), change,
                        roll_sum / max(done_steps, 1), slide,
                        self.recent.max(1),
                        shortfall)
        for i in range(n):
            failed[i] |= not (np.all(np.isfinite(data.qpos[self._own_q[i]]))
                              and np.all(np.isfinite(self.layer._pressure[i])))
        cut = reset_seen & ~failed
        causes: list[str | None] = []
        extra = self._ends()
        for i in range(n):
            flags = {"unstable": failed[i], "contact": forbidden[i], "strike": struck[i],
                     "tilt": worst_tilt[i] > s.tilt_limit, "roll": worst_roll[i] > s.roll_limit,
                     **{name: flag[i] for name, flag in extra.items()}}
            causes.append(None if cut[i] else next((c for c in self.causes if flags[c]), None))
        obs = np.zeros((n, HISTORY * self.frame))
        terminated = np.zeros(n, bool)
        infos: list[dict] = [{} for _ in range(n)]
        for i in range(n):
            rng = self._imu_rngs[i]
            on = self.imu_on[i]
            gyro = gyro_sum[i] / self.steps + rng.uniform(-1.0, 1.0, 3) * GYRO_NOISE * on
            accel = accel_sum[i] / self.steps + rng.uniform(-1.0, 1.0, 3) * ACCEL_NOISE * on
            tilt = self._sensed_tilt(i, sensed.positions[i, SWING], rng)
            if not (cut[i] or failed[i]):
                self._sense(i, sensed, gyro)
                self.history[i, :-1] = self.history[i, 1:]
                self.history[i, -1] = self._frame(i, sensed, gyro, accel, tilt)
            obs[i] = self.history[i].ravel()
            end = self.k[i] * self.period >= self.tasks[i].length - 1e-9
            if causes[i] is not None or cut[i]:
                earned[i] = 0.0
            if causes[i] is not None or cut[i] or end:
                terminated[i] = True
                self.done[i] = True
                self.active[i] = False
                speed = float(np.linalg.norm(motion[i], axis=0).max())
                infos[i] = self.outcome(i, causes[i], pitch[i], float(carried[i]), speed,
                                        bool(bucket[i]), bool(down[i]))
                infos[i]["passed"] &= not cut[i]
                infos[i]["truncated"] = causes[i] is None
                infos[i]["cut_short"] = bool(cut[i])
        return obs, earned, terminated, infos

    def _begin(self, index: int) -> None:
        pass

    def _sense(self, index: int, sensed, gyro: np.ndarray) -> None:
        pass

    def _send(self, commands: np.ndarray) -> None:
        self.layer.act(commands)

    def _prepare(self, times: np.ndarray, commands: np.ndarray) -> None:
        pass

    def _watch(self, step: int, rot: np.ndarray, pitch: np.ndarray, roll: np.ndarray) -> None:
        pass

    def _error(self, tilt_error: np.ndarray, done_steps: int) -> np.ndarray:
        return tilt_error

    def _ends(self) -> dict[str, np.ndarray]:
        return {}

    def reasons(self, info: dict) -> list[str]:
        s = self.settings
        checks = ((info["cause"] is not None, f"early end: {info['cause']}"),
                  (info["hold_share"] < s.hold_share,
                   f"hold share in band under {s.hold_share:.2f}"),
                  (info["landing"] >= s.landing, f"a landing at {s.landing} m/s or faster"),
                  (info["bucket_down"], "bucket on the ground at the end"),
                  (not info["front_down"], "front of the tracks not set down"),
                  (info["front_carried"] < s.carry,
                   f"front carries under {s.carry:.0%} of its load"),
                  (info["front_speed"] >= s.still, f"front moving at {s.still} m/s or faster"))
        return [reason for failed, reason in checks if failed]

    def outcome(self, index: int, cause: str | None, final_tilt: float, carried: float,
                speed: float, bucket_down: bool, front_down: bool) -> dict:
        s = self.settings
        share = self.in_band[index] / max(int(self.hold_steps[index]), 1)
        passed = (cause is None and share >= s.hold_share and self.landing[index] < s.landing
                  and not bucket_down and front_down and carried >= s.carry and speed < s.still)
        return {"seed": self.seeds[index], "cause": cause, "passed": bool(passed),
                "hold_share": float(share), "landing": float(self.landing[index]),
                "bucket_down": bucket_down, "front_down": front_down,
                "final_tilt": float(final_tilt), "front_carried": carried, "front_speed": speed,
                "tilt_error": float(self.err_total[index] / max(int(self.steps_total[index]), 1)),
                "decisions": int(self.k[index])}

    def _front_load(self) -> np.ndarray:
        load = np.zeros((self.n, 2))
        count = self.data.ncon
        if not count:
            return load
        pairs = self.data.contact.geom[:count]
        role = self._role[pairs]
        hits = np.flatnonzero(((role[:, 0] == FRONT) & (role[:, 1] == GROUND))
                              | ((role[:, 1] == FRONT) & (role[:, 0] == GROUND)))
        force = np.zeros(6)
        for c in hits:
            g = pairs[c, 0] if role[c, 0] == FRONT else pairs[c, 1]
            mujoco.mj_contactForce(self.model, self.data, int(c), force)
            load[self._owner[g], self._corner[g] // self._per_track] += force[0]
        return load

    def _contacts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        n = self.n
        bucket, forbidden, strike = (np.zeros(n, bool) for _ in range(3))
        front = np.zeros((n, len(self._points)), bool)
        count = self.data.ncon
        if count:
            pairs = self.data.contact.geom[:count]
            a, b = pairs[:, 0], pairs[:, 1]
            on_ground = self._role[a] == GROUND
            other = np.where(on_ground, b, a)[on_ground | (self._role[b] == GROUND)]
            kind, owner = self._role[other], self._owner[other]
            bucket[owner[kind == BUCKET]] = True
            forbidden[owner[kind == FORBIDDEN]] = True
            feet = kind == FRONT
            front[owner[feet], self._corner[other[feet]]] = True
            part = self._part_of
            hit = ((part[a] == part[b]) & (part[a] >= 0)
                   & ((self._striker[a] & self._struck[b]) | (self._striker[b] & self._struck[a])))
            strike[part[a[hit]]] = True
        return bucket, front, forbidden, strike

    def snapshot(self) -> dict:
        if any(seed is None for seed in self.seeds):
            raise ValueError("every machine must be reset before a snapshot")
        state = np.zeros(mujoco.mj_stateSize(self.model, STATE))
        mujoco.mj_getState(self.model, self.data, state, STATE)
        return {
            "n": self.n, "rung": self.rung, "mode": self.mode, "settings": self.settings,
            "seeds": list(self.seeds), "physics": state, "layer": self.layer.snapshot(),
            "drawn": [list(x) for x in (self.tasks, self.machines, self.grounds, self.frictions,
                                        self.heights)],
            "episode": {name: np.copy(getattr(self, name)) for name in self.episode_arrays},
            "imu": [rng.bit_generator.state for rng in self._imu_rngs],
        }

    def restore(self, snapshot: dict) -> None:
        if (snapshot["n"], snapshot["rung"], snapshot["mode"]) != (self.n, self.rung, self.mode):
            raise ValueError("the snapshot is of a different scene")
        if snapshot["settings"] != self.settings:
            raise ValueError("the snapshot was taken with different settings")
        tasks, machines, grounds, frictions, heights = snapshot["drawn"]
        for index in range(self.n):
            self._apply(index, grounds[index], frictions[index])
            self.layer.machines[index] = machines[index]
            self.layer._apply_mass(index, machines[index])
            self.controllers[index] = self._controller(tasks[index])
        set_const(self.model, self.layer._scratch)
        self.seeds = list(snapshot["seeds"])
        self.tasks, self.machines, self.grounds = list(tasks), list(machines), list(grounds)
        self.frictions, self.heights = list(frictions), list(heights)
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

