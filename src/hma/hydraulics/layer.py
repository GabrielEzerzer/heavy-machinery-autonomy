"""The hydraulics layer: valve commands in [-1, 1] at the control rate become joint speed commands
at the physics rate, and the true joint state becomes delayed, noisy sensor readings. One layer
serves every machine in a scene; machine i is the one named with machine_prefix(i)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import mujoco
import numpy as np

from hma.hydraulics.params import (
    CURVE_SEGMENTS,
    NOMINAL_SPEEDS,
    NOMINAL_SUPPLY,
    PUMP_RESPONSE,
    RANGES,
    Machine,
)
from hma.machine import (
    BODY_NAMES,
    HOME_KEY,
    JOINT_NAMES,
    TRAVEL_NAMES,
    VALVE,
    load_scene,
    machine_prefix,
    tendon_friction,
)
from hma.machine.ground import settle, stand
from hma.machine.model import COUNTERWEIGHT_MASS
from hma.machine.tracks import SIDES, WHEELS, step, wheel_name

CONTROL_HZ = 10.0
JOINTS = len(JOINT_NAMES)
COUNTERWEIGHT = "counterweight"
BASE = BODY_NAMES.index("base")
UPPER = BODY_NAMES.index("upper")
JOINT_SIZES = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}


class Observation(NamedTuple):
    positions: np.ndarray
    velocities: np.ndarray
    loads: np.ndarray


class _Solid(NamedTuple):
    mass: float
    com: np.ndarray
    inertia: np.ndarray


def _quat_matrix(quat: np.ndarray) -> np.ndarray:
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, float))
    return mat.reshape(3, 3)


def _about(solid: _Solid, point: np.ndarray) -> np.ndarray:
    d = solid.com - point
    return solid.inertia + solid.mass * (d @ d * np.eye(3) - np.outer(d, d))


def _combine(a: _Solid, b: _Solid) -> _Solid:
    mass = a.mass + b.mass
    com = (a.mass * a.com + b.mass * b.com) / mass
    return _Solid(mass, com, _about(a, com) + _about(b, com))


def _scaled(solid: _Solid, factor: float) -> _Solid:
    return _Solid(solid.mass * factor, solid.com, solid.inertia * factor)


def _body_solid(model: mujoco.MjModel, body: int) -> _Solid:
    rot = _quat_matrix(model.body_iquat[body])
    inertia = rot @ np.diag(model.body_inertia[body]) @ rot.T
    return _Solid(float(model.body_mass[body]), model.body_ipos[body].copy(), inertia)


def _counterweight_solid(model: mujoco.MjModel) -> _Solid:
    geom = model.geom(machine_prefix(0) + COUNTERWEIGHT).id
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
        raise ValueError("the counterweight must be a box")
    a, b, c = model.geom_size[geom]
    local = COUNTERWEIGHT_MASS / 3.0 * np.diag([b * b + c * c, a * a + c * c, a * a + b * b])
    rot = _quat_matrix(model.geom_quat[geom])
    return _Solid(COUNTERWEIGHT_MASS, model.geom_pos[geom].copy(), rot @ local @ rot.T)


class Hydraulics:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, machines: Sequence[Machine],
                 control_hz: float = CONTROL_HZ):
        if len(machines) == 0:
            raise ValueError("at least one machine is needed")
        self.model, self.data = model, data
        self.n = len(machines)
        self.dt = float(model.opt.timestep)
        if not (np.isfinite(control_hz) and control_hz > 0):
            raise ValueError("control_hz must be a finite positive rate")
        steps = 1.0 / (control_hz * self.dt)
        if round(steps) < 1 or abs(steps - round(steps)) > 1e-9:
            raise ValueError(f"control_hz must divide the physics rate {1 / self.dt:g} Hz")
        self.steps_per_decision = round(steps)
        self.control_hz = control_hz

        def ids(kind, suffix=""):
            return np.array([[kind(machine_prefix(i) + j + suffix).id for j in JOINT_NAMES]
                             for i in range(self.n)])

        self._qpos = np.array([[model.joint(machine_prefix(i) + j).qposadr[0]
                                for j in JOINT_NAMES] for i in range(self.n)])
        self._qvel = np.array([[model.joint(machine_prefix(i) + j).dofadr[0]
                                for j in JOINT_NAMES] for i in range(self.n)])
        self._actuators = ids(model.actuator)
        self._tendons = ids(model.tendon, VALVE)
        self._bodies = np.array([[model.body(machine_prefix(i) + b).id for b in BODY_NAMES]
                                 for i in range(self.n)])
        self._own_joints = [[j for j in range(model.njnt)
                             if model.joint(j).name.startswith(machine_prefix(i))]
                            for i in range(self.n)]
        reference = load_scene(1, weld=True)
        nominal_ids = [reference.body(machine_prefix(0) + b).id for b in BODY_NAMES]
        self._nominal_bodies = [_body_solid(reference, b) for b in nominal_ids]
        self._nominal_body_fields = [
            (reference.body_mass[b].copy(), reference.body_ipos[b].copy(),
             reference.body_inertia[b].copy(), reference.body_iquat[b].copy())
            for b in nominal_ids
        ]
        self._counterweight = _counterweight_solid(reference)

        def wheels(m, i):
            return np.array([m.body(machine_prefix(i) + wheel_name(s, k)).id
                             for s in SIDES for k in range(WHEELS)])

        self._wheels = np.array([wheels(model, i) for i in range(self.n)])
        self._wheel_dofs = model.body_dofadr[self._wheels]
        nominal_wheels = wheels(reference, 0)
        self._wheel_mass = reference.body_mass[nominal_wheels].copy()
        self._wheel_inertia = reference.body_inertia[nominal_wheels].copy()
        self._belt_armature = reference.dof_armature[reference.body_dofadr[nominal_wheels]].copy()
        self._lower_ratio = (self._nominal_bodies[BASE].mass
                             + self._wheel_mass.sum()) / self._nominal_bodies[BASE].mass
        self._scratch = mujoco.MjData(model)

        delay_cap = max([RANGES["delay"][1], *(float(np.max(m.delay)) for m in machines)])
        latency_cap = max([RANGES["sensor_latency"][1], *(m.sensor_latency for m in machines)])
        self._command_len = round(delay_cap / self.dt) + 1
        self._sensor_len = round(latency_cap / self.dt) + 1
        shape = (self.n, JOINTS)
        self._commands = np.zeros((self._command_len, *shape))
        self._command_flat = self._commands.reshape(-1)
        self._delay_index = np.zeros((self._command_len, *shape), np.intp)
        self._sensors = np.zeros((self._sensor_len, 3, *shape))
        self._command_head = 0
        self._sensor_head = 0
        self._held = np.zeros(shape)
        self._spool = np.zeros(shape)
        self._play = np.zeros(shape)
        self._segment_base = np.arange(self.n * JOINTS).reshape(shape) * CURVE_SEGMENTS
        self._inv_nominal_speed = 1.0 / np.array([NOMINAL_SPEEDS[j] for j in JOINT_NAMES])

        self.machines = list(machines)
        self._alpha = np.ones(shape)
        self._deadband = np.zeros(shape)
        self._open_scale = np.ones(shape)
        self._intercept = np.zeros(self.n * JOINTS * CURVE_SEGMENTS)
        self._slope = np.zeros(self.n * JOINTS * CURVE_SEGMENTS)
        self._gain = np.ones(shape)
        self._speed_limit = np.ones(shape)
        self._force_limit = np.ones(shape)
        self._inv_force_limit = np.ones(shape)
        self._force = np.zeros(shape)
        self._ratio = np.ones(self.n)
        self._ratio_rate = self.dt / PUMP_RESPONSE
        self._power = np.ones(self.n)
        self._half_play = np.zeros(shape)
        self._flow = np.ones(self.n)
        self._latency_steps = np.zeros(self.n, np.intp)
        self._noise = np.zeros((3, self.n, 1))
        self._noise_draws = np.zeros((self.n, self.steps_per_decision, 3, JOINTS))
        self._noise_cursor = np.zeros(self.n, np.intp)
        self._machine_index = np.arange(self.n)
        self._rngs: list[np.random.Generator | None] = [None] * self.n
        for i, machine in enumerate(machines):
            self.reset(i, machine)

    def reset(self, index: int, machine: Machine, pose: Sequence[float] | None = None,
              at: Sequence[float] = (0.0, 0.0, 0.0), ground: int | None = None) -> None:
        at = tuple(float(v) for v in np.asarray(at, float).ravel())
        if len(at) != 3 or not all(np.isfinite(at)):
            raise ValueError("at must be a finite x, y and heading")
        if ground is not None and (int(ground) != ground or ground < 0):
            raise ValueError("ground must be a non-negative integer seed")
        if pose is not None:
            pose = np.asarray(pose, float)
            if pose.shape != (JOINTS,) or not np.all(np.isfinite(pose)):
                raise ValueError("pose must be four finite joint angles")
        values = [np.asarray(v, float) for k, v in vars(machine).items() if k != "seed"]
        if not all(np.all(np.isfinite(v)) for v in values):
            raise ValueError("every machine parameter must be finite")
        if np.any(machine.delay < 0) or machine.sensor_latency < 0:
            raise ValueError("delay and sensor latency must not be negative")
        if np.max(machine.delay) > (self._command_len - 1) * self.dt + 1e-12:
            raise ValueError("delay longer than this layer's command buffer")
        if machine.sensor_latency > (self._sensor_len - 1) * self.dt + 1e-12:
            raise ValueError("sensor latency longer than this layer's sensor buffer")
        if np.any(machine.lag <= 0) or machine.flow <= 0:
            raise ValueError("lag and flow must be positive")
        if np.any(machine.deadband < 0) or np.any(machine.deadband >= 1):
            raise ValueError("deadband must be at least 0 and below 1")
        if np.any(machine.gain <= 0) or np.any(machine.speed_limit <= 0):
            raise ValueError("gain and speed limits must be positive")
        if np.any(machine.backlash < 0):
            raise ValueError("backlash must not be negative")
        if np.any(machine.link_mass <= 0) or machine.counterweight <= 0:
            raise ValueError("link mass and counterweight factors must be positive")
        if min(machine.position_noise, machine.velocity_noise, machine.load_noise) < 0:
            raise ValueError("sensor noise must not be negative")
        if np.any(machine.speed_limit > self.model.actuator_ctrlrange[self._actuators[index], 1]):
            raise ValueError("a speed limit exceeds what the joint's actuator accepts")
        if np.any(machine.force_limit <= 0) or machine.power <= 0:
            raise ValueError("force limits and power must be positive")
        if machine.curve.shape != (JOINTS, CURVE_SEGMENTS + 1):
            raise ValueError(f"curve must have shape {(JOINTS, CURVE_SEGMENTS + 1)}")
        if np.any(np.diff(machine.curve, axis=1) < 0):
            raise ValueError("the valve curve must not fall as the spool opens")
        self.machines[index] = machine
        delay_steps = np.round(machine.delay / self.dt).astype(np.intp)
        heads = np.arange(self._command_len)[:, None]
        self._delay_index[:, index] = (((heads - delay_steps) % self._command_len) * self.n
                                       + index) * JOINTS + np.arange(JOINTS)
        self._alpha[index] = 1.0 - np.exp(-self.dt / machine.lag)
        self._deadband[index] = machine.deadband
        self._open_scale[index] = CURVE_SEGMENTS / (1.0 - machine.deadband)
        slope = np.diff(machine.curve, axis=1)
        segments = slice(index * JOINTS * CURVE_SEGMENTS, (index + 1) * JOINTS * CURVE_SEGMENTS)
        self._slope[segments] = slope.ravel()
        self._intercept[segments] = (machine.curve[:, :-1]
                                     - np.arange(CURVE_SEGMENTS) * slope).ravel()
        self._gain[index] = machine.gain
        self._speed_limit[index] = machine.speed_limit
        self._force_limit[index] = machine.force_limit
        self._inv_force_limit[index] = 1.0 / machine.force_limit
        self._power[index] = machine.power
        self._half_play[index] = machine.backlash / 2.0
        self._flow[index] = machine.flow * NOMINAL_SUPPLY
        self._latency_steps[index] = round(machine.sensor_latency / self.dt)
        self._noise[:, index, 0] = (machine.position_noise, machine.velocity_noise,
                                    machine.load_noise)
        self._rngs[index] = np.random.default_rng((machine.seed, 1))
        self._noise_cursor[index] = 0

        self.model.tendon_frictionloss[self._tendons[index]] = machine.force_limit
        self._apply_mass(index, machine)
        mujoco.mj_setConst(self.model, self._scratch)
        self._reset_state(index, pose, at, ground)

    def _draw_noise(self, index: int) -> None:
        shape = (self.steps_per_decision, 3, JOINTS)
        self._noise_draws[index] = self._rngs[index].uniform(-1.0, 1.0, shape)

    def _apply_mass(self, index: int, machine: Machine) -> None:
        lower = machine.link_mass[BASE]
        self.model.body_mass[self._wheels[index]] = self._wheel_mass * lower
        self.model.body_inertia[self._wheels[index]] = self._wheel_inertia * lower
        self.model.dof_armature[self._wheel_dofs[index]] = self._belt_armature * lower
        for k, body in enumerate(self._bodies[index]):
            body = int(body)
            nominal = self._nominal_bodies[k]
            cw_factor = machine.counterweight if k == UPPER else 1.0
            if machine.link_mass[k] == 1.0 and cw_factor == 1.0 and not machine.com_shift[k].any():
                mass, ipos, inertia, iquat = self._nominal_body_fields[k]
                self.model.body_mass[body] = mass
                self.model.body_ipos[body] = ipos
                self.model.body_inertia[body] = inertia
                self.model.body_iquat[body] = iquat
                continue
            solid = _scaled(nominal, machine.link_mass[k])
            if k == UPPER:
                cw = self._counterweight
                rest = _combine(nominal, _scaled(cw, -1.0))
                solid = _combine(_scaled(rest, machine.link_mass[k]), _scaled(cw, cw_factor))
            com = solid.com + machine.com_shift[k] * (self._lower_ratio if k == BASE else 1.0)
            values, axes = np.linalg.eigh(solid.inertia)
            if np.linalg.det(axes) < 0:
                axes[:, 0] = -axes[:, 0]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, axes.ravel())
            self.model.body_mass[body] = solid.mass
            self.model.body_ipos[body] = com
            self.model.body_inertia[body] = values
            self.model.body_iquat[body] = quat

    def _reset_state(self, index: int, pose: Sequence[float] | None,
                     at: tuple[float, float, float], ground: int | None) -> None:
        model, data = self.model, self.data
        key = model.key(HOME_KEY).id
        for joint in self._own_joints[index]:
            q0 = model.jnt_qposadr[joint]
            v0 = model.jnt_dofadr[joint]
            nq, nv = JOINT_SIZES.get(int(model.jnt_type[joint]), (1, 1))
            data.qpos[q0:q0 + nq] = model.key_qpos[key, q0:q0 + nq]
            data.qvel[v0:v0 + nv] = 0.0
            data.qacc_warmstart[v0:v0 + nv] = 0.0
            data.qfrc_applied[v0:v0 + nv] = 0.0
        data.xfrc_applied[self._bodies[index]] = 0.0
        if pose is not None:
            data.qpos[self._qpos[index]] = pose
        data.ctrl[self._actuators[index]] = 0.0
        data.ctrl[[model.actuator(machine_prefix(index) + t).id for t in TRAVEL_NAMES]] = 0.0
        if ground is not None:
            stand(model, data, index, int(ground), *at)
        elif not settle(model, data, index, *at):
            raise RuntimeError("the machine does not come to rest on this ground at this place")
        self._commands[:, index] = 0.0
        self._held[index] = 0.0
        self._spool[index] = 0.0
        self._play[index] = 0.0
        mujoco.mj_forward(model, data)
        first = np.stack(self.true_state())[:, index]
        self._force[index] = first[2] * self._force_limit[index]
        self._ratio[index] = 1.0
        first += self._rngs[index].uniform(-1.0, 1.0, first.shape) * self._noise[:, index]
        self._sensors[:, :, index] = first

    def act(self, commands: np.ndarray) -> None:
        commands = np.asarray(commands, float)
        if commands.shape != (self.n, JOINTS) or not np.all(np.isfinite(commands)):
            raise ValueError(f"commands must be finite with shape {(self.n, JOINTS)}")
        self._held = np.clip(commands, -1.0, 1.0)

    def joint_speed_commands(self) -> np.ndarray:
        head = self._command_head = (self._command_head + 1) % self._command_len
        self._commands[head] = self._held
        spool = self._spool
        spool += self._alpha * (self._command_flat[self._delay_index[head]] - spool)
        opening = np.maximum(np.abs(spool) - self._deadband, 0.0)
        opening *= self._open_scale
        segment = np.minimum(opening.astype(np.intp), CURVE_SEGMENTS - 1)
        segment += self._segment_base
        speed = self._slope[segment] * opening
        speed += self._intercept[segment]
        speed *= self._gain
        np.minimum(speed, 1.0, out=speed)
        speed *= self._speed_limit
        demand = (speed * self._inv_nominal_speed).sum(axis=1)
        speed *= np.minimum(1.0, self._flow / np.maximum(demand, 1e-12))[:, None]
        np.copysign(speed, spool, out=speed)
        power = np.maximum(self._force * speed, 0.0).sum(axis=1)
        error = 1.0 - self._ratio * power / self._power
        self._ratio = np.minimum(self._ratio + self._ratio_rate * error, 1.0)
        speed *= self._ratio[:, None]
        drive = speed * self.dt
        drive += self._play
        self._play = np.minimum(np.maximum(drive, -self._half_play), self._half_play)
        drive -= self._play
        drive /= self.dt
        return drive

    def step(self) -> None:
        self.data.ctrl[self._actuators] = self.joint_speed_commands()
        step(self.model, self.data)
        self._sensor_head = (self._sensor_head + 1) % self._sensor_len
        reading = self._sensors[self._sensor_head]
        self._read_into(reading)
        np.multiply(reading[2], self._force_limit, out=self._force)
        cursor = self._noise_cursor
        for i in np.flatnonzero(cursor == 0):
            self._draw_noise(i)
        reading += self._noise_draws[self._machine_index, cursor].transpose(1, 0, 2) * self._noise
        cursor += 1
        cursor[cursor == self.steps_per_decision] = 0

    def _read_into(self, out: np.ndarray) -> None:
        data = self.data
        out[0] = data.qpos[self._qpos]
        out[1] = data.qvel[self._qvel]
        out[2] = tendon_friction(data, self._tendons) * self._inv_force_limit

    def true_state(self) -> Observation:
        out = np.empty((3, self.n, JOINTS))
        self._read_into(out)
        return Observation(*out)

    def observe(self) -> Observation:
        sample = self._sensors[(self._sensor_head - self._latency_steps) % self._sensor_len,
                               :, np.arange(self.n)]
        return Observation(*np.moveaxis(sample, 1, 0))
