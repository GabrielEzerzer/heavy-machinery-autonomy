"""The hydraulics layer: valve commands in [-1, 1] at the control rate move the spools, two pumps
and the oil in every chamber push the joints, and the true state becomes delayed, noisy sensor
readings. One layer serves every machine in a scene; machine i is named machine_prefix(i)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import mujoco
import numpy as np

from hma.hydraulics import circuit as c
from hma.hydraulics.params import CURVE_SEGMENTS, NOMINAL, RANGES, Machine
from hma.machine import BODY_NAMES, HOME_KEY, JOINT_NAMES, load_scene, machine_prefix
from hma.machine.model import COUNTERWEIGHT_MASS

CONTROL_HZ = 10.0
JOINTS = len(JOINT_NAMES)
NODES = c.PUMPS + 2 * JOINTS
SWING = JOINT_NAMES.index("swing")
COUNTERWEIGHT = "counterweight"
UPPER = BODY_NAMES.index("upper")
JOINT_SIZES = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}
BRAKE_PILOT = 0.02
RELIEF_BAND = 3.0 * c.RELIEF_KNEE
RELIEF_STEP = 0.1 * c.RELIEF_KNEE
CORRECTOR_PASSES = 6
SPEED_STEP = 1e-4
SWING_BLEND = 0.05 * c.SWING_TORQUE * c.SWING_SPEED
PRIORITY_FIRST = JOINT_NAMES.index(c.PRIORITY[0])
SHAPES = {k: np.shape(v) for k, v in vars(NOMINAL).items() if k != "seed"}


class Observation(NamedTuple):
    positions: np.ndarray
    velocities: np.ndarray
    pressures: np.ndarray


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
    a, b, c_ = model.geom_size[geom]
    local = COUNTERWEIGHT_MASS / 3.0 * np.diag([b * b + c_ * c_, a * a + c_ * c_, a * a + b * b])
    rot = _quat_matrix(model.geom_quat[geom])
    return _Solid(COUNTERWEIGHT_MASS, model.geom_pos[geom].copy(), rot @ local @ rot.T)


ORIFICES = slice(0, len(c.ORIFICES))
RELIEFS = slice(len(c.ORIFICES), len(c.ORIFICES) + len(c.RELIEFS))
LEAKS = slice(len(c.ORIFICES) + len(c.RELIEFS), len(c.INCIDENCE))
RELIEF_EDGES = np.arange(RELIEFS.start, RELIEFS.stop)
JOINT_INDEX = np.arange(JOINTS)
DROPS = c.INCIDENCE.T.copy()
NODE_INCIDENCE = c.INCIDENCE[:, :NODES].copy()
NODE_DROPS = NODE_INCIDENCE.T.copy()
INFLOW = -NODE_INCIDENCE
DISPLACEMENT = np.zeros((NODES, JOINTS))
DISPLACEMENT[c.PUMPS::2, :] = -np.diag(c.AREA[:, 0])
DISPLACEMENT[c.PUMPS + 1::2, :] = np.diag(c.AREA[:, 1])
DISPLACEMENT_T = DISPLACEMENT.T.copy()
DEMAND_MAP = np.concatenate([c.TABLES["demand"][0] - c.TABLES["demand"][1],
                             c.TABLES["demand"][1]])
SYSTEM_MAP = np.concatenate([
    np.einsum("ei,ej->eij", NODE_INCIDENCE, NODE_INCIDENCE).reshape(len(c.INCIDENCE), -1),
    np.eye(NODES * NODES)[::NODES + 1]])
MASK_EXTEND, MASK_RETRACT = c.DEMAND_MASK
PLAY_CLOSED = 1.0 - 1e-9
IDENTITY = np.eye(NODES)
VOLUME_SLOPE = np.stack([c.STROKE_AREA[:, 0], -c.STROKE_AREA[:, 1]], axis=-1)
VOLUME_CONSTANT = np.stack([-c.STROKE_AREA[:, 0] * c.STROKE[:, 0],
                            c.STROKE_AREA[:, 1] * c.STROKE[:, 1]], axis=-1)


def _softplus(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.logaddexp(0.0, x), 0.5 * (1.0 + np.tanh(0.5 * x))


def _one_way(gain: np.ndarray, drop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(drop, 0.0)
    shifted = x + c.ROOT_SMOOTHING
    chord = gain * x / (shifted * np.sqrt(shifted))
    return chord * x, chord


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

        def joint_field(field):
            return np.array([[getattr(model.joint(machine_prefix(i) + j), field)[0]
                              for j in JOINT_NAMES] for i in range(self.n)])

        self._qpos = joint_field("qposadr")
        self._qvel = joint_field("dofadr")
        self._actuators = np.array([[model.actuator(machine_prefix(i) + j).id
                                     for j in JOINT_NAMES] for i in range(self.n)])
        self._swing_dofs = self._qvel[:, SWING].copy()
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
        self._scratch = mujoco.MjData(model)
        reference_dofs = [reference.joint(machine_prefix(0) + j).dofadr[0] for j in JOINT_NAMES]
        self._base_armature = np.tile(reference.dof_armature[reference_dofs], (self.n, 1))
        self._base_friction = np.full(self.n, reference.dof_frictionloss[reference_dofs[SWING]])
        model.dof_armature[self._qvel] = self._base_armature
        model.dof_frictionloss[self._swing_dofs] = self._base_friction

        delay_cap = max([RANGES["delay"][1], *(float(np.max(m.delay)) for m in machines)])
        latency_cap = max([RANGES["sensor_latency"][1], *(m.sensor_latency for m in machines)])
        self._command_len = round(delay_cap / self.dt) + 1
        self._sensor_len = round(latency_cap / self.dt) + 1
        n, shape = self.n, (self.n, JOINTS)
        self._commands = np.zeros((self._command_len, *shape))
        self._command_flat = self._commands.reshape(-1)
        self._delay_index = np.zeros((self._command_len, *shape), np.intp)
        self._sensors = np.zeros((self._sensor_len, 4, *shape))
        self._command_head = 0
        self._sensor_head = 0
        self._held = np.zeros(shape)
        self._piloted = np.zeros(n, bool)
        self._spool = np.zeros(shape)
        self._segment_base = np.arange(n * JOINTS).reshape(shape) * CURVE_SEGMENTS

        self.machines = list(machines)
        self._alpha = np.ones(shape)
        self._deadband = np.zeros(shape)
        self._open_scale = np.ones(shape)
        self._intercept = np.zeros(n * JOINTS * CURVE_SEGMENTS)
        self._slope = np.zeros(n * JOINTS * CURVE_SEGMENTS)
        self._gain = np.ones(shape)
        self._spool_flow = np.ones(shape)
        self._pump_max = np.ones(n)
        self._power = np.ones(n)
        self._relief = np.ones((n, len(c.RELIEFS)))
        self._volume0 = np.ones((n, JOINTS, 2))
        self._pump_volume = np.ones(n)
        self._inv_bulk = np.ones(n)
        self._inv_bulk_dt = np.ones((n, 1))
        self._leakage = np.zeros((n, len(c.LEAKS)))
        self._joint = np.arange(JOINTS)
        self._valve = np.zeros((n, 2 * c.OPENINGS))
        self._valve[:, -1] = 1.0
        self._demand_in = np.zeros((n, 2 * JOINTS))
        self._nodes = np.full((n, NODES + 1), c.TANK_PRESSURE)
        self._edge_flow = np.zeros((n, len(c.INCIDENCE)))
        self._weights = np.ones((n, len(c.INCIDENCE) + NODES))
        self._edge_slope = self._weights[:, :len(c.INCIDENCE)]
        self._capacity = self._weights[:, len(c.INCIDENCE):]
        self._edge_slope[:] = 0.0
        self._brake = np.zeros(n)
        self._half_play = np.zeros(shape)
        self._latency_steps = np.zeros(n, np.intp)
        self._noise = np.zeros((4, n, 1))
        self._offset = np.zeros((n, JOINTS, 2))
        self._noise_draws = np.zeros((n, self.steps_per_decision, 4, JOINTS))
        self._noise_cursor = np.zeros(n, np.intp)
        self._machine_index = np.arange(n)
        self._rngs: list[np.random.Generator | None] = [None] * n

        self._pressure = np.zeros((n, NODES))
        self._pump = np.zeros((n, c.PUMPS))
        self._gap = np.zeros(shape)
        self._rod_speed = np.zeros(shape)
        self._accel = np.zeros(shape)
        self._cavity = np.zeros((n, NODES))
        self._held_nodes = np.zeros((n, NODES), bool)
        self._stiff = np.zeros(shape)
        self._torque_scale = np.ones(shape)
        self._neutral = np.zeros(n)
        self._rhs = np.zeros((n, NODES, 1 + JOINTS))
        self._rhs[:, :, 1:] = DISPLACEMENT
        self.pump_energy = np.zeros(n)
        self.lost_energy = np.zeros(n)
        self.work_energy = np.zeros(n)
        self.pump_power = np.zeros(n)
        for i, machine in enumerate(machines):
            self.reset(i, machine)

    def reset(self, index: int, machine: Machine, pose: Sequence[float] | None = None) -> None:
        if pose is not None:
            pose = np.asarray(pose, float)
            if pose.shape != (JOINTS,) or not np.all(np.isfinite(pose)):
                raise ValueError("pose must be four finite joint angles")
        for name, shape in SHAPES.items():
            if np.shape(getattr(machine, name)) != shape:
                raise ValueError(f"{name} must have shape {shape}")
        values = [np.asarray(v, float) for k, v in vars(machine).items() if k != "seed"]
        if not all(np.all(np.isfinite(v)) for v in values):
            raise ValueError("every machine parameter must be finite")
        if np.any(machine.delay < 0) or machine.sensor_latency < 0:
            raise ValueError("delay and sensor latency must not be negative")
        if np.max(machine.delay) > (self._command_len - 1) * self.dt + 1e-12:
            raise ValueError("delay longer than this layer's command buffer")
        if machine.sensor_latency > (self._sensor_len - 1) * self.dt + 1e-12:
            raise ValueError("sensor latency longer than this layer's sensor buffer")
        if np.any(machine.lag <= 0):
            raise ValueError("lag must be positive")
        if np.any(machine.deadband < 0) or np.any(machine.deadband >= 1):
            raise ValueError("deadband must be at least 0 and below 1")
        if np.any(machine.gain <= 0) or np.any(machine.spool_flow <= 0):
            raise ValueError("gain and spool flow must be positive")
        if machine.pump_flow <= 0 or machine.power <= 0:
            raise ValueError("pump flow and power must be positive")
        if machine.main_relief <= c.STANDBY_PRESSURE or np.any(
                machine.port_relief <= c.TANK_PRESSURE):
            raise ValueError("main relief must be above standby and port reliefs above tank")
        if machine.bulk_modulus <= 0 or machine.pump_volume <= 0 or np.any(
                machine.dead_volume <= 0):
            raise ValueError("bulk modulus and oil volumes must be positive")
        if np.any(machine.leakage < 0) or machine.brake < 0:
            raise ValueError("leakage and brake torque must not be negative")
        if np.any(machine.backlash < 0):
            raise ValueError("backlash must not be negative")
        if np.any(machine.link_mass <= 0) or machine.counterweight <= 0:
            raise ValueError("link mass and counterweight factors must be positive")
        if min(machine.position_noise, machine.velocity_noise, machine.pressure_noise) < 0:
            raise ValueError("sensor noise must not be negative")
        if np.any(np.diff(machine.curve, axis=1) < 0):
            raise ValueError("the valve curve must not fall as the spool opens")
        if np.any(machine.curve[:, 0] != 0.0) or np.any(machine.curve[:, -1] != 1.0):
            raise ValueError("the valve curve must run from 0 when shut to 1 when fully open")
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
        self._spool_flow[index] = machine.spool_flow
        self._pump_max[index] = c.PUMP_FLOW * machine.pump_flow
        self._power[index] = machine.power * c.PUMP_EFFICIENCY
        self._relief[index] = np.r_[machine.port_relief.ravel(),
                                    np.full(c.PUMPS, machine.main_relief)] - c.RELIEF_VENT
        self._volume0[index] = machine.dead_volume + VOLUME_CONSTANT
        self._pump_volume[index] = machine.pump_volume
        self._inv_bulk[index] = 1.0 / machine.bulk_modulus
        self._inv_bulk_dt[index] = 1.0 / (machine.bulk_modulus * self.dt)
        self._capacity[index, :c.PUMPS] = machine.pump_volume * self._inv_bulk_dt[index]
        self._leakage[index] = machine.leakage.ravel()
        self._brake[index] = machine.brake
        self._half_play[index] = machine.backlash / 2.0
        self._latency_steps[index] = round(machine.sensor_latency / self.dt)
        self._noise[:, index, 0] = (machine.position_noise, machine.velocity_noise,
                                    machine.pressure_noise, machine.pressure_noise)
        self._offset[index] = machine.pressure_offset
        self._rngs[index] = np.random.default_rng((machine.seed, 1))
        self._noise_cursor[index] = 0

        self._apply_mass(index, machine)
        mujoco.mj_setConst(self.model, self._scratch)
        self._reset_state(index, pose)

    def _draw_noise(self, index: int) -> None:
        shape = (self.steps_per_decision, 4, JOINTS)
        self._noise_draws[index] = self._rngs[index].uniform(-1.0, 1.0, shape)

    def _apply_mass(self, index: int, machine: Machine) -> None:
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
            com = solid.com + machine.com_shift[k]
            values, axes = np.linalg.eigh(solid.inertia)
            if np.linalg.det(axes) < 0:
                axes[:, 0] = -axes[:, 0]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, axes.ravel())
            self.model.body_mass[body] = solid.mass
            self.model.body_ipos[body] = com
            self.model.body_inertia[body] = values
            self.model.body_iquat[body] = quat

    def _reset_state(self, index: int, pose: Sequence[float] | None) -> None:
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
        self._commands[:, index] = 0.0
        self._held[index] = 0.0
        self._piloted[index] = False
        self._spool[index] = 0.0
        mujoco.mj_forward(model, data)
        _, lever = c.geometry(data.qpos[self._qpos[index]])
        force = data.qfrc_bias[self._qvel[index]] / lever
        force[SWING] = 0.0
        area = c.AREA
        head = np.maximum(force, 0.0) / area[:, 0] + c.TANK_PRESSURE * area[:, 1] / area[:, 0]
        rod = np.maximum(-force, 0.0) / area[:, 1] + c.TANK_PRESSURE * area[:, 0] / area[:, 1]
        chambers = np.where(force[:, None] >= 0.0, np.c_[head, np.full(JOINTS, c.TANK_PRESSURE)],
                            np.c_[np.full(JOINTS, c.TANK_PRESSURE), rod])
        chambers[SWING] = c.TANK_PRESSURE
        self._pressure[index, :c.PUMPS] = c.STANDBY_PRESSURE
        self._pressure[index, c.PUMPS:] = chambers.ravel()
        self._pump[index] = c.STANDBY_FLOW
        self._gap[index] = np.where(force >= 0.0, 1.0, -1.0)
        self._rod_speed[index] = 0.0
        self._accel[index] = 0.0
        self._cavity[index] = 0.0
        self._held_nodes[index] = False
        self._stiff[index] = 0.0
        self._torque_scale[index] = 1.0
        self._neutral[index] = c.BRAKE_DELAY
        self.pump_energy[index] = 0.0
        self.lost_energy[index] = 0.0
        self.work_energy[index] = 0.0
        self.pump_power[index] = 0.0
        first = self._true_rows()[:, index]
        first += self._rngs[index].uniform(-1.0, 1.0, first.shape) * self._noise[:, index]
        self._clip_pressure_rows(first)
        self._sensors[:, :, index] = first

    def act(self, commands: np.ndarray) -> None:
        commands = np.asarray(commands, float)
        if commands.shape != (self.n, JOINTS) or not np.all(np.isfinite(commands)):
            raise ValueError(f"commands must be finite with shape {(self.n, JOINTS)}")
        self._held = np.clip(commands, -1.0, 1.0)

    def _openings(self) -> np.ndarray:
        head = self._command_head = (self._command_head + 1) % self._command_len
        self._commands[head] = self._held
        spool = self._spool
        spool += self._alpha * (self._command_flat[self._delay_index[head]] - spool)
        opening = np.maximum(np.abs(spool) - self._deadband, 0.0)
        self._piloted = (opening * self._open_scale > BRAKE_PILOT * CURVE_SEGMENTS).any(1)
        opening *= self._open_scale
        segment = np.minimum(opening.astype(np.intp), CURVE_SEGMENTS - 1)
        segment += self._segment_base
        area = self._slope[segment] * opening
        area += self._intercept[segment]
        area *= self._gain
        np.minimum(area, 1.0, out=area)
        return area

    def step(self) -> None:
        n, dt, data, model = self.n, self.dt, self.data, self.model
        area = self._openings()
        extend = self._spool * c.JOINT_SIGN > 0.0
        wide = extend[..., None]
        qvel = self._qvel
        q = data.qpos[self._qpos]
        qd = data.qvel[qvel]
        length, lever = c.geometry(q)
        pressure = self._pressure

        self._demand_in[:, :JOINTS] = area * extend
        self._demand_in[:, JOINTS:] = area
        demand = self._demand_in @ DEMAND_MAP
        cap = self._power / np.maximum(pressure[:, :c.PUMPS].sum(1), 1.0)
        target = np.minimum(np.maximum(demand, c.STANDBY_FLOW),
                            np.minimum(self._pump_max, cap)[:, None])
        self._pump += (dt / c.PUMP_RESPONSE) * (target - self._pump)

        valve = self._valve
        opening = valve[:, c.OPENINGS:]
        opening[:, :JOINTS] = area * self._spool_flow
        opening[:, c.OPEN_BYPASS:c.OPEN_ALWAYS] = 1.0 - (
            np.where(wide, MASK_EXTEND, MASK_RETRACT) * area[..., None]).max(1)
        valve[:, :JOINTS] = opening[:, :JOINTS] * extend
        gain = valve @ c.GAIN_MAP
        gain[:, c.PRIORITY_EDGES] *= (1.0 - area[:, PRIORITY_FIRST])[:, None]
        self._nodes[:, :NODES] = pressure
        drop = self._nodes @ DROPS
        flow, slope = self._edge_flow, self._edge_slope
        flow[:, ORIFICES], slope[:, ORIFICES] = _one_way(gain, drop[:, ORIFICES])
        flow[:, RELIEFS], slope[:, RELIEFS] = self._relief_flow(drop[:, RELIEFS], self._relief)
        flow[:, LEAKS] = self._leakage * drop[:, LEAKS]
        slope[:, LEAKS] = self._leakage

        force_now = -(pressure @ DISPLACEMENT)
        motoring = np.tanh(force_now[:, SWING] * qd[:, SWING] * (1.0 / SWING_BLEND))
        scale = self._torque_scale
        scale[:, SWING] = c.SWING_MECHANICAL ** motoring
        flow_scale = c.SWING_VOLUMETRIC ** -motoring
        rhs = self._rhs
        rhs[:, :, 1 + SWING] = DISPLACEMENT[:, SWING] * flow_scale[:, None]
        rhs[:, :, 0] = flow @ INFLOW
        rhs[:, :c.PUMPS, 0] += self._pump
        volume = np.maximum(self._volume0 + VOLUME_SLOPE * length[..., None], 1e-6)
        self._capacity[:, c.PUMPS:] = volume.reshape(n, -1) * self._inv_bulk_dt
        system = (self._weights @ SYSTEM_MAP).reshape(n, NODES, NODES)
        solved = np.linalg.solve(system, rhs)

        play = self._half_play * np.abs(lever)
        per_play = np.divide(dt, play, out=np.zeros_like(play), where=play > 0.0)
        rod = lever * qd
        motion = (self._gap, play == 0.0, rod, lever * (qd + dt * self._accel),
                  self._rod_speed, per_play)
        predicted = self._engage(*self._forces(solved, force_now, scale), *motion)[0]
        rise = solved[..., 0] + (solved[..., 1:] @ predicted[..., None])[..., 0]
        self._held_nodes[:] = False
        start = drop[:, RELIEFS]
        reach = start + (rise @ NODE_DROPS)[:, RELIEFS]
        relief_moves = ((np.maximum(start, reach) > self._relief - RELIEF_BAND)
                        & (np.abs(reach - start) > RELIEF_STEP))
        fix = ((pressure + rise < c.CAVITATION_PRESSURE) | (self._cavity > 0.0)).any(1)
        fix |= relief_moves.any(1)
        if fix.any():
            rows = np.flatnonzero(fix)
            self._correct(rows, system, solved, start, rise, predicted, force_now[rows],
                          scale[rows], tuple(m[rows] for m in motion))

        held, coupling = self._forces(solved, force_now, scale)
        _, contact, closing, reach, free_speed, diag, others = self._engage(
            held, coupling, *motion)
        self._gap = np.where(closing, np.sign(reach), self._gap)

        stiff = self._stiff
        stiff[:] = np.where(contact, diag * lever * lever, 0.0)
        data.ctrl[self._actuators] = np.where(contact, lever * (held - others), 0.0) - stiff * qd
        self._neutral = np.where(self._piloted, 0.0, self._neutral + dt)
        model.dof_armature[qvel] = self._base_armature + stiff * dt
        model.dof_frictionloss[self._swing_dofs] = self._base_friction + np.where(
            self._neutral >= c.BRAKE_DELAY, self._brake, 0.0)
        mujoco.mj_step(model, data)
        model.dof_armature[qvel] = self._base_armature
        model.dof_frictionloss[self._swing_dofs] = self._base_friction

        self._accel = (data.qvel[qvel] - qd) * (1.0 / dt)
        joint_rod = lever * data.qvel[qvel]
        room = play * (1.0 / dt)
        free_speed = np.minimum(np.maximum(free_speed, joint_rod - room * (1.0 + self._gap)),
                                joint_rod + room * (1.0 - self._gap))
        speed = np.where(contact, joint_rod, free_speed)
        self._rod_speed = free_speed
        change = solved[..., 0] + (solved[..., 1:] @ speed[..., None])[..., 0]
        held_nodes = self._held_nodes
        cavity = self._cavity
        missing = np.zeros_like(cavity)
        tracked = (held_nodes | (cavity > 0.0)).any(1)
        if tracked.any():
            rows = np.flatnonzero(tracked)
            missing[rows] = self._balance(system[rows], self._rhs[rows], change[rows], speed[rows])
        wrong = (np.where(held_nodes, cavity + missing < 0.0,
                          pressure + change < c.CAVITATION_PRESSURE)).any(1)
        if wrong.any():
            rows = np.flatnonzero(wrong)
            sol, held_nodes[rows], missing[rows] = self._hold(
                rows, system[rows], held_nodes[rows], speed[rows])
            change[rows] = sol[..., 0] + (sol[..., 1:] @ speed[rows][..., None])[..., 0]
        missing *= held_nodes | (cavity > 0.0)
        pressure += change
        used = flow + slope * (change @ NODE_DROPS)
        lost = (used * (pressure @ NODE_DROPS)).sum(1) * dt
        lost -= (pressure * missing).sum(1)
        cavity[:] = np.where(held_nodes, np.maximum(cavity + missing, 0.0), 0.0)
        hydraulic = speed * (pressure @ DISPLACEMENT)
        hydraulic[:, SWING] *= flow_scale
        mechanical = speed * (pressure @ DISPLACEMENT) * scale
        self.lost_energy += lost + (mechanical - hydraulic).sum(1) * dt
        self.work_energy -= mechanical.sum(1) * dt
        gap = self._gap + np.where(contact, 0.0, (free_speed - joint_rod) * per_play)
        self._gap = np.minimum(np.maximum(gap, -1.0, out=gap), 1.0, out=gap)
        self.pump_power = (pressure[:, :c.PUMPS] * self._pump).sum(1)
        self.pump_energy += self.pump_power * dt

        self._sensor_head = (self._sensor_head + 1) % self._sensor_len
        reading = self._sensors[self._sensor_head]
        self._true_rows(reading)
        cursor = self._noise_cursor
        due = cursor == 0
        if due.any():
            for i in np.flatnonzero(due):
                self._draw_noise(i)
        reading += self._noise_draws[self._machine_index, cursor].transpose(1, 0, 2) * self._noise
        self._clip_pressure_rows(reading)
        cursor += 1
        cursor[cursor == self.steps_per_decision] = 0

    @staticmethod
    def _relief_flow(drop: np.ndarray, relief: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        knee, opened = _softplus((drop - relief) * (1.0 / c.RELIEF_KNEE))
        return knee * (c.RELIEF_SLOPE * c.RELIEF_KNEE), opened * c.RELIEF_SLOPE

    def _balance(self, matrix: np.ndarray, rhs: np.ndarray, delta: np.ndarray,
                 speed: np.ndarray) -> np.ndarray:
        balance = (matrix @ delta[..., None])[..., 0] - rhs[..., 0]
        return (balance - (rhs[..., 1:] @ speed[..., None])[..., 0]) * self.dt

    def _hold(self, rows: np.ndarray, matrix: np.ndarray, held: np.ndarray,
              speed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rhs = self._rhs[rows]
        pressure, cavity = self._pressure[rows], self._cavity[rows]
        floor = c.CAVITATION_PRESSURE - pressure
        for attempt in range(NODES):
            given = rhs.copy()
            given[..., 0] = np.where(held, floor, rhs[..., 0] - cavity / self.dt)
            given[..., 1:] *= ~held[..., None]
            solved = np.linalg.solve(np.where(held[..., None], IDENTITY, matrix), given)
            delta = solved[..., 0] + (solved[..., 1:] @ speed[..., None])[..., 0]
            missing = self._balance(matrix, rhs, delta, speed)
            now = np.where(held, cavity + missing > 0.0, delta < floor)
            if (now == held).all() or attempt == NODES - 1:
                break
            held = now
        return solved, held, missing * (held | (cavity > 0.0))

    @staticmethod
    def _forces(solved: np.ndarray, force_now: np.ndarray,
                scale: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        response = DISPLACEMENT_T @ solved
        response *= scale[..., None]
        return force_now * scale - response[..., 0], response[..., 1:]

    @staticmethod
    def _engage(held: np.ndarray, coupling: np.ndarray, gap: np.ndarray, rigid: np.ndarray,
                rod: np.ndarray, ahead: np.ndarray, loose: np.ndarray,
                per_play: np.ndarray) -> tuple[np.ndarray, ...]:
        force_next = held - (coupling @ rod[..., None])[..., 0]
        contact = rigid | ((np.abs(gap) >= PLAY_CLOSED) & (gap * force_next >= 0.0))
        guess = np.where(contact, ahead, loose)
        diag = coupling[:, JOINT_INDEX, JOINT_INDEX]
        others = (coupling @ guess[..., None])[..., 0] - diag * guess
        free = np.divide(held - others, diag, out=rod.copy(), where=diag != 0.0)
        reach = gap + (free - rod) * per_play
        closing = ~contact & (np.abs(reach) >= 1.0)
        contact |= closing
        return np.where(contact, ahead, free), contact, closing, reach, free, diag, others

    def _correct(self, rows: np.ndarray, system: np.ndarray, solved: np.ndarray,
                 start: np.ndarray, rise: np.ndarray, speed: np.ndarray, force_now: np.ndarray,
                 scale: np.ndarray, motion: tuple[np.ndarray, ...]) -> None:
        guess = speed[rows]
        relief = self._relief[rows]
        base = start[rows]
        shift = np.zeros_like(base)
        held = (self._pressure[rows] + rise[rows] < c.CAVITATION_PRESSURE) | (
            self._cavity[rows] > 0.0)
        matrix = system[rows]
        for attempt in range(CORRECTOR_PASSES):
            sol, held, _ = self._hold(rows, matrix, held, guess)
            moved = self._engage(*self._forces(sol, force_now, scale), *motion)[0]
            slid = np.abs(moved - guess) > SPEED_STEP
            guess = np.where(slid, moved, guess)
            delta = sol[..., 0] + (sol[..., 1:] @ guess[..., None])[..., 0]
            reach = base + (delta @ NODE_DROPS)[:, RELIEFS]
            moves = ((np.maximum(base + shift, reach) > relief - RELIEF_BAND)
                     & (np.abs(reach - base - shift) > RELIEF_STEP))
            if not (moves.any() or slid.any()) or attempt == CORRECTOR_PASSES - 1:
                break
            if not moves.any():
                continue
            shift = np.where(moves, reach - base, shift)
            level, tangent = self._relief_flow(base + shift, relief)
            self._edge_flow[rows[:, None], RELIEF_EDGES] = level - tangent * shift
            self._edge_slope[rows[:, None], RELIEF_EDGES] = tangent
            self._rhs[rows, :, 0] = self._edge_flow[rows] @ INFLOW
            self._rhs[rows, :c.PUMPS, 0] += self._pump[rows]
            matrix = (self._weights[rows] @ SYSTEM_MAP).reshape(len(rows), NODES, NODES)
        system[rows] = matrix
        solved[rows] = sol
        self._held_nodes[rows] = held

    def _true_rows(self, out: np.ndarray | None = None) -> np.ndarray:
        if out is None:
            out = np.empty((4, self.n, JOINTS))
        out[0] = self.data.qpos[self._qpos]
        out[1] = self.data.qvel[self._qvel]
        chambers = self._pressure[:, c.PUMPS:].reshape(self.n, JOINTS, 2)
        np.add(chambers, self._offset, out=out[2:].transpose(1, 2, 0))
        return out

    @staticmethod
    def _clip_pressure_rows(rows: np.ndarray) -> None:
        np.maximum(rows[2:], 0.0, out=rows[2:])

    def chamber_pressures(self) -> np.ndarray:
        return self._pressure[:, c.PUMPS:].reshape(self.n, JOINTS, 2).copy()

    def pump_pressures(self) -> np.ndarray:
        return self._pressure[:, :c.PUMPS].copy()

    def oil_energy(self) -> np.ndarray:
        q = self.data.qpos[self._qpos]
        length, _ = c.geometry(q)
        volume = self._volume0 + VOLUME_SLOPE * length[..., None]
        chambers = self._pressure[:, c.PUMPS:].reshape(self.n, JOINTS, 2)
        pumps = self._pressure[:, :c.PUMPS]
        stored = (chambers ** 2 * volume).sum(axis=(1, 2)) + (pumps ** 2).sum(1) * self._pump_volume
        return stored * self._inv_bulk / 2.0

    def true_state(self) -> Observation:
        q = self.data.qpos[self._qpos].copy()
        qd = self.data.qvel[self._qvel].copy()
        return Observation(q, qd, self.chamber_pressures())

    def observe(self) -> Observation:
        sample = self._sensors[(self._sensor_head - self._latency_steps) % self._sensor_len,
                               :, np.arange(self.n)]
        rows = np.moveaxis(sample, 1, 0)
        return Observation(rows[0], rows[1], np.stack([rows[2], rows[3]], axis=-1))
