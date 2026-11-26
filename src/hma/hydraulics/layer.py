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
from hma.machine import (
    BODY_NAMES,
    HOME_KEY,
    JOINT_NAMES,
    TRAVEL_NAMES,
    WELD,
    load_scene,
    machine_prefix,
)
from hma.machine.cylinders import MAIN_RELIEF, seal_friction
from hma.machine.ground import settle, stand
from hma.machine.model import COUNTERWEIGHT_MASS
from hma.machine.tracks import SIDES, WHEELS, wheel_name
from hma.machine.tracks import step as track_step

CONTROL_HZ = 10.0
JOINTS = len(JOINT_NAMES)
NODES = c.PUMPS + 2 * JOINTS
SWING = JOINT_NAMES.index("swing")
COUNTERWEIGHT = "counterweight"
BASE = BODY_NAMES.index("base")
UPPER = BODY_NAMES.index("upper")
JOINT_SIZES = {int(mujoco.mjtJoint.mjJNT_FREE): (7, 6), int(mujoco.mjtJoint.mjJNT_BALL): (4, 3)}
BRAKE_PILOT = 0.02
RELIEF_BAND = 3.0 * c.RELIEF_KNEE
RELIEF_STEP = 0.1 * c.RELIEF_KNEE
RELIEF_MISS = 0.01
CORRECTOR_PASSES = 6
CHECK_PASSES = 12
REDO_PASSES = 3
REDO_SHARE = 0.005
STATE = mujoco.mjtState.mjSTATE_INTEGRATION
REDO_FORCE = REDO_SHARE * c.AREA.max(1) * MAIN_RELIEF
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
CHECKS = np.array([dst != c.TANK for _, dst, *_ in c.ORIFICES])
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


def _settle(cond: np.ndarray, opened: np.ndarray, gain: np.ndarray, end: np.ndarray,
            bound: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    back = CHECKS & (cond > 0.0) & (end < 0.0)
    blocked = CHECKS & (cond == 0.0) & (gain > 0.0) & (end > 0.0) & (not bound)
    halve = back & opened & (not bound)
    cond = np.where(back, np.where(halve, 0.5 * cond, 0.0),
                    np.where(blocked, _one_way(gain, end)[1], cond))
    return cond, (opened | blocked) & ~(back & ~halve), back | blocked


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
        self._bodies = np.array([[model.body(machine_prefix(i) + b).id for b in BODY_NAMES]
                                 for i in range(self.n)])
        self._own_joints = [[j for j in range(model.njnt)
                             if model.joint(j).name.startswith(machine_prefix(i))]
                            for i in range(self.n)]
        self._own_bodies = [[b for b in range(model.nbody)
                             if model.body(b).name.startswith(machine_prefix(i))]
                            for i in range(self.n)]
        self._own_equalities = [[e for e in range(model.neq)
                                 if model.equality(e).name.startswith(machine_prefix(i))
                                 and model.equality(e).name != machine_prefix(i) + WELD]
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
        self._travel = np.array([[model.actuator(machine_prefix(i) + name).id
                                  for name in TRAVEL_NAMES] for i in range(self.n)])
        nominal_wheels = wheels(reference, 0)
        self._wheel_mass = reference.body_mass[nominal_wheels].copy()
        self._wheel_inertia = reference.body_inertia[nominal_wheels].copy()
        self._belt_armature = reference.dof_armature[reference.body_dofadr[nominal_wheels]].copy()
        self._lower_ratio = (self._nominal_bodies[BASE].mass
                             + self._wheel_mass.sum()) / self._nominal_bodies[BASE].mass
        self._scratch = mujoco.MjData(model)
        reference_dofs = [reference.joint(machine_prefix(0) + j).dofadr[0] for j in JOINT_NAMES]
        self._base_armature = np.tile(reference.dof_armature[reference_dofs], (self.n, 1))
        self._base_frictionloss = np.tile(reference.dof_frictionloss[reference_dofs], (self.n, 1))
        self._base_friction = self._base_frictionloss[:, SWING].copy()
        model.dof_armature[self._qvel] = self._base_armature
        model.dof_frictionloss[self._qvel] = self._base_frictionloss

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
        self._seal_scale = np.zeros(n)
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
        self._piston_speed = np.zeros(shape)
        self._accel = np.zeros(shape)
        self._cavity = np.zeros((n, NODES))
        self._held_nodes = np.zeros((n, NODES), bool)
        self._stiff = np.zeros(shape)
        self._torque_scale = np.ones(shape)
        self._neutral = np.zeros(n)
        self._rhs = np.zeros((n, NODES, 1 + JOINTS))
        self._rhs[:, :, 1:] = DISPLACEMENT
        self._state = np.zeros(mujoco.mj_stateSize(model, STATE))
        self.redos = 0
        self.capped = 0
        self.check_bound = 0
        self.pump_energy = np.zeros(n)
        self.lost_energy = np.zeros(n)
        self.work_energy = np.zeros(n)
        self.pump_power = np.zeros(n)
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
        if machine.seal_friction < 0:
            raise ValueError("the seal friction factor must not be negative")
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
        self._seal_scale[index] = machine.seal_friction
        self._latency_steps[index] = round(machine.sensor_latency / self.dt)
        self._noise[:, index, 0] = (machine.position_noise, machine.velocity_noise,
                                    machine.pressure_noise, machine.pressure_noise)
        self._offset[index] = machine.pressure_offset
        self._rngs[index] = np.random.default_rng((machine.seed, 1))
        self._noise_cursor[index] = 0

        self._apply_mass(index, machine)
        mujoco.mj_setConst(self.model, self._scratch)
        self._reset_state(index, pose, at, ground)

    def _draw_noise(self, index: int) -> None:
        shape = (self.steps_per_decision, 4, JOINTS)
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
        data.xfrc_applied[self._own_bodies[index]] = 0.0
        own = self._own_equalities[index]
        data.eq_active[own] = model.eq_active0[own]
        if pose is not None:
            data.qpos[self._qpos[index]] = pose
        data.ctrl[self._actuators[index]] = 0.0
        data.ctrl[self._travel[index]] = 0.0
        if ground is not None:
            stand(model, data, index, int(ground), *at)
        elif not settle(model, data, index, *at):
            raise RuntimeError("the machine does not come to rest on this ground at this place")
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
        self._piston_speed[index] = 0.0
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
        check_start = drop[:, ORIFICES].copy()
        flow[:, ORIFICES], slope[:, ORIFICES] = _one_way(gain, check_start)
        self._orifice_gain = gain
        cond_start = slope[:, ORIFICES].copy()
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
        rigid = play == 0.0
        relief_start = drop[:, RELIEFS]
        saved = (self._edge_flow.copy(), self._weights.copy(), self._rhs.copy(), system.copy(),
                 solved.copy(), self._gap.copy())
        mujoco.mj_getState(model, data, self._state, STATE)
        self._neutral = np.where(self._piloted, 0.0, self._neutral + dt)
        brake = self._base_friction + np.where(self._neutral >= c.BRAKE_DELAY, self._brake, 0.0)
        chambers = pressure[:, c.PUMPS:].reshape(n, JOINTS, 2)
        drag = np.zeros((n, JOINTS))
        drag[:, 1:] = seal_friction(chambers[:, 1:, 0], chambers[:, 1:, 1],
                                    self._piston_speed[:, 1:]) * self._seal_scale[:, None]
        seal = drag * np.abs(lever)
        redo = np.zeros(n, bool)
        redo_held = np.zeros((n, NODES), bool)
        redo_speed = np.zeros((n, JOINTS))
        redo_cond = cond_start.copy()
        best = (np.full(n, np.inf), redo.copy(), redo_held.copy(), redo_speed.copy(),
                np.zeros(n, int), redo_cond.copy())
        cavity = self._cavity
        for attempt in range(REDO_PASSES + 2):
            cond = np.where(redo[:, None], redo_cond, cond_start)
            shift = np.zeros_like(relief_start)
            if redo.any():
                rows = np.flatnonzero(redo)
                system[rows] = self._set_checks(rows, cond[rows], check_start[rows])
            ahead = np.where(redo[:, None], redo_speed, lever * (qd + dt * self._accel))
            loose = np.where(redo[:, None], redo_speed, self._rod_speed)
            motion = (self._gap, rigid, rod, ahead, loose, per_play, drag)
            predicted = self._engage(*self._forces(solved, force_now, scale), *motion)[0]
            predicted = np.where(redo[:, None], redo_speed, predicted)
            rise = solved[..., 0] + (solved[..., 1:] @ predicted[..., None])[..., 0]
            self._held_nodes[:] = False
            reach = relief_start + (rise @ NODE_DROPS)[:, RELIEFS]
            relief_moves = self._relief_moves(slice(None), relief_start, reach)
            start_held = np.where(redo[:, None], redo_held,
                                  (pressure + rise < c.CAVITATION_PRESSURE) | (cavity > 0.0))
            flips = _settle(cond, np.zeros(cond.shape, bool), gain,
                            check_start + (rise @ NODE_DROPS)[:, ORIFICES])[2]
            fix = start_held.any(1) | relief_moves.any(1) | redo | flips.any(1)
            if fix.any():
                rows = np.flatnonzero(fix)
                self._correct(rows, system, solved, relief_start, start_held[rows], predicted,
                              force_now[rows], scale[rows], tuple(m[rows] for m in motion),
                              redo[rows], (cond, gain, check_start), shift)

            held, coupling = self._forces(solved, force_now, scale)
            _, contact, closing, reach, free_speed, diag, others = self._engage(
                held, coupling, *motion)
            self._gap = np.where(closing, np.sign(reach), self._gap)

            stiff = self._stiff
            stiff[:] = np.where(contact, diag * lever * lever, 0.0)
            push = np.where(contact, lever * (held - others), 0.0)
            data.ctrl[self._actuators] = push - stiff * qd
            model.dof_armature[qvel] = self._base_armature + stiff * dt
            friction = self._base_frictionloss + np.where(contact, seal, 0.0)
            friction[:, SWING] = brake
            model.dof_frictionloss[qvel] = friction
            track_step(model, data)
            model.dof_armature[qvel] = self._base_armature
            model.dof_frictionloss[qvel] = self._base_frictionloss

            joint_rod = lever * data.qvel[qvel]
            room = play * (1.0 / dt)
            free_speed = np.minimum(np.maximum(free_speed, joint_rod - room * (1.0 + self._gap)),
                                    joint_rod + room * (1.0 - self._gap))
            speed = np.where(contact, joint_rod, free_speed)
            change = solved[..., 0] + (solved[..., 1:] @ speed[..., None])[..., 0]
            held_nodes = self._held_nodes.copy()
            missing = np.zeros_like(cavity)
            tracked = (held_nodes | (cavity > 0.0)).any(1)
            if tracked.any():
                rows = np.flatnonzero(tracked)
                missing[rows] = self._balance(system[rows], self._rhs[rows], change[rows],
                                              speed[rows])
            wrong = (np.where(held_nodes, cavity + missing < 0.0,
                              pressure + change < c.CAVITATION_PRESSURE)).any(1)
            if wrong.any():
                rows = np.flatnonzero(wrong)
                sol, held_nodes[rows], missing[rows] = self._hold(
                    rows, system[rows], held_nodes[rows], speed[rows])
                change[rows] = sol[..., 0] + (sol[..., 1:] @ speed[rows][..., None])[..., 0]
            opened = np.zeros(cond.shape, bool)
            bounded = np.zeros(n, bool)
            for passes in range(CHECK_PASSES + int(CHECKS.sum()) + CORRECTOR_PASSES):
                end = check_start + (change @ NODE_DROPS)[:, ORIFICES]
                bound = passes >= CHECK_PASSES
                if passes == CHECK_PASSES:
                    bounded = _settle(cond, opened, gain, end)[2].any(1)
                settled, now_opened, flips = _settle(cond, opened, gain, end, bound)
                reach = relief_start + (change @ NODE_DROPS)[:, RELIEFS]
                moves = self._relief_moves(slice(None), relief_start + shift, reach)
                flip = flips.any(1)
                moving = flip | moves.any(1)
                if not moving.any():
                    break
                rows = np.flatnonzero(moving)
                cond[rows], opened[rows] = settled[rows], now_opened[rows]
                shift[rows] = np.where(moves[rows], reach[rows] - relief_start[rows], shift[rows])
                self._set_reliefs(rows, relief_start[rows], shift[rows])
                if flip.any():
                    turn = np.flatnonzero(flip)
                    self._set_checks(turn, cond[turn], check_start[turn])
                system[rows] = self._refresh(rows)
                sol, held_nodes[rows], missing[rows] = self._hold(
                    rows, system[rows], held_nodes[rows], speed[rows])
                change[rows] = sol[..., 0] + (sol[..., 1:] @ speed[rows][..., None])[..., 0]
            applied = (data.ctrl[self._actuators] - stiff * (data.qvel[qvel] - qd)) / lever
            final = -((pressure + change) @ DISPLACEMENT) * scale
            miss = np.where(contact, np.abs(applied - final), 0.0)
            changed = ((held_nodes != self._held_nodes).any(1)
                       | (miss > REDO_FORCE).any(1))
            if not changed.any() or attempt > REDO_PASSES:
                break
            score = (miss / REDO_FORCE).max(1)
            better = score < best[0]
            for kept, now in zip(best, (score, redo, redo_held, redo_speed, attempt, redo_cond),
                                 strict=True):
                kept[better] = now[better] if np.ndim(now) else now
            if attempt == REDO_PASSES:
                self.capped += 1
                if (best[4] == attempt)[changed].all():
                    break
                for now, kept in zip((redo, redo_held, redo_speed, redo_cond),
                                     best[1:4] + best[5:], strict=True):
                    now[changed] = kept[changed]
            else:
                redo_held[changed] = held_nodes[changed]
                relaxed = np.where(redo[:, None], 0.5 * (speed + redo_speed), speed)
                redo_speed[changed] = relaxed[changed]
                redo_cond[changed] = cond[changed]
                redo |= changed
            self.redos += 1
            mujoco.mj_setState(model, data, self._state, STATE)
            self._edge_flow[:], self._weights[:], self._rhs[:] = saved[:3]
            system[:], solved[:] = saved[3:5]
            self._gap = saved[5].copy()

        self.check_bound += int(bounded.any())
        self._accel = (data.qvel[qvel] - qd) * (1.0 / dt)
        self._rod_speed = free_speed
        self._piston_speed = speed
        self._held_nodes[:] = held_nodes
        missing *= held_nodes | (cavity > 0.0)
        pressure += change
        used = flow + slope * (change @ NODE_DROPS)
        lost = (used * (pressure @ NODE_DROPS)).sum(1) * dt
        lost -= (pressure * missing).sum(1)
        cavity[:] = np.where(held_nodes, np.maximum(cavity + missing, 0.0), 0.0)
        hydraulic = speed * (pressure @ DISPLACEMENT)
        hydraulic[:, SWING] *= flow_scale
        mechanical = speed * (pressure @ DISPLACEMENT) * scale
        loose_rods = np.where(contact, 0.0, mechanical)
        self.lost_energy += lost + (mechanical - hydraulic - loose_rods).sum(1) * dt
        self.work_energy -= np.where(contact, mechanical, 0.0).sum(1) * dt
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

    def _set_checks(self, rows: np.ndarray, cond: np.ndarray, start: np.ndarray) -> np.ndarray:
        self._edge_flow[rows, ORIFICES] = cond * start
        self._edge_slope[rows, ORIFICES] = cond
        return self._refresh(rows)

    def _relief_moves(self, rows: np.ndarray | slice, point: np.ndarray,
                      reach: np.ndarray) -> np.ndarray:
        relief = self._relief[rows]
        level, tangent = self._relief_flow(point, relief)
        miss = np.abs(self._relief_flow(reach, relief)[0] - level - tangent * (reach - point))
        near = np.maximum(point, reach) > relief - RELIEF_BAND
        return ((near | (miss > RELIEF_MISS * self._pump_max[rows, None]))
                & (np.abs(reach - point) > RELIEF_STEP))

    def _set_reliefs(self, rows: np.ndarray, base: np.ndarray, shift: np.ndarray) -> None:
        level, tangent = self._relief_flow(base + shift, self._relief[rows])
        self._edge_flow[rows[:, None], RELIEF_EDGES] = level - tangent * shift
        self._edge_slope[rows[:, None], RELIEF_EDGES] = tangent

    def _refresh(self, rows: np.ndarray) -> np.ndarray:
        self._rhs[rows, :, 0] = self._edge_flow[rows] @ INFLOW
        self._rhs[rows, :c.PUMPS, 0] += self._pump[rows]
        return (self._weights[rows] @ SYSTEM_MAP).reshape(len(rows), NODES, NODES)

    def _balance(self, matrix: np.ndarray, rhs: np.ndarray, delta: np.ndarray,
                 speed: np.ndarray) -> np.ndarray:
        balance = (matrix @ delta[..., None])[..., 0] - rhs[..., 0]
        return (balance - (rhs[..., 1:] @ speed[..., None])[..., 0]) * self.dt

    def _hold(self, rows: np.ndarray, matrix: np.ndarray, held: np.ndarray,
              speed: np.ndarray, fixed: np.ndarray | bool = False
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            now = np.where(fixed, held, np.where(held, cavity + missing > 0.0, delta < floor))
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
                per_play: np.ndarray, drag: np.ndarray) -> tuple[np.ndarray, ...]:
        force_next = held - (coupling @ rod[..., None])[..., 0]
        contact = rigid | ((np.abs(gap) >= PLAY_CLOSED) & (gap * force_next >= 0.0))
        guess = np.where(contact, ahead, loose)
        diag = coupling[:, JOINT_INDEX, JOINT_INDEX]
        others = (coupling @ guess[..., None])[..., 0] - diag * guess
        net = held - others
        net = np.sign(net) * np.maximum(np.abs(net) - drag, 0.0)
        free = np.divide(net, diag, out=rod.copy(), where=diag != 0.0)
        reach = gap + (free - rod) * per_play
        closing = ~contact & (np.abs(reach) >= 1.0)
        contact |= closing
        return np.where(contact, ahead, free), contact, closing, reach, free, diag, others

    def _correct(self, rows: np.ndarray, system: np.ndarray, solved: np.ndarray,
                 start: np.ndarray, held: np.ndarray, speed: np.ndarray, force_now: np.ndarray,
                 scale: np.ndarray, motion: tuple[np.ndarray, ...], fixed: np.ndarray,
                 checks: tuple[np.ndarray, ...], shifts: np.ndarray) -> None:
        guess = speed[rows]
        base = start[rows]
        shift = shifts[rows]
        matrix = system[rows]
        cond, gain, check_start = (a[rows] for a in checks)
        opened = np.zeros(cond.shape, bool)
        free = ~fixed[:, None]
        for attempt in range(CORRECTOR_PASSES):
            sol, held, _ = self._hold(rows, matrix, held, guess, fixed[:, None])
            moved = self._engage(*self._forces(sol, force_now, scale), *motion)[0]
            slid = np.abs(moved - guess) > SPEED_STEP
            guess = np.where(slid, moved, guess)
            delta = sol[..., 0] + (sol[..., 1:] @ guess[..., None])[..., 0]
            reach = base + (delta @ NODE_DROPS)[:, RELIEFS]
            moves = self._relief_moves(rows, base + shift, reach)
            end = check_start + (delta @ NODE_DROPS)[:, ORIFICES]
            settled, now_opened, flips = _settle(cond, opened, gain, end)
            flips &= free
            if (not (moves.any() or slid.any() or flips.any())
                    or attempt == CORRECTOR_PASSES - 1):
                break
            if flips.any():
                cond = np.where(flips, settled, cond)
                opened = np.where(flips, now_opened, opened)
                turn = flips.any(1)
                matrix[turn] = self._set_checks(rows[turn], cond[turn], check_start[turn])
            if not moves.any():
                continue
            shift = np.where(moves, reach - base, shift)
            self._set_reliefs(rows, base, shift)
            matrix = self._refresh(rows)
        system[rows] = matrix
        solved[rows] = sol
        self._held_nodes[rows] = held
        checks[0][rows] = cond
        shifts[rows] = shift

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
