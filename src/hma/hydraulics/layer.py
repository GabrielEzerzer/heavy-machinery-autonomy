"""The hydraulics layer: valve commands in [-1, 1] at the control rate move the spools, two pumps
and the oil in every chamber push the joints, and the true state becomes delayed, noisy sensor
readings. One layer serves every machine in a scene; machine i is named machine_prefix(i)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import mujoco
import numpy as np

from hma.hydraulics import circuit as c
from hma.hydraulics import kernels
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
from hma.machine.model import COUNTERWEIGHT_MASS, set_const
from hma.machine.tracks import SIDES, coupled_names
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
CHECKS = np.array([dst != c.TANK for _, dst, *_ in c.ORIFICES])
DROPS = c.INCIDENCE.T.copy()
DISPLACEMENT = np.zeros((NODES, JOINTS))
DISPLACEMENT[c.PUMPS::2, :] = -np.diag(c.AREA[:, 0])
DISPLACEMENT[c.PUMPS + 1::2, :] = np.diag(c.AREA[:, 1])
DEMAND_MAP = np.concatenate([c.TABLES["demand"][0] - c.TABLES["demand"][1],
                             c.TABLES["demand"][1]])
MASK_EXTEND, MASK_RETRACT = c.DEMAND_MASK
PLAY_CLOSED = 1.0 - 1e-9
VOLUME_SLOPE = np.stack([c.STROKE_AREA[:, 0], -c.STROKE_AREA[:, 1]], axis=-1)
VOLUME_CONSTANT = np.stack([-c.STROKE_AREA[:, 0] * c.STROKE[:, 0],
                            c.STROKE_AREA[:, 1] * c.STROKE[:, 1]], axis=-1)
EDGE_ENDS = np.array([e[:2] for e in c.ORIFICES] + c.RELIEFS + c.LEAKS).T.copy()
CIRCUIT_VALUES = np.array([
    c.ROOT_SMOOTHING, 1.0 / c.RELIEF_KNEE, c.RELIEF_SLOPE * c.RELIEF_KNEE, c.RELIEF_SLOPE,
    RELIEF_BAND, RELIEF_STEP, RELIEF_MISS, c.STANDBY_FLOW, c.PUMP_RESPONSE, 1.0 / SWING_BLEND,
    c.SWING_MECHANICAL, c.SWING_VOLUMETRIC, c.CAVITATION_PRESSURE, c.BRAKE_DELAY, PLAY_CLOSED,
    SPEED_STEP, c.GEOMETRY_LAST])
COUNTS = np.array([CORRECTOR_PASSES, CHECK_PASSES, CHECK_PASSES + int(CHECKS.sum())
                   + CORRECTOR_PASSES, SWING, c.PUMPS, c.OPEN_BYPASS, PRIORITY_FIRST,
                   len(c.ORIFICES), len(c.RELIEFS)])
CIRCUIT = (CIRCUIT_VALUES, EDGE_ENDS[0].copy(), EDGE_ENDS[1].copy(), CHECKS, c.GAIN_MAP,
           DEMAND_MAP, DISPLACEMENT, MASK_EXTEND.copy(), MASK_RETRACT.copy(), c.JOINT_SIGN,
           VOLUME_SLOPE, c.PRIORITY_EDGES, c.GEOMETRY_TABLE, c.GEOMETRY_STEP, c.GEOMETRY_START,
           c.GEOMETRY_SCALE, c.GEOMETRY_BASE, REDO_FORCE)
if len(CIRCUIT_VALUES) != kernels.C_SIZE or len(COUNTS) != kernels.P_SIZE:
    raise ImportError("the circuit constants do not match the compiled step")


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
            return np.array([m.body(machine_prefix(i) + name).id
                             for s in SIDES for name in coupled_names(s)])

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
        self._work = kernels.workspace(n, JOINTS, NODES, len(c.ORIFICES), len(c.RELIEFS),
                                       len(c.INCIDENCE))
        self.redos = 0
        self.capped = 0
        self.scores: list[np.ndarray] = []
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
        set_const(self.model, self._scratch)
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
        area = np.empty(self._spool.shape)
        kernels.openings(head, self._commands, self._command_flat, self._held, self._spool,
                         self._alpha, self._delay_index, self._deadband, self._open_scale,
                         self._piloted, self._segment_base, self._slope, self._intercept,
                         self._gain, BRAKE_PILOT * CURVE_SEGMENTS, CURVE_SEGMENTS, area)
        return area

    @property
    def _orifice_gain(self) -> np.ndarray:
        return kernels.orifice_gain(self._work)

    def step(self) -> None:
        n, dt, data, model = self.n, self.dt, self.data, self.model
        area = self._openings()
        chambers = self._pressure[:, c.PUMPS:].reshape(n, JOINTS, 2)
        drag = np.zeros((n, JOINTS))
        drag[:, 1:] = seal_friction(chambers[:, 1:, 0], chambers[:, 1:, 1],
                                    self._piston_speed[:, 1:]) * self._seal_scale[:, None]
        params = (self._pump_max, self._power, self._spool_flow, self._relief, self._volume0,
                  self._inv_bulk_dt, self._leakage, self._half_play, self._base_friction,
                  self._brake, self._base_armature, self._base_frictionloss)
        state = (self._pressure, self._pump, self._gap, self._rod_speed, self._piston_speed,
                 self._accel, self._cavity, self._held_nodes, self._stiff, self._torque_scale,
                 self._neutral, self._rhs, self._edge_flow, self._weights, self._valve,
                 self._demand_in, self._nodes, self._spool, self._piloted)
        energy = state + (self.pump_energy, self.lost_energy, self.work_energy, self.pump_power)
        mj = (data.qpos, data.qvel, data.ctrl, model.dof_armature, model.dof_frictionloss,
              self._qpos, self._qvel, self._actuators)
        work = self._work
        kernels.begin(CIRCUIT, COUNTS, dt, area, params, state, mj, drag, work)
        mujoco.mj_getState(model, data, self._state, STATE)
        self.scores = []
        for attempt in range(REDO_PASSES + 2):
            kernels.before(CIRCUIT, COUNTS, dt, params, state, mj, work)
            track_step(model, data)
            outcome, bound = kernels.after(attempt, REDO_PASSES, CIRCUIT, COUNTS, dt, params,
                                           energy, mj, work)
            if outcome == kernels.DONE:
                break
            self.scores.append(kernels.scores(work))
            if attempt == REDO_PASSES:
                self.capped += 1
            if outcome == kernels.KEPT:
                break
            self.redos += 1
            mujoco.mj_setState(model, data, self._state, STATE)
        self.check_bound += int(bound)

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
