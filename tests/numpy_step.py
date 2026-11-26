"""The hydraulics step and the travel power cap as NumPy code, as they stood before the step was
compiled, kept only to run side by side with the compiled step."""

import mujoco
import numpy as np

from hma.hydraulics import circuit as c
from hma.hydraulics.layer import (
    BRAKE_PILOT,
    CHECK_PASSES,
    CHECKS,
    CORRECTOR_PASSES,
    DEMAND_MAP,
    DISPLACEMENT,
    DROPS,
    JOINTS,
    LEAKS,
    MASK_EXTEND,
    MASK_RETRACT,
    NODES,
    ORIFICES,
    PLAY_CLOSED,
    PRIORITY_FIRST,
    REDO_FORCE,
    REDO_PASSES,
    RELIEF_BAND,
    RELIEF_MISS,
    RELIEF_STEP,
    RELIEFS,
    SPEED_STEP,
    STATE,
    SWING,
    SWING_BLEND,
    VOLUME_SLOPE,
    Hydraulics,
)
from hma.hydraulics.params import CURVE_SEGMENTS
from hma.machine.cylinders import (
    _FIRST_STEP,
    _LAST_STEP,
    _PRESSURE_INDEX,
    _SPEED_INDEX,
    SEAL_PRESSURE,
    SEAL_SCALES,
    SEAL_SPEED,
    SEAL_TABLES,
)
from hma.machine.tracks import (
    FULL_SHARE,
    SHARE_TIME,
    TRACTION,
    TRAVEL_POWER,
    WHEEL_RADIUS,
    drive_forces,
    drives,
    used_power,
)

RELIEF_EDGES = np.arange(RELIEFS.start, RELIEFS.stop)
JOINT_INDEX = np.arange(JOINTS)
NODE_INCIDENCE = c.INCIDENCE[:, :NODES].copy()
NODE_DROPS = NODE_INCIDENCE.T.copy()
INFLOW = -NODE_INCIDENCE
DISPLACEMENT_T = DISPLACEMENT.T.copy()
SYSTEM_MAP = np.concatenate([
    np.einsum("ei,ej->eij", NODE_INCIDENCE, NODE_INCIDENCE).reshape(len(c.INCIDENCE), -1),
    np.eye(NODES * NODES)[::NODES + 1]])
IDENTITY = np.eye(NODES)


def _one_way(gain: np.ndarray, drop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.maximum(drop, 0.0)
    shifted = x + c.ROOT_SMOOTHING
    chord = gain * x / (shifted * np.sqrt(shifted))
    return chord * x, chord


def seal_friction(head, rod, speed) -> np.ndarray:
    head, rod = np.asarray(head, float), np.asarray(rod, float)
    i = np.interp(np.abs(speed), SEAL_SPEED, _SPEED_INDEX)
    p = np.stack([rod, np.abs(head - rod)])
    j = (np.interp(p, SEAL_PRESSURE, _PRESSURE_INDEX)
         + np.minimum(p - SEAL_PRESSURE[0], 0.0) / _FIRST_STEP
         + np.maximum(p - SEAL_PRESSURE[-1], 0.0) / _LAST_STEP)
    i0 = np.minimum(i.astype(np.intp), len(SEAL_SPEED) - 2)
    j0 = np.clip(np.floor(j).astype(np.intp), 0, len(SEAL_PRESSURE) - 2)
    kind = np.arange(2).reshape((2,) + (1,) * i.ndim)
    a, b = i - i0, j - j0
    low = SEAL_TABLES[kind, i0, j0] * (1 - b) + SEAL_TABLES[kind, i0, j0 + 1] * b
    high = SEAL_TABLES[kind, i0 + 1, j0] * (1 - b) + SEAL_TABLES[kind, i0 + 1, j0 + 1] * b
    force = np.maximum(low + a * (high - low), 0.0)
    return (SEAL_SCALES.reshape((2,) + (1,) * (i.ndim - 1) + (-1,)) * force).sum(0)


def _shares(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    lead = 0.5 + data.userdata[:len(drives(model).dofs), None]
    return np.concatenate([lead, 1.0 - lead], axis=1)


def _split_power(belt_speeds: np.ndarray, used: np.ndarray, share: np.ndarray,
                 rate: float) -> np.ndarray:
    full = used >= FULL_SHARE * share * TRAVEL_POWER
    weight = np.abs(belt_speeds) * np.where(full.any(axis=-1, keepdims=True), full, True)
    open_use = np.where(full, 0.0, used)
    rest = TRAVEL_POWER - open_use.sum(axis=-1, keepdims=True)
    total = weight.sum(axis=-1, keepdims=True)
    even = np.full_like(weight, 1.0 / weight.shape[-1])
    part = np.divide(weight, total, out=even, where=total > 0.0)
    target = (open_use + rest * part) / TRAVEL_POWER
    return share + rate * (target - share)


def _drive_caps(belt_speeds: np.ndarray, share: np.ndarray) -> np.ndarray:
    return np.minimum(TRACTION, share * TRAVEL_POWER / np.maximum(np.abs(belt_speeds), 1e-12))


def _limit_power(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    lay = drives(model)
    belt = data.qvel[lay.dofs] * WHEEL_RADIUS
    used = used_power(model, data)
    share = _split_power(belt, used, _shares(model, data), model.opt.timestep / SHARE_TIME)
    data.userdata[:len(lay.dofs)] = share[:, 0] - 0.5
    ahead = data.qvel[lay.valves] * belt > 0.0
    excess = np.where(ahead, _drive_caps(belt, share) - TRACTION, 0.0)
    data.qfrc_applied[lay.dofs] = excess * np.sign(belt) * WHEEL_RADIUS


def track_step(model: mujoco.MjModel, data: mujoco.MjData, nstep: int = 1) -> None:
    for _ in range(nstep):
        _limit_power(model, data)
        lay = drives(model)
        belt = data.qvel[lay.dofs] * WHEEL_RADIUS
        mujoco.mj_step(model, data)
        used_power(model, data)[:] = np.maximum(drive_forces(model, data) * belt, 0.0)


def _softplus(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.logaddexp(0.0, x), 0.5 * (1.0 + np.tanh(0.5 * x))


def _settle(cond: np.ndarray, opened: np.ndarray, gain: np.ndarray, end: np.ndarray,
            bound: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    back = CHECKS & (cond > 0.0) & (end < 0.0)
    blocked = CHECKS & (cond == 0.0) & (gain > 0.0) & (end > 0.0) & (not bound)
    halve = back & opened & (not bound)
    cond = np.where(back, np.where(halve, 0.5 * cond, 0.0),
                    np.where(blocked, _one_way(gain, end)[1], cond))
    return cond, (opened | blocked) & ~(back & ~halve), back | blocked


class NumpyHydraulics(Hydraulics):
    def _refresh(self, rows: np.ndarray) -> np.ndarray:
        self._rhs[rows, :, 0] = self._edge_flow[rows] @ INFLOW
        self._rhs[rows, :c.PUMPS, 0] += self._pump[rows]
        return (self._weights[rows] @ SYSTEM_MAP).reshape(len(rows), NODES, NODES)

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
        self._numpy_gain = gain
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
        self.scores = []
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
            self.scores.append(score)
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
