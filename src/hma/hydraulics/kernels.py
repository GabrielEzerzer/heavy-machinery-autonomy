"""The hydraulics step's per-machine arithmetic compiled with Numba: the oil network before and
after each MuJoCo step, the corrector, the check passes and the bookkeeping. Every circuit constant
comes in as an argument, so a cached compile never keeps a value from an older source."""

from __future__ import annotations

import math

import numpy as np
from numba import njit
from numpy.linalg import LinAlgError

LOGE2 = math.log(2.0)

(C_ROOT_SMOOTHING, C_INV_KNEE, C_RELIEF_LEVEL, C_RELIEF_SLOPE, C_RELIEF_BAND, C_RELIEF_STEP,
 C_RELIEF_MISS, C_STANDBY_FLOW, C_PUMP_RESPONSE, C_INV_SWING_BLEND, C_SWING_MECHANICAL,
 C_SWING_VOLUMETRIC, C_CAVITATION, C_BRAKE_DELAY, C_PLAY_CLOSED, C_SPEED_STEP, C_GEOMETRY_LAST,
 C_SIZE) = range(18)
(P_CORRECTOR, P_CHECK, P_CHECK_TOTAL, P_SWING, P_PUMPS, P_OPEN_BYPASS, P_PRIORITY_FIRST,
 P_ORIFICES, P_RELIEFS, P_SIZE) = range(10)

(J_Q, J_QD, J_LENGTH, J_LEVER, J_FORCE_NOW, J_PLAY, J_PER_PLAY, J_ROD, J_DRAG, J_SEAL,
 J_REDO_SPEED, J_BEST_SPEED, J_AHEAD, J_LOOSE, J_PREDICTED, J_HELD, J_FREE, J_DIAG, J_OTHERS,
 J_REACH, J_JOINT_ROD, J_SPEED, J_SAVED_GAP, J_MISS, J_SLOTS) = range(25)
B_EXTEND, B_RIGID, B_CONTACT, B_CLOSING, B_SLOTS = range(5)
O_GAIN, O_CHECK_START, O_COND_START, O_REDO_COND, O_BEST_COND, O_COND, O_SLOTS = range(7)
N_CHANGE, N_MISSING, N_RISE, N_SLOTS = range(4)
H_REDO, H_BEST, H_HELD, H_START, H_SLOTS = range(5)
R_START, R_SHIFT, R_SLOTS = range(3)
W_FLOW_SCALE, W_BRAKE, W_BEST_SCORE, W_SCORE, W_SLOTS = range(5)
F_REDO, F_BEST_REDO, F_FIX, F_CHANGED, F_SLOTS = range(5)
DONE, KEPT, AGAIN = range(3)


def workspace(n: int, joints: int, nodes: int, orifices: int, reliefs: int,
              edges: int) -> tuple[np.ndarray, ...]:
    return (np.zeros((J_SLOTS, n, joints)), np.zeros((B_SLOTS, n, joints), np.bool_),
            np.zeros((O_SLOTS, n, orifices)), np.zeros((N_SLOTS, n, nodes)),
            np.zeros((H_SLOTS, n, nodes), np.bool_), np.zeros((R_SLOTS, n, reliefs)),
            np.zeros((2, n, edges)), np.zeros((n, edges + nodes)), np.zeros((2, n, nodes, nodes)),
            np.zeros((3, n, nodes, 1 + joints)), np.zeros((n, joints, joints)),
            np.zeros((W_SLOTS, n)), np.zeros((F_SLOTS, n), np.bool_), np.zeros(n, np.int64))


def orifice_gain(work: tuple[np.ndarray, ...]) -> np.ndarray:
    return work[2][O_GAIN]


def scores(work: tuple[np.ndarray, ...]) -> np.ndarray:
    return work[11][W_SCORE].copy()


@njit(cache=True, error_model="numpy")
def openings(head, commands, flat, held, spool, alpha, delay_index, deadband, open_scale, piloted,
             segment_base, slope, intercept, gain, pilot, segments, area):
    n, joints = spool.shape
    for i in range(n):
        for j in range(joints):
            commands[head, i, j] = held[i, j]
    for i in range(n):
        piloted[i] = False
        for j in range(joints):
            spool[i, j] += alpha[i, j] * (flat[delay_index[head, i, j]] - spool[i, j])
            opening = abs(spool[i, j]) - deadband[i, j]
            opening = opening if opening >= 0.0 or opening != opening else 0.0
            piloted[i] = piloted[i] or opening * open_scale[i, j] > pilot
            opening *= open_scale[i, j]
            segment = int(opening) if opening < segments - 1 else segments - 1
            segment += segment_base[i, j]
            value = slope[segment] * opening
            value += intercept[segment]
            value *= gain[i, j]
            area[i, j] = value if value <= 1.0 or value != value else 1.0


@njit(cache=True, error_model="numpy")
def _copy(dst, src):
    for index in np.ndindex(dst.shape):
        dst[index] = src[index]


@njit(cache=True, error_model="numpy")
def _sum(a):
    n = a.shape[0]
    if n < 8:
        res = 0.0
        for i in range(n):
            res += a[i]
        return res
    r0, r1, r2, r3, r4, r5, r6, r7 = a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]
    i = 8
    stop = n - n % 8
    while i < stop:
        r0 += a[i]
        r1 += a[i + 1]
        r2 += a[i + 2]
        r3 += a[i + 3]
        r4 += a[i + 4]
        r5 += a[i + 5]
        r6 += a[i + 6]
        r7 += a[i + 7]
        i += 8
    res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
    while i < n:
        res += a[i]
        i += 1
    return res


@njit(cache=True, error_model="numpy")
def _sign(x):
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0 if x == 0.0 else x


@njit(cache=True, error_model="numpy")
def _logaddexp0(x):
    if x == 0.0:
        return x + LOGE2
    tmp = 0.0 - x
    if tmp > 0.0:
        return 0.0 + math.log1p(math.exp(-tmp))
    if tmp <= 0.0:
        return x + math.log1p(math.exp(tmp))
    return tmp


@njit(cache=True, error_model="numpy")
def _relief_flow(drop, relief, k):
    x = (drop - relief) * k[C_INV_KNEE]
    return (_logaddexp0(x) * k[C_RELIEF_LEVEL],
            0.5 * (1.0 + math.tanh(0.5 * x)) * k[C_RELIEF_SLOPE])


@njit(cache=True, error_model="numpy")
def _one_way(gain, drop, smoothing):
    x = max(drop, 0.0)
    shifted = x + smoothing
    chord = gain * x / (shifted * math.sqrt(shifted))
    return chord * x, chord


@njit(cache=True, error_model="numpy")
def _solve(matrix, rhs, out):
    size, width = rhs.shape
    a = matrix.copy()
    for r in range(size):
        for col in range(width):
            out[r, col] = rhs[r, col]
    for col in range(size):
        pivot = col
        big = abs(a[col, col])
        for r in range(col + 1, size):
            if abs(a[r, col]) > big:
                big = abs(a[r, col])
                pivot = r
        if a[pivot, col] == 0.0:
            raise LinAlgError("Singular matrix")
        if pivot != col:
            for c in range(size):
                a[col, c], a[pivot, c] = a[pivot, c], a[col, c]
            for c in range(width):
                out[col, c], out[pivot, c] = out[pivot, c], out[col, c]
        inverse = 1.0 / a[col, col]
        for r in range(col + 1, size):
            factor = a[r, col] * inverse
            a[r, col] = factor
            if factor != 0.0:
                for c in range(col + 1, size):
                    a[r, c] -= factor * a[col, c]
                for c in range(width):
                    out[r, c] -= factor * out[col, c]
    for c in range(width):
        for r in range(size - 1, -1, -1):
            if out[r, c] != 0.0:
                out[r, c] /= a[r, r]
                value = out[r, c]
                for above in range(r):
                    out[above, c] -= value * a[above, r]


@njit(cache=True, error_model="numpy")
def _assemble(weights, src, dst, system):
    nodes = system.shape[0]
    edges = src.shape[0]
    for a in range(nodes):
        for b in range(nodes):
            system[a, b] = 0.0
    for e in range(edges):
        s, a, b = weights[e], src[e], dst[e]
        if a < nodes:
            system[a, a] += s
        if b < nodes:
            system[b, b] += s
        if a < nodes and b < nodes:
            system[a, b] -= s
            system[b, a] -= s
    for a in range(nodes):
        system[a, a] += weights[edges + a]


@njit(cache=True, error_model="numpy")
def _refresh(flow, weights, pump, src, dst, rhs, system):
    nodes = system.shape[0]
    for a in range(nodes):
        rhs[a, 0] = 0.0
    for e in range(src.shape[0]):
        if src[e] < nodes:
            rhs[src[e], 0] -= flow[e]
        if dst[e] < nodes:
            rhs[dst[e], 0] += flow[e]
    for p in range(pump.shape[0]):
        rhs[p, 0] += pump[p]
    _assemble(weights, src, dst, system)


@njit(cache=True, error_model="numpy")
def _node_drop(values, src, dst, e):
    nodes = values.shape[0]
    drop = 0.0
    if src[e] < nodes:
        drop += values[src[e]]
    if dst[e] < nodes:
        drop -= values[dst[e]]
    return drop


@njit(cache=True, error_model="numpy")
def _delta(solved, speed, out):
    for a in range(solved.shape[0]):
        acc = 0.0
        for j in range(speed.shape[0]):
            acc += solved[a, 1 + j] * speed[j]
        out[a] = solved[a, 0] + acc


@njit(cache=True, error_model="numpy")
def _balance(matrix, rhs, delta, speed, dt, out):
    nodes = matrix.shape[0]
    for a in range(nodes):
        acc = 0.0
        for b in range(nodes):
            acc += matrix[a, b] * delta[b]
        balance = acc - rhs[a, 0]
        flow = 0.0
        for j in range(speed.shape[0]):
            flow += rhs[a, 1 + j] * speed[j]
        out[a] = (balance - flow) * dt


@njit(cache=True, error_model="numpy")
def _hold(matrix, rhs, pressure, cavity, held, speed, fixed, k, dt, solved, missing):
    nodes, width = rhs.shape
    floor = np.empty(nodes)
    for a in range(nodes):
        floor[a] = k[C_CAVITATION] - pressure[a]
    given = np.empty((nodes, width))
    system = np.empty((nodes, nodes))
    delta = np.empty(nodes)
    balance = np.empty(nodes)
    now = np.empty(nodes, np.bool_)
    for attempt in range(nodes):
        for a in range(nodes):
            keep = 0.0 if held[a] else 1.0
            given[a, 0] = floor[a] if held[a] else rhs[a, 0] - cavity[a] / dt
            for c in range(1, width):
                given[a, c] = rhs[a, c] * keep
            for b in range(nodes):
                system[a, b] = (1.0 if a == b else 0.0) if held[a] else matrix[a, b]
        _solve(system, given, solved)
        _delta(solved, speed, delta)
        _balance(matrix, rhs, delta, speed, dt, balance)
        same = True
        for a in range(nodes):
            if fixed:
                now[a] = held[a]
            elif held[a]:
                now[a] = cavity[a] + balance[a] > 0.0
            else:
                now[a] = delta[a] < floor[a]
            same = same and now[a] == held[a]
        if same or attempt == nodes - 1:
            break
        for a in range(nodes):
            held[a] = now[a]
    for a in range(nodes):
        missing[a] = balance[a] * (1.0 if held[a] or cavity[a] > 0.0 else 0.0)


@njit(cache=True, error_model="numpy")
def _forces(solved, force_now, scale, displacement, held, coupling):
    nodes, joints = displacement.shape
    for j in range(joints):
        for c in range(1 + joints):
            acc = 0.0
            for a in range(nodes):
                acc += displacement[a, j] * solved[a, c]
            response = acc * scale[j]
            if c == 0:
                held[j] = force_now[j] * scale[j] - response
            else:
                coupling[j, c - 1] = response


@njit(cache=True, error_model="numpy")
def _engage(held, coupling, gap, rigid, rod, ahead, loose, per_play, drag, closed,
            speed, contact, closing, reach, free, diag, others):
    joints = held.shape[0]
    guess = np.empty(joints)
    for j in range(joints):
        acc = 0.0
        for t in range(joints):
            acc += coupling[j, t] * rod[t]
        force_next = held[j] - acc
        contact[j] = rigid[j] or (abs(gap[j]) >= closed and gap[j] * force_next >= 0.0)
        guess[j] = ahead[j] if contact[j] else loose[j]
    for j in range(joints):
        d = coupling[j, j]
        diag[j] = d
        acc = 0.0
        for t in range(joints):
            acc += coupling[j, t] * guess[t]
        others[j] = acc - d * guess[j]
        net = held[j] - others[j]
        net = _sign(net) * max(abs(net) - drag[j], 0.0)
        free[j] = net / d if d != 0.0 else rod[j]
        reach[j] = gap[j] + (free[j] - rod[j]) * per_play[j]
        closing[j] = (not contact[j]) and abs(reach[j]) >= 1.0
        contact[j] = contact[j] or closing[j]
        speed[j] = ahead[j] if contact[j] else free[j]


@njit(cache=True, error_model="numpy")
def _relief_moves(point, reach, relief, pump_max, k, moves):
    moving = False
    for r in range(relief.shape[0]):
        level, tangent = _relief_flow(point[r], relief[r], k)
        far = _relief_flow(reach[r], relief[r], k)[0]
        miss = abs(far - level - tangent * (reach[r] - point[r]))
        near = max(point[r], reach[r]) > relief[r] - k[C_RELIEF_BAND]
        moves[r] = ((near or miss > k[C_RELIEF_MISS] * pump_max)
                    and abs(reach[r] - point[r]) > k[C_RELIEF_STEP])
        moving = moving or moves[r]
    return moving


@njit(cache=True, error_model="numpy")
def _set_reliefs(base, shift, relief, k, flow, slope):
    for r in range(relief.shape[0]):
        level, tangent = _relief_flow(base[r] + shift[r], relief[r], k)
        flow[r] = level - tangent * shift[r]
        slope[r] = tangent


@njit(cache=True, error_model="numpy")
def _settle(cond, opened, gain, end, checks, bound, smoothing, settled, now_opened, flips):
    flipping = False
    for e in range(cond.shape[0]):
        back = checks[e] and cond[e] > 0.0 and end[e] < 0.0
        blocked = (checks[e] and cond[e] == 0.0 and gain[e] > 0.0 and end[e] > 0.0
                   and not bound)
        halve = back and opened[e] and not bound
        if back:
            settled[e] = 0.5 * cond[e] if halve else 0.0
        elif blocked:
            settled[e] = _one_way(gain[e], end[e], smoothing)[1]
        else:
            settled[e] = cond[e]
        now_opened[e] = (opened[e] or blocked) and not (back and not halve)
        flips[e] = back or blocked
        flipping = flipping or flips[e]
    return flipping


@njit(cache=True, error_model="numpy")
def _drops(values, start, src, dst, first, out):
    for r in range(out.shape[0]):
        out[r] = start[r] + _node_drop(values, src, dst, first + r)


@njit(cache=True, error_model="numpy")
def begin(circuit, p, dt, area, params, state, mj, drag, work):
    (k, src, dst, checks, gain_map, demand_map, displacement, mask_extend, mask_retract,
     joint_sign, volume_slope, priority_edges, geometry_table, geometry_step, geometry_start,
     geometry_scale, geometry_base, redo_force) = circuit
    (pump_max, power, spool_flow, relief, volume0, inv_bulk_dt, leakage, half_play, base_friction,
     brake_torque, base_armature, base_frictionloss) = params
    (pressure, pump, gap, rod_speed, piston_speed, accel, cavity, held_nodes, stiff, scale, neutral,
     rhs, edge_flow, weights, valve, demand_in, nodes, spool, piloted) = state
    qpos, qvel, ctrl, armature, frictionloss, qpos_index, qvel_index, actuators = mj
    wj, wb, wo, wn, wh, wr, we, saved_weights, wm, ws, coupling, ww, wf, best_attempt = work
    n, joints = area.shape
    node_count = pressure.shape[1]
    orifices, reliefs = p[P_ORIFICES], p[P_RELIEFS]
    edges = src.shape[0]
    pumps, swing, openings = p[P_PUMPS], p[P_SWING], valve.shape[1] // 2
    for i in range(n):
        q, qd, length, lever = wj[J_Q, i], wj[J_QD, i], wj[J_LENGTH, i], wj[J_LEVER, i]
        extend = wb[B_EXTEND, i]
        for j in range(joints):
            q[j] = qpos[qpos_index[i, j]]
            qd[j] = qvel[qvel_index[i, j]]
            extend[j] = spool[i, j] * joint_sign[j] > 0.0
        length[0] = q[0]
        lever[0] = 1.0
        for c in range(joints - 1):
            u = (q[c + 1] - geometry_start[c]) * geometry_scale[c]
            u = min(max(u, 0.0), k[C_GEOMETRY_LAST])
            index = int(u) if u >= 0.0 else 0
            u -= index
            index += geometry_base[c]
            length[c + 1] = geometry_table[index, 0] + u * geometry_step[index, 0]
            lever[c + 1] = geometry_table[index, 1] + u * geometry_step[index, 1]

        for j in range(joints):
            demand_in[i, j] = area[i, j] * (1.0 if extend[j] else 0.0)
            demand_in[i, joints + j] = area[i, j]
        supply = 0.0
        for c in range(pumps):
            supply += pressure[i, c]
        cap = power[i] / max(supply, 1.0)
        for c in range(pumps):
            demand = 0.0
            for r in range(2 * joints):
                demand += demand_in[i, r] * demand_map[r, c]
            target = min(max(demand, k[C_STANDBY_FLOW]), min(pump_max[i], cap))
            pump[i, c] += (dt / k[C_PUMP_RESPONSE]) * (target - pump[i, c])

        for j in range(joints):
            valve[i, openings + j] = area[i, j] * spool_flow[i, j]
        for c in range(pumps):
            widest = -np.inf
            for j in range(joints):
                mask = mask_extend[j, c] if extend[j] else mask_retract[j, c]
                widest = max(widest, mask * area[i, j])
            valve[i, openings + p[P_OPEN_BYPASS] + c] = 1.0 - widest
        for j in range(joints):
            valve[i, j] = valve[i, openings + j] * (1.0 if extend[j] else 0.0)
        gain = wo[O_GAIN, i]
        for e in range(orifices):
            acc = 0.0
            for r in range(2 * openings):
                acc += valve[i, r] * gain_map[r, e]
            gain[e] = acc
        for e in priority_edges:
            gain[e] *= 1.0 - area[i, p[P_PRIORITY_FIRST]]
        for a in range(node_count):
            nodes[i, a] = pressure[i, a]
        drop = we[0, i]
        for e in range(edges):
            drop[e] = _node_drop(nodes[i], src, dst, e)
        flow, slope = edge_flow[i], weights[i, :edges]
        check_start, cond_start = wo[O_CHECK_START, i], wo[O_COND_START, i]
        for e in range(orifices):
            check_start[e] = drop[e]
            flow[e], slope[e] = _one_way(gain[e], drop[e], k[C_ROOT_SMOOTHING])
            cond_start[e] = slope[e]
        for r in range(reliefs):
            flow[orifices + r], slope[orifices + r] = _relief_flow(drop[orifices + r],
                                                                  relief[i, r], k)
        for e in range(orifices + reliefs, edges):
            flow[e] = leakage[i, e - orifices - reliefs] * drop[e]
            slope[e] = leakage[i, e - orifices - reliefs]

        force_now = wj[J_FORCE_NOW, i]
        for j in range(joints):
            acc = 0.0
            for a in range(node_count):
                acc += pressure[i, a] * displacement[a, j]
            force_now[j] = -acc
        motoring = math.tanh(force_now[swing] * qd[swing] * k[C_INV_SWING_BLEND])
        scale[i, swing] = k[C_SWING_MECHANICAL] ** motoring
        flow_scale = k[C_SWING_VOLUMETRIC] ** -motoring
        ww[W_FLOW_SCALE, i] = flow_scale
        for a in range(node_count):
            rhs[i, a, 1 + swing] = displacement[a, swing] * flow_scale
        for j in range(joints):
            for side in range(2):
                volume = max(volume0[i, j, side] + volume_slope[j, side] * length[j], 1e-6)
                weights[i, edges + pumps + 2 * j + side] = volume * inv_bulk_dt[i, 0]
        system, solved = wm[0, i], ws[0, i]
        _refresh(flow, weights[i], pump[i], src, dst, rhs[i], system)
        _solve(system, rhs[i], solved)

        play, per_play, rod = wj[J_PLAY, i], wj[J_PER_PLAY, i], wj[J_ROD, i]
        for j in range(joints):
            play[j] = half_play[i, j] * abs(lever[j])
            per_play[j] = dt / play[j] if play[j] > 0.0 else 0.0
            rod[j] = lever[j] * qd[j]
            wb[B_RIGID, i, j] = play[j] == 0.0
        for r in range(reliefs):
            wr[R_START, i, r] = drop[orifices + r]
        _copy(we[1, i], edge_flow[i])
        _copy(saved_weights[i], weights[i])
        _copy(ws[1, i], rhs[i])
        _copy(wm[1, i], system)
        _copy(ws[2, i], solved)
        _copy(wj[J_SAVED_GAP, i], gap[i])
        neutral[i] = 0.0 if piloted[i] else neutral[i] + dt
        ww[W_BRAKE, i] = base_friction[i] + (brake_torque[i] if neutral[i] >= k[C_BRAKE_DELAY]
                                             else 0.0)
        for j in range(joints):
            wj[J_DRAG, i, j] = drag[i, j]
            wj[J_SEAL, i, j] = drag[i, j] * abs(lever[j])
        wf[F_REDO, i] = False
        wh[H_REDO, i, :] = False
        wj[J_REDO_SPEED, i, :] = 0.0
        _copy(wo[O_REDO_COND, i], cond_start)
        ww[W_BEST_SCORE, i] = np.inf
        wf[F_BEST_REDO, i] = False
        wh[H_BEST, i, :] = False
        wj[J_BEST_SPEED, i, :] = 0.0
        best_attempt[i] = 0
        _copy(wo[O_BEST_COND, i], cond_start)


@njit(cache=True, error_model="numpy")
def _correct(rows, circuit, p, dt, params, state, work):
    (k, src, dst, checks, gain_map, demand_map, displacement, mask_extend, mask_retract,
     joint_sign, volume_slope, priority_edges, geometry_table, geometry_step, geometry_start,
     geometry_scale, geometry_base, redo_force) = circuit
    (pump_max, power, spool_flow, relief, volume0, inv_bulk_dt, leakage, half_play, base_friction,
     brake_torque, base_armature, base_frictionloss) = params
    (pressure, pump, gap, rod_speed, piston_speed, accel, cavity, held_nodes, stiff, scale, neutral,
     rhs, edge_flow, weights, valve, demand_in, nodes, spool, piloted) = state
    wj, wb, wo, wn, wh, wr, we, saved_weights, wm, ws, coupling, ww, wf, best_attempt = work
    count = rows.shape[0]
    joints = wj.shape[2]
    node_count = pressure.shape[1]
    orifices, reliefs = p[P_ORIFICES], p[P_RELIEFS]
    guess = np.empty((count, joints))
    held = np.empty((count, node_count), np.bool_)
    opened = np.zeros((count, orifices), np.bool_)
    settled = np.empty((count, orifices))
    now_opened = np.empty((count, orifices), np.bool_)
    flips = np.empty((count, orifices), np.bool_)
    flipped = np.empty(count, np.bool_)
    moves = np.empty((count, reliefs), np.bool_)
    reach = np.empty((count, reliefs))
    point = np.empty(reliefs)
    moved = np.empty(joints)
    delta = np.empty(node_count)
    end = np.empty(orifices)
    missing = np.empty(node_count)
    force = np.empty(joints)
    pull = np.empty((joints, joints))
    contact = np.empty(joints, np.bool_)
    closing = np.empty(joints, np.bool_)
    far = np.empty(joints)
    free = np.empty(joints)
    diag = np.empty(joints)
    others = np.empty(joints)
    for s in range(count):
        i = rows[s]
        _copy(guess[s], wj[J_PREDICTED, i])
        _copy(held[s], wh[H_START, i])
    for attempt in range(p[P_CORRECTOR]):
        any_move = any_slid = any_flip = False
        for s in range(count):
            i = rows[s]
            fixed = wf[F_REDO, i]
            solved, system = ws[0, i], wm[0, i]
            base, shift = wr[R_START, i], wr[R_SHIFT, i]
            _hold(system, rhs[i], pressure[i], cavity[i], held[s], guess[s], fixed, k, dt, solved,
                  missing)
            _forces(solved, wj[J_FORCE_NOW, i], scale[i], displacement, force, pull)
            _engage(force, pull, gap[i], wb[B_RIGID, i], wj[J_ROD, i], wj[J_AHEAD, i],
                    wj[J_LOOSE, i], wj[J_PER_PLAY, i], wj[J_DRAG, i], k[C_PLAY_CLOSED], moved,
                    contact, closing, far, free, diag, others)
            for j in range(joints):
                if abs(moved[j] - guess[s, j]) > k[C_SPEED_STEP]:
                    guess[s, j] = moved[j]
                    any_slid = True
            _delta(solved, guess[s], delta)
            _drops(delta, base, src, dst, orifices, reach[s])
            for r in range(reliefs):
                point[r] = base[r] + shift[r]
            moving = _relief_moves(point, reach[s], relief[i], pump_max[i], k, moves[s])
            any_move = any_move or moving
            _drops(delta, wo[O_CHECK_START, i], src, dst, 0, end)
            _settle(wo[O_COND, i], opened[s], wo[O_GAIN, i], end, checks, False,
                    k[C_ROOT_SMOOTHING], settled[s], now_opened[s], flips[s])
            flipped[s] = False
            for e in range(orifices):
                flips[s, e] = flips[s, e] and not fixed
                flipped[s] = flipped[s] or flips[s, e]
            any_flip = any_flip or flipped[s]
        if not (any_move or any_slid or any_flip) or attempt == p[P_CORRECTOR] - 1:
            break
        if any_flip:
            for s in range(count):
                i = rows[s]
                cond = wo[O_COND, i]
                for e in range(orifices):
                    if flips[s, e]:
                        cond[e] = settled[s, e]
                        opened[s, e] = now_opened[s, e]
                if flipped[s]:
                    for e in range(orifices):
                        edge_flow[i, e] = cond[e] * wo[O_CHECK_START, i, e]
                        weights[i, e] = cond[e]
                    _refresh(edge_flow[i], weights[i], pump[i], src, dst, rhs[i], wm[0, i])
        if not any_move:
            continue
        for s in range(count):
            i = rows[s]
            base, shift = wr[R_START, i], wr[R_SHIFT, i]
            for r in range(reliefs):
                if moves[s, r]:
                    shift[r] = reach[s, r] - base[r]
            _set_reliefs(base, shift, relief[i], k, edge_flow[i, orifices:orifices + reliefs],
                         weights[i, orifices:orifices + reliefs])
            _refresh(edge_flow[i], weights[i], pump[i], src, dst, rhs[i], wm[0, i])
    for s in range(count):
        _copy(held_nodes[rows[s]], held[s])


@njit(cache=True, error_model="numpy")
def before(circuit, p, dt, params, state, mj, work):
    (k, src, dst, checks, gain_map, demand_map, displacement, mask_extend, mask_retract,
     joint_sign, volume_slope, priority_edges, geometry_table, geometry_step, geometry_start,
     geometry_scale, geometry_base, redo_force) = circuit
    (pump_max, power, spool_flow, relief, volume0, inv_bulk_dt, leakage, half_play, base_friction,
     brake_torque, base_armature, base_frictionloss) = params
    (pressure, pump, gap, rod_speed, piston_speed, accel, cavity, held_nodes, stiff, scale, neutral,
     rhs, edge_flow, weights, valve, demand_in, nodes, spool, piloted) = state
    qpos, qvel, ctrl, armature, frictionloss, qpos_index, qvel_index, actuators = mj
    wj, wb, wo, wn, wh, wr, we, saved_weights, wm, ws, coupling, ww, wf, best_attempt = work
    n, node_count = pressure.shape
    joints = wj.shape[2]
    orifices, reliefs = p[P_ORIFICES], p[P_RELIEFS]
    swing = p[P_SWING]
    zeros = np.zeros(orifices, np.bool_)
    settled = np.empty(orifices)
    now_opened = np.empty(orifices, np.bool_)
    flips = np.empty(orifices, np.bool_)
    moves = np.empty(reliefs, np.bool_)
    reach = np.empty(reliefs)
    end = np.empty(orifices)
    force = np.empty(joints)
    pull = np.empty((joints, joints))
    contact = np.empty(joints, np.bool_)
    closing = np.empty(joints, np.bool_)
    far = np.empty(joints)
    free = np.empty(joints)
    diag = np.empty(joints)
    others = np.empty(joints)
    fixing = 0
    for i in range(n):
        redo = wf[F_REDO, i]
        cond = wo[O_COND, i]
        _copy(cond, wo[O_REDO_COND, i] if redo else wo[O_COND_START, i])
        wr[R_SHIFT, i, :] = 0.0
        if redo:
            for e in range(orifices):
                edge_flow[i, e] = cond[e] * wo[O_CHECK_START, i, e]
                weights[i, e] = cond[e]
            _refresh(edge_flow[i], weights[i], pump[i], src, dst, rhs[i], wm[0, i])
        lever, qd = wj[J_LEVER, i], wj[J_QD, i]
        ahead, loose, predicted = wj[J_AHEAD, i], wj[J_LOOSE, i], wj[J_PREDICTED, i]
        for j in range(joints):
            ahead[j] = wj[J_REDO_SPEED, i, j] if redo else lever[j] * (qd[j] + dt * accel[i, j])
            loose[j] = wj[J_REDO_SPEED, i, j] if redo else rod_speed[i, j]
        solved = ws[0, i]
        _forces(solved, wj[J_FORCE_NOW, i], scale[i], displacement, force, pull)
        _engage(force, pull, gap[i], wb[B_RIGID, i], wj[J_ROD, i], ahead, loose,
                wj[J_PER_PLAY, i], wj[J_DRAG, i], k[C_PLAY_CLOSED], predicted, contact, closing,
                far, free, diag, others)
        if redo:
            _copy(predicted, wj[J_REDO_SPEED, i])
        rise = wn[N_RISE, i]
        _delta(solved, predicted, rise)
        held_nodes[i, :] = False
        base = wr[R_START, i]
        _drops(rise, base, src, dst, orifices, reach)
        moving = _relief_moves(base, reach, relief[i], pump_max[i], k, moves)
        start = wh[H_START, i]
        holding = False
        for a in range(node_count):
            start[a] = (wh[H_REDO, i, a] if redo
                        else pressure[i, a] + rise[a] < k[C_CAVITATION] or cavity[i, a] > 0.0)
            holding = holding or start[a]
        _drops(rise, wo[O_CHECK_START, i], src, dst, 0, end)
        flipping = _settle(cond, zeros, wo[O_GAIN, i], end, checks, False, k[C_ROOT_SMOOTHING],
                           settled, now_opened, flips)
        wf[F_FIX, i] = holding or moving or redo or flipping
        if wf[F_FIX, i]:
            fixing += 1
    if fixing:
        rows = np.empty(fixing, np.int64)
        s = 0
        for i in range(n):
            if wf[F_FIX, i]:
                rows[s] = i
                s += 1
        _correct(rows, circuit, p, dt, params, state, work)

    for i in range(n):
        lever, qd = wj[J_LEVER, i], wj[J_QD, i]
        contact, closing, reach_j = wb[B_CONTACT, i], wb[B_CLOSING, i], wj[J_REACH, i]
        held = wj[J_HELD, i]
        _forces(ws[0, i], wj[J_FORCE_NOW, i], scale[i], displacement, held, coupling[i])
        _engage(held, coupling[i], gap[i], wb[B_RIGID, i], wj[J_ROD, i], wj[J_AHEAD, i],
                wj[J_LOOSE, i], wj[J_PER_PLAY, i], wj[J_DRAG, i], k[C_PLAY_CLOSED], far, contact,
                closing, reach_j, wj[J_FREE, i], wj[J_DIAG, i], wj[J_OTHERS, i])
        for j in range(joints):
            if closing[j]:
                gap[i, j] = _sign(reach_j[j])
            stiff[i, j] = wj[J_DIAG, i, j] * lever[j] * lever[j] if contact[j] else 0.0
            push = lever[j] * (held[j] - wj[J_OTHERS, i, j]) if contact[j] else 0.0
            ctrl[actuators[i, j]] = push - stiff[i, j] * qd[j]
            armature[qvel_index[i, j]] = base_armature[i, j] + stiff[i, j] * dt
            friction = base_frictionloss[i, j] + (wj[J_SEAL, i, j] if contact[j] else 0.0)
            frictionloss[qvel_index[i, j]] = ww[W_BRAKE, i] if j == swing else friction


@njit(cache=True, error_model="numpy")
def _finish(circuit, p, dt, params, state, mj, work):
    (k, src, dst, checks, gain_map, demand_map, displacement, mask_extend, mask_retract,
     joint_sign, volume_slope, priority_edges, geometry_table, geometry_step, geometry_start,
     geometry_scale, geometry_base, redo_force) = circuit
    (pressure, pump, gap, rod_speed, piston_speed, accel, cavity, held_nodes, stiff, scale, neutral,
     rhs, edge_flow, weights, valve, demand_in, nodes, spool, piloted, pump_energy, lost_energy,
     work_energy, pump_power) = state
    qpos, qvel, ctrl, armature, frictionloss, qpos_index, qvel_index, actuators = mj
    wj, wb, wo, wn, wh, wr, we, saved_weights, wm, ws, coupling, ww, wf, best_attempt = work
    n, node_count = pressure.shape
    joints = wj.shape[2]
    edges = src.shape[0]
    swing, pumps = p[P_SWING], p[P_PUMPS]
    used = np.empty(edges)
    stored = np.empty(node_count)
    joint_terms = np.empty(joints)
    work_terms = np.empty(joints)
    pump_terms = np.empty(pumps)
    for i in range(n):
        change, missing = wn[N_CHANGE, i], wn[N_MISSING, i]
        held = wh[H_HELD, i]
        speed, free, joint_rod = wj[J_SPEED, i], wj[J_FREE, i], wj[J_JOINT_ROD, i]
        contact = wb[B_CONTACT, i]
        for j in range(joints):
            accel[i, j] = (qvel[qvel_index[i, j]] - wj[J_QD, i, j]) * (1.0 / dt)
            rod_speed[i, j] = free[j]
            piston_speed[i, j] = speed[j]
        for a in range(node_count):
            held_nodes[i, a] = held[a]
            missing[a] *= 1.0 if held[a] or cavity[i, a] > 0.0 else 0.0
            pressure[i, a] += change[a]
        for e in range(edges):
            used[e] = edge_flow[i, e] + weights[i, e] * _node_drop(change, src, dst, e)
            used[e] *= _node_drop(pressure[i], src, dst, e)
        lost = _sum(used) * dt
        for a in range(node_count):
            stored[a] = pressure[i, a] * missing[a]
        lost -= _sum(stored)
        for a in range(node_count):
            cavity[i, a] = max(cavity[i, a] + missing[a], 0.0) if held[a] else 0.0
        for j in range(joints):
            acc = 0.0
            for a in range(node_count):
                acc += pressure[i, a] * displacement[a, j]
            hydraulic = speed[j] * acc
            if j == swing:
                hydraulic *= ww[W_FLOW_SCALE, i]
            mechanical = speed[j] * acc * scale[i, j]
            loose = 0.0 if contact[j] else mechanical
            joint_terms[j] = mechanical - hydraulic - loose
            work_terms[j] = mechanical if contact[j] else 0.0
        lost_energy[i] += lost + _sum(joint_terms) * dt
        work_energy[i] -= _sum(work_terms) * dt
        for j in range(joints):
            moved = gap[i, j] + (0.0 if contact[j] else (free[j] - joint_rod[j])
                                 * wj[J_PER_PLAY, i, j])
            gap[i, j] = min(max(moved, -1.0), 1.0)
        for c in range(pumps):
            pump_terms[c] = pressure[i, c] * pump[i, c]
        pump_power[i] = _sum(pump_terms)
        pump_energy[i] += pump_power[i] * dt


@njit(cache=True, error_model="numpy")
def after(attempt, redo_passes, circuit, p, dt, params, state, mj, work):
    (k, src, dst, checks, gain_map, demand_map, displacement, mask_extend, mask_retract,
     joint_sign, volume_slope, priority_edges, geometry_table, geometry_step, geometry_start,
     geometry_scale, geometry_base, redo_force) = circuit
    (pump_max, power, spool_flow, relief, volume0, inv_bulk_dt, leakage, half_play, base_friction,
     brake_torque, base_armature, base_frictionloss) = params
    (pressure, pump, gap, rod_speed, piston_speed, accel, cavity, held_nodes, stiff, scale, neutral,
     rhs, edge_flow, weights, valve, demand_in, nodes, spool, piloted, pump_energy, lost_energy,
     work_energy, pump_power) = state
    qpos, qvel, ctrl, armature, frictionloss, qpos_index, qvel_index, actuators = mj
    wj, wb, wo, wn, wh, wr, we, saved_weights, wm, ws, coupling, ww, wf, best_attempt = work
    n, node_count = pressure.shape
    joints = wj.shape[2]
    orifices, reliefs = p[P_ORIFICES], p[P_RELIEFS]
    for i in range(n):
        for j in range(joints):
            armature[qvel_index[i, j]] = base_armature[i, j]
            frictionloss[qvel_index[i, j]] = base_frictionloss[i, j]
    opened = np.empty(orifices, np.bool_)
    settled = np.empty(orifices)
    now_opened = np.empty(orifices, np.bool_)
    flips = np.empty(orifices, np.bool_)
    moves = np.empty(reliefs, np.bool_)
    reach = np.empty(reliefs)
    point = np.empty(reliefs)
    end = np.empty(orifices)
    total = np.empty(node_count)
    solved = np.empty((node_count, 1 + joints))
    bound_any = False
    changed_any = False
    for i in range(n):
        lever, qd, play = wj[J_LEVER, i], wj[J_QD, i], wj[J_PLAY, i]
        free, speed, joint_rod = wj[J_FREE, i], wj[J_SPEED, i], wj[J_JOINT_ROD, i]
        contact = wb[B_CONTACT, i]
        change, missing = wn[N_CHANGE, i], wn[N_MISSING, i]
        held = wh[H_HELD, i]
        system = wm[0, i]
        for j in range(joints):
            joint_rod[j] = lever[j] * qvel[qvel_index[i, j]]
            room = play[j] * (1.0 / dt)
            free[j] = min(max(free[j], joint_rod[j] - room * (1.0 + gap[i, j])),
                          joint_rod[j] + room * (1.0 - gap[i, j]))
            speed[j] = joint_rod[j] if contact[j] else free[j]
        _delta(ws[0, i], speed, change)
        tracked = False
        for a in range(node_count):
            held[a] = held_nodes[i, a]
            missing[a] = 0.0
            tracked = tracked or held[a] or cavity[i, a] > 0.0
        if tracked:
            _balance(system, rhs[i], change, speed, dt, missing)
        wrong = False
        for a in range(node_count):
            if held[a]:
                wrong = wrong or cavity[i, a] + missing[a] < 0.0
            else:
                wrong = wrong or pressure[i, a] + change[a] < k[C_CAVITATION]
        if wrong:
            _hold(system, rhs[i], pressure[i], cavity[i], held, speed, False, k, dt, solved,
                  missing)
            _delta(solved, speed, change)
        cond, gain, check_start = wo[O_COND, i], wo[O_GAIN, i], wo[O_CHECK_START, i]
        base, shift = wr[R_START, i], wr[R_SHIFT, i]
        opened[:] = False
        for passes in range(p[P_CHECK_TOTAL]):
            _drops(change, check_start, src, dst, 0, end)
            bound = passes >= p[P_CHECK]
            if passes == p[P_CHECK]:
                bound_any = _settle(cond, opened, gain, end, checks, False, k[C_ROOT_SMOOTHING],
                                    settled, now_opened, flips) or bound_any
            flipping = _settle(cond, opened, gain, end, checks, bound, k[C_ROOT_SMOOTHING],
                               settled, now_opened, flips)
            _drops(change, base, src, dst, orifices, reach)
            for r in range(reliefs):
                point[r] = base[r] + shift[r]
            moving = _relief_moves(point, reach, relief[i], pump_max[i], k, moves)
            if not (flipping or moving):
                break
            _copy(cond, settled)
            _copy(opened, now_opened)
            for r in range(reliefs):
                if moves[r]:
                    shift[r] = reach[r] - base[r]
            _set_reliefs(base, shift, relief[i], k, edge_flow[i, orifices:orifices + reliefs],
                         weights[i, orifices:orifices + reliefs])
            if flipping:
                for e in range(orifices):
                    edge_flow[i, e] = cond[e] * check_start[e]
                    weights[i, e] = cond[e]
            _refresh(edge_flow[i], weights[i], pump[i], src, dst, rhs[i], system)
            _hold(system, rhs[i], pressure[i], cavity[i], held, speed, False, k, dt, solved,
                  missing)
            _delta(solved, speed, change)
        miss = wj[J_MISS, i]
        changed = False
        for a in range(node_count):
            total[a] = pressure[i, a] + change[a]
            changed = changed or held[a] != held_nodes[i, a]
        for j in range(joints):
            applied = (ctrl[actuators[i, j]] - stiff[i, j] * (qvel[qvel_index[i, j]] - qd[j])) \
                / lever[j]
            acc = 0.0
            for a in range(node_count):
                acc += total[a] * displacement[a, j]
            final = -acc * scale[i, j]
            miss[j] = abs(applied - final) if contact[j] else 0.0
            changed = changed or miss[j] > redo_force[j]
        wf[F_CHANGED, i] = changed
        changed_any = changed_any or changed
    if not changed_any or attempt > redo_passes:
        _finish(circuit, p, dt, params, state, mj, work)
        return DONE, bound_any
    for i in range(n):
        score = -np.inf
        for j in range(joints):
            score = max(score, wj[J_MISS, i, j] / redo_force[j])
        ww[W_SCORE, i] = score
        if score < ww[W_BEST_SCORE, i]:
            ww[W_BEST_SCORE, i] = score
            wf[F_BEST_REDO, i] = wf[F_REDO, i]
            _copy(wh[H_BEST, i], wh[H_REDO, i])
            _copy(wj[J_BEST_SPEED, i], wj[J_REDO_SPEED, i])
            best_attempt[i] = attempt
            _copy(wo[O_BEST_COND, i], wo[O_REDO_COND, i])
    if attempt == redo_passes:
        kept = True
        for i in range(n):
            if wf[F_CHANGED, i] and best_attempt[i] != attempt:
                kept = False
        if kept:
            _finish(circuit, p, dt, params, state, mj, work)
            return KEPT, bound_any
        for i in range(n):
            if wf[F_CHANGED, i]:
                wf[F_REDO, i] = wf[F_BEST_REDO, i]
                _copy(wh[H_REDO, i], wh[H_BEST, i])
                _copy(wj[J_REDO_SPEED, i], wj[J_BEST_SPEED, i])
                _copy(wo[O_REDO_COND, i], wo[O_BEST_COND, i])
    else:
        for i in range(n):
            if wf[F_CHANGED, i]:
                _copy(wh[H_REDO, i], wh[H_HELD, i])
                for j in range(joints):
                    speed = wj[J_SPEED, i, j]
                    wj[J_REDO_SPEED, i, j] = (0.5 * (speed + wj[J_REDO_SPEED, i, j])
                                              if wf[F_REDO, i] else speed)
                _copy(wo[O_REDO_COND, i], wo[O_COND, i])
                wf[F_REDO, i] = True
    for i in range(n):
        _copy(edge_flow[i], we[1, i])
        _copy(weights[i], saved_weights[i])
        _copy(rhs[i], ws[1, i])
        _copy(wm[0, i], wm[1, i])
        _copy(ws[0, i], ws[2, i])
        _copy(gap[i], wj[J_SAVED_GAP, i])
    return AGAIN, False
