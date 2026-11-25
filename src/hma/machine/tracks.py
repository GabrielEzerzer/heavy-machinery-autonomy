"""Track contact, travel drive and the physics step that uses them. Contacts between each track and
its machine's ground are computed here from the height field, not by MuJoCo's collision, and added
to every step; step() replaces mujoco.mj_step for any scene from load_scene()."""

from __future__ import annotations

import math
import weakref
from typing import NamedTuple

import mujoco
import numpy as np

from hma.machine.mass import GAUGE, SHOE_WIDTH, TRACK_CHAINS, TUMBLER

RAIL = 0.07
IDLER_RADIUS = 0.35
STATIONS_PER_INTERVAL = 15
INTERVALS = 8
STATIONS = STATIONS_PER_INTERVAL * INTERVALS + 1
ARC_STATIONS = 8
ARC_SPAN = math.radians(100.0)
LATERAL = 3
TIE = 1e-3
SLOTS = 2 * 2 * (INTERVALS + 2)
SLIDING = 0.05
CONTACT_SOLREF = (0.02, 1.0)
CONTACT_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)
BRAKE_SOLIMP = (0.9999, 0.9999, 0.001)
IMPRATIO = 10.0

SIDES = ("left", "right")
BELT = "track_"
TRAVEL = "travel_"
FRICTION = "ground_friction"
TRAVEL_SPEED = 5.5 / 3.6
TRACTION = 102.0e3 / 2
TRACK_PITCH = 0.17145
BELT_ARMATURE = TRACK_CHAINS / 2
NOMINAL_FRICTION = (0.45, 0.50)


def side_sign(side: str) -> float:
    return 1.0 if side == "left" else -1.0


def _sample_points() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u = np.linspace(-TUMBLER, TUMBLER, STATIONS)
    rails = np.zeros((2, 2))
    lateral = np.zeros((2, 2, LATERAL))
    for s, side in enumerate(SIDES):
        centre = side_sign(side) * GAUGE / 2
        for h, half in enumerate((-1.0, 1.0)):
            rails[s, h] = centre + half * RAIL
            lateral[s, h] = np.linspace(centre, centre + half * SHOE_WIDTH / 2, LATERAL + 1)[1:]
    angles = np.linspace(ARC_SPAN / ARC_STATIONS, ARC_SPAN, ARC_STATIONS)
    flat = np.zeros((2, 2, LATERAL, STATIONS, 3))
    flat[..., 0] = u
    flat[..., 1] = lateral[..., None]
    arc = np.zeros((2, 2, 2, LATERAL, ARC_STATIONS, 3))
    centre = np.zeros((2, 2, 2, LATERAL, 3))
    for e, end in enumerate((-1.0, 1.0)):
        arc[:, :, e, ..., 0] = end * (TUMBLER + IDLER_RADIUS * np.sin(angles))
        centre[:, :, e, :, 0] = end * TUMBLER
    arc[..., 1] = lateral[:, :, None, :, None]
    arc[..., 2] = IDLER_RADIUS * (1.0 - np.cos(angles))
    centre[..., 1] = lateral[:, :, None, :]
    centre[..., 2] = IDLER_RADIUS
    rail = np.zeros((2, 2, STATIONS, 3))
    rail[..., 0] = u
    rail[..., 1] = rails[..., None]
    samples = np.concatenate([flat.reshape(-1, 3), arc.reshape(-1, 3), centre.reshape(-1, 3)])
    return u, rails, rail.reshape(-1, 3), samples


STATION_U, RAIL_Y, RAIL_POINTS, SAMPLE_POINTS = _sample_points()
N_FLAT = 4 * LATERAL * STATIONS
N_SAMPLED = N_FLAT + 8 * LATERAL * ARC_STATIONS


def _preference() -> np.ndarray:
    rank = np.full((INTERVALS, STATIONS_PER_INTERVAL + 1), -1.0)
    for k in range(INTERVALS):
        idx = np.arange(STATIONS_PER_INTERVAL + 1) + k * STATIONS_PER_INTERVAL
        if k == 0:
            rank[k] = -idx
        elif k == INTERVALS - 1:
            rank[k] = idx
        else:
            rank[k] = -np.abs(idx - (k + 0.5) * STATIONS_PER_INTERVAL)
            rank[k, -1] = -np.inf
    return rank


PREFERENCE = _preference()
INTERVAL_INDEX = (np.arange(INTERVALS)[:, None] * STATIONS_PER_INTERVAL
                  + np.arange(STATIONS_PER_INTERVAL + 1)[None, :])
RAIL_OFFSET = (np.arange(4) * STATIONS).reshape(2, 2, 1)


class Layout(NamedTuple):
    base: np.ndarray
    geoms: np.ndarray
    weld: np.ndarray
    ground: np.ndarray
    friction: np.ndarray
    belt_qpos: np.ndarray
    belt_dof: np.ndarray
    base_dof: np.ndarray
    origin: np.ndarray
    scale: np.ndarray
    heights: np.ndarray
    shape: tuple[int, int]


_layouts: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def layout(model: mujoco.MjModel) -> Layout:
    cached = _layouts.get(model)
    if cached is not None:
        return cached
    from hma.machine.model import BASE_JOINT, GROUND, WELD, machine_prefix

    n = 0
    while mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, machine_prefix(n) + "base") >= 0:
        n += 1
    p = [machine_prefix(i) for i in range(n)]
    hfields = [model.hfield(q + GROUND) for q in p]
    nrow, ncol = int(hfields[0].nrow[0]), int(hfields[0].ncol[0])
    adr = np.array([int(h.adr[0]) for h in hfields])
    size = model.hfield_size[[h.id for h in hfields]]
    if np.any(np.diff(adr) != nrow * ncol) or np.ptp(size, axis=0).any():
        raise ValueError("the machines' height fields must match and be stored one after another")
    grounds = np.array([model.geom(q + GROUND).id for q in p])
    pos = model.geom_pos[grounds]
    sx, sy, sz, _ = size[0]
    lay = Layout(
        base=np.array([model.body(q + "base").id for q in p]),
        geoms=np.array([[model.geom(q + BELT + s).id for s in SIDES] for q in p]),
        weld=np.array([model.equality(q + WELD).id for q in p]),
        ground=grounds,
        friction=np.array([int(model.numeric(q + FRICTION).adr[0]) for q in p]),
        belt_qpos=np.array([[model.joint(q + BELT + s).qposadr[0] for s in SIDES] for q in p]),
        belt_dof=np.array([[model.joint(q + BELT + s).dofadr[0] for s in SIDES] for q in p]),
        base_dof=np.array([model.joint(q + BASE_JOINT).dofadr[0] for q in p]),
        origin=np.stack([pos[:, 0] - sx, pos[:, 1] - sy, pos[:, 2]], axis=1),
        scale=np.array([(ncol - 1) / (2 * sx), (nrow - 1) / (2 * sy), sz]),
        heights=model.hfield_data[adr[0]:adr[0] + n * nrow * ncol],
        shape=(nrow, ncol),
    )
    _layouts[model] = lay
    return lay


def ground_height(lay: Layout, machines: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    nrow, ncol = lay.shape
    origin = lay.origin[machines].reshape(len(machines), *([1] * (x.ndim - 1)), 3)
    cf = (x - origin[..., 0]) * lay.scale[0]
    rf = (y - origin[..., 1]) * lay.scale[1]
    c = np.clip(np.floor(cf), 0, ncol - 2).astype(np.intp)
    r = np.clip(np.floor(rf), 0, nrow - 2).astype(np.intp)
    u = np.clip(cf - c, 0.0, 1.0)
    v = np.clip(rf - r, 0.0, 1.0)
    idx = r * ncol + c
    idx += (np.asarray(machines) * (nrow * ncol)).reshape(origin.shape[:-1])
    hf = lay.heights
    h00 = hf[idx]
    upper = u >= v
    h_side = np.where(upper, hf[idx + 1], hf[idx + ncol])
    h = h00 + np.abs(u - v) * (h_side - h00) + np.minimum(u, v) * (hf[idx + ncol + 1] - h00)
    return origin[..., 2] + h * lay.scale[2]


class Contacts(NamedTuple):
    machine: np.ndarray
    side: np.ndarray
    slot: np.ndarray
    pos: np.ndarray
    depth: np.ndarray
    frame: np.ndarray


def _empty() -> Contacts:
    return Contacts(np.zeros(0, np.intp), np.zeros(0, np.intp), np.zeros(0, np.intp),
                    np.zeros((0, 3)), np.zeros(0), np.zeros((0, 9)))


def contacts_at(lay: Layout, machines: np.ndarray, pos: np.ndarray, rot: np.ndarray) -> Contacts:
    m = len(machines)
    if m == 0:
        return _empty()
    world = SAMPLE_POINTS @ rot.transpose(0, 2, 1) + pos[:, None, :]
    sampled = world[:, :N_SAMPLED]
    h = ground_height(lay, machines, sampled[..., 0], sampled[..., 1])
    rise = h - sampled[..., 2]

    flat = (rise[:, :N_FLAT] * rot[:, 2, 2, None]).reshape(m, 2, 2, LATERAL, STATIONS).max(axis=3)
    grouped = flat[..., INTERVAL_INDEX]
    best = grouped.max(axis=-1, keepdims=True)
    key = np.where(grouped >= best - TIE, PREFERENCE, -np.inf)
    station = INTERVAL_INDEX[np.arange(INTERVALS), key.argmax(axis=-1)]
    flat_depth = np.take_along_axis(flat, station, axis=-1)
    keep = (flat_depth > 0.0) & (flat_depth < IDLER_RADIUS)
    fm, fs, fh, fk = np.nonzero(keep)
    depth_f = flat_depth[keep]
    rail = RAIL_POINTS[(station + RAIL_OFFSET)[keep]]
    rot_f = rot[fm]
    normal_f = rot_f[:, :, 2]
    pos_f = np.einsum("nij,nj->ni", rot_f, rail) + pos[fm] + 0.5 * depth_f[:, None] * normal_f
    frame_f = np.concatenate([normal_f, rot_f[:, :, 0], rot_f[:, :, 1]], axis=1)

    above = rise[:, N_FLAT:]
    if above.max() > 0.0:
        am, as_, slot_a, pos_a, depth_a, frame_a = _arc(world, h, above, rot, m)
    else:
        am = as_ = slot_a = np.zeros(0, np.intp)
        pos_a, depth_a, frame_a = np.zeros((0, 3)), np.zeros(0), np.zeros((0, 9))

    slot_f = (fs * 2 + fh) * (INTERVALS + 2) + fk
    return Contacts(np.concatenate([machines[fm], machines[am]]), np.concatenate([fs, as_]),
                    np.concatenate([slot_f, slot_a]), np.concatenate([pos_f, pos_a]),
                    np.concatenate([depth_f, depth_a]), np.concatenate([frame_f, frame_a]))


def _arc(world, h, above, rot, m):
    shape = (m, 2, 2, 2, LATERAL, ARC_STATIONS)
    ground = world[:, N_FLAT:N_SAMPLED].copy()
    ground[..., 2] = h[:, N_FLAT:]
    ground = ground.reshape(*shape, 3)
    offset = world[:, N_SAMPLED:].reshape(m, 2, 2, 2, LATERAL, 1, 3) - ground
    gap = np.sqrt(np.einsum("...i,...i->...", offset, offset))
    above = above.reshape(shape)
    below = (above > 0.0) & (offset[..., 2] > 0.0)
    arc = np.where(below, IDLER_RADIUS - gap, -np.inf).reshape(m, 2, 2, 2, -1)
    pick = arc.argmax(axis=-1)
    arc_depth = np.take_along_axis(arc, pick[..., None], axis=-1)[..., 0]
    keep = arc_depth > 0.0
    am, as_, ah, ae = np.nonzero(keep)
    depth_a = arc_depth[keep]
    rows = np.arange(len(depth_a))
    sel = pick[keep]
    g = ground.reshape(m, 2, 2, 2, -1, 3)[keep][rows, sel]
    normal_a = offset.reshape(m, 2, 2, 2, -1, 3)[keep][rows, sel]
    normal_a /= gap.reshape(m, 2, 2, 2, -1)[keep][rows, sel, None]
    along = np.cross(rot[am][:, :, 1], normal_a)
    along /= np.linalg.norm(along, axis=1, keepdims=True)
    pos_a = g + 0.5 * depth_a[:, None] * normal_a
    frame_a = np.concatenate([normal_a, along, np.cross(normal_a, along)], axis=1)
    return am, as_, (as_ * 2 + ah) * (INTERVALS + 2) + INTERVALS + ae, pos_a, depth_a, frame_a


def track_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> Contacts:
    lay = layout(model)
    machines = np.flatnonzero(data.eq_active[lay.weld] == 0)
    bodies = lay.base[machines]
    return contacts_at(lay, machines, data.xpos[bodies], data.xmat[bodies].reshape(-1, 3, 3))


_templates: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _template(model: mujoco.MjModel) -> mujoco.MjContact:
    t = _templates.get(model)
    if t is None:
        t = mujoco.MjContact()
        t.dim = 3
        t.solref[:] = CONTACT_SOLREF
        t.solreffriction[:] = (0.0, 0.0)
        t.solimp[:] = CONTACT_SOLIMP
        t.elem[:] = -1
        t.vert[:] = -1
        t.flex[:] = -1
        _templates[model] = t
    return t


_normals: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _slip(model, data, lay, found):
    base = lay.base[found.machine]
    rot = data.xmat[base].reshape(-1, 3, 3)
    dof = lay.base_dof[found.machine]
    lever = found.pos - data.xpos[base]
    spin = np.einsum("nij,nj->ni", rot, data.qvel[dof[:, None] + np.arange(3, 6)])
    belt = data.qvel[lay.belt_dof[found.machine, found.side]]
    v = data.qvel[dof[:, None] + np.arange(3)] + np.cross(spin, lever)
    v -= belt[:, None] * rot[:, :, 0]
    frame = found.frame.reshape(-1, 3, 3)
    return np.einsum("nij,nj->ni", frame[:, 1:], v), rot, lever


def _sliding_friction(model, data, lay, found, mu, normal):
    slip, rot, lever = _slip(model, data, lay, found)
    sliding = np.hypot(slip[:, 0], slip[:, 1]) > SLIDING
    force = np.zeros(model.nv)
    if not sliding.any():
        return sliding, force
    s, k, n = slip[sliding], mu[sliding], normal[sliding]
    scale = n / np.sqrt((k[:, 0] * s[:, 0]) ** 2 + (k[:, 1] * s[:, 1]) ** 2)
    tangential = -(k**2 * s) * scale[:, None]
    frame = found.frame.reshape(-1, 3, 3)[sliding]
    f = np.einsum("ni,nij->nj", tangential, frame[:, 1:])
    m = found.machine[sliding]
    dof = lay.base_dof[m]
    torque = np.einsum("nji,nj->ni", rot[sliding], np.cross(lever[sliding], f))
    np.add.at(force, dof[:, None] + np.arange(3), f)
    np.add.at(force, dof[:, None] + np.arange(3, 6), torque)
    np.add.at(force, lay.belt_dof[m, found.side[sliding]], -np.einsum("ni,ni->n", f,
                                                                      rot[sliding][:, :, 0]))
    return sliding, force


def add_track_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[Contacts, np.ndarray]:
    found = track_contacts(model, data)
    lay = layout(model)
    n = len(found.depth)
    if n == 0:
        return found, np.zeros(model.nv)
    t = _template(model)
    n0 = data.ncon
    for _ in range(n):
        if mujoco.mj_addContact(model, data, t):
            raise RuntimeError("the contact buffer is full")
    mu = model.numeric_data[lay.friction[found.machine][:, None] + np.arange(2)]
    normals = _normals.get(data)
    if normals is None or normals.shape[0] != len(lay.base):
        normals = _normals[data] = np.zeros((len(lay.base), SLOTS))
    sliding, force = _sliding_friction(model, data, lay, found, mu,
                                       normals[found.machine, found.slot])
    c = data.contact
    ground = lay.ground[found.machine]
    belt = lay.geoms[found.machine, found.side]
    c.geom1[n0:] = ground
    c.geom2[n0:] = belt
    c.geom[n0:, 0] = ground
    c.geom[n0:, 1] = belt
    c.pos[n0:] = found.pos
    c.dist[n0:] = -found.depth
    c.frame[n0:] = found.frame
    c.includemargin[n0:] = 0.0
    c.dim[n0:] = np.where(sliding, 1, 3)
    c.friction[n0:, :2] = mu
    c.friction[n0:, 2:] = 0.0
    return found, force


def _store_normals(model, data, found, n0):
    normals = _normals.get(data)
    if normals is None:
        return
    normals[:] = 0.0
    if len(found.depth):
        rows = data.contact.efc_address[n0:n0 + len(found.depth)]
        normals[found.machine, found.slot] = np.where(rows >= 0, data.efc_force[rows], 0.0)


def step(model: mujoco.MjModel, data: mujoco.MjData, nstep: int = 1) -> None:
    belts = layout(model).belt_qpos.ravel()
    for _ in range(nstep):
        mujoco.mj_checkPos(model, data)
        mujoco.mj_checkVel(model, data)
        mujoco.mj_fwdKinematics(model, data)
        mujoco.mj_makeM(model, data)
        mujoco.mj_factorM(model, data)
        mujoco.mj_collision(model, data)
        n0 = data.ncon
        found, friction = add_track_contacts(model, data)
        mujoco.mj_makeConstraint(model, data)
        mujoco.mj_island(model, data)
        mujoco.mj_projectConstraint(model, data)
        mujoco.mj_transmission(model, data)
        mujoco.mj_sensorPos(model, data)
        mujoco.mj_fwdVelocity(model, data)
        mujoco.mj_sensorVel(model, data)
        applied = data.qfrc_applied.copy()
        data.qfrc_applied += friction
        mujoco.mj_step2(model, data)
        data.qfrc_applied[:] = applied
        _store_normals(model, data, found, n0)
        data.qpos[belts] = np.remainder(data.qpos[belts], TRACK_PITCH)


def _barycentric(point: np.ndarray, tri: np.ndarray) -> np.ndarray | None:
    m = np.array([tri[1] - tri[0], tri[2] - tri[0]]).T
    if abs(np.linalg.det(m)) < 1e-12:
        return None
    b, c = np.linalg.solve(m, point - tri[0])
    return np.array([1.0 - b - c, b, c])


def rest_plane(r: np.ndarray, h: np.ndarray, centre: np.ndarray) -> tuple[float, np.ndarray]:
    slope = np.zeros(2)
    active = [int(np.argmax(h))]
    for _ in range(200):
        a = active[0]
        terms = h - r @ slope
        slack = terms[a] - terms
        if len(active) == 3:
            lam = _barycentric(centre, r[active])
            if lam is None:
                far = np.argmax([np.linalg.norm(r[i] - r[j]) for i, j in
                                 ((active[1], active[2]), (active[0], active[2]),
                                  (active[0], active[1]))])
                active.pop(int(far))
                continue
            if lam.min() >= -1e-9:
                return float(terms[a]), slope
            active.pop(int(np.argmin(lam)))
            continue
        if len(active) == 1:
            d = r[a] - centre
        else:
            e = r[active[1]] - r[a]
            e = np.array([-e[1], e[0]])
            g = (centre - r[a]) @ e
            if abs(g) < 1e-12 * max(1.0, e @ e):
                return float(terms[a]), slope
            d = -np.sign(g) * e
        rate = -(r - r[a]) @ d
        rate[active] = 0.0
        blocking = rate > 1e-12
        if not blocking.any():
            raise ValueError("the machine's centre of mass lies outside its tracks here")
        t = np.where(blocking, slack / np.where(blocking, rate, 1.0), np.inf)
        first = t.min()
        ties = np.flatnonzero(t <= first + 1e-12)
        nxt = int(ties[np.argmax(rate[ties])])
        slope = slope + first * d
        active.append(nxt)
    raise RuntimeError("no resting pose found")


def _machine_com(model: mujoco.MjModel, data: mujoco.MjData, base: int) -> np.ndarray:
    bodies = np.flatnonzero(model.body_rootid == base)
    mass = model.body_mass[bodies]
    return mass @ data.xipos[bodies] / mass.sum()


def place(model: mujoco.MjModel, data: mujoco.MjData, index: int, x: float = 0.0, y: float = 0.0,
          heading: float = 0.0) -> None:
    from hma.machine.model import ANCHOR, BASE_JOINT, machine_prefix

    lay = layout(model)
    base = int(lay.base[index])
    q0 = int(model.joint(machine_prefix(index) + BASE_JOINT).qposadr[0])
    v0 = int(model.joint(machine_prefix(index) + BASE_JOINT).dofadr[0])
    xy = np.array([x, y]) + _machine_offset(model, index)
    turn = np.array([[math.cos(heading), -math.sin(heading), 0.0],
                     [math.sin(heading), math.cos(heading), 0.0], [0.0, 0.0, 1.0]])
    support = SAMPLE_POINTS[:N_FLAT]
    data.qpos[q0:q0 + 3] = (*xy, 0.0)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, turn.ravel())
    data.qpos[q0 + 3:q0 + 7] = quat
    mujoco.mj_kinematics(model, data)
    com = _machine_com(model, data, base) - data.xpos[base]
    com_local = data.xmat[base].reshape(3, 3).T @ com
    rot = turn
    z = 0.0
    machine = np.array([index])
    for _ in range(6):
        offsets = support @ rot.T
        flat = offsets @ turn
        h = ground_height(lay, machine, (xy[0] + offsets[:, 0])[None],
                          (xy[1] + offsets[:, 1])[None])
        c = (rot @ com_local) @ turn
        z, (p, q) = rest_plane(flat[:, :2], h[0], c[:2])
        bx = np.array([1.0, 0.0, p]) / math.hypot(1.0, p)
        bz = np.array([-p, -q, 1.0]) / math.sqrt(1.0 + p * p + q * q)
        rot = turn @ np.column_stack([bx, np.cross(bz, bx), bz])
    found = contacts_at(lay, machine, np.array([[*xy, z]]), rot[None])
    z += float(found.depth.max(initial=0.0))
    mujoco.mju_mat2Quat(quat, rot.ravel())
    data.qpos[q0:q0 + 3] = (*xy, z)
    data.qpos[q0 + 3:q0 + 7] = quat
    data.qvel[v0:v0 + 6] = 0.0
    if data.eq_active[lay.weld[index]]:
        mocap = model.body_mocapid[model.body(machine_prefix(index) + ANCHOR).id]
        data.mocap_pos[mocap] = (*xy, z)
        data.mocap_quat[mocap] = quat
    mujoco.mj_kinematics(model, data)


def _machine_offset(model: mujoco.MjModel, index: int) -> np.ndarray:
    from hma.machine.model import ANCHOR, machine_prefix

    body = model.body(machine_prefix(index) + ANCHOR)
    return model.body_pos[body.id][:2].copy()
