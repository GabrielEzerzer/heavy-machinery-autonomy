import math

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, circuit
from hma.hydraulics import layer as oil_layer
from hma.hydraulics.layer import DISPLACEMENT
from hma.machine import (
    HOME_KEY,
    JOINT_LIMITS,
    JOINT_NAMES,
    TIP_SITE,
    ground_axes,
    ground_shape,
    load_scene,
    machine_prefix,
    write_ground,
)
from hma.machine.collision import LINK_MESH, PIECES, link_triangles
from hma.machine.cylinders import seal_friction
from hma.machine.contact import STEEL_FRICTION_RANGE, draw_friction, set_friction
from hma.machine.load import (
    LEG_WLL,
    RANGES,
    SLING_WLL,
    apply_load,
    block,
    draw_load,
    hang_load,
    hook_force,
    hook_load,
    leg_tensions,
    place_load,
    sling_tension,
    unhook_load,
)
from hma.machine.model import (
    CONTACT_MARGIN,
    CONTACT_SOLIMP,
    CONTACT_SOLREF,
    LOAD,
    LOAD_DENSITY,
    MACHINE_SPACING,
    RING,
    RING_MASS,
    TRACK_FEET,
    ground_geoms,
    ring_at_load,
)
from hma.machine.render import PARTS
from scenes import held, lock_joints

P = machine_prefix(0)
G = 9.81
GRAVITY_BODIES = ("boom", "arm", "bucket")
F6 = np.zeros(6)
LOWER = 0.3
SERVO_GAIN = 1e8


def home(n=1, weld=False):
    model = load_scene(n, weld=weld)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    return model, data


def pose(model, data, angles, index=0):
    for name, value in zip(JOINT_NAMES, angles, strict=True):
        data.qpos[model.joint(machine_prefix(index) + name).qposadr[0]] = value
    mujoco.mj_forward(model, data)


def command(model, data, name, value, index=0):
    layer = held(model, data)
    u = layer._held.copy()
    u[index, JOINT_NAMES.index(name)] = value
    layer.act(u)


def run(model, data, seconds, each=None):
    layer = held(model, data)
    for _ in range(round(seconds / model.opt.timestep)):
        layer.step()
        if each is not None:
            each()


def oil_torque(model, data, name, index=0):
    layer = held(model, data)
    _, lever = circuit.geometry(data.qpos[layer._qpos])
    force = -(layer._pressure @ DISPLACEMENT) * layer._torque_scale
    return float((force * lever)[index, JOINT_NAMES.index(name)])


def seal_torque(model, data, name, index=0):
    layer = held(model, data)
    _, lever = circuit.geometry(data.qpos[layer._qpos])
    chambers = layer.chamber_pressures()[index]
    speed = lever[index] * data.qvel[layer._qvel[index]]
    joint = JOINT_NAMES.index(name)
    return float(seal_friction(chambers[1:, 0], chambers[1:, 1], speed[1:])[joint - 1]
                 * abs(lever[index, joint]))


def at_relief(model, data, index=0):
    return held(model, data).pump_pressures()[index].max() > NOMINAL.main_relief


def below_port_relief(model, data, name, index=0):
    chambers = held(model, data).chamber_pressures()[index, JOINT_NAMES.index(name)]
    return bool(np.all(chambers < 0.99 * NOMINAL.port_relief[JOINT_NAMES.index(name)]))


def body_contacts(model, data, body):
    bid = model.body(P + body).id
    out = []
    for i, c in enumerate(data.contact[:data.ncon]):
        if bid in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
            mujoco.mj_contactForce(model, data, i, F6)
            frame = c.frame.reshape(3, 3)
            force = frame.T @ F6[:3]
            if model.geom_bodyid[c.geom1] == bid:
                force = -force
            out.append((c.pos.copy(), force, c.dist))
    return out


def resultant(contacts):
    if not contacts:
        return np.zeros(3), np.zeros(3)
    pos = np.array([c[0] for c in contacts])
    force = np.array([c[1] for c in contacts])
    weights = np.abs(force[:, 2])
    centre = (weights @ pos) / max(weights.sum(), 1e-12)
    return force.sum(0), centre


def gravity_moment(model, data, bodies, axis_x):
    return sum(model.body_mass[model.body(P + b).id] * G
               * (data.xipos[model.body(P + b).id][0] - axis_x) for b in bodies)


def pivot_x(model, data, joint):
    return float(data.xanchor[model.joint(P + joint).id][0])


def lower_onto_ground(model, data, value=LOWER, seconds=6.0):
    command(model, data, "boom", value)
    run(model, data, seconds)
    command(model, data, "boom", 1.0)
    run(model, data, 1.5)


def test_pressed_bucket_ground_reaction_matches_statics_at_the_boom_cap():
    model, data = home(weld=True)
    set_friction(model, 0, 0.0)
    pose(model, data, (0.0, -0.5, 1.6, 1.0))
    lower_onto_ground(model, data)
    run(model, data, 1.0)
    force, centre = resultant(body_contacts(model, data, "bucket"))
    xp = pivot_x(model, data, "boom")
    tau = abs(oil_torque(model, data, "boom")) - seal_torque(model, data, "boom")
    assert at_relief(model, data)
    for joint in ("arm", "bucket"):
        assert below_port_relief(model, data, joint)
    hand = (tau + gravity_moment(model, data, GRAVITY_BODIES, xp)) / (centre[0] - xp)
    assert force[2] == pytest.approx(hand, rel=0.01)


def test_propped_machine_bucket_reaction_matches_statics():
    model, data = home()
    lower_onto_ground(model, data, LOWER, 4.0)
    command(model, data, "boom", 0.0)
    run(model, data, 3.0)
    base = model.body(P + "base").id
    up = data.xmat[base].reshape(3, 3)[:, 2]
    assert math.degrees(math.acos(up[2])) > 3.0
    bucket, xb = resultant(body_contacts(model, data, "bucket"))
    tracks, xr = resultant(body_contacts(model, data, "base"))
    weight = data.subtree_com[base], model.body_subtreemass[base] * G
    hand = weight[1] * (weight[0][0] - xr[0]) / (xb[0] - xr[0])
    assert bucket[2] == pytest.approx(hand, rel=0.01)
    assert bucket[2] + tracks[2] == pytest.approx(weight[1], rel=0.01)


def propping(boom, arm, seconds):
    model, data = home()
    pose(model, data, (0.0, -0.5, 1.6, 1.0))
    command(model, data, "boom", boom)
    command(model, data, "arm", arm)
    layer = held(model, data)
    base = model.body(P + "base").id
    tip = model.site(P + TIP_SITE).id
    weight = model.body_subtreemass[base] * G
    rows = []

    def moments(body):
        contacts = body_contacts(model, data, body)
        if not contacts:
            return np.zeros(3), np.zeros(3)
        force = np.array([c[1] for c in contacts])
        weights = np.abs(force[:, 2])
        return force.sum(0), np.r_[weights @ np.array([c[0] for c in contacts])[:, [0, 2]],
                                   weights.sum()]

    def record():
        (fb, mb), (_, mr) = moments("bucket"), moments("base")
        pressure = max(layer.pump_pressures()[0].max() - NOMINAL.main_relief,
                       (layer.chamber_pressures()[0] - NOMINAL.port_relief).max())
        rows.append((data.time, fb[0], fb[2], *mb, *mr, data.subtree_com[base][0],
                     data.site_xpos[tip][2], pressure))

    run(model, data, seconds, record)
    return np.array(rows), weight


def periods(rows, start, steps):
    kept = rows[rows[:, 0] > start]
    return kept[:len(kept) // steps * steps].reshape(-1, steps, rows.shape[1])


def reaction_over_hand(rows, weight, start, steps):
    w = periods(rows, start, steps).sum(1)
    fx, fz, gx = w[:, 1], w[:, 2], w[:, 9] / steps
    bx, bz, rx, rz = w[:, 3] / w[:, 5], w[:, 4] / w[:, 5], w[:, 6] / w[:, 8], w[:, 7] / w[:, 8]
    return fz / ((weight * steps * (gx - rx) + fx * (bz - rz)) / (bx - rx))


def test_propped_machine_whose_tip_holds_keeps_contact_and_matches_statics():
    rows, weight = propping(0.6, -0.15, 4.0)
    landed = rows[rows[:, 0] > 1.0]
    assert np.all(landed[:, 2] > 0.0)
    assert np.ptp(landed[:, 10]) < 0.005
    assert np.abs(reaction_over_hand(rows, weight, 1.0, 20) - 1.0).max() < 0.02


@pytest.mark.parametrize(("boom", "until", "mean_bound", "rms_bound", "ripple_bound"),
                         [(0.6, 8.0, 0.03, 0.75e-3, 3.5e-3), (1.0, 3.3, 0.07, 1.5e-3, 7.0e-3)])
def test_propped_machine_whose_tip_slides_hops_within_its_measured_bounds(
        boom, until, mean_bound, rms_bound, ripple_bound):
    rows, weight = propping(boom, 0.0, until)
    start = 2.0 if boom < 1.0 else 1.0
    assert np.abs(reaction_over_hand(rows, weight, start, 100) - 1.0).max() < mean_bound
    tip = periods(rows, start, 20)[..., 10]
    steps = np.arange(20) - 9.5
    slope = (tip * steps).sum(1, keepdims=True) / (steps * steps).sum()
    ripple = tip - tip.mean(1, keepdims=True) - slope * steps
    assert np.sqrt((ripple ** 2).mean()) < rms_bound
    assert np.ptp(ripple, 1).max() < ripple_bound
    assert periods(rows, start, 1)[..., 11].max() < -oil_layer.RELIEF_BAND


def rigid(model, data, seconds, each=None):
    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if each is not None:
            each()


def servo(model, data, name, speed):
    act = model.actuator(P + name).id
    model.actuator_gainprm[act, 0] = SERVO_GAIN
    model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_biasprm[act, :3] = [0.0, 0.0, -SERVO_GAIN]
    data.ctrl[act] = speed


def press(model, data, newtons, lever):
    xp = pivot_x(model, data, "boom")
    hold = gravity_moment(model, data, GRAVITY_BODIES, xp) - newtons * lever
    data.ctrl[model.actuator(P + "boom").id] = -hold
    return xp, hold


def pressed_by_torque(model, data, newtons, mu):
    set_friction(model, 0, mu)
    pose(model, data, (0.0, -0.5, 1.6, 1.0))
    lock_joints(model, data, names=("arm", "bucket"))
    servo(model, data, "swing", 0.0)
    tip = data.site_xpos[model.site(P + TIP_SITE).id]
    press(model, data, newtons, tip[0] - pivot_x(model, data, "boom"))
    rigid(model, data, 2.0)
    _, centre = resultant(body_contacts(model, data, "bucket"))
    xp, hold = press(model, data, newtons, centre[0] - pivot_x(model, data, "boom"))
    rigid(model, data, 1.0)
    _, centre = resultant(body_contacts(model, data, "bucket"))
    return (gravity_moment(model, data, GRAVITY_BODIES, xp) - hold) / (centre[0] - xp)


def drag(model, data, joint, speed, seconds):
    servo(model, data, joint, speed)
    act = model.actuator(P + joint).id
    samples = []

    def sample():
        f, c = resultant(body_contacts(model, data, "bucket"))
        samples.append((f[2], math.hypot(f[0], f[1]), abs(data.actuator_force[act]),
                        math.hypot(*c[:2])))

    rigid(model, data, seconds, sample)
    return np.array(samples[len(samples) // 4:])


def test_dragged_bucket_slides_at_the_friction_force():
    model, data = home(weld=True)
    mu = 0.4
    normal = pressed_by_torque(model, data, 8000.0, mu)
    s = drag(model, data, "swing", 0.02, 4.0)
    radius = s[s[:, 0] > 0, 3].mean()
    assert s[:, 0].mean() == pytest.approx(normal, rel=0.10)
    assert s[:, 1].mean() == pytest.approx(mu * normal, rel=0.10)
    assert s[:, 2].mean() == pytest.approx(mu * normal * radius, rel=0.10)


def hung(mass, angles=(0.0, -1.0, 1.6, 2.3)):
    model, data = home(weld=True)
    apply_load(model, 0, block(mass), mujoco.MjData(model))
    pose(model, data, angles)
    hang_load(model, data, 0)
    return model, data


def hung_on_servos(mass):
    model, data = hung(mass)
    for joint in JOINT_NAMES:
        servo(model, data, joint, 0.0)
    rigid(model, data, 3.0)
    return model, data


def test_hanging_load_swings_at_the_compound_pendulum_period():
    model, data = hung(2000.0)
    run(model, data, 2.0)
    load = model.body(P + "load").id
    hook = model.site(P + "hook").id
    geom = model.geom(P + "load").id
    half = model.geom_size[geom]
    mass = model.body_mass[load]
    d = np.linalg.norm(data.site_xpos[hook] - data.xipos[load])
    inertia = mass / 12.0 * ((2 * half[0]) ** 2 + (2 * half[2]) ** 2)
    hand = 2 * math.pi * math.sqrt((inertia + mass * d * d) / (mass * G * d))
    v0 = model.jnt_dofadr[model.joint(P + "load").id]
    data.qvel[v0 + 4] = 0.05 * 2 * math.pi / hand
    data.qvel[v0 + 0] = data.qvel[v0 + 4] * d
    xs, ts = [], []
    run(model, data, 6 * hand, lambda: (xs.append(data.xipos[load][0] - data.site_xpos[hook][0]),
                                        ts.append(data.time)))
    xs = np.array(xs) - np.mean(xs)
    up = np.where((xs[:-1] < 0) & (xs[1:] >= 0))[0]
    period = np.diff(np.array(ts)[up]).mean()
    assert period == pytest.approx(hand, rel=0.01)


def test_joint_loads_with_a_hanging_load_match_statics():
    model, data = hung_on_servos(3000.0)
    hook = data.site_xpos[model.site(P + "hook").id]
    weight = (model.body_mass[model.body(P + "load").id] + RING_MASS) * G
    block_weight = model.body_mass[model.body(P + "load").id] * G
    assert sling_tension(model, data, 0) == pytest.approx(
        block_weight / math.cos(math.radians(30.0)), rel=0.01)
    chain = {"boom": ("boom", "arm", "bucket"), "arm": ("arm", "bucket"), "bucket": ("bucket",)}
    for joint, bodies in chain.items():
        x = pivot_x(model, data, joint)
        hand = gravity_moment(model, data, bodies, x) + weight * (hook[0] - x)
        sim = data.actuator_force[model.actuator(P + joint).id]
        assert abs(sim) == pytest.approx(abs(hand), rel=0.01)



def test_hung_load_on_the_oil_reads_its_sling_tension_on_the_instant():
    model, data = hung(3000.0)
    run(model, data, 6.0)
    block_weight = model.body_mass[model.body(P + "load").id] * G
    for _ in range(5):
        assert sling_tension(model, data, 0) == pytest.approx(
            block_weight / math.cos(math.radians(30.0)), rel=0.01)
        run(model, data, 1.0)


def bounce(model, data, layer, seconds):
    static = model.body_mass[model.body(P + "load").id] * G / math.cos(math.radians(30.0))
    swing, boom_seals = [], []
    for _ in range(round(seconds / model.opt.timestep)):
        layer.step()
        swing.append(sling_tension(model, data, 0) / static - 1.0)
        _, lever = circuit.geometry(data.qpos[layer._qpos[0]])
        chambers = layer.chamber_pressures()[0]
        speed = lever * data.qvel[layer._qvel[0]]
        boom_seals.append(seal_friction(chambers[1:, 0], chambers[1:, 1], speed[1:])[0]
                          * abs(lever[1]))
    swing = np.abs(swing)
    peaks = np.flatnonzero((swing[1:-1] >= swing[:-2]) & (swing[1:-1] > swing[2:])) + 1
    t = (peaks + 1) * model.opt.timestep
    kept = (t > 1.0) & (t < 3.3) & (swing[peaks] > 0.01)
    return np.polyfit(t[kept], swing[peaks][kept], 1)[0], np.median(np.diff(t[kept])), np.mean(
        boom_seals[200:660])


def test_hung_load_bounce_dies_at_the_rate_the_boom_seals_give_by_hand(monkeypatch):
    with monkeypatch.context() as patch:
        patch.setattr(oil_layer, "seal_friction", lambda head, rod, speed: np.zeros(np.shape(head)))
        model, data = hung(3000.0)
        oil_only, half_period, _ = bounce(model, data, held(model, data), 3.5)
    model, data = hung(3000.0)
    layer = held(model, data)
    with_seals, _, seals = bounce(model, data, layer, 3.5)
    mass = np.zeros((model.nv, model.nv))
    mujoco.mj_fullM(model, data, mass)
    boom = layer._qvel[0, 1]
    reach = data.site_xpos[model.site(P + "hook").id][0] - pivot_x(model, data, "boom")
    inertia = mass[boom, boom] + 3000.0 * reach ** 2
    hand = 4.0 * abs(reach) * seals / (inertia * G) / (2.0 * half_period)
    assert oil_only < 0.0
    assert hand <= oil_only - with_seals <= 1.25 * hand


def test_sling_readings_match_statics_and_a_snatch_reads_over_the_rating():
    model, data = hung_on_servos(5000.0)
    block_weight = model.body_mass[model.body(P + "load").id] * G
    legs = leg_tensions(model, data, 0)
    assert legs == pytest.approx(block_weight / 4 / math.cos(math.radians(30.0)), rel=0.01)
    assert hook_force(model, data, 0) == pytest.approx(block_weight + RING_MASS * G, rel=0.01)
    assert legs.max() < LEG_WLL and hook_force(model, data, 0) < SLING_WLL
    v0 = model.jnt_dofadr[model.joint(P + LOAD).id]
    r0 = model.jnt_dofadr[model.joint(P + RING + "_x").id]
    data.qvel[[v0 + 2, r0 + 2]] = -3.0
    peaks = []
    rigid(model, data, 1.0, lambda: peaks.append((leg_tensions(model, data, 0).max(),
                                                  hook_force(model, data, 0))))
    leg, hook = np.max(peaks, axis=0)
    assert leg > LEG_WLL and hook > SLING_WLL
    assert np.linalg.norm(data.qvel[v0:v0 + 3]) < 0.5

AUDIT_POSES = [
    ("base", (19.6, 152.0, 64.8)),
    ("upper", (-70.0, 152.0, 84.3)),
    ("boom", (-21.1, 152.0, 123.4)),
]


def pieces(model, link, index=0):
    prefix = machine_prefix(index)
    return [model.geom(f"{prefix}{link}_{name}").id for name in PIECES[link]]


def piece_distance(model, data, a, b):
    return min(mujoco.mj_geomDistance(model, data, x, y, 1.0, None)
               for x in pieces(model, a) for y in pieces(model, b))


@pytest.mark.parametrize(("other", "degrees"), AUDIT_POSES)
def test_bucket_clears_where_whole_link_hulls_overlapped(other, degrees):
    model, data = home(weld=True)
    pose(model, data, (0.0, *np.radians(degrees)))
    hull = mujoco.mj_geomDistance(model, data, model.geom(P + "bucket").id,
                                  model.geom(P + LINK_MESH[other]).id, 1.0, None)
    assert hull < -0.1
    assert piece_distance(model, data, "bucket", other) > 0.05


def deepest(model, data):
    load = model.body(P + LOAD).id
    return min([c.dist for c in data.contact[:data.ncon]
                if load not in model.body_rootid[model.geom_bodyid[[c.geom1, c.geom2]]]],
               default=0.0)


def test_bucket_swung_over_the_track_corner_stops_on_it():
    model, data = home(weld=True)
    write_ground(model, 0, np.full(ground_shape(), -5.0))
    pose(model, data, np.radians((45.0, -3.5, 100.0, 40.3)))
    command(model, data, "arm", 1.0)
    worst = [0.0]
    run(model, data, 4.0, lambda: worst.append(deepest(model, data)))
    arm = data.qpos[model.joint(P + "arm").qposadr[0]]
    assert arm < JOINT_LIMITS["arm"][1] - 0.1
    assert at_relief(model, data)
    touching = {(model.body(model.geom_bodyid[c.geom1]).name,
                 model.body(model.geom_bodyid[c.geom2]).name) for c in data.contact[:data.ncon]}
    assert (P + "base", P + "bucket") in touching
    assert min(worst) > -0.002


@pytest.mark.parametrize("swing", [0.0, 45.0, 90.0, 135.0, 180.0])
def test_driving_every_joint_end_to_end_passes_nothing_through_anything(swing):
    model, data = home(weld=True)
    pose(model, data, (math.radians(swing), -0.6, 1.9, 0.6))
    periods = {"swing": 7.0, "boom": 5.0, "arm": 3.0, "bucket": 2.0}
    tip = model.site(P + TIP_SITE).id
    layer = held(model, data)
    worst, fastest, velocity = 0.0, 0.0, np.zeros(6)
    for k in range(round(15.0 / model.opt.timestep)):
        t = k * model.opt.timestep
        layer.act(np.array([[math.copysign(1.0, math.sin(2 * math.pi * t / period))
                             for period in periods.values()]]))
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, tip, velocity, 0)
        fastest = max(fastest, np.linalg.norm(velocity[3:]))
        layer.step()
        worst = min(worst, deepest(model, data))
    assert -worst <= fastest * CONTACT_SOLREF[0] / math.e


def ground_contacts(model, data, link):
    ground = set(ground_geoms(model, 0).tolist())
    body = model.body(P + link).id
    return [c for c in data.contact[:data.ncon]
            if {c.geom1, c.geom2} & ground and body in model.geom_bodyid[[c.geom1, c.geom2]]]


@pytest.mark.parametrize(("link", "start", "edges"), [
    ("bucket", (0.0, -1.0, 0.6, -0.3), (2.5, 10.0, 2.0)),
    ("arm", (0.0, -1.0, 2.6, 2.4), (3.0, 10.0, 2.0)),
    ("boom", (0.0, -1.0, 0.6, 0.0), (1.8, 2.6, 2.0)),
])
def test_each_front_link_lowered_onto_raised_ground_rests_on_it(link, start, edges):
    model, data = home(weld=True)
    xs, _ = ground_axes()
    heights = np.full(ground_shape(), -5.0)
    heights[:, xs > edges[0]] = edges[2]
    heights[:, xs >= edges[1]] = -5.0
    write_ground(model, 0, heights)
    pose(model, data, start)
    lower_onto_ground(model, data, LOWER, 8.0)
    touching = []
    run(model, data, 1.0, lambda: touching.append(ground_contacts(model, data, link)))
    assert np.mean([bool(c) for c in touching]) > 0.9
    assert min(c.dist for step in touching for c in step) > -0.002
    assert at_relief(model, data)


def test_ridge_higher_than_the_ground_clearance_high_centres_the_belly():
    model, data = home()
    xs, ys = ground_axes()
    ridge = 0.6
    heights = np.zeros(ground_shape())
    heights[np.ix_(np.abs(ys) < 0.15, np.abs(xs) < 3.0)] = ridge
    write_ground(model, 0, heights)
    data.qpos[model.joint(P + "base_free").qposadr[0] + 2] = ridge - 0.40
    run(model, data, 3.0)
    base = model.body(P + "base").id
    belly, _ = resultant(body_contacts(model, data, "base"))
    crawler = link_triangles("base").reshape(-1, 3)
    clearance = crawler[np.abs(crawler[:, 1]) < 0.15, 2].min()
    assert data.xpos[base][2] == pytest.approx(ridge - clearance, abs=0.02)
    assert belly[2] == pytest.approx(model.body_subtreemass[base] * G, rel=0.01)
    for c in data.contact[:data.ncon]:
        names = model.geom(c.geom1).name + model.geom(c.geom2).name
        assert "track_" not in names


def test_swung_counterweight_stops_against_a_mound_beside_the_tracks():
    model, data = home(weld=True)
    xs, ys = ground_axes()
    heights = np.zeros(ground_shape())
    heights[np.ix_(np.abs(ys) > 1.85, np.abs(xs) < 1.0)] = 1.2
    write_ground(model, 0, heights)
    pose(model, data, (0.0, -0.8, 1.2, 1.0))
    command(model, data, "swing", 1.0)
    run(model, data, 4.0)
    swing = data.qpos[model.joint(P + "swing").qposadr[0]]
    assert swing < math.radians(45.0)
    rear = model.geom(P + "upper_rear").id
    ground = set(ground_geoms(model, 0).tolist())
    touching = [c for c in data.contact[:data.ncon]
                if rear in (c.geom1, c.geom2) and {c.geom1, c.geom2} & ground]
    assert touching and min(c.dist for c in touching) > -0.002
    assert at_relief(model, data)


def surface_drag(slope_deg, speed, mu=0.4):
    model, data = home(weld=True)
    xs, _ = ground_axes()
    rise = math.tan(math.radians(slope_deg))
    write_ground(model, 0, np.zeros(ground_shape()) + rise * (xs[None, :] - 5.0))
    up = np.array([-rise, 0.0, 1.0]) / math.hypot(rise, 1.0)
    pressed_by_torque(model, data, 8000.0, mu)
    servo(model, data, "swing", speed)
    bucket = model.body(P + "bucket").id
    sums = np.zeros(3)

    def sample():
        for i, c in enumerate(data.contact[:data.ncon]):
            if bucket in (model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]):
                mujoco.mj_contactForce(model, data, i, F6)
                force = c.frame.reshape(3, 3).T @ F6[:3]
                normal = force @ up
                tilted = abs(c.frame[:3] @ up) < math.cos(math.radians(5.0))
                sums[:] += (abs(normal), np.linalg.norm(force - normal * up), tilted * F6[0])

    rigid(model, data, 1.0)
    rigid(model, data, 3.0, sample)
    return sums[1] / sums[0], sums[2] / sums[0]


@pytest.mark.parametrize("slope", [0.0, 10.0])
def test_slow_drag_slides_at_mu_on_flat_and_sloped_ground(slope):
    ratio, tilted = surface_drag(slope, 0.02)
    assert ratio == pytest.approx(0.4, rel=0.10)
    assert tilted < 0.15


def test_bucket_driven_down_at_full_speed_stops_at_the_surface():
    model, data = home(weld=True)
    pose(model, data, (0.8, -1.0, 1.0, 0.3))
    command(model, data, "boom", 1.0)
    layer = held(model, data)
    tip = model.site(P + TIP_SITE).id
    worst, fastest, velocity = 0.0, 0.0, np.zeros(6)
    for _ in range(round(4.0 / model.opt.timestep)):
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, tip, velocity, 0)
        fastest = max(fastest, np.linalg.norm(velocity[3:]))
        layer.step()
        worst = min(worst, deepest(model, data))
    assert -worst <= fastest * CONTACT_SOLREF[0] / math.e
    assert ground_contacts(model, data, "bucket")
    assert deepest(model, data) > -0.002


def load_energy(model, data, body):
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 1)
    spin = data.ximat[body].reshape(3, 3).T @ velocity[:3]
    mass = model.body_mass[body]
    return (mass * G * data.xipos[body][2] + 0.5 * mass * velocity[3:] @ velocity[3:]
            + 0.5 * model.body_inertia[body] @ spin ** 2)


def resting_load(mass):
    model = load_scene(1, weld=True)
    apply_load(model, 0, block(mass), mujoco.MjData(model))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    mujoco.mj_forward(model, data)
    return model, data, model.body(P + LOAD).id, model.geom_size[model.geom(P + LOAD).id]


LANDING_SPEED = 5.5


def land(tilt, phase):
    model, data, body, half = resting_load(3000.0)
    rest = data.xipos[body][2]
    q0, v0 = model.joint(P + LOAD).qposadr[0], model.joint(P + LOAD).dofadr[0]
    lift = 0.5 + phase * LANDING_SPEED * model.opt.timestep
    data.qpos[q0 + 2] += lift
    data.qpos[q0 + 3:q0 + 7] = [math.cos(tilt / 2), math.sin(tilt / 2), 0.0, 0.0]
    r0, w0 = model.joint(P + RING + "_x").qposadr[0], model.joint(P + RING + "_x").dofadr[0]
    data.qpos[r0:r0 + 3] = ring_at_load(model, data, 0)
    data.qvel[v0 + 2] = data.qvel[w0 + 2] = -math.sqrt(LANDING_SPEED ** 2 - 2 * G * lift)
    mujoco.mj_forward(model, data)
    bodies = (body, model.body(P + RING).id)
    energy, low = [sum(load_energy(model, data, b) for b in bodies)], [data.xipos[body][2]]

    def track():
        energy.append(sum(load_energy(model, data, b) for b in bodies))
        low.append(data.xipos[body][2])

    run(model, data, 3.0, track)
    energy = np.array(energy)
    weight = sum(model.body_mass[b] for b in bodies) * G
    held = weight * max(CONTACT_SOLIMP[2], rest - min(low))
    return (energy - np.minimum.accumulate(energy)).max(), held, model, data, body, half


@pytest.mark.parametrize("tilt", [0.0, 0.1, 0.3])
def test_load_landing_at_speed_gains_no_energy_and_comes_to_rest(tilt):
    for phase in np.linspace(0.0, 1.0, 6, endpoint=False):
        rise, held, model, data, body, half = land(tilt, phase)
        assert rise < held
        assert np.abs(data.cvel[body]).max() < 1e-3
    if tilt == 0.0:
        assert data.xipos[body][2] == pytest.approx(half[2] + CONTACT_MARGIN, abs=1e-5)
        assert data.xmat[body][8] > math.cos(math.radians(0.01))


def test_pressed_load_neither_sinks_nor_bounces_back_nor_creeps():
    model, data, body, _ = resting_load(3000.0)
    run(model, data, 0.5)
    rest = data.xipos[body][2]
    data.xfrc_applied[body, 2] = -20.0 * model.body_mass[body] * G
    run(model, data, 1.0)
    assert rest - data.xipos[body][2] < 1e-4
    data.xfrc_applied[body] = 0.0
    heights = []
    run(model, data, 10.0, lambda: heights.append(data.xipos[body][2]))
    assert max(heights) - rest < 1e-5
    assert abs(heights[-1] - heights[len(heights) // 10]) < 1e-6


LIFT_POSE = (0.0, -0.6, 1.6, 2.3)
FEATHER = -0.7


def lift_cycle(model, data, index=0, mass=2000.0):
    load = block(mass)
    apply_load(model, index, load, mujoco.MjData(model))
    pose(model, data, LIFT_POSE, index)
    hook = data.site_xpos[model.site(machine_prefix(index) + "hook").id]
    place_load(model, data, index, hook[0], hook[1] - index * MACHINE_SPACING)
    hook_load(model, data, index)
    trace = []

    def phase(joint, speed, seconds):
        command(model, data, joint, speed, index)
        run(model, data, seconds, lambda: trace.append(data.qpos.copy()))
        command(model, data, joint, 0.0, index)
        run(model, data, 1.0, lambda: trace.append(data.qpos.copy()))

    return load, trace, phase


@pytest.mark.parametrize("weld", [True, False])
def test_load_is_lifted_carried_swung_and_set_down_on_its_sling(weld):
    model, data = home(weld=weld)
    load, _, phase = lift_cycle(model, data)
    body = model.body(P + LOAD).id
    start = data.xpos[body].copy()
    phase("boom", FEATHER, 3.0)
    assert data.xpos[body][2] - start[2] > 1.5
    tension = []
    run(model, data, 3.0, lambda: tension.append(sling_tension(model, data, 0)))
    assert np.mean(tension) == pytest.approx(load.mass * G / math.cos(load.leg_angle), rel=0.01)
    phase("swing", 1.0, 2.0)
    phase("boom", LOWER, 5.0)
    assert sling_tension(model, data, 0) < 1.0
    unhook_load(model, data, 0)
    phase("boom", FEATHER, 2.0)
    moved = data.xpos[body] - start
    assert math.hypot(moved[0], moved[1]) > 6.0
    assert data.xpos[body][2] == pytest.approx(load.half[2] + CONTACT_MARGIN, abs=1e-5)
    assert np.abs(data.cvel[body]).max() < 1e-6


def test_same_draws_repeat_bit_for_bit():
    runs = []
    for _ in range(2):
        model, data = home(weld=True)
        apply_load(model, 0, draw_load(5), mujoco.MjData(model))
        mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
        pressed_by_torque(model, data, 8000.0, draw_friction(5))
        runs.append(drag(model, data, "swing", 0.05, 2.0))
        runs.append(data.qpos.copy())
    np.testing.assert_array_equal(runs[0], runs[2])
    np.testing.assert_array_equal(runs[1], runs[3])


def test_draws_lie_in_their_sourced_ranges():
    for seed in range(200):
        assert STEEL_FRICTION_RANGE[0] <= draw_friction(seed) <= STEEL_FRICTION_RANGE[1]
        load = draw_load(seed)
        assert RANGES["mass"][0] <= load.mass <= RANGES["mass"][1]
        assert RANGES["leg_angle"][0] <= load.leg_angle <= RANGES["leg_angle"][1]
        assert RANGES["friction"][0] <= load.friction <= RANGES["friction"][1]
        assert all(abs(c) <= 0.05 * 2 * h + 1e-12
                   for c, h in zip(load.com, load.half, strict=True))
        assert LOAD_DENSITY * np.prod(2 * np.array(load.half)) == pytest.approx(load.mass)


def test_machines_in_one_scene_never_collide_with_each_other():
    model = load_scene(3)
    owner = [next((i for i in range(3) if model.geom(g).name.startswith(machine_prefix(i))), None)
             for g in range(model.ngeom)]
    for a in range(model.ngeom):
        for b in range(a + 1, model.ngeom):
            if None not in (owner[a], owner[b]) and owner[a] != owner[b]:
                assert not (model.geom_contype[a] & model.geom_conaffinity[b]
                            or model.geom_contype[b] & model.geom_conaffinity[a])


def test_only_contact_shapes_collide_and_drawn_parts_are_no_geom():
    model = load_scene(2)
    expected = set()
    for i in range(2):
        p = machine_prefix(i)
        expected |= {f"{p}{link}_{name}" for link, names in PIECES.items() for name in names}
        expected |= {model.geom(g).name for g in ground_geoms(model, i)}
        expected |= {model.geom(g).name for g in range(model.ngeom)
                     if model.geom(g).name.startswith((p + TRACK_FEET, p + LOAD))}
    colliding = np.flatnonzero((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    assert {model.geom(g).name for g in colliding} == expected
    drawn = [model.mesh(machine_prefix(i) + part[0]).id for i in range(2) for part in PARTS]
    assert not np.isin(model.geom_dataid[model.geom_type == mujoco.mjtGeom.mjGEOM_MESH],
                       drawn).any()


@pytest.mark.parametrize("weld", [True, False])
def test_one_machine_working_in_contact_does_not_change_its_neighbour(weld):
    alone, data = home(1, weld=weld)
    _, solo, phase = lift_cycle(alone, data)
    phase("boom", FEATHER, 3.0)
    pair, data = home(2, weld=weld)
    pose(pair, data, (0.0, -0.5, 1.6, 1.0))
    _, shared, phase = lift_cycle(pair, data, index=1)
    command(pair, data, "boom", LOWER)
    command(pair, data, "swing", LOWER)
    phase("boom", FEATHER, 3.0)
    touched = []
    run(pair, data, 0.1, lambda: touched.append(bool(ground_contacts(pair, data, "bucket"))))
    assert any(touched)
    got, want = np.array(shared), np.array(solo)
    for j in range(alone.njnt):
        width = 7 if alone.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE else 1
        a = alone.jnt_qposadr[j]
        b = pair.joint(machine_prefix(1) + alone.joint(j).name[len(P):]).qposadr[0]
        mine = got[:, b:b + width].copy()
        if width == 7:
            mine[:, 1] -= MACHINE_SPACING
        np.testing.assert_allclose(mine, want[:, a:a + width], atol=1e-6)



def hull_planes(model, geom):
    mesh = model.geom_dataid[geom]
    adr = model.mesh_graphadr[mesh]
    nvert, nface = model.mesh_graph[adr:adr + 2]
    start = adr + 2 + 3 * nvert + 3 * nface
    faces = model.mesh_graph[start:start + 3 * nface].reshape(-1, 3)
    first = model.mesh_vertadr[mesh]
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
    verts = (model.mesh_vert[first:first + model.mesh_vertnum[mesh]] @ rot.reshape(3, 3).T
             + model.geom_pos[geom])
    tri = verts[faces]
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.linalg.norm(normal, axis=1)[:, None]
    offset = np.einsum("ij,ij->i", normal, tri[:, 0])
    inward = normal @ verts.mean(axis=0) > offset
    normal[inward], offset[inward] = -normal[inward], -offset[inward]
    support = (verts @ normal.T - offset).max(axis=0) < 1e-6
    return normal[support], offset[support]


def test_every_mesh_vertex_lies_inside_a_piece_except_the_documented_parts():
    model = load_scene(1)
    for link in PIECES:
        verts = np.unique(link_triangles(link).reshape(-1, 3), axis=0)
        outside = np.full(len(verts), np.inf)
        for geom in pieces(model, link):
            normal, offset = hull_planes(model, geom)
            outside = np.minimum(outside, (verts @ normal.T - offset).max(axis=1))
        bare = verts[outside > 0.01]
        if link == "upper":
            assert np.all(bare[:, 2] > 1.0)
        elif link == "base":
            assert np.all((bare[:, 2] > 0.8) & (np.hypot(bare[:, 0], bare[:, 1]) < 0.77))
        else:
            assert len(bare) == 0
