import math

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, Hydraulics
from hma.machine import (
    HOME_KEY,
    load_scene,
    machine_prefix,
    off_features,
    settle,
    tendon_friction,
)
from hma.machine.contact import STEEL_FRICTION_RANGE, set_friction
from hma.machine.ground import (
    RANGES,
    Box,
    Ground,
    apply_ground,
    draw_ground,
    inside,
    stand,
)
from hma.machine.measure import (
    CASES,
    TIPPED,
    base_frame,
    chart_pose,
    hand_tipping_load,
    machine_com,
    run,
    slope,
    tipping_case,
    tips_under,
)
from hma.machine.measure import (
    released as machine,
)
from hma.machine.model import GROUND, HOME_POSE, JOINT_NAMES, TRAVEL_NAMES, VALVE, ground_parts
from hma.machine.tracks import (
    FRICTION,
    FRICTION_RANGE,
    GAUGE,
    OUTLINE,
    SIDES,
    TRACTION,
    TRAVEL,
    TRAVEL_POWER,
    TRAVEL_SPEED,
    WHEEL_RADIUS,
    WHEEL_X,
    WHEELS,
    contact_names,
    drive_forces,
    drives,
    step,
    wheel_name,
)

P = machine_prefix(0)
G = 9.81
SINK = 1e-5


def energy(model, data):
    mujoco.mj_forward(model, data)
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    return float(data.energy.sum())


def travel(model):
    return [model.actuator(P + name).id for name in TRAVEL_NAMES]


def brakes(model):
    return np.array([model.tendon(P + TRAVEL + side + VALVE).id for side in SIDES])


def pitch(model, data):
    _, rot = base_frame(model, data)
    return math.degrees(math.atan2(-rot[2, 0], rot[2, 2]))


def deepest(data, geom=None):
    return max([-c.dist for c in data.contact[:data.ncon] if geom in (None, c.geom1, c.geom2)],
               default=0.0)


BESIDE = OUTLINE[3] + 0.03

PASSIVE = {
    "sideways into a 0.2 m kerb": (Ground(boxes=(Box(0.0, BESIDE + 0.15, 0.0, 3.0, 0.15, 0.2),)),
                                   1, 1.0, 0.0),
    "sideways into a 0.38 m step": (Ground(boxes=(Box(0.0, BESIDE + 3.0, 0.0, 3.0, 3.0, 0.38),)),
                                    1, 1.0, 0.0),
    "sliding along": (Ground(), 0, 1.0, 0.0),
    "landing from 5 cm moving sideways": (Ground(), 1, 0.2, 0.05),
    "counter-rotating beside a 0.4 m step": (
        Ground(boxes=(Box(0.0, BESIDE + 3.0, 0.0, 3.0, 3.0, 0.4),)), 5, 0.5, 0.0),
    "sliding down a slope": (slope(along=0.9 * math.atan(FRICTION)), 0, -1.0, 0.0),
}
DOWNHILL = {
    "sliding down a slope": 0.9 * math.atan(FRICTION),
}


@pytest.mark.parametrize("name", list(PASSIVE))
def test_passive_runs_never_gain_energy(name):
    ground, dof, speed, drop = PASSIVE[name]
    model, data = machine(ground)
    data.qpos[2] += drop
    data.qvel[dof] = speed
    if name in DOWNHILL:
        data.qvel[[0, 2]] = speed * math.cos(DOWNHILL[name]), speed * math.sin(DOWNHILL[name])
    mujoco.mj_forward(model, data)
    start = energy(model, data)
    rise, reversed_ = 0.0, 0.0
    for _ in range(round(3.0 / model.opt.timestep)):
        step(model, data)
        rise = max(rise, energy(model, data) - start)
        reversed_ = max(reversed_, -data.qvel[dof] * math.copysign(1.0, speed))
    assert rise < 1.0, rise
    if drop:
        assert reversed_ < 1e-3, reversed_


@pytest.mark.parametrize("dof", [0, 1], ids=["along", "across"])
def test_a_sliding_machine_stops_in_the_coulomb_distance(dof):
    model, data = machine()
    start = data.qpos[dof]
    data.qvel[dof] = TRAVEL_SPEED
    run(model, data, 2.0)
    hand = TRAVEL_SPEED**2 / (2 * FRICTION * G)
    assert (data.qpos[dof] - start) / hand == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("factor", [0.995, 1.005])
@pytest.mark.parametrize("name", list(CASES))
def test_tipping_load_is_within_half_a_percent_of_hand_statics(name, factor):
    ground, pose, hand = tipping_case(name)
    model, data = machine(ground, pose, hold=True)
    angle = tips_under(model, data, factor * hand)
    assert (angle > TIPPED) if factor > 1 else (angle < TIPPED), angle


def tipping_about(ground, pose, line, axis, factor, edit=None):
    model, data = machine(ground, pose, hold=True)
    if edit is not None:
        edit(model, data)
        mujoco.mj_forward(model, data)
    origin, rot = base_frame(model, data)
    hand = hand_tipping_load(model, data, origin + rot @ line, rot @ axis)
    return tips_under(model, data, factor * hand)


EDGES = {
    "across the tracks under a wheel": (
        Ground(boxes=(Box(WHEEL_X[-2] - 6.0, 0.0, 0.0, 6.0, 6.0, 1.0),)),
        chart_pose(1.5, 6.0), [WHEEL_X[-2], 0.0, 0.0], [0.0, 1.0, 0.0]),
    "along between the rails": (Ground(boxes=(Box(0.0, GAUGE / 2 - 3.0, 0.0, 6.0, 3.0, 1.0),)),
                                chart_pose(1.5, 4.5, math.pi / 2), [0.0, GAUGE / 2, 0.0],
                                [1.0, 0.0, 0.0]),
}


@pytest.mark.parametrize("factor", [0.98, 1.02])
@pytest.mark.parametrize("name", list(EDGES))
def test_a_ground_edge_under_a_track_becomes_the_tipping_line(name, factor):
    ground, pose, line, axis = EDGES[name]
    angle = tipping_about(ground, pose, line, axis, factor)
    assert (angle > TIPPED) if factor > 1 else (angle < TIPPED), angle


BETWEEN = (WHEEL_X[-2] + WHEEL_X[-1]) / 2


def between_wheels(factor, edit=None):
    return tipping_about(Ground(boxes=(Box(BETWEEN - 6.0, 0.0, 0.0, 6.0, 6.0, 1.0),)),
                         chart_pose(1.5, 6.0), [BETWEEN, 0.0, 0.0], [0.0, 1.0, 0.0], factor, edit)


def test_an_edge_between_two_wheels_holds_the_machine_to_0_93_of_the_hand_tipping_load():
    assert between_wheels(0.93) < TIPPED
    assert between_wheels(1.0) > TIPPED


def propped(mu):
    model, data = machine(pose=chart_pose(1.5, 5.0))
    set_friction(model, 0, mu)
    boom = model.actuator(P + "boom").id
    data.ctrl[boom] = 0.1
    for _ in range(round(8.0 / model.opt.timestep)):
        step(model, data)
        if pitch(model, data) < -25.0:
            break
    data.ctrl[boom] = 0.0
    run(model, data, 1.0)
    return model, data


@pytest.mark.parametrize("mu", STEEL_FRICTION_RANGE)
def test_a_machine_propped_on_its_bucket_stays_put_with_its_brakes_far_from_slipping(mu):
    model, data = propped(mu)
    assert -30.0 < pitch(model, data) < -20.0
    start = data.qpos[:3].copy()
    peak = 0.0
    for _ in range(round(4.0 / model.opt.timestep)):
        step(model, data)
        peak = max(peak, np.abs(tendon_friction(data, brakes(model))).max())
    assert peak < 0.1 * TRACTION
    assert np.linalg.norm(data.qpos[:3] - start) < 1e-3


def wall_push(edit=None):
    wall = Box(4.0, 0.0, 0.0, 0.5, 3.0, 1.2)
    model, data = machine(Ground(boxes=(wall,)))
    if edit is not None:
        edit(model, data)
    data.ctrl[travel(model)] = TRAVEL_SPEED
    run(model, data, 3.0)
    start = data.qpos[0]
    run(model, data, 3.0)
    weight = model.body_subtreemass[model.body(P + "base").id] * G
    pushes = []
    for _ in range(200):
        step(model, data)
        pushes.append(drive_forces(model, data)[0].sum())
    return data.qpos[0] - start, data.qpos[0] + OUTLINE[1] - wall.x + wall.half_length, (
        np.mean(pushes) / (FRICTION * weight))


def test_full_speed_into_a_wall_stalls_at_the_wall():
    moved, gap, push = wall_push()
    assert abs(moved) < 1e-2
    assert gap < SINK
    assert push == pytest.approx(1.0, rel=0.03)


def test_a_parked_machine_pushed_with_120_kn_does_not_pass_through_a_wall():
    wall = Box(0.0, BESIDE + 0.5, 0.0, 3.0, 0.5, 1.2)
    model, data = machine(Ground(boxes=(wall,)))
    data.qvel[1] = 0.2
    sink = 0.0
    for k in range(round(3.0 / model.opt.timestep)):
        data.xfrc_applied[model.body(P + "base").id, 1] = 120e3
        step(model, data)
        if k * model.opt.timestep > 1.0:
            sink = max(sink, deepest(data, ground_parts(model, 0)[1][0]))
    assert data.qpos[1] + OUTLINE[3] < BESIDE + SINK
    assert sink < SINK


def drive_onto_step(height, speed, depth):
    model, data = machine(Ground(boxes=(Box(2.5 + depth, 0.0, 0.0, depth, 6.0, height),)),
                          pose=None)
    data.ctrl[travel(model)] = speed
    steepest = 0.0
    for _ in range(round((5.0 / speed + 4.0) / model.opt.timestep)):
        step(model, data)
        steepest = max(steepest, -pitch(model, data))
        if data.qpos[0] > 2.5 + 2 * OUTLINE[1]:
            break
    data.ctrl[travel(model)] = 0.0
    run(model, data, 1.0)
    return model, data, steepest


@pytest.mark.parametrize("speed, depth", [(0.25, 5.0), (0.5, 5.0), (0.5, 3.0), (1.0, 5.0),
                                          (TRAVEL_SPEED, 5.0)])
def test_driving_onto_a_0_1_m_step_climbs_it_at_any_speed_and_pitches_with_the_ground(speed, depth):
    height = 0.1
    model, data, steepest = drive_onto_step(height, speed, depth)
    assert data.qpos[0] > 2.5 + OUTLINE[1]
    assert data.qpos[2] == pytest.approx(height, abs=1e-3)
    assert abs(pitch(model, data)) < 0.05
    if speed <= 0.5:
        _, com = machine_com(model, data)
        origin, rot = base_frame(model, data)
        lever = (rot.T @ (com - origin))[0] - WHEEL_X[0]
        assert steepest == pytest.approx(math.degrees(math.asin(height / lever)), rel=0.05)


def test_a_0_2_m_step_stops_a_crawling_machine_on_its_edge():
    model, data, _ = drive_onto_step(0.2, 0.25, 5.0)
    assert data.qpos[0] < 2.5 + OUTLINE[1]
    assert -pitch(model, data) > 3.0


SLOPE = math.radians(20.0)


def slope_speed():
    model, data = machine(slope(along=SLOPE))
    data.ctrl[travel(model)] = TRAVEL_SPEED
    run(model, data, 3.0)
    start, _ = base_frame(model, data)
    run(model, data, 1.0)
    return float(np.linalg.norm(base_frame(model, data)[0] - start))


def power_speed():
    model = load_scene(1)
    weight = model.body_subtreemass[model.body(P + "base").id] * G
    return TRAVEL_POWER / (weight * math.sin(SLOPE))


def turning(commands, seconds=3.0):
    model, data = machine()
    parts = {model.geom(P + name).id for side in SIDES for name in contact_names(side)}
    data.ctrl[travel(model)] = commands
    run(model, data, 1.5)
    yaw, belts, power, heights, rises, free = [], [], [], [], [], 0
    for _ in range(round(seconds / model.opt.timestep)):
        before = data.qvel[drives(model).dofs[0]] * WHEEL_RADIUS
        rise = data.qvel[2]
        step(model, data)
        yaw.append(math.degrees(data.qvel[5]))
        belts.append(data.qvel[drives(model).dofs[0]] * WHEEL_RADIUS)
        power.append(np.maximum(drive_forces(model, data)[0] * before, 0.0).sum())
        heights.append(data.qpos[2])
        rises.append((data.qvel[2] - rise) / model.opt.timestep)
        free += not any(c.dist < 0 and {c.geom1, c.geom2} & parts
                        for c in data.contact[:data.ncon])
    lift = (np.ptp(heights), free / len(heights), float(np.sqrt(np.mean(np.square(rises)))))
    return float(np.mean(yaw)), np.mean(belts, axis=0), np.array(power), lift


def test_a_pivot_turn_holds_the_inner_track_still_and_turns():
    yaw, belts, *_ = turning([TRAVEL_SPEED, 0.0])
    assert abs(belts[1]) < 0.01
    assert yaw < -2.0


@pytest.mark.parametrize("commands", [[TRAVEL_SPEED, 0.0], [TRAVEL_SPEED, -TRAVEL_SPEED]])
def test_sliding_tracks_lift_the_machine_within_known_bounds(commands):
    height, free, rise = turning(commands)[3]
    assert height < 2e-3
    assert free < 0.4
    assert rise < 10.0


def test_a_gentle_curve_steers():
    yaw, *_ = turning([TRAVEL_SPEED, 0.75 * TRAVEL_SPEED])
    assert yaw < -0.6


@pytest.mark.parametrize("share", [1 / 3, 1.0])
def test_counter_rotation_never_takes_more_than_the_engine_share(share):
    _, _, power, _ = turning([share * TRAVEL_SPEED, -share * TRAVEL_SPEED])
    assert power.max() <= TRAVEL_POWER * (1 + 1e-9)
    if share == 1.0:
        assert power.mean() > 0.9 * TRAVEL_POWER


def test_on_a_slope_the_speed_is_set_by_the_engine_power():
    assert slope_speed() == pytest.approx(power_speed(), rel=0.03)


def downhill_run(angle):
    model, data = machine(slope(along=angle))
    data.ctrl[travel(model)] = -TRAVEL_SPEED
    run(model, data, 5.0)
    speed = float(np.linalg.norm(data.qvel[:3]))
    start = data.qpos[:3].copy()
    data.ctrl[travel(model)] = 0.0
    run(model, data, 4.0)
    return speed, float(np.linalg.norm(data.qpos[:3] - start)), float(np.abs(data.qvel[:6]).max())


@pytest.mark.parametrize("degrees", [15.0, 20.0])
def test_full_speed_downhill_holds_the_commanded_speed_and_stops(degrees):
    speed, distance, left = downhill_run(math.radians(degrees))
    assert speed == pytest.approx(TRAVEL_SPEED, rel=0.005)
    assert distance < 1.0
    assert left < 1e-3


def test_braking_from_full_speed_on_flat_takes_the_friction_limited_distance():
    model, data = machine()
    data.ctrl[travel(model)] = TRAVEL_SPEED
    run(model, data, 3.0)
    start, speed = data.qpos[0], data.qvel[0]
    data.ctrl[travel(model)] = 0.0
    run(model, data, 2.0)
    assert (data.qpos[0] - start) / (speed**2 / (2 * FRICTION * G)) == pytest.approx(1.0, abs=0.05)


def test_a_track_in_the_air_leaves_the_other_its_full_traction():
    model = load_scene(1, weld=True)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    data.mocap_pos[0, 2] += 1.0
    data.qpos[2] += 1.0
    data.ctrl[travel(model)] = [TRAVEL_SPEED, 0.3]
    load = 0.98 * TRACTION
    data.qfrc_applied[model.joint(P + wheel_name("right", 1)).dofadr[0]] = -load * WHEEL_RADIUS
    step(model, data, round(4.0 / model.opt.timestep))
    belts = data.qvel[drives(model).dofs[0]] * WHEEL_RADIUS
    assert belts == pytest.approx([TRAVEL_SPEED, 0.3], rel=1e-3)
    assert drive_forces(model, data)[0, 1] == pytest.approx(load, rel=1e-3)


def test_positive_drive_power_never_exceeds_the_engine_share():
    rng = np.random.default_rng(2)
    model, data = machine(draw_ground(7))
    peak = 0.0
    for k in range(4000):
        if k % 100 == 0:
            data.ctrl[travel(model)] = rng.uniform(-1.0, 1.0, 2) * TRAVEL_SPEED
        belts = data.qvel[drives(model).dofs[0]] * WHEEL_RADIUS
        step(model, data)
        peak = max(peak, np.maximum(drive_forces(model, data)[0] * belts, 0.0).sum())
    assert TRAVEL_POWER * 0.99 < peak <= TRAVEL_POWER * (1 + 1e-9)


def test_rollouts_sharing_one_model_match_rollouts_on_their_own():
    ground = slope(along=math.radians(15.0))
    shared, _ = machine(ground)
    datas = [mujoco.MjData(shared) for _ in range(2)]
    own = [machine(ground) for _ in range(2)]
    for k, (d, (m, o)) in enumerate(zip(datas, own, strict=True)):
        mujoco.mj_copyData(d, shared, o)
        d.ctrl[travel(shared)] = o.ctrl[travel(m)] = [TRAVEL_SPEED, -TRAVEL_SPEED * k]
    fields = {name: getattr(shared, name).copy() for name in ("tendon_frictionloss", "dof_damping")}
    for _ in range(300):
        for d, (m, o) in zip(datas, own, strict=True):
            step(shared, d)
            step(m, o)
    for d, (_, o) in zip(datas, own, strict=True):
        assert np.array_equal(d.qpos, o.qpos)
    for name, value in fields.items():
        assert np.array_equal(getattr(shared, name), value)


def test_brakes_hold_on_a_slope_below_the_friction_angle():
    model, data = machine(slope(along=0.9 * math.atan(FRICTION)))
    start = data.qpos[:3].copy()
    run(model, data, 4.0)
    assert np.linalg.norm(data.qpos[:3] - start) < 1e-3


def test_a_machine_high_centred_on_a_pad_sits_still():
    model, data = machine(Ground(boxes=(Box(0.0, 0.0, 0.0, 0.4, 3.0, 0.3),)))
    start = data.qpos[:7].copy()
    run(model, data, 3.0)
    assert data.qpos[2] > 0.25
    assert np.abs(data.qpos[:7] - start).max() < 1e-3
    assert np.abs(data.qvel[:6]).max() < 1e-3


@pytest.mark.parametrize("speed", [0.23, 0.5, 1.0, 1.38])
def test_a_dragged_block_reads_the_ground_friction(speed):
    model = load_scene(1, weld=True)
    mu = 0.57
    apply_ground(model, 0, Ground(friction=mu))
    plane = model.geom(P + GROUND).id
    friction, solref, solimp = (" ".join(str(v) for v in getattr(model, k)[plane])
                                for k in ("geom_friction", "geom_solref", "geom_solimp"))
    block = mujoco.MjModel.from_xml_string(f"""<mujoco>
      <option timestep="{model.opt.timestep}" integrator="Euler" cone="elliptic"
              impratio="{model.opt.impratio}" tolerance="{model.opt.tolerance}"/>
      <worldbody>
        <geom type="plane" size="50 50 1" friction="{friction}" solref="{solref}"
              solimp="{solimp}"/>
        <body pos="0 0 0.25">
          <joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 0 1"/>
          <joint type="hinge" axis="0 1 0"/>
          <geom type="box" size="0.5 0.5 0.25" mass="1000" friction="0 0 0"/>
        </body>
      </worldbody></mujoco>""")
    data = mujoco.MjData(block)
    mujoco.mj_step(block, data, 200)
    drag = []
    for k in range(400):
        data.qvel[0] = speed
        mujoco.mj_step(block, data)
        if k >= 100:
            drag.append(-data.qfrc_constraint[0])
    assert np.mean(drag) / (1000 * G) == pytest.approx(mu, rel=0.03)


def test_every_undercarriage_part_meets_the_ground_and_takes_the_drawn_friction():
    model = load_scene(1)
    plane, boxes, bumps = ground_parts(model, 0)
    ground = {plane, *boxes, *bumps}
    for side in SIDES:
        for part in contact_names(side):
            geom = model.geom(P + part).id
            pairs = [k for k in range(model.npair)
                     if geom in (model.pair_geom1[k], model.pair_geom2[k])]
            met = {int(model.pair_geom1[k] + model.pair_geom2[k] - geom) for k in pairs}
            assert met == ground, part
    apply_ground(model, 0, Ground(friction=0.57))
    assert np.all(model.pair_friction[:, :2] == 0.57)
    assert np.all(model.geom_friction[list(ground), 0] == 0.57)


def test_resets_on_drawn_grounds_start_at_rest_with_steady_load_readings():
    model = load_scene(1)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [NOMINAL])
    rng = np.random.default_rng(0)
    for seed in range(300):
        layer.reset(0, NOMINAL, at=(0.0, 0.0, float(rng.uniform(0.0, 2 * math.pi))),
                    ground=seed)
        first = layer.true_state().loads[0].copy()
        peak = 0.0
        for _ in range(20):
            layer.step()
            peak = max(peak, np.abs(data.qvel[:6]).max())
        assert peak < 2e-3, seed
        np.testing.assert_allclose(layer.true_state().loads[0], first, atol=0.01)


def test_a_refused_draw_leaves_no_drift_for_the_next_draw_to_start_from():
    model = load_scene(1)
    data = mujoco.MjData(model)
    joints = [model.joint(P + name).qposadr[0] for name in JOINT_NAMES]
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    apply_ground(model, 0, draw_ground(75, 0))
    assert not settle(model, data, 0)
    assert np.abs(data.qpos[joints] - HOME_POSE).max() > 0.1
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    stand(model, data, 0, 75)
    assert np.abs(data.qpos[joints] - HOME_POSE).max() < 1e-2


def test_restoring_the_integration_state_repeats_bit_for_bit():
    model, data = machine(draw_ground(4))
    data.ctrl[travel(model)] = [TRAVEL_SPEED, 0.3]
    step(model, data, 50)
    kind = mujoco.mjtState.mjSTATE_INTEGRATION
    saved = np.empty(mujoco.mj_stateSize(model, kind))
    mujoco.mj_getState(model, data, saved, kind)

    def later(target):
        mujoco.mj_setState(model, target, saved, kind)
        step(model, target, 100)
        return target.qpos.copy(), target.qvel.copy()

    first = later(data)
    again = later(data)
    fresh = later(mujoco.MjData(model))
    for a, b in ((first, again), (first, fresh)):
        assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def two_machines(busy):
    model = load_scene(2)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    for i in range(2):
        apply_ground(model, i, Ground())
        assert settle(model, data, i)
    rng = np.random.default_rng(1)
    other = [model.actuator(machine_prefix(1) + t).id for t in (*JOINT_NAMES, *TRAVEL_NAMES)]
    for _ in range(200):
        if busy:
            data.ctrl[other] = rng.uniform(-1.0, 1.0, len(other)) * model.actuator_ctrlrange[
                other, 1]
        step(model, data, 20)
    return data.qpos[:7].copy()


def test_one_machine_barely_moves_another_in_the_same_scene():
    assert np.abs(two_machines(True) - two_machines(False)).max() < 1e-6


def placed_points(model):
    plane, boxes, bumps = ground_parts(model, 0)
    normal = np.zeros(3)
    mujoco.mju_rotVecQuat(normal, [0.0, 0.0, 1.0], model.geom_quat[plane])
    points = []
    for geom in boxes:
        if model.geom_pos[geom, 2] < -1.0:
            continue
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, model.geom_quat[geom])
        size = model.geom_size[geom]
        top = np.array([[a * size[0], b * size[1], size[2]] for a in (-1, 1) for b in (-1, 1)])
        height = (model.geom_pos[geom] - model.geom_pos[plane]) @ normal + size[2]
        corners = np.concatenate([top, top - [0.0, 0.0, height]])
        points.append(model.geom_pos[geom] + corners @ rot.reshape(3, 3).T)
    for geom in bumps:
        if model.geom_pos[geom, 2] < -1.0:
            continue
        below = (model.geom_pos[geom] - model.geom_pos[plane]) @ normal
        rim = math.sqrt(model.geom_size[geom, 0] ** 2 - below**2)
        across = np.cross(normal, [1.0, 0.0, 0.0])
        across /= np.linalg.norm(across)
        along = np.cross(across, normal)
        foot = model.geom_pos[geom] - below * normal
        angles = np.linspace(0.0, 2 * math.pi, 64, endpoint=False)
        points.append(foot + rim * (np.outer(np.cos(angles), along)
                                    + np.outer(np.sin(angles), across)))
    return np.concatenate(points) if points else np.zeros((0, 3))


def test_drawn_features_stay_in_the_field_and_the_sun_shadow_as_placed():
    model = load_scene(1)
    xs = []
    counts = []
    for seed in range(300):
        ground = draw_ground(seed)
        assert FRICTION_RANGE[0] <= ground.friction <= FRICTION_RANGE[1]
        assert abs(math.atan(math.hypot(*ground.grade))) <= RANGES["slope"][1] + 1e-12
        try:
            apply_ground(model, 0, ground)
        except ValueError:
            continue
        finally:
            xs += [b.x for b in ground.boxes] + [b.x for b in ground.bumps]
            counts.append((len(ground.boxes), len(ground.bumps)))
        points = placed_points(model)
        assert len(points) == 0 or inside(points), seed
    boxes, bumps = np.mean(counts, axis=0)
    assert abs(boxes - 4.0) < 0.3 and abs(bumps - 3.0) < 0.3
    assert min(xs) < -1.5 and max(xs) > 8.0
    assert draw_ground(5) == draw_ground(5) and draw_ground(5) != draw_ground(6)


def test_off_features_flags_the_machine_that_drove_off_its_own_field():
    model = load_scene(2)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    for i in range(2):
        apply_ground(model, i, Ground())
        assert settle(model, data, i)
    data.ctrl[[model.actuator(machine_prefix(1) + name).id for name in TRAVEL_NAMES]] = (
        TRAVEL_SPEED)
    flagged = []
    for _ in range(round(12.0 / model.opt.timestep)):
        step(model, data)
        flagged.append((off_features(model, data, 0), off_features(model, data, 1)))
    assert not any(first for first, _ in flagged)
    assert not flagged[0][1] and flagged[-1][1]


@pytest.mark.parametrize("heading", [0.0, 1.0, -2.5])
def test_settle_turns_the_machine_to_the_asked_heading(heading):
    _, data = machine(heading=heading)
    w, _, _, z = data.qpos[3:7]
    assert math.atan2(2 * w * z, 1 - 2 * z * z) == pytest.approx(heading, abs=1e-3)


def stop_distance(edit):
    model, data = machine()
    edit(model, data)
    data.qvel[1] = TRAVEL_SPEED
    start = data.qpos[1]
    run(model, data, 2.0)
    return (data.qpos[1] - start) / (TRAVEL_SPEED**2 / (2 * FRICTION * G))


def brake_drift(edit):
    model, data = machine(slope(along=0.9 * math.atan(FRICTION)))
    edit(model, data)
    start = data.qpos[:3].copy()
    run(model, data, 2.0)
    return np.linalg.norm(data.qpos[:3] - start)


def front_tipping(edit, factor):
    ground, pose, _ = tipping_case("front, flat, 6.0 m")
    return tipping_about(ground, pose, [WHEEL_X[-1], 0.0, 0.0], [0.0, 1.0, 0.0], factor, edit)


def double_friction(model, data):
    model.pair_friction[:, :2] *= 2


def no_end_wheel(model, data):
    for side in SIDES:
        geom = model.geom(P + wheel_name(side, WHEELS - 1)).id
        model.pair_margin[(model.pair_geom1 == geom) | (model.pair_geom2 == geom)] = -1.0


def no_coupling(model, data):
    data.eq_active[model.eq_type == mujoco.mjtEq.mjEQ_JOINT] = 0


def end_wheels_out(model, data):
    for side in SIDES:
        for k, shift in ((0, -0.05), (WHEELS - 1, 0.05)):
            model.body_pos[model.body(P + wheel_name(side, k)).id, 0] += shift


def test_each_broken_piece_of_physics_fails_its_check(monkeypatch):
    import hma.machine.tracks as tracks

    assert stop_distance(double_friction) < 0.97
    assert front_tipping(no_end_wheel, 0.995) > TIPPED
    assert front_tipping(end_wheels_out, 1.005) < TIPPED
    assert brake_drift(no_coupling) > 1e-2

    def no_brake(model, data):
        model.tendon_frictionloss[drives(model).tendons] = 0.0

    assert brake_drift(no_brake) > 1e-2

    def no_support(model, data):
        boxes = [model.geom(P + tracks.shell_names(side)[3]).id for side in SIDES]
        model.pair_margin[np.isin(model.pair_geom1, boxes) | np.isin(model.pair_geom2, boxes)] = -1

    assert between_wheels(0.93, no_support) > TIPPED

    def friction_damping(b):
        def edit(model, data):
            model.pair_solreffriction[:, 1] = -b
        return edit

    assert wall_push(friction_damping(20.0))[2] < 0.97
    assert wall_push(friction_damping(100.0))[2] > 1.03

    def capped_brakes(model, data):
        belts = data.qvel[drives(model).dofs] * WHEEL_RADIUS
        model.tendon_frictionloss[drives(model).tendons] = np.minimum(
            TRACTION, TRAVEL_POWER / np.abs(belts).sum(axis=-1, keepdims=True).clip(1e-12))
        data.qfrc_applied[drives(model).dofs] = 0.0

    monkeypatch.setattr(tracks, "limit_power", capped_brakes)
    assert downhill_run(math.radians(20.0))[0] > 1.5 * TRAVEL_SPEED
    assert abs(turning([TRAVEL_SPEED, 0.0])[1][1]) > 0.05
    assert turning([TRAVEL_SPEED, 0.75 * TRAVEL_SPEED])[0] > -0.3
    monkeypatch.undo()
    monkeypatch.setattr(tracks, "TRAVEL_POWER", 1e12)
    assert slope_speed() > 1.1 * power_speed()
