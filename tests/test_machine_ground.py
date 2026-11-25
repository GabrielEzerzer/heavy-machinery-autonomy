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
from hma.machine.ground import (
    RANGES,
    Box,
    Ground,
    apply_ground,
    box_corners,
    draw_ground,
    inside,
)
from hma.machine.measure import (
    CASES,
    HOLD,
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
from hma.machine.model import GROUND, JOINT_NAMES, TRAVEL_NAMES, VALVE, ground_geoms
from hma.machine.tracks import (
    CONTACT_SOLIMP,
    FRICTION,
    FRICTION_RANGE,
    GAUGE,
    OUTLINE,
    SIDES,
    TRAVEL,
    TRAVEL_POWER,
    TRAVEL_SPEED,
    WHEEL_X,
    WHEELS,
    contact_names,
    drives,
    step,
    wheel_name,
)

P = machine_prefix(0)
G = 9.81
SOFTNESS = CONTACT_SOLIMP[2]


def machine(ground=None, pose=None, hold=False, heading=0.0):
    model = load_scene(1)
    apply_ground(model, 0, ground or Ground())
    if hold:
        for name in JOINT_NAMES:
            model.tendon_frictionloss[model.tendon(P + name + VALVE).id] = HOLD
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key(HOME_KEY).id)
    if pose is not None:
        for name, value in zip(JOINT_NAMES, pose, strict=True):
            data.qpos[model.joint(P + name).qposadr[0]] = value
    assert settle(model, data, 0, heading=heading)
    mujoco.mj_forward(model, data)
    return model, data


def energy(model, data):
    mujoco.mj_forward(model, data)
    mujoco.mj_energyPos(model, data)
    mujoco.mj_energyVel(model, data)
    mujoco.mj_subtreeVel(model, data)
    base = model.body(P + "base").id
    gravity_power = model.body_subtreemass[base] * model.opt.gravity @ data.subtree_linvel[base]
    return float(data.energy.sum()) + 0.5 * model.opt.timestep * gravity_power


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
        (0.0, 0.0, 1.0, 0.5), [WHEEL_X[-2], 0.0, 0.0], [0.0, 1.0, 0.0]),
    "along between the rails": (Ground(boxes=(Box(0.0, GAUGE / 2 - 3.0, 0.0, 6.0, 3.0, 1.0),)),
                                chart_pose(0.0, 4.5, math.pi / 2), [0.0, GAUGE / 2, 0.0],
                                [1.0, 0.0, 0.0]),
}


@pytest.mark.parametrize("factor", [0.98, 1.02])
@pytest.mark.parametrize("name", list(EDGES))
def test_a_ground_edge_under_a_track_becomes_the_tipping_line(name, factor):
    ground, pose, line, axis = EDGES[name]
    angle = tipping_about(ground, pose, line, axis, factor)
    assert (angle > TIPPED) if factor > 1 else (angle < TIPPED), angle


def propped():
    model, data = machine(pose=chart_pose(1.5, 5.0))
    bucket = model.geom(P + "bucket").id
    model.geom_contype[bucket] = 1
    model.body_contype[model.geom_bodyid[bucket]] = 1
    model.geom_friction[bucket] = 0.0
    model.geom_priority[bucket] = 1
    boom = model.actuator(P + "boom").id
    data.ctrl[boom] = 0.1
    for _ in range(round(8.0 / model.opt.timestep)):
        step(model, data)
        if pitch(model, data) < -25.0:
            break
    data.ctrl[boom] = 0.0
    run(model, data, 1.0)
    return model, data


def test_a_machine_propped_on_its_bucket_loads_no_brake_and_stays_put():
    model, data = propped()
    assert -30.0 < pitch(model, data) < -20.0
    start = data.qpos[:3].copy()
    peak = 0.0
    for _ in range(round(4.0 / model.opt.timestep)):
        step(model, data)
        peak = max(peak, np.abs(tendon_friction(data, brakes(model))).max())
    assert peak < 1.0
    assert np.linalg.norm(data.qpos[:3] - start) < 1e-3


def test_full_speed_into_a_wall_stalls_at_the_wall():
    wall = Box(4.0, 0.0, 0.0, 0.5, 3.0, 1.2)
    model, data = machine(Ground(boxes=(wall,)))
    data.ctrl[travel(model)] = TRAVEL_SPEED
    run(model, data, 3.0)
    start = data.qpos[0]
    run(model, data, 3.0)
    assert abs(data.qpos[0] - start) < 1e-2
    assert data.qpos[0] + OUTLINE[1] < wall.x - wall.half_length + SOFTNESS


def test_a_parked_machine_pushed_with_120_kn_does_not_pass_through_a_wall():
    wall = Box(0.0, BESIDE + 0.5, 0.0, 3.0, 0.5, 1.2)
    model, data = machine(Ground(boxes=(wall,)))
    data.qvel[1] = 0.2
    sink = 0.0
    for k in range(round(3.0 / model.opt.timestep)):
        data.xfrc_applied[model.body(P + "base").id, 1] = 120e3
        step(model, data)
        if k * model.opt.timestep > 1.0:
            sink = max(sink, deepest(data, ground_geoms(model, 0)[1][0]))
    assert data.qpos[1] + OUTLINE[3] < BESIDE + SOFTNESS
    assert sink < SOFTNESS


def test_driving_onto_a_0_2_m_step_climbs_it_and_pitches_with_the_ground():
    height = 0.2
    model, data = machine(Ground(boxes=(Box(2.5 + 5.0, 0.0, 0.0, 5.0, 6.0, height),)))
    data.ctrl[travel(model)] = 0.5
    steepest = 0.0
    for _ in range(round(14.0 / model.opt.timestep)):
        step(model, data)
        steepest = max(steepest, -pitch(model, data))
    assert data.qpos[0] > 2.5 + OUTLINE[1]
    assert data.qpos[2] == pytest.approx(height, abs=1e-3)
    assert abs(pitch(model, data)) < 0.05
    _, com = machine_com(model, data)
    origin, rot = base_frame(model, data)
    lever = (rot.T @ (com - origin))[0] - WHEEL_X[0]
    assert steepest == pytest.approx(math.degrees(math.asin(height / lever)), rel=0.05)


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


def test_on_a_slope_the_speed_is_set_by_the_engine_power():
    assert slope_speed() == pytest.approx(power_speed(), rel=0.03)


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
    plane, boxes, bumps = ground_geoms(model, 0)
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


def test_drawn_features_stay_in_the_field_and_the_sun_shadow():
    for seed in range(200):
        ground = draw_ground(seed)
        assert FRICTION_RANGE[0] <= ground.friction <= FRICTION_RANGE[1]
        assert abs(math.atan(math.hypot(*ground.grade))) <= RANGES["slope"][1] + 1e-12
        for box in ground.boxes:
            assert inside(box_corners(box)), seed
        for bump in ground.bumps:
            rim = [[bump.x + bump.radius * c, bump.y + bump.radius * s, bump.height]
                   for c, s in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            assert inside(np.array(rim)), seed
    assert draw_ground(5) == draw_ground(5) and draw_ground(5) != draw_ground(6)


def test_off_features_flags_a_machine_that_drove_off_the_drawn_field():
    model, data = machine()
    assert not off_features(model, data, 0)
    data.qpos[0] += 12.0
    mujoco.mj_kinematics(model, data)
    assert off_features(model, data, 0)


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

    def released_brakes(model, data):
        model.tendon_frictionloss[drives(model).tendons] = 0.0

    def no_brake(model, data):
        monkeypatch.setattr(tracks, "write_caps", released_brakes)

    assert brake_drift(no_brake) > 1e-2
    monkeypatch.undo()
    monkeypatch.setattr(tracks, "TRAVEL_POWER", 1e12)
    assert slope_speed() > 1.1 * power_speed()
