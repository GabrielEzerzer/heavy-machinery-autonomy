import math
import pickle
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, Hydraulics
from hma.machine import inverse_kinematics, load_scene
from hma.machine.contact import set_friction
from hma.machine.ground import Ground, apply_ground
from hma.machine.mass import GAUGE
from hma.machine.model import STEEL_FRICTION
from hma.train import footing
from hma.train.play import summary
from hma.train.scene import PropScene
from hma.train.task import DEV_BASE, PropPath, TurnSettings
from hma.train.turn import ACTIONS, TRACK_SPEED, TurnController, TurnScene

SEED = DEV_BASE + 3
VALVES = 4
SPREAD = 0.33


def run(scene, actions):
    seen = []
    for a in actions:
        obs, reward, done, info = scene.step(a)
        seen.append((obs.copy(), reward.copy(), done.copy()))
    return seen


def same(a, b):
    return all(np.array_equal(x, y) for p, q in zip(a, b, strict=True)
               for x, y in zip(p, q, strict=True))


def valve_noise(seed, shape):
    actions = np.random.default_rng(seed).uniform(-0.3, 0.3, (*shape, ACTIONS))
    actions[..., VALVES:] = 0.0
    return actions


def finish(scene, commands):
    while True:
        t = scene.k[0] * scene.period
        _, _, done, info = scene.step(commands(t))
        if done[0]:
            return info[0]


def test_zero_residual_is_the_scripted_turn_bit_for_bit():
    scene = TurnScene(1, "nominal")
    scene.reset(0, SEED)
    task = scene.tasks[0]
    decisions = int(task.turn_span[1] / scene.period) + 10
    run(scene, [np.zeros((1, ACTIONS))] * decisions)

    model = load_scene(1)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [NOMINAL])
    path = PropPath(task, footing.rear_pivot())
    apply_ground(model, 0, Ground())
    footing.park(model, 0)
    set_friction(model, 0, STEEL_FRICTION[0])
    layer.reset(0, scene.machines[0], inverse_kinematics(*path(0.0)))
    mujoco.mj_forward(model, data)
    controller = TurnController(task, footing.rear_pivot(), 10.0)
    turned = False
    for k in range(decisions):
        sensed = layer.observe()
        out = controller(k * 0.1, sensed.positions[0], sensed.pressures[0])
        turned |= bool(np.any(out[VALVES:] != 0.0))
        layer.act(out[None, :VALVES])
        data.ctrl[layer._travel[0]] = out[VALVES:] * TRACK_SPEED
        for _ in range(layer.steps_per_decision):
            layer.step()
    assert turned
    assert np.array_equal(data.qpos, scene.data.qpos)
    assert np.array_equal(layer.chamber_pressures(), scene.layer.chamber_pressures())


def test_same_seed_gives_the_same_turning_episode_bit_for_bit():
    actions = valve_noise(2, (40, 2))
    episodes = []
    for _ in range(2):
        scene = TurnScene(2, "machines")
        scene.reset(0, SEED)
        scene.reset(1, SEED + 1)
        episodes.append(run(scene, actions))
    assert same(*episodes)


def test_save_restore_and_replay_mid_turn_is_bit_for_bit():
    scene = TurnScene(2, "nominal")
    scene.reset(0, SEED)
    scene.reset(1, SEED + 2)
    start = int(min(task.turn_span[0] for task in scene.tasks) / scene.period) + 5
    actions = valve_noise(3, (start + 30, 2)) * 0.2
    run(scene, actions[:start])
    assert scene.turn_started.any()
    saved = pickle.loads(pickle.dumps(scene.snapshot()))
    straight = run(scene, actions[start:])
    other = TurnScene(2, "nominal")
    other.reset(0, 11)
    other.reset(1, 12)
    run(other, actions[:5])
    other.restore(saved)
    assert same(straight, run(other, actions[start:]))
    assert np.array_equal(scene.data.qpos, other.data.qpos)
    assert np.array_equal(scene.heading_est, other.heading_est)
    assert np.array_equal(scene.turn_slide, other.turn_slide)


def test_resetting_one_machine_leaves_its_turning_neighbour_bit_for_bit():
    decisions = 150
    actions = valve_noise(1, (decisions, 2))

    def neighbour(reset_other):
        scene = TurnScene(2, "nominal")
        scene.reset(0, SEED)
        scene.reset(1, SEED + 1)
        seen = []
        for k, a in enumerate(actions):
            if reset_other and k and k % 6 == 0:
                scene.reset(1, 555 + k)
            obs, reward, done, _ = scene.step(a)
            for i in np.flatnonzero(done):
                scene.reset(i, 900 + i)
            seen.append((obs[:1].copy(), reward[:1].copy()))
        return seen, scene.layer.redos, scene.turn_started[0]

    alone, _, turned = neighbour(False)
    disturbed, redos, _ = neighbour(True)
    assert turned
    assert same(alone, disturbed)
    assert redos > 0


def spin(scene):
    task = scene.tasks[0]

    def commands(t):
        out = np.zeros((1, ACTIONS))
        start, end = task.turn_span
        if start <= t < end:
            out[0, VALVES:] = scene.controllers[0](t, np.zeros(4), np.zeros((4, 2)))[VALVES:]
        return out

    return commands


def test_a_spin_turn_with_the_bucket_in_the_air_fails_and_ends_as_stray():
    scene = TurnScene(1, "nominal", "direct")
    scene.reset(0, SEED)
    info = finish(scene, spin(scene))
    assert info["cause"] == "stray"
    assert not info["passed"]

    no_ends = replace(TurnSettings(), stray_end=math.inf, slide_limit=math.inf, overshoot=math.inf)
    blind = TurnScene(1, "nominal", "direct", no_ends)
    blind.reset(0, SEED)
    info = finish(blind, spin(blind))
    reasons = blind.reasons(info)
    assert info["cause"] is None
    assert abs(info["heading"]) > math.radians(2.0)
    assert info["hold_share"] == 0.0
    assert not info["passed"]
    assert any(reason.startswith("hold share") for reason in reasons)
    assert any(reason.startswith("bucket slid") for reason in reasons)
    assert any(reason.startswith("tracks travelled") for reason in reasons)
    assert summary("spin", info, blind).startswith("spin: fail (hold share")


def test_a_failed_propping_episode_prints_its_reasons():
    scene = PropScene(1, "nominal", "direct")
    scene.reset(0, DEV_BASE)
    info = finish(scene, lambda t: np.array([[0.0, 1.0, 0.0, 0.0]]))
    assert info["cause"] == "tilt"
    assert summary("pushed", info, scene).startswith("pushed: fail (early end: tilt")


def test_a_turn_on_the_swing_alone_is_not_counted_as_a_turn_of_the_belts():
    scene = TurnScene(1, "nominal", "direct")
    scene.reset(0, DEV_BASE)
    task = scene.tasks[0]
    start, end = task.turn_span

    def commands(t):
        out = scene.baseline()
        if start <= t < end:
            out[0, VALVES:] = 0.0
            out[0, 0] = -math.copysign(1.0, task.heading)
        return out

    info = finish(scene, commands)
    assert abs(info["heading"]) > math.radians(5.0)
    assert abs(info["belt_heading"]) < math.radians(0.1)
    assert not info["passed"]
    assert any(reason.startswith("heading more than") for reason in scene.reasons(info))


def test_a_turn_held_back_until_the_ramp_down_fails():
    scene = TurnScene(1, "nominal", "direct")
    scene.reset(0, SEED)
    start, end = scene.tasks[0].turn_span
    controller = scene.controllers[0]

    def commands(t):
        out = scene.baseline()
        if start <= t < end:
            out[0, 0] = 0.0
            out[0, VALVES:] = 0.0
        elif t >= end:
            late = controller(t - (end - start), np.zeros(4), np.zeros((4, 2)))
            out[0, 0] = late[0] if t < 2 * end - start else out[0, 0]
            out[0, VALVES:] = late[VALVES:]
        return out

    info = finish(scene, commands)
    assert info["turn_slide"] < scene.settings.turn_slide
    assert info["cause"] == "stray"
    assert not info["passed"]


def test_the_belt_rule_reads_the_drive_wheels_not_the_track_commands():
    scene = TurnScene(1, "nominal", "direct")
    scene.reset(0, DEV_BASE)
    task = scene.tasks[0]
    start, end = task.turn_span
    sign = math.copysign(1.0, task.heading)
    commanded = 0.0

    def commands(t):
        nonlocal commanded
        out = scene.baseline()
        if start <= t < end:
            out[0, 0] = 0.0
            out[0, VALVES:] = (-sign, sign)
        commanded += scene.period * (out[0, VALVES + 1] - out[0, VALVES]) * TRACK_SPEED / GAUGE
        return out

    info = finish(scene, commands)
    assert abs(info["belt_heading"] - commanded) > math.radians(1.0)
    assert not info["passed"]


def test_random_residual_commands_survive_the_rest_and_place_phases():
    scene = TurnScene(1, "nominal")
    scene.reset(0, DEV_BASE)
    rng = np.random.default_rng(7)
    start = scene.tasks[0].turn_span[0]
    while scene.k[0] * scene.period < start:
        _, _, done, info = scene.step(rng.normal(0.0, SPREAD, (1, ACTIONS)))
        assert not done[0], info[0]


def test_driving_both_tracks_forward_after_set_down_fails():
    def drive(scene):
        centre = scene.tasks[0].recentre_span[0]

        def commands(t):
            out = scene.baseline()
            if t >= centre:
                out[0, VALVES:] = 1.0
            return out

        return commands

    scene = TurnScene(1, "nominal", "direct")
    scene.reset(0, SEED)
    info = finish(scene, drive(scene))
    assert info["cause"] == "stray"
    assert not info["passed"]
    blind = TurnScene(1, "nominal", "direct", replace(TurnSettings(), stray_end=math.inf))
    blind.reset(0, SEED)
    info = finish(blind, drive(blind))
    assert info["cause"] is None and info["stray_travel"] > 1.0
    assert not info["passed"]
    assert any(reason.startswith("tracks travelled") for reason in blind.reasons(info))


def test_shuffling_the_tracks_back_and_forth_after_set_down_fails():
    scene = TurnScene(1, "nominal", "direct")
    scene.reset(0, SEED)
    start = scene.tasks[0].recentre_span[0] + 3.0

    def commands(t):
        out = scene.baseline()
        if start <= t < start + 2.0:
            out[0, VALVES:] = 0.3 if int((t - start) / 0.2 + 1e-9) % 2 == 0 else -0.3
        return out

    info = finish(scene, commands)
    reasons = scene.reasons(info)
    assert info["cause"] is None
    assert info["stray_travel"] < TurnSettings().stray_pass
    assert info["stray_distance"] > TurnSettings().distance_pass
    assert not info["passed"]
    assert any(reason.startswith("tracks covered") for reason in reasons)


def test_turning_observations_stay_finite_and_in_bounds_under_random_actions():
    scene = TurnScene(2, "machines", "direct")
    seen = [scene.reset(0, SEED), scene.reset(1, SEED + 1)]
    rng = np.random.default_rng(5)
    for k in range(120):
        obs, _, done, _ = scene.step(rng.uniform(-1.0, 1.0, (2, ACTIONS)))
        seen += list(obs)
        for i in np.flatnonzero(done):
            seen.append(scene.reset(i, SEED + 10 + k))
    seen = np.array(seen)
    assert np.all(np.isfinite(seen))
    assert np.all((seen >= TurnScene.obs_low) & (seen <= TurnScene.obs_high))


@pytest.mark.parametrize(("settings", "cause"), [
    (TurnSettings(), "slide"), (replace(TurnSettings(), slide_limit=math.inf), "stray")])
def test_losing_the_prop_mid_turn_ends_the_episode(settings, cause):
    scene = TurnScene(1, "nominal", "direct", settings)
    scene.reset(0, SEED)
    start, end = scene.tasks[0].turn_span

    def commands(t):
        out = scene.baseline()
        if t >= start + 1.0:
            out[0, 1] = -1.0
        return out

    info = finish(scene, commands)
    assert info["cause"] == cause
    assert not info["passed"]
    assert start + 1.0 < scene.k[0] * scene.period < end


def test_the_scripted_turn_passes_on_a_nominal_seed():
    scene = TurnScene(1, "nominal")
    scene.reset(0, SEED)
    info = finish(scene, lambda t: np.zeros((1, ACTIONS)))
    assert info["passed"], scene.reasons(info)
    assert abs(info["heading_error"]) < math.radians(1.5)
    assert abs(info["heading"] - info["belt_heading"]) < math.radians(0.5)
    assert info["stray_distance"] < TurnSettings().distance_pass / 10


def test_the_turn_refuses_rough_ground_and_keeps_the_prop_spaces():
    with pytest.raises(ValueError, match="level ground"):
        TurnScene(1, "ground")
    assert PropScene.actions == VALVES and TurnScene.actions == ACTIONS
    assert TurnScene.frame > PropScene.frame
    assert TurnScene.obs_low.shape == TurnScene.obs_high.shape


def test_a_checkpoint_keeps_its_task_and_old_checkpoints_read_as_propping():
    from argparse import Namespace

    from hma.train.__main__ import pick_task

    assert pick_task(Namespace(task=None, rung="nominal"), {}) == "prop"
    assert pick_task(Namespace(task=None, rung="nominal"), {"task": "turn"}) == "turn"
    assert pick_task(Namespace(task="turn", rung="machines")) == "turn"
    with pytest.raises(SystemExit, match="trained on the prop task"):
        pick_task(Namespace(task="turn", rung="nominal"), {})
    with pytest.raises(SystemExit, match="level ground"):
        pick_task(Namespace(task="turn", rung="ground"))
