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
from hma.machine.model import STEEL_FRICTION
from hma.train import footing
from hma.train.scene import PropScene
from hma.train.task import DEV_BASE, PropPath, TurnSettings
from hma.train.turn import ACTIONS, TRACK_SPEED, TurnController, TurnScene

SEED = DEV_BASE + 3
VALVES = 4


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


def test_a_spin_turn_with_the_bucket_in_the_air_fails_and_ends_as_lost():
    scene = TurnScene(1, "nominal", "direct")
    scene.reset(0, SEED)
    info = finish(scene, spin(scene))
    assert info["cause"] == "lost"
    assert not info["passed"]

    no_ends = replace(TurnSettings(), lost_tilt=-1.0, slide_limit=math.inf, overshoot=math.inf)
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


def test_losing_the_prop_mid_turn_ends_the_episode():
    scene = TurnScene(1, "nominal", "direct")
    scene.reset(0, SEED)
    start, end = scene.tasks[0].turn_span

    def commands(t):
        out = scene.baseline()
        if t >= start + 1.0:
            out[0, 1] = -1.0
        return out

    info = finish(scene, commands)
    assert info["cause"] == "lost"
    assert not info["passed"]
    assert start + 1.0 < scene.k[0] * scene.period < end


def test_the_scripted_turn_passes_on_a_nominal_seed():
    scene = TurnScene(1, "nominal")
    scene.reset(0, SEED)
    info = finish(scene, lambda t: np.zeros((1, ACTIONS)))
    assert info["passed"], scene.reasons(info)
    assert abs(info["heading_error"]) < math.radians(1.5)
    assert abs(info["heading"] - info["belt_heading"]) < math.radians(0.5)


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
