import math
import pickle
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, RANGES, BaseController, Hydraulics, draw_machine
from hma.hydraulics.params import LITRE
from hma.machine import TRAVEL_NAMES, inverse_kinematics, load_scene, machine_prefix, tracks
from hma.machine.contact import set_friction
from hma.machine.ground import Box, Ground, apply_ground
from hma.machine.model import STEEL_FRICTION
from hma.train import footing
from hma.train.scene import OBS_HIGH, OBS_LOW, PropScene
from hma.train.task import (
    DEV_BASE,
    HELD_OUT_BASE,
    UNFILTERED,
    PropPath,
    Settings,
    dev_seeds,
    held_out_seeds,
    training_seed,
)
from hma.train.turn import TurnScene

SEED = DEV_BASE + 3
BRAKED = DEV_BASE + 30


def run(scene, actions):
    seen = []
    for a in actions:
        obs, reward, done, info = scene.step(a)
        seen.append((obs.copy(), reward.copy(), done.copy()))
    return seen


def same(a, b):
    return all(np.array_equal(x, y) for p, q in zip(a, b, strict=True)
               for x, y in zip(p, q, strict=True))


def model_arrays(model):
    return {name: getattr(model, name) for name in dir(model)
            if not name.startswith("_") and isinstance(getattr(model, name), np.ndarray)}


def test_zero_residual_is_the_plain_baseline_bit_for_bit():
    scene = PropScene(1, "machines")
    scene.reset(0, SEED)
    run(scene, [np.zeros((1, 4))] * 60)

    model = load_scene(1)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [NOMINAL])
    machine, task = scene.machines[0], scene.tasks[0]
    path = PropPath(task, footing.rear_pivot())
    apply_ground(model, 0, Ground())
    footing.park(model, 0)
    set_friction(model, 0, STEEL_FRICTION[0])
    layer.reset(0, machine, inverse_kinematics(*path(0.0)))
    mujoco.mj_forward(model, data)
    controller = BaseController(path, 10.0)
    for k in range(60):
        sensed = layer.observe()
        layer.act(controller(k * 0.1, sensed.positions[0], sensed.pressures[0])[None])
        for _ in range(layer.steps_per_decision):
            layer.step()
    assert np.array_equal(data.qpos, scene.data.qpos)
    assert np.array_equal(layer.chamber_pressures(), scene.layer.chamber_pressures())


def test_same_seed_gives_the_same_episode_bit_for_bit():
    actions = np.random.default_rng(2).uniform(-0.3, 0.3, (40, 2, 4))
    episodes = []
    for _ in range(2):
        scene = PropScene(2, "ground")
        scene.reset(0, SEED)
        scene.reset(1, SEED + 1)
        episodes.append(run(scene, actions))
    assert same(*episodes)


def test_save_restore_and_replay_is_bit_for_bit():
    actions = np.random.default_rng(3).uniform(-0.3, 0.3, (50, 2, 4))
    scene = PropScene(2, "ground")
    scene.reset(0, SEED)
    scene.reset(1, SEED + 1)
    run(scene, actions[:20])
    assert np.all(scene.smooth != 0.0)
    scene.data.userdata[:] = (0.1, -0.2, 3e3, 0.0, 0.0, 5e2)
    saved = pickle.loads(pickle.dumps(scene.snapshot()))
    straight = run(scene, actions[20:])
    other = PropScene(2, "ground")
    other.reset(0, 11)
    other.reset(1, 12)
    run(other, actions[:5])
    other.restore(saved)
    assert same(straight, run(other, actions[20:]))
    assert np.array_equal(scene.data.qpos, other.data.qpos)
    assert np.array_equal(scene.data.userdata, other.data.userdata)
    assert np.array_equal(scene.smooth, other.smooth)
    written = model_arrays(other.model)
    for name, value in model_arrays(scene.model).items():
        assert np.array_equal(value, written[name], equal_nan=True), name


@pytest.mark.parametrize("kind, smoothing", [(PropScene, Settings().smoothing),
                                             (TurnScene, Settings().smoothing),
                                             (PropScene, UNFILTERED)])
def test_a_step_in_the_action_reaches_the_valves_through_the_filter(kind, smoothing):
    assert -0.1 / math.log(1.0 - Settings().smoothing) == pytest.approx(0.3, abs=0.01)
    scene = kind(1, "nominal", settings=replace(kind.settings_type(), smoothing=smoothing))
    scene.reset(0, SEED)
    baselines, sent = [], []
    plain, send = scene.baseline, scene._send
    scene.baseline = lambda: baselines.append(plain()) or baselines[-1]
    scene._send = lambda commands: sent.append(np.copy(commands)) or send(commands)
    run(scene, [np.full((1, scene.actions), 0.5)] * 6)
    scale = scene.settings.scale
    assert len(sent) == len(baselines) == 6
    for k, (b, u) in enumerate(zip(baselines, sent, strict=True), 1):
        if smoothing == UNFILTERED:
            assert np.array_equal(u, np.clip(b + scale * 0.5, -1.0, 1.0))
        else:
            reached = 1.0 - (1.0 - smoothing) ** k
            assert np.allclose(u, np.clip(b + scale * 0.5 * reached, -1.0, 1.0), rtol=0.0,
                               atol=1e-12)


def test_a_reset_clears_its_own_filter_state_and_leaves_its_neighbours():
    scene = PropScene(2, "nominal")
    scene.reset(0, SEED)
    scene.reset(1, SEED + 1)
    run(scene, [np.full((2, 4), 0.5)] * 3)
    assert np.all(scene.smooth != 0.0)
    kept = scene.smooth[0].copy()
    scene.reset(1, SEED + 2)
    assert np.array_equal(scene.smooth[0], kept)
    assert np.all(scene.smooth[1] == 0.0)


@pytest.mark.parametrize("rung, machines, decisions", [("ground", 3, 60), ("nominal", 2, 120)])
def test_resetting_one_machine_leaves_its_neighbours_bit_for_bit(rung, machines, decisions):
    actions = np.random.default_rng(1).uniform(-0.3, 0.3, (decisions, machines, 4))
    others = [i for i in range(machines) if i != 1]

    def neighbours(reset_middle):
        scene = PropScene(machines, rung)
        for i in range(machines):
            scene.reset(i, SEED + i)
        seen = []
        for k, a in enumerate(actions):
            if reset_middle and k and k % 6 == 0:
                scene.reset(1, 555 + k)
            obs, reward, done, _ = scene.step(a)
            for i in np.flatnonzero(done):
                scene.reset(i, 900 + i)
            seen.append((obs[others].copy(), reward[others].copy()))
        return seen, scene.layer.redos

    alone, _ = neighbours(False)
    disturbed, redos = neighbours(True)
    assert same(alone, disturbed)
    assert redos > 0


class Corner(PropScene):
    def __init__(self, machine, *args, **kwargs):
        self.corner = machine
        super().__init__(*args, **kwargs)

    def machine_for(self, index, seed):
        return self.corner


STIFFEST = replace(
    draw_machine(SEED), bulk_modulus=RANGES["bulk_modulus"][1],
    dead_volume=np.full((4, 2), RANGES["dead_volume"][0]),
    pump_volume=RANGES["pump_volume"][0], delay=np.full(4, RANGES["delay"][0]),
    lag=np.full(4, RANGES["lag"][0]), deadband=np.zeros(4), backlash=np.zeros(4),
    leakage=np.zeros((4, 2)) + 0.01 * LITRE / 60.0 / 10e6)


@pytest.mark.parametrize("machine", [NOMINAL, STIFFEST], ids=["nominal", "stiffest"])
def test_observations_stay_finite_and_in_bounds_under_random_actions(machine):
    scene = Corner(machine, 2, "ground", mode="direct")
    seen = [scene.reset(0, SEED), scene.reset(1, SEED + 1)]
    rng = np.random.default_rng(5)
    lowest = Settings().floor - sum(Settings().weights.values())
    for k in range(120):
        obs, reward, done, info = scene.step(rng.uniform(-1.0, 1.0, (2, 4)))
        seen += list(obs)
        assert np.all((reward >= lowest) & (reward <= 1.0))
        for i in np.flatnonzero(done):
            seen.append(scene.reset(i, SEED + 10 + k))
    seen = np.array(seen)
    assert np.all(np.isfinite(seen))
    assert np.all((seen >= OBS_LOW) & (seen <= OBS_HIGH))


@pytest.mark.parametrize(("seed", "push", "cause"), [
    (DEV_BASE, (0.0, 1.0, 0.0, 0.0), "tilt"),
    (DEV_BASE + 2, (0.0, 1.0, 1.0, 1.0), "contact"),
])
def test_ending_early_never_returns_more_than_continuing_with_the_baseline(seed, push, cause):
    switches = [0, 5, 10, 15, 20, 25, None]
    scene = PropScene(len(switches), "nominal", mode="direct")
    for i in range(len(switches)):
        scene.reset(i, seed)
    push = np.array(push)
    rewards = [[] for _ in switches]
    causes = [None] * len(switches)
    live = np.ones(len(switches), bool)
    floor = Settings().floor - sum(Settings().weights.values())
    for k in range(400):
        baseline = scene.baseline()
        actions = np.array([push if s is None or k < s else baseline[i]
                            for i, s in enumerate(switches)])
        _, reward, done, info = scene.step(actions)
        for i in np.flatnonzero(live):
            rewards[i].append(reward[i])
            if done[i]:
                causes[i], live[i] = info[i]["cause"], False
            else:
                assert reward[i] >= floor
        for i in np.flatnonzero(done):
            scene.reset(i, DEV_BASE + 1)
        if not live.any():
            break
    pushed = rewards[-1]
    assert causes[-1] == cause and pushed[-1] == 0.0
    recovered = 0
    for i, start in enumerate(switches[:-1]):
        assert sum(pushed[start:]) <= sum(rewards[i][start:])
        recovered += causes[i] is None
    assert recovered > 0


def test_no_observation_moves_when_only_unsensed_quantities_change():
    actions = np.random.default_rng(7).uniform(-0.3, 0.3, (30, 1, 4))

    def observed(change):
        scene = PropScene(1, "machines")
        scene.reset(0, SEED)
        change(scene)
        return [o for o, _, _ in run(scene, actions)]

    def nothing(scene):
        pass

    def move_load(scene):
        model, data = scene.model, scene.data
        q0 = int(model.joint(machine_prefix(0) + "load").qposadr[0])
        data.qpos[q0:q0 + 2] += (-2.0, 1.5)
        body = model.body(machine_prefix(0) + "load").id
        model.body_mass[body] *= 1.7
        model.geom_friction[model.geom(machine_prefix(0) + "load").id, 0] = 0.2

    def far_ground(scene):
        apply_ground(scene.model, 0, Ground(boxes=(Box(-10.0, 10.0, 0.3, 1.0, 1.0, 0.3),)))

    def clock(scene):
        scene.data.time += 1234.5

    base = observed(nothing)
    for change in (move_load, far_ground, clock):
        assert all(np.array_equal(a, b) for a, b in zip(base, observed(change), strict=True))


def test_dev_and_held_out_seeds_never_meet_training_seeds():
    rng = np.random.default_rng(0)
    drawn = {training_seed(rng) for _ in range(10_000)}
    assert max(drawn) < DEV_BASE
    assert max(dev_seeds()) < HELD_OUT_BASE <= min(held_out_seeds(10))


@pytest.mark.parametrize("latency", [0.05, RANGES["sensor_latency"][1]])
def test_period_readings_are_delayed_means_with_one_full_size_noise_draw_each(latency):
    machine = replace(draw_machine(SEED), sensor_latency=latency)
    model = load_scene(1, weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), [machine])
    commands = np.random.default_rng(8).uniform(-0.5, 0.5, (80, 1, 4))
    lag = round(machine.sensor_latency / layer.dt)
    noise = np.array([machine.position_noise, machine.velocity_noise, machine.pressure_noise])
    truth, worst = [], np.zeros(3)
    for u in commands:
        layer.act(u)
        for _ in range(layer.steps_per_decision):
            layer.step()
            state = layer.true_state()
            pressures = np.maximum(state.pressures[0].ravel() + machine.pressure_offset.ravel(), 0)
            truth.append(np.r_[state.positions[0], state.velocities[0], pressures])
        if len(truth) < layer.steps_per_decision + lag:
            continue
        window = np.array(truth[-layer.steps_per_decision - lag:len(truth) - lag]).mean(0)
        sensed = layer.observe_mean()
        seen = np.r_[sensed.positions[0], sensed.velocities[0], sensed.pressures[0].ravel()]
        off = np.abs(seen - window)
        bounds = np.repeat(noise, [4, 4, 8])
        assert np.all((off <= bounds * (1 + 1e-9)) | (seen == 0.0))
        worst = np.maximum(worst, [off[:4].max(), off[4:8].max(), off[8:].max()])
    assert np.all(worst > 0.9 * noise)


@pytest.mark.train_extra
def test_an_untrained_residual_policy_sends_the_baseline_plus_its_scaled_action():
    from stable_baselines3.common.vec_env import VecNormalize

    from hma.train.__main__ import COMMAND_NOISE, EGLI_HUTTER, SPREAD, build_model
    from hma.train.vec import SceneVecEnv

    scale, smoothing = Settings().scale, Settings().smoothing
    assert SPREAD["residual"] * scale == pytest.approx(COMMAND_NOISE)
    env = SceneVecEnv(1, "nominal", "residual", 0, seeds=[DEV_BASE + 4])
    venv = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=EGLI_HUTTER["gamma"])
    model = build_model(venv, 128, 0, SPREAD["residual"])
    assert np.allclose(model.policy.log_std.exp().detach().numpy(), SPREAD["residual"])
    scene, baselines, sent = env.scene, [], []
    plain, act = scene.baseline, scene.layer.act
    scene.baseline = lambda: baselines.append(plain()) or baselines[-1]
    scene.layer.act = lambda commands: sent.append(np.copy(commands)) or act(commands)
    obs, largest, actions, starts = venv.reset(), 0.0, [], []
    for deterministic in (True, False):
        starts.append(len(actions))
        while True:
            action = model.predict(obs, deterministic=deterministic)[0]
            if deterministic:
                largest = max(largest, float(np.abs(action).max()))
            actions.append(np.clip(np.asarray(action, float), -1.0, 1.0))
            obs, _, done, _ = venv.step(action)
            if done[0]:
                break
    assert largest <= 0.02
    assert len(sent) == len(baselines) == len(actions) > 300
    smooth = np.zeros(4)
    for k, (b, a, u) in enumerate(zip(baselines, actions, sent, strict=True)):
        smooth = np.zeros(4) if k in starts else smooth
        smooth = smooth + smoothing * (a[0] - smooth)
        assert np.array_equal(u, np.clip(b + scale * smooth, -1.0, 1.0))


def test_period_means_clip_each_pressure_sample_before_averaging():
    model = load_scene(1, weld=True)
    layer = Hydraulics(model, mujoco.MjData(model), [NOMINAL])
    layer._mean_noise[:] = 0.0
    layer._latency_steps[:] = 0
    layer._clean_head = layer._clean_len - 1
    signs = np.where(np.arange(layer._clean_len) % 2 == 0, -1.0, 1.0)
    layer._clean[:, 2:] = 1e6 * signs[:, None, None, None]
    assert np.allclose(layer.observe_mean().pressures, 5e5)


def test_no_observation_reads_the_valve_delay_lag_or_deadband():
    machine = draw_machine(SEED)
    slower = replace(machine, delay=machine.delay * 0.5, lag=machine.lag * 0.5,
                     deadband=machine.deadband * 0.5)

    def observed(drawn):
        class Fixed(PropScene):
            def machine_for(self, index, seed):
                return drawn

        scene = Fixed(1, "machines", mode="direct")
        seen = [scene.reset(0, SEED)]
        for _ in range(30):
            seen.append(scene.step(np.zeros((1, 4)))[0][0])
        return np.array(seen)

    assert np.array_equal(observed(machine), observed(slower))


def test_a_propping_episode_holds_its_tracks_on_shut_brakes_and_under_the_solver_limit():
    scene = PropScene(1, "ground")
    scene.reset(0, BRAKED)
    model, data = scene.model, scene.data
    lay = tracks.drives(model)
    wheel = model.jnt_qposadr[model.dof_jntid[lay.dofs[0]]]
    travel = [model.actuator(machine_prefix(0) + name).id for name in TRAVEL_NAMES]
    start = data.qpos[wheel].copy()
    seen = {"iterations": 0, "travel": 0.0}
    step = scene.layer.step

    def watched():
        step()
        seen["iterations"] = max(seen["iterations"], int(data.solver_niter.max()))
        seen["travel"] = max(seen["travel"], float(np.abs(data.ctrl[travel]).max()))

    scene.layer.step = watched
    while not scene.step(np.zeros((1, 4)))[2][0]:
        pass
    assert seen["travel"] == 0.0
    assert np.abs(data.qpos[wheel] - start).max() * tracks.WHEEL_RADIUS < 1e-3
    assert seen["iterations"] < model.opt.iterations
