import pickle
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, RANGES, BaseController, Hydraulics, draw_machine
from hma.hydraulics.params import LITRE
from hma.machine import inverse_kinematics, load_scene, machine_prefix
from hma.train import footing
from hma.train.scene import FIELDS, OBS_HIGH, OBS_LOW, PropScene
from hma.train.task import (
    DEV_BASE,
    HELD_OUT_BASE,
    PropPath,
    Settings,
    dev_seeds,
    held_out_seeds,
    training_seed,
)

SEED = DEV_BASE + 3


def run(scene, actions):
    seen = []
    for a in actions:
        obs, reward, done, info = scene.step(a)
        seen.append((obs.copy(), reward.copy(), done.copy()))
    return seen


def same(a, b):
    return all(np.array_equal(x, y) for p, q in zip(a, b, strict=True)
               for x, y in zip(p, q, strict=True))


def test_zero_residual_is_the_plain_baseline_bit_for_bit():
    scene = PropScene(1, "machines")
    scene.reset(0, SEED)
    run(scene, [np.zeros((1, 4))] * 60)

    model = load_scene(1)
    model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_WARMSTART)
    data = mujoco.MjData(model)
    layer = Hydraulics(model, data, [NOMINAL])
    machine, ground, task = scene.draw(0, SEED)
    path = PropPath(task, footing.rear_pivot(model))
    footing.apply_ground(model, 0, ground)
    layer.reset(0, machine, inverse_kinematics(*path(0.0)))
    footing.Parking(model, 1).park(model, data, 0)
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
    saved = pickle.loads(pickle.dumps(scene.snapshot()))
    straight = run(scene, actions[20:])
    other = PropScene(2, "ground")
    other.reset(0, 11)
    other.reset(1, 12)
    run(other, actions[:5])
    other.restore(saved)
    assert same(straight, run(other, actions[20:]))
    assert np.array_equal(scene.data.qpos, other.data.qpos)


def test_resetting_one_machine_leaves_its_neighbours_bit_for_bit():
    actions = np.random.default_rng(1).uniform(-0.3, 0.3, (60, 3, 4))

    def neighbours(reset_middle):
        scene = PropScene(3, "ground")
        for i in range(3):
            scene.reset(i, SEED + i)
        seen = []
        for k, a in enumerate(actions):
            if reset_middle and k in (17, 41):
                scene.reset(1, 555 + k)
            obs, reward, done, _ = scene.step(a)
            for i in np.flatnonzero(done):
                scene.reset(i, 900 + i)
            seen.append((obs[[0, 2]].copy(), reward[[0, 2]].copy()))
        return seen, scene.layer.redos

    alone, _ = neighbours(False)
    disturbed, redos = neighbours(True)
    assert same(alone, disturbed)
    assert redos > 0


class Corner(PropScene):
    def __init__(self, machine, *args, **kwargs):
        self.corner = machine
        super().__init__(*args, **kwargs)

    def draw(self, index, seed):
        _, ground, task = super().draw(index, seed)
        return self.corner, ground, task


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
    for k in range(120):
        obs, reward, done, info = scene.step(rng.uniform(-1.0, 1.0, (2, 4)))
        seen += list(obs)
        assert np.all((reward >= 0.0) & (reward <= 1.0))
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
        heights = footing.heights(footing.FLAT)
        heights[:20, :20] = -1.0
        hf = scene.model.hfield(machine_prefix(0) + "ground")
        rows = heights.size
        scene.model.hfield_data[int(hf.adr[0]):int(hf.adr[0]) + rows] = (
            (heights + 6.0) / 8.0).ravel()

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


def test_an_untrained_residual_policy_reproduces_the_baseline_episode():
    pytest.importorskip("stable_baselines3")
    from stable_baselines3.common.vec_env import VecNormalize

    from hma.train.__main__ import EGLI_HUTTER, SPREAD, build_model
    from hma.train.vec import SceneVecEnv

    pitch = sum(count for name, count, *_ in FIELDS[:[f[0] for f in FIELDS].index("tilt")])

    def episode(untrained):
        env = SceneVecEnv(1, "nominal", "residual", 0, seeds=[DEV_BASE + 4])
        venv = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=EGLI_HUTTER["gamma"])
        model = build_model(venv, 128, 0, SPREAD["residual"]) if untrained else None
        if model:
            spread = model.policy.log_std.exp().detach().numpy()
            assert np.allclose(spread, SPREAD["residual"])
        obs, largest, tilts = venv.reset(), 0.0, []
        while True:
            action = (model.predict(obs, deterministic=True)[0] if model
                      else np.zeros((1, 4), np.float32))
            largest = max(largest, float(np.abs(action).max()))
            obs, _, done, info = venv.step(action)
            tilts.append(env.scene.history[0, -1, pitch] * np.pi / 2)
            if done[0]:
                return info[0], largest, np.array(tilts)

    policy, largest, policy_tilts = episode(True)
    baseline, _, baseline_tilts = episode(False)
    assert largest <= 0.02
    assert (policy["cause"], policy["passed"]) == (baseline["cause"], baseline["passed"])
    assert abs(policy["tilt_error"] - baseline["tilt_error"]) <= np.radians(0.05)
    assert np.abs(policy_tilts - baseline_tilts).max() <= np.radians(0.5)
