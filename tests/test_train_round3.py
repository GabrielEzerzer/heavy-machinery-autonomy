from dataclasses import replace

import mujoco
import numpy as np
import pytest

from hma.hydraulics import NOMINAL, Hydraulics, draw_machine
from hma.machine import load_scene, machine_prefix
from hma.machine.kinematics import BOOM_PIVOT
from hma.train import evaluate, footing
from hma.train.scene import PropScene
from hma.train.task import (
    DEV_BASE,
    HELD_OUT_BASE,
    PropPath,
    Settings,
    dev_seeds,
    draw_task,
    reachable,
)

SEED = DEV_BASE + 3
STRIKE_POSE = (-2.36, -0.369, 2.501, 0.521)


def finish(scene, index=0):
    while True:
        _, reward, done, info = scene.step(np.zeros((scene.n, 4)))
        if done[index]:
            return info[index], reward[index]


class Reference(PropScene):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.feet = footing.front_feet(self.model, 0)
        self.body = self.model.body(machine_prefix(0) + "base").id
        self.touching, self.worst = set(), 0.0

    def reset(self, index, seed):
        obs = super().reset(index, seed)
        self.touching = self._touching()
        self.worst = 0.0
        return obs

    def _touching(self):
        pairs = self.data.contact.geom[:self.data.ncon]
        return {int(g) for g in pairs.ravel() if g in self.feet}

    def _contacts(self):
        touching = self._touching()
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.body,
                                 velocity, 0)
        rot = self.data.xmat[self.body].reshape(3, 3)
        for g in touching - self.touching:
            corner = self.model.geom_pos[g] - (0.0, 0.0, self.model.geom_size[g, 0])
            point = velocity[3:] + np.cross(velocity[:3], rot @ corner)
            self.worst = max(self.worst, -point[2])
        self.touching = touching
        return super()._contacts()


def test_landing_is_each_front_corners_speed_just_before_its_own_first_contact():
    scene = Reference(1, "nominal")
    scene.reset(0, DEV_BASE + 4)
    info, _ = finish(scene)
    assert scene.worst > 0.1
    assert info["landing"] == pytest.approx(scene.worst, abs=1e-9)


def test_restoring_never_ends_an_episode_over_a_stale_warning_count():
    scene = PropScene(1, "nominal")
    scene.reset(0, SEED)
    for _ in range(3):
        scene.step(np.zeros((1, 4)))
    saved = scene.snapshot()
    other = PropScene(1, "nominal")
    other.reset(0, SEED + 1)
    other.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 5
    other.restore(saved)
    _, _, done, _ = other.step(np.zeros((1, 4)))
    assert not done[0]


@pytest.mark.parametrize("kind", ["acceleration", "velocity"])
def test_a_solver_failure_ends_only_its_own_machine_and_cuts_the_rest_short(kind):
    scene = PropScene(3, "nominal")
    for i in range(3):
        scene.reset(i, SEED + i)
    for _ in range(2):
        scene.step(np.zeros((3, 4)))
    if kind == "acceleration":
        scene.data.xfrc_applied[scene.model.body(machine_prefix(1) + "bucket").id, 2] = 1e30
    else:
        scene.data.qvel[scene._base_v[1]] = np.nan
    _, reward, done, info = scene.step(np.zeros((3, 4)))
    assert done.all() and np.all(reward == 0.0)
    assert info[1]["cause"] == "unstable" and not info[1]["truncated"]
    for i in (0, 2):
        assert info[i]["cause"] is None and info[i]["truncated"] and info[i]["cut_short"]
        assert not info[i]["passed"]


def test_the_bucket_striking_its_own_machine_ends_the_episode_as_a_strike():
    scene = PropScene(1, "nominal", mode="direct")
    scene.reset(0, SEED)
    scene.layer.reset(0, scene.machines[0], STRIKE_POSE)
    scene.parking.park(scene.model, scene.data, 0)
    mujoco.mj_forward(scene.model, scene.data)
    _, reward, done, info = scene.step(np.zeros((1, 4)))
    assert done[0] and info[0]["cause"] == "strike" and reward[0] == 0.0


def test_the_pass_mark_at_each_of_its_boundaries():
    s = Settings()
    scene = PropScene(1, "nominal")
    scene.reset(0, SEED)
    scene.hold_steps[0] = 1000
    tiny = 1e-9

    def passes(share=s.hold_share, landing=0.0, carried=1.0, speed=0.0, cause=None):
        scene.in_band[0], scene.landing[0] = round(share * 1000), landing
        return scene.outcome(0, cause, 0.0, carried, speed)["passed"]

    assert passes() and not passes(share=s.hold_share - 0.001)
    assert passes(landing=s.landing - tiny) and not passes(landing=s.landing)
    assert passes(carried=s.carry) and not passes(carried=s.carry - tiny)
    assert passes(speed=s.still - tiny) and not passes(speed=s.still)
    assert not passes(cause="tilt")


@pytest.mark.train_extra
def test_the_scheduled_end_is_a_truncation_and_an_early_end_is_terminal():
    from hma.train.vec import SceneVecEnv

    env = SceneVecEnv(2, "nominal", "direct", 0, seeds=[DEV_BASE, DEV_BASE])
    env.reset()
    seen = {}
    while len(seen) < 2:
        actions = env.scene.baseline()
        actions[1] = (0.0, 1.0, 0.0, 0.0)
        env.step_async(actions)
        _, reward, done, infos = env.step_wait()
        for i in np.flatnonzero(done):
            seen.setdefault(int(i), (infos[i], float(reward[i])))
    (end, _), (early, early_reward) = seen[0], seen[1]
    assert end["cause"] is None and end["TimeLimit.truncated"]
    assert end["terminal_observation"].shape == env.observation_space.shape
    assert early["cause"] == "tilt" and not early["TimeLimit.truncated"] and early_reward == 0.0


def test_scoring_refuses_held_out_and_repeated_seeds():
    for seeds in ([DEV_BASE, HELD_OUT_BASE], [DEV_BASE, DEV_BASE]):
        with pytest.raises(ValueError):
            evaluate.score("nominal", "residual", seeds, 2, 1, None)


def test_a_solver_failure_while_scoring_reruns_the_other_seeds_to_the_same_result(monkeypatch):
    seeds = dev_seeds(3)
    clean = evaluate.score("nominal", "residual", seeds, 3, 1, None)

    class Failing(PropScene):
        struck = False

        def step(self, actions):
            if not Failing.struck and self.k[1] == 3:
                Failing.struck = True
                self.data.xfrc_applied[self.model.body(machine_prefix(1) + "bucket").id, 2] = 1e30
            return super().step(actions)

    monkeypatch.setattr(evaluate, "PropScene", Failing)
    failed = evaluate.score("nominal", "residual", seeds, 3, 1, None)
    assert Failing.struck
    assert [r["seed"] for r in failed] == seeds and [r["slot"] for r in failed] == [0, 1, 2]
    assert failed[1]["cause"] == "unstable"
    for k in (0, 2):
        assert failed[k] == clean[k]
    assert all(r["scene_size"] == 3 for r in failed)


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
            def draw(self, index, seed):
                _, ground, task = super().draw(index, seed)
                return drawn, ground, task

        scene = Fixed(1, "machines", mode="direct")
        seen = [scene.reset(0, SEED)]
        for _ in range(30):
            seen.append(scene.step(np.zeros((1, 4)))[0][0])
        return np.array(seen)

    assert np.array_equal(observed(machine), observed(slower))


@pytest.mark.train_extra
def test_closing_survives_a_worker_that_died():
    from hma.train.vec import ProcVecEnv

    envs = ProcVecEnv(2, 1, "nominal", "residual", 0)
    try:
        envs.processes[0].terminate()
        envs.processes[0].join(30)
        envs.close()
        assert not any(p.is_alive() for p in envs.processes)
    finally:
        for process in envs.processes:
            if process.is_alive():
                process.terminate()


def test_every_drawn_task_reaches_the_drawn_ground_at_its_press_point():
    pivot = footing.rear_pivot(load_scene(1))
    for seed in dev_seeds():
        ground = footing.draw_ground(seed)
        task = draw_task(seed, Settings(), pivot,
                         lambda x, y, g=ground: footing.surface(g, x, y), 0.1)
        height = footing.surface(ground, task.prop_x, float(BOOM_PIVOT[1]))
        assert reachable(PropPath(task, pivot, height), 0.1), seed


def _ppo(env):
    from stable_baselines3.common.vec_env import VecNormalize

    from hma.train.__main__ import EGLI_HUTTER, build_model

    venv = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=EGLI_HUTTER["gamma"])
    return build_model(venv, 8, 0, 0.3), venv


@pytest.mark.train_extra
def test_a_resumed_run_continues_where_the_checkpoint_was_taken(tmp_path):
    from hma.train.__main__ import latest_checkpoint, load_checkpoint, save_checkpoint
    from hma.train.vec import SceneVecEnv

    env = SceneVecEnv(2, "nominal", "residual", 0)
    model, venv = _ppo(env)
    model._last_obs = venv.reset()
    rng = np.random.default_rng(4)
    for _ in range(5):
        model._last_obs = venv.step(rng.uniform(-0.2, 0.2, (2, 4)))[0]
    model._last_episode_starts = np.array([True, False])
    model.num_timesteps = 10
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, venv, env, {"rung": "nominal"})
    later = rng.uniform(-0.2, 0.2, (6, 2, 4))
    straight = [venv.step(a)[0] for a in later]

    def broken():
        raise OSError("interrupted")

    env.snapshot = broken
    with pytest.raises(OSError):
        save_checkpoint(checkpoint, model, venv, env, {"rung": "nominal"})
    assert latest_checkpoint(checkpoint) == checkpoint

    other = SceneVecEnv(2, "nominal", "residual", 7)
    resumed, rvenv, config = load_checkpoint(checkpoint, other)
    assert config["rung"] == "nominal" and config["timesteps"] == 10
    assert resumed.num_timesteps == 10
    assert np.array_equal(resumed._last_episode_starts, [True, False])
    for a, b in zip(model.policy.state_dict().values(), resumed.policy.state_dict().values(),
                    strict=True):
        assert np.array_equal(a.numpy(), b.numpy())
    replay = [rvenv.step(a)[0] for a in later]
    assert all(np.array_equal(a, b) for a, b in zip(straight, replay, strict=True))

    checkpoint.rename(tmp_path / "checkpoint.old")
    assert latest_checkpoint(checkpoint) == tmp_path / "checkpoint.old"


@pytest.mark.train_extra
def test_steps_is_the_total_across_a_resume():
    from stable_baselines3.common.callbacks import BaseCallback

    from hma.train.__main__ import learn
    from hma.train.vec import SceneVecEnv

    class Stop(BaseCallback):
        def _on_step(self):
            return False

    env = SceneVecEnv(1, "nominal", "residual", 0)
    model, venv = _ppo(env)
    model._last_obs = venv.reset()
    model._last_episode_starts = np.ones(1, bool)
    model.num_timesteps = 64
    learn(model, 100, Stop(), resume=True)
    assert model._total_timesteps == 100


def test_the_score_does_not_depend_on_the_number_of_processes():
    seeds = dev_seeds(2)
    alone = evaluate.score("nominal", "residual", seeds, 1, 1, None)
    spread = evaluate.score("nominal", "residual", seeds, 1, 2, None)
    assert [r["seed"] for r in alone] == seeds
    assert alone == spread
