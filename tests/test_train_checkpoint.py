
import numpy as np
import pytest


def _ppo(env):
    from stable_baselines3.common.vec_env import VecNormalize

    from hma.train.__main__ import EGLI_HUTTER, build_model

    venv = VecNormalize(env, norm_obs=True, norm_reward=False, gamma=EGLI_HUTTER["gamma"])
    return build_model(venv, 8, 0, 0.3), venv


@pytest.mark.train_extra
def test_a_restored_checkpoint_continues_the_same_episodes(tmp_path):
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


@pytest.mark.train_extra
def test_a_resume_draws_fresh_noise_and_fresh_episodes(tmp_path):
    import torch

    from hma.train.__main__ import load_checkpoint, save_checkpoint
    from hma.train.vec import SceneVecEnv

    env = SceneVecEnv(1, "nominal", "residual", 0)
    model, venv = _ppo(env)
    obs = venv.reset()
    first_seeds = [env.scene.seeds[0]] + [env.next_seed() for _ in range(20)]
    started = np.random.random(), torch.rand(4).numpy()
    model._last_obs = obs
    model._last_episode_starts = np.ones(1, bool)
    model.num_timesteps = 10
    save_checkpoint(tmp_path / "checkpoint", model, venv, env, {"rung": "nominal"})

    other = SceneVecEnv(1, "nominal", "residual", 0)
    resumed, _, _ = load_checkpoint(tmp_path / "checkpoint", other)
    after = np.random.random(), torch.rand(4).numpy()
    assert after[0] != started[0] and not np.array_equal(after[1], started[1])
    assert resumed.seed == 0
    assert other.next_seed() not in first_seeds


def test_a_resume_drops_curve_rows_written_after_its_checkpoint(tmp_path):
    from hma.train.__main__ import trim_curve

    path = tmp_path / "curve.csv"
    rows = ["timesteps,seconds", "100,1", "200,2", "300,3"]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    trim_curve(path, 200)
    assert path.read_text(encoding="utf-8").splitlines() == rows[:3]
    trim_curve(tmp_path / "missing.csv", 200)
