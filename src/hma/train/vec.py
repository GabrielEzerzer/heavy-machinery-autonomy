"""Vectorised environments for Stable-Baselines3: one PropScene as a VecEnv whose machines restart
on fresh seeds when they finish, and several scenes in parallel worker processes over pipes."""

from __future__ import annotations

import multiprocessing as mp
import os

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

from hma.train.scene import OBS_HIGH, OBS_LOW, PropScene
from hma.train.task import Settings, training_seed


def spaces_for() -> tuple[spaces.Box, spaces.Box]:
    obs = spaces.Box(OBS_LOW.astype(np.float32), OBS_HIGH.astype(np.float32), dtype=np.float32)
    return obs, spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)


class SceneVecEnv(VecEnv):
    def __init__(self, n: int, rung: str, mode: str, seed: int,
                 settings: Settings | None = None, seeds: list[int] | None = None):
        self.scene = PropScene(n, rung, mode, settings)
        self.stream = np.random.default_rng((seed, 99))
        self.queue = list(seeds) if seeds is not None else None
        self.returns = np.zeros(n)
        self.lengths = np.zeros(n, np.intp)
        self._actions = np.zeros((n, 4))
        self.render_mode = None
        super().__init__(n, *spaces_for())

    def next_seed(self) -> int:
        if self.queue is not None:
            return self.queue.pop(0) if self.queue else -1
        return training_seed(self.stream)

    def _start(self, index: int) -> np.ndarray:
        seed = self.next_seed()
        self.returns[index], self.lengths[index] = 0.0, 0
        return self.scene.reset(index, seed if seed >= 0 else 0)

    def reset(self) -> np.ndarray:
        return np.stack([self._start(i) for i in range(self.num_envs)]).astype(np.float32)

    def step_async(self, actions: np.ndarray) -> None:
        self._actions = np.asarray(actions, float).reshape(self.num_envs, 4)

    def step_wait(self):
        obs, reward, done, infos = self.scene.step(self._actions)
        self.returns += reward
        self.lengths += 1
        for i in np.flatnonzero(done):
            infos[i]["terminal_observation"] = obs[i].astype(np.float32)
            infos[i]["TimeLimit.truncated"] = infos[i]["truncated"]
            infos[i]["episode"] = {"r": float(self.returns[i]), "l": int(self.lengths[i])}
            obs[i] = self._start(i)
        return obs.astype(np.float32), reward.astype(np.float32), done, infos

    def snapshot(self) -> dict:
        return {"scene": self.scene.snapshot(), "stream": self.stream.bit_generator.state,
                "queue": None if self.queue is None else list(self.queue),
                "returns": self.returns.copy(), "lengths": self.lengths.copy()}

    def restore(self, snapshot: dict) -> np.ndarray:
        self.scene.restore(snapshot["scene"])
        self.stream.bit_generator.state = snapshot["stream"]
        self.queue = snapshot["queue"]
        self.returns, self.lengths = snapshot["returns"].copy(), snapshot["lengths"].copy()
        return self.observations()

    def observations(self) -> np.ndarray:
        return self.scene.history.reshape(self.num_envs, -1).astype(np.float32)

    def close(self) -> None:
        pass

    def get_attr(self, attr_name, indices=None):
        return [getattr(self, attr_name)] * len(self._get_indices(indices))

    def set_attr(self, attr_name, value, indices=None) -> None:
        setattr(self, attr_name, value)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        return [getattr(self, method_name)(*args, **kwargs)] * len(self._get_indices(indices))

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * len(self._get_indices(indices))

    def _get_indices(self, indices):
        if indices is None:
            return range(self.num_envs)
        return [indices] if isinstance(indices, int) else indices


ONE_THREAD = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")


def _worker(pipe, args) -> None:
    import torch

    torch.set_num_threads(1)
    env = SceneVecEnv(*args)
    while True:
        command, payload = pipe.recv()
        if command == "step":
            env.step_async(payload)
            pipe.send(env.step_wait())
        elif command == "reset":
            pipe.send(env.reset())
        elif command == "snapshot":
            pipe.send(env.snapshot())
        elif command == "restore":
            env.restore(payload)
            pipe.send(env.observations())
        elif command == "close":
            pipe.send(None)
            pipe.close()
            return


class ProcVecEnv(VecEnv):
    def __init__(self, scenes: int, n: int, rung: str, mode: str, seed: int,
                 settings: Settings | None = None):
        context = mp.get_context("spawn")
        self.n = n
        self.pipes, self.processes = [], []
        saved = {name: os.environ.get(name) for name in ONE_THREAD}
        os.environ.update(dict.fromkeys(ONE_THREAD, "1"))
        try:
            for k in range(scenes):
                here, there = context.Pipe()
                process = context.Process(target=_worker,
                                          args=(there, (n, rung, mode, seed * 1000 + k, settings)),
                                          daemon=True)
                process.start()
                there.close()
                self.pipes.append(here)
                self.processes.append(process)
        finally:
            for name, value in saved.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
        self.closed = False
        super().__init__(scenes * n, *spaces_for())

    @property
    def pids(self) -> list[int]:
        return [p.pid for p in self.processes]

    def _all(self, command: str, payloads=None) -> list:
        for k, pipe in enumerate(self.pipes):
            pipe.send((command, None if payloads is None else payloads[k]))
        return [pipe.recv() for pipe in self.pipes]

    def reset(self) -> np.ndarray:
        return np.concatenate(self._all("reset"))

    def step_async(self, actions: np.ndarray) -> None:
        actions = np.asarray(actions, float).reshape(len(self.pipes), self.n, 4)
        for pipe, chunk in zip(self.pipes, actions, strict=True):
            pipe.send(("step", chunk))

    def step_wait(self):
        results = [pipe.recv() for pipe in self.pipes]
        obs, reward, done, infos = zip(*results, strict=True)
        return (np.concatenate(obs), np.concatenate(reward), np.concatenate(done),
                [info for chunk in infos for info in chunk])

    def snapshot(self) -> list[dict]:
        return self._all("snapshot")

    def restore(self, snapshots: list[dict]) -> np.ndarray:
        return np.concatenate(self._all("restore", snapshots))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for pipe, process in zip(self.pipes, self.processes, strict=True):
            if process.is_alive():
                try:
                    pipe.send(("close", None))
                    pipe.recv()
                except (BrokenPipeError, EOFError, ConnectionResetError):
                    pass
            pipe.close()
        for process in self.processes:
            process.join(timeout=30)

    def get_attr(self, attr_name, indices=None):
        if attr_name == "render_mode":
            return [None] * self.num_envs
        raise NotImplementedError("attributes live in the worker processes")

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError("attributes live in the worker processes")

    def env_method(self, method_name, *args, indices=None, **kwargs):
        raise NotImplementedError("methods run in the worker processes")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * self.num_envs
