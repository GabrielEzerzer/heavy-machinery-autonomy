"""Scoring over the fixed development seeds and throughput timing. Each episode's outcome is joined
with its draws, then grouped overall and by thirds of every drawn parameter into one table per
controller. Held-out seeds are refused here."""

from __future__ import annotations

import csv
import math
import pickle
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from hma.train.scene import CAUSES
from hma.train.task import HELD_OUT_BASE
from hma.train.vec import SceneVecEnv

DRAWN = ("target", "ramp_up", "hold", "ramp_down", "prop_x", "pitch", "friction", "step", "grade",
         "cross", "delay", "lag", "deadband", "sensor_latency", "counterweight", "link_mass")
DEGREES = {"target", "pitch"}


def episode_row(scene, info: dict, controller: str) -> dict:
    index = scene.seeds.index(info["seed"])
    task, ground, machine = scene.tasks[index], scene.grounds[index], scene.machines[index]
    return {
        "controller": controller, "rung": scene.rung, **info,
        "target": task.target, "ramp_up": task.ramp_up, "hold": task.hold,
        "ramp_down": task.ramp_down, "prop_x": task.prop_x, "pitch": task.pitch,
        "friction": ground.friction, "step": ground.step, "grade": ground.grade,
        "cross": ground.cross, "delay": float(np.mean(machine.delay)),
        "lag": float(np.mean(machine.lag)), "deadband": float(np.mean(machine.deadband)),
        "sensor_latency": machine.sensor_latency, "counterweight": machine.counterweight,
        "link_mass": float(np.mean(machine.link_mass)),
    }


def load_policy(checkpoint: Path):
    from stable_baselines3 import PPO

    model = PPO.load(checkpoint / "model.zip", device="cpu")
    with open(checkpoint / "normalize.pkl", "rb") as f:
        normalize = pickle.load(f)
    normalize.training = False
    return lambda obs: model.predict(normalize.normalize_obs(obs), deterministic=True)[0]


def run_seeds(rung: str, mode: str, seeds: list[int], machines: int,
              checkpoint: str | None) -> list[dict]:
    if any(seed >= HELD_OUT_BASE for seed in seeds):
        raise ValueError("held-out seeds are never scored")
    import torch

    torch.set_num_threads(1)
    policy = load_policy(Path(checkpoint)) if checkpoint else None
    controller = "policy" if checkpoint else "baseline"
    env = SceneVecEnv(min(machines, len(seeds)), rung, mode, 0, seeds=seeds)
    wanted, rows = set(seeds), []
    obs = env.reset()
    while len(rows) < len(seeds):
        actions = policy(obs) if policy else np.zeros((env.num_envs, 4))
        obs, _, done, infos = env.scene.step(np.asarray(actions, float))
        for i in np.flatnonzero(done):
            if infos[i]["seed"] in wanted:
                rows.append(episode_row(env.scene, infos[i], controller))
                wanted.discard(infos[i]["seed"])
            obs[i] = env._start(i)
        obs = obs.astype(np.float32)
    return rows


def score(rung: str, mode: str, seeds: list[int], machines: int, procs: int,
          checkpoint: str | None) -> list[dict]:
    chunks = [seeds[k::procs] for k in range(procs) if seeds[k::procs]]
    if len(chunks) == 1:
        return run_seeds(rung, mode, chunks[0], machines, checkpoint)
    with ProcessPoolExecutor(len(chunks), mp_context=get_context("spawn")) as pool:
        parts = pool.map(run_seeds, [rung] * len(chunks), [mode] * len(chunks), chunks,
                         [machines] * len(chunks), [checkpoint] * len(chunks))
        rows = [row for part in parts for row in part]
    return sorted(rows, key=lambda r: (r["controller"], r["seed"]))


def _summary(rows: list[dict]) -> str:
    if not rows:
        return "| 0 | - | - | - |"
    passed = 100.0 * np.mean([r["passed"] for r in rows])
    error = math.degrees(float(np.mean([r["tilt_error"] for r in rows])))
    causes = Counter(r["cause"] for r in rows if r["cause"])
    ends = ", ".join(f"{c} {causes[c]}" for c in CAUSES if causes[c]) or "none"
    return f"| {len(rows)} | {passed:.0f}% | {error:.2f} | {ends} |"


def _value(row: dict, name: str) -> float:
    return math.degrees(row[name]) if name in DEGREES else row[name]


def table(rows: list[dict]) -> str:
    lines = []
    for controller in sorted({r["controller"] for r in rows}):
        mine = [r for r in rows if r["controller"] == controller]
        lines += [f"### {controller}, rung {mine[0]['rung']}", "",
                  "| group | episodes | pass | tilt error (deg) | early ends |",
                  "|---|---|---|---|---|", "| all " + _summary(mine), ""]
        lines += ["| drawn parameter (range) | episodes | pass | tilt error (deg) | early ends |",
                  "|---|---|---|---|---|"]
        for name in DRAWN:
            values = np.array([_value(r, name) for r in mine])
            if np.ptp(values) == 0:
                continue
            edges = np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0])
            for k in range(3):
                upper = values <= edges[k + 1] if k == 2 else values < edges[k + 1]
                picked = [r for r, keep in zip(mine, (values >= edges[k]) & upper, strict=True)
                          if keep]
                label = f"{name} {edges[k]:.3g} to {edges[k + 1]:.3g}"
                lines.append(f"| {label} " + _summary(picked))
        lines.append("")
    return "\n".join(lines)


def write(rows: list[dict], out: Path, name: str) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    path = out / f"{name}.md"
    path.write_text(table(rows), encoding="utf-8")
    return path


def throughput(rung: str, sizes: list[int], seconds: float, seed: int = 0) -> list[dict]:
    results = []
    for n in sizes:
        env = SceneVecEnv(n, rung, "residual", seed)
        env.reset()
        rng = np.random.default_rng(seed)
        for _ in range(3):
            env.step_async(rng.uniform(-0.2, 0.2, (n, 4)))
            env.step_wait()
        decisions, wall0, cpu0 = 0, time.perf_counter(), time.process_time()
        while time.perf_counter() - wall0 < seconds:
            env.step_async(rng.uniform(-0.2, 0.2, (n, 4)))
            env.step_wait()
            decisions += 1
        wall, cpu = time.perf_counter() - wall0, time.process_time() - cpu0
        results.append({"machines": n, "control_steps_per_s": decisions * n / wall,
                        "scene_decisions_per_s": decisions / wall, "cpu_share": cpu / wall,
                        "redos": env.scene.layer.redos})
    return results
